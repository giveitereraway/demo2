from collections import deque
from dataclasses import dataclass
import numpy as np
from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask


# 标准差顺序为高度（m）、航向（°）、机体前向速度（m/s）。
HEADING_MEASUREMENT_NOISE_STD = (
    (0.0, 0.0, 0.0), (10.0, 1.0, 1.0), (25.0, 3.0, 3.0), (50.0, 5.0, 5.0),
)
HEADING_MEASUREMENT_NOISE_PROBABILITIES = (0.2, 0.3, 0.3, 0.2)


@dataclass(frozen=True)
class HeadingDegradationSpec:
    """制导/目标跟踪误差通道的混合退化参数（丢帧率、时延步数、三通道噪声标准差）。"""

    dropout_probability: float = 0.0
    delay_steps: int = 0
    altitude_noise_std_m: float = 0.0
    heading_noise_std_deg: float = 0.0
    speed_noise_std_mps: float = 0.0


# 与 heading_observation_degradation_eval.py 中 E0–E3 严格对齐，E4–E6 留作分布外泛化评估。
HEADING_MIXED_DEGRADATION_SPECS = (
    HeadingDegradationSpec(0.0, 0, 0.0, 0.0, 0.0),
    HeadingDegradationSpec(0.10, 1, 10.0, 1.0, 1.0),
    HeadingDegradationSpec(0.20, 2, 25.0, 3.0, 3.0),
    HeadingDegradationSpec(0.30, 3, 50.0, 5.0, 5.0),
)
HEADING_MIXED_DEGRADATION_PROBABILITIES = (0.25, 0.30, 0.25, 0.20)


class HeadingErrorChannelDegrader:
    """对 Heading 观测的前三维目标误差施加高斯噪声、零阶保持丢帧与队列延迟。"""

    ERROR_SLICE = slice(0, 3)

    def __init__(self, spec: HeadingDegradationSpec, initial_obs: np.ndarray, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.dropout_probability = spec.dropout_probability
        self.noise_std = np.array([
            spec.altitude_noise_std_m / 1000.0,
            np.deg2rad(spec.heading_noise_std_deg),
            spec.speed_noise_std_mps / 340.0,
        ], dtype=np.float64)
        self.last_valid = np.asarray(initial_obs[self.ERROR_SLICE], dtype=np.float64).copy()
        self.delay = deque([self.last_valid.copy() for _ in range(spec.delay_steps)])
        self.dropped_frames = 0

    def transform(self, obs: np.ndarray) -> np.ndarray:
        measured = np.asarray(obs[self.ERROR_SLICE], dtype=np.float64) + self.rng.standard_normal(3) * self.noise_std
        measured[1] = (measured[1] + np.pi) % (2 * np.pi) - np.pi
        if self.rng.random() < self.dropout_probability:
            self.dropped_frames += 1
        else:
            self.last_valid = np.clip(measured, -10.0, 10.0)
        if self.spec.delay_steps > 0:
            self.delay.append(self.last_valid.copy())
            delayed = self.delay.popleft()
        else:
            delayed = self.last_valid
        degraded = np.asarray(obs, dtype=np.float64).copy()
        degraded[self.ERROR_SLICE] = delayed
        return degraded


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        elif taskname == "HumanFreeFly":
            self.task = HumanFreeFlyTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0 # 重置步数计数器
        self.reset_simulators() # 重置仿真器状态：这会重新初始化飞机的位置、姿态、目标状态等
        self.heading_turn_counts = 0 # 重置航向转弯计数
        self.task.reset(self) # 重置任务状态
        obs = self.get_obs() # 获取初始观测
        return self._pack(obs) # 打包并返回观测

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()


class HeadingMeasurementNoiseEnv(SingleControlEnv):
    """在 Heading 的策略观测中加入测量噪声，固定档位用于独立评估。"""

    def __init__(self, config_name, *, noise_level=None, sensor_seed=0):
        self.noise_level = noise_level
        self.sensor_seed = sensor_seed
        self._noise_episode_seeds = np.random.SeedSequence()
        super().__init__(config_name)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        # 独立于飞行初态和目标指令的随机数流。
        self._noise_episode_seeds = np.random.SeedSequence([seeds[0], self.sensor_seed], spawn_key=(1,))
        return seeds

    def reset(self):
        # 每局独立派生噪声流，前一局长度不会改变下一局的噪声档位。
        self._noise_rng = np.random.default_rng(self._noise_episode_seeds.spawn(1)[0])
        level = self.noise_level
        if level is None:
            level = self._noise_rng.choice(
                len(HEADING_MEASUREMENT_NOISE_STD), p=HEADING_MEASUREMENT_NOISE_PROBABILITIES,
            )
        self._noise_std = HEADING_MEASUREMENT_NOISE_STD[level] if level else None
        self._noise_step = None
        return super().reset()

    def get_obs(self):
        # 同一决策步重复读观测（例如切换指令）时，复用同一份测量偏差。
        if self._noise_step != self.current_step:
            self._noise_step = self.current_step
            self._measurement_error = self._noise_rng.normal(scale=self._noise_std) if self._noise_std else None
        return {
            agent_id: self.task.get_obs(self, agent_id, measurement_error=self._measurement_error)
            for agent_id in self.agents
        }


class HeadingTrackingEvalEnv(HeadingMeasurementNoiseEnv):
    record_heading_tracking = True

    def __init__(self, config_name, *, noise_level=0):
        super().__init__(config_name, noise_level=noise_level)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self.episode_seeds = np.random.default_rng(seeds[0])
        return seeds

    def reset(self):
        # 每回合独立取种子，前一回合提前结束也不会改变后续场景。
        super().seed(int(self.episode_seeds.integers(2**32)))
        return super().reset()


class HeadingMixedDegradationEnv(SingleControlEnv):
    """在 Heading 训练采样中按档位混合注入噪声、零阶保持丢帧和时延，不改变底层飞行物理与评分。"""

    def __init__(self, config_name, *, degradation_level=None, sensor_seed=0):
        self.degradation_level = degradation_level
        self.sensor_seed = sensor_seed
        self._degrade_episode_seeds = np.random.SeedSequence()
        self._degrader = None
        self._degrade_step = None
        self._cached_obs = None
        super().__init__(config_name)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self._degrade_episode_seeds = np.random.SeedSequence([seeds[0], self.sensor_seed], spawn_key=(2,))
        return seeds

    def reset(self):
        self._degrade_rng = np.random.default_rng(self._degrade_episode_seeds.spawn(1)[0])
        level = self.degradation_level
        if level is None:
            level = int(
                self._degrade_rng.choice(
                    len(HEADING_MIXED_DEGRADATION_SPECS), p=HEADING_MIXED_DEGRADATION_PROBABILITIES
                )
            )
        self._current_level = int(level)
        self._spec = HEADING_MIXED_DEGRADATION_SPECS[self._current_level]
        self._degrader = None
        self._degrade_step = None
        self._cached_obs = None
        return super().reset()

    def get_obs(self):
        if self._degrade_step != self.current_step or self._cached_obs is None:
            self._degrade_step = self.current_step
            clean_dict = {agent_id: self.task.get_obs(self, agent_id) for agent_id in self.agents}
            if self._current_level == 0:
                self._cached_obs = clean_dict
            else:
                degraded_dict = {}
                for agent_id, clean_obs in clean_dict.items():
                    if self._degrader is None:
                        self._degrader = HeadingErrorChannelDegrader(self._spec, clean_obs, self._degrade_rng)
                    degraded_dict[agent_id] = self._degrader.transform(clean_obs)
                self._cached_obs = degraded_dict
        return {agent_id: obs.copy() for agent_id, obs in self._cached_obs.items()}


class HeadingMixedTrackingEvalEnv(HeadingMixedDegradationEnv):
    """支持定期验证跟踪达标率的混合退化 Heading 评估环境。"""

    record_heading_tracking = True

    def __init__(self, config_name, *, degradation_level=0):
        super().__init__(config_name, degradation_level=degradation_level)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self.episode_seeds = np.random.default_rng(seeds[0])
        return seeds

    def reset(self):
        super().seed(int(self.episode_seeds.integers(2**32)))
        return super().reset()
