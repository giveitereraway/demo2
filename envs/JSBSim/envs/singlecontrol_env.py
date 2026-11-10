import numpy as np
from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask


# 标准差顺序为高度（m）、航向（°）、机体前向速度（m/s）。
HEADING_MEASUREMENT_NOISE_STD = (
    (0.0, 0.0, 0.0), (10.0, 1.0, 1.0), (25.0, 3.0, 3.0), (50.0, 5.0, 5.0),
)
HEADING_MEASUREMENT_NOISE_PROBABILITIES = (0.2, 0.3, 0.3, 0.2)


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
