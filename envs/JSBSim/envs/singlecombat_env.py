from collections import deque
from dataclasses import dataclass
import numpy as np
from .env_base import BaseEnv
from ..core.catalog import Catalog as c
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask, \
    TacticalHierarchicalSingleCombatTask, TacticalHierarchicalSingleCombatDodgeMissileTask, \
    TacticalHierarchicalSingleCombatShootTask
from ..human_task.HumanSingleCombatTask import HumanSingleCombatTask, HumanSingleCombat_shoot_Task


@dataclass(frozen=True)
class RadarDegradationSpec:
    """机载火控雷达目标航迹退化参数（脱锁丢帧率、滤波时延步数、距离/方位角/高度/速度噪声标准差）。"""

    dropout_probability: float = 0.0
    delay_steps: int = 0
    range_noise_std_m: float = 0.0
    azimuth_noise_std_deg: float = 0.0
    altitude_noise_std_m: float = 0.0
    velocity_noise_std_mps: float = 0.0


# E0 干净；E1–E2 用于训练域随机化与分布内测试；E3–E4 用于分布外强退化泛化测试。
RADAR_DEGRADATION_SPECS = (
    RadarDegradationSpec(0.0, 0, 0.0, 0.0, 0.0, 0.0),
    RadarDegradationSpec(0.10, 1, 100.0, 1.5, 30.0, 3.0),
    RadarDegradationSpec(0.20, 2, 200.0, 3.0, 60.0, 6.0),
    RadarDegradationSpec(0.35, 3, 350.0, 5.0, 100.0, 10.0),
    RadarDegradationSpec(0.50, 5, 500.0, 8.0, 150.0, 15.0),
)
RADAR_TRAIN_PROBABILITIES = (0.30, 0.35, 0.35)


def extract_true_enemy_track(env: BaseEnv, agent_id: str) -> np.ndarray:
    """提取敌机真实 8 维运动航迹 [n_m, e_m, u_m, alt_m, vn_mps, ve_mps, vd_mps, u_body_mps]。"""
    enm = env.agents[agent_id].enemies[0]
    enm_pos = enm.get_position()
    props = enm.get_property_values([
        c.position_h_sl_m,
        c.velocities_v_north_mps,
        c.velocities_v_east_mps,
        c.velocities_v_down_mps,
        c.velocities_u_mps,
    ])
    return np.array([
        enm_pos[0], enm_pos[1], enm_pos[2],
        props[0], props[1], props[2], props[3], props[4],
    ], dtype=np.float64)


def get_enemy_radar_track_from_env(env: BaseEnv, agent_id: str) -> np.ndarray | None:
    """统一查询环境中是否存在针对指定智能体的退化雷达敌机航迹覆盖。"""
    overrides = getattr(env, "_radar_track_overrides", None)
    if overrides and agent_id in overrides:
        return overrides[agent_id]
    getter = getattr(env, "get_enemy_radar_track", None)
    if callable(getter):
        return getter(agent_id)
    return None


class RadarTrackDegrader:
    """对机载火控雷达测得的敌机相对距离/方位角/高度与多普勒速度执行加噪、丢帧保持与延迟队列处理。"""

    def __init__(self, spec: RadarDegradationSpec, rng: np.random.Generator):
        if not 0.0 <= spec.dropout_probability <= 1.0:
            raise ValueError("dropout_probability 必须位于 [0, 1]")
        if spec.delay_steps < 0:
            raise ValueError("delay_steps 不能为负数")
        self.spec = spec
        self.rng = rng
        self._last_valid: np.ndarray | None = None
        self._delay_buffer: deque[np.ndarray] = deque()
        self.transform_count = 0
        self.dropout_count = 0

    def reset(self, initial_track: np.ndarray) -> None:
        track = np.asarray(initial_track, dtype=np.float64).copy()
        self._last_valid = track
        self._delay_buffer = deque(track.copy() for _ in range(self.spec.delay_steps))
        self.transform_count = 0
        self.dropout_count = 0

    def _measure_noisy_track(self, ego_pos: np.ndarray, true_track: np.ndarray) -> np.ndarray:
        measured = np.asarray(true_track, dtype=np.float64).copy()
        spec = self.spec
        if spec.range_noise_std_m > 0.0 or spec.azimuth_noise_std_deg > 0.0:
            rel_n = measured[0] - ego_pos[0]
            rel_e = measured[1] - ego_pos[1]
            horiz_range = float(np.hypot(rel_n, rel_e))
            azimuth = float(np.arctan2(rel_e, rel_n))
            if spec.range_noise_std_m > 0.0:
                horiz_range = max(0.0, horiz_range + float(self.rng.normal(0.0, spec.range_noise_std_m)))
            if spec.azimuth_noise_std_deg > 0.0:
                azimuth += float(self.rng.normal(0.0, np.deg2rad(spec.azimuth_noise_std_deg)))
            measured[0] = ego_pos[0] + horiz_range * np.cos(azimuth)
            measured[1] = ego_pos[1] + horiz_range * np.sin(azimuth)

        if spec.altitude_noise_std_m > 0.0:
            alt_err = float(self.rng.normal(0.0, spec.altitude_noise_std_m))
            measured[2] += alt_err
            measured[3] += alt_err

        if spec.velocity_noise_std_mps > 0.0:
            vel_err = self.rng.normal(0.0, spec.velocity_noise_std_mps, size=4)
            measured[4:8] += vel_err
            measured[7] = max(0.0, measured[7])
        return measured

    def transform(self, ego_pos: np.ndarray, true_track: np.ndarray) -> np.ndarray:
        if self._last_valid is None:
            self.reset(true_track)
        if (
            self.spec.dropout_probability == 0.0
            and self.spec.delay_steps == 0
            and self.spec.range_noise_std_m == 0.0
            and self.spec.azimuth_noise_std_deg == 0.0
            and self.spec.altitude_noise_std_m == 0.0
            and self.spec.velocity_noise_std_mps == 0.0
        ):
            self._last_valid = np.asarray(true_track, dtype=np.float64).copy()
            return self._last_valid.copy()

        measured = self._measure_noisy_track(ego_pos, true_track)
        self.transform_count += 1
        if self.rng.random() < self.spec.dropout_probability:
            self.dropout_count += 1
            current_valid = self._last_valid.copy()
        else:
            self._last_valid = measured
            current_valid = measured.copy()

        if self.spec.delay_steps > 0:
            self._delay_buffer.append(current_valid)
            return self._delay_buffer.popleft().copy()
        return current_valid

    @property
    def actual_dropout_rate(self) -> float:
        return self.dropout_count / self.transform_count if self.transform_count else 0.0


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None
        # 会执行BaseEnv的__init__方法，包括load_task(),load_simulator(),seed()

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'tactical_hierarchical_singlecombat':
            self.task = TacticalHierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'tactical_hierarchical_singlecombat_dodge_missile':
            self.task = TacticalHierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'tactical_hierarchical_singlecombat_shoot':
            self.task = TacticalHierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        elif taskname == 'HumanSingleCombat_shoot':
            self.task = HumanSingleCombat_shoot_Task(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side 换边训练
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()


class SingleCombatRadarDegradationEnv(SingleCombatEnv):
    """在 1v1 空战训练与验证中对敌机雷达航迹施加加噪、脱锁丢帧保持与滤波时延，同时作用于观测与战术导引解算。"""

    def __init__(
        self, config_name: str, *, radar_level=None, agent_radar_levels=None, sensor_seed=0,
        shared_episode_level=False,
    ):
        self.radar_level = radar_level
        self.agent_radar_levels = dict(agent_radar_levels) if agent_radar_levels is not None else None
        self.sensor_seed = sensor_seed
        self._radar_episode_seeds = np.random.SeedSequence()
        self._degraders = {}
        self._latest_radar_tracks = {}
        self._radar_track_steps = {}
        super().__init__(config_name)
        self._share_episode_level = (
            shared_episode_level
            and self.config.task == "tactical_hierarchical_singlecombat"
            and not self.task.use_artillery
        )

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self._radar_episode_seeds = np.random.SeedSequence([seeds[0], self.sensor_seed], spawn_key=(3,))
        return seeds

    def _choose_agent_level(self, agent_id: str, rng: np.random.Generator) -> int:
        if self.agent_radar_levels is not None and agent_id in self.agent_radar_levels:
            return int(self.agent_radar_levels[agent_id])
        if self.radar_level is not None:
            return int(self.radar_level)
        return int(rng.choice(len(RADAR_TRAIN_PROBABILITIES), p=RADAR_TRAIN_PROBABILITIES))

    def reset(self) -> np.ndarray:
        episode_rng = np.random.default_rng(self._radar_episode_seeds.spawn(1)[0])
        self._degraders = {}
        self._latest_radar_tracks = {}
        self._radar_track_steps = {}
        self._agent_levels = {}
        episode_level = None
        if self._share_episode_level and self.radar_level is None and self.agent_radar_levels is None:
            # 每局只抽一次档位，双方的传感器随机流仍分别初始化。
            episode_level = int(episode_rng.choice(len(RADAR_TRAIN_PROBABILITIES), p=RADAR_TRAIN_PROBABILITIES))
        for agent_id, child_seed in zip(self.agents.keys(), self._radar_episode_seeds.spawn(len(self.agents))):
            level = episode_level if episode_level is not None else self._choose_agent_level(agent_id, episode_rng)
            self._agent_levels[agent_id] = level
            self._degraders[agent_id] = RadarTrackDegrader(
                RADAR_DEGRADATION_SPECS[level], np.random.default_rng(child_seed)
            )
        return super().reset()

    def get_enemy_radar_track(self, agent_id: str) -> np.ndarray | None:
        if self._agent_levels.get(agent_id, 0) == 0:
            return None
        return self._latest_radar_tracks.get(agent_id)

    def get_obs(self):
        for agent_id, sim in self.agents.items():
            if self._radar_track_steps.get(agent_id) != self.current_step:
                self._radar_track_steps[agent_id] = self.current_step
                if self._agent_levels.get(agent_id, 0) > 0:
                    true_track = extract_true_enemy_track(self, agent_id)
                    self._latest_radar_tracks[agent_id] = self._degraders[agent_id].transform(
                        sim.get_position(), true_track
                    )
        return super().get_obs()
