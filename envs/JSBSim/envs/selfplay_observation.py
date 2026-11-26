from collections import deque
from dataclasses import dataclass

import numpy as np

from .singlecombat_env import (
    RADAR_DEGRADATION_SPECS,
    RadarTrackDegrader,
    SingleCombatEnv,
    extract_true_enemy_track,
)


SELFPLAY_SCENARIO = "1v1/NoWeapon/Selfplay"
TACTICAL_SELFPLAY_SCENARIO = "1v1/NoWeapon/TacticalHierarchySelfplay"
UNARMED_SELFPLAY_TASKS = {
    SELFPLAY_SCENARIO: "singlecombat",
    TACTICAL_SELFPLAY_SCENARIO: "tactical_hierarchical_singlecombat",
}
OWN_SHIP_STATE = slice(0, 9)


@dataclass(frozen=True)
class OwnshipDegradationSpec:
    """合成观测退化的归一化噪声、丢帧概率和时延步数。"""

    noise_std: float
    dropout_probability: float
    delay_steps: int


OWN_SHIP_DEGRADATION_SPECS = (
    OwnshipDegradationSpec(noise_std=0.0, dropout_probability=0.0, delay_steps=0),
    OwnshipDegradationSpec(noise_std=0.01, dropout_probability=0.10, delay_steps=1),
    OwnshipDegradationSpec(noise_std=0.03, dropout_probability=0.20, delay_steps=2),
    # E3–E6 沿用 Heading 的丢帧率和时延阶梯，噪声使用归一化观测单位。
    OwnshipDegradationSpec(noise_std=0.05, dropout_probability=0.30, delay_steps=3),
    OwnshipDegradationSpec(noise_std=0.07, dropout_probability=0.40, delay_steps=4),
    OwnshipDegradationSpec(noise_std=0.09, dropout_probability=0.50, delay_steps=5),
    OwnshipDegradationSpec(noise_std=0.11, dropout_probability=0.60, delay_steps=6),
)
OWN_SHIP_TRAIN_PROBABILITIES = (0.30, 0.35, 0.35)
OWN_SHIP_EVAL_LEVEL = 2
JOINT_DEGRADATION_MAX_LEVEL = min(len(OWN_SHIP_DEGRADATION_SPECS), len(RADAR_DEGRADATION_SPECS)) - 1
JOINT_TRAIN_PROBABILITIES = OWN_SHIP_TRAIN_PROBABILITIES
JOINT_EVAL_LEVEL = OWN_SHIP_EVAL_LEVEL


def validate_unarmed_selfplay(config, config_name: str = SELFPLAY_SCENARIO):
    if config.task != UNARMED_SELFPLAY_TASKS[config_name] or getattr(config, "use_artillery", False):
        raise ValueError(f"观测退化要求场景 {config_name} 使用对应的无武器自博弈任务并关闭火炮")


class OwnshipObservationDegrader:
    """只扰动本机状态的归一化输入，供训练和离线评估共用。"""

    def __init__(self, level: int, initial_obs: np.ndarray, rng: np.random.Generator):
        self.spec = OWN_SHIP_DEGRADATION_SPECS[level]
        self.rng = rng
        self.last_valid = initial_obs[OWN_SHIP_STATE].copy()
        self.delay = deque([self.last_valid] * self.spec.delay_steps)
        self.steps = 0
        self.dropped = 0

    @property
    def actual_dropout_rate(self) -> float:
        return self.dropped / max(self.steps, 1)

    def transform(self, obs: np.ndarray) -> np.ndarray:
        degraded = obs.copy()
        if self.spec == OWN_SHIP_DEGRADATION_SPECS[0]:
            return degraded

        self.steps += 1
        if self.rng.random() < self.spec.dropout_probability:
            self.dropped += 1
        else:
            ownship = obs[OWN_SHIP_STATE]
            noise = self.rng.normal(0.0, self.spec.noise_std, size=ownship.shape)
            self.last_valid = np.clip(ownship + noise, -10.0, 10.0)

        self.delay.append(self.last_valid)
        degraded[OWN_SHIP_STATE] = self.delay.popleft()
        return degraded


class SelfplayOwnshipDegradationEnv(SingleCombatEnv):
    """无武器 1v1 的本机状态混合观测退化，作用于环境返回的策略观测。"""

    def __init__(self, config_name: str, *, degradation_level: int | None = None):
        if config_name not in UNARMED_SELFPLAY_TASKS:
            raise ValueError("本机状态退化仅支持无武器 Selfplay 和 TacticalHierarchySelfplay")
        self._config_name = config_name
        self.degradation_level = degradation_level
        self._episode_level = 0
        self._degraders = {}
        self._sensor_rngs = {}
        self._obs_step = None
        self._cached_obs = None
        super().__init__(config_name)

    def load_task(self):
        validate_unarmed_selfplay(self.config, self._config_name)
        super().load_task()

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self._episode_seeds = np.random.SeedSequence(seeds[0], spawn_key=(4,))
        return seeds

    def reset(self) -> np.ndarray:
        level_seed, *sensor_seeds = self._episode_seeds.spawn(1 + len(self.agents))
        self._episode_level = self.degradation_level
        if self._episode_level is None:
            self._episode_level = int(np.random.default_rng(level_seed).choice(
                len(OWN_SHIP_TRAIN_PROBABILITIES), p=OWN_SHIP_TRAIN_PROBABILITIES,
            ))
        self._sensor_rngs = {
            agent_id: np.random.default_rng(seed)
            for agent_id, seed in zip(self.agents, sensor_seeds)
        }
        self._degraders.clear()
        self._obs_step = None
        self._cached_obs = None
        return super().reset()

    def get_obs(self):
        if self._cached_obs is None or self._obs_step != self.current_step:
            self._cached_obs = super().get_obs()
            self._obs_step = self.current_step
            if self._episode_level:
                for agent_id, obs in self._cached_obs.items():
                    if agent_id not in self._degraders:
                        self._degraders[agent_id] = OwnshipObservationDegrader(
                            self._episode_level, obs, self._sensor_rngs[agent_id],
                        )
                    self._cached_obs[agent_id] = self._degraders[agent_id].transform(obs)
        return {agent_id: obs.copy() for agent_id, obs in self._cached_obs.items()}


class SelfplayJointDegradationEnv(SingleCombatEnv):
    """无武器 1v1 的联合观测退化，本机状态和敌机雷达航迹使用同一退化档位。"""

    def __init__(self, config_name: str, *, degradation_level: int | None = None):
        if config_name not in UNARMED_SELFPLAY_TASKS:
            raise ValueError("联合观测退化仅支持无武器 Selfplay 和 TacticalHierarchySelfplay")
        if degradation_level is not None and not 0 <= degradation_level <= JOINT_DEGRADATION_MAX_LEVEL:
            raise ValueError(f"联合观测退化档位必须在 [0, {JOINT_DEGRADATION_MAX_LEVEL}] 范围内")
        self._config_name = config_name
        self.degradation_level = degradation_level
        self._episode_level = 0
        self._ownship_degraders = {}
        self._ownship_rngs = {}
        self._radar_degraders = {}
        self._latest_radar_tracks = {}
        self._obs_step = None
        self._cached_obs = None
        super().__init__(config_name)

    def load_task(self):
        validate_unarmed_selfplay(self.config, self._config_name)
        super().load_task()

    def seed(self, seed=None):
        seeds = super().seed(seed)
        self._episode_seeds = np.random.SeedSequence(seeds[0], spawn_key=(5,))
        return seeds

    def reset(self) -> np.ndarray:
        num_agents = len(self.agents)
        level_seed, *sensor_seeds = self._episode_seeds.spawn(1 + 2 * num_agents)
        self._episode_level = self.degradation_level
        if self._episode_level is None:
            # 每局红蓝共享同一综合档位，避免不对称退化引入对局方差。
            self._episode_level = int(np.random.default_rng(level_seed).choice(
                len(JOINT_TRAIN_PROBABILITIES), p=JOINT_TRAIN_PROBABILITIES,
            ))
        ownship_seeds = sensor_seeds[:num_agents]
        radar_seeds = sensor_seeds[num_agents:]
        self._ownship_rngs = {
            agent_id: np.random.default_rng(seed)
            for agent_id, seed in zip(self.agents, ownship_seeds)
        }
        radar_spec = RADAR_DEGRADATION_SPECS[self._episode_level]
        self._radar_degraders = {
            agent_id: RadarTrackDegrader(radar_spec, np.random.default_rng(seed))
            for agent_id, seed in zip(self.agents, radar_seeds)
        }
        self._ownship_degraders.clear()
        self._latest_radar_tracks.clear()
        self._obs_step = None
        self._cached_obs = None
        return super().reset()

    def get_enemy_radar_track(self, agent_id: str) -> np.ndarray | None:
        if self._episode_level == 0:
            return None
        return self._latest_radar_tracks.get(agent_id)

    def get_obs(self):
        if self._cached_obs is None or self._obs_step != self.current_step:
            if self._episode_level:
                for agent_id, sim in self.agents.items():
                    true_track = extract_true_enemy_track(self, agent_id)
                    self._latest_radar_tracks[agent_id] = self._radar_degraders[agent_id].transform(
                        sim.get_position(), true_track,
                    )
            else:
                self._latest_radar_tracks.clear()

            # 先经任务读取退化后的敌机航迹计算后 6 维，再对前 9 维施加本机状态退化。
            self._cached_obs = super().get_obs()
            self._obs_step = self.current_step
            if self._episode_level:
                for agent_id, obs in self._cached_obs.items():
                    if agent_id not in self._ownship_degraders:
                        self._ownship_degraders[agent_id] = OwnshipObservationDegrader(
                            self._episode_level, obs, self._ownship_rngs[agent_id],
                        )
                    self._cached_obs[agent_id] = self._ownship_degraders[agent_id].transform(obs)
        return {agent_id: obs.copy() for agent_id, obs in self._cached_obs.items()}
