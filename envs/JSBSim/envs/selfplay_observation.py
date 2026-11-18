from collections import deque
from dataclasses import dataclass

import numpy as np

from .singlecombat_env import SingleCombatEnv


SELFPLAY_SCENARIO = "1v1/NoWeapon/Selfplay"
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


def validate_unarmed_selfplay(config):
    if config.task != "singlecombat" or getattr(config, "use_artillery", False):
        raise ValueError("本机状态退化要求普通无武器 Selfplay 场景")


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
    """普通无武器 1v1 的本机状态混合观测退化。"""

    def __init__(self, config_name: str, *, degradation_level: int | None = None):
        if config_name != SELFPLAY_SCENARIO:
            raise ValueError("本机状态退化仅支持 1v1/NoWeapon/Selfplay")
        self.degradation_level = degradation_level
        self._episode_level = 0
        self._degraders = {}
        self._sensor_rngs = {}
        self._obs_step = None
        self._cached_obs = None
        super().__init__(config_name)

    def load_task(self):
        validate_unarmed_selfplay(self.config)
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
