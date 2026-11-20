import numpy as np

from .singlecombat_env import SingleCombatEnv
from ..utils.utils import get_AO_TA_R


class SelfplayScoreEnv(SingleCombatEnv):
    """在向量环境自动重置前记录单局采样得分。"""

    def reset(self):
        self.advantage_steps = np.zeros(2, dtype=np.int64)
        self.score_steps = 0
        return super().reset()

    def step(self, action):
        obs, rewards, dones, info = super().step(action)
        aircraft = [self.agents[uid] for uid in self.ego_ids + self.enm_ids]
        features = [np.concatenate((sim.get_position(), sim.get_velocity())) for sim in aircraft]
        for index in range(2):
            ao, ta, distance = get_AO_TA_R(features[index], features[1 - index])
            self.advantage_steps[index] += (
                ao <= np.deg2rad(30) and ta <= np.deg2rad(30) and distance <= 3000)
        self.score_steps += 1
        if np.all(dones):
            alive = [sim.is_alive for sim in aircraft]
            occupancy = self.advantage_steps / self.score_steps
            if alive[0] != alive[1]:
                score = float(alive[0])
            elif self.task.use_artillery or not all(alive):
                score = 0.5
            else:
                score = float(0.5 + (occupancy[0] - occupancy[1]) / 2)
            info["selfplay_result"] = {
                "score": score,
                "advantage_occupancy": occupancy.tolist(),
                "alive": alive,
                "use_artillery": self.task.use_artillery,
                "steps": self.score_steps,
            }
        return obs, rewards, dones, info
