import numpy as np
from typing import Dict, List
from abc import ABC, abstractstaticmethod


def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP
    elif algo_name == 'fsp':
        return FSP
    elif algo_name == 'pfsp':
        return PFSP
    elif algo_name == 'mix_pfsp':
        return MixPFSP
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))


class SelfplayAlgorithm(ABC):

    @abstractstaticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        pass

    @abstractstaticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class SP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return list(agents_elo.keys())[-1]

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class FSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return np.random.choice(list(agents_elo.keys()))

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class PFSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
        history_elo = np.array(list(agents_elo.values()))
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        """ meta-solver """
        k = float(len(sample_probs) + 1)
        meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))
        opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
        return opponent_idx

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
        pass


class MixPFSP:
    """以近期得分、历史回落、均匀覆盖和最新快照混合抽样。"""

    def __init__(self, seed, window=50, pool_size=16, weights=(0.5, 0.2, 0.2, 0.1),
                 min_games=20, delta=0.1, max_age=50):
        self.weights = np.asarray(weights, dtype=float)
        if (self.weights.shape != (4,) or not np.isfinite(self.weights).all()
                or (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1)):
            raise ValueError("Mix-PFSP 的四项权重必须非负、有限且和为 1")
        if window < 1 or pool_size < 1 or not 1 <= min_games <= window or max_age < 1:
            raise ValueError("窗口、候选池、统计龄期须为正数，最低局数须在窗口范围内")
        if not np.isfinite(delta) or not 0 <= delta <= 1:
            raise ValueError("遗忘阈值须在 [0, 1] 内")
        self.window = window
        self.pool_size = pool_size
        self.min_games = min_games
        self.delta = delta
        self.max_age = max_age
        self.rng = np.random.default_rng(seed)
        self.stats = {}

    def candidates(self, pool):
        ids = list(pool)
        if not ids:
            raise ValueError("Mix-PFSP 对手池为空")
        if len(ids) <= self.pool_size:
            return ids
        recent_count = (self.pool_size + 1) // 2
        history_count = self.pool_size - recent_count
        history = ids[:-recent_count]
        indices = np.linspace(0, len(history) - 1, history_count, dtype=int)
        return [history[index] for index in indices] + ids[-recent_count:]

    def update(self, opponent_id, score, version):
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("单局得分须在 [0, 1] 内")
        stat = self.stats.setdefault(opponent_id, {
            "results": [], "count": 0, "best": None, "last_version": version})
        # 长期未交手的窗口重新收集，历史最佳值继续用于检测回落。
        if version - stat["last_version"] > self.max_age:
            stat["results"] = []
        stat["results"] = (stat["results"] + [float(score)])[-self.window:]
        stat["count"] += 1
        stat["last_version"] = int(version)
        score_rate = (1 + sum(stat["results"])) / (2 + len(stat["results"]))
        if len(stat["results"]) >= self.min_games:
            stat["best"] = max(score_rate, stat["best"] or 0.0)

    def probabilities(self, pool, version=0):
        ids = self.candidates(pool)
        scores = np.full(len(ids), 0.5)
        forgotten = np.zeros(len(ids))
        for index, opponent_id in enumerate(ids):
            stat = self.stats.get(opponent_id)
            if stat is None or version - stat["last_version"] > self.max_age:
                continue
            scores[index] = (1 + sum(stat["results"])) / (2 + len(stat["results"]))
            if len(stat["results"]) >= self.min_games and stat["best"] is not None:
                forgotten[index] = max(stat["best"] - scores[index] - self.delta, 0)
        uniform = np.full(len(ids), 1 / len(ids))
        curriculum = 4 * scores * (1 - scores)
        curriculum = curriculum / curriculum.sum() if curriculum.sum() else uniform
        forgotten = forgotten / forgotten.sum() if forgotten.sum() else uniform
        latest = np.zeros(len(ids))
        latest[-1] = 1
        probabilities = self.weights @ np.stack((curriculum, forgotten, uniform, latest))
        return ids, probabilities / probabilities.sum()

    def choose(self, pool, version=0):
        ids, probabilities = self.probabilities(pool, version)
        return str(self.rng.choice(ids, p=probabilities))

    def state_dict(self):
        return {
            "config": {"window": self.window, "pool_size": self.pool_size,
                       "weights": self.weights.tolist(), "min_games": self.min_games,
                       "delta": self.delta, "max_age": self.max_age},
            "stats": self.stats, "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["config"] != self.state_dict()["config"]:
            raise ValueError("恢复 Mix-PFSP 时采样参数必须与保存时一致")
        self.stats = state["stats"]
        self.rng.bit_generator.state = state["rng"]
