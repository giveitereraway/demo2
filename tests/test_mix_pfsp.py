import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithms.utils.selfplay import MixPFSP
from config import get_config
from envs.JSBSim.envs.selfplay_score_env import SelfplayScoreEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner
from scripts.train.train_jsbsim import parse_args


def test_probabilities_forgetting_and_restore():
    sampler = MixPFSP(7, window=20, min_games=20)
    pool = dict.fromkeys(["0", "1", "2", "3"], 1000)
    for score in [1.0] * 20 + [0.0] * 20:
        sampler.update("0", score, 0)
    ids, probabilities = sampler.probabilities(pool)
    assert ids == list(pool)
    assert probabilities.sum() == pytest.approx(1)
    assert probabilities.min() >= 0.2 / 4
    assert probabilities[0] > probabilities[1]
    assert probabilities[3] - probabilities[2] == pytest.approx(0.1)
    restored = MixPFSP(7, window=20, min_games=20)
    restored.load_state_dict(copy.deepcopy(sampler.state_dict()))
    assert [sampler.choose(pool) for _ in range(30)] == [restored.choose(pool) for _ in range(30)]
    _, stale = sampler.probabilities(pool, version=51)
    assert stale == pytest.approx([0.225, 0.225, 0.225, 0.325])
    # 未形成历史有效窗口的低分对手没有遗忘证据。
    fresh = MixPFSP(7)
    fresh.update("0", 0, 0)
    assert fresh.stats["0"]["best"] is None
    pool = dict.fromkeys(map(str, range(100)), 1000)
    assert len(sampler.candidates(pool)) == 16
    assert sampler.candidates(pool)[-1] == "99"


def fake_aircraft(position, alive=True):
    return SimpleNamespace(is_alive=alive, get_position=lambda: np.asarray(position, dtype=float),
                           get_velocity=lambda: np.array([200.0, 0, 0]))


@pytest.mark.parametrize("artillery,alive,expected", [
    (True, [True, True], 0.5), (False, [True, True], 0.75),
    (False, [False, True], 0.0), (True, [True, False], 1.0),
    (False, [False, False], 0.5),
])
def test_score_before_vector_reset(monkeypatch, artillery, alive, expected):
    env = SelfplayScoreEnv.__new__(SelfplayScoreEnv)
    env.ego_ids, env.enm_ids = ["a"], ["b"]
    env._jsbsims = {"a": fake_aircraft([0, 0, 0], alive[0]),
                   "b": fake_aircraft([1000, 0, 0], alive[1])}
    env.task = SimpleNamespace(use_artillery=artillery)
    env.advantage_steps, env.score_steps = np.zeros(2, dtype=int), 1
    monkeypatch.setattr(SingleCombatEnv, "step", lambda self, action:
                        (None, None, np.ones((2, 1), dtype=bool), {}))
    result = env.step(None)[3]["selfplay_result"]
    assert result["score"] == pytest.approx(expected)
    assert result["alive"] == alive


def short_env(scenario):
    env = SelfplayScoreEnv(scenario)
    for condition in env.task.termination_conditions:
        if hasattr(condition, "max_steps"):
            condition.max_steps = 2
    return env


@pytest.mark.parametrize("scenario,vector_type", [
    ("1v1/NoWeapon/Selfplay", SubprocVecEnv),
    ("1v1/NoWeapon/TacticalHierarchySelfplay", DummyVecEnv),
])
def test_training_evaluation_and_resume(tmp_path, scenario, vector_type):
    torch.set_num_threads(1)
    args = parse_args([
        "--env-name", "SingleCombat", "--scenario-name", scenario,
        "--use-selfplay", "--selfplay-algorithm", "mix_pfsp", "--use-eval",
        "--n-rollout-threads", "2", "--n-choose-opponents", "2",
        "--n-eval-rollout-threads", "3", "--eval-episodes", "5", "--eval-interval", "1",
        "--save-interval", "1", "--num-env-steps", "24", "--buffer-size", "4",
        "--num-mini-batch", "1", "--ppo-epoch", "1", "--data-chunk-length", "2",
        "--hidden-size", "16 16", "--act-hidden-size", "16 16", "--recurrent-hidden-size", "16",
    ], get_config())
    envs = vector_type([lambda: short_env(scenario) for _ in range(2)])
    eval_envs = DummyVecEnv([lambda: short_env(scenario) for _ in range(3)])
    config = {"all_args": args, "envs": envs, "eval_envs": eval_envs, "device": torch.device("cpu"),
              "render_mode": "txt", "run_dir": tmp_path / "first"}
    try:
        runner = SelfplayJSBSimRunner(config)
        runner.run()
        assert sum(stat["count"] for stat in runner.selfplay_algo.stats.values()) == 12
        assert runner.latest_metrics["eval_episodes_completed"] == 5
        assert len(runner.policy_pool) == 4
        frozen = {key: torch.load(tmp_path / "first" / f"actor_{key}.pt", weights_only=True)
                  for key in runner.policy_pool}
        state = torch.load(tmp_path / "first" / "mix_pfsp_state.pt", weights_only=True)
        args.model_dir = str(tmp_path / "first")
        args.use_eval = False
        args.num_env_steps = 8
        config["run_dir"] = tmp_path / "resumed"
        resumed = SelfplayJSBSimRunner(config)
        assert resumed.selfplay_algo.state_dict()["stats"] == state["sampler"]["stats"]
        resumed.run()
        assert resumed.mix_version == 4
        assert sum(stat["count"] for stat in resumed.selfplay_algo.stats.values()) == 16
        assert len(resumed.policy_pool) == 5
        for key, weights in frozen.items():
            restored = torch.load(tmp_path / "resumed" / f"actor_{key}.pt", weights_only=True)
            assert all(torch.equal(weights[name], restored[name]) for name in weights)
    finally:
        envs.close()
        eval_envs.close()


def test_cli_rejects_unsupported_mode():
    with pytest.raises(SystemExit):
        parse_args(["--selfplay-algorithm", "mix_pfsp"], get_config())
    with pytest.raises(ValueError):
        MixPFSP(0, weights=[1, 1, 0, 0])
