import importlib.util
import json
import sys
from itertools import count
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from config import get_config


@pytest.fixture
def runner_modules(monkeypatch):
    """隔离可选依赖，用轻量环境和策略验证真实 Runner 控制流，不运行飞行仿真。"""
    saved = {}
    torch = ModuleType("torch")
    torch.no_grad = lambda: lambda method: method
    torch.save = lambda state, path: saved.__setitem__(Path(path).name, state.copy())
    buffer = ModuleType("algorithms.utils.buffer")
    buffer.ReplayBuffer = buffer.SharedReplayBuffer = object
    for name, module in [("torch", torch), ("wandb", ModuleType("wandb")),
                         ("algorithms.utils.buffer", buffer)]:
        monkeypatch.setitem(sys.modules, name, module)

    package_name = "_eval_checkpoint_test_runner"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "runner")]
    monkeypatch.setitem(sys.modules, package_name, package)
    modules = {}
    for name in ["base_runner", "jsbsim_runner", "selfplay_jsbsim_runner", "share_jsbsim_runner"]:
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, Path(package.__path__[0]) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    ticks = count(1)
    for name in ["jsbsim_runner", "share_jsbsim_runner"]:
        monkeypatch.setattr(modules[name], "time", SimpleNamespace(time=lambda: next(ticks)))
    return SimpleNamespace(**modules, saved=saved)


def make_runner(modules, tmp_path, kind="jsbsim_runner", score=None, **overrides):
    args = get_config().parse_args([])
    args.scenario_name = "test"
    args.use_eval = True
    args.use_selfplay = kind == "selfplay_jsbsim_runner"
    args.num_env_steps = 20
    args.buffer_size = args.n_rollout_threads = 1
    args.eval_interval = 2
    args.save_interval = 10
    args.log_interval = 100
    for key, value in overrides.items():
        setattr(args, key, value)
    shared = kind == "share_jsbsim_runner"
    runner_type = getattr(modules, kind).__dict__[
        {"jsbsim_runner": "JSBSimRunner", "selfplay_jsbsim_runner": "SelfplayJSBSimRunner",
         "share_jsbsim_runner": "ShareJSBSimRunner"}[kind]]

    class Harness(runner_type):
        def load(self):
            self.updates = 0
            self.evaluated_steps = []
            self.use_selfplay = args.use_selfplay
            self.latest_elo = args.init_elo
            self.policy_pool = {}
            self.policy = SimpleNamespace(
                actor=SimpleNamespace(state_dict=lambda: {"actor_update": self.updates}),
                critic=SimpleNamespace(state_dict=lambda: {"critic_update": self.updates}))
            self.buffer = SimpleNamespace(rewards=np.ones((1, 1, 1, 1)), masks=np.zeros((1, 1, 1, 1)))

        def warmup(self):
            pass

        def collect(self, step):
            return (None,) * 5

        def insert(self, data):
            pass

        def compute(self):
            pass

        def train(self):
            self.updates += 1
            return {}

        def eval(self, total_num_steps):
            if self.use_selfplay:
                assert self.policy_pool
            self.evaluated_steps.append(total_num_steps)
            return score(self) if score else 10 - len(self.evaluated_steps)

    envs = SimpleNamespace(step=lambda actions: (None, None, None, None, []) if shared
                           else (None, None, None, []))
    return Harness({"all_args": args, "envs": envs, "eval_envs": object() if args.use_eval else None,
                    "device": "cpu", "render_mode": "txt", "run_dir": tmp_path})


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf])
def test_invalid_reward_breaks_streak(runner_modules, tmp_path, reward):
    runner = make_runner(runner_modules, tmp_path, "selfplay_jsbsim_runner")
    runner.update_eval_checkpoint(10, 0)
    runner.update_eval_checkpoint(9, 1)
    assert runner.no_improvement_evals == 1
    runner.update_eval_checkpoint(reward, 2)
    assert runner.best_eval_reward == 10
    assert runner.no_improvement_evals == 0


@pytest.mark.parametrize("options", [{"eval_interval": 0}, {"eval_episodes": 0}])
def test_invalid_configuration_fails(runner_modules, tmp_path, options):
    with pytest.raises(ValueError):
        make_runner(runner_modules, tmp_path, **options)


@pytest.mark.parametrize("kind,selfplay", [
    ("jsbsim_runner", False),
    ("selfplay_jsbsim_runner", True),
    ("share_jsbsim_runner", False),
    ("share_jsbsim_runner", True),
])
def test_first_convergence_does_not_stop_training(runner_modules, tmp_path, kind, selfplay):
    def score(runner):
        return 20 if runner.updates == 12 else 10

    runner = make_runner(runner_modules, tmp_path, kind, use_selfplay=selfplay,
                         num_env_steps=24, eval_interval=1, score=score)
    runner.run()
    saved = runner_modules.saved
    assert runner.total_num_steps == 24
    assert saved["actor_latest.pt"] == {"actor_update": 24}
    assert saved["actor_best.pt"] == {"actor_update": 12}

    record_path = tmp_path / "selfplay_convergence.json"
    if kind == "selfplay_jsbsim_runner":
        # 第 12 轮创新高，随后第 22 轮首次连续十次无提升。
        assert saved["actor_converged.pt"] == {"actor_update": 22}
        assert saved["critic_converged.pt"] == {"critic_update": 22}
        assert json.loads(record_path.read_text(encoding="utf-8")) == {
            "training_round": 22,
            "total_num_steps": 22,
            "eval_average_episode_rewards": 10,
            "best_eval_average_episode_rewards": 20,
            "no_improvement_evals": 10,
        }
    else:
        assert not record_path.exists()
        assert "actor_converged.pt" not in saved


def test_later_improvement_updates_best_but_keeps_first_convergence(runner_modules, tmp_path):
    runner = make_runner(runner_modules, tmp_path, "selfplay_jsbsim_runner",
                         num_env_steps=30, eval_interval=1,
                         score=lambda item: 20 if item.updates == 15 else 10)
    runner.run()
    assert runner_modules.saved["actor_converged.pt"] == {"actor_update": 12}
    assert runner_modules.saved["actor_best.pt"] == {"actor_update": 15}
    record = json.loads((tmp_path / "selfplay_convergence.json").read_text(encoding="utf-8"))
    assert record["training_round"] == 12
    assert record["best_eval_average_episode_rewards"] == 10


@pytest.mark.parametrize("kind", ["jsbsim_runner", "selfplay_jsbsim_runner", "share_jsbsim_runner"])
@pytest.mark.parametrize("budget", [1, 5])
def test_normal_end_evaluates_and_saves_final_model(runner_modules, tmp_path, kind, budget):
    runner = make_runner(runner_modules, tmp_path, kind, score=lambda item: item.total_num_steps,
                         num_env_steps=budget, eval_interval=10)
    runner.run()
    assert runner.evaluated_steps[-1] == budget
    assert runner_modules.saved["actor_best.pt"] == {"actor_update": budget}
    assert runner_modules.saved["critic_best.pt"] == {"critic_update": budget}
    assert runner_modules.saved["actor_latest.pt"] == {"actor_update": budget}
    assert runner_modules.saved["critic_latest.pt"] == {"critic_update": budget}


def test_no_evaluation_preserves_training_budget(runner_modules, tmp_path):
    runner = make_runner(runner_modules, tmp_path, use_eval=False)
    runner.run()
    assert runner.total_num_steps == 20
    assert not runner.evaluated_steps
    assert "actor_best.pt" not in runner_modules.saved
    assert runner_modules.saved["actor_latest.pt"] == {"actor_update": 20}
