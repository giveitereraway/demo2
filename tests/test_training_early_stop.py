import importlib.util
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

    package_name = "_early_stop_test_runner"
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
    args.use_eval = args.use_early_stop = True
    args.use_selfplay = kind == "selfplay_jsbsim_runner"
    args.num_env_steps = 20
    args.buffer_size = args.n_rollout_threads = 1
    args.eval_interval = 2
    args.save_interval = 10
    args.log_interval = 100
    args.early_stop_patience = 2
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


def test_defaults_are_opt_in():
    args = get_config().parse_args([])
    assert not args.use_early_stop
    assert args.early_stop_min_steps is None
    assert args.early_stop_patience == 10


def test_minimum_steps_and_improvement_reset(runner_modules, tmp_path):
    runner = make_runner(runner_modules, tmp_path, early_stop_min_steps=100)
    assert not runner.check_early_stopping(-10, 10)
    assert not runner.check_early_stopping(-11, 50)
    assert not runner.check_early_stopping(-11, 99)
    assert runner.no_improvement_evals == 0
    assert not runner.check_early_stopping(-11, 100)
    assert runner.no_improvement_evals == 1
    runner.updates = 1
    assert not runner.check_early_stopping(-9, 101)
    assert runner.no_improvement_evals == 0
    assert not runner.check_early_stopping(-9, 102)
    assert runner.check_early_stopping(-10, 103)
    assert runner_modules.saved["actor_best.pt"] == {"actor_update": 1}
    assert runner_modules.saved["critic_best.pt"] == {"critic_update": 1}


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf])
def test_invalid_rewards_do_not_replace_best_or_advance_patience(runner_modules, tmp_path, reward):
    runner = make_runner(runner_modules, tmp_path, early_stop_min_steps=0)
    runner.check_early_stopping(10, 1)
    assert not runner.check_early_stopping(reward, 2)
    assert runner.best_eval_reward == 10
    assert runner.no_improvement_evals == 0


@pytest.mark.parametrize("options", [
    {"use_eval": False}, {"early_stop_min_steps": -1}, {"early_stop_patience": 0},
    {"eval_interval": 0}, {"eval_episodes": 0},
])
def test_invalid_configuration_fails(runner_modules, tmp_path, options):
    with pytest.raises(ValueError):
        make_runner(runner_modules, tmp_path, **options)


@pytest.mark.parametrize("kind,selfplay", [
    ("jsbsim_runner", False), ("selfplay_jsbsim_runner", True),
    ("share_jsbsim_runner", False), ("share_jsbsim_runner", True),
])
def test_early_stop_keeps_best_and_final_off_save_interval(runner_modules, tmp_path, kind, selfplay):
    runner = make_runner(runner_modules, tmp_path, kind, use_selfplay=selfplay)
    assert runner.early_stop_min_steps == 4
    runner.run()
    assert runner.total_num_steps == 7
    best_step = runner.evaluated_steps[0]
    assert runner_modules.saved["actor_best.pt"] == {"actor_update": best_step}
    assert runner_modules.saved["critic_best.pt"] == {"critic_update": best_step}
    assert runner_modules.saved["actor_latest.pt"] == {"actor_update": 7}
    assert runner_modules.saved["critic_latest.pt"] == {"critic_update": 7}
    if selfplay:
        assert "actor_6.pt" in runner_modules.saved


@pytest.mark.parametrize("kind", ["jsbsim_runner", "selfplay_jsbsim_runner", "share_jsbsim_runner"])
@pytest.mark.parametrize("budget", [1, 5])
def test_normal_end_evaluates_and_saves_final_model(runner_modules, tmp_path, kind, budget):
    runner = make_runner(runner_modules, tmp_path, kind, score=lambda item: item.total_num_steps,
                         num_env_steps=budget, eval_interval=10, use_early_stop=False)
    runner.run()
    assert runner.evaluated_steps[-1] == budget
    assert runner_modules.saved["actor_best.pt"] == {"actor_update": budget}
    assert runner_modules.saved["critic_best.pt"] == {"critic_update": budget}
    assert runner_modules.saved["actor_latest.pt"] == {"actor_update": budget}
    assert runner_modules.saved["critic_latest.pt"] == {"critic_update": budget}


def test_no_evaluation_preserves_training_budget(runner_modules, tmp_path):
    runner = make_runner(runner_modules, tmp_path, use_eval=False, use_early_stop=False)
    runner.run()
    assert runner.total_num_steps == 20
    assert not runner.evaluated_steps
    assert "actor_best.pt" not in runner_modules.saved
    assert runner_modules.saved["actor_latest.pt"] == {"actor_update": 20}
