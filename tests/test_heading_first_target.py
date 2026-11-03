import json

import pytest
import torch

from envs.JSBSim.envs.singlecontrol_env import HeadingTrackingEvalEnv, SingleControlEnv
from envs.env_wrappers import DummyVecEnv
from runner.jsbsim_runner import JSBSimRunner
from scripts.train.train_jsbsim import get_config, parse_args


@pytest.mark.parametrize("policy_loss", ["clip", "spo"])
def test_heading_saves_first_target_and_finishes_training(tmp_path, policy_loss):
    args = parse_args([
        "--env-name", "SingleControl", "--scenario-name", "1/heading",
        "--policy-loss", policy_loss, "--use-eval", "--heading-first-target",
        "--heading-error-limits", "180", "100000", "1000",
        "--eval-interval", "1", "--eval-episodes", "3", "--n-eval-rollout-threads", "2",
        "--n-rollout-threads", "1", "--buffer-size", "4", "--data-chunk-length", "2",
        "--num-env-steps", "8", "--ppo-epoch", "1",
    ], get_config())

    def make_eval_env(max_steps):
        def create():
            env = HeadingTrackingEvalEnv("1/heading")
            env.config.max_steps = max_steps
            env.task.termination_conditions[-1].max_steps = max_steps
            return env
        return create

    envs = DummyVecEnv([lambda: SingleControlEnv("1/heading")])
    eval_envs = DummyVecEnv([make_eval_env(2), make_eval_env(3)])
    try:
        runner = JSBSimRunner(dict(all_args=args, envs=envs, eval_envs=eval_envs,
                                   device=torch.device("cpu"), render_mode="txt", run_dir=tmp_path))
        runner.record_heading_first_target(0, 2)
        record_path = tmp_path / "heading_first_target.json"
        assert json.loads(record_path.read_text())["first_env_steps"] is None
        assert not (tmp_path / "actor_first_tracking_target.pt").exists()

        runner.run()
        record = json.loads(record_path.read_text())
        assert record["first_env_steps"] == 4
        assert record["passed_episodes"] == 3
        assert record["policy_loss"] == policy_loss
        assert runner.total_num_steps == 8
        first = torch.load(tmp_path / "actor_first_tracking_target.pt", weights_only=True)
        latest = torch.load(tmp_path / "actor_latest.pt", weights_only=True)
        assert any(not torch.equal(first[name], latest[name]) for name in first)
        assert (tmp_path / "critic_first_tracking_target.pt").exists()
    finally:
        eval_envs.close()
        envs.close()
