import json

import numpy as np
import pytest
import torch

from algorithms.utils.lstm import LSTMLayer
from experiments.evaluate_lowlevel_controller import build_actor, predict_action
from runner.heading_offpolicy_runner import HeadingOffPolicyRunner
from runner.jsbsim_runner import JSBSimRunner
from scripts.train.train_jsbsim import get_config, make_eval_env, make_train_env, parse_args


def test_lstm_chunks_match_stepwise_rollout_with_resets():
    torch.manual_seed(7)
    layer = LSTMLayer(input_size=3, hidden_size=4, num_layers=2)
    obs = torch.randn(4, 2, 3)
    initial = torch.randn(2, 2, 8)
    masks = torch.tensor([[[1.0], [0.0]], [[1.0], [1.0]], [[0.0], [1.0]], [[1.0], [1.0]]])

    sequence, final_state = layer(obs.flatten(0, 1), initial, masks.flatten(0, 1))
    step_outputs, state = [], initial
    for step_obs, step_mask in zip(obs, masks):
        output, state = layer(step_obs, state, step_mask)
        step_outputs.append(output)
    torch.testing.assert_close(sequence, torch.cat(step_outputs))
    torch.testing.assert_close(final_state, state)

    # 两路记忆都必须在回合边界清零。
    _, reset_state = layer(obs[0], initial, torch.zeros_like(masks[0]))
    _, zero_state = layer(obs[0], torch.zeros_like(initial), torch.ones_like(masks[0]))
    torch.testing.assert_close(reset_state, zero_state)


@pytest.mark.parametrize("algorithm,encoder,policy_loss", [
    ("ppo", "lstm", "clip"), ("ppo", "igru", "clip"), ("ppo", "igru", "spo"),
    ("sac", "mlp", "clip"), ("bdq", "mlp", "clip"),
])
def test_heading_baseline_trains_evaluates_and_reloads(tmp_path, algorithm, encoder, policy_loss):
    torch.manual_seed(8)
    args = parse_args([
        "--env-name", "SingleControl", "--scenario-name", "1/heading",
        "--algorithm-name", algorithm, "--memory-encoder", encoder, "--heading-mixed-degradation",
        "--policy-loss", policy_loss,
        "--n-rollout-threads", "1", "--buffer-size", "4", "--num-env-steps", "8",
        "--data-chunk-length", "2", "--ppo-epoch", "1", "--hidden-size", "8", "--act-hidden-size", "8",
        "--recurrent-hidden-size", "8", "--offpolicy-warmup-steps", "0", "--offpolicy-batch-size", "2",
        "--offpolicy-train-freq", "2", "--offpolicy-replay-size", "6", "--use-eval",
        "--n-eval-rollout-threads", "1", "--eval-episodes", "1", "--eval-interval", "1",
        "--heading-first-target", "--heading-error-limits", "180", "100000", "1000",
    ], get_config())
    envs, eval_envs = make_train_env(args), make_eval_env(args)
    try:
        for env in (envs.envs[0], eval_envs.envs[0]):
            env.config.max_steps = 2
            env.task.termination_conditions[-1].max_steps = 2
        runner_type = JSBSimRunner if algorithm == "ppo" else HeadingOffPolicyRunner
        runner = runner_type(dict(all_args=args, envs=envs, eval_envs=eval_envs, device=torch.device("cpu"),
                                  render_mode="txt", run_dir=tmp_path))
        initial = {key: tensor.clone() for key, tensor in runner.policy.actor.state_dict().items()}
        runner.run()

        history = [json.loads(line) for line in (tmp_path / "heading-evaluation.jsonl").read_text().splitlines()]
        assert [row["step"] for row in history] == [4, 8]
        assert history[-1]["actor_parameters"] > 0
        assert history[-1]["training_elapsed_seconds"] >= history[0]["training_elapsed_seconds"] > 0
        assert history[-1]["heading_tracking_pass_rate"] == 1
        assert (tmp_path / "actor_best.pt").is_file()
        assert (tmp_path / "actor_first_tracking_target.pt").is_file()
        if algorithm != "ppo":
            assert runner.replay.size == 6
            assert runner.latest_metrics["gradient_updates"] == 4

        actor = build_actor(tmp_path / "actor_latest.pt", torch.device("cpu"))
        assert any(not torch.equal(initial[key], tensor) for key, tensor in actor.state_dict().items())
        obs = envs.reset()[0]
        state = np.zeros((1, *actor.state_shape), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            actual, _ = predict_action(actor, obs, state, masks)
            expected, _ = runner.policy.act(obs, state, masks, deterministic=True)
        torch.testing.assert_close(actual, expected)
        assert envs.action_space.contains(actual[0].cpu().numpy())
    finally:
        envs.close()
        eval_envs.close()
