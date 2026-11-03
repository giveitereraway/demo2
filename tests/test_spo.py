from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer
from algorithms.utils.buffer import ReplayBuffer
from config import get_config


def fixed_update(ratios, advantages, policy_loss="spo"):
    """固定 log probability，直接检查实际更新入口的损失及反向梯度。"""
    args = get_config().parse_args(["--policy-loss", policy_loss])
    args.use_max_grad_norm = False
    args.entropy_coef = args.value_loss_coef = 0
    actor = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(ratios).log().reshape(-1, 1))])
    critic = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(actor[0]))])
    policy = SimpleNamespace(actor=actor, critic=critic,
        optimizer=torch.optim.SGD(list(actor.parameters()) + list(critic.parameters()), lr=0))
    policy.evaluate_actions = lambda *unused: (critic[0], actor[0], actor[0] * 0)
    zeros = torch.zeros_like(actor[0])
    sample = (None, None, None, zeros, torch.tensor(advantages).reshape(-1, 1),
              zeros, zeros, None, None)
    result = PPOTrainer(args).ppo_update(policy, sample)
    return result[0].item(), actor[0].grad.flatten()


@pytest.mark.parametrize("ratio,advantage,loss,gradient", [
    (0.6, 2.0, -0.4, -3.6),
    (1.0, 2.0, -2.0, -2.0),
    (1.4, 2.0, -2.0, 2.8),
    (0.6, -2.0, 2.0, -1.2),
    (1.0, -2.0, 2.0, 2.0),
    (1.4, -2.0, 3.6, 8.4),
    (1.2, 2.0, -2.2, 0.0),
    (0.8, -2.0, 1.8, 0.0),
    (1.4, 0.0, 0.0, 0.0),
])
def test_spo_loss_and_log_probability_gradient(ratio, advantage, loss, gradient):
    actual_loss, actual_gradient = fixed_update([ratio], [advantage])
    assert actual_loss == pytest.approx(loss, abs=1e-6)
    assert actual_gradient.item() == pytest.approx(gradient, abs=2e-6)


def test_mixed_advantages_do_not_broadcast_across_samples():
    loss, gradient = fixed_update([1.4, 0.6, 1.0], [2.0, -2.0, 0.0])
    assert loss == pytest.approx(0, abs=1e-6)
    torch.testing.assert_close(gradient, torch.tensor([2.8 / 3, -1.2 / 3, 0.0]))


@pytest.mark.parametrize("ratio,advantage,loss", [(1.4, 2.0, -2.4), (0.6, -2.0, 1.6)])
def test_clip_keeps_its_zero_gradient_beyond_the_boundary(ratio, advantage, loss):
    actual_loss, gradient = fixed_update([ratio], [advantage], "clip")
    assert actual_loss == pytest.approx(loss, abs=1e-6)
    assert gradient.item() == 0


@pytest.mark.parametrize("epsilon", [0, -0.1, float("nan"), float("inf")])
def test_spo_rejects_invalid_epsilon(epsilon):
    args = get_config().parse_args(["--policy-loss", "spo"])
    args.clip_param = epsilon
    with pytest.raises(ValueError, match="有限正数"):
        PPOTrainer(args)


def test_default_and_legacy_arguments_keep_clip():
    args = get_config().parse_args([])
    assert args.policy_loss == PPOTrainer(args).policy_loss == "clip"
    del args.policy_loss
    assert PPOTrainer(args).policy_loss == "clip"


@pytest.mark.parametrize("algorithm,env", [("mappo", "SingleCombat"), ("ppo", "MultipleCombat")])
def test_training_entry_rejects_unsupported_spo(algorithm, env):
    from scripts.train.train_jsbsim import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--policy-loss", "spo", "--algorithm-name", algorithm, "--env-name", env], get_config())
    assert error.value.code == 2


def test_gru_spo_rollout_update_and_diagnostics():
    """使用真实策略、联合动作概率和 GAE 缓冲区完成两轮优化。"""
    torch.manual_seed(12)
    args = get_config().parse_args(["--policy-loss", "spo"])
    args.hidden_size = args.act_hidden_size = "16"
    args.recurrent_hidden_size = 16
    args.n_rollout_threads = 1
    args.buffer_size = 4
    args.data_chunk_length = 2
    args.num_mini_batch = args.ppo_epoch = 2
    obs_space = gym.spaces.Box(-10, 10, shape=(15,))
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    policy = PPOPolicy(args, obs_space, act_space)
    buffer = ReplayBuffer(args, 1, obs_space, act_space)
    rng = np.random.default_rng(12)
    buffer.obs[0] = rng.normal(size=buffer.obs[0].shape)
    for step in range(args.buffer_size):
        with torch.no_grad():
            values, actions, logp, ha, hc = policy.get_actions(
                buffer.obs[step, 0], buffer.rnn_states_actor[step, 0],
                buffer.rnn_states_critic[step, 0], buffer.masks[step, 0])
        buffer.insert(rng.normal(size=buffer.obs[0].shape), actions.numpy()[None],
                      np.full((1, 1, 1), step + 1), np.ones((1, 1, 1)),
                      logp.numpy()[None], values.numpy()[None], ha.numpy()[None], hc.numpy()[None])
    buffer.compute_returns(np.zeros((1, 1, 1)))
    old_logp = buffer.action_log_probs.copy()
    before = [p.detach().clone() for p in policy.actor.parameters()]
    info = PPOTrainer(args).train(policy, buffer)
    np.testing.assert_array_equal(buffer.action_log_probs, old_logp)
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.actor.parameters()))
    assert all(np.isfinite(value) for value in info.values())
    assert info["ratio_deviation"] >= 0
    assert info["approx_kl"] >= -1e-7
    assert 0 <= info["ratio_out_of_bounds"] <= 1
    expected_ev = 1 - np.var(buffer.returns[:-1] - buffer.value_preds[:-1]) / np.var(buffer.returns[:-1])
    assert info["explained_variance"] == pytest.approx(expected_ev)
