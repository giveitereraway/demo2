import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer
from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.temporal_attention import TemporalAttention, get_memory_encoder
from config import get_config


def attention_args(**overrides):
    args = get_config().parse_args(["--memory-encoder", "temporal_attention"])
    args.buffer_size = 3
    args.n_rollout_threads = 2
    args.history_length = 5
    args.ppo_epoch = 1
    args.num_mini_batch = 2
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_padding_reset_environment_isolation_and_history_gradients():
    torch.manual_seed(7)
    space = gym.spaces.Box(-1, 1, shape=(15,))
    layer = TemporalAttention(attention_args(), space)
    obs = torch.randn(2, 15)
    history = torch.zeros(2, *layer.state_shape)
    masks = torch.ones(2, 1)
    reference, _ = layer(obs, history, masks)
    dirty_padding = history.clone()
    dirty_padding[:, :, :-1] = 1e6
    padded, _ = layer(obs, dirty_padding, masks)
    torch.testing.assert_close(reference, padded)
    assert torch.isfinite(padded).all()

    history_length = layer.state_shape[0]
    history[:, :, :-1] = torch.randn(2, history_length, 15)
    history[:, :, -1] = 1
    reset_masks = torch.tensor([[0.0], [1.0]])
    output, state = layer(obs, history, reset_masks)
    torch.testing.assert_close(output[0], reference[0])
    assert state[0, :-1, -1].sum() == 0
    assert state[1, :, -1].sum() == history_length
    other_history = history.clone()
    other_history[1, :, :-1] *= 100
    isolated, _ = layer(obs, other_history, reset_masks)
    torch.testing.assert_close(isolated[0], output[0])

    history.requires_grad_()
    features, _ = layer(obs, history, masks)
    features[:, 0].sum().backward()
    assert history.grad[:, 1:, :-1].abs().sum() > 0
    assert torch.isfinite(history.grad).all()


def test_causal_prefix_is_unchanged_by_future_observations():
    torch.manual_seed(8)
    layer = TemporalAttention(attention_args(), gym.spaces.Box(-1, 1, shape=(15,)))
    sequence = torch.randn(9, 2, 15)

    def run(inputs):
        history = torch.zeros(2, *layer.state_shape)
        outputs = []
        for obs in inputs:
            output, history = layer(obs, history, torch.ones(2, 1))
            outputs.append(output)
        return torch.stack(outputs)

    before = run(sequence)
    sequence[4:] += 100
    after = run(sequence)
    torch.testing.assert_close(before[:4], after[:4])
    assert not torch.allclose(before[4:], after[4:])


@pytest.mark.parametrize("act_space", [gym.spaces.Discrete(12),
    gym.spaces.MultiDiscrete([41, 41, 41, 30]),
    gym.spaces.Tuple([gym.spaces.Discrete(12), gym.spaces.Discrete(2)])])
def test_rollout_prefix_probability_ratio_and_update(act_space):
    """H 大于 rollout 长度；覆盖跨 rollout、独立环境/飞机和回合中途重置。"""
    torch.manual_seed(9)
    args = attention_args()
    space = gym.spaces.Box(-100, 100, shape=(15,))
    policy = PPOPolicy(args, space, act_space)
    buffer = ReplayBuffer(args, 2, space, act_space)
    rng = np.random.default_rng(9)
    buffer.obs[0] = rng.normal(size=buffer.obs[0].shape)
    batch = args.n_rollout_threads * 2
    for rollout in range(2):
        for step in range(args.buffer_size):
            with torch.no_grad():
                values, actions, log_probs, ha, hc = policy.get_actions(
                    buffer.obs[step].reshape(batch, 15),
                    buffer.rnn_states_actor[step].reshape(batch, *policy.actor.state_shape),
                    buffer.rnn_states_critic[step].reshape(batch, *policy.critic.state_shape),
                    buffer.masks[step].reshape(batch, 1))
            masks = np.ones((2, 2, 1), dtype=np.float32)
            if rollout == 1 and step == 0:
                masks[0] = 0
            buffer.insert(rng.normal(size=buffer.obs[0].shape),
                          actions.numpy().reshape(2, 2, -1), np.ones((2, 2, 1)), masks,
                          log_probs.numpy().reshape(2, 2, 1), values.numpy().reshape(2, 2, 1),
                          ha.numpy().reshape(2, 2, *policy.actor.state_shape),
                          hc.numpy().reshape(2, 2, *policy.critic.state_shape))
        buffer.compute_returns(np.zeros((2, 2, 1)))
        # 重排所有时刻后，训练模式的 log_prob/value 应与逐步采样相同。
        policy.prep_training()
        visited = 0
        for obs, actions, masks, old_logp, _, _, old_values, ha, hc in buffer.feed_forward_generator(buffer, 5):
            values, logp, entropy = policy.evaluate_actions(obs, ha, hc, actions, masks)
            torch.testing.assert_close(logp, torch.from_numpy(old_logp), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(values, torch.from_numpy(old_values), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close((logp - torch.from_numpy(old_logp)).exp(), torch.ones_like(logp),
                                       atol=2e-6, rtol=2e-6)
            assert torch.isfinite(entropy).all()
            visited += len(obs)
        assert visited == args.buffer_size * batch
        buffer.after_update()
        if rollout == 0:
            assert np.all(buffer.rnn_states_actor[0, :, :, :, -1].sum(axis=-1) == 3)
    before = policy.actor.base.embedding.weight.detach().clone()
    info = PPOTrainer(args).train(policy, buffer)
    assert all(np.isfinite(value) for value in info.values())
    assert not torch.equal(before, policy.actor.base.embedding.weight)
    assert next(policy.actor.base.parameters()) is not next(policy.critic.base.parameters())


def test_mlp_training_and_legacy_gru_checkpoint():
    space, action = gym.spaces.Box(-1, 1, shape=(15,)), gym.spaces.Discrete(12)
    legacy_args = get_config().parse_args([])
    legacy = PPOPolicy(legacy_args, space, action)
    explicit = PPOPolicy(get_config().parse_args(["--memory-encoder", "gru"]), space, action)
    explicit.actor.load_state_dict(legacy.actor.state_dict(), strict=True)
    explicit.critic.load_state_dict(legacy.critic.state_dict(), strict=True)
    assert get_memory_encoder(legacy_args) == "gru"
    args = attention_args(memory_encoder="mlp")
    policy = PPOPolicy(args, space, action)
    info = PPOTrainer(args).train(policy, ReplayBuffer(args, 1, space, action))
    assert all(np.isfinite(value) for value in info.values())
    assert not hasattr(policy.actor, "rnn")


def test_checkpoint_metadata_and_invalid_options(tmp_path):
    space = gym.spaces.Box(-1, 1, shape=(15,))
    policy = PPOPolicy(attention_args(), space, gym.spaces.Discrete(12))
    with pytest.raises(ValueError, match="缺少"):
        policy.validate_model_config(tmp_path)
    policy.save_model_config(tmp_path)
    policy.validate_model_config(tmp_path)
    config = json.loads((tmp_path / "model_config.json").read_text(encoding="utf-8"))
    assert config["history_length"] == 5
    # 头数改变不会改变权重形状，必须由元数据挡住。
    other = PPOPolicy(attention_args(attention_heads=2), space, gym.spaces.Discrete(12))
    with pytest.raises(ValueError, match="配置不匹配"):
        other.validate_model_config(tmp_path)
    with pytest.raises(ValueError, match="冲突"):
        get_memory_encoder(attention_args(use_recurrent_policy=False))
    with pytest.raises(ValueError):
        TemporalAttention(attention_args(history_length=0), space)
    with pytest.raises(ValueError):
        TemporalAttention(attention_args(attention_heads=3), space)
    with pytest.raises(ValueError):
        get_memory_encoder(attention_args(algorithm_name="mappo"))


@pytest.mark.parametrize("env_name,scenario_name", [
    ("SingleControl", "1/heading"), ("SingleCombat", "1v1/NoWeapon/Selfplay"),
])
def test_attention_training_entry_accepts_supported_environments(env_name, scenario_name):
    from scripts.train.train_jsbsim import parse_args

    args = parse_args(["--env-name", env_name, "--scenario-name", scenario_name,
                       "--memory-encoder", "temporal_attention"], get_config())
    assert args.env_name == env_name
    assert args.memory_encoder == "temporal_attention"


def test_attention_training_entry_rejects_multiple_combat(capsys):
    from scripts.train.train_jsbsim import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--env-name", "MultipleCombat", "--memory-encoder", "temporal_attention"], get_config())
    assert error.value.code == 2
    assert "仅支持 SingleControl 和 SingleCombat" in capsys.readouterr().err


@pytest.mark.parametrize("env_name,scenario_name,selfplay,width", [
    ("SingleCombat", "1v1/NoWeapon/Selfplay", False, 64),
    ("SingleCombat", "1v1/NoWeapon/Selfplay", True, 64),
    ("SingleControl", "1/heading", False, 64),
    ("SingleControl", "1/heading", False, 128),
])
def test_attention_jsbsim_training_evaluation_and_restore(tmp_path, env_name, scenario_name, selfplay, width, monkeypatch):
    """真实飞行仿真闭环：两次更新、跨回合评估、保存/恢复和换对手清空。"""
    from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
    from envs.env_wrappers import DummyVecEnv
    from runner.jsbsim_runner import JSBSimRunner
    from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner

    torch.set_num_threads(1)
    args = attention_args(n_rollout_threads=1, num_env_steps=8, buffer_size=4, history_length=16,
                          attention_hidden_size=width,
                          use_selfplay=selfplay, n_choose_opponents=1, use_eval=True,
                          n_eval_rollout_threads=1, eval_episodes=2, eval_interval=1)
    args.env_name = env_name
    args.scenario_name = scenario_name

    def make_env():
        env_type = SingleControlEnv if env_name == "SingleControl" else SingleCombatEnv
        env = env_type(args.scenario_name)
        # 仅缩短本测试的回合，不改场景文件及物理数据。
        for condition in env.task.termination_conditions:
            if hasattr(condition, "max_steps"):
                condition.max_steps = 3
        return env

    envs, eval_envs = DummyVecEnv([make_env]), DummyVecEnv([make_env])
    try:
        runner_type = SelfplayJSBSimRunner if selfplay else JSBSimRunner
        runner = runner_type(dict(all_args=args, envs=envs, eval_envs=eval_envs,
                                  device=torch.device("cpu"), render_mode="txt", run_dir=tmp_path))
        original_eval = runner.eval
        eval_masks = []

        def record_eval(total_steps):
            hook = runner.policy.actor.base.register_forward_pre_hook(
                lambda module, inputs: eval_masks.append(inputs[2].detach().clone()))
            try:
                return original_eval(total_steps)
            finally:
                hook.remove()

        monkeypatch.setattr(runner, "eval", record_eval)
        runner.run()
        assert [mask[0, 0].item() for mask in eval_masks] == [1, 1, 1, 0, 1, 1]
        assert runner.total_num_steps == 8
        assert (tmp_path / "actor_latest.pt").exists()
        assert (tmp_path / "actor_best.pt").exists()
        assert (tmp_path / "model_config.json").exists()
        runner.model_dir = tmp_path
        runner.restore()
        if selfplay:
            runner.reset_opponent()
            assert not runner.opponent_rnn_states.any()
            assert not runner.buffer.rnn_states_actor.any()
    finally:
        envs.close()
        eval_envs.close()
