import gymnasium as gym
import numpy as np
import pytest
import torch

from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer
from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.gru import InnovationGatedResidualGRU
from algorithms.utils.temporal_attention import get_memory_encoder, get_policy_state_shape
from config import get_config


def igru_args(**overrides):
    """构造新息门控残差 GRU（IGRU）的测试参数。"""
    args = get_config().parse_args(["--memory-encoder", "igru"])
    args.hidden_size = "32 32"
    args.act_hidden_size = "32"
    args.recurrent_hidden_size = 32
    args.recurrent_hidden_layers = 1
    args.data_chunk_length = 3
    args.buffer_size = 6
    args.n_rollout_threads = 2
    args.ppo_epoch = 2
    args.num_mini_batch = 2
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_igru_compact_state_isolation_reset_and_causality():
    """验证紧凑状态尺寸、无效上一帧隔离、单环境 mask=0 重置与严格时间因果性。"""
    torch.manual_seed(23)
    space = gym.spaces.Box(-10.0, 10.0, shape=(15,))
    args = igru_args()
    assert get_memory_encoder(args) == "igru"
    assert get_policy_state_shape(args, space) == (1, 32 + 15 + 1)

    layer = InnovationGatedResidualGRU(args, space)
    obs = torch.randn(2, 15)
    zero_state = torch.zeros(2, *layer.state_shape)
    masks = torch.ones(2, 1)
    clean_out, _ = layer(obs, zero_state, masks)

    # 门控初始偏置为 -2.0，训练起点开度应在 0.12 附近以平滑对齐原版 GRU。
    assert 0.05 < layer.last_in_gate_mean < 0.25
    assert 0.05 < layer.last_gate_mean < 0.25

    # 当上一帧有效位 valid=0 时，即使缓存槽位存在脏数据也不得泄露到差分或先验特征。
    dirty_state = zero_state.clone()
    gru_dim = layer._gru_flat_dim
    dirty_state[:, 0, gru_dim : gru_dim + 15] = 1e6
    dirty_out, _ = layer(obs, dirty_state, masks)
    torch.testing.assert_close(clean_out, dirty_out)

    # 连续运行若干步后仅重置第 0 个环境，必须精确回到首步输出且不干扰第 1 个环境。
    running_state = zero_state.clone()
    for _ in range(4):
        _, running_state = layer(torch.randn(2, 15), running_state, masks)
    env1_expected, _ = layer(obs[1:], running_state[1:], torch.ones(1, 1))
    reset_masks = torch.tensor([[0.0], [1.0]])
    reset_out, after_reset_state = layer(obs, running_state, reset_masks)
    torch.testing.assert_close(reset_out[0], clean_out[0])
    torch.testing.assert_close(reset_out[1:], env1_expected)
    torch.testing.assert_close(after_reset_state[:, 0, gru_dim : gru_dim + 15], obs)
    assert torch.all(after_reset_state[:, 0, -1] == 1.0)

    # 因果前缀检查：改变未来帧观测不会影响历史时刻特征。
    seq = torch.randn(7, 2, 15)

    def unroll(inputs):
        st = torch.zeros(2, *layer.state_shape)
        outs = []
        for frame in inputs:
            out, st = layer(frame, st, torch.ones(2, 1))
            outs.append(out)
        return torch.stack(outs)

    before = unroll(seq.clone())
    altered = seq.clone()
    altered[4:] += 25.0
    after = unroll(altered)
    torch.testing.assert_close(before[:4], after[:4])
    assert not torch.allclose(before[4:], after[4:])


@pytest.mark.parametrize("num_layers", [1, 2])
def test_igru_step_vs_chunk_strict_identity_after_parameter_mutation(num_layers):
    """在 LayerNorm 偏置非零与中途 mask=0 重置下，严格核验单步采样 (T=1) 与序列分块 (T=L) 逐元素恒等。"""
    torch.manual_seed(31)
    space = gym.spaces.Box(-10.0, 10.0, shape=(12,))
    args = igru_args(recurrent_hidden_layers=num_layers, hidden_size="48 40", recurrent_hidden_size=32)
    layer = InnovationGatedResidualGRU(args, space)

    # 主动打乱所有 LayerNorm 的 weight 与 bias，检验先验隐状态掩码顺序不存在偏置泄漏差异。
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "norm" in name:
                param.add_(0.15 * torch.randn_like(param))

    batch_envs = 3
    chunk_len = 5
    init_state = torch.randn(batch_envs, *layer.state_shape)
    init_state[:, 0, -1] = torch.tensor([0.0, 1.0, 1.0])
    obs_seq = torch.randn(chunk_len, batch_envs, 12)
    mask_seq = torch.ones(chunk_len, batch_envs, 1)
    mask_seq[0, 1, 0] = 0.0
    mask_seq[2, 0, 0] = 0.0
    mask_seq[3, 2, 0] = 0.0

    # 逐步执行 (T=1)
    step_state = init_state.clone()
    step_outputs = []
    for t in range(chunk_len):
        out_t, step_state = layer(obs_seq[t], step_state, mask_seq[t])
        step_outputs.append(out_t)
    expected_features = torch.cat(step_outputs, dim=0)

    # 分块一次性并行执行 (T=chunk_len)
    chunk_features, chunk_state = layer(
        obs_seq.reshape(chunk_len * batch_envs, 12),
        init_state.clone(),
        mask_seq.reshape(chunk_len * batch_envs, 1),
    )

    torch.testing.assert_close(chunk_features, expected_features, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chunk_state, step_state, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("policy_loss", ["clip", "spo"])
@pytest.mark.parametrize(
    "act_space",
    [
        gym.spaces.Discrete(12),
        gym.spaces.MultiDiscrete([41, 41, 41, 30]),
        gym.spaces.Tuple([gym.spaces.Discrete(12), gym.spaces.Discrete(2)]),
    ],
)
def test_igru_ppo_zero_update_ratio_and_training(act_space, policy_loss):
    """验证全动作空间下零更新重要性采样比严格为 1，且端到端 PPO/SPO 更新记录门控指标。"""
    torch.manual_seed(43)
    args = igru_args(policy_loss=policy_loss)
    obs_space = gym.spaces.Box(-10.0, 10.0, shape=(15,))
    policy = PPOPolicy(args, obs_space, act_space)
    buffer = ReplayBuffer(args, 1, obs_space, act_space)
    rng = np.random.default_rng(43)
    buffer.obs[0] = rng.normal(size=buffer.obs[0].shape).astype(np.float32)

    for step in range(args.buffer_size):
        with torch.no_grad():
            values, actions, logp, ha, hc = policy.get_actions(
                buffer.obs[step].reshape(2, 15),
                buffer.rnn_states_actor[step].reshape(2, *policy.actor.state_shape),
                buffer.rnn_states_critic[step].reshape(2, *policy.critic.state_shape),
                buffer.masks[step].reshape(2, 1),
            )
        next_masks = np.ones((2, 1, 1), dtype=np.float32)
        if step == 2:
            next_masks[0] = 0.0
        next_obs = rng.normal(size=(2, 1, 15)).astype(np.float32)
        if step == 4:
            # 模拟零阶保持丢帧冻结帧（当前帧完全等于上一帧）
            next_obs[1] = buffer.obs[step, 1]
        buffer.insert(
            next_obs,
            actions.numpy().reshape(2, 1, -1),
            rng.normal(size=(2, 1, 1)).astype(np.float32),
            next_masks,
            logp.numpy().reshape(2, 1, 1),
            values.numpy().reshape(2, 1, 1),
            ha.numpy().reshape(2, 1, *policy.actor.state_shape),
            hc.numpy().reshape(2, 1, *policy.critic.state_shape),
        )

    with torch.no_grad():
        next_val = policy.get_values(
            buffer.obs[-1].reshape(2, 15),
            buffer.rnn_states_critic[-1].reshape(2, *policy.critic.state_shape),
            buffer.masks[-1].reshape(2, 1),
        )
    buffer.compute_returns(next_val.numpy().reshape(2, 1, 1))

    # 更新前检查分块回放的对数概率与价值预测和采样时完全相等（重要性采样比 = 1.0）。
    for sample in ReplayBuffer.recurrent_generator(buffer, args.num_mini_batch, args.data_chunk_length):
        sample_obs, sample_act, sample_masks, old_logp, _, _, old_val, rnn_a, rnn_c = sample
        with torch.no_grad():
            new_val, new_logp, entropy = policy.evaluate_actions(sample_obs, rnn_a, rnn_c, sample_act, sample_masks)
        ratio = torch.exp(new_logp - torch.from_numpy(old_logp))
        torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(new_val, torch.from_numpy(old_val), atol=1e-5, rtol=1e-5)
        assert torch.isfinite(entropy).all()

    trainer = PPOTrainer(args)
    info = trainer.train(policy, buffer)
    assert np.isfinite(info["policy_loss"])
    assert np.isfinite(info["value_loss"])
    assert 0.0 < info["actor_igru_in_gate"] < 1.0
    assert 0.0 < info["actor_igru_res_gate"] < 1.0
    assert 0.0 < info["critic_igru_res_gate"] < 1.0


def test_igru_checkpoint_config_and_eval_loaders(tmp_path):
    """验证 model_config.json 保存校验及 1/heading 与 1v1 评估加载器对 igru 的兼容支持。"""
    from experiments import evaluate_lowlevel_controller as lowlevel_eval
    from experiments import heading_observation_degradation_eval as deg_eval
    import importlib.util

    # 1. Heading (12 维观测, MultiDiscrete([41, 41, 41, 30]))
    h_args = igru_args(hidden_size="32 32", act_hidden_size="32", recurrent_hidden_size=32)
    h_obs = gym.spaces.Box(-10.0, 10.0, shape=(12,))
    h_act = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    h_policy = PPOPolicy(h_args, h_obs, h_act)
    h_dir = tmp_path / "heading_igru"
    h_dir.mkdir()
    torch.save(h_policy.actor.state_dict(), h_dir / "actor_latest.pt")
    h_policy.save_model_config(h_dir)
    h_policy.validate_model_config(h_dir)

    loaded_h_path, loaded_h_actor = deg_eval.load_actor(h_dir, "igru", torch.device("cpu"))
    assert loaded_h_path.name == "actor_latest.pt"
    assert loaded_h_actor.memory_encoder == "igru"
    assert loaded_h_actor.state_shape == (1, 32 + 12 + 1)
    assert lowlevel_eval.build_actor(h_dir / "actor_latest.pt", torch.device("cpu")).memory_encoder == "igru"

    # 2. 1v1 战术分层 (15 维观测, Discrete(12))
    c_args = igru_args(hidden_size="32 32", act_hidden_size="32", recurrent_hidden_size=32)
    c_obs = gym.spaces.Box(-10.0, 10.0, shape=(15,))
    c_act = gym.spaces.Discrete(12)
    c_policy = PPOPolicy(c_args, c_obs, c_act)
    c_dir = tmp_path / "combat_igru"
    c_dir.mkdir()
    torch.save(c_policy.actor.state_dict(), c_dir / "actor_latest.pt")
    c_policy.save_model_config(c_dir)

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("eval_1v1_mod", "experiments/1v1.py")
    eval_1v1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = eval_1v1
    spec.loader.exec_module(eval_1v1)
    built_args = eval_1v1.build_actor_args_from_checkpoint(c_dir / "actor_latest.pt", c_obs, c_act, False)
    assert built_args.memory_encoder == "igru"
    rebuilt_actor = eval_1v1.PPOActor(built_args, c_obs, c_act, device=torch.device("cpu"))
    rebuilt_actor.load_state_dict(torch.load(c_dir / "actor_latest.pt", map_location="cpu"))
    assert rebuilt_actor.state_shape == (1, 32 + 15 + 1)


def test_igru_jsbsim_degraded_heading_and_combat_smoke(tmp_path):
    """在真实 JSBSim 混合退化 Heading 与雷达退化 1v1 环境中验证 IGRU 闭环运行与分层低层兼容。"""
    from envs.JSBSim.envs.singlecontrol_env import HeadingMixedDegradationEnv
    from envs.JSBSim.envs.singlecombat_env import SingleCombatRadarDegradationEnv

    # 1. HeadingMixedDegradationEnv 退化观测交互
    h_env = HeadingMixedDegradationEnv("1/heading", degradation_level=2, sensor_seed=7)
    try:
        h_env.seed(1)
        obs = h_env.reset()
        h_args = igru_args(hidden_size="32 32", act_hidden_size="32", recurrent_hidden_size=32)
        h_policy = PPOPolicy(h_args, h_env.observation_space, h_env.action_space)
        ha = np.zeros((1, *h_policy.actor.state_shape), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        for _ in range(4):
            with torch.no_grad():
                act, ha_t = h_policy.act(obs, ha, masks, deterministic=True)
            ha = ha_t.numpy()
            obs, _, dones, _ = h_env.step(act.numpy())
            masks = np.ones((1, 1), dtype=np.float32) * (1.0 - dones.reshape(1, 1))
    finally:
        h_env.close()

    # 2. SingleCombatRadarDegradationEnv 战术分层（内部同时加载固定原版 GRU 的 actor_heading.pt）
    c_env = SingleCombatRadarDegradationEnv("1v1/NoWeapon/TacticalHierarchySelfplay", radar_level=2, sensor_seed=9)
    try:
        c_env.seed(1)
        obs = c_env.reset()
        c_args = igru_args(hidden_size="32 32", act_hidden_size="32", recurrent_hidden_size=32)
        c_policy = PPOPolicy(c_args, c_env.observation_space, c_env.action_space)
        ha = np.zeros((2, *c_policy.actor.state_shape), dtype=np.float32)
        masks = np.ones((2, 1), dtype=np.float32)
        for _ in range(4):
            with torch.no_grad():
                act, ha_t = c_policy.act(obs, ha, masks, deterministic=True)
            ha = ha_t.numpy()
            obs, _, dones, _ = c_env.step(act.numpy())
            masks = np.ones((2, 1), dtype=np.float32) * (1.0 - dones.reshape(2, 1))
    finally:
        c_env.close()
