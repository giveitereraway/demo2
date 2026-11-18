from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer, get_num_action_heads
from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.temporal_attention import GatedAttentionGRU, get_memory_encoder
from config import get_config


def gru_attn_args(**overrides):
    """构造门控递归-注意力编码器的测试参数。"""
    args = get_config().parse_args(["--memory-encoder", "gru_attention"])
    args.hidden_size = "32 32"
    args.act_hidden_size = "32"
    args.recurrent_hidden_size = 32
    args.recurrent_hidden_layers = 1
    args.attention_hidden_size = 32
    args.attention_heads = 4
    args.history_length = 6
    args.data_chunk_length = 3
    args.buffer_size = 6
    args.n_rollout_threads = 2
    args.ppo_epoch = 1
    args.num_mini_batch = 2
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_gated_attention_gru_padding_isolation_reset_and_causality():
    """验证脏 padding 不泄露到差分与注意力、mask=0 同步清空 GRU 与窗口，且严格满足因果性。"""
    torch.manual_seed(11)
    space = gym.spaces.Box(-10.0, 10.0, shape=(15,))
    args = gru_attn_args()
    layer = GatedAttentionGRU(args, space)

    obs = torch.randn(2, 15)
    zero_state = torch.zeros(2, *layer.state_shape)
    masks = torch.ones(2, 1)
    clean_out, _ = layer(obs, zero_state, masks)

    # 初始门控偏置为 -2.0，开度应在 0.12 附近。
    assert 0.05 < layer.last_gate_mean < 0.25

    # 往无效历史槽位写入极大脏值（保留 valid=0），输出必须严格不变。
    dirty_state = zero_state.clone()
    gru_dim = layer._gru_flat_dim
    hist_view = dirty_state[:, 0, gru_dim:].view(2, args.history_length, 16)
    hist_view[:, :, :-1] = 1e6
    dirty_out, _ = layer(obs, dirty_state, masks)
    torch.testing.assert_close(clean_out, dirty_out)

    # 积累数步真实历史后，仅重置第 0 个环境，应回到初始状态输出且不影响第 1 个环境。
    running_state = zero_state.clone()
    for _ in range(4):
        _, running_state = layer(torch.randn(2, 15), running_state, masks)
    reset_masks = torch.tensor([[0.0], [1.0]])
    reset_out, after_reset_state = layer(obs, running_state, reset_masks)
    torch.testing.assert_close(reset_out[0], clean_out[0])
    reset_hist = after_reset_state[0, 0, gru_dim:].view(args.history_length, 16)
    assert reset_hist[:-1, -1].sum() == 0
    assert reset_hist[-1, -1].item() == 1.0

    # 因果前缀检查：修改未来帧不得改变过去时刻输出。
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
    altered[4:] += 50.0
    after = unroll(altered)
    torch.testing.assert_close(before[:4], after[:4])
    assert not torch.allclose(before[4:], after[4:])


@pytest.mark.parametrize("policy_loss", ["clip", "spo"])
@pytest.mark.parametrize(
    "act_space",
    [
        gym.spaces.Discrete(12),
        gym.spaces.MultiDiscrete([41, 41, 41, 30]),
        gym.spaces.Tuple([gym.spaces.Discrete(12), gym.spaces.Discrete(2)]),
    ],
)
def test_gated_attention_gru_step_vs_chunk_equivalence_and_update(act_space, policy_loss):
    """验证逐步采样 (T=1) 与 recurrent_generator 序列块 (T=L) 在中途重置下的严格等价性。"""
    torch.manual_seed(17)
    args = gru_attn_args(policy_loss=policy_loss)
    obs_space = gym.spaces.Box(-10.0, 10.0, shape=(15,))
    policy = PPOPolicy(args, obs_space, act_space)
    buffer = ReplayBuffer(args, 1, obs_space, act_space)
    rng = np.random.default_rng(17)
    buffer.obs[0] = rng.normal(size=buffer.obs[0].shape).astype(np.float32)

    for rollout in range(2):
        for step in range(args.buffer_size):
            with torch.no_grad():
                values, actions, logp, ha, hc = policy.get_actions(
                    buffer.obs[step].reshape(2, 15),
                    buffer.rnn_states_actor[step].reshape(2, *policy.actor.state_shape),
                    buffer.rnn_states_critic[step].reshape(2, *policy.critic.state_shape),
                    buffer.masks[step].reshape(2, 1),
                )
            next_masks = np.ones((2, 1, 1), dtype=np.float32)
            ha_np = ha.numpy().reshape(2, 1, *policy.actor.state_shape)
            hc_np = hc.numpy().reshape(2, 1, *policy.critic.state_shape)
            # 在序列块中间构造回合终止重置，检验分块回放的一致性。
            if step == 1:
                next_masks[0] = 0.0
                ha_np[0] = 0.0
                hc_np[0] = 0.0
            buffer.insert(
                rng.normal(size=buffer.obs[0].shape).astype(np.float32),
                actions.numpy().reshape(2, 1, -1),
                np.full((2, 1, 1), float(step + 1), dtype=np.float32),
                next_masks,
                logp.numpy().reshape(2, 1, 1),
                values.numpy().reshape(2, 1, 1),
                ha_np,
                hc_np,
            )

        buffer.compute_returns(np.zeros((2, 1, 1), dtype=np.float32))
        policy.prep_training()
        visited = 0
        for sample in ReplayBuffer.recurrent_generator(buffer, args.num_mini_batch, args.data_chunk_length):
            obs_b, act_b, mask_b, old_logp_b, _, _, old_val_b, ha_b, hc_b = sample
            val_pred, logp_pred, entropy = policy.evaluate_actions(obs_b, ha_b, hc_b, act_b, mask_b)
            torch.testing.assert_close(logp_pred, torch.from_numpy(old_logp_b), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(val_pred, torch.from_numpy(old_val_b), atol=1e-5, rtol=1e-5)
            assert torch.isfinite(entropy).all()
            visited += len(obs_b)
        assert visited == args.buffer_size * args.n_rollout_threads
        if rollout == 0:
            buffer.after_update()

    trainer = PPOTrainer(args)
    info = trainer.train(policy, buffer)
    assert all(np.isfinite(val) for val in info.values())
    assert "actor_attention_gate" in info
    assert "geo_approx_kl" in info


def test_spo_geo_unbiased_gradient_and_multidiscrete_trust_region():
    """验证欧拉极限 SPO-Geo 的一阶无偏性、二阶联合 KL 曲率维度不变性、对称欧拉驻点与抗乘积爆炸性质。"""
    act_space_1d = gym.spaces.Discrete(12)
    act_space_2d = gym.spaces.Tuple([gym.spaces.Discrete(12), gym.spaces.Discrete(2)])
    act_space_4d = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    assert get_num_action_heads(act_space_4d) == 4
    assert get_num_action_heads(act_space_1d) == 1
    assert get_num_action_heads(act_space_2d) == 2

    def eval_spo_loss_grad_curv(joint_ratio: float, advantage: float, *, act_space=act_space_4d, adaptive: bool = True):
        cli = ["--policy-loss", "spo", "--clip-param", "0.2"]
        if not adaptive:
            cli.append("--disable-spo-adaptive-dim")
        args = get_config().parse_args(cli)
        args.use_max_grad_norm = False
        args.entropy_coef = args.value_loss_coef = 0.0
        logp = torch.nn.Parameter(torch.tensor([[np.log(joint_ratio)]], dtype=torch.float32))
        val = torch.nn.Parameter(torch.zeros_like(logp))
        policy = SimpleNamespace(
            act_space=act_space,
            actor=torch.nn.ParameterList([logp]),
            critic=torch.nn.ParameterList([val]),
            optimizer=torch.optim.SGD([logp, val], lr=0.0),
            evaluate_actions=lambda *_: (val, logp, logp * 0.0),
        )
        zeros = torch.zeros_like(logp)
        sample = (None, None, None, zeros, torch.tensor([[advantage]], dtype=torch.float32), zeros, zeros, None, None)
        trainer = PPOTrainer(args)
        loss, *_ = trainer.ppo_update(policy, sample)
        return loss.item(), logp.grad.item(), trainer._last_geo_clip_param

    # 1) D=1 (Discrete) 时，SPO-Geo 与原版 SPO 逐位严格等价。
    loss_1d_geo, grad_1d_geo, clip_1d = eval_spo_loss_grad_curv(1.35, 2.0, act_space=act_space_1d, adaptive=True)
    loss_1d_raw, grad_1d_raw, _ = eval_spo_loss_grad_curv(1.35, 2.0, act_space=act_space_1d, adaptive=False)
    assert clip_1d == pytest.approx(0.2)
    assert loss_1d_geo == pytest.approx(loss_1d_raw, abs=1e-6)
    assert grad_1d_geo == pytest.approx(grad_1d_raw, abs=1e-6)

    # 2) 在新旧策略相等 (r=1) 处，任意维数 D 下 SPO-Geo 与标准策略梯度严格一致：d(-r*A)/d(log pi) = -A。
    for sp in (act_space_1d, act_space_2d, act_space_4d):
        _, grad_pos, _ = eval_spo_loss_grad_curv(1.0, 2.0, act_space=sp, adaptive=True)
        _, grad_neg, _ = eval_spo_loss_grad_curv(1.0, -2.0, act_space=sp, adaptive=True)
        assert grad_pos == pytest.approx(-2.0, abs=1e-6)
        assert grad_neg == pytest.approx(2.0, abs=1e-6)

    # 3) 欧拉极限驻点验证：eps=0.2, D=4 时 eps_geo = 0.05，正/负优势极值点严格位于 (1 +- 0.05)^4。
    _, grad_pos_peak, clip_4d = eval_spo_loss_grad_curv(1.05 ** 4, 2.0, act_space=act_space_4d, adaptive=True)
    _, grad_neg_peak, _ = eval_spo_loss_grad_curv(0.95 ** 4, -2.0, act_space=act_space_4d, adaptive=True)
    assert clip_4d == pytest.approx(0.05)
    assert grad_pos_peak == pytest.approx(0.0, abs=1e-5)
    assert grad_neg_peak == pytest.approx(0.0, abs=1e-5)

    # 4) 二阶惩罚曲率维度不变性：纯二次惩罚项在 log r = 0 处的二阶导恒为 |A| / eps = 10.0 (与 D 无关)。
    eps_fd = 1e-3
    for sp, d_heads in ((act_space_1d, 1), (act_space_2d, 2), (act_space_4d, 4)):
        _, g_plus, _ = eval_spo_loss_grad_curv(float(np.exp(eps_fd)), 2.0, act_space=sp, adaptive=True)
        _, g_minus, _ = eval_spo_loss_grad_curv(float(np.exp(-eps_fd)), 2.0, act_space=sp, adaptive=True)
        total_hessian = (g_plus - g_minus) / (2 * eps_fd)
        # 总损失对 log r 的二阶导为 -A / D + |A| / eps，故提取纯惩罚曲率项应恒等于 |A| / eps = 10.0。
        penalty_curvature = total_hessian + 2.0 / d_heads
        assert penalty_curvature == pytest.approx(2.0 / 0.2, rel=1e-3)

    # 5) 抗指数乘积爆炸与合理越界约束：在 r = 1.2^4 = 2.0736 处，
    #    欧拉极限 SPO-Geo 施加温和正向拉回梯度 (+7.20)，既防止旧版过松零梯度导致的 KL 失控，
    #    又显著低于未归一化原始 SPO 的指数爆炸梯度 (+18.12)。
    _, grad_euler_pullback, _ = eval_spo_loss_grad_curv(1.2 ** 4, 2.0, act_space=act_space_4d, adaptive=True)
    _, grad_raw_explosion, _ = eval_spo_loss_grad_curv(1.2 ** 4, 2.0, act_space=act_space_4d, adaptive=False)
    assert grad_euler_pullback == pytest.approx(7.2, abs=1e-4)
    assert grad_raw_explosion > 2.5 * grad_euler_pullback


def test_heading_mixed_degradation_env_isolates_error_channels_and_preserves_physics():
    """验证底层混合退化环境支持零阶保持丢帧与延迟，仅污染前 3 维误差且不改变物理仿真与评分。"""
    from envs.JSBSim.envs.singlecontrol_env import HeadingMixedDegradationEnv, SingleControlEnv
    from scripts.train.train_jsbsim import make_eval_env, make_train_env, parse_args

    clean = SingleControlEnv("1/heading")
    mixed = HeadingMixedDegradationEnv("1/heading", degradation_level=3, sensor_seed=42)
    try:
        clean.seed(5)
        mixed.seed(5)
        clean_obs = clean.reset()
        mixed_obs = mixed.reset()
        agent_id = mixed.ego_ids[0]
        # 后 9 维本机状态实时保持不变，前 3 维受首帧延迟队列与噪声退化。
        np.testing.assert_allclose(mixed_obs[0, 3:], clean_obs[0, 3:])
        for _ in range(4):
            action = np.array([[20, 20, 20, 15]])
            clean_obs, clean_rew, clean_done, _ = clean.step(action)
            mixed_obs, mixed_rew, mixed_done, _ = mixed.step(action)
            np.testing.assert_array_equal(mixed.task.get_obs(mixed, agent_id), clean.get_obs()[agent_id])
            np.testing.assert_allclose(mixed_obs[0, 3:], clean_obs[0, 3:])
            np.testing.assert_array_equal(mixed_rew, clean_rew)
            np.testing.assert_array_equal(mixed_done, clean_done)
            assert not np.array_equal(mixed_obs[0, :3], clean_obs[0, :3])
    finally:
        mixed.close()
        clean.close()

    # 检查训练入口参数互斥与定期评估固定 E2 档构建。
    with pytest.raises(SystemExit):
        parse_args(
            ["--env-name", "SingleControl", "--scenario-name", "1/heading",
             "--heading-measurement-noise", "--heading-mixed-degradation"],
            get_config(),
        )
    args = parse_args(
        ["--env-name", "SingleControl", "--scenario-name", "1/heading",
         "--heading-mixed-degradation", "--use-eval", "--heading-first-target",
         "--n-rollout-threads", "1", "--n-eval-rollout-threads", "1"],
        get_config(),
    )
    train_vec = make_train_env(args)
    eval_vec = make_eval_env(args)
    try:
        assert type(train_vec.envs[0]) is HeadingMixedDegradationEnv
        assert eval_vec.envs[0].degradation_level == 2
        assert getattr(eval_vec.envs[0], "record_heading_tracking", False) is True
    finally:
        train_vec.close()
        eval_vec.close()


def test_tactical_selfplay_radar_shares_training_level_per_episode():
    from scripts.train.train_jsbsim import make_train_env, parse_args

    args = parse_args(
        ["--env-name", "SingleCombat", "--scenario-name", "1v1/NoWeapon/TacticalHierarchySelfplay",
         "--combat-radar-degradation", "--n-rollout-threads", "1", "--seed", "7"],
        get_config(),
    )
    train_vec = make_train_env(args)
    try:
        env = train_vec.envs[0]
        sampled_levels = set()
        for _ in range(16):
            env.reset()
            level_a, level_b = env._agent_levels.values()
            assert level_a == level_b
            sampled_levels.add(level_a)
        assert sampled_levels == {0, 1, 2}
    finally:
        train_vec.close()


def test_radar_degradation_eliminates_tactical_combat_geometry_ground_truth_leak():
    """验证 1v1 战术分层在雷达全丢帧下，观测与 _combat_geometry 同步冻结在初始航迹，不泄露敌机新真值位置。"""
    from envs.JSBSim.envs.singlecombat_env import (
        RadarDegradationSpec,
        RadarTrackDegrader,
        SingleCombatEnv,
        SingleCombatRadarDegradationEnv,
        extract_true_enemy_track,
    )

    # 1) E0 干净档位与原始 SingleCombatEnv 完全一致。
    clean_env = SingleCombatEnv("1v1/NoWeapon/TacticalHierarchySelfplay")
    radar_e0 = SingleCombatRadarDegradationEnv("1v1/NoWeapon/TacticalHierarchySelfplay", radar_level=0)
    try:
        clean_env.seed(21)
        radar_e0.seed(21)
        np.testing.assert_allclose(clean_env.reset(), radar_e0.reset())
        act = np.array([[0], [0]])
        obs_c, rew_c, done_c, _ = clean_env.step(act)
        obs_r, rew_r, done_r, _ = radar_e0.step(act)
        np.testing.assert_allclose(obs_c, obs_r)
        np.testing.assert_allclose(rew_c, rew_r)
        np.testing.assert_array_equal(done_c, done_r)
    finally:
        clean_env.close()
        radar_e0.close()

    # 2) 100% 雷达丢帧覆盖下，敌机飞行数步后，_combat_geometry 的目标向量必须指向冻结旧航迹而非真值。
    env = SingleCombatEnv("1v1/NoWeapon/TacticalHierarchySelfplay")
    try:
        env.seed(33)
        env.reset()
        ego_id = env.ego_ids[0]
        initial_track = extract_true_enemy_track(env, ego_id)
        full_drop = RadarTrackDegrader(RadarDegradationSpec(dropout_probability=1.0), np.random.default_rng(1))
        full_drop.reset(initial_track)

        for _ in range(5):
            env.step(np.array([[6], [8]]))  # 本机平飞加速，敌机左急转改变真实位置

        true_geom = env.task._combat_geometry(env, ego_id)
        frozen_track = full_drop.transform(env.agents[ego_id].get_position(), extract_true_enemy_track(env, ego_id))
        np.testing.assert_allclose(frozen_track, initial_track)

        env._radar_track_overrides = {ego_id: frozen_track}
        try:
            degraded_geom = env.task._combat_geometry(env, ego_id)
            degraded_obs = env.task.get_obs(env, ego_id)
        finally:
            env._radar_track_overrides = None
        clean_obs_now = env.task.get_obs(env, ego_id)

        expected_rel_xy = initial_track[:2] - env.agents[ego_id].get_position()[:2]
        np.testing.assert_allclose(degraded_geom["relative_xy"], expected_rel_xy, atol=1e-4)
        assert not np.allclose(degraded_geom["relative_xy"], true_geom["relative_xy"])
        assert not np.allclose(degraded_obs[9:15], clean_obs_now[9:15])
    finally:
        env.close()


def test_end_to_end_smoke_training_and_evaluators_for_gru_attention(tmp_path):
    """验证底层 heading 与高层 1v1 战术自博弈的训练保存及三大独立评估脚本加载闭环。"""
    from envs.JSBSim.envs.singlecombat_env import SingleCombatRadarDegradationEnv
    from envs.JSBSim.envs.singlecontrol_env import HeadingMixedDegradationEnv
    from envs.env_wrappers import DummyVecEnv
    from experiments import heading_observation_degradation_eval as heading_eval
    from experiments import partial_observability_gru_eval as po_eval
    from runner.jsbsim_runner import JSBSimRunner
    from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner

    torch.set_num_threads(1)

    # A) 底层 heading: SPO-Geo + GatedAttentionGRU 短闭环训练 + 观测退化评估器加载运行
    h_dir = tmp_path / "heading_run"
    h_args = gru_attn_args(
        policy_loss="spo",
        n_rollout_threads=1,
        n_eval_rollout_threads=1,
        num_env_steps=6,
        buffer_size=3,
        data_chunk_length=3,
        num_mini_batch=1,
        use_eval=True,
        eval_episodes=1,
        eval_interval=1,
    )
    h_args.env_name = "SingleControl"
    h_args.scenario_name = "1/heading"

    def make_h_env():
        env = HeadingMixedDegradationEnv("1/heading", degradation_level=1, sensor_seed=7)
        for cond in env.task.termination_conditions:
            if hasattr(cond, "max_steps"):
                cond.max_steps = 3
        return env

    h_envs, h_eval_envs = DummyVecEnv([make_h_env]), DummyVecEnv([make_h_env])
    try:
        runner = JSBSimRunner(
            dict(all_args=h_args, envs=h_envs, eval_envs=h_eval_envs,
                 device=torch.device("cpu"), render_mode="txt", run_dir=h_dir)
        )
        runner.run()
        assert (h_dir / "actor_best.pt").is_file()
        assert (h_dir / "model_config.json").is_file()

        resolved_path, loaded_actor = heading_eval.load_actor(
            h_dir, "gru_attention", torch.device("cpu"), prefer_best=True
        )
        assert resolved_path.name == "actor_best.pt"
        eval_env = heading_eval.heading.SingleControlEnv("1/heading")
        try:
            row, _ = heading_eval.run_trial(
                eval_env, loaded_actor, heading_eval.CONDITIONS[1],
                trial_id=1, seed=1, command=heading_eval.heading.Command(0.0, 15.0, 0.0),
                horizon_steps=4, device=torch.device("cpu"),
            )
            assert "attention_gate_mean" in row
            assert np.isfinite(row["episode_reward"])
        finally:
            eval_env.close()
    finally:
        h_envs.close()
        h_eval_envs.close()

    # B) 高层 1v1 TacticalHierarchySelfplay: SPO + GatedAttentionGRU 短闭环 + 1v1/partial_obs 评估器加载
    c_dir = tmp_path / "tactical_1v1_run"
    c_args = gru_attn_args(
        policy_loss="spo",
        n_rollout_threads=1,
        n_eval_rollout_threads=1,
        num_env_steps=6,
        buffer_size=3,
        data_chunk_length=3,
        num_mini_batch=1,
        use_selfplay=True,
        n_choose_opponents=1,
        use_eval=True,
        eval_episodes=1,
        eval_interval=1,
    )
    c_args.env_name = "SingleCombat"
    c_args.scenario_name = "1v1/NoWeapon/TacticalHierarchySelfplay"

    def make_c_env():
        env = SingleCombatRadarDegradationEnv("1v1/NoWeapon/TacticalHierarchySelfplay", radar_level=1)
        for cond in env.task.termination_conditions:
            if hasattr(cond, "max_steps"):
                cond.max_steps = 3
        return env

    c_envs, c_eval_envs = DummyVecEnv([make_c_env]), DummyVecEnv([make_c_env])
    try:
        sp_runner = SelfplayJSBSimRunner(
            dict(all_args=c_args, envs=c_envs, eval_envs=c_eval_envs,
                 device=torch.device("cpu"), render_mode="txt", run_dir=c_dir)
        )
        sp_runner.run()
        assert (c_dir / "actor_latest.pt").is_file()

        base_1v1 = po_eval.load_base_eval_module()
        po_eval.configure_base_module(base_1v1)
        lowlevel = base_1v1.load_lowlevel_policy(torch.device("cpu"))
        ctrl_a = base_1v1.load_controller(
            "actor_a", c_dir / "actor_latest.pt",
            "1v1/NoWeapon/TacticalHierarchySelfplay", lowlevel, torch.device("cpu"),
        )
        ctrl_b = base_1v1.load_controller(
            "actor_b", c_dir / "actor_latest.pt",
            "1v1/NoWeapon/TacticalHierarchySelfplay", lowlevel, torch.device("cpu"),
        )
        assert ctrl_a.actor.memory_encoder == "gru_attention"
        direct_env = base_1v1.SingleCombatEnv("1v1/NoWeapon/Selfplay")
        direct_env.seed(1)
        for cond in direct_env.task.termination_conditions:
            if hasattr(cond, "max_steps"):
                cond.max_steps = 3
        try:
            base_1v1.validate_setup(direct_env, [ctrl_a, ctrl_b])
            ep_row = po_eval.run_episode(
                base_1v1, po_eval.RELATIVE_CONDITIONS[1], 0,
                direct_env, ctrl_a, ctrl_b, tmp_path / "po_out",
            )
            assert ep_row["steps"] == 3
            assert np.isfinite(ep_row["actor_a_reward"])
        finally:
            direct_env.close()
    finally:
        c_envs.close()
        c_eval_envs.close()
