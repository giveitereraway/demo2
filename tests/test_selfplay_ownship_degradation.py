from types import SimpleNamespace

import numpy as np
import pytest

from config import get_config
from envs.JSBSim.envs.selfplay_observation import (
    JOINT_DEGRADATION_MAX_LEVEL,
    JOINT_TRAIN_PROBABILITIES,
    OWN_SHIP_TRAIN_PROBABILITIES,
    OwnshipObservationDegrader,
    SelfplayJointDegradationEnv,
    validate_unarmed_selfplay,
)
from experiments.partial_observability_gru_eval import (
    joint_conditions,
    ownship_conditions,
    parse_cli_args,
    radar_conditions,
)
from scripts.train.train_jsbsim import parse_args as parse_train_args


def test_ownship_degradation_holds_and_delays_only_ownship_state():
    initial = np.arange(15, dtype=np.float32) / 10
    clean = OwnshipObservationDegrader(0, initial, np.random.default_rng(1))
    np.testing.assert_array_equal(clean.transform(initial), initial)

    draws = iter([1.0, 0.0, 1.0, 1.0])
    rng = SimpleNamespace(
        random=lambda: next(draws),
        normal=lambda mean, std, size: np.full(size, mean + std),
    )
    degraded = OwnshipObservationDegrader(2, initial, rng)
    frames = [initial + i for i in range(1, 5)]
    outputs = [degraded.transform(frame) for frame in frames]

    # 两步延迟先输出初始观测，丢帧使第一帧测量再保持一步。
    np.testing.assert_array_equal(outputs[0][:9], initial[:9])
    np.testing.assert_array_equal(outputs[1][:9], initial[:9])
    np.testing.assert_allclose(outputs[2][:9], frames[0][:9] + 0.03)
    np.testing.assert_array_equal(outputs[3][:9], outputs[2][:9])
    for output, frame in zip(outputs, frames):
        np.testing.assert_array_equal(output[9:], frame[9:])
    assert degraded.actual_dropout_rate == 0.25


def test_ownship_degradation_requires_artillery_disabled():
    validate_unarmed_selfplay(SimpleNamespace(task="singlecombat", use_artillery=False))
    with pytest.raises(ValueError):
        validate_unarmed_selfplay(SimpleNamespace(task="singlecombat", use_artillery=True))


def test_ownship_full_suite_increases_each_factor_and_keeps_quick_mode():
    conditions = ownship_conditions(quick=False)
    assert [condition.condition_id for condition in conditions] == [f"E{i}" for i in range(7)]
    specs = [condition.degradation for condition in conditions]
    assert [spec.delay_steps for spec in specs] == list(range(7))
    assert [spec.dropout_probability for spec in specs] == pytest.approx([i / 10 for i in range(7)])
    assert [spec.noise_std for spec in specs] == pytest.approx([0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11])
    assert ownship_conditions(quick=True) == (conditions[0], conditions[2])
    assert len(OWN_SHIP_TRAIN_PROBABILITIES) == 3


def test_joint_and_radar_conditions_align_with_e0_to_e4_and_quick_mode():
    radar = radar_conditions(quick=False)
    joint = joint_conditions(quick=False)
    expected_ids = [f"E{i}" for i in range(JOINT_DEGRADATION_MAX_LEVEL + 1)]
    assert [c.condition_id for c in radar] == expected_ids
    assert [c.condition_id for c in joint] == expected_ids
    assert all(c.radar_level == i and c.ownship_level is None for i, c in enumerate(radar))
    assert all(c.ownship_level == i and c.radar_level == i for i, c in enumerate(joint))
    assert radar_conditions(quick=True) == (radar[0], radar[2])
    assert joint_conditions(quick=True) == (joint[0], joint[2])
    assert len(JOINT_TRAIN_PROBABILITIES) == 3

    cli = parse_cli_args(["--degradation-mode", "joint", "--degrade-both", "--quick"])
    assert cli.degradation_mode == "joint"
    assert cli.degrade_both is True
    assert cli.quick is True


def test_joint_degradation_cli_validation_and_mutual_exclusion():
    base_cli = [
        "--env-name", "SingleCombat",
        "--algorithm-name", "ppo",
        "--scenario-name", "1v1/NoWeapon/Selfplay",
        "--use-selfplay",
        "--policy-loss", "spo",
        "--memory-encoder", "igru",
        "--selfplay-joint-degradation",
    ]
    parsed = parse_train_args(base_cli, get_config())
    assert parsed.selfplay_joint_degradation is True
    assert parsed.policy_loss == "spo"
    assert parsed.memory_encoder == "igru"

    with pytest.raises(SystemExit):
        parse_train_args(base_cli + ["--selfplay-ownship-degradation"], get_config())
    with pytest.raises(SystemExit):
        parse_train_args(base_cli + ["--combat-radar-degradation"], get_config())
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--env-name", "SingleCombat",
                "--algorithm-name", "ppo",
                "--scenario-name", "1v1/NoWeapon/HierarchySelfplay",
                "--use-selfplay",
                "--selfplay-joint-degradation",
            ],
            get_config(),
        )


def test_joint_degradation_env_rejects_invalid_scenario_or_level():
    with pytest.raises(ValueError):
        SelfplayJointDegradationEnv("1v1/NoWeapon/HierarchySelfplay")
    with pytest.raises(ValueError):
        SelfplayJointDegradationEnv("1v1/NoWeapon/Selfplay", degradation_level=JOINT_DEGRADATION_MAX_LEVEL + 1)


def test_joint_degradation_env_step_perturbs_both_ownship_and_radar_obs():
    pytest.importorskip("jsbsim")
    env = SelfplayJointDegradationEnv("1v1/NoWeapon/Selfplay", degradation_level=2)
    env.seed(2026)
    try:
        obs0 = env.reset()
        assert obs0.shape == (2, 15)
        cached_a = env.get_obs()
        cached_b = env.get_obs()
        for agent_id in env.agents:
            np.testing.assert_array_equal(cached_a[agent_id], cached_b[agent_id])

        # 连续推进数步越过 E2 的 2 步时延后，验证本机与敌机雷达观测均被退化且无 NaN。
        action = np.array([[20, 20, 20, 15], [20, 20, 20, 15]], dtype=np.int64)
        for _ in range(6):
            obs, _, _, _ = env.step(action)
            assert np.isfinite(obs).all()
        assert all(env.get_enemy_radar_track(agent_id) is not None for agent_id in env.agents)
        assert len(env._ownship_degraders) == 2
    finally:
        env.close()
