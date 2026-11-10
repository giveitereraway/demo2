import numpy as np
import pytest

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecontrol_env import HeadingMeasurementNoiseEnv, SingleControlEnv
from scripts.train.train_jsbsim import get_config, make_eval_env, make_train_env, parse_args


def test_heading_measurement_noise_keeps_observation_channels_consistent():
    env = SingleControlEnv("1/heading")
    try:
        env.seed(1)
        env.reset()
        agent_id = env.ego_ids[0]
        sim = env.agents[agent_id]
        heading = sim.get_property_value(c.attitude_psi_deg)
        sim.set_property_value(c.target_heading_deg, (heading + 179.0) % 360.0)
        clean = env.get_obs()[agent_id]
        noisy = env.task.get_obs(env, agent_id, measurement_error=(20.0, -4.0, 3.0))
        expected = clean.copy()
        expected[[0, 2, 3, 8]] += [-20.0 / 1000.0, -3.0 / 340.0, 20.0 / 5000.0, 3.0 / 340.0]
        expected[1] = np.deg2rad(-177.0)
        np.testing.assert_allclose(noisy, expected, atol=1e-12)
        np.testing.assert_array_equal(env.get_obs()[agent_id], clean)
    finally:
        env.close()


def test_heading_noise_changes_observation_but_keeps_flight_and_scoring():
    clean = SingleControlEnv("1/heading")
    noisy = HeadingMeasurementNoiseEnv("1/heading", noise_level=3)
    try:
        clean.seed(12)
        noisy.seed(12)
        clean_obs = clean.reset()
        noisy_obs = noisy.reset()
        agent_id = noisy.ego_ids[0]
        assert not np.array_equal(noisy_obs, clean_obs)
        np.testing.assert_array_equal(noisy.get_obs()[agent_id], noisy_obs[0])
        for _ in range(3):
            action = np.array([[20, 20, 20, 15]])
            clean_obs, clean_reward, clean_done, _ = clean.step(action)
            noisy_obs, noisy_reward, noisy_done, _ = noisy.step(action)
            clean_current = clean.get_obs()[agent_id]
            np.testing.assert_array_equal(noisy.task.get_obs(noisy, agent_id), clean_current)
            np.testing.assert_array_equal(noisy_reward, clean_reward)
            np.testing.assert_array_equal(noisy_done, clean_done)
            assert not np.array_equal(noisy_obs, clean_obs)
            # 若本步切换了目标，绝对高度的测量偏差仍应复用。
            assert noisy.get_obs()[agent_id][3] == noisy_obs[0, 3]
    finally:
        noisy.close()
        clean.close()


def test_heading_noise_episode_stream_is_reproducible():
    first = HeadingMeasurementNoiseEnv("1/heading", noise_level=2)
    second = HeadingMeasurementNoiseEnv("1/heading", noise_level=2)
    try:
        first.seed(8)
        second.seed(8)
        np.testing.assert_array_equal(first.reset(), second.reset())
        first.step(np.array([[20, 20, 20, 15]]))
        first.reset()
        second.reset()

        def altitude_bias(env):
            agent_id = env.ego_ids[0]
            return env.get_obs()[agent_id][3] - env.task.get_obs(env, agent_id)[3]

        assert altitude_bias(first) == pytest.approx(altitude_bias(second), abs=1e-12)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("tracking", [False, True])
def test_heading_periodic_eval_observation(noisy, tracking):
    args = parse_args([
        "--env-name", "SingleControl", "--scenario-name", "1/heading",
        "--use-eval",
        *(["--heading-measurement-noise"] if noisy else []),
        *(["--heading-first-target"] if tracking else []),
        "--n-rollout-threads", "1", "--n-eval-rollout-threads", "1",
    ], get_config())
    train = make_train_env(args)
    evaluation = make_eval_env(args)
    try:
        train_type = HeadingMeasurementNoiseEnv if noisy else SingleControlEnv
        assert type(train.envs[0]) is train_type
        env = evaluation.envs[0]
        assert getattr(env, "record_heading_tracking", False) == tracking
        eval_obs = evaluation.reset(seeds=[2026])
        assert train.reset().shape == eval_obs.shape == (1, 1, 12)
        agent_id = env.ego_ids[0]
        clean_obs = env.task.get_obs(env, agent_id)
        if noisy:
            assert env.noise_level == 2
            assert env._noise_std == (25.0, 3.0, 3.0)
            assert not np.array_equal(eval_obs[0, 0], clean_obs)
        else:
            np.testing.assert_array_equal(eval_obs[0, 0], clean_obs)
        np.testing.assert_array_equal(evaluation.reset(seeds=[2026]), eval_obs)
    finally:
        train.close()
        evaluation.close()


def test_heading_noise_rejects_other_scenarios():
    with pytest.raises(SystemExit):
        parse_args(["--env-name", "SingleCombat", "--heading-measurement-noise"], get_config())
