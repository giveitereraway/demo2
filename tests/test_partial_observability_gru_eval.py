from types import SimpleNamespace

import numpy as np
import pytest

from experiments.partial_observability_gru_eval import (
    AO_INDEX,
    DISTANCE_INDEX,
    RELATIVE_FEATURE_SLICE,
    TA_INDEX,
    DegradationConfig,
    ObservationDegrader,
    action_continuity_metrics,
    maybe_reset_actor_hidden,
)


def make_obs(relative_value: float = 1.0) -> np.ndarray:
    obs = np.zeros(15, dtype=np.float32)
    obs[RELATIVE_FEATURE_SLICE] = relative_value
    obs[AO_INDEX] = 0.5
    obs[TA_INDEX] = 1.0
    obs[DISTANCE_INDEX] = relative_value
    return obs


def test_no_degradation_keeps_observation_unchanged():
    obs = make_obs()
    degrader = ObservationDegrader(DegradationConfig(), np.random.default_rng(1))
    degrader.reset(obs)

    actual = degrader.transform(obs)

    np.testing.assert_allclose(actual, obs)
    assert degrader.actual_dropout_rate == 0.0


def test_full_dropout_holds_last_valid_relative_observation():
    initial = make_obs(relative_value=1.0)
    updated = make_obs(relative_value=2.0)
    degrader = ObservationDegrader(
        DegradationConfig(dropout_probability=1.0),
        np.random.default_rng(2),
    )
    degrader.reset(initial)

    actual = degrader.transform(updated)

    np.testing.assert_allclose(
        actual[RELATIVE_FEATURE_SLICE], initial[RELATIVE_FEATURE_SLICE]
    )
    assert degrader.actual_dropout_rate == 1.0


def test_one_step_delay_returns_previous_relative_observation():
    initial = make_obs(relative_value=1.0)
    first = make_obs(relative_value=2.0)
    second = make_obs(relative_value=3.0)
    degrader = ObservationDegrader(
        DegradationConfig(delay_steps=1),
        np.random.default_rng(3),
    )
    degrader.reset(initial)

    first_output = degrader.transform(first)
    second_output = degrader.transform(second)

    np.testing.assert_allclose(
        first_output[RELATIVE_FEATURE_SLICE], initial[RELATIVE_FEATURE_SLICE]
    )
    np.testing.assert_allclose(
        second_output[RELATIVE_FEATURE_SLICE], first[RELATIVE_FEATURE_SLICE]
    )


def test_noise_is_reproducible_and_only_changes_configured_features():
    obs = make_obs(relative_value=1.0)
    config = DegradationConfig(
        angle_noise_std_deg=5.0,
        range_noise_std_m=300.0,
    )
    first = ObservationDegrader(config, np.random.default_rng(4))
    second = ObservationDegrader(config, np.random.default_rng(4))
    first.reset(obs)
    second.reset(obs)

    first_output = first.transform(obs)
    second_output = second.transform(obs)

    np.testing.assert_allclose(first_output, second_output)
    np.testing.assert_allclose(first_output[:9], obs[:9])
    assert first_output[AO_INDEX] != pytest.approx(obs[AO_INDEX])
    assert first_output[TA_INDEX] != pytest.approx(obs[TA_INDEX])
    assert first_output[DISTANCE_INDEX] != pytest.approx(obs[DISTANCE_INDEX])
    np.testing.assert_allclose(first_output[[9, 10, 14]], obs[[9, 10, 14]])


def test_gru_hidden_reset_obeys_interval():
    controller = SimpleNamespace(rnn_states=np.ones((1, 1, 4), dtype=np.float32))

    assert not maybe_reset_actor_hidden(controller, step=4, interval=5)
    np.testing.assert_allclose(controller.rnn_states, 1.0)
    assert maybe_reset_actor_hidden(controller, step=5, interval=5)
    np.testing.assert_allclose(controller.rnn_states, 0.0)

    controller.rnn_states.fill(1.0)
    assert maybe_reset_actor_hidden(controller, step=0, interval=1)
    np.testing.assert_allclose(controller.rnn_states, 0.0)


def test_action_continuity_metrics_detect_large_jump():
    actions = [
        np.array([0.0, 0.0, 0.0, 0.5]),
        np.array([0.1, 0.0, 0.0, 0.5]),
        np.array([0.5, 0.0, 0.0, 0.5]),
    ]

    metrics = action_continuity_metrics(actions)

    assert metrics["actor_a_action_total_variation"] == pytest.approx(0.25)
    assert metrics["actor_a_large_action_jump_rate"] == pytest.approx(0.5)
    assert metrics["actor_a_action_second_difference"] == pytest.approx(0.3)
