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
    plot_summaries,
    maybe_reset_actor_hidden,
    resolve_actor_path,
    scenario_config_path,
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


def test_actor_directory_resolves_actor_latest(tmp_path):
    actor_path = tmp_path / "wandb" / "run" / "files" / "actor_latest.pt"
    actor_path.parent.mkdir(parents=True)
    actor_path.write_bytes(b"checkpoint")

    assert resolve_actor_path(tmp_path) == actor_path.resolve()


def test_core_selfplay_scenario_exists():
    path = scenario_config_path("1v1/NoWeapon/Selfplay")

    assert path.is_file()
    assert path.name == "Selfplay.yaml"


def test_summary_plot_accepts_advantage_confidence_interval(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import experiments.partial_observability_gru_eval as module

    monkeypatch.setattr(module, "SAVE_PLOTS", True)
    monkeypatch.setattr(module, "PLOT_DPI", 40)
    summaries = []
    for index, condition_id in enumerate(("E0", "E1", "E2", "E3", "E4", "E5")):
        value = 0.4 + index * 0.01
        summaries.append(
            {
                "condition_id": condition_id,
                "win_score": value,
                "win_score_ci_low": value - 0.05,
                "win_score_ci_high": value + 0.05,
                "advantage_window_occupancy": value,
                "advantage_window_occupancy_ci_low": value - 0.05,
                "advantage_window_occupancy_ci_high": value + 0.05,
                "actor_a_mean_ao_deg": 30.0,
                "mean_distance_m": 3000.0,
                "actor_a_action_total_variation": 0.2,
            }
        )

    paths = plot_summaries(summaries, tmp_path)

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
