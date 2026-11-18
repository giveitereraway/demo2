from types import SimpleNamespace

import numpy as np
import pytest

from envs.JSBSim.envs.selfplay_observation import (
    OWN_SHIP_TRAIN_PROBABILITIES,
    OwnshipObservationDegrader,
    validate_unarmed_selfplay,
)
from experiments.partial_observability_gru_eval import ownship_conditions


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
