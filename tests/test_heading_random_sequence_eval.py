from types import SimpleNamespace

import pytest
import torch

from experiments import heading_observation_degradation_eval as evaluation


@pytest.mark.parametrize("miss_at_step", [None, 4, 12])
def test_random_sequence_retains_policy_state_and_checks_each_stage(monkeypatch, miss_at_step):
    heading = evaluation.heading
    env = heading.SingleControlEnv(heading.SCENARIO_NAME)
    actor = SimpleNamespace(memory_encoder="mlp", state_shape=(1, 1))
    device = torch.device("cpu")
    state_history = []
    get_state_record = heading.get_state_record

    def predict_action(actor, obs, state, masks):
        state_history.append(state.item())
        return torch.tensor([[20, 20, 20, 15]]), state + 1

    def record_state(env):
        row = get_state_record(env)
        row["heading_error_deg"] = 11.0 if env.current_step == miss_at_step else 0.0
        row["altitude_error_m"] = 0.0
        row["speed_error_mps"] = 0.0
        return row

    monkeypatch.setattr(heading, "predict_action", predict_action)
    monkeypatch.setattr(heading, "get_state_record", record_state)
    try:
        row, stages = evaluation.run_trial(
            env, actor, evaluation.CONDITIONS[-1], trial_id=1, seed=41001,
            command=None, horizon_steps=12, device=device, training_sequence=True,
        )
    finally:
        env.close()

    steps = 12 if miss_at_step is None else miss_at_step
    assert row["num_steps"] == steps
    assert row["completed_stages"] == steps // 2
    assert state_history == list(range(steps))
    assert row["success"] == (miss_at_step is None)
    assert row["done"] == (steps < 12)
    if miss_at_step is not None:
        assert row["termination_reason"] == stages[-1]["termination_reason"] == "unreach_heading"
        assert not stages[-1]["success"]


def test_random_targets_repeat_with_same_seed_and_build_on_previous_target():
    heading = evaluation.heading
    env = heading.SingleControlEnv(heading.SCENARIO_NAME)
    target_keys = ("target_heading_deg", "target_altitude_m", "target_speed_mps")

    def sample_stages():
        env.seed(41001)
        env.reset()
        return list(evaluation.iter_training_stages(env, horizon_steps=12))

    try:
        stages = sample_stages()
        assert stages == sample_stages()
    finally:
        env.close()

    assert [(stage.start_step, stage.end_step) for stage in stages] == [(i, i + 2) for i in range(0, 12, 2)]
    for previous, stage in zip(stages, stages[1:]):
        prev_heading, prev_altitude, prev_speed = (previous.target_info[key] for key in target_keys)
        assert tuple(stage.target_info[key] for key in target_keys) == pytest.approx((
            (prev_heading + stage.command.delta_heading_deg) % 360.0,
            max(prev_altitude + stage.command.delta_altitude_m, 15000 * heading.FT_TO_M),
            max(prev_speed + stage.command.delta_velocity_mps, 120.0),
        ))
