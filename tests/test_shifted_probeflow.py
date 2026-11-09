"""Shifted-grid arithmetic contracts. Written but not run in this task."""

import pytest
import torch

from flash_rt.sampling import ProbeFlowConfig, ShiftedSigmaSchedule
from flash_rt.sampling.torch_probeflow import sample_probeflow_torch_


def test_shift_five_two_step_midpoint_is_five_sixths():
    schedule = ShiftedSigmaSchedule(5.0)
    assert schedule(0.0) == 0.0
    assert schedule(1.0) == 1.0
    assert schedule(0.5) == pytest.approx(5.0 / 6.0)


def test_two_probe_state_uses_sigma_delta_not_uniform_clock_delta():
    schedule = ShiftedSigmaSchedule(5.0)
    state = torch.zeros(1, 2)
    seen = []
    def evaluate(clock):
        seen.append((clock, state.clone()))
        return torch.ones_like(state)
    result = sample_probeflow_torch_(state, evaluate, ProbeFlowConfig(), time_map=schedule)
    assert seen[1][0] == 0.5
    torch.testing.assert_close(seen[1][1], torch.full((1, 2), 1.0 / 6.0))
    torch.testing.assert_close(state, torch.ones(1, 2))
    assert result['nfe'] == 2
    assert result['integration_times'][1] == pytest.approx(5.0 / 6.0)


@pytest.mark.parametrize('shift', [0, -1, float('nan'), float('inf'), True])
def test_bad_shift_rejected_before_model_use(shift):
    with pytest.raises(ValueError):
        ShiftedSigmaSchedule(shift)


def test_shifted_dense_restart_discards_probe_and_uses_all_sigma_intervals():
    state = torch.zeros(1, 2)
    calls = []
    def evaluate(clock):
        calls.append(clock)
        # Force the dense branch using the first two velocity directions.
        if len(calls) == 2:
            return torch.tensor([[0.0, 1.0]])
        return torch.tensor([[1.0, 0.0]])
    metadata = sample_probeflow_torch_(
        state, evaluate, ProbeFlowConfig(epsilon=0.001, max_steps=4),
        time_map=ShiftedSigmaSchedule(5.0))
    assert calls == [1.0, 0.5, 0.75, 0.5, 0.25]
    assert metadata['nfe'] == 5
    torch.testing.assert_close(state, torch.tensor([[1.0, 0.0]]))
