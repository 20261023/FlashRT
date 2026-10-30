"""Actual tensor scheduler with analytic velocities; no model or GPU required."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flash_rt.models.pi05.probeflow import ProbeFlowConfig, sample_probeflow


def _run(state, config, curved=False):
    from flash_rt.frontends.torch.pi05_probeflow import sample_probeflow_torch_
    times = []

    def evaluate(t):
        times.append(t)
        return torch.tensor([1., 4. * (1. - t) if curved else 0.], dtype=state.dtype)

    metadata = sample_probeflow_torch_(state, evaluate, config)
    return metadata, times


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('curved,steps,nfe,expected', [
    (False, 2, 2, [1., 0.]), (True, 4, 5, [1., 1.5]),
])
def test_actual_tensor_updates_match_analytic_reference(dtype, curved, steps, nfe, expected):
    state = torch.zeros(2, dtype=dtype)
    metadata, times = _run(state, ProbeFlowConfig(epsilon=0.1, max_steps=4), curved)
    np.testing.assert_allclose(state.float().numpy(), expected)
    assert metadata['scheduled_steps'] == steps
    assert metadata['nfe'] == nfe == len(times)
    assert times == ([1., 0.5] if not curved else [1., 0.5, 0.75, 0.5, 0.25])


def test_tensor_schedule_changes_between_calls_and_does_not_keep_warmup_route():
    state = torch.zeros(2)
    first, _ = _run(state, ProbeFlowConfig(epsilon=0.1, max_steps=4), False)
    state.zero_()
    second, _ = _run(state, ProbeFlowConfig(epsilon=0.1, max_steps=4), True)
    assert (first['scheduled_steps'], second['scheduled_steps']) == (2, 4)


def test_tensor_dense_path_matches_state_dependent_reference():
    from flash_rt.frontends.torch.pi05_probeflow import sample_probeflow_torch_
    config = ProbeFlowConfig(epsilon=0.05, max_steps=6, probe_horizon=0.3)
    noise = np.array([0.2, -0.4], dtype=np.float32)
    expected, oracle = sample_probeflow(
        noise, lambda x, t: -np.array([1. + x[0], 4. * (1. - t) - x[1]]), config)
    state = torch.from_numpy(noise.copy())
    result = sample_probeflow_torch_(state, lambda t: torch.stack([
        1. + state[0], 4. * (1. - t) - state[1]]), config)
    np.testing.assert_allclose(state.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert result['nfe'] == oracle['nfe']


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_tensor_nonfinite_inputs_and_velocities_are_rejected(bad):
    from flash_rt.frontends.torch.pi05_probeflow import sample_probeflow_torch_
    with pytest.raises(ValueError, match='finite'):
        sample_probeflow_torch_(torch.tensor([bad]), lambda t: torch.zeros(1), ProbeFlowConfig())
    with pytest.raises(ValueError, match='finite'):
        sample_probeflow_torch_(torch.zeros(1), lambda t: torch.tensor([bad]), ProbeFlowConfig())


def test_adaptive_pipeline_rejects_export_and_cross_frame_kv_reuse():
    from flash_rt.models.pi05.pipeline_rtx_probeflow import Pi05ProbeFlowPipeline
    pipeline = Pi05ProbeFlowPipeline.__new__(Pi05ProbeFlowPipeline)
    for method in ['export_runtime', 'export_model_runtime', 'forward_decode_only']:
        with pytest.raises(NotImplementedError, match='ProbeFlow'):
            getattr(pipeline, method)()


def test_frontend_factory_preserves_euler_and_forwards_adaptive_configuration(monkeypatch):
    from flash_rt.frontends.torch import pi05_rtx
    from flash_rt.models.pi05 import pipeline_rtx_probeflow
    config = ProbeFlowConfig(max_steps=4)
    monkeypatch.setattr(pi05_rtx, 'Pi05Pipeline', lambda **kwargs: ('euler', kwargs))
    monkeypatch.setattr(pipeline_rtx_probeflow, 'Pi05ProbeFlowPipeline',
                        lambda **kwargs: ('probeflow', kwargs))
    frontend = SimpleNamespace(_probeflow_config=None)
    factory = pi05_rtx.Pi05TorchFrontendRtx._create_pipeline
    assert factory(frontend, num_steps=10) == ('euler', {'num_steps': 10})
    frontend._probeflow_config = config
    kind, kwargs = factory(frontend, num_steps=10)
    assert kind == 'probeflow'
    assert kwargs['probeflow_config'] is config
    assert kwargs['num_steps'] == len(config.timesteps())


@pytest.mark.parametrize('horizon', [1. - 1e-13, float(np.nextafter(1., 0.))])
def test_captured_velocity_lookup_does_not_round_positive_time_to_zero(horizon):
    from flash_rt.models.pi05.pipeline_rtx_probeflow import Pi05ProbeFlowPipeline
    config = ProbeFlowConfig(probe_horizon=horizon)
    pipeline = Pi05ProbeFlowPipeline.__new__(Pi05ProbeFlowPipeline)
    pipeline._time_indices = {t: i for i, t in enumerate(config.timesteps())}
    replayed = []
    pipeline._velocity_graphs = {
        index: SimpleNamespace(replay=lambda stream, i=index: replayed.append(i))
        for index in pipeline._time_indices.values()}
    pipeline._velocity_view = object()
    query_time = 1. - horizon
    result = pipeline._evaluate(query_time, stream=17, use_graphs=True)
    assert result is pipeline._velocity_view
    assert replayed == [pipeline._time_indices[query_time]]
