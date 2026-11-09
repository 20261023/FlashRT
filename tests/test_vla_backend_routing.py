"""CPU candidate-to-frontend binding contracts, without loading model weights."""

from types import SimpleNamespace

import pytest

from flash_rt.vla import BackendConfig, load_backend
from flash_rt.vla import registry


class MinimalFrontend:
    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs

    def set_prompt(self, prompt):
        pass

    def infer(self, observation):
        return {'actions': []}


@pytest.mark.parametrize('model, framework, name', [
    ('pi0', 'torch', 'Pi0TorchFrontendRtx'),
    ('pi0', 'jax', 'Pi0JaxFrontendRtx'),
    ('pi05', 'torch', 'Pi05TorchFrontendRtx'),
    ('pi05', 'jax', 'Pi05JaxFrontendRtx'),
    ('hyvla', 'torch', 'HyVLAFlowFrontend'),
])
def test_requested_sampler_reaches_concrete_frontend(monkeypatch, model, framework, name):
    from flash_rt.sampling import ProbeFlowConfig
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module',
                        lambda module: SimpleNamespace(**{name: MinimalFrontend}))
    policy = ProbeFlowConfig(epsilon=0.004, max_steps=6)
    loaded = load_backend(BackendConfig(model=model, checkpoint='/weights',
                                        framework=framework, sampler='probeflow',
                                        probeflow=policy))
    assert loaded.frontend.kwargs['sampler'] == 'probeflow'
    assert loaded.frontend.kwargs['probeflow_config'] is policy
    if model in ('pi0', 'pi05'):
        assert loaded.frontend.kwargs['use_fp8'] is False
    if model == 'pi0':
        assert loaded.frontend.kwargs['use_fp8_decoder'] is False


def test_fixed_euler_budget_is_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(
        Pi0TorchFrontendRtx=MinimalFrontend))
    loaded = load_backend(BackendConfig(model='pi0', checkpoint='/weights', num_steps=5))
    assert loaded.frontend.kwargs['num_steps'] == 5


@pytest.mark.parametrize('kwargs', [
    {'model': 'hyvla', 'framework': 'jax'},
    {'model': 'pi0', 'precision': 'bf16'},
    {'model': 'pi05', 'options': {'use_fp8': True}},
])
def test_unimplemented_request_is_rejected_before_device_or_import(monkeypatch, kwargs):
    def unexpected(*args):
        raise AssertionError('must reject before touching CUDA or importing model')
    monkeypatch.setattr(registry, '_cuda_target', unexpected)
    monkeypatch.setattr(registry, 'import_module', unexpected)
    with pytest.raises(ValueError):
        load_backend(BackendConfig(checkpoint='/weights', **kwargs))


@pytest.mark.parametrize('name', [
    'FVK_PI05_RTX_FORCE_INT8', 'FVK_PI05_RTX_INT8_ENCODER_ONLY',
    'FVK_PI05_RTX_INT8_VISION',
])
def test_bf16_euler_rejects_native_quantization_environment_before_device(monkeypatch, name):
    monkeypatch.setenv(name, '1')
    def unexpected(*args):
        raise AssertionError('must reject conflicting precision before CUDA/import')
    monkeypatch.setattr(registry, '_cuda_target', unexpected)
    monkeypatch.setattr(registry, 'import_module', unexpected)
    with pytest.raises(ValueError, match=name):
        load_backend(BackendConfig(model='pi05', checkpoint='/weights'))


def test_fastwam_binds_its_own_partial_native_path_and_explicit_sigma_grid(monkeypatch):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(
        FastWAMTorchFrontend=MinimalFrontend))
    loaded = load_backend(BackendConfig(
        model='fastwam', checkpoint='/weights/policy.pt', num_steps=12,
        options={'source_dir': '/sources/FastWAM', 'model_config': '/weights/model.json',
                 'backbone_dir': '/weights/wan', 'sigma_shift': 5.0, 'action_horizon': 32}))
    assert loaded.frontend.kwargs['native_linear'] is True
    assert loaded.frontend.kwargs['sigma_shift'] == 5.0
    assert loaded.frontend.kwargs['num_inference_steps'] == 12
    assert loaded.frontend.kwargs['action_horizon'] == 32
    assert 'num_steps' not in loaded.frontend.kwargs


def test_fastwam_requires_schedule_and_geometry_before_device_probe(monkeypatch):
    def unexpected(*args):
        raise AssertionError('incomplete FastWAM request must fail before CUDA/import')
    monkeypatch.setattr(registry, '_cuda_target', unexpected)
    with pytest.raises(ValueError, match='FastWAM requires'):
        load_backend(BackendConfig(model='fastwam', checkpoint='/weights/policy.pt'))


def test_pi05_post_vision_pooling_reaches_frontend_as_a_conditional_knob(monkeypatch):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(
        Pi05TorchFrontendRtx=MinimalFrontend))
    model = load_backend(BackendConfig(model='pi05', checkpoint='/weights',
                                       options={'vision_pool_factor': 2}))
    assert model.frontend.kwargs['vision_pool_factor'] == 2


def test_hyvla_official_local_profile_reaches_frontend(monkeypatch):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(
        HyVLAFlowFrontend=MinimalFrontend))
    backend = load_backend(BackendConfig(model='hyvla', checkpoint='/weights',
                                         options={'numerical_profile': 'official_local_fp32'}))
    assert backend.frontend.kwargs['numerical_profile'] == 'official_local_fp32'


def test_fastwam_cached_plan_is_explicitly_selected_without_changing_model_options(monkeypatch):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(
        FastWAMTorchFrontend=MinimalFrontend))
    backend = load_backend(BackendConfig(model='fastwam', checkpoint='/weights', num_steps=2,
        options={'model_config': '/model.json', 'backbone_dir': '/wan',
                 'sigma_shift': 5., 'action_horizon': 32, 'native_linear_backend': 'cached'}))
    assert backend.frontend.kwargs['native_linear_backend'] == 'cached'
    assert backend.frontend.kwargs['num_inference_steps'] == 2


@pytest.mark.parametrize('kind,native', [('torch', False), ('flashrt', True)])
def test_fastwam_graph_routes_requested_engine_and_returns_honest_backend_label(monkeypatch, kind, native):
    monkeypatch.setattr(registry, '_cuda_target', lambda value: 'cuda_sm80')
    monkeypatch.setattr(registry, 'import_module', lambda module: SimpleNamespace(FastWAMTorchFrontend=MinimalFrontend))
    options = {'model_config': '/model.json', 'backbone_dir': '/wan', 'sigma_shift': 5.,
               'action_horizon': 32, 'use_cuda_graph': True}
    if native: options['native_linear_backend'] = 'cached'
    backend = load_backend(BackendConfig(model='fastwam', checkpoint='/weights', backend=kind,
                                        options=options, num_steps=10))
    assert backend.frontend.kwargs['native_linear'] is native
    assert backend.frontend.kwargs['use_cuda_graph'] is True
    assert backend.frontend.kwargs['num_inference_steps'] == 10
    result = backend.infer({'context': [1]})
    assert result['backend_metadata']['backend'] == kind


@pytest.mark.parametrize('kind,extra', [
    ('torch', {'native_linear_backend': 'cached'}),
    ('torch', {'native_linear_backend': 'legacy'}),
    ('flashrt', {'use_cuda_graph': True}),
    ('flashrt', {'use_cuda_graph': 'true', 'native_linear_backend': 'cached'}),
])
def test_fastwam_graph_engine_conflicts_reject_before_device_and_model(monkeypatch, kind, extra):
    def forbidden(*args): raise AssertionError('must reject before CUDA or model import')
    monkeypatch.setattr(registry, '_cuda_target', forbidden)
    monkeypatch.setattr(registry, 'import_module', forbidden)
    options = {'model_config': '/missing', 'backbone_dir': '/missing', 'sigma_shift': 5., 'action_horizon': 32}
    options.update(extra)
    with pytest.raises(ValueError, match='native_linear_backend|cached|use_cuda_graph'):
        load_backend(BackendConfig(model='fastwam', checkpoint='/missing', backend=kind, options=options))
