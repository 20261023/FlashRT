"""CPU-only boundaries: select the sampler before loading weights or CUDA."""
from types import SimpleNamespace

import pytest

from flash_rt.models.pi05.probeflow import ProbeFlowConfig
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx, _precompute_decoder_styles
from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline


@pytest.mark.parametrize('frontend', [Pi05TorchFrontendRtx, Pi05JaxFrontendRtx])
@pytest.mark.parametrize('kwargs,match', [
    ({'sampler': 'typo'}, 'sampler'),
    ({'sampler': 'euler', 'probeflow_config': ProbeFlowConfig()}, 'probeflow_config'),
    ({'sampler': 'probeflow', 'use_fp8': True}, 'BF16'),
    ({'sampler': 'probeflow', 'use_fp8': False, 'cache_frames': 2}, 'cache_frames'),
    ({'sampler': 'probeflow', 'use_fp8': False, 'sde': True}, 'sde'),
    ({'sampler': 'probeflow', 'use_fp8': False, 'denoise_trace': True}, 'denoise_trace'),
    ({'sampler': 'probeflow', 'use_fp8': False, 'probeflow_config': {}}, 'ProbeFlowConfig'),
])
def test_sampler_validation_happens_before_checkpoint_or_device_access(frontend, kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        frontend('/definitely/not/a/checkpoint', **kwargs)


@pytest.mark.parametrize('times', [[], [float('nan')], [0.], [1.1]])
def test_invalid_explicit_times_raise_before_cuda_allocation(times):
    with pytest.raises(ValueError, match='timesteps'):
        _precompute_decoder_styles({}, 50, timesteps=times)


def _decoder_fixture():
    pipeline = Pi05Pipeline.__new__(Pi05Pipeline)
    pointer = lambda value: SimpleNamespace(ptr=SimpleNamespace(value=value))
    pipeline.bufs = {name: pointer(i + 1) for i, name in enumerate([
        'diffusion_noise', 'decoder_x', 'x_normed_buf', 'gate_buf', 'decoder_action_buf'])}
    pipeline._rms_ones_dec = pointer(100)
    pipeline.weights = {name: 200 for name in [
        'decoder_action_in_proj_w', 'decoder_action_in_proj_b',
        'decoder_action_out_proj_w', 'decoder_action_out_proj_b']}
    pipeline.use_fp8_decoder = False
    pipeline.fp8_calibrated = False
    pipeline.use_int8_decoder = False
    pipeline.denoise_trace = False
    pipeline.sde = False
    pipeline.encoder_seq_len = 2
    pipeline.chunk_size = 1
    pipeline.num_steps = 5
    pipeline.updates = []
    pipeline.layers = []
    pipeline.fvk = SimpleNamespace(
        ada_rms_norm_style=lambda *a, **k: None,
        residual_add=lambda *a, **k: pipeline.updates.append(a))
    pipeline.gemm = SimpleNamespace(bf16_nn=lambda *a, **k: None)
    pipeline._copy_rtc_prefix = lambda *a: None
    pipeline._bias_add_bf16 = lambda *a: None
    pipeline._style_slice_ptr = lambda name, step, *args: step
    pipeline._decoder_layer = lambda layer, step, *args: pipeline.layers.append((layer, step))
    return pipeline


def test_native_velocity_evaluation_does_not_apply_a_hidden_euler_step():
    pipeline = _decoder_fixture()
    pipeline.transformer_decoder(stream=17, _step_indices=(3,), _integrate=False)
    assert pipeline.layers == [(i, 3) for i in range(18)]
    assert pipeline.updates == []


def test_default_decoder_still_integrates_each_fixed_step():
    pipeline = _decoder_fixture()
    pipeline.transformer_decoder(stream=17)
    assert len(pipeline.layers) == 5 * 18
    assert len(pipeline.updates) == 5


@pytest.mark.parametrize('method,kwargs', [
    ('set_rl_mode', {}), ('set_batched_mode', {}), ('reload_weights', {'source': {}}),
])
def test_unsupported_modes_raise_before_mutating_frontend(method, kwargs):
    from threading import RLock
    frontend = Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
    frontend._probeflow_config = ProbeFlowConfig()
    frontend._lifecycle_lock = RLock()
    frontend._reload_failed = False
    with pytest.raises(NotImplementedError, match='ProbeFlow'):
        getattr(frontend, method)(**kwargs)
    assert not hasattr(frontend, '_rl_config')
    assert not hasattr(frontend, '_batched_active')


@pytest.mark.parametrize('frontend_class', [Pi05TorchFrontendRtx, Pi05JaxFrontendRtx])
def test_select_batch_size_rejects_before_losing_live_single_sample_state(frontend_class):
    from threading import RLock
    frontend = frontend_class.__new__(frontend_class)
    frontend._probeflow_config = ProbeFlowConfig()
    frontend._lifecycle_lock = RLock()
    frontend._reload_failed = False
    frontend.pipeline = object()
    frontend.attn_backend = object()
    frontend._last_prompt_call = ('single', 'keep this prompt', None)
    frontend._batch_prompt_texts = object()
    frontend._batched_active = False
    frontend._batch_size = 1
    frontend.current_prompt_len = 42
    frontend.graph_recorded = True
    frontend.calibrated = True
    original = dict(frontend.__dict__)
    with pytest.raises(NotImplementedError, match='ProbeFlow'):
        frontend.select_batch_size(2)
    assert frontend.__dict__ == original
