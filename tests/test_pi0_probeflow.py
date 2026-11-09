"""Pi0 sampler seams; written for later execution, not run in this change.

Native CUDA calls are replaced only at the kernel/allocation boundary. The
decoder, frontend validation/factory and shared vector sampler remain real.
"""
from types import SimpleNamespace
from contextlib import nullcontext
import ctypes
import sys

import numpy as np
import pytest
import torch

from flash_rt.frontends.torch.pi0_rtx import (
    Pi0TorchFrontendRtx, _precompute_time_proj_all,
)
from flash_rt.frontends.jax.pi0_rtx import Pi0JaxFrontendRtx
from flash_rt.models.pi0.pipeline_rtx import Pi0Pipeline
from flash_rt.sampling.probeflow import ProbeFlowConfig


HIGH_STREAM = 0x123456789


@pytest.fixture
def typed_cuda_boundary(monkeypatch):
    """Cross a real ctypes ABI boundary without calling a CUDA function."""
    from flash_rt.core.cuda_buffer import _cudart
    expected = {
        'cudaMemcpyAsync': [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_int, ctypes.c_void_p],
        'cudaMemsetAsync': [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                            ctypes.c_void_p],
        'cudaStreamSynchronize': [ctypes.c_void_p],
    }
    for name, arguments in expected.items():
        function = getattr(_cudart, name)
        assert function.argtypes == arguments
        assert function.restype is ctypes.c_int  # CUDA statuses are signed ints.
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    boundary = SimpleNamespace(runtime=_cudart, calls=[], copy_status=0,
                               memset_status=0, copy_fail_on=None, copy_count=0,
                               on_copy=lambda: None)

    @ctypes.CFUNCTYPE(ctypes.c_int, *expected['cudaMemcpyAsync'])
    def copy(dst, src, nbytes, kind, stream):
        boundary.copy_count += 1
        boundary.calls.append(('copy', dst, src, nbytes, kind, stream))
        boundary.on_copy()
        if boundary.copy_fail_on is None or boundary.copy_count == boundary.copy_fail_on:
            return boundary.copy_status
        return 0

    @ctypes.CFUNCTYPE(ctypes.c_int, *expected['cudaMemsetAsync'])
    def memset(dst, value, nbytes, stream):
        boundary.calls.append(('memset', dst, value, nbytes, stream))
        return boundary.memset_status

    @ctypes.CFUNCTYPE(ctypes.c_int, *expected['cudaStreamSynchronize'])
    def synchronize(stream):
        boundary.calls.append(('sync', stream))
        return 0

    monkeypatch.setattr(_cudart, 'cudaMemcpyAsync', copy)
    monkeypatch.setattr(_cudart, 'cudaMemsetAsync', memset)
    monkeypatch.setattr(_cudart, 'cudaStreamSynchronize', synchronize)
    return boundary


@pytest.fixture
def pi0_runtime_objects(monkeypatch, tmp_path, typed_cuda_boundary):
    """Run real constructors, replacing checkpoint/GPU allocations only."""
    import flash_rt
    from flash_rt.frontends.torch import pi0_rtx as torch_frontend
    from flash_rt.frontends.jax import pi0_rtx as jax_frontend
    from flash_rt.hardware.rtx import attn_backend
    from flash_rt.models.pi0 import pipeline_rtx

    def forbid_private_cdll(*args, **kwargs):
        raise AssertionError('Pi0 must reuse the typed CUDA runtime, not open a new CDLL')

    monkeypatch.setattr(ctypes, 'CDLL', forbid_private_cdll)
    next_pointer = 0x234567890

    def buffer(count, dtype):
        nonlocal next_pointer
        next_pointer += 0x10000
        return SimpleNamespace(ptr=ctypes.c_void_p(next_pointer),
                               nbytes=count * np.dtype(dtype).itemsize)

    monkeypatch.setattr(pipeline_rtx.CudaBuffer, 'device_empty', buffer)
    monkeypatch.setattr(pipeline_rtx.CudaBuffer, 'from_numpy',
                        lambda value: buffer(value.size, value.dtype))
    monkeypatch.setattr(Pi0Pipeline, '_allocate_buffers', lambda self: {
        'diffusion_noise': buffer(self.chunk_size * 32, np.float16)})
    monkeypatch.setattr(Pi0Pipeline, '_build_rope_table', lambda self: None)
    monkeypatch.setattr(Pi0Pipeline, '_build_pos_embed_expanded', lambda self: None)
    native = SimpleNamespace(FvkContext=lambda: object(), GemmRunner=lambda: object(),
                             bias_residual_fp16=lambda *args, **kwargs: None)
    attention = SimpleNamespace(get_ptrs=lambda: {'enc_k_layer_stride_bytes': 2})
    pipeline = Pi0Pipeline(None, native, attention, {}, num_views=1, max_prompt_len=1,
                           use_fp8=False, use_fp8_decoder=False)
    monkeypatch.setattr(pipeline_rtx, 'CUDAGraph', lambda: SimpleNamespace(
        begin_capture=lambda handle: None, end_capture=lambda handle: None))
    pipeline.run_pipeline = lambda **kwargs: None

    monkeypatch.setattr(flash_rt, 'flash_rt_kernels', native, raising=False)
    monkeypatch.setattr(torch_frontend, 'RtxFlashAttnBackend', lambda **kwargs: attention)
    monkeypatch.setattr(attn_backend, 'RtxFlashAttnBackend', lambda **kwargs: attention)
    monkeypatch.setattr(Pi0TorchFrontendRtx, '_load_norm_stats', lambda *args: None)
    monkeypatch.setattr(Pi0JaxFrontendRtx, '_load_norm_stats_jax', lambda *args: None)
    raw = {'embedding_weight': object()}
    monkeypatch.setattr(torch_frontend, 'convert_pi0_safetensors', lambda *args, **kwargs: raw)
    monkeypatch.setattr(jax_frontend, 'convert_pi0_orbax', lambda *args, **kwargs: raw)
    monkeypatch.setattr(torch_frontend, '_precompute_time_proj_all', lambda *args, **kwargs: object())
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: object())
    stream = SimpleNamespace(cuda_stream=HIGH_STREAM, wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    (tmp_path / 'model.safetensors').touch()
    options = dict(num_steps=7, use_fp8=False, use_fp8_decoder=False, use_cuda_graph=False)
    return (pipeline, Pi0TorchFrontendRtx(tmp_path, **options),
            Pi0JaxFrontendRtx(tmp_path, **options))


def test_actual_pi0_objects_use_the_shared_pointer_width_runtime(pi0_runtime_objects, typed_cuda_boundary):
    for instance in pi0_runtime_objects:
        assert instance._cudart is typed_cuda_boundary.runtime


def test_high_stream_reaches_native_memset_and_noise_save_reset(pi0_runtime_objects, typed_cuda_boundary):
    pipeline = pi0_runtime_objects[0]
    pipeline._bias_add_bf16(0x345678901, 0x456789012, 1, 2, HIGH_STREAM)
    pipeline.record_infer_graph(external_stream_int=HIGH_STREAM)
    calls = typed_cuda_boundary.calls
    assert [call[0] for call in calls].count('memset') == 1
    assert [call[0] for call in calls].count('copy') == 6  # one save, five resets
    assert all(call[-1] == HIGH_STREAM for call in calls)
    assert all(call[1] > 0xFFFFFFFF for call in calls if call[0] != 'sync')


@pytest.mark.parametrize('operation', ['memset', 'save', 'reset'])
def test_pipeline_reports_native_memory_errors(pi0_runtime_objects, typed_cuda_boundary, operation):
    pipeline = pi0_runtime_objects[0]
    if operation == 'memset':
        typed_cuda_boundary.memset_status = -7
    else:
        typed_cuda_boundary.copy_status = -7
        typed_cuda_boundary.copy_fail_on = 1 if operation == 'save' else 2
    with pytest.raises(RuntimeError, match='CUDA error -7'):
        if operation == 'memset':
            pipeline._bias_add_bf16(1, 2, 1, 2, HIGH_STREAM)
        else:
            pipeline.record_infer_graph(external_stream_int=HIGH_STREAM)
    if operation == 'memset':
        assert not hasattr(pipeline, '_bias_zero_buf')


@pytest.mark.parametrize('index', [1, 2], ids=['torch', 'jax'])
@pytest.mark.parametrize('status', [0, -7])
def test_frontend_input_copy_keeps_high_stream_and_size_and_checks_status(
        pi0_runtime_objects, typed_cuda_boundary, index, status):
    frontend = pi0_runtime_objects[index]
    nbytes = 0x100000040  # Exercise size_t too, without allocating this memory.
    source = SimpleNamespace(numel=lambda: nbytes // 2, element_size=lambda: 2,
                             data_ptr=lambda: 0x345678901)
    destination = SimpleNamespace(ptr=ctypes.c_void_p(0x456789012), nbytes=nbytes)
    typed_cuda_boundary.copy_status = status
    if status:
        with pytest.raises(RuntimeError, match='CUDA error -7'):
            frontend._copy_tensor_to_pipeline_buf_stream(source, destination, HIGH_STREAM)
    else:
        frontend._copy_tensor_to_pipeline_buf_stream(source, destination, HIGH_STREAM)
    assert typed_cuda_boundary.calls[-1] == (
        'copy', 0x456789012, 0x345678901, nbytes, 3, HIGH_STREAM)


@pytest.mark.parametrize('frontend', [Pi0TorchFrontendRtx, Pi0JaxFrontendRtx])
@pytest.mark.parametrize('kwargs,match', [
    ({'sampler': 'typo'}, 'sampler'),
    ({'probeflow_config': ProbeFlowConfig()}, 'probeflow_config'),
    ({'sampler': 'probeflow'}, 'FP16'),
    ({'sampler': 'probeflow', 'use_fp8': False}, 'FP16'),
    ({'sampler': 'probeflow', 'use_fp8': False, 'use_fp8_decoder': False,
      'probeflow_config': {}}, 'ProbeFlowConfig'),
    ({'use_fp8': False, 'use_fp8_decoder': True}, 'use_fp8_decoder'),
    ({'num_steps': 0}, 'num_steps'),
    ({'num_steps': 1.5}, 'num_steps'),
    ({'num_steps': True}, 'num_steps'),
])
def test_bad_sampler_options_fail_before_checkpoint_or_cuda(frontend, kwargs, match):
    with pytest.raises(ValueError, match=match):
        frontend('/definitely/not/a/checkpoint', **kwargs)


@pytest.mark.parametrize('times', [[], [0.], [1.1], [float('nan')], [True]])
def test_invalid_time_table_fails_before_weight_or_cuda_access(times):
    with pytest.raises(ValueError, match='timesteps'):
        _precompute_time_proj_all({}, 2, timesteps=times)


@pytest.mark.parametrize('num_steps', [0, -1, 1.2, True])
def test_invalid_euler_steps_fail_before_weight_or_cuda_access(num_steps):
    with pytest.raises(ValueError, match='num_steps'):
        _precompute_time_proj_all({}, 2, num_steps=num_steps)


def test_explicit_time_union_projects_each_exact_time_to_fp16(monkeypatch):
    from flash_rt.frontends.torch import pi0_rtx
    monkeypatch.setattr(pi0_rtx, 'DEC_D', 4)
    original_empty, original_linspace = torch.empty, torch.linspace
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs:
                        original_empty(*args, **{**kwargs, 'device': 'cpu'}))
    monkeypatch.setattr(torch, 'linspace', lambda *args, **kwargs:
                        original_linspace(*args, **{**kwargs, 'device': 'cpu'}))

    class HostWeight:
        def __init__(self, value):
            self.value = value

        def to(self, *args):
            return self.value.float()

    weights = {'_action_time_mlp_in_wt_raw': HostWeight(torch.eye(4)),
               '_action_time_mlp_in_b': HostWeight(torch.tensor([1., 2., 3., 4.]))}
    times = ProbeFlowConfig(max_steps=4, probe_horizon=0.3).timesteps()
    actual = _precompute_time_proj_all(weights, 2, timesteps=times)
    expected = []
    for t in (1., 0.75, 0.7, 0.5, 0.25):
        phase = np.array([500. * np.pi, 0.5 * np.pi]) * t
        row = np.concatenate([np.sin(phase), np.cos(phase)]).astype(np.float32)
        row = (row + np.array([1., 2., 3., 4.], dtype=np.float32)).astype(np.float16)
        expected.extend([row, row])
    assert actual.dtype == torch.float16
    np.testing.assert_array_equal(actual.numpy(), np.array(expected))


def _decoder_fixture():
    pipeline = Pi0Pipeline.__new__(Pi0Pipeline)
    pointer = lambda value: SimpleNamespace(ptr=SimpleNamespace(value=value))
    pipeline.bufs = {name: pointer(index + 1) for index, name in enumerate([
        'diffusion_noise', 'decoder_action_buf', 'decoder_x', 'decoder_x_norm'])}
    noise_ptr = pipeline.bufs['diffusion_noise'].ptr.value
    velocity_ptr = pipeline.bufs['decoder_action_buf'].ptr.value
    pipeline.values = {noise_ptr: np.full((1, 32), 10., dtype=np.float16),
                       velocity_ptr: np.full((1, 32), -99., dtype=np.float16)}
    pipeline.weights = {name: 100 for name in [
        'decoder_final_norm_w', 'decoder_action_out_proj_w', 'decoder_action_out_proj_b']}
    pipeline.encoder_seq_len = 2
    pipeline.chunk_size = 1
    pipeline.S_dec = 2
    pipeline.num_steps = 3
    pipeline._ctx = object()
    pipeline.gemm = object()
    pipeline.assembled_steps = []
    pipeline.layers = []
    pipeline.projections = []
    pipeline.projection_inputs = []
    pipeline._assemble_decoder_x = lambda step, stream: pipeline.assembled_steps.append(step)
    pipeline._decoder_layer = lambda layer, *args: pipeline.layers.append(layer)

    def project(ctx, x, weight, output, m, n, k, beta, stream):
        pipeline.projections.append((output, beta))
        pipeline.projection_inputs.append(x)
        pipeline.values[output][:] = 2. + beta * pipeline.values[output]

    def bias(output, *args):
        pipeline.values[output] += 1.

    pipeline.fvk = SimpleNamespace(
        rms_norm_fp16=lambda *args, **kwargs: None,
        gmm_fp16=project, add_bias_fp16=bias)
    return pipeline, noise_ptr, velocity_ptr


def test_velocity_only_writes_separate_scratch_without_integrating_noise():
    pipeline, noise, velocity = _decoder_fixture()
    pipeline.transformer_decoder(stream=17, _step_indices=(2,), _integrate=False)
    np.testing.assert_array_equal(pipeline.values[noise], np.full((1, 32), 10.))
    np.testing.assert_array_equal(pipeline.values[velocity], np.full((1, 32), 3.))
    assert pipeline.projections == [(velocity, 0.)]
    assert pipeline.projection_inputs == [pipeline.bufs['decoder_x_norm'].ptr.value + 2048]
    assert pipeline.assembled_steps == [2]
    assert pipeline.layers == list(range(18))


def test_default_euler_keeps_fused_beta_one_accumulation_for_every_step():
    pipeline, noise, velocity = _decoder_fixture()
    pipeline.transformer_decoder(stream=17)
    np.testing.assert_array_equal(pipeline.values[noise], np.full((1, 32), 19.))
    np.testing.assert_array_equal(pipeline.values[velocity], np.full((1, 32), -99.))
    assert pipeline.projections == [(noise, 1.)] * 3
    assert pipeline.assembled_steps == [0, 1, 2]


def test_native_fp16_decoder_keeps_state_query_attention_bound():
    pipeline, _, _ = _decoder_fixture()
    pointer = lambda value: SimpleNamespace(ptr=SimpleNamespace(value=value))
    pipeline.bufs.update({name: pointer(50 + index) for index, name in enumerate([
        'decoder_QKV', 'decoder_rope_weights', 'dec_attn_proj',
        'decoder_gate_merged', 'decoder_hidden'])})
    pipeline.weights.update({name: [100] * 18 for name in [
        'decoder_attn_qkv_w', 'decoder_attn_o_w', 'decoder_ffn_gate_w',
        'decoder_ffn_up_w', 'decoder_ffn_down_w']})
    pipeline.use_fp8_decoder = False
    pipeline.fp8_calibrated = False
    pipeline._rms_ones_dec = pointer(30)
    pipeline._attn_ptrs = {'dec_Q': 31}
    pipeline._state_nk = 3
    pipeline._enc_kv_layer_ptrs = lambda *args, **kwargs: (32, 33)
    pipeline.gemm = SimpleNamespace(fp16_nn=lambda *args, **kwargs: None)
    pipeline.fvk.qkv_split_rope_kvcache_fp16 = lambda *args: None
    pipeline.fvk.residual_add_fp16 = lambda *args, **kwargs: None
    pipeline.fvk.gate_geglu_fp16 = lambda *args, **kwargs: None
    calls = []

    def attend(site, layer, **kwargs):
        calls.append((site, layer, kwargs))
        return 34

    pipeline.attn = SimpleNamespace(run=attend)
    Pi0Pipeline._decoder_layer(pipeline, 0, 2, 2, 17)
    assert calls == [('decoder', 0, {'q_seq': 2, 'kv_seq': 4,
                                     'stream': 17, 'state_nk': 3})]


def test_disabling_fp8_skips_weight_quantization_entirely():
    frontend = Pi0TorchFrontendRtx.__new__(Pi0TorchFrontendRtx)
    frontend._ckpt_fp16 = {}
    frontend._fp8_store = []
    frontend._fp8_weights = {}
    frontend._use_fp8 = frontend._use_fp8_decoder = False
    frontend._quantize_all_fp8()
    assert frontend._fp8_weights == {}
    assert frontend._fp8_store == []


@pytest.mark.parametrize('output_scale', [None, -1., -1. / 7])
def test_output_projection_is_scaled_from_original_safetensors(monkeypatch, output_scale):
    from flash_rt.frontends.torch import pi0_rtx
    for name in ('VIS_L', 'ENC_L', 'DEC_L'):
        monkeypatch.setattr(pi0_rtx, name, 1)
    monkeypatch.setattr(pi0_rtx, 'DEC_D', 2)
    original_weight = torch.tensor([[1.0003, 0.3126], [-0.0000731, 0.07143]])
    original_bias = torch.tensor([0.10003, -0.06127])

    def tensor(key):
        if key == 'action_out_proj.weight':
            return original_weight.clone()
        if key == 'action_out_proj.bias':
            return original_bias.clone()
        if key.endswith('patch_embedding.weight'):
            return torch.ones((2, 3, 1, 1))
        if key == 'action_time_mlp_in.weight':
            return torch.ones((2, 4))
        if 'norm' in key or key.endswith('.bias'):
            return torch.zeros(2)
        if key.endswith('q_proj.weight'):
            return torch.ones((16, 2))
        return torch.ones((2, 2))

    checkpoint = SimpleNamespace(
        keys=lambda: ['paligemma_with_expert.test'], get_tensor=tensor)
    monkeypatch.setitem(sys.modules, 'safetensors', SimpleNamespace(
        safe_open=lambda *args, **kwargs: checkpoint))
    converted = pi0_rtx.convert_pi0_safetensors('unused.safetensors', output_scale=output_scale)
    if output_scale is None:
        expected_weight = original_weight.half().t() * -0.1
        expected_bias = original_bias.half() * -0.1
    else:
        expected_weight = (original_weight.t() * output_scale).half()
        expected_bias = (original_bias * output_scale).half()
    torch.testing.assert_close(converted['decoder_action_out_proj_w'], expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(converted['decoder_action_out_proj_b'], expected_bias, rtol=0, atol=0)


def test_factory_selects_native_pipeline_and_exact_probe_time_union(monkeypatch):
    from flash_rt.frontends.torch import pi0_rtx
    from flash_rt.models.pi0 import pipeline_rtx_probeflow
    monkeypatch.setattr(pi0_rtx, 'Pi0Pipeline', lambda **kwargs: ('euler', kwargs))
    monkeypatch.setattr(pipeline_rtx_probeflow, 'Pi0ProbeFlowPipeline',
                        lambda **kwargs: ('probeflow', kwargs))
    frontend = Pi0TorchFrontendRtx.__new__(Pi0TorchFrontendRtx)
    frontend._num_steps = 7
    assert frontend._create_pipeline(num_steps=7) == ('euler', {'num_steps': 7})
    frontend._probeflow_config = ProbeFlowConfig(max_steps=4, probe_horizon=0.3)
    kind, options = frontend._create_pipeline(num_steps=7)
    assert kind == 'probeflow'
    assert options['num_steps'] == 5
    assert options['probeflow_config'].timesteps() == (1., 0.75, 0.7, 0.5, 0.25)


def test_native_probe_buffers_are_fp16_action_only_and_retain_owners(monkeypatch):
    from flash_rt.models.pi0.pipeline_rtx_probeflow import Pi0ProbeFlowPipeline
    from flash_rt.sampling import torch_probeflow
    allocations = []

    def base_init(self, *args, **kwargs):
        self.chunk_size = 3
        self.bufs = {'diffusion_noise': object(), 'decoder_action_buf': object()}

    def view(buffer, shape, dtype):
        owner = object()
        allocations.append((buffer, shape, dtype, owner))
        return torch.zeros(shape, dtype=dtype), owner

    monkeypatch.setattr(Pi0Pipeline, '__init__', base_init)
    monkeypatch.setattr(torch_probeflow, 'native_float_view', view)
    config = ProbeFlowConfig(max_steps=4)
    pipeline = Pi0ProbeFlowPipeline(
        probeflow_config=config, num_steps=4, use_fp8=False, use_fp8_decoder=False)
    assert [entry[1:3] for entry in allocations] == [((3, 32), torch.float16)] * 2
    assert pipeline._view_owners == tuple(entry[3] for entry in allocations)
    assert pipeline._state_view.dtype == pipeline._velocity_view.dtype == torch.float16


def test_fp16_shared_sampler_reports_discarded_probe_and_retains_aliased_velocity():
    from flash_rt.sampling.torch_probeflow import sample_probeflow_torch_
    state = torch.zeros((1, 32), dtype=torch.float16)
    scratch = torch.zeros_like(state)
    times = []

    def evaluate(t):
        times.append(t)
        scratch.zero_()
        scratch[0, :2] = torch.tensor([1., 4. * (1. - t)], dtype=torch.float16)
        return scratch

    metadata = sample_probeflow_torch_(
        state, evaluate, ProbeFlowConfig(epsilon=0.1, max_steps=4))
    torch.testing.assert_close(state[0, :2], torch.tensor([1., 1.5], dtype=torch.float16))
    assert metadata['scheduled_steps'] == 4
    assert metadata['nfe'] == 5
    assert times == [1., 0.5, 0.75, 0.5, 0.25]


def test_frontend_metadata_does_not_replace_actual_nfe_with_schedule_length():
    frontend = Pi0TorchFrontendRtx.__new__(Pi0TorchFrontendRtx)
    frontend._num_steps = 7
    frontend.pipeline = SimpleNamespace(last_sampler_metadata={
        'sampler': 'probeflow', 'scheduled_steps': 4, 'nfe': 5,
        'timesteps': [1., 0.5, 0.75, 0.5, 0.25]})
    frontend._probeflow_config = ProbeFlowConfig(max_steps=4)
    assert frontend._sampler_metadata()['nfe'] == 5
    frontend._probeflow_config = None
    assert frontend._sampler_metadata() == {
        'sampler': 'euler', 'scheduled_steps': 7, 'nfe': 7,
        'timesteps': [1., 1. - 1 / 7, 1. - 2 / 7, 1. - 3 / 7,
                      1. - 4 / 7, 1. - 5 / 7, 1. - 6 / 7]}


def test_explicit_noise_is_copied_and_shape_is_validated_without_touching_caller():
    frontend = Pi0TorchFrontendRtx.__new__(Pi0TorchFrontendRtx)
    frontend.chunk_size = 2
    frontend._noise_buf = torch.zeros((2, 32), dtype=torch.float16)
    noise = np.arange(64, dtype=np.float32).reshape(2, 32)
    original = noise.copy()
    frontend._fill_noise_buf(noise)
    np.testing.assert_array_equal(frontend._noise_buf.float().numpy(), original)
    np.testing.assert_array_equal(noise, original)
    with pytest.raises(ValueError, match='noise'):
        frontend._fill_noise_buf(np.zeros((3, 32)))


@pytest.mark.parametrize('frontend_class', [Pi0TorchFrontendRtx, Pi0JaxFrontendRtx])
@pytest.mark.parametrize('copy_status', [0, -7])
def test_eager_infer_without_calibration_keeps_action_semantics_and_metadata(
        monkeypatch, typed_cuda_boundary, frontend_class, copy_status):
    frontend = frontend_class.__new__(frontend_class)
    frontend.chunk_size = 2
    frontend._num_steps = 7
    frontend._probeflow_config = None
    frontend._use_cuda_graph = False
    frontend.graph_recorded = False
    frontend.latency_records = []
    stream = SimpleNamespace(cuda_stream=HIGH_STREAM, wait_stream=lambda other: None)
    frontend._graph_torch_stream = stream
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    frontend._noise_buf = torch.empty((2, 32), dtype=torch.float16)
    frontend._noise_out = torch.empty_like(frontend._noise_buf)
    frontend._img_buf = frontend._state_buf_host = object()
    frontend._fill_img_buf = frontend._fill_state_buf = lambda *args: None
    frontend._copy_tensor_to_pipeline_buf_stream = lambda *args: None
    forwarded_streams = []

    def forward(stream):
        forwarded_streams.append(stream)
        return 123

    frontend.pipeline = SimpleNamespace(
        input_images_buf=object(), input_noise_buf=object(), input_state_buf=object(),
        forward=forward)
    frontend._cudart = typed_cuda_boundary.runtime
    typed_cuda_boundary.on_copy = lambda: frontend._noise_out.fill_(2.)
    typed_cuda_boundary.copy_status = copy_status
    frontend.norm_stats = {'actions': {'q01': [-2.] * 32, 'q99': [2.] * 32}}
    noise = np.zeros((2, 32), dtype=np.float32)
    if copy_status:
        with pytest.raises(RuntimeError, match='CUDA error -7'):
            frontend.infer({}, noise=noise)
        assert typed_cuda_boundary.calls[-1][-1] == HIGH_STREAM
        assert frontend.latency_records == []
        return
    result = frontend.infer({}, noise=noise)
    assert forwarded_streams == [HIGH_STREAM]
    assert typed_cuda_boundary.calls[-2][-1] == HIGH_STREAM
    assert result['actions'].shape == (2, 7)
    np.testing.assert_allclose(result['actions'], np.full((2, 7), 4. + 1.5e-6))
    np.testing.assert_array_equal(result['raw_actions'], np.full((2, 32), 2.))
    assert result['sampler_metadata']['nfe'] == 7
    assert not frontend.graph_recorded
    np.testing.assert_array_equal(noise, np.zeros((2, 32)))


def test_tiny_positive_probe_time_has_an_exact_graph_lookup():
    from flash_rt.models.pi0.pipeline_rtx_probeflow import Pi0ProbeFlowPipeline
    pipeline = Pi0ProbeFlowPipeline.__new__(Pi0ProbeFlowPipeline)
    config = ProbeFlowConfig(probe_horizon=float(np.nextafter(1., 0.)))
    pipeline._time_indices = {t: i for i, t in enumerate(config.timesteps())}
    replayed = []
    pipeline._velocity_graphs = {
        index: SimpleNamespace(replay=lambda stream, i=index: replayed.append(i))
        for index in pipeline._time_indices.values()}
    pipeline._velocity_view = object()
    query = 1. - config.probe_horizon
    assert pipeline._evaluate(query, stream=17, use_graphs=True) is pipeline._velocity_view
    assert replayed == [pipeline._time_indices[query]]


def test_same_length_prompt_keeps_captured_prefix_source_alive():
    pipeline = Pi0Pipeline.__new__(Pi0Pipeline)
    pipeline.max_prompt_len = 2
    old_allocation = SimpleNamespace(nbytes=4096, upload=lambda value: uploads.append(value.copy()))
    uploads = []
    pipeline._lang_embeds_buf = old_allocation
    pipeline._set_decoder_rope_for_prompt = lambda length: None
    pipeline._copy_lang_embeds_to_encoder_x = lambda: None
    embeds = np.ones((1, 2048), dtype=np.float16)
    pipeline.set_language_embeds(embeds)
    assert pipeline._lang_embeds_buf is old_allocation
    np.testing.assert_array_equal(uploads, [embeds])


def test_adaptive_graph_capture_only_records_prefix_and_single_velocities(monkeypatch):
    from flash_rt.models.pi0 import pipeline_rtx_probeflow
    pipeline = pipeline_rtx_probeflow.Pi0ProbeFlowPipeline.__new__(
        pipeline_rtx_probeflow.Pi0ProbeFlowPipeline)
    pipeline._graph = None
    pipeline._velocity_graphs = {}
    pipeline._sampler_times = (1., 0.75, 0.5, 0.25)
    pipeline._check_stream = lambda stream: None
    pipeline._sync = lambda stream: None
    pipeline.autotune_gemms = lambda: None
    prefix_runs, decoder_runs = [], []
    pipeline._run_prefix = lambda stream: prefix_runs.append(stream)

    def forbid_host_routing(*args, **kwargs):
        raise AssertionError('adaptive routing must not run during graph capture')

    pipeline._sample = forbid_host_routing
    monkeypatch.setattr(Pi0Pipeline, 'transformer_decoder',
                        lambda self, **kwargs: decoder_runs.append(kwargs))
    monkeypatch.setattr(pipeline_rtx_probeflow, 'CUDAGraph', lambda: SimpleNamespace(
        begin_capture=lambda handle: None, end_capture=lambda handle: None))
    pipeline.record_infer_graph(external_stream_int=17)
    assert prefix_runs == [17, 17, 17]
    assert decoder_runs == [
        {'stream': 17, '_step_indices': (index,), '_integrate': False}
        for index in (0, 0, 1, 1, 2, 2, 3, 3)]
    assert len(pipeline._velocity_graphs) == 4
    # This fixture's graph handles are not native resources.
    pipeline._graph = None
    pipeline._velocity_graphs = {}


def test_euler_graph_warmups_restore_the_initial_noise(monkeypatch):
    from flash_rt.models.pi0 import pipeline_rtx
    pipeline = Pi0Pipeline.__new__(Pi0Pipeline)
    pipeline.chunk_size = 1
    pipeline.use_fp8 = True
    pipeline.fp8_calibrated = False
    pipeline.bufs = {'diffusion_noise': SimpleNamespace(
        ptr=SimpleNamespace(value=10), nbytes=64)}
    values = {10: 10., 20: -99.}
    observed = []
    capturing = False

    def copy(dst, src, *args):
        values[dst.value] = values[src.value]
        return 0

    def calibrate(**kwargs):
        values[10] = 999.

    def run(**kwargs):
        observed.append(values[10])
        if not capturing:
            values[10] += 1.

    def begin(handle):
        nonlocal capturing
        capturing = True

    def end(handle):
        nonlocal capturing
        capturing = False

    pipeline._cudart = SimpleNamespace(
        cudaMemcpyAsync=copy, cudaStreamSynchronize=lambda *args: 0)
    pipeline.calibrate_fp8 = calibrate
    pipeline.autotune_gemms = lambda: None
    pipeline.run_pipeline = run
    monkeypatch.setattr(pipeline_rtx.CudaBuffer, 'device_empty', lambda *args:
                        SimpleNamespace(ptr=SimpleNamespace(value=20), nbytes=64))
    monkeypatch.setattr(pipeline_rtx, 'CUDAGraph', lambda: SimpleNamespace(
        begin_capture=begin, end_capture=end))
    pipeline.record_infer_graph(external_stream_int=17)
    assert observed == [10., 10., 10., 10.]
    assert values[10] == 10.
