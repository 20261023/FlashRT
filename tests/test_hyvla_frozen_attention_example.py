import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/hy_trace/frozen_attention.py'


@pytest.fixture
def attention():
    spec = importlib.util.spec_from_file_location('hy_frozen_attention', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_qkv_preserves_observed_token_head_and_channel_order(attention):
    original = torch.arange(24.).reshape(1, 2, 12)
    q, k, v = attention.split_qkv(original, heads=2)
    expected_q = torch.tensor([[[[0., 1.], [2., 3.]], [[12., 13.], [14., 15.]]]])
    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(k, expected_q+4, atol=0, rtol=0)
    torch.testing.assert_close(v, expected_q+8, atol=0, rtol=0)
    assert q.shape == (1, 2, 2, 2) and q.stride() == (24, 12, 2, 1)
    torch.testing.assert_close(original, torch.arange(24.).reshape(1, 2, 12), atol=0, rtol=0)


@pytest.mark.parametrize('group', [1, 2, 4])
def test_grouped_sdpa_restores_camera_and_frame_order_on_real_cpu_attention(attention, group):
    qkv = torch.arange(24.).reshape(4, 1, 6)
    # One token gives softmax=1; independently derived output equals V, not Q/K.
    output = attention.attention_arm(qkv, heads=2, group_size=group)
    expected = torch.tensor([[[4., 5.]], [[10., 11.]], [[16., 17.]], [[22., 23.]]])
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(qkv, torch.arange(24.).reshape(4, 1, 6), atol=0, rtol=0)


@pytest.mark.parametrize('shape,heads,group', [((4, 2, 7), 2, 2), ((4, 2, 6), 0, 2),
    ((4, 2, 6), 2, 3), ((4, 2, 6), 2, 0), ((4, 2), 2, 2)])
def test_invalid_qkv_or_group_geometry_rejects_before_attention(attention, shape, heads, group):
    with pytest.raises(ValueError, match='geometry|group'):
        attention.attention_arm(torch.ones(shape), heads=heads, group_size=group)


def test_noncausal_softmax_and_head_scale_on_two_tokens(attention):
    qkv = torch.tensor([[[1., 0., 2.], [1., math.log(3.), 6.]]], dtype=torch.float64)
    result = attention.attention_arm(qkv, heads=1, group_size=1)
    # Scores [0,log3] give weights [.25,.75], hence 2*.25+6*.75=5.
    torch.testing.assert_close(result, torch.tensor([[[5.], [5.]]], dtype=torch.float64), atol=1e-12, rtol=0)


def test_attention_observer_receives_actual_bhnd_group_views_with_live_offsets(attention):
    qkv = torch.arange(48., dtype=torch.float64).reshape(4, 2, 6)
    observed = []
    attention.attention_arm(qkv, heads=2, group_size=2,
        observer=lambda kind, q, k, v: observed.append((kind, q, k, v)))
    assert [item[0] for item in observed] == ['sdpa', 'sdpa']
    assert [item[1].shape for item in observed] == [(2, 2, 2, 1)]*2
    assert [item[1].stride() for item in observed] == [(12, 1, 6, 1)]*2
    assert [[item.storage_offset() for item in row[1:]] for row in observed] == [[0, 2, 4], [24, 26, 28]]
    assert [float(item[1][0, 0, 0, 0]) for item in observed] == [0., 24.]


def test_native_reference_binds_checkpoint_path_even_for_equal_file_stats(attention, tmp_path):
    original, other = tmp_path/'official.safetensors', tmp_path/'different.safetensors'
    original.write_bytes(b'0000'); other.write_bytes(b'1111')
    stat = original.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert other.stat().st_size == stat.st_size and other.stat().st_mtime_ns == stat.st_mtime_ns
    trace = tmp_path/'accepted-trace'
    report = {'kind': 'diagnostic_frozen_block0_operators_not_recorded_block0_or_action_acceptance',
              'engine': 'official', 'job_id': '7', 'checkpoint': str(original.resolve()),
              'checkpoint_stat': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
              'official_trace': str(trace.resolve())}
    attention.validate_reference_report(report, job_id='7', checkpoint=original, official_trace=trace)
    with pytest.raises(ValueError, match='checkpoint|provenance'):
        attention.validate_reference_report(report, job_id='7', checkpoint=other, official_trace=trace)


def test_one_hot_math_probe_runs_real_cpu_attention_and_restores_backend_flags(attention):
    before = attention.sdpa_flags()
    qkv = torch.tensor([[[1., 0., 2.], [1., math.log(3.), 6.]]], dtype=torch.float64)
    output, metadata = attention.probe_sdpa_arm(qkv, heads=1, group_size=1, name='MATH', synchronize=lambda: None)
    torch.testing.assert_close(output, torch.tensor([[[5.], [5.]]], dtype=torch.float64), atol=1e-12, rtol=0)
    assert metadata['status'] == 'executed'
    assert metadata['selected_flags'] == {'flash': False, 'efficient': False, 'cudnn': False, 'math': True}
    actual = metadata['observed_sdpa_inputs'][0]
    assert actual['api'] == 'sdpa' and actual['q']['shape'] == [1, 1, 2, 1]
    assert [actual[key]['storage_offset'] for key in ('q', 'k', 'v')] == [0, 1, 2]
    assert attention.sdpa_flags() == before


def test_unsupported_fused_preflight_does_not_execute_and_restores_flags(attention, monkeypatch):
    before = attention.sdpa_flags()
    def forbidden(*args, **kwargs): raise AssertionError('unsupported arm reached actual CUDA execution')
    monkeypatch.setattr(attention, 'attention_arm', forbidden)
    output, metadata = attention.probe_sdpa_arm(torch.ones(1, 2, 3), heads=1, group_size=1,
        name='FLASH_ATTENTION', synchronize=lambda: None,
        preflight=lambda *args, **kwargs: {'supported': False, 'warnings': ['unsupported fixture layout']})
    assert output is None and metadata['status'] == 'unsupported'
    assert metadata['preflight']['warnings'] == ['unsupported fixture layout']
    assert attention.sdpa_flags() == before


@pytest.mark.parametrize('message', ['CUDA out of memory', 'CUDA illegal memory access',
                                    'CUDA device-side assert', 'cuDNN execution failed'])
def test_backend_probe_never_hides_real_execution_faults_as_unsupported(attention, monkeypatch, message):
    before = attention.sdpa_flags()
    def fail(*args, **kwargs): raise RuntimeError(message)
    monkeypatch.setattr(attention, 'attention_arm', fail)
    with pytest.raises(RuntimeError, match=message):
        attention.probe_sdpa_arm(torch.ones(1, 2, 3), heads=1, group_size=1,
                                 name='MATH', synchronize=lambda: None)
    assert attention.sdpa_flags() == before


@pytest.mark.parametrize('message', ['No available kernel. Aborting execution.',
    'No viable backend for scaled_dot_product_attention was found.',
    'No viable backend for scaled_dot_product_attention was found. This is likely due to turning off both the math kernel and the fused kernels.'])
def test_only_known_dispatch_rejection_is_classified_unsupported(attention, monkeypatch, message):
    def fail(*args, **kwargs): raise RuntimeError(message)
    monkeypatch.setattr(attention, 'attention_arm', fail)
    output, metadata = attention.probe_sdpa_arm(torch.ones(1, 2, 3), heads=1, group_size=1,
                                               name='MATH', synchronize=lambda: None)
    assert output is None and metadata['status'] == 'unsupported' and metadata['dispatch_error'] == message


def test_backend_probe_sync_failure_is_never_a_dispatch_rejection(attention):
    def fail(): raise RuntimeError('No available kernel. Aborting execution.')
    with pytest.raises(RuntimeError, match='No available kernel'):
        attention.probe_sdpa_arm(torch.ones(1, 2, 3), heads=1, group_size=1,
                                 name='MATH', synchronize=fail)


def test_unknown_backend_never_silently_uses_auto_dispatch(attention):
    with pytest.raises(ValueError, match='backend'):
        attention.probe_sdpa_arm(torch.ones(1, 2, 3), heads=1, group_size=1,
                                 name='unknown', synchronize=lambda: None)


def test_dispatch_evidence_reports_operator_and_region_kernel_names_not_timings(attention):
    class Event:
        def __init__(self, name, device_type): self.name, self.device_type = name, device_type
        @property
        def cuda_time_total(self): raise AssertionError('diagnostic summary must not consume timings')
    summary = attention.summarize_dispatch_events([
        Event('aten::scaled_dot_product_attention', 'DeviceType.CPU'),
        Event('aten::_scaled_dot_product_flash_attention', 'DeviceType.CPU'),
        Event('flash_kernel', 'DeviceType.CUDA'), Event('flash_kernel', 'DeviceType.CUDA'),
        Event('cat_kernel', 'DeviceType.CUDA')])
    assert summary['sdpa_operator_counts'] == {'aten::_scaled_dot_product_flash_attention': 1}
    assert summary['cuda_kernel_names_in_region'] == {'flash_kernel': 2, 'cat_kernel': 1}
    assert summary['cuda_activity_observed'] is True


def test_missing_cuda_profiler_activity_is_unknown_not_zero_work(attention):
    summary = attention.summarize_dispatch_events([SimpleNamespace(
        name='aten::_scaled_dot_product_attention_math', device_type='DeviceType.CPU')])
    assert summary['cuda_activity_observed'] is False
    assert summary['cuda_kernel_names_in_region'] == {}
    assert summary['kernel_identity'] == 'unverified_no_cuda_activity'


def test_gpu_user_annotations_are_not_reported_as_cuda_kernels(attention):
    events = [SimpleNamespace(name='user-region', device_type='DeviceType.CUDA', is_user_annotation=True),
              SimpleNamespace(name='actual_kernel', device_type='DeviceType.CUDA', is_user_annotation=False)]
    summary = attention.summarize_dispatch_events(events)
    assert summary['cuda_kernel_names_in_region'] == {'actual_kernel': 1}
    assert summary['ignored_cuda_annotation_names'] == {'user-region': 1}


def test_own_gpu_region_mirror_is_excluded_even_without_annotation_flag(attention):
    events = [SimpleNamespace(name='diagnostic-region', device_type='DeviceType.CPU', is_user_annotation=True),
              SimpleNamespace(name='diagnostic-region', device_type='DeviceType.CUDA', is_user_annotation=False),
              SimpleNamespace(name='actual_kernel', device_type='DeviceType.CUDA', is_user_annotation=False)]
    summary = attention.summarize_dispatch_events(events, region_name='diagnostic-region')
    assert summary['cuda_kernel_names_in_region'] == {'actual_kernel': 1}
    assert summary['ignored_cuda_annotation_names'] == {'diagnostic-region': 1}


def test_only_gpu_region_annotations_do_not_establish_kernel_identity(attention):
    summary = attention.summarize_dispatch_events([SimpleNamespace(
        name='diagnostic-region', device_type='DeviceType.CUDA', is_user_annotation=True)])
    assert summary['cuda_activity_observed'] is False
    assert summary['cuda_kernel_names_in_region'] == {} and summary['kernel_identity'] == 'unverified_no_cuda_activity'


def test_help_and_allocation_guard_without_outputs(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], env=env, text=True, capture_output=True)
    assert result.returncode == 0 and '--engine' in result.stdout
    output = tmp_path/'unused'
    result = subprocess.run([sys.executable, str(SCRIPT), '--engine', 'official',
        '--official-trace', '/missing', '--checkpoint', '/missing', '--output', str(output)],
        env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'allocated GPU srun' in result.stderr
    assert not output.exists()
