import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

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
