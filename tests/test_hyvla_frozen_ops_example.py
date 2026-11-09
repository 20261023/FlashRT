import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch = pytest.importorskip('torch')


@pytest.fixture
def ops():
    path = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/hy_trace/frozen_ops.py'
    spec = importlib.util.spec_from_file_location('hy_frozen_operators', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconstructs_live_view_without_changing_frozen_values(ops):
    values = torch.arange(72.).reshape(3, 2, 3, 4)
    original = values.clone()
    view = ops.rebuild_view(values, (24, 3, 1, 6))
    assert view.stride() == (24, 3, 1, 6)
    assert not view.is_contiguous()
    torch.testing.assert_close(view, original, atol=0, rtol=0)
    torch.testing.assert_close(values, original, atol=0, rtol=0)
    assert values.stride() == (24, 12, 4, 1)


@pytest.mark.parametrize('strides', [(1, 1), (-1, 2), (1,), (0, 1)])
def test_unsafe_or_malformed_strides_reject_before_allocating(ops, strides):
    with pytest.raises(ValueError, match='stride|overlap'):
        ops.rebuild_view(torch.arange(6.).reshape(2, 3), strides)


def test_singleton_stride_is_preserved_not_treated_as_overlap(ops):
    values = torch.arange(6.).reshape(1, 2, 3)
    view = ops.rebuild_view(values, (2, 1, 2))
    assert view.stride() == (2, 1, 2)
    torch.testing.assert_close(view, values, atol=0, rtol=0)


def test_projection_controls_use_live_strides_and_identical_camera_values(ops):
    values = [torch.arange(12.).reshape(1, 2, 2, 3) + 100*i for i in range(3)]
    official, native, contiguous = ops.projection_inputs(values, [(4, 2, 1, 4)]*3,
        {'shape': [3, 4, 3], 'live_stride': [12, 1, 4]})
    assert [view.stride() for view in official] == [(4, 2, 1, 4)]*3
    assert native.stride() == (12, 2, 1, 4)
    assert contiguous.stride() == (12, 6, 3, 1) and contiguous.is_contiguous()
    expected = torch.cat(values)
    torch.testing.assert_close(torch.cat(official), expected, atol=0, rtol=0)
    torch.testing.assert_close(native, expected, atol=0, rtol=0)
    torch.testing.assert_close(contiguous, expected, atol=0, rtol=0)


@pytest.mark.parametrize('metadata', [
    {'shape': [3, 5, 3], 'live_stride': [15, 1, 5]},
    {'shape': [2, 4, 3], 'live_stride': [12, 1, 4]},
    {'shape': [3, 4, 2], 'live_stride': [8, 1, 4]},
])
def test_projection_controls_reject_misaligned_geometry(ops, metadata):
    with pytest.raises(ValueError, match='geometry'):
        ops.projection_inputs([torch.zeros(1, 2, 2, 3)]*3, [(4, 2, 1, 4)]*3, metadata)


@pytest.mark.parametrize('promote', [False, True])
def test_position_controls_preserve_token_channel_order_and_dtype(ops, promote):
    position = torch.tensor([[[0, 16], [2, 18], [8, 24], [10, 26]]], dtype=torch.bfloat16)
    result = ops.resize_position(position, (1, 1), promote=promote)
    torch.testing.assert_close(result, torch.tensor([[[5, 21]]], dtype=torch.bfloat16), atol=0, rtol=0)
    assert result.dtype == position.dtype
    torch.testing.assert_close(position, torch.tensor([[[0, 16], [2, 18], [8, 24], [10, 26]]],
                                                     dtype=torch.bfloat16), atol=0, rtol=0)


def test_position_control_rejects_non_square_learned_grid(ops):
    with pytest.raises(ValueError, match='position geometry'):
        ops.resize_position(torch.zeros(1, 6, 3), (2, 2), promote=False)


def test_cli_help_and_unallocated_guard_without_creating_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/hy_trace/frozen_ops.py'
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
    help_result = subprocess.run([sys.executable, str(script), '--help'], env=env, text=True, capture_output=True)
    assert help_result.returncode == 0 and '--official-trace' in help_result.stdout
    output = tmp_path / 'unused'
    result = subprocess.run([sys.executable, str(script), '--official-trace', '/missing',
        '--native-report', '/missing', '--checkpoint', '/missing', '--label', 'guard', '--output', str(output)],
        env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'allocated GPU srun' in result.stderr
    assert not output.exists()
