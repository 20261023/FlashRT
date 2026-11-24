"""CPU tests for autotune artifacts, call-site capture and external tool adapters."""

import ast
import dataclasses
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from flash_rt.vla.autotune.tools import artifacts as art
from flash_rt.vla.autotune.tools import external
from flash_rt.vla.autotune.tools.capture import (
    CALLS_FILE, CAPTURE_FILE, NUMPY_TENSORS_FILE, capture_calls,
)
from flash_rt.vla.autotune.tools.external import aitune, apxinf, forgekernel, kernelagent

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


FK_PIN = 'b2c8ae8510e02c28576808910a6b8827721ab7dc'
KA_PIN = 'e0647170da36ef9b059ac0bd3d60103aa4ed378b'
APX_PIN = 'b05b40990387e8c48e7d6267a790664981a665e4'
AIT_PIN = '1d3a4236ad8f458669d5633715b6956125f2289c'


# -- fixtures -------------------------------------------------------------------------------


def _linear_capture(tensors_path=None, *, in_cuda_graph=True, bias=True):
    signature = [
        {'name': 'x', 'shape': ['M', 'K'], 'dtype': 'bfloat16', 'role': 'input'},
        {'name': 'weight', 'shape': ['N', 'K'], 'dtype': 'bfloat16', 'role': 'weight'},
    ]
    if bias:
        signature.append({'name': 'bias', 'shape': ['N'], 'dtype': 'bfloat16',
                          'role': 'weight'})
    signature.append({'name': 'out', 'shape': ['M', 'N'], 'dtype': 'bfloat16',
                      'role': 'output'})
    return art.StageCapture(
        model='hyvla', stage_id='kernel.gemm',
        call_site='flash_rt/models/hyvla/pipeline_rtx.py:merger_forward',
        op_kind='linear', signature=signature,
        shapes=[{'name': 'cam2', 'axes': {'M': 392, 'K': 1152, 'N': 2048}},
                {'name': 'cam3', 'axes': {'M': 588, 'K': 1152, 'N': 2048}}],
        dtype='bfloat16', in_cuda_graph=in_cuda_graph, tensors_path=tensors_path,
        notes=['x flattened from [2, 14, 14, 1152]'])


def _attention_capture(tensors_path=None, *, gqa=False):
    hkv = 4 if gqa else 8
    return art.StageCapture(
        model='pi05', stage_id='kernel.attention', call_site='a.py:attend',
        op_kind='attention',
        signature=[
            {'name': 'q', 'shape': ['B', 'H', 'S', 'D'], 'dtype': 'float16', 'role': 'input'},
            {'name': 'k', 'shape': ['B', 'HKV', 'SKV', 'D'], 'dtype': 'float16',
             'role': 'input'},
            {'name': 'v', 'shape': ['B', 'HKV', 'SKV', 'DV'], 'dtype': 'float16',
             'role': 'input'},
            {'name': 'o', 'shape': ['B', 'H', 'S', 'DV'], 'dtype': 'float16',
             'role': 'output'},
        ],
        shapes=[{'name': 's1', 'axes': {'B': 1, 'H': 8, 'HKV': hkv, 'S': 64, 'SKV': 64,
                                        'D': 64, 'DV': 64}}],
        dtype='float16', in_cuda_graph=False, tensors_path=tensors_path)


def _custom_capture(tmp_path, tensors_path=None, body='    return x * 2.0\n'):
    reference = tmp_path / 'ref_impl.py'
    reference.write_text('import torch\n\n\ndef run(x):\n' + body, encoding='utf-8')
    return art.StageCapture(
        model='fastwam', stage_id='kernel.norm_elementwise', call_site='b.py:scale',
        op_kind='custom',
        signature=[{'name': 'x', 'shape': ['x_0', 'x_1'], 'dtype': 'bfloat16',
                    'role': 'input'},
                   {'name': 'y', 'shape': ['x_0', 'x_1'], 'dtype': 'bfloat16',
                    'role': 'output'}],
        shapes=[{'name': 'a', 'axes': {'x_0': 4, 'x_1': 8}}], dtype='bfloat16',
        reference=str(reference), tensors_path=tensors_path)


def _kernel_artifact(**changes):
    fields = dict(
        tool_id='kernel.forgekernel', upstream_commit=FK_PIN, stage_id='kernel.gemm',
        model='hyvla', language='cuda', entry_symbol='run', kernel_path='/runs/best/kernel.cu',
        kernel_sha256='a' * 64, op_metrics={'median_ratio': 1.2}, accepted=True, reasons=[],
        provenance={'run_dir': '/runs/x'})
    fields.update(changes)
    return art.KernelArtifact(**fields)


# -- artifacts ------------------------------------------------------------------------------


def test_artifacts_roundtrip_through_dict_and_file(tmp_path):
    proposal = art.PatchProposal(
        base_commit='1' * 40, patch_ref='sha256:' + 'b' * 64, tool_id='kernel.kernelagent',
        stage_id='kernel.gemm', hypothesis='triton proj1', options={'ka_kernels': ['p']},
        provenance={'session_dir': '/s'})
    for item in (_linear_capture('/abs/t.pt'), _kernel_artifact(), proposal):
        record = item.to_dict()
        assert record['schema'] == item.schema
        assert type(item).from_dict(json.loads(json.dumps(record))) == item
        path = item.save(tmp_path / f'{type(item).__name__}.json')
        assert type(item).load(path) == item
        assert art.load_artifact(path) == item
        assert path.read_text().endswith('\n')


def test_artifacts_reject_invalid_records():
    capture = _linear_capture()
    with pytest.raises(ValueError, match='stage_id'):
        dataclasses.replace(capture, stage_id='Kernel GEMM')
    with pytest.raises(ValueError, match='differ'):
        dataclasses.replace(capture, shapes=[{'name': 'a', 'axes': {'M': 1, 'K': 2, 'N': 3}},
                                             {'name': 'b', 'axes': {'M': 1, 'K': 2}}])
    with pytest.raises(ValueError, match='not an axis'):
        dataclasses.replace(capture, shapes=[{'name': 'a', 'axes': {'M': 1, 'K': 2}}])
    with pytest.raises(ValueError, match='op_kind'):
        dataclasses.replace(capture, op_kind='conv')
    with pytest.raises(ValueError, match='Unknown StageCapture fields'):
        art.StageCapture.from_dict({**capture.to_dict(), 'extra': 1})
    with pytest.raises(ValueError, match='schema'):
        art.StageCapture.from_dict({**capture.to_dict(), 'schema': 'other'})
    with pytest.raises(ValueError, match='no rejection reasons'):
        _kernel_artifact(reasons=['slow'])
    with pytest.raises(ValueError, match='at least one reason'):
        _kernel_artifact(accepted=False)
    with pytest.raises(ValueError, match='all set or all None'):
        _kernel_artifact(kernel_sha256=None)
    with pytest.raises(ValueError, match='upstream_commit'):
        _kernel_artifact(upstream_commit='abc')
    with pytest.raises(ValueError, match='patch_ref'):
        art.PatchProposal(base_commit='1' * 40, patch_ref='abc', tool_id='t', stage_id='s',
                          hypothesis='h', options={}, provenance={})
    with pytest.raises(ValueError, match='Unknown artifact schema'):
        art.from_record({'schema': 'nope'})


# -- capture_calls --------------------------------------------------------------------------


class _Owner:
    def linear(self, x, weight, bias=None):
        y = x @ weight.T
        return y if bias is None else y + bias

    @staticmethod
    def attend(q, k, v):
        scores = np.einsum('bhsd,bhtd->bhst', q, k) / np.sqrt(q.shape[-1])
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        return np.einsum('bhst,bhtd->bhsd', probs, v)


def test_capture_calls_records_numpy_linear_and_restores(tmp_path):
    owner = _Owner()
    weight, bias = np.ones((16, 8), np.float32), np.zeros(16, np.float32)
    with capture_calls(owner, 'linear', model='hyvla', stage_id='kernel.gemm',
                       call_site='x.py:f', op_kind='linear', out_dir=tmp_path, max_calls=3,
                       in_cuda_graph=True) as recorder:
        assert 'linear' in vars(owner)
        for m in (4, 4, 6, 6):
            out = owner.linear(np.ones((2, m, 8), np.float32), weight, bias=bias)
            assert out.shape == (2, m, 16)
        owner.linear(np.ones((2, 4, 8), np.float32)[:, :, ::2].repeat(2, -1), weight)
    assert 'linear' not in vars(owner) and owner.linear.__func__ is _Owner.linear
    assert recorder.total_calls == 5 and len(recorder.calls) == 3
    capture = art.StageCapture.load(tmp_path / CAPTURE_FILE)
    assert capture == recorder.capture
    assert capture.op_kind == 'linear' and capture.in_cuda_graph is True
    assert [(e['name'], e['shape'], e['role']) for e in capture.signature] == [
        ('x', ['M', 'K'], 'input'), ('weight', ['N', 'K'], 'weight'),
        ('bias', ['N'], 'weight'), ('out', ['M', 'N'], 'output')]
    assert [row['axes'] for row in capture.shapes] == [{'M': 8, 'K': 8, 'N': 16},
                                                      {'M': 12, 'K': 8, 'N': 16}]
    assert any('flattened' in note for note in capture.notes)
    assert any('recorded 3 of 5' in note for note in capture.notes)
    calls = json.loads((tmp_path / CALLS_FILE).read_text())
    first = calls['calls'][0]['arguments']
    assert first[0]['strides'] == [32, 8, 1] and first[0]['contiguous'] is True
    assert first[0]['dtype'] == 'float32'
    assert capture.tensors_path == str((tmp_path / NUMPY_TENSORS_FILE).resolve())
    with np.load(capture.tensors_path) as saved:
        assert sorted(saved.files) == ['bias', 'out', 'weight', 'x']
        assert saved['x'].shape == (2, 4, 8)


def test_capture_calls_restores_on_error_and_wraps_static_methods(tmp_path):
    owner = _Owner()
    with pytest.raises(RuntimeError, match='boom'):
        with capture_calls(owner, 'linear', model='m', stage_id='kernel.gemm',
                           call_site='c', op_kind='linear', out_dir=tmp_path):
            owner.linear(np.ones((2, 8), np.float32), np.ones((4, 8), np.float32))
            raise RuntimeError('boom')
    assert 'linear' not in vars(owner)
    assert not (tmp_path / CAPTURE_FILE).exists()

    q = np.random.default_rng(0).standard_normal((1, 2, 5, 4)).astype(np.float32)
    raw = vars(_Owner)['attend']
    with capture_calls(_Owner, 'attend', model='pi05', stage_id='kernel.attention',
                       call_site='d', op_kind='attention') as recorder:
        _Owner().attend(q, q, q)
        _Owner.attend(q, q, q)
    assert vars(_Owner)['attend'] is raw
    capture = recorder.capture
    assert capture.shapes[0]['axes'] == {'B': 1, 'H': 2, 'HKV': 2, 'S': 5, 'SKV': 5,
                                         'D': 4, 'DV': 4}
    assert recorder.tensors_path is None and recorder.capture_path is None


def test_capture_calls_namespace_owner_custom_and_empty(tmp_path):
    def scale(x, factor=2.0):
        return x * factor

    owner = types.SimpleNamespace(scale=scale)
    with capture_calls(owner, 'scale', model='fastwam', stage_id='kernel.norm_elementwise',
                       call_site='e', op_kind='custom') as recorder:
        owner.scale(np.ones((3, 5), np.float16), factor=3.0)
    assert owner.scale is scale
    signature = recorder.capture.signature
    assert [e['shape'] for e in signature] == [['x_0', 'x_1'], ['out_0', 'out_1']]
    assert recorder.capture.dtype == 'float16'
    assert recorder.calls[0]['arguments'][1] == {'name': 'factor', 'kind': 'scalar',
                                                 'value': 3.0}

    with pytest.raises(ValueError, match='captured no call'):
        with capture_calls(owner, 'scale', model='m', stage_id='s', call_site='c',
                           op_kind='custom', out_dir=tmp_path / 'empty'):
            pass
    assert owner.scale is scale
    with pytest.raises(ValueError, match='max_calls'):
        with capture_calls(owner, 'scale', model='m', stage_id='s', call_site='c',
                           op_kind='custom', max_calls=0):
            pass
    with pytest.raises(AttributeError):
        with capture_calls(owner, 'missing', model='m', stage_id='s', call_site='c',
                           op_kind='custom'):
            pass


class _Doubler:
    def __call__(self, x):
        return x * 2.0


class _Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor


@pytest.mark.filterwarnings('ignore::FutureWarning')  # partial as a method (Python 3.13)
@pytest.mark.parametrize('value', [np.negative, functools.partial(np.multiply, 3.0),
                                   _Doubler(), _Scaler(5.0).scale],
                         ids=['ufunc', 'partial', 'callable_instance', 'bound_method'])
def test_capture_calls_keeps_binding_of_non_function_class_attributes(value):
    holder = type('Holder', (), {'op': value})
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = holder().op(x)
    with capture_calls(holder, 'op', model='m', stage_id='kernel.norm_elementwise',
                       call_site='c', op_kind='custom') as recorder:
        np.testing.assert_array_equal(holder().op(x), expected)
        np.testing.assert_array_equal(holder.op(x), expected)
    assert vars(holder)['op'] is value
    np.testing.assert_array_equal(holder().op(x), expected)
    assert recorder.total_calls == 2
    tensors = [entry for entry in recorder.calls[0]['arguments'] if entry['kind'] == 'tensor']
    assert [entry['shape'] for entry in tensors] == [[2, 3]]
    assert [len(entry['shape']) for entry in recorder.capture.signature] == [2, 2]


# -- SPECs ----------------------------------------------------------------------------------


@pytest.mark.parametrize('module,pin,notice,skills', [
    (forgekernel, FK_PIN, True, ['forge-start', 'vla-kernel-candidate', 'vla-operator-gate']),
    (kernelagent, KA_PIN, False, ['vla-kernel-candidate', 'vla-operator-gate']),
    (apxinf, APX_PIN, True, ['vla-model-impl-map']),
    (aitune, AIT_PIN, False, ['aitune-inspect', 'aitune-tune', 'aitune-validate',
                              'aitune-benchmark', 'vla-patch-candidate']),
])
def test_specs_match_research_pins(module, pin, notice, skills):
    spec = module.SPEC
    assert external.SPEC_KEYS <= set(spec) <= external.SPEC_KEYS | external.SPEC_OPTIONAL_KEYS
    assert set(spec['upstream']) == external.UPSTREAM_KEYS
    assert spec['upstream']['commit'] == pin and spec['upstream']['notice'] is notice
    assert spec['upstream']['license'] == 'Apache-2.0'
    assert list(spec['skills']) == skills
    expected_status = 'needs_implementation' if module is aitune else 'needs_setup'
    assert spec['status'] == expected_status and spec['reason']
    assert spec['setup'] and spec['risks']
    assert external.external_specs()[spec['tool_id']] is spec
    tree = ast.parse(Path(module.__file__).read_text())
    imported = {alias.name.split('.')[0] for node in tree.body
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split('.')[0] for node in tree.body
                 if isinstance(node, ast.ImportFrom) and node.level == 0}
    assert not imported & {'torch', 'triton', 'forgekernel', 'triton_kernel_agent', 'utils',
                           'apxinf', 'aitune', 'tensorrt', 'requests', 'urllib', 'subprocess',
                           'socket'}


def test_spec_specifics():
    assert forgekernel.SPEC['lane'] == 'patch'
    assert 'A800' in forgekernel.SPEC['reason'] and 'Claude' in forgekernel.SPEC['reason']
    assert 'first' in kernelagent.SPEC['reason'] and 'A100 SXM4 80GB' in kernelagent.SPEC[
        'reason']
    assert apxinf.SPEC['models'] == ('*',) and apxinf.SPEC['lane'] == 'baseline'
    assert apxinf.SPEC['model_predicate'] == {'path': 'tools.supported', 'contains': 'engine.apxinf'}
    assert len(forgekernel.sm80_enablement()) == 5


def test_aitune_spec_is_documentation_only_and_grounded():
    spec = aitune.SPEC
    assert spec['tool_id'] == 'compile.aitune' and spec['lane'] == 'patch'
    assert spec['models'] == ('*',) and spec['exactness'] == 'approximate'
    assert spec['model_predicate'] and spec['status'] == 'needs_implementation'
    assert spec['stages'][0] == 'denoise.velocity'
    assert {'vision.encode', 'prefix.prefill'} <= set(spec['stages'])
    # No generated script: every documented API comes from source read at the pin.
    assert not hasattr(aitune, 'render_script')
    for key in aitune.API_FACTS:
        path, sha = key.split('@')
        assert path.startswith('aitune/torch/') and len(sha) == 40 and int(sha, 16) >= 0
    text = ' '.join([spec['reason'], *spec['setup'], *spec['risks']])
    for needle in ('CUDA graph', 'A800', 'break-even', 'nn.Module', 'tune_strategy',
                   'ignore_failing_modules', 'sha256'):
        assert needle in text, needle


# -- ForgeKernel ----------------------------------------------------------------------------


def _parse_toml(path):
    if tomllib is None:
        pytest.skip('tomllib needs Python 3.11+')
    with open(path, 'rb') as handle:
        return tomllib.load(handle)


def _check_task_dir(task_dir):
    config = _parse_toml(task_dir / 'task.toml')
    assert set(config) <= set(forgekernel.TASK_KEYS)
    for table, body in config.items():
        assert set(body) <= forgekernel.TASK_KEYS[table], table
    forgekernel.validate_task_config(config)
    run_file, function = config['reference']['run'].split(':')
    assert function == 'run' and (task_dir / run_file).is_file()
    assert not Path(config['reference']['source']).is_absolute()
    shapes = json.loads((task_dir / 'shapes.json').read_text())
    assert isinstance(shapes, list) and shapes
    assert len({frozenset(row['axes']) for row in shapes}) == 1
    for row in shapes:
        assert set(row) == {'name', 'axes'}
        assert all(isinstance(v, int) and v > 0 for v in row['axes'].values())
    run_text = (task_dir / 'reference' / 'run.py').read_text()
    compile(run_text, 'run.py', 'exec')
    description = (task_dir / 'description.md').read_text()
    for section in ('## Signature', '## Semantics', '## Legal inputs'):
        assert section in description
    assert 'REVIEW REQUIRED' in description
    return config, run_text, description


def test_forgekernel_prepare_task_linear(tmp_path):
    task_dir = forgekernel.prepare_task(_linear_capture('/abs/t.pt'),
                                        tmp_path / 'hyvla_proj1', deny=['/abs/FlashRT'],
                                        deny_modules=['flash_rt'])
    config, run_text, description = _check_task_dir(task_dir)
    assert config['task'] == {'language': 'cuda', 'arch': 'sm80'}
    assert config['target'] == {'ratio': 1.1, 'timing': 'device'}
    assert config['budget'] == {'hours': 6, 'max_meta': 3}
    assert config['reference'] == {'run': 'reference/run.py:run', 'source': 'reference',
                                   'deny': ['/abs/FlashRT'], 'deny_modules': ['flash_rt']}
    assert 'def run(x, weight, bias):' in run_text and 'F.linear(x, weight, bias)' in run_text
    assert '// fk-cuda-cflags: -gencode=arch=compute_80,code=sm_80' in description
    assert '`x`: `[M, K]`, bfloat16' in description and 'float32' in description
    assert 'x flattened' in description
    shapes = json.loads((task_dir / 'shapes.json').read_text())
    assert [row['name'] for row in shapes] == ['cam2', 'cam3']


def test_forgekernel_prepare_task_attention_and_options(tmp_path):
    capture = _attention_capture(gqa=True)
    task_dir = forgekernel.prepare_task(capture, tmp_path / 'pi05_attn', language='cutedsl',
                                        target_ratio=None, target_ms=0.25, hours=4.5,
                                        max_meta=2)
    config, run_text, description = _check_task_dir(task_dir)
    assert config['target'] == {'ms': 0.25, 'timing': 'e2e'}
    assert config['budget'] == {'hours': 4.5, 'max_meta': 2}
    assert 'deny' not in config['reference']
    assert 'scaled_dot_product_attention(q, k, v, enable_gqa=True)' in run_text
    assert 'fk-cuda-cflags' not in description and 'CUTE_DSL_ARCH' in description
    with pytest.raises(ValueError, match='exactly one target'):
        forgekernel.prepare_task(capture, tmp_path / 'other', target_ms=0.25)


def test_forgekernel_prepare_task_custom_copies_reference(tmp_path):
    capture = _custom_capture(tmp_path)
    source = tmp_path / 'kernel_src.cu'
    source.write_text('// flashrt kernel\n')
    task_dir = forgekernel.prepare_task(capture, tmp_path / 'fw_scale', timing='device',
                                        reference_source=source)
    config, run_text, description = _check_task_dir(task_dir)
    assert run_text == Path(capture.reference).read_text()
    assert (task_dir / 'reference' / 'source' / 'kernel_src.cu').is_file()
    assert config['target']['timing'] == 'device'
    assert description.count('REVIEW REQUIRED') >= 3
    with pytest.raises(ValueError, match='requires capture.reference'):
        forgekernel.prepare_task(dataclasses.replace(capture, reference=None),
                                 tmp_path / 'fw2')


def test_forgekernel_prepare_task_rejects_bad_inputs(tmp_path):
    capture = _linear_capture()
    for name in ('_hidden', 'run', 'bad name'):
        with pytest.raises(ValueError, match='task name'):
            forgekernel.prepare_task(capture, tmp_path / name)
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    (occupied / 'x').write_text('')
    with pytest.raises(ValueError, match='not empty'):
        forgekernel.prepare_task(capture, occupied)
    with pytest.raises(ValueError, match='arch'):
        forgekernel.prepare_task(capture, tmp_path / 'a1', arch='ampere')
    with pytest.raises(ValueError, match='deny_modules'):
        forgekernel.prepare_task(capture, tmp_path / 'a2', deny_modules=['bad name'])
    with pytest.raises(ValueError, match='max_meta'):
        forgekernel.prepare_task(capture, tmp_path / 'a3', max_meta=-1)
    renamed = dataclasses.replace(capture, signature=[
        {**entry, 'name': 'torch'} if entry['name'] == 'x' else entry
        for entry in capture.signature])
    with pytest.raises(ValueError, match='collides'):
        forgekernel.prepare_task(renamed, tmp_path / 'a4')
    assert not any((tmp_path / name).exists() for name in ('a1', 'a2', 'a3', 'a4'))
    with pytest.raises(ValueError, match='unknown keys'):
        forgekernel.validate_task_config({'task': {'language': 'cuda', 'arch': 'sm80',
                                                   'gpu': 'a800'}})


def test_forgekernel_render_job(tmp_path):
    task_dir = forgekernel.prepare_task(_linear_capture(), tmp_path / 'op1')
    script = forgekernel.render_job(task_dir, forgekernel_root='/opt/ForgeKernel',
                                    runs_dir='/scratch/fk-runs', env_file='/opt/env.sh',
                                    gpu=0)
    assert script.startswith('#!/usr/bin/env bash\n')
    for expected in ('set -euo pipefail', 'source /opt/env.sh',
                     'export FK_DELIVERY=local FK_GPU=0 LB_INSTALL_CUPTI=0',
                     'export FORGEKERNEL_RUNS_DIR=/scratch/fk-runs', 'FORGEKERNEL_LOCK_DIR',
                     f'TASK_DIR={task_dir}', 'cd /opt/ForgeKernel',
                     f'git merge-base --is-ancestor {FK_PIN} HEAD',
                     'forge new "$TASK_DIR" --dry-run', 'forge run "$RUN"', 'FORGE_RC=$RC'):
        assert expected in script, expected
    assert "sed -n 's/^\\[forge\\] run directory \\(.*\\): `forge run.*/\\1/p'" in script
    with pytest.raises(ValueError, match='absolute'):
        forgekernel.render_job(task_dir, forgekernel_root='ForgeKernel', runs_dir='/r')
    with pytest.raises(ValueError, match='task.toml'):
        forgekernel.render_job(tmp_path, forgekernel_root='/f', runs_dir='/r')


def _fk_run(tmp_path, *, ratios=(1.20, 1.21, 1.22), median=1.21, disputed=False,
            monitor='ok', confirmation=True, kernel='kernel.cu', nodes=('n1', 'n2'),
            requested=None):
    run = tmp_path / 'op1_run'
    op_dir = run / 'operators' / 'op1'
    op_dir.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({
        'task': 'op1', 'id': 'r1', 'status': 'stopped', 'stop': 'budget', 'loop_rc': 2,
        'forgekernel_commit': 'c' * 40, 'target_met': False}))
    if confirmation:
        record = {'sha': 'd' * 40, 'window': 'device', 'median_ms': 0.1,
                  'jobs': [{'job': i, 'ratio': r, 'ms': 0.1} for i, r in enumerate(ratios)],
                  'median_ratio': median, 'keep_disputed': disputed, 'nodes': list(nodes)}
        if requested is not None:
            record['requested'] = requested
        if monitor is not None:
            record['correctness_monitor'] = {'status': monitor}
        (op_dir / 'best_confirmation.json').write_text(json.dumps(record))
    (op_dir / 'loop_summary.json').write_text(json.dumps({'stop': 'budget',
                                                          'confirm': {'median_ratio': median}}))
    if kernel:
        (run / 'best').mkdir()
        (run / 'best' / kernel).write_text('// kernel\n')
    return run


def test_forgekernel_harvest_accepts_confirmed_kernel(tmp_path):
    run = _fk_run(tmp_path)
    artifact = forgekernel.harvest(run, stage_id='kernel.gemm', model='hyvla')
    assert artifact.accepted and artifact.reasons == ()
    assert artifact.language == 'cuda' and artifact.entry_symbol == 'run'
    expected = hashlib.sha256(b'// kernel\n').hexdigest()
    assert artifact.kernel_sha256 == expected
    assert artifact.kernel_path == str((run / 'best' / 'kernel.cu').resolve())
    assert artifact.upstream_commit == 'c' * 40
    assert artifact.op_metrics['median_ratio'] == 1.21
    assert artifact.op_metrics['acceptance_threshold'] == pytest.approx(1 + 0.02 / 1.2)
    assert artifact.provenance['run']['loop_rc'] == 2
    art.KernelArtifact.from_dict(artifact.to_dict())


@pytest.mark.parametrize('changes,reason', [
    ({'ratios': (1.00, 1.10, 1.20), 'median': 1.10}, 'median_ratio 1.1 <='),
    ({'ratios': (1.010, 1.012, 1.011), 'median': 1.011}, '1 + max(0.015'),
    ({'disputed': True}, 'keep_disputed'),
    ({'monitor': 'alarm'}, 'alarm'),
    ({'confirmation': False}, 'best_confirmation.json missing'),
    ({'kernel': None}, 'no best/kernel.py'),
])
def test_forgekernel_harvest_rejections(tmp_path, changes, reason):
    artifact = forgekernel.harvest(_fk_run(tmp_path, **changes), stage_id='kernel.gemm',
                                   model='hyvla')
    assert not artifact.accepted
    assert any(reason in item for item in artifact.reasons), artifact.reasons


def test_forgekernel_harvest_records_every_reason(tmp_path):
    run = _fk_run(tmp_path, ratios=(1.0, 1.0), median=1.0, disputed=True, monitor='alarm',
                  kernel='kernel.py', nodes=('n1',))
    artifact = forgekernel.harvest(run, stage_id='kernel.gemm', model='hyvla')
    assert len(artifact.reasons) == 3 and artifact.language == 'cutedsl'
    assert any('one hardware draw' in note for note in artifact.provenance['notes'])
    (run / 'best' / 'kernel.cu').write_text('x')
    assert any('both' in r for r in forgekernel.harvest(
        run, stage_id='kernel.gemm', model='hyvla').reasons)
    empty = forgekernel.harvest(tmp_path / 'nothing', stage_id='kernel.gemm', model='m')
    assert not empty.accepted and empty.upstream_commit == FK_PIN
    assert any('run.json missing' in r for r in empty.reasons)


def test_forgekernel_harvest_rejects_short_or_single_job_confirmation(tmp_path):
    short = forgekernel.harvest(_fk_run(tmp_path / 'a', ratios=(1.02,), median=1.02,
                                        requested=3), stage_id='kernel.gemm', model='hyvla')
    assert not short.accepted
    assert any('short of requested: 1 < 3 jobs' in r for r in short.reasons), short.reasons
    assert any('one confirmation job' in r for r in short.reasons)
    assert short.op_metrics['confirmation_jobs'] == 1
    assert short.op_metrics['confirmation_requested'] == 3
    assert 'acceptance_threshold' not in short.op_metrics
    single = forgekernel.harvest(_fk_run(tmp_path / 'b', ratios=(1.30,), median=1.30),
                                 stage_id='kernel.gemm', model='hyvla')
    assert [r for r in single.reasons if 'one confirmation job' in r] and not any(
        'short of requested' in r for r in single.reasons)
    two_of_three = forgekernel.harvest(_fk_run(tmp_path / 'c', ratios=(1.20, 1.21),
                                               median=1.205, requested=3),
                                       stage_id='kernel.gemm', model='hyvla')
    assert any('short of requested: 2 < 3' in r for r in two_of_three.reasons)
    full = forgekernel.harvest(_fk_run(tmp_path / 'd', requested=3), stage_id='kernel.gemm',
                               model='hyvla')
    assert full.accepted and full.op_metrics['confirmation_requested'] == 3
    bad = forgekernel.harvest(_fk_run(tmp_path / 'e', requested='3'), stage_id='kernel.gemm',
                              model='hyvla')
    assert any("requested is not a positive int: '3'" in r for r in bad.reasons)


def _fake_forgekernel_clone(root, *, index=None, cute_arch=None):
    (root / 'src' / 'forgekernel').mkdir(parents=True)
    if cute_arch is not None:
        (root / 'src' / 'forgekernel' / 'taskcard.py').write_text(
            f'CUTE_ARCH = {cute_arch}\n')
    if index is not None:
        (root / index).parent.mkdir(parents=True)
        (root / index).write_text('# sm80\n')
    return root


def _preflight(script):
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('echo "FORGEKERNEL_HEAD'))
    end = next(i for i, line in enumerate(lines) if line.startswith('mkdir -p'))
    return lines[start + 1:end]


@pytest.mark.parametrize('language', ['cuda', 'cutedsl'])
def test_forgekernel_render_job_checks_sm80_enablement(tmp_path, language):
    task_dir = forgekernel.prepare_task(_linear_capture(), tmp_path / 'op1', language=language)
    script = forgekernel.render_job(task_dir, forgekernel_root='/opt/ForgeKernel',
                                    runs_dir='/r')
    preflight = _preflight(script)
    assert len(preflight) == 2
    assert preflight[0].startswith(f'test -f hw_kb/{language}/sm80/index.md || ')
    assert 'src/forgekernel/taskcard.py' in preflight[1]
    assert script.index(preflight[1]) < script.index('forge new')
    patched = '{"sm90": "sm_90a", "sm80": "sm_80"}'
    for name, kwargs, rc, message in (
            ('unpatched', {'cute_arch': '{"sm90": "sm_90a"}'}, 1, 'index.md'),
            ('no_cute', {'index': f'hw_kb/{language}/sm80/index.md',
                         'cute_arch': '{"sm90": "sm_90a"}'}, 1, 'no sm80 entry'),
            ('other_lang', {'index': 'hw_kb/other/sm80/index.md', 'cute_arch': patched},
             1, 'index.md'),
            ('patched', {'index': f'hw_kb/{language}/sm80/index.md', 'cute_arch': patched},
             0, ''),
            ('single_quotes', {'index': f'hw_kb/{language}/sm80/index.md',
                               'cute_arch': "{'sm80': 'sm_80'}"}, 0, '')):
        clone = _fake_forgekernel_clone(tmp_path / name, **kwargs)
        proc = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + '\n'.join(preflight)],
                              cwd=clone, capture_output=True, text=True)
        assert proc.returncode == rc, (name, proc.stderr)
        assert message in proc.stderr and ('sm80_enablement()' in proc.stderr) == bool(rc)

    other = forgekernel.prepare_task(_linear_capture(), tmp_path / 'op2', arch='sm90a')
    assert _preflight(forgekernel.render_job(other, forgekernel_root='/f', runs_dir='/r')) == []
    toml = task_dir / 'task.toml'
    toml.write_text(toml.read_text().replace(f'language = "{language}"', 'language = "x"'))
    with pytest.raises(ValueError, match='language must be one of'):
        forgekernel.render_job(task_dir, forgekernel_root='/f', runs_dir='/r')


# -- KernelAgent ----------------------------------------------------------------------------


def _assert_spec_phrases(text):
    for phrase in ('kernel_function(', 'float32', 'sm_80', 'no TMA', 'tensor descriptors',
                   'warp_specialize', 'no FP8', '.item()', '.cpu()', 'synchronize',
                   'data-dependent', 'torch.empty', 'At most 4 @triton.autotune configs',
                   'second, different input draw', 'CUDA graph', 'PASS'):
        assert phrase in text, phrase


def test_kernelagent_prepare_problem_linear(tmp_path):
    capture = _linear_capture('/abs/stage_tensors.pt')
    stage = kernelagent.prepare_problem(capture, tmp_path / 'stage', rel_l2=1e-3)
    texts = {name: (stage / name).read_text() for name in ('problem.py', 'test.py',
                                                           'run_ka.py')}
    for name, text in texts.items():
        compile(text, name, 'exec')
    problem, test, runner = texts['problem.py'], texts['test.py'], texts['run_ka.py']
    tree = ast.parse(problem)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '__init__' for n in model.body)
    assert 'def get_init_inputs():\n    return []' in problem
    assert "TENSORS = '/abs/stage_tensors.pt'" in problem and "TENSORS = '/abs/stage_tensors.pt'" in test
    assert 'F.linear(x, weight, bias)' in problem
    test_doc = ast.get_docstring(ast.parse(test))
    _assert_spec_phrases(test_doc)
    assert 'kernel_function(x, weight, bias) -> out' in test_doc
    assert 'assert x.dtype == torch.bfloat16' in test_doc
    assert 'x: [392, 1152] bfloat16 (input)' in test_doc and 'M=588' in test_doc
    assert 'max(0.001, 2 x' in test_doc
    test_tree = ast.parse(test)
    modules = {node.module for node in ast.walk(test_tree) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(test_tree) if isinstance(node, ast.Import)
                for alias in node.names}
    assert 'problem' not in modules and 'kernel' in modules
    for snippet in ('torch.addmm(bias.float(), x.float(), weight.float().t())',
                    'torch.cuda.CUDAGraph()', 'torch.cuda.Stream()', '.copy_(',
                    'graph.replay()', 'torch.isfinite', 'REL_L2_TOL = 0.001',
                    'COSINE_MIN = 0.99999', 'print("FAIL" if failures else "PASS")',
                    'sys.exit(main())', 'json.dumps({METRICS_KEY'):
        assert snippet in test, snippet
    for snippet in ('is_model_available', 'TritonKernelAgent', 'kernel_sha256',
                    'upstream_commit', 'session_dir', 'ka_result.json', 'kernel.py',
                    'generate_default_test=False', f"PINNED_COMMIT = '{KA_PIN}'",
                    f"STAGE_DIR = '{stage}'", 'return 0 if', '        return 3'):
        assert snippet in runner, snippet


def test_kernelagent_prepare_problem_attention_custom_and_errors(tmp_path):
    stage = kernelagent.prepare_problem(_attention_capture('/abs/t.npz', gqa=True),
                                        tmp_path / 'attn')
    test = (stage / 'test.py').read_text()
    compile(test, 'test.py', 'exec')
    assert 'F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True)' \
        in test
    custom = _custom_capture(tmp_path, '/abs/c.pt')
    stage = kernelagent.prepare_problem(custom, tmp_path / 'custom')
    for name in ('problem.py', 'test.py'):
        text = (stage / name).read_text()
        compile(text, name, 'exec')
        assert 'return x * 2.0' in text and '_REFERENCE["run"]' in text
    with pytest.raises(ValueError, match='tensors_path is required'):
        kernelagent.prepare_problem(_linear_capture(), tmp_path / 'no_tensors')
    with pytest.raises(ValueError, match='absolute'):
        kernelagent.prepare_problem(_linear_capture('rel.pt'), tmp_path / 'rel')
    with pytest.raises(ValueError, match='cosine'):
        kernelagent.prepare_problem(_linear_capture('/abs/t.pt'), tmp_path / 'c', cosine=2)
    assert not (tmp_path / 'c').exists()


def test_kernelagent_render_job(tmp_path):
    stage = kernelagent.prepare_problem(_linear_capture('/abs/t.pt'), tmp_path / 'st')
    script = kernelagent.render_job(stage, kernelagent_root='/opt/KernelAgent',
                                    env_file='/opt/ka.env')
    for expected in ('set -euo pipefail', 'source /opt/ka.env', 'cd /opt/KernelAgent',
                     'export PYTHONPATH=/opt/KernelAgent', KA_PIN,
                     f'python {stage}/run_ka.py --stage-dir {stage} --model '
                     'claude-sonnet-4-6 --workers 3 --rounds 8', 'KA_RC=$RC'):
        assert expected in script, expected
    with pytest.raises(ValueError, match='workers <= 3'):
        kernelagent.render_job(stage, kernelagent_root='/opt/KA', workers=4)
    assert '--workers 4' in kernelagent.render_job(stage, kernelagent_root='/opt/KA',
                                                    model='gpt-5', workers=4)


def _ka_stage(tmp_path, *, success=True, sha=None, retest=True, change_test=False):
    stage = kernelagent.prepare_problem(_linear_capture('/abs/t.pt'), tmp_path / 'stage')
    kernel = b'import triton\n\ndef kernel_function(x, weight, bias):\n    pass\n'
    result = {'schema': kernelagent.RESULT_SCHEMA, 'success': success,
              'model': 'claude-sonnet-4-6', 'session_dir': '/logs/session_1', 'worker_id': 1,
              'rounds': 2, 'upstream_commit': KA_PIN,
              'test_sha256': hashlib.sha256((stage / 'test.py').read_bytes()).hexdigest()}
    if success:
        (stage / 'kernel.py').write_bytes(kernel)
        result['kernel_sha256'] = sha or hashlib.sha256(kernel).hexdigest()
        if retest:
            result['retest'] = {'returncode': 0, 'passed': True,
                                'metrics': {'ok': True, 'metrics': {'captured': {
                                    'rel_l2': 1e-3, 'cosine': 0.99999}}}}
    else:
        result['message'] = 'Failed to generate working kernel'
    (stage / 'ka_result.json').write_text(json.dumps(result))
    if change_test:
        (stage / 'test.py').write_text((stage / 'test.py').read_text() + '\n# edited\n')
    return stage


def test_kernelagent_harvest_success(tmp_path):
    stage = _ka_stage(tmp_path)
    artifact = kernelagent.harvest(stage, stage_id='kernel.gemm', model='hyvla')
    assert artifact.accepted and artifact.language == 'triton'
    assert artifact.entry_symbol == 'kernel_function' and artifact.upstream_commit == KA_PIN
    assert artifact.op_metrics['metrics']['captured']['cosine'] == 0.99999
    assert artifact.provenance['ka_result']['session_dir'] == '/logs/session_1'
    assert artifact.provenance['first_valid_not_fastest'] is True


@pytest.mark.parametrize('changes,reason', [
    ({'success': False}, 'no success'),
    ({'sha': 'f' * 64}, 'sha256'),
    ({'change_test': True}, 'test.py changed'),
])
def test_kernelagent_harvest_rejections(tmp_path, changes, reason):
    artifact = kernelagent.harvest(_ka_stage(tmp_path, **changes), stage_id='kernel.gemm',
                                   model='hyvla')
    assert not artifact.accepted
    assert any(reason in item for item in artifact.reasons), artifact.reasons


def test_kernelagent_harvest_metrics_from_output_and_missing(tmp_path):
    stage = _ka_stage(tmp_path, retest=False)
    (stage / 'test_output.txt').write_text(
        'noise\n{"flashrt_kernel_test": {"ok": true, "metrics": {}}}\nPASS\n')
    artifact = kernelagent.harvest(stage, stage_id='kernel.gemm', model='hyvla')
    assert artifact.accepted and artifact.op_metrics == {'ok': True, 'metrics': {}}
    assert any('no re-test' in note for note in artifact.provenance['notes'])
    missing = kernelagent.harvest(tmp_path / 'none', stage_id='kernel.gemm', model='m')
    assert not missing.accepted and missing.kernel_path is None
    assert any('ka_result.json missing' in r for r in missing.reasons)


def _stub_torch(monkeypatch):
    torch = types.ModuleType('torch')
    torch.nn = types.ModuleType('torch.nn')
    torch.nn.functional = types.ModuleType('torch.nn.functional')
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(
        matmul=types.SimpleNamespace(allow_tf32=True)),
        cudnn=types.SimpleNamespace(allow_tf32=True))
    torch.device = lambda name: name
    torch.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda seed: None)
    kernel = types.ModuleType('kernel')
    kernel.kernel_function = lambda *args: None
    for name, module in (('torch', torch), ('torch.nn', torch.nn),
                         ('torch.nn.functional', torch.nn.functional), ('kernel', kernel)):
        monkeypatch.setitem(sys.modules, name, module)


def _run_ka_test(monkeypatch, capsys, test, fail_label=None):
    """Execute a generated test.py main() with torch, the kernel and tensors stubbed."""
    _stub_torch(monkeypatch)
    namespace = {'__name__': 'flashrt_generated_test'}
    exec(compile(test, 'test.py', 'exec'), namespace)
    calls = []

    def call(label, inputs, failures, report, output=namespace['OUTPUT']):
        calls.append((label, inputs, output))
        if label == fail_label:
            failures.append(f'{label}: wrong')

    namespace.update(
        _load_inputs=lambda device: [f'captured_{row[0]}' for row in namespace['INPUTS']],
        _fresh_like=lambda tensor, generator, shape=None: ('fresh', tensor, shape),
        _call=call, _graph_check=lambda inputs, generator, failures, report: None)
    rc = namespace['main']()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('{'))
    return rc, calls, json.loads(line)[kernelagent.METRICS_KEY]


def test_kernelagent_test_verifies_every_captured_shape(tmp_path, monkeypatch, capsys):
    stage = kernelagent.prepare_problem(_linear_capture('/abs/t.pt'), tmp_path / 'stage')
    test = (stage / 'test.py').read_text()
    shapes = ast.literal_eval(test.split('SHAPES = ', 1)[1].split('\n', 1)[0])
    assert [row['name'] for row in shapes] == ['cam2', 'cam3']
    assert shapes[1]['inputs'] == [('x', [588, 1152], 'bfloat16', 'input'),
                                   ('weight', [2048, 1152], 'bfloat16', 'weight'),
                                   ('bias', [2048], 'bfloat16', 'weight')]
    assert shapes[1]['output'] == ('out', [588, 2048], 'bfloat16')
    assert 'Checks 1 and 2 at every other production shape' in ast.get_docstring(
        ast.parse(test))

    rc, calls, metrics = _run_ka_test(monkeypatch, capsys, test)
    assert rc == 0 and metrics['ok'] is True
    assert [label for label, _, _ in calls] == ['captured', 'second_draw', 'shape_1_cam3']
    _, inputs, output = calls[-1]
    assert output == ('out', [588, 2048], 'bfloat16')
    assert inputs == [('fresh', 'captured_x', [588, 1152]),
                      ('fresh', 'captured_weight', [2048, 1152]),
                      ('fresh', 'captured_bias', [2048])]
    assert metrics['shapes_total'] == ['cam2', 'cam3']
    assert metrics['shapes_verified'] == ['cam2', 'cam3']

    rc, _, metrics = _run_ka_test(monkeypatch, capsys, test, fail_label='shape_1_cam3')
    assert rc == 1 and metrics['shapes_verified'] == ['cam2']
    rc, _, metrics = _run_ka_test(monkeypatch, capsys, test, fail_label='captured')
    assert rc == 1 and metrics['shapes_verified'] == ['cam3']


def test_kernelagent_harvest_rejects_unverified_shapes(tmp_path):
    stage = _ka_stage(tmp_path)
    result = json.loads((stage / 'ka_result.json').read_text())
    result['retest']['metrics'].update(shapes_total=['cam2', 'cam3'],
                                       shapes_verified=['cam2'])
    (stage / 'ka_result.json').write_text(json.dumps(result))
    artifact = kernelagent.harvest(stage, stage_id='kernel.gemm', model='hyvla')
    assert not artifact.accepted
    assert any("did not verify captured shapes ['cam3']" in r for r in artifact.reasons)
    assert artifact.op_metrics['shapes_verified'] == ['cam2']

    result['retest']['metrics']['shapes_verified'] = ['cam2', 'cam3']
    (stage / 'ka_result.json').write_text(json.dumps(result))
    artifact = kernelagent.harvest(stage, stage_id='kernel.gemm', model='hyvla')
    assert artifact.accepted
    assert artifact.op_metrics['shapes_total'] == artifact.op_metrics['shapes_verified']

    result['retest']['metrics'].update(shapes_total=[], shapes_verified=[])
    (stage / 'ka_result.json').write_text(json.dumps(result))
    assert any('shapes_total is empty' in r for r in kernelagent.harvest(
        stage, stage_id='kernel.gemm', model='hyvla').reasons)
    old = kernelagent.harvest(_ka_stage(tmp_path / 'old'), stage_id='kernel.gemm',
                              model='hyvla')
    assert old.accepted
    assert any('only the first captured shape' in n for n in old.provenance['notes'])


def _echo_line(script, prefix):
    return next(line for line in script.splitlines() if line.startswith(f'echo "{prefix}'))


def test_rendered_echo_lines_quote_paths(tmp_path):
    root = tmp_path / 'st$(touch INJECTED)`touch TICK`'
    stage = kernelagent.prepare_problem(_linear_capture('/abs/t.pt'), root)
    script = kernelagent.render_job(stage, kernelagent_root='/opt/KA')
    lines = [_echo_line(script, 'KA_RC=')]
    out, python_out = str(tmp_path / 'o$(touch INJECTED).json'), str(tmp_path / 'p"$HOME".json')
    bench = apxinf.render_bench(None, apxinf_root='/opt/ApxInf', out=out, python_out=python_out)
    lines += [_echo_line(bench, 'APXINF_RUST_RC='), _echo_line(bench, 'APXINF_PY_RC=')]
    printed = []
    for line in lines:
        proc = subprocess.run(['bash', '-c', f'set -euo pipefail; RC=0; PY_RC=0; {line}'],
                              cwd=tmp_path, capture_output=True, text=True, check=True)
        printed.append(proc.stdout.strip())
    assert printed == [f'KA_RC=0 STAGE={stage}', f'APXINF_RUST_RC=0 OUT={out}',
                       f'APXINF_PY_RC=0 OUT={python_out}']
    assert not (tmp_path / 'INJECTED').exists() and not (tmp_path / 'TICK').exists()


# -- ApxInf ---------------------------------------------------------------------------------


def test_apxinf_render_bench():
    script = apxinf.render_bench('/ckpt/pi05', apxinf_root='/opt/ApxInf', token_count=21,
                                 out='/out/rust.json', python_out='/out/l1.json')
    for expected in ('cd /opt/ApxInf', 'APXINF_CUDA_ARCH', APX_PIN,
                     'cargo run --release -p apxinf-model --features cuda --example pi05_bench'
                     ' -- /ckpt/pi05 --model-variant bf16 --views 2 --token-count 21 '
                     '--iterations 30 > /out/rust.json',
                     'python scripts/bench_pi05.py --layer l1 --model-variant bf16 '
                     '--model-dir /ckpt/pi05 --warmup 10 --samples 30 --out /out/l1.json',
                     'CUDA preprocessing'):
        assert expected in script, expected
    random_script = apxinf.render_bench(None, apxinf_root='/opt/ApxInf', views=3,
                                        iterations=50, out='/o/r.json',
                                        python_out='/o/p.json')
    assert ('pi05_bench -- random --model-variant bf16 --views 3 --token-count 10 '
            '--iterations 50') in random_script
    assert '--random-weights --views 3 --token-count 10' in random_script
    assert 'bench_pi05' not in apxinf.render_bench(None, apxinf_root='/a', out='/o.json')
    for kwargs, match in (({'variant': 'fp4'}, 'variant'), ({'views': 0}, 'views'),
                          ({'out': 'rel.json'}, 'absolute')):
        args = {'apxinf_root': '/a', 'out': '/o.json', **kwargs}
        with pytest.raises(ValueError, match=match):
            apxinf.render_bench(None, **args)


def test_apxinf_render_bench_passes_one_token_count_to_both_benches():
    assert apxinf.PROTOCOL_TOKEN_COUNT == 10
    for token_count, expected in ((None, 10), (21, 21)):
        kwargs = {} if token_count is None else {'token_count': token_count}
        script = apxinf.render_bench(None, apxinf_root='/opt/ApxInf', out='/o/r.json',
                                     python_out='/o/p.json', **kwargs)
        rust = next(line for line in script.splitlines() if 'pi05_bench' in line)
        python = next(line for line in script.splitlines() if 'bench_pi05.py' in line)
        assert f'--token-count {expected} ' in rust
        assert f'--token-count {expected} ' in python
    checkpoint = apxinf.render_bench('/ckpt/pi05', apxinf_root='/opt/ApxInf', out='/o/r.json',
                                     python_out='/o/p.json')
    assert '--token-count 10 --iterations 30' in checkpoint
    assert 'the prompt must have 10 tokens to match the Rust bench' in checkpoint
    with pytest.raises(ValueError, match='token_count'):
        apxinf.render_bench(None, apxinf_root='/a', out='/o.json', token_count=None)


def test_apxinf_parse_report_rust_python_and_unknown(tmp_path):
    rust = tmp_path / 'rust.json'
    rust.write_text('cargo noise\n' + json.dumps({
        'passed': True, 'model_variant': 'bf16', 'token_count': 21,
        'image_input': {'graph_includes_cuda_preprocess': True,
                        'input_update_plus_graph_latency_ms': None},
        'latency_ms': {'samples_ms': [1.0, 2.0, 3.0], 'p50': 2.0, 'p95': 3.0, 'min': 1.0,
                       'max': 3.0, 'mean': 2.0, 'standard_deviation': 0.8}}) + '\n')
    report = apxinf.parse_report(rust)
    assert report['source_format'] == 'apxinf.pi05_bench' and report['passed'] is True
    assert report['latency_ms'] == {'graph_replay': {'p50': 2.0, 'p95': 3.0, 'min': 1.0,
                                                     'max': 3.0, 'mean': 2.0, 'std': 0.8,
                                                     'samples': 3}}
    assert report['token_count'] == 21 and 'boundary' in report
    assert report['raw']['model_variant'] == 'bf16'

    python = tmp_path / 'py.json'
    python.write_text(json.dumps({
        'schema': 'apxinf.pi05.latency.v2', 'model_variant': 'bf16',
        'workload': {'token_count': 10},
        'layers_ms': {'L1_rust': {'p50': 30.5, 'p95': 31.0, 'samples': 30},
                      'L0_model': {'p50': 'n/a'}}}))
    report = apxinf.parse_report(python)
    assert report['latency_ms'] == {'L1_rust': {'p50': 30.5, 'p95': 31.0, 'samples': 30}}
    assert report['passed'] is None and report['token_count'] == 10

    unknown = tmp_path / 'u.json'
    unknown.write_text(json.dumps({'hello': 1}))
    report = apxinf.parse_report(unknown)
    assert report['source_format'] == 'unknown' and report['latency_ms'] == {}
    garbage = tmp_path / 'g.txt'
    garbage.write_text('not json at all\n')
    with pytest.raises(ValueError, match='no JSON object'):
        apxinf.parse_report(garbage)


def test_capture_calls_saves_torch_tensors_when_torch_is_available(tmp_path):
    torch = pytest.importorskip('torch')
    import torch.nn.functional as F

    owner = types.SimpleNamespace(linear=F.linear)
    x, weight = torch.randn(3, 5, 8), torch.randn(16, 8)
    bias = torch.randn(16)
    with capture_calls(owner, 'linear', model='hyvla', stage_id='kernel.gemm',
                       call_site='t.py:f', op_kind='linear', out_dir=tmp_path) as recorder:
        expected = owner.linear(x, weight, bias)
    assert owner.linear is F.linear
    saved = torch.load(recorder.capture.tensors_path, weights_only=True)
    assert set(saved) == {e['name'] for e in recorder.capture.signature}
    assert torch.equal(saved[recorder.capture.signature[-1]['name']], expected)
    task_dir = forgekernel.prepare_task(recorder.capture, tmp_path / 'task')
    namespace = {}
    exec(compile((task_dir / 'reference' / 'run.py').read_text(), 'run.py', 'exec'), namespace)
    inputs = [saved[e['name']] for e in recorder.capture.inputs()]
    assert torch.allclose(namespace['run'](inputs[0].reshape(-1, 8), *inputs[1:]),
                          expected.reshape(-1, 16), atol=1e-5)
