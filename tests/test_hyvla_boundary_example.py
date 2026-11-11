import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/vla_autotune'


@pytest.fixture
def diag():
    path = EXAMPLES / 'hy_trace/diagnose.py'
    spec = importlib.util.spec_from_file_location('hy_boundary_diag', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_alignment_restores_tokens_and_repeats_heads_not_tiles(diag):
    # Official [layer,batch,prefix,kvhead,width], two distinguishable heads.
    original = np.array([[[[[10], [100]], [[20], [200]], [[30], [300]]]]], dtype=np.float32)
    perm = np.array([2, 0, 1])
    expanded = diag.expand_official_kv(original, perm, query_heads=4, suffix_rows=2)
    assert expanded.shape == (1, 1, 4, 5, 1)
    np.testing.assert_array_equal(expanded[0, 0, :, :3, 0],
                                  [[30, 10, 20], [30, 10, 20], [300, 100, 200], [300, 100, 200]])
    assert not expanded[..., 3:, :].any()
    np.testing.assert_array_equal(diag.canonical_native_kv(expanded, perm, kv_heads=2), original)
    expanded[0, 0, 1, 0, 0] += 1
    with pytest.raises(ValueError, match='repeated'):
        diag.canonical_native_kv(expanded, perm, kv_heads=2)


def test_invalid_permutation_cannot_silently_duplicate_prefix_rows(diag):
    with pytest.raises(ValueError, match='permutation'):
        diag.expand_official_kv(np.zeros((1, 1, 3, 2, 1)), np.array([0, 0, 2]), 4, 2)


def test_guard_rejects_login_and_batch_before_model_import(diag):
    with pytest.raises(RuntimeError):
        diag.require_step({'SLURM_JOB_ID': '1', 'SLURM_STEP_ID': 'batch'}, 'ln02')
    diag.require_step({'SLURM_JOB_ID': '1', 'SLURM_STEP_ID': '0',
                       'SLURMD_NODENAME': 'gpu01', 'CUDA_VISIBLE_DEVICES': '0'}, 'gpu01')


def test_observer_extraction_never_executes_hash_or_module_level_code(diag, tmp_path):
    source = tmp_path / 'old_observer.py'
    source.write_text('raise AssertionError("module executed")\n'
                      'def build_manifest():\n    raise AssertionError("hashing")\n'
                      'def validate_load_report(a,b):\n    return (a,b)\n'
                      '@contextmanager\ndef diagnostic_capture(policy,tensors):\n    yield\n')
    validate, observe = diag.observer_functions(source)
    assert validate(['missing'], []) == (['missing'], [])
    with observe(None, {}):
        pass


def test_method_observation_restores_class_binding_after_failure(diag):
    from contextlib import ExitStack
    class Owner:
        def add(self, value):
            return value + 1
    obj = Owner()
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            diag.wrap_method(stack, obj, 'add', lambda original: lambda value: original(value) * 2)
            assert obj.add(2) == 6
            raise RuntimeError('inference failed')
    assert obj.add(2) == 3 and 'add' not in obj.__dict__


def test_alignment_does_not_mix_layers_batches_or_head_groups(diag):
    native = np.zeros((2, 2, 4, 4, 2), np.float32)
    for layer in range(2):
        for batch in range(2):
            for group in range(2):
                for row in range(3):
                    native[layer, batch, group*2:group*2+2, row] = [1000*layer+100*batch+10*group+2*row,
                                                                 1000*layer+100*batch+10*group+2*row+1]
    original = diag.canonical_native_kv(native, np.array([2, 0, 1]), 2)
    assert original.shape == (2, 2, 3, 2, 2)
    np.testing.assert_array_equal(original[1, 1, :, 1, 1], [1113, 1115, 1111])


def test_cli_help_and_unallocated_guard_without_creating_outputs(tmp_path):
    script = EXAMPLES / 'hy_trace/diagnose.py'
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
    help_result = subprocess.run([sys.executable, str(script), '--help'], env=env, text=True, capture_output=True)
    assert help_result.returncode == 0 and '--engine' in help_result.stdout
    args = [sys.executable, str(script), '--engine', 'official']
    for flag in ('checkpoint', 'fixtures', 'official-source', 'observer-source'):
        args.extend(['--' + flag, '/does-not-exist'])
    args.extend(['--output', str(tmp_path / 'unused')])
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'allocated GPU srun' in result.stderr
    assert not (tmp_path / 'unused').exists()


def admission_inputs():
    import torch
    native = {'prepared.images':torch.zeros(3,2,3,2,2),
              'prepared.state': torch.zeros(1, 4), 'prepared.tokens': torch.tensor([[1, 5, 8]]),
              'prepared.token_mask': torch.tensor([[True, True, False]]),
              'prefix.pad_mask': torch.tensor([[True, True, False]]),
              'prefix.modality_mask': torch.tensor([[True, False, True]])}
    reference = {key: value.clone() for key, value in native.items()}
    reference['actions'] = torch.zeros(1, 2, 3)
    reference['step0.latent_before'] = torch.ones(1, 2, 3)
    data = {'expected_actions32': np.zeros((2, 3), np.float32), 'noise': np.ones((2, 3), np.float32)}
    report = {'variant': 'baseline', 'oracle': {'exact': True}, 'prefix_invariance': {
        name: True for name in ('prepared.images', 'prefix.embeddings', 'prefix.pad_mask',
                                'prefix.modality_mask', 'prefix.k', 'prefix.v')}}
    return reference, native, data, report


@pytest.mark.parametrize('bad', ['actions', 'noise', 'prepared.images', 'prepared.state', 'prepared.tokens',
                                'prepared.token_mask', 'prefix.pad_mask', 'prefix.modality_mask', 'invariance'])
def test_mismatched_reference_never_calls_frozen_evaluator(diag, bad):
    import torch
    reference, native, data, report = admission_inputs()
    if bad == 'invariance':
        report['prefix_invariance']['prefix.k'] = False
    else:
        key = {'actions': 'actions', 'noise': 'step0.latent_before'}.get(bad, bad)
        value = reference[key].reshape(-1)
        value[0] = not bool(value[0]) if value.dtype == torch.bool else value[0] + 1
    calls = []
    with pytest.raises(ValueError, match='reference|input|invariance'):
        diag.admitted_frozen_evaluation(reference, native, data, report, 'baseline', lambda: calls.append('wrong'))
    assert calls == []


def test_admission_allows_the_prefix_numeric_differences_being_investigated(diag):
    import torch
    reference, native, data, report = admission_inputs()
    reference['prefix.k'], native['prefix.k'] = torch.zeros(2), torch.ones(2)
    result = diag.admitted_frozen_evaluation(reference, native, data, report, 'baseline', lambda: 'evaluated')
    assert result == 'evaluated'


def test_reference_input_dtype_changes_cannot_be_hidden_by_float_array_conversion(diag):
    reference,native,data,report=admission_inputs()
    reference['prepared.tokens']=reference['prepared.tokens'].double()
    called=[]
    with pytest.raises(ValueError,match='dtype|input'):
        diag.admitted_frozen_evaluation(reference,native,data,report,'baseline',lambda:called.append(True))
    assert called==[]


def test_folded_cli_requires_historical_source_and_prior_identity_before_loading(diag):
    args=SimpleNamespace(engine='native',folded_proj1=True,variants=None,preflight_report=Path('/proof'),
        source_revision='0'*40,prior_job='309681',reference_job='308690',reference=Path('/official'),prior=Path('/prior'))
    assert diag.validate_trace_options(args)==('baseline','state')
    args.prior_job=None
    with pytest.raises(ValueError,match='prior|identity'):diag.validate_trace_options(args)
    args.prior_job='309681';args.engine='official'
    with pytest.raises(ValueError,match='native|folded'):diag.validate_trace_options(args)


def test_official_attention_capture_cli_is_optin_and_binds_historical_reference(diag):
    args=SimpleNamespace(engine='official',folded_proj1=False,capture_eager_attention=True,
        variants=None,source_revision='0'*40,reference=Path('/official'),reference_job='308690',
        preflight_report=None,prior_job=None)
    assert diag.validate_trace_options(args)==('baseline','state')
    args.engine='native'
    with pytest.raises(ValueError,match='official|capture'):diag.validate_trace_options(args)
    args.engine='official';args.reference_job=None
    with pytest.raises(ValueError,match='reference|capture'):diag.validate_trace_options(args)


def folded_publication(tmp_path):
    import torch
    folder=tmp_path/'state';folder.mkdir()
    return {'folder':folder,'tensors':{'actions':torch.zeros(1,2,1)},
        'frozen':{'velocity':torch.ones(1,2,1)},
        'report':{'variant':'state','diagnostic_cleanup':{'passed':True},
                  'reference_boundaries':{},'frozen_velocity_comparisons':[]}}


def test_frozen_write_failure_never_publishes_completed_result_or_success_print(diag,tmp_path,monkeypatch,capsys):
    import safetensors.torch
    item=folded_publication(tmp_path)
    real_save=safetensors.torch.save_file
    def fail(tensors,path,*args,**kwargs):
        if Path(path).name=='frozen_velocities.safetensors':raise OSError('injected frozen write failure')
        return real_save(tensors,path,*args,**kwargs)
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='frozen write'):
        diag.publish_folded_trace(item)
    assert not (item['folder']/'result.json').exists() and capsys.readouterr().out==''


def test_folded_raw_artifacts_precede_atomic_completion_marker(diag,tmp_path,monkeypatch,capsys):
    import json
    import safetensors.torch
    item=folded_publication(tmp_path);events=[]
    real_save=safetensors.torch.save_file;real_replace=Path.replace
    def save(tensors,path,*args,**kwargs):
        assert not (item['folder']/'result.json').exists()
        real_save(tensors,path,*args,**kwargs);events.append(Path(path).name)
    def replace(path,target):
        assert (item['folder']/'frozen_velocities.safetensors').is_file()
        assert (item['folder']/'trace.safetensors').is_file()
        events.append(Path(target).name);return real_replace(path,target)
    monkeypatch.setattr(safetensors.torch,'save_file',save);monkeypatch.setattr(Path,'replace',replace)
    diag.publish_folded_trace(item)
    assert events==['frozen_velocities.safetensors','trace.safetensors','result.json']
    assert json.loads((item['folder']/'result.json').read_text())['artifact_complete'] is True
    assert 'cleanup_passed' in capsys.readouterr().out


def test_atomic_json_failure_has_no_success_marker_or_print(diag,tmp_path,monkeypatch,capsys):
    item=folded_publication(tmp_path)
    def fail(*args,**kwargs):raise OSError('injected publication failure')
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError,match='publication failure'):diag.publish_folded_trace(item)
    assert not (item['folder']/'result.json').exists() and capsys.readouterr().out==''


def test_live_cache_and_expanded_time_strides_are_not_canonical_copy_strides(diag):
    import torch
    # Expanded repeated heads have zero stride in the live Q-head axis.
    cache = torch.arange(8.).reshape(1, 1, 1, 4, 2).expand(1, 1, 4, 4, 2)
    before, strides = cache.clone(), cache.stride()
    tensors, meta = {}, {}
    diag.record_native_cache(tensors, meta, 'prefix.k', cache, np.array([2, 0, 1]), 1)
    assert meta['prefix.k.live_cache']['live_stride'] == list(strides)
    assert 'live_stride' not in meta['prefix.k']
    assert meta['prefix.k']['artifact_stride'] == list(tensors['prefix.k'].stride())
    torch.testing.assert_close(cache, before, atol=0, rtol=0)
    assert cache.stride() == strides
    class Model:
        def denoise_step(self, state, pad, kv, latent, time):
            return latent * 2, None
    model = Model(); latent = torch.arange(12.).reshape(3, 4).t(); time = torch.tensor(1.).expand(3)
    with diag.observe_step_metadata(model, meta):
        model.denoise_step(None, None, None, latent, time)
    assert meta['step0.latent_before']['live_stride'] == list(latent.stride())
    assert meta['step0.time']['live_stride'] == [0]
    assert 'denoise_step' not in model.__dict__


def test_packaged_observer_does_not_require_a_task_local_harness(diag):
    validate, observe = diag.observer_functions(EXAMPLES / 'hy_trace/reference_observer.py')
    report = validate(['model.dual_tower.expert.lm_head.weight'], [])
    assert report['missing_inference_keys'] == []
    assert callable(observe)
    with pytest.raises(RuntimeError, match='missing inference tensors'):
        validate(['model.action_out_proj.weight'], [])
