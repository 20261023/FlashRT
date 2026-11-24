import json
import math

import pytest

from flash_rt.vla.autotune import stages
from flash_rt.vla.autotune.stages import (
    MODEL_STAGES, PROFILE_KEYS, REFERENCED_TOOL_IDS, STAGES, amdahl_ceiling, attribute,
    missing_seams, parse_seam, rank, seam_defined, stage, stages_for)


def _record(model, intervals, *, wall_ms, nfe=None, sampler='euler', sums=True):
    metrics = {'measurement_kind': 'diagnostic_cuda_event_intervals',
               'separate_from_latency_samples': True, 'wall_ms': wall_ms,
               'interval_ms': intervals,
               'sampler_metadata': None if nfe is None else {'sampler': sampler, 'nfe': nfe},
               'bound_classification': 'unknown_without_kernel_counters_or_dataflow_measurements'}
    if sums:
        metrics['interval_sum_ms'] = {key: sum(values) for key, values in intervals.items()}
    return {'schema_version': 2, 'state': 'measured',
            'candidate': {'candidate_id': 'c', 'workload_id': 'w',
                          'backend': {'model': model, 'checkpoint': '/w', 'sampler': sampler}},
            'stage_metrics': metrics}


def test_every_seam_exists_in_repo():
    seams = stages.all_seams()
    assert len(seams) > 40
    assert missing_seams() == []
    for seam in seams:
        assert seam_defined(seam), seam


def test_seam_checker_rejects_missing_names(tmp_path):
    module = tmp_path / 'pkg' / 'mod.py'
    module.parent.mkdir()
    module.write_text('def top():\n    pass\n\nclass Owner:\n    def method(self):\n        def inner():\n            pass\n')
    assert seam_defined('pkg/mod.py:top', tmp_path)
    assert seam_defined('pkg/mod.py:Owner', tmp_path)
    assert seam_defined('pkg/mod.py:Owner.method', tmp_path)
    assert not seam_defined('pkg/mod.py:inner', tmp_path)
    assert not seam_defined('pkg/mod.py:Owner.missing', tmp_path)
    assert not seam_defined('pkg/mod.py:top.method', tmp_path)
    assert not seam_defined('pkg/absent.py:top', tmp_path)
    for bad in ('pkg/mod.py', '/abs/mod.py:top', '../mod.py:top', 'pkg/mod.txt:top',
                'pkg/mod.py:a.b.c', 'pkg/mod.py:not-id'):
        with pytest.raises(ValueError):
            parse_seam(bad)


@pytest.mark.parametrize('required', [
    'flash_rt/models/pi0/pipeline_rtx.py:Pi0Pipeline.vision_encoder',
    'flash_rt/models/pi0/pipeline_rtx.py:Pi0Pipeline.transformer_encoder',
    'flash_rt/models/pi0/pipeline_rtx.py:Pi0Pipeline.transformer_decoder',
    'flash_rt/models/pi05/pipeline_rtx.py:Pi05Pipeline.vision_encoder',
    'flash_rt/models/pi05/pipeline_rtx.py:Pi05Pipeline.transformer_encoder',
    'flash_rt/models/pi05/pipeline_rtx.py:Pi05Pipeline.transformer_decoder',
    'flash_rt/models/hyvla/pipeline_thor.py:HyVLAThorBF16Pipeline.vit_forward',
    'flash_rt/models/hyvla/pipeline_thor.py:HyVLAThorBF16Pipeline.merger_forward',
    'flash_rt/models/hyvla/pipeline_thor.py:HyVLAThorBF16Pipeline.prefill',
    'flash_rt/models/hyvla/pipeline_thor.py:HyVLAThorBF16Pipeline.evaluate_velocity',
    'flash_rt/frontends/torch/hyvla_thor.py:HyVLATorchFrontendThor._preprocess_images',
    'flash_rt/frontends/torch/hyvla_thor.py:HyVLATorchFrontendThor._assemble_prefix',
    'flash_rt/frontends/torch/hyvla_probeflow.py:HyVLAFlowFrontend._graph_forward',
    'flash_rt/frontends/torch/hyvla_probeflow.py:HyVLAFlowFrontend._sample_latent',
    'flash_rt/models/fastwam/pipeline.py:FastWAMActionPipeline._prepare_condition',
    'flash_rt/models/fastwam/pipeline.py:FastWAMActionPipeline._evaluate_at_timestep',
    'flash_rt/models/fastwam/pipeline.py:FastWAMActionPipeline._euler',
    'flash_rt/sampling/torch_probeflow.py:sample_probeflow_torch_',
    'flash_rt/vla/backend.py:VLABackend.infer',
])
def test_dataflow_seams_are_mapped(required):
    assert required in stages.all_seams()


def test_stage_tables_are_consistent():
    assert set(MODEL_STAGES) == set(PROFILE_KEYS) == {'pi0', 'pi05', 'hyvla', 'fastwam', 'dm05'}
    for model, items in MODEL_STAGES.items():
        assert items[0].stage_id == 'host.input'
        for item in items:
            assert item.stage_id in STAGES
            assert item.seams, (model, item.stage_id)
            for key in item.profiling_keys:
                assert item.stage_id in PROFILE_KEYS[model][key]
        for key, covered in PROFILE_KEYS[model].items():
            assert covered and all(stage_id in STAGES for stage_id in covered)
            mapped = {item.stage_id for item in items}
            assert set(covered) <= mapped, (model, key)
            for item in items:
                assert (key in item.profiling_keys) == (item.stage_id in covered), (model, key, item.stage_id)
    for info in STAGES.values():
        assert info.kind in stages.STAGE_KINDS
        assert info.playbook and all(row in stages.PLAYBOOK for row in info.playbook)
        assert info.default_tools and set(info.default_tools) <= REFERENCED_TOOL_IDS
        assert info.id == info.stage_id
    used = {tool for info in STAGES.values() for tool in info.default_tools}
    assert used == REFERENCED_TOOL_IDS
    assert set(stages.STAGE_IDS) == {
        'host.input', 'vision.encode', 'vision.project', 'prefix.prefill', 'denoise.velocity',
        'denoise.schedule', 'denoise.routing', 'host.launch', 'host.output', 'kernel.gemm',
        'kernel.attention', 'kernel.norm_elementwise', 'kernel.conv', 'precision.weights'}
    assert stage('denoise.schedule').kind == 'count'
    assert stage('kernel.conv').kind == 'kernel'
    assert 'sampler.probeflow' in stage('denoise.schedule').default_tools


def test_profile_keys_follow_contract():
    full = ('vision.encode', 'vision.project', 'prefix.prefill', 'denoise.velocity')
    for model in ('pi0', 'pi05'):
        assert PROFILE_KEYS[model] == {
            'full': full, 'prefix': full[:3], 'velocity': full[3:],
            'vision': ('vision.encode',),
            'prefix_only': ('vision.project', 'prefix.prefill'),
            'denoise_all': ('denoise.velocity',),
            'host_input': ('host.input',), 'host_output': ('host.output',)}
    assert PROFILE_KEYS['hyvla'] == {'vision': ('vision.encode', 'vision.project'),
                                     'prefix': ('prefix.prefill',), 'velocity': ('denoise.velocity',),
                                     'host_input': ('host.input',), 'host_output': ('host.output',)}
    assert PROFILE_KEYS['fastwam'] == {
        'condition_vae_video_prefill': ('host.input', 'vision.encode', 'prefix.prefill'),
        'velocity': ('denoise.velocity',), 'vision': ('vision.encode',),
        'prefix': ('prefix.prefill',), 'condition_tokens': ('prefix.prefill',),
        'condition_context': ('prefix.prefill',), 'condition_mask': ('prefix.prefill',),
        'condition_text_kv': ('prefix.prefill',),
        'host_input': ('host.input',), 'host_output': ('host.output',)}


def test_fastwam_condition_interval_covers_host_input_and_notes_say_so():
    # _prepare_condition starts with _observation_tensors (validation, H2D, T5 encode_prompt).
    host = next(item for item in stages_for('fastwam') if item.stage_id == 'host.input')
    assert set(host.profiling_keys) == {'condition_vae_video_prefill', 'host_input'}
    text = ' '.join(host.notes)
    assert 'encode_prompt' in text and '_initial_state' in text and 'unattributed_ms' in text
    assert 'Not instrumented' not in text
    result = attribute(_record('fastwam', {'condition_vae_video_prefill': [30.0], 'velocity': [2.0] * 4},
                               wall_ms=45.0, nfe=4))
    assert result['groups'][0]['stages'] == ['host.input', 'vision.encode', 'prefix.prefill']
    assert 'host.input' not in result['residual_stages']
    assert result['coverage'] == 'full' and result['unattributed_ms'] == pytest.approx(7.0)
    assert rank(result)[0]['stages'] == ['host.input', 'vision.encode', 'prefix.prefill']


def test_fastwam_host_launch_note_allows_torch_graphs():
    launch = next(item for item in stages_for('fastwam') if item.stage_id == 'host.launch')
    text = ' '.join(launch.notes)
    assert 'backend=torch' in text and 'native_linear_backend=cached' in text
    assert 'requires backend=flashrt' not in text


def test_fastwam_notes_name_text_crossattention_opportunity():
    velocity = next(item for item in stages_for('fastwam') if item.stage_id == 'denoise.velocity')
    text = ' '.join(velocity.notes)
    assert 'cross-attention' in text and 'every' in text
    assert 'text_embedding' in text and 'first-frame video KV is cached once' in text
    assert 'cache.fastwam_text_crossattn_kv' in text
    assert 'cache.fastwam_text_crossattn_kv' in stage('denoise.velocity').default_tools


def test_unknown_lookups_raise():
    with pytest.raises(ValueError, match='Unknown stage'):
        stage('nope')
    with pytest.raises(ValueError, match='Unknown model'):
        stages_for('groot')


def test_attribute_pi05_probeflow_full_coverage():
    record = _record('pi05', {'prefix': [20.0], 'velocity': [3.0, 3.0, 3.0, 3.0]},
                     wall_ms=40.0, nfe=4, sampler='probeflow')
    result = attribute(record)
    assert result['coverage'] == 'full'
    assert result['wall_ms'] == 40.0 and result['nfe'] == 4
    assert result['groups'] == [
        {'stages': ['vision.encode', 'vision.project', 'prefix.prefill'], 'interval_key': 'prefix',
         'ms': 20.0, 'calls': 1},
        {'stages': ['denoise.velocity'], 'interval_key': 'velocity', 'ms': 12.0, 'calls': 4}]
    assert result['per_call_ms'] == {'denoise.velocity': 3.0}
    assert result['unattributed_ms'] == pytest.approx(8.0)
    assert 'host.input' in result['residual_stages'] and 'host.output' in result['residual_stages']
    assert not any('!=' in note for note in result['notes'])
    ranked = rank(result)
    assert [item['stages'] for item in ranked] == [
        ['vision.encode', 'vision.project', 'prefix.prefill'], ['denoise.velocity'], ['unattributed']]
    assert ranked[0]['share'] == pytest.approx(0.5)
    assert sum(item['share'] for item in ranked) == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


def test_attribute_pi0_euler_full_interval_is_not_split():
    result = attribute(_record('pi0', {'full': [30.0]}, wall_ms=33.0, nfe=10))
    assert result['coverage'] == 'full'
    assert result['per_call_ms'] == {}
    assert result['groups'][0]['stages'] == [
        'vision.encode', 'vision.project', 'prefix.prefill', 'denoise.velocity']
    assert any('not separable' in note for note in result['notes'])
    assert result['unattributed_ms'] == pytest.approx(3.0)


def test_attribute_hyvla_is_partial_and_residual_includes_vit():
    # A record written before the ViT+merger graph was timed: still accepted.
    result = attribute(_record('hyvla', {'prefix': [10.0], 'velocity': [2.0] * 5},
                               wall_ms=30.0, nfe=5))
    assert result['coverage'] == 'partial'
    assert result['per_call_ms'] == {'denoise.velocity': 2.0}
    assert result['unattributed_ms'] == pytest.approx(10.0)
    assert {'vision.encode', 'vision.project'} <= set(result['residual_stages'])
    assert any("missing for hyvla/euler: ['vision']" in note for note in result['notes'])


def test_attribute_hyvla_with_vision_interval_is_full():
    result = attribute(_record('hyvla', {'vision': [6.0], 'prefix': [10.0], 'velocity': [2.0] * 5},
                               wall_ms=30.0, nfe=5))
    assert result['coverage'] == 'full'
    assert result['groups'][0] == {'stages': ['vision.encode', 'vision.project'],
                                   'interval_key': 'vision', 'ms': 6.0, 'calls': 1}
    assert result['unattributed_ms'] == pytest.approx(4.0)
    assert not {'vision.encode', 'vision.project'} & set(result['residual_stages'])


def test_attribute_fastwam_velocity_mismatch_and_missing_condition_noted():
    result = attribute(_record('fastwam', {'velocity': [4.0] * 9}, wall_ms=50.0, nfe=10))
    assert result['coverage'] == 'partial'
    assert result['per_call_ms']['denoise.velocity'] == pytest.approx(4.0)
    assert any('velocity interval count 9 != sampler nfe 10' in note for note in result['notes'])
    assert any("missing for fastwam/euler: ['condition_vae_video_prefill']" in note
               for note in result['notes'])


def test_attribute_reports_negative_residual_unclamped():
    result = attribute(_record('fastwam', {'condition_vae_video_prefill': [30.0],
                                           'velocity': [10.0, 10.0]}, wall_ms=45.0, nfe=2))
    assert result['unattributed_ms'] == pytest.approx(-5.0)
    assert result['coverage'] == 'full'
    assert any('exceeds wall' in note and 'unclamped' in note for note in result['notes'])
    ranked = rank(result)
    assert ranked[-1] == {'stages': ['unattributed'], 'ms': pytest.approx(-5.0),
                          'share': pytest.approx(-5.0 / 45.0)}


def test_attribute_coverage_none_cases():
    measured = _record('pi05', {'full': [1.0]}, wall_ms=2.0, nfe=10)
    missing = dict(measured, stage_metrics=None)
    assert attribute(missing)['coverage'] == 'none'
    assert attribute(missing)['unattributed_ms'] is None
    failed = dict(measured, profiling_failure='RuntimeError: boom')
    result = attribute(failed)
    assert result['coverage'] == 'none' and 'boom' in result['notes'][0]
    empty = attribute(_record('pi05', {}, wall_ms=5.0, nfe=10))
    assert empty['coverage'] == 'none' and empty['groups'] == []
    assert rank(attribute(missing)) == []
    no_record_key = {k: v for k, v in measured.items() if k != 'stage_metrics'}
    assert attribute(no_record_key)['coverage'] == 'none'


def test_attribute_without_nfe_notes_unknown():
    result = attribute(_record('pi05', {'prefix': [5.0], 'velocity': [1.0]}, wall_ms=9.0,
                               sampler='probeflow'))
    assert result['nfe'] is None
    assert any('nfe unknown' in note for note in result['notes'])


@pytest.mark.parametrize('mutate, message', [
    (lambda r: (r['stage_metrics']['interval_ms'].update(bogus=[1.0]),
                r['stage_metrics']['interval_sum_ms'].update(bogus=1.0)), 'no stage mapping'),
    (lambda r: r['stage_metrics']['interval_ms'].update(bogus=[1.0]), 'keys must equal'),
    (lambda r: r['stage_metrics'].update(wall_ms=0.0), 'wall_ms'),
    (lambda r: r['stage_metrics'].update(wall_ms=float('nan')), 'wall_ms'),
    (lambda r: r['stage_metrics']['interval_ms'].update(velocity=[-1.0]), 'nonnegative'),
    (lambda r: r['stage_metrics']['interval_ms'].update(velocity=[]), 'nonempty'),
    (lambda r: r['stage_metrics']['interval_sum_ms'].update(velocity=99.0), 'differs'),
    (lambda r: r['stage_metrics'].update(measurement_kind='latency'), 'measurement_kind'),
    (lambda r: r['stage_metrics']['sampler_metadata'].update(nfe=0), 'nfe'),
    (lambda r: r['candidate']['backend'].update(model='groot'), 'No stage map'),
])
def test_attribute_rejects_invalid_records(mutate, message):
    record = _record('pi05', {'prefix': [5.0], 'velocity': [1.0, 1.0]}, wall_ms=9.0, nfe=2,
                     sampler='probeflow')
    mutate(record)
    with pytest.raises(ValueError, match=message):
        attribute(record)


def test_attribute_model_override_must_match():
    record = _record('pi05', {'full': [1.0]}, wall_ms=2.0, nfe=10)
    with pytest.raises(ValueError, match='does not match'):
        attribute(record, model='pi0')
    bare = {'stage_metrics': record['stage_metrics']}
    with pytest.raises(ValueError, match='Model unknown'):
        attribute(bare)
    assert attribute(bare, model='pi05')['coverage'] == 'full'


def test_amdahl_ceiling():
    assert amdahl_ceiling(0.5, 2.0) == pytest.approx(1 / 0.75)
    assert amdahl_ceiling(0.0, 10.0) == 1.0
    assert amdahl_ceiling(1.0, 4.0) == pytest.approx(4.0)
    assert amdahl_ceiling(0.3, 1e12) == pytest.approx(1 / 0.7)
    assert amdahl_ceiling(0.4, 0.5) == pytest.approx(1 / 1.4)
    for share, speedup in ((-0.1, 2.0), (1.1, 2.0), (0.5, 0.0), (0.5, -1.0),
                           (math.nan, 2.0), (0.5, math.inf), (True, 2.0), ('0.5', 2.0)):
        with pytest.raises(ValueError):
            amdahl_ceiling(share, speedup)


def test_rank_rejects_non_attribution():
    with pytest.raises(ValueError):
        rank({'wall_ms': 1.0})


def test_cli_list_show_attribute(tmp_path, capsys):
    assert stages.main(['list']) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item['id'] for item in listed['stages']] == list(stages.STAGE_IDS)
    assert stages.main(['list', '--model', 'hyvla']) == 0
    hy = json.loads(capsys.readouterr().out)
    assert hy['model'] == 'hyvla' and hy['stages'][0]['stage_id'] == 'host.input'
    assert hy['profile_keys']['prefix'] == ['prefix.prefill']
    assert stages.main(['show', 'denoise.velocity']) == 0
    shown = json.loads(capsys.readouterr().out)
    assert 'VLA expert 很小却不快' in shown['playbook_rows']
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(_record('pi05', {'prefix': [5.0], 'velocity': [1.0, 1.0]},
                                       wall_ms=9.0, nfe=2, sampler='probeflow')))
    assert stages.main(['attribute', str(path), '--rank']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['attribution']['per_call_ms'] == {'denoise.velocity': 1.0}
    assert output['rank'][0]['stages'][-1] == 'prefix.prefill'
    assert stages.main(['check-seams']) == 0
    assert json.loads(capsys.readouterr().out)['missing'] == []


def test_cli_errors_return_nonzero(tmp_path, capsys):
    assert stages.main(['list', '--model', 'groot']) == 2
    assert 'Unknown model' in capsys.readouterr().err
    assert stages.main(['show', 'nope']) == 2
    assert 'Unknown stage' in capsys.readouterr().err
    assert stages.main(['attribute', str(tmp_path / 'missing.json')]) == 2
    assert 'Cannot read result JSON' in capsys.readouterr().err
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    assert stages.main(['attribute', str(bad)]) == 2
    invalid = tmp_path / 'invalid.json'
    record = _record('pi05', {'nonsense': [1.0]}, wall_ms=2.0, nfe=1)
    invalid.write_text(json.dumps(record))
    assert stages.main(['attribute', str(invalid)]) == 2
    assert 'no stage mapping' in capsys.readouterr().err
