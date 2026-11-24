"""Onboarding tests use synthetic CPU evidence, never a real-hardware claim."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


@pytest.mark.parametrize('head', ['flow_matching', 'autoregressive_tokens', 'other'])
def test_init_builds_shared_spec_and_reproducible_cpu_fixtures(tmp_path, head):
    from flash_rt.vla.onboard import init_onboarding, generate_fixtures
    from flash_rt.vla.model_spec import get_spec
    from flash_rt.vla.autotune.workloads import NPZWorkload

    root = tmp_path / head
    init_onboarding(root, model='onboard_toy', head_type=head)
    spec = get_spec('onboard_toy', spec_dir=root / 'specs')
    assert spec['head']['type'] == head
    assert spec['identity']['source']['status'] == 'unknown'
    assert spec['identity']['source']['value'] is None
    first = NPZWorkload.from_file(root / 'fixtures/workload.json')
    other = generate_fixtures(spec, tmp_path / 'repeated-fixtures', spec_dir=root / 'specs')
    second = NPZWorkload.from_file(other)
    assert len(first) >= 4
    for index in range(len(first)):
        left, right = first.request(index), second.request(index)
        assert left.request_id == right.request_id
        assert left.prompt == right.prompt
        np.testing.assert_array_equal(left.noise, right.noise)
        np.testing.assert_array_equal(left.observation['state'], right.observation['state'])
        assert left.observation['images'][0].dtype == np.uint8
    assert not np.array_equal(first.request(0).observation['images'][0],
                              first.request(1).observation['images'][0])
    assert read_json(root / 'onboarding.json')['fixture_kind'] == 'synthetic'
    if head == 'other':
        assert spec['head']['kind'] == 'regression'
        assert spec['head']['defaults']['sampler'] == 'none'
        assert np.count_nonzero(first.request(0).noise) == 0
    package = root / 'onboard_toy_onboarding'
    assert (package / 'author_adapter.py').is_file()
    assert (package / 'project_adapter.py').is_file()
    assert (package / 'project_backend.py').is_file()
    assert (root / 'tests/test_onboard_toy_contract.py').is_file()
    assert spec['adapter']['hooks']['binding'] == 'onboard_toy_onboarding.project_adapter:binding'
    if head == 'autoregressive_tokens':
        assert spec['head']['defaults']['sampler'] == 'autoregressive'
        assert spec['capabilities']['samplers'] == ['autoregressive']
        assert spec['head']['max_tokens'] > 0
        assert spec['head']['vocab_size'] > 0
        assert spec['fixture']['noise_role'] == 'unused'
        assert np.count_nonzero(first.request(0).noise) == 0
        assert spec['inputs']['noise']['shape'] == [1]


def test_autoregressive_template_has_no_flow_tools_and_scaffold_tests_run_cpu_only(tmp_path):
    from flash_rt.vla.onboard import init_onboarding
    from flash_rt.vla.autotune.tools import list_tools

    root = tmp_path / 'decode'
    init_onboarding(root, model='onboard_decode', head_type='autoregressive_tokens')
    tools = {tool.tool_id for tool in list_tools(
        model='onboard_decode', spec_dir=root / 'specs', include_unavailable=True)}
    assert not {'sampler.euler_steps', 'sampler.probeflow'} & tools
    process = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(root / 'tests')],
        text=True, capture_output=True)
    assert process.returncode == 0, process.stdout + process.stderr


def test_scaffold_project_adapter_fails_closed_until_implemented(tmp_path, monkeypatch):
    import importlib
    from flash_rt.vla.onboard import init_onboarding
    from flash_rt.vla import BackendConfig

    root = tmp_path / 'scaffold'
    init_onboarding(root, model='onboard_skeleton', head_type='autoregressive_tokens')
    monkeypatch.syspath_prepend(str(root))
    torch_loaded = 'torch' in sys.modules
    adapter = importlib.import_module('onboard_skeleton_onboarding.project_adapter')
    config = BackendConfig(model='onboard_skeleton', checkpoint='/unused',
                           backend='torch', hardware='cuda_sm80', spec_dir=str(root / 'specs'))
    with pytest.raises(NotImplementedError):
        adapter.binding(config, 'cuda_sm80')
    assert ('torch' in sys.modules) == torch_loaded


def test_check_fails_closed_for_template_without_importing_runtime(tmp_path):
    from flash_rt.vla.onboard import init_onboarding, check_onboarding

    root = tmp_path / 'new'
    init_onboarding(root, model='onboard_toy', head_type='flow_matching')
    report = check_onboarding(root)
    assert report['status'] == 'incomplete'
    assert any('identity.source' in issue for issue in report['issues'])
    assert any('author' in issue for issue in report['issues'])
    assert not (root / 'verification.json').exists()
    code = (
        'import sys; from flash_rt.vla.onboard import check_onboarding; '
        'check_onboarding(sys.argv[1]); '
        'assert "torch" not in sys.modules; '
        'assert not any(n.startswith("flash_rt.frontends.") for n in sys.modules)'
    )
    subprocess.run([sys.executable, '-c', code, str(root)], check=True)


def test_init_preserves_existing_work_and_bad_head_creates_nothing(tmp_path):
    from flash_rt.vla.onboard import init_onboarding

    root = tmp_path / 'existing'
    root.mkdir()
    (root / 'keep.txt').write_text('user work')
    with pytest.raises((ValueError, FileExistsError)):
        init_onboarding(root, model='onboard_toy', head_type='flow_matching')
    assert (root / 'keep.txt').read_text() == 'user work'
    bad = tmp_path / 'bad'
    with pytest.raises(ValueError):
        init_onboarding(bad, model='onboard_toy', head_type='invalid-head')
    assert not bad.exists()


def test_fixture_generation_rejects_unknown_shapes_and_unsafe_dtypes(tmp_path):
    from flash_rt.vla.onboard import init_onboarding, generate_fixtures
    from flash_rt.vla.model_spec import get_spec

    init_onboarding(tmp_path / 'new', model='onboard_toy', head_type='other')
    spec = get_spec('onboard_toy', spec_dir=tmp_path / 'new/specs')
    spec['inputs']['state']['shape'] = ['unresolved']
    with pytest.raises(ValueError, match='unresolved'):
        generate_fixtures(spec, tmp_path / 'bad-shape')
    assert not (tmp_path / 'bad-shape').exists()
    spec['inputs']['state']['shape'] = [8]
    spec['inputs']['state']['dtype'] = 'object'
    with pytest.raises(ValueError, match='dtype'):
        generate_fixtures(spec, tmp_path / 'bad-dtype')
    assert not (tmp_path / 'bad-dtype').exists()


def test_cli_check_reports_actionable_incomplete_status(tmp_path):
    from flash_rt.vla.onboard import main

    root = tmp_path / 'cli'
    assert main(['init', '--root', str(root), '--model', 'onboard_toy',
                 '--head', 'flow_matching']) == 0
    assert main(['check', '--root', str(root)]) == 1


@pytest.fixture
def evidence(tmp_path, monkeypatch, request):
    """Full portable author/campaign fixture with synthetic device metadata.

    No model is executed. Integrity and provenance checks are production code;
    every purported measurement below is test data in this temporary directory.
    """
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.onboard import init_onboarding
    from flash_rt.vla.autotune import author
    from flash_rt.vla.autotune.controller import (
        initialize_campaign, mark_submitting, mark_queued, ingest_campaign)
    from flash_rt.vla.autotune.quality import compare_outputs
    from flash_rt.vla.autotune.schema import Candidate
    from flash_rt.vla.autotune.workloads import NPZWorkload
    from flash_rt.vla.autotune.token_usage import SegmentLog

    parameters = getattr(request, 'param', {})
    head_type = parameters.get('head', 'flow_matching')
    channels = parameters.get('channels', [0, 1])
    width = max(channels) + 1
    root = tmp_path / 'onboarding'
    init_onboarding(root, model='onboard_toy', head_type=head_type)
    model_path = root / 'specs/onboard_toy.json'
    spec = read_json(model_path)
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'model.bin').write_bytes(b'CPU test fixture; not a checkpoint')
    checkpoint = author.content_identity(weights)
    source = tmp_path / 'original-author'
    package = source / 'author_pkg'
    package.mkdir(parents=True)
    model_file = package / 'model.py'
    model_file.write_text('# CPU test source fixture\n')
    pin = 'a' * 40
    url = 'https://example.invalid/original-author'
    for key, value in [('source', pin), ('weights', 'b' * 40), ('license', 'Apache-2.0')]:
        spec['identity'][key] = {'status': 'pinned', 'value': value,
                                 'sources': [url]}
    spec['identity']['source'].update(repository=url, revision=pin)
    spec['identity']['weights']['sha256'] = checkpoint['sha256']
    spec['inputs']['image_0']['shape'] = [4, 4, 3]
    spec['inputs']['image_1']['shape'] = [4, 4, 3]
    if head_type == 'flow_matching':
        spec['inputs']['noise']['shape'] = [2, width]
    spec['actions'].update(model_dim=width, effective_dim=2, channels=channels, horizon=2)
    spec['fixture']['dimensions'].update(height=4, width=4, horizon=2, model_dim=width)
    spec['quality']['action_dim'] = 2
    spec['adapter']['hooks']['binding'] = 'unimported_runtime_adapter:binding'
    spec['author'].update(adapter='onboard_test_author', source_package='author_pkg')
    module_root = tmp_path / 'modules'
    module_root.mkdir()
    (module_root / 'onboard_test_author.py').write_text('# CPU-only author adapter fixture\n')
    monkeypatch.syspath_prepend(str(module_root))
    write_json(model_path, spec)
    from flash_rt.vla.onboard import generate_fixtures
    workload_path = generate_fixtures(spec, root / 'measured-fixtures', spec_dir=root / 'specs')
    workload = NPZWorkload.from_file(workload_path)
    fixture_identity = author.workload_identity(workload)
    config = read_json(root / 'onboarding.json')
    config.update(workload='measured-fixtures/workload.json',
                  template_assumptions_confirmed=True, quality_budget_confirmed=True)
    implementation = tmp_path / 'implemented_backend.py'
    implementation.write_text('ONBOARDING_IMPLEMENTED = True\n')
    config['implementation_files'] = dict.fromkeys(
        ('project_adapter', 'project_backend', 'author_adapter'), str(implementation))
    write_json(root / 'onboarding.json', config)
    author_config = {'mode': 'produce', 'source': str(source), 'source_revision': pin,
                     'num_steps': 10 if head_type == 'flow_matching' else 1, 'options': {}}
    write_json(root / 'author_baseline.json', author_config)
    gate = {'scope': 'numerical_replay', 'action_dim': 2,
            **({'action_channels': channels} if channels != [0, 1] else {}),
            'max_relative_l2': 0.05, 'min_cosine': 0.999}
    backend = dict(model='onboard_toy', checkpoint=str(weights), backend='torch',
                   hardware='cuda_sm80', num_steps=10 if head_type == 'flow_matching' else None,
                   spec_dir=str(root / 'specs'))
    baseline = Candidate('graph', BackendConfig(**backend, options={'use_cuda_graph': True}),
                         workload.workload_id)
    eager = Candidate('eager', BackendConfig(**backend, options={'use_cuda_graph': False}),
                      workload.workload_id)
    campaign, results = tmp_path / 'campaign', tmp_path / 'results'
    repeats = parameters.get('repeats', 8)
    plan = initialize_campaign(campaign, comparison_id='onboarding-test', baseline=baseline,
                               candidates=[eager], workload_path=str(workload_path),
                               output_root=str(results), warmup=1, repeats=repeats,
                               quality_gate=gate, author_baseline=author_config)
    job_id = '12345'
    gpu = {'name': 'NVIDIA A800 80GB PCIe', 'uuid': 'GPU-onboarding-CPU-test',
           'capability': [8, 0], 'job_id': job_id, 'host': 'compute-test'}
    ids = [workload.request(i).request_id for i in range(repeats)]
    actions = [np.arange(1, 1 + 2 * width, dtype=np.float32).reshape(2, width) + i % 4
               for i in range(repeats)]
    sampler = spec['head']['defaults']['sampler']
    nfe = 10 if head_type == 'flow_matching' else 5 if head_type == 'autoregressive_tokens' else 1
    observed = ({'prefill_forward_calls': 1, 'decoder_forward_calls': 4,
                 'generated_tokens': 4, 'nfe_provenance': 'observed_forward_calls'}
                if head_type == 'autoregressive_tokens' else {})
    author_dir = results / 'author'
    author_dir.mkdir(parents=True)
    author_outputs = author_dir / 'outputs.npz'
    np.savez_compressed(author_outputs, actions=np.stack(actions), request_ids=np.asarray(ids),
                        latency_ms=np.asarray([10.0] * repeats),
                        output_domain=np.asarray(spec['actions']['worker_domain']))
    author_record = {
        'status': 'complete', 'origin': 'author', 'model': 'onboard_toy',
        'producer_spec': plan['author_baseline']['spec'],
        'producer_command': plan['author_baseline']['producer_command'],
        'identity': {'model': 'onboard_toy', 'workload': fixture_identity, 'checkpoint': checkpoint,
                     'author_source': {'path': str(source), 'revision': pin, 'source_url': url,
                                       'python_files': {'author_pkg/model.py': author.file_sha256(model_file)}},
                     'author_model': {'file': str(model_file), 'sha256': author.file_sha256(model_file)},
                     'output_domain': spec['actions']['worker_domain']},
        'metadata': {'expected_shape': [2, width], **observed},
        'precision': 'bf16', 'sampler': sampler,
        'nfe': nfe, 'boundary': author.BOUNDARY, 'repeats': repeats, 'latency_ms': [10.0] * repeats,
        'request_ids': ids, 'outputs_path': str(author_outputs),
        'outputs_sha256': author.file_sha256(author_outputs), 'implementation': 'test original author',
        'job_id': job_id, 'gpu': gpu,
        'comparison_id': plan['comparison_id'], 'campaign_instance_id': plan['campaign_instance_id']}
    author_path = author_dir / 'result.json'
    write_json(author_path, author_record)
    descriptor = {
        'schema_version': 1, 'origin': 'author', 'result_path': str(author_path),
        'result_sha256': author.file_sha256(author_path),
        'outputs_sha256': author.file_sha256(author_outputs), 'planned': plan['author_baseline'],
        'provenance': 'same_campaign', 'producer_command': author_record['producer_command']}
    control = results / 'control'
    control.mkdir()
    descriptor_path = control / 'author_reference.json'
    write_json(descriptor_path, descriptor)
    author.export_portable_reference(plan, descriptor_path, batch_job_id=job_id)
    reference = {'descriptor': descriptor, 'record': author_record, 'actions': actions, 'request_ids': ids}
    mark_submitting(campaign)
    mark_queued(campaign, job_id)
    records = []
    for entry in plan['entries']:
        directory = results / entry['attempt_id']
        directory.mkdir()
        outputs = directory / 'outputs.npz'
        np.savez_compressed(outputs, request_ids=np.asarray(ids),
                            output_domain=np.asarray(spec['actions']['worker_domain']),
                            workload_digest=np.asarray(fixture_identity['sha256']),
                            **{f'actions_{i:05d}': action for i, action in enumerate(actions)})
        graph = entry['candidate']['backend']['options']['use_cuda_graph']
        record = {
            'schema_version': 2, 'candidate_id': entry['candidate']['candidate_id'],
            'candidate': entry['candidate'], 'campaign_instance_id': plan['campaign_instance_id'],
            'job_id': job_id, 'attempt_id': entry['attempt_id'],
            'comparison_id': plan['comparison_id'], 'workload_id': workload.workload_id,
            'state': 'measured', 'gpu': gpu, 'runtime': {'python': '3.11', 'torch': '2.5.1', 'cuda': '12.4'},
            'measurement': {'warmup': 1, 'repeats': repeats, 'latency_ms': [10.0] * repeats,
                            'load_ms': 1.0, 'prepare_ms': 1.0,
                            'sampler_metadata': [{'sampler': sampler, 'nfe': nfe,
                                                  **({'generated_tokens': 4} if observed else {})}] * repeats,
                            'boundary': author.BOUNDARY},
            'backend_metadata': {'precision': 'bf16', 'action_domain': spec['actions']['domain']},
            'outputs_path': str(outputs), 'output_domain': spec['actions']['worker_domain'],
            'workload_identity': fixture_identity, 'workload_request_ids': sorted(set(ids)),
            'quality': compare_outputs(actions, ids, actions, ids, gate,
                                       domain=spec['actions']['worker_domain']),
            'author_quality': author.compare_reference(actions, ids, reference, gate),
            'execution_metadata': [{**observed, 'use_cuda_graph': graph, 'graph_replays': 1 if graph else 0,
                                    'graph_capture_id': 'test-stable-capture' if graph else None}] * repeats,
            'stage_metrics': {'measurement_kind': 'diagnostic_cuda_event_intervals',
                              'wall_ms': 10.0, 'interval_ms': {'vision': [2.0], 'prefix': [3.0], 'action': [4.5]},
                              'interval_sum_ms': {'vision': 2.0, 'prefix': 3.0, 'action': 4.5},
                              'interval_provenance': {
                                  key: {'stages': covered, 'additive': True}
                                  for key, covered in spec['profiling']['keys'].items()}}}
        if entry['role'] == 'reference':
            record['quality']['reference_role'] = True
        path = directory / 'result.json'
        write_json(path, record)
        author.record_portable_attempt(plan, entry, record, path)
        records.append(path)
    ingest_campaign(campaign, results)
    segments = tmp_path / 'segments.jsonl'
    SegmentLog(segments).start('test-session', model='onboard_toy', phase='onboarding',
                               item='D', summary='CPU test evidence')
    return dict(root=root, campaign=campaign, results=results, graph_result=records[0],
                eager_result=records[1], decisions=tmp_path / 'decisions.jsonl',
                segments=segments, session_id='test-session', records=records, plan=plan)


def verify_args(evidence):
    return {key: value for key, value in evidence.items() if key not in ('root', 'records', 'plan')}


def rebind_test_record(evidence, path, record):
    """Simulate a different legitimate worker envelope, never alter real data."""
    import sqlite3
    from flash_rt.vla.autotune.author import record_portable_attempt
    write_json(path, record)
    entry = next(entry for entry in evidence['plan']['entries']
                 if entry['attempt_id'] == record['attempt_id'])
    record_portable_attempt(evidence['plan'], entry, record, path)
    with sqlite3.connect(evidence['campaign'] / 'ledger.sqlite') as db:
        db.execute('UPDATE attempts SET result_json=? WHERE attempt_id=?',
                   (json.dumps(record), record['attempt_id']))


def test_verify_requires_authoritative_gates_then_records_note_and_closes_segment(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    from flash_rt.vla.autotune.decisions import DecisionLog
    from flash_rt.vla.autotune.token_usage import SegmentLog, segment_intervals

    ledger_before = (evidence['campaign'] / 'ledger.sqlite').read_bytes()
    report = verify_onboarding(evidence['root'], **verify_args(evidence))
    assert report['status'] == 'verified'
    assert report['profile']['coverage_ratio'] == pytest.approx(0.95)
    assert report['graph_eager_quality']['status'] == 'passed'
    assert report['internal_repeats_per_case'] == 2
    notes = DecisionLog(evidence['decisions']).records()
    assert len(notes) == 1 and notes[0]['kind'] == 'note' and notes[0]['actor'] == 'agent'
    assert notes[0]['model'] == 'onboard_toy'
    spans = segment_intervals(SegmentLog(evidence['segments']).records())
    assert len(spans) == 1 and spans[0]['closed']
    assert read_json(evidence['root'] / 'verification.json')['status'] == 'verified'
    assert (evidence['campaign'] / 'ledger.sqlite').read_bytes() == ledger_before


def test_verify_cli_publishes_receipt_after_authoritative_flow_evidence(evidence, capsys):
    from flash_rt.vla.onboard import main

    argv = ['verify']
    for name, value in verify_args(evidence).items():
        argv.extend(['--' + name.replace('_', '-'), str(value)])
    argv.extend(['--root', str(evidence['root']), '--actor', 'agent'])
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'verified'
    assert read_json(evidence['root'] / 'verification.json')['model'] == 'onboard_toy'


@pytest.mark.parametrize('evidence', [{'head': 'autoregressive_tokens'}], indirect=True)
def test_autoregressive_full_campaign_verifies_actual_decoder_nfe(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    report = verify_onboarding(evidence['root'], **verify_args(evidence))
    assert report['status'] == 'verified'
    assert report['head']['type'] == 'autoregressive_tokens'
    graph = read_json(evidence['graph_result'])
    assert graph['candidate']['backend']['num_steps'] is None
    assert graph['measurement']['sampler_metadata'][0]['nfe'] == 5


@pytest.mark.parametrize('evidence', [{'channels': [0, 2]}], indirect=True)
def test_nonprefix_public_channels_use_explicit_shared_quality_selection(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    report = verify_onboarding(evidence['root'], **verify_args(evidence))
    assert report['status'] == 'verified'
    assert report['graph_eager_quality']['action_dim'] == 2
    assert report['graph_eager_quality']['action_channels'] == [0, 2]


@pytest.mark.parametrize('evidence', [{'head': 'other'}], indirect=True)
def test_optional_regression_full_campaign_remains_one_direct_evaluation(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    report = verify_onboarding(evidence['root'], **verify_args(evidence))
    assert report['status'] == 'verified'
    assert report['head']['kind'] == 'regression'
    graph = read_json(evidence['graph_result'])
    assert graph['measurement']['sampler_metadata'][0]['sampler'] == 'none'
    assert graph['measurement']['sampler_metadata'][0]['nfe'] == 1


@pytest.mark.parametrize('evidence', [{'head': 'autoregressive_tokens'}], indirect=True)
def test_autoregressive_authoritative_bundle_with_wrong_decoder_count_is_refused(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    path = evidence['graph_result']
    record = read_json(path)
    record['execution_metadata'][0]['decoder_forward_calls'] = 1
    rebind_test_record(evidence, path, record)
    with pytest.raises(ValueError, match='prefill\\+decoder'):
        verify_onboarding(evidence['root'], **verify_args(evidence))
    assert not evidence['decisions'].exists()


@pytest.mark.parametrize('broken', [
    'no_author', 'untrusted_summary', 'wrong_job', 'wrong_domain', 'low_profile',
    'different_sampler', 'fake_graph', 'nonfinite_actions', 'missing_outputs', 'unconfirmed_budget',
    'wrong_segment_phase', 'bad_profile_sum', 'overlapping_profile', 'missing_runtime',
    'login_host', 'bad_author_gate', 'excessive_drift', 'corrupt_ledger', 'public_channel_drift',
])
def test_verify_fails_without_completion_side_effects(evidence, broken):
    from flash_rt.vla.onboard import verify_onboarding
    from flash_rt.vla.autotune.token_usage import SegmentLog

    if broken == 'no_author':
        (evidence['results'] / 'control/author_outputs.npz').unlink()
    elif broken == 'untrusted_summary':
        (evidence['campaign'] / 'ledger.sqlite').unlink()
        write_json(evidence['campaign'] / 'summary.json', {'complete': True, 'winner': 'graph'})
    elif broken == 'corrupt_ledger':
        (evidence['campaign'] / 'ledger.sqlite').write_bytes(b'not a sqlite database')
    elif broken == 'public_channel_drift':
        path = evidence['root'] / 'specs/onboard_toy.json'
        spec = read_json(path)
        spec['actions']['channels'] = [1, 0]
        write_json(path, spec)
    elif broken in ('wrong_job', 'wrong_domain', 'low_profile', 'different_sampler', 'fake_graph',
                    'bad_profile_sum', 'overlapping_profile', 'missing_runtime', 'login_host',
                    'bad_author_gate', 'excessive_drift'):
        path = evidence['records'][-1] if broken == 'excessive_drift' else evidence['graph_result']
        record = read_json(path)
        if broken == 'wrong_job':
            record['gpu']['job_id'] = '999'
        elif broken == 'wrong_domain':
            record['output_domain'] = 'different-domain'
        elif broken == 'low_profile':
            record['stage_metrics']['wall_ms'] = 20
        elif broken == 'different_sampler':
            record['candidate']['backend']['num_steps'] = 5
        elif broken == 'bad_profile_sum':
            record['stage_metrics']['interval_sum_ms']['vision'] = 5.0
        elif broken == 'overlapping_profile':
            record['stage_metrics']['wall_ms'] = 8.0
        elif broken == 'missing_runtime':
            record['runtime'].pop('torch')
        elif broken == 'login_host':
            record['gpu']['host'] = 'login-node'
        elif broken == 'bad_author_gate':
            record['author_quality']['metrics']['max_relative_l2'] = 0.2
        elif broken == 'excessive_drift':
            record['measurement']['latency_ms'] = [20.0] * record['measurement']['repeats']
        else:
            record['execution_metadata'][0]['graph_replays'] = 0
        if broken in ('low_profile', 'fake_graph', 'bad_profile_sum', 'overlapping_profile',
                      'missing_runtime', 'login_host', 'bad_author_gate', 'excessive_drift'):
            rebind_test_record(evidence, path, record)
        else:
            write_json(path, record)
    elif broken == 'nonfinite_actions':
        np.savez_compressed(evidence['graph_result'].parent / 'outputs.npz',
                            actions_00000=np.asarray([np.nan]))
    elif broken == 'missing_outputs':
        (evidence['graph_result'].parent / 'outputs.npz').unlink()
    elif broken == 'unconfirmed_budget':
        path = evidence['root'] / 'onboarding.json'
        config = read_json(path)
        config['quality_budget_confirmed'] = False
        write_json(path, config)
    else:
        log = SegmentLog(evidence['segments'])
        log.end('test-session')
        log.start('test-session', model='onboard_toy', phase='optimization', item='D',
                  summary='not onboarding')
    before = evidence['segments'].read_bytes()
    with pytest.raises(ValueError):
        verify_onboarding(evidence['root'], **verify_args(evidence))
    assert evidence['segments'].read_bytes() == before
    assert not evidence['decisions'].exists()
    assert not (evidence['root'] / 'verification.json').exists()


def test_check_refuses_generated_unimplemented_files_even_with_confirmed_metadata(evidence):
    from flash_rt.vla.onboard import check_onboarding

    path = evidence['root'] / 'onboarding.json'
    config = read_json(path)
    config['implementation_files']['project_backend'] = 'onboard_toy_onboarding/project_backend.py'
    write_json(path, config)
    report = check_onboarding(evidence['root'])
    assert report['status'] == 'incomplete'
    assert any('project_backend' in issue and 'unimplemented' in issue for issue in report['issues'])


def test_check_rejects_flag_only_implementation_with_unimplemented_infer(evidence):
    from flash_rt.vla.onboard import check_onboarding

    source = evidence['root'] / 'onboard_toy_onboarding/project_backend.py'
    source.write_text(source.read_text().replace('ONBOARDING_IMPLEMENTED = False',
                                               'ONBOARDING_IMPLEMENTED = True'))
    path = evidence['root'] / 'onboarding.json'
    config = read_json(path)
    config['implementation_files']['project_backend'] = str(source)
    write_json(path, config)
    report = check_onboarding(evidence['root'])
    assert report['status'] == 'incomplete'
    assert any('NotImplementedError' in issue for issue in report['issues'])


def test_check_accepts_real_data_manifest_without_synthetic_recipe(evidence):
    from flash_rt.vla.onboard import check_onboarding

    model_path = evidence['root'] / 'specs/onboard_toy.json'
    spec = read_json(model_path)
    spec['fixture']['recipe'] = 'real-dataset-selected-frames'
    write_json(model_path, spec)
    path = evidence['root'] / 'onboarding.json'
    config = read_json(path)
    config['fixture_kind'] = 'dataset'
    config['fixture_sources'] = ['https://example.invalid/fixed-dataset-revision']
    write_json(path, config)
    assert check_onboarding(evidence['root'])['status'] == 'ready_for_measurement'


@pytest.mark.parametrize('nfe,tokens,allowed', [(5, 4, True), (1, 4, False), (0, 4, False), (5, 0, False)])
def test_autoregressive_decoder_nfe_uses_observed_forward_counts_not_flow_steps(tmp_path, nfe, tokens, allowed):
    from flash_rt.vla.onboard import init_onboarding, _sampler_evidence
    from flash_rt.vla.model_spec import get_spec

    root = tmp_path / 'ar'
    init_onboarding(root, model='onboard_decode', head_type='autoregressive_tokens')
    spec = get_spec('onboard_decode', spec_dir=root / 'specs')
    record = {
        'candidate': {'backend': {'sampler': 'autoregressive', 'num_steps': None, 'probeflow': None}},
        'measurement': {'repeats': 2, 'sampler_metadata': [
            {'sampler': 'autoregressive', 'nfe': nfe, 'generated_tokens': tokens}] * 2},
        'execution_metadata': [
            {'decoder_forward_calls': tokens, 'prefill_forward_calls': 1,
             'generated_tokens': tokens, 'nfe_provenance': 'observed_forward_calls'}] * 2}
    if allowed:
        _sampler_evidence(record, spec)
    else:
        with pytest.raises(ValueError):
            _sampler_evidence(record, spec)


@pytest.mark.parametrize('evidence', [{'repeats': 4}], indirect=True)
def test_verify_refuses_single_sample_per_case_even_when_campaign_gates_pass(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    with pytest.raises(ValueError, match='two internal output samples'):
        verify_onboarding(evidence['root'], **verify_args(evidence))
    assert not evidence['decisions'].exists()


def measured_explanation(evidence):
    from flash_rt.vla.autotune.author import file_sha256
    record = read_json(evidence['graph_result'])
    record['stage_metrics']['wall_ms'] = 20.0
    rebind_test_record(evidence, evidence['graph_result'], record)
    artifact = evidence['root'] / 'residual-samples.json'
    write_json(artifact, {'host_launch_ms': [10.4, 10.6], 'clock': 'host_wall',
                          'source': 'Synthetic CPU test evidence only.'})
    path = evidence['root'] / 'residual-explanation.json'
    write_json(path, {
        'schema_version': 1, 'kind': 'onboarding_residual_explanation',
        'binding': {'campaign_instance_id': record['campaign_instance_id'],
                    'attempt_id': record['attempt_id'], 'job_id': record['job_id'],
                    'gpu_uuid': record['gpu']['uuid'],
                    'workload_digest': record['workload_identity']['sha256'],
                    'result_sha256': file_sha256(evidence['graph_result'])},
        'wall_ms': 20.0, 'covered_ms': 9.5, 'unattributed_ms': 10.5,
        'reason': 'Separate repeated host wall intervals measured submission gaps.',
        'limitations': 'Synthetic test only; event intervals do not establish kernel saturation.',
        'observations': [{'stage': 'host.launch', 'clock': 'host_wall',
                          'samples_ms': [10.4, 10.6],
                          'artifact': {'path': artifact.name, 'sha256': file_sha256(artifact)}}]})
    return path


def test_measured_residual_explanation_keeps_actual_low_coverage_visible(evidence):
    from flash_rt.vla.onboard import verify_onboarding
    path = measured_explanation(evidence)
    report = verify_onboarding(evidence['root'], **verify_args(evidence), residual_explanation=path)
    assert report['profile']['status'] == 'measured_residual_explained'
    assert report['profile']['coverage_ratio'] == pytest.approx(0.475)
    assert report['profile']['explained_ms'] == pytest.approx(10.5)


@pytest.mark.parametrize('broken', ['wrong_binding', 'text_only', 'unexplained', 'bad_artifact'])
def test_residual_exception_requires_bound_repeated_quantitative_evidence(evidence, broken):
    from flash_rt.vla.onboard import verify_onboarding
    path = measured_explanation(evidence)
    extra = read_json(path)
    if broken == 'wrong_binding':
        extra['binding']['job_id'] = '98765'
    elif broken == 'text_only':
        extra['observations'] = []
    elif broken == 'unexplained':
        extra['observations'][0]['samples_ms'] = [0.5, 0.6]
    else:
        extra['observations'][0]['artifact']['sha256'] = '0' * 64
    write_json(path, extra)
    with pytest.raises(ValueError):
        verify_onboarding(evidence['root'], **verify_args(evidence), residual_explanation=path)
    assert not evidence['decisions'].exists()
