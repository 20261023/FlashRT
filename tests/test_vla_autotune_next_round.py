"""CPU outer-loop regressions using real controller-ingested worker envelopes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.controller import (
    ingest_campaign, initialize_campaign, mark_queued, mark_submitting)
from flash_rt.vla.autotune.decisions import DecisionLog
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.tools import all_tools


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/vla_autotune/next_round.py'


def _module():
    assert SCRIPT.is_file(), 'the outside-harness next-round planner is missing'
    spec = importlib.util.spec_from_file_location('next_round', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def _campaign(tmp_path, name='r0', *, tool='cache.fastwam_text_crossattn_kv',
              candidate_ms=80.0, bookend_ms=100.0, candidate_quality='passed',
              intervals=None, no_profile=False, baseline_only=False):
    """Real initialize/ingest, not a hand-invented summary schema."""
    backend = BackendConfig(
        model='fastwam', checkpoint='/weights', backend='torch', framework='torch',
        hardware='cuda_sm80', num_steps=10,
        options={'sigma_shift': 5.0, 'action_horizon': 32,
                 'model_config': '/model.json', 'backbone_dir': '/wan'})
    baseline = Candidate('base', backend, 'frames', source_ref=None)
    other = Candidate('variant', backend, 'frames', parent_id='base', origin='search',
                      tool=tool, stage='denoise.velocity', hypothesis='one measured method')
    directory = tmp_path / name
    plan = initialize_campaign(
        directory, comparison_id=name, baseline=baseline,
        candidates=[] if baseline_only else [other], workload_path='/inputs.json',
        output_root=str(directory / 'results'), repeats=2,
        quality_gate={'scope': 'numerical_replay', 'max_relative_l2': 0.05})
    mark_submitting(directory)
    job_id = str(1000 + sum(map(ord, name)))
    mark_queued(directory, job_id)
    for entry in plan['entries']:
        role = entry['role']
        value = candidate_ms if role == 'candidate' else (
            bookend_ms if role == 'bookend' else 100.0)
        status = candidate_quality if role == 'candidate' else 'passed'
        record = {
            'schema_version': 2, 'candidate_id': entry['candidate']['candidate_id'],
            'candidate': entry['candidate'], 'campaign_instance_id': plan['campaign_instance_id'],
            'job_id': job_id, 'attempt_id': entry['attempt_id'], 'comparison_id': name,
            'workload_id': 'frames', 'state': 'measured',
            'measurement': {'warmup': 0, 'repeats': 2, 'latency_ms': [value, value],
                            'load_ms': 1.0, 'prepare_ms': 1.0,
                            'sampler_metadata': [{'sampler': 'euler', 'nfe': 10}] * 2},
            'backend_metadata': {'action_domain': 'normalized'},
            'gpu': {'uuid': 'GPU-one', 'job_id': job_id}, 'outputs_path': '/outputs.npz',
            'quality': {'status': status, 'scope': 'numerical_replay', 'action_dim': None,
                        'gate': {'max_relative_l2': 0.05},
                        'metrics': {'max_relative_l2': 0.1 if status == 'failed' else 0.0}}}
        if not no_profile:
            actual = intervals or {'vision': [10], 'prefix': [14], 'velocity': [7] * 10,
                                   'host_input': [3], 'host_output': [1]}
            record['stage_metrics'] = {
                'measurement_kind': 'diagnostic_cuda_event_intervals',
                'separate_from_latency_samples': True, 'wall_ms': value,
                'interval_ms': {key: [x * value / 100 for x in values]
                                for key, values in actual.items()},
                'expected_interval_keys': list(actual),
                'sampler_metadata': {'sampler': 'euler', 'nfe': 10}}
        _save(directory / 'results' / entry['attempt_id'] / 'result.json', record)
    ingest_campaign(directory, directory / 'results', job_finished=True)
    return directory / 'summary.json'


def _history(tmp_path, summaries=(), *, elapsed=600, actor='agent', **extra):
    rounds = []
    for index, summary in enumerate(summaries):
        data = json.loads(summary.read_text())
        job = {'job_id': data['attempts'][0]['job_id'], 'state': 'COMPLETED',
               'elapsed_seconds': elapsed, 'gpus': 1}
        accounting = _save(tmp_path / f'accounting-{index}.json', job)
        rounds.append({'summary': str(summary), 'actor': actor, 'jobs': [
            {'evidence_ref': str(accounting)}]})
    return _save(tmp_path / 'history.json', {'schema_version': 1, 'rounds': rounds, **extra})


def test_uses_real_attribution_and_feature_table_not_predicted_win(tmp_path):
    summary = _campaign(tmp_path)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['chosen_action']['action'] == 'select_tool'
    assert result['chosen_action']['share'] == pytest.approx(0.70)
    assert result['chosen_action']['stage'] == 'denoise.velocity'
    assert result['measured_round']['winner'] == 'variant'
    assert result['measured_round']['latency_reduction_ratio'] == pytest.approx(0.20)
    assert result['chosen_action']['predicted_latency_ms'] is None
    assert result['features']['flow_matching'] is True
    fp8 = next(row for row in result['tool_table'] if row['tool_id'] == 'quant.fp8')
    assert fp8['status'] == 'blocked'
    assert fp8['reason']
    assert result['profile']['attributed_fraction'] == pytest.approx(0.98)


def test_call_site_timeline_qualifies_attention_without_double_counting(tmp_path):
    summary = _campaign(tmp_path)
    module = _module()
    profile = module._read_round(summary)['incumbent']
    evidence = {
        'schema_version': 1,
        'profile_sha256': hashlib.sha256(json.dumps(
            profile, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest(),
        'regions': [{'tool': 'attention.flex', 'stage': 'denoise.velocity',
                     'site': 'action_video_self',
                     'critical_path_spans_ms': [[4, 8], [7, 10], [20, 24]]}],
    }
    timeline = _save(tmp_path / 'attention.json', evidence)
    exhausted = _save(tmp_path / 'finished.json', {'result': 'other routes evaluated'})
    history = _history(tmp_path, [summary], exhausted_tools=[
        {'tool': tool.tool_id, 'reason': 'already evaluated', 'evidence_ref': str(exhausted)}
        for tool in all_tools() if tool.tool_id != 'attention.flex_attention'])
    chosen = module.recommend(summary, history, operator_profile_path=timeline)['chosen_action']
    assert chosen['tool'] == 'attention.flex_attention'
    assert chosen['call_site'] == 'action_video_self'
    assert chosen['affected_share'] == pytest.approx(10 / 80)

    evidence['regions'][0]['site'] = 'video_self'
    _save(timeline, evidence)
    with pytest.raises(ValueError, match='eligible attention site'):
        module.recommend(summary, history, operator_profile_path=timeline)

    evidence['profile_sha256'] = '0' * 64
    _save(timeline, evidence)
    with pytest.raises(ValueError, match='does not match'):
        module.recommend(summary, history, operator_profile_path=timeline)


@pytest.mark.parametrize('intervals, expected', [
    ({'vision': [10], 'prefix': [10], 'velocity': [6] * 10}, 'profile_coverage_below_90_percent'),
    ({'condition_vae_video_prefill': [96], 'velocity': [0.2] * 10}, 'unsplit_profile_groups'),
    ({'vision': [10], 'prefix': [20], 'velocity': [8] * 10}, 'invalid_profile_accounting'),
])
def test_missing_or_ambiguous_share_acquires_profile_instead_of_guessing(
        tmp_path, intervals, expected):
    summary = _campaign(tmp_path, intervals=intervals)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['chosen_action']['action'] == 'acquire_profile'
    assert result['chosen_action']['reason_code'] == expected
    assert result['chosen_action'].get('tool') is None


def test_no_profile_is_unknown_not_zero_and_baseline_only_does_not_count_as_no_gain(tmp_path):
    summary = _campaign(tmp_path, no_profile=True, baseline_only=True)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['chosen_action']['action'] == 'acquire_profile'
    assert result['profile']['attributed_fraction'] is None
    assert result['measured_round']['latency_reduction_ratio'] is None


def test_sub_five_percent_stage_is_skipped_without_renormalizing_residual(tmp_path):
    summary = _campaign(
        tmp_path, intervals={'vision': [4], 'prefix': [1], 'velocity': [8.7] * 10,
                             'host_input': [3], 'host_output': [1]})
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['chosen_action']['share'] == pytest.approx(.87)
    assert all(row['stage'] != 'vision.encode' for row in result['eligible_directions'])
    assert result['profile']['unattributed_ms'] == pytest.approx(3.2)


def test_rejected_fast_candidate_cannot_supply_gain_or_profile(tmp_path):
    summary = _campaign(tmp_path, candidate_ms=20, candidate_quality='failed')
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['measured_round']['winner'] == 'base'
    assert result['measured_round']['latency_reduction_ratio'] == 0
    assert result['profile']['candidate_id'] == 'base'


def test_high_drift_round_is_not_stagnation_or_an_optimization_win(tmp_path):
    bad = _campaign(tmp_path, 'bad', bookend_ms=115)
    good = _campaign(tmp_path, 'good', candidate_ms=99)
    result = _module().recommend(good, _history(tmp_path, [bad, good]))
    assert result['chosen_action']['action'] != 'stop'
    assert result['round_evidence'][0]['latency_reduction_ratio'] is None
    assert result['round_evidence'][0]['reliability'] == 'unreliable_bookends'


def test_two_comparable_rounds_below_three_percent_stop_model(tmp_path):
    first = _campaign(tmp_path, 'first', candidate_ms=97.5)
    second = _campaign(tmp_path, 'second', candidate_ms=98)
    result = _module().recommend(second, _history(tmp_path, [first, second]))
    assert result['chosen_action']['action'] == 'stop'
    assert result['chosen_action']['reason_code'] == 'two_rounds_below_3_percent'
    assert result['actor_attribution']['agent'] == 2


def test_direction_three_small_rounds_skip_method_without_cherry_picking_model_rounds(tmp_path):
    summaries = []
    for index in range(3):
        summaries.append(_campaign(tmp_path, f'cache{index}', candidate_ms=99))
        summaries.append(_campaign(
            tmp_path, f'compile{index}', tool='compile.inductor', candidate_ms=90))
    result = _module().recommend(summaries[-1], _history(tmp_path, summaries))
    cache = next(row for row in result['tool_table']
                 if row['tool_id'] == 'cache.fastwam_text_crossattn_kv')
    assert cache['exclusion'] == 'three_direction_rounds_below_2_percent'
    assert result['chosen_action']['action'] != 'stop'
    assert result['chosen_action'].get('tool') != 'cache.fastwam_text_crossattn_kv'


def test_gpu_budget_is_actual_elapsed_times_gpu_count_and_never_a_walltime(tmp_path):
    summary = _campaign(tmp_path)
    history = _history(tmp_path, [summary], elapsed=10800)
    data = json.loads(history.read_text())
    path = Path(data['rounds'][0]['jobs'][0]['evidence_ref'])
    accounting = json.loads(path.read_text())
    accounting['gpus'] = 2
    _save(path, accounting)
    result = _module().recommend(summary, history)
    cache = next(row for row in result['tool_table']
                 if row['tool_id'] == 'cache.fastwam_text_crossattn_kv')
    assert cache['gpu_hours'] == 6
    assert cache['exclusion'] == 'method_budget_exhausted'
    assert result['chosen_action'].get('tool') != 'cache.fastwam_text_crossattn_kv'
    assert result['submission_policy']['natural_job_end'] is True
    assert result['submission_policy']['walltime'] is None


@pytest.mark.parametrize('hours', [6.01, 0, -1, True, float('nan')])
def test_cannot_raise_per_method_budget_above_six_or_use_invalid_budget(tmp_path, hours):
    summary = _campaign(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        _module().recommend(summary, _history(tmp_path), method_budget_gpuh=hours)


def test_missing_accounting_requires_evidence_not_zero_hours(tmp_path):
    summary = _campaign(tmp_path)
    history = _history(tmp_path, [summary])
    data = json.loads(history.read_text())
    data['rounds'][0].pop('jobs')
    result = _module().recommend(summary, _save(history, data))
    assert result['chosen_action']['action'] == 'acquire_accounting'
    assert result['chosen_action']['reason_code'] == 'unknown_gpu_cost'


def test_running_job_waits_for_natural_end_not_a_replacement_submission(tmp_path):
    summary = _campaign(tmp_path)
    history = _history(tmp_path, [summary])
    data = json.loads(history.read_text())
    path = Path(data['rounds'][0]['jobs'][0]['evidence_ref'])
    accounting = json.loads(path.read_text())
    accounting['state'] = 'RUNNING'
    _save(path, accounting)
    result = _module().recommend(summary, history)
    assert result['chosen_action']['action'] == 'wait'
    assert result['chosen_action']['reason_code'] == 'job_not_terminal'


def test_duplicate_campaign_cannot_fabricate_a_second_low_gain_round(tmp_path):
    summary = _campaign(tmp_path, candidate_ms=99)
    history = _history(tmp_path, [summary, summary])
    with pytest.raises(ValueError, match='duplicate'):
        _module().recommend(summary, history)


def test_mismatched_diagnostic_profile_is_not_reused_for_current_incumbent(tmp_path):
    summary = _campaign(tmp_path)
    path = next((summary.parent / 'results').glob('*reference/result.json'))
    result = _module().recommend(summary, _history(tmp_path, [summary]), profile_path=path)
    assert result['chosen_action']['action'] == 'acquire_profile'
    assert result['chosen_action']['reason_code'] == 'profile_identity_mismatch'


def test_uncharted_cross_cutting_tool_never_gets_whole_stage_share(tmp_path):
    summary = _campaign(tmp_path)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    kernel = next(row for row in result['tool_table'] if row['tool_id'] == 'kernel.forgekernel')
    assert kernel['share'] is None
    assert kernel['exclusion'] == 'needs_operator_or_launch_attribution'


def test_exhausted_tools_stop_with_evidence_not_a_claim_that_ports_are_installed(tmp_path):
    summary = _campaign(tmp_path)
    evidence = _save(tmp_path / 'exhausted.json', {'review': 'method qualification records'})
    history = _history(tmp_path, [summary], exhausted_tools=[
        {'tool': tool.tool_id, 'reason': 'already evaluated in this evidence scope',
         'evidence_ref': str(evidence)} for tool in all_tools()])
    result = _module().recommend(summary, history)
    assert result['chosen_action']['action'] == 'stop'
    assert result['chosen_action']['reason_code'] == 'applicable_tools_exhausted'


def test_remaining_known_ceiling_below_five_percent_stops_model(tmp_path):
    summary = _campaign(tmp_path, intervals={
        'vision': [60], 'prefix': [30], 'velocity': [.4] * 10,
        'host_input': [3], 'host_output': [1]})
    evidence = _save(tmp_path / 'done.json', {'review': 'method qualification records'})
    history = _history(tmp_path, [summary], exhausted_tools=[
        {'tool': tool.tool_id, 'reason': 'qualified already', 'evidence_ref': str(evidence)}
        for tool in all_tools() if tool.tool_id != 'cache.fastwam_text_crossattn_kv'])
    result = _module().recommend(summary, history)
    assert result['chosen_action']['action'] == 'stop'
    assert result['chosen_action']['reason_code'] == 'remaining_ceilings_below_5_percent'


def test_compile_setup_gpu_time_counts_even_before_a_campaign_exists(tmp_path):
    summary = _campaign(tmp_path)
    accounting = _save(tmp_path / 'cold-compile.json',
                       {'job_id': '9999', 'state': 'FAILED', 'elapsed_seconds': 21600, 'gpus': 1})
    history = _history(tmp_path, [summary], method_jobs=[
        {'tool': 'compile.fastwam_inductor', 'actor': 'agent',
         'jobs': [{'evidence_ref': str(accounting)}]}])
    result = _module().recommend(summary, history)
    compile_row = next(row for row in result['tool_table'] if row['tool_id'] == 'compile.inductor')
    assert compile_row['gpu_hours'] == 6
    assert compile_row['exclusion'] == 'method_budget_exhausted'


def test_unrelated_scheduler_job_does_not_supply_known_budget(tmp_path):
    summary = _campaign(tmp_path)
    history = _history(tmp_path, [summary])
    manifest = json.loads(history.read_text())
    path = Path(manifest['rounds'][0]['jobs'][0]['evidence_ref'])
    job = json.loads(path.read_text())
    job['job_id'] = 'unrelated-job'
    _save(path, job)
    result = _module().recommend(summary, history)
    assert result['chosen_action']['action'] == 'acquire_accounting'


def test_runtime_or_workload_identity_change_resets_stagnation_evidence(tmp_path):
    first = _campaign(tmp_path, 'one', candidate_ms=99)
    second = _campaign(tmp_path, 'two', candidate_ms=99)
    record = json.loads(second.read_text())
    for attempt in record['attempts']:
        attempt['result']['workload_identity'] = {'digest': 'changed-inputs'}
    _save(second, record)
    result = _module().recommend(second, _history(tmp_path, [first, second]))
    assert result['chosen_action']['action'] != 'stop'


def test_profile_cannot_reuse_different_noise_under_same_workload_id(tmp_path):
    summary = _campaign(tmp_path)
    source = next((summary.parent / 'results').glob('*candidate/result.json'))
    record = json.loads(source.read_text())
    record['workload_identity'] = {'digest': 'different-noise'}
    result = _module().recommend(
        summary, _history(tmp_path, [summary]),
        profile_path=_save(tmp_path / 'wrong-noise.json', record))
    assert result['chosen_action']['reason_code'] == 'profile_identity_mismatch'


def test_config_ready_is_not_a_prediction_that_all_velocity_time_is_removable(tmp_path):
    summary = _campaign(tmp_path)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    cache = next(row for row in result['tool_table']
                 if row['tool_id'] == 'cache.fastwam_text_crossattn_kv')
    assert cache['affected_share'] is None
    assert cache['share'] == pytest.approx(.70)
    assert cache['ideal_region_latency_reduction_ceiling'] == pytest.approx(.70)


def test_direction_noise_at_two_percent_cannot_prove_sub_two_percent_stagnation(tmp_path):
    summaries = [_campaign(tmp_path, f'r{i}', candidate_ms=99, bookend_ms=102)
                 for i in range(3)]
    result = _module().recommend(summaries[-1], _history(tmp_path, summaries))
    assert result['chosen_action']['action'] != 'stop'
    cache = next(row for row in result['tool_table']
                 if row['tool_id'] == 'cache.fastwam_text_crossattn_kv')
    assert cache['exclusion'] != 'three_direction_rounds_below_2_percent'


def test_source_variant_in_plan_v2_needs_snapshot_evidence_before_next_method(tmp_path):
    summary = _campaign(tmp_path)
    plan_path = summary.with_name('plan.json')
    plan, record = json.loads(plan_path.read_text()), json.loads(summary.read_text())
    for entry in plan['entries']:
        if entry['role'] == 'candidate':
            entry['candidate']['source_ref'] = '1' * 40
    for attempt in record['attempts']:
        if attempt['candidate_id'] == 'variant':
            attempt['result']['candidate']['source_ref'] = '1' * 40
    _save(plan_path, plan)
    _save(summary, record)
    result = _module().recommend(summary, _history(tmp_path, [summary]))
    assert result['chosen_action']['action'] == 'acquire_source_evidence'
    assert result['chosen_action']['reason_code'] == 'source_requires_plan_v3'
    assert result['measured_round']['source_comparison_qualified'] is False


def test_compile_selection_requires_cold_cache_evidence_in_output(tmp_path):
    summary = _campaign(tmp_path)
    evidence = _save(tmp_path / 'other-methods.json', {'review': 'method qualification records'})
    history = _history(tmp_path, [summary], exhausted_tools=[
        {'tool': tool.tool_id, 'reason': 'already evaluated', 'evidence_ref': str(evidence)}
        for tool in all_tools() if tool.tool_id != 'compile.inductor'])
    result = _module().recommend(summary, history)
    assert result['chosen_action']['tool'] == 'compile.inductor'
    assert any('fresh empty private' in text for text in result['chosen_action']['requirements'])


def test_real_historical_pi05_full_interval_requires_split_despite_96_percent_coverage(tmp_path):
    summary = _campaign(tmp_path, baseline_only=True)
    current = _module()._read_round(summary)
    # Actual job 357483 profile, deidentified; official result remains unchanged.
    record = json.loads(json.dumps(current['incumbent']))
    record['stage_metrics'] = {
        'measurement_kind': 'diagnostic_cuda_event_intervals',
        'separate_from_latency_samples': True, 'wall_ms': 39.001307,
        'interval_ms': {'full': [37.71392059326172]},
        'interval_sum_ms': {'full': 37.71392059326172},
        'sampler_metadata': {'sampler': 'euler', 'scheduled_steps': 8, 'nfe': 8}}
    # Use the actual pi05 mapping without writing historical data as a new campaign.
    record['candidate']['backend']['model'] = 'pi05'
    current['incumbent'] = record
    profile, gap = _module()._profile(record, current, 'job357483-diagnostic')
    assert profile['attributed_fraction'] == pytest.approx(.9669911988647384)
    assert gap == 'unsplit_profile_groups'


def test_cli_writes_codex_agent_decision_through_existing_log_and_refuses_overwrite(tmp_path):
    summary = _campaign(tmp_path)
    history = _history(tmp_path, [summary])
    output, log = tmp_path / 'next.json', tmp_path / 'decisions.jsonl'
    command = [sys.executable, str(SCRIPT), '--summary', str(summary),
               '--history', str(history), '--out', str(output), '--decision-log', str(log)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    record, = DecisionLog(log).records()
    assert record['actor'] == 'agent' and record['model'] == 'fastwam'
    assert record['summary'].startswith('Codex')
    assert record['action'] == 'select_tool'
    assert str(output) in record['evidence_refs']
    assert json.loads(output.read_text())['chosen_action']['tool'] == record['tool']
    repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert len(DecisionLog(log).records()) == 1
