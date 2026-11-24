import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest

from flash_rt.vla.autotune import token_usage as usage


def _tokens(input=10, output=3, reasoning=2, read=100, write=1):
    return {'input': input, 'output': output, 'reasoning': reasoning,
            'cache': {'read': read, 'write': write}}


def _message(id='m1', when=1000, tokens=True, **extra):
    return {'id': id, 'type': 'assistant', 'time': {'created': when},
            **({'tokens': _tokens()} if tokens is True else
               {'tokens': tokens} if tokens is not None else {}), **extra}


def _segment(session='s1', model='hyvla', phase='migration', start=1000, end=2000,
             stage='denoise.velocity', tool='attention.sage', id='seg1'):
    common = dict(segment_id=id, session_id=session, model=model, phase=phase,
                  stage=stage, tool=tool, item='H', summary='Synthetic work')
    return [{**common, 'event': 'start', 'time': usage.utc_time(start)},
            {**common, 'event': 'end', 'time': usage.utc_time(end)}]


def _open_once(path, index):
    try:
        usage.SegmentLog(path).start('same-session', model='shared', phase='framework',
                                     item='H', summary='Concurrent start')
        return True
    except ValueError:
        return False


def test_segments_are_exclusive_and_close_copies_original_labels(tmp_path):
    log = usage.SegmentLog(tmp_path / 'segments.jsonl')
    first = log.start('s1', model='hyvla', phase='migration', stage='denoise.velocity',
                      tool='attention.sage', item='H', summary='Start', time=1000)
    before = log.path.read_bytes()
    with pytest.raises(ValueError, match='open'):
        log.start('s1', model='pi0', phase='optimization', item='H',
                  summary='Switch without close', time=1500)
    assert log.path.read_bytes() == before
    end = log.end('s1', time=2000)
    assert end['segment_id'] == first['segment_id']
    assert end['model'] == 'hyvla' and end['tool'] == 'attention.sage'
    with pytest.raises(ValueError, match='open'):
        log.end('s1', time=2100)
    with pytest.raises(ValueError, match='chronolog'):
        log.start('s1', model='pi0', phase='measurement', item='H',
                  summary='Backdated', time=1900)
    assert len(log.records()) == 2


def test_segment_start_serializes_competing_processes(tmp_path):
    path = str(tmp_path / 'segments.jsonl')
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_open_once, [path] * 8, range(8)))
    assert results.count(True) == 1
    assert len(usage.SegmentLog(path).records()) == 1


@pytest.mark.parametrize('expected', [
    {'expected_model': 'pi0'}, {'expected_phase': 'optimization'},
    {'expected_segment_id': 'stale-segment'},
])
def test_guarded_segment_end_never_closes_a_different_work_segment(tmp_path, expected):
    log = usage.SegmentLog(tmp_path / 'segments.jsonl')
    log.start('s1', model='shared', phase='framework', item='H', summary='Work', time=1000)
    before = log.path.read_bytes()
    with pytest.raises(ValueError, match='expected'):
        log.end('s1', time=2000, **expected)
    assert log.path.read_bytes() == before


def test_segment_end_checks_expectations_after_acquiring_the_append_lock(tmp_path, monkeypatch):
    log = usage.SegmentLog(tmp_path / 'segments.jsonl')
    first = log.start('s1', model='shared', phase='framework', item='H',
                      summary='Old work', time=1000)
    real_flock = usage.fcntl.flock
    rotated = False
    second = None

    def rotate_before_lock(stream, operation):
        nonlocal rotated, second
        if operation == usage.fcntl.LOCK_EX and not rotated:
            rotated = True
            log.end('s1', time=1500)
            second = log.start('s1', model='shared', phase='framework', item='H',
                               summary='Replacement work', time=1600)
        return real_flock(stream, operation)

    monkeypatch.setattr(usage.fcntl, 'flock', rotate_before_lock)
    with pytest.raises(ValueError, match='expected_segment_id'):
        log.end('s1', time=2000, expected_model='shared', expected_phase='framework',
                expected_segment_id=first['segment_id'])
    assert len(log.records()) == 3
    end = log.end('s1', time=2100, expected_model='shared', expected_phase='framework',
                  expected_segment_id=second['segment_id'])
    assert end['segment_id'] == second['segment_id']
    assert 'expected_model' not in end


@pytest.mark.parametrize('change', [
    {'phase': 'precision_debugging'}, {'phase': 'framework', 'model': 'pi0'},
    {'stage': 'does.not.exist'}, {'time': '2026-11-24T01:00:00'},
])
def test_invalid_segment_never_changes_ledger(tmp_path, change):
    log = usage.SegmentLog(tmp_path / 'segments.jsonl')
    fields = dict(model='hyvla', phase='debugging', item='H', summary='Work')
    fields.update(change)
    with pytest.raises(ValueError):
        log.start('s1', **fields)
    assert not log.path.exists()


def test_corrupt_or_truncated_segments_are_refused(tmp_path):
    path = tmp_path / 'segments.jsonl'
    path.write_text(json.dumps(_segment()[0]))  # No durable newline.
    before = path.read_bytes()
    with pytest.raises(ValueError, match='incomplete'):
        usage.SegmentLog(path).end('s1', time=2000)
    assert path.read_bytes() == before


def test_native_projection_deduplicates_settlement_and_excludes_raw_content():
    raw = _message(content=[{'type': 'text', 'text': 'private secret sk-never-publish'}],
                   native={'responseId': 'private-request-id'},
                   metadata={'authorization': 'Bearer private'})
    projected = usage.project_opencode('s1', raw)
    assert projected['counters'] == {
        'input': 10, 'output': 3, 'reasoning': 2, 'cache_read': 100, 'cache_write': 1}
    serialized = json.dumps(projected)
    assert 'private' not in serialized and 'Bearer' not in serialized
    assert 'content' not in projected and 'native' not in projected
    missing = usage.project_opencode('s1', _message(tokens=None))
    rows = usage.deduplicate([missing, projected, projected])
    assert len(rows) == 1 and rows[0]['counters']['cache_read'] == 100
    conflict = usage.project_opencode('s1', _message(tokens=_tokens(input=99)))
    with pytest.raises(ValueError, match='conflicting'):
        usage.deduplicate([projected, conflict])
    assert usage.project_opencode('s1', {'id': 'receipt', 'type': 'user',
                                        'text': 'child used 999 tokens'}) is None


def test_refresh_preserves_failed_usage_and_settles_existing_requests_once():
    previous = [usage.project_opencode('s1', _message(tokens=None)),
                usage.project_opencode('s1', _message('disappeared', 1500, None))]
    current = [usage.project_opencode('s1', _message()),
               usage.project_opencode('s1', _message('later', 4000))]
    rows = usage.merge_collection(previous, current, start=1000, cutoff=3000)
    assert len(rows) == 2
    assert rows[0]['counters']['input'] == 10
    assert rows[1]['message_id'] == 'disappeared'
    assert all(value is None for value in rows[1]['counters'].values())


def test_open_code_collector_follows_short_pages_and_cursor_cycles_fail():
    paths = []

    def fetch(path):
        paths.append(path)
        if len(paths) == 1:
            return {'data': [_message()], 'cursor': {'next': 'cursor with space'}}
        return {'data': [_message('m2', 2000)], 'cursor': {}}

    rows = usage.collect_opencode(['s1'], get=fetch, page_size=200)
    assert len(rows) == 2
    assert 'cursor=cursor+with+space' in paths[1] and 'order=' not in paths[1]

    def cycle(path):
        return {'data': [_message()], 'cursor': {'next': 'same'}}

    with pytest.raises(ValueError, match='cursor'):
        usage.collect_opencode(['s1'], get=cycle)


def test_collection_window_reads_newest_first_and_stops_at_old_page():
    paths = []

    def fetch(path):
        paths.append(path)
        if len(paths) == 1:
            return {'data': [_message('future', 4000), _message('new', 2500)],
                    'cursor': {'next': 'older'}}
        if len(paths) == 2:
            return {'data': [_message('boundary', 2000), _message('old', 1000)],
                    'cursor': {'next': 'do-not-read'}}
        pytest.fail('collector read beyond the requested native time window')

    rows = usage.collect_opencode(['s1'], get=fetch, start=2000, cutoff=3000)
    assert [row['message_id'] for row in rows] == ['new', 'boundary']
    assert 'order=desc' in paths[0] and 'order=' not in paths[1]


def _codex_event(time, input, cached, output, reasoning, last=None):
    counts = dict(input_tokens=input, cached_input_tokens=cached,
                  cache_write_input_tokens=0, output_tokens=output,
                  reasoning_output_tokens=reasoning, total_tokens=input + output)
    return {'timestamp': usage.utc_time(time), 'type': 'event_msg',
            'payload': {'type': 'token_count', 'info': {
                'total_token_usage': counts, 'last_token_usage': last or counts,
                'last_usage_includes_reasoning': True}}}


def test_codex_cumulative_counts_do_not_bill_rate_limit_events_twice(tmp_path):
    path = tmp_path / 'codex.jsonl'
    first = _codex_event(1000, 100, 40, 20, 5)
    second = _codex_event(2000, 180, 90, 32, 8)
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'session_meta', 'payload': {'id': 'codex1', 'api_key': 'private'}},
        first, first, second, second]) + '\n')
    rows = usage.collect_codex([path])
    assert len(rows) == 2
    assert rows[0]['counters'] == {
        'input': 60, 'cache_read': 40, 'cache_write': 0, 'output': 15, 'reasoning': 5}
    assert rows[1]['counters'] == {
        'input': 30, 'cache_read': 50, 'cache_write': 0, 'output': 9, 'reasoning': 3}
    assert 'private' not in json.dumps(rows)
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'session_meta', 'payload': {'id': 'codex1'}}, second, first]) + '\n')
    with pytest.raises(ValueError, match='decreas'):
        usage.collect_codex([path])


def test_codex_missing_usage_and_resumed_prefix_are_explicitly_unknown(tmp_path):
    path = tmp_path / 'partial.jsonl'
    first = _codex_event(1000, 100, 40, 20, 5)
    last = first['payload']['info']['last_token_usage']
    resumed = _codex_event(2000, 180, 90, 32, 8, last=last)
    incomplete = _codex_event(3000, 200, 100, 35, 9)
    incomplete['payload']['info']['total_token_usage']['cache_write_input_tokens'] = None
    incomplete['payload']['info']['last_token_usage']['cache_write_input_tokens'] = None
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'session_meta', 'payload': {'id': 'codex1'}},
        resumed, incomplete]) + '\n')
    rows = usage.collect_codex([path])
    assert rows[0]['kind'] == 'cumulative_carry_in'
    assert rows[0]['timestamp'] is None
    assert rows[0]['counters']['cache_read'] == 50
    assert rows[2]['counters']['cache_write'] is None
    report = usage.build_report(rows, [], run_start=0, cutoff=4000)
    assert report['unresolved_timestamp_records'] == 1
    assert report['current_run']['missing_usage_records'] == 1
    assert report['current_run']['passes_unattributed_gate'] is False


def test_codex_partial_snapshot_recovery_never_counts_the_cumulative_prefix_twice(tmp_path):
    path = tmp_path / 'partial-recovered.jsonl'
    middle = _codex_event(2000, 180, 90, 32, 8)
    middle['payload']['info']['total_token_usage']['cache_write_input_tokens'] = None
    middle['payload']['info']['last_token_usage'] = {
        'input_tokens': 80, 'cached_input_tokens': 50, 'cache_write_input_tokens': None,
        'output_tokens': 12, 'reasoning_output_tokens': 3, 'total_tokens': 92}
    final = _codex_event(3000, 200, 100, 36, 9)
    final['payload']['info']['last_token_usage'] = {
        'input_tokens': 20, 'cached_input_tokens': 10, 'cache_write_input_tokens': 0,
        'output_tokens': 4, 'reasoning_output_tokens': 1, 'total_tokens': 24}
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'session_meta', 'payload': {'id': 'codex1'}},
        _codex_event(1000, 100, 40, 20, 5), middle, final]) + '\n')
    rows = usage.collect_codex([path])
    assert sum(row['counters']['cache_read'] or 0 for row in rows) == 100
    assert sum(row['counters']['output'] or 0 for row in rows) == 27
    assert sum(row['counters']['reasoning'] or 0 for row in rows) == 9
    assert len(rows) == 3
    assert rows[1]['counters']['cache_write'] is None


def test_submillisecond_start_does_not_absorb_an_earlier_message():
    segments = _segment(start=1000, end=2000)
    segments[0]['time'] = '1970-01-01T00:00:01.000270+00:00'
    report = usage.build_report([usage.project_opencode('s1', _message())],
                                segments, run_start=0, cutoff=3000)
    assert report['current_run']['unattributed_fraction'] == 1


def test_message_boundary_missing_usage_and_unattributed_stay_visible():
    rows = [usage.project_opencode('s1', _message()),
            usage.project_opencode('s1', _message('missing', 1500, None)),
            usage.project_opencode('s1', _message('outside', 2000))]
    report = usage.build_report(rows, _segment(), run_start=1000, cutoff=3000)
    assert report['current_run']['total']['all_processed_tokens'] == 232
    assert report['current_run']['missing_usage_records'] == 1
    assert report['current_run']['unattributed_fraction'] == 0.5
    assert report['current_run']['passes_unattributed_gate'] is False
    hy = next(row for row in report['model_lifecycle'] if row['model'] == 'hyvla')
    assert hy['phases']['migration']['all_processed_tokens'] == 116
    assert hy['missing_usage_records'] == 1
    assert next(row for row in report['model_lifecycle']
                if row['model'] == 'unattributed')['total']['all_processed_tokens'] == 116
    assert len([p for p in hy['phases'] if p in usage.PHASES]) == 6


def test_shared_compaction_counts_once_and_fork_copy_is_deduplicated():
    a = usage.project_opencode('s1', _message())
    compaction = usage.project_opencode('s1', {
        'id': 'compact', 'type': 'compaction', 'time': {'created': 1500},
        'status': 'completed', 'tokens': _tokens()})
    copied = usage.project_opencode('fork', _message())
    report = usage.build_report([a, compaction, copied],
                                _segment(model='shared', phase='framework',
                                         stage=None, tool=None),
                                run_start=1000, cutoff=3000)
    assert report['current_run']['total']['all_processed_tokens'] == 232
    assert report['current_run']['unattributed_fraction'] == 0
    assert report['deduplicated_records'] == 1


def test_attribution_gate_uses_known_counts_and_reports_unknown_calls_separately():
    known = usage.project_opencode('s1', _message())
    missing = usage.project_opencode('unlabelled', _message('failed', 1500, None))
    report = usage.build_report([known, missing],
                                _segment(model='shared', phase='framework',
                                         stage=None, tool=None),
                                run_start=1000, cutoff=3000)
    assert report['current_run']['unattributed_fraction'] == 0
    assert report['current_run']['unattributed']['missing_usage_records'] == 1
    assert report['current_run']['gate_uses_known_recorded_counters'] is True
    assert report['current_run']['passes_unattributed_gate'] is True


def test_stage_speedup_uses_explicit_matching_experiment_and_counts_rejects():
    rows = [usage.project_opencode('s1', _message())]
    experiment = {'model': 'hyvla', 'stage': 'denoise.velocity', 'tool': 'attention.sage',
                  'before_ms': 8, 'after_ms': 6, 'outcome': 'keep',
                  'evidence_ref': 'docs/reports/public-result.json'}
    report = usage.build_report(rows, _segment(), run_start=1000, cutoff=3000,
                                experiments=[experiment])
    stage = report['model_stage'][0]
    assert stage['saved_ms'] == 2
    assert stage['ms_saved_per_million_tokens'] == pytest.approx(17241.37931034483)
    assert report['method_model'][0]['total']['all_processed_tokens'] == 116
    report = usage.build_report(rows, _segment(), run_start=1000, cutoff=3000)
    assert report['model_stage'][0]['saved_ms'] is None
    with pytest.raises(ValueError, match='ambiguous'):
        usage.build_report(rows, _segment(), run_start=1000, cutoff=3000,
                           experiments=[experiment, experiment])


def test_missing_stage_or_method_labels_do_not_erase_migration_cost():
    report = usage.build_report(
        [usage.project_opencode('s1', _message())],
        _segment(stage=None, tool=None), run_start=1000, cutoff=3000)
    assert report['model_stage'][0]['stage'] == 'unknown'
    assert report['model_stage'][0]['total']['all_processed_tokens'] == 116
    assert report['method_model'][0]['tool'] == 'unknown'
    assert report['method_model'][0]['total']['all_processed_tokens'] == 116


def _history(cutoff, amount, phases=False):
    result = {'window': {'start_ms': 0, 'cutoff_ms': cutoff},
              'models': {'pi0': {
                  'input': amount, 'output': 0, 'reasoning': 0,
                  'cache_read': 0, 'cache_write': 0,
                  'all_processed_tokens': amount, 'uncached_work_tokens': amount,
                  'usage_records': 1, 'missing_usage_records': 2}}}
    if phases:
        result['stages'] = [{'model': 'pi0', 'stage': 'initial_adaptation',
                             'input': amount, 'output': 0, 'reasoning': 0,
                             'cache_read': 0, 'cache_write': 0}]
    return result


def test_historical_snapshots_preserved_and_never_summed_over_each_other(tmp_path):
    old = tmp_path / 'old.json'
    latest = tmp_path / 'latest.json'
    old.write_text(json.dumps(_history(1000, 100, phases=True)))
    latest.write_text(json.dumps(_history(2000, 150)))
    originals = [old.read_bytes(), latest.read_bytes()]
    rows = [usage.project_opencode('s1', _message('overlap', 1500)),
            usage.project_opencode('s1', _message('gap', 2500)),
            usage.project_opencode('s1', _message('current', 3500))]
    report = usage.build_report(
        rows, _segment(model='pi0', phase='onboarding', start=3000, end=4000,
                       stage=None, tool=None),
        history=[old, latest], run_start=3000, cutoff=4000)
    assert report['grand_total']['all_processed_tokens'] == 382
    assert report['historical_total']['all_processed_tokens'] == 150
    assert report['gap']['total']['all_processed_tokens'] == 116
    assert report['gap']['source'] == '估算'
    assert len(report['historical_snapshots']) == 2
    assert originals == [old.read_bytes(), latest.read_bytes()]
    pi = next(row for row in report['onboarding_trend'] if row['model'] == 'pi0')
    assert pi['historical_onboarding_tokens'] == 100
    assert pi['onboarding_tokens'] == 216
    assert pi['historical_completion_is_proxy'] is True
    assert report['current_run']['unattributed_fraction'] == 0


def test_new_model_onboarding_duration_and_missing_dm_historical_phase(tmp_path):
    root = Path(__file__).resolve().parents[1]
    history = [root / 'docs/reports/2026-11-15-vla-migration-token-usage.json',
               root / 'docs/reports/2026-11-16-vla-development-token-usage-dm05-optimized.json']
    report = usage.build_report([], [], history=history,
                                run_start=1795486880000, cutoff=1795486881000)
    assert report['historical_total']['all_processed_tokens'] == 1741698368
    expected = {'pi0': 16849618, 'pi05': 33433601,
                'fastwam': 32129035, 'hyvla': 17007184}
    assert {row['model']: row['historical_onboarding_tokens']
            for row in report['onboarding_trend'] if row['model'] in expected} == expected
    dm = next(row for row in report['onboarding_trend'] if row['model'] == 'dm05')
    assert dm['historical_onboarding_tokens'] is None
    assert dm['historical_total_tokens'] == 116318116


def test_later_segment_closure_cannot_rewrite_a_past_onboarding_snapshot():
    report = usage.build_report(
        [usage.project_opencode('s1', _message())],
        _segment(model='new-vla', phase='onboarding', stage=None, tool=None, end=5000),
        run_start=1000, cutoff=3000)
    model = next(row for row in report['onboarding_trend'] if row['model'] == 'new-vla')
    assert model['onboarding_tokens'] == 116
    assert model['completed_utc'] is None and model['elapsed_seconds'] is None


def test_nonoverlapping_history_does_not_hide_native_usage_between_windows(tmp_path):
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    first.write_text(json.dumps(_history(1000, 100)))
    later = _history(3000, 200)
    later['window']['start_ms'] = 2000
    second.write_text(json.dumps(later))
    records = [usage.project_opencode('gap', _message(when=1500))]
    report = usage.build_report(records, [], history=[first, second],
                                run_start=4000, cutoff=5000)
    assert report['gap']['total']['all_processed_tokens'] == 116
    assert report['grand_total']['all_processed_tokens'] == 416


def test_report_cli_emits_four_aggregate_tables_without_session_or_prompt(tmp_path):
    records = tmp_path / 'records.jsonl'
    segments = tmp_path / 'segments.jsonl'
    records.write_text(json.dumps(usage.project_opencode(
        'sensitive-session', _message(content=[{'type': 'text', 'text': 'SECRET'}]))) + '\n')
    segments.write_text('\n'.join(json.dumps(row) for row in
                                 _segment(session='sensitive-session')) + '\n')
    prefix = tmp_path / 'report'
    result = subprocess.run(
        [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'report',
         '--records', str(records), '--segments', str(segments),
         '--run-start', usage.utc_time(1000), '--cutoff', usage.utc_time(3000),
         '--output', str(prefix)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for suffix in ('.json', '.zh-CN.md', '-model-lifecycle.csv', '-model-stage.csv',
                   '-method-model.csv', '-onboarding-trend.csv'):
        content = Path(str(prefix) + suffix).read_text()
        assert 'sensitive-session' not in content and 'SECRET' not in content
    markdown = Path(str(prefix) + '.zh-CN.md').read_text()
    assert '反复读取的缓存' in markdown and '来源' in markdown
    assert '新增 token' in markdown and '费用' in markdown


def test_collect_cli_refuses_publishing_message_ledger_under_docs(tmp_path):
    (tmp_path / 'docs/reports').mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'collect-opencode',
         '--session', 'should-not-fetch', '--output', str(tmp_path / 'docs/reports/raw.jsonl')],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'private' in result.stderr
    assert not (tmp_path / 'docs/reports/raw.jsonl').exists()


def test_cli_report_help_formats_the_attribution_threshold():
    result = subprocess.run(
        [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'report', '--help'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '5%' in result.stdout


def test_cli_end_can_guard_all_three_expected_labels(tmp_path):
    log = usage.SegmentLog(tmp_path / 'segments.jsonl')
    first = log.start('s1', model='shared', phase='framework', item='H', summary='Work')
    before = log.path.read_bytes()
    args = [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'end',
            '--segments', str(log.path), '--session', 's1',
            '--expected-model', 'shared', '--expected-phase', 'framework',
            '--expected-segment-id']
    rejected = subprocess.run(args + ['stale'], capture_output=True, text=True)
    assert rejected.returncode == 1
    assert log.path.read_bytes() == before
    closed = subprocess.run(args + [first['segment_id']], capture_output=True, text=True)
    assert closed.returncode == 0, closed.stderr
    assert len(log.records()) == 2


def test_native_collection_does_not_require_the_model_stage_registry():
    result = subprocess.run(
        [sys.executable, '-c',
         "import sys; sys.modules['flash_rt.vla.autotune.stages'] = None; "
         "from flash_rt.vla.autotune import token_usage; "
         "assert token_usage.project_opencode('s', {'id': 'u', 'type': 'user'}) is None"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_report_uses_collection_cutoff_and_refuses_uncollected_later_window(tmp_path):
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(usage.project_opencode('s1', _message())) + '\n')
    Path(str(path) + '.collection.json').write_text(json.dumps({
        'schema': usage.SCHEMA, 'start_utc': usage.utc_time(0),
        'cutoff_utc': usage.utc_time(3000)}))
    segments = tmp_path / 'segments.jsonl'
    segments.write_text('\n'.join(json.dumps(row) for row in _segment()) + '\n')
    args = [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'report',
            '--records', str(path), '--segments', str(segments),
            '--run-start', usage.utc_time(1000), '--output', str(tmp_path / 'report')]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['window']['cutoff_utc'] == usage.utc_time(3000)
    result = subprocess.run(args + ['--cutoff', usage.utc_time(4000)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'collection cutoff' in result.stderr


def test_report_refuses_a_ledger_that_no_longer_matches_its_collection_manifest(tmp_path):
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(usage.project_opencode('s1', _message())) + '\n')
    Path(str(path) + '.collection.json').write_text(json.dumps({
        'schema': usage.SCHEMA, 'cutoff_utc': usage.utc_time(3000),
        'records_sha256': '0' * 64}))
    segments = tmp_path / 'segments.jsonl'
    segments.write_text('')
    result = subprocess.run(
        [sys.executable, '-m', 'flash_rt.vla.autotune.token_usage', 'report',
         '--records', str(path), '--segments', str(segments),
         '--run-start', usage.utc_time(1000), '--output', str(tmp_path / 'report')],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert 'digest' in result.stderr
    assert not (tmp_path / 'report.json').exists()
