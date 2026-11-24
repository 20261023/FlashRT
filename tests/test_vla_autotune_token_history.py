"""Historical packaging must not turn overlapping publications into extra usage."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from flash_rt.vla.autotune import token_usage as usage


def _snapshot(source, cutoff, amount, *, phases=False):
    data = {
        'window': {'start_ms': 0, 'cutoff_ms': cutoff},
        'models': {'pi0': {
            'input': amount, 'output': 0, 'reasoning': 0,
            'cache_read': 0, 'cache_write': 0,
            'all_processed_tokens': amount, 'uncached_work_tokens': amount,
            'usage_records': 3, 'missing_usage_records': 2}},
    }
    if phases:
        data['window']['initial_phase_end_ms'] = 500
        data['stages'] = [{
            'model': 'pi0', 'stage': 'initial_adaptation',
            'input': 60, 'output': 0, 'reasoning': 0,
            'cache_read': 0, 'cache_write': 0,
            'all_processed_tokens': 60, 'uncached_work_tokens': 60,
        }]
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
    return {'source': source, 'sha256': hashlib.sha256(raw).hexdigest(), 'data': data}


def _package(tmp_path):
    snapshots = [
        _snapshot('old.json', 1000, 100, phases=True),
        _snapshot('middle.json', 2000, 130),
        _snapshot('latest.json', 3000, 150),
    ]
    data = {
        'schema': 'flash_rt.autotune.token_history.v1',
        'previous_latest_cumulative_source': 'latest.json',
        'snapshots': snapshots,
    }
    path = tmp_path / 'history.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, data


def test_packaged_snapshot_fusion_keeps_original_cutoffs_counts_and_phase_proxy(tmp_path):
    path, _ = _package(tmp_path)
    before = path.read_bytes()
    report = usage.build_report([], [], history=[path], run_start=4000, cutoff=5000)
    assert report['historical_total']['all_processed_tokens'] == 150
    assert report['grand_total']['all_processed_tokens'] == 150
    assert [(row['source'], row['cutoff_ms']) for row in report['historical_snapshots']] == [
        ('old.json', 1000), ('middle.json', 2000), ('latest.json', 3000)]
    pi = next(row for row in report['model_lifecycle'] if row['model'] == 'pi0')
    assert pi['records'] == 3 and pi['missing_usage_records'] == 2
    assert pi['phases']['onboarding']['all_processed_tokens'] == 60
    assert pi['phases']['unknown']['all_processed_tokens'] == 90
    trend = next(row for row in report['onboarding_trend'] if row['model'] == 'pi0')
    assert trend['historical_onboarding_tokens'] == 60
    assert trend['historical_end_utc'] == '1970-01-01T00:00:00.500000+00:00'
    assert trend['historical_completion_is_proxy'] is True
    assert path.read_bytes() == before


def test_packaged_and_standalone_snapshot_are_deduplicated_by_original_identity(tmp_path):
    path, package = _package(tmp_path)
    standalone = tmp_path / 'old.json'
    standalone.write_text(
        json.dumps(package['snapshots'][0]['data'], ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    report = usage.build_report([], [], history=[path, standalone],
                                run_start=4000, cutoff=5000)
    assert len(report['historical_snapshots']) == 3
    assert report['historical_total']['all_processed_tokens'] == 150


def test_modified_packaged_snapshot_is_rejected_before_it_can_change_old_totals(tmp_path):
    path, package = _package(tmp_path)
    package['snapshots'][-1]['data']['models']['pi0']['input'] = 999
    path.write_text(json.dumps(package), encoding='utf-8')
    with pytest.raises(ValueError, match='historical snapshot digest'):
        usage.build_report([], [], history=[path], run_start=4000, cutoff=5000)


@pytest.mark.parametrize('latest', ['absent.json', 'middle.json'])
def test_previous_latest_pointer_must_bind_the_actual_last_cumulative_snapshot(tmp_path, latest):
    path, package = _package(tmp_path)
    package['previous_latest_cumulative_source'] = latest
    path.write_text(json.dumps(package), encoding='utf-8')
    with pytest.raises(ValueError, match='previous latest cumulative'):
        usage.build_report([], [], history=[path], run_start=4000, cutoff=5000)


@pytest.mark.parametrize('archive_first', [True, False])
def test_same_source_name_with_changed_original_bytes_is_not_silently_replaced(
        tmp_path, archive_first):
    path, package = _package(tmp_path)
    standalone = tmp_path / 'old.json'
    standalone.write_text(json.dumps(_snapshot('old.json', 1500, 120)['data']),
                          encoding='utf-8')
    paths = [path, standalone] if archive_first else [standalone, path]
    with pytest.raises(ValueError, match='conflicting historical source'):
        usage.build_report([], [], history=paths, run_start=4000, cutoff=5000)


def test_standalone_snapshots_with_same_basename_can_cover_disjoint_windows(tmp_path):
    first = tmp_path / 'first' / 'snapshot.json'
    second = tmp_path / 'second' / 'snapshot.json'
    later = _snapshot(second.name, 3000, 200)['data']
    later['window']['start_ms'] = 2000
    for path, data in [(first, _snapshot(first.name, 1000, 100)['data']), (second, later)]:
        path.parent.mkdir()
        path.write_text(json.dumps(data), encoding='utf-8')
    report = usage.build_report([], [], history=[first, second],
                                run_start=4000, cutoff=5000)
    assert report['historical_total']['all_processed_tokens'] == 300
    assert len(report['historical_snapshots']) == 2


def test_previous_latest_cumulative_cutoff_cannot_be_shared_by_another_snapshot(tmp_path):
    path, package = _package(tmp_path)
    package['snapshots'].append(_snapshot('shadow.json', 3000, 150))
    path.write_text(json.dumps(package), encoding='utf-8')
    with pytest.raises(ValueError, match='previous latest cumulative'):
        usage.build_report([], [], history=[path], run_start=4000, cutoff=5000)


def test_repository_package_recovers_original_json_and_csv_bytes_and_latest_counts():
    path = Path(__file__).resolve().parents[1] / 'docs/reports/vla-token-history.json'
    package = json.loads(path.read_text(encoding='utf-8'))
    artifacts = {row['path']: row for row in package['archived_artifacts']}
    for entry in package['snapshots']:
        raw = (json.dumps(entry['data'], ensure_ascii=False, indent=2) + '\n').encode()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        csv_path = 'docs/reports/' + entry['source'].replace('.json', '.csv')
        if csv_path not in artifacts:
            continue
        fields = ['model', *usage.COUNTERS, 'uncached_work_tokens',
                  'all_processed_tokens', 'usage_records', 'missing_usage_records']
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for model, counts in entry['data']['models'].items():
            writer.writerow({'model': model, **{key: counts[key] for key in fields[1:]}})
        encoded = stream.getvalue().encode()
        assert len(encoded) == artifacts[csv_path]['bytes']
        assert hashlib.sha256(encoded).hexdigest() == artifacts[csv_path]['sha256']
    report = usage.build_report([], [], history=[path],
                                run_start=1795486880000, cutoff=1795486881000)
    assert report['historical_total'] == {
        'input': 46287373, 'output': 2314962, 'reasoning': 2812385,
        'cache_read': 1690283648, 'cache_write': 0,
        'uncached_work_tokens': 51414720, 'all_processed_tokens': 1741698368}
    assert [row['cutoff_ms'] for row in report['historical_snapshots']] == [
        1794744759409, 1794818933180, 1794822537044, 1794834164308]
    assert {row['model']: row['total']['all_processed_tokens']
            for row in report['model_lifecycle']} == {
                'pi0': 18303969, 'pi05': 42477950, 'fastwam': 164187442,
                'hyvla': 773133518, 'dm05': 116318116, 'shared': 627277373}
    assert sum(row['missing_usage_records'] for row in report['model_lifecycle']) == 14
    assert sum(row['records'] for row in report['model_lifecycle']) == 4904
    assert next(row for row in report['onboarding_trend']
                if row['model'] == 'dm05')['historical_onboarding_tokens'] is None


def test_report_cli_accepts_one_history_package_and_emits_the_four_tables(tmp_path, capsys):
    history = Path(__file__).resolve().parents[1] / 'docs/reports/vla-token-history.json'
    prefix = tmp_path / 'report'
    status = usage.main([
        'report', '--history', str(history), '--segments', str(tmp_path / 'empty.jsonl'),
        '--run-start', '2026-11-24T02:21:18.867Z', '--cutoff', '2026-11-24T03:57:08Z',
        '--output', str(prefix)])
    assert status == 0
    assert json.loads(capsys.readouterr().out)['grand_total']['all_processed_tokens'] == 1741698368
    for suffix in ('-model-lifecycle.csv', '-model-stage.csv',
                   '-method-model.csv', '-onboarding-trend.csv'):
        output = Path(str(prefix) + suffix).read_bytes()
        assert b'\r\n' not in output
