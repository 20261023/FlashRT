"""Private native usage collection and public VLA development cost aggregates.

OpenCode V2 ``TokenUsage.Info`` already separates uncached input, output,
reasoning and cache. Codex ``token_count`` cumulative input includes cached
input; output includes reasoning when ``last_usage_includes_reasoning`` is
true (the default for the native CLI). Never add cumulative snapshots or
parent-tool receipts as if they were additional requests.

Examples (no torch or model computation):

    python -m flash_rt.vla.autotune.token_usage start --model shared \
        --phase framework --item H --summary "Implement usage accounting"
    python -m flash_rt.vla.autotune.token_usage end
    python -m flash_rt.vla.autotune.token_usage collect-opencode \
        --root ses_example --output devlocal/tokens/native.jsonl
    python -m flash_rt.vla.autotune.token_usage collect-codex \
        --log /private/rollout.jsonl --output devlocal/tokens/codex.jsonl
    python -m flash_rt.vla.autotune.token_usage report \
        --records devlocal/tokens/native.jsonl \
        --run-start 2026-11-24T02:21:18.867Z \
        --history docs/reports/2026-11-15-vla-migration-token-usage.json \
        --history docs/reports/2026-11-16-vla-development-token-usage-dm05-optimized.json \
        --output docs/reports/2026-11-24-vla-token-usage

Collectors only project whitelisted metadata, never prompts, reasoning,
tool inputs/results, credentials, or raw provider responses. Keep their
per-message ledgers private; only ``report`` outputs are for publication.
Historical reports are immutable inputs. The latest cumulative snapshot
replaces overlapping older totals, while older onboarding breakdowns and
every original snapshot's aggregate values remain available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlencode
from uuid import uuid4

PHASES = ('onboarding', 'migration', 'optimization', 'debugging', 'measurement', 'framework')
COUNTERS = ('input', 'output', 'reasoning', 'cache_read', 'cache_write')
DEFAULT_MODELS = ('pi0', 'pi05', 'fastwam', 'hyvla', 'dm05')
SEGMENTS = 'docs/vla_autotune_token_segments.jsonl'
SCHEMA = 'flash_rt.autotune.token_usage.v1'
_OLD_PHASES = {'initial_adaptation': 'onboarding', 'precision_debugging': 'debugging',
               'optimization': 'optimization', 'benchmark_reporting': 'measurement',
               'shared_iteration': 'framework', 'context_compaction': 'framework'}
_LABEL = re.compile(r'^[a-z0-9][a-z0-9_.-]*$')
_SEGMENT_FIELDS = ('segment_id', 'event', 'time', 'session_id', 'model', 'phase',
                   'stage', 'tool', 'item', 'summary')


def utc_ms(value):
    """Accept epoch milliseconds or timezone-aware ISO UTC, preserving microseconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError('time must be finite epoch milliseconds')
        return value
    if not isinstance(value, str):
        raise ValueError('time must be UTC ISO-8601 or epoch milliseconds')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('time must be UTC ISO-8601') from None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('time must be UTC, with Z or +00:00')
    return parsed.timestamp() * 1000


def utc_time(value=None):
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')
    return datetime.fromtimestamp(utc_ms(value) / 1000, timezone.utc).isoformat(
        timespec='microseconds')


def _label(value, field):
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError(f'{field} must be a lowercase identifier')
    return value


def _known_stage(value):
    # Native collection must also work while no model runtime/registry is
    # installed. Only segment and experiment validation need stage IDs.
    from .stages import STAGES
    return value in STAGES


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value


def _count(value):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        raise ValueError('usage counters must be nonnegative integers or null')
    return int(value)


def _vector(values=None):
    return {key: _count((values or {}).get(key)) for key in COUNTERS}


def _zero():
    return dict.fromkeys(COUNTERS, 0)


def _add(target, values):
    for key in COUNTERS:
        if values.get(key) is not None:
            target[key] += values[key]


def _enrich(values):
    return {**values, 'uncached_work_tokens': sum(values[k] for k in COUNTERS[:3]),
            'all_processed_tokens': sum(values[k] for k in COUNTERS)}


def _difference(new, old):
    result = {key: new[key] - old.get(key, 0) for key in COUNTERS}
    if any(value < 0 for value in result.values()):
        raise ValueError('cumulative usage counters decreased; supply nonoverlapping snapshots')
    return result


def _json_line(row):
    # ASCII escaping also prevents U+2028/U+2029 from breaking JSONL consumers.
    return json.dumps(row, ensure_ascii=True, allow_nan=False, sort_keys=True) + '\n'


def _read_jsonl(stream):
    data = stream.read()
    if data and not data.endswith('\n'):
        raise ValueError('ledger has an incomplete final line; refusing to append')
    try:
        return [json.loads(line) for line in data.split('\n') if line]
    except json.JSONDecodeError:
        raise ValueError('ledger contains invalid JSON') from None


def segment_intervals(records, *, cutoff=None):
    """Validate a full event ledger and produce nonoverlapping [start, end) spans."""
    opened, seen, last_time, intervals = {}, set(), {}, []
    for row in records:
        if not isinstance(row, dict) or set(row) != set(_SEGMENT_FIELDS):
            raise ValueError('segment record fields do not match the work-segment schema')
        for field in ('segment_id', 'session_id', 'item', 'summary'):
            _text(row[field], field)
        _label(row['model'], 'model')
        if row['phase'] not in PHASES:
            raise ValueError(f'phase must be one of {PHASES}')
        if row['phase'] == 'framework' and row['model'] != 'shared':
            raise ValueError('framework phase requires model shared')
        if row['stage'] is not None and not _known_stage(row['stage']):
            raise ValueError('stage must be a known stage id')
        if row['tool'] is not None:
            _label(row['tool'], 'tool')
        sid, tag, time = row['session_id'], row['segment_id'], utc_ms(row['time'])
        if time < last_time.get(sid, time):
            raise ValueError('segment events must be chronological within a session')
        last_time[sid] = time
        if row['event'] == 'start':
            if sid in opened:
                raise ValueError('session already has an open segment')
            if tag in seen:
                raise ValueError('duplicate segment_id')
            seen.add(tag)
            opened[sid] = row
        elif row['event'] == 'end':
            if sid not in opened:
                raise ValueError('session has no open segment')
            start = opened[sid]
            if any(row[k] != start[k] for k in _SEGMENT_FIELDS
                   if k not in ('event', 'time', 'summary')):
                raise ValueError('end segment labels must match its start')
            del opened[sid]
            intervals.append({**start, 'start_ms': utc_ms(start['time']),
                              'end_ms': time, 'closed': True})
        else:
            raise ValueError('segment event must be start or end')
    for start in opened.values():
        intervals.append({**start, 'start_ms': utc_ms(start['time']),
                          'end_ms': utc_ms(cutoff) if cutoff is not None else None,
                          'closed': False})
    return intervals


class SegmentLog:
    """Serialize validation plus append under one flock, then flush and fsync."""

    def __init__(self, path=SEGMENTS):
        self.path = Path(path)

    def records(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding='utf-8') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            rows = _read_jsonl(stream)
            segment_intervals(rows)
            return rows

    def start(self, session_id, *, model, phase, item, summary, stage=None,
              tool=None, time=None):
        row = dict(segment_id=uuid4().hex, event='start', time=utc_time(time),
                   session_id=session_id, model=model, phase=phase, stage=stage,
                   tool=tool, item=item, summary=summary)
        segment_intervals([row])  # Invalid input must not create a ledger.
        return self._append(row, default_time=time is None)

    def end(self, session_id, *, time=None, summary=None, expected_model=None,
            expected_phase=None, expected_segment_id=None):
        """Close only the caller's expected segment, checked under the append lock.

        Guards are optional for interactive compatibility. Callers that first
        read the ledger (for example onboarding qualification) should provide
        all three to avoid closing work opened by a competing writer.
        """
        expected = {key: value for key, value in (
            ('model', expected_model), ('phase', expected_phase),
            ('segment_id', expected_segment_id)) if value is not None}
        return self._append(dict(event='end', session_id=session_id, time=utc_time(time),
                                 summary=summary), default_time=time is None,
                            expected=expected)

    def _append(self, row, *, default_time, expected=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a+', encoding='utf-8') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            rows = _read_jsonl(stream)
            spans = segment_intervals(rows)
            if default_time:
                row['time'] = utc_time()
            if row['event'] == 'end':
                current = next((s for s in spans if not s['closed']
                                and s['session_id'] == row['session_id']), None)
                if current is None:
                    raise ValueError('session has no open segment')
                for key, value in (expected or {}).items():
                    if current[key] != value:
                        raise ValueError(f'open segment does not match expected_{key}')
                row = {key: current[key] for key in _SEGMENT_FIELDS} | {
                    'event': 'end', 'time': row['time'],
                    'summary': row['summary'] or current['summary']}
            segment_intervals([*rows, row])
            stream.write(_json_line(row))
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return row


def _fingerprint(native_id):
    return hashlib.sha256(native_id.encode()).hexdigest() if native_id else None


def _task_hints(message):
    """Classify current activity in memory; never retain its underlying text."""
    chunks = []
    for part in message.get('content', []):
        if part.get('type') in ('text', 'reasoning'):
            chunks.append(part.get('text', ''))
        elif part.get('type') == 'tool':
            chunks.append(json.dumps(part.get('state', {}).get('input', {}),
                                     ensure_ascii=False))
    text = '\n'.join(chunks).lower()
    models = {
        'pi05': r'pi0[._]?5|pi05|π0\.5',
        'pi0': r'(?<![\w])(?:pi0|π0)(?![.\w])',
        'hyvla': r'hyvla|hy[-_]embodied|(?<!\w)hy(?!\w)',
        'fastwam': r'fastwam',
        'dm05': r'dm0[._]?5|dm05',
    }
    phases = {'onboarding': r'onboard|接入', 'migration': r'\bport\b|移植|迁移',
              'optimization': r'optimiz|autotun|优化|调优',
              'debugging': r'\bdebug|排查|精度问题|崩溃',
              'measurement': r'benchmark|latency|测量|计时|报告',
              'framework': r'harness|工具表|通用框架'}
    return {'task_models': [m for m, pattern in models.items() if re.search(pattern, text)],
            'task_phases': [p for p, pattern in phases.items() if re.search(pattern, text)]}


def project_opencode(session_id, message):
    """Project actual assistant/compaction counters, with no raw message payload."""
    kind = message.get('type', message.get('kind'))
    if kind not in ('assistant', 'compaction'):
        return None
    tokens = message.get('tokens') or {}
    cache = tokens.get('cache') or {}
    counters = _vector({**{k: tokens.get(k) for k in COUNTERS[:3]},
                        'cache_read': cache.get('read'), 'cache_write': cache.get('write')})
    created = message.get('time', {}).get('created', message.get('created'))
    if created is None:
        raise ValueError('native usage message is missing creation time')
    native_id = (message.get('native') or {}).get('responseId')
    return {'schema': SCHEMA, 'source': 'opencode', 'session_id': session_id,
            'message_id': _text(message.get('id', message.get('message_id')), 'message_id'),
            'timestamp': utc_time(created), 'kind': kind,
            'counters': counters, 'request_fingerprint': _fingerprint(native_id),
            **_task_hints(message)}


def deduplicate(records):
    """Deduplicate actual requests, including fork copies and late settlement.

    Message IDs are global in OpenCode. Native response fingerprints also
    catch the same response imported through a different native collector.
    Disagreeing known counters are an error, never an invitation to sum.
    """
    result, ids, requests = [], {}, {}
    for row in records:
        if row is None:
            continue
        source = row['source']
        key = ((source, row['message_id']) if source == 'opencode' else
               (source, row['session_id'], row['message_id']))
        fingerprint = row.get('request_fingerprint')
        index = ids.get(key)
        if index is None and fingerprint:
            index = requests.get(fingerprint)
        if index is None:
            index = len(result)
            result.append({**row, 'counters': _vector(row.get('counters'))})
        else:
            stored = result[index]
            incoming = _vector(row.get('counters'))
            for counter in COUNTERS:
                old, new = stored['counters'][counter], incoming[counter]
                if old is not None and new is not None and old != new:
                    raise ValueError('conflicting usage for the same native request')
                if old is None and new is not None:
                    stored['counters'][counter] = new
            if fingerprint and not stored.get('request_fingerprint'):
                stored['request_fingerprint'] = fingerprint
        ids[key] = index
        if fingerprint:
            requests[fingerprint] = index
    return result


def merge_collection(previous, current, *, start=None, cutoff=None):
    """Preserve observed failed calls even if the live API later retracts them."""
    start = utc_ms(start) if start is not None else None
    cutoff = utc_ms(cutoff) if cutoff is not None else None
    rows = deduplicate([*previous, *current])
    return [row for row in rows if row.get('timestamp') is None or (
        (start is None or utc_ms(row['timestamp']) >= start)
        and (cutoff is None or utc_ms(row['timestamp']) < cutoff))]


def native_get(endpoint):
    """Use native discovery/authentication, exclusively with HTTP GET."""
    if not endpoint.startswith('/api/'):
        raise ValueError('only native /api/ read endpoints are allowed')
    process = subprocess.run(['opencode', 'api', 'get', endpoint],
                             capture_output=True, text=True)
    if process.returncode:
        # stderr may contain sensitive native diagnostics; do not publish it.
        raise ValueError(f'OpenCode native GET failed (exit {process.returncode})')
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        raise ValueError('OpenCode native GET returned invalid JSON') from None


def _pages(endpoint, *, get, params=None, order='asc'):
    cursor, seen = None, set()
    while True:
        query = dict(params or {})
        query.update({'cursor': cursor} if cursor is not None else {'order': order})
        response = get(endpoint + '?' + urlencode(query))
        data = response.get('data')
        if not isinstance(data, list):
            raise ValueError('native paginated response must contain a data array')
        yield data
        next_cursor = (response.get('cursor') or {}).get('next')
        if not next_cursor:
            break
        if next_cursor in seen:
            raise ValueError('native pagination cursor did not advance')
        seen.add(next_cursor)
        cursor = next_cursor


def discover_sessions(root, *, get=None):
    """Recursively discover children; no session totals are counted as usage."""
    get = get or native_get
    todo, result = [root], []
    seen = set()
    while todo:
        sid = todo.pop(0)
        if sid in seen:
            continue
        seen.add(sid)
        result.append(sid)
        for page in _pages('/api/session', get=get, params={'parentID': sid, 'limit': 200}):
            todo.extend(row['id'] for row in page)
    return result


def collect_opencode(session_ids, *, get=None, page_size=200, start=None, cutoff=None):
    get = get or native_get
    rows = []
    if not isinstance(page_size, int) or not 1 <= page_size <= 1000:
        raise ValueError('page_size must be 1..1000')
    start = utc_ms(start) if start is not None else None
    cutoff = utc_ms(cutoff) if cutoff is not None else None
    if start is not None and cutoff is not None and cutoff <= start:
        raise ValueError('collection cutoff must be after start')
    for sid in dict.fromkeys(session_ids):
        endpoint = '/api/session/' + quote(sid, safe='') + '/message'
        for page in _pages(endpoint, get=get, params={'limit': page_size},
                           order='desc' if start is not None else 'asc'):
            older = False
            for message in page:
                time = utc_ms(message['time']['created'])
                if start is not None and time < start:
                    older = True
                    continue
                if cutoff is None or time < cutoff:
                    rows.append(project_opencode(sid, message))
            if older:
                break
    return deduplicate(rows)


@contextmanager
def _codex_stream(path):
    if Path(path).suffix != '.zst':
        with Path(path).open(encoding='utf-8') as stream:
            yield stream
        return
    # zstd is optional; no pip packages or model runtime are required.
    process = subprocess.Popen(['zstd', '-dc', '--', str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    try:
        yield process.stdout
    except BaseException:
        process.terminate()
        raise
    finally:
        process.stdout.close()
        status = process.wait()
    if status:
        raise ValueError('could not decode compressed Codex log')


def _codex_counters(values, includes_reasoning):
    get = lambda key: _count(values.get(key))
    input, cached = get('input_tokens'), get('cached_input_tokens')
    write = get('cache_write_input_tokens')
    output, reasoning = get('output_tokens'), get('reasoning_output_tokens')
    uncached = input - cached - write if None not in (input, cached, write) else None
    visible = (output - reasoning if None not in (output, reasoning) else None
               ) if includes_reasoning else output
    return _vector(dict(input=uncached, output=visible, reasoning=reasoning,
                        cache_read=cached, cache_write=write))


def collect_codex(paths):
    """Difference cumulative counters and skip unchanged rate-limit snapshots."""
    rows = []
    for path in paths:
        sid, previous = None, None
        with _codex_stream(path) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f'Codex log has invalid JSON at line {line_number}') from None
                payload = event.get('payload') or {}
                if event.get('type') == 'session_meta':
                    sid = payload.get('id', payload.get('session_id'))
                    continue
                if event.get('type') != 'event_msg' or payload.get('type') != 'token_count':
                    continue
                if not sid:
                    raise ValueError('Codex token_count requires preceding session_meta')
                info = payload.get('info') or {}
                cumulative = info.get('total_token_usage')
                last = info.get('last_token_usage')
                includes = info.get('last_usage_includes_reasoning', True)
                if not isinstance(includes, bool):
                    raise ValueError('Codex last_usage_includes_reasoning must be boolean')
                timestamp = utc_time(event['timestamp'])
                if cumulative is not None:
                    current = _codex_counters(cumulative, includes)
                    last_counts = _codex_counters(last or {}, includes)
                    if previous == current:
                        continue
                    if previous is None:
                        if last is None:
                            counts = current
                            timestamp = None
                        else:
                            carry = _vector({
                                key: current[key] - last_counts[key]
                                if current[key] is not None and last_counts[key] is not None else None
                                for key in COUNTERS})
                            if any(value for value in carry.values()):
                                rows.append(dict(
                                    schema=SCHEMA, source='codex', session_id=sid,
                                    message_id='carry-' + _fingerprint(
                                        json.dumps(carry, sort_keys=True)),
                                    timestamp=None, kind='cumulative_carry_in',
                                    counters=carry, request_fingerprint=None))
                            counts = last_counts
                    else:
                        difference = {key: current[key] - previous[key]
                                      if current[key] is not None and previous[key] is not None
                                      else last_counts[key] for key in COUNTERS}
                        if any(value is not None and value < 0 for value in difference.values()):
                            raise ValueError('Codex cumulative usage counters decreased')
                        counts = _vector(difference)
                    # Preserve each known component across partial snapshots.
                    # Resetting the whole baseline would count the resumed
                    # cumulative prefix twice when that snapshot recovers.
                    previous = current
                    if not any(value for value in counts.values()):
                        if all(value is not None for value in counts.values()):
                            continue
                elif last is not None:
                    counts = _codex_counters(last, includes)
                else:
                    counts = _vector()
                identity = json.dumps([timestamp, cumulative, last], sort_keys=True)
                rows.append(dict(schema=SCHEMA, source='codex', session_id=sid,
                                 message_id=_fingerprint(identity), timestamp=timestamp,
                                 kind='assistant' if timestamp is not None else 'cumulative_carry_in',
                                 counters=counts,
                                 request_fingerprint=_fingerprint(payload.get('response_id'))))
    return deduplicate(rows)


def _historical(paths):
    """Freeze input values and select disjoint latest cumulative windows."""
    snapshots = []
    for path in paths:
        data_bytes = Path(path).read_bytes()
        data = json.loads(data_bytes)
        window = data['window']
        start = utc_ms(window.get('start_ms', window.get('start_utc')))
        cutoff = utc_ms(window.get('cutoff_ms', window.get('cutoff_utc')))
        if cutoff <= start:
            raise ValueError('historical window must have positive duration')
        models = {}
        for model, vector in data['models'].items():
            _label(model, 'historical model')
            counters = _vector(vector)
            if any(value is None for value in counters.values()):
                raise ValueError('historical counters must be complete')
            enriched = _enrich(counters)
            for key in ('all_processed_tokens', 'uncached_work_tokens'):
                if key in vector and vector[key] != enriched[key]:
                    raise ValueError('historical derived total disagrees with counters')
            models[model] = {**enriched, 'usage_records': _count(vector.get('usage_records', 0)),
                             'missing_usage_records': _count(vector.get('missing_usage_records', 0))}
        stages = []
        for row in data.get('stages', []):
            # Historical "stage" is a work phase, not a pipeline stage.
            if row.get('stage') not in _OLD_PHASES:
                continue
            counters = _vector(row)
            if any(value is None for value in counters.values()):
                raise ValueError('historical phase counters must be complete')
            stages.append({'model': _label(row['model'], 'historical model'),
                           'phase': _OLD_PHASES[row['stage']], **_enrich(counters)})
        snapshots.append(dict(source=Path(path).name, sha256=hashlib.sha256(data_bytes).hexdigest(),
                              start_ms=start, cutoff_ms=cutoff, models=models, phases=stages,
                              initial_phase_end_ms=window.get('initial_phase_end_ms')))
    snapshots.sort(key=lambda s: (s['start_ms'], s['cutoff_ms']))
    latest = {}
    for snapshot in snapshots:
        start = snapshot['start_ms']
        if start in latest and latest[start]['cutoff_ms'] == snapshot['cutoff_ms']:
            if latest[start]['models'] != snapshot['models']:
                raise ValueError('conflicting historical snapshots at the same cutoff')
        latest[start] = snapshot
    selected = sorted(latest.values(), key=lambda s: s['start_ms'])
    if any(a['cutoff_ms'] > b['start_ms'] for a, b in zip(selected, selected[1:])):
        raise ValueError('overlapping historical scopes require an explicit nonoverlapping snapshot')
    return snapshots, selected


def _bucket():
    return {'total': _zero(), 'records': 0, 'missing_usage_records': 0,
            'missing_counters': dict.fromkeys(COUNTERS, 0), 'sources': set(),
            'source_totals': defaultdict(_zero)}


def _accumulate(bucket, counters, source, *, records=1, missing=None):
    _add(bucket['total'], counters)
    _add(bucket['source_totals'][source], counters)
    bucket['sources'].add(source)
    bucket['records'] += records
    bucket['missing_usage_records'] += (any(v is None for v in counters.values())
                                       if missing is None else missing)
    for key in COUNTERS:
        bucket['missing_counters'][key] += counters.get(key) is None


def _public_bucket(bucket):
    return {'total': _enrich(bucket['total']), 'records': bucket['records'],
            'missing_usage_records': bucket['missing_usage_records'],
            'missing_counters': dict(bucket['missing_counters']),
            'source': '+'.join(sorted(bucket['sources'])) or '账本',
            'source_totals': {key: _enrich(value)
                              for key, value in sorted(bucket['source_totals'].items())}}


def _experiments(rows):
    result = {}
    for row in rows:
        key = (_label(row['model'], 'experiment model'), row['stage'], row.get('tool'))
        if not _known_stage(key[1]):
            raise ValueError('experiment stage must be known')
        if key in result:
            raise ValueError('ambiguous experiments: supply one matching before/after pair per model/stage/tool')
        for field in ('before_ms', 'after_ms'):
            value = row[field]
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('experiment latencies must be positive finite numbers')
        result[key] = {field: row[field] for field in ('before_ms', 'after_ms')}
    return result


def build_report(records, segments, *, run_start, cutoff=None, history=(), experiments=(),
                 models=(), gap_assignments=()):
    """Build four public tables; attribution uses creation time, never completion.

    All sums are recorded lower bounds with missing counts alongside them.
    Gap assignments are private objects keyed by session_id, optional start/end,
    and model/phase/stage/tool. They estimate task attribution, not native counts.
    Otherwise the collector's coarse task hints are used only for the gap.
    """
    run_start, cutoff = utc_ms(run_start), utc_ms(cutoff if cutoff is not None else utc_time())
    if cutoff <= run_start:
        raise ValueError('cutoff must be after run_start')
    spans = [span for span in segment_intervals(segments, cutoff=cutoff)
             if span['start_ms'] < cutoff]
    for span in spans:
        if span['end_ms'] > cutoff:
            span.update(end_ms=cutoff, closed=False)
    snapshots, selected = _historical(history)
    if any(s['cutoff_ms'] > run_start for s in selected):
        raise ValueError('historical snapshot overlaps the current run')
    history_end = max((s['cutoff_ms'] for s in selected), default=run_start)
    history_start = min((s['start_ms'] for s in selected), default=run_start)
    gap_ranges = [(a['cutoff_ms'], b['start_ms'])
                  for a, b in zip(selected, selected[1:]) if a['cutoff_ms'] < b['start_ms']]
    if history_end < run_start:
        gap_ranges.append((history_end, run_start))
    gap_start = min((start for start, end in gap_ranges), default=run_start)
    records = list(records)
    rows = deduplicate(records)
    pairs = _experiments(experiments)
    lifecycle = defaultdict(_bucket)
    phases = defaultdict(lambda: defaultdict(_zero))
    stage_buckets, method_buckets = defaultdict(_bucket), defaultdict(_bucket)
    historical = _zero()
    current, unattributed, gap = _bucket(), _bucket(), _bucket()
    historical_onboarding, historical_model_totals = {}, defaultdict(int)
    historical_times = {}
    for snapshot in selected:
        older_with_phases = [s for s in snapshots if s['start_ms'] == snapshot['start_ms']
                             and s['cutoff_ms'] <= snapshot['cutoff_ms'] and s['phases']]
        phase_snapshot = max(older_with_phases, key=lambda s: s['cutoff_ms'], default=None)
        used = defaultdict(_zero)
        if phase_snapshot:
            for row in phase_snapshot['phases']:
                model, phase = row['model'], row['phase']
                _add(phases[model][phase], row)
                _add(used[model], row)
                if phase == 'onboarding':
                    historical_onboarding[model] = (
                        historical_onboarding.get(model, 0) + row['all_processed_tokens'])
                    historical_times[model] = (phase_snapshot['start_ms'],
                                               phase_snapshot['initial_phase_end_ms'])
        for model, counts in snapshot['models'].items():
            _add(historical, counts)
            historical_model_totals[model] += counts['all_processed_tokens']
            residual = _difference(counts, used[model])
            _add(phases[model]['unknown'], residual)
            _accumulate(lifecycle[model], counts, '账本', records=counts['usage_records'],
                        missing=counts['missing_usage_records'])
    by_session = defaultdict(list)
    for span in spans:
        by_session[span['session_id']].append(span)
    excluded_historical, excluded_after_cutoff, unknown_time = 0, 0, 0
    for row in rows:
        counters = _vector(row.get('counters'))
        time = utc_ms(row['timestamp']) if row.get('timestamp') is not None else None
        if time is None:
            # A resumed Codex cumulative prefix cannot safely be placed after
            # a frozen historical snapshot. Publish it separately, never twice.
            unknown_time += 1
            continue
        if time >= cutoff:
            excluded_after_cutoff += 1
            continue
        if time < history_start or any(s['start_ms'] <= time < s['cutoff_ms'] for s in selected):
            excluded_historical += 1
            continue
        if time < run_start:
            matching = [a for a in gap_assignments if a['session_id'] == row['session_id']
                        and utc_ms(a.get('start', gap_start)) <= time
                        < utc_ms(a.get('end', run_start))]
            if len(matching) > 1:
                raise ValueError('ambiguous gap task assignments')
            hints = row.get('task_models', [])
            phase_hints = row.get('task_phases', [])
            assignment = matching[0] if matching else dict(
                model=hints[0] if len(hints) == 1 else 'shared',
                phase=phase_hints[0] if len(phase_hints) == 1 else 'unknown')
            model = _label(assignment['model'], 'gap model')
            phase = assignment['phase']
            if phase not in (*PHASES, 'unknown'):
                raise ValueError('invalid gap phase')
            if phase == 'framework':
                model = 'shared'
            stage, tool, source = assignment.get('stage'), assignment.get('tool'), '估算'
            _accumulate(gap, counters, source)
        else:
            matches = [span for span in by_session[row['session_id']]
                       if span['start_ms'] <= time
                       and (span['end_ms'] is None or time < span['end_ms'])]
            if len(matches) > 1:
                raise ValueError('overlapping work segments')
            assignment = matches[0] if matches else {}
            model, phase = assignment.get('model', 'unattributed'), assignment.get('phase', 'unknown')
            stage, tool, source = assignment.get('stage'), assignment.get('tool'), '账本'
            _accumulate(current, counters, source)
            if not matches:
                _accumulate(unattributed, counters, source)
        _accumulate(lifecycle[model], counters, source)
        _add(phases[model][phase], counters)
        if phase in ('optimization', 'migration'):
            if stage is not None and not _known_stage(stage):
                raise ValueError('gap stage must be known')
            _accumulate(stage_buckets[(model, stage or 'unknown', tool)], counters, source)
        if phase == 'migration':
            if tool is not None:
                _label(tool, 'method')
            _accumulate(method_buckets[(tool or 'unknown', model, stage or 'unknown')],
                        counters, source)
    all_models = list(dict.fromkeys([*DEFAULT_MODELS, *models, *sorted(lifecycle),
                                    *(span['model'] for span in spans), 'shared']))
    public_lifecycle, total = [], _zero()
    for model in all_models:
        _label(model, 'model')
        bucket = lifecycle[model]
        _add(total, bucket['total'])
        public_lifecycle.append(dict(
            model=model, **_public_bucket(bucket),
            phases={phase: _enrich(phases[model][phase])
                    for phase in (*PHASES, 'unknown')}))
    public_stage = []
    for (model, stage, tool), bucket in sorted(stage_buckets.items(),
                                               key=lambda item: str(item[0])):
        experiment = pairs.get((model, stage, tool))
        saved = experiment['before_ms'] - experiment['after_ms'] if experiment else None
        tokens = sum(bucket['total'].values())
        public_stage.append(dict(
            model=model, stage=stage, tool=tool, **_public_bucket(bucket),
            before_ms=experiment['before_ms'] if experiment else None,
            after_ms=experiment['after_ms'] if experiment else None, saved_ms=saved,
            ms_saved_per_million_tokens=saved * 1_000_000 / tokens
            if saved is not None and tokens and not bucket['missing_usage_records'] else None))
    public_method = [dict(tool=tool, model=model, stage=stage, **_public_bucket(bucket))
                     for (tool, model, stage), bucket in sorted(method_buckets.items(),
                                                               key=lambda item: str(item[0]))]
    trend = []
    for model in all_models:
        if model in ('shared', 'unattributed'):
            continue
        onboard = [s for s in spans if s['model'] == model and s['phase'] == 'onboarding']
        start = min((s['start_ms'] for s in onboard), default=None)
        end = (max(s['end_ms'] for s in onboard) if onboard and all(s['closed'] for s in onboard)
               else None)
        historical_cost = historical_onboarding.get(model)
        known_cost = sum(phases[model]['onboarding'].values())
        trend.append(dict(
            model=model, onboarding_tokens=known_cost
            if historical_cost is not None or onboard else None,
            historical_onboarding_tokens=historical_cost,
            historical_total_tokens=historical_model_totals.get(model),
            historical_completion_is_proxy=historical_cost is not None,
            historical_start_utc=utc_time(historical_times[model][0])
            if model in historical_times else None,
            historical_end_utc=utc_time(historical_times[model][1])
            if model in historical_times and historical_times[model][1] is not None else None,
            start_utc=utc_time(start) if start is not None else None,
            completed_utc=utc_time(end) if end is not None else None,
            elapsed_seconds=(end - start) / 1000 if end is not None else None,
            source='+'.join(sorted(lifecycle[model]['sources'])) or '账本'))
    trend.sort(key=lambda row: (row['historical_start_utc'] or row['start_utc']
                               or '9999', all_models.index(row['model'])))
    denominator, numerator = sum(current['total'].values()), sum(unattributed['total'].values())
    fraction = numerator / denominator if denominator else None
    return dict(
        schema=SCHEMA, window=dict(run_start_utc=utc_time(run_start), cutoff_utc=utc_time(cutoff),
                                  creation_time_rule='start <= message.created < end'),
        current_run={**_public_bucket(current),
                     'unattributed': _public_bucket(unattributed),
                     'unattributed_fraction': fraction,
                     'passes_unattributed_gate': fraction is not None and fraction < .05,
                     'gate_uses_known_recorded_counters': True},
        gap={**_public_bucket(gap), 'source': '估算',
             'start_utc': utc_time(gap_start), 'end_utc': utc_time(run_start),
             'ranges': [{'start_utc': utc_time(start), 'end_utc': utc_time(end)}
                        for start, end in gap_ranges],
             'counter_basis': 'native counters; task attribution estimated'},
        historical_total=_enrich(historical), historical_snapshots=snapshots,
        historical_attribution_is_estimated=True,
        grand_total=_enrich(total), model_lifecycle=public_lifecycle,
        model_stage=public_stage, method_model=public_method, onboarding_trend=trend,
        deduplicated_records=len([r for r in records if r is not None]) - len(rows),
        excluded_historical_overlap_records=excluded_historical,
        excluded_after_cutoff_records=excluded_after_cutoff,
        unresolved_timestamp_records=unknown_time,
        unresolved_timestamp_total=_enrich({
            key: sum((r.get('counters') or {}).get(key) or 0 for r in rows
                     if r.get('timestamp') is None) for key in COUNTERS}),
        note='总处理量包含反复读取的缓存，不等于新增 token，也不能直接换算成费用。',
        missing_usage_policy='缺失 usage 单独计数；计数和总量仅为已记录部分的下界，不把缺失当作 0。',
        privacy='仅发布聚合数据；原生消息、提示词、工具体和逐消息账本不发布。')


def _atomic_write(path, text, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv(rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_report(report, output):
    """Publish only whitelisted aggregate columns, in JSON/CSV/Chinese Markdown."""
    prefix = str(output)
    _atomic_write(prefix + '.json', json.dumps(report, ensure_ascii=False, indent=2,
                                              allow_nan=False) + '\n')
    for name, rows in (('model-lifecycle', report['model_lifecycle']),
                       ('model-stage', report['model_stage']),
                       ('method-model', report['method_model'])):
        flattened = []
        for row in rows:
            fields = ('model', 'source') if name == 'model-lifecycle' else (
                ('model', 'stage', 'tool', 'source') if name == 'model-stage' else
                ('tool', 'model', 'stage', 'source'))
            flat = {key: row[key] for key in fields}
            if name == 'model-lifecycle':
                flat.update({phase: row['phases'][phase]['all_processed_tokens']
                             for phase in (*PHASES, 'unknown')})
            flat.update(row['total'])
            flat['missing_usage_records'] = row['missing_usage_records']
            if name == 'model-stage':
                flat.update({key: row[key] for key in (
                    'before_ms', 'after_ms', 'saved_ms', 'ms_saved_per_million_tokens')})
            flattened.append(flat)
        fields = list(flattened[0]) if flattened else (
            (['model', 'stage', 'tool'] if name == 'model-stage' else ['tool', 'model', 'stage'])
            + ['source', *COUNTERS, 'uncached_work_tokens', 'all_processed_tokens',
               'missing_usage_records']
            + (['before_ms', 'after_ms', 'saved_ms', 'ms_saved_per_million_tokens']
               if name == 'model-stage' else []))
        _atomic_write(prefix + '-' + name + '.csv', _csv(flattened, fields))
    trend_fields = list(report['onboarding_trend'][0]) if report['onboarding_trend'] else ['model']
    _atomic_write(prefix + '-onboarding-trend.csv',
                  _csv(report['onboarding_trend'], trend_fields))
    lines = ['# VLA 开发 Token 统计', '',
             f"统计截点：{report['window']['cutoff_utc']}。", '',
             report['note'], '', report['missing_usage_policy'], '',
             '来源“账本”表示工作段或已封存历史统计；“估算”表示空档的任务归属估算。',
             '原生计数实记；历史模型归属及空档任务归属属于估算。共有工作独立记 shared。',
             '历史阶段无法确认的部分放在 unknown，不追溯伪造阶段。', '',
             '## 模型全过程', '',
             '| 模型 | 来源 | 接入 | 迁移 | 优化 | 排查 | 测量 | 框架 | 阶段未知 | 总处理量 | usage 缺失 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['model_lifecycle']:
        cells = [row['model'], row['source'],
                 *(row['phases'][p]['all_processed_tokens'] for p in (*PHASES, 'unknown')),
                 row['total']['all_processed_tokens'], row['missing_usage_records']]
        lines.append('| ' + ' | '.join(str(x) for x in cells) + ' |')
    lines += ['', '## 模型 × 游', '',
              '| 模型 | 游 | 方法 | 来源 | 总处理量 | 节省 ms | 每百万 token 节省 ms |',
              '|---|---|---|---|---:|---:|---:|']
    display = lambda value: '未知' if value is None else (
        f'{value:.1f}' if isinstance(value, float) else str(value))
    for row in report['model_stage']:
        lines.append('| ' + ' | '.join(display(x) for x in (
            row['model'], row['stage'], row['tool'], row['source'],
            row['total']['all_processed_tokens'], row['saved_ms'],
            row['ms_saved_per_million_tokens'])) + ' |')
    lines += ['', '## 方法 × 模型迁移成本', '',
              '| 方法 | 模型 | 游 | 来源 | 总处理量 | usage 缺失 |',
              '|---|---|---|---|---:|---:|']
    for row in report['method_model']:
        lines.append('| ' + ' | '.join(display(x) for x in (
            row['tool'], row['model'], row['stage'], row['source'],
            row['total']['all_processed_tokens'], row['missing_usage_records'])) + ' |')
    lines += ['', '## 接入成本随时间变化', '',
              '历史 initial_adaptation 是时间窗口代理，未当作逐模型验收时间。'
              'DM0.5 历史仅有全过程总数，其接入成本保留未知。', '',
              '| 模型 | 来源 | 接入 token | 历史接入 token | 历史全过程 token | 开始 | 完成 | 耗时 s |',
              '|---|---|---:|---:|---:|---|---|---:|']
    for row in report['onboarding_trend']:
        lines.append('| ' + ' | '.join(display(row[key]) for key in (
            'model', 'source', 'onboarding_tokens', 'historical_onboarding_tokens',
            'historical_total_tokens', 'start_utc', 'completed_utc', 'elapsed_seconds')) + ' |')
    current = report['current_run']
    ratio = current['unattributed_fraction']
    lines += ['', f"本轮 unattributed：{display(ratio * 100 if ratio is not None else None)}%；"
              f"usage 缺失 {current['missing_usage_records']} 条。", '',
              '五项原生计数、非缓存合计、来源拆分及历史快照校验值见同名 JSON 和 CSV。', '']
    _atomic_write(prefix + '.zh-CN.md', '\n'.join(lines))


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('start', 'end'):
        sub = commands.add_parser(command, help=f'{command} an exclusive work segment')
        sub.add_argument('--segments', default=SEGMENTS)
        sub.add_argument('--session', default=os.environ.get('OPENCODE_SESSION_ID'))
        sub.add_argument('--time')
        sub.add_argument('--summary', required=command == 'start')
        if command == 'start':
            sub.add_argument('--model', required=True)
            sub.add_argument('--phase', required=True, choices=PHASES)
            sub.add_argument('--item', required=True)
            sub.add_argument('--stage')
            sub.add_argument('--tool')
        else:
            sub.add_argument('--expected-model')
            sub.add_argument('--expected-phase', choices=PHASES)
            sub.add_argument('--expected-segment-id')
    for command in ('collect-opencode', 'collect-codex'):
        sub = commands.add_parser(command)
        sub.add_argument('--output', required=True)
        sub.add_argument('--start', help='Inclusive native message creation time (UTC)')
        sub.add_argument('--cutoff', help='Exclusive native message creation time (UTC)')
        if command == 'collect-opencode':
            sub.add_argument('--session', action='append', default=[])
            sub.add_argument('--root', help='Include this root and all descendants')
        else:
            sub.add_argument('--log', action='append', required=True, dest='logs')
    sub = commands.add_parser('report')
    sub.add_argument('--records', action='append', default=[])
    sub.add_argument('--segments', default=SEGMENTS)
    sub.add_argument('--history', action='append', default=[])
    sub.add_argument('--run-start', required=True)
    sub.add_argument('--cutoff')
    sub.add_argument('--experiments', help='JSON list: model/stage/tool/before_ms/after_ms')
    sub.add_argument('--gap-assignments', help='Private JSON list: session_id/model/phase/start/end')
    sub.add_argument('--model', action='append', default=[])
    sub.add_argument('--output', required=True, help='Public output prefix')
    sub.add_argument('--require-attribution', action='store_true',
                     help='Return 2 unless current known unattributed fraction is below 5%%')
    args = parser.parse_args(argv)
    try:
        if args.command in ('start', 'end'):
            if not args.session:
                raise ValueError('--session or OPENCODE_SESSION_ID is required')
            log = SegmentLog(args.segments)
            if args.command == 'start':
                row = log.start(args.session, model=args.model, phase=args.phase,
                                item=args.item, summary=args.summary, stage=args.stage,
                                tool=args.tool, time=args.time)
            else:
                row = log.end(args.session, summary=args.summary, time=args.time,
                              expected_model=args.expected_model,
                              expected_phase=args.expected_phase,
                              expected_segment_id=args.expected_segment_id)
            print(json.dumps(row, ensure_ascii=False))
        elif args.command.startswith('collect-'):
            if 'docs' in Path(args.output).resolve().parts or '.git' in Path(args.output).resolve().parts:
                raise ValueError('per-message usage ledgers must stay private, outside docs and .git')
            collection_cutoff = utc_time(args.cutoff)
            previous = []
            if Path(args.output).exists():
                raw = Path(args.output).read_bytes()
                previous = _read_jsonl(io.StringIO(raw.decode('utf-8')))
                metadata = Path(str(args.output) + '.collection.json')
                if metadata.exists():
                    manifest = _load(metadata)
                    digest = manifest.get('records_sha256')
                    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
                        raise ValueError('usage ledger digest differs from its collection manifest')
            if args.command == 'collect-opencode':
                sessions = args.session + (discover_sessions(args.root) if args.root else [])
                if not sessions:
                    raise ValueError('--session or --root is required')
                rows = collect_opencode(sessions, start=args.start, cutoff=collection_cutoff)
            else:
                rows = collect_codex(args.logs)
            rows = merge_collection(previous, rows, start=args.start, cutoff=collection_cutoff)
            ledger_text = ''.join(_json_line(row) for row in rows)
            _atomic_write(args.output, ledger_text, private=True)
            manifest = dict(
                schema=SCHEMA, source=args.command.removeprefix('collect-'),
                start_utc=utc_time(args.start) if args.start is not None else None,
                cutoff_utc=collection_cutoff, records=len(rows),
                records_sha256=hashlib.sha256(ledger_text.encode('utf-8')).hexdigest(),
                missing_usage_records=sum(any(value is None for value in row['counters'].values())
                                          for row in rows))
            _atomic_write(str(args.output) + '.collection.json',
                          json.dumps(manifest, indent=2) + '\n', private=True)
            print(json.dumps(manifest))
        else:
            records, manifests = [], []
            for path in args.records:
                raw = Path(path).read_bytes()
                records.extend(_read_jsonl(io.StringIO(raw.decode('utf-8'))))
                metadata = Path(str(path) + '.collection.json')
                if metadata.exists():
                    manifest = _load(metadata)
                    digest = manifest.get('records_sha256')
                    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
                        raise ValueError('usage ledger digest differs from its collection manifest')
                    manifests.append(manifest)
            collected_cutoff = min((utc_ms(m['cutoff_utc']) for m in manifests), default=None)
            if args.cutoff is not None and collected_cutoff is not None:
                if utc_ms(args.cutoff) > collected_cutoff:
                    raise ValueError('report cutoff exceeds the native collection cutoff')
            effective_cutoff = args.cutoff if args.cutoff is not None else collected_cutoff
            report = build_report(
                records, SegmentLog(args.segments).records(), run_start=args.run_start,
                cutoff=effective_cutoff, history=args.history,
                experiments=_load(args.experiments) if args.experiments else (),
                gap_assignments=_load(args.gap_assignments) if args.gap_assignments else (),
                models=args.model)
            report['collections'] = [
                {key: manifest[key] for key in (
                    'start_utc', 'cutoff_utc', 'records', 'records_sha256', 'missing_usage_records')
                 if key in manifest}
                for manifest in manifests]
            write_report(report, args.output)
            print(json.dumps({'grand_total': report['grand_total'],
                              'current_run': report['current_run']}, ensure_ascii=False))
            if args.require_attribution and not report['current_run']['passes_unattributed_gate']:
                return 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Parsing exceptions must not quote native payload fragments.
        if isinstance(error, json.JSONDecodeError):
            print('error: invalid JSON input', file=sys.stderr)
        elif isinstance(error, KeyError):
            print('error: input is missing a required metadata field', file=sys.stderr)
        else:
            print(f'error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
