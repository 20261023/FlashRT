"""Append-only decision / experience log (JSONL) recording who led each step.

Every record names its actor: human, agent, human+agent or search. Records
are validated per kind before they are written, the file is only ever opened
for append (one JSON line, flush + fsync, exclusive lock), and a file that
already holds an invalid line is refused rather than extended. Experience
records are conditional context, never universal rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from uuid import UUID, uuid4

from .stages import STAGES


SCHEMA = 'flash_rt.autotune.decision.v1'
ACTORS = ('human', 'agent', 'human+agent', 'search')
KINDS = ('diagnosis', 'decision', 'experience', 'note')
CONFIDENCE = ('low', 'medium', 'high')
PROVENANCE = ('source_fact', 'mechanistic_inference', 'measured_here')
ACTIONS = ('select_stage', 'select_tool', 'run_campaign', 'keep', 'accept', 'reject',
           'redirect', 'stop', 'merge')
DEFAULT_INVALIDATION = ('model/checkpoint', 'source/runtime revision', 'hardware',
                        'shape/precision', 'input distribution', 'timing boundary',
                        'task quality protocol')
HUMAN_ACTORS = ('human', 'human+agent')
HUMAN_ACTIONS = ('accept', 'merge')

_ID = re.compile(r'^[a-z0-9][a-z0-9_.-]*$')
_COMMON = ('schema', 'record_id', 'recorded_at', 'kind', 'actor', 'summary')
_OPTIONAL = ('model', 'stage', 'campaign', 'evidence_refs')

# kind -> {field: (validator, required, default)}
_TEXT, _TEXTS, _ENUM, _ID_FIELD, _IDS, _OBJECT = range(6)
_KIND_FIELDS = {
    'diagnosis': {
        'symptom': (_TEXT, True, None), 'hypothesis': (_TEXT, True, None),
        'alternatives': (_TEXTS, False, ()), 'next_observation': (_TEXT, True, None),
        'candidate_tools': (_IDS, False, ()), 'invalid_when': (_TEXTS, False, ()),
        'confidence': ((_ENUM, CONFIDENCE), True, None),
        'provenance': ((_ENUM, PROVENANCE), True, None)},
    'decision': {
        'action': ((_ENUM, ACTIONS), True, None), 'rationale': (_TEXT, True, None),
        'tool': (_ID_FIELD, False, None), 'candidate_ids': (_TEXTS, False, ()),
        'redirected_from': (_TEXT, False, None)},
    'experience': {
        'context': (_OBJECT, True, None), 'outcome': (_TEXT, True, None),
        'invalidation': (_TEXTS, False, DEFAULT_INVALIDATION)},
    'note': {'text': (_TEXT, False, None)},
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _texts(value, name):
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list of nonempty strings')
    return [_text(item, f'{name}[{index}]') for index, item in enumerate(value)]


def _identifier(value, name):
    if not isinstance(value, str) or not _ID.match(value):
        raise ValueError(f'{name} must match {_ID.pattern}, got {value!r}')
    return value


def _json_object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a JSON object')
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must contain only finite JSON values: {error}') from None


def _check(kind, value, name):
    if kind == _TEXT:
        return _text(value, name)
    if kind == _TEXTS:
        return _texts(value, name)
    if kind == _ID_FIELD:
        return _identifier(value, name)
    if kind == _IDS:
        return [_identifier(item, f'{name}[{index}]') for index, item in enumerate(_texts(value, name))]
    if kind == _OBJECT:
        return _json_object(value, name)
    _, allowed = kind
    if value not in allowed:
        raise ValueError(f'{name} must be one of {list(allowed)}, got {value!r}')
    return value


def _actor(actor):
    if actor is None:
        raise ValueError(f'actor is required; one of {list(ACTORS)}')
    if not isinstance(actor, str) or actor not in ACTORS:
        raise ValueError(f'actor must be one of {list(ACTORS)}, got {actor!r}')
    return actor


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('recorded_at must be an ISO-8601 UTC string')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f'recorded_at is not ISO-8601: {value!r}') from None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f'recorded_at must be UTC (+00:00): {value!r}')
    return parsed


def _cross_checks(record):
    action = record.get('action')
    if action == 'select_tool' and 'tool' not in record:
        raise ValueError('decision action select_tool requires tool')
    if action == 'select_stage' and 'stage' not in record:
        raise ValueError('decision action select_stage requires stage')
    if action == 'redirect' and 'redirected_from' not in record:
        raise ValueError('decision action redirect requires redirected_from')
    if 'redirected_from' in record and record['actor'] not in HUMAN_ACTORS:
        raise ValueError(f'redirected_from records a human redirect; actor must be one of {list(HUMAN_ACTORS)}')
    if action in HUMAN_ACTIONS and record['actor'] not in HUMAN_ACTORS:
        raise ValueError(f'decision action {action} is a human checkpoint; actor must be one of '
                         f'{list(HUMAN_ACTORS)}, got {record["actor"]!r}')


def build_fields(kind, *, actor, summary, **fields):
    """Validate user-supplied fields and return the record body (no id/time)."""
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f'kind must be one of {list(KINDS)}, got {kind!r}')
    reserved = sorted(set(fields) & {'schema', 'record_id', 'recorded_at', 'kind', 'actor', 'summary'})
    if reserved:
        raise ValueError(f'Fields {reserved} are assigned by the log, not by the caller')
    specs = _KIND_FIELDS[kind]
    unknown = sorted(set(fields) - set(specs) - set(_OPTIONAL))
    if unknown:
        raise ValueError(f'Unknown {kind} fields: {unknown}; allowed: '
                         f'{sorted(set(specs) | set(_OPTIONAL))}')
    body = {'kind': kind, 'actor': _actor(actor), 'summary': _text(summary, 'summary')}
    for name in ('model', 'campaign'):
        if fields.get(name) is not None:
            body[name] = (_identifier if name == 'model' else _text)(fields[name], name)
    if fields.get('stage') is not None:
        if not isinstance(fields['stage'], str) or fields['stage'] not in STAGES:
            raise ValueError(f'stage must be a known stage id, got {fields["stage"]!r}')
        body['stage'] = fields['stage']
    body['evidence_refs'] = _texts(fields.get('evidence_refs', []), 'evidence_refs')
    for name, (validator, required, default) in specs.items():
        value = fields.get(name)
        if value is None:
            if required:
                raise ValueError(f'{kind} requires {name}')
            if default is None:
                continue
            value = list(default) if isinstance(default, tuple) else default
        body[name] = _check(validator, value, f'{kind}.{name}')
    _cross_checks(body)
    return body


def validate_record(record):
    """Validate one stored record exactly as written; return it unchanged."""
    if not isinstance(record, dict):
        raise ValueError('decision record must be a JSON object')
    if record.get('schema') != SCHEMA:
        raise ValueError(f'schema must be {SCHEMA!r}, got {record.get("schema")!r}')
    record_id = record.get('record_id')
    if not isinstance(record_id, str) or len(record_id) != 32:
        raise ValueError('record_id must be a uuid4 hex string')
    try:
        if UUID(hex=record_id).hex != record_id:
            raise ValueError
    except ValueError:
        raise ValueError(f'record_id is not a uuid hex: {record_id!r}') from None
    _timestamp(record.get('recorded_at'))
    fields = {key: value for key, value in record.items() if key not in _COMMON}
    body = build_fields(record.get('kind'), actor=record.get('actor'),
                        summary=record.get('summary'), **fields)
    expected = {'schema': SCHEMA, 'record_id': record_id, 'recorded_at': record['recorded_at'], **body}
    if expected != record:
        raise ValueError('decision record is not in canonical form (missing defaults or retyped values)')
    return record


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _escape(text):
    return ' '.join(str(text).split()).replace('|', '\\|')


# Line breaks that str.splitlines() honours but json.dumps(ensure_ascii=False)
# leaves raw; they can only occur inside JSON strings, so escaping is lossless.
_RAW_LINE_BREAKS = {'\u2028': '\\u2028', '\u2029': '\\u2029', '\x85': '\\u0085'}


def _json_line(record):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
    for raw, escaped in _RAW_LINE_BREAKS.items():
        line = line.replace(raw, escaped)
    return line + '\n'


class DecisionLog:
    """Append-only JSONL decision log; one writer at a time (flock)."""

    def __init__(self, path, *, clock=None):
        self.path = Path(path)
        self._clock = clock or _now

    def _read(self):
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError(f'{self.path}: last line is incomplete (no trailing newline); refusing to use')
        records, seen = [], set()
        # Split on b'\n' only: str.splitlines() also breaks on U+2028/U+2029/U+0085,
        # which may legitimately appear in (older, ensure_ascii=False) records.
        for number, raw in enumerate(data.split(b'\n')[:-1], start=1):
            try:
                record = validate_record(json.loads(raw.decode('utf-8')))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'{self.path}:{number}: invalid decision record: {error}') from None
            if record['record_id'] in seen:
                raise ValueError(f'{self.path}:{number}: duplicate record_id {record["record_id"]}')
            seen.add(record['record_id'])
            records.append(record)
        return records

    def append(self, kind, *, actor, summary, **fields):
        body = build_fields(kind, actor=actor, summary=summary, **fields)
        record = {'schema': SCHEMA, 'record_id': uuid4().hex, 'recorded_at': self._clock(), **body}
        validate_record(record)
        line = _json_line(record)
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        with self.path.open('a', encoding='utf-8') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                self._read()
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        if created:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return record

    def records(self, kind=None):
        if kind is not None and kind not in KINDS:
            raise ValueError(f'kind must be one of {list(KINDS)}, got {kind!r}')
        return [record for record in self._read() if kind is None or record['kind'] == kind]

    def summary(self):
        records = self._read()
        by_actor_kind = {actor: {kind: 0 for kind in KINDS} for actor in ACTORS}
        for record in records:
            by_actor_kind[record['actor']][record['kind']] += 1
        return {'path': str(self.path), 'total': len(records),
                'by_actor': {actor: sum(counts.values()) for actor, counts in by_actor_kind.items()},
                'by_kind': {kind: sum(by_actor_kind[actor][kind] for actor in ACTORS) for kind in KINDS},
                'by_actor_kind': by_actor_kind}

    def markdown_table(self):
        """Chronological step table (file order) with the actor of every step."""
        lines = ['| # | time | kind | actor | stage | summary |', '|---|---|---|---|---|---|']
        for index, record in enumerate(self._read(), start=1):
            time = _timestamp(record['recorded_at']).strftime('%Y-%m-%d %H:%M:%SZ')
            lines.append(f'| {index} | {time} | {record["kind"]} | {record["actor"]} | '
                         f'{record.get("stage", "-")} | {_escape(_describe(record))} |')
        return '\n'.join(lines) + '\n'

    def markdown_summary(self):
        counts = self.summary()['by_actor_kind']
        lines = ['| actor | ' + ' | '.join(KINDS) + ' | total |',
                 '|---|' + '---|' * (len(KINDS) + 1)]
        for actor in ACTORS:
            row = [counts[actor][kind] for kind in KINDS]
            lines.append(f'| {actor} | ' + ' | '.join(map(str, row)) + f' | {sum(row)} |')
        return '\n'.join(lines) + '\n'


def _describe(record):
    text = record['summary']
    if record['kind'] == 'decision':
        prefix = record['action'] + (f' {record["tool"]}' if 'tool' in record else '')
        text = f'{prefix}: {text}'
        if 'redirected_from' in record:
            text += f' (redirected from: {record["redirected_from"]})'
    elif record['kind'] == 'diagnosis':
        text = f'{text} (confidence {record["confidence"]}, {record["provenance"]})'
    return text


# --------------------------------------------------------------------------
# CLI.

_TEXT_FLAGS = ('symptom', 'hypothesis', 'next_observation', 'confidence', 'provenance', 'action',
               'rationale', 'tool', 'redirected_from', 'outcome', 'text', 'model', 'stage', 'campaign')
_LIST_FLAGS = {'alternative': 'alternatives', 'candidate_tool': 'candidate_tools',
               'invalid_when': 'invalid_when', 'candidate_id': 'candidate_ids',
               'invalidation': 'invalidation', 'evidence_ref': 'evidence_refs'}


def _load_json(path, name):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read {name} {path}: {error}') from None


def _add_fields(args):
    fields = {}
    if args.json:
        fields = _load_json(args.json, '--json')
        if not isinstance(fields, dict):
            raise ValueError('--json file must contain a JSON object')
    flags = {}
    for name in ('kind', 'actor', 'summary') + _TEXT_FLAGS:
        if getattr(args, name) is not None:
            flags[name] = getattr(args, name)
    for flag, name in _LIST_FLAGS.items():
        if getattr(args, flag):
            flags[name] = list(getattr(args, flag))
    if args.context_json is not None:
        flags['context'] = _load_json(args.context_json, '--context-json')
    clash = sorted(set(fields) & set(flags))
    if clash:
        raise ValueError(f'Fields {clash} given both in --json and as flags')
    fields.update(flags)
    if 'kind' not in fields:
        raise ValueError(f'kind is required; one of {list(KINDS)}')
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m flash_rt.vla.autotune.decisions',
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    add = commands.add_parser('add', help='Append one validated record')
    add.add_argument('--log', required=True)
    add.add_argument('--json', help='JSON object with record fields (kind/actor/summary allowed)')
    add.add_argument('--kind')
    add.add_argument('--actor', help=f'Required: one of {", ".join(ACTORS)}')
    add.add_argument('--summary')
    for name in _TEXT_FLAGS:
        add.add_argument('--' + name.replace('_', '-'), dest=name)
    for flag in _LIST_FLAGS:
        add.add_argument('--' + flag.replace('_', '-'), dest=flag, action='append')
    add.add_argument('--context-json', help='Experience workload signature JSON file')
    listing = commands.add_parser('list', help='Print records (JSON) or the step table')
    listing.add_argument('--log', required=True)
    listing.add_argument('--kind')
    listing.add_argument('--markdown', action='store_true')
    summary = commands.add_parser('summary', help='Counts by actor x kind')
    summary.add_argument('--log', required=True)
    summary.add_argument('--markdown', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'add':
            fields = _add_fields(args)
            kind = fields.pop('kind')
            actor = fields.pop('actor', None)
            text = fields.pop('summary', None)
            output = DecisionLog(args.log).append(kind, actor=actor, summary=text, **fields)
        else:
            log = DecisionLog(args.log)
            if not log.path.exists():
                raise ValueError(f'Decision log {log.path} does not exist')
            if args.command == 'list':
                if args.markdown:
                    if args.kind is not None:
                        raise ValueError('--markdown renders every step; do not combine with --kind')
                    sys.stdout.write(log.markdown_table())
                    return 0
                output = log.records(args.kind)
            else:
                if args.markdown:
                    sys.stdout.write(log.markdown_table() + '\n' + log.markdown_summary())
                    return 0
                output = log.summary()
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
