"""Durable single-controller SQLite ledger; workers never write this database."""

from __future__ import annotations

from collections.abc import Sequence, Set
from contextlib import contextmanager
import json
import sqlite3
from uuid import uuid4

from .schema import Candidate, _json_value, _text, validate_result
from .snapshots import check_snapshot_id, is_ready, snapshot_id, validate_manifest


_TRANSITIONS = {
    'proposed': {'submitting', 'failed'},
    'submitting': {'queued', 'submission_unknown', 'failed'},
    'submission_unknown': {'queued', 'failed'},
    'queued': {'running', 'measured', 'failed'},
    'running': {'measured', 'failed'},
    'measured': {'validated', 'rejected'},
    'validated': set(), 'rejected': set(), 'failed': set(),
}


def _canonical(record):
    return json.dumps(_json_value(record), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


class CampaignLedger:
    """Every mutation commits before returning and is recoverable after reopen.

    Submission is deliberately not performed here. ``submission_unknown`` can
    only be reconciled to queued or failed; retries always need a new ID.
    Reads return detached dictionaries. ``result`` contains the complete worker
    record (or None), not just summary timings.
    """

    def __init__(self, path):
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute('PRAGMA foreign_keys = ON')
            with self._write() as db:
                db.execute('''CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                )''')
                db.execute('''CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    comparison_id TEXT NOT NULL,
                    workload_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    job_id TEXT,
                    failure_kind TEXT,
                    message TEXT,
                    result_json TEXT
                )''')
                db.execute('''CREATE TABLE IF NOT EXISTS campaign_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                )''')
                db.execute('''CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                )''')
                db.execute('INSERT OR IGNORE INTO campaign_metadata(key, value) VALUES (?, ?)',
                           ('instance_id', str(uuid4())))
        except BaseException:
            self.close()
            raise

    def _connection(self):
        if self._db is None:
            raise RuntimeError('CampaignLedger is closed')
        return self._db

    @contextmanager
    def _write(self):
        db = self._connection()
        # Reserve the writer before read/modify/write. A second controller is
        # unsupported operationally, but cannot silently race through our CAS.
        db.execute('BEGIN IMMEDIATE')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        self._connection()
        return self

    def __exit__(self, *exc):
        self.close()

    def register(self, candidate: Candidate):
        if not isinstance(candidate, Candidate):
            raise TypeError('candidate must be a Candidate')
        payload = _canonical(candidate.to_dict())
        with self._write() as db:
            row = db.execute('SELECT record_json FROM candidates WHERE candidate_id = ?',
                             (candidate.candidate_id,)).fetchone()
            if row is not None:
                if row['record_json'] != payload:
                    raise ValueError(f'Conflicting candidate already registered: {candidate.candidate_id}')
                return
            db.execute('INSERT INTO candidates(candidate_id, record_json) VALUES (?, ?)',
                       (candidate.candidate_id, payload))

    def get_candidate(self, candidate_id) -> Candidate:
        row = self._connection().execute(
            'SELECT record_json FROM candidates WHERE candidate_id = ?',
            (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return Candidate.from_dict(json.loads(row['record_json']))

    @staticmethod
    def _snapshot_payload(manifest):
        validate_manifest(manifest)
        if not is_ready(manifest):
            raise ValueError(f'Snapshot {manifest["snapshot_id"]} is not ready (native state '
                             f'{manifest["native"]["status"]})')
        return _canonical(manifest)

    @staticmethod
    def _check_snapshot_rows(db, payloads):
        """Return the snapshots still to insert; reject changed or mixed-harness records."""
        stored = {row['snapshot_id']: row['record_json'] for row in db.execute('SELECT * FROM snapshots')}
        harness = {json.loads(record)['harness_tree'] for record in stored.values()}
        harness.update(json.loads(payload)['harness_tree'] for payload in payloads.values())
        if len(harness) > 1:
            raise ValueError(f'All campaign snapshots must share one harness tree, got {sorted(harness)}')
        missing = []
        for identifier, payload in payloads.items():
            if identifier not in stored:
                missing.append((identifier, payload))
            elif stored[identifier] != payload:
                raise ValueError(f'Conflicting snapshot already registered: {identifier}')
        return missing

    def register_snapshot(self, manifest) -> None:
        """Persist one ready snapshot manifest; identical re-registration is a no-op."""
        payload = self._snapshot_payload(manifest)
        with self._write() as db:
            missing = self._check_snapshot_rows(db, {manifest['snapshot_id']: payload})
            db.executemany('INSERT INTO snapshots(snapshot_id, record_json) VALUES (?, ?)', missing)

    def get_snapshot(self, snapshot_id) -> dict:
        check_snapshot_id(snapshot_id, error=ValueError)
        row = self._connection().execute(
            'SELECT record_json FROM snapshots WHERE snapshot_id = ?', (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return json.loads(row['record_json'])

    def snapshots(self) -> dict:
        return {row['snapshot_id']: json.loads(row['record_json']) for row in self._connection().execute(
            'SELECT snapshot_id, record_json FROM snapshots ORDER BY snapshot_id')}

    @staticmethod
    def _check_source_identity(record, candidate, snapshots):
        """Schema 3: the executed tree must be the ledger's snapshot for this candidate."""
        if not snapshots:
            raise ValueError('Result claims a source snapshot but this campaign registered none')
        identifier = snapshot_id(candidate.source_ref, candidate.patch_ref)
        if identifier not in snapshots:
            raise ValueError(f'Result candidate snapshot {identifier} is not registered in this campaign')
        manifest = snapshots[identifier]
        identity = record.get('source_identity')
        if identity is None:
            return  # Only failed records may lack evidence (enforced by validate_result).
        if not isinstance(identity, dict):
            raise ValueError('source_identity must be a dict')
        expected = {
            'snapshot_id': identifier, 'source_dir': manifest['source_dir'],
            'base_commit': candidate.source_ref, 'patch_ref': candidate.patch_ref,
            'head_commit': manifest['snapshot_commit'], 'head_tree': manifest['snapshot_tree'],
            'harness_tree': manifest['harness_tree'], 'extensions': manifest['native']['artifacts'],
            'tracked_clean': True, 'unexpected_files': [],
        }
        if manifest['patch_ref'] is not None:
            expected['head_parent'] = manifest['base_commit']
        for name, value in expected.items():
            if name not in identity or _canonical(identity[name]) != _canonical(value):
                raise ValueError(f'Result source_identity {name} does not match the registered snapshot')

    @property
    def instance_id(self):
        return self._connection().execute(
            "SELECT value FROM campaign_metadata WHERE key = 'instance_id'").fetchone()['value']

    def validate_execution_identity(self, record):
        """Bind a result to this persisted instance, constructor and confirmed job."""
        validate_result(record)
        if record.get('schema_version') not in (2, 3):
            raise ValueError('Legacy result has no bound execution identity; retain as historical evidence only')
        if record['campaign_instance_id'] != self.instance_id:
            raise ValueError('Result campaign instance does not match ledger')
        attempt = self.get_attempt(record['attempt_id'])
        for name in ('attempt_id', 'candidate_id', 'comparison_id', 'workload_id'):
            if record[name] != attempt[name]:
                raise ValueError(f'Result {name} does not match attempt')
        candidate = self.get_candidate(attempt['candidate_id']).to_dict()
        if _canonical(record['candidate']) != _canonical(candidate):
            raise ValueError('Result candidate execution configuration/source does not match ledger')
        if attempt['job_id'] is None or record.get('job_id') != attempt['job_id']:
            raise ValueError('Result job does not match confirmed attempt job')
        if record['state'] == 'measured' and record['gpu'].get('job_id') != attempt['job_id']:
            raise ValueError('Measured GPU job does not match confirmed attempt job')
        snapshots = self.snapshots()
        if record['schema_version'] == 3:
            self._check_source_identity(record, Candidate.from_dict(candidate), snapshots)
        elif snapshots:
            raise ValueError('Snapshot campaign requires source-verified schema_version 3 results')
        return attempt

    def begin_attempt(self, candidate_id, comparison_id, attempt_id, *, job_id=None) -> dict:
        for name, value in (('candidate_id', candidate_id), ('comparison_id', comparison_id),
                            ('attempt_id', attempt_id)):
            _text(value, name, nonempty=True)
        if job_id is not None:
            _text(job_id, 'job_id', nonempty=True)
        with self._write() as db:
            candidate = self.get_candidate(candidate_id)
            if db.execute('SELECT 1 FROM attempts WHERE attempt_id = ?',
                          (attempt_id,)).fetchone() is not None:
                raise ValueError(f'Attempt already exists: {attempt_id}')
            db.execute('''INSERT INTO attempts (
                attempt_id, candidate_id, comparison_id, workload_id, state, job_id
            ) VALUES (?, ?, ?, ?, 'proposed', ?)''',
                       (attempt_id, candidate_id, comparison_id, candidate.workload_id, job_id))
        return self.get_attempt(attempt_id)

    def initialize_attempts(self, entries, snapshots=None) -> list[dict]:
        """Register candidates and create an entire initial attempt group atomically.

        ``entries`` is a nonempty list of (Candidate, comparison_id, attempt_id).
        ``snapshots`` (snapshot mode only) is a list of ready manifests; every
        candidate must then name one of them by source_ref/patch_ref, and they
        are registered in the same transaction as the attempts.
        Repeating the same group after a commit/publication interruption is
        allowed only while every row is still pristine proposed. Existing
        attempt IDs (and registered snapshot IDs) must match the whole group
        exactly: partial/extra groups reject instead of being extended, reset
        or silently reused. Unrelated
        registered candidates without attempts are allowed. Return detached
        attempt dictionaries in input order; no external plan is written here.
        """
        if not isinstance(entries, list) or not entries:
            raise ValueError('entries must be a nonempty list of initialization entries')
        candidates = {}
        prepared = []
        attempt_ids = set()
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise ValueError('Each entry must contain Candidate, comparison_id, attempt_id')
            candidate, comparison_id, attempt_id = entry
            if not isinstance(candidate, Candidate):
                raise TypeError('candidate must be a Candidate')
            # Validate and snapshot the complete constructor record before any
            # writes, including options that the shallow frozen config contains.
            candidate = Candidate.from_dict(candidate.to_dict())
            payload = _canonical(candidate.to_dict())
            _text(comparison_id, 'comparison_id', nonempty=True)
            _text(attempt_id, 'attempt_id', nonempty=True)
            if attempt_id in attempt_ids:
                raise ValueError(f'Duplicate attempt ID in initialization: {attempt_id}')
            attempt_ids.add(attempt_id)
            previous = candidates.get(candidate.candidate_id)
            if previous is not None and previous != payload:
                raise ValueError(f'Conflicting candidate in initialization: {candidate.candidate_id}')
            candidates[candidate.candidate_id] = payload
            prepared.append((candidate.candidate_id, comparison_id, attempt_id, candidate.workload_id))
        snapshot_payloads = self._prepare_snapshots(snapshots, candidates)

        with self._write() as db:
            existing = {row['attempt_id']: row for row in db.execute('SELECT * FROM attempts')}
            if existing and existing.keys() != attempt_ids:
                raise ValueError('Initialization must match the entire existing attempt group')
            missing_candidates = []
            for candidate_id, payload in candidates.items():
                row = db.execute('SELECT record_json FROM candidates WHERE candidate_id = ?',
                                 (candidate_id,)).fetchone()
                if row is None:
                    if existing:
                        raise ValueError(f'Existing initialization is missing candidate: {candidate_id}')
                    missing_candidates.append((candidate_id, payload))
                elif row['record_json'] != payload:
                    raise ValueError(f'Conflicting candidate already registered: {candidate_id}')
            for candidate_id, comparison_id, attempt_id, workload_id in prepared:
                if not existing:
                    continue
                row = existing[attempt_id]
                if ((row['candidate_id'], row['comparison_id'], row['workload_id'])
                        != (candidate_id, comparison_id, workload_id)):
                    raise ValueError(f'Conflicting initialization identity for attempt: {attempt_id}')
                if row['state'] != 'proposed' or any(
                        row[name] is not None for name in
                        ('job_id', 'result_json', 'failure_kind', 'message')):
                    raise ValueError(f'Initialization attempt is no longer pristine proposed: {attempt_id}')
            if existing:
                # A retry must supply exactly the stored snapshot set; in particular
                # a schema 2 retry (no snapshots) of a snapshot campaign is refused.
                stored = {row['snapshot_id'] for row in db.execute('SELECT snapshot_id FROM snapshots')}
                if stored != set(snapshot_payloads):
                    raise ValueError('Initialization snapshot set must match the existing attempt group: '
                                     f'stored {sorted(stored)}, supplied {sorted(snapshot_payloads)}')
            missing_snapshots = self._check_snapshot_rows(db, snapshot_payloads)
            # Only after every candidate/attempt has been checked may any row
            # be inserted. _write rolls back even a mid-batch SQLite failure.
            db.executemany('INSERT INTO snapshots(snapshot_id, record_json) VALUES (?, ?)',
                           missing_snapshots)
            db.executemany('INSERT INTO candidates(candidate_id, record_json) VALUES (?, ?)',
                           missing_candidates)
            if not existing:
                db.executemany('''INSERT INTO attempts (
                    candidate_id, comparison_id, attempt_id, workload_id, state
                ) VALUES (?, ?, ?, ?, 'proposed')''', prepared)
            attempts = [self.get_attempt(attempt_id) for _, _, attempt_id, _ in prepared]
        return attempts

    def _prepare_snapshots(self, snapshots, candidates):
        if snapshots is None:
            return {}
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError('snapshots must be None or a nonempty list of manifests')
        payloads = {}
        for manifest in snapshots:
            payload = self._snapshot_payload(manifest)
            if payloads.get(manifest['snapshot_id'], payload) != payload:
                raise ValueError(f'Conflicting snapshot in initialization: {manifest["snapshot_id"]}')
            payloads[manifest['snapshot_id']] = payload
        for candidate_id, payload in candidates.items():
            candidate = json.loads(payload)
            identifier = snapshot_id(candidate['source_ref'], candidate['patch_ref'])
            if identifier not in payloads:
                raise ValueError(f'Candidate {candidate_id} has no snapshot {identifier} in this initialization')
        return payloads

    @staticmethod
    def _attempt_dict(row):
        record = dict(row)
        payload = record.pop('result_json')
        record['result'] = None if payload is None else json.loads(payload)
        return record

    def get_attempt(self, attempt_id) -> dict:
        row = self._connection().execute(
            'SELECT * FROM attempts WHERE attempt_id = ?', (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return self._attempt_dict(row)

    def attempts(self) -> list[dict]:
        return [self._attempt_dict(row) for row in self._connection().execute(
            'SELECT * FROM attempts ORDER BY rowid')]

    def transition(self, attempt_id, expected_state, new_state, *, job_id=None,
                   failure_kind=None, message=None):
        if new_state not in _TRANSITIONS.get(expected_state, set()):
            raise ValueError(f'Illegal state transition: {expected_state} -> {new_state}')
        for name, value in (('job_id', job_id), ('failure_kind', failure_kind), ('message', message)):
            if value is not None:
                _text(value, name, nonempty=name != 'message')
        with self._write() as db:
            self.get_attempt(attempt_id)  # Missing IDs are distinct from stale CAS.
            cursor = db.execute('''UPDATE attempts SET state = ?,
                job_id = COALESCE(?, job_id),
                failure_kind = COALESCE(?, failure_kind), message = COALESCE(?, message)
                WHERE attempt_id = ? AND state = ?''',
                                (new_state, job_id, failure_kind, message, attempt_id, expected_state))
            if cursor.rowcount != 1:
                raise ValueError(f'Stale attempt state for {attempt_id}: expected {expected_state}')

    def transition_group(self, attempt_ids, expected_states, new_state, *, job_id=None,
                         failure_kind=None, message=None, allow_already=False) -> None:
        """Validate a whole group, then commit all legal CAS changes together.

        ``attempt_ids`` is a nonempty sequence without duplicates;
        ``expected_states`` is a nonempty sequence or set of known state names.
        Each row must be in an expected state with a legal edge to new_state.
        With allow_already, rows already at new_state are left untouched even
        if that state is not in expected_states; a provided job_id must match
        their existing job_id exactly. Advanced/stale/illegal/missing rows and
        mid-update SQLite failures abort the entire group, never a prefix.
        """
        if (isinstance(attempt_ids, (str, bytes))
                or not isinstance(attempt_ids, Sequence) or not attempt_ids):
            raise ValueError('attempt_ids must be a nonempty sequence of attempt IDs')
        ids = list(attempt_ids)
        for attempt_id in ids:
            _text(attempt_id, 'attempt_id', nonempty=True)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate attempt IDs in transition group')
        if (isinstance(expected_states, (str, bytes))
                or not isinstance(expected_states, (Sequence, Set)) or not expected_states):
            raise ValueError('expected_states must be a nonempty sequence or set of states')
        states = set()
        for state in expected_states:
            _text(state, 'expected_state', nonempty=True)
            if state not in _TRANSITIONS:
                raise ValueError(f'Unknown expected state: {state}')
            states.add(state)
        _text(new_state, 'new_state', nonempty=True)
        if new_state not in _TRANSITIONS:
            raise ValueError(f'Unknown new state: {new_state}')
        if not isinstance(allow_already, bool):
            raise ValueError('allow_already must be a bool')
        for name, value in (('job_id', job_id), ('failure_kind', failure_kind), ('message', message)):
            if value is not None:
                _text(value, name, nonempty=name != 'message')

        with self._write() as db:
            updates = []
            for attempt_id in ids:
                attempt = self.get_attempt(attempt_id)
                current = attempt['state']
                if allow_already and current == new_state:
                    if job_id is not None and attempt['job_id'] != job_id:
                        raise ValueError(f'Conflicting job_id for already {new_state} attempt: {attempt_id}')
                    continue
                if current not in states:
                    raise ValueError(f'Stale attempt state for {attempt_id}: got {current}')
                if new_state not in _TRANSITIONS[current]:
                    raise ValueError(f'Illegal state transition: {current} -> {new_state}')
                updates.append((attempt_id, current))
            # No UPDATE occurs until every row, including idempotent rows, has
            # passed validation. The CAS remains useful for trigger side effects.
            for attempt_id, current in updates:
                cursor = db.execute('''UPDATE attempts SET state = ?,
                    job_id = COALESCE(?, job_id),
                    failure_kind = COALESCE(?, failure_kind), message = COALESCE(?, message)
                    WHERE attempt_id = ? AND state = ?''',
                                    (new_state, job_id, failure_kind, message, attempt_id, current))
                if cursor.rowcount != 1:
                    raise ValueError(f'Stale attempt state for {attempt_id}: expected {current}')

    def ingest(self, record):
        validate_result(record)
        payload = _canonical(record)
        with self._write() as db:
            attempt = self.validate_execution_identity(record)
            stored = db.execute('SELECT result_json FROM attempts WHERE attempt_id = ?',
                                (record['attempt_id'],)).fetchone()['result_json']
            if stored is not None:
                if stored != payload:
                    raise ValueError(f'Conflicting result for attempt {record["attempt_id"]}')
                return  # Never downgrade validated/rejected on duplicate ingestion.
            state = record['state']
            if state not in _TRANSITIONS[attempt['state']]:
                raise ValueError(f'Cannot ingest {state} from attempt state {attempt["state"]}')
            failure = record.get('failure', {}) if state == 'failed' else {}
            cursor = db.execute('''UPDATE attempts SET result_json = ?, state = ?,
                failure_kind = ?, message = ?
                WHERE attempt_id = ? AND state = ? AND result_json IS NULL''',
                                (payload, state, failure.get('kind'), failure.get('message'),
                                 record['attempt_id'], attempt['state']))
            if cursor.rowcount != 1:
                raise ValueError(f'Stale attempt state for {record["attempt_id"]}')
