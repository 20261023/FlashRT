"""Resumable local controller for human/search/agent experiment proposals."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
import time

from .ledger import CampaignLedger
from .schema import Candidate, encode_backend
from .search import enumerate_candidates, select_winner
from .quality import validate_quality_contract


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Campaign/attempt IDs must be readable filename-safe identifiers')
    return value


def _save_json(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def initialize_campaign(root, *, comparison_id, baseline, candidates, workload_path,
                        output_root, warmup=3, repeats=10, quality_gate=None):
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError('Campaign requires warmup >= 0 and repeats > 0 integer budgets')
    if quality_gate is not None:
        if not isinstance(quality_gate, dict) or not isinstance(quality_gate.get('scope'), str) or not quality_gate['scope'].strip():
            raise ValueError('quality_gate requires a nonempty scope')
        unknown = set(quality_gate) - {'scope', 'action_dim', 'max_relative_l2', 'max_abs', 'min_cosine'}
        if unknown:
            raise ValueError(f'Unknown quality gate fields: {sorted(unknown)}')
        if 'action_dim' in quality_gate and (type(quality_gate['action_dim']) is not int or quality_gate['action_dim'] < 1):
            raise ValueError('quality action_dim must be a positive integer')
        for key in ('max_relative_l2', 'max_abs', 'min_cosine'):
            if key not in quality_gate:
                continue
            value = quality_gate[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite numeric quality budget')
            if (key == 'min_cosine' and not -1 <= value <= 1) or (key != 'min_cosine' and value < 0):
                raise ValueError(f'{key} is outside its quality metric range')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _identifier(comparison_id)
    plan_path = root / 'plan.json'
    if plan_path.exists():
        raise FileExistsError('Campaign already initialized; reopen its ledger, do not recreate attempts')
    entries = []
    points = [('reference', baseline), *[('candidate', candidate) for candidate in candidates],
              ('bookend', baseline)]
    candidate_records = {}
    initialization = []
    # Validate the ENTIRE proposal list before persisting any prefix of it.
    for index, (role, candidate) in enumerate(points):
        if not isinstance(candidate, Candidate):
            raise TypeError('Expected Candidate records')
        if candidate.workload_id != baseline.workload_id or candidate.backend.model != baseline.backend.model:
            raise ValueError('One campaign must compare the same model and workload')
        candidate_record = candidate.to_dict()
        if candidate.candidate_id in candidate_records and candidate_records[candidate.candidate_id] != candidate_record:
            raise ValueError('Conflicting candidate IDs within campaign')
        candidate_records[candidate.candidate_id] = candidate_record
        attempt_id = f'{comparison_id}-{index:03d}-{role}'
        initialization.append((candidate, comparison_id, attempt_id))
        entries.append({'role': role, 'attempt_id': attempt_id, 'candidate': candidate_record})
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        ledger.initialize_attempts(initialization)
        instance_id = ledger.instance_id
    plan = {'schema_version': 2, 'comparison_id': comparison_id,
            'campaign_instance_id': instance_id,
            'workload_path': str(workload_path), 'output_root': str(output_root),
            'warmup': warmup, 'repeats': repeats, 'quality_gate': quality_gate,
            'entries': entries}
    _save_json(plan_path, plan)
    return plan


def mark_submitting(root):
    with CampaignLedger(Path(root) / 'ledger.sqlite') as ledger:
        ledger.transition_group([attempt['attempt_id'] for attempt in ledger.attempts()],
                                ('proposed',), 'submitting')


def mark_queued(root, job_id):
    from .slurm import parse_submission
    job_id = parse_submission(job_id)
    with CampaignLedger(Path(root) / 'ledger.sqlite') as ledger:
        pending = []
        for attempt in ledger.attempts():
            if attempt['state'] in ('submitting', 'submission_unknown'):
                pending.append(attempt['attempt_id'])
            elif attempt['job_id'] != job_id:
                raise ValueError('Confirmed job conflicts with the existing attempt/job state')
        if pending:
            ledger.transition_group(pending, ('submitting', 'submission_unknown'), 'queued',
                                    job_id=job_id, allow_already=True)


def mark_submission_unknown(root, message):
    with CampaignLedger(Path(root) / 'ledger.sqlite') as ledger:
        pending = [attempt['attempt_id'] for attempt in ledger.attempts() if attempt['state'] == 'submitting']
        if pending:
            ledger.transition_group(pending, ('submitting',), 'submission_unknown', message=message)


def submit_campaign(root, script, client):
    root = Path(root)
    mark_submitting(root)
    submission = {'script': str(script), 'state': 'submitting'}
    _save_json(root / 'submission.json', submission)
    try:
        job_id = client.submit(script)
        submission.update(job_id=job_id, state='acknowledged')
        _save_json(root / 'submission.json', submission)
        mark_queued(root, job_id)
    except Exception as error:
        submission.update(state='submission_unknown',
                          message=str(error).strip() or type(error).__name__)
        _save_json(root / 'submission.json', submission)
        mark_submission_unknown(root, submission['message'])
        raise
    return job_id


def wait_campaign(root, client, *, interval=15, timeout=3600, sleep=None, clock=None):
    """Monitor confirmed jobs only; scheduler completion never validates results.

    An interrupted/expired wait can be resumed safely. No submission, ingestion,
    rejection or winner selection happens here, even for terminal failed jobs.
    """
    for name, value in (('interval', interval), ('timeout', timeout)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive finite number of seconds')
    root = Path(root)
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        attempts = ledger.attempts()
    if not attempts or any(item['job_id'] is None for item in attempts):
        raise ValueError('wait requires a confirmed job ID for every attempt; reconcile unknown submissions first')
    jobs = sorted({item['job_id'] for item in attempts})
    deadline = clock() + timeout
    while True:
        statuses = [client.status(job_id) for job_id in jobs]
        if any(item.get('job_id') != job_id for job_id, item in zip(jobs, statuses)):
            raise ValueError('Scheduler status does not match the confirmed job IDs')
        with CampaignLedger(root / 'ledger.sqlite') as ledger:
            for status in statuses:
                if status['state'] not in ('RUNNING', 'COMPLETING'):
                    continue
                queued = [item['attempt_id'] for item in ledger.attempts()
                          if item['job_id'] == status['job_id'] and item['state'] == 'queued']
                if queued:
                    ledger.transition_group(queued, ('queued',), 'running')
        terminal = all(item['terminal'] for item in statuses)
        remaining = deadline - clock()
        output = {
            'jobs': statuses, 'timed_out': not terminal and remaining <= 0,
            'scheduler_success': terminal and all(
                item['state'] == 'COMPLETED' and item.get('exit_code') == '0:0' for item in statuses),
            'interpretation': 'Scheduler state only; ingest complete worker bundles before validation/ranking',
        }
        _save_json(root / 'scheduler-status.json', output)
        if terminal or remaining <= 0:
            return output
        sleep(min(interval, remaining))


def ingest_campaign(root, results_dir, *, job_finished=False):
    root, results_dir = Path(root), Path(results_dir)
    plan = json.loads((root / 'plan.json').read_text())
    if plan.get('schema_version') != 2:
        raise ValueError('Legacy campaign/results lack bound execution identity; retain historical evidence, use new attempts to qualify')
    scope = (plan.get('quality_gate') or {}).get('scope', 'numerical_replay')
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        if plan['campaign_instance_id'] != ledger.instance_id:
            raise ValueError('Plan campaign instance does not match ledger')
        records_by_attempt = {}
        # Preflight every supplied envelope before importing any group prefix.
        for entry in plan['entries']:
            path = results_dir / entry['attempt_id'] / 'result.json'
            if path.is_file():
                record = json.loads(path.read_text())
                ledger.validate_execution_identity(record)
                if record['state'] == 'measured':
                    validate_quality_contract(record['quality'], plan.get('quality_gate'))
                records_by_attempt[entry['attempt_id']] = record
        for entry in plan['entries']:
            attempt = ledger.get_attempt(entry['attempt_id'])
            if entry['attempt_id'] in records_by_attempt:
                record = records_by_attempt[entry['attempt_id']]
                ledger.ingest(record)
                attempt = ledger.get_attempt(entry['attempt_id'])
                if attempt['state'] == 'measured':
                    passed = record['quality']['status'] == 'passed' and record['quality']['scope'] == scope
                    ledger.transition(entry['attempt_id'], 'measured', 'validated' if passed else 'rejected',
                                      message=f'Quality scope: {record["quality"]["scope"]}; not task-success validation')
            elif job_finished and attempt['state'] in ('queued', 'running'):
                ledger.transition(entry['attempt_id'], attempt['state'], 'failed',
                                  failure_kind='missing_result', message='Job finished without a complete worker bundle')
        attempts = ledger.attempts()
    complete = all(attempt['state'] in ('validated', 'rejected', 'failed') for attempt in attempts)
    records = [attempt['result'] for attempt in attempts if attempt['result'] is not None]
    winner = select_winner(records, quality_scope=scope) if complete and records else None
    by_attempt = {record['attempt_id']: record for record in records}
    first, last = plan['entries'][0], plan['entries'][-1]
    reference, bookend = by_attempt.get(first['attempt_id']), by_attempt.get(last['attempt_id'])
    drift = None
    if reference and bookend and reference['state'] == bookend['state'] == 'measured':
        base_ms = median(reference['measurement']['latency_ms'])
        drift = median(bookend['measurement']['latency_ms']) / max(base_ms, 1e-12) - 1.0
        if (abs(drift) > 0.10 or any(
                item['quality']['status'] != 'passed' or item['quality']['scope'] != scope
                for item in (reference, bookend))):
            winner = None
    elif complete:
        winner = None
    summary = {'comparison_id': plan['comparison_id'], 'complete': complete,
               'quality_scope': scope, 'winner': winner,
               'winner_interpretation': 'Only within the recorded quality/proxy scope, never an inferred task-success winner',
               'baseline_drift_ratio': drift, 'attempts': attempts}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('init')
    create.add_argument('--root', required=True)
    create.add_argument('--comparison-id', required=True)
    create.add_argument('--baseline', required=True)
    create.add_argument('--workload', required=True)
    create.add_argument('--output-root', required=True)
    create.add_argument('--warmup', type=int, default=3)
    create.add_argument('--repeats', type=int, default=10)
    create.add_argument('--quality-gate')
    create.add_argument('--proposals', help='Optional JSON list of human/agent Candidate records')
    create.add_argument('--enumerate', action='store_true')
    submit = commands.add_parser('submitting')
    submit.add_argument('--root', required=True)
    queued = commands.add_parser('queued')
    queued.add_argument('--root', required=True)
    queued.add_argument('--job-id', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('--root', required=True)
    submit.add_argument('--script', required=True)
    submit.add_argument('--ssh-host')
    status = commands.add_parser('status')
    status.add_argument('--root', required=True)
    status.add_argument('--ssh-host')
    wait = commands.add_parser('wait')
    wait.add_argument('--root', required=True)
    wait.add_argument('--ssh-host')
    wait.add_argument('--interval', type=float, default=15)
    wait.add_argument('--timeout', type=float, default=3600)
    ingest = commands.add_parser('ingest')
    ingest.add_argument('--root', required=True)
    ingest.add_argument('--results', required=True)
    ingest.add_argument('--job-finished', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'init':
        baseline = Candidate.from_dict(json.loads(Path(args.baseline).read_text()))
        points = ([point for point in enumerate_candidates(baseline)
                   if encode_backend(point.backend) != encode_backend(baseline.backend)]
                  if args.enumerate else [])
        if args.proposals:
            points.extend(Candidate.from_dict(record) for record in json.loads(Path(args.proposals).read_text()))
        output = initialize_campaign(args.root, comparison_id=args.comparison_id, baseline=baseline,
                                     candidates=points, workload_path=args.workload, output_root=args.output_root,
                                     warmup=args.warmup, repeats=args.repeats,
                                     quality_gate=json.loads(args.quality_gate) if args.quality_gate else None)
    elif args.command == 'submitting':
        mark_submitting(args.root)
        output = {'state': 'submitting'}
    elif args.command == 'queued':
        mark_queued(args.root, args.job_id)
        output = {'state': 'queued', 'job_id': args.job_id}
    elif args.command == 'submit':
        from .scheduler import SlurmClient
        job_id = submit_campaign(args.root, args.script, SlurmClient(args.ssh_host))
        output = {'state': 'queued', 'job_id': job_id}
    elif args.command == 'status':
        from .scheduler import SlurmClient
        with CampaignLedger(Path(args.root) / 'ledger.sqlite') as ledger:
            jobs = {item['job_id'] for item in ledger.attempts() if item['job_id'] is not None}
        output = {'jobs': [SlurmClient(args.ssh_host).status(job_id) for job_id in sorted(jobs)],
                  'interpretation': 'Scheduler state only; ingest complete results before validation/ranking'}
        _save_json(Path(args.root) / 'scheduler-status.json', output)
    elif args.command == 'wait':
        from .scheduler import SlurmClient
        output = wait_campaign(args.root, SlurmClient(args.ssh_host),
                               interval=args.interval, timeout=args.timeout)
    else:
        output = ingest_campaign(args.root, args.results, job_finished=args.job_finished)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
