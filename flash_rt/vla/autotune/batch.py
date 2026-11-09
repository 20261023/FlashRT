"""A single allocation serially runs isolated candidate worker processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .schema import Candidate, validate_result


def run_batch(plan, *, execute=None, python=None):
    if plan.get('schema_version') != 2 or not plan.get('campaign_instance_id'):
        raise ValueError('Batch requires a bound campaign-instance plan, not a historical unbound plan')
    if execute is None:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Execute campaign batches inside a Slurm allocation')
        execute = subprocess.run
    python = python or sys.executable
    output = Path(plan['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    control = output / 'candidates'
    control.mkdir(exist_ok=True)
    reference = None
    entries = []
    for entry in plan['entries']:
        candidate = Candidate.from_dict(entry['candidate'])
        attempt_id = entry['attempt_id']
        if not attempt_id or Path(attempt_id).name != attempt_id or attempt_id in ('.', '..'):
            raise ValueError('Invalid attempt filename')
        candidate_path = control / f'{attempt_id}.json'
        candidate_path.write_text(json.dumps(candidate.to_dict(), indent=2) + '\n')
        directory = output / attempt_id
        argv = [python, '-B', '-m', 'flash_rt.vla.autotune.worker',
                '--candidate', str(candidate_path), '--workload', plan['workload_path'],
                '--output', str(directory), '--attempt-id', attempt_id,
                '--comparison-id', plan['comparison_id'], '--warmup', str(plan['warmup']),
                '--campaign-instance-id', plan['campaign_instance_id'],
                '--repeats', str(plan['repeats'])]
        if plan.get('quality_gate'):
            argv.extend(['--quality-gate', json.dumps(plan['quality_gate'])])
        if entry['role'] == 'reference':
            argv.append('--baseline-reference')
        elif reference is not None:
            argv.extend(['--reference', str(reference)])
        else:
            raise RuntimeError('Reference must succeed before testing candidates')
        with (output / f'{attempt_id}.log').open('x') as log:
            completed = execute(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
        result_path = directory / 'result.json'
        record = json.loads(result_path.read_text()) if result_path.is_file() else None
        if record is not None:
            validate_result(record)
        entries.append({'attempt_id': attempt_id, 'candidate_id': candidate.candidate_id,
                        'returncode': completed.returncode, 'result_path': str(result_path),
                        'state': None if record is None else record['state']})
        if entry['role'] == 'reference':
            if completed.returncode or record is None or record['state'] != 'measured':
                break
            reference = Path(record['outputs_path'])
    result = {'comparison_id': plan['comparison_id'], 'campaign_instance_id': plan['campaign_instance_id'],
              'job_id': os.environ.get('SLURM_JOB_ID'),
              'entries': entries, 'complete': len(entries) == len(plan['entries'])}
    (output / 'batch_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args(argv)
    result = run_batch(json.loads(Path(args.plan).read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] and all(item['returncode'] == 0 for item in result['entries']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
