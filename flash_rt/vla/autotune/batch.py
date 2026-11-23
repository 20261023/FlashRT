"""A single allocation serially runs isolated candidate worker processes.

Schema 2 plans run every worker from the allocation's own FlashRT checkout.
Schema 3 (snapshot) plans first re-verify every source snapshot, then launch
each attempt with its snapshot as the only FlashRT on PYTHONPATH and a fresh,
empty ``<output_root>/work/<attempt_id>`` as cwd, so nothing an attempt writes
relative to its cwd can dirty the shared snapshot.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .schema import Candidate, validate_result

_STRIPPED_ENV = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE')


def _check_extra_pythonpath(paths):
    if not isinstance(paths, list) or not all(isinstance(path, str) and os.path.isabs(path) for path in paths):
        raise ValueError('runtime.extra_pythonpath must be a list of absolute paths')
    for path in paths:
        if os.path.exists(os.path.join(path, 'flash_rt', '__init__.py')):
            raise ValueError(f'runtime.extra_pythonpath entry contains another flash_rt package: {path}')
    return paths


def preflight_snapshots(plan):
    """Raise before any attempt runs unless every snapshot is ready, unchanged and clean.

    Returns {snapshot_id: manifest} (plan copies without ``manifest_path``).
    """
    from .snapshots import (SnapshotError, SourceIdentityError, check_candidate_source,
                            is_ready, snapshot_id, validate_manifest, verify_source)
    for name in ('workload_path', 'output_root'):
        if not isinstance(plan.get(name), str) or not os.path.isabs(plan[name]):
            raise ValueError(f'Snapshot plan {name} must be an absolute path')
    harness = (plan.get('harness') or {}).get('autotune_tree')
    runtime = plan.get('runtime')
    if not isinstance(runtime, dict) or set(runtime) != {'python', 'extra_pythonpath'}:
        raise ValueError('Snapshot plan runtime must have exactly python and extra_pythonpath')
    if runtime['python'] is not None and (not isinstance(runtime['python'], str)
                                          or not os.path.isabs(runtime['python'])):
        raise ValueError('Snapshot plan runtime.python must be null or an absolute path')
    _check_extra_pythonpath(runtime['extra_pythonpath'])
    if not isinstance(plan.get('snapshots'), dict) or not plan['snapshots']:
        raise ValueError('Snapshot plan requires a nonempty snapshots mapping')
    manifests = {}
    for identifier, copy in plan['snapshots'].items():
        if not isinstance(copy, dict) or not isinstance(copy.get('manifest_path'), str):
            raise ValueError(f'Snapshot {identifier} has no manifest_path in the plan')
        manifest = {key: value for key, value in copy.items() if key != 'manifest_path'}
        path = Path(copy['manifest_path'])
        if not path.is_file():
            raise ValueError(f'Snapshot manifest is missing: {path}')
        try:
            stored = validate_manifest(json.loads(path.read_text()))
        except (SnapshotError, json.JSONDecodeError) as error:
            raise ValueError(f'Snapshot manifest {path} is invalid: {error}') from error
        if stored != manifest or identifier != manifest['snapshot_id']:
            raise ValueError(f'Snapshot manifest {path} differs from the plan copy of {identifier}')
        if not is_ready(stored):
            raise ValueError(f'Snapshot {identifier} is not ready (native {stored["native"]["status"]})')
        if stored['harness_tree'] != harness:
            raise ValueError(f'Snapshot {identifier} harness tree differs from the plan harness')
        try:
            verify_source(stored, import_root=stored['source_dir'])
        except SourceIdentityError as error:
            raise ValueError(f'Snapshot {identifier} failed source verification: {error}') from error
        manifests[identifier] = stored
    for entry in plan['entries']:
        candidate = Candidate.from_dict(entry['candidate'])
        identifier = entry.get('snapshot_id')
        if identifier not in manifests:
            raise ValueError(f'Entry {entry.get("attempt_id")} names unknown snapshot {identifier!r}')
        if snapshot_id(candidate.source_ref, candidate.patch_ref) != identifier:
            raise ValueError(f'Entry {entry["attempt_id"]} candidate refs do not derive snapshot {identifier}')
        check_candidate_source(manifests[identifier], candidate.source_ref, candidate.patch_ref,
                               error=ValueError)
    return manifests


def _snapshot_launch(plan, manifest, attempt_id, control):
    """Per-attempt cwd/env and the --source-manifest arguments for a schema 3 entry.

    The cwd is a new empty directory, never the snapshot: verify_source rejects
    any untracked file, and the reference and bookend share one snapshot.
    PYTHONSAFEPATH keeps it off sys.path; where that is ignored (Python < 3.11)
    ``-m`` prepends it, which is harmless because it is empty.
    """
    source = os.path.realpath(manifest['source_dir'])
    work = Path(plan['output_root']) / 'work' / attempt_id
    work.mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if key not in _STRIPPED_ENV}
    env.update(PYTHONPATH=os.pathsep.join([source, *plan['runtime']['extra_pythonpath']]),
               PYTHONSAFEPATH='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    path = control / f'{attempt_id}.source.json'
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {'cwd': str(work), 'env': env}, ['--source-manifest', str(path)]


def run_batch(plan, *, execute=None, python=None):
    version = plan.get('schema_version')
    if version not in (2, 3) or not plan.get('campaign_instance_id'):
        raise ValueError('Batch requires a bound campaign-instance plan, not a historical unbound plan')
    manifests = preflight_snapshots(plan) if version == 3 else None
    if execute is None:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Execute campaign batches inside a Slurm allocation')
        execute = subprocess.run
    python = python or sys.executable
    if version == 3 and plan['runtime']['python'] is not None:
        python = plan['runtime']['python']
    output = Path(plan['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    control = output / 'candidates'
    control.mkdir(exist_ok=True)
    if version == 3:
        sources = output / 'control'
        sources.mkdir(exist_ok=True)
    reference = None
    author_reference = None
    author_status = None
    entries = []
    if plan.get('author_baseline') is not None:
        from .author import (AuthorReferenceError, export_portable_reference,
                             prepare_reference, record_portable_attempt)
        try:
            author_reference = prepare_reference(plan, execute=execute)
            export_portable_reference(plan, author_reference)
            author_status = {'status': 'verified', 'descriptor_path': author_reference,
                             'mode': plan['author_baseline']['mode'],
                             'portable_manifest': 'control/author_portable.json'}
        except AuthorReferenceError as error:
            result = {'comparison_id': plan['comparison_id'],
                      'campaign_instance_id': plan['campaign_instance_id'],
                      'job_id': os.environ.get('SLURM_JOB_ID'), 'entries': [], 'complete': False,
                      'author_baseline': {'status': 'failed', 'kind': 'author_reference',
                                          'message': str(error),
                                          'producer_command': plan['author_baseline']['producer_command']}}
            (output / 'batch_result.json').write_text(json.dumps(result, indent=2) + '\n')
            return result
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
        if author_reference is not None:
            argv.extend(['--author-reference', author_reference])
        if entry['role'] == 'reference':
            argv.append('--baseline-reference')
        elif reference is not None:
            argv.extend(['--reference', str(reference)])
        else:
            raise RuntimeError('Reference must succeed before testing candidates')
        launch = {}
        if version == 3:
            launch, source_args = _snapshot_launch(plan, manifests[entry['snapshot_id']], attempt_id, sources)
            argv.extend(source_args)
        with (output / f'{attempt_id}.log').open('x') as log:
            completed = execute(argv, stdout=log, stderr=subprocess.STDOUT, check=False, **launch)
        result_path = directory / 'result.json'
        record = json.loads(result_path.read_text()) if result_path.is_file() else None
        if record is not None:
            validate_result(record)
            if author_reference is not None:
                record_portable_attempt(plan, entry, record, result_path)
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
    if author_status is not None:
        result['author_baseline'] = author_status
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
