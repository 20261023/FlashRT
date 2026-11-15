#!/usr/bin/env python3
"""Deploy the controller, verify the selected GPU, then run or resume ForgeKernel."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import deploy


def check_arch(expected, capability):
    actual = f'sm{capability[0]}{capability[1]}'
    if expected != actual:
        raise ValueError(f'Task requires {expected}; selected GPU is {actual}')


def run_once(state, task, create, run):
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state.exists():
            record = json.loads(state.read_text())
            if record['task'] != task:
                raise ValueError(f'Session belongs to a different task: {state}')
            path = Path(record['run'])
        else:
            path = create()
            temporary = state.with_suffix('.tmp')
            temporary.write_text(json.dumps({'task': task, 'run': str(path)}))
            temporary.replace(state)
        print(f'FORGE_RUN {path}', flush=True)
        return run(path)


def probe_gpu(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'GPU probe failed (exit {error.returncode}): {error.stderr}') from error
    return json.loads(result.stdout.strip().splitlines()[-1])


def execute(args):
    from forgekernel.task import load_task
    from forgekernel.localenv import load_machine
    from forgekernel.runs import new_run, open_run
    from forgekernel.launch import cmd_new, cmd_run
    from forgekernel.paths import repo_root

    if repo_root().resolve() != args.repo.resolve():
        raise ValueError('Installed ForgeKernel does not match the selected repository')
    os.chdir(args.repo)
    task = load_task(args.task)
    machine = load_machine()
    # Inspect the snapshot when resuming; configuration changes must not silently retarget it.
    if args.session.exists():
        record = json.loads(args.session.read_text())
        if record['task'] != str(args.task):
            raise ValueError('Session task differs from the requested task')
        task = load_task(open_run(record['run']).task_dir)
    code = ('import json,torch; '
            f'd={int(machine.gpu)}; '
            'x=torch.ones(8,device=f"cuda:{d}"); assert x.sum().item()==8; '
            'print(json.dumps(list(torch.cuda.get_device_capability(d))))')
    if task.language == 'cutedsl':
        code = 'import cutlass; ' + code
    if machine.delivery == 'ssh':
        command = 'python3 -c ' + shlex.quote(code)
        if machine.remote_env_file:
            command = 'source ' + shlex.quote(machine.remote_env_file) + ' && ' + command
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               machine.remote_host, 'bash -lc ' + shlex.quote(command)]
    else:
        cmd = [sys.executable, '-c', code]
    check_arch(task.arch, probe_gpu(cmd))
    if not args.session.exists() and cmd_new([str(args.task), '--dry-run']) != 0:
        raise ValueError('ForgeKernel task validation failed')
    # Authentication and sandbox enforcement remain owned by ForgeKernel's real agent turn.
    return run_once(args.session, str(args.task),
                    lambda: new_run(task, machine=machine).dir,
                    lambda path: cmd_run([str(path)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Remaining arguments are forwarded to deploy.py. Configure local.env before launch.')
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True,
                        help='Persistent launch pointer; reuse to resume without creating another run')
    args, remaining = parser.parse_known_args()
    installation = deploy.parse_args(remaining)
    if installation.dry_run:
        raise ValueError('Use deploy.py for dry-run; start.py performs a real run')
    if installation.role == 'gpu':
        raise ValueError('start.py must run on a controller; use controller or both')
    args.repo = installation.repo.resolve()
    args.task = args.task.resolve()
    args.session = args.session.resolve()
    target_python = installation.env.resolve() / 'bin/python'
    if Path(sys.prefix).resolve() != installation.env.resolve():
        deploy.deploy(installation)
        os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    cli = installation.claude
    if not cli:
        installed = installation.env.resolve() / 'agent/node_modules/.bin/claude'
        cli = installed if installed.is_file() else shutil.which('claude')
    if not cli:
        raise ValueError('Claude CLI is missing; rerun deploy.py')
    os.environ['FK_CLAUDE_BIN'] = str(Path(cli).resolve())
    os.environ['PATH'] = str(target_python.parent) + os.pathsep + os.environ.get('PATH', '')
    if installation.dsl_source_wheels:
        source = str(installation.env.resolve() / 'dsl-source')
        sys.path.insert(0, source)
        os.environ['PYTHONPATH'] = source + os.pathsep + os.environ.get('PYTHONPATH', '')
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
