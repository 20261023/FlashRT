"""Render site-compatible batch scripts and parse replies; never submit jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
import os
from os import fspath
import re
from shlex import quote
import socket


def require_gpu_allocation(env=None, hostname=None):
    """Identify a single-node GPU batch or compute step before model loading."""
    env = os.environ if env is None else env
    host = (hostname or socket.gethostname()).split('.')[0]
    node = env.get('SLURMD_NODENAME') or env.get('SLURM_JOB_NODELIST', '')
    step = env.get('SLURM_STEP_ID', 'batch')
    if (not env.get('SLURM_JOB_ID', '').isdigit() or host.lower().startswith(('ln', 'login'))
            or host != node.split('.')[0] or not (step == 'batch' or step.isdigit())
            or env.get('CUDA_VISIBLE_DEVICES', '').lower() in ('', '-1', 'nodevfiles', 'none', 'void')):
        raise RuntimeError('Requires an allocated Slurm GPU on its compute node')
    return step


def _directive(value, name, *, path=False):
    if path:
        try:
            value = fspath(value)
        except TypeError:
            raise ValueError(f'{name} must be a path string') from None
    if (not isinstance(value, str) or not value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f'{name} must be nonempty text without control characters')
    return value


@dataclass(frozen=True)
class SlurmResources:
    partition: str = 'gpu_a800'
    gpus: int = 1

    def __post_init__(self):
        _directive(self.partition, 'partition')
        if isinstance(self.gpus, bool) or not isinstance(self.gpus, Integral) or self.gpus < 1:
            raise ValueError('gpus must be a positive integer')


def render_sbatch(*, job_name, workdir, output, argv,
                  resources=SlurmResources(), env=None) -> str:
    """Return a shell-quoted script without invoking sbatch or srun.

    N46H2 manual §2.3.1: request GPUs and run the single-node Python command
    directly in the batch script. Slurm assigns CPUs and walltime.
    """
    if not isinstance(resources, SlurmResources):
        raise TypeError('resources must be SlurmResources')
    job_name = _directive(job_name, 'job_name')
    workdir = _directive(workdir, 'workdir', path=True)
    output = _directive(output, 'output', path=True)
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ValueError('argv must be a nonempty sequence of argument strings')
    if any(not isinstance(arg, str) or '\x00' in arg for arg in argv) or not argv[0]:
        raise ValueError('argv must contain strings without NUL and a nonempty executable')
    if env is None:
        env = {}
    if not isinstance(env, Mapping):
        raise ValueError('env must be a mapping of variable names to strings')
    for key, value in env.items():
        if not isinstance(key, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) is None:
            raise ValueError('env keys must be shell variable names')
        if not isinstance(value, str) or '\x00' in value:
            raise ValueError('env values must be strings without NUL')
    directives = [
        ('job-name', job_name), ('partition', resources.partition),
        ('gpus', str(resources.gpus)),
        ('chdir', workdir), ('output', output),
    ]
    lines = ['#!/bin/bash']
    # Slurm's get_argument, unlike a shell, consumes backslash escapes even
    # inside single quotes. Escape those first; argv/exports use normal quote().
    for name, value in directives:
        escaped = quote(value.replace('\\', '\\\\'))
        lines.append(f'#SBATCH --{name}={escaped}')
    lines.extend(['', 'set -euo pipefail'])
    lines.extend(f'export {key}={quote(env[key])}' for key in sorted(env))
    lines.append('exec ' + ' '.join(quote(arg) for arg in argv))
    return '\n'.join(lines) + '\n'


def parse_submission(stdout: str) -> str:
    """Parse sbatch --parsable's jobid[;cluster], not human status messages."""
    if not isinstance(stdout, str):
        raise ValueError('submission output must be text')
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', stdout.strip())
    if match is None:
        raise ValueError('Malformed sbatch --parsable output')
    return match.group(1)
