"""Render site-compatible batch scripts and parse replies; never submit jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from os import fspath
import re
from shlex import quote


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
    cpus: int = 8
    time_limit: str = '00:20:00'

    def __post_init__(self):
        _directive(self.partition, 'partition')
        for name in ('gpus', 'cpus'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (not isinstance(self.time_limit, str)
                or re.fullmatch(r'[0-9]{2}:[0-5][0-9]:[0-5][0-9]', self.time_limit) is None
                or self.time_limit == '00:00:00'):
            raise ValueError('time_limit must be bounded HH:MM:SS between 00:00:01 and 99:59:59')


def render_sbatch(*, job_name, workdir, output, argv,
                  resources=SlurmResources(), env=None) -> str:
    """Return a shell-quoted script without invoking sbatch or srun.

    This site's plugin derives CPU allocation from the GPU request: do not put
    cpus-per-task/per-gpu in the sbatch header. Refuse an insufficient/missing
    allocation at runtime, then request the exact CPU count on the srun step.
    Shell exports are emitted only after the CPU check and after all directives.
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
        ('gres', f'gpu:{resources.gpus}'), ('nodes', '1'), ('ntasks', '1'),
        ('time', resources.time_limit), ('chdir', workdir), ('output', output),
    ]
    lines = ['#!/bin/bash']
    # Slurm's get_argument, unlike a shell, consumes backslash escapes even
    # inside single quotes. Escape those first; argv/exports use normal quote().
    for name, value in directives:
        escaped = quote(value.replace('\\', '\\\\'))
        lines.append(f'#SBATCH --{name}={escaped}')
    lines.extend([
        '', 'set -euo pipefail',
        'allocated_cpus=${SLURM_CPUS_ON_NODE:-0}',
        'if [[ ! "$allocated_cpus" =~ ^[0-9]+$ ]] '
        f'|| (( 10#$allocated_cpus < {resources.cpus} )); then',
        f'    echo "Insufficient allocated CPUs: need {resources.cpus}, '
        'SLURM_CPUS_ON_NODE=$allocated_cpus" >&2',
        '    exit 1', 'fi',
    ])
    lines.extend(f'export {key}={quote(env[key])}' for key in sorted(env))
    lines.append(f'exec srun -n 1 -c {resources.cpus} --cpu-bind=cores '
                 + ' '.join(quote(arg) for arg in argv))
    return '\n'.join(lines) + '\n'


def parse_submission(stdout: str) -> str:
    """Parse sbatch --parsable's jobid[;cluster], not human status messages."""
    if not isinstance(stdout, str):
        raise ValueError('submission output must be text')
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', stdout.strip())
    if match is None:
        raise ValueError('Malformed sbatch --parsable output')
    return match.group(1)
