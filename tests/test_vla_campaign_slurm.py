"""Rendered scripts are exercised locally with a non-submitting srun shim."""

from dataclasses import FrozenInstanceError
from importlib import import_module
import shlex
import subprocess

import pytest


def _slurm():
    return import_module('flash_rt.vla.autotune.slurm')


def test_default_resources_and_site_policy_are_rendered_without_cpu_header():
    slurm = _slurm()
    resources = slurm.SlurmResources()
    assert (resources.partition, resources.gpus) == ('gpu_a800', 1)
    with pytest.raises(FrozenInstanceError):
        resources.gpus = 2
    script = slurm.render_sbatch(
        job_name='campaign', workdir='/repo', output='/results/%j.log',
        argv=['python', '-m', 'worker'])
    directives = [line for line in script.splitlines() if line.startswith('#SBATCH')]
    assert directives == [
        '#SBATCH --job-name=campaign', '#SBATCH --partition=gpu_a800',
        '#SBATCH --gres=gpu:1', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --chdir=/repo',
        '#SBATCH --output=/results/%j.log',
    ]
    assert 'set -euo pipefail' in script
    assert all('--cpus-per-task' not in line and '--cpus-per-gpu' not in line
               for line in directives)
    assert 'exec srun -n 1 python -m worker' in script


def test_header_paths_and_command_arguments_are_quoted_as_literals():
    script = _slurm().render_sbatch(
        job_name="campaign 'quoted'", workdir='/repo with spaces',
        output='/results with spaces/%j.log', argv=['python', 'a b', "it's literal"])
    directives = [shlex.split(line[len('#SBATCH '):]) for line in script.splitlines()
                  if line.startswith('#SBATCH ')]
    assert ['--job-name=campaign \'quoted\''] in directives
    assert ['--chdir=/repo with spaces'] in directives
    assert ['--output=/results with spaces/%j.log'] in directives
    invocation = next(line for line in script.splitlines() if 'srun -n' in line)
    assert shlex.split(invocation) == [
        'exec', 'srun', '-n', '1',
        'python', 'a b', "it's literal",
    ]


def test_directive_backslashes_are_escaped_for_slurms_non_shell_parser():
    # Slurm get_argument consumes backslash escapes even within single quotes.
    # Unlike command argv, a directive must therefore double literal slashes.
    script = _slurm().render_sbatch(
        job_name='campaign', workdir=r'/repo\literal', output=r'/results\literal/%j.log',
        argv=['python', r'argument\literal'])
    assert r"#SBATCH --chdir='/repo\\literal'" in script.splitlines()
    assert r"#SBATCH --output='/results\\literal/%j.log'" in script.splitlines()
    invocation = next(line for line in script.splitlines() if 'srun -n' in line)
    assert shlex.split(invocation)[-1] == r'argument\literal'


@pytest.mark.parametrize('kwargs', [
    {'gpus': 0}, {'gpus': True}, {'gpus': 1.5},
    {'partition': 'gpu\n#SBATCH --exclusive'},
])
def test_resources_reject_invalid_and_injected_values(kwargs):
    with pytest.raises(ValueError):
        _slurm().SlurmResources(**kwargs)


@pytest.mark.parametrize('field', ['job_name', 'workdir', 'output'])
@pytest.mark.parametrize('control', ['\n', '\r', '\x00'])
def test_directive_values_cannot_inject_another_header(field, control):
    kwargs = {'job_name': 'campaign', 'workdir': '/repo', 'output': '/results/%j.log',
              'argv': ['python', '-m', 'worker']}
    kwargs[field] += f'{control}#SBATCH --exclusive'
    with pytest.raises(ValueError):
        _slurm().render_sbatch(**kwargs)


@pytest.mark.parametrize('argv, env', [
    ([], None), ('python -m worker', None), (['python', 1], None),
    (['python', 'a\x00b'], None), (['python'], {'BAD-NAME': 'x'}),
    (['python'], {'BAD\nNAME': 'x'}), (['python'], {'GOOD': 12}),
])
def test_invalid_command_and_environment_are_rejected(argv, env):
    with pytest.raises((TypeError, ValueError)):
        _slurm().render_sbatch(job_name='campaign', workdir='/repo',
                               output='/results/log', argv=argv, env=env)


def _run_locally(tmp_path, allocated_cpus):
    # This shim never launches its arguments. There is no call to Slurm, a
    # model, a GPU, or even Python in the generated command under test.
    shim = tmp_path / 'srun'
    shim.write_text('#!/bin/bash\nprintf "%s\\0" "$@" "$PAYLOAD"\n')
    shim.chmod(0o700)
    script = _slurm().render_sbatch(
        job_name='campaign', workdir=tmp_path, output=tmp_path / '%j.log',
        argv=['python', '-m', 'never_run', 'a b', "it's literal", '$(touch injected)'],
        env={'PAYLOAD': 'one\ntwo $(touch also_injected)'},
    )
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin'}
    if allocated_cpus is not None:
        env['SLURM_CPUS_ON_NODE'] = allocated_cpus
    return subprocess.run(['/bin/bash'], input=script.encode(), cwd=tmp_path,
                          env=env, capture_output=True, timeout=10, check=False)


@pytest.mark.parametrize('allocated_cpus', [None, '1', '32'])
def test_script_inherits_allocation_and_preserves_literal_argv_and_env(tmp_path, allocated_cpus):
    result = _run_locally(tmp_path, allocated_cpus)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(b'\0') == [
        b'-n', b'1', b'python', b'-m',
        b'never_run', b'a b', b"it's literal", b'$(touch injected)',
        b'one\ntwo $(touch also_injected)', b'',
    ]
    assert not (tmp_path / 'injected').exists()
    assert not (tmp_path / 'also_injected').exists()


@pytest.mark.parametrize('stdout, job_id', [
    ('12345', '12345'), ('12345\n', '12345'), ('12345;cluster-a\n', '12345'),
])
def test_parsable_submission_accepts_numeric_id_and_optional_cluster(stdout, job_id):
    assert _slurm().parse_submission(stdout) == job_id


@pytest.mark.parametrize('stdout', [
    '', '0', '-1', 'Submitted batch job 123', '12_3', '12.3',
    '123;cluster;extra', '123;', '123\n456', '123;bad cluster', '123;$(bad)',
])
def test_malformed_submission_output_is_never_mistaken_for_a_job_id(stdout):
    with pytest.raises(ValueError):
        _slurm().parse_submission(stdout)
