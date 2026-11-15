#!/usr/bin/env python3
"""Deploy ForgeKernel dependencies; infrastructure and sandbox policy stay external."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--env', type=Path, required=True)
    p.add_argument('--role', choices=['controller', 'gpu', 'both'], required=True)
    p.add_argument('--dsl-version', help='CuTe DSL version selected for this task')
    p.add_argument('--dsl-source-wheels', type=Path, nargs='+',
                   help='Linux DSL wheels to extract as controller-only API source')
    p.add_argument('--torch-spec', help='Explicit CUDA torch package spec, if absent from image')
    p.add_argument('--index-url', help='Verified pip index for this machine')
    p.add_argument('--wheelhouse', type=Path, help='Offline wheel directory; disables pip indexes')
    p.add_argument('--claude', type=Path, help='Existing executable; required for offline controller setup')
    p.add_argument('--claude-version', help='Version to install when Claude CLI is absent')
    p.add_argument('--npm-registry', help='Verified npm registry')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def plan(a):
    repo, env = a.repo.resolve(), a.env.resolve()
    manifest = repo / 'pyproject.toml'
    if not manifest.is_file() or tomllib.loads(manifest.read_text()).get('project', {}).get('name') != 'forgekernel':
        raise ValueError(f'Not a ForgeKernel checkout: {repo}')
    if sys.version_info < (3, 11):
        raise ValueError('Python 3.11 or newer is required')
    if env.exists() and not (env / 'pyvenv.cfg').is_file():
        raise ValueError(f'Refusing existing non-venv directory: {env}')
    if a.wheelhouse and not a.wheelhouse.is_dir():
        raise ValueError(f'Missing wheelhouse: {a.wheelhouse}')
    if a.wheelhouse and a.index_url:
        raise ValueError('Choose wheelhouse or index-url, not both')
    controller = a.role in ('controller', 'both')
    gpu = a.role in ('gpu', 'both')
    cli = a.claude.resolve() if a.claude else None
    if cli and (not cli.is_file() or not os.access(cli, os.X_OK)):
        raise ValueError(f'Claude executable is unavailable: {cli}')
    if not cli and controller:
        found = None if a.claude_version else shutil.which('claude')
        if found:
            cli = Path(found).resolve()
        elif a.wheelhouse:
            raise ValueError('Offline controller deployment requires --claude')
    if a.dsl_source_wheels:
        if a.role != 'controller':
            raise ValueError('Source-only DSL wheels require controller role')
        for wheel in a.dsl_source_wheels:
            if not wheel.is_file() or wheel.suffix != '.whl':
                raise ValueError(f'Missing DSL wheel: {wheel}')
    commands = []
    if not env.exists():
        commands.append([sys.executable, '-m', 'venv',
                         *(['--system-site-packages'] if gpu else []), str(env)])
    py = str(env / 'bin/python')
    pip = [py, '-m', 'pip', 'install', '--timeout', '30', '--retries', '0']
    if a.wheelhouse:
        pip += ['--no-index', '--find-links', str(a.wheelhouse.resolve())]
    elif a.index_url:
        pip += ['--index-url', a.index_url]
    commands.append(pip + ['setuptools>=61', 'wheel'])
    commands.append(pip + ['--no-build-isolation', '-e', str(repo)])
    if gpu:
        if a.torch_spec:
            commands.append(pip + [a.torch_spec])
        commands.append(pip + ['safetensors'])
    if a.dsl_version:
        commands.append(pip + ['nvidia-cutlass-dsl==' + a.dsl_version])
    for wheel in a.dsl_source_wheels or []:
        commands.append([py, '-m', 'zipfile', '-e', str(wheel.resolve()), str(env / 'dsl-source')])
    if controller:
        if cli is None:
            npm = shutil.which('npm')
            if not npm or not shutil.which('node'):
                commands.append(pip + ['nodejs-wheel'])
                npm = str(env / 'bin/npm')
            cmd = [npm, 'install', '--prefix', str(env / 'agent'), '--fetch-timeout=30000', '--fetch-retries=0']
            if a.npm_registry:
                cmd += ['--registry', a.npm_registry]
            cmd += ['@anthropic-ai/claude-code' + ('@' + a.claude_version if a.claude_version else '')]
            commands.append(cmd)
            cli = env / 'agent/node_modules/.bin/claude'
        commands.append([str(cli), '--help'])
    commands.append([str(env / 'bin/forge'), '--help'])
    if gpu:
        code = ('import torch; '
                'assert torch.cuda.is_available(), "CUDA PyTorch is unavailable; supply --torch-spec for this image"; '
                f'd={a.gpu}; x=torch.ones(8,device=f"cuda:{{d}}"); '
                'assert x.sum().item()==8; '
                'print("GPU_OK",torch.cuda.get_device_name(d),torch.cuda.get_device_capability(d))')
        if a.dsl_version:
            code += '; import cutlass; print("CUTLASS_OK")'
        commands.append([py, '-c', code])
    return commands


def deploy(a):
    commands = plan(a)
    for cmd in commands:
        if a.dry_run:
            print(shlex.join(cmd))
        else:
            subprocess.run(cmd, check=True, env=dict(os.environ,
                PATH=str(a.env.resolve() / 'bin') + os.pathsep + os.environ.get('PATH', '')))
    if not a.dry_run:
        print(json.dumps({'status': 'dependencies_ready', 'role': a.role,
                          'env': str(a.env.resolve()),
                          'authentication_verified': False, 'sandbox_verified': False}))


if __name__ == '__main__':
    deploy(parse_args())
