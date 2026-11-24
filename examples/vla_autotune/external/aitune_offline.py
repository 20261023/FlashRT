#!/usr/bin/env python3
"""Build an explicit CU12 metapackage or install AITune in a clean offline venv.

Public packaging work is SETUP_FAILED on error, not an authorization BLOCKED
state. No authentication, model execution, Slurm, sudo, or global installation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


TENSORRT_VERSION = '10.15.1.29'
LOCAL_TENSORRT_VERSION = '10.15.1.29+flashrtcu12.1'


def cuda12_metapackage_text(text):
    """Adapt only the public meta dependency; label the changed bytes locally."""
    version = f'version="{TENSORRT_VERSION}"'
    dependency = f'"tensorrt_cu13=={TENSORRT_VERSION}"'
    if text.count(version) != 1 or text.count(dependency) != 1:
        raise ValueError('unexpected TensorRT metapackage source/version')
    return text.replace(version, f'version="{LOCAL_TENSORRT_VERSION}"').replace(
        dependency, f'"tensorrt_cu12=={TENSORRT_VERSION}"')


def create_clean_venv(base_python, destination, *, with_pip=True):
    """Never inherit system/user site packages or replace an existing venv."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    args = [str(base_python), '-m', 'venv']
    if not with_pip:
        args.append('--without-pip')
    subprocess.run([*args, str(destination)], check=True, capture_output=True)
    return destination / 'bin/python'


def lock_constraints(lock):
    """Pin already hash-verified wheels without imposing hashes on editable Git."""
    pins = re.findall(r'^([A-Za-z0-9_.-]+==[^\s\\]+)', Path(lock).read_text(), re.M)
    if not pins:
        raise ValueError('wheel lock must contain exact package==version pins')
    return '\n'.join(pins) + '\n'


def _common():
    spec = importlib.util.spec_from_file_location(
        '_external_setup_common', Path(__file__).with_name('setup.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(root, base_python, wheelhouse, lock, source_repo):
    """Install a complete hash-locked wheel closure, then the unchanged source."""
    common = _common()
    root = Path(root).absolute()
    if root.exists():
        raise FileExistsError(root)
    base = common.runtime_metadata(base_python)
    if base['version'][:2] != [3, 11]:
        raise ValueError('offline profile requires Python 3.11')
    root.mkdir(parents=True)
    for directory in ('tmp', 'cache/pip', 'cache/aitune', 'cache/triton'):
        (root / directory).mkdir(parents=True)
    env = common._install_env(root, wheelhouse=wheelhouse)
    env['NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP'] = '1'
    result = {'tool_id': 'compile.aitune', 'dependency_setup': 'SETUP_IN_PROGRESS',
              'gpu_validation': 'NOT_RUN', 'functional_validation': 'NOT_RUN',
              'root': str(root), 'system_site_packages': False}
    try:
        python = create_clean_venv(base_python, root / 'venv')
        pip = [python, '-m', 'pip', '--disable-pip-version-check']
        common._run([*pip, 'install', '--require-hashes', '-r', lock], env=env)
        source = root / 'source'
        common._run(['git', 'clone', '--no-hardlinks', source_repo, source], env=env)
        pin = common.TOOLS['aitune']['commit']
        common._run(['git', '-C', source, 'checkout', '--detach', pin], env=env)
        # The source has no packaging patch. The explicitly local TensorRT
        # metapackage supplies the CU12 dependency to both AITune and TorchTRT.
        constraints = root / 'constraints.txt'
        constraints.write_text(lock_constraints(lock))
        common._run([*pip, 'install', '--no-build-isolation', '-c', constraints, '-e',
                     f'{source}[torch211,onnxruntime-gpu-cuda12]'], env=env)
        check = common._run([*pip, 'check'], env=env).decode().strip()
        common._run(['git', '-C', source, 'diff', '--exit-code', 'HEAD'], env=env)
        result.update({'dependency_setup': 'INSTALLED_PIP_CHECK_PASS',
                       'upstream_commit': pin, 'source': str(source),
                       'python': str(python), 'runtime': common.runtime_metadata(python),
                       'pip_check': check})
        (root / 'packages.json').write_bytes(
            common._run([*pip, 'list', '--format=json'], env=env))
        import shlex
        q = shlex.quote
        (root / 'activate.sh').write_text('\n'.join([
            f'export PATH={q(str(root / "venv/bin"))}:"$PATH"',
            'export PYTHONNOUSERSITE=1 PYTHON_DOTENV_DISABLED=1',
            'unset PYTHONPATH AUTOWRAPT_BOOTSTRAP',
            f'export AITUNE_CACHE_DIR={q(str(root / "cache/aitune"))}',
            f'export TRITON_CACHE_DIR={q(str(root / "cache/triton"))}',
        ]) + '\n')
    except Exception as error:
        result['dependency_setup'] = 'SETUP_FAILED'
        result['failure_type'] = type(error).__name__
        if isinstance(error, subprocess.CalledProcessError):
            result['failure_returncode'] = error.returncode
            # This dedicated invocation has no provider secrets, and all index
            # access is disabled. Keep useful offline install errors in its log.
            (root / 'install-error.log').write_bytes(error.stderr or b'')
        raise
    finally:
        (root / 'setup-result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    adapt = sub.add_parser('adapt-metapackage')
    adapt.add_argument('--setup-py', type=Path, required=True)
    prepare = sub.add_parser('install')
    for name in ('root', 'base-python', 'wheelhouse', 'lock', 'source-repo'):
        prepare.add_argument(f'--{name}', required=True)
    args = parser.parse_args(argv)
    if args.action == 'adapt-metapackage':
        original = args.setup_py.read_text()
        args.setup_py.write_text(cuda12_metapackage_text(original))
    else:
        result = install(args.root, args.base_python, args.wheelhouse, args.lock,
                         args.source_repo)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
