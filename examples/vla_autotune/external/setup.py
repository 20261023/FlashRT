#!/usr/bin/env python3
"""Install pinned external tools into new, disposable source/venv directories.

Only dependency installation and CPU metadata checks run here. This script never
authenticates, imports a model, submits a job, or starts a kernel search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ASSETS = Path(__file__).resolve().parent
TOOLS = {
    'forgekernel': {
        'tool_id': 'kernel.forgekernel',
        'repo': 'https://github.com/OpenBMB/ForgeKernel.git',
        'commit': 'b2c8ae8510e02c28576808910a6b8827721ab7dc',
    },
    'kernelagent': {
        'tool_id': 'kernel.kernelagent',
        'repo': 'https://github.com/meta-pytorch/KernelAgent.git',
        'commit': 'e0647170da36ef9b059ac0bd3d60103aa4ed378b',
    },
    'aitune': {
        'tool_id': 'compile.aitune',
        'repo': 'https://github.com/ai-dynamo/aitune.git',
        'commit': '1d3a4236ad8f458669d5633715b6956125f2289c',
    },
}
KA_DEPENDENCIES = ('openai', 'anthropic', 'jinja2', 'python-dotenv', 'requests',
                   'numpy', 'omegaconf', 'pandas')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _run(args, *, cwd=None, env=None, data=None):
    return subprocess.run([str(a) for a in args], cwd=cwd, env=env, input=data,
                          check=True, capture_output=True).stdout


def _inside(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'path escapes source: {name}')
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f'symlink in source path: {name}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path escapes source: {name}')
    return path


def apply_patch_bundle(repository, bundle=ASSETS / 'forgekernel-sm80'):
    """Check pin, patch hashes and every file before a single git-apply call.

    Mixed/pre-edited states are rejected. An exact postimage is an idempotent
    no-op. Additional source edits (including new files) are rejected.
    """
    root, bundle = Path(repository).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    commit = _run(['git', '-C', root, 'rev-parse', 'HEAD']).decode().strip()
    if commit != manifest['commit']:
        raise ValueError(f'wrong source revision: expected commit {manifest["commit"]}')
    patches = []
    for row in manifest['patches']:
        content = _inside(bundle, row['path']).read_bytes()
        if _sha(content) != row['sha256']:
            raise ValueError(f'patch sha256 mismatch: {row["path"]}')
        patches.append(content)
    states = []
    for name, hashes in manifest['files'].items():
        path = _inside(root, name)
        digest = _sha(path.read_bytes()) if path.exists() else None
        if digest == hashes['after']:
            states.append('after')
        elif digest == hashes['before']:
            states.append('before')
        else:
            raise ValueError(f'modified source / preimage drift: {name}')
    changed = set(_run(['git', '-C', root, 'diff', '--name-only', 'HEAD', '-z'])
                  .decode().strip('\0').split('\0')) - {''}
    untracked = set(_run(['git', '-C', root, 'ls-files', '--others',
                          '--exclude-standard', '-z']).decode().strip('\0').split('\0')) - {''}
    if (changed | untracked) - set(manifest['files']):
        raise ValueError('source has unrelated modified or untracked files')
    if all(state == 'after' for state in states):
        state = 'already_applied'
    elif all(state == 'before' for state in states):
        patch = b''.join(patches)
        _run(['git', '-C', root, 'apply', '--check', '-'], data=patch)
        _run(['git', '-C', root, 'apply', '-'], data=patch)
        state = 'applied'
    else:
        raise ValueError('mixed preimage/postimage drift; use a fresh dedicated clone')
    for name, hashes in manifest['files'].items():
        if _sha(_inside(root, name).read_bytes()) != hashes['after']:
            raise ValueError(f'postimage mismatch: {name}')
    return {'state': state, 'commit': commit,
            'bundle_sha256': _sha((bundle / 'manifest.json').read_bytes()),
            'files': {k: v['after'] for k, v in manifest['files'].items()}}


def provider_presence(environ=None):
    """Report inherited standard credential presence without reading auth files."""
    env = os.environ if environ is None else environ
    return {
        'anthropic_api_key_present': bool(env.get('ANTHROPIC_API_KEY')),
        'claude_oauth_present': bool(env.get('CLAUDE_CODE_OAUTH_TOKEN')),
        'openai_api_key_present': bool(env.get('OPENAI_API_KEY')),
        'relay_url_present': bool(env.get('LLM_RELAY_URL')),
        'authentication_verified': False,
    }


def runtime_metadata(python):
    code = (
        'import importlib.metadata as m,json,sys; '
        'names=("torch","triton","numpy"); '
        'versions={}; '
        '\nfor name in names:\n'
        ' try: versions[name]=m.version(name)\n'
        ' except m.PackageNotFoundError: versions[name]=None\n'
        'print(json.dumps({"python":sys.executable,"version":list(sys.version_info[:3]),'
        '"packages":versions}))'
    )
    return json.loads(_run([python, '-I', '-c', code]))


def constraints_text(packages):
    for name in ('torch', 'triton', 'numpy'):
        if not packages.get(name):
            raise ValueError(f'base runtime has no {name}; supply the FlashRT runtime Python')
    return ''.join(f'{name}=={packages[name]}\n' for name in ('torch', 'triton', 'numpy'))


def aitune_requirement(source, torch_version):
    """Use upstream's declared extra for the baseline Torch minor version."""
    import tomllib
    match = re.match(r'^(\d+)\.(\d+)\.', torch_version)
    extra = f'torch{match[1]}{match[2]}' if match else ''
    config = tomllib.loads((Path(source) / 'pyproject.toml').read_text())
    if not extra or extra not in config['project'].get('optional-dependencies', {}):
        raise ValueError('pinned AITune has no matching Torch version extra')
    return f'{source}[{extra}]'


def _install_env(root, *, wheelhouse=None):
    env = os.environ.copy()
    # Package installation has no reason to receive model-provider credentials.
    for name in list(env):
        if (name.startswith(('PIP_', 'UV_', 'ANTHROPIC_', 'OPENAI_', 'CLAUDE_', 'LLM_'))
                or name in ('PYTHONPATH', 'AUTOWRAPT_BOOTSTRAP')):
            env.pop(name)
    env.update({
        'PIP_CONFIG_FILE': os.devnull, 'PIP_INDEX_URL': 'https://pypi.org/simple',
        'PIP_CACHE_DIR': str(root / 'cache/pip'), 'TMPDIR': str(root / 'tmp'),
        'XDG_CACHE_HOME': str(root / 'cache'), 'PYTHONNOUSERSITE': '1',
        'PYTHON_DOTENV_DISABLED': '1',
    })
    if wheelhouse is not None:
        wheels = Path(wheelhouse).absolute()
        if not wheels.is_dir():
            raise ValueError('wheelhouse must be an existing directory')
        env.update({'PIP_NO_INDEX': '1', 'PIP_FIND_LINKS': str(wheels)})
    return env


def install(tool, root, base_python, *, source_repo=None, wheelhouse=None):
    """Make one new installation; leave existing installations untouched."""
    root = Path(root).absolute()
    if root.exists():
        raise FileExistsError(f'install destination already exists: {root}')
    spec = TOOLS[tool]
    base = runtime_metadata(base_python)
    if tuple(base['version'][:2]) < (3, 11) or (
            tool == 'kernelagent' and tuple(base['version'][:2]) >= (3, 13)):
        raise ValueError('use Python 3.11 or 3.12 for the shared tool runtimes')
    constraint = constraints_text(base['packages']) if tool != 'forgekernel' else ''
    root.mkdir(parents=True)
    for name in ('tmp', 'cache/pip', 'cache/triton', 'cache/aitune', 'runs', 'logs'):
        (root / name).mkdir(parents=True, exist_ok=True)
    env = _install_env(root, wheelhouse=wheelhouse)
    source, venv = root / 'source', root / 'venv'
    result = {'tool_id': spec['tool_id'], 'upstream': spec, 'base_runtime': base,
              'root': str(root), 'source': str(source), 'python': str(venv / 'bin/python'),
              'dependency_setup': 'SETUP_IN_PROGRESS', 'functional_validation': 'NOT_RUN',
              'gpu_validation': 'NOT_RUN', 'provider': provider_presence()}
    try:
        _run(['git', 'clone', '--no-hardlinks', source_repo or spec['repo'], source], env=env)
        _run(['git', '-C', source, 'checkout', '--detach', spec['commit']], env=env)
        if tool == 'forgekernel':
            result['sm80'] = apply_patch_bundle(source)
        _run([base_python, '-m', 'venv', '--system-site-packages', venv], env=env)
        python = venv / 'bin/python'
        (root / 'constraints.txt').write_text(constraint)
        pip = [python, '-m', 'pip', '--disable-pip-version-check']
        _run([*pip, 'install', '--no-deps', '-e', source], env=env)
        if tool == 'kernelagent':
            args = [*pip, 'install', '-c', root / 'constraints.txt', *KA_DEPENDENCIES]
        elif tool == 'aitune':
            args = [*pip, 'install', '-c', root / 'constraints.txt',
                    '--extra-index-url', 'https://pypi.nvidia.com',
                    '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
                    '-e', aitune_requirement(source, base['packages']['torch'])]
        else:
            args = None
        if args:
            _run(args, env=env)
        current = runtime_metadata(python)
        if constraint and current['packages'] != base['packages']:
            raise ValueError('install changed the pinned Torch/Triton/numpy runtime')
        result['runtime'] = current
        if tool == 'kernelagent':
            result['gpu_name'] = 'NVIDIA A100 SXM4 80GB'
            shutil.copyfile(ASSETS / 'kernelagent-sm80.json', root / 'kernelagent-sm80.json')
        result['dependency_setup'] = 'INSTALLED_METADATA_ONLY'
        freeze = _run([*pip, 'list', '--format=json'], env=env)
        (root / 'packages.json').write_bytes(freeze)
    except (subprocess.CalledProcessError, ValueError) as error:
        result['dependency_setup'] = 'SETUP_FAILED'
        # Deliberately never serialize exception text or raw subprocess output:
        # dependency tooling can include credentials/URLs in error messages.
        result['failure_type'] = type(error).__name__
        if isinstance(error, subprocess.CalledProcessError):
            result['failure_returncode'] = error.returncode
            result['failure_step'] = str(error.cmd[0])
        raise
    finally:
        (root / 'setup-result.json').write_text(json.dumps(result, indent=2) + '\n')
        # No secrets, no global shell changes, no PYTHONPATH import bypass for workers.
        import shlex
        q = shlex.quote
        lines = [
            f'export PATH={q(str(venv / "bin"))}:"$PATH"',
            'export PYTHONNOUSERSITE=1 PYTHON_DOTENV_DISABLED=1',
            f'export TRITON_CACHE_DIR={q(str(root / "cache/triton"))}',
            f'export AITUNE_CACHE_DIR={q(str(root / "cache/aitune"))}',
            f'export XDG_CACHE_HOME={q(str(root / "cache"))}',
        ]
        if tool == 'forgekernel':
            lines += ['export FK_DELIVERY=local FK_GPU=0 LB_INSTALL_CUPTI=0',
                      f'export FORGEKERNEL_RUNS_DIR={q(str(root / "runs"))}']
        elif tool == 'kernelagent':
            lines += [f'export KERNELAGENT_GPU_NAME={q("NVIDIA A100 SXM4 80GB")}']
        (root / 'activate.sh').write_text('\n'.join(lines) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    apply = sub.add_parser('apply-sm80')
    apply.add_argument('--source', required=True, type=Path)
    prepare = sub.add_parser('install')
    prepare.add_argument('--tool', choices=TOOLS, required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--base-python', required=True)
    prepare.add_argument('--source-repo', help='optional local clone or git bundle of the pin')
    prepare.add_argument('--wheelhouse', type=Path, help='offline wheels, including build dependencies')
    sub.add_parser('presence')
    args = parser.parse_args(argv)
    if args.action == 'apply-sm80':
        result = apply_patch_bundle(args.source)
    elif args.action == 'presence':
        result = provider_presence()
    else:
        try:
            result = install(args.tool, args.root, args.base_python, source_repo=args.source_repo,
                             wheelhouse=args.wheelhouse)
        except subprocess.CalledProcessError:
            print(json.dumps({'status': 'SETUP_FAILED', 'root': str(args.root),
                              'reason': 'dependency/source action failed; see setup-result.json'}))
            return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
