"""Content-addressed source snapshots that make Candidate.source_ref/patch_ref executable.

A snapshot is one immutable FlashRT source tree: a full base commit plus an
optional stored patch. Its lifecycle is explicit and every step is a separate
command; this module never submits jobs and never runs a native build.

1. ``materialize``: create a detached git worktree at its final path
   ``<store>/snapshots/<id>`` (never renamed). A patch is applied with
   ``git apply --index`` and committed deterministically, so equal inputs give
   the same commit and tree. Patches touching the measurement harness
   ``flash_rt/vla/autotune``, git metadata, or any path outside the patchable
   tree (``check_patch_paths``) are rejected before any worktree exists; a
   patch that creates a symlink or gitlink is rejected before publication.
   The manifest ``<store>/snapshots/<id>.json`` is written last.
2. Native extensions: ``adopt_native`` copies prebuilt ``flash_rt/*.so`` from a
   checkout whose native inputs (csrc, CMake files, GEMM plan builder) are
   identical; otherwise ``build_script`` renders a build job ending in
   ``record-built``. ``none_expected`` marks a snapshot that loads no extension.
   Only then is the manifest ready (``is_ready``).
3. ``controller init --snapshot-store`` loads the ready manifests, requires one
   shared harness tree, registers them in the ledger and writes plan schema 3.
4. ``batch`` re-checks every manifest and source tree before running anything
   and launches each attempt with its snapshot as the only FlashRT on
   PYTHONPATH and an empty per-attempt working directory as cwd.
5. The worker calls ``verify_source`` before loading the backend and records
   the identity in a schema 3 result, which the ledger checks at ingest.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


SNAPSHOT_SCHEMA = 'flash_rt.autotune.snapshot.v1'
HARNESS_PATH = 'flash_rt/vla/autotune'
NATIVE_INPUTS = ('csrc', 'CMakeLists.txt', 'cmake', 'scripts/build_gemm_plan.py')
FORBIDDEN_PATCH_PREFIXES = ('flash_rt/vla/autotune/', '.git', '.gitignore', '.gitmodules', '.gitattributes')
# The snapshot root is first on every worker's PYTHONPATH, so a patch may only
# touch entries Python never imports from there: the flash_rt package and
# top-level directories/files that hold no top-level module (check_patch_paths).
PATCHABLE_DIRECTORIES = ('flash_rt', 'csrc', 'cmake', 'scripts', 'docs', 'tests', 'examples', 'benchmarks')
PATCHABLE_FILES = ('CMakeLists.txt', 'pyproject.toml', 'README.md', 'USAGE.md', 'CONTRIBUTING.md',
                   'LICENSE', 'NOTICE', 'CITATION.cff')
NATIVE_STATUSES = ('pending', 'reused', 'built', 'none')
READY_STATUSES = ('reused', 'built', 'none')
IDENTITY_ENV_PREFIXES = ('FLASHRT_', 'FVK_', 'PI0_')

_COMMIT = re.compile(r'[0-9a-f]{40}')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_PATCH_REF = re.compile(r'sha256:([0-9a-f]{64})')
_SNAPSHOT_ID = re.compile(r'[0-9a-f]{20}')
_ARTIFACT = re.compile(r'flash_rt/[^/]+\.so')
_GPU_ARCH = re.compile(r'[0-9]+[a-z]?')
_EPOCH = '1970-01-01T00:00:00Z'
_GIT_NAME = 'flash_rt-autotune'
_GIT_EMAIL = 'autotune@localhost'
# Tree entries whose executed content is not the blob the tree records.
_SPECIAL_MODES = {'120000': 'symlink', '160000': 'gitlink'}
_MANIFEST_FIELDS = frozenset({
    'schema', 'snapshot_id', 'source_dir', 'base_commit', 'patch_ref', 'snapshot_commit',
    'snapshot_tree', 'snapshot_parent', 'harness_tree', 'native_inputs', 'native',
})
_NATIVE_FIELDS = frozenset({'status', 'fingerprint', 'artifacts'})


class SnapshotError(ValueError):
    """A snapshot cannot be created, loaded or updated as requested."""


class SourceIdentityError(RuntimeError):
    """The running source tree is not the snapshot an attempt was bound to."""


# --------------------------------------------------------------------------- formats

def check_commit(value, name='commit', *, error=SnapshotError) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise error(f'{name} must be a full 40-hex lowercase git object id, got {value!r}')
    return value


def check_patch_ref(value, *, error=SnapshotError):
    """Return the 64-hex digest of ``sha256:<hex>`` or None for no patch."""
    if value is None:
        return None
    match = _PATCH_REF.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise error(f'patch_ref must be null or sha256:<64 lowercase hex>, got {value!r}')
    return match.group(1)


def check_snapshot_id(value, *, error=SnapshotError) -> str:
    if not isinstance(value, str) or not _SNAPSHOT_ID.fullmatch(value):
        raise error(f'snapshot_id must be 20 lowercase hex characters, got {value!r}')
    return value


def snapshot_id(base_commit, patch_ref=None) -> str:
    check_commit(base_commit, 'base_commit')
    digest = check_patch_ref(patch_ref)
    text = f'flash_rt-snapshot-v1\n{base_commit}\n{digest or "-"}\n'
    return hashlib.sha256(text.encode('ascii')).hexdigest()[:20]


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


# --------------------------------------------------------------------------- git

def _git_env(extra=None):
    # Inherited GIT_* variables (GIT_DIR, GIT_INDEX_FILE, GIT_AUTHOR_*, config
    # injection) would redirect or change these commands; drop all of them.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    env.update(extra or {})
    return env


def _git(cwd, *args, check=True, env=None, error=SnapshotError):
    if shutil.which('git') is None:
        raise error('git executable not found on PATH')
    argv = ['git', '-c', f'user.name={_GIT_NAME}', '-c', f'user.email={_GIT_EMAIL}',
            '-c', 'commit.gpgsign=false', '-c', f'core.hooksPath={os.devnull}',
            '-c', 'advice.detachedHead=false', '-C', str(cwd), *args]
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=_git_env(env), check=False)
    if check and completed.returncode != 0:
        message = completed.stderr.decode('utf-8', 'replace').strip() or 'no error output'
        raise error(f'git {" ".join(args)} failed in {cwd} (exit {completed.returncode}): {message}')
    return completed


def _git_text(cwd, *args, error=SnapshotError) -> str:
    return _git(cwd, *args, error=error).stdout.decode('utf-8', 'surrogateescape').strip()


def _optional_object(cwd, spec, *, error=SnapshotError):
    """Object id of ``spec`` or None when it does not exist."""
    completed = _git(cwd, 'rev-parse', '--verify', '--quiet', spec, check=False, error=error)
    if completed.returncode == 1:
        return None
    if completed.returncode != 0:
        raise error(f'git rev-parse {spec} failed in {cwd}: '
                    f'{completed.stderr.decode("utf-8", "replace").strip()}')
    return check_commit(completed.stdout.decode('ascii').strip(), spec, error=error)


def _toplevel(path, *, error=SnapshotError) -> str:
    if not os.path.isdir(path):
        raise error(f'Not a directory: {path}')
    return os.path.realpath(_git_text(path, 'rev-parse', '--show-toplevel', error=error))


def _native_inputs(cwd, commit, *, error=SnapshotError) -> dict:
    return {path: _optional_object(cwd, f'{commit}:{path}', error=error) for path in NATIVE_INPUTS}


def resolve_commit(repo, ref) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise SnapshotError('ref must be a nonempty string')
    commit = _git_text(repo, 'rev-parse', '--verify', '--end-of-options', f'{ref}^{{commit}}')
    return check_commit(commit, f'resolved {ref!r} (only SHA-1 repositories are supported)')


# --------------------------------------------------------------------------- patches

def store_patch(store, patch_bytes) -> str:
    if not isinstance(patch_bytes, (bytes, bytearray)) or not patch_bytes:
        raise SnapshotError('patch_bytes must be nonempty bytes')
    patch_bytes = bytes(patch_bytes)
    digest = hashlib.sha256(patch_bytes).hexdigest()
    directory = Path(store) / 'patches'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{digest}.patch'
    if path.exists():
        if path.read_bytes() != patch_bytes:
            raise SnapshotError(f'Stored patch {path} does not contain the bytes of sha256:{digest}')
        return f'sha256:{digest}'
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=f'.{digest}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(patch_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return f'sha256:{digest}'


def patch_file(store, patch_ref) -> Path:
    digest = check_patch_ref(patch_ref)
    if digest is None:
        raise SnapshotError('patch_ref is required')
    path = Path(store) / 'patches' / f'{digest}.patch'
    if not path.is_file():
        raise SnapshotError(f'Patch {patch_ref} is not in the store: {path}')
    if _sha256_file(path) != digest:
        raise SnapshotError(f'Stored patch {path} does not hash to {patch_ref}')
    return path


def _numstat_paths(repo, patch_file, *reverse):
    output = _git(repo, 'apply', *reverse, '--numstat', '-z', str(patch_file)).stdout
    paths = []
    for token in output.decode('utf-8', 'surrogateescape').split('\0'):
        if not token:
            continue
        fields = token.split('\t', 2)
        if len(fields) != 3 or not fields[2]:
            raise SnapshotError(f'Unparseable git apply --numstat entry: {token!r}')
        paths.append(fields[2])
    return paths


def patch_paths(repo, patch_file) -> list[str]:
    """Every path a patch touches, including rename/copy sources; applies nothing.

    ``git apply --numstat`` reports only the destination of a rename, so the
    reversed patch is listed as well to obtain the source side.
    """
    paths = _numstat_paths(repo, patch_file) + _numstat_paths(repo, patch_file, '-R')
    return list(dict.fromkeys(paths))


def check_patch_paths(paths) -> None:
    """Reject harness/git-metadata/escaping paths and anything outside the patchable tree.

    The measurement harness ``flash_rt/vla/autotune`` must stay pinned. Because
    the snapshot root is first on the worker's PYTHONPATH (ahead of the
    standard library), a patch may not add anything Python could import from
    that root other than ``flash_rt``: a multi-segment path's first segment
    must be one of PATCHABLE_DIRECTORIES, a top-level file must be one of the
    non-Python PATCHABLE_FILES (so no ``sitecustomize.py``, ``json.py``,
    ``*.pth``, ``*.so`` or ``setup.py``), and no patchable directory other than
    ``flash_rt`` may gain an ``__init__`` module that would make it a regular
    top-level package. Extend the allowlists deliberately, never by default.
    Candidate code under ``flash_rt`` still runs in the worker process, and
    ``flash_rt/__init__.py`` runs before ``verify_source``: the pin guards the
    harness files against accidental edits, not against deliberate tampering.
    """
    if not isinstance(paths, (list, tuple)) or not paths:
        raise SnapshotError('Patch changes no paths')
    for path in paths:
        if not isinstance(path, str) or not path:
            raise SnapshotError(f'Invalid patch path: {path!r}')
        parts = path.split('/')
        if path.startswith('/') or '\\' in path or '..' in parts or '.' in parts or '' in parts:
            raise SnapshotError(f'Patch path must be a normalized relative path: {path!r}')
        if path == HARNESS_PATH or any(path.startswith(prefix) for prefix in FORBIDDEN_PATCH_PREFIXES):
            raise SnapshotError(f'Patch touches a pinned path ({HARNESS_PATH} harness or git metadata): {path}')
        if any(part.startswith('.git') for part in parts):
            raise SnapshotError(f'Patch touches git metadata: {path}')
        if len(parts) == 1:
            if path not in PATCHABLE_FILES:
                raise SnapshotError(f'Patch touches top-level file {path!r}, which is not one of the patchable '
                                    f'non-Python files {list(PATCHABLE_FILES)}; a top-level module or .pth '
                                    'file would be importable from the snapshot root')
        elif parts[0] not in PATCHABLE_DIRECTORIES:
            raise SnapshotError(f'Patch touches top-level directory {parts[0]!r}, which is not one of the '
                                f'patchable directories {list(PATCHABLE_DIRECTORIES)}: {path}')
        elif parts[0] != 'flash_rt' and len(parts) == 2 and parts[1].split('.', 1)[0] == '__init__':
            raise SnapshotError(f'Patch would make {parts[0]!r} an importable top-level package: {path}')


# --------------------------------------------------------------------------- manifests

def manifest_path(store, snapshot_id) -> Path:
    check_snapshot_id(snapshot_id)
    return Path(store) / 'snapshots' / f'{snapshot_id}.json'


def _native_record(native, error):
    if not isinstance(native, dict) or set(native) != _NATIVE_FIELDS:
        raise error(f'manifest native must have exactly the fields {sorted(_NATIVE_FIELDS)}')
    status, fingerprint, artifacts = native['status'], native['fingerprint'], native['artifacts']
    if status not in NATIVE_STATUSES:
        raise error(f'native.status must be one of {NATIVE_STATUSES}, got {status!r}')
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
        raise error('native.fingerprint must be null or a nonempty string')
    if not isinstance(artifacts, dict):
        raise error('native.artifacts must be an object')
    for relpath, digest in artifacts.items():
        if not isinstance(relpath, str) or not _ARTIFACT.fullmatch(relpath):
            raise error(f'native artifact path must be flash_rt/<name>.so, got {relpath!r}')
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise error(f'native artifact {relpath} must have a 64-hex sha256')
    if status == 'pending' and (fingerprint is not None or artifacts):
        raise error('pending native state cannot carry a fingerprint or artifacts')
    if status in ('reused', 'built') and (fingerprint is None or not artifacts):
        raise error(f'{status} native state requires a fingerprint and artifacts')
    if status == 'none' and artifacts:
        raise error('native state none cannot carry artifacts')


def validate_manifest(manifest, *, error=SnapshotError) -> dict:
    """Strict structural validation of one snapshot manifest; returns it unchanged."""
    if not isinstance(manifest, dict):
        raise error('snapshot manifest must be a JSON object')
    missing, unknown = _MANIFEST_FIELDS - set(manifest), set(manifest) - _MANIFEST_FIELDS
    if missing or unknown:
        raise error(f'snapshot manifest fields: missing {sorted(missing)}, unknown {sorted(map(str, unknown))}')
    if manifest['schema'] != SNAPSHOT_SCHEMA:
        raise error(f'snapshot manifest schema must be {SNAPSHOT_SCHEMA}')
    check_snapshot_id(manifest['snapshot_id'], error=error)
    source = manifest['source_dir']
    if not isinstance(source, str) or not os.path.isabs(source):
        raise error('snapshot source_dir must be an absolute path')
    for name in ('base_commit', 'snapshot_commit', 'snapshot_tree', 'harness_tree'):
        check_commit(manifest[name], name, error=error)
    check_patch_ref(manifest['patch_ref'], error=error)
    if snapshot_id(manifest['base_commit'], manifest['patch_ref']) != manifest['snapshot_id']:
        raise error('snapshot_id does not match base_commit/patch_ref')
    if manifest['patch_ref'] is None:
        if manifest['snapshot_parent'] is not None or manifest['snapshot_commit'] != manifest['base_commit']:
            raise error('unpatched snapshot must be the base commit itself with a null snapshot_parent')
    elif (manifest['snapshot_parent'] != manifest['base_commit']
          or manifest['snapshot_commit'] == manifest['base_commit']):
        raise error('patched snapshot must be a new commit whose snapshot_parent is base_commit')
    inputs = manifest['native_inputs']
    if not isinstance(inputs, dict) or set(inputs) != set(NATIVE_INPUTS):
        raise error(f'native_inputs must have exactly the paths {list(NATIVE_INPUTS)}')
    for path, value in inputs.items():
        if value is not None:
            check_commit(value, f'native_inputs[{path}]', error=error)
    _native_record(manifest['native'], error)
    return manifest


def load_manifest(store, snapshot_id) -> dict:
    path = manifest_path(store, snapshot_id)
    if not path.is_file():
        raise SnapshotError(f'Snapshot {snapshot_id} has no manifest (not materialized): {path}')
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SnapshotError(f'Snapshot manifest {path} is not valid JSON: {error}') from error
    validate_manifest(manifest)
    if manifest['snapshot_id'] != snapshot_id:
        raise SnapshotError(f'Manifest {path} describes snapshot {manifest["snapshot_id"]}')
    return manifest


def is_ready(manifest) -> bool:
    return manifest['native']['status'] in READY_STATUSES


@contextlib.contextmanager
def _locked(store, identifier):
    check_snapshot_id(identifier)
    directory = Path(store) / 'snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / f'{identifier}.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SnapshotError(f'Snapshot {identifier} is locked ({lock}); remove the lock only if no '
                            'snapshot command is still running') from None
    try:
        try:
            os.write(descriptor, f'{os.getpid()}\n'.encode('ascii'))
        finally:
            os.close(descriptor)
        yield
    finally:
        os.unlink(lock)


# --------------------------------------------------------------------------- materialize

def _registered_worktrees(repo) -> dict:
    output = _git_text(repo, 'worktree', 'list', '--porcelain')
    worktrees, current = {}, None
    for line in output.splitlines():
        if line.startswith('worktree '):
            current = os.path.realpath(line[len('worktree '):])
            worktrees[current] = {'locked': False}
        elif current is not None and (line == 'locked' or line.startswith('locked ')):
            worktrees[current]['locked'] = True
    return worktrees


def _discard_incomplete(repo, source):
    """A snapshot directory without a manifest is incomplete: unregister and remove it."""
    registered = _registered_worktrees(repo).get(os.path.realpath(source))
    if registered is not None:
        if registered['locked']:
            _git(repo, 'worktree', 'unlock', str(source))
        if os.path.lexists(source):
            _git(repo, 'worktree', 'remove', '--force', '--force', str(source))
        else:
            _git(repo, 'worktree', 'prune')
    elif os.path.lexists(source):
        raise SnapshotError(f'Incomplete snapshot directory {source} is not a registered worktree '
                            f'of {repo}; remove it manually')


def _remove_failed(repo, source, ref):
    problems = []
    if os.path.realpath(source) in _registered_worktrees(repo):
        _git(repo, 'worktree', 'unlock', str(source), check=False)
        removed = _git(repo, 'worktree', 'remove', '--force', '--force', str(source), check=False)
        if removed.returncode:
            problems.append(removed.stderr.decode('utf-8', 'replace').strip())
    _git(repo, 'update-ref', '-d', ref, check=False)
    return problems


def _head_identity(cwd, *, error=SnapshotError) -> dict:
    return {'commit': check_commit(_git_text(cwd, 'rev-parse', '--verify', 'HEAD^{commit}', error=error),
                                   'HEAD', error=error),
            'tree': check_commit(_git_text(cwd, 'rev-parse', '--verify', 'HEAD^{tree}', error=error),
                                 'HEAD tree', error=error),
            'parent': _optional_object(cwd, 'HEAD^1', error=error),
            'harness': _optional_object(cwd, f'HEAD:{HARNESS_PATH}', error=error)}


def _patched_special_paths(source, base_commit) -> list[str]:
    """Paths the patched HEAD turns into a symlink or gitlink relative to base_commit."""
    output = _git(source, 'diff-tree', '-r', '-z', '--no-commit-id', '--no-renames', base_commit, 'HEAD').stdout
    tokens = output.decode('utf-8', 'surrogateescape').split('\0')
    if tokens and tokens[-1] == '':
        tokens.pop()
    special = []
    for index in range(0, len(tokens), 2):
        fields = tokens[index].split(' ')
        if not tokens[index].startswith(':') or len(fields) != 5 or index + 1 >= len(tokens):
            raise SnapshotError(f'Unparseable git diff-tree entry: {tokens[index]!r}')
        if fields[1] in _SPECIAL_MODES:
            special.append(f'{tokens[index + 1]} ({_SPECIAL_MODES[fields[1]]})')
    return special


def _tracked_special_entries(root, *, error) -> list[str]:
    """HEAD entries that are symlinks or gitlinks: the tree records only a link target or a commit id."""
    output = _git(root, 'ls-tree', '-r', '-z', '--full-tree', 'HEAD', error=error).stdout
    special = []
    for token in output.decode('utf-8', 'surrogateescape').split('\0'):
        if not token:
            continue
        meta, separator, path = token.partition('\t')
        if not separator or not path:
            raise error(f'Unparseable git ls-tree entry: {token!r}')
        mode = meta.split(' ', 1)[0]
        # The repository ships Claude skill aliases pointing to the real,
        # tracked .agents/skills/<name>/SKILL.md in this very same tree.
        # This exemption is restricted to those exact documentation aliases;
        # product patches still cannot introduce links (_patched_special_paths).
        match = re.fullmatch(r'\.claude/skills/([A-Za-z0-9_-]+)', path)
        if mode == '120000' and match:
            target = f'.agents/skills/{match[1]}'
            link = Path(root) / path
            if (link.is_symlink() and os.readlink(link) == f'../../{target}'
                    and (Path(root) / target).resolve() == Path(root).resolve() / target
                    and _git_text(root, 'ls-tree', 'HEAD', '--', f'{target}/SKILL.md',
                                  error=error).startswith('100644 blob ')):
                continue
        if mode in _SPECIAL_MODES:
            special.append(f'{path} ({_SPECIAL_MODES[mode]})')
    return special


def _flagged_index_entries(root, *, error) -> list[str]:
    """Index entries git status would not compare (skip-worktree, assume-unchanged, unmerged, ...)."""
    output = _git(root, 'ls-files', '-v', '-z', error=error).stdout
    flagged = []
    for token in output.decode('utf-8', 'surrogateescape').split('\0'):
        if not token:
            continue
        if len(token) < 3 or token[1] != ' ':
            raise error(f'Unparseable git ls-files -v entry: {token!r}')
        if token[0] != 'H':
            flagged.append(token)
    return flagged


def _sparse_checkout(root, *, error) -> bool:
    completed = _git(root, 'config', '--type=bool', '--get', 'core.sparseCheckout', check=False, error=error)
    if completed.returncode == 1:
        return False
    if completed.returncode != 0:
        raise error(f'git config --get core.sparseCheckout failed in {root}: '
                    f'{completed.stderr.decode("utf-8", "replace").strip()}')
    return completed.stdout.strip() == b'true'


def _check_worktree_metadata(root, *, error):
    """Refuse trees and indexes whose files git status/diff would not fully compare to HEAD."""
    special = _tracked_special_entries(root, error=error)
    if special:
        raise error(f'Snapshot tracks symlinks or gitlinks, whose content is outside its tree identity: '
                    f'{special[:10]}')
    flagged = _flagged_index_entries(root, error=error)
    if flagged:
        raise error(f'Snapshot index has entries hidden from git status (git ls-files -v tag other than H, '
                    f'e.g. skip-worktree S or assume-unchanged h): {flagged[:10]}')
    if _sparse_checkout(root, error=error):
        raise error(f'Snapshot worktree {root} has core.sparseCheckout enabled')


def _create_snapshot(repo, source, identifier, base_commit, patch_ref, patch, base_harness):
    ref = f'refs/autotune/snapshots/{identifier}'
    _git(repo, 'worktree', 'add', '--detach', str(source), base_commit)
    try:
        if patch is not None:
            _git(source, 'apply', '--index', '--whitespace=nowarn', str(patch))
            if _git(source, 'diff', '--cached', '--quiet', check=False).returncode == 0:
                raise SnapshotError(f'Patch {patch_ref} produces no change on {base_commit}')
            _git(source, 'commit', '--no-verify', '--no-gpg-sign', '--quiet', '-m', f'autotune-patch {patch_ref}',
                 env={'GIT_AUTHOR_DATE': _EPOCH, 'GIT_COMMITTER_DATE': _EPOCH,
                      'GIT_AUTHOR_NAME': _GIT_NAME, 'GIT_AUTHOR_EMAIL': _GIT_EMAIL,
                      'GIT_COMMITTER_NAME': _GIT_NAME, 'GIT_COMMITTER_EMAIL': _GIT_EMAIL})
        head = _head_identity(source)
        if patch is None and head['commit'] != base_commit:
            raise SnapshotError(f'Worktree HEAD {head["commit"]} is not base {base_commit}')
        if patch is not None and head['parent'] != base_commit:
            raise SnapshotError(f'Patched snapshot parent {head["parent"]} is not base {base_commit}')
        if head['harness'] != base_harness:
            raise SnapshotError(f'Snapshot changed the pinned {HARNESS_PATH} tree')
        if patch is not None:
            special = _patched_special_paths(source, base_commit)
            if special:
                raise SnapshotError(f'Patch {patch_ref} creates symlinks or gitlinks, whose content is '
                                    f'outside the snapshot tree identity: {special[:10]}')
        if _git_text(source, 'status', '--porcelain=v1', '--untracked-files=all'):
            raise SnapshotError('Snapshot worktree is not clean after materialization')
        _check_worktree_metadata(source, error=SnapshotError)
        _git(repo, 'update-ref', ref, head['commit'])
        _git(repo, 'worktree', 'lock', '--reason', f'autotune {identifier}', str(source))
        native_inputs = _native_inputs(source, 'HEAD')
    except BaseException as error:
        problems = _remove_failed(repo, source, ref)
        if problems:
            raise SnapshotError(f'{error}; cleanup of {source} also failed: {"; ".join(problems)}') from error
        raise
    return {'schema': SNAPSHOT_SCHEMA, 'snapshot_id': identifier,
            'source_dir': os.path.realpath(source), 'base_commit': base_commit, 'patch_ref': patch_ref,
            'snapshot_commit': head['commit'], 'snapshot_tree': head['tree'],
            'snapshot_parent': None if patch is None else base_commit,
            'harness_tree': head['harness'], 'native_inputs': native_inputs,
            'native': {'status': 'pending', 'fingerprint': None, 'artifacts': {}}}


def _check_materialized(manifest, repo, base_commit, patch_ref):
    if manifest['base_commit'] != base_commit or manifest['patch_ref'] != patch_ref:
        raise SnapshotError('Existing manifest does not describe the requested base/patch')
    source = manifest['source_dir']
    if not os.path.isdir(source) or _toplevel(source) != os.path.realpath(source):
        raise SnapshotError(f'Ready manifest exists but its worktree is missing: {source}')
    head = _head_identity(source)
    expected = {'commit': manifest['snapshot_commit'], 'tree': manifest['snapshot_tree'],
                'harness': manifest['harness_tree']}
    if any(head[key] != value for key, value in expected.items()):
        raise SnapshotError(f'Existing snapshot worktree {source} no longer matches its manifest')
    ref = _optional_object(repo, f'refs/autotune/snapshots/{manifest["snapshot_id"]}')
    if ref != manifest['snapshot_commit']:
        raise SnapshotError(f'refs/autotune/snapshots/{manifest["snapshot_id"]} does not point at the snapshot')


def materialize(repo, store, base_commit, patch_ref=None) -> dict:
    """Create (or re-validate) the snapshot for base_commit + patch_ref."""
    identifier = snapshot_id(base_commit, patch_ref)
    repo = _toplevel(repo)
    if resolve_commit(repo, base_commit) != base_commit:
        raise SnapshotError(f'{base_commit} is not a commit of {repo}')
    patch = None
    if patch_ref is not None:
        patch = patch_file(store, patch_ref)
        check_patch_paths(patch_paths(repo, patch))
    base_harness = _optional_object(repo, f'{base_commit}:{HARNESS_PATH}')
    if base_harness is None:
        raise SnapshotError(f'Base {base_commit} has no {HARNESS_PATH} measurement harness')
    Path(store, 'snapshots').mkdir(parents=True, exist_ok=True)
    store = os.path.realpath(store)
    with _locked(store, identifier):
        path = manifest_path(store, identifier)
        if path.exists():
            manifest = load_manifest(store, identifier)
            _check_materialized(manifest, repo, base_commit, patch_ref)
            return manifest
        source = Path(store) / 'snapshots' / identifier
        _discard_incomplete(repo, source)
        manifest = _create_snapshot(repo, source, identifier, base_commit, patch_ref, patch, base_harness)
        validate_manifest(manifest)
        _write_json_atomic(path, manifest)  # Publication: the manifest is written last.
        return manifest


# --------------------------------------------------------------------------- native extensions

def _extension_files(source) -> dict:
    directory = Path(source) / 'flash_rt'
    return {f'flash_rt/{path.name}': _sha256_file(path)
            for path in sorted(directory.glob('*.so')) if path.is_file()}


def _update_native(store, manifest, native):
    current = manifest['native']
    if current['status'] != 'pending':
        if current == native:
            return manifest
        raise SnapshotError(f'Snapshot {manifest["snapshot_id"]} native state is already '
                            f'{current["status"]}; materialize a new snapshot instead of overwriting it')
    updated = {**manifest, 'native': native}
    validate_manifest(updated)
    _write_json_atomic(manifest_path(store, manifest['snapshot_id']), updated)
    return updated


def _fingerprint(value, *, required):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError('fingerprint must be a nonempty string describing the native build')
    return value


def _copy_artifact(source_file, destination, digest) -> bool:
    """Copy one extension; return whether this call created ``destination``."""
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise SnapshotError(f'Refusing to overwrite different extension bytes at {destination}')
        return False
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp')
    os.close(descriptor)
    try:
        shutil.copyfile(source_file, temporary)
        os.chmod(temporary, 0o555)
        if _sha256_file(temporary) != digest:
            raise SnapshotError(f'Extension {source_file} changed while it was copied')
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return True


def adopt_native(store, snapshot_id, prebuilt_root, *, fingerprint, none_expected=False) -> dict:
    """Record the snapshot's native extension state (reuse prebuilt or declare none).

    A snapshot whose native state is no longer pending is immutable: an exactly
    identical repeat returns its manifest, anything else is refused before any
    file is copied. Extensions copied by a call that then fails are removed.
    """
    if not isinstance(none_expected, bool):
        raise SnapshotError('none_expected must be a bool')
    fingerprint = _fingerprint(fingerprint, required=not none_expected)
    if not none_expected and prebuilt_root is None:
        raise SnapshotError('prebuilt_root is required unless none_expected')
    with _locked(store, snapshot_id):
        manifest = load_manifest(store, snapshot_id)
        source = Path(manifest['source_dir'])
        if none_expected:
            present = _extension_files(source)
            if present:
                raise SnapshotError(f'none_expected but extension modules exist: {sorted(present)}')
            return _update_native(store, manifest, {'status': 'none', 'fingerprint': fingerprint, 'artifacts': {}})
        prebuilt = os.path.realpath(prebuilt_root)
        if _toplevel(prebuilt) != prebuilt:
            raise SnapshotError(f'prebuilt_root must be the top level of a git checkout: {prebuilt}')
        if manifest['native']['status'] != 'pending':
            # Refuse (or accept an identical repeat) before touching the ready snapshot.
            return _update_native(store, manifest, {'status': 'reused', 'fingerprint': fingerprint,
                                                    'artifacts': _extension_files(prebuilt)})
        inputs = _native_inputs(prebuilt, 'HEAD')
        if inputs != manifest['native_inputs']:
            changed = sorted(path for path in NATIVE_INPUTS if inputs[path] != manifest['native_inputs'][path])
            raise SnapshotError(f'native inputs differ: rebuild required (changed: {changed})')
        dirty = _git_text(prebuilt, 'status', '--porcelain=v1', '--untracked-files=all', '--', *NATIVE_INPUTS)
        if dirty:
            raise SnapshotError('native inputs differ: rebuild required (prebuilt checkout has '
                                f'uncommitted native input changes: {dirty.splitlines()[:5]})')
        built = _extension_files(prebuilt)
        if not built:
            raise SnapshotError(f'No prebuilt extensions in {prebuilt}/flash_rt; build them or pass none_expected')
        extra = sorted(set(_extension_files(source)) - set(built))
        if extra:
            raise SnapshotError(f'Snapshot contains extensions not adopted from {prebuilt}: {extra}')
        copied = []
        try:
            for relpath, digest in built.items():
                if _copy_artifact(Path(prebuilt) / relpath, source / relpath, digest):
                    copied.append(source / relpath)
            return _update_native(store, manifest, {'status': 'reused', 'fingerprint': fingerprint,
                                                    'artifacts': built})
        except BaseException:
            for path in copied:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
            raise


def record_built(store, snapshot_id, *, fingerprint) -> dict:
    """After an external in-place build, hash and record the snapshot's extensions."""
    fingerprint = _fingerprint(fingerprint, required=True)
    with _locked(store, snapshot_id):
        manifest = load_manifest(store, snapshot_id)
        source = manifest['source_dir']
        dirty = _git_text(source, 'status', '--porcelain=v1', '--untracked-files=all')
        if dirty:
            raise SnapshotError(f'Build left the snapshot worktree unclean: {dirty.splitlines()[:5]}')
        artifacts = _extension_files(source)
        if not artifacts:
            raise SnapshotError(f'No built extensions in {source}/flash_rt; the build did not complete')
        return _update_native(store, manifest, {'status': 'built', 'fingerprint': fingerprint, 'artifacts': artifacts})


def build_script(manifest, *, gpu_arch='80', cutlass_dir, python='python', extra_cmake_args=(),
                 fingerprint=None) -> str:
    """Bash text for an explicit build job; never executed by this library."""
    validate_manifest(manifest)
    if not isinstance(gpu_arch, str) or not _GPU_ARCH.fullmatch(gpu_arch):
        raise SnapshotError(f'gpu_arch must look like 80 or 120a, got {gpu_arch!r}')
    if not isinstance(cutlass_dir, str) or not os.path.isabs(cutlass_dir):
        raise SnapshotError('cutlass_dir must be an absolute path (third_party/cutlass is not in a worktree)')
    if not isinstance(python, str) or not python.strip():
        raise SnapshotError('python must be a nonempty interpreter path')
    if isinstance(extra_cmake_args, str) or not all(isinstance(item, str) and item for item in extra_cmake_args):
        raise SnapshotError('extra_cmake_args must be a sequence of nonempty strings')
    extra = list(extra_cmake_args)
    if fingerprint is None:
        description = {'native_inputs': manifest['native_inputs'], 'gpu_arch': gpu_arch,
                       'cutlass_dir': cutlass_dir, 'python': python, 'extra_cmake_args': extra}
        fingerprint = 'cmake-sha256:' + hashlib.sha256(
            json.dumps(description, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    _fingerprint(fingerprint, required=True)
    source = manifest['source_dir']
    store = str(Path(source).parent.parent)
    quote = shlex.quote
    cmake = ['cmake', '-S', '"$SRC"', '-B', '"$SRC/build"', f'-DGPU_ARCH={quote(gpu_arch)}',
             f'-DCUTLASS_DIR={quote(cutlass_dir)}', f'-DPython3_EXECUTABLE={quote(python)}',
             *map(quote, extra)]
    lines = [
        '#!/usr/bin/env bash',
        '# Generated by flash_rt.vla.autotune.snapshots build-script; review before running.',
        f'# Snapshot {manifest["snapshot_id"]}: base {manifest["base_commit"]}, '
        f'patch {manifest["patch_ref"] or "none"}.',
        '# Module hints (cluster specific, adjust): module load gcc/11.4.0 cuda/12.8',
        '# GPU_ARCH is pinned: CMake silently defaults it on nodes without a visible GPU.',
        '# FastWAM cached GEMM plans additionally need (not run here):',
        f'#   {quote(python)} "$SRC/scripts/build_gemm_plan.py" --output-dir "$SRC/flash_rt" --cuda-home "$CUDA_HOME"',
        'set -euo pipefail',
        f'SRC={quote(source)}',
        f'STORE={quote(store)}',
        'cd "$SRC"',
        ' '.join(cmake),
        'cmake --build "$SRC/build" -j "${SLURM_CPUS_ON_NODE:-$(nproc)}"',
        f'PYTHONPATH="$SRC" PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 {quote(python)} -B '
        f'-m flash_rt.vla.autotune.snapshots record-built --store "$STORE" '
        f'--snapshot-id {manifest["snapshot_id"]} --fingerprint {quote(fingerprint)}',
    ]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- verification

def check_candidate_source(manifest, source_ref, patch_ref, *, error=SourceIdentityError):
    """The candidate's claimed refs must be exactly the manifest's base/patch."""
    if manifest['base_commit'] != source_ref or manifest['patch_ref'] != patch_ref:
        raise error(f'Candidate source_ref/patch_ref ({source_ref}, {patch_ref}) do not match snapshot '
                    f'{manifest["snapshot_id"]} ({manifest["base_commit"]}, {manifest["patch_ref"]})')


def _status_entries(root, *args):
    output = _git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all', *args,
                  error=SourceIdentityError).stdout.decode('utf-8', 'surrogateescape')
    return [entry for entry in output.split('\0') if entry]


def _allowed_ignored(path, artifacts):
    parts = path.rstrip('/').split('/')
    return path in artifacts or parts[0] == 'build' or '__pycache__' in parts


def verify_source(manifest, import_root=None) -> dict:
    """Prove the imported flash_rt is exactly the clean, ready snapshot; return its identity."""
    error = SourceIdentityError
    try:
        validate_manifest(manifest)
    except SnapshotError as problem:
        raise error(f'Invalid source manifest: {problem}') from problem
    if not is_ready(manifest):
        raise error(f'Snapshot {manifest["snapshot_id"]} native state is {manifest["native"]["status"]}, not ready')
    if import_root is None:
        import flash_rt
        flash_file = os.path.realpath(flash_rt.__file__)
        root = os.path.dirname(os.path.dirname(flash_file))
    else:
        root = os.path.realpath(import_root)
        flash_file = os.path.join(root, 'flash_rt', '__init__.py')
        if not os.path.isfile(flash_file):
            raise error(f'No flash_rt package under import root {root}')
    expected_root = os.path.realpath(manifest['source_dir'])
    if root != expected_root:
        raise error(f'flash_rt is imported from {root}, not snapshot {manifest["snapshot_id"]} at {expected_root}')
    if _toplevel(root, error=error) != root:
        raise error(f'Import root {root} is not the top level of its own git worktree')
    head = _head_identity(root, error=error)
    for key, name in (('commit', 'snapshot_commit'), ('tree', 'snapshot_tree'), ('harness', 'harness_tree')):
        if head[key] != manifest[name]:
            raise error(f'HEAD {key} {head[key]} does not match manifest {name} {manifest[name]}')
    if manifest['patch_ref'] is not None and head['parent'] != manifest['base_commit']:
        raise error(f'Patched snapshot HEAD^1 {head["parent"]} is not base {manifest["base_commit"]}')
    if manifest['patch_ref'] is None and head['commit'] != manifest['base_commit']:
        raise error(f'Unpatched snapshot HEAD {head["commit"]} is not base {manifest["base_commit"]}')
    changed = _status_entries(root)
    if changed:
        raise error(f'Snapshot worktree has tracked or untracked changes: {changed[:10]}')
    if _git(root, 'diff', '--quiet', 'HEAD', check=False, error=error).returncode != 0:
        raise error('Snapshot worktree differs from HEAD (git diff --quiet HEAD failed)')
    _check_worktree_metadata(root, error=error)
    artifacts = manifest['native']['artifacts']
    ignored = [entry[3:] for entry in _status_entries(root, '--ignored=matching') if entry.startswith('!! ')]
    unexpected = [path for path in ignored if not _allowed_ignored(path, artifacts)]
    if unexpected:
        raise error(f'Snapshot contains unexpected ignored files: {unexpected[:10]}')
    extensions = _extension_files(root)
    if extensions != artifacts:
        raise error(f'Snapshot extensions {extensions} do not match manifest artifacts {artifacts}')
    return {'snapshot_id': manifest['snapshot_id'], 'source_dir': root, 'flash_rt_file': flash_file,
            'base_commit': manifest['base_commit'], 'patch_ref': manifest['patch_ref'],
            'head_commit': head['commit'], 'head_tree': head['tree'], 'head_parent': head['parent'],
            'harness_tree': head['harness'], 'tracked_clean': True, 'unexpected_files': [],
            'extensions': extensions,
            'python': {'version': platform.python_version(), 'executable': sys.executable},
            'env': {key: value for key, value in sorted(os.environ.items())
                    if key.startswith(IDENTITY_ENV_PREFIXES)}}


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description='Materialize, prepare and verify autotune source snapshots.')
    commands = parser.add_subparsers(dest='command', required=True)
    resolve = commands.add_parser('resolve', help='Resolve a ref to a full commit id')
    resolve.add_argument('--repo', required=True)
    resolve.add_argument('--ref', required=True)
    store = commands.add_parser('store-patch', help='Store patch bytes content-addressed')
    store.add_argument('--store', required=True)
    store.add_argument('--patch', required=True, help='Patch file (git diff --binary --full-index), or - for stdin')
    create = commands.add_parser('materialize', help='Create the snapshot worktree and its manifest')
    create.add_argument('--repo', required=True)
    create.add_argument('--store', required=True)
    create.add_argument('--base', required=True, help='Base ref; resolved to a full commit first')
    source = create.add_mutually_exclusive_group()
    source.add_argument('--patch-ref')
    source.add_argument('--patch', help='Patch file to store before materializing')
    adopt = commands.add_parser('adopt-native', help='Reuse prebuilt extensions or declare none')
    adopt.add_argument('--store', required=True)
    adopt.add_argument('--snapshot-id', required=True)
    adopt.add_argument('--prebuilt-root')
    adopt.add_argument('--fingerprint')
    adopt.add_argument('--none-expected', action='store_true')
    built = commands.add_parser('record-built', help='Record extensions after an in-place build')
    built.add_argument('--store', required=True)
    built.add_argument('--snapshot-id', required=True)
    built.add_argument('--fingerprint', required=True)
    verify = commands.add_parser('verify', help='Verify a snapshot source tree')
    verify.add_argument('--store', required=True)
    verify.add_argument('--snapshot-id', required=True)
    verify.add_argument('--import-root', help='Defaults to the snapshot source_dir')
    script = commands.add_parser('build-script', help='Render the bash build job for a snapshot')
    script.add_argument('--store', required=True)
    script.add_argument('--snapshot-id', required=True)
    script.add_argument('--cutlass-dir', required=True)
    script.add_argument('--gpu-arch', default='80')
    script.add_argument('--python', default='python')
    script.add_argument('--cmake-arg', action='append', default=[])
    script.add_argument('--fingerprint')
    script.add_argument('--out', help='Write the script here instead of embedding it in JSON')
    show = commands.add_parser('show', help='Print a manifest and whether it is ready')
    show.add_argument('--store', required=True)
    show.add_argument('--snapshot-id', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'resolve':
            output = {'ref': args.ref, 'commit': resolve_commit(args.repo, args.ref)}
        elif args.command == 'store-patch':
            data = sys.stdin.buffer.read() if args.patch == '-' else Path(args.patch).read_bytes()
            output = {'patch_ref': store_patch(args.store, data)}
        elif args.command == 'materialize':
            patch_ref = (store_patch(args.store, Path(args.patch).read_bytes()) if args.patch
                         else args.patch_ref)
            output = materialize(args.repo, args.store, resolve_commit(args.repo, args.base), patch_ref)
        elif args.command == 'adopt-native':
            output = adopt_native(args.store, args.snapshot_id, args.prebuilt_root,
                                  fingerprint=args.fingerprint, none_expected=args.none_expected)
        elif args.command == 'record-built':
            output = record_built(args.store, args.snapshot_id, fingerprint=args.fingerprint)
        elif args.command == 'verify':
            manifest = load_manifest(args.store, args.snapshot_id)
            output = verify_source(manifest, import_root=args.import_root or manifest['source_dir'])
        elif args.command == 'build-script':
            text = build_script(load_manifest(args.store, args.snapshot_id), gpu_arch=args.gpu_arch,
                                cutlass_dir=args.cutlass_dir, python=args.python,
                                extra_cmake_args=args.cmake_arg, fingerprint=args.fingerprint)
            if args.out:
                with open(args.out, 'x') as stream:
                    stream.write(text)
                output = {'snapshot_id': args.snapshot_id, 'script_path': os.path.realpath(args.out)}
            else:
                output = {'snapshot_id': args.snapshot_id, 'script': text}
        else:
            manifest = load_manifest(args.store, args.snapshot_id)
            output = {**manifest, 'manifest_path': str(manifest_path(args.store, args.snapshot_id)),
                      'ready': is_ready(manifest)}
    except (SnapshotError, SourceIdentityError) as error:
        print(json.dumps({'error': type(error).__name__, 'message': str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
