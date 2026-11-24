"""Source snapshots: content-addressed git worktrees bound to Candidate refs.

Every test builds its own temporary git repository with isolated git config;
nothing touches the FlashRT checkout that runs the tests.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune import snapshots as snap
from flash_rt.vla.autotune.schema import Candidate, validate_result


pytestmark = pytest.mark.skipif(shutil.which('git') is None, reason='git executable required')

# The fixture repository's measurement "harness" is this stub worker. It is
# importable as ``python -m flash_rt.vla.autotune.worker`` from a snapshot and
# writes a schema-3 bundle that quotes its own __file__: a measured one for the
# reference (so the batch proceeds) and a failed one for every other attempt.
STUB_WORKER = '''\
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ('--candidate', '--workload', '--output', '--attempt-id', '--comparison-id',
                 '--campaign-instance-id', '--warmup', '--repeats', '--reference', '--quality-gate',
                 '--source-manifest'):
        parser.add_argument(name)
    parser.add_argument('--baseline-reference', action='store_true')
    args = parser.parse_args()
    with open(args.candidate) as stream:
        candidate = json.load(stream)
    with open(args.source_manifest) as stream:
        manifest = json.load(stream)
    import flash_rt
    here = os.path.realpath(__file__)
    os.makedirs(args.output, exist_ok=True)
    record = {'schema_version': 3, 'candidate_id': candidate['candidate_id'],
              'attempt_id': args.attempt_id, 'comparison_id': args.comparison_id,
              'workload_id': candidate['workload_id'], 'candidate': candidate,
              'campaign_instance_id': args.campaign_instance_id, 'job_id': None,
              'stub_file': here, 'stub_flash_rt': os.path.realpath(flash_rt.__file__),
              'stub_cwd': os.path.realpath(os.getcwd()),
              'stub_pythonpath': os.environ.get('PYTHONPATH')}
    if args.baseline_reference:
        identity = {'snapshot_id': manifest['snapshot_id'], 'source_dir': manifest['source_dir'],
                    'flash_rt_file': os.path.realpath(flash_rt.__file__),
                    'base_commit': manifest['base_commit'], 'patch_ref': manifest['patch_ref'],
                    'head_commit': manifest['snapshot_commit'], 'head_tree': manifest['snapshot_tree'],
                    'head_parent': manifest['snapshot_parent'], 'harness_tree': manifest['harness_tree'],
                    'tracked_clean': True, 'unexpected_files': [],
                    'extensions': manifest['native']['artifacts'],
                    'python': {'version': sys.version.split()[0]}, 'env': {}}
        record.update(state='measured', source_identity=identity,
                      measurement={'warmup': 0, 'repeats': 1, 'latency_ms': [1.0], 'load_ms': 0.0,
                                   'prepare_ms': 0.0, 'sampler_metadata': [{'sampler': 'euler', 'nfe': 1}]},
                      backend_metadata={}, gpu={}, outputs_path=os.path.join(args.output, 'outputs.npz'),
                      quality={'status': 'passed', 'scope': 'numerical_replay', 'metrics': {}})
    else:
        record.update(state='failed', failure={'kind': 'stub', 'message': here})
    with open(os.path.join(args.output, 'result.json'), 'w') as stream:
        json.dump(record, stream)
    return 0 if record['state'] == 'measured' else 1


if __name__ == '__main__':
    raise SystemExit(main())
'''

FILES = {
    'flash_rt/__init__.py': '',
    'flash_rt/vla/__init__.py': '',
    'flash_rt/vla/autotune/__init__.py': '',
    'flash_rt/vla/autotune/worker.py': STUB_WORKER,
    'flash_rt/model.py': 'STEPS = 10\n',
    'csrc/k.cu': '__global__ void k() {}\n',
    'CMakeLists.txt': 'cmake_minimum_required(VERSION 3.18)\n',
    '.gitignore': '*.so\nbuild/\n__pycache__/\n',
}


# --------------------------------------------------------------------------- helpers

def git(cwd, *args, check=True):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    completed = subprocess.run(
        ['git', '-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost',
         '-c', 'commit.gpgsign=false', '-c', 'init.defaultBranch=main', '-C', str(cwd), *args],
        capture_output=True, env=env, check=False)
    if check and completed.returncode:
        raise AssertionError(completed.stderr.decode())
    return completed


def make_repo(path):
    path.mkdir(parents=True)
    for relpath, text in FILES.items():
        target = path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(path, 'init', '-q')
    git(path, 'add', '-A')
    git(path, 'commit', '-q', '-m', 'fixture base')
    return path


def head(repo):
    return git(repo, 'rev-parse', 'HEAD').stdout.decode().strip()


def make_patch(repo, edits, renames=()):
    """Bytes of `git diff --binary --full-index` for edits applied to HEAD; HEAD stays clean."""
    for old, new in renames:
        git(repo, 'mv', old, new)
    for relpath, text in edits.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(repo, 'add', '-A')
    patch = git(repo, 'diff', '--cached', '--binary', '--full-index', '-M').stdout
    git(repo, 'reset', '-q', '--hard')
    git(repo, 'clean', '-q', '-fdx', '-e', '*.so')
    assert patch
    return patch


def registered_worktrees(repo):
    output = git(repo, 'worktree', 'list', '--porcelain').stdout.decode()
    return [os.path.realpath(line.split(' ', 1)[1]) for line in output.splitlines() if line.startswith('worktree ')]


def ready_snapshot(repo, store, patch_edits=None, *, base=None):
    """Materialize (optionally patched) and mark it as loading no extension."""
    base = base or head(repo)
    patch_ref = None if patch_edits is None else snap.store_patch(store, make_patch(repo, patch_edits))
    manifest = snap.materialize(repo, store, base, patch_ref)
    return snap.adopt_native(store, manifest['snapshot_id'], None, fingerprint=None, none_expected=True)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / 'repo')


@pytest.fixture
def store(tmp_path):
    return tmp_path / 'store'


# --------------------------------------------------------------------------- A. provenance fields

def _candidate(**changes):
    return Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames', **changes)


def test_candidate_provenance_fields_are_emitted_only_when_set():
    plain = _candidate()
    assert 'tool' not in plain.to_dict() and 'stage' not in plain.to_dict()
    assert list(plain.to_dict()) == ['candidate_id', 'backend', 'workload_id', 'parent_id', 'hypothesis',
                                     'origin', 'patch_ref', 'source_ref']
    tagged = _candidate(origin='agent', tool='kernel.forgekernel', stage='denoise.velocity')
    record = json.loads(json.dumps(tagged.to_dict()))
    assert record['tool'] == 'kernel.forgekernel' and record['stage'] == 'denoise.velocity'
    assert Candidate.from_dict(record) == tagged
    assert Candidate.from_dict({**record, 'stage': None}).to_dict().get('stage', 'absent') == 'absent'


@pytest.mark.parametrize('field,value', [('tool', 'Kernel'), ('tool', ''), ('stage', '-x'),
                                         ('stage', 'a b'), ('tool', 3)])
def test_candidate_provenance_fields_reject_non_identifiers(field, value):
    with pytest.raises(ValueError, match=field):
        _candidate(**{field: value})


# --------------------------------------------------------------------------- B. refs and patches

def test_resolve_commit_returns_full_hex_and_rejects_unknown_refs(repo):
    commit = snap.resolve_commit(repo, 'HEAD')
    assert commit == head(repo) and len(commit) == 40
    assert snap.resolve_commit(repo, 'main') == commit
    with pytest.raises(snap.SnapshotError, match='rev-parse'):
        snap.resolve_commit(repo, 'no-such-branch')
    with pytest.raises(snap.SnapshotError):
        snap.resolve_commit(repo, '')


def test_store_patch_is_content_addressed_idempotent_and_detects_conflicts(store):
    data = b'diff --git a/x b/x\n'
    ref = snap.store_patch(store, data)
    assert ref == 'sha256:' + hashlib.sha256(data).hexdigest()
    assert snap.store_patch(store, data) == ref
    path = snap.patch_file(store, ref)
    assert path.read_bytes() == data and not os.access(path, os.W_OK)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_bytes(b'tampered')
    with pytest.raises(snap.SnapshotError, match='does not contain'):
        snap.store_patch(store, data)
    with pytest.raises(snap.SnapshotError, match='does not hash'):
        snap.patch_file(store, ref)
    with pytest.raises(snap.SnapshotError):
        snap.store_patch(store, b'')
    with pytest.raises(snap.SnapshotError, match='not in the store'):
        snap.patch_file(store, 'sha256:' + '0' * 64)


def test_snapshot_id_is_deterministic_and_validated():
    base = 'a' * 40
    ref = 'sha256:' + 'b' * 64
    expected = hashlib.sha256(f'flash_rt-snapshot-v1\n{base}\n{"b" * 64}\n'.encode()).hexdigest()[:20]
    assert snap.snapshot_id(base, ref) == expected == snap.snapshot_id(base, ref)
    assert snap.snapshot_id(base) == hashlib.sha256(f'flash_rt-snapshot-v1\n{base}\n-\n'.encode()).hexdigest()[:20]
    assert snap.snapshot_id(base) != expected
    for bad_base, bad_ref in (('main', None), ('A' * 40, None), (base, 'b' * 64), (base, 'sha256:xyz')):
        with pytest.raises(snap.SnapshotError):
            snap.snapshot_id(bad_base, bad_ref)


@pytest.mark.parametrize('path', ['flash_rt/vla/autotune/worker.py', 'flash_rt/vla/autotune', '.gitignore',
                                  '.gitattributes', '.git/config', 'flash_rt/.gitignore', '/abs/x', '../x',
                                  'flash_rt/../x', 'flash_rt//x', ''])
def test_check_patch_paths_rejects_harness_git_metadata_and_escaping_paths(path):
    with pytest.raises(snap.SnapshotError):
        snap.check_patch_paths(['flash_rt/model.py', path])


def test_check_patch_paths_accepts_source_and_native_paths():
    snap.check_patch_paths(['flash_rt/model.py', 'csrc/k.cu', 'CMakeLists.txt', 'flash_rt/vla/other.py'])
    snap.check_patch_paths(['cmake/FindX.cmake', 'scripts/build_gemm_plan.py', 'docs/notes.md',
                            'tests/test_x.py', 'examples/run.py', 'benchmarks/bench.py', 'README.md',
                            'flash_rt/sub/__init__.py', 'csrc/attention/pkg/__init__.py'])
    with pytest.raises(snap.SnapshotError, match='no paths'):
        snap.check_patch_paths([])


@pytest.mark.parametrize('path', ['sitecustomize.py', 'usercustomize.py', 'json.py', 'evil.pth', 'json.so',
                                  'numpy.cpython-312-x86_64-linux-gnu.so', 'setup.py', 'json/__init__.py',
                                  'numpy/core.py', 'flash_wm/src/x.py', 'training/x.py', 'docs/__init__.py',
                                  'tests/__init__.py', 'scripts/__init__.pyc'])
def test_check_patch_paths_rejects_paths_importable_from_the_snapshot_root(path):
    # The snapshot root is first on the worker's PYTHONPATH: a top-level module,
    # .pth/.so file or package other than flash_rt would run ahead of the stdlib.
    with pytest.raises(snap.SnapshotError, match='top-level'):
        snap.check_patch_paths(['flash_rt/model.py', path])


def test_patch_adding_a_root_module_is_rejected_before_any_worktree(repo, store):
    ref = snap.store_patch(store, make_patch(repo, {'sitecustomize.py': 'print("first")\n',
                                                   'flash_rt/model.py': 'STEPS = 4\n'}))
    before = registered_worktrees(repo)
    with pytest.raises(snap.SnapshotError, match="top-level file 'sitecustomize.py'"):
        snap.materialize(repo, store, head(repo), ref)
    assert registered_worktrees(repo) == before
    assert not (store / 'snapshots').exists() or not any((store / 'snapshots').iterdir())


def test_patch_paths_lists_both_sides_of_a_rename(repo, store):
    ref = snap.store_patch(store, make_patch(repo, {}, renames=[('flash_rt/model.py', 'flash_rt/moved.py')]))
    assert set(snap.patch_paths(repo, snap.patch_file(store, ref))) == {'flash_rt/model.py', 'flash_rt/moved.py'}


# --------------------------------------------------------------------------- B. materialize

def test_materialize_is_deterministic_and_records_identity(repo, tmp_path):
    base = head(repo)
    patch = make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'})
    manifests = []
    for name in ('store-a', 'store-b'):
        store = tmp_path / name
        manifests.append(snap.materialize(repo, store, base, snap.store_patch(store, patch)))
    first, second = manifests
    assert first['snapshot_id'] == second['snapshot_id']
    assert (first['snapshot_commit'], first['snapshot_tree']) == (second['snapshot_commit'], second['snapshot_tree'])
    assert first['source_dir'] != second['source_dir']
    assert first['schema'] == snap.SNAPSHOT_SCHEMA and first['base_commit'] == base
    assert first['snapshot_parent'] == base and first['snapshot_commit'] != base
    assert first['harness_tree'] == git(repo, 'rev-parse', f'{base}:flash_rt/vla/autotune').stdout.decode().strip()
    assert first['native'] == {'status': 'pending', 'fingerprint': None, 'artifacts': {}}
    assert first['native_inputs']['cmake'] is None and first['native_inputs']['scripts/build_gemm_plan.py'] is None
    assert first['native_inputs']['csrc'] is not None and first['native_inputs']['CMakeLists.txt'] is not None
    assert first['source_dir'] == os.path.realpath(first['source_dir'])
    source = Path(first['source_dir'])
    assert (source / 'flash_rt/model.py').read_text() == 'STEPS = 4\n'
    assert git(source, 'log', '-1', '--format=%an|%ae|%aI|%s').stdout.decode().strip() == (
        f'flash_rt-autotune|autotune@localhost|1970-01-01T00:00:00Z|autotune-patch {first["patch_ref"]}')
    ref = git(repo, 'rev-parse', f'refs/autotune/snapshots/{first["snapshot_id"]}').stdout.decode().strip()
    assert ref == first['snapshot_commit']
    worktrees = git(repo, 'worktree', 'list', '--porcelain').stdout.decode()
    assert f'locked autotune {first["snapshot_id"]}' in worktrees
    store = tmp_path / 'store-a'
    assert snap.load_manifest(store, first['snapshot_id']) == first
    assert snap.materialize(repo, store, base, first['patch_ref']) == first  # Idempotent re-validation.
    assert not (store / 'snapshots' / f'{first["snapshot_id"]}.lock').exists()
    assert snap.is_ready(first) is False


def test_unpatched_and_differently_patched_snapshots_have_distinct_ids(repo, store):
    base = head(repo)
    plain = snap.materialize(repo, store, base)
    assert plain['snapshot_commit'] == base and plain['snapshot_parent'] is None and plain['patch_ref'] is None
    one = snap.materialize(repo, store, base, snap.store_patch(store, make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'})))
    two = snap.materialize(repo, store, base, snap.store_patch(store, make_patch(repo, {'flash_rt/model.py': 'STEPS = 5\n'})))
    assert len({plain['snapshot_id'], one['snapshot_id'], two['snapshot_id']}) == 3
    assert len({one['snapshot_tree'], two['snapshot_tree'], plain['snapshot_tree']}) == 3
    assert one['harness_tree'] == two['harness_tree'] == plain['harness_tree']


@pytest.mark.parametrize('edits,renames', [
    ({'flash_rt/vla/autotune/worker.py': 'print("changed timing boundary")\n'}, ()),
    ({'flash_rt/vla/autotune/extra.py': 'X = 1\n'}, ()),
    ({'.gitignore': '*.so\nbuild/\n__pycache__/\n*.py\n'}, ()),
    ({}, [('flash_rt/vla/autotune/worker.py', 'flash_rt/worker.py')]),
])
def test_patch_touching_harness_or_gitignore_is_rejected_before_any_worktree(repo, store, edits, renames):
    ref = snap.store_patch(store, make_patch(repo, edits, renames))
    before = registered_worktrees(repo)
    with pytest.raises(snap.SnapshotError, match='pinned|git metadata'):
        snap.materialize(repo, store, head(repo), ref)
    assert registered_worktrees(repo) == before
    assert not (store / 'snapshots').exists() or not any((store / 'snapshots').iterdir())


def _symlink_patch(repo, outside):
    """A patch replacing flash_rt/model.py by a symlink to a file outside the repository."""
    (repo / 'flash_rt' / 'model.py').unlink()
    (repo / 'flash_rt' / 'model.py').symlink_to(outside)
    git(repo, 'add', '-A')
    patch = git(repo, 'diff', '--cached', '--binary', '--full-index').stdout
    git(repo, 'reset', '-q', '--hard')
    return patch


def _gitlink_patch(repo, outside):
    """A patch adding a gitlink (submodule entry) under flash_rt."""
    git(repo, 'update-index', '--add', '--cacheinfo', f'160000,{head(repo)},flash_rt/sub')
    patch = git(repo, 'diff', '--cached', '--binary', '--full-index').stdout
    git(repo, 'reset', '-q', '--hard')
    return patch


@pytest.mark.parametrize('make,kind', [(_symlink_patch, 'symlink'), (_gitlink_patch, 'gitlink')])
def test_patch_creating_a_symlink_or_gitlink_is_refused(repo, store, tmp_path, make, kind):
    outside = tmp_path / 'outside.py'
    outside.write_text('STEPS = 1\n')
    patch = make(repo, outside)
    assert b'mode 120000' in patch or b'mode 160000' in patch
    ref = snap.store_patch(store, patch)
    identifier = snap.snapshot_id(head(repo), ref)
    before = registered_worktrees(repo)
    with pytest.raises(snap.SnapshotError, match=f'symlinks or gitlinks.*{kind}'):
        snap.materialize(repo, store, head(repo), ref)
    assert registered_worktrees(repo) == before
    assert not snap.manifest_path(store, identifier).exists()
    assert not os.path.lexists(store / 'snapshots' / identifier)


@pytest.mark.parametrize('target,valid', [
    ('../../.agents/skills/demo', True),
    ('../../.agents/skills/elsewhere', False),
])
def test_snapshot_only_allows_exact_tracked_local_skill_alias(repo, store, target, valid):
    directory = repo / '.agents/skills/demo'
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text('# Fixture skill\n')
    link = repo / '.claude/skills/demo'
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    git(repo, 'add', '.agents', '.claude')
    git(repo, 'commit', '-qm', 'install skill alias')
    if valid:
        manifest = snap.materialize(repo, store, head(repo))
        assert Path(manifest['source_dir'], '.claude/skills/demo/SKILL.md').is_file()
    else:
        with pytest.raises(snap.SnapshotError, match='symlinks or gitlinks'):
            snap.materialize(repo, store, head(repo))


def test_base_tracking_a_symlink_cannot_be_materialized(repo, store, tmp_path):
    (tmp_path / 'outside.py').write_text('X = 1\n')
    (repo / 'flash_rt' / 'linked.py').symlink_to(tmp_path / 'outside.py')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'track a symlink')
    with pytest.raises(snap.SnapshotError, match='symlinks or gitlinks'):
        snap.materialize(repo, store, head(repo))
    assert registered_worktrees(repo) == [os.path.realpath(repo)]


def test_verify_source_rejects_a_tracked_symlink(repo, store, tmp_path, monkeypatch):
    # Defence in depth: a snapshot that tracks a symlink (created here with the
    # materialize check disabled) never verifies, whatever the link target holds.
    outside = tmp_path / 'outside.py'
    outside.write_text('STEPS = 1\n')
    with monkeypatch.context() as patched:
        patched.setattr(snap, '_patched_special_paths', lambda source, base: [])
        patched.setattr(snap, '_check_worktree_metadata', lambda root, *, error: None)
        manifest = snap.materialize(repo, store, head(repo), snap.store_patch(store, _symlink_patch(repo, outside)))
        manifest = snap.adopt_native(store, manifest['snapshot_id'], None, fingerprint=None, none_expected=True)
    assert os.path.islink(Path(manifest['source_dir']) / 'flash_rt' / 'model.py')
    with pytest.raises(snap.SourceIdentityError, match=r"symlinks or gitlinks.*flash_rt/model.py \(symlink\)"):
        snap.verify_source(manifest, import_root=manifest['source_dir'])


def test_non_applying_patch_fails_cleanly_and_can_be_retried(repo, store):
    base = head(repo)
    patch = make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'}).replace(b'-STEPS = 10', b'-STEPS = 99')
    ref = snap.store_patch(store, patch)
    identifier = snap.snapshot_id(base, ref)
    before = registered_worktrees(repo)
    with pytest.raises(snap.SnapshotError, match='apply'):
        snap.materialize(repo, store, base, ref)
    assert registered_worktrees(repo) == before
    assert not snap.manifest_path(store, identifier).exists()
    assert not (store / 'snapshots' / identifier).exists()
    assert not (store / 'snapshots' / f'{identifier}.lock').exists()
    assert git(repo, 'rev-parse', '--verify', '--quiet', f'refs/autotune/snapshots/{identifier}',
               check=False).returncode == 1
    good = snap.materialize(repo, store, base, snap.store_patch(store, make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'})))
    assert snap.load_manifest(store, good['snapshot_id']) == good


def test_incomplete_snapshot_without_manifest_is_removed_and_redone(repo, store):
    base = head(repo)
    ref = snap.store_patch(store, make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'}))
    first = snap.materialize(repo, store, base, ref)
    snap.manifest_path(store, first['snapshot_id']).unlink()  # Crash before publication.
    (Path(first['source_dir']) / 'partial.txt').write_text('left over')
    again = snap.materialize(repo, store, base, ref)
    assert again == first and not (Path(again['source_dir']) / 'partial.txt').exists()


def test_concurrent_materialize_is_refused_by_the_lock(repo, store):
    base = head(repo)
    identifier = snap.snapshot_id(base)
    (store / 'snapshots').mkdir(parents=True)
    (store / 'snapshots' / f'{identifier}.lock').write_text('123\n')
    with pytest.raises(snap.SnapshotError, match='locked'):
        snap.materialize(repo, store, base)
    assert not (store / 'snapshots' / identifier).exists()


def test_load_manifest_rejects_missing_and_inconsistent_records(repo, store):
    with pytest.raises(snap.SnapshotError, match='not materialized'):
        snap.load_manifest(store, '0' * 20)
    manifest = snap.materialize(repo, store, head(repo))
    path = snap.manifest_path(store, manifest['snapshot_id'])
    for change in ({'snapshot_tree': 'x'}, {'patch_ref': 'sha256:' + 'c' * 64}, {'extra': 1},
                   {'native': {'status': 'built', 'fingerprint': None, 'artifacts': {}}}):
        path.write_text(json.dumps({**manifest, **change}))
        with pytest.raises(snap.SnapshotError):
            snap.load_manifest(store, manifest['snapshot_id'])


# --------------------------------------------------------------------------- B. native extensions

def test_adopt_native_reuses_prebuilt_extensions_when_native_inputs_are_equal(repo, store):
    (repo / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'\x7fELF prebuilt kernels')
    manifest = snap.materialize(repo, store, head(repo),
                                snap.store_patch(store, make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'})))
    adopted = snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80-cuda12.8')
    digest = hashlib.sha256(b'\x7fELF prebuilt kernels').hexdigest()
    assert adopted['native'] == {'status': 'reused', 'fingerprint': 'sm80-cuda12.8',
                                 'artifacts': {'flash_rt/flash_rt_kernels.so': digest}}
    assert snap.is_ready(adopted) and snap.load_manifest(store, manifest['snapshot_id']) == adopted
    assert (Path(adopted['source_dir']) / 'flash_rt/flash_rt_kernels.so').read_bytes() == b'\x7fELF prebuilt kernels'
    assert snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80-cuda12.8') == adopted
    with pytest.raises(snap.SnapshotError, match='already reused'):
        snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='other-build')
    identity = snap.verify_source(adopted, import_root=adopted['source_dir'])
    assert identity['extensions'] == {'flash_rt/flash_rt_kernels.so': digest}


def test_adopt_native_requires_a_rebuild_when_csrc_differs(repo, store):
    (repo / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'prebuilt')
    manifest = snap.materialize(repo, store, head(repo),
                                snap.store_patch(store, make_patch(repo, {'csrc/k.cu': '__global__ void k2() {}\n'})))
    with pytest.raises(snap.SnapshotError, match='native inputs differ: rebuild required'):
        snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    assert snap.load_manifest(store, manifest['snapshot_id'])['native']['status'] == 'pending'
    assert not (Path(manifest['source_dir']) / 'flash_rt/flash_rt_kernels.so').exists()
    # An uncommitted native edit in the prebuilt checkout is not "equal" either.
    plain = snap.materialize(repo, store, head(repo))
    (repo / 'csrc' / 'k.cu').write_text('// edited after build\n')
    with pytest.raises(snap.SnapshotError, match='rebuild required'):
        snap.adopt_native(store, plain['snapshot_id'], repo, fingerprint='sm80')
    # The external build path: build in place, then record.
    (Path(manifest['source_dir']) / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'rebuilt')
    built = snap.record_built(store, manifest['snapshot_id'], fingerprint='sm80')
    assert built['native']['status'] == 'built'
    assert built['native']['artifacts'] == {'flash_rt/flash_rt_kernels.so': hashlib.sha256(b'rebuilt').hexdigest()}


@pytest.mark.parametrize('none_expected', [True, False])
def test_adopt_native_on_a_ready_snapshot_changes_no_file(repo, store, none_expected):
    manifest = ready_snapshot(repo, store, base=head(repo)) if none_expected else None
    if not none_expected:
        (repo / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'kernels v1')
        manifest = snap.materialize(repo, store, head(repo))
        manifest = snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    source = Path(manifest['source_dir'])
    before = snap._extension_files(source)
    # A rebuilt prebuilt checkout with a new and a changed extension.
    (repo / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'kernels v2')
    (repo / 'flash_rt' / 'flash_rt_extra.so').write_bytes(b'extra kernels')
    with pytest.raises(snap.SnapshotError, match='native state is already'):
        snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    assert snap._extension_files(source) == before
    assert snap.load_manifest(store, manifest['snapshot_id']) == manifest
    snap.verify_source(manifest, import_root=manifest['source_dir'])


def test_adopt_native_failure_removes_the_extensions_it_copied(repo, store, monkeypatch):
    (repo / 'flash_rt' / 'a_kernels.so').write_bytes(b'first')
    (repo / 'flash_rt' / 'b_kernels.so').write_bytes(b'second')
    manifest = snap.materialize(repo, store, head(repo))
    copy = snap._copy_artifact

    def copy_then_fail(source_file, destination, digest):
        if destination.name == 'b_kernels.so':
            raise snap.SnapshotError('injected copy failure')
        return copy(source_file, destination, digest)
    monkeypatch.setattr(snap, '_copy_artifact', copy_then_fail)
    with pytest.raises(snap.SnapshotError, match='injected'):
        snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    assert snap._extension_files(manifest['source_dir']) == {}
    assert snap.load_manifest(store, manifest['snapshot_id'])['native']['status'] == 'pending'
    monkeypatch.setattr(snap, '_copy_artifact', copy)
    adopted = snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    assert set(adopted['native']['artifacts']) == {'flash_rt/a_kernels.so', 'flash_rt/b_kernels.so'}


def test_record_built_without_output_leaves_the_snapshot_not_ready(repo, store):
    manifest = snap.materialize(repo, store, head(repo))
    with pytest.raises(snap.SnapshotError, match='did not complete'):
        snap.record_built(store, manifest['snapshot_id'], fingerprint='sm80')
    assert snap.is_ready(snap.load_manifest(store, manifest['snapshot_id'])) is False


def test_none_expected_marks_pure_python_snapshots_ready(repo, store):
    manifest = ready_snapshot(repo, store)
    assert manifest['native'] == {'status': 'none', 'fingerprint': None, 'artifacts': {}}
    assert snap.is_ready(manifest)


def test_build_script_pins_arch_and_records_the_build(repo, store):
    manifest = snap.materialize(repo, store, head(repo))
    script = snap.build_script(manifest, cutlass_dir='/shared/cutlass', python='/env/bin/python',
                               extra_cmake_args=['-DFLASHRT_ENABLE_HYVLA=ON'])
    assert '-DGPU_ARCH=80' in script and '-DCUTLASS_DIR=/shared/cutlass' in script
    assert '-DPython3_EXECUTABLE=/env/bin/python' in script and '-DFLASHRT_ENABLE_HYVLA=ON' in script
    assert f'record-built --store "$STORE" --snapshot-id {manifest["snapshot_id"]}' in script
    assert 'module load' in script
    if shutil.which('bash'):
        assert subprocess.run(['bash', '-n'], input=script.encode(), check=False).returncode == 0
    with pytest.raises(snap.SnapshotError):
        snap.build_script(manifest, cutlass_dir='relative/cutlass')
    with pytest.raises(snap.SnapshotError):
        snap.build_script(manifest, cutlass_dir='/c', gpu_arch='80; rm -rf /')


# --------------------------------------------------------------------------- B. verify_source

def test_verify_source_passes_on_the_clean_snapshot(repo, store):
    manifest = ready_snapshot(repo, store, {'flash_rt/model.py': 'STEPS = 4\n'})
    (Path(manifest['source_dir']) / 'build').mkdir()
    (Path(manifest['source_dir']) / 'build' / 'CMakeCache.txt').write_text('ignored build output')
    identity = snap.verify_source(manifest, import_root=manifest['source_dir'])
    assert identity['snapshot_id'] == manifest['snapshot_id']
    assert identity['head_commit'] == manifest['snapshot_commit'] and identity['head_tree'] == manifest['snapshot_tree']
    assert identity['head_parent'] == manifest['base_commit'] and identity['harness_tree'] == manifest['harness_tree']
    assert identity['tracked_clean'] is True and identity['unexpected_files'] == [] and identity['extensions'] == {}
    assert identity['flash_rt_file'] == os.path.join(manifest['source_dir'], 'flash_rt', '__init__.py')
    assert set(identity['python']) == {'version', 'executable'}


def test_verify_source_resolves_symlinked_import_roots(repo, store, tmp_path):
    manifest = ready_snapshot(repo, store)
    link = tmp_path / 'link-to-snapshot'
    link.symlink_to(manifest['source_dir'])
    assert snap.verify_source(manifest, import_root=link)['source_dir'] == manifest['source_dir']


def _tracked_edit(source):
    (source / 'flash_rt' / 'model.py').write_text('STEPS = 1\n')


def _stray_python(source):
    (source / 'flash_rt' / 'injected.py').write_text('import os\n')


def _stray_extension(source):
    (source / 'flash_rt' / 'stray.so').write_bytes(b'unexpected extension')


@pytest.mark.parametrize('mutate,message', [(_tracked_edit, 'changes'), (_stray_python, 'changes'),
                                            (_stray_extension, 'unexpected ignored')])
def test_verify_source_rejects_modified_snapshots(repo, store, mutate, message):
    manifest = ready_snapshot(repo, store)
    mutate(Path(manifest['source_dir']))
    with pytest.raises(snap.SourceIdentityError, match=message):
        snap.verify_source(manifest, import_root=manifest['source_dir'])


def _skip_worktree(source):
    git(source, 'update-index', '--skip-worktree', 'flash_rt/vla/autotune/worker.py')
    (source / 'flash_rt' / 'vla' / 'autotune' / 'worker.py').write_text('# changed timing boundary\n')


def _assume_unchanged(source):
    git(source, 'update-index', '--assume-unchanged', 'flash_rt/model.py')
    (source / 'flash_rt' / 'model.py').write_text('STEPS = 1\n')


def _sparse_checkout(source):
    git(source, 'config', 'core.sparseCheckout', 'true')


@pytest.mark.parametrize('mutate,message', [(_skip_worktree, 'hidden from git status.*S flash_rt/vla'),
                                            (_assume_unchanged, 'hidden from git status.*h flash_rt/model'),
                                            (_sparse_checkout, 'sparseCheckout')])
def test_verify_source_rejects_index_flags_that_hide_modifications(repo, store, mutate, message):
    manifest = ready_snapshot(repo, store)
    mutate(Path(manifest['source_dir']))
    with pytest.raises(snap.SourceIdentityError, match=message):
        snap.verify_source(manifest, import_root=manifest['source_dir'])


def test_verify_source_rejects_a_foreign_import_root(repo, store, tmp_path):
    manifest = ready_snapshot(repo, store)
    with pytest.raises(snap.SourceIdentityError, match='imported from'):
        snap.verify_source(manifest, import_root=repo)
    other = ready_snapshot(repo, store, {'flash_rt/model.py': 'STEPS = 4\n'})
    with pytest.raises(snap.SourceIdentityError, match='imported from'):
        snap.verify_source(manifest, import_root=other['source_dir'])
    with pytest.raises(snap.SourceIdentityError, match='No flash_rt'):
        snap.verify_source(manifest, import_root=tmp_path)
    with pytest.raises(snap.SourceIdentityError, match='imported from'):
        snap.verify_source(manifest)  # The test process imports the FlashRT checkout, not the snapshot.


def test_verify_source_rejects_wrong_extension_bytes_and_unready_snapshots(repo, store):
    (repo / 'flash_rt' / 'flash_rt_kernels.so').write_bytes(b'kernels v1')
    manifest = snap.materialize(repo, store, head(repo))
    with pytest.raises(snap.SourceIdentityError, match='not ready'):
        snap.verify_source(manifest, import_root=manifest['source_dir'])
    adopted = snap.adopt_native(store, manifest['snapshot_id'], repo, fingerprint='sm80')
    snap.verify_source(adopted, import_root=adopted['source_dir'])
    library = Path(adopted['source_dir']) / 'flash_rt' / 'flash_rt_kernels.so'
    os.chmod(library, stat.S_IRUSR | stat.S_IWUSR)
    library.write_bytes(b'kernels v2')
    with pytest.raises(snap.SourceIdentityError, match='extensions'):
        snap.verify_source(adopted, import_root=adopted['source_dir'])
    library.unlink()
    with pytest.raises(snap.SourceIdentityError, match='extensions'):
        snap.verify_source(adopted, import_root=adopted['source_dir'])


def test_verify_source_rejects_a_manifest_for_another_commit(repo, store):
    manifest = ready_snapshot(repo, store)
    forged = {**manifest, 'snapshot_tree': '0' * 40}
    with pytest.raises(snap.SourceIdentityError, match='tree'):
        snap.verify_source(forged, import_root=manifest['source_dir'])


# --------------------------------------------------------------------------- B. CLI

def test_cli_round_trip_reports_json_and_errors(repo, store, tmp_path, capsys):
    def run(*argv):
        code = snap.main([str(item) for item in argv])
        captured = capsys.readouterr()
        return code, (json.loads(captured.out) if captured.out else json.loads(captured.err))
    code, resolved = run('resolve', '--repo', repo, '--ref', 'HEAD')
    assert code == 0 and resolved['commit'] == head(repo)
    patch = tmp_path / 'change.patch'
    patch.write_bytes(make_patch(repo, {'flash_rt/model.py': 'STEPS = 4\n'}))
    code, stored = run('store-patch', '--store', store, '--patch', patch)
    assert code == 0 and stored['patch_ref'].startswith('sha256:')
    code, manifest = run('materialize', '--repo', repo, '--store', store, '--base', 'main', '--patch', patch)
    assert code == 0 and manifest['patch_ref'] == stored['patch_ref']
    identifier = manifest['snapshot_id']
    code, shown = run('show', '--store', store, '--snapshot-id', identifier)
    assert code == 0 and shown['ready'] is False and shown['manifest_path'].endswith(f'{identifier}.json')
    code, error = run('verify', '--store', store, '--snapshot-id', identifier)
    assert code == 2 and error['error'] == 'SourceIdentityError'
    code, adopted = run('adopt-native', '--store', store, '--snapshot-id', identifier, '--none-expected')
    assert code == 0 and adopted['native']['status'] == 'none'
    code, identity = run('verify', '--store', store, '--snapshot-id', identifier)
    assert code == 0 and identity['head_commit'] == manifest['snapshot_commit']
    code, script = run('build-script', '--store', store, '--snapshot-id', identifier, '--cutlass-dir', '/c')
    assert code == 0 and 'cmake --build' in script['script']


# --------------------------------------------------------------------------- C. schema 3 results

def measured_v3(manifest, candidate, **changes):
    identity = {'snapshot_id': manifest['snapshot_id'], 'source_dir': manifest['source_dir'],
                'flash_rt_file': os.path.join(manifest['source_dir'], 'flash_rt', '__init__.py'),
                'base_commit': manifest['base_commit'], 'patch_ref': manifest['patch_ref'],
                'head_commit': manifest['snapshot_commit'], 'head_tree': manifest['snapshot_tree'],
                'head_parent': manifest['snapshot_parent'], 'harness_tree': manifest['harness_tree'],
                'tracked_clean': True, 'unexpected_files': [], 'extensions': {},
                'python': {'version': '3.11.0'}, 'env': {}}
    record = {'schema_version': 3, 'candidate_id': candidate.candidate_id, 'attempt_id': 'a1',
              'comparison_id': 'c1', 'workload_id': candidate.workload_id, 'candidate': candidate.to_dict(),
              'campaign_instance_id': 'instance', 'job_id': '123', 'state': 'measured',
              'source_identity': identity,
              'measurement': {'warmup': 0, 'repeats': 1, 'latency_ms': [1.0], 'load_ms': 0.0, 'prepare_ms': 0.0,
                              'sampler_metadata': [{'sampler': 'euler', 'nfe': 2}]},
              'backend_metadata': {}, 'gpu': {'job_id': '123'}, 'outputs_path': '/x/outputs.npz',
              'quality': {'status': 'passed', 'scope': 'numerical_replay', 'metrics': {}}}
    record.update(changes)
    return record


def test_validate_result_accepts_complete_v3_and_requires_identity_when_measured():
    manifest = {'snapshot_id': snap.snapshot_id('a' * 40), 'source_dir': '/store/snapshots/x',
                'base_commit': 'a' * 40, 'patch_ref': None, 'snapshot_commit': 'a' * 40,
                'snapshot_tree': 'b' * 40, 'snapshot_parent': None, 'harness_tree': 'c' * 40}
    candidate = _candidate(source_ref='a' * 40)
    record = measured_v3(manifest, candidate)
    validate_result(record)
    for broken in ({'source_identity': None},
                   {'source_identity': {**record['source_identity'], 'tracked_clean': False}},
                   {'source_identity': {**record['source_identity'], 'head_tree': 'short'}},
                   {'source_identity': {**record['source_identity'], 'patch_ref': 'sha256:' + 'd' * 64}},
                   {'source_identity': {key: value for key, value in record['source_identity'].items()
                                        if key != 'extensions'}}):
        with pytest.raises(ValueError, match='source_identity'):
            validate_result({**record, **broken})
    failed = {key: value for key, value in record.items()
              if key not in ('measurement', 'backend_metadata', 'gpu', 'outputs_path', 'quality', 'source_identity')}
    validate_result({**failed, 'state': 'failed', 'failure': {'kind': 'source_identity', 'message': 'wrong tree'}})
    with pytest.raises(ValueError, match='schema_version'):
        validate_result({**record, 'schema_version': 4})
