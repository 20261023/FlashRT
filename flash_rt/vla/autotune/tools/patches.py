"""Turn a source change in a git worktree into a stored PatchProposal and a Candidate.

``proposal_from_worktree`` stores ``git diff --binary --full-index <base>`` of
the worktree: committed (since base), staged and unstaged changes to TRACKED
files. Untracked, non-ignored files are refused unless ``include_untracked``;
then they are marked intent-to-add (``git add -N``) in a temporary copy of the
index (``GIT_INDEX_FILE``), so the diff contains them as new files while the
worktree's real index stays untouched. Before storing, the patch must touch
only patchable paths (``snapshots.check_patch_paths``: never the measurement
harness ``flash_rt/vla/autotune`` or git metadata, and only allowlisted top-level
directories and files, so nothing importable lands at the snapshot root) and
must apply to ``base`` (checked
with ``git apply --cached --check`` against a temporary index read from base).

``candidate_from_proposal`` builds the snapshot-mode Candidate: 40-hex
``source_ref`` = base, ``sha256:`` ``patch_ref``, ``origin`` (default 'agent';
pass 'human' for a patch the human wrote). The patch
becomes executable only after ``snapshots.materialize`` + native adoption/build
and ``controller init --snapshot-store``.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import tempfile

from flash_rt.vla.registry import validate_config

from .. import snapshots
from ..schema import Candidate, decode_backend, encode_backend
from ..stages import STAGE_IDS
from .artifacts import PatchProposal


ORIGINS = ('human', 'search', 'agent')
DIFF_ARGS = ('diff', '--binary', '--full-index', '--no-renames', '--no-ext-diff', '--no-textconv',
             '--no-color', '--src-prefix=a/', '--dst-prefix=b/')


def _git(worktree, *args, env=None, check=True):
    return snapshots._git(worktree, *args, env=env, check=check, error=ValueError)


def _untracked(worktree) -> list:
    output = _git(worktree, 'ls-files', '--others', '--exclude-standard', '-z').stdout
    return [path for path in output.decode('utf-8', 'surrogateescape').split('\0') if path]


def _temporary_index(worktree, directory, *, copy):
    index = Path(directory) / 'index'
    if copy:
        real = Path(snapshots._git_text(worktree, 'rev-parse', '--git-path', 'index', error=ValueError))
        if not real.is_absolute():
            real = Path(worktree) / real
        if real.exists():
            shutil.copyfile(real, index)
    return {'GIT_INDEX_FILE': str(index)}


def worktree_patch(worktree, base_commit, *, include_untracked=False) -> tuple:
    """(patch bytes, untracked paths included) for the worktree against base; stores nothing."""
    untracked = _untracked(worktree)
    if untracked and not include_untracked:
        raise ValueError(f'Worktree {worktree} has untracked, non-ignored files {untracked}; '
                         'commit/stage them, ignore them, or pass include_untracked=True')
    with tempfile.TemporaryDirectory(prefix='flashrt-patch-') as directory:
        env = None
        if untracked:
            env = _temporary_index(worktree, directory, copy=True)
            _git(worktree, 'add', '-N', '--', *untracked, env=env)
        patch = _git(worktree, *DIFF_ARGS, base_commit, '--', env=env).stdout
    return patch, untracked


def _check_patch(worktree, base_commit, patch):
    with tempfile.TemporaryDirectory(prefix='flashrt-patch-check-') as directory:
        path = Path(directory) / 'change.patch'
        path.write_bytes(patch)
        paths = snapshots.patch_paths(worktree, path)
        snapshots.check_patch_paths(paths)
        env = _temporary_index(worktree, directory, copy=False)
        _git(worktree, 'read-tree', base_commit, env=env)
        completed = _git(worktree, 'apply', '--cached', '--check', '--whitespace=nowarn', str(path),
                         env=env, check=False)
        if completed.returncode != 0:
            raise ValueError(f'Patch does not apply to {base_commit}: '
                             f'{completed.stderr.decode("utf-8", "replace").strip()}')
        return paths


def proposal_from_worktree(worktree, base_commit, *, store, tool_id, stage_id, hypothesis,
                           options=None, include_untracked=False, provenance=None) -> PatchProposal:
    """Store the worktree's change against ``base_commit`` and describe it (see module docstring).

    ``base_commit`` may be any ref of the worktree's repository; the proposal
    records its full 40-hex commit. ``options`` are frontend options the patch
    needs (merged into the baseline's by candidate_from_proposal).
    """
    if stage_id not in STAGE_IDS:
        raise ValueError(f'stage_id must be a known stage id, got {stage_id!r}')
    worktree = snapshots._toplevel(worktree, error=ValueError)
    base = snapshots.resolve_commit(worktree, base_commit)
    patch, untracked = worktree_patch(worktree, base, include_untracked=include_untracked)
    if not patch:
        raise ValueError(f'Worktree {worktree} has no tracked change against {base}')
    paths = _check_patch(worktree, base, patch)
    head = snapshots._git_text(worktree, 'rev-parse', '--verify', 'HEAD^{commit}', error=ValueError)
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError('provenance must be a mapping')
    record = {
        'source': 'worktree', 'worktree': os.path.realpath(worktree), 'head_commit': head,
        'diff': 'git ' + ' '.join(DIFF_ARGS) + f' {base}', 'include_untracked': bool(include_untracked),
        'untracked_included': untracked, 'paths': paths,
        **dict(provenance or {}),
    }
    patch_ref = snapshots.store_patch(store, patch)
    return PatchProposal(base_commit=base, patch_ref=patch_ref, tool_id=tool_id, stage_id=stage_id,
                         hypothesis=hypothesis, options=dict(options or {}), provenance=record)


def candidate_from_proposal(baseline, proposal, candidate_id=None, *, validate=True,
                            origin='agent') -> Candidate:
    """Snapshot-mode Candidate for ``proposal`` against ``baseline``.

    The baseline must run the proposal's base commit (``source_ref``) without a
    patch, so the comparison isolates the change. ``proposal.options`` are
    merged over the baseline options. With ``validate=True`` the backend is
    checked with ``registry.validate_config`` of THIS checkout; a patch that
    adds new frontend options is known only to its snapshot's registry, so pass
    ``validate=False`` for it (the snapshot worker still validates at load).
    ``origin`` records who wrote the patch: 'human', 'search' or 'agent'.
    """
    if not isinstance(origin, str) or origin not in ORIGINS:
        raise ValueError(f'origin must be one of {list(ORIGINS)}, got {origin!r}')
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    if not isinstance(proposal, PatchProposal):
        raise TypeError('proposal must be a PatchProposal')
    if baseline.source_ref != proposal.base_commit:
        raise ValueError(f'baseline source_ref {baseline.source_ref!r} must equal the patch base '
                         f'{proposal.base_commit}: the reference must run the unpatched base')
    if baseline.patch_ref is not None:
        raise ValueError('baseline must not carry a patch_ref; stacked patches are not supported')
    snapshots.check_commit(proposal.base_commit, 'proposal.base_commit', error=ValueError)
    digest = snapshots.check_patch_ref(proposal.patch_ref, error=ValueError)
    record = encode_backend(baseline.backend)
    record['options'] = {**record['options'], **proposal.options}
    backend = decode_backend(record)
    if validate:
        validate_config(backend)
    return Candidate(
        candidate_id=candidate_id or f'{baseline.candidate_id}-patch-{digest[:12]}',
        backend=backend, workload_id=baseline.workload_id, parent_id=baseline.candidate_id,
        hypothesis=proposal.hypothesis, origin=origin, patch_ref=proposal.patch_ref,
        source_ref=proposal.base_commit, tool=proposal.tool_id, stage=proposal.stage_id)
