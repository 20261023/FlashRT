#!/usr/bin/env python3
"""Verify (and optionally re-sync) the vendored third-party agent-skill files.

Reads ``manifest.json`` next to this script. Every entry names one upstream file
(``repo``, ``commit``, ``upstream_path``, ``blob``, ``bytes``) and where FlashRT
keeps it (``local_path``, relative to the repository root).

Default mode (offline, stdlib only):
  * re-hash every ``local_path`` as a git blob, sha1(b"blob <n>\\0" + data), and
    compare it with ``blob`` and ``bytes``;
  * check that every blob line in each ``.agents/skills/<name>/NOTICE`` and every
    file row in each ``third_party/agent-skills/<project>/VENDOR.md`` agrees with
    the manifest, and that each vendored skill folder holds no file the manifest
    does not list (besides the FlashRT-added NOTICE).

``--check-upstream`` additionally asks GitHub (``gh api``) that every blob is
reachable in the upstream repository and that ``upstream_path`` at ``commit``
still resolves to it. ``--resync`` rewrites local files from upstream blobs, but
only after the fetched bytes hash to the pinned blob SHA. Downloaded content is
treated as data: it is written to disk and never executed or imported.

Exit status: 0 when everything matches, 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FLASHRT_ADDED = {'NOTICE'}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def load_manifest() -> list:
    return json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))['entries']


def gh_json(endpoint: str):
    out = subprocess.run(['gh', 'api', endpoint], check=True, capture_output=True).stdout
    return json.loads(out)


def fetch_blob(repo: str, blob: str) -> bytes:
    payload = gh_json(f'repos/{repo}/git/blobs/{blob}')
    if payload.get('encoding') != 'base64':
        raise ValueError(f'{repo} blob {blob}: unexpected encoding {payload.get("encoding")!r}')
    return base64.b64decode(payload['content'])


def check_local(entries: list) -> list:
    errors = []
    for e in entries:
        path = ROOT / e['local_path']
        if not path.is_file():
            errors.append(f'missing: {e["local_path"]}')
            continue
        data = path.read_bytes()
        got = git_blob_sha(data)
        if got != e['blob'] or len(data) != e['bytes']:
            errors.append(f'hash mismatch: {e["local_path"]} has {got} ({len(data)} B), '
                          f'manifest {e["blob"]} ({e["bytes"]} B)')
    return errors


def check_side_manifests(entries: list) -> list:
    errors = []
    by_local = {e['local_path']: e for e in entries}
    skills = sorted({Path(e['local_path']).parts[2] for e in entries if e['local_path'].startswith('.agents/skills/')})
    for name in skills:
        folder = ROOT / '.agents' / 'skills' / name
        prefix = f'.agents/skills/{name}/'
        notice = folder / 'NOTICE'
        if not notice.is_file():
            errors.append(f'{prefix}NOTICE missing')
            continue
        listed = {}
        for line in notice.read_text(encoding='utf-8').splitlines():
            m = re.fullmatch(r'([0-9a-f]{40})\s+(\d+)\s+(\S.*)', line.strip())
            if m:
                listed[prefix + m.group(3)] = (m.group(1), int(m.group(2)))
        expected = {k: (v['blob'], v['bytes']) for k, v in by_local.items()
                    if k.startswith(prefix) and not k.endswith('/LICENSE.txt')}
        if listed != expected:
            errors.append(f'{prefix}NOTICE file list disagrees with manifest.json')
        on_disk = {p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()}
        extra = sorted(p for p in on_disk - set(by_local) if Path(p).name not in FLASHRT_ADDED)
        if extra:
            errors.append(f'{prefix}: files not in manifest.json: {extra}')
    for vendor in sorted(HERE.glob('*/VENDOR.md')):
        for line in vendor.read_text(encoding='utf-8').splitlines():
            m = re.fullmatch(r'\| `([^`]+)` \| `([^`]+)` \| `([0-9a-f]{40})` \| (\d+) \|', line.strip())
            if not m:
                continue
            e = by_local.get(m.group(1))
            if e is None or (e['upstream_path'], e['blob'], e['bytes']) != (m.group(2), m.group(3), int(m.group(4))):
                errors.append(f'{vendor.relative_to(ROOT)}: row for {m.group(1)} disagrees with manifest.json')
    return errors


def check_upstream(entries: list, resync: bool) -> list:
    errors = []
    trees = {}
    for e in entries:
        key = (e['repo'], e['commit'])
        if key not in trees:
            tree = gh_json(f'repos/{e["repo"]}/git/trees/{e["commit"]}?recursive=1')
            trees[key] = {t['path']: t['sha'] for t in tree['tree'] if t['type'] == 'blob'}
        if trees[key].get(e['upstream_path']) != e['blob']:
            errors.append(f'upstream {e["repo"]}@{e["commit"][:12]}:{e["upstream_path"]} is not blob {e["blob"]}')
            continue
        if resync:
            data = fetch_blob(e['repo'], e['blob'])
            if git_blob_sha(data) != e['blob'] or len(data) != e['bytes']:
                errors.append(f'fetched bytes for {e["upstream_path"]} do not hash to {e["blob"]}')
                continue
            path = ROOT / e['local_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check-upstream', action='store_true', help='also verify blobs against GitHub (needs gh)')
    parser.add_argument('--resync', action='store_true', help='rewrite local files from verified upstream blobs (implies --check-upstream)')
    args = parser.parse_args(argv)
    entries = load_manifest()
    errors = []
    if args.check_upstream or args.resync:
        errors += check_upstream(entries, resync=args.resync)
    errors += check_local(entries)
    errors += check_side_manifests(entries)
    for err in errors:
        print('FAIL', err)
    projects = sorted({e['project'] for e in entries})
    print(f'{len(entries)} vendored files from {len(projects)} upstreams ({", ".join(projects)}): '
          f'{"OK" if not errors else f"{len(errors)} problem(s)"}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
