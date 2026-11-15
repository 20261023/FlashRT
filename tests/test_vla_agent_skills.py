"""Static checks for the repo agent skills (.agents/skills) and the vendored third-party files.

Stdlib + pytest only. The frontmatter parser below handles the subset of YAML the
skills use (plain/quoted scalars, folded '>' / '>-' and literal '|' blocks, one
nested string map, block or flow lists, '#' comments). When PyYAML is importable
it is used as a second opinion on FlashRT-authored skills.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / '.agents' / 'skills'
CLAUDE_SKILLS = ROOT / '.claude' / 'skills'
VENDOR = ROOT / 'third_party' / 'agent-skills'
MANIFEST = VENDOR / 'manifest.json'

NAME_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
VENDORED_MARKER = re.compile(r'^Status: vendored unmodified$', re.M)
AUTHORED_KEYS = {'name', 'description', 'license', 'metadata'}
MANIFEST_KEYS = {'project', 'repo', 'commit', 'upstream_path', 'local_path', 'blob', 'bytes', 'license'}
VENDORED = {'forge-start', 'perf-torch-cuda-graphs', 'perf-torch-sync-free', 'perf-nsight-systems',
            'perf-nsight-compute-analysis', 'perf-workload-profiling', 'kernel-triton-writing',
            'kernel-numerics-verifier', 'aitune-inspect', 'aitune-tune', 'aitune-benchmark', 'aitune-validate'}
ADAPTED = {'vla-model-impl-map', 'vla-operator-gate', 'vla-copy-drift-audit', 'vla-port-workflow'}
FLASHRT_ADDED = {'NOTICE', 'LICENSE.txt'}


# --------------------------------------------------------------------------- helpers

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return bytes(text[1:-1], 'utf-8').decode('unicode_escape') if '\\' in text else text[1:-1]
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def _block(lines: list, style: str) -> str:
    indents = [len(l) - len(l.lstrip(' ')) for l in lines if l.strip()]
    cut = min(indents) if indents else 0
    body = [l[cut:] if l.strip() else '' for l in lines]
    if style.startswith('|'):
        text = '\n'.join(body)
    else:  # folded: lines in one paragraph join with spaces, blank lines become newlines
        paragraphs, cur = [], []
        for l in body:
            if l:
                cur.append(l.strip())
            else:
                paragraphs.append(' '.join(cur)); cur = []
        paragraphs.append(' '.join(cur))
        text = '\n'.join(paragraphs)
    text = text.rstrip('\n')
    return text if style.endswith('-') else text + '\n'


def parse_frontmatter(text: str):
    """Return (fields, body_lines, body_start_index). Raises ValueError on malformed input."""
    lines = text.split('\n')
    if not lines or lines[0] != '---':
        raise ValueError("line 1 must be exactly '---'")
    try:
        end = lines.index('---', 1)
    except ValueError:
        raise ValueError("no closing '---'") from None
    raw = lines[1:end]
    fields, i = {}, 0
    while i < len(raw):
        line = raw[i]
        if not line.strip() or line.lstrip().startswith('#'):
            i += 1
            continue
        m = re.match(r'^([A-Za-z0-9_-]+):(?:\s+(.*))?$', line)
        if not m or line.startswith(' '):
            raise ValueError(f'unparsable frontmatter line {i + 2}: {line!r}')
        key, rest = m.group(1), (m.group(2) or '').strip()
        if key in fields:
            raise ValueError(f'duplicate key {key!r}')
        j = i + 1
        child = []
        while j < len(raw) and (raw[j].startswith((' ', '\t')) or not raw[j].strip()
                                or (raw[j].startswith('- ') and not rest)):
            child.append(raw[j]); j += 1
        while child and not child[-1].strip():
            child.pop(); j -= 1
        if rest in ('>', '>-', '|', '|-'):
            fields[key] = _block(child, rest)
        elif rest.startswith('[') and rest.endswith(']'):
            fields[key] = [_scalar(x) for x in rest[1:-1].split(',') if x.strip()]
        elif rest and not rest.startswith('#'):
            if child:
                raise ValueError(f'{key}: multi-line plain scalars are not supported')
            fields[key] = _scalar(rest)
        else:
            items = [c for c in child if c.strip() and not c.lstrip().startswith('#')]
            if items and all(c.lstrip().startswith('- ') for c in items):
                fields[key] = [_scalar(c.lstrip()[2:]) for c in items]
            else:
                sub = {}
                for c in items:
                    cm = re.match(r'^\s+([A-Za-z0-9_.-]+):\s*(.*)$', c)
                    if not cm:
                        raise ValueError(f'{key}: unparsable nested line {c!r}')
                    sub[cm.group(1)] = _scalar(cm.group(2))
                fields[key] = sub
        i = j
    return fields, lines[end + 1:], end + 1


def skill_dirs() -> list:
    if not SKILLS.is_dir():
        return []
    return sorted(p for p in SKILLS.iterdir() if p.is_dir() and not p.name.startswith('.'))


def is_vendored(folder: Path) -> bool:
    notice = folder / 'NOTICE'
    return notice.is_file() and bool(VENDORED_MARKER.search(notice.read_text(encoding='utf-8')))


def load_skill(folder: Path):
    path = folder / 'SKILL.md'
    if not path.is_file():
        pytest.fail(f'{folder.name}: SKILL.md missing (skill folder exists but is incomplete)')
    data = path.read_bytes()
    return data, data.decode('utf-8')


def manifest_entries() -> list:
    return json.loads(MANIFEST.read_text(encoding='utf-8'))['entries']


DIRS = skill_dirs()
IDS = [p.name for p in DIRS]


# --------------------------------------------------------------------------- parser self-test

def test_frontmatter_parser_handles_used_forms():
    text = ('---\n# SPDX comment\nname: x-y\ndescription: >-\n  one\n  two\n\n  three\nlicense: "Apache-2.0"\n'
            'tags: [a, b]\nlist:\n  - c\n  - d\nmetadata:\n  author: \'N\'\n  upstream-commit: "abc"\n---\nbody\n')
    fields, body, start = parse_frontmatter(text)
    assert fields == {'name': 'x-y', 'description': 'one two\nthree', 'license': 'Apache-2.0', 'tags': ['a', 'b'],
                      'list': ['c', 'd'], 'metadata': {'author': 'N', 'upstream-commit': 'abc'}}
    assert body[0] == 'body' and start == 17
    assert parse_frontmatter('---\nd: >\n  a\n  b\n---\n')[0]['d'] == 'a b\n'
    with pytest.raises(ValueError):
        parse_frontmatter('\n---\nname: x\n---\n')


# --------------------------------------------------------------------------- every active skill

def test_skills_directory_exists():
    assert DIRS, '.agents/skills has no skill folders'


@pytest.mark.parametrize('folder', DIRS, ids=IDS)
def test_skill_frontmatter_and_body(folder):
    data, text = load_skill(folder)
    assert b'\r' not in data, 'SKILL.md must use LF line endings (no CR)'
    fields, body, _ = parse_frontmatter(text)
    name, desc = fields.get('name'), fields.get('description')
    assert isinstance(name, str) and name, 'frontmatter needs a name'
    assert isinstance(desc, str) and desc.strip(), 'frontmatter needs a description'
    assert name == folder.name, f'name {name!r} != folder {folder.name!r}'
    assert NAME_RE.fullmatch(name) and len(name) <= 64, f'bad skill name {name!r}'
    assert len(desc) <= 1024, f'description has {len(desc)} chars (max 1024)'
    assert '<' not in desc and '>' not in desc, 'description must not contain < or >'
    assert not desc.startswith('[TODO:'), 'description is an unfinished TODO placeholder'
    in_fence = False
    for n, line in enumerate(body, 1):
        if line.lstrip().startswith(('```', '~~~')):
            in_fence = not in_fence
        elif not in_fence and line.lstrip(' ').startswith('[TODO:'):
            pytest.fail(f'unfinished [TODO: placeholder in body line {n}: {line!r}')


@pytest.mark.parametrize('folder', DIRS, ids=IDS)
def test_skill_text_files_use_lf(folder):
    bad = [p.relative_to(folder).as_posix() for p in folder.rglob('*')
           if p.is_file() and p.suffix in {'.md', '.yaml', '.yml', '.py', '.txt', ''} and b'\r\n' in p.read_bytes()]
    assert not bad, f'CRLF line endings in {bad}'


@pytest.mark.parametrize('folder', [d for d in DIRS if not is_vendored(d)],
                         ids=[d.name for d in DIRS if not is_vendored(d)])
def test_flashrt_skill_rules(folder):
    """FlashRT-authored or adapted skills: portable frontmatter keys only, string metadata, < 500 lines."""
    _, text = load_skill(folder)
    fields, _, _ = parse_frontmatter(text)
    extra = set(fields) - AUTHORED_KEYS
    assert not extra, f'frontmatter keys {sorted(extra)} outside {sorted(AUTHORED_KEYS)}'
    for key in ('name', 'description', 'license'):
        assert key not in fields or isinstance(fields[key], str), f'{key} must be a string'
    meta = fields.get('metadata', {})
    assert isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values()), 'metadata must map str -> str'
    n_lines = text.count('\n') + (0 if text.endswith('\n') else 1)
    assert n_lines < 500, f'SKILL.md has {n_lines} lines; move material into references/'
    try:
        import yaml
    except ImportError:
        return
    raw = text.split('\n---\n', 1)[0][4:]
    loaded = yaml.safe_load(raw)
    assert isinstance(loaded, dict) and loaded.get('name') == fields['name']
    assert ' '.join(str(loaded.get('description')).split()) == ' '.join(fields['description'].split())


def _is_adapted(folder: Path, fields: dict) -> bool:
    meta = fields.get('metadata') if isinstance(fields.get('metadata'), dict) else {}
    return (folder.name in ADAPTED or any(k.startswith('upstream-') for k in meta)
            or (folder / 'NOTICE').exists() or (folder / 'LICENSE.txt').exists())


@pytest.mark.parametrize('folder', [d for d in DIRS if not is_vendored(d)],
                         ids=[d.name for d in DIRS if not is_vendored(d)])
def test_adapted_skill_provenance(folder):
    data, text = load_skill(folder)
    fields, body, _ = parse_frontmatter(text)
    if not _is_adapted(folder, fields):
        pytest.skip('FlashRT-authored skill (no upstream material)')
    assert (folder / 'LICENSE.txt').is_file(), 'adapted skill needs LICENSE.txt (verbatim upstream license)'
    assert (folder / 'NOTICE').is_file(), 'adapted skill needs NOTICE (provenance + list of changes)'
    meta = fields.get('metadata', {})
    for key in ('upstream-repo', 'upstream-path', 'upstream-commit'):
        assert meta.get(key), f'metadata.{key} missing'
    assert re.fullmatch(r'[0-9a-f]{40}', meta['upstream-commit']), 'upstream-commit must be a full SHA'
    assert 'LICENSE.txt' in fields.get('license', ''), "license should read '<SPDX>. Complete terms in LICENSE.txt'"
    first = next((l for l in body if l.strip()), '')
    assert first.startswith('<!--'), 'modification notice (HTML comment) must be the first line after the frontmatter'


# --------------------------------------------------------------------------- vendored skills + manifest

@pytest.mark.parametrize('name', sorted(VENDORED))
def test_vendored_skill_present_and_marked(name):
    folder = SKILLS / name
    assert folder.is_dir(), f'vendored skill {name} missing'
    assert is_vendored(folder), f"{name}/NOTICE lacks the 'Status: vendored unmodified' line"
    assert (folder / 'LICENSE.txt').is_file()


def test_adapted_skills_are_not_marked_vendored():
    for name in ADAPTED:
        if (SKILLS / name).is_dir():
            assert not is_vendored(SKILLS / name), f'{name} is adapted, its NOTICE must not claim unmodified'


def test_manifest_schema_and_hashes():
    entries = manifest_entries()
    assert entries
    seen = set()
    for e in entries:
        assert set(e) == MANIFEST_KEYS, f'manifest entry keys {sorted(e)}'
        assert re.fullmatch(r'[0-9a-f]{40}', e['blob']) and re.fullmatch(r'[0-9a-f]{40}', e['commit'])
        assert e['local_path'] not in seen, f'duplicate local_path {e["local_path"]}'
        seen.add(e['local_path'])
        path = ROOT / e['local_path']
        assert path.is_file(), f'missing vendored file {e["local_path"]}'
        data = path.read_bytes()
        assert len(data) == e['bytes'], f'{e["local_path"]}: {len(data)} bytes, manifest {e["bytes"]}'
        assert git_blob_sha(data) == e['blob'], f'{e["local_path"]}: blob hash mismatch'


@pytest.mark.parametrize('name', sorted(VENDORED))
def test_vendored_folder_matches_manifest(name):
    folder = SKILLS / name
    if not folder.is_dir():
        pytest.fail(f'vendored skill {name} missing')
    by_local = {e['local_path']: e for e in manifest_entries()}
    files = sorted(p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file())
    unlisted = [f for f in files if f not in by_local and Path(f).name != 'NOTICE']
    assert not unlisted, f'files not in manifest.json (vendored folders must stay byte-identical): {unlisted}'
    listed = [k for k in by_local if k.startswith(f'.agents/skills/{name}/')]
    assert f'.agents/skills/{name}/SKILL.md' in listed and f'.agents/skills/{name}/LICENSE.txt' in listed
    notice = (folder / 'NOTICE').read_text(encoding='utf-8')
    for k in listed:
        if not k.endswith('/LICENSE.txt'):
            assert by_local[k]['blob'] in notice, f'NOTICE does not list {k}'


def test_vendored_paths_not_gitignored():
    paths = [e['local_path'] for e in manifest_entries()]
    paths += [p.relative_to(ROOT).as_posix() for d in DIRS for p in d.rglob('*') if p.is_file()]
    try:
        res = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-v', '--no-index', '--stdin'],
                             input='\n'.join(paths), text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f'git unavailable: {exc}')
    if res.returncode not in (0, 1):
        pytest.skip(f'git check-ignore failed: {res.stderr.strip()}')
    assert not res.stdout.strip(), f'skill/vendored files would be ignored by .gitignore:\n{res.stdout}'


def test_archive_is_not_discoverable():
    """Nothing under third_party/agent-skills may be picked up as a live skill or instruction file."""
    bad = []
    for p in VENDOR.rglob('*'):
        rel = p.relative_to(VENDOR).as_posix()
        parts = rel.split('/')
        if any(a in ('.claude', '.agents', '.codex') for a in parts):
            bad.append(rel)
        if p.is_file() and p.name in ('AGENTS.md', 'CLAUDE.md', 'CLAUDE.local.md'):
            bad.append(rel)
    assert not bad, f'archive paths that tools could discover: {bad}'
    for project in ('apxinf', 'forgekernel', 'tensorrt-llm', 'torchtitan', 'aitune'):
        assert (VENDOR / project / 'VENDOR.md').is_file() and (VENDOR / project / 'LICENSE').is_file()


def test_root_notice_lists_every_upstream():
    text = (ROOT / 'NOTICE').read_text(encoding='utf-8')
    assert 'Third-party agent skills' in text
    for e in manifest_entries():
        assert e['commit'] in text and e['repo'] in text, f'root NOTICE misses {e["repo"]}@{e["commit"]}'
    assert 'Copyright 2026 The ForgeKernel Authors' in text


# --------------------------------------------------------------------------- tool registry + Claude links

def test_tool_registry_skills_exist():
    from flash_rt.vla.autotune.tools import all_tools
    referenced = sorted({s for tool in all_tools() for s in tool.skills})
    assert referenced
    missing = [s for s in referenced if not (SKILLS / s / 'SKILL.md').is_file()]
    assert not missing, f'skills referenced by flash_rt/vla/autotune/tools but missing in .agents/skills: {missing}'


def test_claude_skill_symlinks():
    if not CLAUDE_SKILLS.exists():
        pytest.skip('.claude/skills does not exist yet (the symlinks are created in a later step)')
    entries = sorted(CLAUDE_SKILLS.iterdir())
    for entry in entries:
        assert entry.is_symlink(), f'.claude/skills/{entry.name} must be a symlink'
        target = os.readlink(entry)
        assert target == f'../../.agents/skills/{entry.name}', f'{entry.name} -> {target}'
        assert entry.resolve() == (SKILLS / entry.name).resolve() and (entry / 'SKILL.md').is_file()
    linked = {e.name for e in entries}
    missing = [d.name for d in DIRS if d.name not in linked]
    assert not missing, f'active skills without a .claude/skills link: {missing}'
