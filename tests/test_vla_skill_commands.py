"""Every documented `decisions add` command in the FlashRT skills and pipeline doc runs.

Commands are taken from fenced code blocks of the FlashRT-authored and adapted
skills under .agents/skills (skills whose NOTICE says 'Status: vendored
unmodified' are upstream text and skipped) and docs/vla_autotune_pipeline.zh-CN.md.
Backslash-continued lines are joined, `<...>` placeholders and `$VARS` get
temporary values, the line is split with shlex, and decisions.main(argv) must
return 0 against a fresh temporary log.
"""

from pathlib import Path
import re
import shlex

import pytest

from flash_rt.vla.autotune import decisions


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / '.agents' / 'skills'
DOCS = (ROOT / 'docs' / 'vla_autotune_pipeline.zh-CN.md',)
VENDORED = 'Status: vendored unmodified'
ENTRY = 'flash_rt.vla.autotune.decisions'

_FENCE = re.compile(r'^\s*(```|~~~)')
_PLACEHOLDER = re.compile(r'<[^<>\n]+>')
_VARIABLE = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
_SENTINEL = 'PLACEHOLDER'
# Flags whose value is validated as an identifier / known stage: a placeholder gets a valid value.
_FLAG_VALUES = {'--model': 'pi05', '--stage': 'denoise.velocity', '--tool': 'launch.cuda_graph',
                '--candidate-tool': 'launch.cuda_graph'}
_FILE_FLAGS = ('--json', '--context-json')


def _vendored(path):
    skill = path.relative_to(SKILLS).parts[0]
    notice = SKILLS / skill / 'NOTICE'
    return notice.is_file() and any(line.strip() == VENDORED for line in notice.read_text().splitlines())


def documents():
    paths = [path for path in sorted(SKILLS.rglob('*.md')) if not _vendored(path)]
    return paths + [path for path in DOCS]


def _logical_lines(text):
    """(line number, joined command line) for every line inside fenced code blocks."""
    inside, pending, start = False, '', None
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            inside, pending, start = not inside, '', None
            continue
        if not inside:
            continue
        if start is None:
            start = number
        if line.rstrip().endswith('\\'):
            pending += line.rstrip()[:-1] + ' '
            continue
        yield start, pending + line
        pending, start = '', None


def commands():
    found = []
    for path in documents():
        for number, line in _logical_lines(path.read_text(encoding='utf-8')):
            if f'{ENTRY} add' in line:
                found.append((f'{path.relative_to(ROOT)}:{number}', line))
    return found


COMMANDS = commands()


def to_argv(line, directory):
    """argv for decisions.main from one documented command line."""
    text = _PLACEHOLDER.sub(_SENTINEL, line)
    text = _VARIABLE.sub(lambda match: str(directory / match.group(1).lower()), text)
    tokens = shlex.split(text, comments=True)
    tokens = tokens[tokens.index(ENTRY) + 1:]
    if tokens[:1] != ['add']:
        raise ValueError(f'not a decisions add command: {line!r}')
    argv = []
    for index, token in enumerate(tokens):
        flag = tokens[index - 1] if index else None
        if flag == '--log':
            token = str(directory / 'decisions.jsonl')
        elif flag in _FILE_FLAGS:
            target = directory / Path(token.replace(_SENTINEL, 'file')).name
            target.write_text('{}\n')
            token = str(target)
        elif flag in _FLAG_VALUES and _SENTINEL in token:
            token = _FLAG_VALUES[flag]
        else:
            token = token.replace(_SENTINEL, 'x')
        argv.append(token)
    return argv


def test_commands_are_found_in_skills_and_docs():
    assert not any('forge-start' in str(path) or 'perf-' in str(path) for path in documents())
    assert any('vla-patch-candidate' in str(path) for path in documents())
    sources = {location.split(':')[0] for location, _ in COMMANDS}
    assert len(COMMANDS) >= 15
    assert 'docs/vla_autotune_pipeline.zh-CN.md' in sources
    assert '.agents/skills/vla-autotune/SKILL.md' in sources
    assert '.agents/skills/vla-autotune/references/decision-log.md' in sources


def test_extractor_handles_quoting_placeholders_and_variables(tmp_path):
    line = ('py -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind experience '
            "--actor agent --summary \"<model> 'quoted' on <hw>\" --model <model> --stage <stage> "
            '--context-json signature.json --outcome "a b" --evidence-ref <campaign root>/s.json  # c')
    argv = to_argv(line, tmp_path)
    assert argv[:3] == ['add', '--log', str(tmp_path / 'decisions.jsonl')]
    assert argv[argv.index('--summary') + 1] == "x 'quoted' on x"
    assert argv[argv.index('--model') + 1] == 'pi05'
    assert argv[argv.index('--stage') + 1] == 'denoise.velocity'
    assert argv[argv.index('--context-json') + 1] == str(tmp_path / 'signature.json')
    assert argv[argv.index('--evidence-ref') + 1] == 'x/s.json' and argv[-1] == 'x/s.json'


@pytest.mark.parametrize('location, line', COMMANDS, ids=[location for location, _ in COMMANDS])
def test_documented_decisions_add_command_runs(tmp_path, capsys, location, line):
    argv = to_argv(line, tmp_path)
    code = decisions.main(argv)
    error = capsys.readouterr().err
    assert code == 0, f'{location}: {argv!r} -> rc={code}: {error.strip()}'
    assert len(decisions.DecisionLog(tmp_path / 'decisions.jsonl').records()) == 1
