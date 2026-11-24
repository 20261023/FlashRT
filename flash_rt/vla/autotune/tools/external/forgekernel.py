"""ForgeKernel adapter: task directory in, confirmed kernel out.

OpenBMB/ForgeKernel runs one operator per campaign: a teacher/student pair of
Claude Code agents optimizes a kernel for hours against a reference callable
timed in the same GPU job. This module only writes the task directory
(``prepare_task``), renders a job script (``render_job``) and reads a finished
run (``harvest``); it never runs ``forge`` or imports ForgeKernel.

The acceptance rule ignores ``forge run``'s exit code (2 also means "beat the
reference but missed the target") and reads ``best_confirmation.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

from ..artifacts import KernelArtifact, StageCapture
from . import (
    dims_text, fresh_dir, jsonable, quote, require_absolute, sha256_file,
    template_contract,
)
from . import reference_source as custom_reference_text


UPSTREAM_COMMIT = 'b2c8ae8510e02c28576808910a6b8827721ab7dc'
TOOL_ID = 'kernel.forgekernel'

SPEC = {
    'tool_id': TOOL_ID,
    'title': 'ForgeKernel: autonomous teacher/student kernel campaign for one operator',
    'lane': 'patch',
    'stages': ('kernel.gemm', 'kernel.attention', 'kernel.norm_elementwise', 'kernel.conv',
               'vision.encode', 'prefix.prefill', 'denoise.velocity'),
    'models': ('*',),
    'exactness': 'approximate',
    'status': 'needs_setup',
    'reason': (
        'A800 (sm_80) is not supported as shipped: five ForgeKernel patches are needed '
        '(no hw_kb sm80 tree, CUTE_ARCH has no sm80 entry, CUDA mode compiles for '
        'sm_100a, the SASS self-check accepts only SM90 families, PM-sampling intervals '
        'are below the GA100 minimum of 20000 cycles; see sm80_enablement()). Agents are '
        'Claude Code only (backend "claude", sandboxed, outbound HTTPS to the Claude API). '
        'Each operator costs hours of exclusive GPU time plus frontier-model tokens, and '
        'agent quality on sm80 is unvalidated.'),
    'upstream': {
        'repo': 'https://github.com/OpenBMB/ForgeKernel',
        'commit': UPSTREAM_COMMIT,
        'license': 'Apache-2.0',
        'notice': True,
    },
    'skills': ('forge-start', 'vla-kernel-candidate', 'vla-operator-gate'),
    'setup': [
        'git clone https://github.com/OpenBMB/ForgeKernel.git into a dedicated clone (runs '
        'write hw_kb/ and docs/kb-provenance/ into it) and git checkout '
        f'{UPSTREAM_COMMIT}',
        'python3.11 -m venv .venv && . .venv/bin/activate && pip install -e . (the install '
        'must be editable: paths resolve from the clone)',
        'for language="cutedsl": pip install nvidia-cutlass-dsl==<the GPU image version>',
        'apply the five sm80 enablement patches (sm80_enablement()) on a local branch and '
        'commit them; record that commit',
        'cp local.env.example local.env; set FK_DELIVERY=local, FK_GPU=0, '
        'LB_INSTALL_CUPTI=0 (offline nodes) and CUTLASS_DIR for CUDA tasks; keep '
        'credentials out of the repository',
        'on the node running forge: Claude Code CLI authenticated (ANTHROPIC_API_KEY or '
        'CLAUDE_CODE_OAUTH_TOKEN; FK_CLAUDE_BIN selects the binary), sandbox support '
        '(bubblewrap on Linux) and outbound HTTPS to the Claude API',
        'on the GPU: the bare python3 first on PATH imports CUDA torch; nvcc, ninja and '
        'CUTLASS headers for language="cuda" (skills/job-env-check/job_env_check.py '
        '--cuda checks this); ncu optional (without counter permission the run is '
        'timing-only)',
        'optional launcher skill: cp -R integrations/claude-code/forge-start '
        '~/.claude/skills/ (never into the clone\'s skills/)',
        'prepare_task(...) then `forge new <task_dir> --dry-run` before any GPU job',
    ],
    'risks': [
        'agent quality on sm80 is unknown: the hw_kb sm80 tree starts empty and the '
        'released kernels are Hopper/Blackwell only (WGMMA/TMA/tcgen05)',
        'an operator-level win may not show up end to end (off the critical path, hidden '
        'under CUDA Graph, error compounding across denoise steps x layers); only the '
        'FlashRT controller gate (rel-L2 <= 5%, cosine >= 0.999) decides',
        'forge run exits 2 whenever the target is missed, even with a faster confirmed '
        'kernel; harvest reads best_confirmation.json instead of the exit code',
        'task runs keep the float64 precision gate off; tolerance is 2x the reference '
        'error vs the teacher-written judge, floored at 2 units of rounding',
        'confirmation jobs on one node are one hardware draw (see provenance nodes)',
        'a wrong description.md makes judge calibration fail after 3 attempts: a human '
        'reviews every REVIEW REQUIRED line',
        'runs write new knowledge-base files into the ForgeKernel clone; review them '
        'before committing',
        'cost: hours of exclusive GPU per operator plus frontier-model tokens; upstream '
        'plans a simpler v1.5 harness, so keep the SHA pinned',
    ],
}

TASK_KEYS = {
    'task': frozenset({'language', 'arch'}),
    'reference': frozenset({'run', 'source', 'blackbox', 'deny', 'deny_urls', 'deny_modules'}),
    'target': frozenset({'ratio', 'ms', 'timing'}),
    'budget': frozenset({'hours', 'max_meta', 'reference_hours'}),
}
LANGUAGES = ('cuda', 'cutedsl')
TIMINGS = ('device', 'e2e')
RESERVED_TASK_NAMES = frozenset({'init', 'new', 'run', 'status', 'harvest', 'export'})
NOISE_FLOOR_MIN = 0.015
ENTRY_SYMBOL = 'run'
# Clone-relative file holding the CUTE_ARCH map (sm80_enablement() item 2).
CUTE_ARCH_FILE = 'src/forgekernel/taskcard.py'

_TASK_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_ARCH_RE = re.compile(r'^sm(\d+)([a-z]?)$')
_MODULE_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')
_COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')
# key = "basic" or key = 'literal' string (no escapes), optional trailing comment.
_TOML_STRING_RE = re.compile(
    r'''^([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(?:"([^"\\]*)"|'([^']*)')\s*(?:#.*)?$''')


def sm80_enablement() -> list:
    """The ForgeKernel changes an A800 (GA100, sm_80) campaign needs (§2.3)."""
    return [
        'hw_kb: create hw_kb/cuda/sm80/index.md (language="cuda") or '
        'hw_kb/cutedsl/sm80/index.md (language="cutedsl"); task.py needs the directory '
        'and the loop start needs index.md (exit 14 EXIT_KB_ARCH_MISSING otherwise); the '
        'tree starts with no certified sm80 recipes',
        'taskcard.py: add "sm80": "sm_80" to CUTE_ARCH (the fallback yields '
        'CUTE_DSL_ARCH="sm80a", which CuTe DSL does not accept; the "a" suffix is only '
        'for major >= 9)',
        'CUDA mode: card_template/eval/leaderboard.py CUDA_GENCODE defaults to '
        '-gencode=arch=compute_100a,code=sm_100a; patch it to compute_80/sm_80 or keep the '
        'kernel line "// fk-cuda-cflags: -gencode=arch=compute_80,code=sm_80" (the '
        'generated description requires it)',
        'skills/sass/sass_tools.py: analyze() defaults to SM90 tensor families '
        '(HGMMA/QGMMA/...), so HMMA/LDGSTS kernels get SASS-FAIL; pass '
        'families=("HMMA", "LDGSTS", "LDSM") or waive with a SASS-WAIVED: line',
        'PM sampling: pm_timeline_run.py and stall_price_run.py use 8192/2048-cycle '
        'intervals (retries down to 128); GA100 needs >= 20000 cycles, so pass '
        '--interval/--pm-interval >= 20000 or accept PMTL-FAIL/ROLE-PARTIAL',
    ]


def _gencode(arch):
    digits, suffix = _ARCH_RE.match(arch).groups()
    return f'-gencode=arch=compute_{digits}{suffix},code=sm_{digits}{suffix}'


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number, got {value!r}')
    return value


def _toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('task.toml numbers must be finite')
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a valid TOML basic string (ASCII, escaped)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(item) for item in value) + ']'
    raise ValueError(f'{type(value).__name__} is not a task.toml value')


def validate_task_config(config) -> None:
    """Reject tables/keys ForgeKernel's task.py would reject, and bad values."""
    if not isinstance(config, dict):
        raise ValueError('task config must be a dict of tables')
    for table, body in config.items():
        if table not in TASK_KEYS:
            raise ValueError(f'task.toml: unknown table [{table}]')
        if not isinstance(body, dict):
            raise ValueError(f'task.toml: [{table}] must be a table')
        unknown = set(body) - TASK_KEYS[table]
        if unknown:
            raise ValueError(f'task.toml: unknown keys {sorted(unknown)} in [{table}]')
    task = config.get('task', {})
    if task.get('language') not in LANGUAGES:
        raise ValueError(f'[task] language must be one of {list(LANGUAGES)}')
    if not isinstance(task.get('arch'), str) or not _ARCH_RE.match(task['arch']):
        raise ValueError(f'[task] arch must match sm<digits>[a-z], got {task.get("arch")!r}')
    reference = config.get('reference')
    if reference is not None:
        for key in ('run', 'source'):
            value = reference.get(key)
            if value is None:
                continue
            path = value.split(':', 1)[0] if key == 'run' else value
            if Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError(f'[reference] {key} must stay inside the task directory')
        for module in reference.get('deny_modules', []):
            if not isinstance(module, str) or not _MODULE_RE.match(module):
                raise ValueError(f'[reference] deny_modules entry {module!r} is not a '
                                 f'package name')
        for path in reference.get('deny', []):
            if not isinstance(path, str) or not path:
                raise ValueError('[reference] deny entries must be nonempty paths')
    target = config.get('target', {})
    if ('ratio' in target) == ('ms' in target):
        raise ValueError('[target] needs exactly one of ratio or ms')
    for key in ('ratio', 'ms'):
        if key in target:
            _positive_number(target[key], f'[target] {key}')
    if target.get('timing') not in TIMINGS:
        raise ValueError(f'[target] timing must be one of {list(TIMINGS)}')
    budget = config.get('budget', {})
    if 'hours' in budget:
        _positive_number(budget['hours'], '[budget] hours')
    if 'max_meta' in budget:
        value = budget['max_meta']
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'[budget] max_meta must be an int >= 0, got {value!r}')


def task_toml_text(config, header=()) -> str:
    validate_task_config(config)
    lines = [f'# {line}' for line in header]
    for table in ('task', 'reference', 'target', 'budget'):
        body = config.get(table)
        if body is None:
            continue
        if lines:
            lines.append('')
        lines.append(f'[{table}]')
        lines += [f'{key} = {_toml_value(value)}' for key, value in body.items()]
    return '\n'.join(lines) + '\n'


def shapes_records(capture) -> list:
    """shapes.json rows: [{"name", "axes"}], one axis-name set across shapes."""
    return [{'name': row['name'], 'axes': dict(row['axes'])} for row in capture.shapes]


def _one_line(text):
    return ' '.join(str(text).split())


def _linear_semantics(capture, inputs, outputs):
    x, weight = inputs[0]['name'], inputs[1]['name']
    y = outputs[0]['name']
    bias = f' + {inputs[2]["name"]}[n]' if len(inputs) == 3 else ''
    lines = [
        f'For every m < M and n < N: `{y}[m, n] = sum_k {x}[m, k] * {weight}[n, k]{bias}`.',
        f'`{weight}` is stored [N, K] (the torch `F.linear` layout) and is consumed as '
        f'stored: no transposed copy outside the kernel.',
        'Products accumulate in float32' + (', the bias is added in float32' if bias else '')
        + f', and the result is rounded to {outputs[0]["dtype"]} once, at the end.',
        f'The reference is `torch.nn.functional.linear` on {capture.dtype} tensors (the '
        f'production cuBLAS path, float32 accumulation).',
    ]
    return lines


def _attention_semantics(capture, inputs, outputs):
    q, k, v = (entry['name'] for entry in inputs)
    o = outputs[0]['name']
    return [
        f'For every b, h, s: `{o}[b, h, s, :] = sum_j p[j] * {v}[b, g, j, :]` with '
        f'`p = softmax_j(scale * {q}[b, h, s, :] . {k}[b, g, j, :])`, `g = h // (H / HKV)` '
        f'and `scale = 1 / sqrt(D)`.',
        'No mask (non-causal), no dropout. Softmax statistics and both products '
        f'accumulate in float32; the output is rounded to {outputs[0]["dtype"]} once.',
        'The reference is `torch.nn.functional.scaled_dot_product_attention` on '
        f'{capture.dtype} tensors.',
        'REVIEW REQUIRED: confirm the mask/causality, the scale and the [B, H, S, D] '
        f'layout against the call site {_one_line(capture.call_site)}; a call site that '
        'uses [B, S, H, D] needs the description and reference/run.py changed.',
    ]


def _custom_semantics(capture):
    return [
        'REVIEW REQUIRED: write the exact formula with every convention (masking, padding, '
        'layouts, accumulation precision). The callable in reference/run.py was copied '
        f'from {_one_line(capture.reference)}; its behaviour is not automatically the '
        'definition of correct.',
    ]


def description_text(capture, task_name, *, language, arch, timing) -> str:
    inputs, outputs = template_contract(capture)
    lines = [
        f'# {task_name}', '',
        f'FlashRT stage `{capture.stage_id}` of model `{capture.model}` ({capture.op_kind}), '
        f'captured at `{_one_line(capture.call_site)}`.', '',
        '> REVIEW REQUIRED: a human checks every line marked REVIEW REQUIRED before '
        '`forge new`; the teacher writes the correctness judge from this file alone.', '',
        '## Signature', '',
        '- inputs, in order (the order of `reference/run.py:run`):',
    ]
    lines += [f'  - `{entry["name"]}`: `{dims_text(entry)}`, {entry["dtype"]} '
              f'({entry["role"]})' for entry in inputs]
    lines.append('- outputs (freshly allocated and returned):')
    lines += [f'  - `{entry["name"]}`: `{dims_text(entry)}`, {entry["dtype"]}'
              for entry in outputs]
    lines += ['', f'Axes {", ".join(capture.axes)} take the values of each shape in '
              'shapes.json; one kernel serves every shape.', '', '## Semantics', '']
    if capture.op_kind == 'linear':
        lines += _linear_semantics(capture, inputs, outputs)
    elif capture.op_kind == 'attention':
        lines += _attention_semantics(capture, inputs, outputs)
    else:
        lines += _custom_semantics(capture)
    lines += [
        '', '## Legal inputs', '',
        'Any finite values of the declared dtypes. Every shape in shapes.json is a '
        'production shape recorded at the call site.',
        'REVIEW REQUIRED: describe the value distribution (real activations have outlier '
        'channels; weights are trained). No input groups ship with this task, so the '
        'teacher generates them'
        + (f'; the first captured call is saved at {_one_line(capture.tensors_path)} '
           '(not part of the task).' if capture.tensors_path else '.'),
        '', '## Build constraints', '',
        f'Target GPU architecture {arch}; timing window `{timing}`'
        + (' (the call site runs inside a CUDA graph).' if capture.in_cuda_graph else '.'),
    ]
    if language == 'cuda':
        lines.append(f'The kernel file must contain the line `// fk-cuda-cflags: '
                     f'{_gencode(arch)}`: ForgeKernel otherwise compiles for sm_100a.')
    else:
        lines.append(f'CuTe DSL kernels need CUTE_DSL_ARCH for {arch} (ForgeKernel\'s '
                     'CUTE_ARCH map must have an entry for it; see sm80_enablement()).')
    if arch == 'sm80':
        lines.append('sm_80 has no TMA, WGMMA, tcgen05 or FP8 tensor cores: use mma.sync '
                     '(HMMA), ldmatrix and cp.async.')
    if capture.notes:
        lines += ['', '## Capture notes', '']
        lines += [f'- {_one_line(note)}' for note in capture.notes]
    return '\n'.join(lines) + '\n'


def _linear_reference(inputs):
    params = ', '.join(entry['name'] for entry in inputs)
    return [
        f'def run({params}):',
        '    """Production path: cuBLAS with float32 accumulation; a fresh output."""',
        f'    return F.linear({params})',
    ]


def _attention_reference(capture, inputs):
    params = ', '.join(entry['name'] for entry in inputs)
    gqa = any(row['axes']['H'] != row['axes']['HKV'] for row in capture.shapes)
    call = f'F.scaled_dot_product_attention({params}' + (', enable_gqa=True)' if gqa else ')')
    lines = [
        f'def run({params}):',
        '    """Non-causal SDPA with the default 1/sqrt(D) scale; a fresh output."""',
        f'    return {call}',
    ]
    return lines


def reference_text(capture) -> str:
    """reference/run.py for linear/attention captures (custom copies its file)."""
    inputs, _ = template_contract(capture)
    header = [
        f'"""ForgeKernel reference for FlashRT {capture.model} {capture.stage_id} '
        f'({capture.op_kind}).',
        '',
        f'Captured at {_one_line(capture.call_site)}. Generated by FlashRT; returns freshly',
        'allocated outputs. Timed beside the kernel in the same GPU job.',
        '"""',
        '',
        'import torch  # noqa: F401',
        'import torch.nn.functional as F',
        '', '',
    ]
    body = (_linear_reference(inputs) if capture.op_kind == 'linear'
            else _attention_reference(capture, inputs))
    return '\n'.join(header + body) + '\n'


def prepare_task(capture, out_dir, *, language='cuda', arch='sm80', target_ratio=1.10,
                 target_ms=None, timing=None, hours=6, max_meta=3, reference_source=None,
                 deny=(), deny_modules=()) -> Path:
    """Write a ForgeKernel task directory at ``out_dir`` (its name is the task name).

    Writes task.toml, description.md (with REVIEW REQUIRED lines), shapes.json and
    reference/run.py (linear: ``F.linear``; attention: SDPA; custom: a copy of
    ``capture.reference``). ``reference_source`` (file or directory) is copied to
    reference/source/ for the teacher to read. ``target_ms`` replaces the ratio
    target and then requires ``target_ratio=None``. ``timing`` defaults to
    'device' when the call site runs in a CUDA graph, else 'e2e'.
    """
    if not isinstance(capture, StageCapture):
        raise TypeError('capture must be a StageCapture')
    out_dir = Path(out_dir)
    name = out_dir.name
    if not _TASK_NAME_RE.match(name) or name.startswith('_') or name in RESERVED_TASK_NAMES:
        raise ValueError(f'task directory name {name!r} is not a ForgeKernel task name '
                         f'(letters, digits, _ . -; not starting with _; not a forge command)')
    if language not in LANGUAGES:
        raise ValueError(f'language must be one of {list(LANGUAGES)}, got {language!r}')
    if not isinstance(arch, str) or not _ARCH_RE.match(arch):
        raise ValueError(f'arch must match sm<digits>[a-z], got {arch!r}')
    if timing is None:
        timing = 'device' if capture.in_cuda_graph else 'e2e'
    if (target_ratio is None) == (target_ms is None):
        raise ValueError('give exactly one target: target_ratio or target_ms (with '
                         'target_ratio=None)')
    target = ({'ratio': float(_positive_number(target_ratio, 'target_ratio'))}
              if target_ms is None else {'ms': float(_positive_number(target_ms, 'target_ms'))})
    target['timing'] = timing
    template_contract(capture)
    reference = {'run': 'reference/run.py:run', 'source': 'reference'}
    if deny:
        reference['deny'] = [str(path) for path in deny]
    if deny_modules:
        reference['deny_modules'] = list(deny_modules)
    config = {
        'task': {'language': language, 'arch': arch},
        'reference': reference,
        'target': target,
        'budget': {'hours': hours, 'max_meta': max_meta},
    }
    header = [
        'ForgeKernel task generated by FlashRT (flash_rt.vla.autotune.tools.external.'
        'forgekernel).',
        f'Stage {capture.model} {capture.stage_id} at {_one_line(capture.call_site)}.',
        f'Pinned ForgeKernel {UPSTREAM_COMMIT}; review description.md before `forge new`.',
    ]
    toml_text = task_toml_text(config, header)
    description = description_text(capture, name, language=language, arch=arch,
                                   timing=timing)
    run_text = (custom_reference_text(capture) if capture.op_kind == 'custom'
                else reference_text(capture))
    compile(run_text, 'reference/run.py', 'exec')
    if reference_source is not None and not Path(reference_source).exists():
        raise ValueError(f'reference_source {reference_source} does not exist')

    fresh_dir(out_dir, 'task directory')
    (out_dir / 'task.toml').write_text(toml_text, encoding='utf-8')
    (out_dir / 'description.md').write_text(description, encoding='utf-8')
    (out_dir / 'shapes.json').write_text(
        json.dumps(shapes_records(capture), indent=2) + '\n', encoding='utf-8')
    reference_dir = out_dir / 'reference'
    reference_dir.mkdir()
    (reference_dir / 'run.py').write_text(run_text, encoding='utf-8')
    if reference_source is not None:
        source = Path(reference_source)
        target_dir = reference_dir / 'source'
        if source.is_dir():
            shutil.copytree(source, target_dir)
        else:
            target_dir.mkdir()
            shutil.copy2(source, target_dir / source.name)
    return out_dir


def _task_table(task_dir):
    """(language, arch) from task.toml's [task] table (string values, as prepare_task writes)."""
    path = Path(task_dir) / 'task.toml'
    table, values = None, {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            table = line.split('#', 1)[0].strip()
            continue
        match = _TOML_STRING_RE.match(line) if table == '[task]' else None
        if match and match.group(1) in ('language', 'arch'):
            basic, literal = match.group(2), match.group(3)
            values[match.group(1)] = literal if basic is None else basic
    language, arch = values.get('language'), values.get('arch')
    if language not in LANGUAGES:
        raise ValueError(f'{path}: [task] language must be one of {list(LANGUAGES)}, got '
                         f'{language!r}')
    if arch is None or not _ARCH_RE.match(arch):
        raise ValueError(f'{path}: [task] arch must match sm<digits>[a-z], got {arch!r}')
    return language, arch


def _sm80_preflight(language):
    """Bash that stops before `forge new` when the clone lacks the sm80 enablement."""
    index = f'hw_kb/{language}/sm80/index.md'
    hint = 'apply sm80_enablement() (flash_rt.vla.autotune.tools.external.forgekernel)'
    return [
        f"test -f {index} || {{ echo 'ForgeKernel clone has no {index}: {hint}' >&2; "
        f"exit 1; }}",
        'grep -Eq "[\\"\']sm80[\\"\'][[:space:]]*:" ' + CUTE_ARCH_FILE + ' || '
        f"{{ echo 'CUTE_ARCH in {CUTE_ARCH_FILE} has no sm80 entry: {hint}' >&2; exit 1; }}",
    ]


def render_job(task_dir, *, forgekernel_root, runs_dir, env_file=None, gpu='0') -> str:
    """Bash for one GPU allocation running ``forge new`` + ``forge run`` (never executed).

    Mode A of the report: controller and GPU jobs in one allocation, FK_DELIVERY=local.
    The allocation must cover [budget] hours plus reconnaissance, three confirmation
    jobs and the correctness monitor (budget about hours + 4h). For arch sm80 the
    script first checks the clone for the hw_kb sm80 index and the CUTE_ARCH sm80
    entry (sm80_enablement() items 1-2) and exits 1 without them.
    """
    task_dir = require_absolute(task_dir, 'task_dir')
    forgekernel_root = require_absolute(forgekernel_root, 'forgekernel_root')
    runs_dir = require_absolute(runs_dir, 'runs_dir')
    if env_file is not None:
        env_file = require_absolute(env_file, 'env_file')
    gpu = str(gpu)
    if not re.match(r'^[0-9]+$', gpu):
        raise ValueError(f'gpu must be a device index, got {gpu!r}')
    if not (Path(task_dir) / 'task.toml').is_file():
        raise ValueError(f'{task_dir} has no task.toml; run prepare_task first')
    language, arch = _task_table(task_dir)
    lines = [
        '#!/usr/bin/env bash',
        '# ForgeKernel campaign generated by FlashRT; FlashRT never executes this script.',
        f'# Pinned OpenBMB/ForgeKernel {UPSTREAM_COMMIT} plus the sm80 enablement patches.',
        '# Single-node submission on the N46H cluster:',
        '#   sbatch -p gpu_a800 --gpus=1 <this script>',
        '# The exit code of `forge run` is NOT the acceptance signal: harvest() reads',
        '# operators/<task>/best_confirmation.json.',
        'set -euo pipefail',
    ]
    if env_file is not None:
        lines.append(f'source {quote(env_file)}')
    lines += [
        f'export FK_DELIVERY=local FK_GPU={gpu} LB_INSTALL_CUPTI=0',
        f'export FORGEKERNEL_RUNS_DIR={quote(runs_dir)}',
        'export FORGEKERNEL_LOCK_DIR="${FORGEKERNEL_LOCK_DIR:-/tmp/fk-locks}"  # node-local',
        f'TASK_DIR={quote(task_dir)}',
        f'cd {quote(forgekernel_root)}',
        f'git merge-base --is-ancestor {UPSTREAM_COMMIT} HEAD || '
        f'{{ echo "ForgeKernel clone is not based on {UPSTREAM_COMMIT}" >&2; exit 1; }}',
        'echo "FORGEKERNEL_HEAD=$(git rev-parse HEAD)"',
    ]
    if arch == 'sm80':
        lines += _sm80_preflight(language)
    lines += [
        'mkdir -p "$FORGEKERNEL_RUNS_DIR" "$FORGEKERNEL_LOCK_DIR"',
        'forge new "$TASK_DIR" --dry-run',
        "RUN=$(forge new \"$TASK_DIR\" | sed -n 's/^\\[forge\\] run directory \\(.*\\): "
        "`forge run.*/\\1/p')",
        'if [ -z "$RUN" ]; then echo "forge new printed no run directory" >&2; exit 1; fi',
        'echo "FORGE_RUN=$RUN"',
        'set +e',
        'forge run "$RUN"',
        'RC=$?',
        'set -e',
        'echo "FORGE_RC=$RC RUN=$RUN"  # resume: rerun `forge run "$RUN"` in a new allocation',
        'exit 0',
    ]
    return '\n'.join(lines) + '\n'


def _read_json(path, reasons, label):
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except FileNotFoundError:
        reasons.append(f'{label} missing: {path}')
    except (OSError, ValueError) as error:
        reasons.append(f'{label} unreadable: {error}')
    return None


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _confirmation_checks(confirmation, reasons, metrics, notes):
    if not isinstance(confirmation, dict):
        reasons.append('best_confirmation.json is not an object')
        return
    jobs = confirmation.get('jobs')
    ratios = []
    if not isinstance(jobs, list) or not jobs:
        reasons.append('best_confirmation.json has no jobs')
    else:
        for index, job in enumerate(jobs):
            ratio = job.get('ratio') if isinstance(job, dict) else None
            if not _number(ratio) or ratio <= 0:
                reasons.append(f'jobs[{index}].ratio is not a positive number: {ratio!r}')
            else:
                ratios.append(ratio)
    requested = confirmation.get('requested')
    if requested is not None and (isinstance(requested, bool) or not isinstance(
            requested, int) or requested < 1):
        reasons.append(f'requested is not a positive int: {requested!r}')
        requested = None
    if isinstance(jobs, list):
        metrics['confirmation_jobs'] = len(jobs)
        if requested is not None:
            metrics['confirmation_requested'] = requested
            if len(jobs) < requested:
                reasons.append(f'confirmation short of requested: {len(jobs)} < {requested} '
                               f'jobs (upstream: provisional)')
    median = confirmation.get('median_ratio')
    if not _number(median):
        reasons.append(f'median_ratio missing or not a number: {median!r}')
    if len(ratios) == 1 and len(jobs) == 1:
        reasons.append('one confirmation job: its spread cannot estimate the noise floor '
                       '(need >= 2 jobs)')
    if len(ratios) >= 2 and len(ratios) == len(jobs):
        spread = (max(ratios) - min(ratios)) / min(ratios)
        threshold = 1.0 + max(NOISE_FLOOR_MIN, spread)
        metrics.update({'job_ratios': ratios, 'noise_floor': max(NOISE_FLOOR_MIN, spread),
                        'acceptance_threshold': threshold})
        if _number(median) and not median > threshold:
            reasons.append(f'median_ratio {median} <= 1 + max({NOISE_FLOOR_MIN}, spread '
                           f'{spread:.6g}) = {threshold:.6g}')
    if _number(median):
        metrics['median_ratio'] = median
    if confirmation.get('keep_disputed') is not False:
        reasons.append(f'keep_disputed is {confirmation.get("keep_disputed")!r}, not false')
    monitor = confirmation.get('correctness_monitor')
    if monitor is None:
        notes.append('no correctness_monitor in best_confirmation.json')
    elif not isinstance(monitor, dict):
        reasons.append('correctness_monitor is not an object')
    else:
        metrics['correctness_monitor_status'] = monitor.get('status')
        if monitor.get('status') == 'alarm':
            reasons.append('correctness_monitor.status is alarm')
    for key in ('median_ms', 'keep_ms', 'node_band_pct'):
        if _number(confirmation.get(key)):
            metrics[key] = confirmation[key]
    if isinstance(confirmation.get('window'), str):
        metrics['window'] = confirmation['window']
    nodes = confirmation.get('nodes')
    if isinstance(nodes, list):
        metrics['nodes'] = [str(node) for node in nodes]
        if len(nodes) == 1:
            notes.append('confirmation jobs ran on one node: one hardware draw')


def harvest(run_dir, *, stage_id, model) -> KernelArtifact:
    """Apply FlashRT's acceptance rule to a ForgeKernel run directory (read-only).

    Accepted only if best_confirmation.json exists with at least two jobs and no
    fewer than it requested, median_ratio > 1 + max(0.015, (max - min) / min of
    jobs[].ratio), keep_disputed is false, the correctness monitor did not alarm
    and exactly one of best/kernel.py, best/kernel.cu exists.
    Every failed check is recorded in ``reasons``; the exit code is ignored.
    """
    run_dir = Path(run_dir).resolve()
    reasons, notes, metrics = [], ['forge run exit code ignored by design'], {}
    record = _read_json(run_dir / 'run.json', reasons, 'run.json')
    task_name = record.get('task') if isinstance(record, dict) else None
    confirmation = None
    if isinstance(task_name, str) and task_name:
        confirmation_path = run_dir / 'operators' / task_name / 'best_confirmation.json'
        confirmation = _read_json(confirmation_path, reasons, 'best_confirmation.json')
    elif record is not None:
        reasons.append('run.json names no task')
    if confirmation is not None:
        _confirmation_checks(confirmation, reasons, metrics, notes)
    kernels = [path for path in (run_dir / 'best' / 'kernel.py', run_dir / 'best' / 'kernel.cu')
               if path.is_file()]
    kernel_path = kernel_sha256 = language = None
    if len(kernels) == 1:
        kernel_path = str(kernels[0])
        kernel_sha256 = sha256_file(kernels[0])
        language = 'cutedsl' if kernels[0].suffix == '.py' else 'cuda'
    elif not kernels:
        reasons.append('no best/kernel.py or best/kernel.cu')
    else:
        reasons.append('both best/kernel.py and best/kernel.cu exist')
    run_commit = record.get('forgekernel_commit') if isinstance(record, dict) else None
    if isinstance(run_commit, str) and _COMMIT_RE.match(run_commit):
        upstream_commit, commit_source = run_commit, 'run.json forgekernel_commit'
    else:
        upstream_commit, commit_source = UPSTREAM_COMMIT, 'SPEC pin (run.json had none)'
    if upstream_commit != UPSTREAM_COMMIT:
        notes.append(f'run used ForgeKernel {upstream_commit}, pinned {UPSTREAM_COMMIT} '
                     f'(expected when sm80 patches are committed)')
    provenance = {
        'run_dir': str(run_dir), 'task': task_name, 'pinned_commit': UPSTREAM_COMMIT,
        'upstream_commit_source': commit_source, 'notes': notes,
    }
    if isinstance(record, dict):
        provenance['run'] = {key: record[key] for key in (
            'id', 'status', 'stop', 'target_met', 'loop_rc', 'tolerance',
            'snapshot_sha256', 'forgekernel_commit', 'machine', 'ncu_available')
            if key in record}
    if isinstance(confirmation, dict) and isinstance(confirmation.get('sha'), str):
        provenance['best_sha'] = confirmation['sha']
    if isinstance(task_name, str) and task_name:
        summary = _read_json(run_dir / 'operators' / task_name / 'loop_summary.json', [],
                             'loop_summary.json')
        if isinstance(summary, dict):
            provenance['loop_summary'] = {key: summary[key] for key in ('stop', 'confirm')
                                          if key in summary}
    return KernelArtifact(
        tool_id=TOOL_ID, upstream_commit=upstream_commit, stage_id=stage_id, model=model,
        language=language, entry_symbol=ENTRY_SYMBOL, kernel_path=kernel_path,
        kernel_sha256=kernel_sha256, op_metrics=jsonable(metrics), accepted=not reasons,
        reasons=reasons, provenance=jsonable(provenance))
