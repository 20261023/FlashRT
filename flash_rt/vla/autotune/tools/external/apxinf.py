"""ApxInf adapter: an alternative whole-model pi0.5 engine as a latency baseline.

infinigence/ApxInf runs pi0.5 as one CUDA graph (CUDA preprocessing of uint8
images, vision, prefix, all flow steps, actions). It is a baseline to compare
against, not a FlashRT candidate. ``render_bench`` returns the documented
benchmark commands (never executed here) and ``parse_report`` reads their JSON
without inventing numbers. Compare latency boundaries before comparing ms.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import jsonable, quote, require_absolute


UPSTREAM_COMMIT = 'b05b40990387e8c48e7d6267a790664981a665e4'
TOOL_ID = 'engine.apxinf'
REPORT_SCHEMA = 'flash_rt.autotune.apxinf_report.v1'
RUST_VARIANTS = ('bf16', 'fp8_static', 'int8_dynamic')
# ApxInf's protocol prompt length (T=10; T=21 is the other documented point). The
# Rust pi05_bench defaults to 200 and bench_pi05.py to 10, so it is always passed.
PROTOCOL_TOKEN_COUNT = 10
PYTHON_VARIANTS = ('bf16', 'fp8_dynamic', 'fp8_static', 'int8_dynamic')
BOUNDARY_NOTE = (
    'ApxInf graph replay = steady-state CUDA Graph launch + synchronize; the graph '
    'includes CUDA preprocessing of resized uint8 images, all flow steps and the action '
    'output, and excludes H2D, image decoding, rotation and CPU resize. "Input update + '
    'graph" adds the in-place input update. Match FlashRT\'s timing boundary before '
    'comparing milliseconds.')

SPEC = {
    'tool_id': TOOL_ID,
    'title': 'ApxInf: whole-model CUDA-graph engine as an external pi0.5 latency baseline',
    'lane': 'baseline',
    'stages': ('host.input', 'vision.encode', 'vision.project', 'prefix.prefill',
               'denoise.velocity', 'denoise.routing', 'host.launch'),
    'models': ('pi05',),
    'exactness': 'n/a',
    'status': 'needs_setup',
    'reason': (
        'Needs a Rust + CUDA build (maturin wheel or cargo) on a GPU node; A800 (sm_80) is '
        'not validated upstream (no tactics database, no CI runner; FP8 static is '
        'unavailable on sm_80); covers pi0.5 only (no pi0 flow, Hy-VLA or FastWAM). Its '
        'graph includes CUDA preprocessing, so latency boundaries must be matched before '
        'comparing ms.'),
    'upstream': {
        'repo': 'https://github.com/infinigence/ApxInf',
        'commit': UPSTREAM_COMMIT,
        'license': 'Apache-2.0',
        'notice': True,
    },
    'skills': ('vla-model-impl-map',),
    'setup': [
        'git clone https://github.com/infinigence/ApxInf.git and git checkout '
        f'{UPSTREAM_COMMIT}',
        'toolchain on a Linux GPU node: NVIDIA driver, full CUDA toolkit (nvcc, cuBLAS and '
        'cuBLASLt dev libraries), C/C++ compiler, ar, git, pkg-config, Python 3, stable Rust',
        'build on the GPU node (the build probes the visible GPU) or export '
        'APXINF_CUDA_ARCH=sm_80',
        'Python wheel: CARGO_TARGET_DIR=target/wheel maturin build --release --features '
        'cuda --auditwheel skip -m crates/apxinf-py/Cargo.toml && pip install '
        'target/wheel/wheels/apxinf_py-*.whl && pip install -e "python/apxinf[serving]"',
        'pi0.5 checkpoint in openpi_pytorch layout (model.safetensors + metadata.pt + '
        'assets/<asset_id>/norm_stats.json) or lerobot layout (config.json type pi05 + '
        'model.safetensors)',
        'run once with --autotune per shape and toolkit, from a dedicated working '
        'directory (recipes persist under configs/tuning/ relative to the cwd)',
        'render_bench(...) then parse_report(...)',
    ],
    'risks': [
        'sm_80 is unvalidated: BF16 probably builds and runs; INT8 falls back to '
        'non-CUTLASS providers and FP8 static is unavailable',
        'boundary mismatch: the graph includes CUDA preprocessing and excludes H2D and CPU '
        'image work, unlike a FlashRT end-to-end wall time',
        'P50 index rules differ: Rust rounds (n-1)*f, the Python bench floors it',
        'implicit infer falls back to eager when capture fails; check the report\'s passed '
        'flag and graph fields',
        'warm-start / one-step sampler results are closed-loop techniques that a static '
        'rel-L2 gate rejects by design',
    ],
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive int, got {value!r}')
    return value


def render_bench(model_dir, *, apxinf_root, variant='bf16', views=2,
                 token_count=PROTOCOL_TOKEN_COUNT, iterations=30, out,
                 python_out=None) -> str:
    """Bash for ApxInf's documented pi0.5 benchmarks (never executed by FlashRT).

    Always renders the Rust graph-replay bench (``pi05_bench``, one JSON line on
    stdout redirected to ``out``). With ``python_out`` it also renders the Python
    layered bench at layer l1 (``ModelRunner.infer_rgb``). ``model_dir=None``
    benchmarks deterministic random weights (latency depends on shapes only).
    ``--token-count`` is always passed to the Rust bench and to the random-weight
    Python bench; a Python checkpoint run takes T from its prompt, which must have
    ``token_count`` tokens for the two reports to be comparable (check the
    ``token_count`` field ``parse_report`` returns for each).
    """
    apxinf_root = require_absolute(apxinf_root, 'apxinf_root')
    out = require_absolute(out, 'out')
    if python_out is not None:
        python_out = require_absolute(python_out, 'python_out')
        if python_out == out:
            raise ValueError('python_out must differ from out')
    if model_dir is not None:
        model_dir = require_absolute(model_dir, 'model_dir')
    if variant not in RUST_VARIANTS:
        raise ValueError(f'variant must be one of {list(RUST_VARIANTS)}, got {variant!r}')
    _positive_int(views, 'views')
    _positive_int(iterations, 'iterations')
    _positive_int(token_count, 'token_count')
    source = 'random' if model_dir is None else model_dir
    rust = (f'cargo run --release -p apxinf-model --features cuda --example pi05_bench -- '
            f'{quote(source)} --model-variant {variant} --views {views} '
            f'--token-count {token_count} --iterations {iterations} > {quote(out)}')
    lines = [
        '#!/usr/bin/env bash',
        '# ApxInf pi0.5 latency baseline generated by FlashRT; FlashRT never executes it.',
        f'# Pinned infinigence/ApxInf {UPSTREAM_COMMIT}; A800 (sm_80) is unvalidated upstream.',
        f'# Boundary: {BOUNDARY_NOTE}',
        '# The Rust bench runs 10 untimed replays, then times `iterations` replays.',
        'set -euo pipefail',
        f'cd {quote(apxinf_root)}',
        'export APXINF_CUDA_ARCH="${APXINF_CUDA_ARCH:-sm_80}"',
        'echo "APXINF_HEAD=$(git rev-parse HEAD)"',
        f'mkdir -p {quote(str(Path(out).parent))}',
    ]
    if variant == 'fp8_static':
        lines.append('# fp8_static needs FP8 tensor cores (sm_89+): not available on A800.')
    lines += ['set +e', rust, 'RC=$?', 'set -e',
              f'echo "APXINF_RUST_RC=$RC OUT="{quote(out)}']
    if python_out is not None:
        python = (f'python scripts/bench_pi05.py --layer l1 --model-variant {variant}')
        if model_dir is None:
            python += f' --random-weights --views {views} --token-count {token_count}'
        else:
            python += f' --model-dir {quote(model_dir)}'
            lines.append('# Python checkpoint mode: views and token count come from the '
                         'checkpoint and prompt, not from --views/--token-count; the '
                         f'prompt must have {token_count} tokens to match the Rust bench.')
        python += f' --warmup 10 --samples {iterations} --out {quote(python_out)}'
        lines += [f'mkdir -p {quote(str(Path(python_out).parent))}', 'set +e', python,
                  'PY_RC=$?', 'set -e', f'echo "APXINF_PY_RC=$PY_RC OUT="{quote(python_out)}',
                  'if [ "$PY_RC" -ne 0 ] && [ "$RC" -eq 0 ]; then RC=$PY_RC; fi']
    lines.append('exit $RC')
    return '\n'.join(lines) + '\n'


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _stats(record):
    """Numeric latency statistics present in one stats object; nothing invented."""
    if not isinstance(record, dict):
        return None
    names = {'p50': 'p50', 'p95': 'p95', 'min': 'min', 'max': 'max', 'mean': 'mean',
             'std': 'std', 'standard_deviation': 'std', 'samples': 'samples'}
    stats = {target: record[source] for source, target in names.items()
             if _finite(record.get(source))}
    if 'samples' not in stats and isinstance(record.get('samples_ms'), list):
        stats['samples'] = len(record['samples_ms'])
    return stats or None


def _read_document(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        document = json.loads(text)
    except ValueError:
        document = None
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith('{'):
                try:
                    document = json.loads(line)
                    break
                except ValueError:
                    continue
    if not isinstance(document, dict):
        raise ValueError(f'{path} holds no JSON object')
    return document


def parse_report(path) -> dict:
    """Read an ApxInf benchmark report (Rust pi05_bench or Python latency.v2).

    Returns {"schema", "path", "source_format", "latency_ms": {segment: stats},
    "passed", "model_variant", "token_count", "boundary", "notes", "raw"}. Only
    numbers present in the report are extracted; missing fields stay absent.
    """
    document = _read_document(path)
    latency, notes = {}, []
    if document.get('schema') == 'apxinf.pi05.latency.v2':
        source_format = 'apxinf.pi05.latency.v2'
        layers = document.get('layers_ms')
        if isinstance(layers, dict):
            for name, record in layers.items():
                stats = _stats(record)
                if stats:
                    latency[str(name)] = stats
        segments = document.get('l3_segments_ms')
        if isinstance(segments, dict):
            for name, record in segments.items():
                stats = _stats(record)
                if stats:
                    latency[f'L3.{name}'] = stats
        token_count = (document.get('workload') or {}).get('token_count') if isinstance(
            document.get('workload'), dict) else None
        passed = None
        notes.append('Python bench P50 is nearest-rank int(0.5*(n-1))')
    elif 'latency_ms' in document or 'image_input' in document:
        source_format = 'apxinf.pi05_bench'
        stats = _stats(document.get('latency_ms'))
        if stats:
            latency['graph_replay'] = stats
        image_input = document.get('image_input')
        if isinstance(image_input, dict):
            stats = _stats(image_input.get('input_update_plus_graph_latency_ms'))
            if stats:
                latency['input_update_plus_graph'] = stats
            if image_input.get('graph_includes_cuda_preprocess') is True:
                notes.append('graph includes CUDA preprocessing')
        token_count = document.get('token_count')
        passed = document.get('passed') if isinstance(document.get('passed'), bool) else None
        notes.append('Rust bench P50 index is round((n-1)*0.5)')
    else:
        source_format = 'unknown'
        token_count, passed = None, None
        notes.append('unrecognized report format; no latency extracted')
    if not latency and source_format != 'unknown':
        notes.append('no latency statistics found in the report')
    variant = document.get('model_variant')
    return {
        'schema': REPORT_SCHEMA,
        'path': str(Path(path).resolve()),
        'source_format': source_format,
        'latency_ms': latency,
        'passed': passed,
        'model_variant': variant if isinstance(variant, str) else None,
        'token_count': token_count if _finite(token_count) else None,
        'boundary': BOUNDARY_NOTE,
        'notes': notes,
        'raw': jsonable(document),
    }
