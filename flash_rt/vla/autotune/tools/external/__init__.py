"""Adapters that prepare inputs for, and harvest outputs of, external autotune projects.

Adapters never run an external tool, never import torch/triton or the external
project at import time and never touch the network: ``prepare_*``/``render_*``
only write files or return text, ``harvest``/``parse_*`` only read files.
Each adapter module exposes a ``SPEC`` dict (see ``SPEC_KEYS``).

This module also holds the small helpers the adapters share (stdlib only).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import keyword
import math
from pathlib import Path, PurePosixPath
import shlex

from ..artifacts import StageCapture


ADAPTERS = {
    'kernel.forgekernel': 'flash_rt.vla.autotune.tools.external.forgekernel',
    'kernel.kernelagent': 'flash_rt.vla.autotune.tools.external.kernelagent',
    'engine.apxinf': 'flash_rt.vla.autotune.tools.external.apxinf',
    'compile.aitune': 'flash_rt.vla.autotune.tools.external.aitune',
}
SPEC_KEYS = frozenset({
    'tool_id', 'title', 'lane', 'stages', 'models', 'exactness', 'status', 'reason',
    'upstream', 'skills', 'setup', 'risks',
})
SPEC_OPTIONAL_KEYS = frozenset({
    'model_predicate', 'status_rules', 'frameworks', 'samplers', 'min_capability',
    'hardware_only', 'aliases', 'requirements_path',
})
UPSTREAM_KEYS = frozenset({'repo', 'commit', 'license', 'notice'})
FLOAT_DTYPES = ('bfloat16', 'float16', 'float32')
# Names the generated Python files define or import at module level.
RESERVED_NAMES = frozenset({
    'torch', 'nn', 'F', 'np', 'json', 'math', 'os', 'sys', 'pathlib', 'run', 'Model',
    'kernel_function', 'main',
})


def adapter_module(tool_id):
    """Import one adapter module by tool id (stdlib-only imports)."""
    if tool_id not in ADAPTERS:
        raise ValueError(f'Unknown external tool {tool_id!r}; known: {sorted(ADAPTERS)}')
    return importlib.import_module(ADAPTERS[tool_id])


def external_specs() -> dict:
    """tool_id -> SPEC of every external adapter."""
    return {tool_id: adapter_module(tool_id).SPEC for tool_id in ADAPTERS}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require_absolute(value, name) -> str:
    """An absolute POSIX path string (the job may run on another machine)."""
    text = str(value) if isinstance(value, Path) else value
    if not isinstance(text, str) or not text or not PurePosixPath(text).is_absolute():
        raise ValueError(f'{name} must be an absolute path, got {value!r}')
    return text


def quote(value) -> str:
    return shlex.quote(str(value))


def fresh_dir(path, name) -> Path:
    """Create ``path`` (parents included); refuse a non-empty existing directory."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'{name} {path} exists and is not a directory')
        if any(path.iterdir()):
            raise ValueError(f'{name} {path} is not empty; refusing to overwrite')
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_contract(capture) -> tuple:
    """(inputs, outputs) of a capture whose op_kind a generated template supports.

    ``linear``: x[M, K], weight[N, K], optional bias[N] -> one [M, N] output.
    ``attention``: q, k, v (4-D, SDPA layout) -> one 4-D output.
    ``custom``: needs ``capture.reference``, a Python file defining ``run``.
    """
    if not isinstance(capture, StageCapture):
        raise TypeError('capture must be a StageCapture')
    inputs, outputs = capture.inputs(), capture.outputs()
    for entry in capture.signature:
        if entry['name'] in RESERVED_NAMES:
            raise ValueError(f'signature name {entry["name"]!r} collides with a name the '
                             f'generated files define; rename it in the capture')
    if capture.op_kind == 'linear':
        ranks = [len(entry['shape']) for entry in inputs]
        if ranks not in ([2, 2], [2, 2, 1]) or len(outputs) != 1 or len(
                outputs[0]['shape']) != 2:
            raise ValueError(f'linear needs x[M, K], weight[N, K], optional bias[N] and one '
                             f'[M, N] output; got input ranks {ranks} and {len(outputs)} '
                             f'output(s)')
    elif capture.op_kind == 'attention':
        ranks = [len(entry['shape']) for entry in inputs]
        if ranks != [4, 4, 4] or len(outputs) != 1 or len(outputs[0]['shape']) != 4:
            raise ValueError(f'attention needs 4-D q, k, v and one 4-D output; got input '
                             f'ranks {ranks} and {len(outputs)} output(s)')
    else:
        reference_source(capture)
    if capture.op_kind != 'custom':
        for entry in inputs + outputs:
            if entry['dtype'] not in FLOAT_DTYPES:
                raise ValueError(f'{capture.op_kind} template needs {list(FLOAT_DTYPES)} '
                                 f'tensors; {entry["name"]} is {entry["dtype"]}')
    return inputs, outputs


def reference_source(capture) -> str:
    """Text of a custom capture's reference file; it must define ``run(<inputs>)``."""
    if capture.reference is None:
        raise ValueError('op_kind custom requires capture.reference (a Python file '
                         'defining run(<inputs in signature order>))')
    path = Path(capture.reference)
    if path.suffix != '.py' or not path.is_file():
        raise ValueError(f'capture.reference {path} is not an existing .py file')
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text, filename=str(path))
    runs = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'run']
    if not runs:
        raise ValueError(f'capture.reference {path} defines no top-level function run')
    arguments = runs[-1].args
    positional = arguments.posonlyargs + arguments.args
    expected = len(capture.inputs())
    if arguments.vararg is None and len(positional) != expected:
        raise ValueError(f'capture.reference run takes {len(positional)} positional '
                         f'arguments; the signature has {expected} inputs')
    return text


def jsonable(value):
    """Copy of data read from a tool's files that FlashRT artifacts can hold:
    nonfinite floats and non-JSON objects become their repr strings."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


def is_identifier(name) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def dims_text(entry) -> str:
    return '[' + ', '.join(str(dim) for dim in entry['shape']) + ']'
