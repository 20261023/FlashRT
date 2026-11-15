"""Capture the operator contract of one call site while a workload replays.

``capture_calls`` temporarily replaces ``owner.attr`` with a recording wrapper,
then always restores it. It records shapes, dtypes, strides and contiguity of
tensor-like arguments (anything with ``.shape`` and ``.dtype``; numpy arrays
count), optionally saves the first call's tensors, and on a clean exit derives
a :class:`StageCapture`. Torch is imported only when torch tensors are saved.

Run captures eagerly, never inside CUDA graph capture: saving copies tensors
to the host. ``in_cuda_graph`` records how the call site runs in production.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import functools
import inspect
import math
from pathlib import Path
import zipfile

from .artifacts import OP_KINDS, ROLES, StageCapture, dtype_name, write_json


CAPTURE_FILE = 'stage_capture.json'
CALLS_FILE = 'stage_calls.json'
CALLS_SCHEMA = 'flash_rt.autotune.stage_calls.v1'
TORCH_TENSORS_FILE = 'stage_tensors.pt'
NUMPY_TENSORS_FILE = 'stage_tensors.npz'

_WEIGHT_NAMES = ('weight', 'bias', 'w', 'b')


def is_tensor_like(value) -> bool:
    return hasattr(value, 'shape') and hasattr(value, 'dtype') and not isinstance(
        value, type)


def _is_torch(value) -> bool:
    return type(value).__module__.split('.')[0] == 'torch'


def _strides(value):
    stride = getattr(value, 'stride', None)
    if callable(stride):
        return [int(item) for item in stride()]
    strides = getattr(value, 'strides', None)
    itemsize = getattr(value, 'itemsize', None)
    if strides is not None and itemsize:
        return [int(item) // int(itemsize) for item in strides]
    return None


def _contiguous(value):
    method = getattr(value, 'is_contiguous', None)
    if callable(method):
        return bool(method())
    flags = getattr(value, 'flags', None)
    if flags is not None:
        return bool(flags['C_CONTIGUOUS'])
    return None


def describe_value(name, value) -> dict:
    """JSON description of one argument or output."""
    if is_tensor_like(value):
        record = {
            'name': name, 'kind': 'tensor', 'shape': [int(dim) for dim in value.shape],
            'dtype': dtype_name(value.dtype), 'strides': _strides(value),
            'contiguous': _contiguous(value),
        }
        device = getattr(value, 'device', None)
        record['device'] = 'cpu' if device is None else str(device)
        return record
    if value is None or isinstance(value, (bool, int, str)):
        return {'name': name, 'kind': 'scalar', 'value': value}
    if isinstance(value, float):
        return {'name': name, 'kind': 'scalar',
                'value': value if math.isfinite(value) else repr(value)}
    kind = type(value)
    return {'name': name, 'kind': 'object', 'type': f'{kind.__module__}.{kind.__qualname__}'}


def _named_arguments(signature, args, kwargs):
    """(name, value) pairs in call order; positional names fall back to argN."""
    if signature is not None:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            pairs = []
            for name, value in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_POSITIONAL:
                    pairs += [(f'{name}{index}', item) for index, item in enumerate(value)]
                elif kind is inspect.Parameter.VAR_KEYWORD:
                    pairs += list(value.items())
                else:
                    pairs.append((name, value))
            return pairs
    return [(f'arg{index}', value) for index, value in enumerate(args)] + list(kwargs.items())


def _named_outputs(result, taken):
    """(name, tensor) pairs of a result; names never collide with ``taken``."""
    if is_tensor_like(result):
        pairs = [('out', result)]
    elif isinstance(result, (tuple, list)):
        pairs = [(f'out{index}', item) for index, item in enumerate(result)
                 if is_tensor_like(item)]
    elif isinstance(result, Mapping):
        pairs = [(str(key), item) for key, item in result.items() if is_tensor_like(item)]
    else:
        pairs = []
    named, used = [], set(taken)
    for name, item in pairs:
        while name in used:
            name = f'{name}_out'
        used.add(name)
        named.append((name, item))
    return named


def _host_copy(value):
    if _is_torch(value):
        return value.detach().to('cpu', copy=True)
    import numpy as np
    return np.array(value, copy=True)


def _save_tensors(out_dir, tensors) -> Path:
    """torch.save when any tensor is a torch tensor, numpy .npz otherwise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(_is_torch(value) for value in tensors.values()):
        import torch
        path = out_dir / TORCH_TENSORS_FILE
        torch.save({name: value if _is_torch(value) else torch.from_numpy(value)
                    for name, value in tensors.items()}, path)
        return path
    import numpy as np
    path = out_dir / NUMPY_TENSORS_FILE
    # Written entry by entry: np.savez(**names) breaks on an argument named 'file'.
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in tensors.items():
            with archive.open(f'{name}.npy', 'w', force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)
    return path


class CallRecorder:
    """What a ``capture_calls`` block saw; ``capture`` is set on a clean exit."""

    def __init__(self, owner, attr, max_calls):
        self.owner_type = f'{type(owner).__module__}.{type(owner).__qualname__}'
        self.attr = attr
        self.max_calls = max_calls
        self.total_calls = 0
        self.calls = []
        self.tensors_path = None
        self.capture = None
        self.capture_path = None


def _wrap(function, recorder, out_dir, save_tensors):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        index = recorder.total_calls
        recorder.total_calls += 1
        if index >= recorder.max_calls:
            return function(*args, **kwargs)
        named = _named_arguments(signature, args, kwargs)
        saved = None
        if index == 0 and out_dir is not None and save_tensors:
            saved = {name: _host_copy(value) for name, value in named
                     if is_tensor_like(value)}
        result = function(*args, **kwargs)
        outputs = _named_outputs(result, [name for name, _ in named])
        recorder.calls.append({
            'index': index,
            'arguments': [describe_value(name, value) for name, value in named],
            'outputs': [describe_value(name, value) for name, value in outputs],
        })
        if saved is not None:
            for name, value in outputs:
                saved[name] = _host_copy(value)
            recorder.tensors_path = str(_save_tensors(out_dir, saved).resolve())
        return result

    return wrapper


class _BindingWrapper:
    """Class attribute that binds like ``raw`` (its type's ``__get__``) and records calls."""

    def __init__(self, raw, recorder, out_dir, save_tensors):
        self._raw = raw
        self._recorder = recorder
        self._out_dir = out_dir
        self._save_tensors = save_tensors

    def __get__(self, instance, owner=None):
        bound = type(self._raw).__get__(self._raw, instance, owner)
        return _wrap(bound, self._recorder, self._out_dir, self._save_tensors)


@contextmanager
def capture_calls(owner, attr, *, model, stage_id, call_site, op_kind, out_dir=None,
                  max_calls=8, in_cuda_graph=None, roles=None, reference=None,
                  save_tensors=True):
    """Record calls of ``owner.attr`` inside the block; restore it on exit.

    Works on modules, classes (plain, static and class methods, and non-binding
    callables such as builtins, partials and callable objects) and instances.
    Yields a :class:`CallRecorder`. On a clean exit the first recorded call
    defines the signature and every recorded call contributes a shape;
    ``recorder.capture`` holds the :class:`StageCapture`, and with ``out_dir``
    the capture is written to ``stage_capture.json`` (plus ``stage_calls.json``
    with strides/contiguity per call and the first call's tensors). A clean exit
    with no recorded call raises ValueError when ``out_dir`` is given.
    """
    if not isinstance(attr, str) or not attr.isidentifier():
        raise ValueError(f'attr must be an attribute name, got {attr!r}')
    if op_kind not in OP_KINDS:
        raise ValueError(f'op_kind must be one of {list(OP_KINDS)}, got {op_kind!r}')
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError(f'max_calls must be a positive int, got {max_calls!r}')
    if roles is not None:
        if not isinstance(roles, Mapping):
            raise ValueError('roles must map argument names to roles')
        for name, role in roles.items():
            if role not in ROLES:
                raise ValueError(f'roles[{name!r}] must be one of {list(ROLES)}, got {role!r}')
    # Validate identity fields before patching anything.
    StageCapture(model=model, stage_id=stage_id, call_site=call_site, op_kind=op_kind,
                 signature=[{'name': 'x', 'shape': [1], 'dtype': 'float32', 'role': 'input'},
                            {'name': 'y', 'shape': [1], 'dtype': 'float32', 'role': 'output'}],
                 shapes=[{'name': 'probe', 'axes': {'a': 1}}], dtype='float32',
                 in_cuda_graph=in_cuda_graph, reference=reference)
    raw = inspect.getattr_static(owner, attr)  # AttributeError when missing
    function = getattr(owner, attr)
    if not callable(function):
        raise ValueError(f'{attr!r} of {type(owner).__name__} is not callable')
    try:
        own = attr in vars(owner)
        previous = vars(owner)[attr] if own else None
    except TypeError:  # __slots__ instance: the value lives on the instance
        own, previous = True, function
    out_path = None if out_dir is None else Path(out_dir)
    recorder = CallRecorder(owner, attr, max_calls)
    descriptor = isinstance(owner, type) and isinstance(raw, (staticmethod, classmethod))
    if descriptor:
        replacement = type(raw)(_wrap(raw.__func__, recorder, out_path, save_tensors))
    elif isinstance(owner, type) and not inspect.isfunction(raw):
        # A plain wrapper function would bind ``self``; builtins, partials, bound
        # methods and callable objects on a class must keep their own binding.
        # The descriptor protocol looks on the type: a bound method proxies
        # ``__get__`` to its function, yet never rebinds.
        if hasattr(type(raw), '__get__'):
            replacement = _BindingWrapper(raw, recorder, out_path, save_tensors)
        else:
            replacement = staticmethod(_wrap(raw, recorder, out_path, save_tensors))
    else:
        replacement = _wrap(function, recorder, out_path, save_tensors)
    setattr(owner, attr, replacement)
    try:
        yield recorder
    finally:
        if own:
            setattr(owner, attr, previous)
        else:
            delattr(owner, attr)
    if not recorder.calls:
        if out_path is not None:
            raise ValueError(f'captured no call of {recorder.owner_type}.{attr}; '
                             f'nothing to write')
        return
    recorder.capture = build_capture(
        recorder.calls, model=model, stage_id=stage_id, call_site=call_site,
        op_kind=op_kind, in_cuda_graph=in_cuda_graph, roles=roles, reference=reference,
        tensors_path=recorder.tensors_path, total_calls=recorder.total_calls)
    if out_path is not None:
        write_json(out_path / CALLS_FILE, {
            'schema': CALLS_SCHEMA, 'owner_type': recorder.owner_type, 'attr': attr,
            'call_site': call_site, 'max_calls': max_calls,
            'total_calls': recorder.total_calls, 'calls': recorder.calls,
        })
        recorder.capture_path = recorder.capture.save(out_path / CAPTURE_FILE)


def _tensors(entries):
    return [entry for entry in entries if entry['kind'] == 'tensor']


def _role(name, op_kind, position, roles):
    if roles is not None and name in roles:
        return roles[name]
    if op_kind == 'linear':
        return 'input' if position == 0 else 'weight'
    if name in _WEIGHT_NAMES or name.endswith(('_weight', '_bias')):
        return 'weight'
    return 'input'


def _linear(call):
    inputs, outputs = _tensors(call['arguments']), _tensors(call['outputs'])
    if len(inputs) not in (2, 3) or len(outputs) != 1:
        raise ValueError(f'call {call["index"]}: linear needs 2 or 3 tensor arguments '
                         f'(x, weight[, bias]) and 1 tensor output, got {len(inputs)} and '
                         f'{len(outputs)}')
    x, weight = inputs[0], inputs[1]
    if not x['shape'] or len(weight['shape']) != 2:
        raise ValueError(f'call {call["index"]}: linear needs x[..., K] and weight[N, K]')
    n, k = weight['shape']
    if x['shape'][-1] != k:
        raise ValueError(f'call {call["index"]}: x last dim {x["shape"][-1]} != weight K {k}')
    m = math.prod(x['shape'][:-1])
    if len(inputs) == 3 and inputs[2]['shape'] != [n]:
        raise ValueError(f'call {call["index"]}: bias shape {inputs[2]["shape"]} != [{n}]')
    if outputs[0]['shape'] != x['shape'][:-1] + [n]:
        raise ValueError(f'call {call["index"]}: output shape {outputs[0]["shape"]} != '
                         f'{x["shape"][:-1] + [n]}')
    dims = [['M', 'K'], ['N', 'K'], ['N']][:len(inputs)] + [['M', 'N']]
    notes = []
    if len(x['shape']) != 2:
        notes.append(f'{x["name"]} flattened from {x["shape"]} to [M, K] = [{m}, {k}]; '
                     f'the output is [M, N] and the call site reshapes it back')
    return inputs, outputs, dims, {'M': m, 'K': k, 'N': n}, notes


def _attention(call):
    inputs, outputs = _tensors(call['arguments']), _tensors(call['outputs'])
    if len(inputs) != 3 or len(outputs) != 1:
        raise ValueError(f'call {call["index"]}: attention needs 3 tensor arguments '
                         f'(q, k, v) and 1 tensor output, got {len(inputs)} and {len(outputs)}')
    q, k, v = (entry['shape'] for entry in inputs)
    out = outputs[0]['shape']
    if any(len(shape) != 4 for shape in (q, k, v, out)):
        raise ValueError(f'call {call["index"]}: attention tensors must be 4-D [B, H, S, D]')
    b, h, s, d = q
    _, hkv, skv, dv = v
    if k != [b, hkv, skv, d] or v[:3] != [b, hkv, skv] or out != [b, h, s, dv]:
        raise ValueError(f'call {call["index"]}: inconsistent attention shapes q={q} k={k} '
                         f'v={v} out={out}')
    if h % hkv:
        raise ValueError(f'call {call["index"]}: query heads {h} not a multiple of kv heads '
                         f'{hkv}')
    dims = [['B', 'H', 'S', 'D'], ['B', 'HKV', 'SKV', 'D'], ['B', 'HKV', 'SKV', 'DV'],
            ['B', 'H', 'S', 'DV']]
    axes = {'B': b, 'H': h, 'HKV': hkv, 'S': s, 'SKV': skv, 'D': d, 'DV': dv}
    return inputs, outputs, dims, axes, []


def _custom(call):
    inputs, outputs = _tensors(call['arguments']), _tensors(call['outputs'])
    tensors = inputs + outputs
    if any(not entry['shape'] for entry in tensors):
        raise ValueError(f'call {call["index"]}: 0-d tensors cannot be described by axes')
    dims = [[f'{entry["name"]}_{axis}' for axis in range(len(entry['shape']))]
            for entry in tensors]
    axes = {name: size for entry, names in zip(tensors, dims)
            for name, size in zip(names, entry['shape'])}
    return inputs, outputs, dims, axes, []


_DERIVE = {'linear': _linear, 'attention': _attention, 'custom': _custom}


def _shape_name(axes):
    return '-'.join(f'{axis}{size}' for axis, size in axes.items())


def build_capture(calls, *, model, stage_id, call_site, op_kind, in_cuda_graph=None,
                  roles=None, reference=None, tensors_path=None, total_calls=None):
    """Derive a StageCapture from recorded calls (first call = signature)."""
    derive = _DERIVE[op_kind]
    inputs, outputs, dims, _, notes = derive(calls[0])
    tensors = inputs + outputs
    roles_in_order = [_role(entry['name'], op_kind, position, roles)
                      for position, entry in enumerate(inputs)] + ['output'] * len(outputs)
    signature = [{'name': entry['name'], 'shape': shape, 'dtype': entry['dtype'],
                  'role': role}
                 for entry, shape, role in zip(tensors, dims, roles_in_order)]
    shapes, seen = [], set()
    for call in calls:
        call_inputs, call_outputs, _, axes, _ = derive(call)
        call_tensors = call_inputs + call_outputs
        names = [entry['name'] for entry in call_tensors]
        if names != [entry['name'] for entry in tensors]:
            raise ValueError(f'call {call["index"]}: tensor arguments {names} differ from the '
                             f'first call\'s {[entry["name"] for entry in tensors]}')
        key = tuple(axes.items())
        if key not in seen:
            seen.add(key)
            shapes.append({'name': _shape_name(axes), 'axes': axes})
        for entry in call_tensors:
            if entry['contiguous'] is False:
                notes.append(f'call {call["index"]}: {entry["name"]} is non-contiguous '
                             f'(strides {entry["strides"]})')
    if total_calls is not None and total_calls > len(calls):
        notes.append(f'recorded {len(calls)} of {total_calls} calls (max_calls)')
    first_input = next(entry for entry in signature if entry['role'] != 'output')
    return StageCapture(
        model=model, stage_id=stage_id, call_site=call_site, op_kind=op_kind,
        signature=signature, shapes=shapes, dtype=first_input['dtype'],
        in_cuda_graph=in_cuda_graph, tensors_path=tensors_path, reference=reference,
        notes=notes)
