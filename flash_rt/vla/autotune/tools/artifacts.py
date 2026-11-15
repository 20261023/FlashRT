"""Immutable JSON artifacts exchanged between FlashRT and external autotune tools.

``StageCapture`` describes one captured call site (the operator contract a tool
is asked to beat), ``KernelArtifact`` is what a kernel tool returned after the
deterministic acceptance rule ran, and ``PatchProposal`` is a source change
stored by content hash. Each class validates on construction, round-trips
through ``to_dict``/``from_dict`` and saves canonical JSON atomically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import keyword
import math
import os
from pathlib import Path
import re
import tempfile


STAGE_CAPTURE_SCHEMA = 'flash_rt.autotune.stage_capture.v1'
KERNEL_ARTIFACT_SCHEMA = 'flash_rt.autotune.kernel_artifact.v1'
PATCH_PROPOSAL_SCHEMA = 'flash_rt.autotune.patch_proposal.v1'

OP_KINDS = ('linear', 'attention', 'custom')
ROLES = ('input', 'weight', 'output')
LANGUAGES = ('triton', 'cuda', 'cutedsl')
# ForgeKernel's judge dtypes plus uint8 (raw image inputs).
DTYPES = (
    'float64', 'float32', 'float16', 'bfloat16', 'float8_e4m3fn', 'float8_e5m2',
    'int64', 'int32', 'int16', 'int8', 'uint8', 'bool',
)

_ID_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]*$')
_SHAPE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_AXIS_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_PATCH_REF_RE = re.compile(r'^sha256:[0-9a-f]{64}$')


def _json_value(value, name):
    """Detached copy of JSON data; rejects nonfinite floats and opaque objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{name} holds a nonfinite number')
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f'{name} object keys must be strings')
        return {key: _json_value(item, name) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name) for item in value]
    raise ValueError(f'{name} holds a non-JSON value: {type(value).__name__}')


def _text(value, name, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f'{name} must be a {"nonempty " if nonempty else ""}string')
    return value


def _optional_text(value, name):
    return None if value is None else _text(value, name)


def _identifier(value, name):
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f'{name} must match {_ID_RE.pattern}, got {value!r}')
    return value


def _choice(value, choices, name):
    if value not in choices:
        raise ValueError(f'{name} must be one of {list(choices)}, got {value!r}')
    return value


def _schema(value, expected, name):
    if value != expected:
        raise ValueError(f'{name} schema must be {expected!r}, got {value!r}')
    return value


def _strings(value, name):
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f'{name} must be a list of strings')
    for item in value:
        _text(item, f'{name} item')
    return tuple(value)


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a dict')
    return _json_value(value, name)


def _fields(record, cls, name):
    if not isinstance(record, Mapping):
        raise ValueError(f'{name} must be a dict')
    allowed = set(cls.__dataclass_fields__)
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f'Unknown {name} fields: {sorted(map(str, unknown))}')
    return dict(record)


def dtype_name(dtype) -> str:
    """Canonical dtype name ('bfloat16') of a torch/numpy dtype or a string."""
    text = str(dtype)
    if text.startswith('torch.'):
        text = text[len('torch.'):]
    aliases = {'half': 'float16', 'float': 'float32', 'double': 'float64', 'bool_': 'bool'}
    return aliases.get(text, text)


def _signature_entry(entry, index, axes):
    where = f'signature[{index}]'
    if not isinstance(entry, Mapping) or set(entry) != {'name', 'shape', 'dtype', 'role'}:
        raise ValueError(f'{where} must have exactly the keys name, shape, dtype, role')
    name = entry['name']
    if (not isinstance(name, str) or not name.isidentifier()
            or keyword.iskeyword(name)):
        raise ValueError(f'{where} name must be a Python identifier, got {name!r}')
    shape = entry['shape']
    if isinstance(shape, str) or not isinstance(shape, (list, tuple)):
        raise ValueError(f'{where} ({name}) shape must be a list')
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, (int, str)):
            raise ValueError(f'{where} ({name}) dims must be axis names or positive ints')
        if isinstance(dim, int) and dim <= 0:
            raise ValueError(f'{where} ({name}) fixed dims must be positive, got {dim}')
        if isinstance(dim, str) and dim not in axes:
            raise ValueError(f'{where} ({name}) axis {dim!r} is not an axis of shapes '
                             f'{sorted(axes)}')
        dims.append(dim)
    dtype = _choice(entry['dtype'], DTYPES, f'{where} ({name}) dtype')
    role = _choice(entry['role'], ROLES, f'{where} ({name}) role')
    return {'name': name, 'shape': dims, 'dtype': dtype, 'role': role}


def _shape_entries(shapes):
    if isinstance(shapes, str) or not isinstance(shapes, (list, tuple)) or not shapes:
        raise ValueError('shapes must be a nonempty list of {name, axes}')
    rows, names, axis_set = [], set(), None
    for index, row in enumerate(shapes):
        where = f'shapes[{index}]'
        if not isinstance(row, Mapping) or set(row) != {'name', 'axes'}:
            raise ValueError(f'{where} must have exactly the keys name, axes')
        name = row['name']
        if not isinstance(name, str) or not _SHAPE_NAME_RE.match(name):
            raise ValueError(f'{where} name must match {_SHAPE_NAME_RE.pattern}, got {name!r}')
        if name in names:
            raise ValueError(f'{where} duplicate shape name {name!r}')
        names.add(name)
        axes = row['axes']
        if not isinstance(axes, Mapping) or not axes:
            raise ValueError(f'{where} ({name}) axes must be a nonempty dict')
        for axis, size in axes.items():
            if not isinstance(axis, str) or not _AXIS_RE.match(axis):
                raise ValueError(f'{where} ({name}) axis name {axis!r} is not an identifier')
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f'{where} ({name}) axis {axis!r} = {size!r} is not a '
                                 f'positive int')
        if axis_set is None:
            axis_set = set(axes)
        elif set(axes) != axis_set:
            raise ValueError(f'{where} ({name}) axes {sorted(axes)} differ from the first '
                             f"shape's {sorted(axis_set)}")
        rows.append({'name': name, 'axes': dict(axes)})
    return tuple(rows), axis_set


class _Artifact:
    """Shared JSON codec: to_dict/from_dict/save/load."""

    def to_dict(self) -> dict:
        record = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            record[name] = _json_value(value, name)
        return record

    @classmethod
    def from_dict(cls, record):
        return cls(**_fields(record, cls, cls.__name__))

    def save(self, path) -> Path:
        """Write canonical JSON atomically (temp file + os.replace)."""
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_json(path))


@dataclass(frozen=True)
class StageCapture(_Artifact):
    """One captured call site and the operator contract derived from it."""

    model: str
    stage_id: str
    call_site: str
    op_kind: str
    signature: tuple
    shapes: tuple
    dtype: str
    in_cuda_graph: bool | None = None
    tensors_path: str | None = None
    reference: str | None = None
    notes: tuple = ()
    schema: str = STAGE_CAPTURE_SCHEMA

    def __post_init__(self):
        _schema(self.schema, STAGE_CAPTURE_SCHEMA, 'StageCapture')
        _text(self.model, 'model')
        _identifier(self.stage_id, 'stage_id')
        _text(self.call_site, 'call_site')
        _choice(self.op_kind, OP_KINDS, 'op_kind')
        shapes, axes = _shape_entries(self.shapes)
        if isinstance(self.signature, str) or not isinstance(self.signature, (list, tuple)):
            raise ValueError('signature must be a list')
        signature = tuple(_signature_entry(entry, index, axes)
                          for index, entry in enumerate(self.signature))
        names = [entry['name'] for entry in signature]
        if len(set(names)) != len(names):
            raise ValueError(f'signature names must be unique, got {names}')
        roles = [entry['role'] for entry in signature]
        if 'output' not in roles or all(role == 'output' for role in roles):
            raise ValueError('signature needs at least one output and one non-output entry')
        _choice(self.dtype, DTYPES, 'dtype')
        if self.in_cuda_graph is not None and not isinstance(self.in_cuda_graph, bool):
            raise ValueError('in_cuda_graph must be a bool or None')
        _optional_text(self.tensors_path, 'tensors_path')
        _optional_text(self.reference, 'reference')
        object.__setattr__(self, 'signature', signature)
        object.__setattr__(self, 'shapes', shapes)
        object.__setattr__(self, 'notes', _strings(self.notes, 'notes'))

    @property
    def axes(self) -> tuple:
        """Axis names shared by every shape, in first-shape order."""
        return tuple(self.shapes[0]['axes'])

    def inputs(self) -> tuple:
        """Non-output signature entries, in call order."""
        return tuple(entry for entry in self.signature if entry['role'] != 'output')

    def outputs(self) -> tuple:
        return tuple(entry for entry in self.signature if entry['role'] == 'output')

    def concrete_shape(self, entry, shape_index=0) -> list:
        """Integer dims of a signature entry at one of the captured shapes."""
        axes = self.shapes[shape_index]['axes']
        return [axes[dim] if isinstance(dim, str) else dim for dim in entry['shape']]


@dataclass(frozen=True)
class KernelArtifact(_Artifact):
    """A kernel returned by an external tool, after FlashRT's acceptance rule.

    ``accepted`` is True exactly when ``reasons`` is empty; a rejected artifact
    lists every failed check. ``kernel_path``/``kernel_sha256``/``language`` are
    None only when the tool produced no kernel file.
    """

    tool_id: str
    upstream_commit: str
    stage_id: str
    model: str
    language: str | None
    entry_symbol: str
    kernel_path: str | None
    kernel_sha256: str | None
    op_metrics: dict
    accepted: bool
    reasons: tuple
    provenance: dict
    schema: str = KERNEL_ARTIFACT_SCHEMA

    def __post_init__(self):
        _schema(self.schema, KERNEL_ARTIFACT_SCHEMA, 'KernelArtifact')
        _identifier(self.tool_id, 'tool_id')
        if not isinstance(self.upstream_commit, str) or not _COMMIT_RE.match(
                self.upstream_commit):
            raise ValueError(f'upstream_commit must be 40 lowercase hex, got '
                             f'{self.upstream_commit!r}')
        _identifier(self.stage_id, 'stage_id')
        _text(self.model, 'model')
        _text(self.entry_symbol, 'entry_symbol')
        has_kernel = self.kernel_path is not None
        if (self.kernel_sha256 is not None) != has_kernel or (
                self.language is not None) != has_kernel:
            raise ValueError('kernel_path, kernel_sha256 and language are all set or all None')
        if has_kernel:
            _text(self.kernel_path, 'kernel_path')
            if not isinstance(self.kernel_sha256, str) or not _SHA256_RE.match(
                    self.kernel_sha256):
                raise ValueError('kernel_sha256 must be 64 lowercase hex')
            _choice(self.language, LANGUAGES, 'language')
        if not isinstance(self.accepted, bool):
            raise ValueError('accepted must be a bool')
        reasons = _strings(self.reasons, 'reasons')
        if self.accepted and (reasons or not has_kernel):
            raise ValueError('an accepted artifact has a kernel and no rejection reasons')
        if not self.accepted and not reasons:
            raise ValueError('a rejected artifact must list at least one reason')
        object.__setattr__(self, 'reasons', reasons)
        object.__setattr__(self, 'op_metrics', _mapping(self.op_metrics, 'op_metrics'))
        object.__setattr__(self, 'provenance', _mapping(self.provenance, 'provenance'))


@dataclass(frozen=True)
class PatchProposal(_Artifact):
    """A source change against ``base_commit`` stored as ``patch_ref``."""

    base_commit: str
    patch_ref: str
    tool_id: str
    stage_id: str
    hypothesis: str
    options: dict
    provenance: dict
    schema: str = PATCH_PROPOSAL_SCHEMA

    def __post_init__(self):
        _schema(self.schema, PATCH_PROPOSAL_SCHEMA, 'PatchProposal')
        if not isinstance(self.base_commit, str) or not _COMMIT_RE.match(self.base_commit):
            raise ValueError(f'base_commit must be 40 lowercase hex, got {self.base_commit!r}')
        if not isinstance(self.patch_ref, str) or not _PATCH_REF_RE.match(self.patch_ref):
            raise ValueError(f'patch_ref must be sha256:<64 hex>, got {self.patch_ref!r}')
        _identifier(self.tool_id, 'tool_id')
        _identifier(self.stage_id, 'stage_id')
        _text(self.hypothesis, 'hypothesis')
        object.__setattr__(self, 'options', _mapping(self.options, 'options'))
        object.__setattr__(self, 'provenance', _mapping(self.provenance, 'provenance'))


_BY_SCHEMA = {
    STAGE_CAPTURE_SCHEMA: StageCapture,
    KERNEL_ARTIFACT_SCHEMA: KernelArtifact,
    PATCH_PROPOSAL_SCHEMA: PatchProposal,
}


def from_record(record):
    """Decode any artifact by its ``schema`` key."""
    if not isinstance(record, Mapping):
        raise ValueError('artifact record must be a dict')
    cls = _BY_SCHEMA.get(record.get('schema'))
    if cls is None:
        raise ValueError(f'Unknown artifact schema: {record.get("schema")!r}')
    return cls.from_dict(record)


def load_artifact(path):
    return from_record(read_json(path))


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, record) -> Path:
    """Atomically write canonical JSON (sorted keys, indent 2, trailing newline)."""
    path = Path(path)
    text = json.dumps(_json_value(record, 'record'), indent=2, sort_keys=True,
                      allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
    return path
