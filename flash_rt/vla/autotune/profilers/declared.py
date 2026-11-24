"""Generic method/graph paths for external model declarations (stdlib only)."""

from collections.abc import Mapping
from importlib import import_module


def _targets(owner, path):
    """Yield existing terminal attributes/items; absent paths remain unknown."""
    parts = path.split('.')
    if not parts or any(part != '*' and not part.isidentifier() for part in parts):
        raise ValueError(f'Invalid profiling path {path!r}')

    def walk(current, remaining):
        part, *rest = remaining
        if part == '*':
            if isinstance(current, Mapping):
                children = list(current.items())
            elif isinstance(current, (list, tuple)):
                children = list(enumerate(current))
            else:
                return
            item = True
        elif isinstance(current, Mapping):
            children = [(part, current[part])] if part in current else []
            item = True
        else:
            children = [(part, getattr(current, part))] if hasattr(current, part) else []
            item = False
        for key, value in children:
            if value is None:
                continue
            if rest:
                yield from walk(value, rest)
            else:
                yield current, key, value, item

    yield from walk(owner, parts)


def install(session):
    declared = session.declaration.get('profiling', {})
    strategy = declared.get('strategy', {})
    kind = strategy.get('kind')
    if kind == 'adapter':
        reference = strategy.get('hook')
        if not reference:
            raise ValueError('Adapter profiling strategy requires a hook')
        module, name = reference.split(':', 1)
        return getattr(import_module(module), name)(session)
    if kind not in ('graphs', 'methods'):
        session.notes.append('No profiling adapter or method/graph strategy declared: timing is unknown')
        return
    mapping = declared.get('keys', {})
    session.expected = list(declared.get('expected_keys', {}).get(session.backend.config.sampler, ()))
    paths = strategy.get(kind, [])
    for item in paths:
        key = item.get('key') or item.get('key_by_sampler', {}).get(session.backend.config.sampler)
        if key is None:
            continue
        if key not in mapping:
            raise ValueError(f'Profiling strategy key {key!r} has no declared stages')
        path = item['path']
        options = {key: item[key] for key in ('additive', 'scope') if key in item}
        found = list(_targets(session.frontend, path))
        if not found:
            session.notes.append(f'Profiling path {path!r} is unavailable: {key!r} remains unknown')
        for owner, name, value, is_item in found:
            if kind == 'graphs':
                timed = session.graph(value, key, mapping[key], f'Declared graph replay: {path}', **options)
            elif item.get('clock', 'cuda_event') == 'perf_counter_ns':
                if is_item:
                    raise ValueError('Host method profiling requires an instance method path')
                session.host_method(owner, name, key, mapping[key], f'Declared host method: {path}', **options)
                continue
            elif item.get('clock', 'cuda_event') == 'cuda_event':
                timed = session.timed(value, key, mapping[key], f'Declared method call: {path}', **options)
            else:
                raise ValueError(f'Unsupported profiling clock {item["clock"]!r}')
            if is_item:
                session.replace_item(owner, name, timed)
            else:
                session.replace(owner, name, timed)
