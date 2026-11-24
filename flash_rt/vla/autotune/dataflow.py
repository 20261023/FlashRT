"""Declared source/runtime ledgers; never profiler-derived HBM measurements."""

from importlib import import_module

from ..model_spec import get_spec


def describe_dataflow(backend):
    config = getattr(backend, 'config', None)
    if config is None or getattr(backend, 'frontend', None) is None:
        return None
    spec = get_spec(config.model, spec_dir=getattr(config, 'spec_dir', None))
    hook = spec.get('adapter', {}).get('hooks', {}).get('dataflow')
    hook = hook or spec.get('profiling', {}).get('dataflow')
    if hook is None:
        return None
    module, name = hook.split(':', 1)
    return getattr(import_module(module), name)(backend)
