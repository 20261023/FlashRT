"""Lazy model-specific policy hooks for the declarative VLA registry.

Importing a spec never imports this package. Hook modules must keep validation
CPU-only; runtime libraries belong inside binding targets or runtime functions.
"""

from importlib import import_module


def get_hook(spec, name):
    """Resolve a declared callable only when its operation is explicitly used."""
    reference = spec['adapter']['hooks'].get(name)
    if reference is None:
        return None
    module, symbol = reference.split(':')
    hook = getattr(import_module(module), symbol)
    if not callable(hook):
        raise TypeError(f'Adapter hook {reference!r} is not callable')
    return hook


def frontend_kwargs(config):
    kwargs = dict(config.options)
    kwargs.update(sampler=config.sampler, probeflow_config=config.probeflow)
    if config.num_steps is not None:
        kwargs['num_steps'] = config.num_steps
    return kwargs
