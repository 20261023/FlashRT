"""CPU-only declarative capabilities and lazy adapter/frontend construction."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module

from .adapters import get_hook
from .backend import VLABackend
from .config import BackendConfig
from .model_spec import discover_specs, get_spec, validate_options


def _capability_view(spec):
    """Keep the historical capability keys as a view of the validated spec."""
    caps = spec['capabilities']
    result = {**spec, 'precision': caps['precision'],
              'frameworks': tuple(caps['frameworks']), 'hardware': tuple(caps['hardware']),
              'options': tuple(caps['options']), 'action_domain': spec['actions']['domain']}
    coverage = caps['execution_coverage'].get('flashrt')
    if coverage is not None:
        result['execution_coverage'] = coverage
    return result


# Compatibility for existing CPU tool consumers. This is derived data, never a
# second source of model names, dimensions, policies or frontend routing.
_CAPABILITIES = {model: _capability_view(spec) for model, spec in discover_specs().items()}


def _spec_for(config):
    if config.spec_dir is None:
        return _CAPABILITIES[config.model]
    return _capability_view(get_spec(config.model, spec_dir=config.spec_dir))


def _execution_precision(config, spec):
    hook = get_hook(spec, 'precision')
    return hook(config) if hook is not None else spec['capabilities']['precision']


def describe_backend(config: BackendConfig) -> dict:
    """Describe a declaration without checkpoint access, CUDA, Torch or loading."""
    spec = _spec_for(config)
    caps, actions = spec['capabilities'], spec['actions']
    result = {
        'model': config.model, 'name': spec['name'], 'backend': config.backend,
        'sampler': config.sampler, 'framework': config.framework, 'hardware': config.hardware,
        'precision': _execution_precision(config, spec) if config.precision == 'auto' else config.precision,
        'num_steps': config.num_steps,
        'probeflow': None if config.probeflow is None else asdict(config.probeflow),
        'frameworks': list(caps['frameworks']), 'hardware_paths': list(caps['hardware']),
        'action_domain': ('resolved_from_checkpoint_on_load' if caps['action_domain_resolution'] == 'checkpoint'
                          else actions['domain']),
        'default_action_domain': actions['domain'], 'options': dict(config.options),
        'head_type': spec['head']['type'], 'model_spec_version': spec['schema_version'],
        'worker_action_domain': actions['worker_domain'],
        'effective_action_dim': actions['effective_dim'], 'action_channels': actions['channels'],
        'identity': spec['identity'], 'runtime_verified': False,
    }
    coverage = caps['execution_coverage'].get(config.backend)
    if coverage is not None:
        result['execution_coverage'] = coverage
    if 'blocker' in caps:
        result['blocker'] = caps['blocker']
    if config.spec_dir is not None:
        result['spec_dir'] = config.spec_dir
    return result


# CUDA capability -> hardware label, also read by CPU tool filtering.
_CUDA_TARGETS = {(8, 0): 'cuda_sm80', (8, 7): 'rtx_sm87', (8, 9): 'rtx_sm89',
                 (11, 0): 'thor', (12, 0): 'rtx_sm120'}


def _cuda_target(requested):
    import torch
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None):
        raise RuntimeError('This first VLA backend requires an NVIDIA CUDA device')
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in _CUDA_TARGETS:
        raise ValueError(f'No VLA backend path for CUDA capability {capability}')
    actual = _CUDA_TARGETS[capability]
    if requested != 'auto' and requested != actual:
        raise ValueError(f'Requested hardware={requested!r}, current device is {actual!r}')
    return actual


def validate_config(config: BackendConfig) -> dict:
    """Validate a declaration and adapter policy before touching any device.

    The returned spec retains historical capability keys for existing callers.
    Adapter validation must remain CPU-only; checkpoint/device checks belong in
    loading. The same validation always runs at the start of ``load_backend``.
    """
    spec = _spec_for(config)
    caps = spec['capabilities']
    if 'blocker' in caps:
        raise NotImplementedError(caps['blocker'])
    if config.framework not in caps['frameworks']:
        raise ValueError(f'{config.model} has no {config.framework} frontend in this backend')
    if config.framework not in caps['backends'].get(config.backend, []):
        raise ValueError(f'Unsupported {config.model} backend/framework combination')
    precision = _execution_precision(config, spec)
    if config.precision not in ('auto', precision):
        raise ValueError(f'{config.model} backend uses {precision}, not {config.precision}')
    validate_options(spec, config.options)
    validator = get_hook(spec, 'validate')
    if validator is not None:
        validator(config)
    return spec


def load_backend(config: BackendConfig) -> VLABackend:
    """Construct the explicitly declared backend; never substitute a model."""
    spec = validate_config(config)
    binding = get_hook(spec, 'binding')
    if binding is None:
        raise NotImplementedError(f'{config.model} declares no runtime adapter binding')
    precision = _execution_precision(config, spec)
    decoder_hook = get_hook(spec, 'decoder')
    decoder = decoder_hook(config) if decoder_hook is not None else None
    hardware = _cuda_target(config.hardware)
    if hardware not in spec['capabilities']['hardware']:
        raise NotImplementedError(f'{config.model} {precision} path is not adapted for {hardware}')
    module, name, kwargs = binding(config, hardware)
    frontend = getattr(import_module(module), name)(config.checkpoint, **kwargs)
    return VLABackend(config, frontend, hardware=hardware, precision=precision,
                      action_domain=decoder.action_domain if decoder is not None else spec['actions']['domain'],
                      action_decoder=decoder)
