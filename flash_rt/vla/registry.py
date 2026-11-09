"""Lightweight model capabilities and lazy FlashRT frontend construction."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from numbers import Integral
import os

from .backend import VLABackend
from .config import BackendConfig


_CAPABILITIES = {
    'pi0': {
        'name': 'Physical Intelligence pi0', 'precision': 'fp16',
        'frameworks': ('torch', 'jax'),
        'hardware': ('cuda_sm80', 'rtx_sm89', 'rtx_sm120'),
        'action_domain': 'frontend-decoded-libero-7d',
        'options': ('num_views', 'chunk_size', 'max_prompt_len', 'use_cuda_graph'),
    },
    'pi05': {
        'name': 'Physical Intelligence pi0.5', 'precision': 'bf16',
        'frameworks': ('torch', 'jax'),
        'hardware': ('cuda_sm80', 'rtx_sm87', 'rtx_sm89', 'rtx_sm120'),
        'action_domain': 'frontend-decoded-libero-7d',
        'options': ('num_views', 'chunk_size', 'max_prompt_len', 'use_cuda_graph',
                    'state_prompt_mode', 'vision_pool_factor'),
    },
    'hyvla': {
        'name': 'Hy-Embodied-0.5-VLA', 'precision': 'bf16',
        'frameworks': ('torch',),
        'hardware': ('cuda_sm80', 'rtx_sm87', 'rtx_sm89', 'thor', 'rtx_sm120'),
        'action_domain': 'normalized-padded-model-actions',
        'options': ('use_cuda_graph', 'numerical_profile'),
    },
    'fastwam': {
        'name': 'FastWAM', 'precision': 'bf16',
        'frameworks': ('torch',),
        'hardware': ('cuda_sm80', 'rtx_sm87', 'rtx_sm89', 'thor', 'rtx_sm120'),
        'action_domain': 'normalized-model-actions-batched',
        'options': ('source_dir', 'model_config', 'backbone_dir',
                    'sigma_shift', 'action_horizon', 'native_linear_backend'),
        'execution_coverage': 'partial-native-linear-upstream-other-operators',
    },
}


def describe_backend(config: BackendConfig) -> dict:
    """Describe a candidate without importing Torch, loading weights or running it."""
    spec = _CAPABILITIES[config.model]
    return {
        'model': config.model, 'name': spec['name'], 'backend': config.backend,
        'sampler': config.sampler, 'framework': config.framework,
        'hardware': config.hardware,
        'precision': spec['precision'] if config.precision == 'auto' else config.precision,
        'num_steps': config.num_steps,
        'probeflow': None if config.probeflow is None else asdict(config.probeflow),
        'frameworks': list(spec['frameworks']),
        'hardware_paths': list(spec['hardware']),
        'action_domain': ('resolved_from_checkpoint_on_load' if config.model in ('pi0', 'pi05')
                          else spec['action_domain']),
        'default_action_domain': spec['action_domain'],
        'options': dict(config.options),
        'runtime_verified': False,
        **({'execution_coverage': spec['execution_coverage']}
           if 'execution_coverage' in spec else {}),
        **({'blocker': spec['blocker']} if 'blocker' in spec else {}),
    }


def _cuda_target(requested):
    import torch
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None):
        raise RuntimeError('This first VLA backend requires an NVIDIA CUDA device')
    capability = tuple(torch.cuda.get_device_capability())
    targets = {(8, 0): 'cuda_sm80', (8, 7): 'rtx_sm87', (8, 9): 'rtx_sm89',
               (11, 0): 'thor', (12, 0): 'rtx_sm120'}
    if capability not in targets:
        raise ValueError(f'No VLA backend path for CUDA capability {capability}')
    actual = targets[capability]
    if requested != 'auto' and requested != actual:
        raise ValueError(f'Requested hardware={requested!r}, current device is {actual!r}')
    return actual


def load_backend(config: BackendConfig) -> VLABackend:
    """Construct the requested backend; never substitute another model/method."""
    spec = _CAPABILITIES[config.model]
    if 'blocker' in spec:
        raise NotImplementedError(spec['blocker'])
    if config.framework not in spec['frameworks']:
        raise ValueError(f'{config.model} has no {config.framework} frontend in this backend')
    precision = spec['precision']
    if config.precision not in ('auto', precision):
        raise ValueError(f'{config.model} backend uses {precision}, not {config.precision}')
    unknown = set(config.options) - set(spec['options'])
    if unknown:
        raise ValueError(f'Unsupported {config.model} options: {sorted(unknown)}')
    for name in ('num_views', 'chunk_size', 'max_prompt_len'):
        if name in config.options:
            value = config.options[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
    if 'use_cuda_graph' in config.options and not isinstance(config.options['use_cuda_graph'], bool):
        raise ValueError('use_cuda_graph must be a bool')
    if config.options.get('state_prompt_mode', 'exact') not in ('exact', 'fixed'):
        raise ValueError('state_prompt_mode must be exact or fixed')
    if 'vision_pool_factor' in config.options:
        factor = config.options['vision_pool_factor']
        if type(factor) is not int or factor not in (1, 2, 4):
            raise ValueError('vision_pool_factor must be 1, 2 or 4')
    if config.model == 'hyvla' and config.options.get('numerical_profile', 'semantic_bf16') not in (
            'semantic_bf16', 'official_local_fp32'):
        raise ValueError('Unknown HyVLA numerical_profile')
    if config.model == 'fastwam':
        if config.options.get('native_linear_backend', 'legacy') not in ('legacy', 'cached'):
            raise ValueError('native_linear_backend must be legacy or cached')
        missing = {'sigma_shift', 'action_horizon', 'model_config', 'backbone_dir'} - set(config.options)
        if missing:
            raise ValueError(f'FastWAM requires explicit options: {sorted(missing)}')
        from flash_rt.models.fastwam.config import positive_integer, validate_sampler
        positive_integer(config.options['action_horizon'], 'action_horizon')
        validate_sampler(config.sampler, config.options['sigma_shift'],
                         config.num_steps, config.probeflow)
    if config.model == 'pi05':
        for name in ('FVK_PI05_RTX_FORCE_INT8', 'FVK_PI05_RTX_INT8_ENCODER_ONLY',
                     'FVK_PI05_RTX_INT8_VISION'):
            if os.environ.get(name, '0') == '1':
                raise ValueError(f'The BF16 VLA backend conflicts with {name}=1')
        for name, allowed in (
                ('FLASHRT_PI05_PREFIX_PRECISION', ('fp8',)),
                ('FLASHRT_PI05_DECODER_PRECISION', ('fp8',)),
                ('FLASHRT_PI05_DECODER_KERNEL', ('auto', 'cublaslt'))):
            if name in os.environ and os.environ[name] not in allowed:
                raise ValueError(f'The BF16 VLA backend conflicts with {name}={os.environ[name]}')
    decoder = None
    if config.model in ('pi0', 'pi05'):
        from .processing import load_action_decoder
        decoder = load_action_decoder(config.checkpoint)
    hardware = _cuda_target(config.hardware)
    if hardware not in spec['hardware']:
        raise NotImplementedError(f'{config.model} {precision} path is not adapted for {hardware}')
    kwargs = dict(config.options)
    kwargs.update(sampler=config.sampler, probeflow_config=config.probeflow)
    if config.num_steps is not None and config.model != 'fastwam':
        kwargs['num_steps'] = config.num_steps
    if config.model == 'fastwam':
        module = 'flash_rt.frontends.torch.fastwam'
        name = 'FastWAMTorchFrontend'
        # This binding is deliberately PARTIAL native execution, recorded in
        # both candidate descriptors and per-request execution metadata.
        kwargs.update(native_linear=True, dtype='bf16', device='cuda',
                      num_inference_steps=config.num_steps)
    elif config.model == 'hyvla':
        module = 'flash_rt.frontends.torch.hyvla_probeflow'
        name = 'HyVLAFlowFrontend'
        kwargs['hardware'] = hardware
    else:
        module = f'flash_rt.frontends.{config.framework}.{config.model}_rtx'
        name = ('Pi0' if config.model == 'pi0' else 'Pi05') + (
            'TorchFrontendRtx' if config.framework == 'torch' else 'JaxFrontendRtx')
        kwargs['use_fp8'] = False
        if config.model == 'pi0':
            kwargs['use_fp8_decoder'] = False
        else:
            kwargs['hardware'] = hardware
            # These internal selector labels only affect quantized branches;
            # use_fp8=False keeps actual execution BF16. Pin them so no ambient
            # NVFP4/skinny selection silently changes the requested path.
            kwargs.update(prefix_precision='fp8', decoder_precision='fp8',
                          decoder_kernel='cublaslt')
    frontend = getattr(import_module(module), name)(config.checkpoint, **kwargs)
    return VLABackend(config, frontend, hardware=hardware, precision=precision,
                      action_domain=decoder.action_domain if decoder is not None else spec['action_domain'],
                      action_decoder=decoder)
