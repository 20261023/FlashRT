"""Pi0.5 precision/environment interactions and concrete frontend bindings."""

import os

from . import frontend_kwargs


def precision(config):
    return ('mixed_int8_bf16' if config.options.get('int8_mode', 'none') in (
        'encoder', 'encoder_decoder') else 'bf16')


def validate(config):
    from flash_rt.models.pi05.int8 import validate_int8_mode
    mode = config.options.get('int8_mode', 'none')
    if mode is None:
        raise ValueError('int8_mode must explicitly name none, encoder, or encoder_decoder')
    validate_int8_mode(mode, sampler=config.sampler)
    if mode != 'none' and (config.framework != 'torch' or config.hardware not in ('auto', 'cuda_sm80')):
        raise ValueError('Pi0.5 W8A8 requires the Torch SM80 frontend')
    for name, allowed in (
            ('FLASHRT_PI05_PREFIX_PRECISION', ('fp8',)),
            ('FLASHRT_PI05_DECODER_PRECISION', ('fp8',)),
            ('FLASHRT_PI05_DECODER_KERNEL', ('auto', 'cublaslt'))):
        if name in os.environ and os.environ[name] not in allowed:
            raise ValueError(f'The BF16 VLA backend conflicts with {name}={os.environ[name]}')
    env_mode = os.environ.get('FLASHRT_PI05_STATE_PROMPT_MODE')
    requested = config.options.get('state_prompt_mode', 'exact')
    if env_mode is not None and env_mode != requested:
        raise ValueError(f'The VLA backend option state_prompt_mode={requested} conflicts with '
                         f'FLASHRT_PI05_STATE_PROMPT_MODE={env_mode}')


def binding(config, hardware):
    kwargs = frontend_kwargs(config)
    kwargs.update(use_fp8=False, hardware=hardware, prefix_precision='fp8',
                  decoder_precision='fp8', decoder_kernel='cublaslt')
    if config.framework == 'torch':
        kwargs.setdefault('int8_mode', 'none')
    else:
        kwargs.pop('int8_mode', None)
    module = f'flash_rt.frontends.{config.framework}.pi05_rtx'
    name = 'Pi05' + ('TorchFrontendRtx' if config.framework == 'torch' else 'JaxFrontendRtx')
    return module, name, kwargs
