"""Hy flow frontend-specific policy."""

from . import frontend_kwargs


def validate(config):
    fusion = config.options.get('fused_rope_qknorm_kvwrite', False)
    if fusion and config.hardware not in ('auto', 'cuda_sm80'):
        raise ValueError('Hy fused_rope_qknorm_kvwrite requires SM80')


def binding(config, hardware):
    kwargs = frontend_kwargs(config)
    kwargs['hardware'] = hardware
    return 'flash_rt.frontends.torch.hyvla_probeflow', 'HyVLAFlowFrontend', kwargs
