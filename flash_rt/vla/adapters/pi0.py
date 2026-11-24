"""Pi0 frontend policy, separate from model-independent registry mechanics."""

from . import frontend_kwargs


def binding(config, hardware):
    kwargs = frontend_kwargs(config)
    kwargs.update(use_fp8=False, use_fp8_decoder=False)
    module = f'flash_rt.frontends.{config.framework}.pi0_rtx'
    name = 'Pi0' + ('TorchFrontendRtx' if config.framework == 'torch' else 'JaxFrontendRtx')
    return module, name, kwargs
