"""Fixed DM0.5 profile policy."""

from . import frontend_kwargs


def validate(config):
    optimization = config.options.get('project_optimization', 'none')
    if optimization != 'none' and config.backend != 'torch':
        raise ValueError('DM05 project_optimization requires backend="torch"')
    if config.sampler != 'euler' or config.num_steps not in (None, 10):
        raise ValueError('DM05 first profile requires fixed Euler10')
    source = config.options.get('source_dir')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('DM05 requires source_dir pointing to the pinned read-only OpenDM CPU transforms')


def binding(config, hardware):
    kwargs = frontend_kwargs(config)
    kwargs['backend'] = config.backend
    return 'flash_rt.frontends.torch.dm05_cuda_sm80', 'Dm05TorchFrontendCudaSm80', kwargs
