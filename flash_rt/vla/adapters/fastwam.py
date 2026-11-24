"""FastWAM schedule, compilation and partial-native frontend policy."""

from . import frontend_kwargs


def validate(config):
    compiled = config.options.get('compile_action_infer', False)
    if compiled and (config.backend != 'torch' or config.options.get('use_cuda_graph', False)):
        raise ValueError('Compiled FastWAM requires backend="torch" and use_cuda_graph=False')
    if config.backend == 'torch' and 'native_linear_backend' in config.options:
        raise ValueError('native_linear_backend is incompatible with backend="torch"')
    if (config.backend == 'flashrt' and config.options.get('use_cuda_graph', False)
            and config.options.get('native_linear_backend', 'legacy') != 'cached'):
        raise ValueError('FastWAM native CUDA graph requires native_linear_backend="cached"')
    missing = {'sigma_shift', 'action_horizon', 'model_config', 'backbone_dir'} - set(config.options)
    if missing:
        raise ValueError(f'FastWAM requires explicit options: {sorted(missing)}')
    from flash_rt.models.fastwam.config import positive_integer, validate_sampler
    positive_integer(config.options['action_horizon'], 'action_horizon')
    validate_sampler(config.sampler, config.options['sigma_shift'], config.num_steps, config.probeflow)


def binding(config, hardware):
    kwargs = frontend_kwargs(config)
    kwargs.pop('num_steps', None)
    kwargs.update(native_linear=config.backend == 'flashrt', dtype='bf16', device='cuda',
                  num_inference_steps=config.num_steps)
    return 'flash_rt.frontends.torch.fastwam', 'FastWAMTorchFrontend', kwargs
