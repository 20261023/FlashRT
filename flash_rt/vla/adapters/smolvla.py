"""SmolVLA onboarding policy. Only the explicit upstream Torch path exists."""


def validate(config):
    if config.backend != 'torch' or config.framework != 'torch':
        raise ValueError('SmolVLA onboarding supports the Torch backend only')
    if config.sampler != 'euler':
        raise ValueError('SmolVLA onboarding implements the author Euler route only')
    if config.num_steps is not None and (type(config.num_steps) is not int or config.num_steps < 1):
        raise ValueError('SmolVLA num_steps must be a positive integer')
    for name in ('source_dir', 'vlm_dir'):
        value = config.options.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'SmolVLA requires explicit {name}')
    if 'use_cuda_graph' in config.options and type(config.options['use_cuda_graph']) is not bool:
        raise ValueError('use_cuda_graph must be bool')
    capacity = config.options.get('max_graph_profiles', 4)
    if type(capacity) is not int or capacity < 1:
        raise ValueError('max_graph_profiles must be a positive integer')


def binding(config, hardware):
    if hardware != 'cuda_sm80':
        raise ValueError('SmolVLA onboarding has only a cuda_sm80 path')
    kwargs = dict(config.options)
    kwargs.update(num_steps=10 if config.num_steps is None else config.num_steps,
                  sampler=config.sampler)
    return ('flash_rt.frontends.torch.smolvla_cuda_sm80',
            'SmolVLATorchFrontendCudaSm80', kwargs)


def install_profile(session):
    """Place disjoint events on the actual upstream methods; no inferred costs."""
    import torch
    frontend = session.frontend
    pipeline = frontend.pipeline
    if frontend.use_cuda_graph and not session.take_reference():
        return
    profile = frontend.last_profile
    if profile is None:
        raise RuntimeError('Prepare a real SmolVLA request before profiling')
    captured = profile.graph is not None
    descriptions = {
        'vision': (['vision.encode', 'vision.project'], 'Upstream embed_image: vision encoder and connector'),
        'prefix': (['prefix.prefill'], 'Upstream forward(fill_kv_cache=True)'),
        'velocity': (['denoise.velocity'], 'Upstream denoise_step including action projection'),
    }
    pairs = {}
    if captured:
        for key, count in (('vision', 2), ('prefix', 1), ('velocity', frontend.num_steps)):
            stages, boundary = descriptions[key]
            pairs[key] = [session.pair(key, stages, boundary, external=True) for _ in range(count)]

    def wrap(owner, name, key, *, predicate=None):
        original = getattr(owner, name)
        cursor = [0]
        def timed(*args, **kwargs):
            if predicate is not None and not predicate(kwargs):
                return original(*args, **kwargs)
            if captured:
                events = pairs[key][cursor[0]]
                cursor[0] += 1
            else:
                events = session.pair(key, *descriptions[key])
            events[0].record()
            value = original(*args, **kwargs)
            events[1].record()
            return value
        session.replace(owner, name, timed)

    model = pipeline.policy.model
    wrap(model.vlm_with_expert, 'embed_image', 'vision')
    wrap(model.vlm_with_expert, 'forward', 'prefix', predicate=lambda kw: kw.get('fill_kv_cache'))
    wrap(model, 'denoise_step', 'velocity')
    if captured:
        # A private diagnostic graph retains the production graph and its pool.
        output = []
        def body():
            output.append(pipeline.run_static(profile))
        graph = session.capture(body)
        session.replace(profile, 'graph', graph)
        session.replace(profile, 'output', output[0])
    session.start_host_input = True
    session.boundary(pipeline, 'run', output=True)
    session.expected.extend(('vision', 'prefix', 'velocity', 'host_input', 'host_output'))
