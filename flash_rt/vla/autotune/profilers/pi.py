"""Pi-family diagnostics; Euler splitting never replaces a production graph."""

import inspect


def _split_euler(session, pipe):
    if not session.fixed_noise():
        session.notes.append('Diagnostic graph split requires supplied noise; retaining unsplit evidence')
        return False
    methods = ('vision_encoder', 'transformer_encoder', 'transformer_decoder', 'run_pipeline', 'forward')
    if (getattr(session.backend.config, 'framework', 'torch') != 'torch'
            or not all(callable(getattr(pipe, name, None)) for name in methods)
            or '_step_indices' not in inspect.signature(pipe.transformer_decoder).parameters):
        session.notes.append('Euler split hook unavailable for the active program; retaining whole-graph evidence')
        return False
    session.take_reference()
    scope = 'diagnostic_same_program_recapture'
    for name, key, stages, boundary in (
            ('vision_encoder', 'vision', ('vision.encode',), 'Pi vision encoder only'),
            ('transformer_encoder', 'prefix_only', ('vision.project', 'prefix.prefill'),
             'Pi multimodal projector and prefix transformer; excludes vision encoder')):
        pair = session.pair(key, stages, boundary, external=True, scope=scope)
        session.gpu_method(pipe, name, key, stages, boundary, pair=pair, scope=scope)
    decoder = pipe.transformer_decoder
    velocity_pairs = [
        session.pair('velocity', ('denoise.velocity',),
                     'One unchanged Euler decoder step including its fused integration',
                     external=True, scope=scope) for _ in range(pipe.num_steps)]

    def steps(*args, **kwargs):
        if '_step_indices' in kwargs:
            raise RuntimeError('Diagnostic Euler wrapper expected the complete production schedule')
        for step, pair in enumerate(velocity_pairs):
            session.timed(
                decoder, 'velocity', ('denoise.velocity',),
                'One unchanged Euler decoder step including its fused integration',
                pair=pair, scope=scope)(*args, _step_indices=(step,), **kwargs)

    session.replace(pipe, 'transformer_decoder', steps)
    stream = getattr(session.frontend, '_graph_torch_stream', None)
    stream_int = stream.cuda_stream if stream is not None else 0
    graph = session.capture(lambda: pipe.run_pipeline(stream=stream_int), stream=stream)

    def forward(*args, **kwargs):
        # frontend has just refreshed all public request buffers. The private
        # graph references those same stable input/output addresses.
        if stream is None:
            graph.replay()
        else:
            import torch
            with torch.cuda.stream(stream):
                graph.replay()
        return pipe.input_noise_buf.ptr.value

    session.replace(pipe, 'forward', forward)
    session.expected = ['vision', 'prefix_only', 'velocity']
    session.notes.append(
        'Euler split uses a private diagnostic graph and the original single-step API; '
        'velocity includes the production fused integration; public output parity is required')
    return True


def install(session):
    pipe = session.frontend.pipeline
    sampler = session.backend.config.sampler
    graph = getattr(pipe, '_graph', None)
    split = sampler == 'euler' and graph is not None and _split_euler(session, pipe)
    if not split:
        if graph is not None:
            key = 'prefix' if sampler == 'probeflow' else 'full'
            stages = ('vision.encode', 'vision.project', 'prefix.prefill')
            if key == 'full':
                stages += ('denoise.velocity',)
            session.replace(pipe, '_graph', session.graph(
                graph, key, stages, f'Production Pi {key} graph replay'))
        for key, velocity in getattr(pipe, '_velocity_graphs', {}).copy().items():
            session.replace_item(pipe._velocity_graphs, key, session.graph(
                velocity, 'velocity', ('denoise.velocity',), 'One production ProbeFlow velocity graph replay'))
        session.expected = ['prefix', 'velocity'] if sampler == 'probeflow' else ['full']
        if graph is None:
            # These are source calls of the actual eager program. Euler's
            # decoder contains all evaluations, so preserve that aggregate.
            session.gpu_method(pipe, 'vision_encoder', 'vision',
                               ('vision.encode',), 'Eager Pi vision encoder')
            session.gpu_method(pipe, 'transformer_encoder', 'prefix_only',
                               ('vision.project', 'prefix.prefill'), 'Eager Pi projector and prefix')
            if sampler == 'probeflow':
                session.gpu_method(pipe, '_evaluate', 'velocity',
                                   ('denoise.velocity',), 'One eager ProbeFlow field evaluation')
            else:
                session.gpu_method(pipe, 'transformer_decoder', 'denoise_all',
                                   ('denoise.velocity',), 'All eager Euler steps including fused integration')
            session.expected = ['vision', 'prefix_only',
                                'velocity' if sampler == 'probeflow' else 'denoise_all']
    session.start_host_input = session.boundary(pipe, 'forward', output=True)
    if session.start_host_input:
        session.expected += ['host_input', 'host_output']


def describe_dataflow(backend):
    pipe = backend.frontend.pipeline
    geometry = {
        key: getattr(pipe, key) for key in
        ('encoder_seq_len', 'chunk_size', 'vision_seq', 'vision_seq_enc', 'num_steps')
        if isinstance(getattr(pipe, key, None), int)}
    return {
        'basis': 'runtime_geometry_and_source_seams',
        'measurement_kind': 'static_dataflow_ledger', 'runtime_geometry': geometry,
        'notes': ['Pi prefix KV is reused within one request',
                  'No measured HBM traffic or kernel bottleneck classification'],
    }


def describe_pi05_dataflow(backend):
    """Source-verified Pi0.5 cost model, selected explicitly by its declaration."""
    pipe = backend.frontend.pipeline
    prefix = int(pipe.encoder_seq_len)
    actions = int(pipe.chunk_size)
    layers = 18 if getattr(pipe, 'prefix_export', False) else 17
    decoder_int8 = getattr(pipe, 'use_int8_decoder', False)
    mixed_int8 = decoder_int8 or getattr(pipe, 'use_int8_encoder', False)
    return {
        'basis': ('actual_runtime_rows_and_source_verified_pi05_w8a8_bf16_geometry'
                  if mixed_int8 else 'actual_runtime_rows_and_source_verified_pi05_bf16_geometry'),
        'measurement_kind': 'static_dataflow_ledger',
        'prefix_rows': prefix, 'action_rows': actions,
        'vision_tokens_before_pool': int(pipe.vision_seq),
        'vision_tokens_after_pool': int(pipe.vision_seq_enc),
        'prefix_ffn': {'layers': layers, 'M': prefix, 'K': 2048, 'N': 16384,
                       'linear_flops': layers * 6 * prefix * 2048 * 16384},
        'action_per_evaluation': {
            'layers': 18, 'M': actions, 'hidden': 1024,
            'q_width': 2048, 'kv_width': 256, 'ffn_width': 4096,
            'projection_and_ffn_flops': 18 * (
                2 * actions * 1024 * 2560 + 2 * actions * 2048 * 1024
                + 6 * actions * 1024 * 4096),
            'attention_product_flops': 18 * 4 * actions * (prefix + actions) * 2048,
            'logical_linear_weight_bytes': 18 * (1 if decoder_int8 else 2) * (
                1024 * 2560 + 2048 * 1024 + 3 * 1024 * 4096)},
        'notes': ['FMA=2 FLOPs; norm/activation/softmax omitted',
                  'Logical expert matrix bytes exclude quantization scales, retained copies and HBM traffic',
                  'Prefix KV reused across evaluations, not across observations',
                  'Post-ViT pooling reduces prefix rows, not the completed ViT work'],
    }
