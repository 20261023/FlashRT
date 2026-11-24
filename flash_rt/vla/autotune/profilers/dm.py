"""DM0.5 exact-prefix diagnostics using a separate private capture."""


def install(session):
    frontend = session.frontend
    pipe = frontend.pipeline
    # prepare() resolved the current request's exact token length. Never pick a
    # stale last/LRU graph belonging to a different observation.
    profile = frontend.profiles[frontend.last_prepared['input_ids'].shape[1]]
    captured = profile.graph is not None
    if captured and not session.take_reference():
        session.replace(profile, 'graph', session.graph(
            profile.graph, 'full', ('vision.encode', 'vision.project', 'prefix.prefill', 'denoise.velocity'),
            'DM whole vision+prefix+Euler graph; supplied-noise parity unavailable'))
        session.expected = ['full']
    else:
        scope = 'diagnostic_same_program_recapture' if captured else 'diagnostic_request'
        vision_pair = session.pair(
            'vision', ('vision.encode', 'vision.project'), 'DM vision, pooling and projector',
            external=captured, scope=scope)
        prefix_pair = session.pair(
            'prefix', ('prefix.prefill',), 'DM prefix transformer after vision returns',
            external=captured, scope=scope)
        velocity_pairs = [
            session.pair('velocity', ('denoise.velocity',), 'One raw DM action field evaluation',
                         external=captured, scope=scope) for _ in range(10)]
        vision, prefill, velocity = pipe.vision, pipe.prefill, pipe.velocity

        def timed_vision(*args, **kwargs):
            vision_pair[0].record()
            value = vision(*args, **kwargs)
            vision_pair[1].record()
            # prefill calls vision first. Its interval starts *after* that
            # nested call, so the two components are sequential and disjoint.
            prefix_pair[0].record()
            return value

        def timed_prefill(*args, **kwargs):
            value = prefill(*args, **kwargs)
            prefix_pair[1].record()
            return value

        def timed_velocity(state, active_profile, step):
            return session.timed(
                velocity, 'velocity', ('denoise.velocity',), 'One raw DM action field evaluation',
                pair=velocity_pairs[step], scope=scope)(state, active_profile, step)

        for name, fn in (('vision', timed_vision), ('prefill', timed_prefill), ('velocity', timed_velocity)):
            session.replace(pipe, name, fn)
        if captured:
            graph = session.capture(lambda: pipe.forward(profile))
            session.replace(profile, 'graph', graph)
        session.expected = ['vision', 'prefix', 'velocity']
    session.start_host_input = session.boundary(profile, 'run', output=True)
    if session.start_host_input:
        session.expected += ['host_input', 'host_output']


def describe_dataflow(backend):
    frontend = backend.frontend
    profile = getattr(frontend, 'last_prepared', {}).get('input_ids')
    return {
        'basis': 'runtime_exact_prefix_and_source_verified_fixed_geometry',
        'measurement_kind': 'static_dataflow_ledger',
        'runtime_geometry': {'prefix_rows': None if profile is None else int(profile.shape[1]),
                             'action_rows': 50, 'padded_action_dim': 32,
                             'vision_views': 3, 'pooled_vision_tokens': 768,
                             'prefix_layers': 34, 'action_layers': 34},
        'notes': ['Fixed Euler10; raw velocity intervals exclude integration',
                  'Prefix KV is reused across all ten action field evaluations',
                  'No measured HBM traffic or kernel bound classification'],
    }
