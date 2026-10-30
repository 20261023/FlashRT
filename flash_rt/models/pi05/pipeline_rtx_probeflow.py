"""Opt-in BF16 ProbeFlow execution over the native Pi0.5 RTX kernels.

Prefix and velocity graphs are static; their per-call selection is not. This
avoids capturing a warmup routing decision as if it were a fixed Euler loop.
"""
import ctypes

from flash_rt.core.cuda_graph import CUDAGraph
from flash_rt.models.pi05.pipeline_rtx import ACTION_DIM, Pi05Pipeline
from flash_rt.models.pi05.probeflow import ProbeFlowConfig


class Pi05ProbeFlowPipeline(Pi05Pipeline):
    def __init__(self, *args, probeflow_config: ProbeFlowConfig, **kwargs):
        if not isinstance(probeflow_config, ProbeFlowConfig):
            raise ValueError('probeflow_config must be a ProbeFlowConfig')
        unsupported = ('use_fp8', 'use_fp8_decoder', 'use_int8_decoder',
                       'use_int8_encoder', 'use_int8_vision', 'use_int8_vision_static',
                       'sde', 'denoise_trace')
        if any(kwargs.get(key, key in ('use_fp8', 'use_fp8_decoder')) for key in unsupported):
            raise ValueError('ProbeFlow requires the single-sample BF16 path')
        self.sampler_config = probeflow_config
        self._sampler_times = probeflow_config.timesteps()
        if kwargs.get('num_steps') != len(self._sampler_times):
            raise ValueError('ProbeFlow num_steps must match its exact time-table size')
        self._time_indices = {t: i for i, t in enumerate(self._sampler_times)}
        self._velocity_graphs = {}
        self.last_sampler_metadata = {}
        super().__init__(*args, **kwargs)
        from flash_rt.frontends.torch.pi05_probeflow import native_bf16_view
        shape = (self.chunk_size, ACTION_DIM)
        self._state_view, state_owner = native_bf16_view(self.input_noise_buf, shape)
        self._velocity_view, velocity_owner = native_bf16_view(self.bufs['decoder_action_buf'], shape)
        self._view_owners = (state_owner, velocity_owner)

    def _check_stream(self, stream):
        import torch
        current = torch.cuda.current_stream().cuda_stream
        if int(stream) != current:
            raise RuntimeError('ProbeFlow requires the requested Torch CUDA stream to be current')

    def _sync(self, stream):
        rc = self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
        if rc:
            raise RuntimeError(f'ProbeFlow CUDA stream synchronization failed: {rc}')

    def _run_prefix(self, stream):
        self._copy_lang_embeds_to_encoder_x(stream=stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        if self.prefix_export:
            self._export_prefix_hidden(stream)

    def _evaluate(self, t, stream, use_graphs):
        index = self._time_indices[t]
        if use_graphs:
            self._velocity_graphs[index].replay(ctypes.c_void_p(stream))
        else:
            Pi05Pipeline.transformer_decoder(
                self, stream=stream, _step_indices=(index,), _integrate=False)
        return self._velocity_view

    def _sample(self, stream, use_graphs=False):
        from flash_rt.frontends.torch.pi05_probeflow import sample_probeflow_torch_
        self.last_sampler_metadata = {}
        self.last_sampler_metadata = sample_probeflow_torch_(
            self._state_view, lambda t: self._evaluate(t, stream, use_graphs),
            self.sampler_config)

    def transformer_decoder(self, stream=0, rtc_prefix_len=0):
        if rtc_prefix_len:
            raise NotImplementedError('ProbeFlow does not support RTC prefix guidance yet')
        self._check_stream(stream)
        self._sample(stream)

    def run_pipeline(self, stream=0):
        self._check_stream(stream)
        self._run_prefix(stream)
        self._sample(stream)

    def record_infer_graph(self, external_stream_int=None):
        if external_stream_int is None:
            raise ValueError('ProbeFlow graph capture requires the frontend Torch CUDA stream')
        stream = int(external_stream_int)
        self._check_stream(stream)
        self.autotune_gemms()
        self._release_sampler_graphs()
        handle = ctypes.c_void_p(stream)
        self._graph_stream = handle
        for _ in range(2):
            self._run_prefix(stream)
        self._sync(stream)
        graph = CUDAGraph()
        graph.begin_capture(handle)
        self._run_prefix(stream)
        graph.end_capture(handle)
        self._graph = graph
        for index in range(len(self._sampler_times)):
            Pi05Pipeline.transformer_decoder(
                self, stream=stream, _step_indices=(index,), _integrate=False)
            self._sync(stream)
            graph = CUDAGraph()
            graph.begin_capture(handle)
            Pi05Pipeline.transformer_decoder(
                self, stream=stream, _step_indices=(index,), _integrate=False)
            graph.end_capture(handle)
            self._velocity_graphs[index] = graph
        self._sync(stream)

    def forward(self, stream=None):
        import torch
        stream = torch.cuda.current_stream().cuda_stream if stream is None else int(stream)
        self._check_stream(stream)
        use_graphs = self._graph is not None
        if use_graphs:
            self._graph.replay(ctypes.c_void_p(stream))
        else:
            self._run_prefix(stream)
        self._sample(stream, use_graphs=use_graphs)
        self._sync(stream)
        return self.input_noise_buf.ptr.value

    def forward_decode_only(self, stream=None):
        raise NotImplementedError('ProbeFlow does not support cross-frame K/V reuse yet')

    def export_runtime(self, *args, **kwargs):
        raise NotImplementedError('ProbeFlow requires host routing; native runtime export is not supported')

    def export_model_runtime(self, *args, **kwargs):
        raise NotImplementedError('ProbeFlow requires host routing; native model runtime export is not supported')

    def _release_sampler_graphs(self):
        graphs = [getattr(self, '_graph', None), *getattr(self, '_velocity_graphs', {}).values()]
        if not any(graphs):
            return
        handle = self._graph_stream
        self._sync(handle.value)
        for graph in graphs:
            if graph is None:
                continue
            for function, pointer in [('cudaGraphExecDestroy', graph._graph_exec),
                                      ('cudaGraphDestroy', graph._graph)]:
                if pointer.value:
                    rc = getattr(self._cudart, function)(pointer)
                    if rc:
                        raise RuntimeError(f'ProbeFlow {function} failed: {rc}')
                    pointer.value = None
            graph._captured = False
        self._graph = None
        self._velocity_graphs = {}

    def __del__(self):
        try:
            self._release_sampler_graphs()
        except Exception:
            pass  # CUDA may already have shut down during interpreter teardown.
