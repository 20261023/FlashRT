"""FP16 ProbeFlow over Pi0's native RTX kernels.

Only prefix/individual velocity evaluations are captured. Input-dependent
routing and action integration stay on the host/Torch side, on the same CUDA
stream. Pi0's state token and its state-masked attention are left unchanged.
"""
from __future__ import annotations

import ctypes

from flash_rt.core.cuda_graph import CUDAGraph
from flash_rt.models.pi0.pipeline_rtx import ACTION_DIM, Pi0Pipeline
from flash_rt.sampling.probeflow import ProbeFlowConfig


class Pi0ProbeFlowPipeline(Pi0Pipeline):
    def __init__(self, *args, probeflow_config: ProbeFlowConfig, **kwargs):
        if not isinstance(probeflow_config, ProbeFlowConfig):
            raise ValueError('probeflow_config must be a ProbeFlowConfig')
        if kwargs.get('use_fp8', True) or kwargs.get('use_fp8_decoder', True):
            raise ValueError('Pi0 ProbeFlow requires FP16: disable use_fp8 and use_fp8_decoder')
        self.sampler_config = probeflow_config
        self._sampler_times = probeflow_config.timesteps()
        if kwargs.get('num_steps') != len(self._sampler_times):
            raise ValueError('ProbeFlow num_steps must match its exact time-table size')
        self._time_indices = {t: index for index, t in enumerate(self._sampler_times)}
        self._velocity_graphs = {}
        self.last_sampler_metadata = {}
        super().__init__(*args, **kwargs)

        import torch
        from flash_rt.sampling.torch_probeflow import native_float_view
        shape = (self.chunk_size, ACTION_DIM)
        self._state_view, state_owner = native_float_view(
            self.input_noise_buf, shape, torch.float16)
        self._velocity_view, velocity_owner = native_float_view(
            self.bufs['decoder_action_buf'], shape, torch.float16)
        self._view_owners = (state_owner, velocity_owner)

    def _check_stream(self, stream):
        import torch
        if int(stream) != torch.cuda.current_stream().cuda_stream:
            raise RuntimeError('ProbeFlow requires the requested Torch CUDA stream to be current')

    def _sync(self, stream):
        status = self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
        if status:
            raise RuntimeError(f'ProbeFlow CUDA stream synchronization failed: {status}')

    def _run_prefix(self, stream):
        self._copy_lang_embeds_to_encoder_x(stream=stream)
        self._state_project(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)

    def _evaluate(self, t, stream, use_graphs):
        index = self._time_indices[t]
        if use_graphs:
            self._velocity_graphs[index].replay(ctypes.c_void_p(stream))
        else:
            Pi0Pipeline.transformer_decoder(
                self, stream=stream, _step_indices=(index,), _integrate=False)
        # The sampler clones the first probe before this scratch is reused.
        return self._velocity_view

    def _sample(self, stream, use_graphs=False):
        from flash_rt.sampling.torch_probeflow import sample_probeflow_torch_
        self.last_sampler_metadata = {}
        self.last_sampler_metadata = sample_probeflow_torch_(
            self._state_view, lambda t: self._evaluate(t, stream, use_graphs),
            self.sampler_config)

    def transformer_decoder(self, stream=0):
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
        # Prefix and velocity warmups never integrate/mutate diffusion_noise.
        for _ in range(2):
            self._run_prefix(stream)
        self._sync(stream)
        graph = CUDAGraph()
        graph.begin_capture(handle)
        self._run_prefix(stream)
        graph.end_capture(handle)
        self._graph = graph
        for index in range(len(self._sampler_times)):
            Pi0Pipeline.transformer_decoder(
                self, stream=stream, _step_indices=(index,), _integrate=False)
            self._sync(stream)
            graph = CUDAGraph()
            graph.begin_capture(handle)
            Pi0Pipeline.transformer_decoder(
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

    def _release_sampler_graphs(self):
        graphs = [getattr(self, '_graph', None),
                  *getattr(self, '_velocity_graphs', {}).values()]
        if not any(graphs):
            return
        self._sync(self._graph_stream.value)
        for graph in graphs:
            if graph is None:
                continue
            for function, pointer in [('cudaGraphExecDestroy', graph._graph_exec),
                                      ('cudaGraphDestroy', graph._graph)]:
                if pointer.value:
                    status = getattr(self._cudart, function)(pointer)
                    if status:
                        raise RuntimeError(f'ProbeFlow {function} failed: {status}')
                    pointer.value = None
            graph._captured = False
        self._graph = None
        self._velocity_graphs = {}

    def __del__(self):
        try:
            self._release_sampler_graphs()
        except Exception:
            pass  # CUDA may already be unavailable at interpreter teardown.
