"""Native graph ownership contracts; no model or CUDA is executed by fixtures.

Written but not run under the user's no-test-run instruction.
"""

import ctypes
from types import SimpleNamespace

from flash_rt.vla.resources import release_pi_frontend_graphs


def test_close_releases_unique_full_decoder_velocity_and_cached_graphs_once():
    calls = []
    def record(name):
        def call(pointer):
            calls.append((name, pointer.value))
            return 0
        return call
    runtime = SimpleNamespace(
        cudaStreamSynchronize=record('sync'),
        cudaGraphExecDestroy=record('exec'), cudaGraphDestroy=record('graph'))
    def graph(value):
        return SimpleNamespace(_graph_exec=ctypes.c_void_p(value),
                               _graph=ctypes.c_void_p(value + 100), _captured=True)
    full, decoder, velocity = graph(0x123456789), graph(2), graph(3)
    one = SimpleNamespace(_cudart=runtime, _graph_stream=ctypes.c_void_p(0x234567891),
                          _graph=full, _decoder_only_graph=decoder,
                          _velocity_graphs={0: velocity})
    two = SimpleNamespace(_cudart=runtime, _graph_stream=ctypes.c_void_p(0x234567891),
                          _graph=velocity)
    frontend = SimpleNamespace(pipeline=one, _fixed_pipeline=one,
                               _prompt_pipeline_cache={20: one, 30: two})
    release_pi_frontend_graphs(frontend)
    release_pi_frontend_graphs(frontend)
    assert [value for name, value in calls if name == 'exec'] == [0x123456789, 2, 3]
    assert len([name for name, value in calls if name == 'graph']) == 3
    assert calls[0] == ('sync', 0x234567891)
    assert full._graph_exec.value is None
    assert not full._captured


def test_no_graphs_does_not_require_a_cuda_runtime():
    release_pi_frontend_graphs(SimpleNamespace(pipeline=None))


def test_pi0_replaced_prompt_pipeline_is_released_before_final_close():
    import numpy as np
    from flash_rt.vla import BackendConfig, VLABackend
    destroyed = []
    def destroy(pointer):
        destroyed.append(pointer.value)
        return 0
    runtime = SimpleNamespace(cudaStreamSynchronize=lambda pointer: 0,
                              cudaGraphExecDestroy=destroy, cudaGraphDestroy=destroy)
    class Frontend:
        calibrated = True
        pipeline = None
        def set_prompt(self, prompt):
            value = 10 if prompt == 'first' else 20
            self.pipeline = SimpleNamespace(
                _cudart=runtime, _graph_stream=ctypes.c_void_p(1),
                _graph=SimpleNamespace(_graph_exec=ctypes.c_void_p(value),
                                       _graph=ctypes.c_void_p(value + 1), _captured=True))
        def infer(self, observation):
            return {'actions': np.zeros((10, 7))}
    model = VLABackend(BackendConfig(model='pi0', checkpoint='/weights'), Frontend(),
                       hardware='cuda_sm80', precision='fp16',
                       action_domain='frontend-decoded-libero-7d')
    model.infer({}, prompt='first')
    model.infer({}, prompt='second')
    assert destroyed == [10, 11]
    model.close()
    model.close()
    assert destroyed == [10, 11, 20, 21]


def _replace_failure_backend(*, setter_fails=False, cleanup_fails=False):
    import numpy as np
    from flash_rt.vla import BackendConfig, VLABackend
    destroyed, failed = [], []
    def destroy(pointer):
        if cleanup_fails and pointer.value == 10 and not failed:
            failed.append(pointer.value)
            return 7
        destroyed.append(pointer.value)
        return 0
    runtime = SimpleNamespace(cudaStreamSynchronize=lambda pointer: 0,
                              cudaGraphExecDestroy=destroy, cudaGraphDestroy=destroy)
    class Frontend:
        pipeline = None
        def set_prompt(self, prompt):
            value = 10 if prompt == 'first' else 20
            self.pipeline = SimpleNamespace(
                _cudart=runtime, _graph_stream=ctypes.c_void_p(1),
                _graph=SimpleNamespace(_graph_exec=ctypes.c_void_p(value),
                                       _graph=ctypes.c_void_p(value + 1), _captured=True))
            if setter_fails and prompt == 'second':
                raise ValueError('embedding upload failed')
        def infer(self, observation):
            return {'actions': np.zeros((10, 7))}
    model = VLABackend(BackendConfig(model='pi0', checkpoint='/weights'), Frontend(),
                       hardware='cuda_sm80', precision='fp16',
                       action_domain='frontend-decoded-libero-7d')
    return model, destroyed


def test_setter_failure_keeps_original_error_and_cleanup_failure_is_retried():
    import pytest
    model, destroyed = _replace_failure_backend(setter_fails=True, cleanup_fails=True)
    model.infer({}, prompt='first')
    with pytest.raises(ValueError, match='embedding upload failed'):
        model.infer({}, prompt='second')
    model.close()
    assert destroyed == [10, 11, 20, 21]


def test_cleanup_failure_after_successful_setter_is_retried_by_close():
    import pytest
    model, destroyed = _replace_failure_backend(cleanup_fails=True)
    model.infer({}, prompt='first')
    with pytest.raises(RuntimeError, match='CUDA error 7'):
        model.infer({}, prompt='second')
    model.close()
    assert destroyed == [10, 11, 20, 21]
