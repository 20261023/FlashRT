"""Observable adapter behavior; no CUDA dependencies. Not executed this round."""

import numpy as np
import pytest

from flash_rt.vla import BackendConfig, VLABackend


class RecordingPiFrontend:
    """Small boundary fixture: no weights/device operations or model substitute."""

    def __init__(self):
        self.calibrated = False
        self.prompts = []
        self.calibrations = 0
        self.last_noise = None

    def set_prompt(self, prompt, state=None):
        self.prompts.append((prompt, None if state is None else state.copy()))

    def calibrate_with_real_data(self, samples):
        self.calibrations += 1
        self.calibrated = True

    def infer(self, observation, *, noise=None):
        self.last_noise = noise
        return {'actions': np.zeros((50, 7)),
                'sampler_metadata': {'sampler': 'probeflow', 'nfe': 2}}


def backend(frontend):
    return VLABackend(BackendConfig(model='pi05', checkpoint='/weights',
                                    sampler='probeflow'), frontend,
                      hardware='cuda_sm80', precision='bf16',
                      action_domain='frontend-decoded-libero-7d')


def test_new_state_is_retokenized_and_noise_is_not_dropped():
    frontend = RecordingPiFrontend()
    model = backend(frontend)
    noise = np.ones((50, 32), dtype=np.float32)
    obs = {'images': [], 'state': np.array([0.25])}
    first = model.infer(obs, prompt='pick', noise=noise)
    obs['state'][0] = -0.5
    second = model.infer(obs, noise=noise)
    assert frontend.prompts[0][1].tolist() == [0.25]
    assert frontend.prompts[1][1].tolist() == [-0.5]
    assert frontend.calibrations == 1
    assert frontend.last_noise is noise
    assert first['sampler_metadata']['nfe'] == second['sampler_metadata']['nfe'] == 2
    assert second['backend_metadata']['action_domain'] == 'frontend-decoded-libero-7d'


def test_same_prompt_state_does_not_repeat_setup():
    frontend = RecordingPiFrontend()
    model = backend(frontend)
    obs = {'state': np.zeros(2)}
    model.prepare(obs, prompt='pick')
    model.infer(obs)
    assert len(frontend.prompts) == 1
    assert frontend.calibrations == 1


def test_obs_noise_is_forwarded_without_mutating_input_dict():
    frontend = RecordingPiFrontend()
    model = backend(frontend)
    noise = np.zeros((50, 32), dtype=np.float32)
    obs = {'prompt': 'pick', 'noise': noise}
    model.infer(obs)
    assert frontend.last_noise is noise
    assert obs['noise'] is noise


def test_duplicate_noise_is_rejected():
    model = backend(RecordingPiFrontend())
    with pytest.raises(ValueError, match='noise'):
        model.infer({'prompt': 'pick', 'noise': np.zeros(1)}, noise=np.zeros(1))


def test_closed_backend_cannot_reuse_released_buffers():
    model = backend(RecordingPiFrontend())
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match='closed'):
        model.infer({}, prompt='pick')


def test_explicit_prompt_overrides_stale_observation_prompt_without_cache_drift():
    class ObservationPromptFrontend(RecordingPiFrontend):
        def infer(self, observation, *, noise=None):
            if observation.get('prompt') != self.prompts[-1][0]:
                self.set_prompt(observation['prompt'])
            return {'actions': self.prompts[-1][0],
                    'sampler_metadata': {'sampler': 'probeflow', 'nfe': 2}}

    frontend = ObservationPromptFrontend()
    model = backend(frontend)
    old = {'prompt': 'old task'}
    result = model.infer(old, prompt='new task')
    following = model.infer({})
    assert result['actions'] == following['actions'] == 'new task'
    assert len(frontend.prompts) == 1
    assert old['prompt'] == 'old task'


def test_prepare_warms_frontend_without_a_separate_calibration_api():
    class FlowFrontend:
        def __init__(self):
            self.calls = 0
        def set_prompt(self, prompt):
            self.prompt = prompt
        def infer(self, observation):
            self.calls += 1
            assert observation['prompt'] == self.prompt
            return {'actions': np.zeros((40, 32)),
                    'sampler_metadata': {'sampler': 'probeflow', 'nfe': 2}}
    frontend = FlowFrontend()
    model = VLABackend(BackendConfig(model='hyvla', checkpoint='/weights',
                                     sampler='probeflow'), frontend,
                       hardware='cuda_sm80', precision='bf16',
                       action_domain='normalized-padded-model-actions')
    model.prepare({}, prompt='pick')
    assert frontend.calls == 1
    model.infer({})
    assert frontend.calls == 2


def test_fastwam_prepared_context_needs_no_text_setter_or_prompt():
    class PreparedContextFrontend:
        def infer(self, observation, noise=None):
            assert 'prompt' not in observation
            assert observation['context'] == 'prepared T5'
            return {'actions': np.zeros((1, 32, 7)),
                    'sampler_metadata': {'sampler': 'probeflow', 'nfe': 2},
                    'execution_metadata': {'fully_native': False}}
    model = VLABackend(BackendConfig(model='fastwam', checkpoint='/weights',
                                     sampler='probeflow'), PreparedContextFrontend(),
                       hardware='cuda_sm80', precision='bf16',
                       action_domain='normalized-model-actions-batched')
    result = model.infer({'context': 'prepared T5', 'context_mask': 'mask'})
    assert result['actions'].shape == (1, 32, 7)
    assert result['backend_metadata']['execution_coverage']['fully_native'] is False


def test_fastwam_does_not_silently_ignore_prompt_with_prepared_context():
    class ContextFrontend:
        def infer(self, observation):
            raise AssertionError('conflicting conditioning must fail before execution')
    model = VLABackend(BackendConfig(model='fastwam', checkpoint='/weights'),
                       ContextFrontend(), hardware='cuda_sm80', precision='bf16',
                       action_domain='normalized-model-actions-batched')
    with pytest.raises(ValueError, match='mutually exclusive'):
        model.infer({'context': 'prepared'}, prompt='a different task')
