"""SmolVLA SM80 onboarding frontend: explicit upstream compute and graph IO."""

import inspect
from pathlib import Path
import sys

import numpy as np
import torch

from flash_rt.models.smolvla.inputs import (
    CAMERAS, checkpoint_profile, validate_noise, validate_observation)
from flash_rt.models.smolvla.pipeline_cuda_sm80 import SmolVLAProfile, SmolVLAPipelineCudaSm80
from flash_rt.models.smolvla.source import POLICY_SHA256, admit_source, sha256


class SmolVLATorchFrontendCudaSm80:
    execution_coverage = 'upstream-lerobot-neural-operators-project-owned-graph-and-io'

    def __init__(self, checkpoint, *, source_dir, vlm_dir, num_steps=10, sampler='euler',
                 use_cuda_graph=True, max_graph_profiles=4):
        if sampler != 'euler' or type(num_steps) is not int or num_steps < 1:
            raise ValueError('SmolVLA onboarding requires positive fixed Euler steps')
        if type(use_cuda_graph) is not bool or type(max_graph_profiles) is not int or max_graph_profiles < 1:
            raise ValueError('Invalid SmolVLA graph configuration')
        if tuple(torch.cuda.get_device_capability()) != (8, 0):
            raise ValueError('This SmolVLA frontend is the explicit SM80 path')
        self.checkpoint = str(Path(checkpoint).resolve(strict=True))
        checkpoint_profile(self.checkpoint)
        if sha256(Path(self.checkpoint) / 'model.safetensors') != POLICY_SHA256:
            raise ValueError('SmolVLA onboarding requires the pinned LIBERO weight bytes')
        source = admit_source(source_dir)
        sys.path.insert(0, str(source / 'src'))
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        if not Path(inspect.getfile(SmolVLAPolicy)).resolve().is_relative_to(source / 'src/lerobot'):
            raise ValueError('SmolVLA imported policy differs from the declared pinned author source')
        config = SmolVLAConfig.from_pretrained(self.checkpoint, local_files_only=True)
        config.vlm_model_name = str(Path(vlm_dir).resolve(strict=True))
        config.device, config.num_steps, config.compile_model = 'cuda', num_steps, False
        policy = SmolVLAPolicy.from_pretrained(
            self.checkpoint, config=config, strict=True, local_files_only=True).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, pretrained_path=self.checkpoint,
            preprocessor_overrides={
                'tokenizer_processor': {'tokenizer_name': config.vlm_model_name},
                'device_processor': {'device': 'cuda'}})
        self.pipeline = SmolVLAPipelineCudaSm80(policy)
        self.num_steps = num_steps
        self.use_cuda_graph = use_cuda_graph
        self.max_graph_profiles = max_graph_profiles
        self.profiles = {}
        self.last_profile = None
        self._prompt = None
        self._closed = False

    def set_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('SmolVLA requires nonempty task text')
        self._prompt = prompt

    def _host_input(self, observation):
        images, state = validate_observation(observation)
        prompt = observation.get('prompt', self._prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('SmolVLA requires task text')
        batch = {'observation.state': torch.from_numpy(state.copy()), 'task': prompt}
        for name, image in zip(CAMERAS, images):
            batch[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        return self.preprocess(batch)

    @staticmethod
    def _profile_key(batch):
        return tuple((name, tuple(value.shape), str(value.dtype)) for name, value in sorted(batch.items())
                     if isinstance(value, torch.Tensor))

    def _update_profile(self, batch, noise):
        tensors = {name: value for name, value in batch.items() if isinstance(value, torch.Tensor)}
        key = self._profile_key(tensors)
        profile = self.profiles.get(key)
        if profile is None:
            if len(self.profiles) >= self.max_graph_profiles:
                raise RuntimeError('SmolVLA graph profile capacity exceeded; configure max_graph_profiles')
            profile = SmolVLAProfile(
                {name: value.clone() for name, value in tensors.items()}, noise.clone())
            if self.use_cuda_graph:
                self.pipeline.capture(profile)
            self.profiles[key] = profile
        else:
            for name, value in tensors.items():
                profile.batch[name].copy_(value)
            profile.noise.copy_(noise)
        self.last_profile = profile
        return profile

    @torch.inference_mode()
    def infer(self, observation, noise=None):
        if self._closed:
            raise RuntimeError('SmolVLA frontend is closed')
        batch = self._host_input(observation)
        supplied = observation.get('noise') if noise is None else noise
        # VLABackend.prepare has no noise argument. Its untimed setup preserves
        # the upstream optional-noise behavior; measured requests pass host noise.
        if supplied is None:
            fixed_noise = self.pipeline.policy.model.sample_noise((1, 50, 32), 'cuda')
        else:
            fixed_noise = torch.from_numpy(validate_noise(supplied).copy()).to('cuda')
        profile = self._update_profile(batch, fixed_noise)
        action, nfe = self.pipeline.run(profile)
        actions = self.postprocess(action).float().cpu().numpy()[0].copy()
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise RuntimeError('SmolVLA returned invalid public action output')
        return {
            'actions': actions,
            'sampler_metadata': {'sampler': 'euler', 'scheduled_steps': self.num_steps, 'nfe': nfe,
                                 'nfe_evidence': 'observed-capture-route' if self.use_cuda_graph
                                 else 'observed-eager-route'},
            'execution_metadata': {
                'engine': 'torch', 'coverage': self.execution_coverage,
                'use_cuda_graph': self.use_cuda_graph,
                'graph_replays': 1 if self.use_cuda_graph else 0,
                'graph_total_replays': profile.replays,
                'graph_capture_id': profile.capture_id,
                'nfe': nfe, 'native_operators': [],
                'noise_contract': 'explicit-float32-host-noise' if supplied is not None
                                  else 'upstream-cuda-rng-untimed-prepare',
            },
        }

    def close(self):
        if self._closed:
            return
        torch.cuda.synchronize()
        for profile in self.profiles.values():
            if profile.graph is not None:
                profile.graph.reset()
        self.profiles.clear()
        self.last_profile = None
        self.pipeline = None
        self.preprocess = self.postprocess = None
        self._closed = True
