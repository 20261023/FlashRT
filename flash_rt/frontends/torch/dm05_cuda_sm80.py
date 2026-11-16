"""DM0.5 host IO and exact-prefix graph ownership for A800/SM80 only."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
from pathlib import Path
import sys
from threading import RLock
import time

import numpy as np
import torch

from flash_rt.models.dm05.pipeline_cuda_sm80 import Dm05PipelineSm80, Dm05ProfileSm80
from flash_rt.models.dm05.weights import SOURCE_REVISION


_OPENDM_PYTHON_SHA256 = '28617ae3b6db5a887d6291e32bfac0406666f79188ec759dad5e22e1404dd462'


def _cpu_policy(checkpoint, source_dir):
    source = Path(source_dir).expanduser().resolve()
    package = source / 'opendm'
    if not (package / 'exp/dm05_exp.py').is_file():
        raise FileNotFoundError(f'DM05 source_dir is not an OpenDM checkout: {source}')
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(path.relative_to(package).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    if digest.hexdigest() != _OPENDM_PYTHON_SHA256:
        raise ValueError(f'DM05 requires the pinned OpenDM Python source {SOURCE_REVISION}; source hash differs')
    sys.path.insert(0, str(source))
    module = importlib.import_module('opendm.exp.dm05_exp')
    if Path(module.__file__).resolve() != package / 'exp/dm05_exp.py':
        raise RuntimeError('An OpenDM from another source_dir was already imported in this process')
    exp = module.DM05Exp(task='inference', logger_level='WARNING')
    exp.model_config.model_name_or_path = str(Path(checkpoint).resolve())
    exp.model_config.chunk_size = 50
    exp.data_config.is_history = False
    exp.inference_config.backend = 'default'
    exp.inference_config.diffusion_steps = 10
    exp.inference_config.enable_vlm_chat = False
    # This public author entry loads CPU transforms only, NOT GPU model weights.
    exp.initialize_inference_preprocessing(device='cpu')
    return exp.inference_config, {'source_revision': SOURCE_REVISION,
                                  'opendm_python_sha256': digest.hexdigest()}


class Dm05TorchFrontendCudaSm80:
    """Fixed base/Euler10/BF16 only; direct and VLABackend calls serialize.

    observation: images=[three PIL RGB or uint8 arrays], state=float[14],
    prompt=text, robot_type='DOS W1'. No history, batch or prepared GPU inputs.
    noise (optional): finite CPU or CUDA float [1,50,32] (or [50,32]), rounded
    to BF16 before masking; absent noise uses a fresh CUDA BF16 draw.
    actions: independent, complete absolute FP32 CPU ndarray[50,14].
    """

    def __init__(self, checkpoint, *, source_dir, backend='flashrt', num_steps=10,
                 sampler='euler', probeflow_config=None, use_cuda_graph=True,
                 max_graph_profiles=2, num_views=3):
        if backend not in ('torch', 'flashrt'):
            raise ValueError('DM05 backend must be torch or flashrt')
        if sampler != 'euler' or probeflow_config is not None or num_steps != 10:
            raise ValueError('DM05 first profile requires fixed Euler10, without ProbeFlow')
        if num_views != 3:
            raise ValueError('DM05 fixed base profile requires exactly three cameras')
        if not isinstance(use_cuda_graph, bool):
            raise TypeError('use_cuda_graph must be bool')
        if type(max_graph_profiles) is not int or max_graph_profiles < 1:
            raise ValueError('max_graph_profiles must be a positive integer')
        self._lock = RLock()
        self._checkpoint_path = str(checkpoint)
        self.backend = backend
        self._num_steps = 10
        self.use_cuda_graph = use_cuda_graph
        self.max_graph_profiles = max_graph_profiles
        self._closed = False
        self._prompt = None
        self.profiles = OrderedDict()
        self.requests = 0
        self.capture_s = 0.0
        with torch.inference_mode():
            self.policy, self.source_identity = _cpu_policy(checkpoint, source_dir)
            self.pipeline = Dm05PipelineSm80(checkpoint, native=backend == 'flashrt')

    def set_prompt(self, prompt, state=None):
        # State is intentionally NOT tokenized/cached here. infer() always runs
        # the pinned CPU transform on the CURRENT observation, including a
        # same-prompt state change that VLABackend does not special-case.
        if not isinstance(prompt, str):
            raise TypeError('DM05 prompt must be text')
        with self._lock:
            self._prompt = prompt

    def _host_input(self, observation):
        if not isinstance(observation, dict):
            raise TypeError('DM05 observation must be a dict')
        if observation.get('history_images') or observation.get('history_pixel_values') is not None:
            raise ValueError('DM05 first profile does not support history')
        prompt = observation.get('prompt', self._prompt)
        if not isinstance(prompt, str):
            raise ValueError('DM05 prompt is required')
        images = observation.get('images')
        if images is None:
            images = [observation.get(name) for name in ('image', 'wrist_image', 'wrist_image_right')]
        if not isinstance(images, (list, tuple)) or len(images) != 3 or any(i is None for i in images):
            raise ValueError('DM05 requires exactly three current images')
        state = np.asarray(observation.get('state'), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('DM05 requires finite state[14]')
        if observation.get('robot_type', 'DOS W1') != 'DOS W1':
            raise ValueError('DM05 first profile supports robot_type="DOS W1" only')
        data = self.policy._prepare_model_input(
            text=prompt, images=list(images), states=state.tolist(), robot_type='DOS W1')
        original_state = np.array(data['state'], copy=True)
        prepared = self.policy.input_transform(data)
        return data, original_state, prepared

    def _update_profile(self, prepared, noise):
        ids = prepared['input_ids']
        types = prepared['token_type_ids']
        pixels = prepared['pixel_values']
        if ids.device.type != 'cpu' or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('DM05 host tokenizer must return CPU batch-one IDs')
        length = ids.shape[1]
        if not 1 <= length <= 1024 or tuple(pixels.shape) != (3, 3, 448, 448):
            raise ValueError('DM05 requires P<=1024 and pixels[3,3,448,448]')
        valid = prepared['attention_mask'].numpy().astype(bool)
        tokens = ids.numpy()
        if not valid.all() or np.isin(tokens, (0, 7)).any():
            raise ValueError('DM05 first profile requires unpadded current-image prefixes (no history slots)')
        rows = np.flatnonzero(tokens.reshape(-1) == 262144)
        if rows.size != 768:
            raise ValueError(f'DM05 expected 768 image tokens, got {rows.size}')
        if length not in self.profiles:
            if len(self.profiles) >= self.max_graph_profiles:
                raise ValueError('DM05 exact-prefix profile capacity exhausted; load a larger explicit capacity')
            self.profiles[length] = Dm05ProfileSm80(self.pipeline, length)
        profile = self.profiles[length]
        image = types.numpy()[0] == 1
        starts = image & ~np.r_[False, image[:-1]]
        groups = np.cumsum(starts)
        bidirectional = image[:, None] & image[None, :] & (groups[:, None] == groups[None, :])
        mask = np.tril(np.ones((length, length), dtype=bool)) | bidirectional
        profile.ids.copy_(ids)
        profile.image_rows.copy_(torch.from_numpy(rows))
        profile.attention.prefix_mask.copy_(torch.from_numpy(mask)[None, None])
        profile.pixels.copy_(pixels)
        if noise is None:
            profile.noise.normal_()
        else:
            supplied = torch.as_tensor(noise)
            if tuple(supplied.shape) == (50, 32):
                supplied = supplied[None]
            if tuple(supplied.shape) != (1, 50, 32) or not supplied.is_floating_point():
                raise ValueError('DM05 noise must be floating [1,50,32] or [50,32]')
            # No per-step device readback. CPU noise is checked at the host
            # boundary; CUDA callers own finiteness until the final CPU output.
            if supplied.device.type == 'cpu' and not torch.isfinite(supplied).all():
                raise ValueError('DM05 noise must be finite')
            profile.noise.copy_(supplied)
        return profile

    def execution_metadata(self, profile):
        native = self.backend == 'flashrt'
        return {
            'engine': 'flashrt-dm05-owned-pipeline' if native else 'project-dm05-owned-torch-pipeline',
            'coverage': 'partial-native-gemm-fa2-gelu-torch-norm-rope-prefix-attention'
                        if native else 'project-owned-torch-operators',
            'native_operators': ['CachedGemmRunner.bf16_nt_bias', 'flash_rt_fa2.fwd_bf16',
                                 'bias_gelu_bf16_strict'] if native else [],
            'torch_regions': ['CPU preprocessing/postprocessing', 'patch convolution', 'LayerNorm',
                              'Gemma plus-one RMSNorm', 'adaptive RMSNorm', 'half-split RoPE',
                              'prefix masked SDPA', 'pooling', 'residual/gates', 'Euler'],
            'native_gemm_per_request': 2801 if native else 0,
            'native_fa2_per_request': 367 if native else 0,
            'native_gelu_per_request': 401 if native else 0,
            'setup_time_modulation_gemm': 710 if native else 0,
            'native_enqueue_stats': self.pipeline.runner.stats() if native else None,
            'prefix_length': profile.length, 'vision_patch_tokens_per_image': 1024,
            'pooled_image_tokens': 768, 'use_cuda_graph': self.use_cuda_graph,
            'graph_scope': 'vision+prefix+Euler10' if self.use_cuda_graph else 'none',
            'graph_profiles': len(self.profiles), 'profile_replays': profile.replays,
            'static_pointers': {name: getattr(profile, name).data_ptr()
                                for name in ('ids', 'pixels', 'noise', 'output', 'image_rows')},
            'requests': self.requests, 'capture_total_s': self.capture_s,
            'checkpoint': self.pipeline.identity, 'cpu_source': self.source_identity,
            'reference_math': 'author-hot-suffix-BF16-not-first-eager-FP32',
            'noise_contract': 'provided-noise-or-fresh-CUDA-BF16; no reseeding in frontend',
            'release_acceptance': 'not-run-user-waived-tests-and-extra-reviews',
        }

    @torch.inference_mode()
    def infer(self, observation, *, noise=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('DM05 frontend is closed; load a new frontend')
            data, original_state, prepared = self._host_input(observation)
            # The formal driver may persist the inputs of an actual measured
            # request afterwards; this does not execute a second preparation.
            self.last_prepared = prepared
            profile = self._update_profile(prepared, noise)
            if self.use_cuda_graph and profile.graph is None:
                start = time.perf_counter()
                profile.capture(self.pipeline)
                self.capture_s += time.perf_counter() - start
            profile.run(self.pipeline)
            raw = profile.output.float().cpu().numpy()[0, :, :14].copy()
            output = self.policy.output_transform({
                'action': raw.copy(), 'state': original_state, 'meta_data': data['meta_data']})
            actions = np.asarray(output['action'], dtype=np.float32).copy()
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise RuntimeError('DM05 returned nonfinite or incomplete public actions')
            self.requests += 1
            return {'actions': actions, 'raw_actions': raw,
                    'sampler_metadata': {'sampler': 'euler', 'nfe': 10, 'scheduled_steps': 10},
                    'execution_metadata': self.execution_metadata(profile)}

    def close(self):
        with self._lock:
            if not self._closed:
                torch.cuda.synchronize()
                for profile in self.profiles.values():
                    if profile.graph is not None:
                        profile.graph.reset()
                self.profiles.clear()
                self.pipeline = None
                self.policy = None
                self._closed = True
        return 0
