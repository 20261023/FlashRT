"""Original LeRobot SmolVLA entrypoint; independent of the project frontend.

All neural calls below belong to the pinned author package. The shared author
producer owns provenance, checkpoint/workload binding and measurement.
"""

import inspect
from pathlib import Path
import sys

import numpy as np


REVISION = 'b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf'
CAMERAS = ('observation.images.image', 'observation.images.image2')


def validate_options(options):
    value = options.get('vlm_dir')
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('SmolVLA author requires an explicit absolute vlm_dir')


def _load_config(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError('SmolVLA author requires a SmolVLA checkpoint config')
    return config


def load(spec, torch):
    if spec['source_revision'] != REVISION:
        raise ValueError(f'SmolVLA author adapter requires LeRobot revision {REVISION}')
    root = Path(spec['source']).resolve(strict=True)
    sys.path.insert(0, str(root / 'src'))
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if not Path(inspect.getfile(SmolVLAPolicy)).resolve().is_relative_to(root / 'src/lerobot'):
        raise ValueError('Another LeRobot policy package is already imported')
    config = _load_config(spec['checkpoint'])
    config.vlm_model_name = str(Path(spec['options']['vlm_dir']).resolve(strict=True))
    config.device = 'cuda'
    config.num_steps = spec['num_steps']
    config.compile_model = False
    if (config.chunk_size != 50 or config.max_action_dim != 32
            or config.action_feature.shape != (7,) and list(config.action_feature.shape) != [7]):
        raise ValueError('SmolVLA author checkpoint differs from the pinned LIBERO profile')
    policy = SmolVLAPolicy.from_pretrained(
        spec['checkpoint'], config=config, local_files_only=True, strict=True).eval()
    preprocess, postprocess = make_pre_post_processors(
        config, pretrained_path=spec['checkpoint'],
        preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': config.vlm_model_name},
            'device_processor': {'device': 'cuda'}})
    original = policy.model.denoise_step
    calls = [0]
    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)
    policy.model.denoise_step = counted
    observations = []

    def infer(request):
        images = request.observation['images']
        state = np.asarray(request.observation['state'])
        if len(images) != 2 or state.shape != (8,) or state.dtype != np.float32:
            raise ValueError('SmolVLA author requires exactly two cameras and actual float32 state[8]')
        batch = {'observation.state': torch.from_numpy(state.copy()), 'task': request.prompt}
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError('SmolVLA author requires real task text')
        for name, image in zip(CAMERAS, images):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.shape != (256, 256, 3):
                raise ValueError('SmolVLA author requires raw RGB uint8 HWC [256,256,3]')
            batch[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        noise = np.asarray(request.noise)
        if (noise.dtype != np.float32 or noise.shape != (1, 50, 32)
                or not np.isfinite(noise).all()):
            raise ValueError('SmolVLA author requires provided finite float32 host noise [1,50,32]')
        prepared = preprocess(batch)
        fixed_noise = torch.from_numpy(noise.copy()).to('cuda')
        calls[0] = 0
        action = policy.predict_action_chunk(prepared, noise=fixed_noise)
        if calls[0] != spec['num_steps']:
            raise RuntimeError(f'SmolVLA author observed NFE={calls[0]}, expected {spec["num_steps"]}')
        observations.append({'request_id': request.request_id, 'nfe': calls[0]})
        return postprocess(action).float().cpu().numpy()[0].copy()

    from flash_rt.vla.autotune.author import content_identity
    return policy, infer, {
        'implementation': 'original LeRobot SmolVLAPolicy.predict_action_chunk + checkpoint processors',
        'expected_shape': [50, 7], 'precision': 'mixed_bf16_fp32', 'sampler': 'euler',
        'noise_contract': 'provided host float32 noise passed explicitly to author predict_action_chunk',
        'parameter_loading': {'loader': 'SmolVLAPolicy.from_pretrained', 'strict': True},
        'nfe_observations': observations,
        'nfe_evidence': 'actual denoise_step calls observed on every original-author request',
        'parameter_dtypes': sorted({str(parameter.dtype) for parameter in policy.parameters()}),
        'auxiliary_assets': {'vlm': content_identity(config.vlm_model_name)},
        'additional_work': 'raw uint8 HWC to CHW float32/255; original newline, tokenization, normalization, resize; action unnormalization and CPU transfer',
    }
