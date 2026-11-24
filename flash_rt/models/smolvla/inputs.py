"""CPU contracts for the pinned HuggingFaceVLA SmolVLA LIBERO profile."""

import json
from pathlib import Path

import numpy as np


CAMERAS = ('observation.images.image', 'observation.images.image2')


def validate_observation(observation):
    images = observation.get('images')
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValueError('SmolVLA requires exactly two RGB cameras in checkpoint order')
    for image in images:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (256, 256, 3)):
            raise ValueError('SmolVLA LIBERO images must be uint8 HWC RGB [256,256,3]')
    state = np.asarray(observation.get('state'))
    if state.shape != (8,) or state.dtype != np.float32:
        raise ValueError('SmolVLA LIBERO state must be float32[8]')
    if not np.isfinite(state).all():
        raise ValueError('SmolVLA state must be finite')
    return images, state


def validate_noise(noise):
    value = np.asarray(noise)
    if (value.dtype != np.float32 or value.shape not in ((50, 32), (1, 50, 32))
            or not np.isfinite(value).all()):
        raise ValueError('SmolVLA requires explicit finite float32 noise [1,50,32]')
    return np.ascontiguousarray(value.reshape(1, 50, 32))


def checkpoint_profile(checkpoint):
    config = json.loads((Path(checkpoint) / 'config.json').read_text())
    expected_inputs = {name: {'type': 'VISUAL', 'shape': [3, 256, 256]} for name in CAMERAS}
    expected_inputs['observation.state'] = {'type': 'STATE', 'shape': [8]}
    required = {
        'type': 'smolvla', 'chunk_size': 50, 'max_action_dim': 32, 'max_state_dim': 32,
        'input_features': expected_inputs,
        'output_features': {'action': {'type': 'ACTION', 'shape': [7]}},
        'resize_imgs_with_padding': [512, 512],
        'adapt_to_pi_aloha': False, 'use_delta_joint_actions_aloha': False,
        'attention_mode': 'cross_attn', 'self_attn_every_n_layers': 2,
        'empty_cameras': 0, 'use_cache': True,
    }
    if any(config.get(name) != value for name, value in required.items()):
        raise ValueError('Checkpoint differs from the pinned SmolVLA LIBERO onboarding profile')
    return {'action_dim': 7, 'model_dim': 32, 'horizon': 50, 'state_dim': 8, 'config': config}
