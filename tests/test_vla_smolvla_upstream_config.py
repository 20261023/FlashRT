"""Optional CPU regression against the actual pinned LeRobot and checkpoint.

Set SMOLVLA_AUTHOR_SOURCE and SMOLVLA_CHECKPOINT to run. No neural module is
constructed; ordinary CPU CI without these private paths reports a skip.
"""

import hashlib
import importlib
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('module', [
    'flash_rt.vla.author_adapters.smolvla',
    'flash_rt.frontends.torch.smolvla_cuda_sm80',
])
def test_actual_pinned_smolvla_checkpoint_dispatches_type_without_cuda(module, monkeypatch):
    source, checkpoint = os.environ.get('SMOLVLA_AUTHOR_SOURCE'), os.environ.get('SMOLVLA_CHECKPOINT')
    if not source or not checkpoint:
        pytest.skip('Actual pinned LeRobot/checkpoint paths were not supplied')
    torch = pytest.importorskip('torch')
    source, checkpoint = Path(source).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    monkeypatch.syspath_prepend(str(source / 'src'))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    config_path = checkpoint / 'config.json'
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()
    loaded = importlib.import_module(module)._load_config(checkpoint)
    assert isinstance(loaded, SmolVLAConfig)
    assert loaded.chunk_size == 50
    assert loaded.max_action_dim == 32
    assert list(loaded.action_feature.shape) == [7]
    assert list(loaded.input_features['observation.state'].shape) == [8]
    assert 'observation.images.image2' in loaded.input_features
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == before
    assert not torch.cuda.is_initialized()
    assert not any(name.startswith('flash_rt.flash_rt_kernels') for name in sys.modules)
