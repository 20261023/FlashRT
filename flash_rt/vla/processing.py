"""Honor a checkpoint's declared action processor, not the presence of q01 keys.

The resident LeRobot pi0.5 release declares ACTION=MEAN_STD while storing both
mean/std and quantiles. Legacy FlashRT frontends always quantile-decode. This
small CPU boundary corrects the public output from the already returned raw
actions; model math and normalized outputs are untouched.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActionDecoder:
    mode: str
    dimension: int
    first: np.ndarray | None
    second: np.ndarray | None
    eps: float
    source: str | None = None

    @property
    def action_domain(self):
        return f'checkpoint-ACTION-{self.mode}-{self.dimension}d'

    def __call__(self, raw):
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] < self.dimension:
            raise ValueError('Raw action width does not cover the declared processor dimension')
        values = values[..., :self.dimension]
        if self.mode == 'IDENTITY':
            return values.copy()
        if self.mode == 'MEAN_STD':
            return values * self.second + self.first
        denominator = self.second - self.first
        denominator = np.where(denominator == 0, np.float32(self.eps), denominator)
        if self.mode == 'MIN_MAX':
            return (values + np.float32(1.0)) / np.float32(2.0) * denominator + self.first
        return (values + np.float32(1.0)) * denominator / np.float32(2.0) + self.first


def _processor(record):
    steps = [step for step in record.get('steps', []) if step.get('registry_name') == 'unnormalizer_processor']
    if len(steps) != 1:
        raise ValueError('Expected one explicit checkpoint unnormalizer_processor')
    step = steps[0]
    config = step['config']
    actions = [(name, spec) for name, spec in config['features'].items() if spec['type'] == 'ACTION']
    if len(actions) != 1:
        raise ValueError('Expected one declared ACTION feature')
    feature, spec = actions[0]
    dimension = spec['shape'][-1]
    if type(dimension) is not int or dimension < 1:
        raise ValueError('Invalid checkpoint action dimension')
    return step, feature, dimension, config['norm_map'].get('ACTION', 'IDENTITY')


def action_decoder_from_record(record, stats, *, source=None):
    step, feature, dimension, mode = _processor(record)
    pairs = {'MEAN_STD': ('mean', 'std'), 'MIN_MAX': ('min', 'max'),
             'QUANTILES': ('q01', 'q99'), 'QUANTILE10': ('q10', 'q90')}
    if mode not in (*pairs, 'IDENTITY'):
        raise ValueError(f'Unsupported declared action normalization mode: {mode}')
    values = []
    if mode != 'IDENTITY':
        for suffix in pairs[mode]:
            key = f'{feature}.{suffix}'
            if key not in stats:
                raise ValueError(f'Missing checkpoint processor statistic: {key}')
            value = np.asarray(stats[key], dtype=np.float32).reshape(-1)
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint processor statistic shape/value: {key}')
            values.append(value.copy())
    else:
        values = [None, None]
    return ActionDecoder(mode, dimension, *values, float(step['config'].get('eps', 1e-8)), source)


def load_action_decoder(checkpoint):
    root = Path(checkpoint)
    path = root / 'policy_postprocessor.json'
    if not path.is_file():
        return None  # Preserve the existing OpenPI/Orbax quantile contract.
    record = json.loads(path.read_text())
    step, feature, dimension, mode = _processor(record)
    if mode == 'IDENTITY':
        return action_decoder_from_record(record, {}, source=str(path))
    from safetensors import safe_open
    with safe_open(str(root / step['state_file']), framework='np') as tensors:
        stats = {name: tensors.get_tensor(name) for name in tensors.keys() if name.startswith(feature + '.')}
    return action_decoder_from_record(record, stats, source=str(path))
