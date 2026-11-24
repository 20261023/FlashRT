"""Author producers and file-backed workers share the model's declared contract."""
import importlib
import json

import numpy as np
import pytest

from flash_rt.vla.autotune.author import _validate_spec
from flash_rt.vla.autotune.quality import output_domain
from flash_rt.vla.autotune.workloads import NPZWorkload
from flash_rt.vla.model_spec import get_spec


def test_declared_public_channels_can_skip_padding_between_arms():
    from flash_rt.vla.autotune.quality import compare_outputs, validate_quality_contract
    reference = np.array([[1., 0., 1., 0.]])
    gate = {'scope': 'bimanual', 'action_dim': 2, 'action_channels': [0, 2],
            'max_relative_l2': 0.05, 'min_cosine': 0.999}
    def compare(actual):
        return compare_outputs([np.array([actual])], ['frame'], [reference], ['frame'], gate)
    good = compare([1., 8., 1., -3.])
    assert good['status'] == 'passed'
    assert compare([1., 0., -1., 0.])['status'] == 'failed'
    with pytest.raises(ValueError, match='contract'):
        validate_quality_contract(good, dict(gate, action_channels=[0, 1]))


def test_new_model_can_use_declared_author_and_action_domain(tmp_path):
    spec = get_spec('hyvla')
    spec.update(model='example_vla', name='Example VLA', aliases=[])
    spec['actions']['worker_domain'] = 'example-normalized-actions'
    directory = tmp_path / 'specs'
    directory.mkdir()
    (directory / 'example_vla.json').write_text(json.dumps(spec))
    producer = {'model': 'example_vla', 'spec_dir': str(directory),
                'source_revision': spec['identity']['source']['revision'],
                'source': '/author', 'checkpoint': '/weights',
                'num_steps': spec['author']['default_steps'], 'options': {}}
    assert _validate_spec(producer) == producer
    assert callable(importlib.import_module(spec['author']['adapter']).load)
    assert output_domain('example_vla', spec_dir=directory) == 'example-normalized-actions'
    fixture = tmp_path / 'frame.npz'
    np.savez(fixture, state=np.zeros(1), noise=np.ones((1, 2)))
    manifest = {'model': 'example_vla', 'workload_id': 'frames', 'schema_version': 1,
                'requests': [{'request_id': 'frame', 'npz_path': str(fixture),
                              'fields': {'state': 'state'}, 'noise_key': 'noise'}]}
    assert NPZWorkload(manifest, tmp_path, spec_dir=directory).request(0).noise.shape == (1, 2)
    for value in [dict(producer, num_steps=2), dict(producer, source_revision='1' * 40)]:
        with pytest.raises(ValueError):
            _validate_spec(value)
