"""CPU contracts; actual inference is checked separately on real LIBERO inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _config(**changes):
    values = dict(backend='torch', sampler='euler', num_steps=10, framework='torch',
                  hardware='cuda_sm80', options={'source_dir': '/author', 'vlm_dir': '/vlm'})
    values.update(changes)
    return SimpleNamespace(**values)


def test_smolvla_routes_vision_attention_and_preserves_mixed_precision_exclusions():
    from flash_rt.vla.autotune.tools import get_tool

    spec_dir = Path(__file__).resolve().parents[1] / 'flash_rt/models/smolvla'
    sites = get_tool('attention.sage').attention_sites(
        'smolvla', hardware='cuda_sm80', spec_dir=spec_dir)
    vision, = sites['eligible_sites']
    assert (vision['id'], vision['mask_kind'], vision['head_dim']) == ('vision', 'none', 64)
    assert vision['shapes']['q'] == [1, 12, 1024, 64]
    flex = get_tool('attention.flex').attention_sites(
        'smolvla', hardware='cuda_sm80', spec_dir=spec_dir)
    masked = {site['id']: site for site in flex['ineligible_sites']}
    assert set(masked) == {'prefix', 'action_self', 'action_cross'}
    assert all(site['mask_kind'] == 'arbitrary' and site['gqa'] for site in masked.values())
    assert all(any('precision' in reason for reason in site['reasons']) for site in masked.values())


def test_smolvla_binding_preserves_graph_and_fixed_noise_options():
    from flash_rt.vla.adapters import smolvla

    config = _config(options={'source_dir': '/author', 'vlm_dir': '/vlm',
                              'use_cuda_graph': False, 'max_graph_profiles': 4})
    smolvla.validate(config)
    module, name, kwargs = smolvla.binding(config, 'cuda_sm80')
    assert (module, name) == ('flash_rt.frontends.torch.smolvla_cuda_sm80',
                              'SmolVLATorchFrontendCudaSm80')
    assert kwargs['use_cuda_graph'] is False
    assert kwargs['max_graph_profiles'] == 4
    assert kwargs['num_steps'] == 10
    assert kwargs['source_dir'] == '/author'


@pytest.mark.parametrize('changes,match', [
    ({'backend': 'flashrt'}, 'Torch'),
    ({'sampler': 'probeflow'}, 'Euler'),
    ({'num_steps': 0}, 'positive'),
    ({'options': {'source_dir': '/author'}}, 'vlm_dir'),
    ({'options': {'source_dir': '/author', 'vlm_dir': '/vlm', 'use_cuda_graph': 'yes'}}, 'bool'),
])
def test_smolvla_rejects_unimplemented_or_ambiguous_execution(changes, match):
    from flash_rt.vla.adapters import smolvla

    with pytest.raises((TypeError, ValueError), match=match):
        smolvla.validate(_config(**changes))


def test_smolvla_observation_rejects_camera_loss_and_nonfinite_state():
    from flash_rt.models.smolvla.inputs import validate_observation

    camera = np.zeros((256, 256, 3), dtype=np.uint8)
    observation = {'images': [camera, camera.copy()], 'state': np.arange(8, dtype=np.float32)}
    images, state = validate_observation(observation)
    assert images[0].shape == (256, 256, 3)
    np.testing.assert_array_equal(state, np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError, match='two'):
        validate_observation({**observation, 'images': [camera]})
    with pytest.raises(ValueError, match='finite'):
        validate_observation({**observation, 'state': np.full(8, np.nan, dtype=np.float32)})
    with pytest.raises(ValueError, match='uint8'):
        validate_observation({**observation, 'images': [camera.astype(np.float32), camera]})
    with pytest.raises(ValueError, match='8'):
        validate_observation({**observation, 'state': np.zeros(7, dtype=np.float32)})


def test_smolvla_noise_is_typed_explicit_and_shape_checked():
    from flash_rt.models.smolvla.inputs import validate_noise

    noise = np.arange(1600, dtype=np.float32).reshape(50, 32)
    np.testing.assert_array_equal(validate_noise(noise), noise[None])
    for bad in (noise.astype(np.float64), np.zeros((50, 7), dtype=np.float32),
                np.full((1, 50, 32), np.inf, dtype=np.float32)):
        with pytest.raises(ValueError):
            validate_noise(bad)


def test_smolvla_checkpoint_contract_rejects_other_profile(tmp_path):
    from flash_rt.models.smolvla.inputs import checkpoint_profile

    config = {
        'type': 'smolvla', 'chunk_size': 50, 'max_action_dim': 32, 'max_state_dim': 32,
        'input_features': {
            'observation.images.image': {'type': 'VISUAL', 'shape': [3, 256, 256]},
            'observation.images.image2': {'type': 'VISUAL', 'shape': [3, 256, 256]},
            'observation.state': {'type': 'STATE', 'shape': [8]}},
        'output_features': {'action': {'type': 'ACTION', 'shape': [7]}},
        'resize_imgs_with_padding': [512, 512],
        'adapt_to_pi_aloha': False, 'use_delta_joint_actions_aloha': False,
        'attention_mode': 'cross_attn', 'self_attn_every_n_layers': 2,
        'empty_cameras': 0, 'use_cache': True,
    }
    (tmp_path / 'config.json').write_text(json.dumps(config))
    assert checkpoint_profile(tmp_path)['action_dim'] == 7
    config['input_features']['observation.state']['shape'] = [14]
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='profile'):
        checkpoint_profile(tmp_path)


def test_smolvla_source_refuses_dirty_or_wrong_export(tmp_path):
    from flash_rt.models.smolvla.source import admit_source

    (tmp_path / 'src/lerobot/policies/smolvla').mkdir(parents=True)
    (tmp_path / 'AUTHOR_GIT_COMMIT').write_text('0' * 40)
    with pytest.raises(ValueError, match='revision'):
        admit_source(tmp_path)


def test_smolvla_real_workload_preserves_pixels_state_prompt_and_varies_noise(tmp_path):
    from examples.vla_autotune.smolvla_workload import write_workload
    from flash_rt.vla.autotune.workloads import NPZWorkload

    # Small controlled data tests serialization; this is never measurement evidence.
    pixels = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    cases = [{'images': [pixels, np.flip(pixels, axis=1).copy()],
              'state': np.arange(8, dtype=np.float32) + i,
              'prompt': f'LIBERO task {i}', 'episode': i, 'frame': 20}
             for i in range(4)]
    path = write_workload(cases, tmp_path / 'workload', dataset_identity={
        'repository': 'test-fixture', 'revision': '0' * 40, 'files': {}}, seed=7)
    workload = NPZWorkload.from_file(path)
    assert workload.model == 'smolvla'
    assert len(workload) == 4
    request = workload.request(2)
    np.testing.assert_array_equal(request.observation['images'][0], pixels)
    np.testing.assert_array_equal(request.observation['state'], np.arange(8, dtype=np.float32) + 2)
    assert request.prompt == 'LIBERO task 2'
    assert request.noise.shape == (1, 50, 32)
    assert not np.array_equal(workload.request(0).noise, request.noise)
    with pytest.raises(FileExistsError):
        write_workload(cases, path.parent, dataset_identity={}, seed=7)


def test_smolvla_dataset_provenance_rejects_modified_or_missing_bytes(tmp_path):
    from examples.vla_autotune.smolvla_workload import verify_dataset_files

    expected = {'sample': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}
    (tmp_path / 'sample').write_bytes(b'abc')
    assert verify_dataset_files(tmp_path, expected) == expected
    (tmp_path / 'sample').write_bytes(b'abd')
    with pytest.raises(ValueError, match='bytes'):
        verify_dataset_files(tmp_path, expected)
    with pytest.raises(ValueError, match='missing'):
        verify_dataset_files(tmp_path, {'missing-file': expected['sample']})


def test_smolvla_control_workload_preserves_real_requests_and_changes_one_input(tmp_path):
    from examples.vla_autotune.smolvla_workload import write_workload
    from flash_rt.vla.autotune.workloads import NPZWorkload

    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    cases = [{'images': [image, image], 'state': np.arange(8, dtype=np.float32),
              'prompt': f'Pick object {index}', 'episode': index, 'frame': 0}
             for index in range(4)]
    originals = write_workload(cases, tmp_path / 'real', dataset_identity={}, seed=7)
    path = write_workload(cases, tmp_path / 'extended', dataset_identity={}, seed=7,
                          include_controls=True)
    real, extended = NPZWorkload.from_file(originals), NPZWorkload.from_file(path)
    assert len(extended) == 16 and extended.workload_id != real.workload_id
    for index in range(4):
        reference, original = real.request(index), extended.request(index)
        assert original.request_id == reference.request_id
        np.testing.assert_array_equal(original.noise, reference.noise)
        np.testing.assert_array_equal(original.observation['images'], reference.observation['images'])
        assert original.prompt == reference.prompt
        noise, dark, prompt = (extended.request(4 + 3 * index + k) for k in range(3))
        assert not np.array_equal(noise.noise, original.noise)
        np.testing.assert_array_equal(noise.observation['images'], original.observation['images'])
        assert noise.prompt == original.prompt
        np.testing.assert_array_equal(dark.noise, original.noise)
        assert np.all(dark.observation['images'][0] == 115)
        assert dark.prompt == original.prompt
        np.testing.assert_array_equal(prompt.noise, original.noise)
        np.testing.assert_array_equal(prompt.observation['images'], original.observation['images'])
        assert prompt.prompt == original.prompt + '.'
        for control in (noise, dark, prompt):
            np.testing.assert_array_equal(control.observation['state'], original.observation['state'])
