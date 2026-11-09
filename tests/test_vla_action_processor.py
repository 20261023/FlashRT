import numpy as np
import pytest

from flash_rt.vla.processing import action_decoder_from_record


def record(mode):
    return {'steps': [{'registry_name': 'unnormalizer_processor',
                      'state_file': 'statistics.safetensors',
                      'config': {'eps': 1e-8, 'features': {'action': {'type': 'ACTION', 'shape': [2]}},
                                 'norm_map': {'ACTION': mode}}}]}


def test_explicit_mean_std_is_not_replaced_by_available_quantile_stats():
    stats = {'action.mean': np.array([100, 200]), 'action.std': np.array([2, 3]),
             'action.q01': np.array([-1, -1]), 'action.q99': np.array([1, 1])}
    decoder = action_decoder_from_record(record('MEAN_STD'), stats)
    raw = np.array([[1.0, 2.0, 999.0]], dtype=np.float32)
    np.testing.assert_array_equal(decoder(raw), [[102.0, 206.0]])
    np.testing.assert_array_equal(raw, [[1.0, 2.0, 999.0]])
    assert decoder.mode == 'MEAN_STD' and decoder.dimension == 2


@pytest.mark.parametrize('mode, keys', [('MIN_MAX', ('min', 'max')),
                                     ('QUANTILES', ('q01', 'q99')),
                                     ('QUANTILE10', ('q10', 'q90'))])
def test_range_processor_uses_its_declared_statistics(mode, keys):
    stats = {f'action.{keys[0]}': np.array([10, 20]), f'action.{keys[1]}': np.array([30, 40])}
    decoder = action_decoder_from_record(record(mode), stats)
    np.testing.assert_array_equal(decoder(np.array([[-1.0, 1.0]], np.float32)), [[10.0, 40.0]])


def test_unsupported_declared_processor_is_not_silently_quantile():
    with pytest.raises(ValueError, match='normalization mode'):
        action_decoder_from_record(record('UNKNOWN'), {})


def test_backend_decodes_from_raw_once_and_reports_the_actual_processor():
    from flash_rt.vla import BackendConfig, VLABackend
    decoder = action_decoder_from_record(record('MEAN_STD'),
                                         {'action.mean': [100, 200], 'action.std': [2, 3]})
    class Frontend:
        def set_prompt(self, prompt):
            pass
        def infer(self, observation):
            return {'actions': np.array([[-1, -1]], np.float32),
                    'raw_actions': np.array([[1, 2, 999]], np.float32),
                    'sampler_metadata': {'sampler': 'euler', 'nfe': 2}}
    backend = VLABackend(BackendConfig(model='pi05', checkpoint='/weights'), Frontend(),
                         hardware='cuda_sm80', precision='bf16',
                         action_domain=decoder.action_domain, action_decoder=decoder)
    result = backend.infer({}, prompt='pick')
    np.testing.assert_array_equal(result['actions'], [[102, 206]])
    np.testing.assert_array_equal(result['raw_actions'], [[1, 2, 999]])
    assert result['backend_metadata']['action_processor']['mode'] == 'MEAN_STD'
