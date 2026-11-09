"""Backend selection contracts. Written without running, per user request."""

import pytest

from flash_rt.vla import BackendConfig, describe_backend


@pytest.mark.parametrize('alias, expected', [
    ('pi0', 'pi0'), ('pi0.5', 'pi05'), ('pi05', 'pi05'),
    ('hy-embodied-vla', 'hyvla'), ('hy-embadding-vla', 'hyvla'),
    ('fastwam', 'fastwam'),
])
def test_model_aliases_select_the_intended_family(alias, expected):
    config = BackendConfig(model=alias, checkpoint='/weights')
    assert config.model == expected


def test_pi0fast_is_not_substituted_for_fastwam():
    with pytest.raises(ValueError, match='model'):
        BackendConfig(model='pi0fast', checkpoint='/weights')


def test_probeflow_requires_a_valid_policy_not_an_ignored_dict():
    with pytest.raises((TypeError, ValueError), match='ProbeFlowConfig'):
        BackendConfig(model='pi05', checkpoint='/weights',
                      sampler='probeflow', probeflow={})


def test_euler_does_not_accept_an_unused_probeflow_policy():
    from flash_rt.sampling import ProbeFlowConfig
    with pytest.raises(ValueError, match='probeflow'):
        BackendConfig(model='pi05', checkpoint='/weights',
                      probeflow=ProbeFlowConfig())


def test_describe_needs_no_checkpoint_access_or_gpu():
    result = describe_backend(BackendConfig(
        model='pi05', checkpoint='/does/not/exist',
        hardware='cuda_sm80', sampler='probeflow'))
    assert result['model'] == 'pi05'
    assert result['backend'] == 'flashrt'
    assert result['sampler'] == 'probeflow'
    assert result['runtime_verified'] is False


@pytest.mark.parametrize('steps', [0, -1, True, 2.5])
def test_invalid_fixed_step_budget_is_rejected(steps):
    with pytest.raises(ValueError, match='num_steps'):
        BackendConfig(model='pi0', checkpoint='/weights', num_steps=steps)
