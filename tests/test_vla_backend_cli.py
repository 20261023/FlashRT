"""CPU description CLI contracts; no checkpoint or CUDA execution is needed."""

import json

import pytest

from flash_rt.vla.__main__ import main


@pytest.mark.parametrize('option, value', [
    ('--epsilon', '0.001'), ('--probe-horizon', '0.3'), ('--max-steps', '6'),
])
def test_euler_rejects_explicit_unused_probeflow_options(option, value):
    with pytest.raises(SystemExit) as exc:
        main(['--model', 'pi05', '--checkpoint', '/weights', option, value])
    assert exc.value.code == 2


def test_probeflow_options_are_visible_in_description(capsys):
    main(['--model', 'pi05', '--checkpoint', '/weights', '--sampler', 'probeflow',
          '--epsilon', '0.001', '--max-steps', '6'])
    result = json.loads(capsys.readouterr().out)
    assert result['probeflow']['epsilon'] == 0.001
    assert result['probeflow']['max_steps'] == 6
