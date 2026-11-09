import importlib.util
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/vla_autotune'


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location('raw_velocity_graph_pilot', EXAMPLES / 'fast_graph/velocity_graph.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs():
    return {'latents_action': torch.ones(1, 2, 1), 'timestep_action': torch.tensor([.5]),
            'context': torch.full((1, 3, 1), 3.), 'context_mask': torch.ones(1, 3, dtype=torch.bool),
            'video_cache_k': [torch.full((1, 2, 1), 4.), torch.full((1, 2, 1), 6.)],
            'video_cache_v': [torch.full((1, 2, 1), 5.), torch.full((1, 2, 1), 7.)],
            'action_attention_mask': torch.ones(1, 2, 4, dtype=torch.bool)}


def head(**data):
    return (2 * data['latents_action'] + data['timestep_action'].reshape(1, 1, 1) + data['context'].mean()
            + data['context_mask'].float().sum() + sum(x.mean() for x in data['video_cache_k'])
            + sum(x.mean() for x in data['video_cache_v']) + data['action_attention_mask'].float().sum())


def capture(fn, static):
    output = torch.zeros_like(static['latents_action'])
    def replay():
        output.copy_(fn(**static))
    return replay, output, lambda: None, {'warmup_evaluations': 0, 'captured_evaluators': 1,
                                          'captured_native_linear_calls': {}}


def test_reuses_graph_but_refreshes_time_latent_and_reused_condition_pointers(module):
    driver = module.VelocityGraph(head, capture=capture)
    data = inputs()
    driver.begin_request()
    first = driver(**data).clone()
    # x=1 gives 2 + .5 + 3 + 3 + (4+6) + (5+7) + 8 = 38.5.
    torch.testing.assert_close(first, torch.full_like(first, 38.5), rtol=0, atol=0)
    data['latents_action'].fill_(2); data['timestep_action'].fill_(1)
    second = driver(**data)
    torch.testing.assert_close(second, torch.full_like(second, 41.), rtol=0, atol=0)
    assert driver.snapshot()['replays'] == 2 and driver.snapshot()['captures'] == 1
    data['context'].fill_(9); data['context_mask'][0, -1] = False
    data['video_cache_k'][0].fill_(10); data['video_cache_v'][1].zero_()
    data['action_attention_mask'][0, 0, 0] = False
    data['latents_action'].fill_(3); data['timestep_action'].fill_(-.25)
    driver.begin_request()
    third = driver(**data)
    torch.testing.assert_close(third, torch.full_like(third, 44.75), rtol=0, atol=0)
    torch.testing.assert_close(first, torch.full_like(first, 38.5), rtol=0, atol=0)
    assert driver.snapshot()['captures'] == 0 and driver.snapshot()['replays'] == 1


def test_geometry_change_rejects_instead_of_using_stale_graph(module):
    driver = module.VelocityGraph(head, capture=capture)
    data = inputs(); driver.begin_request(); driver(**data)
    data['latents_action'] = torch.ones(1, 3, 1)
    driver.begin_request()
    with pytest.raises(ValueError, match='geometry'):
        driver(**data)


def test_cleanup_is_owned_and_idempotent_and_closed_driver_rejects(module):
    closed = []
    def factory(fn, static):
        replay, output, _, meta = capture(fn, static)
        return replay, output, lambda: closed.append('one'), meta
    driver = module.VelocityGraph(head, capture=factory)
    driver.begin_request(); driver(**inputs()); driver.close(); driver.close()
    assert closed == ['one']
    with pytest.raises(RuntimeError, match='closed'):
        driver.begin_request()


def test_disabled_mode_is_real_eager_not_a_graph_replay(module):
    driver = module.VelocityGraph(head, capture=lambda *args: pytest.fail('must not capture'))
    driver.enabled = False; driver.begin_request()
    actual = driver(**inputs())
    torch.testing.assert_close(actual, torch.full_like(actual, 38.5), atol=0, rtol=0)
    assert driver.snapshot()['replays'] == 0
