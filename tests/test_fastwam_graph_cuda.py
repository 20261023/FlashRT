"""Small real-CUDA regressions, explicitly enabled in an allocated test step.

The normal CPU suite never queries CUDA for this module. No policy weights,
upstream inference or task-quality claims are involved.
"""
import os

import pytest

if os.environ.get('FLASHRT_RUN_ALLOCATED_FAST_GRAPH_TESTS') != '1':
    pytest.skip('enable only in an explicitly allocated CUDA test step', allow_module_level=True)
if not os.environ.get('SLURM_STEP_ID', '').isdigit():
    pytest.skip('allocated CUDA regressions require an srun compute step', allow_module_level=True)

torch = pytest.importorskip('torch')


def test_real_raw_capture_refresh_and_cleanup_do_not_flush_global_caches(monkeypatch):
    from flash_rt.models.fastwam.velocity_graph import FastWAMVelocityGraph
    from test_fastwam_velocity_graph import head, inputs
    def cuda_tree(value):
        if isinstance(value, dict): return {key: cuda_tree(item) for key, item in value.items()}
        if isinstance(value, list): return [cuda_tree(item) for item in value]
        return value.cuda()
    def forbidden(*args, **kwargs): pytest.fail('owned raw capture must not flush global caches')
    monkeypatch.setattr(torch.cuda, 'empty_cache', forbidden)
    monkeypatch.setattr(torch._C, '_host_emptyCache', forbidden, raising=False)
    data = cuda_tree(inputs())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = FastWAMVelocityGraph(head)
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            graph.begin_request(); first = graph(**data).clone(); graph.end_request()
            data['context'].fill_(9); data['latents_action'].fill_(2); data['timestep_action'].fill_(1)
            graph.begin_request(); second = graph(**data).clone(); graph.end_request()
        stream.synchronize()
        torch.testing.assert_close(first.cpu(), torch.full((1, 2, 1), 38.5), atol=0, rtol=0)
        torch.testing.assert_close(second.cpu(), torch.full((1, 2, 1), 47.), atol=0, rtol=0)
        assert graph.snapshot()['total_captures'] == 1 and graph.snapshot()['total_replays'] == 2
    finally:
        graph.close()


def test_real_eager_close_completes_async_velocity_before_releasing_time_bank():
    from flash_rt.frontends.torch.fastwam import FastWAMTorchFrontend
    from flash_rt.models.fastwam.pipeline import FastWAMCondition
    from test_fastwam_adapter import _tiny_model
    device = torch.device('cuda', torch.cuda.current_device())
    model = _tiny_model(torch)
    model.device = device
    model.action_expert.freqs = model.action_expert.freqs.to(device)
    def delayed(**kwargs):
        # Test-only delay makes the in-flight ownership race observable without
        # loading policy weights or relying on an enormous matmul allocation.
        torch.cuda._sleep(50_000_000)
        return kwargs['latents_action'] + kwargs['timestep_action']
    model._denoise_action_with_video_cache = delayed
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, num_inference_steps=2)
    state = torch.ones(1, 2, 2, device=device)
    condition = FastWAMCondition(torch.zeros(1, 1, 4, device=device),
        torch.ones(1, 1, dtype=torch.bool, device=device), [], [], torch.ones(1, 1, dtype=torch.bool, device=device))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    done = torch.cuda.Event()
    try:
        with torch.cuda.stream(stream):
            result = frontend.pipeline.evaluate_velocity(state, 1., condition)
            done.record(stream)
        frontend.close()
        assert done.query(), 'close returned before the independent eager CUDA work completed'
        assert frontend.pipeline._time_bank is None
        torch.full((64,), -999., device=device)  # permit immediate allocator reuse
        torch.testing.assert_close(result.cpu(), torch.full((1, 2, 2), 1001.), atol=0, rtol=0)
    finally:
        frontend.close()
