"""Native HyVLA sampler contracts, including CPU fake-graph fixtures.

All fixtures are small CPU tensors or fake graph replays. No checkpoint,
tokenizer download, CUDA inference, or GPU is needed to run these tests later.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_probeflow as hy
    from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)

from flash_rt.sampling.probeflow import ProbeFlowConfig


@pytest.fixture
def tiny_pipeline():
    def weight(rows, cols):
        return (torch.arange(rows * cols).reshape(rows, cols).float()
                .remainder(13).sub(6).div(40).to(torch.bfloat16))

    width, action_dim = 4, 32
    w = SimpleNamespace(
        _state_w=weight(width, 3), _state_b=weight(1, width)[0],
        _ain_w=weight(width, action_dim), _ain_b=weight(1, width)[0],
        _atmlp_in_w=weight(width, 2 * width), _atmlp_in_b=weight(1, width)[0],
        _atmlp_out_w=weight(width, width), _atmlp_out_b=weight(1, width)[0],
        _exp_final_norm_w=torch.ones(width, dtype=torch.bfloat16),
        _aout_w=weight(action_dim, width), _aout_b=weight(1, action_dim)[0],
        _qk_norm_q=[None] * 32, _qk_norm_k=[None] * 32,
    )
    pipe = HyVLAThorBF16Pipeline(w)
    pipe._exp_w = lambda _li: None

    def block(hs, n_vis, w_text, *_args, **_kwargs):
        assert n_vis == 3  # the expert sees one state row + two action rows
        assert w_text is None
        return hs + hs[:, :1] * 0.01

    pipe._block = block
    return pipe


def _head_inputs():
    x = torch.linspace(-1, 1, 64).reshape(1, 2, 32)
    state = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.bfloat16)
    time = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16)
    kv = torch.zeros(32, 1)
    return x, state, time, kv


def _legacy_raw_head(pipe, x, time, state_emb):
    """Original BF16 head math with the fixture's cheap expert blocks."""
    f = torch.nn.functional
    w = pipe.W
    action = f.linear(x.to(torch.bfloat16), w._ain_w, w._ain_b)
    ate = torch.cat([action, time.expand_as(action)], 2)
    ate = f.linear(ate, w._atmlp_in_w, w._atmlp_in_b)
    ate = f.linear(f.silu(ate), w._atmlp_out_w, w._atmlp_out_b)
    hs = torch.cat([state_emb, ate], 1)
    for _ in range(32):
        hs = hs + hs[:, :1] * 0.01
    hs = f.rms_norm(hs, (hs.shape[-1],), w._exp_final_norm_w, pipe.rms_eps)
    return f.linear(hs[:, -x.shape[1]:], w._aout_w, w._aout_b)


def test_velocity_head_is_raw_bf16_and_does_not_integrate(tiny_pipeline):
    pipe = tiny_pipeline
    x, state, time, kv = _head_inputs()
    initial = x.clone()
    state_emb = pipe.embed_state(state)
    raw = pipe.evaluate_velocity(x, time, state_emb, None, None, None, kv, kv, 7)
    assert raw.dtype == torch.bfloat16
    assert raw.shape == x.shape  # never return the state-token row
    assert torch.equal(x, initial)
    assert torch.equal(raw, _legacy_raw_head(pipe, initial, time, state_emb))


def test_extracted_head_keeps_legacy_euler_multiply_then_add(tiny_pipeline):
    pipe = tiny_pipeline
    x, state, time, kv = _head_inputs()
    times = torch.stack([time, time * 0.5, time * 0.25])
    expected = x.clone()
    state_emb = pipe.embed_state(state)
    for row in times:
        expected.add_((-1.0 / 3) * _legacy_raw_head(pipe, expected, row, state_emb).float())
    result = pipe.denoise(state, x, times, None, None, None, kv, kv, 7, num_steps=3)
    assert result is x
    assert torch.equal(result, expected)


def test_euler_embeds_state_only_once(tiny_pipeline, monkeypatch):
    x, state, time, kv = _head_inputs()
    original = tiny_pipeline.embed_state
    calls = []

    def embed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(tiny_pipeline, "embed_state", embed)
    tiny_pipeline.denoise(state, x, torch.stack([time, time]),
                          None, None, None, kv, kv, 7, num_steps=2)
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs, error", [
    ({"sampler": "other"}, "sampler"),
    ({"num_steps": 0}, "num_steps"),
    ({"num_steps": True}, "num_steps"),
    ({"num_steps": 2.5}, "num_steps"),
    ({"probeflow_config": {}}, "ProbeFlowConfig"),
    ({"probeflow_config": ProbeFlowConfig()}, "sampler"),
    ({"sampler": "probeflow", "num_steps": 4}, "num_steps"),
    ({"use_fp8": True}, "BF16"),
    ({"use_fp4": True}, "BF16"),
    ({"use_fused": True}, "BF16"),
    ({"use_fp8_vit": True}, "BF16"),
    ({"use_cuda_graph": "false"}, "use_cuda_graph"),
    ({"hardware": "invented_arch"}, "hardware"),
])
def test_invalid_options_fail_before_device_or_checkpoint(monkeypatch, kwargs, error):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid options reached the hardware probe")

    monkeypatch.setattr(hy.HyVLAFlowFrontend, "_require_arch", forbidden)
    with pytest.raises((ValueError, TypeError), match=error):
        hy.HyVLAFlowFrontend("/checkpoint-must-not-be-read", **kwargs)


def test_unknown_kwargs_are_not_silently_swallowed():
    with pytest.raises(TypeError, match="unexpected keyword"):
        hy.HyVLAFlowFrontend("/unused", use_int8=True)


@pytest.mark.parametrize("cap", [(8, 0), (8, 7), (8, 9), (11, 0), (12, 0)])
def test_arch_gate_uses_real_supported_capabilities(monkeypatch, cap):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj.hardware = "auto"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: cap)
    obj._require_arch()
    assert obj.capability == cap


def test_force_arch_cannot_override_real_hardware(monkeypatch):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj.hardware = "auto"
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    with pytest.raises(RuntimeError, match="capability"):
        obj._require_arch()


def test_no_cuda_rejected_even_with_force_arch(monkeypatch):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj.hardware = "auto"
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        obj._require_arch()


def test_explicit_arch_does_not_spoof_a_different_device(monkeypatch):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj.hardware = "cuda_sm80"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    with pytest.raises(RuntimeError, match="hardware"):
        obj._require_arch()


def _bare_frontend(sampler="probeflow", policy=None):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj.device = "cpu"
    obj._numerical_profile = "semantic_bf16"
    obj.sampler = sampler
    obj.probeflow_config = policy or ProbeFlowConfig(max_steps=4)
    obj._requested_num_steps = 3 if sampler == "euler" else None
    obj.num_steps = 10  # checkpoint default, not the size of a time union
    obj.proj_width = 4
    obj.chunk, obj.max_action_dim, obj.max_state_dim = 2, 32, 3
    obj.n_heads, obj.head_dim, obj.d_vlm = 1, 2, 4
    obj.use_cuda_graph = False
    obj._graph_cache = {}
    obj._last_sampler_metadata = None
    obj._precompute_time_embs()
    return obj


def test_exact_time_union_is_not_the_euler_step_budget():
    policy = ProbeFlowConfig(probe_horizon=0.37, max_steps=7, step_increment=3)
    obj = _bare_frontend(policy=policy)
    assert obj._model_times == policy.timesteps()
    assert 1. - policy.probe_horizon in obj._time_index
    assert obj._time_embs.shape[0] == len(policy.timesteps())
    assert obj.num_steps == 10
    euler = _bare_frontend("euler")
    assert euler.num_steps == 3
    assert euler._model_times == tuple(1. + i * (-1. / 3) for i in range(3))


def _request(obj):
    sp = 2
    ss = 1 + obj.chunk
    return (
        sp, 1, torch.zeros(1, sp, obj.d_vlm, dtype=torch.bfloat16),
        torch.ones(1, 1, sp, sp, dtype=torch.bool),
        torch.ones(1, 1, sp, obj.head_dim, dtype=torch.bfloat16),
        torch.zeros(1, 1, sp, obj.head_dim, dtype=torch.bfloat16),
        torch.ones(1, 1, ss, sp + ss, dtype=torch.bool),
        torch.ones(1, 1, ss, obj.head_dim, dtype=torch.bfloat16),
        torch.zeros(1, 1, ss, obj.head_dim, dtype=torch.bfloat16),
        torch.zeros(1, obj.max_state_dim, dtype=torch.bfloat16),
        torch.zeros(1, obj.chunk, obj.max_action_dim),
    )


class _FakePipe:
    def __init__(self, head):
        self.head = head
        self.prefixes = self.states = self.evaluations = 0

    def prefill(self, pe, _n_vis, _pmask, _pcos, _psin, kbuf, _vbuf):
        self.prefixes += 1
        kbuf.fill_(pe.float().sum())

    def embed_state(self, state):
        self.states += 1
        return state[:, :1, None].expand(1, 1, 4)

    def evaluate_velocity(self, x, time, state_emb, _smask, _scos, _ssin,
                          kbuf, _vbuf, _sp):
        self.evaluations += 1
        assert x.dtype == torch.float32
        assert x.shape == (1, 2, 32)
        return self.head(x, time, state_emb, kbuf).to(torch.bfloat16)


def test_eager_probeflow_converts_head_to_fp32_and_negates_once():
    obj = _bare_frontend()
    obj.pipe = _FakePipe(lambda x, *_: torch.ones_like(x))
    request = _request(obj)
    result = obj._graph_forward(*request, use_graph=False)
    assert torch.equal(result, torch.full_like(result, -1.))
    assert result.dtype == torch.float32
    assert torch.equal(request[-1], torch.zeros_like(request[-1]))
    assert obj.pipe.prefixes == obj.pipe.states == 1
    assert obj.pipe.evaluations == obj._last_sampler_metadata["nfe"] == 2
    assert obj._last_sampler_metadata["scheduled_steps"] == 2


def test_dense_route_counts_discarded_probe_and_reroutes_next_request():
    obj = _bare_frontend(policy=ProbeFlowConfig(epsilon=0.001, max_steps=4))
    obj.pipe = _FakePipe(lambda x, *_: torch.where(x >= -0.1, 1., -1.))
    obj._graph_forward(*_request(obj), use_graph=False)
    assert obj._last_sampler_metadata["scheduled_steps"] == 4
    assert obj._last_sampler_metadata["nfe"] == 5
    obj.pipe.head = lambda x, *_: torch.ones_like(x)
    obj._graph_forward(*_request(obj), use_graph=False)
    assert obj._last_sampler_metadata["scheduled_steps"] == 2
    assert obj._last_sampler_metadata["nfe"] == 2


def test_eager_euler_reports_actual_budget():
    obj = _bare_frontend("euler")
    obj.pipe = _FakePipe(lambda x, *_: torch.ones_like(x))
    result = obj._graph_forward(*_request(obj), use_graph=False)
    expected = torch.zeros_like(result)
    for _ in range(3):
        expected.add_((-1. / 3) * torch.ones_like(result))
    assert torch.equal(result, expected)
    assert obj._last_sampler_metadata["nfe"] == 3
    assert obj._last_sampler_metadata["timesteps"] == list(obj._model_times)
    assert obj.pipe.prefixes == obj.pipe.states == 1


def _fake_graph_builder(obj, monkeypatch):
    builds = []

    class Replay:
        def __init__(self, body):
            self.body = body

        def replay(self):
            self.body()

    def build(key):
        builds.append(key)
        sp, n_vis, _times = key
        b = obj._allocate_flow_buffers(sp)
        bank = {
            t: Replay(lambda index=i: obj._velocity_body(b, index, sp))
            for i, t in enumerate(obj._model_times)
        }
        return {"buf": b, "prefix_graph": Replay(lambda: obj._prefix_body(b, n_vis)),
                "velocity_graphs": bank}

    monkeypatch.setattr(obj, "_build_graph", build)
    return builds


def test_fake_graph_bank_reads_new_prefix_state_masks_and_noise(monkeypatch):
    obj = _bare_frontend()
    obj.pipe = _FakePipe(
        lambda x, _time, state, kv: torch.ones_like(x) * (1 + state.sum() + kv[0, 0, 0, 0, 0]))
    builds = _fake_graph_builder(obj, monkeypatch)
    first = _request(obj)
    out1 = obj._graph_forward(*first, use_graph=True)
    second = list(_request(obj))
    second[2].fill_(0.25)  # same shape, new prompt/image embeddings
    second[3].fill_(False)
    second[4].fill_(0.5)
    second[6].fill_(False)
    second[7].fill_(0.5)
    second[-2].fill_(0.5)
    second[-1].fill_(2.)
    out2 = obj._graph_forward(*second, use_graph=True)
    assert len(builds) == 1
    assert obj.pipe.prefixes == obj.pipe.states == 2
    assert not torch.equal(out1, out2)
    assert torch.equal(out1, torch.full_like(out1, -1.))  # output is owned, not scratch
    b = next(iter(obj._graph_cache.values()))["buf"]
    for name, index in (("pe", 2), ("pmask", 3), ("pcos", 4),
                        ("smask", 6), ("scos", 7), ("state", 9)):
        assert torch.equal(b[name], second[index])


def test_fake_graph_bank_matches_eager_and_routes_again_per_request(monkeypatch):
    obj = _bare_frontend(policy=ProbeFlowConfig(epsilon=0.001, max_steps=4))
    obj.pipe = _FakePipe(lambda x, *_: torch.where(x >= -0.1, 1., -1.))
    builds = _fake_graph_builder(obj, monkeypatch)
    eager = obj._graph_forward(*_request(obj), use_graph=False)
    captured = obj._graph_forward(*_request(obj), use_graph=True)
    assert torch.equal(eager, captured)
    assert obj._last_sampler_metadata["nfe"] == 5
    obj.pipe.head = lambda x, *_: torch.ones_like(x)
    second = obj._graph_forward(*_request(obj), use_graph=True)
    assert obj._last_sampler_metadata["nfe"] == 2
    assert torch.equal(second, torch.full_like(second, -1.))
    assert len(builds) == 1


def test_global_eager_flag_reaches_both_vit_and_denoise(monkeypatch):
    obj = _bare_frontend()
    calls = []

    def predict(_self, images, prompt=None, state=None, noise=None, use_graph=True):
        calls.append((images, prompt, state, noise, use_graph))
        return np.zeros((1, 2, 32), dtype=np.float32)

    monkeypatch.setattr(hy.HyVLATorchFrontendThor, "predict_actions", predict)
    noise = np.zeros((1, 2, 32), dtype=np.float32)
    obj.predict_actions("images", prompt="task", state=[0.], noise=noise)
    obj.predict_actions("images", use_graph=True)
    assert calls[0][3] is noise
    assert all(call[-1] is False for call in calls)


def test_native_camera_dict_order_and_uint8_helper_are_preserved(monkeypatch):
    obj = _bare_frontend()
    obj.image_keys = ["right", "left"]
    calls = []

    def predict(_self, images, **_kwargs):
        calls.append(images)
        return np.zeros((1, 2, 32), dtype=np.float32)

    monkeypatch.setattr(hy.HyVLATorchFrontendThor, "predict_actions", predict)
    obj.predict_actions({"left": np.zeros((2, 2, 3), dtype=np.uint8),
                         "right": np.full((2, 2, 3), 255, dtype=np.uint8)})
    assert calls[0].shape == (2, 1, 3, 2, 2)
    assert torch.equal(calls[0][0], torch.ones_like(calls[0][0]))
    assert torch.equal(calls[0][1], torch.zeros_like(calls[0][1]))


def test_infer_preserves_raw_actions_and_adds_real_sampler_metadata(monkeypatch):
    obj = _bare_frontend()
    raw = np.ones((2, 32), dtype=np.float32)

    def infer(_self, _obs):
        _self._last_sampler_metadata = {"sampler": "probeflow", "nfe": 5}
        return {"actions": raw}

    monkeypatch.setattr(hy.HyVLATorchFrontendThor, "infer", infer)
    result = obj.infer({"noise": np.zeros((1, 2, 32))})
    assert result["actions"] is raw
    assert result["sampler_metadata"] == {"sampler": "probeflow", "nfe": 5}
