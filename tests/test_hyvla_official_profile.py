"""CPU proofs for opt-in Hy numerical hooks, not end-to-end official parity.

The native extension is replaced only at the import boundary. Isolated official
math methods may be extracted from a local source checkout without importing or
constructing its model, tokenizer, training dependencies, or CUDA runtime.
"""

import ast
import importlib
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _native_modules():
    import flash_rt

    boundary = ModuleType("flash_rt.flash_rt_kernels")

    def forbidden(name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"CPU profile tests must not call native kernel {name}")

    boundary.__getattr__ = forbidden
    with patch.dict(sys.modules, {boundary.__name__: boundary}), patch.dict(
            flash_rt.__dict__, {"flash_rt_kernels": boundary}):
        frontend = importlib.import_module("flash_rt.frontends.torch.hyvla_probeflow")
        pipeline = importlib.import_module("flash_rt.models.hyvla.pipeline_thor")
    return frontend, pipeline


hy, native = _native_modules()


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("These are local CPU tests, not a GPU allocation")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.fixture
def official_math():
    configured = os.environ.get("HYVLA_OFFICIAL_SOURCE")
    if not configured:
        pytest.skip("Set HYVLA_OFFICIAL_SOURCE for optional source-matched math proofs")
    root = Path(configured) / "hy_vla"
    if not (root / "modeling_dual_tower.py").is_file():
        pytest.skip("Optional local official source is unavailable")
    namespace = {"torch": torch, "nn": torch.nn, "math": math, "Tensor": torch.Tensor}
    selections = [
        (root / "modeling_dual_tower.py", "HyDualTower", "eager_attention_forward", "attention"),
        (root / "hunyuan_vl_mot/modeling_hunyuan_vl_mot.py",
         "HunYuanVLMoTRMSNorm", "forward", "rms"),
        (root / "hunyuan_vl_mot/modeling_hunyuan_vl_mot.py",
         "HunYuanVLMoTRotaryEmbedding", "forward", "rope"),
        (root / "modeling_hy_vla.py", None, "_get_safe_dtype", "_get_safe_dtype"),
        (root / "modeling_hy_vla.py", None, "create_sinusoidal_pos_embedding", "sinusoidal"),
    ]
    for path, class_name, method_name, symbol in selections:
        tree = ast.parse(path.read_text(), filename=str(path))
        container = tree if class_name is None else next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        method = next(node for node in container.body
                      if isinstance(node, ast.FunctionDef) and node.name == method_name)
        method.name = symbol
        method.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{symbol: namespace[symbol] for symbol in ("attention", "rms", "rope", "sinusoidal")})


def _pipeline(profile="official_local_fp32"):
    return native.HyVLAThorBF16Pipeline(SimpleNamespace(numerical_profile=profile))


def _frontend(profile="official_local_fp32", sampler="euler", steps=10):
    obj = object.__new__(hy.HyVLAFlowFrontend)
    obj._numerical_profile = profile
    obj.device = "cpu"
    obj.cfg = {"vlm_config_dict": {"text_config": {
        "rope_scaling": {"type": "dynamic", "alpha": 1000.0},
        "max_position_embeddings": 262144,
    }}}
    obj.head_dim, obj.proj_width = 128, 1024
    obj.rope_theta = 10000.0
    obj._rope_base = 10000.0 * 1000.0 ** (128 / 126)
    obj.sampler, obj.num_steps = sampler, steps
    obj._requested_num_steps = steps if sampler == "euler" else None
    obj.probeflow_config = hy.ProbeFlowConfig(max_steps=4) if sampler == "probeflow" else None
    obj.n_heads, obj.d_vlm = 16, 2048
    obj.chunk, obj.max_action_dim, obj.max_state_dim = 2, 32, 32
    return obj


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_official_rms_matches_source_cast_before_weight(official_math, dtype):
    # Moving the weight multiplication before the BF16 cast changes this result.
    generator = torch.Generator().manual_seed(102)
    x = torch.randn(3, 7, generator=generator).to(dtype)
    weight = torch.randn(7, generator=generator).to(torch.bfloat16)
    expected = official_math.rms(SimpleNamespace(weight=weight, variance_epsilon=1e-5), x)
    actual = _pipeline()._rms_norm(x, weight)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    if dtype == torch.bfloat16:
        assert not torch.equal(expected, torch.nn.functional.rms_norm(x, (7,), weight, 1e-5))


def test_default_norm_stays_on_original_torch_path():
    x = torch.tensor([[0.2, 1.3, -2.7, 0.1]], dtype=torch.bfloat16)
    weight = torch.tensor([0.7, 1.2, 0.9, 0.3], dtype=torch.bfloat16)
    assert torch.equal(_pipeline("semantic_bf16")._rms_norm(x, weight),
                       torch.nn.functional.rms_norm(x, (4,), weight, 1e-5))


def test_official_rotary_keeps_bf16_inv_freq_storage_but_fp32_tables(official_math):
    obj = _frontend()
    positions = torch.tensor([[0, 1, 137, 239, 280]])
    # The official constructor initializes on CPU in FP32, then .to(bfloat16)
    # rounds its nonpersistent buffer before forward widens it back to FP32.
    inverse = (1. / (obj._rope_base ** (torch.arange(0, 128, 2).float() / 128))).bfloat16()
    ref = SimpleNamespace(inv_freq=inverse, attention_scaling=1.0)
    expected = official_math.rope(ref, torch.zeros(1).float(), positions)
    actual = obj._rope_cos_sin(positions)
    assert all(t.dtype == torch.float32 for t in actual)
    for left, right in zip(actual, expected):
        assert torch.equal(left[:, 0], right)
    # Unrounded FP32 inv_freq is *not* this checkpoint's BF16-buffer profile.
    unrounded = inverse.float() + 0.0001
    other = official_math.rope(SimpleNamespace(inv_freq=unrounded, attention_scaling=1.),
                               torch.zeros(1), positions)[0]
    assert not torch.equal(actual[0][:, 0], other)


def test_semantic_rotary_is_bitwise_legacy():
    obj = _frontend("semantic_bf16")
    positions = torch.tensor([[0, 137, 239]])
    actual = obj._rope_cos_sin(positions)
    expected = hy.HyVLATorchFrontendThor._rope_cos_sin(obj, positions)
    assert all(t.dtype == torch.bfloat16 for t in actual)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_fp32_qk_norm_and_fp32_key_cache_survive_real_block(official_math):
    pipe = _pipeline()
    pipe.n_heads = pipe.n_kv = 1
    pipe.head_dim = pipe.q_dim = pipe.kv_dim = 4
    hs = torch.tensor([[[0.2, -0.3, 0.7, 1.1], [-0.9, 0.4, 0.3, 0.6]]], dtype=torch.bfloat16)
    identity = torch.eye(4, dtype=torch.bfloat16)
    zero = torch.zeros_like(identity)
    norm = torch.tensor([0.8, 1.2, 1.1, 0.7], dtype=torch.bfloat16)
    weights = (torch.cat([identity] * 3), zero, torch.cat([zero, zero]), zero, norm, norm)
    cos = torch.tensor([[[[0.9, 0.8, 0.9, 0.8], [0.7, 0.6, 0.7, 0.6]]]])
    sin = torch.tensor([[[[0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4]]]])
    kbuf = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
    vbuf = torch.zeros(1, 1, 3, 4, dtype=torch.bfloat16)
    kbuf[:, :, 0].fill_(3.25)
    mask = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    result = pipe._block(hs, 3, None, weights, (norm, norm), mask, cos, sin, kbuf, vbuf, 1)
    normalized = official_math.rms(SimpleNamespace(weight=norm, variance_epsilon=1e-5), hs)
    raw = normalized[:, None]
    rotated = raw * cos + torch.cat([-raw[..., 2:], raw[..., :2]], -1) * sin
    key = official_math.rms(SimpleNamespace(weight=norm, variance_epsilon=1e-5), rotated)
    assert torch.equal(kbuf[:, :, 1:], key)
    assert torch.equal(vbuf[:, :, 1:], raw)
    assert torch.equal(kbuf[:, :, 0], torch.full_like(kbuf[:, :, 0], 3.25))
    assert not torch.equal(key, key.bfloat16().float())
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, hs)  # zero output/FFN weights, real attention/norm


def test_official_attention_matches_source_and_rounds_probability_to_bf16(official_math):
    generator = torch.Generator().manual_seed(13)
    q = torch.randn(1, 4, 3, 8, generator=generator)
    k = torch.randn(1, 2, 5, 8, generator=generator)
    v = torch.randn(1, 2, 5, 8, generator=generator).bfloat16()
    mask = torch.tensor([[[[True, False, True, True, False],
                           [True, True, False, True, True],
                           [False, False, False, False, False]]]])
    context = SimpleNamespace(config=SimpleNamespace(vlm_config=SimpleNamespace(
        num_attention_heads=4, num_key_value_heads=2)))
    expected, probabilities = official_math.attention(
        context, mask[:, 0], 1, 8, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    pipe = _pipeline()
    pipe.head_dim = 8
    actual = pipe._attn(q, k, v, mask)
    expected = expected.reshape(1, 3, 4, 8).transpose(1, 2)
    assert actual.dtype == probabilities.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    # A completely masked row uses a finite sentinel, hence uniform attention,
    # NOT SDPA's all-masked-row zero behavior.
    assert torch.count_nonzero(actual[:, :, -1]) > 0
    fp32_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.float().repeat_interleave(2, 1), attn_mask=mask).bfloat16()
    assert not torch.equal(actual, fp32_sdpa)


def test_official_qk_path_is_not_silently_downcast_by_ambient_autocast(official_math):
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(1, 2, 7, 8, generator=generator)
    k = torch.randn(1, 1, 7, 8, generator=generator)
    v = torch.randn(1, 1, 7, 8, generator=generator).bfloat16()
    mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    context = SimpleNamespace(config=SimpleNamespace(vlm_config=SimpleNamespace(
        num_attention_heads=2, num_key_value_heads=1)))
    expected, _ = official_math.attention(
        context, mask[:, 0], 1, 8, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    pipe = _pipeline()
    pipe.head_dim = 8
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = pipe._attn(q, k, v, mask)
    assert torch.equal(actual, expected.reshape(1, 7, 2, 8).transpose(1, 2))


@pytest.mark.parametrize("profile, expected_key_dtype, expected_rope_dtype", [
    ("semantic_bf16", torch.bfloat16, torch.bfloat16),
    ("official_local_fp32", torch.float32, torch.float32),
])
def test_profile_buffers_keep_values_and_large_gemm_inputs_bf16(profile, expected_key_dtype, expected_rope_dtype):
    buffers = _frontend(profile)._allocate_flow_buffers(3)
    assert buffers["kbuf"].dtype == expected_key_dtype
    assert buffers["pcos"].dtype == buffers["scos"].dtype == expected_rope_dtype
    assert buffers["vbuf"].dtype == buffers["state"].dtype == buffers["pe"].dtype == torch.bfloat16
    assert buffers["x"].dtype == buffers["velocity"].dtype == torch.float32


def test_official_euler_uses_fp32_recurrent_times_and_bf16_product(official_math):
    obj = _frontend(steps=10)
    obj._precompute_time_embs()
    assert obj._model_times == tuple(1. + i * (-1. / 10) for i in range(10))
    time = torch.tensor(1., dtype=torch.float32)
    dt = torch.tensor(-0.1, dtype=torch.float32)
    actual_times = []
    for _ in range(10):
        actual_times.append(float(time))
        time += dt
    assert obj._embedding_times == tuple(actual_times)
    expected_embeddings = official_math.sinusoidal(
        torch.tensor(actual_times, dtype=torch.float32), 1024, 4e-3, 4.0,
        device=torch.device("cpu")).bfloat16()
    assert torch.equal(obj._time_embs[:, 0], expected_embeddings)
    # Large enough vectors distinguish BF16-product rounding from FP32 updates.
    raw = torch.linspace(-1.7, 2.1, 64).reshape(1, 2, 32).bfloat16()
    initial = torch.full((1, 2, 32), 0.2)
    expected = initial.clone()
    for _ in range(10):
        expected += dt * raw
    b = {"x": initial.clone()}
    result = obj._sample_latent(b, lambda _t: -raw.float())
    assert torch.equal(result, expected)
    metadata = obj._last_sampler_metadata
    assert metadata["numerical_profile"] == "official_local_fp32"
    assert metadata["operation_dtypes"]["euler_increment"] == "bfloat16"
    assert metadata["embedding_timesteps"] == actual_times
    assert metadata["nfe"] == 10


def test_probeflow_keeps_exact_union_keys_and_fp32_integrator():
    obj = _frontend(sampler="probeflow")
    obj._precompute_time_embs()
    assert obj._model_times == obj.probeflow_config.timesteps()
    assert obj._embedding_times == tuple(float(torch.tensor(t, dtype=torch.float32)) for t in obj._model_times)
    raw = torch.linspace(-1.7, 2.1, 64).reshape(1, 2, 32).bfloat16()
    initial = torch.zeros(1, 2, 32)
    result = obj._sample_latent({"x": initial}, lambda _t: -raw.float())
    assert torch.equal(result, -raw.float())
    assert obj._last_sampler_metadata["nfe"] == 2
    assert obj._last_sampler_metadata["operation_dtypes"]["integration"] == "float32"


def test_nondefault_probe_union_retains_callback_keys_and_source_time_embedding(official_math):
    obj = _frontend(sampler="probeflow")
    obj.probeflow_config = hy.ProbeFlowConfig(probe_horizon=0.37, max_steps=7, step_increment=3)
    obj._precompute_time_embs()
    keys = obj.probeflow_config.timesteps()
    assert obj._model_times == keys
    assert 1. - 0.37 in obj._time_index
    assert len(obj._time_index) == len(keys)
    expected = official_math.sinusoidal(torch.tensor(keys, dtype=torch.float32), 1024,
                                        4e-3, 4.0, device=torch.device("cpu")).bfloat16()
    assert torch.equal(obj._time_embs[:, 0], expected)


def test_untraced_rope_configuration_is_rejected_before_weight_loader(monkeypatch):
    obj = _frontend()
    obj.cfg["vlm_config_dict"]["text_config"]["rope_scaling"] = {"type": "linear", "factor": 2.}
    monkeypatch.setattr(hy.HyVLATorchFrontendThor, "_load_weights",
                        lambda _self: pytest.fail("untraced profile reached the weight loader"))
    with pytest.raises(NotImplementedError, match="dynamic RoPE"):
        obj._load_weights()


def _tiny_native_frontend(sampler):
    """Real 32-layer native towers, with tiny synthetic BF16 weights on CPU."""
    obj = _frontend(sampler=sampler, steps=2)
    if sampler == "probeflow":
        obj.probeflow_config = hy.ProbeFlowConfig(max_steps=2)
    obj.head_dim, obj.proj_width, obj.max_state_dim = 4, 4, 4
    obj.d_vlm, obj.n_heads = 4, 1
    obj._graph_cache = {}
    generator = torch.Generator().manual_seed(217)

    def weight(rows, cols):
        return (0.2 * torch.randn(rows, cols, generator=generator)).bfloat16()

    for branch in ("_vlm", "_exp"):
        for modality in (("_t", "_v") if branch == "_vlm" else ("_v",)):
            for name, rows, cols in (("qkv", 12, 4), ("o", 4, 4),
                                     ("gu", 8, 4), ("d", 4, 4)):
                setattr(obj, f"{branch}_{name}{modality}", [weight(rows, cols) for _ in range(32)])
            for name in ("ln_in", "ln_post"):
                setattr(obj, f"{branch}_{name}{modality}",
                        [torch.ones(4, dtype=torch.bfloat16) for _ in range(32)])
    obj._qk_norm_q = obj._qk_norm_k = [torch.ones(4, dtype=torch.bfloat16) for _ in range(32)]
    obj._exp_final_norm_w = torch.ones(4, dtype=torch.bfloat16)
    for name, rows, cols in (("state", 4, 4), ("ain", 4, 32), ("aout", 32, 4),
                             ("atmlp_in", 4, 8), ("atmlp_out", 4, 4)):
        setattr(obj, f"_{name}_w", weight(rows, cols))
        setattr(obj, f"_{name}_b", torch.zeros(rows, dtype=torch.bfloat16))
    obj.pipe = native.HyVLAThorBF16Pipeline(obj)
    obj.pipe.n_heads = obj.pipe.n_kv = 1
    obj.pipe.head_dim = obj.pipe.q_dim = obj.pipe.kv_dim = 4
    obj._precompute_time_embs()
    return obj


def test_real_native_prefix_and_head_use_same_profile_for_euler2_and_probeflow2():
    initial = torch.linspace(-0.7, 0.9, 64).reshape(1, 2, 32)
    outputs = []
    for sampler in ("euler", "probeflow"):
        obj = _tiny_native_frontend(sampler)
        pc, ps = obj._rope_cos_sin(torch.tensor([[0, 1, 2]]))
        sc, ss = obj._rope_cos_sin(torch.tensor([[3, 4, 5]]))
        result = obj._graph_forward(
            3, 1, torch.tensor([[[0.2, -0.1, 0.4, 0.5]] * 3], dtype=torch.bfloat16),
            torch.ones(1, 1, 3, 3, dtype=torch.bool), pc, ps,
            torch.ones(1, 1, 3, 6, dtype=torch.bool), sc, ss,
            torch.tensor([[0.2, 0.4, -0.3, 0.1]], dtype=torch.bfloat16),
            initial, use_graph=False)
        outputs.append(result)
        assert obj.pipe.numerical_profile == obj._last_sampler_metadata["numerical_profile"] == "official_local_fp32"
        assert obj._last_sampler_metadata["nfe"] == 2
        assert obj._last_sampler_metadata["operation_dtypes"]["key_cache"] == "float32"
        assert torch.isfinite(result).all()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(initial, torch.linspace(-0.7, 0.9, 64).reshape(1, 2, 32))


@pytest.mark.parametrize("profile", ["official_local_fp16", "official_exact", None, {}])
def test_invalid_numerical_profile_rejected_before_io(monkeypatch, profile):
    monkeypatch.setattr(hy.HyVLAFlowFrontend, "_require_arch", lambda _self: pytest.fail("reached CUDA probe"))
    with pytest.raises(ValueError, match="numerical_profile"):
        hy.HyVLAFlowFrontend("/must-not-be-read", numerical_profile=profile)


def test_constructor_default_profile_and_explicit_opt_in(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(hy.HyVLAFlowFrontend, "_require_arch", lambda _self: None)
    monkeypatch.setattr(hy.HyVLAFlowFrontend, "_load_weights", lambda _self: None)
    monkeypatch.setattr(hy.HyVLAFlowFrontend, "_precompute_time_embs", lambda _self: None)
    default = hy.HyVLAFlowFrontend(str(tmp_path), use_cuda_graph=False)
    official = hy.HyVLAFlowFrontend(str(tmp_path), numerical_profile="official_local_fp32", use_cuda_graph=False)
    assert default.numerical_profile == default.pipe.numerical_profile == "semantic_bf16"
    assert official.numerical_profile == official.pipe.numerical_profile == "official_local_fp32"
    with pytest.raises(AttributeError):
        official.numerical_profile = "semantic_bf16"  # fixed profile per graph-owning instance
