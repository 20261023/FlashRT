"""FastWAM CPU adapter regression contracts.

These use tiny CPU fixtures/fake native pointers, not real upstream model
execution, weights, a hub, compiled extension or CUDA device. Real FastWAM
parity/kernel accuracy/performance still require separately authorized tests.
"""

from contextlib import nullcontext
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from flash_rt.frontends.torch.fastwam import FastWAMTorchFrontend
from flash_rt.models.fastwam.config import shifted_sigma, validate_sampler, vae_spatial_factor
from flash_rt.models.fastwam.loader import (
    _validate_full_payload, load_local_model, read_model_config, resolve_local_paths,
)
from flash_rt.sampling.probeflow import ProbeFlowConfig


def test_shifted_sigma_is_not_a_1000_scaled_integration_delta():
    grid = [shifted_sigma(1 - i / 4, 5) for i in range(5)]
    assert grid == pytest.approx([1, 15 / 16, 5 / 6, 5 / 8, 0])
    assert sum(b - a for a, b in zip(grid, grid[1:])) == pytest.approx(-1)
    assert [shifted_sigma(1 - i / 4, 1) for i in range(5)] == [1, .75, .5, .25, 0]


@pytest.mark.parametrize("kwargs, error, match", [
    ({"sigma_shift": 0}, ValueError, "sigma_shift"),
    ({"sigma_shift": float("nan")}, ValueError, "sigma_shift"),
    ({"sigma_shift": True}, ValueError, "sigma_shift"),
    ({"sampler": "unknown"}, ValueError, "sampler"),
    ({"num_inference_steps": 0}, ValueError, "num_inference_steps"),
    ({"num_inference_steps": 2.5}, ValueError, "num_inference_steps"),
    ({"action_horizon": True}, ValueError, "action_horizon"),
    ({"native_linear": "true"}, TypeError, "native_linear"),
    ({"native_linear": True, "device": "cpu"}, ValueError, "CUDA"),
    ({"native_linear": True, "dtype": "fp16"}, ValueError, "BF16"),
    ({"use_cuda_graph": "true"}, TypeError, "use_cuda_graph"),
    ({"use_cuda_graph": True, "device": "cpu"}, ValueError, "CUDA"),
    ({"use_cuda_graph": True, "native_linear": True}, ValueError, "cached"),
])
def test_invalid_options_fail_before_files_or_torch(monkeypatch, kwargs, error, match):
    frontend = importlib.import_module("flash_rt.frontends.torch.fastwam")

    def forbidden(*args, **kw):
        raise AssertionError("invalid options reached configuration/files")

    monkeypatch.setattr(frontend, "read_model_config", forbidden)
    options = dict(sigma_shift=1, action_horizon=8)
    options.update(kwargs)
    with pytest.raises(error, match=match):
        FastWAMTorchFrontend("/must-not-be-accessed", **options)


def test_sampler_options_do_not_silently_override_one_another():
    with pytest.raises(ValueError, match="Euler-only"):
        validate_sampler("probeflow", 1, 20, None)
    with pytest.raises(TypeError, match="ProbeFlowConfig"):
        validate_sampler("probeflow", 1, None, {})
    with pytest.raises(ValueError, match="requires sampler"):
        validate_sampler("euler", 1, None, ProbeFlowConfig())
    assert validate_sampler("euler", 5, None, None) == (20, None)
    assert isinstance(validate_sampler("probeflow", 5, None, None)[1], ProbeFlowConfig)


def _config():
    return {
        "_target_": "fastwam.runtime.create_fastwam",
        "video_dit_config": {
            "video_attention_mask_mode": "first_frame_causal",
            "seperated_timestep": True, "fuse_vae_embedding_in_latents": True,
            "patch_size": [1, 2, 2],
        },
        "action_dit_config": {"action_dim": 2},
        "action_scheduler": {"train_shift": 1, "infer_shift": 1, "num_train_timesteps": 1000},
        "load_text_encoder": False,
    }


def test_unresolved_or_wrong_model_configs_are_rejected():
    config = _config()
    config["action_dit_config"]["action_dim"] = "${data.action_dim}"
    with pytest.raises(ValueError, match="interpolation"):
        read_model_config(config)
    config = _config()
    config["_target_"] = "motus.runtime.create_model"
    with pytest.raises(ValueError, match="Only FastWAM"):
        read_model_config(config)
    config = _config()
    config["load_text_encoder"] = True
    with pytest.raises(ValueError, match="prepared context"):
        read_model_config(config)
    with pytest.raises(ValueError, match="resolved JSON"):
        read_model_config("/no-file.yaml")


def test_local_loader_paths_fail_without_model_or_cuda_import(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        resolve_local_paths(checkpoint, tmp_path)
    checkpoint.touch()
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE"):
        resolve_local_paths(checkpoint, tmp_path)
    (tmp_path / "Wan2.2_VAE.pth").touch()
    with pytest.raises(ValueError, match="exactly one"):
        resolve_local_paths(tmp_path, tmp_path)


def test_loader_uses_real_local_only_factory_arguments(monkeypatch, tmp_path):
    """A controlled factory verifies the API, not the real model execution."""
    events = []
    model = SimpleNamespace(
        mot=SimpleNamespace(state_dict=lambda: {}), proprio_encoder=None,
    )
    model.load_checkpoint = lambda path: events.append(("checkpoint", path)) or {"mot": {}}
    model.eval = lambda: model
    model.to = lambda **kwargs: events.append(("to", kwargs)) or model

    class Factory:
        @staticmethod
        def from_wan22_pretrained(**kwargs):
            events.append(("factory", kwargs))
            return model

    monkeypatch.setattr("flash_rt.models.fastwam.loader.importlib.import_module",
                        lambda name: SimpleNamespace(FastWAM=Factory))
    result = load_local_model(_config(), tmp_path / "policy.pt", tmp_path.resolve(), None,
                              device="sentinel-device", dtype="sentinel-dtype")
    kwargs = events[0][1]
    assert kwargs["device"] == "cpu"
    assert kwargs["model_id"] == str(tmp_path.resolve())
    assert kwargs["redirect_common_files"] is False
    assert kwargs["load_text_encoder"] is False
    assert kwargs["skip_dit_load_from_pretrain"] is True
    assert kwargs["action_dit_pretrained_path"] is None
    assert [event[0] for event in events] == ["factory", "checkpoint", "to"]
    assert result.device == "sentinel-device"


def test_partial_checkpoint_cannot_leave_random_action_weights():
    expected = SimpleNamespace(shape=(2, 3))
    model = SimpleNamespace(mot=SimpleNamespace(state_dict=lambda: {"mixtures.action.head.weight": expected}),
                            proprio_encoder=None)
    with pytest.raises(ValueError, match="full FastWAM"):
        _validate_full_payload(model, {"dit": {}})
    with pytest.raises(ValueError, match="Incomplete"):
        _validate_full_payload(model, {"mot": {}})
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_full_payload(model, {"mot": {"mixtures.action.head.weight": SimpleNamespace(shape=(3, 2))}})


@pytest.mark.parametrize("location_kind", ["file", "spec", "path", "both"])
def test_same_source_regular_and_namespace_modules_allow_repeated_scope(monkeypatch, tmp_path, location_kind):
    """Fixtures only: no upstream imports or namespace module execution."""
    loader = importlib.import_module("flash_rt.models.fastwam.loader")
    root = tmp_path / "FastWAM"
    directory = root / "src/fastwam/models/wan22/schedulers"
    module = SimpleNamespace(
        __file__=str(directory / "scheduler_continuous.py") if location_kind == "file" else None,
        __spec__=SimpleNamespace(submodule_search_locations=[str(directory)]
                                 if location_kind in ("spec", "both") else None),
        __path__=[str(directory)] if location_kind in ("path", "both") else None,
    )
    fake_sys = SimpleNamespace(modules={"fastwam.models.wan22.schedulers": module}, path=["original"])
    monkeypatch.setattr(loader, "sys", fake_sys)
    for _ in range(2):
        with loader._source_import_path(root):
            assert fake_sys.path == [str(root / "src"), "original"]
        assert fake_sys.path == ["original"]


@pytest.mark.parametrize("location_kind", ["file", "spec", "path", "mixed", "unknown"])
def test_different_source_or_unlocatable_namespace_modules_still_reject(monkeypatch, tmp_path, location_kind):
    loader = importlib.import_module("flash_rt.models.fastwam.loader")
    root, other = tmp_path / "FastWAM", tmp_path / "OtherFastWAM"
    valid = root / "src/fastwam/models/wan22/schedulers"
    outside = other / "src/fastwam/models/wan22/schedulers"
    module = SimpleNamespace(
        __file__=str(outside / "scheduler_continuous.py") if location_kind == "file" else None,
        __spec__=SimpleNamespace(submodule_search_locations=[str(outside)]
                                 if location_kind in ("spec", "mixed") else None),
        __path__=([str(valid)] if location_kind == "mixed" else
                  [str(outside)] if location_kind == "path" else None),
    )
    fake_sys = SimpleNamespace(modules={"fastwam.models.wan22.schedulers": module}, path=["original"])
    monkeypatch.setattr(loader, "sys", fake_sys)
    with pytest.raises(ImportError, match="different fastwam"):
        with loader._source_import_path(root):
            raise AssertionError("foreign namespace entered the import scope")
    assert fake_sys.path == ["original"]


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _tiny_model(torch, *, field=None, proprio_dim=None, vae_factor=16, factor_on_inner=False):
    """Small upstream-shaped seam, not FastWAM weights or an inference model."""
    model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32,
                            text_dim=4, proprio_dim=proprio_dim, evaluations=[], prefills=0,
                            text_encoder=None, tokenizer=None, encodes=0, prepares=0)
    model.eval = lambda: model
    model.action_expert = SimpleNamespace(action_dim=2, freqs=torch.zeros(1024, 1))
    model.vae = SimpleNamespace(model=SimpleNamespace())
    setattr(model.vae.model if factor_on_inner else model.vae, "upsampling_factor", vae_factor)

    def video_prepare(**kwargs):
        model.prepares += 1
        context = kwargs["context"]
        f, h, w = (size // patch for size, patch in zip(kwargs["x"].shape[-3:], model.video_expert.patch_size))
        tokens = torch.zeros(1, f * h * w, 4, dtype=kwargs["x"].dtype)
        return tokens, None, torch.zeros(1), context, kwargs["context_mask"], None, f, h, w, h * w

    model.video_expert = SimpleNamespace(
        video_attention_mask_mode="first_frame_causal", fuse_vae_embedding_in_latents=True,
        seperated_timestep=True, patch_size=(1, 2, 2), prepare=video_prepare,
    )

    def encode(**kwargs):
        model.encodes += 1
        scale_h, scale_w = (vae_factor, vae_factor) if isinstance(vae_factor, int) else vae_factor[-2:]
        return kwargs["input_image"][..., ::scale_h, ::scale_w].unsqueeze(2)

    def append(context, mask, proprio):
        token = torch.zeros(1, 1, 4)
        token[..., :proprio.shape[1]] = proprio
        return torch.cat([context, token], 1), torch.cat([mask, torch.ones(1, 1, dtype=torch.bool)], 1)

    def build_mask(video_seq_len, action_seq_len, **kwargs):
        mask = torch.ones(video_seq_len + action_seq_len, video_seq_len + action_seq_len, dtype=torch.bool)
        mask[:video_seq_len, video_seq_len:] = False
        return mask

    def prefill(**kwargs):
        model.prefills += 1
        return [kwargs["video_tokens"].clone()], [kwargs["video_tokens"].clone()]

    def denoise(**kwargs):
        state, time = kwargs["latents_action"], kwargs["timestep_action"]
        model.evaluations.append((time.item(), state.clone(), id(kwargs["video_cache_k"]),
                                  kwargs["context"].shape[1], kwargs["action_attention_mask"].shape))
        return torch.ones_like(state) if field is None else field(state, time)

    def forbidden(*args, **kwargs):
        raise AssertionError("An integrated upstream infer path was called as a velocity")

    model._encode_input_image_latents_tensor = encode
    model._append_proprio_to_context = append
    model._build_mot_attention_mask = build_mask
    model._denoise_action_with_video_cache = denoise
    model.infer_action = model.infer = model.infer_joint = forbidden
    model.mot = SimpleNamespace(prefill_video_cache_tensor=prefill)

    class Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def build_inference_schedule(num_inference_steps, device, dtype, shift_override):
            u = torch.linspace(1, 0, num_inference_steps + 1, device=device)
            sigma = shift_override * u / (1 + (shift_override - 1) * u)
            return (sigma[:-1] * 1000).to(dtype), (sigma[1:] - sigma[:-1]).to(dtype)

        @staticmethod
        def step(velocity, delta, state):
            return state + velocity * delta

    model.infer_action_scheduler = Scheduler()
    return model


def _obs(torch, proprio=None):
    result = {"input_image": torch.zeros(3, 32, 64), "context": torch.zeros(3, 4),
              "context_mask": torch.ones(3, dtype=torch.bool)}
    if proprio is not None:
        result["proprio"] = proprio
    return result


def test_euler_scales_model_time_but_not_integration(torch):
    model = _tiny_model(torch, field=lambda state, time: torch.ones_like(state) * (time / 1000))
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=3, num_inference_steps=2)
    noise = torch.zeros(1, 3, 2)
    result = frontend.infer(_obs(torch), noise)
    assert np.allclose(result["actions"], -.75)
    assert torch.equal(noise, torch.zeros_like(noise))
    assert [call[0] for call in model.evaluations] == [1000, 500]
    assert model.prefills == 1 and model.encodes == 1
    assert len({call[2] for call in model.evaluations}) == 1
    assert result["sampler_metadata"]["nfe"] == 2
    assert result["execution_metadata"]["backend"] == "upstream_torch_reference"
    assert result["execution_metadata"]["fully_native"] is False


def _attach_cpu_graph(frontend, torch):
    """Inject only capture/stream boundaries, leaving real pipeline math intact."""
    from flash_rt.models.fastwam.velocity_graph import FastWAMVelocityGraph
    released = []
    def capture(fn, static):
        output = torch.empty_like(static['latents_action'])
        return (lambda: output.copy_(fn(**static)), output, lambda: released.append(True),
                {'warmup_evaluations': 0, 'encoded_bodies': 1, 'captured_native_linear_calls': {}})
    frontend.pipeline._velocity_graph = FastWAMVelocityGraph(
        frontend.model._denoise_action_with_video_cache, capture=capture)
    return released


@pytest.mark.parametrize('sampler,dense', [('euler', False), ('probeflow', False), ('probeflow', True)])
def test_pipeline_graph_matches_euler10_and_both_probe_branches_with_borrowed_output(torch, sampler, dense):
    def field(state, time):
        return torch.ones_like(state) * ((1 if time.item() > 500 else -1) if dense else time/1000)
    options = dict(sigma_shift=1, action_horizon=3, sampler=sampler)
    if sampler == 'euler': options['num_inference_steps'] = 10
    else: options['probeflow_config'] = ProbeFlowConfig(epsilon=.1, max_steps=4)
    eager = FastWAMTorchFrontend(model=_tiny_model(torch, field=field), **options)
    graphed = FastWAMTorchFrontend(model=_tiny_model(torch, field=field), **options)
    original = graphed.model._denoise_action_with_video_cache
    released = _attach_cpu_graph(graphed, torch)
    noise = torch.zeros(1, 3, 2)
    expected = eager.infer(_obs(torch), noise)
    actual = graphed.infer(_obs(torch), noise)
    np.testing.assert_array_equal(actual['actions'], expected['actions'])
    assert actual['sampler_metadata'] == expected['sampler_metadata']
    expected_nfe = 10 if sampler == 'euler' else 5 if dense else 2
    assert actual['sampler_metadata']['nfe'] == expected_nfe
    assert actual['execution_metadata']['velocity_graph']['replays'] == expected_nfe
    assert actual['execution_metadata']['cuda_graph'] is True
    assert graphed.model._denoise_action_with_video_cache is original
    assert graphed.model.prefills == graphed.model.encodes == 1
    condition = graphed.pipeline.prepare_condition(_obs(torch))
    raw = graphed.pipeline.evaluate_velocity(noise, 1., condition)
    assert torch.equal(raw, torch.ones_like(noise))
    assert graphed.pipeline._velocity_graph.snapshot()['replays'] == expected_nfe
    saved = actual['actions'].copy()
    graphed.infer(_obs(torch), torch.ones_like(noise))
    np.testing.assert_array_equal(actual['actions'], saved)
    graphed.close(); graphed.close(); assert released == [True]


def test_pipeline_graph_refreshes_changed_context_and_ends_scope_after_bad_input(torch):
    model = _tiny_model(torch)
    model._denoise_action_with_video_cache = lambda **kw: torch.ones_like(kw['latents_action']) * kw['context'].mean()
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, num_inference_steps=2)
    _attach_cpu_graph(frontend, torch)
    obs = _obs(torch); noise = torch.zeros(1, 2, 2)
    first = frontend.infer(obs, noise)
    obs['context'].fill_(3)
    second = frontend.infer(obs, noise)
    np.testing.assert_array_equal(first['actions'], np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(second['actions'], np.full((1, 2, 2), -3))
    assert second['execution_metadata']['velocity_graph']['condition_refreshes'] == 1
    assert model.prefills == 2
    with pytest.raises(ValueError, match='finite'):
        frontend.infer(obs, torch.full_like(noise, float('nan')))
    frontend.infer(obs, noise)  # admission failure did not leave an active request


def test_direct_frontend_close_retries_graph_reset_before_dropping_model(torch):
    frontend = FastWAMTorchFrontend(model=_tiny_model(torch), sigma_shift=1, action_horizon=2)
    model = frontend.model; events = []
    class Graph:
        def close(self):
            events.append('reset')
            if len(events) == 1: raise RuntimeError('reset failed')
    frontend.pipeline._velocity_graph = Graph()
    with pytest.raises(RuntimeError, match='reset failed'): frontend.close()
    assert frontend.model is model and frontend.pipeline.model is model
    with pytest.raises(RuntimeError, match='clos'): frontend.infer(_obs(torch))
    frontend.close(); frontend.close()
    assert events == ['reset', 'reset'] and frontend.model is frontend.pipeline.model is None
    assert model.eval() is model  # externally owned model was not destroyed


def test_pipeline_orders_all_public_operations_on_owned_stream(torch, monkeypatch):
    frontend = FastWAMTorchFrontend(model=_tiny_model(torch), sigma_shift=1, action_horizon=2, num_inference_steps=2)
    pipe = frontend.pipeline; events = []
    class Stream:
        def __init__(self, name): self.name = name
        def wait_stream(self, other): events.append((self.name, 'wait', other.name))
        def synchronize(self): events.append((self.name, 'sync'))
    caller, owned = Stream('caller'), Stream('owned')
    from contextlib import contextmanager
    @contextmanager
    def scope(stream):
        events.append(('enter', stream.name)); yield; events.append(('exit', stream.name))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: caller)
    monkeypatch.setattr(torch.cuda, 'stream', scope)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    pipe._execution_stream = owned
    condition = pipe.prepare_condition(_obs(torch))
    pipe.evaluate_velocity(torch.zeros(1, 2, 2), 1., condition)
    pipe.infer(_obs(torch))
    assert events == [('owned', 'wait', 'caller'), ('enter', 'owned'), ('exit', 'owned'),
                      ('caller', 'wait', 'owned')]*3
    frontend.close()
    assert events[-1] == ('owned', 'sync')


def test_graph_stream_waits_for_construction_stream_before_cross_stream_first_infer(torch, monkeypatch):
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline
    events = []
    origin = SimpleNamespace(name='load-stream')
    owned = SimpleNamespace(wait_stream=lambda source: events.append(source.name))
    monkeypatch.setattr(FastWAMActionPipeline, '_validate_model', lambda self: None)
    monkeypatch.setattr(FastWAMActionPipeline, '_build_time_bank', lambda self: None)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'Stream', lambda device: owned)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: origin)
    model = SimpleNamespace(eval=lambda: None, _denoise_action_with_video_cache=lambda **kw: None)
    pipe = FastWAMActionPipeline(model, action_horizon=2, sigma_shift=1, sampler='euler',
        num_inference_steps=2, probeflow_config=None, native_linear=False,
        device=torch.device('cuda:0'), dtype=torch.bfloat16, use_cuda_graph=True)
    assert pipe._execution_stream is owned and events == ['load-stream']


def test_graph_replay_failure_ends_request_and_rejects_next_prefill(torch):
    model = _tiny_model(torch)
    def fail(**kwargs): raise RuntimeError('head failed')
    model._denoise_action_with_video_cache = fail
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    _attach_cpu_graph(frontend, torch)
    with pytest.raises(RuntimeError, match='head failed'): frontend.infer(_obs(torch))
    assert not frontend.pipeline._velocity_graph._active
    with pytest.raises(RuntimeError, match='failed'): frontend.infer(_obs(torch))
    assert model.prefills == 1
    frontend.close()


def test_eager_velocity_close_drains_caller_stream_and_retries_before_dropping_banks(torch, monkeypatch):
    frontend = FastWAMTorchFrontend(model=_tiny_model(torch), sigma_shift=1, action_horizon=2)
    pipe = frontend.pipeline
    condition = pipe.prepare_condition(_obs(torch))
    bank, model = pipe._time_bank, pipe.model
    events = []
    class Stream:
        cuda_stream = 99
        def wait_stream(self, other): events.append(('wait', other.name))
        def synchronize(self):
            events.append('sync')
            if events.count('sync') == 1: raise RuntimeError('in-flight sync failed')
    stream = Stream()
    load = SimpleNamespace(name='construction-stream')
    pipe.device = torch.device('cuda:0')  # CUDA API boundary only; tensor math remains CPU
    pipe._last_eager_stream = load
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    result = pipe.evaluate_velocity(torch.zeros(1, 2, 2), 1., condition)
    torch.testing.assert_close(result, torch.ones(1, 2, 2))
    assert events == [('wait', 'construction-stream')]
    with pytest.raises(RuntimeError, match='in-flight sync failed'): frontend.close()
    assert pipe._time_bank is bank and pipe.model is model and frontend.model is model
    with pytest.raises(RuntimeError, match='clos'): pipe.evaluate_velocity(torch.zeros(1, 2, 2), 1., condition)
    frontend.close(); frontend.close()
    assert events == [('wait', 'construction-stream'), 'sync', 'sync']
    assert pipe._time_bank is pipe.model is frontend.model is None


def test_euler_preserves_original_shift_five_grid(torch):
    model = _tiny_model(torch)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=5, action_horizon=2, num_inference_steps=2)
    result = frontend.infer(_obs(torch), torch.zeros(1, 2, 2))
    assert np.allclose(result["actions"], -1)
    assert [call[0] for call in model.evaluations] == pytest.approx([1000, 1000 * 5 / 6])


def test_probeflow_uses_real_negative_velocity_and_one_cache(torch):
    model = _tiny_model(torch, proprio_dim=2)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=3, sampler="probeflow")
    result = frontend.infer(_obs(torch, torch.tensor([.1, .2])), torch.zeros(1, 3, 2))
    assert result["actions"].shape == (1, 3, 2)
    assert np.allclose(result["actions"], -1)
    assert [call[0] for call in model.evaluations] == [1000, 500]
    assert torch.all(model.evaluations[1][1] == -.5)
    assert all(call[3:] == (4, torch.Size([3, 5])) for call in model.evaluations)
    assert len({call[2] for call in model.evaluations}) == 1
    assert model.prefills == 1
    assert result["sampler_metadata"]["nfe"] == 2
    assert result["sampler_metadata"]["action_infer_mode"] == "first_frame"


@pytest.mark.parametrize("shift", [1.0, 5.0])
def test_shifted_two_probe_matches_same_scalar_grid_euler(torch, shift):
    def field(state, time):
        return torch.ones_like(state) * (time / 1000)

    model = _tiny_model(torch, field=field)
    euler_model = _tiny_model(torch, field=field)
    adaptive = FastWAMTorchFrontend(model=model, sigma_shift=shift, action_horizon=2,
                                   sampler="probeflow")
    fixed = FastWAMTorchFrontend(model=euler_model, sigma_shift=shift, action_horizon=2,
                                num_inference_steps=2)
    noise = torch.zeros(1, 2, 2)
    result = adaptive.infer(_obs(torch), noise)
    baseline = fixed.infer(_obs(torch), noise)
    midpoint = shifted_sigma(.5, shift)
    assert np.allclose(result["actions"], baseline["actions"], rtol=1e-6)
    assert torch.allclose(model.evaluations[1][1], torch.full_like(noise, -(1 - midpoint)))
    assert [call[0] for call in model.evaluations] == [call[0] for call in euler_model.evaluations]
    assert result["sampler_metadata"]["timesteps"] == [1, .5]
    assert result["sampler_metadata"]["integration_times"] == pytest.approx([1, midpoint])
    assert result["sampler_metadata"]["time_grid"] == "mapped"
    if shift != 1:
        assert result["sampler_metadata"]["adaptation"] == "shifted_sigma_grid_not_original_paper_validation"


def test_shift_five_dense_probe_restart_uses_physical_sigma_deltas(torch):
    def field(state, time):
        return torch.ones_like(state) * (1 if time.item() > 900 else -1)

    model = _tiny_model(torch, field=field)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=5, action_horizon=2,
                                   sampler="probeflow",
                                   probeflow_config=ProbeFlowConfig(epsilon=.1, max_steps=4))
    result = frontend.infer(_obs(torch), torch.zeros(1, 2, 2))
    assert result["sampler_metadata"]["nfe"] == 5
    assert result["sampler_metadata"]["integration_times"] == pytest.approx([1, 5 / 6, 15 / 16, 5 / 6, 5 / 8])
    assert torch.allclose(model.evaluations[1][1], torch.full((1, 2, 2), -1 / 6))
    assert torch.all(model.evaluations[2][1] == -1 / 16)
    assert np.allclose(result["actions"], 2 / 3)


def test_dense_probeflow_counts_discarded_probe_and_restarts(torch):
    def field(state, time):
        return torch.ones_like(state) * (1 if time.item() > 500 else -1)

    model = _tiny_model(torch, field=field)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, sampler="probeflow",
                                   probeflow_config=ProbeFlowConfig(epsilon=.1, max_steps=4))
    result = frontend.infer(_obs(torch), torch.zeros(1, 2, 2))
    assert result["sampler_metadata"]["nfe"] == 5
    assert result["sampler_metadata"]["scheduled_steps"] == 4
    assert [call[0] for call in model.evaluations] == [1000, 500, 750, 500, 250]
    assert torch.all(model.evaluations[2][1] == -.25)
    assert np.allclose(result["actions"], 0)


def test_velocity_does_not_integrate_or_mutate_state(torch):
    model = _tiny_model(torch)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, num_inference_steps=8)
    condition = frontend.pipeline.prepare_condition(_obs(torch))
    state = torch.zeros(1, 2, 2)
    velocity = frontend.pipeline.evaluate_velocity(state, .125, condition)
    assert torch.equal(state, torch.zeros_like(state))
    assert torch.equal(velocity, torch.ones_like(state))
    assert model.evaluations[-1][0] == 125


def test_observation_failures_precede_vae_encoding(torch):
    model = _tiny_model(torch, proprio_dim=2)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    with pytest.raises(ValueError, match="proprio is required"):
        frontend.infer(_obs(torch), torch.zeros(1, 2, 2))
    obs = _obs(torch, torch.zeros(2))
    obs["input_image"] = torch.full((3, 32, 64), 255.)
    with pytest.raises(ValueError, match="normalized"):
        frontend.infer(obs, torch.zeros(1, 2, 2))
    obs = _obs(torch, torch.zeros(2))
    obs["prompt"] = "already have context"
    with pytest.raises(ValueError, match="mutually exclusive"):
        frontend.infer(obs, torch.zeros(1, 2, 2))
    assert model.encodes == 0 and model.prefills == 0


def test_implicit_cuda_device_is_pinned_before_ambient_device_changes(torch, monkeypatch):
    """Fake parameters/validation only; never allocates a CUDA tensor."""
    pipeline_module = importlib.import_module("flash_rt.models.fastwam.pipeline")
    real_pipeline = pipeline_module.FastWAMActionPipeline
    current = [3]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    model = _tiny_model(torch)
    model.device = torch.device("cuda")
    parameter = SimpleNamespace(device=torch.device("cuda:3"), dtype=torch.float32,
                                is_floating_point=lambda: True)
    model.named_parameters = lambda: [("actual_weight", parameter)]

    def no_move(*args, **kwargs):
        raise AssertionError("Injected model weights were silently moved")

    model.to = no_move

    class ValidationOnlyPipeline:
        def __init__(self, model, **kwargs):
            validation = object.__new__(real_pipeline)
            validation.model, validation.action_horizon = model, kwargs["action_horizon"]
            validation.device, validation.dtype = kwargs["device"], kwargs["dtype"]
            validation._validate_model()  # host attributes only; skip time-bank upload
            self.device = validation.device

    monkeypatch.setattr(pipeline_module, "FastWAMActionPipeline", ValidationOnlyPipeline)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, device="cuda")
    current[0] = 5
    assert frontend.device == torch.device("cuda:3")
    assert frontend.pipeline.device == torch.device("cuda:3")
    assert model.device == torch.device("cuda:3")
    assert model.named_parameters()[0][1] is parameter
    assert parameter.device == torch.device("cuda:3")


@pytest.mark.parametrize("mismatch", ["model_attribute", "component_attribute", "parameter", "buffer"])
def test_fixed_cuda_device_rejects_actual_tensor_and_declared_device_mismatches(torch, mismatch):
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline

    model = _tiny_model(torch)
    model.device = torch.device("cuda:2")
    target = object.__new__(FastWAMActionPipeline)
    target.model, target.action_horizon = model, 2
    target.device, target.dtype = torch.device("cuda:2"), torch.float32
    if mismatch == "model_attribute":
        model.device = torch.device("cuda:1")
    elif mismatch == "component_attribute":
        model.vae.device = torch.device("cuda:1")
    elif mismatch == "parameter":
        model.named_parameters = lambda: [("weight", SimpleNamespace(device=torch.device("cuda:1"),
                                                                    dtype=torch.float32,
                                                                    is_floating_point=lambda: True))]
    else:
        model.named_buffers = lambda: [("scale", SimpleNamespace(device=torch.device("cuda:1")))]
    with pytest.raises(ValueError, match="device"):
        target._validate_model()


@pytest.mark.parametrize("size", [16, 48, 240])
def test_vae38_patch_geometry_rejects_non_32_aligned_images_before_encode(torch, size):
    model = _tiny_model(torch)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    obs = _obs(torch)
    obs["input_image"] = torch.zeros(3, size, size)
    with pytest.raises(ValueError, match="actual VAE factor"):
        frontend.pipeline.prepare_condition(obs)
    assert model.encodes == 0 and model.prepares == 0


@pytest.mark.parametrize("size", [32, 224])
def test_vae38_patch_geometry_accepts_valid_spatial_boundaries(torch, size):
    model = _tiny_model(torch)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    obs = _obs(torch)
    obs["input_image"] = torch.zeros(3, size, size)
    frontend.pipeline.prepare_condition(obs)
    assert frontend.pipeline.image_multiple == (32, 32)
    assert model.encodes == 1 and model.prepares == 1


@pytest.mark.parametrize("factor,inner,expected", [
    (16, False, (32, 32)), ((16, 8), False, (32, 16)), ((4, 16, 8), True, (32, 16)),
])
def test_vae_factor_is_read_from_real_wrapper_or_inner_declaration(torch, factor, inner, expected):
    model = _tiny_model(torch, vae_factor=factor, factor_on_inner=inner)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    assert frontend.pipeline.image_multiple == expected
    frontend.pipeline.prepare_condition(_obs(torch))
    assert model.prepares == 1


@pytest.mark.parametrize("factor", [None, True, 0, -1, (16,), (4, 16, 16, 16)])
def test_unknown_or_invalid_vae_factor_is_not_guessed(factor):
    with pytest.raises(ValueError, match="upsampling_factor"):
        vae_spatial_factor(SimpleNamespace(upsampling_factor=factor, model=SimpleNamespace()))


def test_latent_patch_divisibility_is_checked_before_video_prepare(torch):
    model = _tiny_model(torch)
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    model._encode_input_image_latents_tensor = lambda **kwargs: torch.zeros(1, 48, 1, 3, 4)
    with pytest.raises(ValueError, match="latent dimensions must be divisible"):
        frontend.pipeline.prepare_condition(_obs(torch))
    assert model.prepares == 0


@pytest.mark.parametrize("sampler", ["euler", "probeflow"])
def test_banked_velocity_hotpath_has_no_scalar_reads_or_timestep_allocations(torch, monkeypatch, sampler):
    model = _tiny_model(torch)
    kwargs = {"num_inference_steps": 8} if sampler == "euler" else {
        "probeflow_config": ProbeFlowConfig(max_steps=4)
    }
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=5, action_horizon=2,
                                   sampler=sampler, **kwargs)
    pipe = frontend.pipeline
    condition = pipe.prepare_condition(_obs(torch))
    state = torch.zeros(1, 2, 2)
    seen = []

    def field(**values):
        seen.append(id(values["timestep_action"]))
        return torch.ones_like(values["latents_action"])

    model._denoise_action_with_video_cache = field

    def forbidden(*args, **kwargs):
        raise AssertionError("Per-NFE host read or variable timestep allocation")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        patch.setattr(torch, "tensor", forbidden)
        patch.setattr(torch, "full", forbidden)
        for clock in (*pipe._clock_keys, *pipe._clock_keys):
            pipe._evaluate_clock_velocity(state, clock, condition)
            pipe.evaluate_velocity(state, pipe._sigma_by_clock[clock], condition)
        if sampler == "euler":
            _, metadata = pipe._euler(state, condition)
    width = 2 * len(pipe._clock_keys)
    assert seen[:width] == seen[width:2 * width]
    if sampler == "euler":
        assert metadata["model_timesteps"] == [pipe._model_time_by_clock[t] for t in pipe._clock_keys]


def test_timebank_rejects_unrepresentable_positive_time_before_upload(torch, monkeypatch):
    model = _tiny_model(torch)
    model.torch_dtype = torch.float16

    def no_upload(*args, **kwargs):
        raise AssertionError("Invalid CPU timestep bank reached a device upload")

    monkeypatch.setattr(torch.Tensor, "to", no_upload)
    with pytest.raises(ValueError, match="representable"):
        FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, sampler="probeflow",
                             probeflow_config=ProbeFlowConfig(probe_horizon=1 - 2**-52))
    assert model.encodes == 0 and not model.evaluations


def test_request_end_nonfinite_action_guard_is_retained(torch):
    model = _tiny_model(torch, field=lambda state, time: torch.full_like(state, float("nan")))
    frontend = FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2, num_inference_steps=2)
    with pytest.raises(ValueError, match="integration produced nonfinite actions"):
        frontend.infer(_obs(torch), torch.zeros(1, 2, 2))
    assert len(model.evaluations) == 2  # checked once at request end, not per NFE


def test_native_install_preserves_aliases_parameters_and_state_dict(torch, monkeypatch):
    native = importlib.import_module("flash_rt.models.fastwam.native_linear")

    class FakeNativeLinear(torch.nn.Module):
        def __init__(self, linear, runner):
            super().__init__()
            self.weight, self.bias = linear.weight, linear.bias
            self.in_features, self.out_features = linear.in_features, linear.out_features
            self._runner = runner

    monkeypatch.setattr(native, "FlashRTLinear", FakeNativeLinear)
    action = torch.nn.Module()
    action.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 6))])
    action.action_encoder = torch.nn.Linear(2, 4)
    action.head = torch.nn.Linear(4, 2)
    video = torch.nn.Module()
    video.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 6))])
    video.head = torch.nn.Linear(4, 2)  # unused by first-frame video prefill
    model = SimpleNamespace(action_expert=action, video_expert=video,
                            mot=SimpleNamespace(mixtures={"action": action, "video": video}))
    weight = action.blocks[0][0].weight
    before = {key: value.clone() for key, value in action.state_dict().items()}
    installed, skipped = native.install_native_linears(model, runner=object())
    assert len(installed) == 4 and skipped == ()
    assert model.mot.mixtures["action"].blocks[0][0].weight is weight
    assert isinstance(video.head, torch.nn.Linear)
    assert set(before) == set(action.state_dict())
    assert all(torch.equal(before[key], value) for key, value in action.state_dict().items())


def test_native_forward_uses_nk_weight_layout_and_actual_stream(torch, monkeypatch):
    native = importlib.import_module("flash_rt.models.fastwam.native_linear")
    records, pointers = [], []

    class Tensor:
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        def __init__(self, shape, pointer):
            self.shape, self.pointer = tuple(shape), pointer
            self.ndim = len(self.shape)

        def is_floating_point(self):
            return True

        def to(self, **kwargs):
            assert kwargs == {"dtype": torch.bfloat16}
            return self

        def contiguous(self):
            return self

        def reshape(self, *shape):
            if shape == (-1, 4):
                return Tensor((6, 4), self.pointer)
            return Tensor(shape[0], self.pointer)

        def data_ptr(self):
            return self.pointer

        def record_stream(self, stream):
            pointers.append((self.pointer, stream.cuda_stream))

    runner = SimpleNamespace(bf16_nt_bias=lambda *args: records.append(args))
    module = SimpleNamespace(training=False, in_features=4, out_features=8,
                             weight=Tensor((8, 4), 22), bias=Tensor((8,), 33), _runner=runner, calls=0)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: Tensor(shape, 44))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=99))
    output = native.FlashRTLinear.forward(module, Tensor((2, 3, 4), 11))
    assert records == [(11, 22, 44, 33, 6, 8, 4, 99)]
    assert output.shape == (2, 3, 8)
    assert pointers == [(11, 99), (22, 99), (33, 99), (44, 99)]
    assert module.calls == 1
