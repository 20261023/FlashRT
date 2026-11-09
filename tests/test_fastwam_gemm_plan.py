"""CPU boundary tests for explicit cached GEMM selection and request accounting.

No compiled extension, real model, CUDA query or tensor is used. C++ numerical,
bias-pointer and stream correctness are tested by the allocation-only smoke,
not by asserting source strings or claiming these doubles validate cuBLASLt.
"""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("backend,factory_name", [("legacy", "GemmRunner"), ("cached", "CachedGemmRunner")])
def test_explicit_backend_selects_only_its_runner(monkeypatch, backend, factory_name):
    native = importlib.import_module("flash_rt.models.fastwam.native_linear")
    imported = []
    runner = SimpleNamespace(bf16_nt_bias=lambda *args: None, stats=lambda: {
        "plan_builds": 0, "heuristic_queries": 0, "launches": 0,
    })

    def selected_module(name):
        imported.append(name)
        return SimpleNamespace(**{factory_name: lambda: runner})

    monkeypatch.setattr(native, "importlib", SimpleNamespace(import_module=selected_module), raising=False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    result = native.create_native_runner(backend, torch.device("cuda:4"))
    assert result is runner
    assert imported == ["flash_rt.flash_rt_gemm_plan" if backend == "cached" else "flash_rt.flash_rt_kernels"]


def test_missing_cached_extension_fails_without_legacy_fallback(monkeypatch):
    native = importlib.import_module("flash_rt.models.fastwam.native_linear")
    attempts = []

    def unavailable(name):
        attempts.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(native, "importlib", SimpleNamespace(import_module=unavailable), raising=False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    with pytest.raises(ImportError, match="build_gemm_plan.py"):
        native.create_native_runner("cached", torch.device("cuda:0"))
    assert attempts == ["flash_rt.flash_rt_gemm_plan"]


@pytest.mark.parametrize("kwargs", [
    {"native_linear_backend": "other"},
    {"native_linear_backend": "cached", "native_linear": False},
])
def test_invalid_cached_options_fail_before_config_files_or_cuda(monkeypatch, kwargs):
    frontend = importlib.import_module("flash_rt.frontends.torch.fastwam")

    def forbidden(*args, **kw):
        raise AssertionError("invalid cached options reached files/device")

    monkeypatch.setattr(frontend, "read_model_config", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    with pytest.raises(ValueError, match="native_linear"):
        frontend.FastWAMTorchFrontend("/unread", sigma_shift=5, action_horizon=32, **kwargs)


def test_legacy_remains_frontend_default(monkeypatch):
    frontend = importlib.import_module("flash_rt.frontends.torch.fastwam")
    pipeline_module = importlib.import_module("flash_rt.models.fastwam.pipeline")
    captured = {}

    class BoundaryPipeline:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pipeline_module, "FastWAMActionPipeline", BoundaryPipeline)
    model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32)
    frontend.FastWAMTorchFrontend(model=model, sigma_shift=1, action_horizon=2)
    assert captured["native_linear_backend"] == "legacy"


def test_request_metadata_uses_plan_stat_deltas_not_cumulative_identity():
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline
    import threading

    values = {"plan_builds": 9, "heuristic_queries": 9, "launches": 100}
    reads = []

    def stats():
        reads.append(dict(values))
        return dict(values)

    runner = SimpleNamespace(stats=stats)
    module = SimpleNamespace(calls=10, _runner=runner)
    pipe = object.__new__(FastWAMActionPipeline)
    pipe._lock = threading.RLock()
    pipe._native_modules, pipe._native_skipped = {"action_expert.head": module}, ()
    pipe._native_runner = runner
    pipe.native_linear_backend = "cached"
    pipe.device, pipe.image_multiple = torch.device("cpu"), (32, 32)
    pipe.sampler, pipe.sigma_shift, pipe.time_scale = "euler", 5.0, 1000
    pipe._initial_state = lambda noise: torch.zeros(1, 2, 2)
    pipe.prepare_condition = lambda obs: None

    def euler(state, condition):
        if values["launches"] == 100:
            values["plan_builds"] += 1
            values["heuristic_queries"] += 1
        values["launches"] += 3
        module.calls += 3
        return state, {"sampler": "euler", "nfe": 2}

    pipe._euler = euler
    identity_before = pipe.execution_metadata()
    first = pipe.infer({})["execution_metadata"]
    second = pipe.infer({})["execution_metadata"]
    assert first["native_plan_stats"] == {"plan_builds": 1, "heuristic_queries": 1, "launches": 3}
    assert second["native_plan_stats"] == {"plan_builds": 0, "heuristic_queries": 0, "launches": 3}
    assert first["native_linear_calls"] == {"action_expert.head": 3}
    assert len(reads) == 4  # once before/after each request, never per Linear
    assert pipe.execution_metadata() == identity_before
    assert identity_before["native_linear_backend"] == "cached"
    assert "native_plan_stats" not in identity_before
    assert identity_before["fully_native"] is False


def test_pipeline_passes_cached_selection_and_strongly_owns_shared_runner(monkeypatch):
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline
    native = importlib.import_module("flash_rt.models.fastwam.native_linear")
    runner = SimpleNamespace(stats=lambda: {"plan_builds": 0, "heuristic_queries": 0, "launches": 0})
    module = SimpleNamespace(_runner=runner, calls=0)
    model = SimpleNamespace(eval=lambda: None)
    selected = []

    def install(actual_model, *, backend):
        assert actual_model is model
        selected.append(backend)
        return {"action_expert.head": module}, ()

    def validate(self):
        self.image_multiple = (32, 32)

    monkeypatch.setattr(native, "install_native_linears", install)
    monkeypatch.setattr(FastWAMActionPipeline, "_validate_model", validate)
    monkeypatch.setattr(FastWAMActionPipeline, "_build_time_bank", lambda self: None)
    pipe = FastWAMActionPipeline(model, action_horizon=2, sigma_shift=5, sampler="euler",
                                num_inference_steps=2, probeflow_config=None, native_linear=True,
                                device=torch.device("cpu"), dtype=torch.bfloat16,
                                native_linear_backend="cached")
    assert selected == ["cached"]
    assert pipe._native_runner is runner
    assert pipe._native_modules["action_expert.head"]._runner is runner
    assert pipe.execution_metadata()["native_linear_backend"] == "cached"


def test_legacy_request_never_asks_for_cached_stats():
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline

    def forbidden():
        raise AssertionError("legacy mode accessed cached-only stats")

    pipe = object.__new__(FastWAMActionPipeline)
    pipe.native_linear_backend = "legacy"
    pipe._native_runner = SimpleNamespace(stats=forbidden)
    assert pipe._native_plan_snapshot() is None


def test_cpu_stats_contract_rejects_missing_or_invalid_counters():
    from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline

    pipe = object.__new__(FastWAMActionPipeline)
    pipe.native_linear_backend = "cached"
    for bad in ({}, {"plan_builds": 0, "heuristic_queries": -1, "launches": 2},
                {"plan_builds": 0, "heuristic_queries": 0, "launches": True}):
        pipe._native_runner = SimpleNamespace(stats=lambda: bad)
        with pytest.raises(ValueError, match="counter"):
            pipe._native_plan_snapshot()


def test_build_dry_run_uses_host_cpp_and_target_package_only(tmp_path):
    """Exercise CLI output against directory fixtures, never execute a compiler."""
    root = Path(__file__).resolve().parents[1]
    cuda, headers = tmp_path / "cuda", tmp_path / "torch-include"
    (cuda / "include").mkdir(parents=True)
    (cuda / "lib64").mkdir()
    (headers / "pybind11").mkdir(parents=True)
    for path in (cuda / "include/cublasLt.h", cuda / "include/cuda_runtime_api.h",
                 headers / "pybind11/pybind11.h"):
        path.touch()
    output = tmp_path / "isolated-code/flash_rt"
    result = subprocess.run([
        sys.executable, "-B", str(root / "scripts/build_gemm_plan.py"),
        "--cuda-home", str(cuda), "--torch-include", str(headers),
        "--output-dir", str(output), "--cxx", "/not-executed/g++", "--dry-run",
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["command"][0] == "/not-executed/g++"
    assert "-lcublasLt" in plan["command"] and "-lcudart" in plan["command"]
    assert "-std=c++17" in plan["command"]
    assert str(root / "csrc/gemm/bf16_nt_plan.cpp") in plan["command"]
    assert Path(plan["output"]).parent == output
    assert Path(plan["output"]).name.startswith("flash_rt_gemm_plan.")
    assert not output.exists()  # dry-run does not create binaries/directories


def test_smoke_rejects_login_before_any_torch_import(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("SLURM"):
            env.pop(name)
    result = subprocess.run([sys.executable, "-B", str(root / "examples/gemm_plan_smoke.py")],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Slurm" in result.stderr


def test_default_build_header_discovery_never_executes_torch(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    cuda, package = tmp_path / "cuda", tmp_path / "packages/torch"
    (cuda / "include").mkdir(parents=True)
    (cuda / "lib64").mkdir()
    (package / "include/pybind11").mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('Torch package must not execute during header discovery')\n")
    for path in (cuda / "include/cublasLt.h", cuda / "include/cuda_runtime_api.h",
                 package / "include/pybind11/pybind11.h"):
        path.touch()
    result = subprocess.run([
        sys.executable, "-B", str(root / "scripts/build_gemm_plan.py"),
        "--cuda-home", str(cuda), "--output-dir", str(tmp_path / "code/flash_rt"), "--dry-run",
    ], text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(package.parent)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["torch_headers_only"] == str(package / "include")
