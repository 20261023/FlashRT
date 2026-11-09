#!/usr/bin/env python3
"""Allocation-only cached BF16 GEMM smoke; no model, weights or hub loading.

Checks real cuBLASLt math against torch.nn.functional.linear at FastWAM shapes,
changed bias/weight pointers, same-shape cache reuse and a real 64-bit CUDA
stream. Optional host enqueue timings are NOT GPU or end-to-end speedups.
No Torch or extension imports occur until the Slurm compute/GPU guard passes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time


SHAPES = (
    (1, 1024, 256),          # action time embedding
    (1, 6144, 1024),         # action time projection
    (32, 1024, 7),           # normalized action encoder, small odd K
    (32, 7, 1024),           # normalized action head, small odd N
    (32, 3072, 1024),        # action Q/K/V width != action hidden width
    (32, 4096, 1024),        # action FFN up
    (32, 1024, 4096),        # action FFN down
    (98, 3072, 3072),        # first-frame two-camera video attention
    (98, 14336, 3072),       # video FFN up
    (98, 3072, 14336),       # video FFN down
    (98, 18432, 3072),       # separated video timestep modulation
    (129, 3072, 4096),       # text/proprio context projection
    (129, 1024, 4096),       # action text/proprio context projection
)


def require_allocation(environ=None, hostname=None):
    env = os.environ if environ is None else environ
    host = (hostname or socket.gethostname()).split(".")[0]
    node = env.get("SLURMD_NODENAME", "").split(".")[0]
    job = env.get("SLURM_JOB_ID", "")
    if (not job.isdigit() or not node or host != node
            or host.lower().startswith(("ln", "login"))):
        raise RuntimeError("Requires the main Slurm allocation's compute host; never execute GPU smoke on login")
    ids = env.get("SLURM_STEP_GPUS") or env.get("SLURM_JOB_GPUS")
    count = env.get("SLURM_GPUS_ON_NODE", "")
    if not ids and not (count.isdigit() and int(count) > 0):
        raise RuntimeError("Requires Slurm GPU-allocation metadata")
    return {"job_id": job, "node": node, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")}


def metrics(actual, reference):
    error = actual.float() - reference.float()
    return {"max_abs": error.abs().max().item(),
            "mean_relative": (error.abs() / (reference.float().abs() + 1e-3)).mean().item(),
            "relative_l2": (error.norm() / reference.float().norm().clamp_min(1e-20)).item()}


def exercise(args, allocation):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Slurm allocated process has no CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Allocated GPU does not support BF16")
    cached_module = importlib.import_module("flash_rt.flash_rt_gemm_plan")
    runners = {"cached": cached_module.CachedGemmRunner()}
    if args.include_legacy:
        legacy = importlib.import_module("flash_rt.flash_rt_kernels")
        runners["legacy"] = legacy.GemmRunner()
    stream = torch.cuda.Stream(device=device)
    if int(stream.cuda_stream) <= 0xFFFFFFFF:
        raise RuntimeError("Actual CUDA stream did not exercise upper 32 pointer bits; high-64-bit coverage is incomplete")
    results = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for m, n, k in SHAPES:
            x = torch.randn((m, k), device=device, generator=generator).to(torch.bfloat16)
            w = (torch.randn((n, k), device=device, generator=generator) / math.sqrt(k)).to(torch.bfloat16)
            bias = (torch.randn((n,), device=device, generator=generator) * .1).to(torch.bfloat16)
            changed_bias = (bias.float() + .5).to(torch.bfloat16)
            changed_weight = (w.float() * .75).to(torch.bfloat16)
            changed_input = (x.float() - .25).to(torch.bfloat16)
            for backend, runner in runners.items():
                before = dict(runner.stats()) if backend == "cached" else None
                records = []
                after_first = None
                for index, (activation, weight, current_bias) in enumerate((
                    (x, w, bias), (x, w, changed_bias),
                    (changed_input, changed_weight, changed_bias), (x, w, bias),
                )):
                    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
                    runner.bf16_nt_bias(activation.data_ptr(), weight.data_ptr(), output.data_ptr(),
                                        current_bias.data_ptr(), m, n, k, int(stream.cuda_stream))
                    reference = F.linear(activation, weight, current_bias)
                    stream.synchronize()
                    error = metrics(output, reference)
                    if not torch.isfinite(output).all().item() or error["mean_relative"] >= args.relative_tolerance:
                        raise AssertionError(f"{backend} M/N/K={m,n,k}, call={index}: {error}")
                    records.append(error)
                    if backend == "cached":
                        now = dict(runner.stats())
                        if index == 0:
                            after_first = now
                            if now["plan_builds"] != before["plan_builds"] + 1 or now["heuristic_queries"] != before["heuristic_queries"] + 1:
                                raise AssertionError("New representative shape did not build/query exactly one cached plan")
                        elif now["heuristic_queries"] != after_first["heuristic_queries"] or now["plan_builds"] != after_first["plan_builds"]:
                            raise AssertionError("Changing bias/weight/input pointers rebuilt a same-shape plan")
                        if now["launches"] != before["launches"] + index + 1:
                            raise AssertionError("Cached successful-enqueue count disagrees with actual calls")
                item = {"backend": backend, "M": m, "N": n, "K": k, "errors": records,
                        "new_bias_pointer": bias.data_ptr() != changed_bias.data_ptr(),
                        "new_weight_pointer": w.data_ptr() != changed_weight.data_ptr()}
                if args.host_repeats:
                    samples = []
                    for _ in range(args.host_repeats):
                        stream.synchronize()
                        start = time.perf_counter_ns()
                        runner.bf16_nt_bias(x.data_ptr(), w.data_ptr(), output.data_ptr(), bias.data_ptr(),
                                            m, n, k, int(stream.cuda_stream))
                        samples.append((time.perf_counter_ns() - start) / 1e3)
                        stream.synchronize()
                    item["host_enqueue_us_samples"] = samples
                    item["host_enqueue_us_median"] = statistics.median(samples)
                    item["timing_scope"] = "host_Python_plus_enqueue_only_not_GPU_latency_or_model_speedup"
                if backend == "cached":
                    after = dict(runner.stats())
                    item["native_plan_stats"] = {key: after[key] - before[key] for key in before}
                results.append(item)
    stream.synchronize()
    return {"status": "passed", "allocation": allocation, "shapes": results,
            "stream_pointer": int(stream.cuda_stream), "stream_upper_32_bits_exercised": True,
            "relative_tolerance": args.relative_tolerance,
            "tolerance_source": "tests/test_bf16_nt_bias.py mean_relative < 1e-2, engineering GEMM check only",
            "dtype": "bfloat16", "workspace_bytes_cached": 64 * 1024 * 1024,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "model_speedup_claimed": False, "task_success_evaluated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON report path; never overwrite an earlier result")
    parser.add_argument("--include-legacy", action="store_true", help="Also exercise unchanged legacy GEMM in the same allocation")
    parser.add_argument("--host-repeats", type=int, default=0, help="Optional per-shape host enqueue samples, 0 disables")
    parser.add_argument("--relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args(argv)
    try:
        allocation = require_allocation()
        if not 0 <= args.host_repeats <= 1000:
            raise ValueError("host-repeats must be in [0,1000]")
        if not math.isfinite(args.relative_tolerance) or args.relative_tolerance <= 0:
            raise ValueError("relative-tolerance must be positive and finite")
        if args.output and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        report = exercise(args, allocation)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as exc:
        parser.exit(1, f"GEMM plan smoke failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
