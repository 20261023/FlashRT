"""Time the fixed-revision author's DM0.5 API, not a FlashRT implementation.

Model work is restricted to a Slurm GPU step. No tests are invoked. This script
keeps decoded CPU actions, prepared input tensors and exact initial noise private.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
from pathlib import Path
import statistics
import sys
import time


SOURCE_REVISION = "03f799120d425ead51d9482bb1da950b6dabade9"
WEIGHT_REVISION = "5cd18734814abb075a9ccfd9ad6d16777b5cf10e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    node = os.environ.get('SLURMD_NODENAME') or os.environ.get('SLURM_JOB_NODELIST', '')
    if (not os.environ.get('SLURM_JOB_ID', '').isdigit() or host.startswith(('ln', 'login'))
            or host != node.split('.')[0] or os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1')):
        raise RuntimeError("Model computation requires an allocated Slurm GPU compute node")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from opendm.exp.dm05_exp import DM05Exp

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    torch.set_num_threads(2)
    exp = DM05Exp(task="inference", logger_level="WARNING")
    exp.model_config.model_name_or_path = str(args.checkpoint.resolve())
    exp.model_config.vision_attn_implementation = "sdpa"
    exp.model_config.liger_kernel = False
    exp.model_config.chunk_size = 50
    exp.inference_config.backend = "default"
    exp.inference_config.diffusion_steps = 10
    exp.inference_config.enable_vlm_chat = False
    exp.data_config.is_history = False
    t0 = time.perf_counter()
    exp._initialize_inference_runtime()
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t0
    policy = exp.inference_config
    model = policy.model
    images = [Image.open(args.source / "assets/demo/images/episode0" / c / "0.jpg").convert("RGB")
              for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    original_state = [0.0] * 14
    original_state[6] = original_state[13] = 1.0
    changed_state = original_state.copy()
    changed_state[0], changed_state[7] = 0.05, -0.05
    prompts = ["Hold the roller for smoothing materials with both arms to pick it up",
               "Pick up the roller using both arms"]
    cases = [
        {"id": "demo-seed0", "seed": 0, "state": original_state, "prompt": prompts[0]},
        {"id": "demo-seed17", "seed": 17, "state": original_state, "prompt": prompts[0]},
        {"id": "state-seed0", "seed": 0, "state": changed_state, "prompt": prompts[0]},
        {"id": "prompt-seed0", "seed": 0, "state": original_state, "prompt": prompts[1]},
    ]
    records, case_metadata = [], []
    for case in cases:
        target = args.output / case["id"]
        target.mkdir()
        os.environ["DM05_INFERENCE_SEED"] = str(case["seed"])

        def request():
            data = policy._prepare_model_input(
                text=case["prompt"], images=images, states=case["state"], robot_type="DOS W1")
            return policy._predict(data)

        cold_t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.warmup):
                request()
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - cold_t0
        actions, samples, core_samples = [], [], []
        with torch.inference_mode():
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                action = request()
                # The original author's _predict has already transferred the
                # full action block to CPU and denormalized/reconstructed it.
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if action.shape != (50, 14) or not np.isfinite(action).all():
                    raise RuntimeError(f"Invalid full CPU action output: {action.shape}")
                actions.append(np.array(action, copy=True))
                samples.append(elapsed)
                core_samples.append(policy.last_model_latency_sec * 1000)
                records.append({"case": case["id"], "repeat": repeat,
                                "full_policy_ms": elapsed, "author_model_ms": core_samples[-1]})
        np.save(target / "actions.npy", np.stack(actions))
        np.save(target / "samples_ms.npy", samples)
        # Reference capture is outside the steady-state samples. Preserve the
        # author's prepared tensors and RNG contract for a later native port.
        data = policy._prepare_model_input(
            text=case["prompt"], images=images, states=case["state"], robot_type="DOS W1")
        prepared = policy.input_transform(data)
        torch.save({k: v.detach().cpu() for k, v in prepared.items() if isinstance(v, torch.Tensor)},
                   target / "prepared.pt")
        torch.manual_seed(case["seed"])
        torch.cuda.manual_seed_all(case["seed"])
        with torch.inference_mode():
            noise = torch.randn(1, 50, 32, dtype=model.model.action_in_proj.weight.dtype,
                                device=policy.device)
            torch.save(noise.cpu(), target / "initial_noise.pt")
        case_metadata.append({**case, "prefix_length": prepared["input_ids"].shape[1],
                              "pixel_values_shape": list(prepared["pixel_values"].shape),
                              "warmup_s": warmup_s,
                              "full_policy_median_ms": statistics.median(samples),
                              "author_model_median_ms": statistics.median(core_samples)})
        print(json.dumps(case_metadata[-1]), flush=True)

    timings = [r["full_policy_ms"] for r in records]
    core = [r["author_model_ms"] for r in records]
    report = {
        "model": "dm05", "variant": "Dexmal/DM05 base", "origin": "author",
        "execution": "official default backend; vision SDPA; no Liger",
        "source_revision": SOURCE_REVISION, "checkpoint_revision": WEIGHT_REVISION,
        "source": str(args.source.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(), "job_id": os.environ["SLURM_JOB_ID"],
        "dtype": "BF16 mixed (author bf16_mixed)", "steps": 10, "nfe": 10,
        "batch": 1, "cameras": 3, "history": 0, "chunk": 50,
        "output_action_dim": 14, "internal_action_dim": 32, "image_size": [448, 448],
        "boundary": "host PIL RGB+state+prompt -> full denormalized absolute FP32 CPU action; excludes file decode/network",
        "inner_boundary": "author _predict last_model_latency_sec: prepared GPU tensors -> synchronized GPU normalized actions",
        "warmup_per_case": args.warmup, "samples_per_case": args.repeats,
        "setup_s": setup_s, "full_policy_median_ms": statistics.median(timings),
        "full_policy_p95_ms": float(np.percentile(timings, 95)),
        "author_model_median_ms": statistics.median(core),
        "cases": case_metadata, "samples": records,
        "suffix_graph_profiles": len(model._suffix_graph_profiles),
        "suffix_graph_disabled_profiles": len(model._suffix_graph_disabled_profiles),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "torchvision", "transformers", "tokenizers", "peft", "numpy")},
        "tests_run": False, "closed_loop_success": "NR",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("full_policy_median_ms", "full_policy_p95_ms",
                                            "author_model_median_ms", "setup_s",
                                            "suffix_graph_profiles", "suffix_graph_disabled_profiles")}), flush=True)


if __name__ == "__main__":
    main()
