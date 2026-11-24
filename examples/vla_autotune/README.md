# VLA experiment and validation examples

These are source-checkout examples above `flash_rt.vla`, not a claim that every
model, hardware path or numerical profile is qualified. Keep model weights,
fixtures and generated outputs outside the source tree. Experiments must use
an immutable source snapshot and explicitly named local inputs/checkpoints.

## Selected methods versus author Native latency

`latency_comparison.py` validates evidence labels and statistics for the model
latency comparison. **Official Native means the author's original execution
stack**. Published numbers with a different GPU/input/sampling/statistical/timing
protocol are context-only; the helper refuses to manufacture a speedup ratio
from them. Missing values are NR, not zero latency. Read the dated
[experiment notes](../../docs/vla_autotune_experiments.md) and
[reports](../../docs/reports) for measured selections and their input scope.

`official_hyvla_latency.py` is an allocation-only timing wrapper around the
unmodified author `HyVLA` APIs: original `prepare_images/state/language` followed
by original `policy.model.sample_actions` with fixed noise, then completed CPU
raw action output. Author code is exported from a fixed Git revision and import
paths/load completeness are checked. One untimed NFE control restores its
observer before formal warmups/samples. This is an official-implementation
self-test with an explicit host-to-CPU boundary. Cold load and warmup remain
separate; raw arrays are written before a completed JSON result.

`official_native_latency.py` adds direct author OpenPI JAX/PyTorch and FastWAM
A800 measurements. It records author-default and selected-matched Euler step
counts separately. OpenPI uses its original model/Policy/model transforms;
the resident LeRobot π0.5 checkpoint uses a temporary outer `model.` loading
namespace and strict shared-weight-aware safetensors loading for its tied
embedding. H50 and discrete-state settings are explicit.
FastWAM uses the author's original compiled `infer_action` and cached T5 inputs.
Cold compile/load is excluded and reported separately.

`official_dm05_latency.py` measures the author OpenDM implementation;
`dm05_project_latency.py` measures the project implementation.
`prepare_dm05_workload.py` prepares the declared workload. Author-baseline
configuration and execution are described in [author_baseline.md](author_baseline.md).

For an explicit FastWAM compiled candidate, use `backend="torch"` with
`options={..., "compile_action_infer": True, "use_cuda_graph": False}` in the
existing candidate JSON. This opt-in compiles both video prefill and raw denoise.
Keep step count, shift, inputs and precision explicit; measure against the
author's compiled API. Compile time is separate from steady-state latency.

## HyVLA recorded-oracle validation

The shared tradeoff contract is
[`hyvla_tradeoff_quality.json`](hyvla_tradeoff_quality.json): normalized public20
relative L2 ≤ 0.05 and cosine ≥ 0.999 against a fixed native BF16 semantic
Euler10 reference. All requests/replays must pass. Minimize measured end-to-end
latency; old-runtime or intermediate-tensor EXACT is not required. This is a
scoped runtime-deviation proxy. The 12 padding channels and original official
elementwise gate remain separate diagnostics. See the
[current objective](../../docs/vla_autotune.md#hy-performancequality-objective-2026-11-13-revision).

`validate_hyvla.py` remains the revision-bound public-fixture validator:
three uint8 camera histories `[6,480,640,3]`, normalized state20, supplied FP32
noise `[40,32]`, and recorded normalized action oracles. Its `--require-oracle`
flag checks the historical gate, not the new speed-search admission gate.

Run model execution **inside an allocated Slurm GPU srun step**. The validator
checks the actual single A800/SM80 device. For example, after selecting the
site's existing runtime in a one-GPU batch script:

```bash
export PYTHONPATH="/snapshot/FlashRT${PYTHONPATH:+:$PYTHONPATH}"
srun -n 1 -c 8 python -B -u /snapshot/FlashRT/examples/vla_autotune/validate_hyvla.py \
  --checkpoint /local/Hy-Embodied-0.5-VLA-RoboTwin \
  --fixture /local/public-reference/baseline.npz \
  --sampler euler --num-steps 10 --numerical-profile official_local_fp32 \
  --use-graph --warmup 1 --repeats 2 --require-oracle --output /runs/hy-baseline
```

Use the site's allocation and thread conventions; `-c 8` is an example.
The runner performs no submission, model download, checksum scan or permission
changes. It refuses to overwrite an output directory.

The original recorded gate remains every element within
`0.01 + 0.01 * abs(reference)` and cosine at least `.999`, with finite outputs.
`--atol`/`--rtol` only affect internal graph/eager or saved-run comparisons.
`--require-oracle` returns exit3 on Euler10 mismatch. Other step counts are
diagnostic relative to the Euler10 oracle.

`semantic_bf16` is the unchanged default. `official_local_fp32` matches verified
local dtype/cast boundaries and does not guarantee all historical fixture gates.
CPU-only operations need no Slurm allocation:

```bash
python examples/vla_autotune/validate_hyvla.py --help
python examples/vla_autotune/validate_hyvla.py \
  --compare /runs/euler2 /runs/pf2 --exact --output /runs/comparison
```

Saved-run comparison checks checkpoint/input/profile/source provenance before
reporting differences. Inspect the complete result and its stated gate.

## FastWAM pinned first-frame validation

The FastWAM examples require a **local pinned upstream checkout**, the released
full MoT/proprio checkpoint, Wan2.2 VAE and local T5/tokenizer for context
preparation. They do not download weights or vendor the upstream repository.
The resolved JSON fixes action width7, proprio8, horizon32, 30layers, action
hidden1024, attention width3072 and explicit shift5.

Prepare the source checkout with git and copy its provenance sidecar. The
sidecar records a contract, not independent content attestation:

```bash
export REPO=/snapshot/FlashRT
export VLA_TASK_ROOT=/runs/vla-experiments
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$VLA_TASK_ROOT/fastwam-source"
git clone https://github.com/yuantianyuan01/FastWAM "$VLA_TASK_ROOT/fastwam-source/upstream"
git -C "$VLA_TASK_ROOT/fastwam-source/upstream" switch --detach 7faa71108368fbb3b6885649f112af607427a2d4
cp "$REPO/examples/vla_autotune/fastwam-source/SOURCE.json" "$VLA_TASK_ROOT/fastwam-source/SOURCE.json"
```

Use an existing compatible runtime. Execute one model per process inside an
allocated GPU compute step:

```bash
V="$REPO/examples/vla_autotune/validate_fastwam.py"
S="$VLA_TASK_ROOT/fastwam-source/upstream"
M="$REPO/examples/vla_autotune/fastwam-source/libero_model_resolved.json"
# Set FAST_CHECKPOINT and WAN_BACKBONE to resident local assets.
srun -n 1 -c 8 python -B -u "$V" --prepare-context \
  --source-dir "$S" --backbone-dir "$WAN_BACKBONE" --output "$VLA_TASK_ROOT/context"

COMMON=(--source-dir "$S" --model-config "$M" --checkpoint "$FAST_CHECKPOINT"
        --backbone-dir "$WAN_BACKBONE" --context "$VLA_TASK_ROOT/context/context.npz"
        --warmups 3 --repeats 20 --seed 20261109)
srun -n 1 -c 8 python -B -u "$V" --reference "${COMMON[@]}" --output "$VLA_TASK_ROOT/upstream"
srun -n 1 -c 8 python -B -u "$V" --run --native-linear-backend cached \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/cached"
```

Cached native mode needs `scripts/build_gemm_plan.py` built in that exact
Python/CUDA/source environment; it never silently falls back. Native execution
is partial. The fixture is a deterministic synthetic two-camera pattern.
Euler defaults to 2steps in this validator; `--num-steps 10` selects a different
budget. ProbeFlow `--pf-max` selects its explicit adaptive budget.

Use the library-owned raw-velocity graph with either Torch or cached native
linears:

```bash
srun -n 1 -c 8 python -B -u "$V" --run --backend torch --num-steps 10 \
  --use-cuda-graph "${COMMON[@]}" --output "$VLA_TASK_ROOT/torch-library-graph10"
srun -n 1 -c 8 python -B -u "$V" --run --native-linear-backend cached \
  --sampler probeflow --pf-max 10 --use-cuda-graph \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/cached-library-graph-pf"
```

Torch uses the common pipeline; `--reference` calls the independent upstream
oracle. Do not supply `--native-linear-backend` with `--backend torch`.
Library graph runs perform six same-weight graph/eager refresh controls after
formal samples, retain actions and changed inputs, and reject parity or
input-freshness failures. Actual callbacks establish sampler NFE; captured
Python enqueue counts are distinct from GPU replay work. Cleanup must succeed
before a passing report is written.

## FastWAM diagnostic profiler

`fast_profile/profile_fastwam.py --runner "$V" ... --repeats 1` profiles a
warmed request with CPU/CUDA traces. `vla-stage-profile` still uses this diagnostic
runner. Its output is tagged diagnostic before atomic publication.
Shape/call FLOPs and logical weight bytes are not HBM counters; kernel busy
union avoids overlap double counting. Profiler overhead, host waits and omitted
copies prevent treating those timings as unprofiled latency or proof of a
compute/memory bottleneck. Missing CUDA events mean unknown GPU time.

Fast saved-output comparison is diagnostic unless a caller supplies a gate.
An explicit allclose failure does not currently enforce a nonzero compare exit
code; do not use that exit code as a quality acceptance signal.

## Archived diagnostics

The old Hy EXACT/localization scripts and fixed-Euler2 FastWAM graph pilot were
retired after checking consumers, provenance and current measured paths.
Their complete source and dedicated tests remain in Git at cleanup baseline
`199376281801427abccbb307056b27e74eab1429`. For example:

```bash
git show 199376281801427abccbb307056b27e74eab1429:examples/vla_autotune/hy_trace/diagnose.py
git show 199376281801427abccbb307056b27e74eab1429:examples/vla_autotune/fast_graph/velocity_graph.py
```

The [cleanup inventory](../../docs/vla_autotune_cleanup.zh-CN.md) maps every
retired script and test to its reason and import watermark. The
[measured notes](../../docs/vla_autotune_experiments.md) and
[research record](../../docs/reports/2026-11-12-vla-autotune-research-process.zh-CN.md)
retain historical results and failures. Run historical reproductions from their
recorded immutable source, runtime and input artifacts.
