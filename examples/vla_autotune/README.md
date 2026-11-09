# VLA experiment and validation examples

These are source-checkout examples above `flash_rt.vla`, not a claim that every
model, hardware path or numerical profile is qualified. Keep model weights,
fixtures and generated outputs outside the source tree. Experiments must use
an immutable source snapshot and explicitly named local inputs/checkpoints.

## HyVLA recorded-oracle validation

`validate_hyvla.py` operates on the revision-bound public fixture contract:
three uint8 camera histories `[6,480,640,3]`, normalized state20, supplied FP32
noise `[40,32]`, and recorded normalized action oracles. It is not a generic
robot observation adapter or a task-success benchmark.

Run model execution **inside an allocated Slurm GPU srun step**, not a login
node. The current runner checks the actual single A800/SM80 device. For example,
inside your site's one-GPU batch script after selecting its existing runtime:

```bash
export PYTHONPATH="/snapshot/FlashRT${PYTHONPATH:+:$PYTHONPATH}"
srun -n 1 -c 8 python -B -u /snapshot/FlashRT/examples/vla_autotune/validate_hyvla.py \
  --checkpoint /local/Hy-Embodied-0.5-VLA-RoboTwin \
  --fixture /local/public-reference/baseline.npz \
  --sampler euler --num-steps 10 --numerical-profile official_local_fp32 \
  --use-graph --warmup 1 --repeats 2 --require-oracle --output /runs/hy-baseline
```

Use the site's own allocation and thread conventions; `-c 8` is an example for
the tested A800 setup, not a portable Slurm requirement. The runner performs no
submission, model download, checksum scan or permission changes. It refuses to
overwrite an output directory.

The original recorded gate remains every element within
`0.01 + 0.01 * abs(reference)` and cosine at least `.999`, with finite outputs.
`--atol`/`--rtol` only affect internal graph/eager or saved-run comparisons, never
the original oracle gate. `--require-oracle` returns exit3 on Euler10 mismatch.
Other step counts are only diagnostic relative to the Euler10 oracle.

`semantic_bf16` is the unchanged default. `official_local_fp32` matches verified
local dtype/cast boundaries, not the complete official implementation; its name
does **not** mean official-exact or guarantee all fixture gates pass.

CPU-only operations need no Slurm allocation:

```bash
python examples/vla_autotune/validate_hyvla.py --help
python examples/vla_autotune/validate_hyvla.py \
  --compare /runs/euler2 /runs/pf2 --exact --output /runs/comparison
```

Saved-run comparison checks checkpoint/input/profile/source provenance before
reporting numerical differences. Original oracle acceptance, reproducibility,
graph/eager parity, numerical replay proxies, and robot success are separate
claims. Inspect the complete result, not just a scheduler exit code.

## FastWAM pinned first-frame validation

The FastWAM examples require a **local pinned upstream source checkout** and
the released full MoT/proprio checkpoint, Wan2.2 VAE and (for context preparation)
the local T5/tokenizer. They do not vendor an upstream repository or download
weights. The resolved JSON fixes the released action width7, proprio8, horizon32,
30layers, action hidden1024, attention width3072 and explicit shift5.

Prepare the small source checkout with git and copy its provenance sidecar; the
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

Use an existing compatible inference runtime; no full upstream training stack is
installed by these scripts. Model execution below belongs inside an allocated
GPU compute step, serially with one model per process:

```bash
V="$REPO/examples/vla_autotune/validate_fastwam.py"
S="$VLA_TASK_ROOT/fastwam-source/upstream"
M="$REPO/examples/vla_autotune/fastwam-source/libero_model_resolved.json"
# Set FAST_CHECKPOINT and WAN_BACKBONE to your resident local assets.
srun -n 1 -c 8 python -B -u "$V" --prepare-context \
  --source-dir "$S" --backbone-dir "$WAN_BACKBONE" --output "$VLA_TASK_ROOT/context"

COMMON=(--source-dir "$S" --model-config "$M" --checkpoint "$FAST_CHECKPOINT"
        --backbone-dir "$WAN_BACKBONE" --context "$VLA_TASK_ROOT/context/context.npz"
        --warmups 3 --repeats 20 --seed 20261109)
srun -n 1 -c 8 python -B -u "$V" --reference "${COMMON[@]}" --output "$VLA_TASK_ROOT/upstream"
srun -n 1 -c 8 python -B -u "$V" --run --native-linear-backend cached \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/cached"
```

Cached native mode needs the separate extension built by
`scripts/build_gemm_plan.py` in that exact Python/CUDA/source environment; it
never silently falls back. Native execution is **partial**, not a full native
port. The default fixture is a deterministic synthetic two-camera pattern, not
a representative LIBERO scene or a task-success measurement. Euler uses2steps in
this validator; ProbeFlow `--pf-max` can select a separate explicit budget.

## Raw-velocity graph pilot and profiler

`fast_graph/run_velocity_graph.py` is an **experimental fixed-geometry Euler2
pilot**, not an enabled library option. Both cached-native and upstream-reference
paths can use the same capture strategy for a fair execution comparison:

```bash
G="$REPO/examples/vla_autotune/fast_graph/run_velocity_graph.py"
srun -n 1 -c 8 python -B -u "$G" --runner "$V" --run --native-linear-backend cached \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/cached-graph"
srun -n 1 -c 8 python -B -u "$G" --runner "$V" --reference \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/upstream-graph"
```

Only the non-integrating raw velocity body is captured. Input validation, VAE,
video prefill, sampling/integration and CPU output remain outside capture but
inside whole-infer timing. Conditions refresh every request; latent/time refresh
each evaluation. One geometry is supported; changes reject rather than silently
replay stale data. Six same-weight graph/eager controls (original plus changed
noise/proprio/image/context/mask) run outside formal samples, retain raw outputs,
and require exact equality. Graph-owned cleanup must succeed before success is
published. Cold warmup/encoding counts are separate from steady sampler NFE and
CPU cuBLAS enqueue counters; graph replay is not zero GPU work.

`fast_profile/profile_fastwam.py --runner "$V" ... --repeats 1` profiles a warmed
request with CPU/CUDA traces. Its output is tagged diagnostic **before** atomic
publication. Shape/call FLOPs and logical weight bytes are not HBM counters;
kernel busy union avoids overlap double counting. Profiler overhead, host waits
and omitted memory copies prevent treating those timings as unprofiled latency
or compute/memory-bound proof. No CUDA events means unknown, never zero GPU time.

Fast saved-output comparison is diagnostic unless a caller supplies its own
gate. In particular, inspect the recorded metrics: an explicit allclose failure
does not currently enforce a nonzero compare exit code. Do not use that exit code
as a quality acceptance signal.

## Hy frozen-boundary diagnosis

`hy_trace/diagnose.py` and `hy_trace/reference_observer.py` capture the actual
baseline/state/noise public fixtures in separate old-official/new-native runtime
processes. The official trace must reproduce its recorded oracle exactly, native
eager must reproduce its prior graph output, and current state/tokens/masks,
initial noise and oracle must match before frozen-KV evaluation is admitted.
Reference prefix invariance is mandatory. Numeric prefix/KV equality is **not**
required because its contribution is the subject of the experiment.

Pass explicit local `--official-source`, `--observer-source`, `--checkpoint`,
`--fixtures` and fresh `--output` directories. Native mode also needs `--reference`
and `--prior` artifacts. Preserve each runtime's actual source revision and dtype
profile. CPU-canonical artifact strides are labeled separately from live tensor
strides; intermediate differences are not a new action-acceptance tolerance.
These host-copy-heavy observations are not performance measurements.

### Frozen early-vision operator controls

`hy_trace/frozen_ops.py` narrows an admitted boundary observation without loading
the complete policy. In each allocated A800 process it reads only three selected
checkpoint tensors (patch weight/bias and learned position), accepted official
trace values, and the native report's **live** merger layout. Run the exact same
immutable script in the recorded official and native runtimes, serially in one
allocation, with fresh output directories:

```bash
srun -n 1 -c 8 python -B -u "$REPO/examples/vla_autotune/hy_trace/frozen_ops.py" \
  --official-trace /runs/accepted-trace/official/baseline \
  --native-report /runs/accepted-trace/native/baseline/result.json \
  --checkpoint /local/Hy-Embodied-0.5-VLA-RoboTwin/model.safetensors \
  --label official-runtime --output /runs/frozen-ops/official-runtime
```

The three independent controls are patch convolution per-camera batch6 versus
combined batch18; direct BF16 positional interpolation versus FP32 then BF16;
and frozen proj1 values in official per-camera/non-contiguous, native combined/
non-contiguous, and a separately labeled contiguous batch layout. Values are
held fixed within each control. Output includes raw operator tensors, exactness,
unequal-element counts, numerical differences, and CPU lowering operation counts
(not profiler timings). Overlapping layouts and mismatched merger geometries
reject instead of constructing an invalid view. `--help` needs no Torch or GPU;
actual execution refuses an unallocated/login-node call before reading weights
or creating output. This runner does not submit jobs or change model behavior.

Within-runtime A/B comparisons isolate their stated variable. Cross-runtime
differences also include environment/kernel changes, and even a reference match
does not constitute an action gate, seven-fixture qualification, or performance
measurement. Keep the original action acceptance thresholds unchanged.
