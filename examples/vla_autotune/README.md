# VLA experiment and validation examples

These are source-checkout examples above `flash_rt.vla`, not a claim that every
model, hardware path or numerical profile is qualified. Keep model weights,
fixtures and generated outputs outside the source tree. Experiments must use
an immutable source snapshot and explicitly named local inputs/checkpoints.

## HyVLA recorded-oracle validation

**Current optimization goal (2026-11-13):** use the shared v2 tradeoff contract
[`hyvla_tradeoff_quality.json`](hyvla_tradeoff_quality.json), normalized public20
relative L2<=0.05 and cosine>=0.999 against a fixed native BF16 semantic Euler10
reference. All requests/replays must pass. Minimize measured end-to-end latency;
do not require old-runtime or intermediate-tensor EXACT first. This is a scoped
runtime-deviation proxy, not an independent robot-quality oracle. The12padding
channels and the original official elementwise gate are separate diagnostics.
The following recorded-oracle runner remains available for historical parity
checks; `--require-oracle` is NOT the new speed-search admission gate. See the
[current objective](../../docs/vla_autotune.md#hy-performancequality-objective-2026-11-13-revision).

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
a representative LIBERO scene or a task-success measurement. Euler defaults to
2steps in this validator; `--num-steps 10` selects a different explicit budget.
ProbeFlow `--pf-max` selects its separate explicit adaptive budget.

The validator also accepts the new **library** path, separately from the old
fixed-Euler2 pilot below:

```bash
srun -n 1 -c 8 python -B -u "$V" --run --backend torch --num-steps 10 \
  --use-cuda-graph "${COMMON[@]}" --output "$VLA_TASK_ROOT/torch-library-graph10"
srun -n 1 -c 8 python -B -u "$V" --run --native-linear-backend cached \
  --sampler probeflow --pf-max 10 --use-cuda-graph \
  "${COMMON[@]}" --output "$VLA_TASK_ROOT/cached-library-graph-pf"
```

Torch uses the common pipeline, **not** the independent direct-upstream oracle.
Do not supply `--native-linear-backend` with `--backend torch`. Library graph
runs automatically perform six same-weight graph/eager refresh controls after
formal samples, retain actions and changed inputs, and reject a parity failure
or a control with no observed output effect. Actual pipeline callbacks establish
sampler NFE independently of metadata; captured head Python enqueues are not
misreported as GPU work. Cleanup must succeed before a passing report is written.
These additions are not retroactive qualification by the earlier pilot results.

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

### Frozen first spatial-attention controls

`hy_trace/frozen_attention.py` follows the patch/position exclusions with a
smaller block0 probe. It reconstructs block0 input values from the accepted
patch+position tensors, loads only norm1/QKV weight and bias (four checkpoint
tensors), and compares camera6 versus batch18 LayerNorm and QKV Linear. The two
Linear arms share identical frozen normalized values rather than conflating a
possible upstream norm difference.

Run `--engine official` in the recorded official runtime, then `--engine native`
in the native runtime in the **same allocated job**, passing the first process's
fresh output as `--reference`. Other required arguments are `--official-trace`,
`--checkpoint` and fresh `--output`. The official process compares FA2's actual
`flash_attn_func` and Torch SDPA, each with camera6/batch18 grouping and identical
QKV. The native process imports no FA2 binary: it consumes the saved official
QKV, compares its own recomputed QKV separately, and evaluates Torch SDPA.
Reference admission binds job, canonical checkpoint path/stat, trace path and
reconstructed input values; it performs no content-hash scan.

The reconstructed patch layout is labeled source-derived, not a measured live
block0 layout. No recorded block0 output oracle exists in the earlier traces:
FA2 here is an **operator API arm**, not independently reproduced block0/policy
acceptance. Tensor layouts, package/runtime provenance and raw differences are
saved, not benchmark timings. No attention backend, profile, default or action
threshold is changed by this diagnostic.

Add `--probe-sdpa-backends` to **both fresh same-job processes** to additionally
test one-hot Torch `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`
and `MATH` at grouping6/18 on the same official QKV. Defaults are unchanged.
Preflight rejection and only exact known host-dispatch rejection strings are
recorded as unsupported; OOM, device/cuDNN execution and synchronization faults
still fail instead of being hidden as skips. Each FA2-exact candidate is repeated
once and retained. No backend is silently substituted.

Before forced contexts, a separate AUTO profiler exports observed SDPA operator
names and CUDA kernel names/counts in its region, never a timing table. Missing
CUDA activity is unverified kernel identity, not zero GPU work. Actual BNHD/BHND
arguments, strides and storage offsets are observed immediately before API calls;
runtime/cuDNN versions, enabled flags and math-reduction flags are recorded.
Backend contexts must restore flags before successful publication. These flags
are used only in the serial diagnostic process, never installed as a product
default or a new action-acceptance threshold. Torch's Flash backend is not assumed
to be the external FA2 binary or bitwise-equivalent to it.
Mirrored GPU `record_function` annotations are counted separately, never as
kernel names. An annotation-only trace still has unverified kernel identity.

### Isolated camera-PROJ1 ablation prototype

`hy_trace/proj1_camera_ablation.py` is a process-exclusive **example override**,
not a registered backend option. It keeps `official_local_fp32`, Euler10,
attention and every merger operation after PROJ1 unchanged. Only that first
Linear uses per-camera calls on the existing weights; pooler/predictor/proj2
remain batched. It never patches global `F.linear`, casts input differently, or
adds `.contiguous()`.

First run `--arm preflight` in its own allocated process with `--checkpoint`,
`--official-trace`, `--native-report`, historical `--profile-qkv`, the frozen Git
`--source-revision`, and fresh `--output`. It rebuilds native B3 values/layout,
checks just the current checkpoint's two PROJ1 tensors against the accepted
frozen weights, then requires the selected candidate to reproduce the saved
official projection **exactly**. The default uses actual native B1 slices with
singleton leading stride225792, not an assumed official196. That candidate was
rejected in job309625 before any full policy was constructed. It also requires real
CUDA activity for the separately filtered name-only profiler confirmation.

The optional `--fold-spatial` follow-up removes only that singleton dimension
before each Linear: an HWD view retaining non-contiguous spatial/channel strides,
original storage and per-camera offset, then restores B1 **after** Linear. It
does not use contiguous or 2D+bias fusion. This targets Torch's leading-stride
fold predicate, not a different mathematical layer. The tiny precondition saves
the candidate and both rejected native B1/B3 controls, raw arrays, unequal counts
and lowering operation counts (not timings). CPU `mm`+separate `add_` evidence
does not guarantee GPU equality. Use the same flag in preflight and both full
arms; mismatched proof/mode is rejected before model construction. Binding is
fixed before capture and restored on exit. The baseline still applies **no**
merger override; its metadata explicitly reports `native_b3`.

Full-model arms consume that same-job/source/checkpoint `--preflight-report`.
Run `--arm baseline` on all seven original fixtures first, adding `--prior` for
the corresponding old validation **directory**; every current baseline must
match saved inputs/oracles/actions/config/runtime/checkpoint/fixture exactly.
Changed source path/stat provenance is reported, not forged equal. Only then
run `--arm camera_proj1` on those fixtures with `--check-graph-eager`.
Forward the ordinary validator's `--fixture`, `--sampler euler --num-steps 10`,
`--numerical-profile official_local_fp32 --use-graph --warmup 1 --repeats 2
--require-oracle`, plus explicit checkpoint/output/source revision. Final parsed
checkpoint/output and actual factory checkpoint are checked against admission;
validator argument abbreviations cannot redirect them.

Binding precedes any prepare/capture. Cleanup drains work, resets owned ViT,
prefix and per-time graphs before restoring the instance method and closing the
backend. Failed cleanup retains owners and prevents successful publication.
Experimental/source/precondition/cleanup metadata is attached to the **first**
result write and printed record. This exclusive-process factory override is not
a thread-safe serving integration.

Use `classify_exit` to continue only completed mechanical checks with exit0 gate
pass or verified exit3 original-oracle mismatch. `aggregate` requires complete,
distinct seven-fixture records per arm, matching baseline/candidate identities
and one admitted experiment; a missing/wrong/duplicated fixture cannot become
7/7. Any candidate failure keeps aggregate exit nonzero. Acceptance remains the
original **padded40x32** elementwise/cosine gate; public40x20 is separately
reported, never silently substituted. No old failed job is relabeled or default
promoted, even if this new prototype eventually passes its recorded gate.

Measured outcome of the fold follow-up: job309681 passed the frozen EXACT gate
but only6/7original full-policy fixtures, versus4/7for the exactly reproduced
baseline. State still failed3public/padded elements; the aggregate stayed nonzero.
Thus `--fold-spatial` is a partial diagnostic result, **not** a qualified product
fix or a recommended default. Keep original tolerances and all failed records.

### Folded candidate state-residual trace

The existing `hy_trace/diagnose.py` has an opt-in `--folded-proj1` native mode;
its original default diagnostic remains unchanged. This mode uses **eager**
execution explicitly, binds the same HWD example before inference, and requires
a fresh same-job/source/checkpoint `--preflight-report` plus `--source-revision`.
Pass historical accepted official traces with `--reference`/`--reference-job`
and completed folded candidate outputs with `--prior`/`--prior-job`. Variant
directories under this candidate prior are direct names, not `euler10-*` names.
It defaults to baseline/state, or accepts explicit `--variants`.

The actual folded actions must exactly reproduce the saved candidate before
interpretation. Reference admission checks images, state, tokens/masks, noise,
oracle, dtype and historical producer identity. Actual time **embeddings** are
recorded; reconstructed scalar schedule values are labeled, not called observed
scalar inputs. The historical BF16 scalar-increment replay is characterized on
CPU solely to validate trace integration, never installed as a new policy solver.

Frozen evaluation first replays the native trajectory exactly using independent
state/mask/RoPE/KV snapshots. Then each official latent and precise official time
embedding is shared by two arms, changing only native versus sorted/expanded
official prefix KV. Scratch is independent, read-only inputs and prefix rows must
remain unchanged, and all raw results are saved. This measures a conditional KV
effect on a fixed trajectory, not free-running policy quality or a new gate.

Final traces are published only after scoped drain/reset/restore/close succeeds.
Frozen tensors and the trace file must both finish before an atomic final JSON
completion marker is published; disk/serialization failure cannot leave a final
success marker or success print.
The previous6/7 original-gate failure remains explicit even if this diagnostic
completes; profiling/trace host copies are not latency benchmarks. It constructs
one native policy and reuses historical official tensors, not a second model or
an incompatible FA2 binary. No new package download or content hash is required.

### Frozen suffix and first Expert query controls

`hy_trace/frozen_suffix.py` loads only13selected checkpoint tensors and constructs
no policy: state/action/time-MLP parameters, Expert layer0 input RMSNorm and Q/K/V
weights, and the shared layer0 query-norm weight. Its old/new runtime processes
consume explicitly historical accepted official and folded-native traces.

For baseline/state at **step0 only**, reconstructed time/RoPE tables must match
the observed native tables, the suffix must exactly match the recorded official
suffix, and batched versus flattened input normalization must match. Only then
does the runner compare separate Q/K/V calls with the native concatenated GEMM
on identical normalized values. Both outputs are rotated and query-normalized
against the actual recorded `query_norm.layer0.1` boundary (call0 is prefix).
It saves raw values and differences, not latency samples. Input admission failure
is nonzero; absence of a QKV arm matching the recorded query is a negative
operator finding, never a changed policy gate or a reason to loosen tolerances.

Use `--engine official` in the recorded2.7.1 runtime and `--engine native` in2.11,
serially in one allocation, with explicit `--checkpoint`, `--official`/`--official-job`,
`--native`/`--native-job`, frozen `--source-revision` and fresh `--output`. The
historical producers retain their IDs; current operator execution has its own job
ID. No old FA2 binary, full-policy import, model rewrite or new default is used.
Atomic result publication follows successful raw tensor saving.

### Recorded eager-attention layout/order controls

The official `hy_trace/diagnose.py` mode accepts opt-in
`--capture-eager-attention`, with historical `--reference`/`--reference-job` and
the current frozen `--source-revision`. An instance-only observer counts exactly
32 eager calls per prefix/suffix phase and10suffix steps, recording layer0 actual
Q/K/V/mask/output/probabilities with live layouts. It preserves mathematics and
restores bindings on failure. Actions, inputs, every latent/time/velocity, the
query-norm boundaries and original prefix/masks must exactly match the historical
accepted trace before an atomic completed record is published.

After that full original producer exits, `hy_trace/frozen_eager_attention.py`
uses the **fresh same-job** capture, loading no model or parameters. Run it
serially in recorded Torch2.7.1 and2.11 with `--engine`, `--reference`, explicit
`--official-attention-source`, frozen `--source-revision` and fresh `--output`.
Only the selected attention method bodies are extracted from revision-bound
sources; module-level code and model imports are never executed.

Four arms hold observed official Q/K/V fixed: original API at restored live
layouts; native local-FP32 attention with matching expanded official layouts;
native contiguous preexpanded KV in original order; and the same contiguous KV
with the observed modality-prefix permutation and matching mask-column order.
Suffix columns/query order remain unchanged. Original2.7 output **and**
probabilities must reproduce the actual captured oracle exactly. New-runtime
differences are measurements, not a reason to loosen that reference precondition.

The contiguous controls model the observed native KV-cache layout, not an
unobserved native Q layout or complete native policy. Native QKV projection and
Qnorm differences are deliberately excluded by freezing official values. These
controls isolate conditional implementation/layout/order contributions, not
robot quality or free-running policy error. All inputs stay read-only; outputs,
contrasts and layouts are retained, with raw files before atomic final JSON.

### Full-Expert frozen prefix-order intervention

`hy_trace/suffix_key_order.py` is a zero-weight native-runtime precondition:
from explicitly historical actual attention captures, it sorts K/V/mask into
native cache order, applies the same inverse-prefix adapter intended for Expert
evaluation, and requires exact recorded contexts plus exact old sorted-control
reproduction at all20baseline/state steps. Use explicit `--reference`,
`--reference-ops`, `--reference-job`, `--reference-source-revision`,
`--official-attention-source`, current `--source-revision` and fresh `--output`.

After both this and the usual HWD precondition pass, folded native `diagnose.py`
can opt into `--probe-suffix-key-order --order-preflight-report ...` with explicit
historical `--order-baseline`/`--order-baseline-job`. It accepts eager only, empty
graph caches, fixed suffix geometry and the local-FP32 profile. The unchanged
sorted counterfactual tensors must exactly reproduce that historical baseline,
and restoring its suffix mask must preserve official attention eligibility.

Native trajectory replay remains outside the intervention. Only then do the
same official latents/time embeddings run with two KV conditions and two orders;
extra tensors use an `original_order.` namespace and comparison rows explicitly
tag `suffix_key_order`. The adapter only gathers prefix K/V/mask columns; it does
not modify its Q argument, suffix order, weights, casts or integration. Earlier
layer changes can nevertheless propagate to later hidden states and Q values:
Q is not numerically fixed across both full-Expert runs. It drains before binding
restore and retains owners on cleanup failure. Its eager attention-call count is
neither sampler NFE nor CUDA kernel count.

This remains a conditional full-Expert diagnostic, **not** a free-running policy
ablation or a new seven-fixture qualification. The historical6/7 failure stays
binding. No graph support, recapture/fallback or production option is introduced.

### Recorded Expert0 MLP / GU controls

Official eager capture can explicitly add `--capture-expert-mlp` to observe the
real `expert.model.layers[0].mlp_v`, without replacing mathematical operations.
The observer checks ten ordered suffix calls and records actual2D BF16 input,
gate/up outputs, down-projection input (SiLU-times-up) and final MLP output.
Gate/up must consume the same input; biases, shapes or event-order changes reject.
Module hooks are removed on both success and failure. Existing historical
actions/trajectories and attention admission remain mandatory.

`hy_trace/expert_mlp.py` then runs serially in the original2.7 and native2.11
runtime using the fresh same-job capture and only **three** selected gate/up/down
weights. Pass `--engine`, `--reference`, `--checkpoint`, revision-bound
`--official-mlp-source`, current `--source-revision` and new `--output`. Only the
original MLP forward body is extracted, after source/job/checkpoint admission;
no full policy, native extension or old FA2 library is imported in the consumer.

Separate Linears and native concatenated-GU GEMM share the observed normalized
input, BF16 weights, SiLU/multiplication and down projection. Original2.7 API
output and all separate gate/up/activation/output values must reproduce actual
records exactly. Fusion differences and runtime differences are saved, not
promoted as a whole-Expert cause or a new robot-quality gate. Scope is Expert0
on ten observed suffix inputs; the prior6/7 failure remains unchanged.

### Recorded32-layer Expert boundaries

`--capture-expert-layers` adds module hooks to admitted official eager capture,
recording each Expert layer's actual input, first residual and MLP output, plus
the final-norm input. It validates the ordered32layer/10suffix-step sequence and
that each residual+MLP output exactly equals the **next observed** hidden input;
the first entry must also match the saved step0 suffix. No math is replaced and
all hooks are removed on failure. Raw boundary traces stay private scratch.

`hy_trace/expert_layers.py` runs the selected actual native `_block`, RMS,
attention and rotate bodies without importing the pipeline/native extension or
constructing a model. It streams11checkpoint tensors per layer with the same
QKV/GU fusion and shared VLM QK norms as the loader;8suffix and3final-head tensors
bring the total to363selected keys. Quantization/fused kernels/graph stay off.

Each block uses sorted **official prefix KV** with independent suffix scratch.
Two conditions distinguish per-layer same-input error (official H_i each time)
from propagated hidden error (native output feeds the next layer, reset at each
official timestep). Both actual native residual and returned hidden are compared
to recorded boundaries; native MLP deltas are not inferred by subtracting BF16
residuals. These errors need not add and are not causal percentages.

All20native suffix starts must exactly match the observed official H0, using
the actual recorded native state/time/latent. The2.11 propagated velocities must
then exactly reproduce explicit historical `--baseline`/`--baseline-job`/
`--baseline-source-revision` official-KV frozen outputs. Failure is nonzero with
an explicit unqualified status, not a relaxed tolerance. The2.7 consumer measures
the same native body's runtime effect; it is not called a historical old-native
replay. Pass the usual `--engine`, `--reference`, `--checkpoint`,
`--official-attention-source`, current `--source-revision` and fresh `--output`.

This fixes external official trajectories and Gold KV only: no native-vision KV
claim, sampler free-run, new seven-fixture gate, robot-success result or changed
production default follows. The historical6/7 failure remains binding.

### Admitted Expert QKV projection geometry

The layer consumer defaults to unchanged `--projection fused`. An explicit
`--projection separate` changes only the unquantized Expert QKV matmul to three
same-weight row-sliced Q/K/V GEMMs followed by concatenation. The selected source
expression must match exactly; source drift rejects, with no fallback. GU,
attention order, RMS, casts, layer-input conditions and output head stay fixed.

For historical layer inputs, pass explicit `--reference-job` and
`--reference-source-revision` plus the paired `--layer-baseline`,
`--layer-baseline-job` and `--layer-baseline-source-revision`. The unchanged
fused run must reproduce all1300same-runtime historical tensors exactly, in
addition to the existing native2.11 velocity and all-step H0 gates. Both old/new
fused processes precede any split process.

The split process also requires `--control-baseline` from the **current job**
and source. Its header/source/runtime/case and actual1300fused tensors are
rechecked before alternate-body compilation. Split-versus-fused effects use the
same raw keys; fixed-Gold output-head values must remain exact as a negative
control. Split velocities are intended interventions, so they are explicitly
not required to equal the unmodified historical velocity. This non-required
comparison is not a quality pass or a weakened gate for the fused reference.
All raw files precede atomic final JSON. No full model or production option is
introduced; original6/7 qualification is unchanged.

### Order interaction with QKV already separate

`expert_layers.replay_core` shares the unchanged numeric loop with the older
layer CLI. Each call creates fresh propagated state and streams the same363keys;
the default remains sorted. The old fused/separate gates and raw keys stay intact.

`hy_trace/expert_order_replay.py` is a dedicated manifest-based driver fixed to
**separate QKV**. Its explicit history chain checks314589fused against314485,
the native fused velocities against313766, and the actual split/fused head
negative controls. Every history raw bundle must contain the full1300finite
BF16 tensors with exact shapes and the full selected-key set. Historical source
records remain historical, not falsely equated to current snapshot paths.

Pass `--engine`, `--arm sorted|original`, the fixed `--inputs` manifest, current
`--source-revision` and fresh `--output`. The provided batch requires both runtime
sorted arms to exactly reproduce all1300historical split-sorted tensors before
either original arm. A standalone driver checks its own runtime; it is not the
cross-runtime barrier.
Current source/runtime/case/checkpoint and input-file path/stat records must match,
and raw control arrays are rechecked before alternate owner compilation. The
historical/current split controls and historical fused reference have distinct
report fields; no historical result is relabeled as a current producer.

The pure adapter inversely gathers prefix K/V and mask columns only, while cache
storage remains sorted. A fresh zero-weight20-context Gold preflight must be exact.
The scope drains and restores `_attn`, retaining owners and primary exceptions on
cleanup failure. Actual Q layouts are passively recorded; Q is not materialized
or reordered. Propagated Q values can nevertheless change after earlier layers.
Its640attention calls per case are neither sampler NFE nor GPU-kernel counts.

Preflight raw and full replay raw precede atomic final JSON. Original-order
effects and fixed-Gold head negatives are retained even when there is no
improvement. No new full-policy, native-vision KV, sampler free-run or robot gate
is implied; historical6/7 remainsFAILED and production defaults are unchanged.

### Fixed-input RMS runtime stages

`hy_trace/expert_rms_stages.py` predeclares four probes: baseline/state at
`(step0,layer1)` and `(step2,layer0)`. It replays the actual separate-QKV,
original-order block and first requires exact residual/output reproduction of
the corresponding historical315024same-runtime teacher records. Only22unique
layer weights are read (44read events); no model or native extension is loaded.

An instance observer calls the real `_rms_norm` once and records its four input,
weight and output boundaries without replacing the returned math. Standalone
diagnostic stages then record FP32 input, square, variance, variance+epsilon,
`rsqrt`, scaling, cast-back and weighted output. That final output must exactly
replay the actual same-runtime RMS on the restored live input layout.

Run old2.7 then native2.11 serially with explicit `--history-inputs` pointing to
the historical order manifest, `--history-job`/`--history-source-revision`, current
`--source-revision`, fresh `--output` and engine label. Native additionally needs
the **current-job** old-runtime `--reference-rms`. It measures both its own block
inputs and the identical captured old inputs/weights, separating upstream input
changes from same-input runtime arithmetic. Historical IDs are never forged as
current producers. Raw files precede atomic final JSON.
Before the native owner is compiled, the old reference must contain the exact
184role keys with finite shape/dtype/layout metadata, and all16weighted stage
endpoints must exactly equal the actual observed RMS outputs. The final summary
computes whether any stage differs; an all-exact negative result stays negative.

No alternative normalization, FP64 arithmetic, primitive replacement or relaxed
threshold is installed. First-stage differences are measurements, not an assumed
`mean`/`rsqrt` bug or an all-layer explained-error fraction. Scope stays at these
four probes and the prior6/7 policy failure remains unchanged.

### Standalone RMS intermediate handoffs

`hy_trace/expert_rms_handoff.py` consumes the retained old184/native328 arrays
from315256, not a new model run. It first admits complete historical roles,
actual RMS endpoints, live input/weight metadata, job/source identity, source/
checkpoint stats and native runtime. All144recomputed native fixed-old arrays
must exactly reproduce that historical baseline before any cut is interpreted.

On the same16norms, five independent cuts inject old values at `variance`,
`variance_plus_epsilon`, `inverse_rsqrt`, `scaled` and `cast_back`. Only each
cut's unchanged downstream suffix runs. Injection uses the **current native
intermediate layout**, because old intermediate live strides were not recorded;
this does not claim old-layout equivalence. A same-native-value reinjection
control must reproduce every baseline suffix exactly. The784raw arrays include
144baseline,320control and320old-cut tensors; endpoint exactness and remaining
stage differences are measured outcomes, never assumed success criteria.

Use explicit `--history-root`, historical job/revision, current source revision
and a fresh output under one allocated native2.11 A800 process. No checkpoint
tensor is read, block/model constructed, FP64 alternative installed or policy
normalization replaced. Conditional local sufficiency, if observed, is not an
all-layer/full-policy/native-vision/free-running/robot qualification; historical
6/7FAILED and production defaults remain unchanged.

### Guarded variance handoff in actual selected blocks

`hy_trace/expert_rms_block_handoff.py` asks whether the standalone RMS cut
composes through the same four actual separate-QKV/original-order Expert blocks.
It admits the old184/native328 history, selected checkpoint/source/runtime and
KV/mask/RoPE records, then requires each native block's residual/output and all
12actual norm boundary tensors to reproduce315256EXACT before intervention.
The two historical input reports must first match the report stats retained by
315256, and their manifest/Gold-trace/native-trace records must bind to the exact
paths actually opened. Empty records or valid records for other files cannot
reach owner creation or checkpoint tensor reads.

The intervention is scoped only to that standalone owner's `_rms_norm`. Each
actual input and weight, including live layout, must exactly match the old
captured role **before** an old variance can be used. It observes the original
native RMS once, injects old variance in the current variance layout, and runs
the unchanged suffix. A nonmatching input/weight/layout stops that probe with
`input_mismatch`, retains the failed boundary, and never falls back to applying
an unrelated old variance. Other probes still run; unexpected failures prevent
final publication. Bindings and counts are checked and restored.
A mismatch accompanied by synchronization/cleanup failure is an unexpected
failure and aborts at that probe boundary, rather than continuing other probes.

Complete probes compare actual residual/output to the conditional old block
oracle; exactness is a measured outcome, not a diagnostic acceptance criterion.
All native boundaries and intervention inputs/native outputs/injected variances/
returned outputs remain in raw arrays. Four complete probes contain144tensors;
partial mismatch reports have an explicit smaller role set. Only22unique layer
weights are read(44events), no model/extension constructed or production norm
changed. This does not test propagated32layers, full policy, native vision KV,
free-running trajectories or robot success; historical6/7FAILED remains intact.

### Isolated FP32 mean launch observations

`hy_trace/mean_launch.py` observes the same16fixed-old norm inputs without a
block/model/weight read. Both runtimes first admit315256's fullold184/native328
roles, source/checkpoint stats and exact runtime/Git/flags. FP32 conversion and
square must be EXACT old; mean before and during profiling must be EXACT the
historical same-runtime variance. Each runtime saves64raw arrays and16isolated
CPU+CUDA trace windows, with manual live float/square shape/stride/offset and
pointer-mod16 metadata. Full pointer values are not added to the report.

Each window contains one labeled mean call; every full CUDA kernel name and
grid/block is retained, not filtered to a predicted implementation. Two warm
calls precede each window, and profiling does not measure candidate latency.
Shape/stack/memory/flops tracing stays off. Missing CPUmean, kernel or launch
metadata rejects final publication; all raw Chrome traces stay private.
The unique label is the CPU `user_annotation`; a same-named
`gpu_user_annotation` projection is retained but is not a second CPU range.
Two actual CPU labels or a GPU-only label still fail admission.

Actual launch observations can test conditional source-based predictions, but
equal symbols and launch dimensions still do not reveal runtime `ReduceConfig`
vectorization or warp-shuffle branches. This is not a compatible mean, an error
fraction, a library-bug claim or a policy gate; historical6/7FAILED and production
math/dtype/runtime/defaults/thresholds remain unchanged.

### Source-explicit FP32 mean tree diagnostic

`hy_trace/mean_tree.py` models32lanes and four independent FP32 accumulators
using only eager tensor additions, then explicit ascending/descending shuffle
trees and one final power-of-two factor. It predeclares four scalar/vector4
grouping × shuffle-order arms. Source-old hypotheses are scalar-ascending for
128wide rows and vector-ascending for1024; source-native is vector-descending.
These are hypotheses to test against actual arrays, not asserted dispatch facts.

The function receives only the direct square tensor and two bools, never a
stored variance. It calls no native `sum`/`mean`, uses no FP64/JIT/build and
rejects nonfinite/negative/non-FP32 inputs, widths outside128/1024, nonunit last
stride or non16byte-aligned pointers. A reshape may copy logical rows without
changing values. It is deliberately a multi-kernel arithmetic diagnostic, not a
fast or generic production normalization implementation.

The native-only driver first admits315256history and315622actual launch capture,
exact runtime/source/checkpoint stats, direct square layout and unmodified
native mean values. It records96arrays:16squares,16native baselines,64tree means;
every arm is compared to both old/native arrays. Hypothesis equality is an
outcome, not a success requirement. No block/model/checkpoint tensor is read,
no candidate latency measured, and no internal compiled branch, full-policy or
robot qualification follows. Historical6/7FAILED and production defaults stay.

### LIVEinput source-tree RMS across the conditional full Expert

`hy_trace/expert_mean_tree.py` wraps only a standalone native AST owner's norm.
`tree_rms(x,w)` derives its FP32 variance from the current input using the tested
source-old arithmetic tree, then applies the unchanged epsilon/rsqrt/scaling/
cast-back/weight order. It receives no captured norm input or old variance and
has no silent native fallback. R31's complete96raw support and actual16selected
old/native means must be admitted before owner creation, not header flags alone.

The shared `replay_core` runs unchanged separate-QKV/original-order GoldKV,
32layers,10steps,baseline/state with teacher and propagated hidden modes. All
1300control tensors/case must EXACT the prior native-original315024records
before the prototype is interpreted. Every prototype tensor is compared to the
old2.7original oracle; equality is an outcome. Final norm and both velocity heads
participate, so fixed-Gold head output is NOT an unchanged negative control here.

Each arm checks5160actual norm calls and scoped restoration:2560FP32/width128,
2600BF16/width1024 including40heads. It reads363unique checkpoint keys/arm,
726events across the two serial arms, constructs no model, and publishes all
5200BF16arrays plus every old comparison. Expected historical report/raw and
actual manifest/Gold/native trace/frozen-input paths/stats are bound before owner
and checkpoint tensor reads. H0/order eligibility and attention cleanup remain
hard gates. Large raw data stay private; no content hashes or new builds.

This intentionally multi-kernel prototype is not a latency candidate or a new
production norm. GoldKV/fixed external trajectory excludes native vision KV,
sampler integration/free-running, seven-fixture and robot qualification; original
6/7FAILED, runtime defaults and quality thresholds remain unchanged.

### Expert-phase-only native seven-fixture policy diagnostic

`hy_trace/expert_policy_scope.py` observes each real prefix assembly and its
actual permutation. It verifies sorted embeddings/masks, fresh request generation,
native prefill/cache identity and read-only prefix KV. Only inside the actual
nine-argument `evaluate_velocity` call does the candidate rebind a source-guarded
separate-QKV block to the **real pipe**, invert prefix attention columns and use
LIVEinput source-old RMS including the final head. Vision/VLM/prefill remain
native, including their128-wide Q/K norms. No Gold KV, latent, norm or velocity
is injected. Every NFE requires32block/32attention/129RMS calls; original RMS
delegation is129for control and0for candidate. Graphs/quantized/fused/mega paths,
stale permutations, input/prefix mutation and nested/reentrant execution reject.
Class-descriptor absence as well as own bindings are restored on failure.

`hy_trace/native_policy_tree.py` reuses the HWD scoped backend and existing
Euler10/frontend/integrator, with fresh supplied noise and both graph caches
empty. It admits the complete R32source support before constructing one model.
All7historical HWD controls run twice and must reproduce old actions/6of7gate
**before any candidate request**; then the candidate runs the same7cases twice.
That is28requests/280NFE, without prepare, warmup, profiling or latency claims.

Old seven-fixture `actions.npz` files do not contain step trajectories. Only
baseline/state have separate historical313766step traces; those are compared.
The other5traces are explicitly CURRENT control captures with exact two-replay
checks, never relabelled as historical velocity evidence. Candidate native prefix,
conditions and time embeddings must equal current control; latent trajectories
are genuinely free-running. The original padded40x32 abs/cos gate is reused,
public40x20 reported separately, and candidate failures remain valid outcomes.

All raw request inputs/conditions/prefix KV/ten-step latents,time embeddings,BF16
raw velocities and FP32actions are saved after request cleanup. Their sidecars
say pending-backend-cleanup, not completed-policy success. The final report is
atomic and only follows all28requests plus successful backend close/cleanup;
failed controls stop the candidate and retain unqualified partial artifacts.
Even7/7would mean only this existing synthetic numerical proxy, not robot,
generalization, performance or production-default qualification. Historical6/7
FAILED records and all thresholds remain unchanged.

### Frozen complete Merger stages

`hy_trace/merger_stages.py` separates different upstream vision values from
complete Merger arithmetic. With only eight BF16 checkpoint tensors, it runs the
actual official AST Tensor-forward/pooler once per camera and the unchanged
native HWD `camera_merger` (per-camera PROJ1, batched remaining operations).
Four combinations use accepted Gold308690 or historical native313766 inputs,
each reconstructed in both recorded input-layout forms. No model is constructed.

Fourteen stages span input, PROJ1, spatial grouping/mean, predictor, softmax,
products/sum, activation, PROJ2 and merged tokens. Each reconstructed endpoint
must EXACT its own actual API. Official camera live layouts are recorded before
concatenating artifact values; no batched live layout is invented for them.
Each runtime saves60arrays, including four actual endpoints. Old2.7 official on
Gold input must reproduce accepted Gold image tokens; new2.11 native HWD on the
historical native input must reproduce native `vision.merged` and the verified
R33 image-token output bridge. Other batching/runtime comparisons are outcomes.

The input manifest binds actual report/raw paths, file stats, source, checkpoint
and runtime identity before weight reads. Baseline/state shared vision values,
prepared inputs and the R33 output bridge are checked. Historical layouts omit
storage offsets: reconstruction uses bounded zero offsets, explicitly not an
observation of the old offset. R33 actual vision input was NOT captured; a
historical input and current output bridge must not be relabelled as one capture.
The original6/7 policy failure, precision/runtime/defaults and gates remain;
this is not a policy fix, error-fraction, performance or robot qualification.

### Frozen PROJ2 call-geometry controls

`hy_trace/proj2_calls.py` isolates the last Linear on two R34 activations. It
admits both complete R34 sixty-role captures, runtime/source/checkpoint identity,
same-value form activations and historical endpoint anchors before reading only
PROJ2 weight and bias. Seven predeclared arms compare B3, per-camera B1/HWD/flat49,
flat147, and batched/per-camera explicit BF16 matmul-then-bias expressions.
The explicit bias addition is a different rounding expression, not a claim of
the same fused epilogue. No activation, weight or output is precision-promoted.

Each arm makes two calls outside profiling, then one CPU-only profiler window
with actual aten input shapes. It saves all fourteen Chrome CPU traces and
thirty arrays (two activations, fourteen unprofiled and fourteen profiled outputs).
Profiled outputs must EXACT their unprofiled controls. B3+bias and per-camera
B1+bias must reproduce R34's respective historical endpoints; all other form
matches or failures are measured outcomes against both references.

This tests the row-geometry/lowering hypothesis, not actual cuBLAS algorithm,
epilogue implementation, GPU kernel duration or speed. No model, full Merger,
native policy or robot run follows from these operator results. Original6/7FAILED,
runtime/precision/defaults and quality thresholds remain unchanged.

### Single-PROJ2 native-policy diagnostic

`hy_trace/proj2_policy.py` changes only the last biased Linear in the existing
HWD camera Merger: B3 becomes three B1 camera views. A guarded AST transform
requires exactly the original signature/last assignment/return and proves an
inverse transform changes nothing else. It binds to the real pipe, does not
patch global linear, and restores instance-method presence and PROJ1 observation
metadata on every exit. Actual inputs/weights are read-only; view storage/shape/
offset evidence is recorded. No forced nonzero effect is required.

The `preflight` arm uses only eight weights and both admitted R34 input origins.
Unobserved original and observed-control whole Merger endpoints must EXACT R34
native HWD; the observed candidate must EXACT R34 official. Twenty raw arrays
retain separate actual control/candidate activations, outputs and a fresh HWD
PROJ1 child check against captured Gold projection values. The complete proof
has its own kind; its old-kind child is only the actual PROJ1 check, not an
invented attention-profile or full-policy qualification. All raw, source,
runtime, checkpoint, callback/view and cleanup checks precede model construction.

The `policy` arm reuses bounded input/request helpers with original Expert in
BOTH arms (`expert_enabled=False`, no separate template). It runs controls for
all seven cases twice before any candidate, then the candidate twice:28requests/
280NFE, no prepare/warmup/profile/timing. Actual Merger input and pre-PROJ2
activation, non-image prefix, masks/permutation/RoPE, suffix conditions/noise/time
embeddings must stay exact across arms. Image embeddings, all context-dependent
prefix KV and later trajectories may change and are measured; prefix KV remains
read-only inside each request. Actual modality masks, not a fixed prefix length,
bind each arm's merged output to its own visual prefix rows.

Every request saves64fields: the previous60 plus actual Merger input, actual
PROJ2 activation/output and merged tokens. No `prepared.images` recapture is
claimed. Full1792request tensors, twenty preflight arrays, two replays, original
padded gate/public slice and close/cleanup must be independently audited. A
candidate quality failure is retained as a completed diagnostic, never a reason
to loosen the gate or combine the earlier Expert triple. No production default,
runtime/precision, robot/generalization or unmeasured performance change follows.

### Expert × PROJ2 four-cell interaction

`hy_trace/policy_factorial.py` fixes four uniform groups in order: base(E0P0),
Eonly(E1P0), Ponly(E0P1), both(E1P1). E is the existing velocity-only Expert
bundle; P is the biased PROJ2 B3-to-three-B1 intervention. All seven cases get
the same configuration, never a case-dependent router. This is a post-selected
mechanistic interaction experiment, not held-out generalization; E's three
components cannot be separately identified by this matrix.

A fresh R36 complete-Merger preflight remains a separate first process. The
second process admits R32's full5200-array Expert support, R33/R36 source,
runtime/checkpoint/fixture identity and both historical replays before constructing
one eager model backend. All three known groups must reproduce their raw fields
before any both request. base/Ponly match R36's64fields; Eonly matches R33's60
old fields, and its four NEW actual Vision fields match current base, not
imaginary historical R33 inputs. The shared control/candidate request guard is
unchanged: each fixed group has its own root, explicit E/P flags and globally
unique operator identity. Repeated local candidate IDs are not global IDs.

Each request retains64typed finite arrays. Fixed input/activation/layout,
non-image prefix, masks/segments/permutation/RoPE, suffix/noise/time conditions
stay exact across groups. Fixed P with E toggled must preserve all Vision and
prefix/KV values; fixed E with P toggled may change visual prefix, KV and later
trajectory. One native prefill and all per-NFE Expert/Merger counters, internal
KV read-only checks and cleanup remain required.

After backend close, CPU FP64 analysis computes actual finite differences
`I=(both-Ponly)-(Eonly-base)` for first raw velocity and final actions, saving
all fourteen interaction arrays. This is not an FP64 GPU policy, an interaction
inferred from RMSE/pass counts, or an explained-error percentage. Completion
requires56requests/560NFE, all3584request arrays plus20preflight and14interaction
arrays, known-cell admission, both replays and all cleanup. The old28-request
publisher is not relaxed or reused. If both fails the unchanged seven-case gate,
stop this combination search and localize upstream ViT next. Even7/7 is only
the existing proxy scope, not default, robot, speed or generalization promotion.

### Frozen actual ViT prefix localization

After the rejected R37 joint candidate, `hy_trace/frozen_vit_prefix.py` localizes
only blocks0–3 on one admitted three-camera/six-frame visual context. Four blocks
are the shortest prefix reaching the first captured Gold block endpoint. It
does not run a policy, read all27layers, alter kernels, or restart combination
search. Header preflight identifies exactly51BF16tensors/161436544bytes; the actual
reader admits their1152/4304 geometry and never loads a complete checkpoint.

The old runtime imports source-bound original `_create_hyvit2(select_layer=4)`,
`SpaceTimeBlock` and the video function. Actual module hooks capture9fields per
block, block3 time PE/temporal V and the actual current-frame return separately
for each camera. Three complete six-frame block3 outputs, Patch and positional
outputs must EXACT Gold. No new block0 input is misrepresented as a historical
capture: it is observed now, rather than reconstructed from stored patch+position.

The native runtime imports no official FA2 module. Eight actual methods retain
the original `vit_forward` loop; an instance stop occurs AFTER block3 returns,
before block4 can touch any weight. This is an actual block3 return, NOT a
complete27-block API output. A stop-only control must EXACT the observed prefix.
Inverse-checked AST observers replace only calls with wrappers that invoke the
same original Torch operation and return it unchanged; no global patch or staged
loop substitutes for the real methods. Hooks/own attributes restore on errors,
inputs and weights remain read-only, and partial captures never become complete.

Official124/native42 typed arrays (166total) retain actual chain/residual and
current-frame/control bridges. Block3 norm1 input AND output are4D before their
callers' view. Analysis concatenates only on CPU for camera/frame comparisons;
the earliest unequal observed boundary is not automatically a primitive cause.
Here relative L2 normalizes old-minus-native by the native endpoint norm;
equality, maxabs and RMSE are symmetric. This is not the policy quality gate.
Old/new runtime AND per-camera/combined FA2/SDPA forms are confounded here. Old
internal layouts/offsets were not historically captured; the observed image
stride is retained and zero storage offset is explicitly a reconstruction
assumption. New native internal captures are not claimed historical313766 data.

Import/source/old Gold anchor failure stops the experiment without installing
dependencies or falling back to approximate reconstruction. If these four blocks
match, report "not localized in this prefix"; do not automatically expand to
blocks4–26. All166arrays and source/call/cleanup evidence need independent audit
after a bounded old-then-new allocation. No policy-error fraction, generalization,
robot quality, unmeasured speed or production-default qualification follows.

### Actual attention context / projection bridge

`hy_trace/frozen_vit_attention_bridge.py` splits the R38 block0 post-projection
boundary, using only four BF16 QKV/projection tensors (10626048bytes). It runs
the original source-bound `_ViTAttention.forward` for three camera6 inputs and
the original native `_vit_qkv/_vit_spatial_attn` for combined18. Neither a full
prefix nor a policy is constructed. Fresh QKV/API/control/self-projection values
must reproduce their respective actual R38 endpoints before any local cut.

The R38 helper is imported from its ORIGINAL round038 path, retaining `__file__`
and its original guards; the R39 copy is not used to relax those guards. Current
native source must match R38 bytes. R38's full typed166raw, source/runtime/Git/
flags/checkpoint and observed norm1 stride/offset are admitted before the four
reads. The native phase also admits the same-job official21fields before its
weight reads. Old FA2 never enters the new runtime. Newly captured projection
prehook contexts are NEW data, not retroactive R38 observations or legacy loop
outputs renamed as actual API captures.

Four fixed native-runtime operator cells use N/O (new/old actual context) and
18/6 (one combined / three camera projections): N18, N6, O18, O6. Controlled
`copy_` substitutions use the SAME actual native projection-input buffer and
restore its values/layout afterward; actual camera views share that storage.
All calls keep original biased `F.linear`, dtype and weights, with no flatten,
bias split or new attention backend. N18 must exactly reproduce native output
BEFORE other cuts; restoration and post-restore projection are hard gates.
O6 matching the old endpoint is an outcome, not a forced gate, because remaining
runtime/layout/lowering effects are still possible. This is a local operator
cut, not a revival of rejected Expert×PROJ2 policy search.

Official21/native17 BF16 arrays plus one CPU FP64 finite difference
`I=(O6-O18)-(N6-N18)` make39complete arrays. A negative old corner, zero effect
or nonzero interaction may all be valid diagnostic outcomes. The exact raw
roles, actual source contexts, counters, views, own API anchors and cleanup must
pass independent audit. Failures merge already captured API fields with cut
partials and preserve the original exception even when evidence writes fail.
New actual contexts are separately compared with retained309182/309409 contexts;
a mismatch never triggers another backend search. No performance, robot,
generalization, default or explained-policy-error-fraction qualification follows.

### Weightless actual-QKV row-LSE observation

`hy_trace/frozen_vit_lse.py` observes only existing Flash LSE returns on R39's
actual BF16 QKV. It reads **zero learned checkpoint or trace-weight tensors**.
R39's original namespace is retained for metadata-only guards and validators;
its full tensor loader is deliberately NOT called, since that recursively reads
Gold Merger weights. A role-allowlisted reader checks the entire header before
any tensor read, admitting only166R38/38R39 BF16 arrays plus the saved FP64
interaction. A full real-disk CPU admission test audits all205 nonweight reads
and rejects learned roles, source changes and malformed observed layouts before
the API consumer. This is not a claim of old/new CUDA API parity.

Only groups6 and18 are observed. Old actual camera6 QKV views and native actual
combined18 QKV layout/offset are restored; old combined18 is explicitly a NEW
grouping control, while native camera6 uses slices of the original buffer. Both
plain and LSE-capture contexts must EXACT their own R39 context before LSE has
interpretive standing. No arbitrary contiguous-input, backend, precision, flag,
runtime or dependency fallback is provided.

Old `flash_attn 2.7.4.post1` uses public `return_attn_probs=False/True` with
dropout0. Installed wrapper AST must keep `return_softmax and dropout_p > 0`;
the empty `S_dmask` is NOT a probability matrix. New AUTO is controlled against
version-bound `aten::_scaled_dot_product_flash_attention.default`, nine returns,
debugFalse. Exact installed schema/signature/module/binary stat checks are
required; source metadata is not proof of a compiled branch or a kernel cause.
Dense native `cum_seq_q/k` are undefined Tensor placeholders exposed as Python
`None` at positions2/3; this does not make context/LSE/debug optional. Defined
RNG/unused tensors have metadata recorded without reading uninitialized values.
Logical head_dim72 sets the scale, not the internal rounded96.

Each engine retains8arrays: QKV before/after, control/capture contexts and FP32
`[18,16,196]` LSE at both groupings. One CPU FP64 mathematical LSE reference
(with FP32 scale promoted to FP64) makes17total arrays; it does not emulate
fused Tensor Core/FP32 accumulation order. LSE comparisons keep numerical and
actual int32 bit-pattern equality separate, including signed zero, without
hashing. Camera/frame/head metrics and context-any-channel row cross-tabs use
explicit B/H/N/D mapping. Reference distances are not kernel correctness rankings
or a replacement quality gate. Equal stored LSE does not prove equal scores or
probabilities; different LSE cannot identify QK versus softmax/exp/log rounding
or explain policy-error fractions. These are outcomes, not forced passes.

Partial control/capture outputs, LSE and a computed FP64 reference remain
unqualified evidence after late failure; raw precedes atomic completion JSON.
Returned context/LSE snapshots precede parser qualification, retaining original
shape/dtype/layout and NaNs under failure-only roles. Snapshot failures are notes
on the original parser error, never a reason to hide it or qualify the capture.
Unsupported schema/API, changed context, mutation or missing arrays stops the
observation. No model/prefix/policy/PE/more-blocks, kernel build, profiler, warmup
or speed/default/robot/generalization qualification is introduced.
