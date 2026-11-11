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
