# VLA autotune: measured iteration notes

These are engineering observations from a bounded A800 iteration, **not** a
blanket hardware qualification, a model leaderboard, or robot task-success
results. Runnable validation, graph-pilot and profiling scripts are under
[`examples/vla_autotune`](../examples/vla_autotune/README.md). Weights, private
environments, raw tensors, traces and generated result directories are not
distributed with these notes.

## Experimental boundary

- One NVIDIA A800-SXM4-80GB allocation; serial fresh processes for compared points.
- Existing native execution runtime: Python3.11, Torch2.11+cu128. Hy recorded
  official references use their separate Torch2.7.1 runtime.
- Formal latency includes frontend validation/transfers/model/completed CPU
  output; load, preparation/capture, warmup and post-return artifact checks are
  separate. Independent instrumented traces are never formal latency samples.
- Action arrays and actual NFE are retained. Fixed-input numerical similarity
  is a scoped proxy, not robot performance. No tolerance was relaxed to rescue
  a fast failing point.
- Earlier runs used version1 result records. Their historical observations were
  not backfilled into version2 identities. The strict v2 campaign below was
  actually run with instance/full-candidate/confirmed-job binding.
- Runs used immutable work-in-progress source snapshots. Later Git checkpoints
  package the source; they do not retroactively replace the original provenance.

## Pi0: real FP16/JAX seam, not an observed acceleration

Job307979 compared Euler2, ProbeFlow2 and Euler2 bookend on the resident Pi0
checkpoint. Ten samples per point: medians26.1060/26.7000/26.3815ms, both2NFE.
ProbeFlow public7 relativeL2=.000375709 and cosine=.999999945 passed that run's
explicit numerical proxy, but was slower. Euler2 was retained as the control.
The slight FP16 beta-zero head/external-integration rounding difference was
measured, not asserted to be bitwise equality.

This was a two-step execution comparison, not a qualification of two-step
sampling against the policy's default quality or a complete Pi0 search.

## Pi0.5: multi-request gating reverses a single-input choice

The first campaign (job307967,20samples/point) measured:

| Point | Actual NFE | Median ms | Public7 relativeL2 | Proxy |
|---|---:|---:|---:|---|
| Euler10 control | 10 | 41.9109 | reference | control |
| Euler2 | 2 | 28.5946 | .224762 | fail |
| Euler5 | 5 | 33.5839 | .0607605 | fail |
| ProbeFlow eps=.002,max20 | 21 | 60.7568 | .0342706 | pass, slower |
| ProbeFlow eps=.008,max20 | 13 | 46.7985 | .0162128 | pass, slower |

The fixed synthetic fixture had911prefix rows:768vision and143prompt/state
rows. Diagnostic prefix intervals were about24ms; each full Action Expert
evaluation about1.65–1.67ms. These intervals include submission/dependency gaps
and are not kernel-duration sums or proof of compute/memory boundedness.

The source/runtime cost ledger gives3.11794TFLOPs for the17prefix FFNs alone
at M911,K2048,N16384, versus31.1427GFLOPs projections/FFNs plus7.08526GFLOPs
attention products per Action Expert evaluation at M50. Logical linear-weight
bytes are not measured HBM traffic.

Round2 (job308132) found Euler7 at36.9639ms and2.56%relativeL2 for one input.
Post-ViT pool2 really changed911->335prefix rows and recorded28.8652ms, but its
33.21%relativeL2 failed. It reduces prefix GEMM M, not the ViT already performed.
The fastest raw point was therefore rejected rather than promoted.

The strict v2 round3 (job308244) broadened to original input, two new fixed
noises and RGBbrightness+8, keeping prompt/state/tensor geometry unchanged.
Each point had8warmups and40total samples,10per request. Gate was explicitly
**normalized public7 relativeL2<=.05 and cosine>=.999** across all four requests.
This is an engineering sensitivity budget, not a universal VLA tolerance.

| Point | Actual NFE | Median ms | Worst relativeL2 | Min cosine | Disposition |
|---|---|---:|---:|---:|---|
| Euler10 reference | 10 | 41.6211 | reference | reference | control |
| Euler7 | 7 | 36.8776 | .665398 | .767542 | reject |
| **Euler8** | **8** | **38.4342** | **.0361504** | **.999482** | **select in this scope** |
| PF eps=.012,max10 | 7/9 | 38.8245 | .129385 | .992178 | reject |
| PF eps=.016,max10 | 5/7 | 37.0234 | .129385 | .992178 | reject |
| Euler10 bookend | 10 | 41.6857 | .0 | ~1 | pass |

Bookend drift was+.155%. The new noise20261111 invalidated Euler7 and both PF
points. Euler8 retained about7.7%lower median latency only within this four-case
proxy. The controller's full-workload coverage and worst-case gate mattered;
the cause of every coarse-trajectory error was not claimed to be isolated.

The real LeRobot release declares ACTION=MEAN_STD despite storing quantiles.
The unified backend now decodes from raw actions using that declaration. Earlier
normalized raw arrays remain numerical evidence, but old legacy-decoded outputs
must not be described as corrected robot commands.

## FastWAM: remove repeated planning, then test graph execution fairly

This adapter is **partial native Linear execution** on the pinned upstream
first-frame action graph, not a full native port. Real T5/VAE/MoT/proprio assets
were used with a deterministic synthetic two-camera input, normalized proprio
and supplied CPU-generated noise. No future-video rollout or task success was
evaluated. All numbers here use Euler2/shift5, not default-policy quality.

The legacy BF16 NT+bias call rebuilt descriptors/layouts/preferences and queried
eight heuristics per enqueue. Job308343 performed20samples/point after3warmups:

| Point | Median ms | P95 ms |
|---|---:|---:|
| Partial-native legacy | 202.5082 | 204.6320 |
| Partial-native cached | 154.7789 | 157.3116 |
| Upstream Torch | 122.5195 | 125.1433 |
| Legacy bookend | 201.9628 | 203.4542 |

All2NFE/oneprefill. Cached and legacy actions were bitwise equal, as were their
bookend outputs. Cached firstcall built19plans; all20steady requests recorded
0builds/queries and919native Linear enqueues across612modules. The47.729ms
median reduction is23.57%; upstream was still faster. Workspace limits differ
(cached64MiB versus legacy256MiB), so the entire difference cannot be assigned
numerically to host planning without an independent algorithm/workspace ablation.

Diagnostic job308460 then found identical920Linear-call/1.22055TFLOP ledgers
(cached919native+oneTorch proprio, upstream920Torch). Logical weights14.0704GB
are not HBM bytes. Both instrumented requests had about51.79ms summed CUDA
kernel duration but long different gaps/spans. cProfile and Kineto overhead
prevent using those walltimes as formal latency or a roofline conclusion.

The private raw-velocity graph pilot was therefore applied **equally** to both
engines in job308611,20samples each, one allocation:

| Point | Median ms | P95 ms |
|---|---:|---:|
| Cached eager | 147.8591 | 149.3722 |
| Cached rawhead graph | 75.8655 | 76.4702 |
| Upstream eager | 118.8005 | 120.3240 |
| **Upstream rawhead graph** | **69.0292** | **69.3196** |
| Cached eager bookend | 148.0505 | 149.6756 |

Graph/eager outputs were bitwise equal within each engine, including six
same-weight original/noise/proprio/image/context/mask controls. Every changed
control affected the output, ruling out a vacuous unchanged-input check. Each
steady request refreshed conditions once and replayed the velocity graph twice;
sampler/integration, VAE and per-request video prefill stayed outside capture.
Owned graph cleanup passed. Coldwarmup/encoding work was retained separately.

Cached graph reduced its eager median48.69%; upstream graph41.89%. The fastest
tested path was **upstream+graph**, not a preset native winner. Cached/upstream
retain their pre-existing1.45%relativeL2 difference; this experiment only proves
graph preservation of each engine, not cross-engine official parity.

The graph pilot remains fixed-geometry/Euler2 experimental code. It is not a
registered production option or new default; adaptive PF, ordinary step counts,
other geometries/hardware and broader quality need validation before integration.

## Hy: keep mechanical checks and official acceptance separate

Semantic BF16 execution, fixed-noise replay and Euler2/PF2/graph-eager mechanics
passed their measured checks. It was not assumed that BF16 labels imply parity.

Job308140 tested the optional official-local cast profile under the original
elementwise `.01+.01*abs(reference)` and cosine>=.999 gate. Baseline,
episode_start,current,prompt passed; history/state/noise failed with4/2/6public
elements outside tolerance. Job status correctly remained FAILED/1:0. The gate
was not loosened and the option was not made default or called official-exact.

The corrected frozen-boundary diagnostic (job308690) reproduced three old
Torch2.7.1 oracles exactly and native eager/prior graph exactly, matched input
identity and preserved prefix invariance before frozen evaluation. Prepared
images/state/tokens/masks matched. Sorting-before prefix embeddings already
differed; replacing native KV with official KV reduced but did not eliminate
raw-velocity differences. This is localization evidence, not a seven-case fix.

CPU inspection of saved tensors found all190464text/special-prefix elements
exact, while266483/301056vision-prefix elements differed. Both implementations'
vision-prefix slices exactly equaled their own merger outputs. Merger inputs
were already different (per-camera relativeL2 about.0057–.0075), so a transformer
sorting/projection hypothesis cannot explain the entire earlier visual mismatch.
Live merger layouts are non-contiguous in both implementations; canonical CPU
artifact strides must never be substituted for them in a layout experiment.

The next **frozen single-operator** diagnostic (job309016, source checkpoint
`d91ef23b`) ran the recorded Torch2.7.1 and native Torch2.11 runtimes serially in
one A800 allocation. It constructed no full policy and changed no action gate.
All seven saved operator arms were independently reread on CPU and were bitwise
identical **across these two runtimes**. Relative to the accepted official trace:

| Frozen control | Unequal elements | Relative L2 | Maxabs |
|---|---:|---:|---:|
| PatchConv, per-camera batch6 | 0 | 0 | 0 |
| PatchConv, combined batch18 | 0 | 0 | 0 |
| Position interpolation, direct BF16 | 0 | 0 | 0 |
| Position interpolation, FP32 then BF16 | 0 | 0 | 0 |
| proj1, official per-camera live views | 0 | 0 | 0 |
| proj1, native combined non-contiguous layout | 314260/1204224 | .002465559 | 1 |
| proj1, contiguous combined batch control | 440808/1204224 | .003142936 | 1 |

Thus patch batching and the interpolation cast hypothesis do **not** explain the
observed difference for these exact frozen inputs, weights and targets. No
speculative interpolation/profile change was made. This is not a universal
statement about other input geometry or hardware.

For proj1, CPU profiler lowering actually recorded three `mm` plus three `add_`
calls for the official views, versus one `bmm` plus one `add_` for the native
batched layout. The contiguous control used one `addmm`. These are operation
counts, not GPU timings or a measured reduction-kernel/algorithm attribution.
They prove an independent batching/layout contribution with values held fixed;
blindly adding `.contiguous()` does not restore official equality here.

This does **not** repair the earlier mismatch already present at the ViT exit,
or qualify the seven original action fixtures. Attention/block arithmetic after
the exact patch/position inputs still needs localization before a product fix.

## Remaining acceptance and testing limits

- These input scopes are synthetic numerical/implementation probes. No robot
  task-success or representative evaluation suite was run in this iteration.
- The graph pilot and causal diagnostics are examples, not product capability
  declarations. Failed jobs and partial source coverage remain explicit.
- Selected CPU suites and staged-only clean exports passed; native import skips
  are not GPU proofs. Real A800 artifacts establish only the stated per-run scope.
- The full-repository CPU pytest attempt had optional/native dependency
  collection failures; it was **not** reported as all-green. Dependency-specific
  environment changes can change that count without qualifying GPU paths.
- Raw tensor/output domains and decoded controls are different. Use the exact
  recorded numerical quality scope, never an inferred task-quality guarantee.
