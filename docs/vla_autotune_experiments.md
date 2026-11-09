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

### Opt-in library-owned raw-field graph

Job309097 later exercised the **new library integration**, frozen at `a5140aac`,
not the pilot's upstream method override. Default eager behavior was unchanged.
One A800 allocation ran fresh serial processes, three additional whole-request
warmups after the separately measured first call, and ten formal samples/point:

| Point | Actual NFE | Median ms | P95 ms |
|---|---:|---:|---:|
| Independent direct-upstream Euler10 | 10 | 411.4261 | 412.9479 |
| Common Torch pipeline Euler10 eager | 10 | 389.6625 | 392.2531 |
| **Common Torch pipeline Euler10 graph** | **10** | **146.6673** | **147.4356** |
| Partial-native cached Euler10 eager | 10 | 515.0250 | 515.6989 |
| Partial-native cached Euler10 graph | 10 | 158.2449 | 159.1946 |
| Torch PF max10,eps=.008 eager | 2 | 114.5456 | 115.4969 |
| Torch PF max10,eps=.008 graph | 2 | 66.0323 | 66.3118 |
| Cached PF max10,eps=.008 eager | 2 | 147.6405 | 148.5308 |
| Cached PF max10,eps=.008 graph | 2 | 77.0178 | 79.1293 |
| Torch Euler10 eager bookend | 10 | 388.8984 | 390.8952 |

Independent CPU rereads verified actual input arrays, checkpoint path/stat and
resolved model configuration identity across all ten points. All four same-engine
graph/eager formal action arrays and sampler metadata were exact. Each graph
also passed six same-weight graph/eager original/noise/proprio/image/context/mask
pairs; every changed control affected the output. Each request had one video
prefill. Owned cleanup succeeded in every library point.

Torch Euler10 graph was also bitwise equal to the independent direct-upstream
Euler10 original-input output. Cached Euler10 eager/graph retained the same
cross-engine difference, relativeL2=.00594835 and maxabs=.0078125; that is not
graph error or an official-parity guarantee. Torch eager bookend drift was
-.1961%. Graph reduced its own Torch Euler10 median62.36% and cached Euler10
median69.27%; Torch+graph remained faster than partial-native+graph.

Cached Euler10 steady metadata recorded305CPU native enqueues (prefill), but
ten raw graph replays each encoded307native Linear calls:3070logical native
Linear calls in the action field, **not zero GPU work or a kernel-count claim**.
Steady plan builds/heuristic queries were zero. The first capture's three eager
head warmups and one encoded body were recorded separately from sampler NFE.

Two additional **real-CUDA** regressions passed in this allocation: raw capture/
refresh/cleanup without device/host global cache flushing, and eager standalone
velocity completion before close released internal timestep storage. This
followed RED->GREEN fixes for both findings from independent static review.

The PF configuration above was adaptive, but **all six tested requests routed
to2NFE**. It does not validate dense restart. Its original-input output differed
from same-engine Euler10 by26.36%relativeL2 (Torch) or25.93%(cached). No robot
quality or FastWAM sampler-approximation acceptance gate was defined, so the
66ms point must not be selected as a quality-qualified policy merely by speed.
Dense PF, default20steps, other geometries/hardware and robot success remain
unqualified by this run. The option is still disabled by default.

Two subsequent parameter-only jobs reused the **unchanged** `round012/a5140aac`
source; neither edited or retrospectively relabeled job309097:

| Job / point | Scheduled steps | Actual NFE | Eager median ms | Graph median ms |
|---|---:|---:|---:|---:|
| 309138 Torch PF max10,eps=.0001 | 8 | 9 | 390.1538 | 140.6916 |
| 309138 cached PF max10,eps=.0001 | 8 | 9 | 489.7465 | 151.5507 |
| 309187 Torch Euler20 | 20 | 20 | 804.6938 | 255.0711 |
| 309187 cached Euler20 | 20 | 20 | 971.2947 | 262.6851 |

Job309138 really exercised discarded-probe/dense restart: both engines selected
8steps/9NFE on the original input; changed inputs selected8/10steps with9/11NFE.
Torch proprio selected10/11 while cached proprio selected8/9, illustrating that
small engine numerical differences can change an adaptive routing decision.
Same-engine graph/eager arrays, schedules and all twelve refresh pairs were
exact; all ten mutants affected output and cleanup passed. On identity-checked
inputs, numerical-only comparison to saved same-engine Euler10 refresh outputs
gave worst relativeL2=.0408773 (Torch) and.0436711 (cached) across six cases.
That comparison spans jobs and is **not** a cross-job latency ranking or a robot
quality gate.

Job309187 covered the library's default **20-step budget** with independent
direct-upstream Euler20 (median797.6525ms), Torch and cached eager/graph. Actual
NFE20, one prefill/request, complete raw sample equality, both six-case refresh
sets and owned cleanup all passed. Torch original-input output was bitwise equal
to the independent upstream control; cached retained relativeL2=.00471522 versus
that control, identically in eager/graph. Own-engine median reductions were
68.30% and72.96%; cached graph still was not faster than Torch graph.

For cached Euler20,305steady host enqueues do not replace the20graph replays
containing6140logical native Linear calls. Cold3head warmups/oneencoded body remain
separate. Euler10 original-input output differed from same-engine Euler20 by
6.55%relativeL2 (Torch) or6.51%(cached), so even an exactly preserved faster
10-step graph is not automatically qualified as a20-step policy replacement.
The measurements establish execution preservation, not task success or a
universal sampler budget. Other geometry/hardware remain unqualified.

### Bound v2 FastWAM graph/sampler search

Job309300 exercised the actual controller/worker path after fixing graph replay
counters being mistaken for immutable backend identity. Its immutable source
was `round014/baef55ff`; full Candidate, campaign-instance and confirmed job
binding were enforced, not backfilled into older standalone validation records.

Four requests held geometry/context/proprio fixed: original prepared input/noise,
new CPU-Torch noises20261110/20261111, and imagebrightness+.1 clipped to[-1,1].
This is synthetic sensitivity, not representative task evaluation. Before the
run, the explicit engineering gate was **all7normalized channels, worst relative
L2<=.05 and minimum cosine>=.999 versus Torch graph Euler20 across all four**.
It is not a universal VLA tolerance; no threshold was loosened after rejection.
Hy's original oracle gate and all library defaults remained unchanged.

Each point used8excluded warmups and40total measured requests,10per case, in
fresh serial processes within one A800 allocation:

| Graph point | Actual NFE | Median ms | Worst relative L2 | Min cosine | Disposition |
|---|---|---:|---:|---:|---|
| Torch Euler20 reference | 20 | 252.0941 | 0 | ~1 | control |
| Torch Euler10 | 10 | 150.3535 | .107391 | .994934 | reject |
| **Torch Euler14** | **14** | **191.7414** | **.0372622** | **.999441** | **select in this scope** |
| Torch Euler16 | 16 | 211.8832 | .0220205 | .999802 | pass, slower |
| Torch Euler18 | 18 | 232.7558 | .0119571 | .999930 | pass, slower |
| Torch PF eps=.00005,max20 | 15/17/21 | 221.6153 | .0303350 | .999632 | pass, slower |
| Torch PF eps=.0001,max20 | 9/21 | 139.4244 | .104311 | .995595 | reject |
| Cached native Euler16 | 16 | 223.8490 | .0203627 | .999831 | pass, slower |
| Cached native Euler18 | 18 | 243.2042 | .0110135 | .999942 | pass, slower |
| Cached native Euler20 | 20 | 263.3780 | .00555848 | .999985 | pass, slower |
| Torch Euler20 bookend | 20 | 251.8910 | 0 | ~1 | pass |

All11complete raw bundles were fetched. Independent CPU checks covered every
saved action, request coverage, deterministic repeats, numerical metrics, full
candidate/instance/job identity, actual NFE/replay equality, one prefill/request,
and steady zero captures/builds/queries before strict ingestion. The ledger
recorded9validated/2rejected; bookend drift was-.08054%. The selected median was
23.94%lower than the graph Euler20 reference **within this four-case proxy**.
The139ms PF point was rejected rather than called the fastest acceptable policy.

The separate profile retained raw-velocity CUDA-event intervals and explicitly
unknown bound classification; missing prefix-stage evidence was not zeroed or
inferred from a residual walltime. No robot success, global parameter optimum,
cross-job performance ranking, or automatic default promotion is claimed.

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
