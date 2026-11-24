# VLA diagnostic profiling contract

`flash_rt.vla.autotune.profiling.profile_request` observes a **separate warmed
request**, never a formal latency sample. Model discovery, stage ownership and
expected interval keys come from `model_spec.get_spec(model, spec_dir=...)`.
An external JSON declaration does not require a core model-name branch.

## Declaring an observation

Use `adapter.hooks.profile = "python.module:install"` for an adapter that receives
a `DiagnosticSession`. Its wrappers are temporary instance attributes/items,
restored on both success and failure. Alternatively, `profiling.strategy` can
declare `methods` or `graphs`, using frontend-relative paths such as
`pipeline.predict` or `profiles.*.graph`. Missing paths remain unknown.

`profiling.keys` maps every request interval to its declared stage ids; the
inverse `stages[].profiling_keys` must agree. An external stage id also declares
`kind` (`compute`, `host`, `count`, `kernel`, `precision`) and optionally `title`.
The `other` / `kind=regression` action head needs no artificial denoising NFE.

Method/graph strategies may explicitly set `additive: false`. The observation is
retained, but it cannot establish request time or stage coverage. This is useful
for an overlapping host helper, not permission to invent its stage mapping.
Optional dataflow hooks return static source/runtime ledgers, not measured HBM
traffic or roofline evidence.

## Five builtin adapter boundaries

| Adapter | GPU event intervals | Host wall intervals and remaining limits |
|---|---|---|
| π0 / π0.5 Euler | Private diagnostic capture: `vision`, `prefix_only`, one `velocity` per original decoder step | `host_input` before `forward`; `host_output` after completed compute. Velocity includes original fused integration. |
| π0 / π0.5 ProbeFlow | Production `prefix` graph plus actual `velocity` replays | Prefix still groups vision/projector/prefill. Adaptive routing remains residual. |
| Hy | `vision` (ViT + merger), `prefix`, actual `velocity` replays; eager source bodies are also supported | Input before ViT and between ViT/flow; output after completed flow. Routing/integration and explicit drain waits remain residual. |
| DM0.5 | Private diagnostic capture: `vision`, `prefix` starting **after** nested vision returns, ten raw `velocity` intervals | Input before the selected exact-prefix profile's `run`; output after completed compute, including D2H and action reconstruction. Integration remains residual. |
| FastWAM | Separate first-frame VAE `vision`, `condition_tokens`, video `prefix`, optional context/mask/text-KV helpers, actual `velocity` calls | Input validation/noise/context staging helpers; output starts after the final finite check. Untimed bookkeeping/routing remains residual. Compiled prefill is timed at its compiled callable, not an unused eager method. |

π Euler and captured DM splits require supplied noise and bit-exact public
output parity against the original graph. Otherwise the adapter retains only
whole-graph evidence (`full`), without inventing a split. Private diagnostic
graphs own their own resources and never reset/adopt the production graph.
External event timestamps must be newer than the request epoch; setup/capture
timestamps cannot be reported as observed request work.

## Clocks, additivity and reproducible coverage

The historical `measurement_kind="diagnostic_cuda_event_intervals"` envelope is
kept for compatibility, but every new interval has `interval_provenance`:

- `device="cuda", clock="cuda_event"` measures a **stream interval**, including
  submission/dependency gaps, not a sum of kernel durations.
- `device="cpu", clock="perf_counter_ns"` measures a **host wall region**. It can
  include that region's own IO/validation waits; it is not CPU-instruction time.
  Builtin host boundaries drain preceding compute outside the IO interval.
- `additive=true` contributes to request attribution.
- `additive=false` is returned under `nonadditive_diagnostics`, not `groups`.
  `scope="operator_probe"` additionally requires `additive=false` and is returned
  separately under `operator_probes`. Probes never establish request coverage.

New records retain `interval_spans_ms` and name the CPU/GPU epoch domains.
Attribution checks non-overlap **within each clock domain**, original interval
sums, stage ownership, and parent/child aggregate double counts. CPU/GPU clocks
are not interchangeable; builtin disjoint boundary semantics remain part of the
adapter contract. External adapters must provide truthful disjoint ownership
or mark an observation non-additive.

```bash
python -m flash_rt.vla.autotune.stages attribute result.json --rank
# For an external model, use the record's BackendConfig.spec_dir or --spec-dir.
```

`coverage_evidence` records additive/excluded keys, expected/missing keys,
`attributed_ms`, `wall_ms`, `residual_ms`, `fraction`, `within_wall` and the
span-validation status. The arithmetic is:

```text
attributed_ms = sum(groups[].ms)             # never sum probes/overlays
residual_ms   = diagnostic wall_ms - attributed_ms
fraction      = attributed_ms / diagnostic wall_ms
```

Residuals and fractions are **not clamped**. A fraction above one is inconsistent
time coverage, not an improvement. `coverage="full"` means the declared compute
stages have interval evidence; it does **not** mean 100% of wall time, all host
work, a separable operator decomposition or hardware qualification. Onboarding
must additionally validate its numeric wall-coverage threshold or its bound
measured residual explanation.

CPU fixture tests verify declaration routing, event/host ownership, restores,
step order, parity failures, stale setup events and coverage arithmetic. Their
synthetic event durations are **not GPU latency measurements**. Real graph
capture, replay parity and hardware coverage remain unverified until a permitted
Slurm GPU run under the frozen unified harness.
