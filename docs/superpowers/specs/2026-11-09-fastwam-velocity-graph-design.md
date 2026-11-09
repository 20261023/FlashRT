# Opt-in FastWAM raw-velocity graph

## Intent and evidence

Reduce repeated host submission without changing the model, solver or defaults.
The A800 Euler2 pilot preserved each engine exactly on six same-weight input
controls. Upstream+graph was faster than partial-native+graph. This motivates a
safe execution choice, not a claim that native is always faster, official parity
between engines, or robot-quality qualification.

The user authorized continued autonomous iteration and ordinary bounded sbatch
validation while unavailable, and frequent verified git-only checkpoints to the
existing fork. The following implementation choices are the builder's calls,
not user-prescribed interfaces. No new default, gate relaxation or new repository.

## Scope and exclusions

- Add `use_cuda_graph=False` to the existing frontend and pipeline; opt-in only.
- Allow explicit `BackendConfig(model='fastwam', backend='torch')`, with honest
  descriptors/results and no native extension load. Other models stay FlashRT.
- First graph combinations: Torch operators, or native Linear backend `cached`.
  Reject native legacy+graph before filesystem/device access; keep eager legacy.
- Capture only the seven-tensor-argument non-integrating raw-velocity head.
  VAE, per-request video prefill, all sampler decisions/integration and CPU output
  remain outside capture and inside whole-infer timing.
- One graph/geometry per pipeline, no automatic recapture, fallback, compiler,
  multi-geometry cache, Slurm guard in the library, extra hashes, weight copies,
  new model math, or global memory/cache operations.

## Ownership and interfaces

`flash_rt/models/fastwam/velocity_graph.py` contains an internal
`FastWAMVelocityGraph(fn, *, capture=None)` and typed
`VelocityGeometryMismatch(ValueError)`. `capture` is a contract-test injection
point, not a public CPU execution option. Production admits same-device CUDA
Tensor leaves only; the module import itself remains lightweight.

Methods: `begin_request()`, `__call__(**seven_arguments)`, `end_request()`,
`snapshot()`, `close()`. Calls outside an active request and nested requests
reject. Every first request evaluation refreshes context/masks/video KV, even
when caller pointers are unchanged. Every evaluation refreshes x/time.

Geometry compares exact tree structure, Tensor shape/stride/dtype/device; no
hashing. Static tensors use a checked layout-preserving allocation; unsupported
overlapping layouts reject instead of silently changing stride. Geometry mismatch
does not replay, fallback or discard the valid graph. Outputs are borrowed scratch
valid only until the next evaluation. PF's existing cloned first probe and
Euler's multiply-then-add order remain unchanged.

Pipeline owns the driver and one non-default execution stream. All graph-enabled
pipeline operations, including prefill and public eager velocity evaluation, run
serially under its existing RLock and on that stream. Entry waits for the caller
stream and records incoming CUDA Tensor lifetimes; exit orders the caller after
the owned stream. No other pipeline path may race its native runner workspace.
CPU tests inject stream/capture boundaries only where CUDA is unavailable.

The pipeline routes a private evaluator, never monkeypatches the upstream model.
Standalone `evaluate_velocity` stays eager and serialized. `infer` brackets graph
request lifetime with `try/finally` around the unchanged sampling loop.

## Failure and cleanup

Capture publishes only a fully validated graph/output. On runtime capture/replay
failure, reject later inference; cleanup remains possible. Partial graph owners
must be retained if release fails, so `close` can retry. Geometry errors are
admission failures, not destructive runtime faults. `close` rejects further work
before teardown, synchronizes owned work, resets only owned graph, then drops
static buffers/callable/model/native references. Failed reset retains references
and can be retried. Successful close is idempotent. An externally held injected
model is not destroyed or modified by cleanup.

Frontend gains `close()` forwarding to pipeline, clearing its model only after
pipeline close succeeds. `VLABackend.close()` already forwards cleanup and keeps
the frontend on failure. No `empty_cache` or global graph-pool operation.

Initial capture uses PyTorch global capture safety and requires a quiescent CUDA
process. A pipeline instance lock cannot coordinate arbitrary other GPU threads;
document explicit `prepare` before concurrent serving. Do not weaken capture mode.

## Metadata and qualification

Keep sampler NFE as actual logical field evaluations. Retain existing counters
but label Python Linear forwards and successful CPU GEMM enqueue counts as such.
Separately report per-request replays, condition refreshes, captures, cold warmup
evaluations and encoded bodies; dynamically observe native Linear calls encoded
per body and derive logical Linear work in replay. Do not hardcode307 or say that
fewer enqueues mean less GPU arithmetic. CPU output is a complete copied result,
never graph scratch aliased into a later request.

Selected CPU contracts and a clean staged export must pass before checkpoints.
Full-repository optional/native collection failures stay explicit. Before a
runtime-qualified claim, run actual A800 Euler10 and adaptive PF with original
and changed inputs, graph/eager equality per engine, same noise, actual NFE,
prefill count, cold/steady counters and cleanup. Diagnostic tests do not replace
quality acceptance; broader geometry/hardware and robot success stay unqualified.

Reference: PyTorch2.11 CUDA semantics (fixed-address storage, side-stream warmup,
no concurrent non-captured CUDA work during capture, explicit stream ordering).
