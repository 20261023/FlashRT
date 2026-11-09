# FastWAM Velocity Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose a safe opt-in graph for the existing FastWAM raw field without changing the solver, defaults or quality gates.

**Architecture:** A pipeline-owned single-geometry driver refreshes fixed-address
inputs and owns its capture/output lifecycle. An owned ordered stream serializes
prefill, graph and integration; explicit Torch and partial-native cached paths
share this orchestration while retaining truthful metadata.

**Tech Stack:** Python, Torch2.11 CUDA graphs, existing NumPy/PF and pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-fastwam-velocity-graph-design.md`.

## Global Constraints

- Default eager/legacy behavior unchanged; no sampler or model math changes.
- Capture only raw field; no upstream monkeypatch, Slurm guard or global flush.
- One geometry, typed mismatch, no recapture/fallback/hash/weight copy.
- Graph supports Torch and cached native only; reject legacy+graph statically.
- Borrowed head output; actual sampler NFE separate from cold encoding/enqueues.
- Tests precede code; selected clean-export verification and git-only checkpoints.
- Actual A800 Euler10/PF/multi-input execution required before runtime claims.

## Review Focus

- Same pointers with changed condition values refresh every request (Task1).
- Non-dense/overlapping or changed leaf layouts do not produce stale replay (Task1).
- Failed cleanup retains graph and model references for retry (Tasks1/2).
- Caller stream changes cannot race the cached runner workspace (Task2).
- Backend='torch' never requests native execution or returns a FlashRT label (Task3).

### Task 1: Owned raw-velocity graph contract

**Files:** Create `flash_rt/models/fastwam/velocity_graph.py` and
`tests/test_fastwam_velocity_graph.py`.

**Interfaces:** Consumes seven head kwargs from the current pipeline; produces
`FastWAMVelocityGraph(fn, *, capture=None)`, `begin_request()`, `__call__(**kwargs)`,
`end_request()`, `snapshot()`, `close()`, `VelocityGeometryMismatch`.

- [ ] Write behavior tests: same-pointer refresh; time/x updates; end/nesting
  admission; borrowed output alias and explicit clone preservation; shape/stride/
  dtype/list mismatch; unsupported overlap; capture/output/replay/release failures;
  retry-safe idempotent close; dynamically derived cold and replay counters.
- [ ] Run `cpu-env/bin/python -m pytest -q tests/test_fastwam_velocity_graph.py`.
  Expected missing module/API RED. CPU tests inject capture only, no fake CUDA
  qualification; real production path must reject CPU tensors.
- [ ] Implement checked tensor trees and owned capture resource; use side-stream
  capture on pipeline's current stream, three warmups, global capture safety,
  strong references and no publication until output validation succeeds.
- [ ] Rerun the test command. Expected all pass, no CUDA initialization.
- [ ] Commit only driver/tests with actual test evidence; push existing fork.

### Task 2: Pipeline ownership, ordered execution and frontend cleanup

**Files:** Modify `flash_rt/models/fastwam/pipeline.py`,
`flash_rt/frontends/torch/fastwam.py`; extend `tests/test_fastwam_adapter.py` and
`tests/test_fastwam_velocity_graph.py`.

**Interfaces:** Consume Task1 driver. Produce frontend/pipeline option
`use_cuda_graph: bool=False`, explicit `close()`, graph execution metadata;
standalone `evaluate_velocity()` remains eager/locked.

- [ ] Write tests for static option errors before paths/device probe, unchanged
  default, no model method replacement, Euler10 and PF against eager using real
  CPU toy model plus injected capture, per-request prefill/refresh, NFE and alias
  safety, finally-ended request on error, stream ordering and retry cleanup.
- [ ] Run those test files; expect unsupported option/API or wrong metadata RED.
- [ ] Add driver dispatch/request scope, one owned non-default execution stream,
  current-stream waits/lifetime recording and shared lock for every public model
  operation. Preserve eager code paths and arithmetic. Close graph before model
  and runner references; frontend clears its model only after successful close.
- [ ] Rerun focused files and selected VLA/Fast/PF suite; expect all green.
- [ ] Commit/push the independently tested integration (not runtime-qualified).

### Task 3: Honest backend selection and measured integration validation

**Files:** Modify `flash_rt/vla/{config,registry,backend}.py`,
`tests/test_vla_{config,registry,backend}.py` as applicable, Fast graph example
runner/docs, `docs/vla_backend.md`.

**Interfaces:** Consume Task2 `use_cuda_graph`; produce FastWAM-only
`backend='torch'`, native flag selected from backend, accurate descriptors and
return metadata; no change to Candidate identity or numerical gates.

- [ ] Write tests for Torch Fast route/metadata, default partial-native route,
  non-Fast Torch refusal, Torch/native-option conflicts and legacy graph refusal
  before filesystem/CUDA access. Expect current FlashRT-only RED.
- [ ] Add minimal config/registry routing and metadata; extend runner with explicit
  Euler10 and adaptive PF configuration, preserving independent upstream oracle.
- [ ] Run focused tests, full selected suite, bare pytest (report dependency
  failures by name), whitespace check and staged-only clean export.
- [ ] Commit/push, then create new immutable source snapshot and run bounded
  single-A800 serial eager/graph for each engine, same fixed noise and changed
  noise/proprio/image/context/mask. Require same-engine exact arrays, actual NFE,
  one prefill/request, counter separation and successful cleanup. Do not alter
  failed snapshots or describe this as robot-quality acceptance.
- [ ] One fresh whole-change review; concrete Important fixes use RED->GREEN.
  Preserve failures/unknowns and commit/push factual evidence/docs.
