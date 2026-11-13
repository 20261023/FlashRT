# Runtime-backed VLA experiment loop

`flash_rt.vla.autotune` is a small controller/worker layer above the
[execution backend](vla_backend.md), not a new inference engine or serving daemon.
See [measured iteration notes](vla_autotune_experiments.md) for scoped outcomes,
including failed candidates, and [source-checkout examples](../examples/vla_autotune/README.md)
for validation and explicitly experimental graph/diagnostic runners.

For the consolidated Chinese report covering all four models, measured outcomes,
rejected candidates and acceptance limits, see
[中文阶段报告](reports/2026-11-09-vla-autotune-report.zh-CN.md).

```text
Candidate + workload + dataflow hypothesis + source/patch reference
 -> SQLite controller ledger + explicit submission intent
 -> one Slurm allocation, serial isolated worker processes
 -> raw metrics/arrays/NFE + separate profiling
 -> scoped quality gate + reference bookend check
 -> retain/reject + next hypothesis
```

## Modules

| Module | Responsibility |
|---|---|
| `schema` | Explicit backend codec, immutable candidate records, bound v2 result envelopes |
| `ledger` | Single-controller SQLite transactions, state CAS, restart recovery, idempotent ingestion |
| `search` | Conditional Euler/ProbeFlow enumeration and scoped quality-gated median selection |
| `quality` | Exact channel/threshold contract and controller-side metric budget verification |
| `slurm` | Validated, site-compatible sbatch rendering |
| `scheduler` | Explicit local/SSH Slurm transport; local-cluster queue queries and full accounting states |
| `workloads` | Explicit NPZ field/noise mappings and fresh request arrays |
| `worker` | One candidate; load/prepare/warmup/steady separation and raw outputs |
| `batch` | Serial subprocesses in one allocation, reference/candidate/reference ordering |
| `profiling` | An extra diagnostic request with graph/evaluator event intervals |
| `dataflow` | Source-specific FLOPs/logical-weight accounting using actual runtime rows |
| `controller` | Campaign creation/reopen, submission-state recording, ingestion and selection |

The coding agent remains outside the measurement process. A proposal uses
`origin="agent"`, `parent_id`, `hypothesis`, `patch_ref` and `source_ref`; human and
deterministic-search proposals use the same interface. The initial search does
not train a cost model, recursively call an LLM or author skills.

## Explicit workload

```json
{
  "schema_version": 1, "workload_id": "pi05-fixed-input", "model": "pi05",
  "requests": [{
    "request_id": "frame-0", "npz_path": "inputs.npz",
    "fields": {"images": "rgb", "state": "normalized_state"},
    "noise_key": "noise", "prompt": "pick up the block"
  }]
}
```

Fields map frontend observation names to array keys; a list constructs a camera
list. No universal robot preprocessing is guessed. File IO and request host
copies are outside timing; frontend processing/transfers/model/routing/completed
CPU output remain inside. This is not HTTP or robot-control latency.

## Use

```bash
python -m flash_rt.vla.autotune.controller init \
  --root campaign-r1 --comparison-id r1 --baseline baseline.json \
  --workload /remote/workload.json --output-root /remote/results/r1 \
  --enumerate --warmup 5 --repeats 20 \
  --quality-gate '{"scope":"normalized-action-replay","action_dim":7,"max_relative_l2":0.05}'
# Explicit submission records intent before calling sbatch. Do NOT first call
# `submitting` when using `submit`: it owns that transition itself.
python -m flash_rt.vla.autotune.controller submit \
  --root campaign-r1 --script /remote/round1.sbatch --ssh-host gpu
python -m flash_rt.vla.autotune.controller wait \
  --root campaign-r1 --ssh-host gpu --interval 15 --timeout 14400
# Inside one allocated compute step, never on the login node:
python -m flash_rt.vla.autotune.batch --plan /remote/plan.json
# After fetching the output tree and confirming job completion:
python -m flash_rt.vla.autotune.controller ingest \
  --root campaign-r1 --results fetched-results --job-finished
```

`baseline.json` is a `Candidate.to_dict()` record. `--proposals` accepts a JSON
list of extra human/agent candidates. `submit` and `--ssh-host` explicitly opt
into scheduler/SSH actions; other commands never submit. `wait` only monitors
confirmed IDs and persists scheduler state, atomically promoting queued attempts
to running. A terminal scheduler state does not ingest, validate, reject or rank
model outputs; timeout is resumable and does not fail the worker attempts.

The ledger has `submission_unknown` for lost scheduler responses. Reconcile an
externally confirmed ID with `queued --root campaign-r1 --job-id 12345`, which is
idempotent for that same job; never automatically resubmit unknown submissions.
The low-level `submitting`/`queued` pair remains available for externally managed
submissions. Initialization and group transitions are atomic; failed proposal
validation cannot leave a partially initialized campaign. Retries use new IDs.
This is a single-controller tool, not a distributed scheduler.

## Execution identity and historical results

Each controller ledger owns a persisted random `campaign_instance_id` (not a
content hash). New plans and worker result envelopes are version2. A result
contains the full Candidate constructor record, including backend/sampler/options,
checkpoint and source/patch references, plus the actual Slurm job ID. Ingestion
checks these against the ledger's immutable candidate and confirmed job; the GPU
job must agree too. All supplied envelopes are preflighted before importing any
prefix of the group. Reusing readable IDs in another campaign cannot attach its
old bundles to the new execution.

Version1 records remain readable historical evidence, but the current controller
does **not** ingest them as newly qualified results or invent missing identity
fields after the fact. In-flight old snapshots still emit their original format;
inspect those artifacts as historical measurements and use a new bound attempt
for qualification under the stronger contract. This is provenance consistency,
not content-hash attestation; source snapshots and input artifacts must remain
unchanged during their experiments.

## Comparison meaning

- Fresh worker processes prevent retained model/graph residency across candidates.
- Slurm completion alone is insufficient. Missing bundles are incomplete/failed,
  never zero-latency successes. Workers do not write the controller DB.
- Raw action arrays are retained. `action_dim` explicitly selects trained/public
  channels; padded untrained channels are not silently used as a quality target.
- `repeats` is the total measured request count, not repetitions per request.
  It must be at least the number of manifest requests, and saved request IDs must
  cover the whole workload. Prepare/capture runs do not count as quality evidence.
- Quality is passed/failed/unknown in an **exact scope**. Numerical replay
  L2/cosine/max-absolute metrics are proxies, not task success. The initial
  baseline is explicitly a reference role, not an independent accuracy oracle.
- Budgets such as relative L2 0.05 are caller-selected engineering choices, not
  universal VLA tolerances. Unknown/failed quality cannot win.
- The controller requires exactly its configured scope, action channels and
  threshold contract, then rechecks passed metrics against its own budgets.
  A passed result from a looser worker plan is rejected, even with the same scope
  name. This applies to reference bookends as well as candidates.
- Mismatched workload/comparison/GPU/action domains reject grouping. Incomplete
  campaigns have no winner. Baseline bookend median drift over 10% makes the
  comparison inconclusive rather than choosing the most optimistic run.
- Load, prepare and steady samples are separate. Actual NFE and mapped clocks
  are preserved in `measurement.sampler_metadata`; top-level aliases remain.
- Per-request native enqueue/Linear counts and raw-field graph counters remain
  complete in `execution_metadata`, not the worker's immutable backend-identity
  comparison. Engine, graph-mode flag, native implementation and installed module
  coverage still must remain unchanged; growing replay totals are not a backend
  substitution.
- Failed cleanup retains samples as partial evidence, not valid measurement.

### Hy performance/quality objective (2026-11-13 revision)

Hy now follows the same optimization objective as the other models: minimize
end-to-end latency within an explicit final-action deviation budget, rather than
reproduce old-runtime intermediate tensors. The source-checkout contract is
[`hyvla_tradeoff_quality.json`](../examples/vla_autotune/hyvla_tradeoff_quality.json):
relative L2 <=0.05 and cosine >=0.999 on the normalized **20 trained/public
channels**, for every request and measured replay. The12padding channels do not
veto selection. Maxabs and the original elementwise official comparison may be
reported separately, but are not this optimization objective.

Use a newly measured, fixed native BF16 `semantic_bf16` Euler10 reference and
bookend on the same source/checkpoint/runtime/workload/noise. Like the other
campaign references, this measures deviation from an execution baseline, NOT
independent official correctness or robot success. Freeze the new scope before
running candidates; all seven existing Hy cases remain in coverage. Select the
fastest passed point using the existing controller, retaining raw arrays, actual
NFE, median/tail latency, source identity and reference drift.

LSE/FMA/SASS and old-runtime EXACT research are no longer optimization blockers
or the active workstream. No global production precision/default was changed.
Historical `validate_hyvla.py` oracle budgets and prior failed results remain
unchanged; a new criterion must never retroactively relabel them as passed.

## Profiling and dataflow

Profiling adds a request outside formal samples. CUDA event intervals can include
submission/dependency gaps; they are **not kernel-duration sums** or evidence of
a compute/memory bound. Missing stage evidence remains unknown, not zero.

The pi0.5 ledger records actual prefix/action rows with source-verified matrix
geometry, FLOPs and logical weights, not measured HBM bytes. Fewer NFE change
repetition count, not each head's GEMM shapes. Post-ViT pooling changes prefix M,
not the ViT work already completed. A new proposal should state which dataflow it
expects to change and be retained or rejected using the measured evidence.
