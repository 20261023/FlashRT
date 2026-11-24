---
name: vla-port-workflow
description: Port lane of the FlashRT VLA autotune loop. Use it to bring a method, kernel or backend into a FlashRT-native, GPU-resident, CUDA-Graph-capturable path for one model, stage and GPU. Examples are an SM80 INT8 build for pi0.5, Hy fused RoPE/QK-norm/KV-write on SM80, and a FastWAM per-request text cross-attention K/V cache. Covers the preflight tuple, private reference capture, execution ledger, operator gate for each new device path, capture acceptance (eager vs graph parity, repeated replay, changed-input propagation, stable addresses, host-transfer audit) and the functional-versus-optimization status split. It ends by handing a snapshot candidate with tool and stage ids to the patch lane. Use when a registry tool is needs_implementation, or when porting a reference implementation or arch path. Chinese triggers - 移植, port lane, 原生路径, 新架构适配, SM80 适配, 执行台账, 图捕获验收.
license: Apache-2.0. Complete terms in LICENSE.txt
metadata:
  upstream-repo: "https://github.com/infinigence/ApxInf"
  upstream-path: "skills/model-port-workflow"
  upstream-commit: "b05b40990387e8c48e7d6267a790664981a665e4"
---
<!-- Adapted by FlashRT (2026-11-15) from ApxInf skills/model-port-workflow/SKILL.md @ b05b40990387e8c48e7d6267a790664981a665e4 (Apache-2.0). Modifications: renamed; scope narrowed from whole-model ports into ApxInf crates to the FlashRT autotune port lane (one method/backend per model, stage and GPU); ApxInf docs, Rust module ownership, MLX/CPU and LLM-serving specifics replaced by FlashRT's repo contract, registry, frontends, pipelines, csrc bindings, CMake gates and snapshots; added registry tool ids, the operator gate, mandatory snapshot checklist and the hand-off into vla-patch-candidate; ledger and capture-acceptance checklists moved to references/. See NOTICE. -->

# VLA port workflow

The **port** lane brings a method or backend into a FlashRT-native path. Typical
ports are a precision route, a kernel that exists for another architecture, or a
cache that removes repeated per-step work. The other two lanes differ:

- **config** changes options on the same code tree and never uses a worktree;
- **patch** rewrites one hotspot behind a tensor contract.

A port usually needs a few limited source variants, a build change and new
acceptance evidence. It ends exactly like a patch: as a **snapshot candidate**
measured by the existing controller against a bookended reference. This skill
supplies the evidence that has to exist before that campaign makes sense.

## Mandatory snapshot checklist (B)

- [ ] Resolve the registered tool lane and compare every point's source/patch refs
  with the baseline. Non-config or unknown `Candidate.tool`, differing `source_ref`,
  or any non-null `patch_ref` (baseline included) requires `--snapshot-store`.
  Null source refs describe an implicit checkout; they do not inherit a pinned ref.
- [ ] Implement each source variant in a separate worktree, pin the 40-hex base
  commit, and store the real source diff as a content-addressed patch.
- [ ] Materialize and ready the baseline and every candidate, record native identity,
  and ensure the reference/candidates/bookend share one harness tree.
- [ ] Inspect plan schema 3, per-attempt `snapshot_id` and ledger manifests; require
  batch/worker import and source verification as part of campaign acceptance.
- [ ] Retain operator-gate, capture-acceptance and CPU controller/snapshot evidence.
  Rejected initialization must leave no plan or partial ledger records; log the actor.

Same-source unpatched registered config tools and pre-tool legacy config records
retain plan v2. `analysis.lever_pricing` is analysis, so its same-source config levers
still use plan v3 under the non-config rule.

Forbidden patterns: implementing a port in the live checkout; toggling source edits
in place between candidates; sharing a mutable variant directory; copying `.py`/`.so`
into verified snapshots; deleting `tool`/`source_ref`/`patch_ref` or assigning unknown
`config.*` names to obtain plan v2; missing baseline snapshots; schema downgrades,
fabricated `source_identity` or `PYTHONPATH` import bypasses. Harness edits belong
only to separately authorized infrastructure maintenance, never a port patch.

## Required reading

Resolve the checkout or candidate worktree you will modify first. Every path
below is relative to it. Then read:

1. `docs/adding_new_model.md`:
   - §0, the repository contract: one file per (model, hardware) pipeline and
     frontend, no runtime architecture forks, and `_PIPELINE_MAP` is one-to-one;
   - §4.3, kernel binding and CMake ownership;
   - §4.4, fvk names are ABI.
2. `docs/kernel_fusion.md` and `docs/kernel_catalog.md`, before adding any
   `fvk.*` entry. Reuse existing ones where you can.
3. `docs/vla_autotune.md`, for the campaign, workloads and the `--quality-gate`
   contract.
4. The registry entry for the tool you are implementing:
   `python -m flash_rt.vla.autotune.tools show <tool_id>`. Its `reason` field
   names the blocker the port must remove. The current port and port-like tool
   ids are in [references/port-checklists.md](references/port-checklists.md).
5. The `vla-model-impl-map` report for the affected stage. Write it first if it
   does not exist.

Read each document completely and treat it as instructions.

## Assign ownership before implementation

| responsibility | FlashRT placement |
|---|---|
| Capability, options and validation | `flash_rt/vla/registry.py`: `_CAPABILITIES` and `validate_config`. A new option must be admitted here, or the candidate cannot validate |
| Request IO, preprocessing, graph caches, static buffers, sampler wiring | the frontend, `flash_rt/frontends/torch/<model>_*.py` |
| Forward order and per-hardware compute | `flash_rt/models/<model>/pipeline_<hw>.py`. No `if arch == ...` forks inside a pipeline |
| Device operation | `csrc/` kernel plus its `m.def` in `csrc/bindings.cpp` (or a sibling bindings file). The binding must be gated exactly like the `.cu` |
| Build gate | `CMakeLists.txt`: `GPU_ARCH` lists and options such as `ENABLE_SM80_INT8_CUTLASS` or `FLASHRT_ENABLE_HYVLA` |
| Autotune harness, tool registry, stage map | `flash_rt/vla/autotune/`. **Never** part of a port patch, because snapshots refuse it. Update a tool's status separately, after the port is accepted and merged |
| Reference drivers, captures, reports | `devlocal/<feat>/` (git-ignored), or `examples/vla_autotune/` only when they are meant to ship |

A port that lives in an upstream checkout outside FlashRT, such as FastWAM
`source_dir`, is invisible to snapshot identity. Implement it on the FlashRT
side, in the pipeline or a FlashRT module wrapping the upstream module, or pin
the upstream checkout separately and say so.

## Workflow

1. **Preflight tuple.** Reuse the task's confirmed inputs. Default to
   hands-off for an implementation request unless the user asks for hands-on.
   Ask only for missing choices that change the port. Record the confirmed
   tuple, and never pick a user choice silently:
   - reference revision (FlashRT `base_commit`, plus the upstream revision
     when porting from an external source);
   - checkpoint identity;
   - target GPU and architecture (A800 is `cuda_sm80`, `GPU_ARCH=80`);
   - precision;
   - representative inputs, meaning the workload file and fixed noise;
   - tolerances: the operator gate thresholds and the model-level
     `--quality-gate`;
   - performance goal, which is best effort unless declared a release gate;
   - public API: `BackendConfig` options and frontend kwargs.

   Log it as a `decision` (`--action select_tool --tool <id> --stage <stage>`)
   with the real `--actor`.
   Completion: every tuple field and threshold is explicit.
2. **Private reference capture.** Run the reference privately: the current
   FlashRT path, or the upstream implementation (FastWAM
   `validate_fastwam.py --reference` is the direct-upstream mode). Capture
   deterministic inputs, outputs and diagnostic tensors with `capture_calls` at
   each affected call site. A baseline worker result carries
   `source_identity` in snapshot mode.
   Completion: the reference loads, and repeated captures are attributable to
   the same source, environment, weights and noise.
3. **Inventory semantics and weight transformations.** Cover scales,
   calibration, layout and repacking, op order (RoPE before QK-norm, say),
   cache layout (pre-expanded KV), eps, and what is constant per request versus
   per step.
   Completion: every required computation is understood independently of its
   framework operator name.
4. **Execution ledger.** Map reference semantics to device calls, one row per
   computation. Use the columns from
   [references/port-checklists.md](references/port-checklists.md):
   `Reference computation | Tensor contract | Frequency | Preferred call |
   Buffers and lifetime | Host traffic | Evidence`. Check maintained FlashRT
   paths and existing fused fvk entries before proposing new ones. Name the
   graph boundaries (which captured graphs, keyed how), the stable buffers, the
   input-update mechanism and the capture blockers.
   Completion: every row resolves to an existing call, a new device path with a
   named exit criterion, or a concrete blocker.
5. **Classify coverage and close gaps.** For a real kernel gap, either use the
   kernel lane (`vla-kernel-candidate`: capture, then ForgeKernel or
   KernelAgent) or write the kernel by hand. Either way, every new device path
   must pass `vla-operator-gate` against the captured references before it is
   wired in.
   - A host or CPU scaffold is a named correctness scaffold with an exit
     criterion.
   - A repeated hot-path host round trip inside a graph-captured stage is a
     blocker, not performance debt.
6. **Implement in a candidate worktree.** Never implement in the live
   checkout. Use `git worktree add --detach <dir> <base_commit>`, following
   the repository contract. Build for the target architecture:
   `python -m flash_rt.vla.autotune.snapshots build-script ...` renders the
   CMake job, and `record-built` or `adopt-native` records the result. For any
   binding or CMake change, validate every affected `GPU_ARCH`, as in
   `docs/adding_new_model.md` §4.3. Add a binding-level test when an argument
   mapping changes (§4.4).
7. **Public integration.** Admit new options in `registry.validate_config`,
   pass them through `load_backend` to the frontend, and keep
   `describe_backend` truthful. Nothing must fall back silently: an unbuilt
   symbol or unsupported architecture raises a clear error. The CPU suites
   `tests/test_vla_backend_*.py` and `tests/test_vla_action_processor.py` must
   still pass.
   Completion: registry validation, frontend construction and public `infer`
   agree in the same change.
8. **Verify.** Verify operators (the gate), weight transformations,
   intermediate checkpoints against the private reference, eager and captured
   inference, and public `VLABackend.infer`. Then run **capture acceptance**
   (checklist in the references): eager versus graph parity, repeated replay,
   changed-input propagation, stable addresses, capture-safe workspace, and a
   host-transfer audit. The host-transfer audit lists every `.item()`, `.cpu()`
   and synchronize per request and per step, before and after the port
   (`perf-torch-sync-free`, `perf-torch-cuda-graphs`, `perf-nsight-systems`).
   Report two statuses separately:
   - **functional acceptance:** pass or fail;
   - **optimization status:** `target met`, `best effort with performance
     debt`, or `blocked`.

   Performance is best effort unless declared a release gate, but you must
   investigate the existing optimized paths that apply.
9. **Hand off to the patch lane.** Produce a diff that contains only product
   changes: no devlocal artifacts, no harness edits. Build the proposal from
   the worktree with
   `tools.patches.proposal_from_worktree(worktree, base_commit, store=STORE,
   tool_id=<registry id>, stage_id=<primary stage>, hypothesis=..., options=...)`.
   Then turn it into a candidate with `candidate_from_proposal(baseline,
   proposal, validate=False)`; use `validate=False` only when the patch adds
   options the base registry does not know. From there `vla-patch-candidate`
   materializes the snapshot and runs the reference/candidates/bookend
   campaign. Attach the ledger, the gate report and the capture-acceptance
   report as evidence refs.
   Completion: a snapshot candidate exists with `tool` and `stage` set, and
   its evidence is linked in the decision log.

## Stop conditions

In hands-off mode, intermediate builds, numerical checkpoints and uncommitted
worktree changes are continuation points. Stop only at:

- the completion criteria;
- a concrete blocker;
- an approval you cannot grant, such as a GPU allocation, a protected-path
  change, or a push.

In hands-on mode, pause at the named checkpoints. Each pause asks a concrete
question and proposes a default next action.

Stop with a concrete blocker when:

- the reference cannot run;
- semantics remain unknown;
- the operator gate fails with no correct path;
- the build gate cannot be satisfied on the target architecture;
- the public `load_backend` path cannot be exercised;
- a graph-captured stage would need a per-step host round trip.

Missing native GPU coverage is a blocker. It does not authorize a CPU partition
or an external engine in the measured path; ApxInf (`engine.apxinf`) is a
baseline lane, not a port target. A performance gap alone is not a stop
condition unless performance is a declared gate: exhaust the applicable existing
paths, measure the gap in the campaign, and report the remaining debt.

A partial foundation is not a completion condition. While safe in-scope work
remains, continue through integration, capture acceptance and the hand-off,
rather than ending with a list of unfinished components.
