---
name: vla-stage-profile
description: Profile a FlashRT VLA candidate (pi0, pi0.5, Hy-Embodied-0.5-VLA, FastWAM), attribute its latency to pipeline stages with flash_rt.vla.autotune.stages, compute Amdahl ceilings and pick the bottleneck stage before choosing a tool. Use when asked where the time goes, which stage to optimize next, why fewer steps did not help, whether a stage is worth a kernel, or when stage coverage is partial or none. Escalates to Nsight Systems, Nsight Compute, CUDA events or sync audits only for a few kernels. Chinese triggers - 测 profile, 时间归到哪一游, 瓶颈在哪, Amdahl 上限, 选 stage.
license: Apache-2.0
metadata:
  short-description: "Attribute VLA latency to stages and pick one"
---

# VLA stage profile

Goal: turn one measured result into a ranked list of stages with known shares,
decide whether any stage is worth investing in (Amdahl ceiling first), and log
who picked the stage. This skill never chooses a tool or a winner; it hands one
stage to `vla-config-sweep`, `vla-patch-candidate` or `vla-kernel-candidate`.

Shell prefix used below (local checkout, CPU only):

```bash
cd <flashrt checkout>
py() { env UV_CACHE_DIR=/tmp/autotune-uv PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy python -B "$@"; }
LOG=<campaign root>/decisions.jsonl          # one append-only log per optimization thread
```

On the cluster use the runtime's own `python -B` instead of `py`.

## Rules carried from the design

- Profiling is a separate diagnostic request outside the formal samples. CUDA event
  intervals include submission and dependency gaps: they are not kernel-duration sums
  and not evidence of a compute or memory bound.
- Missing evidence is unknown, never zero. `coverage: none|partial` is a reason to plan
  an observation, not to guess.
- A diagnostic run (ProbeFlow split, eager run, nsys) is not a candidate. It never
  enters winner selection; label it `diagnostic:` in its hypothesis.
- Amdahl ceiling before investment: ideal total speedup is `1 / ((1-p) + p/s)`. It ranks
  priorities only. Without a measured `p`, the ceiling is unknown.
- When the evidence cannot separate two hypotheses, write the minimal discriminating
  observation instead of declaring a bottleneck.

## Step 1. Get a profiled, measured result

1. Every campaign worker profiles unless disabled: `batch` never passes `--no-profile`,
   and `worker.py` only skips profiling with `--no-profile`. So the reference attempt of
   any campaign already holds `record['stage_metrics']`. The file is
   `<output_root>/<attempt_id>/result.json` (fetch the tree, then `controller ingest`).
2. If you have no campaign yet, run a reference-only campaign through `vla-config-sweep`
   (baseline only, no proposals) and use its reference attempt.
3. Check the record before attributing:
   - `state == "measured"`; a failed record carries `failure`.
   - `profiling_failure` present -> attribution is `coverage: none`; read the message.
   - pi0/pi0.5/Hy with `use_cuda_graph=False` record no intervals (only graphs are wrapped).
     FastWAM is method-wrapped (`_prepare_condition`, `_evaluate_clock_velocity`) and records
     intervals in eager mode too, which is its default.
4. Log the evidence reference (actor = whoever ran or chose the run):

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind note --actor agent \
  --summary "profiled reference for <model> on cuda_sm80" --model <model> \
  --evidence-ref <output_root>/<attempt_id>/result.json --text "workload <id>, campaign <cid>"
```

## Step 2. Attribute and rank

```bash
py -m flash_rt.vla.autotune.stages attribute <result.json> --rank > attribution.json
# --model M only when the record has no candidate.
py -m flash_rt.vla.autotune.stages list --model <model>     # seams + notes per stage
```

Read, in this order: `coverage`, `notes`, `rank` (shares incl. `unattributed`),
`per_call_ms["denoise.velocity"]`, `nfe`, `unattributed_ms`, `residual_stages`.

Interval keys and what they can and cannot separate:

| model | keys | coverage limit |
|---|---|---|
| pi0 / pi0.5, Euler + graph | `full` | one interval for vision+prefix+velocity: unsplittable. See Step 3. |
| pi0 / pi0.5, ProbeFlow | `prefix`, `velocity` | prefix includes vision.encode and vision.project |
| hyvla | `vision`, `prefix`, `velocity` | `vision` = ViT + merger graph (vision.encode + vision.project); old records without it are `partial` |
| fastwam | `condition_vae_video_prefill`, `velocity` | VAE encode and video prefill share one interval |

Everything else (host.input, host.output, denoise.routing, host.launch) lands in
`unattributed_ms`. A negative residual is reported as-is with a note; do not clamp it.
A note that the velocity interval count differs from `nfe` means per-call time is an
average over a different count; say so in the diagnosis.

## Step 3. Close coverage gaps with the smallest observation

- pi0/pi0.5 Euler `full` only: do not split it by hand. Run one ProbeFlow diagnostic
  candidate (same checkpoint, workload, graph on) whose record has `prefix` + `velocity`;
  that gives the prefix vs per-call velocity split under the same graph boundary.
  Alternatively an eager diagnostic with CUDA events / NVTX (`perf-workload-profiling`),
  remembering eager adds launch gaps the graph path does not have.
- FastWAM: if VAE vs video prefill matters, the next observation is a timeline
  (`perf-nsight-systems`) or CUDA events around `_prepare_condition` internals, not a
  new candidate. `examples/vla_autotune/fast_profile/profile_fastwam.py` is the
  existing FastWAM diagnostic runner.
- Large `unattributed_ms`: host work or routing. Look for host syncs (ProbeFlow cosine
  `.item()`, `.cpu()`, finite checks) with `perf-torch-sync-free`, and for timeline gaps
  with `perf-nsight-systems`.

Log every gap as a diagnosis before running anything:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind diagnosis --actor agent \
  --summary "pi05 Euler graph interval is unsplittable" --model pi05 --stage prefix.prefill \
  --symptom "coverage partial: only 'full' interval" \
  --hypothesis "prefix dominates the full interval" \
  --alternative "velocity x NFE dominates" \
  --next-observation "one ProbeFlow diagnostic run with prefix+velocity intervals" \
  --candidate-tool profile.stage_intervals --invalid-when "different checkpoint or views" \
  --confidence low --provenance mechanistic_inference --evidence-ref attribution.json
```

Human checkpoint: spending a GPU allocation on a diagnostic run. The human confirms
the run is allowed and on which partition.

## Step 4. Amdahl ceilings before investing

For each ranked group with share `p`, compute ceilings for an optimistic and a
realistic local speedup:

```bash
py -c "from flash_rt.vla.autotune.stages import amdahl_ceiling as a; p=<share>; print({s: round(a(p, s), 3) for s in (1.2, 1.5, 2.0, 4.0)})"
```

- `denoise.schedule` is a count: fewer steps scale the velocity group by
  `new_nfe / old_nfe`, plus any new probe/routing cost. Use the velocity share as `p`.
- Cross-cutting stages (`kernel.*`, `host.launch`, `precision.weights`) have no interval
  of their own. Their `p` is a fraction of the compute groups and needs kernel-level
  evidence (Step 6) before it is anything but `unknown`.
- If the ceiling at the optimistic `s` is below the required gain, drop that stage and
  record why. If no stage can reach the target, report that (decision `stop`) rather
  than searching.

## Step 5. Pick the stage and log who picked it

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent \
  --action select_stage --stage denoise.velocity --model fastwam \
  --summary "denoise.velocity: largest measured share, ceiling <c>x at s=2" \
  --rationale "velocity share <p> measured_here; prefix share <q>; schedule already minimal" \
  --evidence-ref attribution.json
```

- `--actor agent` when the agent chose alone, `human` when the human named the stage,
  `human+agent` when the agent proposed and the human confirmed.
- If the human overrides the agent's stage, log `--action redirect
  --redirected-from <the agent's stage>`; redirect needs actor `human` or `human+agent`.
- Human checkpoint: the target metric (median vs tail, which boundary) and the quality
  budget come from the human before the first stage is selected.

Then list the tools for that stage and hand over to a lane skill:

```bash
py -m flash_rt.vla.autotune.tools list --model <model> --hardware cuda_sm80 --stage <stage> --all
```

Config tools -> `vla-config-sweep`. Patch/port tools -> `vla-patch-candidate`.
`kernel.*` tools and hot single operators -> `vla-kernel-candidate`. `compile.aitune`
(Torch modules of FastWAM; Hy only after a packaging patch) -> `aitune-inspect`, then
`vla-patch-candidate`.

## Step 6. Kernel-level evidence, only for a few kernels

Escalation order: static dataflow and existing records -> stage intervals -> a short
timeline -> hardware counters on a handful of kernels. Never collect everything.

| question | skill |
|---|---|
| which code runs in this stage, which graph caches, which fvk binding | `vla-model-impl-map` |
| CPU/GPU timeline, launch gaps, stream overlap, host waits | `perf-nsight-systems` |
| why one kernel is slow (compute, memory, occupancy, stalls) | `perf-nsight-compute-analysis` |
| CUDA events / NVTX ranges around a region, eager splits | `perf-workload-profiling` |
| host syncs (`.item()`, `.cpu()`, NumPy on device tensors) | `perf-torch-sync-free` |
| graph capture problems, recapture, static buffers | `perf-torch-cuda-graphs` |

A800 (GA100, sm_80) notes:
- `ncu` needs GPU performance-counter permission on the node
  (`ERR_NVGPUCTRPERM` otherwise); ask the cluster admin or use a node where it is
  enabled. Tool `profile.ncu_kernel` is `needs_setup` for this reason.
- GA100 PM sampling needs an interval of at least 20000 cycles; smaller intervals fail.
- Counter replay serializes kernels and changes timing: use ncu for why, never for ms.
- FP8/NVFP4 tools are not available on sm_80; do not plan around them.

Record the kernel finding as a diagnosis with `--provenance measured_here` and the
report path as `--evidence-ref`. Record a measured share that turned out small as an
`experience` record so the next round does not re-profile it:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind experience --actor agent \
  --summary "<kernel> is a small share of <stage>" --model <model> --stage kernel.gemm \
  --context-json signature.json --outcome "share <p> of wall; ceiling below target, not worth a kernel" \
  --evidence-ref <ncu report>
```

## Output of this skill

- `attribution.json` (+ any diagnostic records), referenced from the log.
- One `select_stage` decision (or `stop`) with the actor set.
- Diagnoses for every gap that is still open, each with a `next_observation`.
