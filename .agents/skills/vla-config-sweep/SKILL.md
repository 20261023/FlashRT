---
name: vla-config-sweep
description: Config lane of the FlashRT VLA autotune loop - sweep existing backend knobs (Euler steps, ProbeFlow epsilon and max_steps, CUDA Graph, FastWAM linear backend, pi0.5 vision pooling and state prompt mode, Hy numerical profile) for one chosen stage with flash_rt.vla.autotune.tools propose / levers, run a reference plus bookend controller campaign on Slurm, ingest, read summary.json and decide keep or accept. Use when the knob already exists in the registry and no source change is needed; config candidates never get a worktree. Chinese triggers - 调参数, 扫步数, ProbeFlow epsilon, 配置搜索, 跑一轮 campaign, 选 winner.
license: Apache-2.0
metadata:
  short-description: "Sweep existing VLA backend knobs via controller campaigns"
---

# VLA config sweep

Input: one selected stage (from `vla-stage-profile`), a baseline `Candidate` record,
a workload, and a quality budget the human has fixed. Output: a campaign summary and
a logged `keep` / `accept` / `reject` decision with the actor set.

```bash
cd <flashrt checkout>
py() { env UV_CACHE_DIR=/tmp/autotune-uv PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy python -B "$@"; }
LOG=<campaign root>/decisions.jsonl
```

## Rules carried from the design

- Same code tree: config candidates never get a worktree or a snapshot. Plan schema 2.
- Deterministic code expands parameters; the agent only chooses the tool, the stage and
  the bounds, and explains the outcome.
- KEEP vs best: the fastest passed point is not automatically adopted. If the reference
  wins, or the gain is inside bookend drift, the decision is `keep`.
- Exact vs approximate: `exact` tools (graph, linear backend, state prompt mode) should
  match the reference within the gate trivially; `approximate` tools (steps, ProbeFlow,
  pooling) trade quality and are judged only inside the recorded quality scope.
- Mechanism counts (NFE, pooled tokens) are not the effect. Only measured latency in the
  same campaign is.

## Step 1. Find the ready config tools for the stage

```bash
py -m flash_rt.vla.autotune.tools list --model <model> --hardware cuda_sm80 --stage <stage>
py -m flash_rt.vla.autotune.tools list --model <model> --hardware cuda_sm80 --stage <stage> --all --markdown
py -m flash_rt.vla.autotune.tools show <tool_id>     # parameters, exactness, reason
```

Without `--all` only `ready` tools are listed. A `patch`/`port` tool in the `--all`
list means the knob does not exist yet: go to `vla-patch-candidate`. FP8/NVFP4 are never
available on `cuda_sm80`.

Human checkpoint H2: if `tools show <tool_id>` reports `exactness: approximate`
(step counts, ProbeFlow, vision pooling), stop and get the human's approval before
generating candidates, and log the choice with `--actor human+agent` (or `human`).
Only exact tools (for example `launch.cuda_graph`) may be selected with `--actor agent`.

Log the tool choice:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent \
  --action select_tool --tool sampler.euler_steps --stage denoise.schedule --model <model> \
  --summary "sweep Euler steps on the velocity-dominated path" \
  --rationale "approximate tool: human approved (H2); velocity share <p> measured_here; ceiling <c>x" \
  --evidence-ref attribution.json
```

## Step 2. Generate candidates

`baseline.json` is one `Candidate.to_dict()` record (the current production config,
with `source_ref`/`patch_ref` as recorded). Parameter values are JSON.

```bash
# One tool, default or explicit parameter grid:
py -m flash_rt.vla.autotune.tools propose --baseline baseline.json \
  --tool sampler.euler_steps --param 'steps=[6,7,8]' --out proposals.json
py -m flash_rt.vla.autotune.tools propose --baseline baseline.json \
  --tool sampler.probeflow --param 'epsilons=[0.002,0.004]' --param 'max_steps=10' --out pf.json
py -m flash_rt.vla.autotune.tools propose --baseline baseline.json \
  --tool prefix.state_prompt_mode --param 'modes=["fixed"]' --out spm.json
```

Parameters per tool (`tools show` is authoritative): `sampler.euler_steps steps`,
`sampler.probeflow epsilons, max_steps`, `gemm.fastwam_linear_backend variants`,
`prefix.vision_pool factors`, `prefix.state_prompt_mode modes`,
`precision.hy_numerical_profile profiles`. `launch.cuda_graph` and
`compile.fastwam_inductor` take none and emit nothing when the baseline already uses them.

Coupled knobs: price them as levers instead of a full grid. Each lever alone, all
together and all-minus-one are emitted; identical encoded backends are dropped:

```bash
py -m flash_rt.vla.autotune.tools levers --baseline baseline.json --stage host.launch \
  --lever 'graph={"options":{"use_cuda_graph":true,"native_linear_backend":"cached"}}' \
  --lever 'steps={"num_steps":12}' --out levers.json
```

Lever fields: `num_steps`, `sampler`, `probeflow`, `backend`, `options`. Two levers that
set the same field differently are an error, never a precedence rule.

Merge several proposal files into one JSON list if you run them in one campaign. Keep
each campaign small enough for one allocation (reference, candidates, bookend run
serially in one job).

## Step 3. Initialize the campaign (plan v2)

Human checkpoint: the quality gate (scope name, `action_dim` trained channels,
`max_relative_l2`, `min_cosine`) and the run budget are the human's choice. Freeze them
before any candidate runs; never loosen them after seeing results.

```bash
py -m flash_rt.vla.autotune.controller init \
  --root <campaign root> --comparison-id <cid> --baseline baseline.json \
  --workload /abs/workload.json --output-root /abs/results/<cid> \
  --proposals proposals.json --warmup 5 --repeats 20 \
  --quality-gate '{"scope":"<scope>","action_dim":<n>,"max_relative_l2":0.05,"min_cosine":0.999}'
```

- Do not pass `--snapshot-store`: that is the patch lane (plan v3).
- `--enumerate` adds the controller's own Euler/ProbeFlow grid (no tool provenance);
  prefer tool proposals so every candidate carries `tool` and `stage`.
- `repeats` is the total measured request count and must cover every workload request.
- Hy: use `examples/vla_autotune/hyvla_tradeoff_quality.json` as the gate (20 public
  channels) and a freshly measured native BF16 `semantic_bf16` Euler10 reference.

Log it:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent \
  --action run_campaign --tool sampler.euler_steps --campaign <cid> --model <model> \
  --candidate-id <id1> --candidate-id <id2> \
  --summary "campaign <cid>: Euler 6/7/8 vs reference" --rationale "bounded grid, 1 allocation"
```

## Step 4. Render, submit, wait, run, ingest

The allocation runs `batch` from a checkout on the cluster. Render the sbatch with the
existing renderer (never hand-write directives; this site derives CPUs from the GPU
request):

```bash
py -c "
from flash_rt.vla.autotune.slurm import render_sbatch, SlurmResources
print(render_sbatch(job_name='at-<cid>', workdir='/abs/flashrt', output='/abs/results/<cid>/slurm-%j.out',
      argv=['python', '-B', '-m', 'flash_rt.vla.autotune.batch', '--plan', '/abs/<campaign root>/plan.json'],
      resources=SlurmResources(partition='gpu_a800', gpus=1)))
" > /abs/round.sbatch
```

The plan must be at the path the job reads (copy `plan.json` to the cluster if the
controller root is local). Then:

```bash
py -m flash_rt.vla.autotune.controller submit --root <campaign root> --script /abs/round.sbatch --ssh-host <host>
py -m flash_rt.vla.autotune.controller wait --root <campaign root> --ssh-host <host> --interval 30
# fetch /abs/results/<cid> to a local directory, then:
py -m flash_rt.vla.autotune.controller ingest --root <campaign root> --results <fetched dir> --job-finished
```

- `submit` owns the submitting transition; do not call `submitting` first.
- A lost scheduler response leaves `submission_unknown`; reconcile with
  `controller queued --root <root> --job-id <id>`. Never resubmit automatically.
- `batch` refuses to run outside a Slurm allocation. Never run models on the login node.

## Step 5. Read summary.json and apply the winner rules

`<campaign root>/summary.json` holds `complete`, `quality_scope`, `winner`,
`baseline_drift_ratio` and every attempt. The controller already enforces:

- incomplete campaign -> no winner;
- reference/bookend median drift beyond 10% -> no winner (inconclusive, rerun);
- reference or bookend not passed in the exact scope -> no winner;
- only measured, quality-passed points in exactly that scope can win.

Then the agent and human apply KEEP vs best:

1. `winner` null: log `reject` (or `stop`) with the reason (drift, incomplete,
   all failed quality). Do not pick the "best looking" attempt by hand.
2. `winner` is the reference attempt: log `keep`.
3. `winner` is a candidate whose gain is not clearly larger than `|baseline_drift_ratio|`:
   log `keep` and, if it matters, plan a wider rerun (more requests, more repeats).
4. Otherwise: human checkpoint, then `accept`. The human confirms the quality scope
   is the one they care about; a numerical replay proxy is never task success.

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human \
  --action accept --tool sampler.euler_steps --campaign <cid> --candidate-id <winner> \
  --summary "accept <winner> within <scope>" \
  --rationale "median <a> ms vs <b> ms; drift <d>; all requests passed" \
  --evidence-ref <campaign root>/summary.json
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind experience --actor agent \
  --summary "<model> <tool> outcome on cuda_sm80" --model <model> --stage <stage> \
  --context-json signature.json --outcome "<what was faster/slower/failed and why>" \
  --evidence-ref <campaign root>/summary.json
```

`signature.json` = workload signature (model, checkpoint, source revision, hardware,
precision, shapes, views, steps). Accepted configs are not written into skills; they live
in the log and the baseline you carry forward.

## Cautions from earlier campaigns (re-measure; never reuse as numbers)

- pi0.5: the prefix dominates wall time, so removing denoising steps buys little end to
  end. Check the velocity share from `vla-stage-profile` before sweeping steps.
- FastWAM: a step count chosen on 4 inputs was overturned by 12 inputs. Always widen the
  workload (more requests, different scenes) before accepting a step count.
- FastWAM + ProbeFlow: the default epsilon routed requests to 2 steps with about 26%
  error. Treat the ProbeFlow defaults as a starting grid, not a safe setting, and read
  the actual NFE in `measurement.sampler_metadata`.
- Fewer steps can add routing cost (ProbeFlow probes, `.item()` syncs). If latency does
  not follow NFE, go back to `vla-stage-profile` (unattributed / routing).
- `gemm.fastwam_linear_backend=flashrt-legacy` turns the velocity graph off: compare it
  with the graph-on baseline, not an eager one.
- `precision.hy_numerical_profile` is an alignment knob, not a speed knob.
- `compile.fastwam_inductor` (torch.compile, `compile_action_infer=True`) and CUDA graphs are
  mutually exclusive, and the compiled path needs `backend=torch`. The graph and native-linear
  tools switch compile off when they propose from a compiled baseline. Record the cold compile
  time separately: it is not in the steady-state samples, so work out the break-even call count.

## Stop rules

Stop the config lane for a stage (decision `stop`, with the actor) when the remaining
grid cannot beat the Amdahl ceiling, two consecutive campaigns end in `keep`, or the
human budget is spent. Then return to `vla-stage-profile` for the next stage or move to
`vla-patch-candidate` if the knob you need does not exist.
