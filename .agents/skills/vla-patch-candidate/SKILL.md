---
name: vla-patch-candidate
description: Patch lane of the FlashRT VLA autotune loop - turn a source change (cache, fusion, new backend option, kernel wiring, AITune integration) into a measured candidate. Make the change in a separate git worktree from the baseline commit, store it as a content-addressed PatchProposal, materialize source snapshots, adopt or build native extensions, run a snapshot-mode (plan v3) controller campaign whose batch and workers verify source identity, then decide and hand a winner to a human merge. Use when a needed knob does not exist, for tools with lane patch or port, or after vla-kernel-candidate produced a kernel. Chinese triggers - 改源码, worktree 候选, patch 候选, 快照, 源码隔离, 合并决定.
license: Apache-2.0
metadata:
  short-description: "Source-change candidates via worktrees and snapshots"
---

# VLA patch candidate

Input: a selected stage, a hypothesis that needs a source change, and a baseline
`Candidate` that runs an exact commit. Output: a snapshot campaign summary and a logged
decision; a winner only becomes code after a drift audit and a human merge decision.

```bash
cd <flashrt checkout>                       # the controller/login checkout
py() { env UV_CACHE_DIR=/tmp/autotune-uv PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy python -B "$@"; }
LOG=<campaign root>/decisions.jsonl
STORE=/abs/shared/autotune-store           # snapshot store on a filesystem the GPU nodes see
```

## Rules carried from the design

- Worktrees are for source changes only, one per hypothesis; config knobs go to
  `vla-config-sweep`. Keep the number of concurrent source variants small.
- One change per patch. A diagnostic ablation (e.g. turning something off to measure its
  share) is labelled `diagnostic:` and is never accepted as a candidate.
- Harness pin: a patch must never touch `flash_rt/vla/autotune/` or `.git*` files.
  `snapshots` and `patches` refuse such patches; do not work around it.
- A new frontend option must be added to `flash_rt/vla/registry.py` `_CAPABILITIES` in
  the SAME patch, or the snapshot worker rejects the backend.
- Exactness: state in the hypothesis whether the change is `exact` (same math, e.g.
  caching a per-request constant) or `approximate` (different numerics or semantics).
  The campaign quality gate judges both; `exact` changes that fail it are bugs.
- KEEP vs best and first-valid vs fastest: a patch that runs is not a winner; only the
  campaign decides speed, and the reference can win.

## Step 1. Pin the base commit

```bash
py -m flash_rt.vla.autotune.snapshots resolve --repo . --ref <baseline ref>   # -> 40-hex BASE
```

The baseline Candidate must have `source_ref` = that 40-hex commit and no `patch_ref`
(stacked patches are not supported). Rewrite `baseline.json` if it held a short ref.

Log the tool/hypothesis choice:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent \
  --action select_tool --tool cache.fastwam_text_crossattn_kv --stage denoise.velocity --model fastwam \
  --summary "cache per-request text cross-attention K/V" \
  --rationale "recomputed every step though constant per request; velocity share <p>"
```

## Step 2. Make the change in its own worktree

```bash
git worktree add --detach /abs/wt/<slug> <BASE>
cd /abs/wt/<slug> && git switch -c autotune/<slug>
# edit; keep the diff minimal and inside the stage's seams (stages list --model M)
git add -A && git commit -m "autotune(<stage>): <one-line hypothesis>"
```

- Seams per stage: `python -m flash_rt.vla.autotune.stages list --model <model>`;
  call paths and graph caches: `vla-model-impl-map`.
- If the patch changes what a CUDA graph captures, check capture/replay rules with
  `perf-torch-cuda-graphs`; operator-level numerics with `vla-operator-gate`.
- Run the CPU tests that cover the touched files inside the worktree before storing.
- Untracked files are refused unless `include_untracked=True`; prefer committing them.

## Step 3. Store the PatchProposal

```bash
py -c "
from flash_rt.vla.autotune.tools.patches import proposal_from_worktree
pr = proposal_from_worktree('/abs/wt/<slug>', '<BASE>', store='$STORE',
        tool_id='cache.fastwam_text_crossattn_kv', stage_id='denoise.velocity',
        hypothesis='<what changes and why it is faster>; exactness=exact',
        options={}, provenance={'author': 'agent'})
pr.save('$STORE/proposals/<slug>.json'); print(pr.patch_ref)"
```

`options` are frontend options the patch needs (merged over the baseline options).
The diff covers committed, staged and unstaged tracked changes against BASE and is
checked against the patchable-path rule and with `git apply --check` before it is stored.
The rule (`snapshots.check_patch_paths`) refuses the harness `flash_rt/vla/autotune/`, git
metadata, symlinks/gitlinks, and any top-level path outside an allowlist (`flash_rt`, `csrc`,
`cmake`, `scripts`, `docs`, `tests`, `examples`, `benchmarks` and a few top-level text files),
so no module lands where it could be imported from the snapshot root.

## Step 4. Materialize snapshots (baseline AND candidate)

Plan v3 runs every point, including the reference, from its own snapshot:

```bash
py -m flash_rt.vla.autotune.snapshots materialize --repo . --store $STORE --base <BASE>
py -m flash_rt.vla.autotune.snapshots materialize --repo . --store $STORE --base <BASE> --patch-ref sha256:<hex>
```

Each prints its manifest (`snapshot_id`, `source_dir`, `native.status: pending`).

## Step 5. Native extensions

Check which paths the patch touches (`pr.provenance['paths']`, or `git diff --stat`).
Native inputs are `csrc`, `CMakeLists.txt`, `cmake`, `scripts/build_gemm_plan.py`.

- Python-only patch (no native input touched): reuse a built checkout of the same
  native inputs, for both snapshots:

```bash
py -m flash_rt.vla.autotune.snapshots adopt-native --store $STORE --snapshot-id <id> \
  --prebuilt-root /abs/built-flashrt --fingerprint "<toolchain + arch + build id>"
```

  It refuses when the native inputs differ ("rebuild required").
- Native change (csrc / CMake): render the build job, run it in its own GPU allocation,
  and let its last line record the result:

```bash
py -m flash_rt.vla.autotune.snapshots build-script --store $STORE --snapshot-id <id> \
  --cutlass-dir /abs/cutlass --gpu-arch 80 --python /abs/venv/bin/python --out /abs/build-<id>.sh
py -c "
from flash_rt.vla.autotune.slurm import render_sbatch
print(render_sbatch(job_name='build-<id>', workdir='$STORE/snapshots/<id>', output='/abs/build-<id>.log',
      argv=['bash', '/abs/build-<id>.sh']))" > /abs/build-<id>.sbatch
# submit with your site's sbatch (not a controller campaign); the script ends with
# python -m flash_rt.vla.autotune.snapshots record-built --store ... --snapshot-id <id> --fingerprint ...
```

- Confirm both are ready: `snapshots show --store $STORE --snapshot-id <id>`, and on a
  node: `snapshots verify --store $STORE --snapshot-id <id>`.

Human checkpoint: a native build costs an allocation; the human approves it.

## Step 6. Candidate record and snapshot campaign (plan v3)

```bash
py -c "
import json
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.tools.artifacts import PatchProposal
from flash_rt.vla.autotune.tools.patches import candidate_from_proposal
base = Candidate.from_dict(json.load(open('baseline.json')))
cand = candidate_from_proposal(base, PatchProposal.load('$STORE/proposals/<slug>.json'),
                               '<base-id>-<slug>', validate=True)
json.dump([cand.to_dict()], open('proposals.json', 'w'), indent=2)"
```

Use `validate=False` only when the patch adds a new option (this checkout's registry
does not know it; the snapshot worker validates it at load).

```bash
py -m flash_rt.vla.autotune.controller init \
  --root <campaign root> --comparison-id <cid> --baseline baseline.json \
  --workload /abs/workload.json --output-root /abs/results/<cid> \
  --proposals proposals.json --warmup 5 --repeats 20 --quality-gate '<frozen gate JSON>' \
  --snapshot-store $STORE --python /abs/venv/bin/python
```

- Requires absolute workload/output paths and ready snapshots sharing one harness tree.
- Render/submit/wait/ingest exactly as in `vla-config-sweep` Step 4. `batch` preflights
  every snapshot (re-verifies the source) before launching anything, runs each attempt
  with cwd = its snapshot and only that FlashRT on `PYTHONPATH`, and passes
  `--source-manifest`; the worker verifies the imported tree before loading the backend.
- An attempt with `failure.kind == "source_identity"` means the tree on the node is not
  the snapshot (dirty tree, wrong `.so`, other import root). Fix the snapshot; never
  relabel the result.

Log `run_campaign` with `--candidate-id` and the actor, as in the config lane.

## Step 7. Decide

Read `summary.json` with the same winner rules (complete, bookend drift within 10%,
exact quality scope, KEEP when the gain is inside drift). Log one of:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor agent \
  --action reject --tool <tool> --campaign <cid> --candidate-id <id> \
  --summary "<slug> slower or failed quality" --rationale "<numbers from summary.json>" \
  --evidence-ref <campaign root>/summary.json
```

`accept` needs a human checkpoint (`--actor human` or `human+agent`). Add an
`experience` record either way, including failures (compile failure, unsupported,
timeout, quality failure, no improvement, insufficient measurement are different
outcomes; say which).

## Step 8. Promote a winner (human merge only)

1. The baseline branch has usually moved since BASE. Run `vla-copy-drift-audit` on
   the candidate branch against the moved baseline (vendored references, copied code).
2. Rebase the worktree branch onto the new baseline; if anything beyond trivial
   context changed, re-run Steps 3-7 on the new base: a result is bound to its base.
3. Human checkpoint: the merge. The human reviews the diff, the summary and the audit and
   decides; log it:

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human \
  --action merge --tool <tool> --candidate-id <id> --summary "merge <slug> into <branch>" \
  --rationale "accepted in <cid>; drift audit clean on <new base>" --evidence-ref <audit report>
```

4. Clean up: `git worktree remove /abs/wt/<slug>` for rejected branches once the log and
   the stored patch reference them. Snapshot worktrees under `$STORE` are locked; leave
   them while any campaign references them.

### Chaining a winner into the next round (before or without a merge)

Patches cannot stack on a patched baseline, because `candidate_from_proposal` refuses a
baseline with `patch_ref`. To continue the outer loop from a patch winner:

1. Commit the winner's worktree branch.
2. Get its commit: `py -m flash_rt.vla.autotune.snapshots resolve --repo /abs/wt/<slug> --ref HEAD`.
3. Write a new `baseline.json` with `source_ref` set to that 40-hex commit, `patch_ref: null`
   and the winner's backend options.
4. `snapshots materialize --base <that commit>` (no patch), then adopt or build its native
   extensions.
5. Start the next proposal from a fresh worktree at that commit.

Merging into the main branch remains the separate human step 3 above.

## Sources of patches

- Registry tools with lane `patch`/`port` (`tools list --all --lane patch`), e.g.
  `cache.fastwam_text_crossattn_kv`, `fusion.hy_rope_qknorm_kvwrite`, `quant.int8_pi05`.
- Kernels from `vla-kernel-candidate` (wired behind a new backend option).
- `compile.aitune`: a frontend option that wraps Torch submodules and loads a pinned
  `.ait` artifact; record its path and sha256 in `provenance`. See
  `python -m flash_rt.vla.autotune.tools show compile.aitune` for setup and risks.
- Port lane (method/backend ports with capability updates): follow `vla-port-workflow`,
  which ends in this skill from Step 3.
