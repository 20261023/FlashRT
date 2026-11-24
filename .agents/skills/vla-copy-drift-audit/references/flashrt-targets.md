# FlashRT drift-audit targets: concrete spellings

**Read the section for your target before you start.** These are case studies
taken from the tree on 2026-11-15, at commit `728e5e2` plus the working tree.
They are not contracts. Re-read each provenance record at the revision you
audit, and substitute your own paths.

## (a) A candidate whose base is older than the current baseline

A patch or port candidate is a snapshot: `base_commit` + `patch_ref`. Its
manifest is at `<store>/snapshots/<id>.json`, and its patch at
`snapshots.patch_file(store, patch_ref)`.

```bash
python -m flash_rt.vla.autotune.snapshots show --store "$STORE" --snapshot-id "$ID"   # base_commit, patch_ref, harness_tree
BASE=<base_commit>; T=<current baseline source_ref or promotion tip>
git merge-base --is-ancestor "$BASE" "$T" || echo "STOP: baseline is not a descendant of the candidate base"
git apply --numstat <patch file>                        # P paths (snapshots.patch_paths also checks renames)
git diff --name-status --find-renames "$BASE" "$T"      # what moved under the candidate
git diff "$BASE" "$T" -- <each P path>                  # drift hunks on the copied paths
git diff "$BASE" "$T" -- flash_rt/vla/autotune          # harness moved? the candidate needs a new snapshot anyway
```

- **Text probe per file.** `P` is the file at the snapshot commit
  (`git show <snapshot_commit>:<path>`), `B` is `git show $BASE:<path>`, and the
  tip version is `git show $T:<path>`. Run
  `git merge-file -p P.txt B.txt T.txt > merged.txt` in a devlocal scratch
  directory.
- **Semantic neighbors to check even when the patched files did not move:**
  - callers and callees of the patched functions;
  - `flash_rt/vla/registry.py` (`validate_config`, `_CAPABILITIES`), if the
    patch adds options;
  - `CMakeLists.txt` and `csrc/bindings.cpp`, if the patch calls fvk;
  - `snapshots.NATIVE_INPUTS`: if those moved, the native build must be redone.
- **Absorbing.** Rebase in a scratch worktree with
  `git worktree add --detach devlocal/<feat>/rebase "$T"`, then
  `git apply --3way <patch>`. Resolve hunk by hunk, then rebuild the proposal
  from that worktree (`tools.patches.proposal_from_worktree(..., base_commit=T)`).
  That gives a new `patch_ref`, a new snapshot, a new gate and a new campaign.
  Remove the scratch worktree afterwards.

## (b) Vendored upstream reference code

| copy | provenance record | source pin | lineage |
|---|---|---|---|
| `training/_vendor/openpi_pi0_pytorch/*.py` | its `VENDOR.md` | **none recorded**: names "openpi (PyTorch port subtree)", capture date 2026-04-25, no URL or commit. Add commit `A` = `53e8973` (initial release) | text, with 5 documented import/JAX rewrites to undo before diffing. `gemma_config.py` and `pi0_config.py` are FlashRT shims (copy-only) |
| `csrc/attention/flash_attn_2_src/` | its `VENDOR.md` | flash-attention tag `v2.7.4.post1`; CUTLASS `c506e16788cb08416a4a57e11a9067beeee29420` | text, a subset (fwd only), plus FlashRT's own `fa2_wrapper.cu` |
| `csrc/attention/flash_attn_4_src/`, `csrc/attention/flashinfer_xqa_src/` | their `VENDOR.md` | as recorded there | read each record's modification list |
| FastWAM upstream model (not vendored; `source_dir`) | `examples/vla_autotune/fastwam-source/SOURCE.json` | `yuantianyuan01/FastWAM@7faa71108368fbb3b6885649f112af607427a2d4` | **contract pair, semantic-review-only**. `flash_rt/models/fastwam/*` relies on upstream seams that `FastWAMActionPipeline._validate_model` checks: `_encode_input_image_latents_tensor`, `_append_proprio_to_context`, `_build_mot_attention_mask`, `_denoise_action_with_video_cache`, `mot.prefill_video_cache_tensor`, and the video expert's attention-mask mode and patch size |
| historical Hy observers (`hy_trace`) | each file's docstring at its recorded Git revision; see `docs/vla_autotune_cleanup.zh-CN.md` | official Hy source `af57e7507ec5964b52fdf6296741e553cbcd3288`; the repository URL is not in the observer | retired from the current tree; semantic-review-only when reviewing a historical snapshot |

- **openpi today: `blocked: no upstream pin`.** Resolving it is a separate task.
  Find the upstream repository and a commit whose files, with the VENDOR.md
  rewrites undone, hash to the local bytes, then record the pin in VENDOR.md.
  Do not guess a revision.
- **Upstream change list** (once a pin exists), with the file at each end
  fetched as data:

  ```bash
  gh api "repos/<o>/<r>/compare/<pin>...<T>" --jq '.files[] | [.status,.filename,.previous_filename] | @tsv'
  gh api "repos/<o>/<r>/contents/<path>?ref=<pin>" --jq .content | base64 -d > devlocal/<feat>/drift/pin/<file>
  gh api "repos/<o>/<r>/contents/<path>?ref=<T>"   --jq .content | base64 -d > devlocal/<feat>/drift/tip/<file>
  ```
- **No implicit copy pairs.** The retired FastWAM Euler2 example graph pilot
  (added in `73b381b`; see `docs/vla_autotune_cleanup.zh-CN.md`) and
  `flash_rt/models/fastwam/velocity_graph.py` (added in `e5028b9`) are similar,
  but no record declares one a copy of the other. When reviewing their historical
  trees, the audit still needs an explicit mapping.

## (c) Agent skills

Start with the integrity check. It stops at the pin:

```bash
python3 third_party/agent-skills/verify_vendor.py                    # offline: blobs, bytes, NOTICE/VENDOR rows vs manifest
python3 third_party/agent-skills/verify_vendor.py --check-upstream   # each pinned commit still serves each blob (needs gh)
```

Then look for drift beyond the pin, per `(repo, commit)` group in
`third_party/agent-skills/manifest.json`:

```bash
T=$(gh api repos/<o>/<r>/commits/<default-branch> --jq .sha)
gh api "repos/<o>/<r>/compare/<commit>...$T" --jq '.files[] | [.status,.filename,.previous_filename] | @tsv' \
  | grep -F -f <(python3 -c 'import json;[print(e["upstream_path"]) for e in json.load(open("third_party/agent-skills/manifest.json"))["entries"] if e["repo"]=="<o>/<r>"]')
```

Then handle each changed file according to its kind:

- **Vendored-unmodified** (`.agents/skills/<name>/` and the
  `third_party/agent-skills/` archive):
  - `P` must equal `B` byte for byte; `verify_vendor.py` proves that.
  - Each upstream hunk is `absorb-now` (a pin bump), `decline` (keep the old
    pin, with a reason) or `pending`.
  - A pin bump changes the manifest `commit`/`blob`/`bytes`, the VENDOR.md row,
    the skill NOTICE blob line and the bytes, all together. Then
    `verify_vendor.py --check-upstream` must pass.
- **Adapted `vla-*` skills** (`vla-model-impl-map`, `vla-operator-gate`,
  `vla-copy-drift-audit`, `vla-port-workflow`):
  - `A` is the `metadata.upstream-commit` in the skill's frontmatter.
  - `B` is the verbatim archive at
    `third_party/agent-skills/apxinf/skills/<upstream name>/SKILL.md`.
  - `P` is `.agents/skills/vla-*/SKILL.md` plus its `references/`.
  - All of them are **semantic-review-only**, because they were heavily
    rewritten.
  - Read each upstream hunk. Ask whether the method changed (a new check, a
    corrected rule) or only ApxInf specifics did.
  - For `absorb-now`, edit the FlashRT text by hand. Then bump
    `upstream-commit`, add the change to the skill's NOTICE, and update the
    archive and manifest as for a vendored file.
- **Upstream-only new skills** next to the vendored ones: classify each as
  `not-needed` (with a reason) or `missing-copy`. A new upstream skill is a
  finding even when it is unrelated.
