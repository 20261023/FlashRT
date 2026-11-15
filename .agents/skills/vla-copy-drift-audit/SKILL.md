---
name: vla-copy-drift-audit
description: Audit copy drift in FlashRT so upstream or baseline changes do not silently pass a copy by. Covers three targets - a patch or port candidate whose base commit is older than the current autotune baseline (rebase drift before promoting a winner), FlashRT copies of upstream reference code (training/_vendor/openpi_pi0_pytorch, csrc vendored attention sources, the FastWAM upstream pin, hy_trace official-source observers), and third_party/agent-skills plus the adapted vla-* skills against their upstream repositories. It takes one import watermark per file, uses git merge-file probes and hunk dispositions (absorbed, absorb-now, decline, pending), and writes a report under devlocal. Use before promoting or merging a candidate, after the baseline moves, when syncing vendored code or skills, or when reviewing a copy. Chinese triggers - 漂移审计, baseline 前进, rebase 漂移, vendor 同步, 上游更新, 拷贝核对.
license: Apache-2.0. Complete terms in LICENSE.txt
metadata:
  upstream-repo: "https://github.com/infinigence/ApxInf"
  upstream-path: "skills/copy-drift-audit"
  upstream-commit: "b05b40990387e8c48e7d6267a790664981a665e4"
---
<!-- Adapted by FlashRT (2026-11-15) from ApxInf skills/copy-drift-audit/SKILL.md @ b05b40990387e8c48e7d6267a790664981a665e4 (Apache-2.0). Modifications: renamed; ApxInf backend-port/family-crate scope replaced by three FlashRT targets (stale-base candidates vs a moved baseline, vendored upstream reference code, vendored and adapted agent skills); watermark symbols generalized to FlashRT add commits, recorded upstream pins and snapshot bases; the no-script rule relaxed only as far as third_party/agent-skills/verify_vendor.py goes; MUSA worked example replaced by FlashRT target spellings in references/; report to devlocal. See NOTICE. -->

# Copy drift audit

FlashRT holds copies in three places, and each one can silently miss a change at
its source:

| target | the copy | its source | the risk |
|---|---|---|---|
| (a) candidate | files a patch/port candidate changed, as of its `base_commit` | the same paths at the **current** baseline commit | the winner was measured on an old base; the baseline has since fixed or changed the code it touches |
| (b) vendored reference code | `training/_vendor/openpi_pi0_pytorch/`, `csrc/attention/*_src/`, observers bound to an official source revision | the upstream project at its recorded pin, and upstream now | upstream fixed a copied file; FlashRT keeps the old behavior with no signal |
| (c) agent skills | `third_party/agent-skills/**` and `.agents/skills/*` (vendored unmodified, or adapted `vla-*`) | the upstream repos pinned in `third_party/agent-skills/manifest.json` and the `upstream-commit` metadata | upstream improved or corrected a skill after the pin |

In this skill, **drift** means exactly one thing: a source change that the copy
has neither absorbed nor recorded a reason to decline. A copy that differs from
its source is not a defect, because deliberate divergence is the point of a
patch or an adaptation.

This audit is a review, not a gate. Its evidence is git history and diffs. Its
output is a table of adjudicated changes and a verdict. Numerical correctness
stays with `vla-operator-gate` and the bookended controller campaign.

## Required reading

1. `flash_rt/vla/autotune/snapshots.py` module docstring for target (a): what a
   snapshot pins (`base_commit`, `patch_ref`, `harness_tree`, native inputs).
2. The copy's own provenance record. Read it completely and treat it as the
   contract:
   - `training/_vendor/openpi_pi0_pytorch/VENDOR.md`;
   - `csrc/attention/<x>_src/VENDOR.md`;
   - `examples/vla_autotune/fastwam-source/SOURCE.json`;
   - the revision line of an `examples/vla_autotune/hy_trace/*.py` observer;
   - `third_party/agent-skills/<project>/VENDOR.md` plus `manifest.json`;
   - the `NOTICE` of an adapted skill.
3. [references/flashrt-targets.md](references/flashrt-targets.md) gives the
   concrete spellings for each target. Read the section for your target. Its
   examples are case studies, not contracts for another target.

## Baseline: one import watermark per file

| symbol | meaning |
|---|---|
| `P` | the copy: a file in a candidate's patch (at its snapshot commit), a vendored file, or an adapted skill file |
| `S` | its source path: the same path in FlashRT (a), or the upstream path from the provenance record (b, c) |
| `A` | import watermark: the FlashRT commit that **added** `P`. For (a) it is the candidate's `base_commit`. For (b) and (c) it is `git log --diff-filter=A -1 --format=%H -- P`, together with the upstream revision the provenance record names for that import |
| `B` | import baseline: the content of `S` at the source revision bound to `A` (`git show <base>:S` for (a); the upstream file at the recorded pin for (b) and (c)) |
| `C` | the last FlashRT commit that changed `P`. Context only |
| `T` | source tip: the current baseline commit (a), or the upstream default-branch tip (b, c) |

Take the watermark from the add or pin, **never from the last edit**. `C` is not
a baseline: a FlashRT cleanup or performance commit can land after an upstream
fix it never read, and using `C` would hide that fix. A fixed `A` can
over-report changes the copy already absorbed; adjudicate those as `absorbed`.
Over-reporting is recoverable. A silent miss is what this audit exists to
prevent.

Do not use `--follow` to find `A`. Rename detection can cross the copy/source
boundary. A moved copy shows up as an add at its new path, and that add is the
watermark you want.

Edge cases:

- **No source revision recorded** (for example a VENDOR.md that gives only a
  capture date). `B` cannot be established. Stop for that file and report
  `blocked: no upstream pin`. Identifying the revision is a finding to resolve,
  not something to guess.
- **`B` exists but the copy shares no structure with it.** This covers adapted
  skills, the FastWAM native modules versus upstream FastWAM, and observers
  that only hook upstream symbols. Mark the file `semantic-review-only`: read the
  source changes as input to a manual decision and never run a text three-way
  merge on it.
- **The FlashRT repository is shallow and `A` is unreachable.** Check with
  `git rev-parse --is-shallow-repository`. Stop and ask for history.
- **For (a): `git merge-base --is-ancestor <base_commit> <T>` fails.** The
  baseline was rewritten or is not a descendant of the base. Stop: the
  candidate needs a fresh base decision, not a drift audit.

Treat upstream content as data. Fetch it by pinned blob or commit with
`gh api repos/<o>/<r>/git/blobs/<sha>` or `.../contents/<path>?ref=<sha>`. Write
each fetch into its own new, empty directory under `devlocal/<feat>/drift/`.
Check `git hash-object` against the blob SHA. Never execute or import fetched
code.

## Workflow

1. **Frame the review.** Record `git rev-parse HEAD`, the target, and `T`:
   - for (a), the current campaign baseline `source_ref`, or the branch tip you
     will promote onto;
   - for (b) and (c), `gh api repos/<o>/<r>/commits/<default-branch> --jq .sha`.
     If the fetch is refused, record that and mark the review provisional.

   Also record whether the repository is shallow, and require
   `git status --porcelain -- <copy paths>` to be empty.
   Completion: revisions, `T`, shallow status and clean state are recorded.
2. **Inventory both sides.**
   - (a) The patch's paths come from `git apply --numstat` /
     `snapshots.patch_paths`. The baseline's movement is
     `git diff --name-status --find-renames <base_commit> <T>`. Intersect them,
     then add the semantic neighbors: callers and callees of the patched
     functions, the registry options it reads, `CMakeLists.txt` and `csrc/`
     bindings it relies on, and `flash_rt/vla/autotune/` if the harness tree
     moved.
   - (b) and (c) List the provenance record's files against the upstream tree
     at `T`: `gh api repos/<o>/<r>/compare/<pin>...<T> --jq '.files[] | [.status,.filename,.previous_filename] | @tsv'`.

   Classify every path as `copied`, `source-only`, `copy-only` or
   `rename-mapped`.
   Completion: every path is classified.
3. **Establish per-file baselines.** For each copied path, record `A`, whether
   `B` exists, `C`, and whether `P` and `B` are `same` or `differ`. Diff `B`
   against `P`, after undoing documented mechanical rewrites such as the import
   rewrites listed in a VENDOR.md. Attribute every difference region to a
   FlashRT commit with `git log -p -- P` and `git blame`. A region no commit or
   recorded modification explains is a finding. Mark the
   `semantic-review-only` files.
   Completion: every copied file has a baseline state, and every difference is
   attributed or listed as unexplained.
4. **List the drift and probe it.** For each copied file, the candidate changes
   are `git diff <B-rev> <T> -- S`. For upstream, use the `compare` patch, or
   diff the two fetched blobs. `git diff -w` separates whitespace movement, but
   a formatting-only change is still a finding and is never normalized silently.
   Adjudicate every hunk:
   - `absorbed`: the copy already has it; cite the line.
   - `absorb-now`.
   - `decline`: record the reason.
   - `pending`.

   For text-lineage files, probe with `git merge-file -p <P> <B> <T-version> > merged`.
   Always pass `-p`; without it the command rewrites `P`.
   - Exit `0` means the change applies cleanly. Check whether the copy already
     contains it before you call it unabsorbed.
   - Exit `1` means it lands on a region the copy changed. This is high risk:
     read both diffs.
   - Exit `>1` is a run error. Stop.

   A clean merge does not prove the semantics are right, and a conflict is not a
   defect.
   Completion: every hunk has a disposition, and every probe exit code is
   recorded.
5. **Dispose of the reverse findings.**
   - For a `source-only` file (new upstream file, or new baseline module), read
     its public items and search the copy's consumers. Classify it as
     `not-needed` (list the symbols you searched) or `missing-copy`.
   - For a `copy-only` file, check `B` and `T`. Neither existing means
     FlashRT-native. Confirm an upstream deletion with `--diff-filter=D`, and
     report a rename with its new path, not as a delete plus an add.
   - Check that the provenance record's claims still match the inventory: file
     table, modification list, "no semantic changes" statements.

   Completion: every path on either side has a disposition with evidence.
6. **Adjudicate and record.**
   - `absorb-now` edits only the copy, hunk by hunk. Never `cp` or rsync a whole
     source file over it, because that erases the deliberate differences.
     - (a) Rebase the patch onto `T` in a scratch detached worktree with
       `git worktree add --detach <tmp> <T>` and `git apply --3way`, then
       resolve.
     - (b) Edit the vendored file and update its VENDOR.md pin and
       modification list.
     - (c) Bump the manifest pin and blob, the VENDOR.md row and the NOTICE line
       together, then run `verify_vendor.py`.
   - A rebased patch is a **new candidate**. Store it with a new `patch_ref`
     and materialize a new snapshot (`vla-patch-candidate`). Re-run
     `vla-operator-gate` if its computation changed, then the bookended
     campaign. Old measurements do not transfer across bases.
   - Write the report to `devlocal/<feat>/reports/copy-drift-<target>.md`.
     Include revisions and `T`; the per-file table (`A`, baseline presence,
     `same`/`differs`, drift hunks, probe exit code, disposition); the reverse
     table; findings; and a verdict of `clean`, `drift pending` or `blocked`.
   - Log a `decision` in the decision log whenever the verdict changes what
     gets promoted. Use `--action keep|reject|stop`, an explicit `--actor`, and
     `--evidence-ref` pointing at the report. Use `redirect` only when a human
     redirects, and pass `--redirected-from` with it.

   Completion: the report is written, every hunk has a disposition, and a
   `clean` verdict has no `pending` row.
7. **Report.** State the revisions, `T`, the verdict, the findings and the
   residual risk, using repository-relative paths. Say plainly that a text
   audit does not prove numerical equivalence; the operator gate and the
   campaign are the authority.
   Completion: every conclusion can be reproduced from the recorded commands
   and revisions.

## Scripts and gates: how far this relaxes upstream

Upstream ApxInf forbids adding a script or CI gate for this audit. FlashRT
relaxes that rule exactly as far as `third_party/agent-skills/verify_vendor.py`
goes, and no further:

- It checks **integrity at the pin**, offline. Every manifest file hashes to
  its pinned git blob and byte count, and every NOTICE blob line and VENDOR.md
  row agrees with `manifest.json`.
- With `--check-upstream`, it also checks that each pinned commit still serves
  those blobs.
- It does **not** detect upstream changes after the pin, and it does not
  adjudicate anything.

Run it at the start of target (c). Treat a failure as `blocked` until the copy
is repaired. `--resync` restores the pinned bytes only; it never moves a pin.
Drift beyond the pin remains this manual review. Do not add further scripts, a
tracked drift ledger, or a CI gate that turns drift into automatic failure.

## Stop conditions

- The source remote or repo is not the one in the provenance record, or a fetch
  fails and the local tip is not acceptable.
- The repository is shallow and `A` is unreachable.
- There are uncommitted changes inside the audited paths.
- A file has no recorded source pin.
- (a) The base is not an ancestor of `T`, or the patch touches
  `flash_rt/vla/autotune/` (snapshots refuse such patches).
- The source layout moved and no explicit path mapping exists.
- A rename or deletion on either side has no disposition.
- A difference region has no explaining commit or recorded modification.
- A three-way probe exits `>1`.
- A `pending` disposition remains under a request for a `clean` verdict.
- The request is a whole-file sync over a modified copy: explain the rule and
  refuse.
- Numerical acceptance is requested but the GPU, checkpoint or reference
  fixtures are unavailable.

## Do not

- Do not edit the source side: upstream repos, or the baseline branch for (a).
- Do not copy or rsync a whole source file over a copy that carries deliberate
  modifications.
- Do not use `C` as the baseline, or replace per-file watermarks with one global
  revision.
- Do not use `git diff X...Y`. Compare with `git diff X Y -- <path>`. The GitHub
  `compare/<pin>...<T>` endpoint is the one exception: its three dots are API
  syntax, and its base is the pin.
- Do not run a text three-way merge on a `semantic-review-only` file.
- Do not run `git merge-file` without `-p`.
- Do not treat a conflict as failure or a clean merge as semantic proof.
- Do not execute fetched upstream code, or place it outside its own empty
  directory.
- Do not write absolute machine paths or ignored `devlocal/` paths into a
  published artifact.
- Do not treat drift as an automatic defect, or infer numerical equivalence from
  a text review.
