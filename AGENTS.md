# Repository instructions for coding agents

These notes apply to Codex, Claude Code and other agents working in this repository.
Read `CONTRIBUTING.md` for general contribution rules.

## VLA autotune

The agent-driven VLA autotune pipeline lives in `flash_rt/vla/autotune/`. Its procedures
are repo-scoped skills under `.agents/skills/`. Claude Code reaches the same folders
through the symlinks in `.claude/skills/`.

- Start from `.agents/skills/vla-autotune/SKILL.md`, the master loop:
  profile, attribute to stages, pick a tool per stage, run a campaign, decide, iterate.
- Tool registry: `python -m flash_rt.vla.autotune.tools list --all`
- Stage map: `python -m flash_rt.vla.autotune.stages list`
- Record every optimization decision with
  `python -m flash_rt.vla.autotune.decisions add --actor <human|agent|human+agent|search> ...`.
  Reports must be able to show which steps a human led.
- Patches must never modify `flash_rt/vla/autotune/`. The measurement harness is pinned
  and `snapshots` rejects such patches.

### Mandatory source-candidate checklist

Complete this checklist before initializing or measuring any candidate:

- [ ] Read `tools show <tool>` and use its registered lane. A non-config or unknown
  `Candidate.tool`, any `source_ref` differing from the baseline, or any non-null
  `patch_ref` (including the baseline) **requires `--snapshot-store` and plan schema 3**.
  An implicit `source_ref: null` does not inherit a concrete ref from another point.
- [ ] Implement each source hypothesis in its own worktree. Pin 40-hex base commits
  and content-addressed `sha256:` patches; materialize the baseline and every candidate.
- [ ] Make every snapshot ready with `adopt-native` or `record-built`, verify native
  identity, and require one harness tree across reference/candidates/bookend.
- [ ] Check that plan v3 binds every attempt to `snapshot_id`, that the ledger records
  those manifests, and that batch/worker source verification succeeds before ranking.
- [ ] Run the relevant CPU controller/snapshot tests; a rejected proposal must leave
  no plan or partial candidate/attempt/snapshot registrations. Record the evidence and
  real actor in the decision log.

Plan v2 remains for unpatched, same-source registered config tools and legacy config
records without `tool`. `analysis.lever_pricing` is registered as analysis: even when it
only changes existing config knobs, its candidates require snapshots under the
non-config rule. Existing ready config knobs need no implementation worktree.
Explicit infrastructure maintenance may edit the harness in the main checkout;
that authorization never makes harness edits part of an optimization patch.

Forbidden patterns:

- Editing the live checkout between candidates and reporting it as a source comparison.
- Dropping/rewriting `tool`, `source_ref` or `patch_ref` to obtain plan v2; treating unknown
  `config.*` names or all analysis tools as config; treating missing refs as inheritance.
- Reusing a mutable source directory for different variants, copying changed `.py`/`.so`
  files into a verified snapshot, or omitting the baseline's ready snapshot.
- Downgrading plan/result schemas, fabricating `source_identity`, bypassing import checks
  with `PYTHONPATH`, or changing timing/quality harness code inside a candidate patch.

## Where things go

- Edit skills in `.agents/skills/<name>/`, never through the `.claude/skills/<name>` symlinks.
- Third-party skills keep their upstream bytes, `LICENSE.txt` and `NOTICE`. Verbatim copies
  and provenance live in `third_party/agent-skills/`. Check them with
  `python third_party/agent-skills/verify_vendor.py`.
- Put scratch reports, probes and logs in `devlocal/<feature>/`. It is git-ignored; never
  force-add it.
- Run model computation only inside an allocated Slurm GPU step, never on a login node.

## Tests

The CPU tests need only numpy and pytest. From the repo root:

```bash
python -m pytest -q tests/test_vla_campaign_*.py tests/test_vla_experiment_*.py tests/test_vla_autotune_*.py tests/test_vla_agent_skills.py
```
