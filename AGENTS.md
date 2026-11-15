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
