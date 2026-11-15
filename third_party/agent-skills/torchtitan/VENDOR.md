# torchtitan (vendored agent-skill material)

- Upstream: https://github.com/pytorch/torchtitan
- Pinned commit: `631d544b1f6794ecfeca734a7bba861e03ddd5e3`
- License: BSD-3-Clause
- Fetched: 2026-11-15
- Method: `gh api repos/pytorch/torchtitan/git/trees/<commit>?recursive=1`, then each file by blob SHA via `gh api repos/pytorch/torchtitan/git/blobs/<blob>`; `git hash-object` of every local file equals the upstream blob SHA.
- Local modifications: none (every file below is byte-identical to its upstream blob).

Path mapping: the hidden `.claude` component of `torchtitan/rl/.claude/skills/...` is stored as `dot-claude` (content
unchanged), so Claude Code never discovers the archived skill as a nested project skill.

## Files

| local path (repo-relative) | upstream path | blob | bytes |
|---|---|---|---|
| `.agents/skills/kernel-numerics-verifier/LICENSE.txt` | `LICENSE` | `6d1df98ff6031178472ac043e5b4e11ea39568b2` | 1495 |
| `.agents/skills/kernel-numerics-verifier/SKILL.md` | `.claude/skills/kernel-numerics-verifier/SKILL.md` | `24a5e8776347e6d8fc9dc8d50584f03e7ac49612` | 5129 |
| `third_party/agent-skills/torchtitan/LICENSE` | `LICENSE` | `6d1df98ff6031178472ac043e5b4e11ea39568b2` | 1495 |
| `third_party/agent-skills/torchtitan/torchtitan/rl/dot-claude/skills/inference_perf_hillclimb/SKILL.md` | `torchtitan/rl/.claude/skills/inference_perf_hillclimb/SKILL.md` | `a6c69067f4b2df4b32a9abc742004e55b75d8dd2` | 12426 |

## .gitignore

Every local path above was checked with `git check-ignore -v --no-index`: none is ignored.
