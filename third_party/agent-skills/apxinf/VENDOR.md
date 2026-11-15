# ApxInf (vendored agent-skill material)

- Upstream: https://github.com/infinigence/ApxInf
- Pinned commit: `b05b40990387e8c48e7d6267a790664981a665e4`
- License: Apache-2.0
- Fetched: 2026-11-15
- Method: `gh api repos/infinigence/ApxInf/git/trees/<commit>?recursive=1`, then each file by blob SHA via `gh api repos/infinigence/ApxInf/git/blobs/<blob>`; `git hash-object` of every local file equals the upstream blob SHA.
- Local modifications: none (every file below is byte-identical to its upstream blob).

Path mapping: upstream `AGENTS.md` is stored as `AGENTS.upstream.md` (content unchanged) so that Codex/Claude Code never
load ApxInf repo instructions as instructions for this directory. Everything here is an archive: nothing under
`third_party/agent-skills/` is discovered as a skill. FlashRT adapts these skills in `.agents/skills/vla-*`.

## Files

| local path (repo-relative) | upstream path | blob | bytes |
|---|---|---|---|
| `third_party/agent-skills/apxinf/AGENTS.upstream.md` | `AGENTS.md` | `0b6c04e9b4fbdddfd483b7336f657f17b1baff44` | 2459 |
| `third_party/agent-skills/apxinf/LICENSE` | `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11357 |
| `third_party/agent-skills/apxinf/NOTICE` | `NOTICE` | `7bcf3b783517301c325d88a461784db92f6fd78a` | 7324 |
| `third_party/agent-skills/apxinf/doc/agentic-optimization-workflow.md` | `doc/agentic-optimization-workflow.md` | `ae33b07a02921b749f54fff8c248dab0940544d7` | 8050 |
| `third_party/agent-skills/apxinf/skills/analyze-model-impl/SKILL.md` | `skills/analyze-model-impl/SKILL.md` | `f82f756c946ea32677bfd64e03a05eef60b31acc` | 9902 |
| `third_party/agent-skills/apxinf/skills/analyze-model-impl/agents/openai.yaml` | `skills/analyze-model-impl/agents/openai.yaml` | `5aa28a8db835943ea9527534c1255005fa7a4879` | 345 |
| `third_party/agent-skills/apxinf/skills/copy-drift-audit/SKILL.md` | `skills/copy-drift-audit/SKILL.md` | `fde0e504564e39c158e0145170c39df52bad90cf` | 14723 |
| `third_party/agent-skills/apxinf/skills/l3-reference-testing/SKILL.md` | `skills/l3-reference-testing/SKILL.md` | `e1ca0fc07fe66d84988504b39e2992973fdab86b` | 4178 |
| `third_party/agent-skills/apxinf/skills/model-port-workflow/SKILL.md` | `skills/model-port-workflow/SKILL.md` | `388aed6842a8abf7de234fb4f4b578817421275c` | 14001 |

## .gitignore

Every local path above was checked with `git check-ignore -v --no-index`: none is ignored.
