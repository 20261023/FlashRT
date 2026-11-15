# AITune (vendored agent-skill material)

- Upstream: https://github.com/ai-dynamo/aitune
- Pinned commit: `1d3a4236ad8f458669d5633715b6956125f2289c`
- License: Apache-2.0
- Fetched: 2026-11-15
- Method: `gh api repos/ai-dynamo/aitune/git/trees/<commit>?recursive=1`, then each file by blob SHA via `gh api repos/ai-dynamo/aitune/git/blobs/<blob>`; `git hash-object` of every local file equals the upstream blob SHA.
- Local modifications: none (every file below is byte-identical to its upstream blob).

No NOTICE exists at this commit; the verbatim root `COPYRIGHT` file is kept next to `LICENSE`.

## Files

| local path (repo-relative) | upstream path | blob | bytes |
|---|---|---|---|
| `.agents/skills/aitune-benchmark/LICENSE.txt` | `LICENSE` | `895657b9a966424100865826cff85a6a1b4c8bb9` | 10140 |
| `.agents/skills/aitune-benchmark/SKILL.md` | `.claude/skills/aitune-benchmark/SKILL.md` | `4f99ebda41b381a55f32df8d50841aef6557faff` | 3795 |
| `.agents/skills/aitune-inspect/LICENSE.txt` | `LICENSE` | `895657b9a966424100865826cff85a6a1b4c8bb9` | 10140 |
| `.agents/skills/aitune-inspect/SKILL.md` | `.claude/skills/aitune-inspect/SKILL.md` | `bed7e35883a923aa169a0fb00c8681ecb03c8f08` | 1776 |
| `.agents/skills/aitune-tune/LICENSE.txt` | `LICENSE` | `895657b9a966424100865826cff85a6a1b4c8bb9` | 10140 |
| `.agents/skills/aitune-tune/SKILL.md` | `.claude/skills/aitune-tune/SKILL.md` | `1bfd018f683d9f37d16970c62fec4411438f4e4b` | 13680 |
| `.agents/skills/aitune-tune/common-errors.md` | `.claude/skills/aitune-tune/common-errors.md` | `546d2e5dae4f06c1eb65aade552c95a684d026d5` | 998 |
| `.agents/skills/aitune-tune/how-to-aot-tune.md` | `.claude/skills/aitune-tune/how-to-aot-tune.md` | `4b23f4d88b526d13b4d8e5158fa4606d5b08d259` | 1026 |
| `.agents/skills/aitune-tune/how-to-jit-tune.md` | `.claude/skills/aitune-tune/how-to-jit-tune.md` | `da46784d221936d4a7a859a88aaa754effb20407` | 2697 |
| `.agents/skills/aitune-validate/LICENSE.txt` | `LICENSE` | `895657b9a966424100865826cff85a6a1b4c8bb9` | 10140 |
| `.agents/skills/aitune-validate/SKILL.md` | `.claude/skills/aitune-validate/SKILL.md` | `c5195c5c2a9d1c7fc31ad773b0bf377dee0b1bef` | 1216 |
| `third_party/agent-skills/aitune/COPYRIGHT` | `COPYRIGHT` | `9031d2eca5569e269c71fc2f52109e619e8ea5fd` | 654 |
| `third_party/agent-skills/aitune/LICENSE` | `LICENSE` | `895657b9a966424100865826cff85a6a1b4c8bb9` | 10140 |

## Validator notes

Vendored-unmodified skills whose upstream frontmatter or size falls outside the FlashRT rules for
authored/adapted skills. They are kept byte-identical on purpose:

- `aitune-inspect`: YAML comment lines (SPDX header) inside the frontmatter
- `aitune-tune`: YAML comment lines (SPDX header) inside the frontmatter
- `aitune-benchmark`: YAML comment lines (SPDX header) inside the frontmatter; fine for YAML parsers
- `aitune-validate`: YAML comment lines (SPDX header) inside the frontmatter

## .gitignore

Every local path above was checked with `git check-ignore -v --no-index`: none is ignored.
