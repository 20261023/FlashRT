# ForgeKernel (vendored agent-skill material)

- Upstream: https://github.com/OpenBMB/ForgeKernel
- Pinned commit: `b2c8ae8510e02c28576808910a6b8827721ab7dc`
- License: Apache-2.0
- Fetched: 2026-11-15
- Method: `gh api repos/OpenBMB/ForgeKernel/git/trees/<commit>?recursive=1`, then each file by blob SHA via `gh api repos/OpenBMB/ForgeKernel/git/blobs/<blob>`; `git hash-object` of every local file equals the upstream blob SHA.
- Local modifications: none (every file below is byte-identical to its upstream blob).

## Files

| local path (repo-relative) | upstream path | blob | bytes |
|---|---|---|---|
| `.agents/skills/forge-start/LICENSE.txt` | `LICENSE` | `d645695673349e3947e8e5ae42332d0ac3164cd7` | 11358 |
| `.agents/skills/forge-start/SKILL.md` | `integrations/claude-code/forge-start/SKILL.md` | `68ea8b12aae253be4fe1273a8235116fcb4bc631` | 15770 |
| `.agents/skills/forge-start/scripts/deploy.py` | `integrations/claude-code/forge-start/scripts/deploy.py` | `fc8a8abdcb7e8b93b752b0bca7d37422717a0ca0` | 5802 |
| `.agents/skills/forge-start/scripts/start.py` | `integrations/claude-code/forge-start/scripts/start.py` | `b61913d9f2ebb9fded2e89f1b8dc3cfbd6d53ef2` | 5382 |
| `third_party/agent-skills/forgekernel/LICENSE` | `LICENSE` | `d645695673349e3947e8e5ae42332d0ac3164cd7` | 11358 |
| `third_party/agent-skills/forgekernel/NOTICE` | `NOTICE` | `c0e04a577321b1135fabb75a11006fbce3288bbe` | 1322 |
| `third_party/agent-skills/forgekernel/skills/README.md` | `skills/README.md` | `7cd98b3041e35c52b50f6cb88cdab5f36efb5a31` | 1442 |
| `third_party/agent-skills/forgekernel/skills/job-env-check/SKILL.md` | `skills/job-env-check/SKILL.md` | `1fa710c82e0979b21f498c917067b42668ba1c82` | 622 |
| `third_party/agent-skills/forgekernel/skills/lever-pricing/SKILL.md` | `skills/lever-pricing/SKILL.md` | `ce9c7d6168f29bae551f2d84b7cbee0e9533ba89` | 1278 |
| `third_party/agent-skills/forgekernel/skills/ncu-trace/SKILL.md` | `skills/ncu-trace/SKILL.md` | `bdc8bb60ee48c17aa29fd32c6f8475811f6e2755` | 6671 |
| `third_party/agent-skills/forgekernel/skills/orchestration/SKILL.md` | `skills/orchestration/SKILL.md` | `48a05557230b9393a7eea34f5f4a57c1ce4e6125` | 1090 |
| `third_party/agent-skills/forgekernel/skills/sass/SKILL.md` | `skills/sass/SKILL.md` | `6fdc8c7b2ac948d631077dc2aad5dbd976846c78` | 1433 |
| `third_party/agent-skills/forgekernel/skills/stall/SKILL.md` | `skills/stall/SKILL.md` | `c4e751ce4968a7d02da161e62931a2e672caf824` | 818 |
| `third_party/agent-skills/forgekernel/skills/timeline/SKILL.md` | `skills/timeline/SKILL.md` | `a06f6d0e1afb72397cda1bb25814685196f7b1e2` | 1119 |

## Upstream files referenced but NOT vendored

Helper scripts of the internal ForgeKernel skills (run inside ForgeKernel workspaces as `notes/<tool>.py`) and the
forge-start unit tests. Not copied; referenced here by path@commit.

| upstream path@commit | blob | bytes |
|---|---|---|
| `integrations/claude-code/forge-start/tests/test_deploy.py@b2c8ae8510e0` | `c1c7aab35c1b4faa4b17a04c0dffca051c7a6654` | 4323 |
| `integrations/claude-code/forge-start/tests/test_start.py@b2c8ae8510e0` | `21f0bd64176454d131154033e06abbcabaa54a03` | 2545 |
| `skills/job-env-check/job_env_check.py@b2c8ae8510e0` | `9b4190170eb48f05f35d9999eeb4aa0484bda9c5` | 3570 |
| `skills/lever-pricing/lever_pricing_run.py@b2c8ae8510e0` | `3c9afe143c775ac7a59df712db3675961542e688` | 67820 |
| `skills/orchestration/orch_census_run.py@b2c8ae8510e0` | `858c470cbb07bca2511d2f91e0078ebe5135224a` | 13722 |
| `skills/orchestration/orch_tools.py@b2c8ae8510e0` | `916b5c8a52478883f6953cd89772797e4be7b2a6` | 58253 |
| `skills/orchestration/role_exposure_tools.py@b2c8ae8510e0` | `c5e8d6a6425ec624084b0175a809f97d685d6efc` | 18676 |
| `skills/sass/launch_tools.py@b2c8ae8510e0` | `0eb80a1dbf2bde9bfe61bce8c918388efabc291d` | 11222 |
| `skills/sass/sass_census.py@b2c8ae8510e0` | `8479e3820dc2213554d3393c07b1f0f538df3234` | 12770 |
| `skills/sass/sass_tools.py@b2c8ae8510e0` | `721d7319e85f13d8bb5ea506fda58f535f2b7f24` | 11042 |
| `skills/sass/sched_live_tools.py@b2c8ae8510e0` | `ceb3e7de4d29190e3d6ec7888135cf2129e2d32b` | 39871 |
| `skills/sass/struct_tools.py@b2c8ae8510e0` | `23bfd6557d109c05747d699ca52ba749359e7849` | 24953 |
| `skills/stall/stall_price_run.py@b2c8ae8510e0` | `75ccb691e9386306403cf5f284207a6330d228e1` | 20689 |
| `skills/stall/stall_tools.py@b2c8ae8510e0` | `c1c56c5f0478bfe923788ee10d77d9bfd6659617` | 31933 |
| `skills/timeline/pm_timeline_run.py@b2c8ae8510e0` | `8677538fb9394306aa2a7f55360f5984acff1024` | 3181 |
| `skills/timeline/pm_timeline_tools.py@b2c8ae8510e0` | `554c6fab7a0baa80fc58520bc86475e66404df88` | 15545 |
| `skills/timeline/timeline_probe.py@b2c8ae8510e0` | `dfd925c0bc23816959d4f3a63989cdb74c3b3ea4` | 83965 |

## .gitignore

Every local path above was checked with `git check-ignore -v --no-index`: none is ignored.
