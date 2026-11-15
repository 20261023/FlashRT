# TensorRT-LLM (vendored agent-skill material)

- Upstream: https://github.com/NVIDIA/TensorRT-LLM
- Pinned commit: `ef83ce246944131336e657681a951fa1c786bc6d`
- License: Apache-2.0
- Fetched: 2026-11-15
- Method: `gh api repos/NVIDIA/TensorRT-LLM/git/trees/<commit>?recursive=1`, then each file by blob SHA via `gh api repos/NVIDIA/TensorRT-LLM/git/blobs/<blob>`; `git hash-object` of every local file equals the upstream blob SHA.
- Local modifications: none (every file below is byte-identical to its upstream blob).

No NOTICE exists at this commit. `ATTRIBUTIONS-CPP-aarch64.md`, `ATTRIBUTIONS-CPP-x86_64.md` and `ATTRIBUTIONS-Python.md`
cover the library's third-party dependencies, not `.claude/skills`, and are not vendored. `LICENSE` here is the verbatim
root LICENSE (Apache-2.0 plus the list of third-party portions of the TensorRT-LLM code base).

## Files

| local path (repo-relative) | upstream path | blob | bytes |
|---|---|---|---|
| `.agents/skills/kernel-triton-writing/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/kernel-triton-writing/SKILL.md` | `.claude/skills/kernel-triton-writing/SKILL.md` | `1c1739b85159a9ea44a809e338d74526ddf54929` | 15233 |
| `.agents/skills/kernel-triton-writing/references/api-core.md` | `.claude/skills/kernel-triton-writing/references/api-core.md` | `f57ebb2a430aca112b26a3db5d239bfd589bb8e4` | 10382 |
| `.agents/skills/kernel-triton-writing/references/api-language.md` | `.claude/skills/kernel-triton-writing/references/api-language.md` | `db920415ef5d24de7b39957fe3a41eed266cc21e` | 12123 |
| `.agents/skills/kernel-triton-writing/references/concepts-semantics.md` | `.claude/skills/kernel-triton-writing/references/concepts-semantics.md` | `3f74cfb5c42e70e8736d0647ff26e13c8b6feac7` | 7892 |
| `.agents/skills/kernel-triton-writing/references/operator-routing.md` | `.claude/skills/kernel-triton-writing/references/operator-routing.md` | `d8d5ec17cf769f2d299870bf11b48a4017da5fe5` | 4428 |
| `.agents/skills/kernel-triton-writing/references/patterns-advanced.md` | `.claude/skills/kernel-triton-writing/references/patterns-advanced.md` | `8bed0f81f27b80e7c01e19a0af333bf3e182fdde` | 11758 |
| `.agents/skills/kernel-triton-writing/references/patterns-basic.md` | `.claude/skills/kernel-triton-writing/references/patterns-basic.md` | `d08f98e95f2ce7d0bfb54bed7fec3099055bc1ff` | 8812 |
| `.agents/skills/kernel-triton-writing/references/patterns-fusion.md` | `.claude/skills/kernel-triton-writing/references/patterns-fusion.md` | `ffd59dafc3f0b2abeaac0a873673e59b9fd241a0` | 10665 |
| `.agents/skills/kernel-triton-writing/references/patterns-gemm.md` | `.claude/skills/kernel-triton-writing/references/patterns-gemm.md` | `770026823a4902ae3f141ac968c2b3a3ceb2ebcd` | 12553 |
| `.agents/skills/kernel-triton-writing/references/troubleshooting.md` | `.claude/skills/kernel-triton-writing/references/troubleshooting.md` | `214ca6483a6c5a6dddea7ac4d9fd1dda2e1947b6` | 10767 |
| `.agents/skills/kernel-triton-writing/scripts/__init__.py` | `.claude/skills/kernel-triton-writing/scripts/__init__.py` | `df97ad0411e9cdba0fa2b7d3f8f2eac08587776a` | 684 |
| `.agents/skills/kernel-triton-writing/scripts/benchmark_kernel.py` | `.claude/skills/kernel-triton-writing/scripts/benchmark_kernel.py` | `2bc8881d3d1141e92b45547942ed4bd8c46e9260` | 9626 |
| `.agents/skills/kernel-triton-writing/scripts/verify_kernel.py` | `.claude/skills/kernel-triton-writing/scripts/verify_kernel.py` | `9d316ea29e771620a4c1f5153c16b0e7edd52a6f` | 12615 |
| `.agents/skills/perf-nsight-compute-analysis/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/perf-nsight-compute-analysis/SKILL.md` | `.claude/skills/perf-nsight-compute-analysis/SKILL.md` | `9d064b12357224edf82ca0d45114aead01bc5272` | 18526 |
| `.agents/skills/perf-nsight-compute-analysis/references/advanced-profiling.md` | `.claude/skills/perf-nsight-compute-analysis/references/advanced-profiling.md` | `6e0f71f7b08cec46f6e36165ed508be87d2e5419` | 11910 |
| `.agents/skills/perf-nsight-compute-analysis/references/bottleneck-guide.md` | `.claude/skills/perf-nsight-compute-analysis/references/bottleneck-guide.md` | `ac41c0c9900b590c06d3a46a17080a48ef7ec172` | 6672 |
| `.agents/skills/perf-nsight-compute-analysis/references/cli-reference.md` | `.claude/skills/perf-nsight-compute-analysis/references/cli-reference.md` | `27b1cb14120eca3c7918ab702d70902238792833` | 12204 |
| `.agents/skills/perf-nsight-compute-analysis/references/memory-analysis.md` | `.claude/skills/perf-nsight-compute-analysis/references/memory-analysis.md` | `8fb6e8e32e73d7e6b139b2d4bffac044a3b5d16c` | 6795 |
| `.agents/skills/perf-nsight-compute-analysis/references/metrics-guide.md` | `.claude/skills/perf-nsight-compute-analysis/references/metrics-guide.md` | `31ea64c82abe8ee5a274e5399c045251392fadff` | 7736 |
| `.agents/skills/perf-nsight-compute-analysis/references/python-report-api.md` | `.claude/skills/perf-nsight-compute-analysis/references/python-report-api.md` | `2cef83d8d0d1b29bf555d2665f52d87f21c686bc` | 7469 |
| `.agents/skills/perf-nsight-compute-analysis/references/roofline-analysis.md` | `.claude/skills/perf-nsight-compute-analysis/references/roofline-analysis.md` | `5204f905d7fb7f081e8b9e7f16394f1def1d2e15` | 5767 |
| `.agents/skills/perf-nsight-compute-analysis/references/sections-guide.md` | `.claude/skills/perf-nsight-compute-analysis/references/sections-guide.md` | `9e2a39cdf90daccd23af37a1907a5f89d69969f8` | 9085 |
| `.agents/skills/perf-nsight-systems/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/perf-nsight-systems/SKILL.md` | `.claude/skills/perf-nsight-systems/SKILL.md` | `9061440a8e73756fef7c8f27f10321fe288ad39a` | 14718 |
| `.agents/skills/perf-nsight-systems/references/app-preparation.md` | `.claude/skills/perf-nsight-systems/references/app-preparation.md` | `440bd4ba2c84e2fb1eb62c9718ddd7057ec90e0a` | 5922 |
| `.agents/skills/perf-nsight-systems/references/cli-post-collection.md` | `.claude/skills/perf-nsight-systems/references/cli-post-collection.md` | `26cbaa98c61e7a6d3a10268d2b4ecdd68d8d7717` | 6606 |
| `.agents/skills/perf-nsight-systems/references/cli-profiling.md` | `.claude/skills/perf-nsight-systems/references/cli-profiling.md` | `b1a9b55b25182a57fc6c4b2ae7033db53d8f4d9e` | 8890 |
| `.agents/skills/perf-nsight-systems/references/expert-systems.md` | `.claude/skills/perf-nsight-systems/references/expert-systems.md` | `9274ddee1836261732215d0ce2beb4e90d53cbd4` | 5256 |
| `.agents/skills/perf-nsight-systems/references/nvtx-analysis.md` | `.claude/skills/perf-nsight-systems/references/nvtx-analysis.md` | `e3c6a58719f3bfe304705758a3e6a6f689d70814` | 5670 |
| `.agents/skills/perf-nsight-systems/references/recipes-dl.md` | `.claude/skills/perf-nsight-systems/references/recipes-dl.md` | `7bc19ec563a4d6c2912fa19edc6ae9abb835e02d` | 6850 |
| `.agents/skills/perf-nsight-systems/references/stats-reports.md` | `.claude/skills/perf-nsight-systems/references/stats-reports.md` | `2417d6e9b13dcc66136098bfef3dda1da0adc13d` | 5796 |
| `.agents/skills/perf-torch-cuda-graphs/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/perf-torch-cuda-graphs/SKILL.md` | `.claude/skills/perf-torch-cuda-graphs/SKILL.md` | `150121861a4422fb79dab8f33d13a9c6b0b42c5e` | 24111 |
| `.agents/skills/perf-torch-cuda-graphs/references/api-pytorch.md` | `.claude/skills/perf-torch-cuda-graphs/references/api-pytorch.md` | `30a114a58395b4ea0dc2d6b5ab11ad666c6c9fb7` | 7586 |
| `.agents/skills/perf-torch-cuda-graphs/references/api-te-megatron.md` | `.claude/skills/perf-torch-cuda-graphs/references/api-te-megatron.md` | `5035b0d06598b63fc9822cb83fd8cb6ab78b108d` | 8362 |
| `.agents/skills/perf-torch-cuda-graphs/references/patterns-compatibility.md` | `.claude/skills/perf-torch-cuda-graphs/references/patterns-compatibility.md` | `8106e8e0a8aeb44c63f2e6c8cba98f3b0abed764` | 4640 |
| `.agents/skills/perf-torch-cuda-graphs/references/patterns-dynamic.md` | `.claude/skills/perf-torch-cuda-graphs/references/patterns-dynamic.md` | `543d60cb3e105d1c9fd48dbded4d76f0a3fc69c2` | 8123 |
| `.agents/skills/perf-torch-cuda-graphs/references/troubleshooting.md` | `.claude/skills/perf-torch-cuda-graphs/references/troubleshooting.md` | `dae4f1a5f3c779b0be8b6867fd565621e007ce06` | 8474 |
| `.agents/skills/perf-torch-cuda-graphs/scripts/verify_workload.py` | `.claude/skills/perf-torch-cuda-graphs/scripts/verify_workload.py` | `23c15ddab60af3ad901ffb708c953587b66fd86a` | 7017 |
| `.agents/skills/perf-torch-sync-free/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/perf-torch-sync-free/SKILL.md` | `.claude/skills/perf-torch-sync-free/SKILL.md` | `30ab16bd340d10474b08327e678e1cedcd5c9776` | 10091 |
| `.agents/skills/perf-torch-sync-free/references/sync-patterns.md` | `.claude/skills/perf-torch-sync-free/references/sync-patterns.md` | `2dee697aeca61cd8fabc0194f4d3dd76008bcb38` | 15932 |
| `.agents/skills/perf-torch-sync-free/scripts/verify_workload.py` | `.claude/skills/perf-torch-sync-free/scripts/verify_workload.py` | `39facbc77cf9aa7df17a4c0381588304ccc7c913` | 7001 |
| `.agents/skills/perf-workload-profiling/LICENSE.txt` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |
| `.agents/skills/perf-workload-profiling/SKILL.md` | `.claude/skills/perf-workload-profiling/SKILL.md` | `60d04ee451da528c844b1d93aa98fba6a6f47731` | 7859 |
| `.agents/skills/perf-workload-profiling/references/benchmarking-patterns.md` | `.claude/skills/perf-workload-profiling/references/benchmarking-patterns.md` | `d5cdf636befbac8072de881871e22e4eb4449660` | 5173 |
| `.agents/skills/perf-workload-profiling/references/nvtx-api.md` | `.claude/skills/perf-workload-profiling/references/nvtx-api.md` | `b26b11c92026b34ab47fcda7348628482cb5ccf2` | 2264 |
| `.agents/skills/perf-workload-profiling/references/pytorch-profiler-api.md` | `.claude/skills/perf-workload-profiling/references/pytorch-profiler-api.md` | `f0929c692ad196744383b4f6078e0d10900d6425` | 2061 |
| `third_party/agent-skills/tensorrt-llm/LICENSE` | `LICENSE` | `0b58d015e42efd81c98a44b91bb2a6dd7293e9f2` | 40678 |

## Validator notes

Vendored-unmodified skills whose upstream frontmatter or size falls outside the FlashRT rules for
authored/adapted skills. They are kept byte-identical on purpose:

- `perf-torch-cuda-graphs`: frontmatter key `tags`; SKILL.md has 634 lines (over the 500-line guideline)
- `perf-torch-sync-free`: frontmatter key `tags`
- `perf-nsight-compute-analysis`: frontmatter keys `tags`, `compatibility` (Codex quick_validate rejects both)
- `kernel-triton-writing`: frontmatter key `tags` (outside name/description/license/allowed-tools/metadata)

## .gitignore

Every local path above was checked with `git check-ignore -v --no-index`: none is ignored.
