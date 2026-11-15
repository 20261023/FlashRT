# Third-party agent skills

FlashRT's agent skills for the VLA autotune pipeline (`flash_rt/vla/autotune/`) live in
`.agents/skills/<name>/SKILL.md` (Codex discovers them there). Claude Code reaches the same folders through
`.claude/skills/<name> -> ../../.agents/skills/<name>` directory symlinks. This directory holds the provenance of every
third-party file behind those skills:

- `<project>/VENDOR.md`: upstream URL, pinned commit, fetch date, and one row per file (local path, upstream path,
  git blob SHA, bytes). Also lists upstream files that were deliberately not vendored and any validator notes.
- `<project>/LICENSE` and `NOTICE`/`COPYRIGHT` where upstream has them, copied verbatim.
- Verbatim archives of upstream skills that FlashRT adapted or only learned from. Nothing under `third_party/` is
  discovered as a skill.
- `manifest.json`: the machine-readable list of every vendored file, active or archived. Each entry has
  `{project, repo, commit, upstream_path, local_path, blob, bytes, license}`.
- `verify_vendor.py`: re-hashes every file in the manifest and cross-checks the NOTICE and VENDOR.md tables.

FlashRT has changed none of the vendored files. The vendored-unmodified active skills each carry two files that
FlashRT added: `LICENSE.txt` (a verbatim copy of the upstream root license) and `NOTICE` (provenance, the upstream
notices that apply, and blob SHAs). The skills adapted from ApxInf (`vla-model-impl-map`, `vla-operator-gate`,
`vla-copy-drift-audit`, `vla-port-workflow`) are FlashRT-modified derivatives. Each one has its own `LICENSE.txt`,
a `NOTICE` that lists its changes, and `metadata.upstream-*` frontmatter. The originals are archived in `apxinf/`.

## (a) What is vendored, where, and why

The pipeline profiles a candidate, attributes latency to stages (`flash_rt/vla/autotune/stages.py`), picks a tool for
the bottleneck stage, and runs a config, port, patch or kernel campaign. These are the third-party skills each step
uses:

| pipeline need | stage ids / lane | skill (active path) | upstream (path @ commit) | license |
|---|---|---|---|---|
| CUDA Graph capture and replay, graph-compatibility fixes | `host.launch`, `launch.cuda_graph` lever, port lane acceptance | `.agents/skills/perf-torch-cuda-graphs/` | NVIDIA/TensorRT-LLM `.claude/skills/perf-torch-cuda-graphs` @ `ef83ce24` | Apache-2.0 |
| Host-device sync removal (`.item()`, `.cpu()`, ProbeFlow routing) | `denoise.routing`, `host.input`, `host.output` | `.agents/skills/perf-torch-sync-free/` | TensorRT-LLM `.claude/skills/perf-torch-sync-free` @ `ef83ce24` | Apache-2.0 |
| Timeline profiling (GPU idle gaps, launch gaps, NVTX ranges) | all stages, used by `vla-stage-profile` | `.agents/skills/perf-nsight-systems/` | TensorRT-LLM `.claude/skills/perf-nsight-systems` @ `ef83ce24` | Apache-2.0 |
| Kernel-level metrics (SOL%, roofline, stalls) | `kernel.gemm`, `kernel.attention`, `kernel.norm_elementwise`, `kernel.conv` | `.agents/skills/perf-nsight-compute-analysis/` | TensorRT-LLM `.claude/skills/perf-nsight-compute-analysis` @ `ef83ce24` | Apache-2.0 |
| Timing instrumentation (CUDA events, warmup, NVTX annotation) | measurement for every campaign | `.agents/skills/perf-workload-profiling/` | TensorRT-LLM `.claude/skills/perf-workload-profiling` @ `ef83ce24` | Apache-2.0 |
| Writing Triton kernels | kernel lane (`vla-kernel-candidate`) | `.agents/skills/kernel-triton-writing/` | TensorRT-LLM `.claude/skills/kernel-triton-writing` @ `ef83ce24` | Apache-2.0 |
| Kernel numerics (fwd and bwd against an fp64 oracle and an eager same-dtype baseline) | operator gate (`vla-operator-gate`) | `.agents/skills/kernel-numerics-verifier/` | pytorch/torchtitan `.claude/skills/kernel-numerics-verifier` @ `631d544b` | BSD-3-Clause |
| Compiler and backend autotune (inspect, tune, benchmark, validate a PyTorch module across backends) | port lane, `precision.weights`, `vision.encode` / `prefix.prefill` backend swaps | `.agents/skills/aitune-{inspect,tune,benchmark,validate}/` | ai-dynamo/aitune `.claude/skills/aitune-*` @ `1d3a4236` | Apache-2.0 |
| Kernel campaigns (launch a ForgeKernel operator optimization run) | kernel lane, tool `kernel.forgekernel` | `.agents/skills/forge-start/` (SKILL.md and scripts/; upstream tests/ omitted) | OpenBMB/ForgeKernel `integrations/claude-code/forge-start` @ `b2c8ae85` | Apache-2.0 |

These are archive only: not discovered, kept as sources for FlashRT-adapted skills or as reference.

| archive path | upstream @ commit | used by |
|---|---|---|
| `apxinf/skills/{analyze-model-impl,copy-drift-audit,l3-reference-testing,model-port-workflow}/`, `apxinf/AGENTS.upstream.md` (upstream `AGENTS.md`), `apxinf/doc/agentic-optimization-workflow.md` | infinigence/ApxInf @ `b05b4099` | adapted into `vla-model-impl-map`, `vla-copy-drift-audit`, `vla-operator-gate`, `vla-port-workflow` |
| `forgekernel/skills/README.md`, `forgekernel/skills/{job-env-check,lever-pricing,ncu-trace,orchestration,sass,stall,timeline}/SKILL.md` | OpenBMB/ForgeKernel @ `b2c8ae85` | reference for `vla-kernel-candidate`. Upstream has no frontmatter, and the helper scripts are not vendored (listed in `forgekernel/VENDOR.md`) |
| `torchtitan/torchtitan/rl/dot-claude/skills/inference_perf_hillclimb/SKILL.md` (upstream `torchtitan/rl/.claude/skills/...`) | pytorch/torchtitan @ `631d544b` | reference for the hill-climb loop in `vla-autotune` |

Two path deviations keep archives from being picked up as live instructions. The file contents are unchanged:
ApxInf's `AGENTS.md` is stored as `AGENTS.upstream.md`, and torchtitan's hidden `.claude` path segment is stored as
`dot-claude`. meta-pytorch/KernelAgent @ `e0647170` (Apache-2.0) ships no skills, so nothing was vendored from it.

## (b) Considered but not vendored

These are from the 2026-11-15 survey (`gh api` trees and frontmatter of each repository). Licenses are the
GitHub-reported SPDX id at the listed commit. `none` means no license file. `NOASSERTION` means GitHub could not
classify the license.

| repo | path(s) | commit | license | reason not vendored |
|---|---|---|---|---|
| NVIDIA/TensorRT-LLM | `.claude/skills/{perf-analysis,perf-optimization,perf-optimization-casebook,perf-regression-cookbook,perf-instability-cookbook,perf-host-analysis,perf-host-optimization}` | `ef83ce246944131336e657681a951fa1c786bc6d` | Apache-2.0 (root GitHub id NOASSERTION) | product-specific to TRT-LLM serving (trtllm-bench, PyExecutor, serve configs) |
| NVIDIA/TensorRT-LLM | `.claude/skills/{exec-env-check,exec-local-compile,exec-local-docker,exec-local-slurm,exec-remote-slurm,exec-slurm-compile}` | `ef83ce246944131336e657681a951fa1c786bc6d` | Apache-2.0 | TRT-LLM build and cluster execution, product-specific |
| NVIDIA/TensorRT-LLM | `.claude/skills/{kernel-cute-writing,kernel-tileir-optimization}` | `ef83ce246944131336e657681a951fa1c786bc6d` | Apache-2.0 | CuTe DSL / TileIR. FlashRT kernels are CUDA C++ (fvk) and Triton; can be added if a kernel lane needs them |
| NVIDIA/TensorRT-LLM | `.claude/skills/trtllm-*` (case-executor, code-contribution, codebase-exploration, flashinfer-upgrade, model-onboard-multimodal, moe-develop, serve-config-guide, test-script-builder, test-specialist, visualgen-bench, visualgen-lpips-golden-refresh) | `ef83ce246944131336e657681a951fa1c786bc6d` | Apache-2.0 | product-specific to TRT-LLM |
| pytorch/torchtitan | `.claude/skills/{cuda_graph_trace_compaction,numerics_debugging,local_compile_perf,fake_distributed_backend,torch_bisect,pr-description,pr-review}` | `631d544b1f6794ecfeca734a7bba861e03ddd5e3` | BSD-3-Clause | training-only or torchtitan repo maintenance. Underscore names fail the name rule |
| pytorch/torchtitan | `torchtitan/rl/.claude/skills/inference_perf_hillclimb/SKILL.md` | `631d544b1f6794ecfeca734a7bba861e03ddd5e3` | BSD-3-Clause | archived only (RL generator on vLLM, product-specific); the loop shape informs `vla-autotune` |
| ai-dynamo/aitune | `.claude/skills/{aitune-commit,aitune-docs-update,aitune-release}` | `1d3a4236ad8f458669d5633715b6956125f2289c` | Apache-2.0 | AITune repo maintenance |
| infinigence/ApxInf | `skills/*` | `b05b40990387e8c48e7d6267a790664981a665e4` | Apache-2.0 | archived; FlashRT ships adapted `vla-*` versions, not the ApxInf-specific originals |
| OpenBMB/ForgeKernel | `skills/{job-env-check,lever-pricing,ncu-trace,orchestration,sass,stall,timeline}` | `b2c8ae8510e02c28576808910a6b8827721ab7dc` | Apache-2.0 | frontmatter-less harness tools for ForgeKernel's own workspaces (`notes/<tool>.py`); archived as reference |
| sgl-project/sglang | `.agents/skills/{llm-torch-profiler-analysis,generate-profile,kl-consistency-test,debug-cuda-crash,sglang-multi-graph-check,add-jit-kernel}`, `python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-{performance,benchmark-profile,modelopt-quant,cache-dit}` | `d9ec74eb72b856419c6eb44733e00193d28ed840` | Apache-2.0 | product-specific to SGLang serving |
| vllm-project/vllm | `.agents/skills/{kernel-microbenchmark,model-microbenchmark,triton-kernel-writing,debug-ima}` (mirrored in `.claude/skills/`) | `1489434bd56a33c068b946eca3a24804e28859b8` | Apache-2.0 | product-specific to vLLM. TRT-LLM `kernel-triton-writing` covers Triton without vLLM coupling |
| vllm-project/vllm-omni | `.claude/skills/{diffusion-perf-opt,diffusion-accuracy-test,diffusion-perf-test,omni-perf-ceiling,quantization}` | `22721f0acdfe129ba2a8937057dd321efecf8aa1` | Apache-2.0 | product-specific to vLLM-Omni serving |
| flashinfer-ai/flashinfer | `.claude/skills/{benchmark-kernel,add-cuda-kernel,debug-cuda-crash}` | `0132cbc015787b0fcb04f2357ee31d405dc2b9c5` | Apache-2.0 | tied to FlashInfer's build and benchmark harness |
| pytorch/pytorch | `.agents/skills/{pt2-bug-basher,cuda-streams,aoti-debug}` | `1dcfb27239b3682a375f49580550b2e9866a2180` | NOASSERTION | license id unclear, and aimed at PyTorch core development |
| pytorch/ao | (skills surveyed in tree) | `4ab9d61da2937831743bc3f7904975fe1b32de7f` | NOASSERTION | license id unclear; torchao-internal |
| NVIDIA/Megatron-LM | `skills/mcore-determinism-debug` | `0be53c791ffb70fe746bc0cdeaedb1409b1e23d9` | NOASSERTION | training-only; license id unclear |
| NVIDIA/cutlass | (tree surveyed) | `b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd` | NOASSERTION | license id unclear; no VLA-relevant skill |
| NVIDIA/TileGym | (tree surveyed) | `e5f28379af8036e711dee42af929a04837509703` | NOASSERTION | license id unclear; cuTile-specific |
| NVIDIA/Model-Optimizer | `skills/{modelopt-benchmark-model-kernels,modelopt-compare-results,modelopt-evaluation,modelopt-ptq,modelopt-quant-recipe-search,ptq}` | `ef893e77084198c161cc908eb91380b3a47eccad` | Apache-2.0 | product-specific to the ModelOpt quantization flow; FlashRT quant levers are native |
| NVIDIA/skills | `skills/{compute-sanitizer-diagnose-cuda-bugs,deepstream-profile-pipeline,fastgen-inference,jetson-inference-mem-tune,jetson-llm-benchmark,mcore-run-on-slurm,nemo-mbridge-perf-cuda-graphs,nemo-relay-plugin-adaptive-tuning,nemo-rl-auto-research,newton-performance,rag-perf,tilegym-*,warp-compile-time-optimizer}` | `894f2d7dfee49085e84e21c1aad2513bc44a5b6d` | Apache-2.0 | product-specific (DeepStream, Jetson LLM, NeMo training, TileGym, Warp, Newton, RAG) |
| NVIDIA/omniperf | `.agents/skills/{diagnose-perf,nsys-analyze,nvtx-python,perf-tuning,profiling}` | `a68f3894bee4d31e84d7cdb32de48a005e8d51a5` | Apache-2.0 | product-specific to Omniverse Kit / Isaac Sim |
| NVIDIA/TensorRT | `.agents/skills/{trt-perf-analysis,trt-torch-quickstart}` | `72fc1a3d23a163216893b363fb562b9464865afb` | Apache-2.0 | TensorRT-engine workflow, outside the current autotune lanes |
| NVIDIA/trt-samples-for-hackathon-cn | `.claude/skills/{trt-debug-mismatch,trt-perf-analysis}` | `4ab69af5ef4468b0202e649ad92c6a020e5ecba4` | Apache-2.0 | TensorRT-engine workflow, sample-repo specific |
| NVIDIA/TransformerEngine | (tree surveyed) | `4f5c08e48b3e190d27c731c1f9515f021dec255a` | Apache-2.0 | training-oriented, no VLA-relevant skill |
| NVIDIA/Isaac-GR00T | (tree surveyed) | `3b49067943eb1ab2eb2203b93d461c4390c78da3` | Apache-2.0 | no performance skill |
| huggingface/lerobot | (tree surveyed) | `fe1c54594f6ba275fccf210b9f7b11f0e2324b89` | Apache-2.0 | no performance skill |
| Physical-Intelligence/openpi | (tree surveyed) | `215abfb217dbac7d5f1273282331b9b1866c0479` | Apache-2.0 | no agent skills |
| triton-lang/triton | (tree surveyed) | `1c53e566ef8f719ff9b69132a72ad943e5409492` | MIT | no agent skills relevant to kernel authoring workflow |
| huggingface/skills | (tree surveyed) | `25910b3d0ba31a2ba102f901cdc9e008c278a69c` | Apache-2.0 | Hub/product skills, not perf |
| huggingface/kernels | `kernel-builder/skills/{cuda-kernels,triton-kernels}` | `b214d7994770b2a93afc80e1a7bc607a9bfe2719` | Apache-2.0 | tied to kernel-builder Hub packaging |
| PaddlePaddle/FastDeploy | `.claude/skills/{nsys-capture,benchmark-compare}` | `d923caf15e1eb0da8e2efd75b0152e65cefeeddd` | Apache-2.0 | product-specific to FastDeploy serving |
| vipshop/cache-dit | `.agents/skills/{cache-dit-custom-kernel,cache-dit-quantization}` | `e085ea6ac723bc6876f2a32d5334f3211b5b43c6` | Apache-2.0 | product-specific to cache-dit |
| alibaba/atrex-kernel-agent | `src/aka/resources/skills/{gpu-measurement,gpu-kernel-baseline,gpu-kernel-episode-loop,autonomous-gpu-kernel-timeline}` | `64322a9775af04c5cb3b7b144e1c239b17883c6d` | Apache-2.0 | internal resources of its own agent harness |
| modular/skills | `profile-model` | `d6000ab8766edeffc273b2d73efe7701397ecd8b` | NOASSERTION | license id unclear; Modular MAX-specific |
| BBuf/AI-Infra-Auto-Driven-SKILLS | `skills/{llm-torch-profiler-analysis,torch-profiler-layer-track,model-compute-simulation}` | `01cfd0aa5efe103b361414c4e6380716f175cae2` | none | no license; LLM-serving oriented |
| wenyi-li/awesome-agent-kernel-skills | (index repo) | `c13be0b381fbb1308de947b2835e724127c94eb0` | none | no license; link list only |
| freemty/vla-profiler | `.claude/skills/project-skill` | `b54d3cf61c699b44f1092f44f1973b43c26d3480` | none | no license; generic project template |
| BytedTsinghua-SIA/CUDA-Agent | `agent_workdir/SKILL.md` | `473025c8af7878e928525138b5cb327d6bcdf9dd` | none | no license; no frontmatter; harness-internal |
| 0satan0/KernelMem | (tree surveyed) | `8b57ccc9adc2ae2f11fc487fd458a7ecc1ea014d` | none | no license |
| AMD-AGI/Apex | `tools/skills/{aiter-reflection,gpu-architecture-fundamentals,hip-kernel-optimization,kernel-exp-history,mi300-cdna3-architecture,mi300-hip-programming-insights,mi300-hip-vs-nvidia,pytorch-kernel-optimization,rocprof-compute,skill-creator,triton-hip-reference-kernel-search,triton-kernel-optimization,triton-kernel-reflection-prompts}` | `ecfad4d74812b5825a8bb5adf4da5e8a01abf188` | MIT | AMD/ROCm-only (MI300, HIP, aiter, rocprof) or generic |
| cfregly/gpu-perf-tune | `plugins/profile-and-optimize/skills/{inference-kernel-ncu-profile,inference-quantize-calibrate,inference-graph-diff,perf-baseline-record}` | `25d74d031732e2f76d30a43e44c3434b588f06b3` | MIT | tied to its own plugin harness; overlaps the TRT-LLM perf skills |
| slowlyC/agent-gpu-skills | (tree surveyed) | `ae02d076fd424f3c134a5738a0e5cc5f28e747c3` | MIT | generic; overlaps vendored skills |
| lucifer1004/VeloQ | (tree surveyed) | `d69af566b9a2ae2ad30e46f1b5c5c27ce2e63131` | MIT | repo-specific |
| tensormux/kernel-skills | (tree surveyed) | `7b7337a123f8711aa8e3d0452351d8fd30dde4b7` | MIT | generic; overlaps vendored skills |
| Bruce-Lee-LY/cuda_auto_tune | `cuda-auto-tune/SKILL.md` | `16bef8dc9bb7b3d3a8d733739d0ed4ddedb6b9bf` | MIT | tied to its own kernel templates |
| KernelFlow-ops/cuda-optimized-skill | `skills/cuda-kernel-optimizer/SKILL.md` | `114a6cba4c194e18d3fe23a4fc2251c982f34309` | MIT | standalone kernel loop; ForgeKernel/KernelAgent cover the kernel lane |
| Orchestra-Research/AI-Research-SKILLs | `18-multimodal/openpi/SKILL.md` | `773a52944ba4747a18bd4ae9ade53fff041adcbc` | MIT | openpi usage and fine-tuning, not inference performance |
| TongmingLAIC/AKO4ALL | `SKILL.md` | `bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e` | MIT | generic kernel loop; overlaps the kernel lane |
| K-Dense-AI/scientific-agent-skills | `skills/optimize-for-gpu` | `2d2065fb4520fa3f3f2bcf61cff26ec8b7eba4b5` | MIT | scientific Python (CuPy/Numba), not VLA inference |
| Amey-Thakur/AI-SKILLS | `skills/gpu-ai-infrastructure/{kernel-profiling-nsight,quantization-deployment}` | `c8c84206e1a05825d8b9dac12a31b3c6c345c649` | MIT | generic and shallow; TRT-LLM Nsight skills are deeper |
| ScaleDockHQ/llm-skills | `skills/{model-quantization,torchao}` | `f47ae4187a162a76758c565629c0aae21c9ffded` | MIT | generic LLM quantization |
| ZaowenChen/nsight-training-profile | `skills/nsight-training-profile` | `040a2b3a82d9b983d2fc147c1e70ae57771150f2` | MIT | training-only |
| a5c-ai/babysitter | `library/specializations/gpu-programming/skills/{cublas-cudnn,cuda-graphs,cutlass-triton,gpu-benchmarking,gpu-memory-analysis,nsight-profiler,tensorrt-optimization}` | `feb68abe397acc14f32b34984975c48fedba2b33` | MIT | generic generated library; TRT-LLM equivalents are deeper |
| drivelineresearch/autoresearch-claude-code | `skills/autoresearch` | `e05758fa8c07a26c63f370661c9edae39b964da1` | MIT | generic research loop; `vla-autotune` owns the loop |
| justinchuby/onnx-genai | `.agents/skills/profiling`, `.github/skills/cuda-perf-measurement` | `49590f7c22da2ee554aa131e394e8c5f50ef7961` | MIT | ONNX Runtime GenAI-specific |
| windog-labs/edge-fm-x | `.codex/skills/{cuda-skill,edge-fm-cuda-kernel-optimizer}` (+ nested `vendor/` sub-skills) | `83251d56a1a682729ef145535f6c4de23a4ce250` | Apache-2.0 | repo-specific; the nested vendored sub-skills have unclear provenance |
| meta-pytorch/KernelAgent | (no skills) | `e0647170da36ef9b059ac0bd3d60103aa4ed378b` | Apache-2.0 | ships no skills; driven as an external tool from `vla-kernel-candidate` |

## (c) Re-sync and verification

Offline check: hashes every file in `manifest.json` and cross-checks the NOTICE and VENDOR.md tables. Exits 1 on any
mismatch.

```bash
python3 -B third_party/agent-skills/verify_vendor.py
```

Online check, which needs `gh auth`: confirms that each `upstream_path` at the pinned commit still resolves to the
recorded blob.

```bash
python3 -B third_party/agent-skills/verify_vendor.py --check-upstream
```

Re-fetch every file from its pinned upstream blob, writing it only if it hashes correctly:

```bash
python3 -B third_party/agent-skills/verify_vendor.py --resync
```

To move to a new upstream commit, repeat for each file:

1. `gh api 'repos/<owner>/<repo>/git/trees/<commit>?recursive=1'`. Find the file and note its `sha` (blob) and `size`.
2. `gh api repos/<owner>/<repo>/git/blobs/<blob> --jq .content | base64 -d > <local_path>`.
3. `git hash-object <local_path>` must equal the blob SHA.
4. `git check-ignore -v --no-index <local_path>` must print nothing. FlashRT ignores `build*/`, `*.so`, `*.bin`,
   `*.npz`, `.cache/`, `notes/` and `results/` at any depth.
5. Update `manifest.json`, the project's `VENDOR.md`, the skill's `NOTICE` file list and the root `NOTICE`. Then run
   `verify_vendor.py` and `tests/test_vla_agent_skills.py`.

Never run fetched scripts as part of syncing. Treat them as data until they have been reviewed.
