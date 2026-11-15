---
name: vla-model-impl-map
description: Trace the ACTIVE implementation of a FlashRT VLA candidate (pi0, pi0.5, Hy-Embodied-0.5-VLA, FastWAM) from BackendConfig and registry.load_backend through the frontend class, model pipeline, torch ops or fvk pybind kernels, cuBLASLt GEMM runners, SDPA/FA2 backend selection, CUDA Graph caches and keys, host syncs and per-stage dtypes, and tag every row with an autotune stage id. Use for implementation walkthroughs, which-kernel-runs-where questions, fusion or graph-capture analysis, the static half of stage attribution, or before choosing a call site for a kernel, patch or port candidate. Chinese triggers - 执行地图, 实现走读, 哪个 kernel, 算子在哪个游, 静态归因, CUDA Graph 缓存.
license: Apache-2.0. Complete terms in LICENSE.txt
metadata:
  upstream-repo: "https://github.com/infinigence/ApxInf"
  upstream-path: "skills/analyze-model-impl"
  upstream-commit: "b05b40990387e8c48e7d6267a790664981a665e4"
---
<!-- Adapted by FlashRT (2026-11-15) from ApxInf skills/analyze-model-impl/SKILL.md @ b05b40990387e8c48e7d6267a790664981a665e4 (Apache-2.0). Modifications: renamed; ApxInf Rust/C-ABI/crate locations replaced by FlashRT's Python frontends, model pipelines, fvk pybind bindings, GemmRunner/cuBLASLt, SDPA/FA2 and CUDA Graph caches; table columns re-labelled; added stage-id tagging against flash_rt/vla/autotune/stages.py, snapshot-candidate scope and the devlocal report rule; FlashRT entry points moved to references/. See NOTICE. -->

# VLA model implementation map

Produce an implementation-backed execution map of one FlashRT VLA candidate and
explain what it implies for precision, memory, graph capture and performance.
Map the requested build and execution path: a kernel existing in `csrc/`, a
design note or an option name does not prove that the candidate executes it.
An analysis request does not by itself require model changes, a rebuild, a GPU
allocation or new benchmarks.

In the autotune loop this map is the **static half of stage attribution**: every
row carries a stage id from `flash_rt/vla/autotune/stages.py`. The dynamic half
(measured interval time per stage) is `vla-stage-profile`. The map says *what*
runs in a stage. Only a measurement says *how long* it takes.

## Establish the implementation being analyzed

- Resolve the checkout. Record `git rev-parse HEAD` and `git status --porcelain`.
  If the candidate is a snapshot (`source_ref` + `patch_ref`), map the snapshot's
  `source_dir` from its manifest
  (`python -m flash_rt.vla.autotune.snapshots show --store S --snapshot-id ID`), not the
  live checkout.
- Record the candidate's `BackendConfig` (`flash_rt/vla/config.py`): model,
  backend, sampler, framework, hardware label (`cuda_sm80`, `rtx_sm87`,
  `rtx_sm89`, `thor`, `rtx_sm120`), precision, num_steps, probeflow and
  options. Run the CPU-only checks first: `registry.validate_config(config)`
  must accept it, and `registry.describe_backend(config)` shows the resolved
  description. A config that fails validation has no implementation to map.
- Record every environment variable that changes dispatch. Examples:
  `FVK_RTX_FA2`, `FVK_RTX_FA2_SITES`, `FVK_PI05_RTX_*` and `FLASHRT_PI05_*`.
  Use the environment of the node that ran the measurement, not only the
  login shell's.
- Identify the checkpoint and read dimensions from its config. Do not take
  them from docstring constants: those are context, and you verify them.
- Distinguish the default path, the configuration of an accepted run, and
  experimental alternatives. When the request leaves this unclear, say which
  path you mapped. Follow option defaults, environment precedence, shape
  restrictions and architecture gates. Remember the CMake gates as well: a
  binding in `csrc/bindings.cpp` can be compiled out for the target `GPU_ARCH`.
- Read the relevant docs (`docs/vla_autotune.md`,
  `examples/vla_autotune/README.md`, module docstrings) for context. Then verify
  every claim against executable code. Label older timings, proposals and stale
  comments; do not present them as current facts.
- Preserve the user's precision and checkpoint choices. Name any dispatch branch
  you cannot resolve, and any missing artifact (an unbuilt `.so`, an absent
  checkpoint). Keep mapping the paths you can establish locally.

## Trace the execution chain

Start at `flash_rt/vla/registry.py:load_backend` and the frontend's `infer`.
Follow the concrete path down to the device and record the condition that
selects each branch:

1. **Frontend class.** `load_backend` imports one class per model and pins some
   kwargs. For example, `use_fp8=False` keeps pi0/pi0.5 in BF16/FP16, and pi0.5
   also pins `prefix_precision`, `decoder_precision` and `decoder_kernel`. The
   per-model table is in
   [references/flashrt-execution-paths.md](references/flashrt-execution-paths.md).
   Read it before tracing a model you have not mapped in this session.
2. **Model pipeline.** `flash_rt/models/<model>/pipeline*.py` owns forward
   order. For pi0/pi0.5 the order is
   `vision_encoder -> transformer_encoder -> transformer_decoder`, for Hy it is
   `vit_forward`, `merger_forward`, `prefill`, then `evaluate_velocity` per step,
   and for FastWAM `_prepare_condition`, then `_evaluate_at_timestep` per step.
   Construction (weight loading, quantization, calibration, GEMM autotune, graph
   capture) and repeated inference are traced separately.
3. **Operation.** For each computation, record what is actually called. That is
   either a torch op (`F.linear`, `@`, `F.scaled_dot_product_attention`,
   `F.rms_norm`, elementwise) or an fvk call: a `flash_rt.flash_rt_kernels`
   function taking raw `data_ptr()`/stream ints, defined by `m.def("<name>", ...)`
   in `csrc/bindings.cpp` and implemented under `csrc/kernels/`, `csrc/gemm/`,
   `csrc/attention/` and the other `csrc/` subtrees. Other extension modules
   exist too: `flash_rt_fa2` (`csrc/fa2_bindings.cpp`), `flash_rt_fp4` and
   `flash_rt_gemm_plan` (`csrc/gemm/bf16_nt_plan.cpp`).
4. **Selected kernel or backend.**
   - For a custom fvk kernel, follow the binding into the launcher and record
     the `__global__` function and the template or config arguments that matter.
     Read the binding's checks too (supported `hd`, offsets and the like).
   - For `GemmRunner` (`csrc/gemm/gemm_runner.h`) and `CachedGemmRunner`,
     record the API (`bf16_nn`, `bf16_nt_bias`, `fp8_nt_dev`, ...), the operand
     and output dtypes, and how the algorithm is chosen: cuBLASLt heuristic,
     `autotune_bf16_nn` before capture, or a cached plan.
   - For SDPA, record the backend list the code allows (e.g.
     `sdpa_kernel([EFFICIENT_ATTENTION, MATH])`) and the mask, dtype and GQA
     conditions. Torch makes the final choice at runtime.
   - Do not invent a cuBLAS or SDPA device-kernel symbol from a source call, and
     do not reuse an old profiler name as proof of current dispatch. A device
     kernel name comes only from an nsys/ncu trace of the same build and config
     (`perf-nsight-systems`, `profile.ncu_kernel`). Label anything else as
     "library call, kernel not observed".
5. **Host work.** Name computation that runs on the CPU or in another backend:
   tokenization, prompt embedding caches, ProbeFlow routing, upstream FastWAM
   modules that the native path does not replace
   (`execution_coverage: partial-native-linear-upstream-other-operators`).

Use `rg --files` and `rg -n '<symbol>'` to find real files before you assume a
layout. Read call sites and launcher bodies, not only definitions. A name like
`rope_qknorm_kvwrite` or `bf16_nt_bias` does not tell you which steps are fused
or how many launches happen. Read the body.

## Map computations and data representation

- **Dimensions.** Verify hidden width, query/KV widths (do not assume
  q_dim == hidden), head grouping (GQA), intermediate width, number of views,
  vision tokens, prefix length, chunk/horizon and steps from the checkpoint and
  config. Track them where they explain a kernel choice.
- **Phases.** Loading covers the safetensors read, layout transforms,
  quantization, scale calibration, persistent buffers and workspaces. Per
  request covers host input, vision, projection, prefix and KV fill. Per
  denoise step covers velocity. Host output covers the D2H copy and action
  decode. Keep these tables separate when their graphs differ.
- **Frequency.** Make it explicit: once per load, per prompt change
  (`set_prompt`), per request, per camera, per layer, or per action-expert call
  (NFE, which can vary per request with ProbeFlow).
- **Precision.** Per stage, trace weight storage, activation dtype, accumulator,
  output dtype and rounding boundaries. Some frontends describe their own
  dtypes; Hy's `_operation_dtypes()` is one. Verify such a description against
  the code rather than quoting it. A fused op can keep the intermediate
  rounding; check before claiming fusion changes numerics. Locate norms folded
  into the previous layer's kernel instead of adding duplicate rows.

## Inspect graph capture and memory behavior

Follow the runtime's real loop and capture lifecycle:

- **Graph caches and keys.** Find what is captured, the cache key, and where a
  miss captures during a request: prompt length, `(S_p, n_vis, times)`,
  `(num_cam, K)`, velocity geometry. A miss inside a timed request is
  first-request latency, not steady state. The known caches are listed in the
  references file.
- **Capture steps.** Separate eager execution, warmup (usually 2-3 runs on a
  side stream), capture, instantiation and replay. Note which inputs are
  copied into static buffers before each replay, and which outputs are
  borrowed scratch that the next replay overwrites.
- **What stays outside the graph.** For example, ProbeFlow cosine routing
  (`.item()`) and Hy/FastWAM integration, plus prefix assembly and host
  preprocessing.
- **Host syncs and transfers.** Look for `.item()`, `.cpu()`, `.tolist()`,
  `torch.cuda.synchronize()`, `cudaStreamSynchronize` and `torch.isfinite(...).all().item()`
  validation, with their frequency. `perf-torch-sync-free` covers how to remove
  them.
- **Allocations.** Note per-call `torch.empty` and `record_stream` in the hot
  path, KV/cache buffers versus scratch, what grows with views, prompt length or
  steps, and how many graph instances each key creates (per-time velocity
  graphs, for example).
- **Repeated or wasted work.** For example, text cross-attention K/V recomputed
  every step although it is constant per request (FastWAM; tool
  `cache.fastwam_text_crossattn_kv`).

If launch counts help, report explicit device launches and library calls
separately, label them as static counts, and state their scope. One Python call
can launch several kernels, and one kernel can perform several model
computations. Do not count load or capture work as steady state.

## Tag rows with stage ids

Run `python -m flash_rt.vla.autotune.stages list --model <model>` and
`... show <stage_id>`. Each `MODEL_STAGES` entry lists the seams
(`file:Class.method`) that belong to a stage. For each row:

- Give it its primary stage (`host.input`, `vision.encode`, `vision.project`,
  `prefix.prefill`, `denoise.velocity`, `denoise.routing`, `host.output`), plus
  cross-cutting tags where they apply: `kernel.gemm`, `kernel.attention`,
  `kernel.norm_elementwise`, `kernel.conv`, `precision.weights` and
  `host.launch`. `denoise.schedule` is a count stage: record NFE, not a row.
- If a row's function is not a listed seam, flag it as a **stage-map gap**. Do
  not force-fit it. That is a finding for the `stages.py` owner.
- Note which profiling interval key (`PROFILE_KEYS`) times each row, and which
  rows fall into an unsplittable or unattributed interval. Examples: pi0/pi0.5
  Euler plus graph uses `full`, and Hy uses `vision`, `prefix` and `velocity`.
  `python -m flash_rt.vla.autotune.stages check-seams` must pass before you rely
  on the seam list.

## Present the mapping and assessment

Lead with the main implementation choice and its scope: revision, snapshot id,
candidate, hardware, checkpoint and env. Then give tables in execution order,
one per phase:

| Stage | Model computation and frequency | Python module/function | torch op or fvk binding | Selected kernel/backend | Shape, precision, fusion, memory, graph detail |
|---|---|---|---|---|---|

Use actual symbols and `path:symbol` locations, so a reader can open
orchestration, binding and launcher. Define repeated template names once as
aliases. A compact dependency graph is useful when the order alone hides a
dependency.

Then explain what the map establishes:

- active fusions and the materializations or casts that remain;
- where routing and integration run;
- weight reuse and repeated per-step work;
- what scales with views, prompt or steps;
- the graph's remaining host dependencies.

Separate implemented-and-enabled paths from build-gated or disabled ones and
from proposals. Treat bottleneck claims as hypotheses until measured.
Source-level counts do not predict a speedup. For candidate tools, cite the
registry (`python -m flash_rt.vla.autotune.tools list --model M --all`).

When you discuss acceptance, keep these apart: operator numerics
(`vla-operator-gate`), model-level action quality (the controller's
`--quality-gate`), eager-versus-graph parity, and benchmark milestones. Say
which fixtures and configuration were validated.

**Where the report goes.** Write task reports to
`devlocal/<feat>/reports/<model>-impl-map.md`. FlashRT's `.gitignore` ignores
`/devlocal/`; confirm with `git check-ignore -v devlocal/<feat>/reports/x.md`.
Link the report in your answer. Promote it to `docs/` only when the task asks for
that. Machine paths, checkpoint flags, measured speeds and this session's kernel
choices stay in the report; they never go into a skill. Optionally record the
headline as a `diagnosis` in the decision log with `--provenance source_fact`
(`python -m flash_rt.vla.autotune.decisions add --log L --kind diagnosis --actor agent ...`).

## Check the result

- Every selected path follows the recorded config, env and build gates.
- Every kernel name is either a device kernel you located or a clearly labeled
  library call.
- The tables cover entry and output work, layer repetition and per-step
  frequency, and fused norms without double counting.
- Every row has a stage id, or is flagged as a stage-map gap.
- Source links and any counts are valid.
- Say where the evidence came from: source inspection, recovered measurements,
  or newly executed runs. Never claim runtime validation for a static analysis.
