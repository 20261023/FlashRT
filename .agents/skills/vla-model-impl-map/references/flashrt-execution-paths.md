# FlashRT VLA execution paths: where to start reading

**Read this when** you map a model you have not traced in this session, or when you
need the graph caches, host syncs and dispatch switches of a candidate.

These are source facts, verified by reading the files at commit `728e5e2` plus the
2026-11-15 working tree. They are **starting points, not conclusions**. Re-open
every file at the revision you are mapping. Symbols are given rather than line
numbers because line numbers drift. Nothing here is a measured cost.

## 1. Loader: `flash_rt/vla/registry.py`

- `validate_config(config)` runs the CPU-only checks in this order:
  1. capability blocker, framework and precision, option names and values;
  2. FastWAM sampler and option contract;
  3. pi0.5 env conflicts: `FVK_PI05_RTX_FORCE_INT8`,
     `FVK_PI05_RTX_INT8_ENCODER_ONLY` and `FVK_PI05_RTX_INT8_VISION` must not be
     `1`; `FLASHRT_PI05_PREFIX_PRECISION`, `FLASHRT_PI05_DECODER_PRECISION` and
     `FLASHRT_PI05_DECODER_KERNEL` are restricted.
- `load_backend(config)` calls `validate_config`, loads the action decoder
  (pi0/pi0.5), resolves the hardware label from the real device
  (`_CUDA_TARGETS`), then imports one frontend:

| model | frontend module : class | pinned kwargs |
|---|---|---|
| pi0 | `flash_rt/frontends/torch/pi0_rtx.py : Pi0TorchFrontendRtx` (JAX: `frontends/jax/pi0_rtx.py`) | `use_fp8=False`, `use_fp8_decoder=False` |
| pi05 | `flash_rt/frontends/torch/pi05_rtx.py : Pi05TorchFrontendRtx` | `use_fp8=False`, `hardware`, `prefix_precision='fp8'`, `decoder_precision='fp8'`, `decoder_kernel='cublaslt'` (selector labels only; execution stays BF16 with `use_fp8=False`) |
| hyvla | `flash_rt/frontends/torch/hyvla_probeflow.py : HyVLAFlowFrontend` (subclass of `hyvla_thor.py : HyVLATorchFrontendThor`) | `hardware`; all quantized/fused flags must be `False` |
| fastwam | `flash_rt/frontends/torch/fastwam.py : FastWAMTorchFrontend` | `native_linear = (backend == 'flashrt')`, `dtype='bf16'`, `num_inference_steps` |

`flash_rt/vla/backend.py : VLABackend.infer` wraps the frontend, so it is part of
`host.output`.

## 2. pi0 / pi0.5 (RTX pipelines)

- Pipelines:
  - pi0: `flash_rt/models/pi0/pipeline_rtx.py : Pi0Pipeline`, plus
    `pipeline_rtx_probeflow.py : Pi0ProbeFlowPipeline`;
  - pi0.5: `flash_rt/models/pi05/pipeline_rtx.py : Pi05Pipeline`, plus
    `pipeline_rtx_probeflow.py : Pi05ProbeFlowPipeline`.
- Order: `run_pipeline` runs `_copy_lang_embeds_to_encoder_x`, then
  `vision_encoder` (27-layer SigLIP), then `transformer_encoder` (Gemma prefix,
  which fills the encoder K/V), then `transformer_decoder` (all Euler steps).
- Operations are raw-pointer fvk calls on CUDA streams. Examples:
  `fvk.patch_im2col` and `fvk.bias_residual`, `GemmRunner.bf16_nn` (cuBLASLt),
  `fvk.quantize_*`, and the `fvk.pi05_dec_skinny_*` decoder kernels (only when
  FP8 decoder plus skinny is selected).
- Attention goes through `flash_rt/hardware/rtx/attn_backend.py :
  RtxFlashAttnBackend`:
  - vendored FA2 (`flash_rt.flash_rt_fa2`, `csrc/fa2_bindings.cpp`) when
    `FVK_RTX_FA2=1`, which is the default;
  - per-site selection with `FVK_RTX_FA2_SITES` (`siglip`, `encoder`,
    `decoder`);
  - pip `flash_attn` when `FVK_RTX_FA2=0`.
  - FA2 is compiled in only when `ENABLE_FA2` is set by `GPU_ARCH` in
    `CMakeLists.txt` (80/86/87/89/120/121; 110 only with
    `FLASHRT_ENABLE_THOR_FA2`).
- GEMM algorithm choice: `autotune_gemms()` runs `gemm.autotune_bf16_nn(...)` per
  shape before capture, and is idempotent.
- CUDA Graph (`flash_rt/core/cuda_graph.py : CUDAGraph`, raw cudart, relaxed
  capture mode):
  - `record_infer_graph` captures the whole `run_pipeline`, plus a decoder-only
    graph used for temporal K/V reuse (`forward_decode_only`).
  - ProbeFlow pipelines capture the prefix once, plus one velocity graph per
    sampler time (`_velocity_graphs`). The routing in
    `sample_probeflow_torch_` stays outside the graphs.
- Capture keys on the pi0.5 frontend:
  - `state_prompt_mode='exact'` keeps one pipeline (and graph) per prompt
    length in `_prompt_pipeline_cache`;
  - `'fixed'` uses one `_fixed_pipeline`;
  - a new prompt length triggers build, calibration and capture.
- Host syncs:
  - `cudaStreamSynchronize` after replay in `forward`;
  - `self._noise_out.float().cpu().numpy()` in `infer`, the action download;
  - the prompt embedding round-trip `.cpu().numpy()` in `_embed_prompt_cached`,
    which runs per prompt change. Task-only prompts are cached by
    `_PromptEmbedCache`. A dynamic state in the prompt bypasses the cache, so
    then it runs every time the prompt is set;
  - ProbeFlow `F.cosine_similarity(...).item()` and the
    `torch.isfinite(...).all().item()` checks.
- Static stage note: with Euler plus a graph, the profiling interval is `full`,
  and velocity time per call cannot be separated from it.

## 3. Hy-Embodied-0.5-VLA

- Frontend chain: `HyVLAFlowFrontend` adds ProbeFlow/Euler sampling and
  numerical profiles to `HyVLATorchFrontendThor`. That class does checkpoint
  IO, `_preprocess_images`, `_tokenize`, `_assemble_prefix` and `_vit_merge`,
  and binds `flash_rt/models/hyvla/pipeline_thor.py : HyVLAThorBF16Pipeline`.
- Pipeline methods:
  - `vit_forward`, the 27-block ViT;
  - `merger_forward`: proj1, pooler, GELU, proj2;
  - `prefill`, the 32-layer MoT VLM over sorted `[vision|text]`, which writes
    `kbuf`/`vbuf`;
  - `evaluate_velocity`, the expert tower and action head, once per NFE;
  - `embed_state`.
- Operations in the default `semantic_bf16` path:
  - torch BF16 GEMMs (`@ w.t()`, `F.linear`);
  - attention: `F.scaled_dot_product_attention` under
    `sdpa_kernel([EFFICIENT_ATTENTION, MATH])` with KV pre-expanded to the query
    heads (the code comment explains that a bool mask with `enable_gqa` forces
    the math backend);
  - `torch.rms_norm` / `F.rms_norm`;
  - RoPE through `_rot_half`.
- `official_local_fp32` changes RoPE and QK-norm, eager-attention casts and
  integration (see `_operation_dtypes()`).
- Fused option: `fvk.hyvla_rope_qknorm_kvwrite_bf16` is used only when
  `_fused_attn` is set.
  - `HyVLAFlowFrontend` rejects `use_fused`.
  - `CMakeLists.txt` compiles the kernel only with `FLASHRT_ENABLE_HYVLA=ON` and
    `GPU_ARCH` 110 or 87/120/121, so it is absent on SM80.
  - Registry tool `fusion.hy_rope_qknorm_kvwrite`.
- CUDA Graph caches:
  - `_vit_graph_cache[(num_cam, K)]` holds `{graph, img, out}` for
    ViT+merger, built in `_build_vit_graph` (warmup ×3 on a side stream).
  - `_graph_cache[(S_p, n_vis, model_times)]` holds `{buf, prefix_graph,
    velocity_graphs{time: graph}}`, built in `HyVLAFlowFrontend._build_graph`.
    Every graph keeps its own private pool. Inputs (`pe`, `pmask`, `pcos`,
    `psin`, `smask`, `scos`, `ssin`, `state`, `x`) are copied in before each
    replay.
- Host syncs and host work:
  - `int(mm.sum().item())` for `n_vis` in `predict_actions`, per request;
  - `x_t.float().cpu().numpy()`;
  - routing and Euler/ProbeFlow integration in `_sample_latent`, which run
    outside the graphs.
- Profiling keys: `vision`, `prefix` and `velocity`. The `vision` key comes
  from the `_vit_graph_cache` graph wrapper in `profiling.py`. A graph that is
  first built inside a profiled request is not wrapped.

## 4. FastWAM

- Path: `FastWAMTorchFrontend` creates
  `flash_rt/models/fastwam/pipeline.py : FastWAMActionPipeline` over the
  **pinned upstream FastWAM model**. The source comes from `source_dir`; the
  pin is in `examples/vla_autotune/fastwam-source/SOURCE.json`.
- Order: `infer` runs `_initial_state`, then `_prepare_condition` (VAE
  first-frame encode, video-expert prefill and video K/V cache), then the
  `_euler` loop (or `sample_probeflow_torch_`), which calls
  `_evaluate_at_timestep` and then `model._denoise_action_with_video_cache`.
- Native coverage is **partial**. `native_linear.install_native_linears`
  replaces only ordinary `nn.Linear` modules on the active paths with
  `FlashRTLinear`. These are the `blocks.*` modules, `text_embedding.`,
  `time_embedding.` and `time_projection.`, and the action expert's
  `action_encoder` and `head`. `FlashRTLinear` calls
  `runner.bf16_nt_bias(A, W[N,K], D, bias, M, N, K, stream)`.
  - `native_linear_backend='legacy'` uses `flash_rt_kernels.GemmRunner`.
  - `'cached'` uses `flash_rt.flash_rt_gemm_plan.CachedGemmRunner`, built by
    `scripts/build_gemm_plan.py`; it has no fallback.
  - Every other operator (attention, norms, VAE convolutions) is upstream
    torch.
- Each call to `FlashRTLinear.forward` allocates its output (`torch.empty`) and
  calls `record_stream`.
- CUDA Graph: `flash_rt/models/fastwam/velocity_graph.py : FastWAMVelocityGraph`
  (only with `use_cuda_graph=True`, plus torch or cached native linear).
  - It captures a single geometry; a different geometry raises
    `VelocityGeometryMismatch` and is not recaptured.
  - Per request, `begin_request` is called, the condition fields are copied
    once, and each step copies only `latents_action` and `timestep_action`
    before `replay`.
  - Sampler and integration are not captured.
   - The retired Euler2 example pilot is historical source, not an active
     execution path. Consult `docs/vla_autotune_cleanup.zh-CN.md` for its Git
     watermark; inspect the library-owned driver above for current behavior.
- Repeated work: every step passes the same `context`. The upstream model
  recomputes `text_embedding(context)` and every layer's cross-attention K/V
  (tool `cache.fastwam_text_crossattn_kv`). This lives in upstream source, so
  read it at the pinned commit in `source_dir`.
- Host syncs per request:
  - `torch.isfinite(...).all().item()` and `image.min()/max().item()` in
    `_observation_tensors` / `_floating_tensor` / `_initial_state`;
  - the final `isfinite` check in `infer`;
  - ProbeFlow cosine `.item()`.
- Profiling keys: `condition_vae_video_prefill` (covering vision.encode and
  prefix.prefill together) and `velocity`.

## 5. Shared pieces

- Sampling:
  - `flash_rt/sampling/torch_probeflow.py : sample_probeflow_torch_` runs the
    routing (cosine `.item()`) and integration for all four models;
  - `flash_rt/sampling/probeflow.py : ProbeFlowConfig` gives the time tables.
- fvk module: `PYBIND11_MODULE(flash_rt_kernels, m)` in `csrc/bindings.cpp`.
  - A `m.def("name", ...)` lambda usually validates its arguments and then
    calls a launcher declared in a `csrc/**.cuh`.
  - Check for `#ifdef FLASHRT_HAVE_*` guards and the matching
    `target_sources(...)` condition in `CMakeLists.txt` before you claim a
    binding exists on the target arch.
- Stage seams: `python -m flash_rt.vla.autotune.stages list --model M` and
  `check-seams`.
- Dataflow descriptor (pi0.5): `flash_rt/vla/autotune/dataflow.py`, which
  `record['dataflow']` stores in worker results (tool `profile.pi05_dataflow`).
