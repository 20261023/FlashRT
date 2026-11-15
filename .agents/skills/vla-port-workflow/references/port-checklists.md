# Port lane checklists and current port tools

**Read this when** you write the preflight tuple or the execution ledger, run
capture acceptance, or start one of the registry tools listed in section 4.

## 1. Preflight tuple (record in the report and the decision log)

| field | value | source of truth |
|---|---|---|
| tool id / lane / primary stage | e.g. `quant.int8_pi05` / port / `precision.weights` | `python -m flash_rt.vla.autotune.tools show <id>` |
| FlashRT base commit | 40-hex | `git rev-parse <ref>^{commit}`, or `snapshots resolve` |
| upstream reference revision | when porting from external code | `SOURCE.json`, `VENDOR.md`, observer docstring |
| checkpoint | path + identity (manifest/sha) | the workload / candidate JSON |
| target | GPU model, capability, `GPU_ARCH`, hardware label | `nvidia-smi`, `registry._CUDA_TARGETS` |
| precision | weights / activations / accumulate / output | the ledger |
| inputs | workload file, fixed noise, prompt(s), views | `flash_rt/vla/autotune/workloads.py` format |
| tolerances | operator gate (`vla-operator-gate`), model `--quality-gate` | both explicit, chosen before measuring |
| performance goal | e.g. stage share or end-to-end target vs bookended reference; release gate yes/no | the user |
| public API | new `BackendConfig.options` keys, defaults, validation errors | `registry.validate_config` |
| mode | hands-off (default for implementation) / hands-on | the user |

## 2. Execution ledger template

One row per reference computation, in execution order, tagged with a stage id:

| Stage | Reference computation | Tensor contract | Frequency | Preferred call | Buffers and lifetime | Host traffic | Evidence |
|---|---|---|---|---|---|---|---|
| `prefix.prefill` | example: QKV split, RoPE, QK-RMSNorm, KV write (Hy) | `qkv[S, q+2kv]` bf16, `cos/sin[S,hd]`, `kbuf/vbuf[32,1,H,S_tot,hd]` at `off` | per layer per request | `fvk.hyvla_rope_qknorm_kvwrite_bf16` (needs SM80 build) or the unfused torch path | `kbuf/vbuf` graph-owned static; `q` per call | none | `pipeline_thor.py:_block`; gate report |

- **Reference computation:** the operator's meaning, not the framework op
  name.
- **Tensor contract:** shapes with axis names, dtypes, layout and strides,
  in-place targets.
- **Frequency:** per load, per prompt change, per request, per layer, per
  NFE.
- **Preferred call:** the existing fvk or torch entry you will use, or `NEW:
  <name>` with its exit criterion, or `BLOCKED: <reason>`.
- **Buffers and lifetime:** who allocates the buffer, whether it is
  graph-static, when it is refreshed, and whether its address is stable.
- **Host traffic:** H2D, D2H, `.item()` and syncs. These must be zero inside
  captured or per-step rows.
- **Evidence:** a source location, a capture file, a gate report or a profiler
  trace.

## 3. Capture acceptance checklist

Run each check with the production `use_cuda_graph` setting and with `False`,
on the same fixed workload and noise.

1. **Eager versus graph parity.** For identical inputs, graph and eager
   outputs agree within the operator gate thresholds. For an exact port,
   require bitwise equality where the eager and captured paths run the same
   kernels.
2. **Repeated replay.** Two or more replays with unchanged inputs give
   identical outputs.
3. **Changed-input propagation.** Change the image, the prompt (same length,
   so the same key), the state and the noise, one at a time. Each output
   changes, and each still matches eager on the new input. This catches stale
   static buffers, such as a condition copied once while a cache key was
   reused.
4. **Stable addresses.** Static input and output buffers keep their
   `data_ptr()` across requests. The graph cache does not grow for a repeated
   key. A new key captures once, and only once.
5. **Capture-safe workspace.** No allocation, host sync or `.item()` inside
   the captured body. Every workspace is allocated before capture, or comes
   from the graph's private pool.
6. **Host-transfer audit.** List every `.item()`, `.cpu()`, `.tolist()`,
   `torch.cuda.synchronize()`, `cudaStreamSynchronize` and `copy_` from host,
   per request and per step, for the reference and the port. A new per-step
   host transfer is a blocker in a captured stage. Use `perf-torch-sync-free`,
   then confirm in an nsys trace (`perf-nsight-systems`).
7. **Public path.** `registry.load_backend(config)` followed by
   `VLABackend.infer` exercises the port, and the frontend's execution
   metadata or call counters prove it ran. Examples are FastWAM
   `execution_metadata()` and native linear `calls`.

Report each item as pass, fail or not applicable, with evidence. A port is
functionally accepted only when every applicable item passes.

## 4. Current port and port-like tools

Run `python -m flash_rt.vla.autotune.tools show <id>` for the live reason. What
follows was verified against the source on 2026-11-15; it is not a result.

### `quant.int8_pi05`: lane port, stages `precision.weights`, `kernel.gemm`; approximate; needs_implementation

- **INT8 already exists in the pi0.5 RTX frontend and pipeline**, behind env
  flags:
  - `FVK_PI05_RTX_FORCE_INT8`, `FVK_PI05_RTX_INT8_ENCODER_ONLY` and
    `FVK_PI05_RTX_INT8_VISION`, read in `Pi05TorchFrontendRtx.__init__`;
  - `_quantize_decoder_int8`, `_quantize_encoder_int8` and
    `_quantize_vision_int8`;
  - pipeline `_int8_gemm*` calls into `fvk.cutlass_int8_rowwise_bf16out` and
    `fvk.cutlass_int8_silu_gated_bf16out`.
- **Blockers:**
  1. The CUTLASS INT8 sources and bindings exist only with
     `ENABLE_SM80_INT8_CUTLASS` (`#ifdef ENABLE_SM80_INT8_CUTLASS` in
     `csrc/bindings.cpp`). That defaults to ON only for `GPU_ARCH 87`, so an
     SM80 build needs `-DENABLE_SM80_INT8_CUTLASS=ON`.
  2. `registry.validate_config` rejects those env vars for the BF16 VLA
     backend.
- **The port needs:**
  - an explicit option, instead of an ambient env var, admitted by the
    registry;
  - a precision label that `describe_backend` reports truthfully;
  - per-row weight and activation scale semantics in the ledger;
  - an operator gate per GEMM site that applies the decoded INT8 values and
    the same scales;
  - the model-level quality gate, because the tool is approximate.

### `fusion.hy_rope_qknorm_kvwrite`: registry lane patch, port-like work; stages `kernel.norm_elementwise`, `prefix.prefill`, `denoise.velocity`; approximate

- `fvk.hyvla_rope_qknorm_kvwrite_bf16` is used in `HyVLAThorBF16Pipeline._block`
  when `_fused_attn` is set.
- `CMakeLists.txt` builds its sources only with `FLASHRT_ENABLE_HYVLA=ON`, and
  only for `GPU_ARCH 110` or `87/120/121`. The binding is guarded by
  `FLASHRT_HAVE_HYVLA_THOR`/`_ORIN`.
- `HyVLAFlowFrontend` rejects `use_fused`, and the registry's `hyvla` options
  are only `use_cuda_graph` and `numerical_profile`.
- **The port needs:**
  - arch 80 added with the binding guard kept in lockstep (§4.3);
  - an admitted option;
  - an operator gate against the unfused path, covering `q_out`, the KV
    window at `off`, untouched rows, `kv_rep`, and both numerical profiles if
    both are offered;
  - capture acceptance for the per-time velocity graphs.

### `cache.fastwam_text_crossattn_kv`: lane patch, port-like work; stage `denoise.velocity`; exact

- Every step passes the same `context` to
  `model._denoise_action_with_video_cache`, through
  `FastWAMActionPipeline._evaluate_at_timestep`. The pinned upstream model then
  recomputes `text_embedding(context)` and every layer's cross-attention K/V.
- Only the first-frame video K/V is cached, in `_prepare_condition`.
- **Upstream code lives in `source_dir`**, pinned by
  `examples/vla_autotune/fastwam-source/SOURCE.json`, outside snapshot
  identity. Implement the cache on the FlashRT side, as a per-request
  pre-projection owned by the pipeline or a FlashRT module wrapping the
  upstream module, or pin and record the upstream edit separately.
- **With `use_cuda_graph=True`**, `FastWAMVelocityGraph` copies the condition
  fields once per request and only `latents_action`/`timestep_action` per
  step. Cached K/V must become captured static inputs that refresh once per
  request, and its geometry is fixed: a mismatch raises.
- Exact means the math is unchanged. Gate it bitwise where the same kernels
  run, otherwise with the operator gate thresholds, and say which.

## 5. Status lines for the report

- **Functional acceptance:** `pass` / `fail`, listing the failing items.
- **Optimization status:** `target met` / `best effort with performance debt
  (<measured gap>, <paths tried>)` / `blocked (<dependency>)`.
- **Hand-off:**
  - PatchProposal `patch_ref`, `tool_id`, `stage_id`;
  - snapshot id;
  - evidence refs (ledger, gate report, capture acceptance);
  - the decision-log record id.
