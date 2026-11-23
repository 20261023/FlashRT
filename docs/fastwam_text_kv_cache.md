# FastWAM request text cross-attention K/V

`FastWAMTorchFrontend(..., cache_text_crossattn_kv=True)` opts into request-scoped
ActionDiT text K/V reuse. The default is `False`. Set
`BackendConfig.options` to include `{"cache_text_crossattn_kv": true}`. The autotune tool is
`cache.fastwam_text_crossattn_kv`, primary stage `denoise.velocity`.

The source contract is FastWAM commit
`7faa71108368fbb3b6885649f112af607427a2d4`. The FlashRT implementation is in
`flash_rt/models/fastwam/text_kv_cache.py`. It verifies token fingerprints of
the upstream methods it replaces; whitespace and comments may differ. Unknown,
wrapped, precompiled or overridden seams raise, as do custom cross-attention
projections, text-embedding variants, and hooks on cached/bypassed modules.
The upstream source checkout is read-only, and its MIT notice is preserved in
the new module.

## Execution and lifetime

The pipeline first validates the observation and appends the checkpoint's
proprio token to context. It owns a snapshot of that context and its boolean
mask. After first-frame VAE/video prefill, it computes ActionDiT
`text_embedding(context)` once, then each layer's
`cross_attn.norm_k(cross_attn.k(embedded))` and `cross_attn.v(embedded)` once.
Keys and values retain the upstream `[B,L,num_heads*head_dim]` layout, device
and dtype. Masks still apply at attention; masked tokens are not removed.

Every `infer` creates fresh text K/V, even if an observation object, tensor
address or prompt is reused. Prompt, mask and proprio changes therefore reach
the next request. Image changes refresh the video prefill as before. The
adapter and model hold no request tensors. A failed preparation cannot publish
a partial text cache. Returned `prepare_condition` values are immutable
snapshots by contract; callers must not mutate their tensors. A condition from
another pipeline, or one without the required text K/V, is rejected.

| Stage | Computation | Frequency with cache | Storage and host traffic |
|---|---|---|---|
| `host.input` | Validate/cast context, mask and proprio; append proprio | Once per request | Existing transfers/scalar validation; context/mask cloned on device |
| `vision.encode`, `prefix.prefill` | VAE, video prepare, video self-attention cache | Once per request | Existing video cache and compile-output clones |
| `denoise.velocity` preparation | Action text embedding | Once per request | Eager device tensor, discarded after projections |
| `denoise.velocity` preparation | Per-layer normalized text K and V | Once per request | Request-owned tensor lists; no new host reads |
| `denoise.velocity` | Action prepare, self-attention, query projection, text SDPA, output projection and FFN | Every NFE | Explicit video/text K/V arguments; no new per-step device/host transfer or scalar read |
| `denoise.routing`, integration | Euler/ProbeFlow choices and sigma updates | Existing schedule | Unchanged sampler boundary |

The static metadata fields `cache_text_crossattn_kv`,
`text_crossattn_kv_cache_scope`, `text_crossattn_kv_cache_layers` and
`text_crossattn_kv_precompute` identify the path. The per-inference
`text_crossattn_kv_cache_request` record separately reports completed preparation
calls, key/value projection and normalization counts, context token count,
logical denoise evaluations (actual sampler NFE) and eager/graph/compiled mode.
Request counters do not mutate compiled bodies or the static backend identity.
The Torch cache path reports `backend="flashrt_torch_text_crossattn_kv"`;
native Linear keeps `backend="flashrt_bf16_linear_partial"` with the cache flag.
Native Linear counts still measure actual Python forwards/enqueues; graph metadata counts the smaller captured
body. The operation ordering and projections are exact algebraic reuse.
Numerical and performance acceptance require measurement.

## CUDA Graph and compile boundaries

Manual `use_cuda_graph=True` adds required `text_cache_k` and `text_cache_v`
tensor lists to the raw-field graph input tree. The graph owns fixed-address
copies. `begin_request` invalidates condition readiness; the first NFE copies
all condition inputs, including text K/V, and each later NFE copies only action
state and timestep. A new shape, stride, dtype, device or list geometry raises
`VelocityGeometryMismatch`. It does not recapture or fall back.

`compile_action_infer=True` still compiles video prefill and raw denoise with
`mode="reduce-overhead", fullgraph=True`. With caching enabled, raw denoise
compiles the FlashRT text-cache tensor core. Text projection stays eager,
after the existing cloned video-output ownership boundary. Its K/V are
ordinary eager outputs passed explicitly into compiled denoise and remain live
across `cudagraph_mark_step_begin()` calls. No mutable module cache is consulted
by a compiled body. The existing rule excluding manual graphs/native Linear
from compile mode remains in force; compile errors propagate.

Eager execution supports the fixed floating model dtypes already admitted by
the frontend. Manual graphs require CUDA; native Linear requires CUDA BF16 and
the cached native backend when graphed. Cache mode requires eval/inference
execution and rejects ambient autocast so a prepared condition cannot silently
cross precision regimes. Model structure, weights and hooks must remain fixed
for a prepared condition and for the lifetime of captured/compiled execution.

## Measured scope

A800 job `357482` exercised the ordinary public backend on eight fixed BF16
Euler20 inputs in eager and compiled modes. Against the faster uncached
bookend, the cache reduced median latency by 10.28% in eager mode and 6.97%
in compiled mode. Eager actions matched elementwise. Compiled actions had
worst relative L2 0.003945 and minimum cosine 0.999992.
See the [campaign report](reports/2026-11-23-fastwam-text-kv-campaign.zh-CN.md).

The model-level gate must cover fixed-noise cached/uncached output comparisons,
per-layer text K/V against uncached upstream projections, Euler and both
ProbeFlow branches, manual capture/repeated replay, same-shape changed
context/mask/proprio/image/noise propagation, fixed static addresses, geometry
rejection, native cached Linear if used, and compiled execution if used.
Retain a prepared condition while preparing another to check ownership too.
Use the registry option `cache_text_crossattn_kv` for public backend measurements.
