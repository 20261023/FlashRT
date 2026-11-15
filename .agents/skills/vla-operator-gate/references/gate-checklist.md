# Operator gate checklist and metric definitions

**Read this when** you write or review an operator gate script or test.

## 1. Metrics

Compute every metric over the flattened output, in the oracle dtype (fp64),
after upcasting the candidate output. Call the oracle `e` and the candidate `a`.

| metric | definition |
|---|---|
| max scaled element error | `max_i |a_i - e_i| / max(1, |e_i|)`: absolute error below magnitude 1, relative error above it |
| relative L2 | `sqrt(sum (a_i - e_i)^2 / max(sum e_i^2, 1e-20))` |
| cosine | `sum a_i e_i / sqrt(sum a_i^2 * sum e_i^2)`. It is 1 when both norms are ~0 and 0 when exactly one is ~0 |
| finite | every `a_i` is finite wherever `e_i` is finite. A non-finite `e` is a fixture error: fix the fixture |

These are the definitions in ApxInf
`crates/apxinf-cuda-new/native/adapters/gemm/reference.cpp : compare_reference`
at commit `b05b40990387e8c48e7d6267a790664981a665e4`, with the thresholds 0.08,
0.05 and 0.9999. They were read from that file and re-expressed here in our own
words.

KernelAgent adapter (`flash_rt/vla/autotune/tools/external/kernelagent.py`):

- the fp32 reference uses the same rel-L2 and cosine definitions;
- `rel_l2_tol = max(REL_L2_TOL, 2 * native_rel_l2)`, with `REL_L2_TOL = 2e-3` by
  default;
- `cosine_min = min(COSINE_MIN, 1 - 2 * (1 - native_cosine))`, with
  `COSINE_MIN = 0.99999` by default;
- `native` is the same-dtype torch path computed from the same inputs.

**Gate validity.** Compute the native baseline's error against the oracle. If the
native error alone breaks the operator default thresholds, the inputs are
ill-conditioned for this metric. Report `INCONCLUSIVE` and run the conditioning
analysis in `kernel-numerics-verifier`, which compares FP32 reduction orders,
sweeps lengths and seeds, and checks for bias. Do not loosen the thresholds.

## 2. Oracle recipes per `op_kind`

Each recipe applies to the StageCapture `op_kind`, and to the axis names that
`capture.build_capture` assigns.

- **linear** (`M, K, N`; `x` flattened to `[M, K]`): `y = x64 @ W64.T + b64`.
  - Take the layout from the call. `FlashRTLinear` / `bf16_nt_bias` uses
    `W[N, K]`; `GemmRunner.bf16_nn` takes `B[K, N]`.
  - Native: `F.linear` in the production dtype, or the fvk call it replaces.
  - Quantized: decode `W8 * w_scale` and `x8 * x_scale` exactly as the call
    receives them.
- **attention** (`B, H, HKV, S, SKV, D, DV`; SDPA layout `[B, H, S, D]`):
  - Repeat KV heads `H / HKV` times, then compute
    `softmax((q @ k.T) * scale + mask_bias) @ v` in fp64. For a bool mask,
    disallowed positions become `-inf`, and a row with every position masked is
    a fixture error.
  - Native: `F.scaled_dot_product_attention` with the backend list the call site
    uses, or `flash_rt_fa2` / the fvk attention it replaces.
  - Confirm the layout before writing the oracle. The capture marks it
    `REVIEW REQUIRED`.
- **custom** (axes `<name>_<i>`): write the oracle from the call site's Python
  semantics. Example: Hy fused RoPE + QK-norm + KV-write.
  - **Inputs:** `qkv[S, q_dim + 2 * kv_dim]`, `cos`/`sin[S, hd]`, the QK-norm
    weights, `kbuf`/`vbuf` (pre-expanded by `kv_rep`), offset `off`, `eps`.
  - **Oracle:** split q/k/v, then `x * cos + rot_half(x) * sin`, then RMSNorm
    over `hd` with the weight.
  - **Writes:** `q_out`, plus K/V written at rows `[off, off + S)` of the cache
    and repeated `kv_rep` times.
  - **Checks:** compare `q_out`, the written window, and that rows outside the
    window are untouched.
  - **Native:** the unfused branch of `HyVLAThorBF16Pipeline._block`.
  - **Profile:** `official_local_fp32` changes the RoPE/QK dtypes. Gate the
    profile the candidate targets.

## 3. Draws and shapes

- **Draw 1:** the captured tensors (`stage_tensors.pt` or `.npz`), byte-exact.
- **Draw 2:** seeded, with the same shape and dtype. Match the mean and std of
  each captured input; KernelAgent's `_fresh_like` does this. Keep masks, RoPE
  tables and offsets semantically valid: regenerate them from their definition,
  never with random noise.
- **Shapes:** the production shape (from `StageCapture.concrete_shape`), the
  smallest legal shape, and sizes just below and above the kernel's tile and
  alignment sizes, when the semantic admits them.
- **Adversarial values, where relevant:** near-zero variance for norms, large
  logits for softmax, cancellation for reductions, and the edge offsets
  `off = 0` and `off + S = capacity` for KV writes.

## 4. CUDA Graph parity procedure

1. Allocate static input and output buffers. Fill the outputs with a sentinel,
   for example NaN for floating outputs.
2. Warm up three times on a side stream, then make the main stream wait for it.
3. Capture one call with `torch.cuda.graph(g)`, or `flash_rt.core.cuda_graph.CUDAGraph`
   for raw-stream fvk calls.
4. Copy draw 1 into the static inputs, replay, and compare to the eager result
   on draw 1. The sentinel must be fully overwritten inside the output window.
5. Copy draw 2 in, replay, and compare to the eager result on draw 2. This
   proves the changed input propagated.
6. Replay again with unchanged inputs; the result must be identical.
7. Any capture error, any illegal host sync (`.item()`, `.cpu()`,
   `torch.cuda.synchronize()` inside the body), or any allocation that breaks
   capture is a FAIL.

## 5. Test skeleton (in the candidate worktree)

```python
# tests/test_<model>_<semantic>_operator_gate.py
import pytest
torch = pytest.importorskip("torch")

def _require():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    fvk = pytest.importorskip("flash_rt.flash_rt_kernels")
    if not hasattr(fvk, "<binding>"):
        pytest.skip("<binding> not built for this GPU_ARCH")
    return fvk

THRESHOLDS = dict(max_scaled=0.08, rel_l2=0.05, cosine=0.9999)   # declared up front, never per-candidate
CANDIDATES = ("<id-a>", "<id-b>")                                 # expected set; assert all visited

def test_<semantic>_all_candidates_match_oracle():
    fvk = _require()
    visited = set()
    for draw in (draw_captured_or_skip(), draw_seeded(seed=1234)):
        oracle = oracle_fp64(draw)
        native_err = metrics(native(draw), oracle)
        for cid in CANDIDATES:
            out = run_candidate(cid, draw)
            assert_finite_like(out, oracle)
            check(metrics(out, oracle), THRESHOLDS, native_err)
            check_graph_parity(cid, draw)
            visited.add(cid)
    assert visited == set(CANDIDATES)
```

## 6. Report row template

| semantic | candidate | draw | mode | oracle | native err (scaled / rel-L2 / cos) | candidate err | thresholds | gate valid | verdict |
|---|---|---|---|---|---|---|---|---|---|
