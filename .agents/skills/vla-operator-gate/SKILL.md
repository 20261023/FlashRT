---
name: vla-operator-gate
description: Operator-level numerical gate for FlashRT kernel candidates (ForgeKernel, KernelAgent, hand-written or ported fvk/Triton/CUDA kernels) before any model-level autotune campaign. Builds deterministic fixtures from a StageCapture, uses an independent fp64/fp32 torch oracle from the exact typed inputs plus the eager same-dtype production baseline, runs every candidate with no silent skip, and rejects NaN/Inf. It checks max scaled element error, relative L2 and cosine on two input draws, plus CUDA Graph capture and replay parity, then records the verdict in the decision log. Use when a KernelArtifact is harvested, when a patch or port replaces a call site, or when reviewing whether a kernel test uses an independent oracle. Chinese triggers - 算子验收, 算子级 gate, kernel 正确性, 数值对齐, 参考实现.
license: Apache-2.0. Complete terms in LICENSE.txt
metadata:
  upstream-repo: "https://github.com/infinigence/ApxInf"
  upstream-path: "skills/l3-reference-testing"
  upstream-commit: "b05b40990387e8c48e7d6267a790664981a665e4"
---
<!-- Adapted by FlashRT (2026-11-15) from ApxInf skills/l3-reference-testing/SKILL.md @ b05b40990387e8c48e7d6267a790664981a665e4 (Apache-2.0). Modifications: renamed; ApxInf L3/Rust/checked-in-fixture specifics replaced by FlashRT StageCapture fixtures, KernelArtifact candidates, fvk/torch call sites and devlocal artifacts; added the eager same-dtype baseline and conditioning idea (credited to torchtitan kernel-numerics-verifier, BSD-3-Clause, re-expressed, no text copied), second input draw, CUDA Graph parity, KernelAgent adapter tolerances, model-level gate distinction and decision-log recording. See NOTICE. -->

# VLA operator gate

This gate answers one question: **does this kernel candidate compute the same
operator as the call site it would replace?** It runs before a candidate enters a
model-level snapshot campaign.

Where it sits in the loop:

- Kernel lane: `vla-kernel-candidate` captures a call site and runs ForgeKernel
  or KernelAgent. This gate comes next, then `vla-patch-candidate` wires the
  kernel in.
- Port lane: `vla-port-workflow` runs this gate for every new device path it
  introduces.

Organize gates by **operator semantics at one call site**, not by tool or dtype.
Examples: `hyvla.rope_qknorm_kvwrite`, `fastwam.linear_bf16_nt_bias`,
`pi05.decoder_attention`. Keep one clearly named gate per semantic. Every
candidate for that semantic (ForgeKernel, KernelAgent, hand-written, each dtype
or variant) goes through the same gate, with the same thresholds.

## Scope and what the gate does not prove

- Passing proves operator numerics on the gated shapes, dtypes and draws. It
  does **not** prove that the model calls the kernel. Wiring is shown by the
  `vla-model-impl-map` row for the snapshot, plus the frontend's execution
  metadata or native call counters (FastWAM `execution_metadata()`, for
  example).
- Operator thresholds are **not** model tolerances. Model acceptance is the
  controller campaign. It uses reference, candidates and a reference bookend,
  with the caller's `--quality-gate` on normalized actions; the Hy objective,
  for example, uses `max_relative_l2` 0.05 and `min_cosine` 0.999. It also
  needs eager-versus-graph and public-path evidence. A kernel can pass this gate
  and still fail there through accumulated drift, and the opposite can happen
  too.
- A generator's own test is not this gate. KernelAgent's `test.py` is the spec
  the LLM optimized against. ForgeKernel's correctness monitor belongs to
  ForgeKernel. Re-run the check yourself with your own oracle, a fresh input
  draw, and FlashRT code under your control. Never reuse the generator's
  reference as the oracle.

## Fixtures from a StageCapture

1. Capture the call site eagerly. Never capture inside CUDA Graph capture:
   `capture_calls` copies tensors to the host. Use
   `flash_rt.vla.autotune.tools.capture.capture_calls(owner, attr, model=...,
   stage_id=..., call_site=..., op_kind='linear'|'attention'|'custom', out_dir=...,
   in_cuda_graph=<how production runs it>)` around one real request. It writes
   `stage_capture.json` (a `StageCapture`), `stage_calls.json` (strides and
   contiguity) and `stage_tensors.pt` or `.npz` (the first call's tensors).
   - `capture_calls` records tensor-like arguments only. fvk bindings take raw
     `data_ptr()` ints, so capture at the nearest Python seam that still holds
     tensors. Examples are `HyVLAThorBF16Pipeline._block`/`_attn` and
     `FlashRTLinear.forward`. For pointer-only pipelines (pi0/pi0.5 RTX),
     reconstruct tensors from the `CudaBuffer`s in a small shim, and say so in
     the report.
2. Draw 1 is the captured real tensors, byte-exact. Draw 2 is a seeded fresh
   draw with the same shapes, dtypes and value statistics. It catches kernels
   that memorized or special-cased draw 1. Add the smallest legal shape and the
   sizes just below and above the kernel's tile and alignment boundaries when
   the semantic admits them.
3. Store the fixtures under `devlocal/<feat>/operator-gate/<semantic>/`, which
   is git-ignored. `*.npz` and `*.bin` are ignored anywhere in this repo. Never
   check tensor fixtures into git. Record the sha256 of every fixture file, and
   the seed of every synthetic draw, in the gate report.

## Oracle contract

Compare three implementations on identical inputs:

| name | what | dtype |
|---|---|---|
| `oracle` | independent eager torch expression of the operator's public meaning | fp64 (fp32 only where fp64 is unsupported) |
| `native` | the production path being replaced at this call site, run eagerly: the current torch op or fvk binding | production dtype |
| `candidate` | the kernel under test | production dtype |

- Build the oracle inputs by losslessly upcasting the **exact stored typed
  inputs**. For BF16, use the stored BF16 values. For FP8/INT8, decode the
  stored values and apply the same scales the call receives. Never compare
  against pre-quantization fp32 values: quantizer error belongs to a separate
  quantization-operator gate.
- Write the oracle from the operator's meaning. Do not mirror the candidate's
  tiling, packing, intermediate buffers or launch split. For example, the
  oracle for fused RoPE + QK-norm + KV-write is RoPE and RMSNorm on the logical
  `[S, heads, hd]` tensors, followed by a write into the KV window at offset
  `off`. It is not a copy of the kernel's thread layout.
- Include every operand that changes the numbers: bias, scales, eps, masks,
  RoPE tables, KV offset and capacity, GQA repetition, output dtype and in-place
  targets. Check every output of a multi-output or in-place kernel. Also check
  that bytes outside the window the kernel is supposed to write are unchanged:
  fill the buffer with a sentinel first.
- The `native` error against the oracle is the floor that the production path
  already accepts today. Use it to tell kernel error from conditioning (see the
  thresholds). Use `kernel-numerics-verifier` for its full conditioning
  procedure when the native error is itself large.

## Candidate coverage and acceptance

- Run **every** candidate that matches the device, shape, alignment and
  execution mode: each `KernelArtifact`, plus every config the patch can select.
  Before running, list the expected set and assert that each one was visited.
  A candidate skipped without a reason is a gate failure, not a pass.
- Before applying any tolerance, compare shape and dtype, plus NaN and ±Inf
  locations. Upcast candidate output to the oracle dtype only for comparison.
  Any NaN or Inf in a candidate output where the oracle is finite is a failure.
- Apply the same immutable thresholds to every candidate and draw (next
  section). Declare them in the report before the first run. A new candidate
  must never loosen them, and never retune them to make a failing kernel pass.
  Diagnose first: check indexing and masks, accumulator dtype, early casts,
  reduction order and stabilization.
- **CUDA Graph parity**, whenever production runs the call site inside a graph
  (`in_cuda_graph` true, or the map row says it is graph-captured):
  1. warm up on a side stream;
  2. capture one call on static buffers;
  3. copy draw 2 into those buffers and replay;
  4. replay again.

  Every replay must pass the same gate, must match the eager result for the
  same input, and must show that the changed input propagated. Any host sync,
  allocation or `.item()` that breaks capture is a failure.
- If the kernel is nondeterministic (atomics, split-K), repeat identical inputs
  and require every result to pass.

## Thresholds

| gate | max scaled element error | relative L2 | cosine | role |
|---|---|---|---|---|
| ApxInf L3 operator default (upstream) | `<= 0.08` | `<= 0.05` | `>= 0.9999` | default FlashRT operator gate for every candidate |
| KernelAgent adapter (`tools/external/kernelagent.py`, `prepare_problem(rel_l2=2e-3, cosine=0.99999)`) | n/a | `<= max(2e-3, 2 x native rel-L2)` | `>= min(0.99999, 1 - 2 x (1 - native cosine))` | stricter generation-time spec; re-applied here when the candidate came from KernelAgent |
| model-level controller gate (not an operator gate) | n/a | `max_relative_l2` (e.g. 0.05) | `min_cosine` (e.g. 0.999) | normalized action replay in `controller init --quality-gate`; caller-selected |

Metric definitions (elementwise in the oracle dtype) are in
[references/gate-checklist.md](references/gate-checklist.md). Read it before
writing a gate. A candidate passes only when it meets **both** the operator
default and the stricter row that applies to its source. When the native
baseline alone violates the operator default (an ill-conditioned reduction,
say), the gate is invalid for that draw. Report `INCONCLUSIVE` with the
conditioning evidence instead of passing or failing it.

Verdicts: `PASS`, `FAIL`, or `INCONCLUSIVE`. Never give a partial pass by
dropping a draw, a shape or a candidate.

## Where gate code lives

- **While iterating**, put the gate script next to its fixtures in
  `devlocal/<feat>/operator-gate/<semantic>/`. It may load captured tensors
  from there.
- **When the candidate is promoted**, the patch carries a pytest file in the
  candidate worktree, so the test travels with the change. Use the flat FlashRT
  layout: `tests/test_<model>_<semantic>_operator_gate.py`. There is no
  `tests/autotune/` package, and do not use the `test_vla_campaign_*` prefix,
  which the harness suites glob.
  - Follow the existing kernel tests (`tests/test_bf16_nt_bias.py`):
    `pytest.importorskip("torch")`, then skip without CUDA or without the fvk
    symbol.
  - Regenerate the draws from fixed seeds inside the test. A captured-tensor
    path may come from an environment variable, and the test skips when it is
    absent.
  - The patch must not touch `flash_rt/vla/autotune/`
    (`snapshots.FORBIDDEN_PATCH_PREFIXES`).
- Run the focused test on the target GPU first (a Slurm allocation on the
  cluster). Then run the CPU suites the patch could affect.

## Record the result

- Write `devlocal/<feat>/operator-gate/<semantic>/report.json` and a markdown
  table. Include one row per candidate x draw x mode (eager, graph), with
  columns oracle dtype, native error, candidate error, thresholds, gate validity
  and verdict. Add fixture hashes, the source revision, GPU, `nvidia-smi`
  driver, torch/CUDA versions and the expected-versus-visited candidate list.
- Put the metrics into the candidate's provenance when you build the
  `PatchProposal` (`provenance={'operator_gate': ...}`). `KernelArtifact.op_metrics`
  is the generator's own claim; keep it separate from the gate result.
- Log the decision, with `--actor` naming who actually decided:
  `python -m flash_rt.vla.autotune.decisions add --log L --kind decision
  --actor agent|human|human+agent --action keep|reject --summary "operator gate
  <semantic>: PASS|FAIL ..." --rationale ... --stage <stage> --model <model>
  --tool <tool id> --candidate-id <id> --evidence-ref <report path>`.
  An operator-level PASS is `keep`: the kernel may proceed to the patch lane. `accept`
  (a model-level winner) and `merge` must be logged with actor `human` or
  `human+agent`; the decision log rejects them from `agent`.
  For `INCONCLUSIVE`, log `--kind diagnosis --provenance measured_here` with the
  next observation instead.

## Do not

- Do not check tensor fixtures into git, or weaken a threshold for one candidate.
- Do not use the candidate, the generator's reference, or a shared intermediate
  as the oracle.
- Do not compare against pre-quantization values when the call receives
  quantized inputs.
- Do not report a model-level quality claim from an operator gate, or the
  reverse.
- Do not benchmark before the gate passes. Speed claims belong to the snapshot
  campaign (`vla-patch-candidate`) and its bookended reference.
