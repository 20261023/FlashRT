---
name: vla-kernel-candidate
description: Kernel lane of the FlashRT VLA autotune loop - replace one hot operator with a generated kernel. Capture the real call site with flash_rt.vla.autotune.tools.capture inside an allocated GPU step, prepare a ForgeKernel task (CUDA or CuTe, hours, Claude agents, sm80 enablement patches) or a KernelAgent problem (Triton, first valid kernel), run it as its own allocation, harvest a KernelArtifact, pass vla-operator-gate, then wire the kernel behind a new backend option and measure it through vla-patch-candidate. Use when stage attribution plus kernel evidence points at one GEMM, attention, norm/elementwise or conv operator. Chinese triggers - 换 kernel, 写算子, ForgeKernel, KernelAgent, 热点算子, 融合 kernel.
license: Apache-2.0
metadata:
  short-description: "Generated kernels for one hot VLA operator"
---

# VLA kernel candidate

Input: one hot operator with evidence (from `vla-stage-profile` Step 6 and
`vla-model-impl-map`). Output: a `KernelArtifact`, an operator-gate verdict and, if both
pass, a patch candidate measured end to end. A kernel never wins on its own numbers.

```bash
cd <flashrt checkout>
py() { env UV_CACHE_DIR=/tmp/autotune-uv PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy python -B "$@"; }
LOG=<campaign root>/decisions.jsonl
WORK=/abs/shared/kernel-work/<slug>        # visible from GPU nodes; one directory per operator
```

## Rules carried from the design

- Amdahl first: operator share x realistic kernel speedup must still move the end-to-end
  target. Profile shares come from measured intervals and a few ncu/nsys kernels.
- First-valid is not fastest: KernelAgent returns the first kernel that passes its test;
  ForgeKernel's acceptance is a timing rule on its own boundary. FlashRT's controller,
  inside FlashRT's CUDA graphs, decides speed.
- Expected payoff: cuBLAS is hard to beat on plain GEMMs (for example the Hy merger PROJ1
  projection). Prefer fusions (norm + rope + KV write, bias + activation chains,
  dtype conversions) or removing repeated work in the style of
  `cache.fastwam_text_crossattn_kv` over rewriting a single GEMM.
- Generated code is untrusted: review it before it enters a worktree; never run it on
  the login node.

## Mandatory snapshot checklist (B)

- [ ] Check the actual registered `Candidate.tool` lane. Kernel tools are non-config;
  non-config/unknown tools, a source ref differing from the baseline, or any non-null
  patch ref require `--snapshot-store`, even when the options look config-only.
- [ ] Gate the generated artifact, then wire each source hypothesis in its own
  worktree against a full 40-hex base commit. Store the actual diff and kernel hash.
- [ ] Materialize the baseline and every wired candidate, adopt/build native
  extensions and verify readiness plus one shared harness tree. A null source ref
  is an implicit checkout, never inheritance of another point's pinned ref.
- [ ] Confirm plan schema 3, every attempt's `snapshot_id` and ledger manifest.
  Batch/worker source and import checks must succeed before model results count.
- [ ] Run operator/capture gates and relevant CPU controller/snapshot tests. A
  rejected initialization must leave no plan or partial ledger registrations; log
  the real actor and the artifact, gate and campaign evidence.

Only same-source unpatched registered config tools and legacy config records
without tool provenance retain plan v2. `analysis.lever_pricing` remains analysis
and also needs snapshots; no general analysis-lane exception exists.

Forbidden patterns: swapping kernel sources in the live checkout between candidates;
reusing a mutable source directory; copying changed `.py`/`.so` into a verified
snapshot; deleting or falsifying `tool`/`source_ref`/`patch_ref` to obtain plan v2;
trusting an unknown `config.*` tool; omitting the baseline snapshot; downgrading
schemas, fabricating `source_identity` or bypassing import checks with `PYTHONPATH`.
Harness changes require separate infrastructure authorization and never enter a
kernel optimization patch.

## Step 1. Choose the operator and log it

```bash
py -m flash_rt.vla.autotune.tools list --model <model> --hardware cuda_sm80 --stage kernel.gemm --all
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent \
  --action select_tool --tool kernel.kernelagent --stage kernel.norm_elementwise --model <model> \
  --summary "<op> at <call site>: fused kernel candidate" \
  --rationale "op share <p> of <stage> measured_here (ncu/nsys ref); ceiling <c>x" \
  --evidence-ref <report>
```

Human checkpoint: kernel generation spends hours of exclusive GPU time and LLM tokens.
The human approves the operator, the tool and the budget before Step 3.

## Step 2. Capture the real call site (inside an allocated GPU step)

The capture wraps a Python attribute, so run the model with `use_cuda_graph=False`
(graph replay bypasses Python) and record `in_cuda_graph=True` when production runs the
op inside a graph. Choose the narrowest owner/attribute whose tensor arguments are
exactly the operator's inputs (find it with `vla-model-impl-map`); wrapping a module-wide
function such as `F.linear` records every call of that function in order.

```python
# capture_op.py - run with the cluster runtime inside srun, never on the login node
import json
from flash_rt.vla import load_backend
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.workloads import NPZWorkload
from flash_rt.vla.autotune.tools.capture import capture_calls

cand = Candidate.from_dict(json.load(open('/abs/baseline-eager.json')))   # use_cuda_graph False
backend = load_backend(cand.backend)
request = NPZWorkload.from_file('/abs/workload.json').request(0)
owner = backend.frontend            # replace with the object that owns the hot method
with capture_calls(owner, '<method>', model=cand.backend.model, stage_id='kernel.norm_elementwise',
                   call_site='<file.py:Class.method>', op_kind='custom',
                   reference='/abs/ref_<op>.py', out_dir='/abs/shared/kernel-work/<slug>/capture',
                   max_calls=4, in_cuda_graph=True) as recorder:
    backend.infer(request.observation, prompt=request.prompt, noise=request.noise)
print(recorder.total_calls, recorder.capture_path, recorder.tensors_path)
backend.close()
```

- `op_kind`: `linear` (x[M,K], weight[N,K], optional bias -> [M,N]), `attention`
  (SDPA layout [B,H,S,D]; the description is marked REVIEW REQUIRED), or `custom` with a
  `reference` Python file defining `run(<inputs in signature order>)`.
- Output: `stage_capture.json`, `stage_calls.json` (strides/contiguity) and
  `stage_tensors.pt`. Check the recorded shapes against the stage map before going on.

## Step 3a. ForgeKernel (CUDA / CuTe, hours, Claude Code agents)

Use for CUDA/CuTe kernels and fusions worth a long search. Kernel-writing guidance and
the launcher: `forge-start`.

```bash
py -c "from flash_rt.vla.autotune.tools.external import forgekernel as f; print('\n'.join(f.sm80_enablement()))"
```

A800 needs all five ForgeKernel patches above (hw_kb sm80 tree, CUTE_ARCH sm80, CUDA
gencode compute_80/sm_80, SASS families HMMA/LDGSTS/LDSM, PM interval >= 20000 cycles)
in your ForgeKernel checkout. Then:

```bash
py -c "
from flash_rt.vla.autotune.tools.artifacts import StageCapture
from flash_rt.vla.autotune.tools.external import forgekernel as f
cap = StageCapture.load('$WORK/capture/stage_capture.json')
task = f.prepare_task(cap, '$WORK/fk-task-<slug>', language='cuda', arch='sm80', target_ratio=1.10, hours=6)
open('$WORK/fk-job.sh', 'w').write(f.render_job(task, forgekernel_root='/abs/ForgeKernel', runs_dir='$WORK/fk-runs'))"
```

Review `description.md` (lines marked REVIEW REQUIRED) and `reference/run.py`, submit
`fk-job.sh` as its own allocation, then harvest. Acceptance (applied by `harvest`):
`best_confirmation.json` exists, median ratio above `1 + max(0.015, spread of job
ratios)`, not disputed, correctness monitor not in alarm, exactly one kernel file.

```bash
py -c "
from flash_rt.vla.autotune.tools.external import forgekernel as f
a = f.harvest('$WORK/fk-runs/<run>', stage_id='kernel.norm_elementwise', model='<model>')
a.save('$WORK/kernel_artifact.json'); print(a.accepted, a.reasons)"
```

## Step 3b. KernelAgent (Triton, first valid kernel)

Use for Triton kernels where a correct kernel quickly is the goal. Triton writing help:
`kernel-triton-writing`.

```bash
py -c "
from flash_rt.vla.autotune.tools.artifacts import StageCapture
from flash_rt.vla.autotune.tools.external import kernelagent as k
cap = StageCapture.load('$WORK/capture/stage_capture.json')
d = k.prepare_problem(cap, '$WORK/ka-<slug>')
open('$WORK/ka-job.sh', 'w').write(k.render_job(d, kernelagent_root='/abs/KernelAgent', workers=3, rounds=8))"
```

- `test.py` is the whole contract the LLM sees (signature, dtype, sm_80, rel-L2 and
  cosine against an fp32 reference, a second input draw, CUDA-graph capture + replay).
  Review it before submitting.
- Keep `workers <= 3` with Anthropic models (seed temperatures above 1.0 silently swap
  in mock kernels). The run needs LLM egress from the GPU node and scoped API keys.
- The optional optimization pipeline needs ncu counter permission and
  `gpu_name="NVIDIA A100 SXM4 80GB"` on A800 (missing from its GPU database).
- Submit `ka-job.sh` as its own allocation, then:

```bash
py -c "
from flash_rt.vla.autotune.tools.external import kernelagent as k
a = k.harvest('$WORK/ka-<slug>', stage_id='kernel.norm_elementwise', model='<model>')
a.save('$WORK/kernel_artifact.json'); print(a.accepted, a.reasons)"
```

`harvest` rejects a changed `test.py`, a failed re-test or a kernel hash mismatch.

Log the outcome either way (`accepted` false is an `experience`, not a silent retry):

```bash
py -m flash_rt.vla.autotune.decisions add --log $LOG --kind experience --actor agent \
  --summary "<tool> on <op>: accepted=<bool>" --model <model> --stage <stage> \
  --context-json signature.json --outcome "<reasons / op_metrics>" --evidence-ref $WORK/kernel_artifact.json
```

## Step 4. Operator gate

Run `vla-operator-gate` on the artifact against the captured tensors: numerics against
an fp32 reference on several input draws and the captured shapes, CUDA-graph replay,
and the dtype/layout contract. For numerics methodology use `kernel-numerics-verifier`.
A kernel that fails the gate stops here (decision `reject`, actor set).

Human checkpoint: reading the generated kernel. KernelAgent's anti-cheat misses
`@`/`torch.addmm`/method forms; a human confirms the kernel computes the op rather than
calling the library it is meant to replace.

## Step 5. Wire it behind a new backend option -> vla-patch-candidate

1. Worktree from the baseline's 40-hex commit (`vla-patch-candidate` Steps 1-2).
2. Add the kernel source and a new frontend option that selects it (default off), and
   add that option to `flash_rt/vla/registry.py` `_CAPABILITIES` for the model in the
   SAME patch. Keep the old path as the default; never edit `flash_rt/vla/autotune/`.
3. Triton kernels: compile/warm up before graph capture so capture sees no JIT work.
   CUDA kernels under `csrc` need the native build path (`snapshots build-script`).
4. Store the proposal with `options={'<new option>': <value>}` and
   `provenance={'kernel_artifact': '$WORK/kernel_artifact.json', 'kernel_sha256': ...}`,
   build the candidate with `candidate_from_proposal(..., validate=False)` (this checkout
   does not know the new option), and run the snapshot campaign.
5. The end-to-end campaign decides; KEEP when the gain is inside bookend drift.

## Routing

| need | skill |
|---|---|
| where the op runs, graph caches, fvk binding | `vla-model-impl-map` |
| kernel timeline / counters | `perf-nsight-systems`, `perf-nsight-compute-analysis` |
| ForgeKernel launch and task authoring | `forge-start` |
| writing or fixing a Triton kernel | `kernel-triton-writing` |
| operator numerics and acceptance | `vla-operator-gate`, `kernel-numerics-verifier` |
| measuring it end to end | `vla-patch-candidate` |
