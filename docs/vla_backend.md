# VLA execution backend: FlashRT + Euler / ProbeFlow

This is the experimental execution layer for model-level VLA optimization.
It separates **model execution** from **sampling policy**. The
[runtime-backed experiment layer](vla_autotune.md) adds persistent campaigns,
serial workers and scoped selection above it. This execution layer does not
call a coding agent or implement BAC/SpeCa/token pruning.

The initial source implementation was developed without execution at the user's
request. On 2026-11-09 the user authorized sbatch verification and experiments.
Per-run evidence lives in the task reports; support is not inferred merely from
the existence of a source path. Existing model evidence does not automatically
qualify newly adapted paths.

## Candidate description versus execution

```python
from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla import BackendConfig, describe_backend, load_backend

candidate = BackendConfig(
    model="pi0.5",                 # canonical ID: pi05
    checkpoint="/local/pi05_pytorch",
    hardware="cuda_sm80",         # A800 / A100; auto detects the current GPU
    sampler="probeflow",
    probeflow=ProbeFlowConfig(
        epsilon=0.008,
        probe_horizon=0.5,
        max_steps=20,
        step_increment=2,
    ),
    options={"num_views": 2, "chunk_size": 50},
)

description = describe_backend(candidate)  # no Torch, weights or GPU access
backend = load_backend(candidate)         # constructs the real frontend
backend.prepare(observation, prompt="pick up the block")  # warmup/capture may run
result = backend.infer(observation, noise=initial_noise)
actions = result["actions"]
metadata = result["sampler_metadata"]      # actual nfe, not just scheduled steps
backend.close()
```

`prepare` is an actual runtime operation, not a dry run. Pi frontends use their
existing calibration/graph preparation entry point; frontends without that API
warm one full request and discard its output. `infer` automatically
prepares a new prompt or calibration bucket when needed, without adding a second
Hy-VLA/FastWAM request merely for warmup. A later performance
runner must explicitly choose whether to measure cold preparation or steady-state
requests; the loader does not invent latency numbers.

The source-only description CLI never loads a model:

```bash
python -m flash_rt.vla --model pi05 --checkpoint /local/weights \
    --hardware cuda_sm80 --sampler probeflow
```

These are usage examples, not commands executed during development.

## Model and input contracts

| Model ID | Neural path / precision in this API | Input and output boundary |
|---|---|---|
| `pi0` | Pi0 native RTX-family pipeline, FP16 | Existing Pi0 image/state frontend and LIBERO-decoded actions; state token remains separate from action rows |
| `pi05` (`pi0.5`) | Pi0.5 native RTX-family pipeline, BF16 | Existing Pi0.5 image frontend and state-in-prompt processing; LIBERO-decoded actions |
| `hyvla` | Hy-Embodied-0.5-VLA native semantic BF16 pipeline | Camera history/prefix/state handling from Hy-VLA; normalized padded model actions, **not decoded robot commands** |
| `fastwam` | Upstream first-frame model + **partial native FlashRT BF16 Linear binding**, or explicit Torch operators; eager default, opt-in raw-field graph | Prepared Wan/T5 inputs; normalized batched `[1,H,A]` actions, not robot commands |

The first API deliberately uses nonquantized paths. `precision="auto"` resolves
to FP16 for Pi0 and BF16 for the other models. An incompatible explicit precision
raises; quantization options cannot be slipped through `options`.

`framework="jax"` for Pi0/Pi0.5 means the existing **Orbax weight loader** driving
the same FlashRT CUDA math, not a separate JAX inference runtime. Hy-VLA/FastWAM
use Torch source bindings.

`observation` remains model-specific. The common adapter does not guess camera
order, robot state normalization, history construction, dataset stats or action
units. In particular, do not feed Hy-VLA's padded normalized actions directly to a
robot, and do not assume Pi frontends' existing LIBERO conversion is appropriate
for a different embodiment. `backend_metadata.action_domain` labels this boundary.

## Pi0 versus Pi0.5

```python
# Fixed Euler, with both time table AND output scale set for five steps.
pi0 = load_backend(BackendConfig(
    model="pi0", checkpoint="/local/pi0_pytorch", num_steps=5,
    options={"num_views": 2, "chunk_size": 50},
))

# Adaptive Pi0 uses FP16 action buffers, not a BF16 reinterpretation.
pi0_adaptive = load_backend(BackendConfig(
    model="pi0", checkpoint="/local/pi0_pytorch", sampler="probeflow",
    options={"num_views": 2, "chunk_size": 50},
))
```

Pi0 has `[state; actions]` decoder tokens and an action/time MLP. The state query's
attention mask is preserved. Only action rows are projected into velocity and
integrated. Pi0.5 instead uses timestep-dependent AdaRMS modulation and may encode
state in prompt tokens. These are different evaluators sharing one sampler, not
one model pretending to be the other.

Pi0's existing Euler path fuses the scaled output projection into the latent via
GEMM `beta=1`. The velocity-only path instead writes a separate output with
`beta=0` and uses weights scaled by `-1`, not `-1/N`. Thus sampler integration does
not accidentally apply the Euler step twice.

## ProbeFlow execution contract

`ProbeFlowConfig` and the CPU reference live under `flash_rt.sampling`; old Pi0.5
import paths remain supported. The Torch integrator takes:

```python
sample_probeflow_torch_(state, evaluate, config)
# evaluate(model_time) -> progress velocity = -v_theta(state, model_time)
```

- `evaluate` must not integrate or mutate `state`.
- Its output must have the same shape, dtype and device as state.
- Model time is `1-progress`; velocity is negated exactly once.
- The first probe is cloned because native evaluator output may alias scratch.
- Two-probe schedule: NFE=2. Dense N-step schedule: NFE=N+1 because the lookahead
  evaluation is discarded after routing.
- For `probe_horizon=0.5`, the two-probe branch uses the uniform Euler-2 grid.
  The routing signal is heuristic, not a certified ODE error bound.
- Prefix KV is computed once **per request** and shared by all evaluations; this
  does not imply cross-frame prefix reuse.
- Reachable timestep unions are precomputed exactly; union table size is not the
  selected integration budget. No truncating/interpolating a ten-step table.
- A mapped clock can supply actual sigma deltas through the shared integrator's
  `time_map` argument. The callback still receives enumerable logical clock
  keys, evaluates at the corresponding physical sigma and does not multiply
  velocity by a model-time scale. This is used for shifted FastWAM grids.
- Routing/integration remain outside captured neural graphs. Do not capture a
  warmup's chosen schedule and treat it as input-adaptive execution.

Threshold quality and latency on newly adapted models have not been measured.
Moving a sampler to another head is not evidence that its default epsilon is a
good operating point there.

## Hardware and dependencies

This API has its own explicit target selection so SM80 does not masquerade as
SM89/SM87. A requested target is checked against the actual CUDA capability.
Source paths depend on matching, locally built FlashRT extensions and each
frontend's existing tokenizer/checkpoint dependencies. No binaries or weights are
downloaded by this implementation task.

The general `flash_rt.load_model` API retains its previous dispatch/precision
behavior. Use `flash_rt.vla.load_backend` for this experimental model/sampler
boundary rather than assuming new sampler flags are silently forwarded there.

## FastWAM

FastWAM is `yuantianyuan01/FastWAM`, a Wan video prefix plus ActionDiT. It is not
the autoregressive Pi0-FAST model and not FlashRT Motus. Its upstream already has
the desired first-frame K/V prefill and nonintegrating action head boundary.

Original action schedules may use shifted sigma:

```text
sigma(u) = shift*u / (1 + (shift-1)*u)
model_timestep = num_train_timesteps * sigma
action_update = (sigma_next - sigma) * raw_velocity
```

Multiplying model time by 1000 does NOT multiply the integration step by 1000.
The original unconditioned release weights use shift=5; optional-IDM weights use
shift=1. Choosing first-frame mode must be explicit for optional-IDM models.
The shifted-grid ProbeFlow adaptation uses sigma differences for both probe and
dense-restart updates. With a half-clock probe, its two-probe branch matches the
shifted two-step Euler grid, rather than silently switching to uniform sigma.
This adaptation and its default threshold are not validated by the original
ProbeFlow paper or by runtime measurements in this development round.

```python
fastwam = load_backend(BackendConfig(
    model="fastwam",
    checkpoint="/local/fastwam/policy.pt",
    sampler="probeflow",
    options={
        "source_dir": "/local/source/FastWAM",  # or an already installed upstream
        "model_config": "/local/fastwam/resolved-model.json",
        "backbone_dir": "/local/wan",         # contains Wan2.2_VAE.pth
        "sigma_shift": 5.0,                    # match the chosen release
        "action_horizon": 32,                  # match its checkpoint processor
    },
))
result = fastwam.infer({
    "input_image": prepared_image,       # [-1,1], [3,H,W] or [1,3,H,W]
    "context": prepared_t5_context,
    "context_mask": prepared_boolean_mask,
    "proprio": normalized_proprio,
}, noise=initial_action_noise)            # [1,action_horizon,action_dim]
coverage = result["execution_metadata"]
```

`model_config` is a **resolved upstream model-section mapping or JSON**, not an
unresolved Hydra training config. The local loader uses the actual upstream
`from_wan22_pretrained` and `load_checkpoint` APIs, with the local VAE and a full
trained `mot` state. It does not load/download T5: pass prepared upstream context
and mask. It requires the upstream FastWAM source/dependencies already available;
no installation is performed by this task.

The unified FlashRT backend explicitly enables `native_linear=True`, replacing
supported active VideoDiT/ActionDiT `nn.Linear` modules with
`GemmRunner.bf16_nt_bias` using original `[N,K]` weights. No second weight
transpose or global module monkeypatch is required. Per-request
`execution_metadata` lists installed modules and their actual native call counts.
VAE, patch Conv3d, attention/SDPA, norms, RoPE and activations/modulation remain
upstream operations. **This is a concrete partial native port, not a fully native
FastWAM backend and not an automatic performance win.** See the measured iteration
notes for the specific A800 outcomes and their numerical/experimental limits.

The direct `FastWAMTorchFrontend` additionally accepts an initialized upstream
`model=` for advanced integration; this is separate from the unified local-file
loader. Its default `native_linear=False` is explicitly labeled
`upstream_torch_reference`, not FlashRT execution. Do not use that reference-only
mode as evidence of native acceleration.

### Explicit Torch and raw-velocity graph selection

`BackendConfig(model="fastwam", backend="torch", ...)` runs Torch operators
through the common first-frame pipeline, without installing native Linear or
loading a native GEMM extension. Its descriptor and result backend label are
`torch`. This is **not** the independent upstream `infer_action` sampler oracle.
Other model families do not gain a Torch backend from this addition. Do not pass
`native_linear_backend` with this engine; unused native selections reject.

Both engines accept `options={..., "use_cuda_graph": True}`. The default remains
`False`. FlashRT+graph additionally requires `"native_linear_backend": "cached"`;
legacy+graph is rejected before model/device loading, not silently upgraded.

Only the non-integrating raw-velocity body is captured. VAE and video KV prefill
run once per request outside capture. Conditions are copied anew at the first
evaluation of **every** request, even for reused input pointers; latent/time
refresh on every evaluation. Euler and adaptive PF routing/integration stay
outside capture. One fixed tensor-tree shape/stride/dtype/device geometry is
supported; a change raises `VelocityGeometryMismatch`, without replay, automatic
recapture or eager fallback. A new backend instance can select a new geometry.

The pipeline owns an ordered side stream and serializes all public GPU model
operations on it. **First preparation/capture must run while the process is CUDA
quiescent:** no unrelated non-captured work on other threads during capture.
Call `prepare` before concurrent serving. The instance lock cannot coordinate
arbitrary CUDA users in the process. Inference failures stop further graph work;
`close()` remains available and failed graph resets can be retried.
Eager public operations also chain caller streams, retaining the final stream
until `close()` drains it before dropping borrowed timestep/model storage.
The graph uses raw `CUDAGraph.capture_begin`/`capture_end` with global capture
safety rather than Torch2.11's convenience context, which implicitly flushes
process-wide allocator caches. Only owned-stream synchronization/reset is used.

`execution_metadata.velocity_graph` separates per-request replays, condition
refreshes, captures, three cold warmup evaluations, encoded bodies and dynamically
observed native Linear calls encoded per replay. Logical replay work is not the
Python Linear-forward or successful CPU GEMM-enqueue counters. Sampler NFE counts
only logical field evaluations, never capture encoding or warmup. Raw head output
is borrowed scratch internally; returned complete CPU action arrays are retained
independently. No global CUDA cache flush or second weight copy is introduced.

The library integration is opt-in engineering functionality. The earlier A800
Euler2 **example pilot** is not evidence of Euler10/adaptive-PF qualification for
this new ownership/stream implementation. Consult per-run measurements before
making that claim; other geometry/hardware and robot success remain unqualified.

## Explicit numerical and kernel profiles

HyVLA defaults to `options={"numerical_profile": "semantic_bf16"}`. The opt-in
`official_local_fp32` profile preserves BF16 projection weights while matching
verified official-local RoPE/QK/norm/attention cast boundaries. FP32 QK/logits,
BF16 probabilities/PV and the official Euler BF16 increment are intentional;
ProbeFlow retains its shared FP32 integrator. Metadata records these boundaries.
The option is not named official-exact and is not a promise of oracle parity;
it may use more memory and be slower. Existing semantic behavior is unchanged.

FastWAM defaults to `options={"native_linear_backend": "legacy"}`. Explicit
`"cached"` selects the separate `flash_rt.flash_rt_gemm_plan` module. Build it
only in the intended task source/environment:

```bash
python scripts/build_gemm_plan.py --output-dir /task/source/flash_rt
```

This compiles one host-C++ cuBLASLt/pybind module, not full FlashRT. It reuses
shape descriptors/heuristics and refreshes the current bias pointer. Missing
cached extension fails clearly; there is no silent legacy/Torch fallback.
It remains **partial** native execution with unchanged tensor math/FLOPs and
64MiB owned workspace. `execution_metadata.native_plan_stats` records per-request
plan-build/query/enqueue deltas, not GPU timings. Actual GEMM and whole-model
comparisons are required before selecting this candidate for performance.

## Current boundaries

- One stateful backend serializes its calls. Independent candidates should use
  independent backend instances/processes, not share mutable graph buffers.
- `close()` explicitly destroys instance-owned Pi native graph/exec handles,
  including prompt bucket aliases, without destroying framework-owned streams
  or flushing process-global CUDA caches. Do not use a retained frontend or
  externally exported runtime after its owning backend has been closed.
- FastWAM direct frontend and backend cleanup drain the owned execution stream
  and release its raw-field graph before dropping model/runner references; reset
  failure retains owners and rejects inference until a successful close retry.
- No automated search, performance winner selection, task-success evaluation or
  agent-driven code mutation is run by this execution layer.
- The initial source-only phase did not execute models. Later verification and
  experiments must be read per run/model/profile; one passing fixture is not
  blanket hardware, task-quality or performance qualification.
