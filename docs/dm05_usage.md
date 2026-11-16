# DM0.5 base on A800 / SM80

The first DM0.5 integration is a **project-owned BF16/Euler10 pipeline**, not a
renaming of the author's complete inference wrapper and not an ApxInf binding.
`backend="torch"` uses project Torch operators and CUDA graphs;
`backend="flashrt"` uses real cached FlashRT BF16 GEMM, vendored BF16 FA2 for
vision/action attention, and strict BF16 GELU. The latter is **partial native /
mixed Torch coverage**, not a fully native model: patch convolution, norms,
RoPE, masked prefix SDPA, pooling, gates, Euler and host transforms remain Torch.
There is no quantization or reduced-step path in this integration.

## Opt-in project compiler method

`backend="torch"` also admits `options={..., "project_optimization": "compiled"}`.
The default is `"none"`; `backend="flashrt"` rejects `"compiled"`, so selecting
this method never silently changes the original mixed/native comparator.

The compiler method is implemented in this project's existing SM80 model entry:
packed QKV and MLP gate/up projections, direct tensor SDPA (native GQA at the
unmasked action site), and three reusable Inductor layer functions for vision,
prefix and action. Arbitrary image/text prefix masks remain supported by the
masked SDPA site. There is no call to ApxInf or the author's complete GPU model.
BF16, Euler10, image geometry, current-state preprocessing and the public CPU
action boundary are unchanged.

Each unit uses `fullgraph=True`, `dynamic=False` and default Inductor GEMM
selection, with compiler-owned CUDA graphs explicitly disabled. Weights and
time modulation tensors are arguments; layer/step indices are not compile
arguments. The frontend prewarms all regions/strides for each exact-prefix
profile **before** its existing whole-policy CUDA graph capture. Compiler errors
raise; there is no suppressed-error or eager-on-compiler-failure path. Small
uncompiled tensor glue is still inside that same whole-model CUDA graph.

Execution metadata reports `project_optimization`, the compile configuration,
and per-profile compile-plus-first-forward setup seconds. A new prefix length
pays compilation/capture costs, and packed weights and graph pools require
additional memory. This is an opt-in candidate, not a supported superiority or
release-quality claim without paired full-policy timing and output evidence.

Select the candidate through the VLA options API (the top-level `load_model`
signature is not extended by this option):

```python
from flash_rt.vla import BackendConfig

config = BackendConfig(
    model="dm05", checkpoint="/local/DM05-base", backend="torch",
    hardware="cuda_sm80", precision="bf16", num_steps=10,
    options={"source_dir": "/local/pinned-opendm", "num_views": 3,
             "use_cuda_graph": True, "max_graph_profiles": 2,
             "project_optimization": "compiled"},
)
```

## Fixed first profile

- NVIDIA compute capability8.0 (`cuda_sm80`), Torch only.
- Dexmal/DM05 base revision `5cd18734814abb075a9ccfd9ad6d16777b5cf10e`;
  `model.safetensors` SHA256
  `b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd`.
- Three current448x448 cameras, no history, batch1, horizon50,
  internal32 / public14 action channels, robot `DOS W1`, Euler10.
- OpenDM `03f799120d425ead51d9482bb1da950b6dabade9` is required **read-only
  for CPU preprocessing and action reconstruction only**. `source_dir` is
  explicit; the frontend verifies its Python source identity. No downloads,
  upstream GPU model construction or external inference engine are performed.
- BF16 action projection and Euler rounding match the author's measured **hot
  suffix graph**, not the author's first eager FP32 projection branch.
- Unsupported checkpoint hashes, geometry, precision, robot, padding/history,
  architecture and sampling options raise; no fallback is selected.

```python
from flash_rt.vla import BackendConfig, load_backend

config = BackendConfig(
    model="dm05", checkpoint="/local/DM05-base",
    backend="flashrt",  # or "torch" for the project-owned Torch path
    hardware="cuda_sm80", precision="bf16", num_steps=10,
    options={"source_dir": "/local/pinned-opendm", "num_views": 3,
             "use_cuda_graph": True, "max_graph_profiles": 2},
)
with load_backend(config) as policy:
    result = policy.infer(
        {"images": [head_rgb, left_wrist_rgb, right_wrist_rgb],
         "state": state14, "robot_type": "DOS W1"},
        prompt="Pick up the roller using both arms", noise=initial_noise,
    )
    actions = result["actions"]  # complete absolute float32 NumPy [50,14]
    coverage = result["execution_metadata"]
```

Images are PIL RGB or uint8 arrays; state is finite14D. `noise` is optional,
floating `[1,50,32]` or `[50,32]`, and is rounded to BF16 before use. If omitted,
a fresh CUDA BF16 draw is used without resetting global RNG. CPU noise must be
finite; CUDA-noise finiteness is checked through the final completed output.
Saved author RNG draws can be supplied without substituting model math.

Every inference runs the current observation's CPU tokenization, so **same
prompt plus changed state refreshes state tokens**, even at the same prefix
length. The frontend owns bounded exact-prefix graph profiles and refreshes
pixels, IDs, image-row indexes, masks and noise on every call. A previously
unseen prefix length pays setup/capture; exceeding `max_graph_profiles` raises.
There is no unbounded graph cache or silent eager fallback. Calls are serialized
and outputs are independent completed CPU arrays.

The top-level API routes to the same frontend:

```python
import flash_rt

model = flash_rt.load_model(
    "/local/DM05-base", config="dm05", framework="torch",
    hardware="cuda_sm80", backend="flashrt", num_views=3,
    use_fp8=False, precision="bf16", num_steps=10,
    source_dir="/local/pinned-opendm", use_cuda_graph=True,
)
actions = model.predict([head_rgb, left_wrist_rgb, right_wrist_rgb],
                        prompt="Pick up the roller using both arms", state=state14)
# For fixed-noise requests, model.infer(observation, noise=...) exposes the
# frontend dictionary contract; predict retains its existing actions-only API.
```

## Native artifacts and measurement

No new CUDA kernel or binding is introduced. The FlashRT variant requires
existing `flash_rt_kernels`, `flash_rt_fa2`, and the standalone
`flash_rt_gemm_plan` module. The latter can be built with the existing
`scripts/build_gemm_plan.py --output-dir <candidate>/flash_rt` in an allocated
CUDA12.8 environment. It does not rebuild or overwrite the production kernels
binary. All native modules used in formal measurements are separately hashed.

`examples/vla_autotune/dm05_project_latency.py` times construction through
`BackendConfig/load_backend` and complete public `VLABackend.infer` outputs.
Each path retains all four author reference cases, exact saved noise,
five warmups and ten formal outputs per case. The timer includes CPU
preprocessing/tokenization, H2D, complete model, D2H and absolute reconstruction;
it excludes file decode/network and separates load/capture. Formal output
arithmetic uses unchanged public14 relL2<=.05 and cosine>=.999 thresholds.
Original-author timing includes CUDA RNG, whereas these paths use provided
noise; retain this caveat when comparing the rows.

Tests, operator gates and extra reviews were explicitly waived for this task;
formal inference evidence is not a claim of release/robot-task acceptance.
First valid baseline is not a fastest/winner claim.
