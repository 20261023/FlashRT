# Pi0.5 FlashRT + ProbeFlow (dev, opt-in BF16)

ProbeFlow reduces Action Expert evaluations using the initial/lookahead velocity
direction. FlashRT still executes the neural model; the sampler and backend are
orthogonal choices. This adaptation requires no training or checkpoint edits.

## Frontend interface

```python
from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
from flash_rt.models.pi05.probeflow import ProbeFlowConfig

model = Pi05JaxFrontendRtx(
    checkpoint_dir,
    num_views=2,
    chunk_size=50,
    use_fp8=False,
    sampler="probeflow",
    probeflow_config=ProbeFlowConfig(
        epsilon=0.008, probe_horizon=0.5, max_steps=20, step_increment=2,
    ),
)
model.set_prompt(prompt)
model.calibrate_with_real_data([observation])
result = model.infer(observation, noise=initial_noise)
print(result["sampler_metadata"])
```

The Torch safetensors frontend accepts the same sampling parameters. Omit both
new options to retain the previous fixed-Euler path. `num_steps` still controls
Euler; ProbeFlow's integration budget is controlled by its config, not that
fixed-step parameter. The top-level `load_model` dispatcher is not extended in
this first dev version: choose the RTX frontend explicitly.

Robot-specific input/tokenization/action decoding must remain with the robot
adapter. The standard frontend returns its existing LIBERO output schema; do
not call that schema UR5 actions. The existing UR5 adapter's norm/tokenizer and
30x7 public action conversion remain authoritative.

## Execution and counters

- OpenPI model time is t=1-s, and the progress velocity is -v. The opt-in output
  projection is scaled by -1 rather than baking a fixed -1/num_steps into it.
- The two-probe branch reuses both evaluations. A denser N-step branch restarts
  from the original noise, reuses the first velocity, and discards the lookahead
  velocity: NFE=2 for N=2, otherwise NFE=N+1.
- Actual time embeddings/styles are precomputed for the union of the supported
  schedules. N is never implemented as a truncation of a fixed ten-step table.
  Time-table and lookup keys use the exact same float expressions as runtime;
  positive near-endpoint probe times are not rounded to zero.
- Prefix and per-time velocity graphs are captured separately. The host makes
  a fresh routing decision each invocation; that scalar synchronization and
  Torch vector operations are part of latency, not excluded accounting.
- Every call recomputes the full visual/language prefix. No cross-frame K/V
  reuse is enabled by this sampler.

Initial limits: single-sample BF16, existing cublaslt decoder; no OSC,
quantization, SDE, fixed denoise trace, CFG/batching, hot reload or native-runtime
export. Unsupported combinations raise instead of silently selecting Euler.
Exact/fixed state-prompt execution remains owned by the existing frontend.
Both batching entry points (`set_batched_mode` and `select_batch_size`) reject
ProbeFlow before changing the active pipeline, backend or prompt state.

## Checks

CPU analytic/reference and routing checks:

```bash
python -m unittest discover -s tests -p test_pi05_probeflow_math.py -v
python -m pytest tests/test_pi05_probeflow_math.py \
  tests/test_pi05_probeflow_frontend.py tests/test_pi05_probeflow_runtime.py
```

`examples/pi05_probeflow_smoke.py --help` documents the bounded UR5 fixture
acceptance check. Run it only in an allocated Slurm `srun` step. It checks
graph/eager equality, seeded replay, live threshold routing and equality of the
two-probe h=0.5 branch with native two-step Euler. Use an isolated output folder.

Fixture timing is **not** a raw/FlashRT/ProbeFlow matched Mock P50, full
`run_policy` latency, HTTP GET→POST latency, task success or a speedup result.
The later matched experiment must fix checkpoint, observations/noise and timing
boundaries, exclude warmup/capture, and include all routing overhead. Recorded
Mock final camera images cannot measure action-dependent closed-loop outcomes.

### Complete recorded-episode comparison

`examples/pi05_probeflow_episode_benchmark.py` adds an offline `prepare` /
`worker` / `summarize` harness. It does not start a robot, client or HTTP server.
Prepare the existing recorded episode once, then run `raw_euler`, `flash_euler`
and `flash_probeflow` serially inside the **same one-GPU Slurm allocation**.
Each worker uses the same real checkpoint, prompt, frame sequence and shared
nearest-even BF16 noise. Raw explicitly verifies a JAX GPU backend and model
parameter residency through `Array.devices()`; JAX CPU Orbax loading is not a
GPU Raw baseline.

The primary wall-time boundary is host shared RGB224 + raw UR5 state7 + noise
to synchronized host public actions30x7. It includes state/token processing,
model transfers/execution, every probe/lookahead, host routing and public action
decoding. It excludes file preparation, capture/warmup and measurement-file
writes; it is **not** HTTP GET→POST latency. Per-call JSONL, P50/P99, actual NFE,
normalized/public output arrays and an independent BF16 integration oracle are
retained. Different samplers are not required to match Euler outputs.

UR5 normalization must retain the original fork's **float64 JSON quantiles
until tokenization**. Casting them to float32 first can change a state bin at
the -1/+1 boundary and therefore change prompt tokens. The harness verifies
exact token/mask equivalence with the original Raw transform before timing.
First/last camera montages are shared recorded inputs, not predictions or
closed-loop outcomes attributable to a method.

## Attribution

An independent implementation of the equations in [ProbeFlow v3, Algorithm 1](https://arxiv.org/html/2603.17850v3#S3).
The authors' [Evo-1 reference](https://github.com/zfan00666/probeflow/blob/bc0037d4745b15c8f3ff39c700dc98921c5bb313/probeflow/model/action_head/flow_matching.py)
was used to verify routing/NFE semantics. No Evo-1 network, dependencies or
weights are imported.
