# SmolVLA LIBERO onboarding

This first-light adapter exposes the pinned LeRobot **Torch** implementation
through the common VLA backend and author-baseline interfaces. It does not
claim FlashRT-native neural kernels or closed-loop task success.

## Pinned assets

| Asset | Source / revision |
|---|---|
| Original implementation | [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf), `b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf` |
| Policy | [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102), `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| VLM / tokenizer | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/4cde9d208e19519c36917aa170fffc7e381aa0ad) |
| Real fixture source | [lerobot/libero](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4), `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |

The implementation, policy card, VLM card and dataset card state Apache-2.0.
The model directory retains the published configuration, processor JSONs and
both processor state files. Local path overrides are constructor arguments,
not edits to published checkpoint assets.

Policy `model.safetensors` SHA256:
`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`.
The declaration separately binds the whole canonical checkpoint directory.

## Explicit contract

- One A800/SM80, Torch backend, original mixed BF16/FP32 parameter dtypes.
- Two current `uint8[256,256,3]` RGB images in checkpoint order
  (`observation.images.image`, then `observation.images.image2`).
- Original `float32[8]` state and task text; checkpoint processors own newline,
  tokenization, normalization and image preprocessing.
- Explicit `float32[1,50,32]` host noise for measured requests.
- Original Euler route; actual `denoise_step` calls are counted.
- Output is a complete `float32[50,7]` CPU action chunk after the checkpoint
  unnormalizer. Queue-based consumption of one action is not benchmarked.
- Public numerical budget: relative L2 ≤ 0.05 and cosine ≥ 0.999 on all seven
  denormalized channels. This is not a task-success metric.

The model declaration is initially external:
`flash_rt/models/smolvla/model_spec.json`. Pass
`spec_dir="flash_rt/models/smolvla"` explicitly; discovery does not load
checkpoints or import neural implementations.

```python
from flash_rt.vla import BackendConfig, load_backend

config = BackendConfig(
    model="smolvla",
    checkpoint="/absolute/pinned/checkpoint",
    backend="torch",
    framework="torch",
    hardware="cuda_sm80",
    num_steps=10,
    spec_dir="flash_rt/models/smolvla",
    options={
        "source_dir": "/absolute/clean/pinned/lerobot",
        "vlm_dir": "/absolute/pinned/vlm",
        "use_cuda_graph": True,
        "max_graph_profiles": 4,
    },
)
with load_backend(config) as backend:
    result = backend.infer(observation, prompt=task_text, noise=fixed_host_noise)
```

Model execution must run inside an allocated Slurm GPU batch. The frontend
owns bounded graph profiles keyed by typed input shapes, updates static input
storage before replay, and releases its own graphs. It reports actual graph
replay telemetry separately from neural-operator coverage.

## Real input generation and acceptance

`examples/vla_autotune/smolvla_workload.py` extracts episodes 0/1, frames 0/40
from a local pinned LeRobot v3 LIBERO subset. It verifies dataset file bytes
before decoding RGB frames, preserves state/task values and records provenance.
Only initial Gaussian diffusion noise is generated.

```bash
python -m examples.vla_autotune.smolvla_workload \
  --dataset /absolute/pinned/libero-subset \
  --output /absolute/private/real-workload \
  --spec-dir flash_rt/models/smolvla
```

The independent original-author adapter is
`flash_rt.vla.author_adapters.smolvla`. It imports no project neural frontend;
the unified author producer verifies the actual author class/source, checkpoint
and workload identities and records actual argv and hardware.

Use `vla-onboard-model` and `python -m flash_rt.vla.onboard --help` for the
author-bound reference/graph/eager/bookend campaign and final verification.
Keep raw images, states, noise, actions and cluster-local paths private.
Subsequent source optimizations require committed worktree candidates, ready
baseline/candidate snapshots and plan v3; this initial adapter is not a speedup
claim.
