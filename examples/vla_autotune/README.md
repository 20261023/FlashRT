# VLA experiment and validation examples

These are source-checkout examples above `flash_rt.vla`, not a claim that every
model, hardware path or numerical profile is qualified. Keep model weights,
fixtures and generated outputs outside the source tree. Experiments must use
an immutable source snapshot and explicitly named local inputs/checkpoints.

## HyVLA recorded-oracle validation

`validate_hyvla.py` operates on the revision-bound public fixture contract:
three uint8 camera histories `[6,480,640,3]`, normalized state20, supplied FP32
noise `[40,32]`, and recorded normalized action oracles. It is not a generic
robot observation adapter or a task-success benchmark.

Run model execution **inside an allocated Slurm GPU srun step**, not a login
node. The current runner checks the actual single A800/SM80 device. For example,
inside your site's one-GPU batch script after selecting its existing runtime:

```bash
export PYTHONPATH="/snapshot/FlashRT:$PYTHONPATH"
srun -n 1 -c 8 python -B -u /snapshot/FlashRT/examples/vla_autotune/validate_hyvla.py \
  --checkpoint /local/Hy-Embodied-0.5-VLA-RoboTwin \
  --fixture /local/public-reference/baseline.npz \
  --sampler euler --num-steps 10 --numerical-profile official_local_fp32 \
  --use-graph --warmup 1 --repeats 2 --require-oracle --output /runs/hy-baseline
```

Use the site's own allocation and thread conventions; `-c 8` is an example for
the tested A800 setup, not a portable Slurm requirement. The runner performs no
submission, model download, checksum scan or permission changes. It refuses to
overwrite an output directory.

The original recorded gate remains every element within
`0.01 + 0.01 * abs(reference)` and cosine at least `.999`, with finite outputs.
`--atol`/`--rtol` only affect internal graph/eager or saved-run comparisons, never
the original oracle gate. `--require-oracle` returns exit3 on Euler10 mismatch.
Other step counts are only diagnostic relative to the Euler10 oracle.

`semantic_bf16` is the unchanged default. `official_local_fp32` matches verified
local dtype/cast boundaries, not the complete official implementation; its name
does **not** mean official-exact or guarantee all fixture gates pass.

CPU-only operations need no Slurm allocation:

```bash
python examples/vla_autotune/validate_hyvla.py --help
python examples/vla_autotune/validate_hyvla.py \
  --compare /runs/euler2 /runs/pf2 --exact --output /runs/comparison
```

Saved-run comparison checks checkpoint/input/profile/source provenance before
reporting numerical differences. Original oracle acceptance, reproducibility,
graph/eager parity, numerical replay proxies, and robot success are separate
claims. Inspect the complete result, not just a scheduler exit code.
