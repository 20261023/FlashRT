# Pinned external tooling (setup only)

These assets install independent editable sources and virtual environments.
They never launch an external agent, authenticate, submit Slurm work or claim
kernel/model correctness.

| Registry tool ID | Pinned upstream |
|---|---|
| `kernel.forgekernel` | OpenBMB/ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc` |
| `kernel.kernelagent` | meta-pytorch/KernelAgent `e0647170da36ef9b059ac0bd3d60103aa4ed378b` |
| `compile.aitune` | ai-dynamo/aitune `1d3a4236ad8f458669d5633715b6956125f2289c` |

## Install

Use an absolute **new** destination per tool and the actual FlashRT runtime
Python (3.11 or 3.12 for this setup). The venv inherits that runtime's packages
read-only using `--system-site-packages`; installs go into the new venv only.
Torch, Triton and NumPy must retain their exact base versions. This is dependency
isolation, **not a sandbox** for generated code.

```bash
python examples/vla_autotune/external/setup.py install \
  --tool kernelagent --root /shared/external/new-kernelagent \
  --base-python /shared/flashrt-env/bin/python
```

The same command accepts `--tool forgekernel` or `--tool aitune`. For offline
hosts, transfer a public upstream Git bundle/clone and a wheelhouse containing
the matching platform/Python dependencies **and build dependencies**:

```bash
python examples/vla_autotune/external/setup.py install \
  --tool forgekernel --root /shared/external/new-forgekernel \
  --base-python /shared/flashrt-env/bin/python \
  --source-repo /shared/bundles/forgekernel.bundle \
  --wheelhouse /shared/wheels
```

`--wheelhouse` sets `PIP_NO_INDEX` and `PIP_FIND_LINKS` for pip's isolated build
subprocess too. Installation removes inherited provider credentials from pip's
environment. It does not read login files, source other sessions' `local.env`,
change HOME, install personal CLI/skill files, use sudo, or print credential
values. Failure leaves the new directory and a truthful `setup-result.json`.
Use a fresh destination rather than overwriting a partial installation.

Outputs: `activate.sh`, `setup-result.json`, `constraints.txt`, `packages.json`
on successful dependency installation, isolated caches, pinned `source/` and
`venv/`. Package versions beyond Torch/Triton/NumPy must be archived from the
actual installation; an online resolve is not a lockfile.

## ForgeKernel SM80 changes

`forgekernel-sm80/manifest.json` binds five reviewed patches, every preimage
and postimage, and the exact upstream pin:

1. CUDA and CuTe DSL `hw_kb/.../sm80/index.md`, explicitly **no certified recipes**.
2. `CUTE_ARCH["sm80"] = "sm_80"`.
3. CUDA evaluator defaults to `compute_80` / `sm_80` (explicit flags still win).
4. SASS accepts HMMA/LDGSTS/LDSM while retaining instruction count, dominant
   kernel and profiler-error checks. Hopper families remain accepted.
5. PM collection, overrides and finer retry never go below 20000 cycles.
   Short-kernel sample starvation remains FAIL/PARTIAL, never fabricated data.

This bundle is for a **dedicated SM80 installation**, not a global default change
for other hardware. No GPU execution or performance claim follows from it.

```bash
python examples/vla_autotune/external/setup.py apply-sm80 \
  --source /shared/new-forgekernel/source
/shared/new-forgekernel/venv/bin/python \
  examples/vla_autotune/external/cpu_smoke.py --tool forgekernel \
  --source /shared/new-forgekernel/source
```

`apply-sm80` rejects the wrong revision, tampered patch bytes, user changes,
mixed patch states, unrelated untracked files and symlink escapes. An exact
already-applied tree is a no-op. It never commits or pushes.

## KernelAgent A800 label

`kernelagent-sm80.json` records the upstream optimization **worker keyword**
`gpu_name="NVIDIA A100 SXM4 80GB"`. `KERNELAGENT_GPU_NAME` in `activate.sh` is a
convenience for the caller, **not an upstream auto-read setting**:

```python
import json
from triton_kernel_agent.opt_manager import OptimizationManager
settings = json.load(open("/shared/new-kernelagent/kernelagent-sm80.json"))
# Only in a later explicitly authorized generation/optimization task:
# manager = OptimizationManager(..., **settings["optimization_worker_kwargs"])
```

Run with cwd at the pinned KernelAgent source, not inside a FlashRT worker.
Disable dotenv lookup with `PYTHON_DOTENV_DISABLED=1`. Keep Anthropic workers
at most 3. This alias does not mislabel recorded Slurm GPU evidence: the real
device must still be measured as A800/SM80. Provider presence is not successful
authentication; absent authorized credentials is BLOCKED.

## AITune scope

`compile.aitune` remains a patch/needs-implementation tool. Installation does
not integrate it with a model. Use the pinned source's matching Torch version
extra and exact runtime constraints; never upgrade the baseline Torch to make
resolution succeed. FP8/MX/FP4 paths are not SM80 paths.

All actual `inspect -> wrap(strategy=...) -> tune(ignore_failing_modules=False)`
work, compilation, graph compatibility, numerical validation and `.ait` hashing
belong to a later allocated GPU task. A tuned artifact must be loaded before
FlashRT graph capture from a source-identified candidate worktree, never tuned
inside the measurement worker. Do not enable `AUTOWRAPT_BOOTSTRAP`.

## Licenses and next task

Public source URLs/pins, copied license digests and adaptations are in
`source-copies.json`; verbatim Apache-2.0 licenses and ForgeKernel's transitive
NOTICE are under `licenses/`. Upstream runtime sources are not vendored here.
Generated kernels remain untrusted and require their own provenance/license
review.

Next task, **separate model-specific segment**:
real call-site capture in Slurm -> external generation in its own allocation
(only after authorized credentials/budget) -> review/harvest -> independent
operator gate -> snapshot source patch -> reference/candidate/bookend campaign.
Only that last campaign can establish end-to-end speedup.
