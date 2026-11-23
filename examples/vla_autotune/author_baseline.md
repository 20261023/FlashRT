# Author references in ordinary campaigns

This is an opt-in measurement-harness change. Land it as harness maintenance
before creating source snapshots; it is not a candidate patch. Existing plans
without `author_baseline` retain the internal reference/candidate/bookend flow.

`controller init --author-baseline /absolute/author.json` (or
`initialize_campaign(..., author_baseline=dict)`) adds a separate author producer
before the internal reference. The ordinary `batch` and `controller ingest`
commands handle it; no separate GPU job or manual comparison step is required.
Batch runs the author in an isolated process, serially with the other attempts.
All model computation still requires an allocated Slurm GPU `srun` step.

For example, a DM author config is:

```json
{
  "mode": "produce",
  "source": "/shared/author/opendm",
  "source_revision": "03f799120d425ead51d9482bb1da950b6dabade9",
  "python": "/shared/envs/opendm/bin/python",
  "num_steps": 10,
  "options": {
    "noise_seeds": {
      "demo-seed0": 0,
      "demo-seed17": 17,
      "state-seed0": 0,
      "prompt-seed0": 0
    }
  }
}
```

The checkpoint comes from the internal baseline Candidate. Workload, output
root, checkpoint and author source paths must be absolute and locally accessible
to the relevant controller/worker processes. The controller reads the workload
on initialization; it does not load a model.

Use the environment's `bin/python` path as `python`. When an author runtime
needs a dependency overlay, list its absolute site-packages paths in
`extra_pythonpath`. The producer starts with those explicit dependencies.

```bash
python -m flash_rt.vla.autotune.controller init \
  --root /shared/campaigns/dm-author \
  --comparison-id dm-author \
  --baseline /shared/dm-baseline.json \
  --workload /shared/dm-workload.json \
  --output-root /shared/results/dm-author \
  --warmup 5 --repeats 40 \
  --quality-gate '{"scope":"absolute-public14","action_dim":14,"max_relative_l2":0.05,"min_cosine":0.999}' \
  --author-baseline /shared/author.json
```

Use the existing campaign submission/queue bookkeeping and Slurm batch script.
Its allocated step runs:

```bash
python -m flash_rt.vla.autotune.batch --plan /shared/campaigns/dm-author/plan.json
```

Then ingest normally:

```bash
python -m flash_rt.vla.autotune.controller ingest \
  --root /shared/campaigns/dm-author \
  --results /shared/results/dm-author --job-finished
```

For SSH/fetch, retain the original controller `plan.json` and ledger locally.
Batch copies the original author JSON and NPZ into `output_root/control/`, with
the executed plan, descriptor and `author_portable.json`. The manifest binds
their SHA256 digests, original plan/instance/job, and each finished attempt's
`result.json`, `outputs.npz` and recorded runtime. Fetch `control/` plus every
attempt's JSON/NPZ, keeping their relative paths; source/checkpoint trees are
not needed for ingest. For example:

```bash
rsync -a --include='*/' --include='*.json' --include='*.npz' --exclude='*' \
  cluster:/shared/results/dm-author/ /local/results/dm-author/
python -m flash_rt.vla.autotune.controller ingest \
  --root /local/campaigns/dm-author --results /local/results/dm-author --job-finished
```

Local ingest verifies the fetched plan against the original plan and ledger,
all bound file digests, candidate/attempt/job identities and GPU UUIDs. It
recomputes both numerical gates from fetched CPU actions. It never reloads
cluster checkpoints, fixtures, author packages or producer executables. Worker
verification of the actual remote checkpoint, fixed inputs and source remains
in place. A partial fetch, changed file, swapped campaign/attempt or missing
portable manifest blocks qualification; fetch the complete evidence and resume
ingestion. Earlier complete bundles can be exported on the trusted cluster host
without a GPU or model imports, using the archived batch job identity:

```python
import json
from pathlib import Path
from flash_rt.vla.autotune.author import export_portable_campaign
plan = json.loads(Path("/shared/campaigns/dm-author/plan.json").read_text())
export_portable_campaign(plan)
```

This requires the original `batch_result.json`, descriptor and author/worker
outputs; it does not manufacture missing runtime or timing evidence. Hashes
verify transport consistency; they are not signatures of an untrusted host.

`plan.author_baseline.producer_command` contains a complete executable argv for
`examples/vla_autotune/author_baseline.py`, including its exact source spec,
workload, output, budgets and campaign identity. This command can also be run
directly inside an allocated GPU step. `options` is model specific:

| Model | Actual author API / reused loader | Options and output domain |
|---|---|---|
| π0 | OpenPI `Pi0Config.load` / original JAX `Policy.infer(noise=...)`, same loading contract as `official_native_latency.py` | Optional `action_horizon` (50), `max_token_len`; normalized padded raw actions, normally `[50,32]` |
| π0.5 | OpenPI `PI0Pytorch` / original PyTorch `Policy.infer(noise=...)`, strict shared-weight safetensors loading | Same options; normalized padded raw actions, normally `[50,32]` |
| Hy | `official_hyvla_latency.author_request`, `admit_author`, original `HyVLA` and safetensors loader | `{}`; normalized padded `[40,32]`; original Torch 2.7.1 environment and author export marker required |
| FastWAM | `validate_fastwam.resolved_model_config` / `load_upstream_reference`, original `FastWAM.infer_action` | Required `backbone_dir`, `noise_seeds`; optional resolved `model_config` JSON, `action_horizon` (32), `sigma_shift` (5), `compile_action_infer` (true); normalized batched `[1,32,7]` |
| DM | Original `DM05Exp` default backend `_prepare_model_input` → `_predict`, same API as `official_dm05_latency.py` | Required `noise_seeds`; absolute reconstructed DOS W1 CPU `[50,14]` |

Author source revisions for the existing Hy, FastWAM and DM profiles are pinned
to `af57e7507ec5964b52fdf6296741e553cbcd3288`,
`7faa71108368fbb3b6885649f112af607427a2d4`, and
`03f799120d425ead51d9482bb1da950b6dabade9`, respectively. OpenPI requires the
caller's exact 40-character source commit. A clean author Git checkout, marked
author export, or the independently pinned DM Python bytes must identify the
source. FastWAM's existing loader also requires its adjacent `SOURCE.json`.
The loaded model class must come from that author package; loading FlashRT
neural modules in the producer is an error.
OpenPI JAX restores the actual Orbax `params` tree as `jnp.bfloat16` and loads
with `remove_extra_params=False`, enforcing the complete parameter structure.
PyTorch inspects the saved tensor keys to choose an unprefixed or `model.`
namespace and uses strict, shared-weight-aware loading. Checkpoint model
variants/dimensions are read from its config rather than silently assumed;
loading evidence is recorded in producer metadata.

Workload fields follow `NPZWorkload`, not a second fixture format. Pi and DM
accept three ordered `images` arrays and `state`, with the manifest prompt.
Hy accepts its three ordered camera histories in checkpoint camera order plus
state and prompt. FastWAM accepts exactly `input_image`, `proprio`, `context`,
and `context_mask`; cached T5 is part of the supplied workload and not timed.

OpenPI and Hy receive the supplied noise directly. FastWAM and DM retain their
original seeded APIs. Every request needs an explicit `noise_seeds` entry; an
untimed observation verifies the author's actual `torch.randn` draw against the
workload noise in the author's dtype. No RNG result or sampler is replaced.
Arbitrary noise that those APIs cannot reproduce is rejected.

The producer keeps the existing author-style `result.json` plus stacked
`outputs.npz` (`actions`, `request_ids`, `latency_ms`), adding explicit
`output_domain`, source/model provenance, actual producer argv and content
digests. Checkpoint assets are streamed through SHA256, including auxiliary
FastWAM VAE weights. This costs host IO outside measured latency. Typed fixture
values, prompts, request order and fixed noise are hashed independently of NPZ
container paths. Workers recheck these before loading their model.

To reuse a bound result from any of the five producers, change the config to
`"mode": "saved"` and add `"result_path": "/shared/previous/author/result.json"`.
Keep its source spec and options. The original producer/campaign/job identity
is preserved, and the result/output digests are pinned at campaign init.
The source paths in the archived spec are provenance; current checkpoint bytes
must remain accessible and match.

`mode=saved` also accepts the original four-case DM `official_dm05_latency.py`
`result.json` directly. Keep the original case IDs in the loop workload.
The author interpreter performs a CPU-only binding: it checks the original
script SHA, pinned Python and base weight bytes, all four images/prompts/states,
saved noise, action/timing files, and exact reproduction of saved `prepared.pt`
using original author CPU transforms. It preserves the old samples and records
the runnable original producer command as reconstructed from the verified
script and recorded arguments; the old interpreter path was not recorded.
No new inference is performed by this binding and no workload is exported.
Historical loaded-weight identity remains limited by the old report's revision
claim; that producer did not hash its loaded weights. The current base weight
pin and exact prepared/noise checks are recorded explicitly.

Other old, unbound author reports are refused with a concrete regeneration
argv. Missing files, changed digests, different fixtures/checkpoints/domains,
incomplete coverage or producer failure never become a numerical pass. Batch
writes an explicit author-reference failure and does not start internal
attempts if author production/binding fails.

`result.quality` always describes the internal reference comparison; its
reference point remains an internal self-reference. `result.author_quality`
compares every measured output to actual author outputs, including the
reference and bookend themselves. Both gates use the declared numerical
budget and channel slice and must pass to qualify a winner. The summary labels
an internal Euler10 baseline separately from the author.

`summary.author_baseline.latency_comparisons` reports both measured medians.
It emits a strict ratio only for the same campaign, job, GPU UUID, measurement
boundary, output domain, precision, sample coverage, sampler and actual NFE.
The author gate must also pass, and recorded noise work and CPU action return
fields/domains/shapes/dtypes must agree. In particular, seeded author RNG and
provided-noise execution differ, as do an author raw-only return and a worker
that also decodes public actions. Those cases show medians and reasons without
a strict ratio. Older results without this work contract cannot supply one.
Saved references and different step/precision configurations retain explicit
incomparability reasons. Numerical quality and timing evidence remain separate;
neither implies closed-loop task success.

DM worker bundles now save/compare `actions` in the absolute public domain.
Pi0/Pi0.5 continue saving raw model outputs. DM's volatile enqueue statistics,
prefix lengths, graph addresses, capture/compile data and request counters are
retained per request, while only a stable execution whitelist is compared as
backend identity. Older DM internal NPZ references without an explicit output
domain must be regenerated.
