# Declarative VLA model catalog

The five built-in models have one JSON declaration each in
`flash_rt/vla/specs/`. `flash_rt.vla.model_spec` validates and discovers these
files using only the Python standard library. Discovery does not load model
code, import adapters, inspect checkpoints, download assets or probe a GPU.

```python
from flash_rt.vla.model_spec import (
    discover_specs, get_spec, list_models, load_spec, resolve_model, validate_spec,
)

names = list_models()
spec = get_spec("π0.5")
assert spec["model"] == "pi05"
assert spec["actions"]["worker_domain"] == "normalized-padded-model-actions"
```

`get_spec` and `discover_specs` return detached JSON-compatible dictionaries.
Modifying a returned dictionary does not change later discovery. `list_models`
returns canonical ids in deterministic directory order. Aliases are resolved
case-insensitively.

An explicit additional directory admits external declarations:

```python
spec = get_spec("my_robot", spec_dir="/local/my-model-specs")
from flash_rt.vla import BackendConfig, describe_backend

config = BackendConfig(
    model="my_robot", checkpoint="/local/weights",
    spec_dir="/local/my-model-specs",
)
description = describe_backend(config)
```

The directory supplements the builtins; duplicate model ids or aliases fail.
It is never read from an ambient environment variable. `BackendConfig.spec_dir`
travels through backend/candidate JSON encoding and decoding when set. Omitted
`spec_dir` does not add a field to legacy records, preserving their canonical
candidate identities.

The v1 declaration contains:

| Field | Contract |
|---|---|
| `model`, `name`, `aliases` | Canonical identity, display name and aliases. |
| `identity` | Source repository/revision, weight identity and license provenance. |
| `inputs` | Shapes, dtypes and layout/normalization semantics. Strings in shapes are symbolic dimensions. |
| `head` | `flow_matching`, `diffusion`, `autoregressive_tokens` or `other`, and head-specific defaults. A regression head uses `other` with `kind="regression"` and sampler `none`. |
| `actions` | Model width, effective channels, horizon, public domain, worker comparison domain and field. |
| `stages` | Stage ids, relative Python seams, interval keys and timing limitations. |
| `profiling` | Aggregate/split interval-to-stage mappings, expected legacy keys and graph/method strategy metadata. Active diagnostics record their precise expected intervals. |
| `attention` | Per-site mask, head dimension, Q/KV head counts, GQA and backend, with sources. |
| `capabilities` | Backend/framework/hardware/precision/sampler support and typed option schemas. |
| `features`, `tools` | Feature predicates, tool metadata/defaults and implemented compiler recipes. |
| `adapter` | Lazy import hooks and explicit prompt/context/resource policies. |
| `author` | Official implementation adapter module, package path and request options. |
| `fixture` | Deterministic CPU fixture recipe and provenance limitations. |
| `quality` | Numerical scope, channels and explicit thresholds or an unknown budget. |

Unknown facts are explicit objects, for example:

```json
{
  "status": "unknown",
  "value": null,
  "sources": ["local/source-record.md"],
  "notes": "The selected checkpoint has not supplied this fact."
}
```

`status="pinned"` requires a nonempty identity value. Source pins require a full
lowercase Git revision and repository URL. A source pin does not attest loaded
weight bytes or imply a weight license. License records remain unknown where
independently pinned code/weight license texts are unavailable. A documented
source-code license can be recorded separately without asserting the weight
license.

The validator rejects malformed/duplicate JSON keys, unsupported schema
versions, nonfinite values, contradictory revisions, invalid dimensions or
action channels, inconsistent attention head counts/GQA, unknown stage
references, unsafe seam/import syntax, invalid option defaults and inconsistent
compiler recipes. It checks declaration consistency; actual asset identity,
runtime numerical quality and hardware support require their normal execution
gates.

π0 and π0.5 declare the OpenPI author measurement commit
`dd5c5ca93ce229017477a75fdda9f16654ab91ef`. Their frontend constructor defaults
use two views and horizon10, whereas `profiles.measured_libero_h50` records the
three-view/horizon50 geometry in the historical comparison. The fixture recipe
uses that H50 geometry but generates synthetic tensors: it does not reproduce
historical input bytes or assert new measured model results.

`actions.domain` is the public frontend output domain.
`actions.worker_domain` is the comparison domain before numerical gates.
`worker_field="raw_actions_if_available"` preserves normalized Pi/Hy outputs;
DM0.5 explicitly uses `worker_field="actions"` for absolute public14 outputs and
declares its distinct normalized `raw_domain`. Callers must retain this
distinction when binding author outputs.

Model-specific behavior lives in `flash_rt/vla/adapters/`. The common registry
validates declared option types, invokes the selected CPU validation hook,
resolves hardware and imports the declared frontend binding. Hooks use
`python.module:callable` references; `binding(config, hardware)` returns
`(module, class_name, kwargs)`. A declaration without a binding supports CPU
discovery and description, but runtime loading fails before a CUDA probe.
Author modules use `author.adapter` and are maintained separately from product
frontend adapters.

Feature flags describe implementation opportunities as well as implemented
product paths. For example, a quantization feature may match a blocked tool,
and a batched frontend may match an unimplemented unified API extension.
Neither a matching flag nor a synthetic fixture is measurement evidence.
