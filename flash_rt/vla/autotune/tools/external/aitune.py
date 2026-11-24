"""AITune adapter: per-module backend selection for the Torch parts of a pipeline.

NVIDIA AITune (``ai-dynamo/aitune``) inspects a callable for executed
``torch.nn.Module`` members (``inspect``), replaces chosen submodules in place
with its ``Module`` proxy (``wrap``: ``parent.set_wrapped(name, proxy)``),
builds and measures backend candidates for each (``tune``: TensorRT,
Torch-TensorRT, TorchAO, Inductor, ONNX Runtime, eager) and stores the result
as an ``.ait`` checkpoint (``save``/``load``).

Lane: **patch**, not config. AITune changes which implementation executes a
Torch submodule, FlashRT's registry has no option that enables it, and the
wrap/load call has to sit inside the frontend (before FlashRT captures its own
CUDA graphs). That integration is a source change in a snapshot worktree
(vla-patch-candidate), measured against the unpatched base.

What can actually be wrapped in FlashRT:
- ``fastwam``: the upstream FastWAM Torch modules held by
  ``FastWAMActionPipeline.model`` (``video_expert``, ``action_expert``, ``mot``,
  ``vae``; flash_rt/models/fastwam/pipeline.py). Not pi0/pi0.5, whose RTX
  pipelines run native fvk kernels on raw buffers.
- ``hyvla``: nothing as shipped. The Hy frontends hold weights as plain tensors
  and call ``torch.nn.functional`` plus fvk kernels, so ``inspect`` finds no
  ``nn.Module`` to wrap. A patch must first package a pure-Torch region (e.g. the
  ViT blocks in flash_rt/models/hyvla/pipeline_thor.py ``vit_forward``:
  F.linear / SDPA / layer_norm) as an ``nn.Module``; fvk calls on ``data_ptr()``
  cannot be traced or exported by any AITune backend.

This module is SPEC + documentation only. ``render_script`` is deliberately
absent: backend config constructors (``TensorRTBackendConfig`` kwargs) and the
``InspectedModulesInfo`` methods appear only in upstream skill text, not in the
fetched source, and the dataset contract for a FlashRT submodule depends on the
patch. ``API_FACTS`` records the signatures that WERE read from source.
Nothing here imports torch or aitune or touches the network.
"""

from __future__ import annotations


UPSTREAM_COMMIT = '1d3a4236ad8f458669d5633715b6956125f2289c'
TOOL_ID = 'compile.aitune'

# Signatures read from upstream source at UPSTREAM_COMMIT (path @ git blob sha).
API_FACTS = {
    'aitune/torch/__init__.py@27d4dba8bf117c9b1570af80c2dc9a287e497177': (
        'aitune.torch exports inspect, wrap, tune, save, load, OneBackendStrategy, '
        'FirstWinsStrategy, MaxThroughputStrategy, MinLatencyStrategy, '
        'LatencyBudgetStrategy, DynamicDim, BatchDim, jit_config, patch_for_jit_tuning'),
    'aitune/torch/inspecting/inspecting.py@2a4f6d7a61f5c62f7aaf6b040f2ef5f6bd0b17b8': (
        'inspect(obj, dataset, inference_function=None, number_of_iterations=..., '
        'warmup_iterations=..., min_depth=0, max_depth=5, call_tree=False) -> '
        'InspectedModulesInfo; finds nn.Module members executed by obj'),
    'aitune/torch/inspecting/wrapping.py@93b99040fa2ffbb54488fd18b6d5c8d3e294d042': (
        'wrap(obj, modules, strategy=None, strategies=None): replaces submodules in place '
        '(parent.set_wrapped) or returns a Module proxy when the root is selected; clears '
        'the global MODULE_REGISTRY first'),
    'aitune/torch/tuning.py@39f97352d9340f121a2ca1f99f73669500202316': (
        'tune(func, dataset, batch_sizes=None, max_num_batches_per_batch_size=None, '
        'device=None, dry_run=False, disable_external_logging=False, clear_cache=False, '
        'ignore_failing_modules=True) runs func under torch.no_grad, tunes each registered '
        'module with the others deactivated, and puts a failing module in passthrough; '
        'save(module, path, storage=None); load(module, path, storage=None, '
        'device_map=None, disable_external_logging=True)'),
    'aitune/torch/backend/__init__.py@023fba07e9338c22426098c505915e29a1af6fe3': (
        'backends: TensorRTBackend, TorchTensorRTAotBackend, TorchTensorRTJitBackend, '
        'TorchAOBackend, TorchInductorAotBackend, TorchInductorJitBackend, '
        'ONNXRuntimeBackend, TorchEagerBackend, KernelSelectorBackend (+ *Config classes)'),
}

SPEC = {
    'tool_id': TOOL_ID,
    'title': 'AITune: per-module TensorRT / Torch-TensorRT / TorchAO / Inductor backend '
             'selection for Torch submodules',
    'lane': 'patch',
    'stages': ('denoise.velocity', 'vision.encode', 'prefix.prefill', 'kernel.gemm',
               'kernel.attention', 'kernel.conv'),
    'models': ('*',),
    'model_predicate': {'any': [
        {'path': 'features.torch_modules', 'equals': True},
        {'path': 'features.functional_torch', 'equals': True},
    ]},
    'frameworks': ('torch',),
    'exactness': 'approximate',
    'status': 'needs_implementation',
    'reason': (
        'No FlashRT wrapper/artifact-loader integration exists: implement that source patch '
        'before environment setup is sufficient. Wraps torch.nn.Module submodules only. FastWAM: the upstream Torch modules '
        '(video_expert, action_expert, mot, vae) can be wrapped once integrated by a patch; '
        'Hy: the frontends use functional Torch ops and fvk kernels on plain tensors, so a '
        'patch must first package a pure-Torch region (ViT blocks) as an nn.Module; '
        'pi0/pi0.5 native pipelines have nothing to wrap. Needs aitune (Python 3.11+, '
        'PyTorch 2.8+) and TensorRT/Torch-TensorRT wheels matching FlashRT\'s torch/CUDA in '
        'the WORKER runtime (it runs in-process), writable AITUNE_CACHE_DIR/TRITON_CACHE_DIR, '
        'and a GPU allocation for tuning. Its backends can capture their own CUDA graphs, '
        'which conflicts with FlashRT\'s graphs. No FlashRT backend option exists: '
        'integration is a source patch, and the tuned .ait artifact must be pinned.'),
    'upstream': {
        'repo': 'https://github.com/ai-dynamo/aitune',
        'commit': UPSTREAM_COMMIT,
        'license': 'Apache-2.0',
        'notice': False,
    },
    'skills': ('aitune-inspect', 'aitune-tune', 'aitune-validate', 'aitune-benchmark',
               'vla-patch-candidate'),
    'setup': [
        'pip install --extra-index-url https://pypi.nvidia.com aitune (pin the version that '
        f'matches upstream {UPSTREAM_COMMIT}) into a dedicated worker venv built on '
        'FlashRT\'s exact torch/CUDA; add tensorrt / torch_tensorrt only for those backends',
        'check on the allocated A800: nvidia-smi; python -c "import torch; '
        'print(torch.cuda.get_device_capability())"; python -c "import aitune.torch"; '
        'python -c "import tensorrt"',
        'export AITUNE_CACHE_DIR and TRITON_CACHE_DIR to fresh writable directories per '
        'tuning job (outside the source tree)',
        'offline tuning job (its own Slurm allocation): run the aitune-inspect procedure on '
        'the FastWAM pipeline.model with inference_function calling the pipeline, wrap the '
        'chosen modules with strategy=OneBackendStrategy(...) (wrap has no tune_strategy '
        'keyword), tune with ignore_failing_modules=False, then save a .ait artifact and '
        'record compile_time_s',
        'patch (vla-patch-candidate): a new frontend option (added to registry _CAPABILITIES '
        'in the same patch) that wraps the same modules and loads the pinned .ait by '
        'absolute path + sha256 before FlashRT graph capture; never tune inside the worker',
        'campaign: controller init --snapshot-store ... --python <aitune venv python> so the '
        'baseline and the candidate share the aitune runtime',
    ],
    'risks': [
        'CUDA graphs: TensorRTBackend use_cuda_graphs=True captures inside the module call; '
        'inside FlashRT\'s FastWAM velocity graph or Hy ViT/prefix/velocity graphs that is '
        'nested capture: disable one side and gate graph replay with vla-operator-gate',
        'turning FlashRT graphs off to make room for AITune must be compared against the '
        'graph-on baseline, not against an eager baseline',
        'A800 (sm_80): TensorRT supports Ampere, but no TensorRT ships with FlashRT; FP8 '
        'presets need SM 8.9+ and MX/NVFP4 need SM 10.0+, so only fp32/fp16/bf16/int8 apply',
        'compile time is not in FlashRT steady-state latency: record compile_time_s and the '
        'break-even call count compile_time / (baseline_ms - tuned_ms) per call before '
        'accepting; a worker that re-tunes on every fresh process is not deployable',
        'tune(ignore_failing_modules=True) is the default and silently leaves failing modules '
        'in eager passthrough: set it False or check the log for every wrapped module',
        'tune needs at least 2 batch sizes to infer a batch axis; FlashRT requests are '
        'batch 1: pass fixed batch_sizes and static shapes, and reject shape-dependent engines '
        'when the workload shapes vary',
        'FastWAM _validate_model checks owners/dtypes/devices of video_expert/action_expert/'
        'mot/vae: wrapping after construction may need the patch to re-run or relax checks',
        'the .ait artifact and AITune caches are outside the snapshot source identity: record '
        'their path and sha256 in the PatchProposal provenance and treat a changed artifact '
        'as a new candidate',
        'upstream aitune-benchmark Step 2 passes tune_strategy= to ait.wrap, which wrap does '
        'not accept (strategy= / strategies=)',
        'JIT mode (AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning) patches modules globally and '
        'is not recorded in source identity (only FLASHRT_/FVK_/PI0_ env vars are): never '
        'use it in FlashRT workers',
        'AITune may retrieve external materials (upstream NOTICE AND DISCLAIMER): run tuning '
        'jobs without network egress where possible',
        'exactness is approximate: autocast fp16/bf16 and TensorRT tactics change numerics; '
        'the campaign quality gate decides, never AITune\'s own atol check',
    ],
}
