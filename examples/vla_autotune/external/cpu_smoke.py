#!/usr/bin/env python3
"""CPU-only external installation probes; never authenticate or execute a model."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forgekernel(source):
    """Exercise the five patches without importing CUDA, Torch, or the agent."""
    source = Path(source).resolve()
    for language in ('cuda', 'cutedsl'):
        assert (source / f'hw_kb/{language}/sm80/index.md').is_file()
    tree = ast.parse((source / 'src/forgekernel/taskcard.py').read_text())
    arch = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'CUTE_ARCH' for t in n.targets))
    assert arch['sm80'] == 'sm_80'
    # Only the pure flag parser is compiled. Importing the full evaluator would
    # pull Torch, optional precision packages and GPU-related module side effects.
    tree = ast.parse((source / 'src/forgekernel/card_template/eval/leaderboard.py').read_text())
    names = {'CUDA_DEFAULT_CFLAGS', 'CUDA_GENCODE', 'CUDA_FLAGS_LINE', 'CUDA_FLAG_OK'}
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in n.targets))
        or isinstance(n, ast.FunctionDef) and n.name == 'cuda_kernel_flags']
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<cuda-flags>', 'exec'), namespace)
    assert namespace['cuda_kernel_flags']('__global__ void run() {}')[-1] == (
        '-gencode=arch=compute_80,code=sm_80')
    assert namespace['cuda_kernel_flags']('// fk-cuda-cflags: -arch=sm_90\n')[-1] == '-arch=sm_90'
    sass = _module('_external_sm80_sass', source / 'skills/sass/sass_tools.py')
    with contextlib.redirect_stdout(io.StringIO()):
        for opcode in ('HMMA.16816.F32', 'LDGSTS.E.BYPASS.128', 'LDSM.16.M88.4'):
            text = '\n'.join(f'/*{i * 16:04x}*/ {opcode} R0, R1, R2;' for i in range(600))
            assert sass.analyze(text, 'target', [('rng', 1), ('target', 10)])
        assert not sass.analyze('/*0000*/ HMMA.16816.F32 R0, R1, R2;',
                                'target', [('target', 10)])
        assert not sass.analyze(text, 'rng', [('target', 10), ('rng', 1)])
    timeline = _module('_external_sm80_timeline', source / 'skills/timeline/pm_timeline_run.py')
    stall = _module('_external_sm80_stall', source / 'skills/stall/stall_price_run.py')
    for interval in (128, 2048, 8192, 20000, 32768, 65536):
        for command in (timeline.build_cmd(interval), stall.build_cmd(True, interval),
                        stall.build_pm_cmd(interval)):
            actual = int(command[command.index('--pm-sampling-interval') + 1])
            assert actual == max(20000, interval)
        for samples in (0, 1, 32, 64, 128):
            finer = stall.finer_interval(interval, samples, 64)
            assert finer is None or 20000 <= finer < interval
    return {'status': 'CPU_PASS', 'checked': ['hw_kb', 'CUTE_ARCH', 'CUDA_GENCODE',
                                             'SASS_positive_and_negative', 'PM_floor_and_retry']}


def kernelagent(source):
    """Import real APIs and check the A800 alias, without creating any agent."""
    import inspect
    sys.path.insert(0, str(Path(source).resolve()))
    from triton_kernel_agent import TritonKernelAgent  # noqa: F401
    from triton_kernel_agent.opt_manager import OptimizationManager  # noqa: F401
    from triton_kernel_agent.opt_worker import OptimizationWorker
    from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import get_gpu_specs
    specs = get_gpu_specs('NVIDIA A100 SXM4 80GB')
    assert specs and specs['name'] == 'NVIDIA A100 SXM4 80GB'
    assert 'gpu_name' in inspect.signature(OptimizationWorker.__init__).parameters
    return {'status': 'CPU_IMPORT_PASS', 'gpu_name': specs['name'],
            'optimization_worker_accepts_gpu_name': True}


def aitune(source, *, cu128=False):
    """Check real API import/signatures; do not inspect, wrap, or tune a module."""
    import inspect
    import aitune.torch as ait
    assert Path(ait.__file__).resolve().is_relative_to(Path(source).resolve())
    assert 'strategy' in inspect.signature(ait.wrap).parameters
    assert 'tune_strategy' not in inspect.signature(ait.wrap).parameters
    assert 'ignore_failing_modules' in inspect.signature(ait.tune).parameters
    result = {'status': 'CPU_IMPORT_PASS', 'api': ['inspect', 'wrap', 'tune', 'save', 'load']}
    if cu128:
        result.update(_aitune_cu128_versions(ait))
    return result


def _aitune_cu128_versions(ait):
    import importlib.metadata as metadata
    import site

    import cupy
    import modelopt
    import nvtx
    import onnxruntime
    import tensorrt
    import torch
    import torchao
    import triton

    versions = {name: metadata.version(name) for name in (
        'aitune', 'torch', 'triton', 'numpy', 'nvtx', 'nvidia-modelopt',
        'tensorrt', 'tensorrt-cu12', 'tensorrt-cu12-bindings', 'tensorrt-cu12-libs',
        'torch-tensorrt', 'torchao', 'cupy-cuda12x', 'onnxruntime-gpu')}
    installed = [dist.metadata['Name'].lower().replace('_', '-')
                 for dist in metadata.distributions()]
    cuda13 = [name for name in installed if 'cu13' in name or 'cuda13' in name]
    assert not cuda13, f'CUDA13 packages in CUDA12 environment: {cuda13}'
    assert torch.version.cuda == '12.8'
    assert versions['torch'] == '2.11.0+cu128'
    assert versions['triton'] == '3.6.0'
    assert versions['nvidia-modelopt'] == '0.43.0'
    assert versions['cupy-cuda12x'] == '13.6.0'
    assert versions['tensorrt-cu12-libs'] == '10.15.1.29'
    assert tensorrt.__version__ == '10.15.1.29'
    assert versions['torch-tensorrt'] == '2.11.0+cu128'
    assert not site.ENABLE_USER_SITE
    assert 'include-system-site-packages = false' in (Path(sys.prefix) / 'pyvenv.cfg').read_text()
    for module in (torch, triton, tensorrt, torchao, modelopt, cupy, nvtx, onnxruntime):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    from aitune.torch.backend import torch_tensorrt_aot_backend
    trt_torch = torch_tensorrt_aot_backend.torch_tensorrt
    return {'versions': versions, 'torch_compiled_cuda': torch.version.cuda,
            'tensorrt_runtime': tensorrt.__version__,
            'torch_tensorrt_cpu_import': 'PASS' if trt_torch is not None else 'BACKEND_UNAVAILABLE',
            'cuda13_packages': cuda13, 'user_site_enabled': site.ENABLE_USER_SITE,
            'python': sys.version.split()[0],
            'prefix': sys.prefix, 'torch_path': torch.__file__,
            'aitune_path': ait.__file__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', choices=('forgekernel', 'kernelagent', 'aitune'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--aitune-cu128', action='store_true',
                        help='require the clean Python 3.11 / Torch 2.11 CU128 wheel profile')
    args = parser.parse_args(argv)
    if args.aitune_cu128 and args.tool != 'aitune':
        parser.error('--aitune-cu128 requires --tool aitune')
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ.pop('AUTOWRAPT_BOOTSTRAP', None)
    kwargs = {'cu128': True} if args.aitune_cu128 else {}
    result = globals()[args.tool](args.source, **kwargs)
    result.update({'authentication_verified': False, 'gpu_validation': 'NOT_RUN',
                   'functional_validation': 'NOT_RUN'})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
