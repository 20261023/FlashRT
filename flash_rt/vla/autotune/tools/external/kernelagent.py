"""KernelAgent adapter: a self-describing Triton problem in, a verified kernel out.

meta-pytorch/KernelAgent (``TritonKernelAgent.generate_kernel``) samples kernel
seeds and refines them until one passes a test script; the kernel-writing LLM
sees only the test, so the generated ``test.py`` carries the whole contract in
its docstring and checks numerics against an fp32 reference, a second input
draw, CUDA-graph replay and every other captured shape. The first passing kernel
wins: generation does no timing (first-valid is not fastest); FlashRT's
controller measures speed.

This module writes ``problem.py``/``test.py``/``run_ka.py`` (``prepare_problem``),
renders a job (``render_job``) and reads the result (``harvest``). It never
imports KernelAgent, torch or triton.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from ..artifacts import KernelArtifact, StageCapture
from . import (
    fresh_dir, jsonable, quote, reference_source, require_absolute, sha256_file,
    template_contract,
)


UPSTREAM_COMMIT = 'e0647170da36ef9b059ac0bd3d60103aa4ed378b'
TOOL_ID = 'kernel.kernelagent'
ENTRY_SYMBOL = 'kernel_function'
METRICS_KEY = 'flashrt_kernel_test'
RESULT_SCHEMA = 'flash_rt.autotune.ka_result.v1'
A800_GPU_NAME_ALIAS = 'NVIDIA A100 SXM4 80GB'
MAX_ANTHROPIC_WORKERS = 3

SPEC = {
    'tool_id': TOOL_ID,
    'title': 'KernelAgent: LLM-generated Triton kernel verified by a FlashRT test',
    'lane': 'patch',
    'stages': ('kernel.gemm', 'kernel.attention', 'kernel.norm_elementwise', 'kernel.conv',
               'vision.encode', 'prefix.prefill', 'denoise.velocity'),
    'models': ('*',),
    'exactness': 'approximate',
    'status': 'needs_setup',
    'reason': (
        'Needs a separate KernelAgent venv (FlashRT\'s exact torch/triton), LLM '
        'credentials and egress from the GPU node. generate_kernel returns the first '
        'kernel that passes the test and does no timing: first-valid is not fastest. The '
        'hardware-guided optimization pipeline needs NCU counter permission (rounds are '
        f'skipped without it) and gpu_name="{A800_GPU_NAME_ALIAS}" on A800, which is '
        'missing from its GPU specs database. Prompts lean towards Hopper (TMA).'),
    'upstream': {
        'repo': 'https://github.com/meta-pytorch/KernelAgent',
        'commit': UPSTREAM_COMMIT,
        'license': 'Apache-2.0',
        'notice': False,
    },
    'skills': ('vla-kernel-candidate', 'vla-operator-gate'),
    'setup': [
        'git clone https://github.com/meta-pytorch/KernelAgent.git and git checkout '
        f'{UPSTREAM_COMMIT}',
        'create a venv with FlashRT\'s exact torch and triton wheels, then pip install '
        '--no-deps -e . and pip install openai anthropic jinja2 python-dotenv requests '
        'numpy omegaconf pandas',
        'always run with cwd=$KA_ROOT and PYTHONPATH=$KA_ROOT (utils/ and scripts/ are not '
        'packages); never import KernelAgent into a FlashRT worker (logging/dotenv side '
        'effects)',
        'credentials: ANTHROPIC_API_KEY for registered Claude ids (claude-sonnet-4-6, '
        'claude-opus-4-6) or OPENAI_API_KEY; other ids go to the relay at LLM_RELAY_URL',
        'GPU node with Triton sm_80 support and LLM egress (or a relay/proxy)',
        'optional optimization: ncu with counter permission (or '
        'KERNELAGENT_NCU_USE_SUDO=1) and OptimizationManager(gpu_name='
        f'"{A800_GPU_NAME_ALIAS}") on A800',
        'capture_calls(...) -> prepare_problem(...) -> render_job(...) -> harvest(...)',
    ],
    'risks': [
        'first valid kernel wins; speed is unmeasured until the FlashRT controller runs',
        'Anthropic provider: max_tokens is capped at 8192 (long kernels truncate) and seed '
        'temperatures 0.8 + 0.1 i exceed 1.0 from the 4th seed, which silently swaps in mock '
        'kernels: keep workers <= 3',
        'the kernel LLM sees only test.py, so the spec is the oracle; KernelAgent\'s regex '
        'anti-cheat misses @, torch.addmm and method forms, so a human reviews kernel.py',
        'its NCU roofline reads the wrong tensor-core metric (uses_tensor_cores is always '
        'False), misleading GEMM diagnosis in the optimization pipeline',
        'optimization timing uses do_bench per call (launch latency included) while FlashRT '
        'replays CUDA graphs: small-kernel rankings can disagree',
        'generated code runs with the full environment, API keys included; use scoped keys',
        'cuBLAS is hard to beat on plain GEMMs; fusions are the realistic target',
        'LLM sampling is non-deterministic: persist session_dir and hashes',
    ],
}

_MODEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:/@-]*$')
_COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')


def _tolerance(value, name, *, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            value) or not 0 < value <= upper:
        raise ValueError(f'{name} must be in (0, {upper}], got {value!r}')
    return float(value)


def _one_line(text):
    return ' '.join(str(text).split())


def _contract(capture):
    inputs, outputs = template_contract(capture)
    if len(outputs) != 1:
        raise ValueError(f'KernelAgent problems return one tensor; the capture has '
                         f'{len(outputs)} outputs')
    if capture.tensors_path is None:
        raise ValueError('KernelAgent problems load real tensors: capture.tensors_path is '
                         'required (capture_calls with out_dir saves them)')
    path = Path(capture.tensors_path)
    if not path.is_absolute() or path.suffix not in ('.pt', '.npz'):
        raise ValueError(f'capture.tensors_path must be an absolute .pt or .npz path, got '
                         f'{capture.tensors_path!r}')
    for entry in inputs + outputs:
        if entry['dtype'] not in ('bfloat16', 'float16', 'float32'):
            raise ValueError(f'{entry["name"]} is {entry["dtype"]}; KernelAgent templates '
                             f'need floating tensors')
    return inputs, outputs[0]


def _input_rows(capture, inputs, shape_index=0):
    return [(entry['name'], capture.concrete_shape(entry, shape_index), entry['dtype'],
             entry['role'])
            for entry in inputs]


def _output_row(capture, output, shape_index=0):
    return (output['name'], capture.concrete_shape(output, shape_index), output['dtype'])


def _shape_rows(capture, inputs, output):
    """test.py SHAPES: every captured shape with its concrete inputs and output."""
    return [{'name': row['name'], 'inputs': _input_rows(capture, inputs, index),
             'output': _output_row(capture, output, index)}
            for index, row in enumerate(capture.shapes)]


def _gqa(capture):
    return capture.op_kind == 'attention' and any(
        row['axes']['H'] != row['axes']['HKV'] for row in capture.shapes)


def _semantics(capture, inputs, output):
    names = [entry['name'] for entry in inputs]
    out = output['name']
    if capture.op_kind == 'linear':
        bias = f' + {names[2]}[n]' if len(names) == 3 else ''
        return [
            f'{out}[m, n] = sum_k {names[0]}[m, k] * {names[1]}[n, k]{bias}',
            f'{names[1]} is stored [N, K] (the torch F.linear layout): read it as stored, '
            f'never transpose-copy it.',
            'Accumulate in float32' + (', add the bias in float32' if bias else '')
            + f', round to {output["dtype"]} once at the end.',
        ]
    if capture.op_kind == 'attention':
        q, k, v = names
        return [
            f'{out}[b, h, s, :] = softmax_j(scale * {q}[b, h, s, :] . {k}[b, g, j, :]) @ '
            f'{v}[b, g, j, :] with scale = 1/sqrt(D) and g = h // (H // HKV).',
            'Layout [B, heads, sequence, head_dim]; no mask (non-causal), no dropout.',
            f'Softmax statistics and both products in float32; round to {output["dtype"]} '
            f'once.',
        ]
    return [
        'Exactly reference_native(...) below (the FlashRT reference embedded in this file); '
        'the float32 oracle runs the same code on float32 copies of the inputs.',
    ]


def _spec_docstring(capture, inputs, output, *, rel_l2, cosine):
    rows = _input_rows(capture, inputs)
    params = ', '.join(name for name, _, _, _ in rows)
    first_name, _, first_dtype, _ = rows[0]
    axes = capture.shapes[0]['axes']
    lines = [
        f'FlashRT kernel specification: {capture.model} {capture.stage_id} '
        f'({capture.op_kind}), captured at {_one_line(capture.call_site)}.',
        '',
        'This file is the complete specification. Write kernel.py defining',
        '',
        f'    kernel_function({params}) -> {output["name"]}',
        '',
        'Signature (the captured call; dtypes are exact, inputs are contiguous):',
    ]
    lines += [f'    {name}: {shape} {dtype} ({role})' for name, shape, dtype, role in rows]
    lines += [
        f'    returns {output["name"]}: {capture.concrete_shape(output)} {output["dtype"]}, '
        f'a new tensor allocated with torch.empty on every call',
        '    axes: ' + ', '.join(f'{axis}={size}' for axis, size in axes.items()),
    ]
    if len(capture.shapes) > 1:
        lines.append('Other production shapes (one kernel must serve all of them):')
        lines += ['    ' + ', '.join(f'{axis}={size}' for axis, size in row['axes'].items())
                  for row in capture.shapes[1:]]
    lines += ['', 'Semantics:']
    lines += [f'    {line}' for line in _semantics(capture, inputs, output)]
    lines += [
        '',
        'Implementation rules:',
        '    - Triton only: @triton.jit kernels launched from kernel_function. No torch '
        'compute in kernel.py (no torch.nn / F.*, torch.matmul/mm/bmm/addmm, @, einsum); '
        'torch only allocates the output and reads shapes/strides.',
        f'    - kernel_function starts with: assert {first_name}.dtype == torch.{first_dtype}',
        f'    - Load {first_dtype} operands, accumulate in float32 (tl.dot with a float32 '
        f'accumulator), store {output["dtype"]}.',
        '    - Target NVIDIA A800/A100 (sm_80): no TMA, no tensor descriptors '
        '(tl.make_tensor_descriptor / TensorDescriptor), no warp_specialize, no FP8, '
        'nothing Hopper/Blackwell-only; tl.dot on bf16/fp16 operands lowers to mma.sync.',
        '    - CUDA-graph capturable: no .item(), .cpu(), .tolist(), torch.cuda.synchronize(), '
        'no host reads of device data and no data-dependent Python branching; allocate '
        'the output with torch.empty inside every call; never cache results across calls.',
        '    - At most 4 @triton.autotune configs, keyed only on shapes (autotuning finishes '
        'during warmup, before graph capture).',
        '    - Never modify the inputs; never return an input.',
        '',
        'Checks (all must pass; the script prints one JSON metrics line, then PASS or FAIL, '
        'and exits 0 or 1):',
        '    1. Output shape, dtype and device exact; all values finite; no input modified.',
        f'    2. Relative L2 error vs a float32 reference <= max({rel_l2:g}, 2 x the relative '
        f'L2 error of the native {first_dtype} torch path); cosine similarity >= '
        f'min({cosine:g}, 1 - 2 x (1 - native cosine)).',
        '    3. The same checks on a second, different input draw of the same shapes.',
        '    4. CUDA graph: 3 warmup calls on a side stream, capture one call, copy new '
        'values into the static inputs, replay, and repeat check 2 on the replayed output.',
        '    5. Checks 1 and 2 at every other production shape, on fresh inputs drawn with '
        'the mean and standard deviation of the captured tensor of the same name.',
    ]
    return '\n'.join(lines)


def _loader_lines():
    return [
        'def _load_inputs(device):',
        '    """Captured FlashRT tensors, reshaped to the signature, in signature order."""',
        '    if TENSORS.endswith(".npz"):',
        '        import numpy as np',
        '        with np.load(TENSORS, allow_pickle=False) as data:',
        '            raw = {row[0]: torch.from_numpy(data[row[0]]) for row in INPUTS}',
        '    else:',
        '        raw = torch.load(TENSORS, map_location="cpu", weights_only=True)',
        '    tensors = []',
        '    for name, shape, dtype, _role in INPUTS:',
        '        tensor = raw[name].reshape(shape).to(getattr(torch, dtype)).contiguous()',
        '        tensors.append(tensor if device is None else tensor.to(device))',
        '    return tensors',
    ]


def _reference_lines(capture, inputs):
    params = ', '.join(entry['name'] for entry in inputs)
    floats = ', '.join(f'{entry["name"]}.float()' for entry in inputs)
    gqa = ', enable_gqa=True' if _gqa(capture) else ''
    if capture.op_kind == 'linear':
        if len(inputs) == 3:
            x, weight, bias = (entry['name'] for entry in inputs)
            fp32 = f'torch.addmm({bias}.float(), {x}.float(), {weight}.float().t())'
        else:
            x, weight = (entry['name'] for entry in inputs)
            fp32 = f'{x}.float() @ {weight}.float().t()'
        return [
            f'def reference_fp32({params}):',
            f'    return {fp32}',
            '', '',
            f'def reference_native({params}):',
            f'    return F.linear({params})',
        ]
    if capture.op_kind == 'attention':
        return [
            f'def reference_fp32({params}):',
            f'    return F.scaled_dot_product_attention({floats}{gqa})',
            '', '',
            f'def reference_native({params}):',
            f'    return F.scaled_dot_product_attention({params}{gqa})',
        ]
    return [
        f'_REFERENCE_SOURCE = {reference_source(capture)!r}',
        '_REFERENCE = {}',
        'exec(compile(_REFERENCE_SOURCE, "<flashrt-reference>", "exec"), _REFERENCE)',
        '', '',
        f'def reference_fp32({params}):',
        f'    return _REFERENCE["run"]({floats})',
        '', '',
        f'def reference_native({params}):',
        f'    return _REFERENCE["run"]({params})',
    ]


def problem_text(capture, *, rel_l2=2e-3, cosine=0.99999) -> str:
    """KernelBench-style problem.py: parameter-free Model, all tensors from get_inputs()."""
    inputs, output = _contract(capture)
    params = ', '.join(entry['name'] for entry in inputs)
    doc = _spec_docstring(capture, inputs, output, rel_l2=rel_l2, cosine=cosine)
    lines = [
        '"""' + doc.replace('\\', '\\\\').replace('"""', '\\"\\"\\"'), '"""', '',
        'import torch',
        'import torch.nn as nn',
        'import torch.nn.functional as F',
        '',
        f'TENSORS = {capture.tensors_path!r}',
        f'INPUTS = {_input_rows(capture, inputs)!r}',
        '', '',
    ]
    lines += _reference_lines(capture, inputs)
    lines += ['', ''] + _loader_lines() + ['', '']
    lines += [
        'class Model(nn.Module):',
        '    """Parameter-free: weights arrive as inputs from get_inputs()."""',
        '',
        f'    def forward(self, {params}):',
        f'        return reference_native({params})',
        '', '',
        'def get_inputs():',
        '    return _load_inputs(None)',
        '', '',
        'def get_init_inputs():',
        '    return []',
    ]
    return '\n'.join(lines) + '\n'


_TEST_BODY = '''
def _metrics(out, ref):
    out64, ref64 = out.double().flatten(), ref.double().flatten()
    diff = out64 - ref64
    ref_norm = float(torch.linalg.vector_norm(ref64))
    out_norm = float(torch.linalg.vector_norm(out64))
    rel_l2 = float(torch.linalg.vector_norm(diff)) / max(ref_norm, 1e-300)
    cosine = float(torch.dot(out64, ref64)) / max(out_norm * ref_norm, 1e-300)
    return {"rel_l2": rel_l2, "cosine": cosine, "max_abs": float(diff.abs().max())}


def _fresh_like(tensor, generator, shape=None):
    """A different draw with the same dtype, mean and scale (and shape unless given)."""
    values = tensor.float()
    mean = float(values.mean())
    std = float(values.std()) if values.numel() > 1 else 1.0
    if not math.isfinite(std) or std == 0.0:
        std = 1.0
    if not math.isfinite(mean):
        mean = 0.0
    draw = torch.randn(values.shape if shape is None else shape, generator=generator,
                       device=values.device)
    return (draw * std + mean).to(tensor.dtype).contiguous()


def _check(label, out, inputs, failures, report, output=OUTPUT):
    name, shape, dtype = output
    if not isinstance(out, torch.Tensor):
        failures.append(f"{label}: kernel_function returned {type(out).__name__}, not a tensor")
        return
    if list(out.shape) != shape:
        failures.append(f"{label}: {name} shape {list(out.shape)} != {shape}")
        return
    if out.dtype != getattr(torch, dtype):
        failures.append(f"{label}: {name} dtype {out.dtype} != torch.{dtype}")
        return
    if out.device != inputs[0].device:
        failures.append(f"{label}: {name} on {out.device}, inputs on {inputs[0].device}")
        return
    if any(out.data_ptr() == tensor.data_ptr() for tensor in inputs):
        failures.append(f"{label}: {name} aliases an input")
        return
    if not bool(torch.isfinite(out).all()):
        failures.append(f"{label}: {name} has non-finite values")
        return
    reference = reference_fp32(*inputs)
    native = _metrics(reference_native(*inputs), reference)
    metrics = _metrics(out, reference)
    rel_l2_tol = max(REL_L2_TOL, 2.0 * native["rel_l2"])
    cosine_min = min(COSINE_MIN, 1.0 - 2.0 * (1.0 - native["cosine"]))
    metrics.update({"native_rel_l2": native["rel_l2"], "native_cosine": native["cosine"],
                    "rel_l2_tolerance": rel_l2_tol, "cosine_min": cosine_min})
    report[label] = metrics
    if not metrics["rel_l2"] <= rel_l2_tol:
        failures.append(f"{label}: rel_l2 {metrics['rel_l2']:.3e} > {rel_l2_tol:.3e}")
    if not metrics["cosine"] >= cosine_min:
        failures.append(f"{label}: cosine {metrics['cosine']:.9f} < {cosine_min:.9f}")


def _call(label, inputs, failures, report, output=OUTPUT):
    before = [tensor.clone() for tensor in inputs]
    out = kernel_function(*inputs)
    torch.cuda.synchronize()
    if any(not torch.equal(a, b) for a, b in zip(before, inputs)):
        failures.append(f"{label}: kernel_function modified its inputs")
    _check(label, out, inputs, failures, report, output)


def _shape_check(index, row, captured, generator, failures, report):
    """Checks 1-2 at another captured shape, on draws with the captured statistics."""
    by_name = {spec[0]: tensor for spec, tensor in zip(INPUTS, captured)}
    inputs = [_fresh_like(by_name[name], generator, shape)
              for name, shape, _dtype, _role in row["inputs"]]
    _call(f"shape_{index}_{row['name']}", inputs, failures, report, row["output"])


def _graph_check(inputs, generator, failures, report):
    static = [tensor.clone() for tensor in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            kernel_function(*static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            static_out = kernel_function(*static)
    except Exception as error:  # report capture failures as a failed check
        failures.append(f"cuda_graph: capture failed: {type(error).__name__}: {error}")
        return
    for tensor, row in zip(static, INPUTS):
        if row[3] == "input":
            tensor.copy_(_fresh_like(tensor, generator))
    graph.replay()
    torch.cuda.synchronize()
    _check("cuda_graph_replay", static_out, [tensor.clone() for tensor in static], failures,
           report)


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA is required")
        print("FAIL")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    failures, report, verified = [], {}, []
    inputs = _load_inputs(device)
    _call("captured", inputs, failures, report)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260)
    second = [_fresh_like(tensor, generator) for tensor in inputs]
    _call("second_draw", second, failures, report)
    _graph_check(inputs, generator, failures, report)
    if not failures:
        verified.append(SHAPES[0]["name"])
    for index, row in enumerate(SHAPES[1:], start=1):
        before = len(failures)
        _shape_check(index, row, inputs, generator, failures, report)
        if len(failures) == before:
            verified.append(row["name"])
    print(json.dumps({METRICS_KEY: {"ok": not failures, "failures": failures,
                                    "metrics": report,
                                    "shapes_total": [row["name"] for row in SHAPES],
                                    "shapes_verified": verified}}, sort_keys=True))
    for failure in failures:
        print(f"FAIL: {failure}")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def test_text(capture, *, rel_l2=2e-3, cosine=0.99999) -> str:
    """Self-contained test.py: the full spec in its docstring, no import of problem.py."""
    inputs, output = _contract(capture)
    rel_l2 = _tolerance(rel_l2, 'rel_l2', upper=1.0)
    cosine = _tolerance(cosine, 'cosine', upper=1.0)
    doc = _spec_docstring(capture, inputs, output, rel_l2=rel_l2, cosine=cosine)
    lines = [
        '"""' + doc.replace('\\', '\\\\').replace('"""', '\\"\\"\\"'), '"""', '',
        'import json',
        'import math',
        'import sys',
        '',
        'import torch',
        'import torch.nn.functional as F',
        '',
        'from kernel import kernel_function',
        '',
        f'TENSORS = {capture.tensors_path!r}',
        f'INPUTS = {_input_rows(capture, inputs)!r}',
        f'OUTPUT = {_output_row(capture, output)!r}',
        f'SHAPES = {_shape_rows(capture, inputs, output)!r}',
        f'REL_L2_TOL = {rel_l2!r}',
        f'COSINE_MIN = {cosine!r}',
        'GRAPH_WARMUP = 3',
        f'METRICS_KEY = {METRICS_KEY!r}',
        '', '',
    ]
    lines += _reference_lines(capture, inputs)
    lines += ['', ''] + _loader_lines() + ['']
    return '\n'.join(lines) + _TEST_BODY


_RUNNER = '''"""Run KernelAgent on one FlashRT stage directory (generated by FlashRT).

Run from the KernelAgent clone, in its venv, on a GPU node with LLM egress:
    cd "$KA_ROOT" && PYTHONPATH="$KA_ROOT" python <stage>/run_ka.py --stage-dir <stage>
Writes kernel.py (on success) and ka_result.json, then re-runs test.py on the winner.
Exit 0: a kernel passed KernelAgent and the re-test; 3: no verified kernel; 2: usage.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

PINNED_COMMIT = __PINNED__
RESULT_SCHEMA = __SCHEMA__
METRICS_KEY = __METRICS__
STAGE_DIR = __STAGE__
MAX_ANTHROPIC_WORKERS = __MAXW__


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and len(head) == 40 else None


def _write_result(stage, result):
    text = json.dumps(result, indent=2, sort_keys=True, default=str) + "\\n"
    (stage / "ka_result.json").write_text(text, encoding="utf-8")


def _retest(stage, timeout_s):
    try:
        proc = subprocess.run([sys.executable, str(stage / "test.py")], cwd=str(stage),
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "passed": False, "metrics": None,
                "error": f"test.py timed out after {timeout_s} s"}
    log = proc.stdout + ("\\n--- stderr ---\\n" + proc.stderr if proc.stderr else "")
    (stage / "test_output.txt").write_text(log, encoding="utf-8")
    metrics = None
    for line in proc.stdout.splitlines():
        if line.startswith("{") and METRICS_KEY in line:
            try:
                metrics = json.loads(line)[METRICS_KEY]
            except (ValueError, KeyError):
                metrics = None
    passed = proc.returncode == 0 and "PASS" in proc.stdout.split()
    return {"returncode": proc.returncode, "passed": passed, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", default=STAGE_DIR)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--test-timeout-s", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1 or args.rounds < 1:
        parser.error("--workers and --rounds must be positive")
    if args.model.startswith("claude") and args.workers > MAX_ANTHROPIC_WORKERS:
        parser.error("Anthropic models: --workers must be <= 3 (seed temperatures above 1.0 "
                     "make KernelAgent fall back to mock kernels)")
    stage = pathlib.Path(args.stage_dir).resolve()
    for name in ("kernel.py", "ka_result.json", "test_output.txt"):
        (stage / name).unlink(missing_ok=True)
    result = {
        "schema": RESULT_SCHEMA, "success": False, "model": args.model,
        "workers": args.workers, "max_rounds": args.rounds,
        "upstream_commit": _git_head(), "upstream_commit_pinned": PINNED_COMMIT,
        "stage_dir": str(stage), "test_sha256": _sha256(stage / "test.py"),
        "problem_sha256": _sha256(stage / "problem.py"),
    }
    from utils.providers import is_model_available
    if not is_model_available(args.model):
        result["error"] = f"model {args.model!r} is not available (credentials or registry)"
        _write_result(stage, result)
        return 3
    from triton_kernel_agent import TritonKernelAgent
    agent = TritonKernelAgent(num_workers=args.workers, max_rounds=args.rounds,
                              log_dir=str(stage / "ka_logs"), model_name=args.model,
                              test_timeout_s=args.test_timeout_s)
    try:
        res = agent.generate_kernel(
            problem_description=(stage / "problem.py").read_text(encoding="utf-8"),
            test_code=(stage / "test.py").read_text(encoding="utf-8"),
            generate_default_test=False)
    finally:
        agent.cleanup()
    result.update({key: value for key, value in res.items() if key != "kernel_code"})
    result["success"] = bool(res.get("success"))
    result["session_dir"] = res.get("session_dir")
    if result["success"]:
        code = res["kernel_code"].encode("utf-8")
        (stage / "kernel.py").write_bytes(code)
        result["kernel_sha256"] = hashlib.sha256(code).hexdigest()
        result["retest"] = _retest(stage, args.test_timeout_s)
    _write_result(stage, result)
    return 0 if result["success"] and result["retest"]["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
'''


def runner_text(stage_dir) -> str:
    """run_ka.py (report §5.3) with this stage directory baked in as the default."""
    stage_dir = require_absolute(stage_dir, 'stage_dir')
    return (_RUNNER.replace('__PINNED__', repr(UPSTREAM_COMMIT))
            .replace('__SCHEMA__', repr(RESULT_SCHEMA))
            .replace('__METRICS__', repr(METRICS_KEY))
            .replace('__STAGE__', repr(stage_dir))
            .replace('__MAXW__', repr(MAX_ANTHROPIC_WORKERS)))


def prepare_problem(capture, out_dir, *, rel_l2=2e-3, cosine=0.99999) -> Path:
    """Write problem.py, test.py and run_ka.py into a new stage directory."""
    if not isinstance(capture, StageCapture):
        raise TypeError('capture must be a StageCapture')
    stage = Path(out_dir).resolve()
    texts = {
        'problem.py': problem_text(capture, rel_l2=rel_l2, cosine=cosine),
        'test.py': test_text(capture, rel_l2=rel_l2, cosine=cosine),
        'run_ka.py': runner_text(str(stage)),
    }
    for name, text in texts.items():
        compile(text, name, 'exec')
    fresh_dir(stage, 'stage directory')
    for name, text in texts.items():
        (stage / name).write_text(text, encoding='utf-8')
    return stage


def render_job(stage_dir, *, kernelagent_root, model='claude-sonnet-4-6', workers=3,
               rounds=8, env_file=None) -> str:
    """Bash running run_ka.py from the KernelAgent clone (never executed by FlashRT)."""
    stage_dir = require_absolute(stage_dir, 'stage_dir')
    kernelagent_root = require_absolute(kernelagent_root, 'kernelagent_root')
    if env_file is not None:
        env_file = require_absolute(env_file, 'env_file')
    if not isinstance(model, str) or not _MODEL_RE.match(model):
        raise ValueError(f'model must be a model id, got {model!r}')
    for name, value in (('workers', workers), ('rounds', rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive int, got {value!r}')
    if model.startswith('claude') and workers > MAX_ANTHROPIC_WORKERS:
        raise ValueError(f'Anthropic models need workers <= {MAX_ANTHROPIC_WORKERS}: seed '
                         f'temperatures above 1.0 make KernelAgent fall back to mock kernels')
    if not (Path(stage_dir) / 'run_ka.py').is_file():
        raise ValueError(f'{stage_dir} has no run_ka.py; run prepare_problem first')
    stage = quote(stage_dir)
    lines = [
        '#!/usr/bin/env bash',
        '# KernelAgent generation for one FlashRT stage; FlashRT never executes this script.',
        f'# Pinned meta-pytorch/KernelAgent {UPSTREAM_COMMIT}. Needs a GPU, the KernelAgent',
        '# venv (FlashRT\'s torch/triton) and LLM credentials with egress (or LLM_RELAY_URL).',
        'set -euo pipefail',
    ]
    if env_file is not None:
        lines.append(f'source {quote(env_file)}')
    lines += [
        f'cd {quote(kernelagent_root)}',
        f'export PYTHONPATH={quote(kernelagent_root)}',
        'export KA_PROCESS_USE_SYS_EXECUTABLE=1',
        f'export TRITON_CACHE_DIR={quote(str(Path(stage_dir) / "triton_cache"))}',
        f'if [ "$(git rev-parse HEAD)" != {UPSTREAM_COMMIT} ]; then '
        f'echo "warning: KernelAgent HEAD differs from {UPSTREAM_COMMIT}" >&2; fi',
        'set +e',
        f'python {quote(str(Path(stage_dir) / "run_ka.py"))} --stage-dir {stage} '
        f'--model {quote(model)} --workers {workers} --rounds {rounds}',
        'RC=$?',
        'set -e',
        f'echo "KA_RC=$RC STAGE="{stage}',
        'exit $RC',
    ]
    return '\n'.join(lines) + '\n'


def _metrics_from_output(path):
    """The test's metrics JSON line from test_output.txt, or None."""
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        if line.startswith('{') and METRICS_KEY in line:
            try:
                return json.loads(line)[METRICS_KEY]
            except (ValueError, KeyError, TypeError):
                return None
    return None


def _shape_checks(metrics, reasons, notes):
    total, verified = metrics.get('shapes_total'), metrics.get('shapes_verified')
    if not isinstance(total, list) or not isinstance(verified, list):
        notes.append('test metrics list no shapes_total/shapes_verified: only the first '
                     'captured shape is known to be verified')
        return
    missing = [name for name in total if name not in verified]
    if not total:
        reasons.append('test metrics shapes_total is empty')
    elif missing:
        reasons.append(f'test.py did not verify captured shapes {missing} '
                       f'(verified {verified})')


def harvest(stage_dir, *, stage_id, model) -> KernelArtifact:
    """Read a stage directory after run_ka.py (read-only).

    Accepted only if ka_result.json reports success, kernel.py exists and its
    sha256 equals ka_result's kernel_sha256, test.py is unchanged since the run,
    the runner's re-test (when recorded) passed, and the test metrics (when they
    list shapes) verified every captured shape.
    """
    stage = Path(stage_dir).resolve()
    reasons, notes = [], []
    result = None
    try:
        result = json.loads((stage / 'ka_result.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        reasons.append('ka_result.json missing')
    except (OSError, ValueError) as error:
        reasons.append(f'ka_result.json unreadable: {error}')
    if result is not None and not isinstance(result, dict):
        reasons.append('ka_result.json is not an object')
        result = None
    if result is not None and result.get('success') is not True:
        detail = result.get('error') or result.get('message') or 'no detail'
        reasons.append(f'KernelAgent reported no success: {detail}')
    kernel = stage / 'kernel.py'
    kernel_path = kernel_sha256 = language = None
    if kernel.is_file():
        kernel_path, kernel_sha256, language = str(kernel), sha256_file(kernel), 'triton'
        expected = None if result is None else result.get('kernel_sha256')
        if result is not None and expected != kernel_sha256:
            reasons.append(f'kernel.py sha256 {kernel_sha256} != ka_result kernel_sha256 '
                           f'{expected!r}')
    else:
        reasons.append('kernel.py missing')
    test = stage / 'test.py'
    if result is not None and result.get('test_sha256') is not None:
        if not test.is_file() or sha256_file(test) != result['test_sha256']:
            reasons.append('test.py changed since KernelAgent ran')
    retest = None if result is None else result.get('retest')
    metrics = None
    if isinstance(retest, dict):
        if retest.get('passed') is not True:
            reasons.append(f're-test of kernel.py failed (returncode '
                           f'{retest.get("returncode")!r})')
        metrics = retest.get('metrics')
    elif result is not None and result.get('success') is True:
        notes.append('no re-test recorded in ka_result.json')
    if metrics is None:
        metrics = _metrics_from_output(stage / 'test_output.txt')
    if isinstance(metrics, dict):
        _shape_checks(metrics, reasons, notes)
    run_commit = None if result is None else result.get('upstream_commit')
    if isinstance(run_commit, str) and _COMMIT_RE.match(run_commit):
        upstream_commit, commit_source = run_commit, 'ka_result upstream_commit'
        if run_commit != UPSTREAM_COMMIT:
            notes.append(f'run used KernelAgent {run_commit}, pinned {UPSTREAM_COMMIT}')
    else:
        upstream_commit, commit_source = UPSTREAM_COMMIT, 'SPEC pin (ka_result had none)'
    provenance = {'stage_dir': str(stage), 'pinned_commit': UPSTREAM_COMMIT,
                  'upstream_commit_source': commit_source, 'notes': notes,
                  'first_valid_not_fastest': True}
    if result is not None:
        provenance['ka_result'] = {key: result[key] for key in (
            'model', 'workers', 'max_rounds', 'rounds', 'worker_id', 'session_dir',
            'test_sha256', 'problem_sha256', 'message', 'error') if key in result}
    return KernelArtifact(
        tool_id=TOOL_ID, upstream_commit=upstream_commit, stage_id=stage_id, model=model,
        language=language, entry_symbol=ENTRY_SYMBOL, kernel_path=kernel_path,
        kernel_sha256=kernel_sha256,
        op_metrics=jsonable(metrics) if isinstance(metrics, dict) else {},
        accepted=not reasons, reasons=reasons, provenance=jsonable(provenance))
