#!/usr/bin/env python3
"""Build ONLY the standalone host-C++ GEMM plan extension, no Torch import.

Run real compilation in the main session's allocated environment after loading
gcc/11.4.0 and cuda/12.8. --dry-run is safe for CPU inspection/tests. Output is
an explicitly selected task-owned package directory, never inferred from an
installed flash_rt environment. No nvcc, Torch C++ ABI or CMake is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
import tempfile


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=True, type=Path,
                        help="Explicit task-source flash_rt package directory")
    result.add_argument("--cuda-home", type=Path,
                        default=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"))
    result.add_argument("--torch-include", type=Path,
                        help="Directory containing pybind11/pybind11.h; discovered without importing Torch by default")
    result.add_argument("--cxx", default=os.environ.get("CXX", "g++"),
                        help="Host compiler executable, normally module gcc/11.4.0's g++")
    result.add_argument("--dry-run", action="store_true", help="Print JSON build plan, do not compile or write")
    result.add_argument("--force", action="store_true", help="Explicitly replace an existing task-local cached extension")
    return result


def torch_header_directory(explicit):
    if explicit is not None:
        root = explicit.expanduser().resolve()
    else:
        # A top-level find_spec discovers package files without running torch's
        # __init__, loading CUDA, or executing a model.
        spec = importlib.util.find_spec("torch")
        locations = list(spec.submodule_search_locations or ()) if spec is not None else []
        if not locations:
            raise ValueError("Torch header files were not located; supply --torch-include (no install is attempted)")
        root = Path(locations[0]).resolve() / "include"
    if not (root / "pybind11/pybind11.h").is_file():
        raise FileNotFoundError(f"Existing pybind11 headers missing: {root / 'pybind11/pybind11.h'}")
    return root


def build_plan(args):
    cuda = args.cuda_home
    if cuda is None:
        # Locating nvcc's file is an inventory operation, not an nvcc execution.
        nvcc = shutil.which("nvcc")
        if nvcc:
            cuda = Path(nvcc).resolve().parent.parent
        else:
            raise ValueError("Provide --cuda-home or CUDA_HOME for the allocated CUDA 12.8 toolkit")
    cuda = cuda.expanduser().resolve()
    for header in ("cublasLt.h", "cuda_runtime_api.h"):
        if not (cuda / "include" / header).is_file():
            raise FileNotFoundError(f"CUDA header missing: {cuda / 'include' / header}")
    library = cuda / "lib64"
    if not library.is_dir():
        raise FileNotFoundError(f"CUDA library directory missing: {library}")
    headers = torch_header_directory(args.torch_include)
    python_include = Path(sysconfig.get_path("include"))
    if not (python_include / "Python.h").is_file():
        raise FileNotFoundError(f"Python development headers missing: {python_include / 'Python.h'}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("This Python interpreter did not provide EXT_SUFFIX")
    source = Path(__file__).resolve().parents[1] / "csrc/gemm/bf16_nt_plan.cpp"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = args.output_dir.expanduser().resolve() / f"flash_rt_gemm_plan{suffix}"
    command = [args.cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-pthread",
               "-Wall", "-Wextra", f"-I{python_include}", f"-I{headers}", f"-I{cuda / 'include'}",
               str(source), "-o", str(output), f"-L{library}", f"-Wl,-rpath,{library}",
               "-lcublasLt", "-lcudart"]
    return {"command": command, "output": str(output), "source": str(source),
            "cuda_home": str(cuda), "python_extension_suffix": suffix,
            "torch_headers_only": str(headers), "imports_torch": False,
            "full_flashrt_rebuild": False}


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        output = Path(plan["output"])
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; select a fresh task directory or use --force explicitly")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix=".gemm-plan-", suffix=".so", dir=output.parent)
        os.close(descriptor)
        temporary = Path(filename)
        command = list(plan["command"])
        command[command.index("-o") + 1] = str(temporary)
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, check=True)
            if args.force:
                temporary.replace(output)
            else:
                # Publish without overwriting even if another build created the
                # destination after our preflight check. Same filesystem as temp.
                os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)  # only this invocation's own temporary output
        print(json.dumps({**plan, "built": True}, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        arguments.exit(1, f"GEMM plan build failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
