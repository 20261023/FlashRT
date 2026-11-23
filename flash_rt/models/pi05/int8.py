"""CPU-only contract for the opt-in Pi0.5 SM80 W8A8 path.

``encoder`` quantizes the Gemma prefix projections; ``encoder_decoder`` also
quantizes the action expert projections. SigLIP, attention, residuals, style
modulation and action input/output projections retain BF16 in both modes.
Activation scales are recomputed per row on the device on every forward.
"""

from __future__ import annotations

import os


INT8_MODES = ("none", "encoder", "encoder_decoder")
INT8_ENV_FLAGS = (
    "FVK_PI05_RTX_FORCE_INT8",
    "FVK_PI05_RTX_INT8_ENCODER_ONLY",
    "FVK_PI05_RTX_INT8_VISION",
    "FVK_PI05_RTX_INT8_ENCODER_STATIC",
)


def validate_int8_mode(
    mode: str | None, *, use_fp8: bool = False, sampler: str = "euler",
    prefix_precision: str | None = None, decoder_precision: str | None = None,
    decoder_kernel: str | None = None,
) -> None:
    """Validate explicit selection without importing Torch or touching CUDA.

    ``None`` preserves the older frontend's environment-controlled behavior.
    Public VLA callers should pass ``"none"`` for BF16 or a named INT8 mode.
    This helper deliberately has no Torch dependency so the registry can share
    exactly the frontend's option/environment checks.
    """
    if mode is None:
        return
    if not isinstance(mode, str) or mode not in INT8_MODES:
        raise ValueError(f"int8_mode must be one of {INT8_MODES}, got {mode!r}")
    flags = INT8_ENV_FLAGS + (("FVK_PI05_RTX_FORCE_BF16",) if mode != "none" else ())
    for name in flags:
        if os.environ.get(name, "0") == "1":
            raise ValueError(f"Explicit int8_mode={mode!r} conflicts with {name}=1")
    if mode == "none":
        return
    if use_fp8:
        raise ValueError("int8_mode requires use_fp8=False")
    if sampler != "euler":
        raise ValueError("Pi0.5 INT8 currently supports sampler='euler' only")
    for name, value, default, allowed in (
        ("prefix_precision", prefix_precision, "fp8", ("fp8",)),
        ("decoder_precision", decoder_precision, "fp8", ("fp8",)),
        ("decoder_kernel", decoder_kernel, "cublaslt", ("auto", "cublaslt")),
    ):
        selected = value or os.environ.get(f"FLASHRT_PI05_{name.upper()}", default)
        if selected not in allowed:
            raise ValueError(f"int8_mode is incompatible with {name}={selected!r}")
