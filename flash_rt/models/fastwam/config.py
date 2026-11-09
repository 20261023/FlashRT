"""CPU-only validation for the pinned FastWAM adapter."""

from __future__ import annotations

import math
from numbers import Integral, Real

from flash_rt.sampling.probeflow import ProbeFlowConfig
from flash_rt.sampling.schedules import ShiftedSigmaSchedule


UPSTREAM_COMMIT = "7faa71108368fbb3b6885649f112af607427a2d4"
UPSTREAM_URL = "https://github.com/yuantianyuan01/FastWAM"


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def vae_spatial_factor(vae) -> tuple[int, int]:
    """Read declared VAE spatial scaling, never infer it from a model name.

    Prefer the public VAE wrapper attribute (WanVideoVAE38 declares 16 there),
    with an inner-model attribute as an explicit alternative. A 2-tuple is
    (height,width); a 3-tuple is (time,height,width), not a scalar product.
    """
    factor = getattr(vae, "upsampling_factor", None)
    if factor is None:
        factor = getattr(getattr(vae, "model", None), "upsampling_factor", None)
    if isinstance(factor, Integral) and not isinstance(factor, bool):
        scale = positive_integer(factor, "vae.upsampling_factor")
        return scale, scale
    if isinstance(factor, (tuple, list)) and len(factor) in (2, 3):
        factors = tuple(positive_integer(value, "vae.upsampling_factor") for value in factor)
        return factors[-2:]
    raise ValueError("A known positive VAE upsampling_factor (scalar, HW or THW tuple) is required")


def validate_sampler(sampler, sigma_shift, num_inference_steps, probeflow_config):
    """Validate explicit schedule choices before checkpoint/device setup.

    FastWAM's released Euler grid is phi(u)=shift*u/(1+(shift-1)*u).
    Both samplers use the shared scalar schedule. Shifted ProbeFlow is an
    explicit grid adaptation, not a claim of original-paper validation.
    """
    if sampler not in ("euler", "probeflow"):
        raise ValueError("sampler must be 'euler' or 'probeflow'")
    if (isinstance(sigma_shift, bool) or not isinstance(sigma_shift, Real)
            or not math.isfinite(sigma_shift) or sigma_shift <= 0):
        raise ValueError("sigma_shift must be an explicit positive finite real number")
    if sampler == "probeflow":
        if num_inference_steps is not None:
            raise ValueError("num_inference_steps is Euler-only; use probeflow_config.max_steps")
        if probeflow_config is not None and not isinstance(probeflow_config, ProbeFlowConfig):
            raise TypeError("probeflow_config must be a ProbeFlowConfig")
        return None, probeflow_config or ProbeFlowConfig()
    if probeflow_config is not None:
        raise ValueError("probeflow_config requires sampler='probeflow'")
    steps = 20 if num_inference_steps is None else positive_integer(
        num_inference_steps, "num_inference_steps"
    )
    return steps, None


def shifted_sigma(u: float, shift: float) -> float:
    """Alias of the shared scalar Euler/ProbeFlow grid, not model timestep."""
    return ShiftedSigmaSchedule(shift)(u)
