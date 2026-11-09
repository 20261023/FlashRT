"""Local-only loading through FastWAM's real from_wan22_pretrained API.

Reference: upstream commit 7faa711, models/wan22/{fastwam,helpers/io}.py.
No model identifier is resolved on a hub and no download setting is patched.
Only resolved model-section mappings/JSON are accepted, not Hydra execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import threading


_IMPORT_LOCK = threading.RLock()
_TARGETS = {
    "fastwam.runtime.create_fastwam": ("fastwam", "FastWAM"),
    "fastwam.runtime.create_fastwam_optional_idm": ("fastwam_optional_idm", "FastWAMOptionalIDM"),
}
_CONFIG_KEYS = {
    "_target_", "model_id", "tokenizer_model_id", "tokenizer_max_len",
    "load_text_encoder", "proprio_dim", "redirect_common_files",
    "mot_checkpoint_mixed_attn", "compile_training_denoise",
    "action_dit_pretrained_path", "skip_dit_load_from_pretrain",
    "video_dit_config", "action_dit_config", "video_scheduler",
    "action_scheduler", "loss", "action_idm_prob", "video_cond_noise_prob",
}


def read_model_config(model_config) -> dict:
    if isinstance(model_config, (str, Path)):
        path = Path(model_config).expanduser()
        if path.suffix.lower() != ".json":
            raise ValueError("model_config path must be resolved JSON, not an unresolved Hydra YAML")
        with path.open(encoding="utf-8") as stream:
            model_config = json.load(stream)
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be a resolved upstream model-section mapping or JSON path")
    config = deepcopy(dict(model_config))
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown FastWAM model_config keys: {sorted(unknown)}")

    def check_resolved(value):
        if isinstance(value, str) and "${" in value:
            raise ValueError("model_config contains unresolved Hydra interpolation")
        if isinstance(value, Mapping):
            for child in value.values():
                check_resolved(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check_resolved(child)

    check_resolved(config)
    target = config.get("_target_", "fastwam.runtime.create_fastwam")
    if target not in _TARGETS:
        raise ValueError("Only FastWAM and FastWAMOptionalIDM first_frame model targets are supported")
    if config.get("load_text_encoder", False):
        raise ValueError("Local FastWAM loading requires load_text_encoder=False and prepared context/context_mask")
    if config.get("mot_checkpoint_mixed_attn", False) or config.get("compile_training_denoise", False):
        raise ValueError("Training checkpointing/compilation is unsupported by this inference adapter")
    for key in ("video_dit_config", "action_dit_config", "action_scheduler"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"model_config.{key} must be a resolved mapping")
    for key in ("video_scheduler", "loss"):
        if key in config and not isinstance(config[key], Mapping):
            raise ValueError(f"model_config.{key} must be a resolved mapping")
    video, action = config["video_dit_config"], config["action_dit_config"]
    if video.get("video_attention_mask_mode") != "first_frame_causal":
        raise ValueError("FastWAM action-only mode requires video_attention_mask_mode='first_frame_causal'")
    if not video.get("seperated_timestep") or not video.get("fuse_vae_embedding_in_latents"):
        raise ValueError("FastWAM requires seperated_timestep and fuse_vae_embedding_in_latents")
    if tuple(video.get("patch_size", ()))[:1] != (1,):
        raise ValueError("FastWAM first-frame conditioning requires temporal patch_size=1")
    if video.get("use_gradient_checkpointing") or action.get("use_gradient_checkpointing"):
        raise ValueError("Expert gradient checkpointing must be disabled for inference")
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    if not required.issubset(config["action_scheduler"]):
        raise ValueError(f"action_scheduler requires {sorted(required)}")
    if target.endswith("optional_idm"):
        if "action_idm_prob" not in config:
            raise ValueError("FastWAMOptionalIDM requires action_idm_prob")
    elif "action_idm_prob" in config or "video_cond_noise_prob" in config:
        raise ValueError("IDM options require the explicit FastWAMOptionalIDM target")
    return config


def resolve_local_paths(checkpoint_dir, backbone_dir, source_dir=None):
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must identify a local full FastWAM checkpoint")
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if checkpoint.is_dir():
        candidates = sorted({*checkpoint.glob("*.pt"), *checkpoint.glob("*.pth")})
        if len(candidates) != 1:
            raise ValueError("checkpoint_dir must contain exactly one .pt/.pth file; otherwise provide its exact path")
        checkpoint = candidates[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {checkpoint}")
    if backbone_dir is None:
        raise ValueError("backbone_dir must be a local Wan directory containing Wan2.2_VAE.pth")
    backbone = Path(backbone_dir).expanduser().resolve()
    if not (backbone / "Wan2.2_VAE.pth").is_file():
        raise FileNotFoundError(f"Local Wan VAE not found: {backbone / 'Wan2.2_VAE.pth'}")
    source = None
    if source_dir is not None:
        source = Path(source_dir).expanduser().resolve()
        if not (source / "src/fastwam/models/wan22/fastwam.py").is_file():
            raise FileNotFoundError(f"source_dir is not a FastWAM source checkout: {source}")
    return checkpoint, backbone, source


@contextmanager
def _source_import_path(source):
    """Scope sys.path changes to import; never replace an already loaded package."""
    with _IMPORT_LOCK:
        if source is None:
            yield
            return
        src = (source / "src").resolve()
        for name, module in tuple(sys.modules.items()):
            if name == "fastwam" or name.startswith("fastwam."):
                filename = getattr(module, "__file__", None)
                if filename is not None:
                    locations = [filename]
                else:
                    # schedulers/ has no __init__.py in the pinned checkout:
                    # a legitimate namespace module has no concrete __file__.
                    spec = getattr(module, "__spec__", None)
                    locations = list(getattr(spec, "submodule_search_locations", None) or ())
                    locations.extend(getattr(module, "__path__", None) or ())
                if not locations or any(not Path(location).resolve().is_relative_to(src)
                                        for location in locations):
                    raise ImportError("A different fastwam package is already imported; use a fresh process")
        entry = str(src)
        sys.path.insert(0, entry)
        try:
            yield
        finally:
            sys.path.remove(entry)


def _validate_full_payload(model, payload):
    """Upstream load_checkpoint uses strict=False; forbid random leftovers."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("mot"), Mapping):
        raise ValueError("A full FastWAM checkpoint with a 'mot' state is required; legacy video-only 'dit' is unsupported")
    expected, actual = model.mot.state_dict(), payload["mot"]
    missing, unexpected = set(expected) - set(actual), set(actual) - set(expected)
    if missing or unexpected:
        raise ValueError(f"Incomplete/incompatible FastWAM mot state: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}")
    for key, value in expected.items():
        if getattr(actual[key], "shape", None) != value.shape:
            raise ValueError(f"FastWAM checkpoint shape mismatch for {key}")
    if model.proprio_encoder is not None:
        state = payload.get("proprio_encoder")
        if not isinstance(state, Mapping):
            raise ValueError("FastWAM checkpoint is missing required proprio_encoder weights")
        model.proprio_encoder.load_state_dict(state, strict=True)
    elif "proprio_encoder" in payload:
        raise ValueError("Checkpoint has proprio_encoder weights but model_config.proprio_dim is disabled")


def load_local_model(config, checkpoint, backbone, source, *, device, dtype):
    """Build on CPU, validate the complete checkpoint, then transfer to device.

    The explicit absolute model_id and existing VAE file make upstream
    ModelConfig.require_downloading() false. DiT pretraining and T5/tokenizer
    loading are disabled, so no other hub resolution/download path is used.
    Checkpoints must be trusted: upstream load_checkpoint owns deserialization.
    """
    target = config.get("_target_", "fastwam.runtime.create_fastwam")
    module_name, class_name = _TARGETS[target]
    try:
        with _source_import_path(source):
            module = importlib.import_module(f"fastwam.models.wan22.{module_name}")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "FastWAM or one of its dependencies is unavailable. Provide source_dir "
            "for the pinned checkout or an already installed upstream package; this adapter never installs it."
        ) from exc
    cls = getattr(module, class_name)
    action_scheduler = config["action_scheduler"]
    video_scheduler = config.get("video_scheduler", {})
    loss = config.get("loss", {})
    kwargs = {
        "device": "cpu", "torch_dtype": dtype,
        "model_id": str(backbone), "redirect_common_files": False,
        "load_text_encoder": False, "skip_dit_load_from_pretrain": True,
        "action_dit_pretrained_path": None,
        "video_dit_config": dict(config["video_dit_config"]),
        "action_dit_config": dict(config["action_dit_config"]),
        "proprio_dim": config.get("proprio_dim"),
        "video_train_shift": video_scheduler.get("train_shift", 5.0),
        "video_infer_shift": video_scheduler.get("infer_shift", 5.0),
        "video_num_train_timesteps": video_scheduler.get("num_train_timesteps", 1000),
        "action_train_shift": action_scheduler["train_shift"],
        "action_infer_shift": action_scheduler["infer_shift"],
        "action_num_train_timesteps": action_scheduler["num_train_timesteps"],
        "loss_lambda_video": loss.get("lambda_video", 1.0),
        "loss_lambda_action": loss.get("lambda_action", 1.0),
    }
    if target.endswith("optional_idm"):
        kwargs.update(action_idm_prob=config["action_idm_prob"],
                      video_cond_noise_prob=config.get("video_cond_noise_prob", 0.5))
    model = cls.from_wan22_pretrained(**kwargs)
    payload = model.load_checkpoint(str(checkpoint))
    _validate_full_payload(model, payload)
    del payload
    model.eval().to(device=device, dtype=dtype)
    # Upstream .to() does not update these attributes used by its helpers.
    model.device, model.torch_dtype = device, dtype
    return model
