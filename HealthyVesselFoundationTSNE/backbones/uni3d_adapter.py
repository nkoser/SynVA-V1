"""Uni3D adapter.

This adapter targets the official Uni3D code layout where `models.uni3d`
contains `create_uni3d(args=...)` and the resulting model exposes `encode_pc`.
If your local Uni3D checkout uses a different factory, set
`embedding.uni3d.callable` to your own wrapper or adjust `model_name`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .common import (
    add_repo_path,
    extract_tensor,
    import_torch,
    load_checkpoint_into_model,
    tensor_to_numpy,
    to_namespace,
    xyzrgb_numpy,
)
from .pointnet2_fallback import install_pointnet2_fallback


_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


UNI3D_SCALE_DEFAULTS = {
    "base": {
        "pc_model": "eva02_base_patch14_448",
        "pc_feat_dim": 768,
        "embed_dim": 768,
        "pc_encoder_dim": 512,
        "num_group": 512,
        "group_size": 64,
    },
    "large": {
        "pc_model": "eva02_large_patch14_448",
        "pc_feat_dim": 1024,
        "embed_dim": 1024,
        "pc_encoder_dim": 512,
        "num_group": 512,
        "group_size": 64,
    },
    "giant": {
        "pc_model": "eva_giant_patch14_560",
        "pc_feat_dim": 1408,
        "embed_dim": 1024,
        "pc_encoder_dim": 512,
        "num_group": 512,
        "group_size": 64,
    },
}


def _build_args(config: Mapping[str, Any]) -> Any:
    scale = str(config.get("scale", "giant")).lower()
    args = dict(UNI3D_SCALE_DEFAULTS.get(scale, UNI3D_SCALE_DEFAULTS["giant"]))
    args.update(
        {
            "pretrained_pc": config.get("pretrained_pc") or "",
            "drop_path_rate": float(config.get("drop_path_rate", 0.0)),
            "patch_dropout": float(config.get("patch_dropout", 0.0)),
            "ckpt_path": config.get("checkpoint") or "",
            "model": config.get("model_name") or "create_uni3d",
        }
    )
    args.update(dict(config.get("args", {}) or {}))
    return to_namespace(args)


def _load_uni3d_model(config: Mapping[str, Any], device: str):
    add_repo_path(config.get("repo_path"))
    torch, torch_device = import_torch(device)
    checkpoint = str(config.get("checkpoint") or "")
    model_name = str(config.get("model_name") or "create_uni3d")
    key = (model_name, checkpoint, str(torch_device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], torch, torch_device

    try:
        if bool(config.get("use_pointnet2_fallback", True)):
            install_pointnet2_fallback(force=bool(config.get("force_pointnet2_fallback", False)))
        from models import uni3d as uni3d_models
    except Exception as exc:
        raise ImportError(
            "Uni3D modules are not importable. Set embedding.uni3d.repo_path "
            "to the official Uni3D checkout. Underlying import error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    factory = getattr(uni3d_models, model_name, None)
    if factory is None:
        raise AttributeError(f"models.uni3d has no factory named {model_name!r}")

    args = _build_args(config)
    model = factory(args=args)
    load_checkpoint_into_model(model, config.get("checkpoint"), strict=bool(config.get("strict_checkpoint", False)))
    model.to(torch_device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model, torch, torch_device


def embed_point_clouds(points: np.ndarray, config: Mapping[str, Any], device: str = "cuda") -> np.ndarray:
    model, torch, torch_device = _load_uni3d_model(config, device)
    pc_np = xyzrgb_numpy(points, config, channels_first=False)
    pc = torch.from_numpy(pc_np).to(torch_device)
    with torch.no_grad():
        if hasattr(model, "encode_pc"):
            output = model.encode_pc(pc)
        else:
            output = model(pc)
    tensor = extract_tensor(output, preferred_keys=config.get("output_keys"))
    return tensor_to_numpy(tensor, pool=str(config.get("pool", "auto")))
