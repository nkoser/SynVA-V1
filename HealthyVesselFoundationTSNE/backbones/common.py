"""Shared utilities for point-cloud foundation-model adapters."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


def add_repo_path(repo_path: Optional[str]) -> None:
    if not repo_path:
        return
    path = Path(str(repo_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"repo_path does not exist: {path}")
    path_str = path.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def resolve_callable(spec: str) -> Callable[..., Any]:
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    else:
        module_name, attr = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Resolved object is not callable: {spec}")
    return obj


def import_torch(device: str):
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch, torch.device(device)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center = pts.mean(axis=1, keepdims=True)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=-1, keepdims=True), axis=1, keepdims=True)
    pts = pts / np.maximum(scale, 1e-8)
    return pts.astype(np.float32)


def resample_points(points: np.ndarray, n_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    n_points = int(n_points)
    if n_points <= 0:
        return pts
    bsz, current, channels = pts.shape
    if current == n_points:
        return pts
    if current > n_points:
        idx = np.linspace(0, current - 1, n_points).round().astype(np.int64)
        return pts[:, idx, :]

    reps = int(np.ceil(n_points / max(current, 1)))
    tiled = np.tile(pts, (1, reps, 1))
    return tiled[:, :n_points, :].reshape(bsz, n_points, channels)


def xyzrgb_numpy(points: np.ndarray, config: Mapping[str, Any], channels_first: bool = False) -> np.ndarray:
    n_points = int(config.get("num_points", points.shape[1]))
    xyz = resample_points(points, n_points)
    if bool(config.get("adapter_normalize", True)):
        xyz = normalize_unit_sphere(xyz)

    rgb_value = np.asarray(config.get("rgb", [0.4, 0.4, 0.4]), dtype=np.float32).reshape(1, 1, 3)
    rgb = np.broadcast_to(rgb_value, xyz.shape).astype(np.float32)
    pc = np.concatenate([xyz, rgb], axis=-1).astype(np.float32)
    if channels_first:
        pc = np.transpose(pc, (0, 2, 1))
    return pc


def xyznormal_numpy(points: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    n_points = int(config.get("num_points", points.shape[1]))
    xyz = resample_points(points, n_points)
    if bool(config.get("adapter_normalize", True)):
        xyz = normalize_unit_sphere(xyz)

    input_channels = int(config.get("input_channels", 6))
    if input_channels <= 3:
        return xyz.astype(np.float32)
    normal_value = np.asarray(config.get("normal_fill", [0.0, 0.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
    normals = np.broadcast_to(normal_value, xyz.shape).astype(np.float32)
    pc = np.concatenate([xyz, normals], axis=-1)
    if input_channels > 6:
        pad = np.zeros((pc.shape[0], pc.shape[1], input_channels - 6), dtype=np.float32)
        pc = np.concatenate([pc, pad], axis=-1)
    return pc[:, :, :input_channels].astype(np.float32)


def to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(v) for v in value]
    return value


def extract_tensor(output: Any, preferred_keys: Optional[Sequence[str]] = None) -> Any:
    keys = list(preferred_keys or [])
    keys += [
        "embeddings",
        "embedding",
        "features",
        "feature",
        "feat",
        "pc_feat",
        "point_feat",
        "shape_embed",
        "latents",
        "z",
    ]
    if hasattr(output, "detach"):
        return output
    if isinstance(output, Mapping):
        for key in keys:
            if key in output:
                return extract_tensor(output[key], preferred_keys)
        for value in output.values():
            try:
                return extract_tensor(value, preferred_keys)
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if hasattr(x, "detach")]
        if tensors:
            return tensors[0]
        for value in output:
            try:
                return extract_tensor(value, preferred_keys)
            except TypeError:
                continue
    raise TypeError(f"Could not extract embedding tensor from output type {type(output)!r}")


def pool_embedding(tensor: Any, pool: str = "auto") -> Any:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        pool = str(pool or "auto")
        if pool in {"cls", "first"}:
            return tensor[:, 0]
        if pool in {"last"}:
            return tensor[:, -1]
        return tensor.mean(dim=1)
    if tensor.ndim > 3:
        return tensor.flatten(start_dim=1)
    return tensor.reshape(tensor.shape[0], -1)


def tensor_to_numpy(tensor: Any, pool: str = "auto") -> np.ndarray:
    pooled = pool_embedding(tensor, pool=pool)
    return pooled.detach().float().cpu().numpy().astype(np.float32)


def load_checkpoint_into_model(model: Any, checkpoint: Optional[str], strict: bool = False) -> None:
    if not checkpoint:
        return
    path = Path(str(checkpoint)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    import torch

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt
    for key in ("state_dict", "model", "module", "model_state_dict", "ema", "net"):
        if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if strict:
        return
    if missing:
        print(f"[adapter] checkpoint loaded with {len(missing)} missing keys")
    if unexpected:
        print(f"[adapter] checkpoint loaded with {len(unexpected)} unexpected keys")


def call_first_working_model_method(model: Any, methods: Iterable[str], tensor: Any, config: Mapping[str, Any]) -> Any:
    errors = []
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        candidates.append(inner)

    for candidate in candidates:
        for method_name in methods:
            fn = candidate if method_name in {"__call__", "forward"} else getattr(candidate, method_name, None)
            if fn is None:
                continue
            call_patterns = [
                lambda: fn(tensor),
                lambda: fn(tensor, return_latents=True),
                lambda: fn({"surface": tensor}),
                lambda: fn(surface=tensor),
                lambda: fn(pc=tensor),
                lambda: fn(points=tensor),
            ]
            for call in call_patterns:
                try:
                    return call()
                except TypeError as exc:
                    errors.append(f"{candidate.__class__.__name__}.{method_name}: {exc}")
                    continue
    raise RuntimeError("No compatible model forward/encode method worked. Last errors: " + " | ".join(errors[-5:]))
