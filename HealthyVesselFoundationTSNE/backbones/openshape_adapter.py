"""OpenShape adapter.

Expected input from the main pipeline: [B, N, 3] float32 xyz point clouds.
OpenShape's public demo support package exposes `openshape.load_pc_encoder`,
and the loaded encoder expects [B, 6, N] xyzrgb tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .common import add_repo_path, extract_tensor, import_torch, tensor_to_numpy, xyzrgb_numpy


_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _load_openshape_model(config: Mapping[str, Any], device: str):
    add_repo_path(config.get("repo_path"))
    torch, torch_device = import_torch(device)

    try:
        import openshape
    except Exception as exc:
        raise ImportError(
            "OpenShape is not importable. Install the official OpenShape package "
            "or set embedding.openshape.repo_path to the local OpenShape repo."
        ) from exc

    model_name = str(config.get("model_name") or "openshape-pointbert-vitg14-rgb")
    key = (model_name, str(torch_device))
    if key not in _MODEL_CACHE:
        model = openshape.load_pc_encoder(model_name)
        model.to(torch_device)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key], torch, torch_device


def embed_point_clouds(points: np.ndarray, config: Mapping[str, Any], device: str = "cuda") -> np.ndarray:
    model, torch, torch_device = _load_openshape_model(config, device)
    pc_np = xyzrgb_numpy(points, config, channels_first=True)
    pc = torch.from_numpy(pc_np).to(torch_device)
    with torch.no_grad():
        output = model(pc)
    tensor = extract_tensor(output, preferred_keys=config.get("output_keys"))
    return tensor_to_numpy(tensor, pool=str(config.get("pool", "auto")))
