"""Template for plugging a real point-cloud foundation model into the t-SNE pipeline.

Copy this file or import from it when wiring Uni3D, OpenShape, Michelangelo, or
another model. The pipeline expects a callable with this interface:

    embed_point_clouds(points, config, device) -> np.ndarray

where points has shape [B, N, 3] and the returned array has shape [B, D].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def embed_point_clouds(points: np.ndarray, config: Mapping[str, Any], device: str = "cuda") -> np.ndarray:
    """Replace this body with the real model forward pass.

    Parameters
    ----------
    points:
        Float32 point clouds with shape [batch, points_per_mesh, 3]. They are
        already normalized by the main pipeline.
    config:
        Backend-specific config block, for example embedding.uni3d.
    device:
        Preferred torch device string from the config.

    Returns
    -------
    np.ndarray
        Embeddings with shape [batch, embedding_dim].
    """
    checkpoint = config.get("checkpoint")
    model_name = config.get("model_name")
    raise NotImplementedError(
        "Wire the selected foundation model here. "
        f"model_name={model_name!r}, checkpoint={checkpoint!r}, device={device!r}"
    )


def require_file(path_like: Any, name: str) -> Path:
    """Small helper for adapters that need a checkpoint or config file."""
    if not path_like:
        raise ValueError(f"Missing required path for {name}")
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path
