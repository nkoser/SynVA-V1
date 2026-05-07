"""Small pure-PyTorch fallback for the pointnet2_ops functions Uni3D needs.

The official Uni3D repo imports:

    from pointnet2_ops import pointnet2_utils

Only two functions are used during inference: `furthest_point_sample` and
`gather_operation`. Building the CUDA extension can be fragile on very new
CUDA/PyTorch stacks, so this module can register a compatible fallback in
`sys.modules` before Uni3D imports it.
"""

from __future__ import annotations

import sys
import types


def _furthest_point_sample(xyz, npoint):
    torch = __import__("torch")
    if xyz.ndim != 3:
        raise ValueError(f"Expected xyz [B, N, C], got {tuple(xyz.shape)}")
    bsz, n_points, _ = xyz.shape
    npoint = int(npoint)
    if n_points <= 0:
        raise ValueError("Cannot sample from an empty point cloud.")

    device = xyz.device
    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.full((bsz, n_points), 1e10, dtype=xyz.dtype, device=device)
    farthest = torch.randint(0, n_points, (bsz,), dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def _gather_operation(features, idx):
    if features.ndim != 3:
        raise ValueError(f"Expected features [B, C, N], got {tuple(features.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"Expected idx [B, S], got {tuple(idx.shape)}")
    _, channels, _ = features.shape
    expanded = idx.long().unsqueeze(1).expand(-1, channels, -1)
    return features.gather(dim=2, index=expanded)


def install_pointnet2_fallback(force: bool = False) -> None:
    if not force:
        try:
            from pointnet2_ops import pointnet2_utils  # noqa: F401

            return
        except Exception:
            pass

    package = types.ModuleType("pointnet2_ops")
    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    utils.furthest_point_sample = _furthest_point_sample
    utils.gather_operation = _gather_operation
    package.pointnet2_utils = utils

    sys.modules["pointnet2_ops"] = package
    sys.modules["pointnet2_ops.pointnet2_utils"] = utils
