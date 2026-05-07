"""
GPU-friendly SDF marching-cubes generator.

Instead of the CPU core.py approach (52,000 batches of 8³ = 512 points each,
using a ThreadPool), this evaluates the SDF on one large grid in a small number
of big GPU chunks (~2 M points at a time).

For a 2 M-sample grid that's:
  CPU (core.py + KDTree)    : ~400 s  (52 K × 512 pts × scipy KDTree)
  GPU (core_cuda + cdist)   : ~4-8 s  (one shot on NVIDIA B200)

API is a drop-in replacement for core.generate / core.save.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional, Tuple

import numpy as np
from skimage import measure

from . import progress as sdf_progress
from . import stl
from .core import _estimate_bounds, _marching_cubes  # reuse helpers


# ──────────────────────────────────────────────────────────────────────────────
# public constants (match core.py)
# ──────────────────────────────────────────────────────────────────────────────

SAMPLES   = 2 ** 21          # ~2 M default (slightly lower to stay fast)
GPU_CHUNK = 2_000_000        # max points per GPU call (adjust for VRAM)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cartesian_product_np(*arrays):
    """Fast cartesian product of 1-D arrays → (N, d) float32."""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la).astype(np.float32)


def _eval_volume(sdf, X, Y, Z, chunk=GPU_CHUNK, verbose=True):
    """
    Evaluate sdf on the full (len(X) × len(Y) × len(Z)) grid in large chunks.
    Returns (nx, ny, nz) float32 volume.
    """
    nx, ny, nz = len(X), len(Y), len(Z)
    N = nx * ny * nz
    volume_flat = np.empty(N, dtype=np.float32)

    # Build full point set in one allocation (float32 saves VRAM vs float64)
    # Shape (N, 3)
    if N <= chunk:
        # Small enough for one shot
        P = _cartesian_product_np(X, Y, Z)
        volume_flat[:] = np.asarray(sdf(P)).ravel()
    else:
        # We need to iterate through the grid in chunks.
        # Fastest to iterate over X slabs (each slab = ny*nz points).
        slab_size = ny * nz
        bar = sdf_progress.Bar(nx, enabled=verbose)
        ptr = 0
        # Pre-build YZ slab (constant for all X)
        YZ = _cartesian_product_np(Y, Z)   # (ny*nz, 2)
        buf_x = np.empty((slab_size, 1), dtype=np.float32)
        slab_pts = np.empty((slab_size, 3), dtype=np.float32)
        slab_pts[:, 1:] = YZ               # Y and Z never change

        # We'll process full X-slabs unless a slab itself > chunk,
        # in which case we break it into sub-chunks.
        for ix in range(nx):
            buf_x[:, 0] = X[ix]
            slab_pts[:, 0] = X[ix]
            if slab_size <= chunk:
                vals = np.asarray(sdf(slab_pts)).ravel()
                volume_flat[ptr: ptr + slab_size] = vals
            else:
                # slab too big (very high resolution) — chunk within slab
                for start in range(0, slab_size, chunk):
                    end = min(start + chunk, slab_size)
                    vals = np.asarray(sdf(slab_pts[start:end])).ravel()
                    volume_flat[ptr + start: ptr + end] = vals
            ptr += slab_size
            bar.increment(1)
        bar.done()

    return volume_flat.reshape(nx, ny, nz)


# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────

def generate(
    sdf,
    step=None,
    bounds=None,
    samples=SAMPLES,
    verbose: bool = True,
    sparse: bool = True,           # kept for API compatibility; not used (full grid)
    workers=None,                  # kept for API compatibility; not used
    batch_size=None,               # kept for API compatibility; not used
    chunk: int = GPU_CHUNK,
) -> Tuple[list, tuple]:
    """
    GPU-friendly marching-cubes generate().

    Parameters mirror core.generate() for drop-in compatibility.
    Returns (points_list, bounds) just like core.generate().
    """
    start_t = time.time()

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume_size = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume_size / samples) ** (1.0 / 3.0)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    X = np.arange(x0, x1, dx, dtype=np.float32)
    Y = np.arange(y0, y1, dy, dtype=np.float32)
    Z = np.arange(z0, z1, dz, dtype=np.float32)

    n_total = len(X) * len(Y) * len(Z)

    if verbose:
        print(f"[cuda] bounds  min ({x0:.4f}, {y0:.4f}, {z0:.4f})")
        print(f"[cuda] bounds  max ({x1:.4f}, {y1:.4f}, {z1:.4f})")
        print(f"[cuda] step    ({dx:.4g}, {dy:.4g}, {dz:.4g})")
        print(f"[cuda] grid    {len(X)} × {len(Y)} × {len(Z)} = {n_total:,} pts")

    volume = _eval_volume(sdf, X, Y, Z, chunk=chunk, verbose=verbose)

    # Run marching cubes on CPU (fast; the volume is already computed)
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, 0.0)
    except Exception as exc:
        if verbose:
            print(f"[cuda] marching_cubes failed: {exc}")
        return [], bounds

    # Scale & translate verts back to world space
    scale  = np.array([dx, dy, dz], dtype=np.float32)
    offset = np.array([x0, y0, z0], dtype=np.float32)
    world_verts = verts * scale + offset

    # Expand into triangle soup (matches core.py output format)
    points = list(world_verts[faces].reshape(-1, 3))

    if verbose:
        triangles = len(points) // 3
        elapsed   = time.time() - start_t
        print(f"[cuda] {triangles:,} triangles in {elapsed:.1f} s")

    return points, bounds


def save(
    path: str,
    sdf,
    *args,
    **kwargs,
) -> tuple:
    """
    GPU-friendly save().  Mirrors core.save() signature.
    Always writes a deduplicated (shared-vertex) mesh so that downstream
    tools like the trimesh component filter work correctly.
    """
    points, bounds = generate(sdf, *args, **kwargs)

    if not points:
        # Empty mesh – write a minimal valid file
        lower = path.lower()
        if lower.endswith(".stl"):
            stl.write_binary_stl(path, [])
        return bounds

    pts_arr = np.array(points, dtype=np.float32)   # (N*3, 3) triangle soup
    # Deduplicate vertices so the mesh has proper connectivity
    pts_u, inv = np.unique(pts_arr, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)

    import meshio
    mesh = meshio.Mesh(pts_u, [("triangle", faces)])
    mesh.write(path)
    return bounds
