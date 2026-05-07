"""Mesh extraction from SDF via marching cubes and post-processing.

Pipeline:
  1. Marching cubes at level=0 on the SDF volume.
  2. Filter to largest connected component.
  3. Taubin smoothing (optional).
"""

import numpy as np
import trimesh


def extract_mesh(
    volume: np.ndarray,
    origin: np.ndarray,
    step: float,
    smooth_iterations: int = 20,
    smooth_lambda: float = 0.5,
    min_component_ratio: float = 0.01,
) -> trimesh.Trimesh:
    """Extract and clean a mesh from an SDF volume.

    Parameters
    ----------
    volume : (Nx, Ny, Nz) float array — SDF values (negative inside).
    origin : (3,) — world position of volume[0, 0, 0].
    step : float — grid spacing.
    smooth_iterations : int — Taubin smoothing iterations (0 to skip).
    smooth_lambda : float — Taubin smoothing λ parameter.
    min_component_ratio : float — remove components with fewer than this
        fraction of the largest component's faces.

    Returns
    -------
    trimesh.Trimesh (may be empty on failure).
    """
    from skimage.measure import marching_cubes

    # Marching cubes
    try:
        verts, faces, _, _ = marching_cubes(volume, level=0.0, spacing=(step, step, step))
    except (ValueError, RuntimeError):
        return trimesh.Trimesh()

    if len(verts) == 0 or len(faces) == 0:
        return trimesh.Trimesh()

    # Translate to world coordinates
    verts += origin

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Component filtering — keep largest + anything close in size
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        sizes = np.array([len(c.faces) for c in components])
        max_size = sizes.max()
        threshold = max_size * min_component_ratio
        keep = [c for c, s in zip(components, sizes) if s >= threshold]
        if keep:
            mesh = trimesh.util.concatenate(keep)
        else:
            mesh = components[np.argmax(sizes)]

    # Taubin smoothing
    if smooth_iterations > 0 and len(mesh.vertices) > 0:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=smooth_lambda,
            nu=-smooth_lambda * 1.01,  # slightly asymmetric for stability
            iterations=smooth_iterations,
        )

    mesh.fix_normals()
    return mesh
