#!/usr/bin/env python3
"""
reconstruct.py — Main entry point for SplineInterpolationMesh.

Loads ground-truth (or generated) vessel-tree tokens, densely interpolates
cross-section splines between nodes, and reconstructs a triangle mesh
via Screened Poisson Surface Reconstruction (or Marching-Cubes SDF).

Usage:
    python SplineInterpolationMesh/reconstruct.py \
        --config SplineInterpolationMesh/configs/reconstruct_gt.yaml

    # Single file:
    python SplineInterpolationMesh/reconstruct.py \
        --input path/to/tree.npy \
        --output_dir output/meshes
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import (
    deserialize,
    local_geometry_tree_to_absolute,
)
from reconstruct_mesh import get_segments

from SplineInterpolationMesh.interpolation import interpolate_tree_segments
from SplineInterpolationMesh.mesh_generation import sdf_reconstruction, sdf_fast_reconstruction, loft_reconstruction, hybrid_reconstruction, poisson_reconstruction, poisson_perseg_reconstruction, smooth_mesh, decimate_mesh


# ─── Config loading ──────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─── Data loading & conversion ───────────────────────────────────────────


def load_tree(path: str, convert_to_absolute: bool = True, n_cp: int = 8):
    """Load a .npy tree file, optionally converting from local to absolute.

    Parameters
    ----------
    n_cp : int
        Number of B-spline control points per axis (8 → k=39, 16 → k=71).

    Returns
    -------
    tree : Tree object (binary tree)
    abs_data : (N, k+1) array in absolute coordinates
    """
    n_knot = n_cp + 4
    k = 3 + 3 * n_cp + n_knot  # geom dims per node
    row_len = k + 1             # +1 for kcount column

    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape((-1, row_len))

    # Build control-point slices: cols 4..4+n_cp, 4+n_cp..4+2*n_cp, etc.
    cp_start = 4
    cp_slices = (
        (cp_start, cp_start + n_cp),
        (cp_start + n_cp, cp_start + 2 * n_cp),
        (cp_start + 2 * n_cp, cp_start + 3 * n_cp),
    )

    if convert_to_absolute:
        abs_data = local_geometry_tree_to_absolute(
            data,
            position_slice=(1, 4),
            control_point_slices=cp_slices,
            relative_positions=True,
            node_local_control_points=True,
        )
    else:
        abs_data = data.copy()

    serial = list(abs_data.flatten())
    tree = deserialize(serial, mode="pre_order_kcount", k=k)
    return tree, abs_data


# ─── Single-file processing ─────────────────────────────────────────────


def process_file(
    path: str,
    output_dir: str,
    params: dict,
) -> tuple[str, str]:
    """Process one .npy file → triangle mesh.

    Returns (status, output_path) where status is "ok", "skip", or "fail".
    """
    base = os.path.splitext(os.path.basename(path))[0]
    output_ext = params.get("output_ext", ".stl")
    out_path = os.path.join(output_dir, base + output_ext)
    overwrite = bool(params.get("overwrite", False))

    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path

    # ── Load & convert ───────────────────────────────────────────────
    convert = bool(params.get("convert_to_absolute", True))
    n_cp = int(params.get("n_cp", 8))
    try:
        tree, abs_data = load_tree(path, convert_to_absolute=convert, n_cp=n_cp)
    except Exception as e:
        print(f"  [FAIL] Could not load {path}: {e}")
        return "fail", out_path

    if tree is None:
        print(f"  [FAIL] Empty tree: {path}")
        return "fail", out_path

    # ── Extract segments ─────────────────────────────────────────────
    n_knot = n_cp + 4
    k_geom = 3 + 3 * n_cp + n_knot
    k = k_geom + 1  # cols per row in the *data* array (k_children + geom)
    segments = get_segments(tree, k)
    if not segments:
        print(f"  [FAIL] No segments: {path}")
        return "fail", out_path

    # ── Interpolate ──────────────────────────────────────────────────
    target_spacing = float(params.get("target_spacing", 0.005))
    n_ring_pts = int(params.get("n_ring_pts", 64))

    seg_data = interpolate_tree_segments(
        segments,
        target_spacing=target_spacing,
        n_ring_pts=n_ring_pts,
        n_cp=n_cp,
    )

    total_rings = sum(len(s["rings"]) for s in seg_data)
    if total_rings < 4:
        print(f"  [FAIL] Too few rings ({total_rings}): {path}")
        return "fail", out_path

    # ── Mesh reconstruction ─────────────────────────────────────────
    method = params.get("method", "sdf")

    if method == "loft":
        mesh = loft_reconstruction(
            seg_data,
            cap_ends=bool(params.get("loft_cap_ends", True)),
            bif_smooth_iterations=int(params.get("bif_smooth_iterations", 30)),
            bif_smooth_radius_factor=float(params.get("bif_smooth_radius_factor", 2.5)),
            bif_smooth_lambda=float(params.get("bif_smooth_lambda", 0.5)),
            complex_threshold=float(params.get("complex_threshold", 1.3)),
            complex_iterations=int(params.get("complex_iterations", 120)),
            complex_radius_factor=float(params.get("complex_radius_factor", 4.0)),
            verbose=True,
        )
    elif method == "hybrid":
        mesh = hybrid_reconstruction(
            seg_data,
            complexity_threshold=float(params.get("hybrid_complexity_threshold", 2.0)),
            sdf_rings_range=int(params.get("hybrid_sdf_rings_range", 25)),
            sdf_grid_resolution=int(params.get("hybrid_sdf_grid_resolution", 128)),
            sdf_smooth_k=float(params.get("hybrid_sdf_smooth_k", 0.015)),
            bif_smooth_iterations=int(params.get("bif_smooth_iterations", 30)),
            bif_smooth_radius_factor=float(params.get("bif_smooth_radius_factor", 2.5)),
            bif_smooth_lambda=float(params.get("bif_smooth_lambda", 0.5)),
            verbose=True,
        )
    elif method == "sdf_fast":
        mesh = sdf_fast_reconstruction(
            seg_data,
            grid_resolution=int(params.get("mc_resolution", 256)),
            padding=float(params.get("mc_padding", 0.02)),
            n_neighbors=int(params.get("mc_n_neighbors", 3)),
            smooth_k=float(params.get("mc_smooth_k", 0.005)),
            level=float(params.get("mc_level", 0.0)),
            narrow_band_factor=float(params.get("mc_narrow_band_factor", 3.0)),
            sdf_smooth_sigma=float(params.get("sdf_smooth_sigma", 1.5)),
            sdf_erosion=float(params.get("sdf_erosion", 0.003)),
            verbose=True,
        )
    elif method == "poisson":
        mesh = poisson_reconstruction(
            seg_data,
            depth=int(params.get("poisson_depth", 8)),
            linear_fit=bool(params.get("poisson_linear_fit", False)),
            verbose=True,
        )
    elif method == "poisson_perseg":
        mesh = poisson_perseg_reconstruction(
            seg_data,
            depth=int(params.get("poisson_depth", 8)),
            n_cap_radial=int(params.get("poisson_n_cap_radial", 8)),
            repoisson=bool(params.get("poisson_repoisson", True)),
            repoisson_depth=int(params.get("poisson_repoisson_depth", 9)),
            repoisson_samples=int(params.get("poisson_repoisson_samples", 120000)),
            iso_remesh=bool(params.get("iso_remesh", False)),
            iso_remesh_factor=float(params.get("iso_remesh_factor", 2.0)),
            iso_remesh_iter=int(params.get("iso_remesh_iter", 3)),
            iso_taubin_iter=int(params.get("iso_taubin_iter", 30)),
            verbose=True,
        )
    else:
        mesh = sdf_reconstruction(
            seg_data,
            grid_resolution=int(params.get("mc_resolution", 256)),
            padding=float(params.get("mc_padding", 0.02)),
            n_neighbors=int(params.get("mc_n_neighbors", 3)),
            smooth_k=float(params.get("mc_smooth_k", 0.01)),
            level=float(params.get("mc_level", 0.0)),
            narrow_band_factor=float(params.get("mc_narrow_band_factor", 3.0)),
            chunk_size=int(params.get("mc_chunk_size", 200_000)),
            sdf_smooth_sigma=float(params.get("sdf_smooth_sigma", 0.0)),
            sdf_erosion=float(params.get("sdf_erosion", 0.0)),
            verbose=True,
        )

    if mesh is None or len(mesh.vertices) == 0:
        print(f"  [FAIL] Reconstruction returned empty mesh: {path}")
        return "fail", out_path

    # ── Optional post-processing ─────────────────────────────────────
    smooth_iterations = int(params.get("smooth_iterations", 0))
    if smooth_iterations > 0:
        mesh = smooth_mesh(mesh, iterations=smooth_iterations)

    decimate_ratio = float(params.get("decimate_ratio", 0))
    if 0 < decimate_ratio < 1:
        mesh = decimate_mesh(mesh, ratio=decimate_ratio)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ext = output_ext.lstrip(".").lower()
    mesh.export(out_path, file_type=ext)
    print(f"  [OK] {out_path}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    return "ok", out_path


# ─── Batch processing ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SplineInterpolationMesh reconstruction")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--input", type=str, default=None, help="Input .npy file or directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # ── Merge config + CLI ───────────────────────────────────────────
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = args.input or paths.get("input")
    output_dir = args.output_dir or paths.get("output_dir", "SplineInterpolationMesh/output")

    if input_path is None:
        parser.error("Provide --input or paths.input in config")

    pattern = params.get("pattern", "*.npy")
    max_files = params.get("max_files")

    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, pattern)))
    else:
        files = [input_path]

    if params.get("skip_rotations", False):
        before = len(files)
        files = [f for f in files if not os.path.basename(f).startswith("rot")]
        print(f"  skip_rotations: {before} → {len(files)} files")

    # Optionally exclude files matching glob pattern(s) — comma-separated
    exclude_pattern = params.get("exclude_pattern")
    if exclude_pattern:
        import fnmatch
        pats = [p.strip() for p in exclude_pattern.split(",")]
        before = len(files)
        files = [f for f in files if not any(fnmatch.fnmatch(os.path.basename(f), p) for p in pats)]
        print(f"  exclude_pattern '{exclude_pattern}': {before} → {len(files)} files")

    if max_files:
        files = files[: int(max_files)]

    print(f"SplineInterpolationMesh — {len(files)} file(s)")
    method = params.get("method", "sdf")
    print(f"  Method: {method}")
    if method == "sdf":
        print(f"  Grid resolution: {params.get('mc_resolution', 256)}")
    print(f"  Target spacing: {params.get('target_spacing', 0.005)}")
    print(f"  Ring points: {params.get('n_ring_pts', 64)}")
    print(f"  Output: {output_dir}")
    print()

    stats = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()

    for i, f in enumerate(files):
        name = os.path.basename(f)
        print(f"[{i + 1}/{len(files)}] {name}")
        status, _ = process_file(f, output_dir, params)
        stats[status] += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — OK: {stats['ok']}, Skip: {stats['skip']}, Fail: {stats['fail']}")


if __name__ == "__main__":
    main()
