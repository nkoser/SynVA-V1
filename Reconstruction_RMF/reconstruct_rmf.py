#!/usr/bin/env python3
"""
Batch mesh reconstruction using Rotation-Minimizing Frames (RMF) directly
from B-spline cross-section rings.

Usage
-----
    python reconstruct_rmf.py --config reconstruct_rmf_config.yaml
    python reconstruct_rmf.py --input file.npy --output_dir /out/

Config keys (all optional, shown with defaults)
------------------------------------------------
paths:
  input:       <path to .npy file or directory>
  output_dir:  <output directory>

params:
  k:                    39       # feature dimension
  mode:                 pre_order_kcount
  pattern:              "*.npy"
  max_files:            null
  overwrite:            false

  # Ring sampling
  n_pts:                32       # vertices per cross-section ring
  target_spacing:       0.003    # arc-length spacing between densified rings
  use_densify:          true     # insert extra rings between nodes
  max_rings:            800      # safety cap per branch segment

  # Post-processing
  smooth_iterations:    10       # Taubin smoothing passes (0 = none)
  smooth_relaxation:    0.5      # Taubin lambda
  min_component_ratio:  0.01     # discard components < this fraction of total
  open_ends:            false    # leave branch ends uncapped

  output_ext:           ".obj"
  file_progress:        true
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob
from typing import Dict, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML required: pip install pyyaml")

try:
    import trimesh
except ImportError:
    raise RuntimeError("trimesh required: pip install trimesh")

from tree_functions import deserialize

# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_inputs(path: str, pattern: str):
    if os.path.isfile(path):
        return [path]
    return sorted(glob(os.path.join(path, pattern)))


def _smooth_mesh(mesh: "trimesh.Trimesh", iterations: int,
                 relaxation: float) -> "trimesh.Trimesh":
    """In-place Taubin smoothing (non-shrinking)."""
    if iterations <= 0:
        return mesh
    import trimesh.smoothing as tsm
    try:
        tsm.filter_taubin(mesh, lamb=relaxation, iterations=iterations)
    except Exception:
        pass
    return mesh


def _keep_largest(mesh: "trimesh.Trimesh", ratio: float) -> "trimesh.Trimesh":
    """Keep only connected components with >= ratio * total_faces."""
    if ratio is None or ratio <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    total = mesh.faces.shape[0]
    threshold = max(1, int(total * ratio))
    kept = [p for p in parts if p.faces.shape[0] >= threshold]
    if not kept:
        return max(parts, key=lambda p: p.faces.shape[0])
    if len(kept) == len(parts):
        return mesh
    return trimesh.util.concatenate(kept)


# ─── single-file reconstruction ───────────────────────────────────────────────

def reconstruct_file(npy_path: str, output_dir: str, params: Dict[str, Any]):
    """
    Reconstruct one .npy file and write the mesh.

    Returns ('ok', out_path) or ('skip', None) or ('error', err_msg).
    """
    # Parse params
    k       = int(params.get("k", 39))
    mode    = str(params.get("mode", "pre_order_kcount"))
    overwrite = bool(params.get("overwrite", False))
    ext     = str(params.get("output_ext", ".obj"))

    n_pts           = int(params.get("n_pts", 32))
    target_spacing  = float(params.get("target_spacing", 0.003))
    use_densify     = bool(params.get("use_densify", True))
    max_rings       = int(params.get("max_rings", 800))
    open_ends       = bool(params.get("open_ends", False))
    smooth_iter     = int(params.get("smooth_iterations", 10))
    smooth_relax    = float(params.get("smooth_relaxation", 0.5))
    min_ratio       = params.get("min_component_ratio", 0.01)
    if min_ratio is not None:
        min_ratio = float(min_ratio)
    verbose         = bool(params.get("verbose", False))

    base     = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(output_dir, base + ext)

    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path

    os.makedirs(output_dir, exist_ok=True)

    # Load & deserialize tree
    try:
        data = np.load(npy_path)
        if data.ndim == 1:
            data = data.reshape((-1, k))
        serial = list(data.flatten())
        tree = deserialize(serial, mode=mode, k=k)
    except Exception as e:
        return "error", f"load failed: {e}"

    # Reconstruct
    from Reconstruction_RMF.rmf_mesh import reconstruct_tree
    try:
        mesh = reconstruct_tree(
            tree, k,
            n_pts=n_pts,
            target_spacing=target_spacing,
            use_densify=use_densify,
            max_rings_per_segment=max_rings,
            bifurcation_patches=False,   # not yet wired in; tubes share junction node
            open_ends=open_ends,
        )
    except Exception as e:
        return "error", f"reconstruct failed: {e}"

    if mesh is None or len(mesh.vertices) == 0:
        return "skip", None

    # Post-processing
    if smooth_iter > 0:
        _smooth_mesh(mesh, smooth_iter, smooth_relax)
    if min_ratio:
        mesh = _keep_largest(mesh, min_ratio)

    if verbose:
        print(f"  {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, "
              f"watertight={mesh.is_watertight}")

    # Export
    try:
        mesh.export(out_path)
    except Exception as e:
        return "error", f"export failed: {e}"

    return "ok", out_path


# ─── batch runner ─────────────────────────────────────────────────────────────

def run_batch(config_path: str):
    cfg     = _load_config(config_path)
    paths   = cfg.get("paths", {})
    params  = cfg.get("params", {})

    input_path  = paths.get("input", ".")
    output_dir  = paths.get("output_dir", "./rmf_output")
    pattern     = params.get("pattern", "*.npy")
    max_files   = params.get("max_files")
    file_prog   = bool(params.get("file_progress", True))

    files = _iter_inputs(input_path, pattern)
    if max_files:
        files = files[:int(max_files)]

    print(f"RMF reconstruction: {len(files)} files → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    n_ok = n_skip = n_err = 0
    t_start = time.time()

    for i, npy_path in enumerate(files, 1):
        t0 = time.time()
        status, result = reconstruct_file(npy_path, output_dir, params)
        elapsed = time.time() - t0

        if status == "ok":
            n_ok += 1
            if file_prog:
                print(f"  [{i}/{len(files)}] ok ({elapsed:.1f}s)  {os.path.basename(result)}")
        elif status == "skip":
            n_skip += 1
            if file_prog:
                print(f"  [{i}/{len(files)}] skip  {os.path.basename(npy_path)}")
        else:
            n_err += 1
            print(f"  [{i}/{len(files)}] ERROR: {result}")

    total = time.time() - t_start
    print(f"\ndone: {n_ok} ok, {n_skip} skipped, {n_err} errors  ({total:.1f}s total)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RMF-based vessel mesh reconstruction")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--input",  help="Input .npy file or directory")
    parser.add_argument("--output_dir", help="Output directory")
    args = parser.parse_args()

    if args.config:
        run_batch(args.config)
    elif args.input:
        params = {}
        out_dir = args.output_dir or "./rmf_output"
        status, result = reconstruct_file(args.input, out_dir, params)
        print(f"{status}: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
