"""CLI entry point for cross-section SDF mesh reconstruction.

Usage:
    python -m CrossSectionSDF.reconstruct --config CrossSectionSDF/default_config.yaml
"""

import argparse
import glob
import os
import time

import numpy as np
import yaml

from .tree import load_tree_segments
from .interpolate import interpolate_segment, evaluate_bspline_ring
from .sdf import CrossSectionSDF
from .extract import extract_mesh


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def process_file(npy_path: str, params: dict, verbose: bool = True):
    """Full pipeline: .npy → segments → SDF → mesh."""
    k = params.get("k", 39)
    step = params.get("sdf_step", 0.004)
    smooth_k = params.get("smooth_k", 0.005)
    target_spacing = params.get("target_spacing", 0.003)
    n_angular = params.get("n_angular", 128)
    k_neighbors = params.get("k_neighbors", 5)
    smooth_iters = params.get("smooth_iterations", 20)
    smooth_lam = params.get("smooth_lambda", 0.5)
    min_comp = params.get("min_component_ratio", 0.01)
    batch_slices = params.get("batch_slices", 40)
    radius_cap_factor = params.get("radius_cap_factor", 1.7)

    # 1. Parse tree → segments
    segments = load_tree_segments(npy_path, k=k)
    if not segments:
        return None
    if verbose:
        print(f"  Parsed {len(segments)} segments")

    # 1b. Compute global radius cap from all nodes across all segments
    all_radii = []
    for seg in segments:
        for i in range(len(seg.centers)):
            ring = evaluate_bspline_ring(seg.coeffs[i])
            if ring is not None:
                r = float(np.max(np.linalg.norm(ring - seg.centers[i], axis=1)))
                all_radii.append(r)
    if all_radii:
        global_median = float(np.median(all_radii))
        global_radius_cap = max(global_median * radius_cap_factor, 0.02)
    else:
        global_radius_cap = None
    if verbose and global_radius_cap is not None:
        print(f"  Global radius: median={global_median:.4f}, cap={global_radius_cap:.4f}")

    # 2. Densely interpolate each segment
    dense_segments = []
    total_stations = 0
    for seg in segments:
        ds = interpolate_segment(seg.centers, seg.coeffs,
                                 target_spacing=target_spacing,
                                 n_angular=n_angular,
                                 global_radius_cap=global_radius_cap)
        if ds is not None:
            dense_segments.append(ds)
            total_stations += len(ds.centers)

    if not dense_segments:
        return None
    if verbose:
        print(f"  Interpolated → {len(dense_segments)} dense segments, {total_stations} stations")

    # 3. Build SDF evaluator
    sdf_eval = CrossSectionSDF(dense_segments, smooth_k=smooth_k, k_neighbors=k_neighbors)
    if verbose:
        lo, hi = sdf_eval.bounds()
        extent = hi - lo
        print(f"  Bounds: {lo} → {hi}  (extent {extent})")
        print(f"  Max radius: {sdf_eval.max_radius:.4f}")

    # 4. Evaluate SDF on grid
    t0 = time.time()
    volume, origin = sdf_eval.evaluate_grid(step=step, batch_slices=batch_slices, verbose=verbose)
    t_sdf = time.time() - t0
    if verbose:
        print(f"  SDF evaluation: {t_sdf:.1f}s")

    # 5. Extract mesh
    t0 = time.time()
    mesh = extract_mesh(volume, origin, step,
                        smooth_iterations=smooth_iters,
                        smooth_lambda=smooth_lam,
                        min_component_ratio=min_comp)
    t_mesh = time.time() - t0
    if verbose:
        print(f"  Mesh extraction: {t_mesh:.1f}s  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})")

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Cross-section SDF mesh reconstruction")
    parser.add_argument("--config", default="CrossSectionSDF/default_config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override input path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = args.input or paths.get("input")
    output_dir = args.output_dir or paths.get("output_dir")

    if not input_path or not output_dir:
        raise SystemExit("Error: input and output_dir are required (in config or --input/--output-dir).")

    os.makedirs(output_dir, exist_ok=True)

    pattern = params.get("pattern", "*.npy")
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = sorted(glob.glob(os.path.join(input_path, pattern)))

    filter_cases = params.get("filter_cases")
    if filter_cases:
        allowed = set(filter_cases)
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]

    max_files = params.get("max_files")
    if max_files is not None:
        files = files[:int(max_files)]

    ext = params.get("output_ext", ".obj")
    overwrite = params.get("overwrite", True)
    total = len(files)
    written = 0

    print(f"Processing {total} files → {output_dir}")

    for idx, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, name + ext)

        if os.path.exists(out_path) and not overwrite:
            print(f"[{idx}/{total}] {name}: skip (exists)")
            continue

        print(f"[{idx}/{total}] {name}")
        t0 = time.time()

        try:
            mesh = process_file(path, params)
            if mesh is not None and not mesh.is_empty:
                mesh.export(out_path)
                elapsed = time.time() - t0
                print(f"  → {out_path} ({elapsed:.1f}s)")
                written += 1
            else:
                print(f"  → FAILED (empty mesh)")
        except Exception as e:
            print(f"  → ERROR: {e}")

    print(f"\nDone: {written}/{total} written")


if __name__ == "__main__":
    main()
