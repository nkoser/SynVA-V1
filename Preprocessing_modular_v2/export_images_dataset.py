import argparse
import os
import sys
import traceback
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize
from view_trees import (
    collect_nodes_edges,
    collect_branches,
    render_custom_plot,
    compute_edge_radii,
    radius_from_node,
    check_root_zero,
)


K_MODES_WITH_COUNT = {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def iter_files(input_path, pattern):
    if os.path.isdir(input_path):
        return sorted(glob(os.path.join(input_path, pattern)))
    return [input_path]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def default_output_dir(input_path):
    if os.path.isdir(input_path):
        return os.path.join(os.path.dirname(input_path), "TreeImages")
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "TreeImages")


def load_tree(file_path, k, mode):
    data = np.load(file_path)
    node_dim = (k + 1) if mode in K_MODES_WITH_COUNT else k

    if data.ndim == 1:
        if data.size % node_dim != 0:
            raise ValueError(
                f"Tree array size ({data.size}) is not divisible by node_dim ({node_dim}) for mode={mode}, k={k}"
            )
        data = data.reshape((-1, node_dim))

    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    if tree is None:
        raise ValueError("Tree deserialization returned None")
    return tree


def build_output_path(output_dir, stem, suffix):
    return os.path.join(output_dir, f"{stem}{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description="Batch export tree visualizations as PNG images.")
    parser.add_argument("--config", default="views_dataset_export_images_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = paths.get("input")
    if not input_path:
        raise ValueError("paths.input is required")

    pattern = params.get("pattern", "*.npy")
    files = iter_files(input_path, pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern '{pattern}' in {input_path}")

    output_dir = paths.get("output_dir") or default_output_dir(input_path)
    ensure_dir(output_dir)

    k = int(params.get("k", 39))
    mode = params.get("mode", "post_order")
    overwrite = bool(params.get("overwrite", False))
    continue_on_error = bool(params.get("continue_on_error", True))
    traceback_on_error = bool(params.get("traceback_on_error", False))

    viewer = params.get("viewer", "combined")
    figsize = params.get("figsize", [8, 7])
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        figsize = (float(figsize[0]), float(figsize[1]))
    else:
        figsize = (8.0, 7.0)

    normalize_xyz = bool(params.get("normalize_xyz", False))
    image_suffix = str(params.get("image_suffix", ""))
    dpi = int(params.get("dpi", 200))

    draw_edges_flag = bool(params.get("draw_edges", True))
    draw_splines_flag = bool(params.get("draw_splines", True))
    draw_centerline_spline_flag = bool(params.get("draw_centerline_spline", False))
    draw_spheres_flag = bool(params.get("draw_spheres", False))

    radius_mode = params.get("radius_mode", "edge")
    radius_fixed = float(params.get("radius_fixed", 0.01))
    radius_scale = float(params.get("radius_scale", 0.1))
    radius_min = float(params.get("radius_min", 0.0))
    radius_max = params.get("radius_max")
    radius_max = float(radius_max) if radius_max is not None else None

    success = 0
    skipped = 0
    failed = 0
    total = len(files)

    for idx, file_path in enumerate(files, start=1):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        out_path = build_output_path(output_dir, stem, image_suffix)
        print(f"[{idx}/{total}] Processing {stem}")

        if os.path.exists(out_path) and not overwrite:
            print(f"  skip exists: {out_path}")
            skipped += 1
            continue

        try:
            tree = load_tree(file_path, k=k, mode=mode)
            nodes = []
            edges = []
            collect_nodes_edges(tree, nodes, edges)
            branches = []
            collect_branches(tree, [], branches)

            xyz = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in nodes], dtype=np.float32)
            root_pos = np.array([tree.data["x"], tree.data["y"], tree.data["z"]], dtype=np.float32)

            if bool(params.get("check_root_zero", False)):
                tol = float(params.get("root_zero_tol", 1e-6))
                check_root_zero(root_pos, tol)

            if normalize_xyz:
                max_abs = np.max(np.abs(xyz))
                if max_abs > 0:
                    xyz = xyz / max_abs

            if radius_mode == "edge":
                edge_lengths = [[] for _ in range(len(nodes))]
                for i, j in edges:
                    dist = float(np.linalg.norm(xyz[i] - xyz[j]))
                    edge_lengths[i].append(dist)
                    edge_lengths[j].append(dist)
                radii = compute_edge_radii(edge_lengths, radius_scale)
            else:
                radii = []
                for n in nodes:
                    r = radius_from_node(n, k, radius_mode, radius_fixed)
                    r = abs(r) * radius_scale
                    radii.append(r)
                radii = np.array(radii, dtype=np.float32)

            if radius_max is not None:
                radii = np.minimum(radii, radius_max)
            radii = np.maximum(radii, radius_min)

            if viewer == "custom":
                render_custom_plot(
                    os.path.basename(file_path),
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    draw_edges_flag,
                    draw_spheres_flag,
                    draw_splines_flag,
                    draw_centerline_spline_flag,
                    figsize,
                )
            elif viewer == "legacy_splines":
                render_custom_plot(
                    os.path.basename(file_path) + " (splines)",
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    False,
                    False,
                    True,
                    draw_centerline_spline_flag,
                    figsize,
                )
            elif viewer == "combined":
                render_custom_plot(
                    os.path.basename(file_path) + " (combined)",
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    True,
                    False,
                    True,
                    draw_centerline_spline_flag,
                    figsize,
                )
            elif viewer == "all":
                render_custom_plot(
                    os.path.basename(file_path) + " (custom)",
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    draw_edges_flag,
                    draw_spheres_flag,
                    False,
                    draw_centerline_spline_flag,
                    figsize,
                )
                render_custom_plot(
                    os.path.basename(file_path) + " (legacy_splines)",
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    False,
                    False,
                    True,
                    draw_centerline_spline_flag,
                    figsize,
                )
                render_custom_plot(
                    os.path.basename(file_path) + " (combined)",
                    xyz,
                    edges,
                    radii,
                    root_pos,
                    nodes,
                    params,
                    normalize_xyz,
                    True,
                    False,
                    True,
                    draw_centerline_spline_flag,
                    figsize,
                )
            else:
                raise ValueError("Unsupported viewer mode. Use custom, legacy_splines, combined, or all.")

            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close("all")
            print(f"  wrote: {out_path}")
            success += 1

        except Exception as exc:
            failed += 1
            plt.close("all")
            print(f"  failed: {file_path} ({exc})")
            if traceback_on_error:
                traceback.print_exc()
            if not continue_on_error:
                raise

    print(
        f"Done. cases={total}, success={success}, failed={failed}, skipped_outputs={skipped}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
