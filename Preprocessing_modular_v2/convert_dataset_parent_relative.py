import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tree_functions import absolute_tree_to_local_geometry, local_geometry_tree_to_absolute


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_files(folder, pattern):
    return sorted(glob.glob(os.path.join(folder, pattern)))


def reshape_tree(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and arr.size % 40 == 0:
        return arr.reshape((-1, 40))
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def summarize_position_delta(src_files, dst_files):
    abs_values = []
    rel_values = []
    for src, dst in zip(src_files, dst_files):
        src_arr = reshape_tree(np.load(src))
        dst_arr = reshape_tree(np.load(dst))
        if src_arr.shape[0] != dst_arr.shape[0]:
            raise ValueError(f"Row mismatch between {src} and {dst}")
        abs_values.append(np.asarray(src_arr[:, 1:4], dtype=np.float32))
        rel_values.append(np.asarray(dst_arr[:, 1:4], dtype=np.float32))
    if not abs_values:
        return {"files": 0}
    abs_cat = np.concatenate(abs_values, axis=0)
    rel_cat = np.concatenate(rel_values, axis=0)
    return {
        "files": int(len(src_files)),
        "nodes": int(abs_cat.shape[0]),
        "absolute_std": [float(v) for v in abs_cat.std(axis=0)],
        "relative_std": [float(v) for v in rel_cat.std(axis=0)],
        "absolute_norm_median": float(np.median(np.linalg.norm(abs_cat, axis=1))),
        "relative_norm_median": float(np.median(np.linalg.norm(rel_cat, axis=1))),
    }


def summarize_control_point_scale(dst_files):
    local_cp = []
    for dst in dst_files:
        dst_arr = reshape_tree(np.load(dst))
        local_cp.append(np.asarray(dst_arr[:, 4:28], dtype=np.float32))
    if not local_cp:
        return {"files": 0}
    cp_cat = np.concatenate(local_cp, axis=0)
    return {
        "control_points_std": [float(v) for v in cp_cat.std(axis=0)[:6]],
        "control_points_abs_median": float(np.median(np.abs(cp_cat))),
    }


def convert_split(
    input_dir,
    output_dir,
    pattern,
    overwrite,
    position_slice,
    relative_positions,
    node_local_control_points,
    control_point_slices,
):
    os.makedirs(output_dir, exist_ok=True)
    files = iter_files(input_dir, pattern)
    written = 0
    skipped = 0
    outputs = []

    for src in files:
        dst = os.path.join(output_dir, os.path.basename(src))
        outputs.append(dst)
        if os.path.exists(dst) and not overwrite:
            skipped += 1
            continue

        arr = reshape_tree(np.load(src))
        rel = absolute_tree_to_local_geometry(
            arr,
            position_slice=position_slice,
            control_point_slices=control_point_slices,
            relative_positions=relative_positions,
            node_local_control_points=node_local_control_points,
            copy=True,
        )
        recon = local_geometry_tree_to_absolute(
            rel,
            position_slice=position_slice,
            control_point_slices=control_point_slices,
            relative_positions=relative_positions,
            node_local_control_points=node_local_control_points,
            copy=True,
        )
        err = float(np.max(np.abs(recon[:, 1:28] - arr[:, 1:28])))
        if err > 1e-5:
            raise RuntimeError(f"Relative position roundtrip failed for {src} with max error {err:.6g}")
        np.save(dst, rel.astype(np.float32, copy=False))
        written += 1

    return files, outputs, written, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert preorder k-count tree datasets to parent-relative xyz positions."
    )
    parser.add_argument("--config", default="convert_dataset_parent_relative_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    train_dir = paths.get("train_dir")
    val_dir = paths.get("val_dir")
    test_dir = paths.get("test_dir")
    output_root = paths.get("output_root")
    metadata_path = paths.get("metadata_path")

    if not train_dir or not val_dir or not test_dir or not output_root:
        raise ValueError("paths.train_dir, paths.val_dir, paths.test_dir and paths.output_root are required.")

    pattern = str(params.get("pattern", "*.npy"))
    overwrite = bool(params.get("overwrite", False))
    position_slice_cfg = params.get("position_slice", [1, 4])
    if not isinstance(position_slice_cfg, (list, tuple)) or len(position_slice_cfg) != 2:
        raise ValueError("params.position_slice must be a two-element list like [1, 4].")
    position_slice = (int(position_slice_cfg[0]), int(position_slice_cfg[1]))
    control_point_slices_cfg = params.get("control_point_slices", [[4, 12], [12, 20], [20, 28]])
    if not isinstance(control_point_slices_cfg, (list, tuple)) or len(control_point_slices_cfg) != 3:
        raise ValueError("params.control_point_slices must contain exactly three slice pairs.")
    control_point_slices = tuple((int(v[0]), int(v[1])) for v in control_point_slices_cfg)
    relative_positions = bool(params.get("relative_positions", True))
    node_local_control_points = bool(params.get("node_local_control_points", False))

    split_paths = {
        "train": (train_dir, os.path.join(output_root, "train")),
        "val": (val_dir, os.path.join(output_root, "val")),
        "test": (test_dir, os.path.join(output_root, "test")),
    }

    summary = {
        "config_path": os.path.abspath(args.config),
        "position_slice": list(position_slice),
        "control_point_slices": [list(v) for v in control_point_slices],
        "pattern": pattern,
        "overwrite": overwrite,
        "relative_positions": relative_positions,
        "node_local_control_points": node_local_control_points,
        "splits": {},
    }

    for split_name, (input_dir, output_dir) in split_paths.items():
        src_files, out_files, written, skipped = convert_split(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            overwrite=overwrite,
            position_slice=position_slice,
            relative_positions=relative_positions,
            node_local_control_points=node_local_control_points,
            control_point_slices=control_point_slices,
        )
        split_summary = summarize_position_delta(src_files, out_files)
        if node_local_control_points:
            split_summary.update(summarize_control_point_scale(out_files))
        split_summary["input_dir"] = input_dir
        split_summary["output_dir"] = output_dir
        split_summary["written"] = int(written)
        split_summary["skipped"] = int(skipped)
        summary["splits"][split_name] = split_summary
        print(
            f"{split_name}: files={len(src_files)} written={written} skipped={skipped} "
            f"| abs_std={split_summary.get('absolute_std')} rel_std={split_summary.get('relative_std')}"
        )

    if not metadata_path:
        metadata_path = os.path.join(output_root, "parent_relative_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
