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


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize
from view_trees import collect_nodes_edges, collect_branches, export_vtp


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


def default_output_dir(input_path):
    if os.path.isdir(input_path):
        return os.path.join(os.path.dirname(input_path), "Views")
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "Views")


def build_out_path(output_dir, stem, suffix):
    return os.path.join(output_dir, f"{stem}{suffix}.vtp")


def main():
    parser = argparse.ArgumentParser(
        description="Batch export centerline/spline VTP views using the exact view_trees export logic."
    )
    parser.add_argument("--config", default="views_dataset_export_config.yaml", help="Path to YAML config")
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

    save_combined = bool(params.get("save_combined", True))
    save_centerline = bool(params.get("save_centerline", False))
    save_splines = bool(params.get("save_splines", False))

    if not (save_combined or save_centerline or save_splines):
        raise ValueError("Enable at least one of save_combined/save_centerline/save_splines")

    combined_suffix = str(params.get("combined_suffix", ""))
    centerline_suffix = str(params.get("centerline_suffix", "_centerline"))
    splines_suffix = str(params.get("splines_suffix", "_splines"))

    normalize_xyz = bool(params.get("normalize_xyz", False))

    success = 0
    skipped = 0
    failed = 0
    total = len(files)

    for idx, file_path in enumerate(files, start=1):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        print(f"[{idx}/{total}] Processing {stem}")

        try:
            tree = load_tree(file_path, k=k, mode=mode)

            nodes = []
            edges = []
            collect_nodes_edges(tree, nodes, edges)

            branches = []
            collect_branches(tree, [], branches)

            root_pos = np.array([tree.data["x"], tree.data["y"], tree.data["z"]], dtype=np.float32)

            targets = []
            if save_combined:
                targets.append((build_out_path(output_dir, stem, combined_suffix), None, None))
            if save_centerline:
                targets.append((build_out_path(output_dir, stem, centerline_suffix), True, False))
            if save_splines:
                targets.append((build_out_path(output_dir, stem, splines_suffix), False, True))

            case_written = False
            for out_path, include_centerline, include_splines in targets:
                if os.path.exists(out_path) and not overwrite:
                    print(f"  skip exists: {out_path}")
                    skipped += 1
                    continue

                export_vtp(
                    out_path,
                    branches,
                    nodes,
                    root_pos,
                    params,
                    normalize_xyz,
                    include_centerline=include_centerline,
                    include_splines=include_splines,
                )
                print(f"  wrote: {out_path}")
                case_written = True

            if case_written:
                success += 1

        except Exception as exc:
            failed += 1
            print(f"  failed: {file_path} ({exc})")
            if bool(params.get("traceback_on_error", False)):
                traceback.print_exc()
            if not continue_on_error:
                raise

    print(
        f"Done. cases={total}, success={success}, failed={failed}, skipped_outputs={skipped}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
