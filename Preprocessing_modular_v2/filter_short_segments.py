"""Remove degenerate short parent->child segments from prepared TreesSplines.

Inputs are (N, 40) arrays with columns:
    [0]      : k_count (0/1/2)
    [1:4]    : node position (x,y,z)
    [4:12]   : 8 control points x
    [12:20]  : 8 control points y
    [20:28]  : 8 control points z
    [28:40]  : 12 knots

Topology stored in pre-order DFS (root first, left subtree, right subtree).

We merge a child C into its parent P when
    dist(P, C) / radius(P_ring)  <  alpha
with two thresholds depending on parent's k_count:
    parent k=1 (continuation) → alpha_cont (default 0.5)
    parent k=2 (bifurcation)  → alpha_bif  (default 0.15)

Merge cases (C has k_C children):
    k_C == 0 (leaf)  : delete C, parent.k -= 1
    k_C == 1         : replace C with its single child (grandchild promoted)
    k_C == 2         : only if parent.k == 1 (would be valid binary).
                       Otherwise skip (would create 3 children).

The filter iterates per-parent until no violator remains, then recurses.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tree_functions import Tree, deserialize, serialize_pre_order_kcount  # noqa: E402


# ── (de)serialization helpers ──────────────────────────────────────────────

def array_to_tree(arr: np.ndarray, k: int = 39) -> Optional[Tree]:
    """Deserialize a (N, k+1) pre-order kcount array into a Tree."""
    if arr.ndim != 2 or arr.shape[1] != k + 1:
        raise ValueError(f"Expected shape (N, {k + 1}), got {arr.shape}")
    serial = arr.flatten().astype(np.float32).tolist()
    return deserialize(serial, mode="pre_order_kcount", k=k)


def tree_to_array(tree: Optional[Tree], k: int = 39) -> np.ndarray:
    """Serialize a tree back to a (N, k+1) pre-order kcount array."""
    if tree is None:
        return np.zeros((0, k + 1), dtype=np.float32)
    serial = serialize_pre_order_kcount(tree, k=k)
    flat = np.asarray(serial, dtype=np.float32)
    if flat.size % (k + 1) != 0:
        raise RuntimeError(
            f"Serialized length {flat.size} is not a multiple of row size {k + 1}"
        )
    return flat.reshape(-1, k + 1)


# ── geometry helpers ───────────────────────────────────────────────────────

def _ring_radius(node: Tree) -> float:
    r = node.data["r"]
    cp = np.asarray(r[:24], dtype=np.float64).reshape(3, 8)
    centroid = cp.mean(axis=1, keepdims=True)
    centered = cp - centroid
    return float(np.sqrt(((centered ** 2).sum(0)).mean()))


def _pos(node: Tree) -> np.ndarray:
    return np.array([node.data["x"], node.data["y"], node.data["z"]], dtype=np.float64)


def _k_count(node: Optional[Tree]) -> int:
    if node is None:
        return 0
    return int(node.left is not None) + int(node.right is not None)


# ── filtering ──────────────────────────────────────────────────────────────

class FilterStats:
    def __init__(self) -> None:
        self.removed_leaf = 0
        self.removed_single = 0
        self.removed_bif_promoted = 0
        self.skipped_bif_under_bif = 0

    def total(self) -> int:
        return self.removed_leaf + self.removed_single + self.removed_bif_promoted

    def asdict(self) -> dict:
        return {
            "removed_leaf": self.removed_leaf,
            "removed_single": self.removed_single,
            "removed_bif_promoted": self.removed_bif_promoted,
            "skipped_bif_under_bif": self.skipped_bif_under_bif,
            "removed_total": self.total(),
        }


def _filter_node(
    node: Optional[Tree],
    alpha_cont: float,
    alpha_bif: float,
    eps_radius: float,
    stats: FilterStats,
) -> None:
    """Iteratively merge violating children of `node`, then recurse."""
    if node is None:
        return

    rp = _ring_radius(node)
    if rp <= eps_radius:
        # No reliable scale → just recurse.
        _filter_node(node.left, alpha_cont, alpha_bif, eps_radius, stats)
        _filter_node(node.right, alpha_cont, alpha_bif, eps_radius, stats)
        return

    parent_pos = _pos(node)

    # Iterate until both slots are stable.
    changed = True
    safety = 0
    while changed:
        changed = False
        safety += 1
        if safety > 64:
            break
        kp = _k_count(node)
        threshold = alpha_cont if kp == 1 else alpha_bif

        for slot in ("left", "right"):
            child: Optional[Tree] = getattr(node, slot)
            if child is None:
                continue
            d = float(np.linalg.norm(_pos(child) - parent_pos))
            if d >= threshold * rp:
                continue

            kc = _k_count(child)
            if kc == 0:
                setattr(node, slot, None)
                stats.removed_leaf += 1
                changed = True
                break
            if kc == 1:
                grand = child.left if child.left is not None else child.right
                setattr(node, slot, grand)
                stats.removed_single += 1
                changed = True
                break
            # kc == 2
            if kp == 1:
                # Parent has only this child → promote both grandchildren.
                other = "right" if slot == "left" else "left"
                setattr(node, slot, child.left)
                setattr(node, other, child.right)
                stats.removed_bif_promoted += 1
                changed = True
                break
            # Parent already has another child → can't promote, skip merge.
            stats.skipped_bif_under_bif += 1
            # Don't loop forever on this slot.
            continue

    _filter_node(node.left, alpha_cont, alpha_bif, eps_radius, stats)
    _filter_node(node.right, alpha_cont, alpha_bif, eps_radius, stats)


def filter_tree(
    arr: np.ndarray,
    alpha_cont: float = 0.5,
    alpha_bif: float = 0.15,
    eps_radius: float = 1e-6,
    k: int = 39,
) -> Tuple[np.ndarray, FilterStats]:
    tree = array_to_tree(arr, k=k)
    stats = FilterStats()
    if tree is not None:
        _filter_node(tree, alpha_cont, alpha_bif, eps_radius, stats)
    return tree_to_array(tree, k=k), stats


# ── split processing ───────────────────────────────────────────────────────

def process_split(
    input_dir: str,
    output_dir: str,
    pattern: str,
    overwrite: bool,
    alpha_cont: float,
    alpha_bif: float,
    eps_radius: float,
    k: int,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "n_files": len(files),
        "n_written": 0,
        "n_skipped": 0,
        "nodes_in": 0,
        "nodes_out": 0,
        "removed_leaf": 0,
        "removed_single": 0,
        "removed_bif_promoted": 0,
        "skipped_bif_under_bif": 0,
    }
    for src in files:
        dst = os.path.join(output_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            summary["n_skipped"] += 1
            continue
        arr = np.load(src).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, k + 1)
        out, stats = filter_tree(
            arr,
            alpha_cont=alpha_cont,
            alpha_bif=alpha_bif,
            eps_radius=eps_radius,
            k=k,
        )
        np.save(dst, out.astype(np.float32, copy=False))
        summary["n_written"] += 1
        summary["nodes_in"] += int(arr.shape[0])
        summary["nodes_out"] += int(out.shape[0])
        summary["removed_leaf"] += stats.removed_leaf
        summary["removed_single"] += stats.removed_single
        summary["removed_bif_promoted"] += stats.removed_bif_promoted
        summary["skipped_bif_under_bif"] += stats.skipped_bif_under_bif
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter degenerate short segments from prepared TreesSplines arrays."
    )
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    params = cfg.get("params", {})

    alpha_cont = float(params.get("alpha_cont", 0.5))
    alpha_bif = float(params.get("alpha_bif", 0.15))
    eps_radius = float(params.get("eps_radius", 1e-6))
    pattern = str(params.get("pattern", "*.npy"))
    overwrite = bool(params.get("overwrite", False))
    k = int(params.get("k", 39))

    out_root = paths["output_root"]
    os.makedirs(out_root, exist_ok=True)
    metadata_path = paths.get(
        "metadata_path", os.path.join(out_root, "filter_short_segments_metadata.json")
    )

    splits_summary = {}
    for split in ("train", "val", "test"):
        in_key = f"{split}_dir"
        if in_key not in paths:
            continue
        out_dir = os.path.join(out_root, split)
        t0 = time.time()
        s = process_split(
            input_dir=paths[in_key],
            output_dir=out_dir,
            pattern=pattern,
            overwrite=overwrite,
            alpha_cont=alpha_cont,
            alpha_bif=alpha_bif,
            eps_radius=eps_radius,
            k=k,
        )
        s["elapsed_s"] = round(time.time() - t0, 2)
        splits_summary[split] = s
        nodes_kept_pct = (
            100.0 * s["nodes_out"] / s["nodes_in"] if s["nodes_in"] else 0.0
        )
        print(
            f"[{split:5}] files={s['n_files']:5d} written={s['n_written']:5d} "
            f"skipped={s['n_skipped']:5d} nodes {s['nodes_in']:>7,} -> "
            f"{s['nodes_out']:>7,} ({nodes_kept_pct:.2f}% kept) "
            f"removed leaf={s['removed_leaf']} single={s['removed_single']} "
            f"bif_promoted={s['removed_bif_promoted']} "
            f"bif_skipped={s['skipped_bif_under_bif']}  ({s['elapsed_s']}s)"
        )

    metadata = {
        "config_path": os.path.abspath(args.config),
        "params": {
            "alpha_cont": alpha_cont,
            "alpha_bif": alpha_bif,
            "eps_radius": eps_radius,
            "pattern": pattern,
            "overwrite": overwrite,
            "k": k,
        },
        "splits": splits_summary,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
