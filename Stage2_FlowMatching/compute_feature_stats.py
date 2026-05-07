"""
Compute per-feature mean and std from the training data.

Iterates over all .npy files in the training folder, collects all valid
geometry features [1:40] (the 39 geom dims), and outputs per-feature
mean and std.  These are used for z-score normalization in Flow Matching.

Usage:
    python Stage2_FlowMatching/compute_feature_stats.py \
        --train_dir derived_data/TreesSplines_k_count_100depth_prepared_norm_v4_relpos_nodecp_v1/train \
        --output Stage2_FlowMatching/feature_stats.npz

    # For absolute positions (no parent-relative):
    python Stage2_FlowMatching/compute_feature_stats.py \
        --train_dir derived_data/TreesSplines_k_count_100depth_prepared_norm_v4_relpos_nodecp_v1/train \
        --output Stage2_FlowMatching/feature_stats_abspos.npz \
        --absolute_positions
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree_functions import parent_relative_positions_to_absolute

GEOM_START, GEOM_END = 1, 40  # same as train_flow_matching.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to training .npy folder")
    parser.add_argument("--output", default="Stage2_FlowMatching/feature_stats.npz",
                        help="Output path for feature_stats.npz")
    parser.add_argument("--absolute_positions", action="store_true",
                        help="Convert parent-relative positions to absolute before computing stats")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    files = sorted(
        train_dir / f
        for f in os.listdir(train_dir)
        if f.endswith(".npy") and not f.startswith(".")
    )
    print(f"Found {len(files)} .npy files in {train_dir}")
    if args.absolute_positions:
        print("Mode: ABSOLUTE positions (converting from parent-relative)")
    else:
        print("Mode: RELATIVE positions (raw file format)")

    # Welford's online algorithm for mean/variance (memory-efficient)
    n_total = 0
    mean = np.zeros(39, dtype=np.float64)
    M2 = np.zeros(39, dtype=np.float64)

    for i, fp in enumerate(files):
        arr = np.load(fp).astype(np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        # Strip trailing all-zero rows (same as dataset)
        valid = ~(np.all(np.abs(arr[:, 1:]) < 1e-8, axis=1))
        arr = arr[valid]
        if arr.shape[0] == 0:
            continue

        # Optionally convert parent-relative → absolute positions
        if args.absolute_positions:
            arr = parent_relative_positions_to_absolute(
                arr, position_slice=(1, 4), copy=True
            ).astype(np.float64)

        geom = arr[:, GEOM_START:GEOM_END]  # [N, 39]

        for row in geom:
            n_total += 1
            delta = row - mean
            mean += delta / n_total
            delta2 = row - mean
            M2 += delta * delta2

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(files)}, {n_total} nodes so far")

    variance = M2 / max(n_total - 1, 1)
    std = np.sqrt(variance)

    # Clamp std to avoid division by zero (min std = 1e-6)
    std = np.maximum(std, 1e-6)

    print(f"\nTotal nodes: {n_total}")
    print(f"Feature stats (39 dims):")
    print(f"  mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"  std  range: [{std.min():.6f}, {std.max():.6f}]")

    # Print per-group stats
    pos_label = "abs_pos" if args.absolute_positions else "rel_pos"
    groups = {
        f"{pos_label} [0:3]": (0, 3),
        "cp_x [3:11]": (3, 11),
        "cp_y [11:19]": (11, 19),
        "cp_z [19:27]": (19, 27),
        "knots [27:39]": (27, 39),
    }
    for name, (a, b) in groups.items():
        print(f"  {name}: mean={mean[a:b].mean():.6f} ± {mean[a:b].std():.6f}, "
              f"std={std[a:b].mean():.6f} ± {std[a:b].std():.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
