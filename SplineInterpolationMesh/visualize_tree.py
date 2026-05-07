#!/usr/bin/env python3
"""
visualize_tree.py — PNG visualisation of vessel tree with original
B-spline rings (left) and interpolated splines + centerlines (right).

Same matplotlib style as the Stage2 comparison PNGs.

Usage:
    python SplineInterpolationMesh/visualize_tree.py \
        --input derived_data/.../aneux_C0028a.npy \
        --output SplineInterpolationMesh/output/images/aneux_C0028a.png

    # Batch (all val files):
    python SplineInterpolationMesh/visualize_tree.py \
        --input derived_data/.../val \
        --output_dir SplineInterpolationMesh/output/images
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize, local_geometry_tree_to_absolute
from reconstruct_mesh import get_segments
from SplineInterpolationMesh.interpolation import interpolate_tree_segments


# ── Helpers ──────────────────────────────────────────────────────────────


def load_tree(path: str):
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape((-1, 40))
    abs_data = local_geometry_tree_to_absolute(
        data,
        position_slice=(1, 4),
        control_point_slices=((4, 12), (12, 20), (20, 28)),
        relative_positions=True,
        node_local_control_points=True,
    )
    serial = list(abs_data.flatten())
    tree = deserialize(serial, mode="pre_order_kcount", k=39)
    return tree, abs_data


def sample_spline(coeffs, n_samples=50):
    """Sample points from B-spline coefficients (36 values)."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if len(coeffs) < 36:
        return None
    t = coeffs[24:36].copy()
    t = np.where(np.abs(t - 1.0) < 0.01, 1.0, t)
    cx, cy, cz = coeffs[0:8], coeffs[8:16], coeffs[16:24]
    tck = (t, [cx, cy, cz], 3)
    u = np.linspace(0, 1, n_samples)
    try:
        x, y, z = splev(u, tck)
        pts = np.column_stack((x, y, z))
        if np.any(np.abs(pts) > 100):
            return None
        return pts
    except Exception:
        return None


def draw_tree_splines(tree, ax, color='b', n_samples=60, s=0.5):
    """Recursively draw original B-spline rings as scatter."""
    if tree is None:
        return
    d = tree.data
    if hasattr(d, 'get') and "r" in d and d["r"] is not None:
        coeffs = d["r"]
        if isinstance(coeffs, (list, np.ndarray)) and len(coeffs) == 36:
            pts = sample_spline(np.array(coeffs), n_samples=n_samples)
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, marker='.', s=s)
    draw_tree_splines(tree.left, ax, color=color, n_samples=n_samples, s=s)
    draw_tree_splines(tree.right, ax, color=color, n_samples=n_samples, s=s)


def draw_tree_centerlines(tree, ax, color='gray', linewidth=1.0, alpha=0.6,
                          parent_pos=None):
    """Draw edges from parent to child node positions."""
    if tree is None:
        return
    d = tree.data
    if hasattr(d, 'get'):
        pos = np.array([d["x"], d["y"], d["z"]], dtype=float)
        if parent_pos is not None:
            ax.plot([parent_pos[0], pos[0]],
                    [parent_pos[1], pos[1]],
                    [parent_pos[2], pos[2]],
                    color=color, linewidth=linewidth, alpha=alpha)
        draw_tree_centerlines(tree.left, ax, color=color, linewidth=linewidth,
                              alpha=alpha, parent_pos=pos)
        draw_tree_centerlines(tree.right, ax, color=color, linewidth=linewidth,
                              alpha=alpha, parent_pos=pos)


def sync_3d_axes(*axes):
    all_lims = []
    for ax in axes:
        all_lims.extend([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    lo = min(l[0] for l in all_lims)
    hi = max(l[1] for l in all_lims)
    m = (hi - lo) * 0.05
    for ax in axes:
        ax.set_xlim3d(lo - m, hi + m)
        ax.set_ylim3d(lo - m, hi + m)
        ax.set_zlim3d(lo - m, hi + m)


# ── Plot one tree ────────────────────────────────────────────────────────


def plot_tree_png(
    input_path: str,
    output_path: str,
    target_spacing: float = 0.005,
    n_ring_pts: int = 64,
    ring_stride: int = 4,
):
    tree, abs_data = load_tree(input_path)
    name = os.path.splitext(os.path.basename(input_path))[0]
    n_nodes = abs_data.shape[0]

    segments_raw = get_segments(tree, k=40)
    seg_data = interpolate_tree_segments(
        segments_raw,
        target_spacing=target_spacing,
        n_ring_pts=n_ring_pts,
    )
    total_interp = sum(len(s["centers"]) for s in seg_data)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"{name}  —  {n_nodes} nodes, {total_interp} interpolated stations",
                 fontsize=12)

    # ── Left: Original tree (same style as Stage2) ───────────────────
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Tree (B-Spline Rings)")
    draw_tree_centerlines(tree, ax1, color='darkblue', linewidth=1.2, alpha=0.5)
    draw_tree_splines(tree, ax1, color='blue', n_samples=60, s=0.5)

    # ── Right: Interpolated ──────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Interpolated (Centerline + Cross-Sections)")

    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(len(seg_data), 1)))

    for si, sd in enumerate(seg_data):
        centers = sd["centers"]
        rings = sd["rings"]
        col = seg_colors[si % len(seg_colors)]

        # Interpolated centerline
        ax2.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                 color=col, linewidth=1.5, alpha=0.8)

        # Interpolated rings (every ring_stride-th)
        for ri in range(0, len(rings), ring_stride):
            ring = rings[ri]
            ring_closed = np.vstack([ring, ring[0]])
            ax2.plot(ring_closed[:, 0], ring_closed[:, 1], ring_closed[:, 2],
                     color=col, linewidth=0.4, alpha=0.4)

        # Original node positions as markers
        seg_raw = segments_raw[si] if si < len(segments_raw) else None
        if seg_raw is not None:
            nodes = seg_raw[:, :3]
            ax2.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        c='green', marker='*', s=30, zorder=10)

    sync_3d_axes(ax1, ax2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npy file or directory")
    parser.add_argument("--output", default=None, help="Output PNG (single file mode)")
    parser.add_argument("--output_dir", default=None, help="Output dir (batch mode)")
    parser.add_argument("--target_spacing", type=float, default=0.005)
    parser.add_argument("--n_ring_pts", type=int, default=64)
    parser.add_argument("--ring_stride", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.npy")))
        if args.max_files:
            files = files[:args.max_files]
        out_dir = args.output_dir or "SplineInterpolationMesh/output/images"
        print(f"Batch: {len(files)} files → {out_dir}")
        for i, f in enumerate(files):
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(out_dir, base + ".png")
            print(f"  [{i+1}/{len(files)}] {base}")
            try:
                plot_tree_png(f, out_path,
                              target_spacing=args.target_spacing,
                              n_ring_pts=args.n_ring_pts,
                              ring_stride=args.ring_stride)
            except Exception as e:
                print(f"    FAIL: {e}")
    else:
        out_path = args.output
        if out_path is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            out_path = f"SplineInterpolationMesh/output/images/{base}.png"
        plot_tree_png(args.input, out_path,
                      target_spacing=args.target_spacing,
                      n_ring_pts=args.n_ring_pts,
                      ring_stride=args.ring_stride)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
