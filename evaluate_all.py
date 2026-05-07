#!/usr/bin/env python3
"""
Comprehensive evaluation of all vessel tree generation models.

Computes:
  A. Per-tree reconstruction metrics (gen vs paired GT):
       pos_mae, cp_mae, knot_mae, total_mse, radius_ratio
  B. Physiological plausibility (per-tree + average):
       Murray's law violation, bifurcation angles, tapering violations,
       symmetry ratio
  C. Geometric feature distributions → distributional metrics:
       MMD, Coverage, 1-NNA between generated and full GT validation set
  D. Diversity: intra-set pairwise distance

Usage:
    conda run -n vmtk_2 python evaluate_all.py [--gt_val_dir DIR]

Output:
    evaluation_results/
        summary_table.csv        – one row per model, all metrics
        per_tree_metrics.csv     – per-tree breakdown
        distributional.csv       – MMD / COV / 1-NNA
        summary_table.txt        – pretty-printed table
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree_functions import preorder_kcount_parent_indices, local_geometry_tree_to_absolute


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration: registered models and their generated directories
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "v8 (Baseline)": "Stage2_FlowMatching/generated/flow_v8_fix5/npy",
    "Physio v1":     "Stage2_FlowMatching_Physio/generated/physio_v1/npy",
    "Physio v2":     "Stage2_FlowMatching_Physio/generated/physio_v2/npy",
    "Physio v3":     "Stage2_FlowMatching_Physio/generated/physio_v3/npy",
    "Physio v4":     "Stage2_FlowMatching_Physio/generated/physio_v4/npy",
    "Physio v5":     "Stage2_FlowMatching_Physio/generated/physio_v5/npy",
    "AR FM v1":      "Stage2_AutoregressiveFM/generated/ar_fm_v1/npy",
    "Latent v1":     "Stage2_LatentTreeDiffusion/generated/latent_v1/npy",
    "Hier FM v1":    "Stage2_HierarchicalFM/generated/hier_v1/npy",
    "Hier FM v3":    "Stage2_HierarchicalFM/generated/hier_v3/npy",
    "Hier FM v4":    "Stage2_HierarchicalFM/generated/hier_v4/npy",
    "Branch FM v1":  "Stage2_BranchFM/generated/branch_v1/npy",
    "AR-Level v1":   "Stage2_ARLevel/generated/ar_level_v1/npy",
    "AR-Level v2":   "Stage2_ARLevel/generated/ar_level_v2/npy",
    "Guided v1":     "Stage2_FlowMatching/generated/guided_v1/npy",
    "Guided v2":     "Stage2_FlowMatching/generated/guided_v2/npy",
    "WFM v1":        "Stage2_WavefrontFM/generated/wfm_v1/npy",
    "WFM v2":        "Stage2_WavefrontFM/generated/wfm_v2/npy",
    "WFM v2 Guided": "Stage2_WavefrontFM/generated/wfm_v2_guided/npy",
    "WFM v2 200s":  "Stage2_WavefrontFM/generated/wfm_v2_200steps/npy",
    "AneuCond v1":   "Stage2_AneuCondFM/generated/aneucond_v1/npy",
    "AneuCond v2":   "Stage2_AneuCondFM/generated/aneucond_v2/npy",
    "WFM+Aneu v1":   "Stage2_AneuCondFM/generated/wfm_aneucond_v1/npy",
    "TwoStage v1":   "Stage2_TwoStageFM/generated/twostage_v1/npy",
}

DEFAULT_GT_VAL_DIR = "derived_data/TreesSplines_k_count_100depth_prepared_norm_v4_relpos_nodecp_v1/val"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_generated_pairs(npy_dir):
    """
    Load generated .npy and paired GT from a model's npy/ directory.
    Returns list of (name, gen_abs [N,40], gt_abs [N,40]).
    """
    npy_dir = Path(npy_dir)
    if not npy_dir.exists():
        print(f"  WARNING: {npy_dir} does not exist, skipping")
        return []

    pairs = []
    seen = set()
    for f in sorted(npy_dir.iterdir()):
        if not f.name.endswith(".npy"):
            continue
        # Skip GT / local variants
        if "_gt" in f.stem or "_local" in f.stem:
            continue
        name = f.stem
        if name in seen:
            continue
        seen.add(name)

        gen = np.load(f).astype(np.float32)
        gt_path = npy_dir / f"{name}_gt.npy"
        if not gt_path.exists():
            continue
        gt = np.load(gt_path).astype(np.float32)

        if gen.ndim == 1:
            gen = gen.reshape(-1, 40)
        if gt.ndim == 1:
            gt = gt.reshape(-1, 40)
        if gen.shape[1] != 40 or gt.shape[1] != 40:
            continue

        pairs.append((name, gen, gt))

    return pairs


def load_full_gt_validation(val_dir, max_trees=None):
    """
    Load the full GT validation set (raw local-coordinate .npy files)
    and convert to absolute coordinates. Returns list of [N, 40] arrays.
    """
    val_dir = Path(val_dir)
    files = sorted(val_dir.glob("*.npy"))
    if max_trees:
        files = files[:max_trees]

    trees = []
    for fp in files:
        arr = np.load(fp).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        # Filter zero-padding
        valid = ~(np.all(np.abs(arr[:, 1:]) < 1e-8, axis=1))
        arr = arr[valid]
        if arr.shape[0] < 2:
            continue
        # Convert local → absolute
        arr_abs = local_geometry_tree_to_absolute(
            arr,
            position_slice=(1, 4),
            control_point_slices=((4, 12), (12, 20), (20, 28)),
            relative_positions=True,
            node_local_control_points=True,
            copy=True,
        )
        trees.append(arr_abs)

    return trees


# ═══════════════════════════════════════════════════════════════════════════════
# A. Per-tree reconstruction metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reconstruction_metrics(gen_abs, gt_abs):
    """Per-tree reconstruction metrics between generated and GT."""
    n = min(len(gen_abs), len(gt_abs))
    pos_gen = gen_abs[:n, 1:4]
    pos_gt = gt_abs[:n, 1:4]
    cp_gen = gen_abs[:n, 4:28]
    cp_gt = gt_abs[:n, 4:28]
    knot_gen = gen_abs[:n, 28:40]
    knot_gt = gt_abs[:n, 28:40]

    pos_mae = np.abs(pos_gen - pos_gt).mean()
    cp_mae = np.abs(cp_gen - cp_gt).mean()
    knot_mae = np.abs(knot_gen - knot_gt).mean()
    total_mse = ((gen_abs[:n] - gt_abs[:n]) ** 2).mean()

    # CP radius (absolute coords: subtract node position)
    cp_x_gen = cp_gen[:, 0:8] - pos_gen[:, 0:1]
    cp_y_gen = cp_gen[:, 8:16] - pos_gen[:, 1:2]
    cp_z_gen = cp_gen[:, 16:24] - pos_gen[:, 2:3]
    gen_radius = np.sqrt(cp_x_gen**2 + cp_y_gen**2 + cp_z_gen**2).mean()

    cp_x_gt = cp_gt[:, 0:8] - pos_gt[:, 0:1]
    cp_y_gt = cp_gt[:, 8:16] - pos_gt[:, 1:2]
    cp_z_gt = cp_gt[:, 16:24] - pos_gt[:, 2:3]
    gt_radius = np.sqrt(cp_x_gt**2 + cp_y_gt**2 + cp_z_gt**2).mean()

    return {
        "pos_mae": float(pos_mae),
        "cp_mae": float(cp_mae),
        "knot_mae": float(knot_mae),
        "total_mse": float(total_mse),
        "gen_radius": float(gen_radius),
        "gt_radius": float(gt_radius),
        "radius_ratio": float(gen_radius / max(gt_radius, 1e-8)),
    }


def _safe_histogram_l1(hist_a, hist_b):
    hist_a = np.asarray(hist_a, dtype=np.float64)
    hist_b = np.asarray(hist_b, dtype=np.float64)
    if hist_a.shape != hist_b.shape:
        n = max(hist_a.shape[0], hist_b.shape[0])
        hist_a = np.pad(hist_a, (0, n - hist_a.shape[0]))
        hist_b = np.pad(hist_b, (0, n - hist_b.shape[0]))
    return float(np.abs(hist_a - hist_b).sum())


def _safe_wasserstein(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    values_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if values_a.size == 0 and values_b.size == 0:
        return 0.0
    if values_a.size == 0:
        values_a = np.array([0.0], dtype=np.float64)
    if values_b.size == 0:
        values_b = np.array([0.0], dtype=np.float64)
    return float(wasserstein_distance(values_a, values_b))


def _safe_f1(precision, recall):
    denom = precision + recall
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _compute_tree_span(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    return float(cdist(positions, positions).max())


def _safe_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float(default)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(finite.mean())


def _close_curve(points_2d):
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if len(points_2d) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if np.allclose(points_2d[0], points_2d[-1]):
        return points_2d
    return np.vstack([points_2d, points_2d[0]])


def _project_cross_section_to_2d(cp_xyz):
    cp_xyz = np.asarray(cp_xyz, dtype=np.float64)
    centroid = cp_xyz.mean(axis=0, keepdims=True)
    centered = cp_xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    coords_2d = centered @ basis
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    order = np.argsort(angles)
    return coords_2d[order]


def _sample_closed_polyline(points_2d, n_samples=64):
    curve = _close_curve(points_2d)
    if len(curve) <= 1:
        return np.zeros((max(n_samples, 1), 2), dtype=np.float64)
    segs = curve[1:] - curve[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    total = seg_lens.sum()
    if total < 1e-12:
        return np.repeat(curve[:1], max(n_samples, 1), axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    ts = np.linspace(0.0, total, max(n_samples, 2), endpoint=False, dtype=np.float64)
    out = []
    seg_idx = 0
    for t in ts:
        while seg_idx + 1 < len(cum) and t > cum[seg_idx + 1]:
            seg_idx += 1
        local_len = seg_lens[min(seg_idx, len(seg_lens) - 1)]
        alpha = 0.0 if local_len < 1e-12 else (t - cum[seg_idx]) / local_len
        out.append(curve[seg_idx] + alpha * segs[seg_idx])
    return np.asarray(out, dtype=np.float64)


def _polygon_area_perimeter(points_2d):
    curve = _close_curve(points_2d)
    if len(curve) <= 2:
        return 0.0, 0.0
    x = curve[:-1, 0]
    y = curve[:-1, 1]
    x_next = curve[1:, 0]
    y_next = curve[1:, 1]
    area = 0.5 * abs(np.sum(x * y_next - x_next * y))
    perimeter = float(np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum())
    return float(area), perimeter


def _eccentricity_from_points(points_2d):
    pts = np.asarray(points_2d, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    cov = np.cov(pts.T)
    eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    major = max(eigvals[0], 1e-12)
    minor = max(eigvals[-1], 0.0)
    ratio = np.clip(minor / major, 0.0, 1.0)
    return float(np.sqrt(max(1.0 - ratio, 0.0)))


def _cross_section_shape_stats(points_2d):
    area, perimeter = _polygon_area_perimeter(points_2d)
    circularity = 0.0
    if perimeter > 1e-12:
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
    eccentricity = _eccentricity_from_points(points_2d)
    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "eccentricity": float(eccentricity),
    }


def compute_cross_section_metrics(gen_abs, gt_abs, n_samples=64):
    n = min(len(gen_abs), len(gt_abs))
    chamfers = []
    hausdorffs = []
    area_errors = []
    perimeter_errors = []
    circularity_errors = []
    eccentricity_errors = []

    for idx in range(n):
        gen_cp = np.stack(
            [gen_abs[idx, 4:12], gen_abs[idx, 12:20], gen_abs[idx, 20:28]],
            axis=-1,
        )
        gt_cp = np.stack(
            [gt_abs[idx, 4:12], gt_abs[idx, 12:20], gt_abs[idx, 20:28]],
            axis=-1,
        )

        gen_2d = _project_cross_section_to_2d(gen_cp)
        gt_2d = _project_cross_section_to_2d(gt_cp)
        gen_samples = _sample_closed_polyline(gen_2d, n_samples=n_samples)
        gt_samples = _sample_closed_polyline(gt_2d, n_samples=n_samples)
        pairwise = chamfer_and_hausdorff(gen_samples, gt_samples)
        gen_stats = _cross_section_shape_stats(gen_2d)
        gt_stats = _cross_section_shape_stats(gt_2d)

        chamfers.append(pairwise["chamfer_mean"])
        hausdorffs.append(pairwise["hausdorff"])
        area_errors.append(abs(gen_stats["area"] - gt_stats["area"]))
        perimeter_errors.append(abs(gen_stats["perimeter"] - gt_stats["perimeter"]))
        circularity_errors.append(abs(gen_stats["circularity"] - gt_stats["circularity"]))
        eccentricity_errors.append(abs(gen_stats["eccentricity"] - gt_stats["eccentricity"]))

    return {
        "cross_section_2d_chamfer": _safe_mean(chamfers),
        "cross_section_2d_hausdorff": _safe_mean(hausdorffs),
        "cross_section_area_error": _safe_mean(area_errors),
        "cross_section_perimeter_error": _safe_mean(perimeter_errors),
        "cross_section_circularity_error": _safe_mean(circularity_errors),
        "cross_section_eccentricity_error": _safe_mean(eccentricity_errors),
    }


def sample_tree_centerline_points(tree_abs, step=0.05):
    """
    Sample centerline points along parent-child edges at roughly fixed spacing.
    Falls back to node positions for degenerate/short edges.
    """
    k_counts, parents = _get_tree_topology(tree_abs)
    positions = tree_abs[:, 1:4]
    points = [positions[0]]
    for i in range(1, len(tree_abs)):
        p = parents[i]
        if p < 0:
            points.append(positions[i])
            continue
        a = positions[p]
        b = positions[i]
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 1e-8:
            points.append(b)
            continue
        n_steps = max(2, int(np.ceil(length / max(step, 1e-4))) + 1)
        ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        seg_points = a[None, :] + ts[:, None] * edge[None, :]
        points.extend(seg_points[1:])
    return np.asarray(points, dtype=np.float64)


def chamfer_and_hausdorff(points_a, points_b):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_a_to_b = tree_b.query(points_a, workers=-1)[0]
    d_b_to_a = tree_a.query(points_b, workers=-1)[0]
    return {
        "chamfer_mean": float(0.5 * (d_a_to_b.mean() + d_b_to_a.mean())),
        "chamfer_a_to_b": float(d_a_to_b.mean()),
        "chamfer_b_to_a": float(d_b_to_a.mean()),
        "hausdorff": float(max(d_a_to_b.max(), d_b_to_a.max())),
        "hausdorff_p95": float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95))),
    }


def compute_centerline_metrics(gen_abs, gt_abs, step=0.05):
    gen_points = sample_tree_centerline_points(gen_abs, step=step)
    gt_points = sample_tree_centerline_points(gt_abs, step=step)
    pairwise = chamfer_and_hausdorff(gen_points, gt_points)

    gen_tree = cKDTree(gen_points)
    gt_tree = cKDTree(gt_points)
    d_gen_to_gt = gt_tree.query(gen_points, workers=-1)[0]
    d_gt_to_gen = gen_tree.query(gt_points, workers=-1)[0]

    gt_span = _compute_tree_span(gt_abs[:, 1:4])
    tau_1pct = max(gt_span * 0.01, 1e-4)
    tau_2pct = max(gt_span * 0.02, 2e-4)

    prec_1 = float((d_gen_to_gt <= tau_1pct).mean())
    rec_1 = float((d_gt_to_gen <= tau_1pct).mean())
    prec_2 = float((d_gen_to_gt <= tau_2pct).mean())
    rec_2 = float((d_gt_to_gen <= tau_2pct).mean())

    return {
        "centerline_acd": pairwise["chamfer_mean"],
        "centerline_cd": pairwise["chamfer_mean"],
        "centerline_hd": pairwise["hausdorff"],
        "centerline_hd95": pairwise["hausdorff_p95"],
        "centerline_precision_1pct": prec_1,
        "centerline_recall_1pct": rec_1,
        "centerline_f1_1pct": _safe_f1(prec_1, rec_1),
        "centerline_precision_2pct": prec_2,
        "centerline_recall_2pct": rec_2,
        "centerline_f1_2pct": _safe_f1(prec_2, rec_2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# B. Physiological plausibility metrics (numpy versions)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_node_radii_np(tree_abs):
    """
    Compute per-node RMS cross-section radius using centroid-based method.
    This matches physio_losses.py: radius = RMS distance from CP centroid.
    Translation-invariant — works on both local and absolute coordinates.
    """
    cp_x = tree_abs[:, 4:12]     # [N, 8]
    cp_y = tree_abs[:, 12:20]
    cp_z = tree_abs[:, 20:28]

    # Stack to [N, 8, 3]
    cps = np.stack([cp_x, cp_y, cp_z], axis=-1)

    # Centroid per node
    centroid = cps.mean(axis=1, keepdims=True)  # [N, 1, 3]
    centered = cps - centroid                     # [N, 8, 3]

    # RMS radius
    r2 = (centered**2).sum(axis=-1).mean(axis=-1)  # [N]
    return np.sqrt(r2 + 1e-10)


def _get_tree_topology(tree_abs):
    """Extract k_counts, parents from absolute tree data."""
    k_counts = np.clip(np.rint(tree_abs[:, 0]), 0, 2).astype(np.int64)
    parents = preorder_kcount_parent_indices(k_counts)
    return k_counts, parents


def _find_bifurcations(k_counts, parents, n_nodes):
    """Find bifurcation nodes and their two children."""
    bif_mask = (k_counts == 2)
    child_map = defaultdict(list)
    for i in range(n_nodes):
        p = parents[i]
        if p >= 0:
            child_map[p].append(i)

    bifs = []  # list of (parent_idx, child1_idx, child2_idx)
    for p_idx in range(n_nodes):
        if bif_mask[p_idx] and len(child_map[p_idx]) >= 2:
            bifs.append((p_idx, child_map[p_idx][0], child_map[p_idx][1]))
    return bifs


def _compute_tree_depths(parents):
    depths = np.zeros(len(parents), dtype=np.int64)
    for i in range(len(parents)):
        if parents[i] >= 0:
            depths[i] = depths[parents[i]] + 1
    return depths


def _build_child_map(parents):
    child_map = defaultdict(list)
    for idx, parent in enumerate(parents):
        if parent >= 0:
            child_map[parent].append(idx)
    return child_map


def _compute_components_and_betti(n_nodes, parents):
    adjacency = [[] for _ in range(n_nodes)]
    n_edges = 0
    for i, p in enumerate(parents):
        if p >= 0:
            adjacency[p].append(i)
            adjacency[i].append(p)
            n_edges += 1

    seen = np.zeros(n_nodes, dtype=bool)
    n_components = 0
    for start in range(n_nodes):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

    beta0 = n_components
    beta1 = max(n_edges - n_nodes + beta0, 0)
    return beta0, beta1


def _compute_laplacian_spectrum(tree_abs, n_eigs=10):
    k_counts, parents = _get_tree_topology(tree_abs)
    n_nodes = len(tree_abs)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, p in enumerate(parents):
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0
    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj
    eigvals = np.linalg.eigvalsh(lap)
    eigvals = np.sort(np.real(eigvals))
    if len(eigvals) < n_eigs:
        eigvals = np.pad(eigvals, (0, n_eigs - len(eigvals)), mode="constant")
    return eigvals[:n_eigs]


def extract_distribution_descriptors(tree_abs):
    """
    Per-tree descriptors for scalar and graph-distribution metrics.
    """
    k_counts, parents = _get_tree_topology(tree_abs)
    n_nodes = len(tree_abs)
    positions = tree_abs[:, 1:4]
    radii = _compute_node_radii_np(tree_abs)
    depths = _compute_tree_depths(parents)
    beta0, beta1 = _compute_components_and_betti(n_nodes, parents)

    child_counts = np.clip(k_counts.astype(np.int64), 0, 2)
    node_degrees = child_counts.copy()
    node_degrees[parents >= 0] += 1

    edge_lengths = []
    theta_vals = []
    phi_vals = []
    tortuosities = []
    for i, p in enumerate(parents):
        if p < 0:
            continue
        vec = positions[i] - positions[p]
        length = np.linalg.norm(vec)
        edge_lengths.append(length)
        if length > 1e-8:
            unit = vec / length
            theta_vals.append(np.arccos(np.clip(unit[2], -1.0, 1.0)))
            phi_vals.append(np.arctan2(unit[1], unit[0]))
        else:
            theta_vals.append(0.0)
            phi_vals.append(0.0)

        chain_len = length
        cur = i
        while child_counts[cur] == 1:
            nxt = np.where(parents == cur)[0]
            if len(nxt) != 1:
                break
            nxt = int(nxt[0])
            seg = np.linalg.norm(positions[nxt] - positions[cur])
            chain_len += seg
            cur = nxt
        straight = np.linalg.norm(positions[cur] - positions[p])
        tortuosities.append(float(chain_len / max(straight, 1e-8)))

    return {
        "x": positions[:, 0].astype(np.float64),
        "y": positions[:, 1].astype(np.float64),
        "z": positions[:, 2].astype(np.float64),
        "depth": depths.astype(np.float64),
        "degree": node_degrees.astype(np.float64),
        "radius": radii.astype(np.float64),
        "edge_length": np.asarray(edge_lengths if edge_lengths else [0.0], dtype=np.float64),
        "theta": np.asarray(theta_vals if theta_vals else [0.0], dtype=np.float64),
        "phi": np.asarray(phi_vals if phi_vals else [0.0], dtype=np.float64),
        "tortuosity": np.asarray(tortuosities if tortuosities else [1.0], dtype=np.float64),
        "beta0": np.asarray([beta0], dtype=np.float64),
        "beta1": np.asarray([beta1], dtype=np.float64),
        "lap_spec": _compute_laplacian_spectrum(tree_abs, n_eigs=10).astype(np.float64),
    }


def compute_topology_metrics(tree_abs):
    k_counts, parents = _get_tree_topology(tree_abs)
    n_nodes = len(tree_abs)
    child_counts = np.clip(k_counts.astype(np.int64), 0, 2)
    depths = _compute_tree_depths(parents)
    beta0, beta1 = _compute_components_and_betti(n_nodes, parents)
    child_map = _build_child_map(parents)

    degrees = child_counts.copy()
    degrees[parents >= 0] += 1
    degree_hist = np.bincount(np.clip(degrees, 0, 3), minlength=4).astype(np.float64)
    if n_nodes > 0:
        degree_hist /= float(n_nodes)

    n_edges = int(np.sum(parents >= 0))
    n_leaves = int(np.sum(child_counts == 0))
    n_bifurcations = int(np.sum(child_counts == 2))
    max_depth = int(depths.max()) if n_nodes > 0 else 0

    branch_counts_by_order = np.bincount(depths[child_counts == 2], minlength=max_depth + 1).astype(np.float64)

    bif_angles = []
    positions = tree_abs[:, 1:4]
    for parent_idx, children in child_map.items():
        if len(children) < 2:
            continue
        c1_idx, c2_idx = children[0], children[1]
        dir1 = positions[c1_idx] - positions[parent_idx]
        dir2 = positions[c2_idx] - positions[parent_idx]
        d1 = np.linalg.norm(dir1)
        d2 = np.linalg.norm(dir2)
        if d1 < 1e-10 or d2 < 1e-10:
            continue
        dir1 /= d1
        dir2 /= d2
        angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(dir1, dir2)), np.dot(dir1, dir2)))
        bif_angles.append(float(angle))

    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "n_terminal": float(n_leaves),
        "n_bifurcations_topo": float(n_bifurcations),
        "max_depth_topo": float(max_depth),
        "beta0_topo": float(beta0),
        "beta1_topo": float(beta1),
        "degree_hist": degree_hist,
        "branch_order_counts": branch_counts_by_order,
        "bif_angles": np.asarray(bif_angles, dtype=np.float64),
    }


def compute_topology_comparison_metrics(gen_abs, gt_abs):
    gen_topo = compute_topology_metrics(gen_abs)
    gt_topo = compute_topology_metrics(gt_abs)

    return {
        "node_count_abs_err": abs(gen_topo["n_nodes"] - gt_topo["n_nodes"]),
        "edge_count_abs_err": abs(gen_topo["n_edges"] - gt_topo["n_edges"]),
        "terminal_count_abs_err": abs(gen_topo["n_terminal"] - gt_topo["n_terminal"]),
        "bifurcation_count_abs_err": abs(gen_topo["n_bifurcations_topo"] - gt_topo["n_bifurcations_topo"]),
        "max_depth_abs_err": abs(gen_topo["max_depth_topo"] - gt_topo["max_depth_topo"]),
        "beta0_abs_err": abs(gen_topo["beta0_topo"] - gt_topo["beta0_topo"]),
        "beta1_abs_err": abs(gen_topo["beta1_topo"] - gt_topo["beta1_topo"]),
        "degree_hist_l1": _safe_histogram_l1(gen_topo["degree_hist"], gt_topo["degree_hist"]),
        "branch_order_count_l1": _safe_histogram_l1(gen_topo["branch_order_counts"], gt_topo["branch_order_counts"]),
        "branch_angle_w1": _safe_wasserstein(gen_topo["bif_angles"], gt_topo["bif_angles"]),
        "gt_beta0": gt_topo["beta0_topo"],
        "gt_beta1": gt_topo["beta1_topo"],
        "gen_beta0": gen_topo["beta0_topo"],
        "gen_beta1": gen_topo["beta1_topo"],
        "gt_terminal_count": gt_topo["n_terminal"],
        "gen_terminal_count": gen_topo["n_terminal"],
    }


def _extract_edge_lengths(tree_abs):
    _, parents = _get_tree_topology(tree_abs)
    positions = tree_abs[:, 1:4]
    lengths = []
    for idx, parent in enumerate(parents):
        if parent >= 0:
            lengths.append(np.linalg.norm(positions[idx] - positions[parent]))
    return np.asarray(lengths if lengths else [0.0], dtype=np.float64)


def _extract_tortuosity_values(tree_abs):
    desc = extract_distribution_descriptors(tree_abs)
    return np.asarray(desc["tortuosity"], dtype=np.float64)


def compute_geometry_distribution_metrics(gen_abs, gt_abs):
    return {
        "edge_length_w1": _safe_wasserstein(_extract_edge_lengths(gen_abs), _extract_edge_lengths(gt_abs)),
        "tortuosity_w1": _safe_wasserstein(_extract_tortuosity_values(gen_abs), _extract_tortuosity_values(gt_abs)),
    }


def compute_physio_metrics(tree_abs):
    """
    Compute physiological metrics for a single tree (absolute coords).
    Returns dict with:
        murray_violation: mean relative Murray's law violation
        bif_angles_deg:   list of bifurcation angles (degrees)
        mean_bif_angle:   mean bifurcation angle
        tapering_violation_frac: fraction of nodes violating tapering
        mean_symmetry_ratio: mean r_minor/r_major at bifurcations
    """
    k_counts, parents = _get_tree_topology(tree_abs)
    n_nodes = len(tree_abs)
    radii = _compute_node_radii_np(tree_abs)

    bifs = _find_bifurcations(k_counts, parents, n_nodes)
    positions = tree_abs[:, 1:4]  # [N, 3]

    # ── Murray's Law ──
    murray_violations = []
    for p_idx, c1_idx, c2_idx in bifs:
        rp = radii[p_idx]
        rc1 = radii[c1_idx]
        rc2 = radii[c2_idx]
        gamma = 3.0
        rp_g = rp**gamma + 1e-10
        rc_sum = rc1**gamma + rc2**gamma + 1e-10
        murray_violations.append(abs(rp_g - rc_sum) / rp_g)

    # ── Bifurcation Angles ──
    bif_angles = []
    for p_idx, c1_idx, c2_idx in bifs:
        dir1 = positions[c1_idx] - positions[p_idx]
        dir2 = positions[c2_idx] - positions[p_idx]
        d1_norm = np.linalg.norm(dir1)
        d2_norm = np.linalg.norm(dir2)
        if d1_norm < 1e-10 or d2_norm < 1e-10:
            continue
        dir1 = dir1 / d1_norm
        dir2 = dir2 / d2_norm
        cross = np.cross(dir1, dir2)
        cross_norm = np.linalg.norm(cross)
        dot = np.dot(dir1, dir2)
        angle_rad = np.arctan2(cross_norm, dot)
        bif_angles.append(np.degrees(angle_rad))

    # ── Radius Tapering ──
    n_with_parent = 0
    n_violations = 0
    for i in range(n_nodes):
        p = parents[i]
        if p < 0:
            continue
        n_with_parent += 1
        if radii[i] > radii[p] * 1.05:  # 5% tolerance
            n_violations += 1

    # ── Symmetry Ratio ──
    sym_ratios = []
    for p_idx, c1_idx, c2_idx in bifs:
        rc1 = radii[c1_idx]
        rc2 = radii[c2_idx]
        r_max = max(rc1, rc2) + 1e-10
        r_min = min(rc1, rc2)
        sym_ratios.append(r_min / r_max)

    return {
        "murray_violation": float(np.median(murray_violations)) if murray_violations else 0.0,
        "murray_violation_mean": float(np.mean(np.clip(murray_violations, 0, 10))) if murray_violations else 0.0,
        "murray_violations": murray_violations,
        "bif_angles_deg": bif_angles,
        "mean_bif_angle": float(np.mean(bif_angles)) if bif_angles else 0.0,
        "median_bif_angle": float(np.median(bif_angles)) if bif_angles else 0.0,
        "std_bif_angle": float(np.std(bif_angles)) if bif_angles else 0.0,
        "tapering_violation_frac": float(n_violations / max(n_with_parent, 1)),
        "mean_symmetry_ratio": float(np.mean(sym_ratios)) if sym_ratios else 0.0,
        "n_bifurcations": len(bifs),
        "n_nodes": n_nodes,
        "mean_radius": float(radii.mean()),
        "std_radius": float(radii.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# C. Geometric feature extraction (per-tree → fixed-length vector)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tree_features(tree_abs):
    """
    Extract a fixed-length feature vector summarizing a tree's geometry.
    Used for distributional metrics (MMD, Coverage, 1-NNA).

    Returns: np.array of shape [D] with features:
        [0]  n_nodes
        [1]  max_depth
        [2]  n_bifurcations
        [3]  total_branch_length (sum of edge lengths)
        [4]  mean_radius
        [5]  std_radius
        [6]  median_radius
        [7]  mean_bif_angle
        [8]  std_bif_angle
        [9]  murray_violation
        [10] tapering_violation_frac
        [11] mean_symmetry_ratio
        [12] mean_edge_length
        [13] std_edge_length
        [14] tree_span (max distance between any two node positions)
        [15] compactness (span / total_branch_length)
    """
    k_counts, parents = _get_tree_topology(tree_abs)
    n_nodes = len(tree_abs)
    positions = tree_abs[:, 1:4]
    radii = _compute_node_radii_np(tree_abs)

    # Depth
    depths = _compute_tree_depths(parents)
    max_depth = int(depths.max()) if n_nodes > 0 else 0

    # Edge lengths
    edge_lengths = []
    for i in range(n_nodes):
        if parents[i] >= 0:
            d = np.linalg.norm(positions[i] - positions[parents[i]])
            edge_lengths.append(d)
    edge_lengths = np.array(edge_lengths) if edge_lengths else np.array([0.0])
    total_branch_length = edge_lengths.sum()
    mean_edge_length = edge_lengths.mean()
    std_edge_length = edge_lengths.std() if len(edge_lengths) > 1 else 0.0

    # Tree span
    if n_nodes > 1:
        dists = cdist(positions, positions)
        tree_span = dists.max()
    else:
        tree_span = 0.0
    compactness = tree_span / (total_branch_length + 1e-10)

    # Physio metrics
    physio = compute_physio_metrics(tree_abs)

    features = np.array([
        n_nodes,                          # 0
        max_depth,                        # 1
        physio["n_bifurcations"],         # 2
        total_branch_length,              # 3
        float(radii.mean()),              # 4
        float(radii.std()),               # 5
        float(np.median(radii)),          # 6
        physio["median_bif_angle"],       # 7
        physio["std_bif_angle"],          # 8
        physio["murray_violation"],       # 9  (median)
        physio["tapering_violation_frac"],# 10
        physio["mean_symmetry_ratio"],    # 11
        mean_edge_length,                 # 12
        std_edge_length,                  # 13
        tree_span,                        # 14
        compactness,                      # 15
    ], dtype=np.float64)

    return features


FEATURE_NAMES = [
    "n_nodes", "max_depth", "n_bifurcations", "total_branch_length",
    "mean_radius", "std_radius", "median_radius",
    "median_bif_angle", "std_bif_angle", "median_murray_violation",
    "tapering_violation_frac", "mean_symmetry_ratio",
    "mean_edge_length", "std_edge_length", "tree_span", "compactness",
]


# ═══════════════════════════════════════════════════════════════════════════════
# D. Distributional metrics: MMD, Coverage, 1-NNA
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_features(gen_feats, gt_feats):
    """Z-score normalize both sets using GT statistics."""
    mu = gt_feats.mean(axis=0)
    sigma = gt_feats.std(axis=0) + 1e-8
    return (gen_feats - mu) / sigma, (gt_feats - mu) / sigma


def compute_mmd(gen_feats, gt_feats, sigma=None):
    """
    Maximum Mean Discrepancy with Gaussian kernel.
    gen_feats: [M, D], gt_feats: [N, D]
    """
    gen_n, gt_n = _normalize_features(gen_feats.copy(), gt_feats.copy())

    if sigma is None:
        # Median heuristic
        all_feats = np.vstack([gen_n, gt_n])
        dists = cdist(all_feats, all_feats, metric='euclidean')
        sigma = np.median(dists[dists > 0]) + 1e-8

    def rbf(X, Y):
        dists = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-dists / (2.0 * sigma**2))

    K_gg = rbf(gen_n, gen_n)
    K_rr = rbf(gt_n, gt_n)
    K_gr = rbf(gen_n, gt_n)

    m = len(gen_n)
    n = len(gt_n)

    mmd2 = (K_gg.sum() / (m * m) + K_rr.sum() / (n * n) - 2 * K_gr.sum() / (m * n))
    return float(max(mmd2, 0.0))


def compute_coverage(gen_feats, gt_feats):
    """
    Coverage: fraction of GT trees that have at least one generated
    nearest neighbor within some threshold.
    Here we just report the fraction with a unique NN.
    """
    gen_n, gt_n = _normalize_features(gen_feats.copy(), gt_feats.copy())
    # For each GT, find nearest gen
    D = cdist(gt_n, gen_n, metric='euclidean')
    nn_indices = D.argmin(axis=1)
    # Coverage = fraction of GT covered (i.e., how many *distinct* generated
    # trees are NN to some GT tree, divided by min(M, N))
    unique_nn = len(set(nn_indices.tolist()))
    cov = unique_nn / min(len(gen_n), len(gt_n))
    return float(min(cov, 1.0))


def compute_1nna(gen_feats, gt_feats):
    """
    1-Nearest Neighbor Accuracy: leave-one-out.
    Perfect generative model → 50% accuracy (can't distinguish).
    """
    gen_n, gt_n = _normalize_features(gen_feats.copy(), gt_feats.copy())
    all_feats = np.vstack([gen_n, gt_n])
    m = len(gen_n)
    n = len(gt_n)
    labels = np.array([0] * m + [1] * n)

    D = cdist(all_feats, all_feats, metric='euclidean')
    np.fill_diagonal(D, np.inf)  # exclude self

    nn_labels = labels[D.argmin(axis=1)]
    # Accuracy = fraction correctly classified by 1-NN
    acc = (nn_labels == labels).mean()
    return float(acc)


def compute_diversity(gen_feats):
    """Mean pairwise L2 distance between generated trees (in normalized space)."""
    if len(gen_feats) < 2:
        return 0.0
    # Self-normalize
    mu = gen_feats.mean(axis=0)
    sigma = gen_feats.std(axis=0) + 1e-8
    normed = (gen_feats - mu) / sigma
    dists = cdist(normed, normed, metric='euclidean')
    # Upper triangle only
    n = len(normed)
    upper = dists[np.triu_indices(n, k=1)]
    return float(upper.mean())


def compute_histogram_kl(gen_values, gt_values, bins=64, value_range=None):
    gen_values = np.asarray(gen_values, dtype=np.float64).reshape(-1)
    gt_values = np.asarray(gt_values, dtype=np.float64).reshape(-1)
    if value_range is None:
        lo = min(gen_values.min(initial=0.0), gt_values.min(initial=0.0))
        hi = max(gen_values.max(initial=1.0), gt_values.max(initial=1.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        pad = 1e-6 * max(1.0, abs(hi - lo))
        value_range = (lo - pad, hi + pad)
    p_hist = np.histogram(gen_values, bins=bins, range=value_range, density=False)[0].astype(np.float64)
    q_hist = np.histogram(gt_values, bins=bins, range=value_range, density=False)[0].astype(np.float64)
    eps = 1e-8
    p = (p_hist + eps) / (p_hist.sum() + eps * len(p_hist))
    q = (q_hist + eps) / (q_hist.sum() + eps * len(q_hist))
    return float(np.sum(p * np.log(p / q)))


def compute_set_scalar_mmd(gen_values, gt_values):
    gen_values = np.asarray(gen_values, dtype=np.float64).reshape(-1, 1)
    gt_values = np.asarray(gt_values, dtype=np.float64).reshape(-1, 1)
    return compute_mmd(gen_values, gt_values)


def compute_set_mean_coverage(gen_scalar_lists, gt_scalar_lists):
    gen_means = np.array([np.mean(v) for v in gen_scalar_lists], dtype=np.float64).reshape(-1, 1)
    gt_means = np.array([np.mean(v) for v in gt_scalar_lists], dtype=np.float64).reshape(-1, 1)
    return compute_coverage(gen_means, gt_means)


def compute_distributional_breakdown(gen_desc_list, gt_desc_list):
    scalar_keys = [
        "x", "y", "z", "depth", "degree", "radius",
        "edge_length", "theta", "phi", "tortuosity", "beta0", "beta1",
    ]
    out = {}
    for key in scalar_keys:
        gen_values = np.concatenate([d[key] for d in gen_desc_list], axis=0)
        gt_values = np.concatenate([d[key] for d in gt_desc_list], axis=0)
        out[f"KL_{key}"] = compute_histogram_kl(gen_values, gt_values)
        out[f"MMD_{key}"] = compute_set_scalar_mmd(gen_values, gt_values)
        out[f"COV_{key}"] = compute_set_mean_coverage([d[key] for d in gen_desc_list], [d[key] for d in gt_desc_list])

    gen_spec = np.array([d["lap_spec"] for d in gen_desc_list], dtype=np.float64)
    gt_spec = np.array([d["lap_spec"] for d in gt_desc_list], dtype=np.float64)
    gen_betti = np.array([[d["beta0"][0], d["beta1"][0]] for d in gen_desc_list], dtype=np.float64)
    gt_betti = np.array([[d["beta0"][0], d["beta1"][0]] for d in gt_desc_list], dtype=np.float64)
    out["MMD_lap_spec"] = compute_mmd(gen_spec, gt_spec)
    out["COV_lap_spec"] = compute_coverage(gen_spec, gt_spec)
    out["MMD_betti"] = compute_mmd(gen_betti, gt_betti)
    out["COV_betti"] = compute_coverage(gen_betti, gt_betti)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name, npy_dir, gt_val_trees, output_dir):
    """Evaluate a single model. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Directory:  {npy_dir}")
    print(f"{'='*60}")

    pairs = load_generated_pairs(npy_dir)
    if not pairs:
        print(f"  No generated files found, skipping.")
        return None

    print(f"  Loaded {len(pairs)} generated trees")

    # ── A. Reconstruction metrics ──
    recon_metrics = []
    for name, gen, gt in pairs:
        m = compute_reconstruction_metrics(gen, gt)
        m.update(compute_centerline_metrics(gen, gt))
        m.update(compute_cross_section_metrics(gen, gt))
        m.update(compute_topology_comparison_metrics(gen, gt))
        m.update(compute_geometry_distribution_metrics(gen, gt))
        m["name"] = name
        recon_metrics.append(m)

    avg_recon = {
        k: np.mean([m[k] for m in recon_metrics])
        for k in [
            "pos_mae", "cp_mae", "knot_mae", "total_mse", "radius_ratio",
            "centerline_acd", "centerline_cd", "centerline_hd", "centerline_hd95",
            "centerline_f1_1pct", "centerline_f1_2pct",
            "cross_section_2d_chamfer", "cross_section_2d_hausdorff",
            "cross_section_area_error", "cross_section_perimeter_error",
            "cross_section_circularity_error", "cross_section_eccentricity_error",
            "node_count_abs_err", "edge_count_abs_err", "terminal_count_abs_err",
            "bifurcation_count_abs_err", "max_depth_abs_err", "beta0_abs_err", "beta1_abs_err",
            "degree_hist_l1", "branch_order_count_l1", "branch_angle_w1",
            "edge_length_w1", "tortuosity_w1",
        ]
    }
    med_rr = np.median([m["radius_ratio"] for m in recon_metrics])
    avg_recon["radius_ratio_median"] = med_rr
    avg_recon["|radius_ratio-1|"] = np.mean([abs(m["radius_ratio"] - 1.0) for m in recon_metrics])

    print(f"\n  Reconstruction (n={len(pairs)}):")
    print(f"    pos_mae  = {avg_recon['pos_mae']:.4f}")
    print(f"    cp_mae   = {avg_recon['cp_mae']:.4f}")
    print(f"    knot_mae = {avg_recon['knot_mae']:.4f}")
    print(f"    radius   = {avg_recon['radius_ratio']:.3f}× (median {med_rr:.3f}×)")
    print(f"    ctr_ACD  = {avg_recon['centerline_acd']:.4f}")
    print(f"    ctr_CD   = {avg_recon['centerline_cd']:.4f}")
    print(f"    ctr_HD95 = {avg_recon['centerline_hd95']:.4f}")
    print(f"    ctr_F1@1% = {avg_recon['centerline_f1_1pct']:.3f}")
    print(f"    ctr_F1@2% = {avg_recon['centerline_f1_2pct']:.3f}")
    print(f"    xsec CD2D = {avg_recon['cross_section_2d_chamfer']:.4f}")
    print(f"    xsec HD2D = {avg_recon['cross_section_2d_hausdorff']:.4f}")
    print(f"    xsec area/perim = {avg_recon['cross_section_area_error']:.4f} / {avg_recon['cross_section_perimeter_error']:.4f}")
    print(f"    xsec circ/ecc = {avg_recon['cross_section_circularity_error']:.4f} / {avg_recon['cross_section_eccentricity_error']:.4f}")
    print(f"    topo Δnodes = {avg_recon['node_count_abs_err']:.2f}")
    print(f"    topo Δleafs = {avg_recon['terminal_count_abs_err']:.2f}")
    print(f"    topo Δdepth = {avg_recon['max_depth_abs_err']:.2f}")
    print(f"    topo deg_L1 = {avg_recon['degree_hist_l1']:.4f}")
    print(f"    geom W1(len/tort) = {avg_recon['edge_length_w1']:.4f} / {avg_recon['tortuosity_w1']:.4f}")

    # ── B. Physio metrics (on generated trees) ──
    gen_physio_list = []
    for name, gen, gt in pairs:
        pm = compute_physio_metrics(gen)
        pm["name"] = name
        gen_physio_list.append(pm)

    # Also compute physio on paired GT for reference
    gt_physio_list = []
    for name, gen, gt in pairs:
        pm = compute_physio_metrics(gt)
        pm["name"] = name
        gt_physio_list.append(pm)

    avg_gen_physio = {
        "murray_violation": np.median([p["murray_violation"] for p in gen_physio_list]),
        "murray_violation_mean_clip": np.mean([p["murray_violation_mean"] for p in gen_physio_list]),
        "mean_bif_angle": np.mean([p["mean_bif_angle"] for p in gen_physio_list]),
        "median_bif_angle": np.median([p["median_bif_angle"] for p in gen_physio_list]),
        "tapering_violation_frac": np.mean([p["tapering_violation_frac"] for p in gen_physio_list]),
        "mean_symmetry_ratio": np.mean([p["mean_symmetry_ratio"] for p in gen_physio_list]),
    }
    avg_gt_physio = {
        "murray_violation": np.median([p["murray_violation"] for p in gt_physio_list]),
        "murray_violation_mean_clip": np.mean([p["murray_violation_mean"] for p in gt_physio_list]),
        "mean_bif_angle": np.mean([p["mean_bif_angle"] for p in gt_physio_list]),
        "median_bif_angle": np.median([p["median_bif_angle"] for p in gt_physio_list]),
        "tapering_violation_frac": np.mean([p["tapering_violation_frac"] for p in gt_physio_list]),
        "mean_symmetry_ratio": np.mean([p["mean_symmetry_ratio"] for p in gt_physio_list]),
    }

    print(f"\n  Physiology (gen / GT):")
    print(f"    Murray viol (med) = {avg_gen_physio['murray_violation']:.4f} / {avg_gt_physio['murray_violation']:.4f}")
    print(f"    Murray viol (μ,c) = {avg_gen_physio['murray_violation_mean_clip']:.4f} / {avg_gt_physio['murray_violation_mean_clip']:.4f}")
    print(f"    Bif angle (mean)  = {avg_gen_physio['mean_bif_angle']:.1f}° / {avg_gt_physio['mean_bif_angle']:.1f}°")
    print(f"    Bif angle (med)   = {avg_gen_physio['median_bif_angle']:.1f}° / {avg_gt_physio['median_bif_angle']:.1f}°")
    print(f"    Tapering viol.    = {avg_gen_physio['tapering_violation_frac']:.3f} / {avg_gt_physio['tapering_violation_frac']:.3f}")
    print(f"    Symmetry ratio    = {avg_gen_physio['mean_symmetry_ratio']:.3f} / {avg_gt_physio['mean_symmetry_ratio']:.3f}")

    # ── C. Feature extraction → distributional metrics ──
    gen_features = []
    gen_desc = []
    for name, gen, gt in pairs:
        gen_features.append(extract_tree_features(gen))
        gen_desc.append(extract_distribution_descriptors(gen))
    gen_features = np.array(gen_features)

    # Build GT feature matrix from full validation set
    gt_features = []
    gt_desc = []
    for gt_tree in gt_val_trees:
        gt_features.append(extract_tree_features(gt_tree))
        gt_desc.append(extract_distribution_descriptors(gt_tree))
    gt_features = np.array(gt_features)

    print(f"\n  Features: gen={gen_features.shape}, gt={gt_features.shape}")

    mmd_val = compute_mmd(gen_features, gt_features)
    cov_val = compute_coverage(gen_features, gt_features)
    nna_val = compute_1nna(gen_features, gt_features)
    div_val = compute_diversity(gen_features)
    breakdown = compute_distributional_breakdown(gen_desc, gt_desc)

    print(f"  Distributional:")
    print(f"    MMD   = {mmd_val:.6f}")
    print(f"    COV   = {cov_val:.3f}")
    print(f"    1-NNA = {nna_val:.3f}  (ideal=0.50)")
    print(f"    Diversity = {div_val:.3f}")
    print(f"    MMD_radius = {breakdown['MMD_radius']:.6f}")
    print(f"    MMD_length = {breakdown['MMD_edge_length']:.6f}")
    print(f"    MMD_tortu  = {breakdown['MMD_tortuosity']:.6f}")
    print(f"    MMD_deg    = {breakdown['MMD_degree']:.6f}")
    print(f"    MMD_spec   = {breakdown['MMD_lap_spec']:.6f}")

    # ── Assemble summary ──
    summary = {
        "model": model_name,
        "n_trees": len(pairs),
        # Reconstruction
        "pos_mae": avg_recon["pos_mae"],
        "cp_mae": avg_recon["cp_mae"],
        "knot_mae": avg_recon["knot_mae"],
        "total_mse": avg_recon["total_mse"],
        "radius_ratio": avg_recon["radius_ratio"],
        "radius_ratio_median": avg_recon["radius_ratio_median"],
        "|radius-1|": avg_recon["|radius_ratio-1|"],
        "centerline_acd": avg_recon["centerline_acd"],
        "centerline_cd": avg_recon["centerline_cd"],
        "centerline_hd": avg_recon["centerline_hd"],
        "centerline_hd95": avg_recon["centerline_hd95"],
        "centerline_f1_1pct": avg_recon["centerline_f1_1pct"],
        "centerline_f1_2pct": avg_recon["centerline_f1_2pct"],
        "cross_section_2d_chamfer": avg_recon["cross_section_2d_chamfer"],
        "cross_section_2d_hausdorff": avg_recon["cross_section_2d_hausdorff"],
        "cross_section_area_error": avg_recon["cross_section_area_error"],
        "cross_section_perimeter_error": avg_recon["cross_section_perimeter_error"],
        "cross_section_circularity_error": avg_recon["cross_section_circularity_error"],
        "cross_section_eccentricity_error": avg_recon["cross_section_eccentricity_error"],
        "node_count_abs_err": avg_recon["node_count_abs_err"],
        "edge_count_abs_err": avg_recon["edge_count_abs_err"],
        "terminal_count_abs_err": avg_recon["terminal_count_abs_err"],
        "bifurcation_count_abs_err": avg_recon["bifurcation_count_abs_err"],
        "max_depth_abs_err": avg_recon["max_depth_abs_err"],
        "beta0_abs_err": avg_recon["beta0_abs_err"],
        "beta1_abs_err": avg_recon["beta1_abs_err"],
        "degree_hist_l1": avg_recon["degree_hist_l1"],
        "branch_order_count_l1": avg_recon["branch_order_count_l1"],
        "branch_angle_w1": avg_recon["branch_angle_w1"],
        "edge_length_w1": avg_recon["edge_length_w1"],
        "tortuosity_w1": avg_recon["tortuosity_w1"],
        # Physio (generated) — median Murray
        "murray_viol": avg_gen_physio["murray_violation"],
        "murray_viol_μc": avg_gen_physio["murray_violation_mean_clip"],
        "bif_angle_deg": avg_gen_physio["mean_bif_angle"],
        "bif_angle_med": avg_gen_physio["median_bif_angle"],
        "tapering_viol": avg_gen_physio["tapering_violation_frac"],
        "symmetry_ratio": avg_gen_physio["mean_symmetry_ratio"],
        # Physio (GT reference)
        "murray_viol_gt": avg_gt_physio["murray_violation"],
        "bif_angle_deg_gt": avg_gt_physio["mean_bif_angle"],
        # Distributional
        "MMD": mmd_val,
        "COV": cov_val,
        "1-NNA": nna_val,
        "diversity": div_val,
    }
    summary.update(breakdown)

    # ── Save per-tree metrics ──
    per_tree_path = Path(output_dir) / f"per_tree_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    with open(per_tree_path, "w", newline="") as f:
        keys = list(recon_metrics[0].keys()) + [
            "murray_viol_med", "murray_viol_mean_clip", "mean_bif_angle",
            "tapering_viol", "symmetry_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for rm, pm in zip(recon_metrics, gen_physio_list):
            row = dict(rm)
            row["murray_viol_med"] = pm["murray_violation"]
            row["murray_viol_mean_clip"] = pm["murray_violation_mean"]
            row["mean_bif_angle"] = pm["mean_bif_angle"]
            row["tapering_viol"] = pm["tapering_violation_frac"]
            row["symmetry_ratio"] = pm["mean_symmetry_ratio"]
            writer.writerow(row)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate all vessel tree generation models")
    parser.add_argument("--gt_val_dir", type=str, default=DEFAULT_GT_VAL_DIR,
                        help="Path to full GT validation directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--max_gt", type=int, default=None,
                        help="Max GT trees to load for distributional metrics (None=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load full GT validation set ──
    print(f"Loading GT validation set from {args.gt_val_dir} ...")
    gt_val_trees = load_full_gt_validation(args.gt_val_dir, max_trees=args.max_gt)
    print(f"  Loaded {len(gt_val_trees)} GT trees\n")

    # ── Also compute GT physio baseline ──
    print("Computing GT physiological baseline ...")
    gt_physio_all = []
    for t in gt_val_trees[:200]:  # subsample for speed
        gt_physio_all.append(compute_physio_metrics(t))
    gt_murray = np.median([p["murray_violation"] for p in gt_physio_all])
    gt_murray_mc = np.mean([p["murray_violation_mean"] for p in gt_physio_all])
    gt_bif = np.mean([p["mean_bif_angle"] for p in gt_physio_all])
    gt_bif_med = np.median([p["median_bif_angle"] for p in gt_physio_all])
    gt_taper = np.mean([p["tapering_violation_frac"] for p in gt_physio_all])
    gt_sym = np.mean([p["mean_symmetry_ratio"] for p in gt_physio_all])
    print(f"  GT Murray (median)  = {gt_murray:.4f}")
    print(f"  GT Murray (μ,clip)  = {gt_murray_mc:.4f}")
    print(f"  GT Bif angle (mean) = {gt_bif:.1f}°")
    print(f"  GT Bif angle (med)  = {gt_bif_med:.1f}°")
    print(f"  GT Tapering viol.   = {gt_taper:.3f}")
    print(f"  GT Symmetry ratio   = {gt_sym:.3f}")

    # ── Evaluate each model ──
    all_summaries = []
    for model_name, npy_dir in MODELS.items():
        full_dir = ROOT / npy_dir
        summary = evaluate_model(model_name, full_dir, gt_val_trees, output_dir)
        if summary:
            all_summaries.append(summary)

    # ── Add GT baseline row ──
    gt_row = {
        "model": "GT (validation)",
        "n_trees": len(gt_val_trees),
        "pos_mae": 0.0,
        "cp_mae": 0.0,
        "knot_mae": 0.0,
        "total_mse": 0.0,
        "radius_ratio": 1.0,
        "radius_ratio_median": 1.0,
        "|radius-1|": 0.0,
        "centerline_acd": 0.0,
        "centerline_cd": 0.0,
        "centerline_hd": 0.0,
        "centerline_hd95": 0.0,
        "centerline_f1_1pct": 1.0,
        "centerline_f1_2pct": 1.0,
        "cross_section_2d_chamfer": 0.0,
        "cross_section_2d_hausdorff": 0.0,
        "cross_section_area_error": 0.0,
        "cross_section_perimeter_error": 0.0,
        "cross_section_circularity_error": 0.0,
        "cross_section_eccentricity_error": 0.0,
        "node_count_abs_err": 0.0,
        "edge_count_abs_err": 0.0,
        "terminal_count_abs_err": 0.0,
        "bifurcation_count_abs_err": 0.0,
        "max_depth_abs_err": 0.0,
        "beta0_abs_err": 0.0,
        "beta1_abs_err": 0.0,
        "degree_hist_l1": 0.0,
        "branch_order_count_l1": 0.0,
        "branch_angle_w1": 0.0,
        "edge_length_w1": 0.0,
        "tortuosity_w1": 0.0,
        "murray_viol": gt_murray,
        "murray_viol_μc": gt_murray_mc,
        "bif_angle_deg": gt_bif,
        "bif_angle_med": gt_bif_med,
        "tapering_viol": gt_taper,
        "symmetry_ratio": gt_sym,
        "murray_viol_gt": gt_murray,
        "bif_angle_deg_gt": gt_bif,
        "MMD": 0.0,
        "COV": 1.0,
        "1-NNA": 0.5,
        "diversity": compute_diversity(np.array([extract_tree_features(t) for t in gt_val_trees[:200]])),
    }
    gt_desc = [extract_distribution_descriptors(t) for t in gt_val_trees]
    gt_row.update(compute_distributional_breakdown(gt_desc, gt_desc))
    all_summaries.append(gt_row)

    # ── Save summary table ──
    if all_summaries:
        csv_path = output_dir / "summary_table.csv"
        keys = list(all_summaries[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\nSaved CSV: {csv_path}")

        sota_csv_path = output_dir / "summary_table_sota.csv"
        _write_selected_csv(
            all_summaries,
            sota_csv_path,
            [
                "model", "n_trees",
                "pos_mae", "cp_mae", "knot_mae",
                "centerline_acd", "centerline_hd95", "centerline_f1_1pct", "centerline_f1_2pct",
                "cross_section_2d_chamfer", "cross_section_2d_hausdorff",
                "cross_section_area_error", "cross_section_perimeter_error",
                "cross_section_circularity_error", "cross_section_eccentricity_error",
                "node_count_abs_err", "terminal_count_abs_err", "bifurcation_count_abs_err",
                "max_depth_abs_err", "beta1_abs_err", "degree_hist_l1", "branch_order_count_l1",
                "branch_angle_w1", "edge_length_w1", "tortuosity_w1",
                "murray_viol", "tapering_viol", "symmetry_ratio",
                "MMD", "MMD_radius", "MMD_edge_length", "MMD_tortuosity", "MMD_degree",
                "COV", "1-NNA", "diversity",
            ],
        )
        print(f"Saved CSV: {sota_csv_path}")

        # Pretty-print table
        txt_path = output_dir / "summary_table.txt"
        _print_table(all_summaries, txt_path)
        _print_sota_table(all_summaries, output_dir / "summary_table_sota.txt")

    print(f"\n{'='*60}")
    print(f"  All results saved to {output_dir}/")
    print(f"{'='*60}")


def _print_table(summaries, txt_path=None):
    """Pretty-print a comparison table."""

    # Key columns for display
    display_cols = [
        ("Model",            "model",             "s",   18),
        ("N",                "n_trees",           "d",   5),
        ("pos_mae↓",         "pos_mae",           ".4f", 9),
        ("cp_mae↓",          "cp_mae",            ".4f", 9),
        ("CtrCD↓",           "centerline_cd",     ".4f", 9),
        ("XSecCD↓",          "cross_section_2d_chamfer", ".4f", 9),
        ("r_ratio→1",        "radius_ratio",      ".3f", 10),
        ("|r-1|↓",           "|radius-1|",        ".3f", 7),
        ("Murray↓",          "murray_viol",       ".4f", 9),
        ("BifAngle",         "bif_angle_med",     ".1f", 9),
        ("Taper↓",           "tapering_viol",     ".3f", 8),
        ("SymR",             "symmetry_ratio",    ".3f", 6),
        ("MMD↓",             "MMD",               ".5f", 9),
        ("MMD_r↓",           "MMD_radius",        ".5f", 9),
        ("MMD_l↓",           "MMD_edge_length",   ".5f", 9),
        ("MMD_t↓",           "MMD_tortuosity",    ".5f", 9),
        ("MMD_deg↓",         "MMD_degree",        ".5f", 9),
        ("COV↑",             "COV",               ".3f", 6),
        ("1-NNA→.5",         "1-NNA",             ".3f", 9),
        ("Div",              "diversity",         ".2f", 6),
    ]

    lines = []

    # Header
    header = " | ".join(f"{col[0]:>{col[3]}}" for col in display_cols)
    sep = "-+-".join("-" * col[3] for col in display_cols)
    lines.append(header)
    lines.append(sep)

    for s in summaries:
        parts = []
        for _, key, fmt, width in display_cols:
            val = s.get(key, "")
            if fmt == "s":
                parts.append(f"{str(val):>{width}}")
            elif fmt == "d":
                parts.append(f"{int(val):>{width}d}")
            else:
                parts.append(f"{float(val):>{width}{fmt}}")
        lines.append(" | ".join(parts))

    table_str = "\n".join(lines)
    print(f"\n{table_str}\n")

    if txt_path:
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")
        print(f"Saved table: {txt_path}")


def _write_selected_csv(rows, csv_path, keys):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _print_sota_table(summaries, txt_path=None):
    display_cols = [
        ("Model",        "model",                    "s",   24),
        ("CtrACD↓",      "centerline_acd",           ".4f", 9),
        ("HD95↓",        "centerline_hd95",          ".4f", 9),
        ("CF1@1%↑",      "centerline_f1_1pct",       ".3f", 8),
        ("XSCD↓",        "cross_section_2d_chamfer", ".4f", 8),
        ("XSHD↓",        "cross_section_2d_hausdorff", ".4f", 8),
        ("Area↓",        "cross_section_area_error", ".4f", 8),
        ("Perim↓",       "cross_section_perimeter_error", ".4f", 8),
        ("Circ↓",        "cross_section_circularity_error", ".4f", 8),
        ("Ecc↓",         "cross_section_eccentricity_error", ".4f", 8),
        ("dNodes↓",      "node_count_abs_err",       ".2f", 8),
        ("dLeaf↓",       "terminal_count_abs_err",   ".2f", 8),
        ("dDepth↓",      "max_depth_abs_err",        ".2f", 8),
        ("DegL1↓",       "degree_hist_l1",           ".4f", 8),
        ("AngW1↓",       "branch_angle_w1",          ".3f", 8),
        ("LenW1↓",       "edge_length_w1",           ".4f", 8),
        ("TortW1↓",      "tortuosity_w1",            ".4f", 9),
        ("Murray↓",      "murray_viol",              ".4f", 9),
        ("MMD↓",         "MMD",                      ".5f", 9),
        ("COV↑",         "COV",                      ".3f", 6),
        ("1-NNA",        "1-NNA",                    ".3f", 6),
    ]

    lines = []
    header = " | ".join(f"{col[0]:>{col[3]}}" for col in display_cols)
    sep = "-+-".join("-" * col[3] for col in display_cols)
    lines.append(header)
    lines.append(sep)

    for summary in summaries:
        parts = []
        for _, key, fmt, width in display_cols:
            value = summary.get(key, "")
            if fmt == "s":
                parts.append(f"{str(value):>{width}}")
            else:
                parts.append(f"{float(value):>{width}{fmt}}")
        lines.append(" | ".join(parts))

    table_str = "\n".join(lines)
    print(f"\n{table_str}\n")

    if txt_path:
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")
        print(f"Saved table: {txt_path}")


if __name__ == "__main__":
    main()
