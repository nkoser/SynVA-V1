"""
Generate vessel trees using the trained Autoregressive Tree Flow Matching model.

The model generates trees level-by-level: root first, then children, then
grandchildren, etc.  Each level is denoised via Flow Matching conditioned
on already-generated ancestor nodes.

Usage:
    python Stage2_AutoregressiveFM/generate_trees.py \
        --config Stage2_AutoregressiveFM/configs/generate_ar_fm.yaml
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import warnings as _w
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import splev, splprep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    raise RuntimeError("pip install pyyaml")

from Stage2_AutoregressiveFM.model import AutoregressiveTreeFlowNet
from tree_functions import (
    Tree,
    preorder_kcount_parent_indices,
    local_geometry_tree_to_absolute,
    parent_relative_positions_to_absolute,
    deserialize_pre_order_kcount,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def load_feature_stats(path, device="cpu"):
    data = np.load(path)
    mean = torch.from_numpy(data["mean"]).float().to(device)
    std = torch.from_numpy(data["std"]).float().to(device)
    return mean, std


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(d):
    if isinstance(d, int):
        d = f"cuda:{d}"
    if isinstance(d, str) and d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def compute_depths_and_child_slots(parents):
    parents = np.asarray(parents).ravel()
    depths = np.zeros_like(parents)
    child_slots = np.zeros_like(parents)
    cc = {}
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            continue
        depths[i] = depths[p] + 1
        s = cc.get(p, 0)
        child_slots[i] = 1 if s == 0 else 2
        cc[p] = s + 1
    return depths, child_slots


# ── Data loading ─────────────────────────────────────────────────────────────

def load_val_trees(val_dir, n_samples=20, shuffle=True):
    files = sorted(
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir)
        if f.endswith(".npy") and not f.startswith(".")
    )
    if shuffle:
        random.shuffle(files)
    if n_samples and n_samples < len(files):
        files = files[:n_samples]

    trees, names = [], []
    for fp in files:
        arr = np.load(fp).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        valid = ~(np.all(np.abs(arr[:, 1:]) < 1e-8, axis=1))
        arr = arr[valid]
        if arr.shape[0] < 2:
            continue
        trees.append(arr)
        names.append(Path(fp).stem)
    return trees, names


# ── Visualization ────────────────────────────────────────────────────────────

def data_to_tree(data):
    n = data.shape[0]
    serial = []
    for i in range(n):
        kc = int(round(data[i, 0]))
        serial.append(float(kc))
        serial.append(float(data[i, 1]))
        serial.append(float(data[i, 2]))
        serial.append(float(data[i, 3]))
        serial.extend(float(v) for v in data[i, 4:40])
    tree, _ = deserialize_pre_order_kcount(serial, k=39)
    return tree


def sample_spline(coeffs, n_samples=50):
    coeffs = list(coeffs)
    if len(coeffs) < 36:
        return None
    t = np.array(coeffs[24:36])
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8:(i + 1) * 8]) for i in range(3)]
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    try:
        x, y, z = splev(u, tck)
        pts = np.column_stack((x, y, z))
        if np.any(np.abs(pts) > 100):
            return None
        return pts
    except Exception:
        return None


def draw_tree_splines_scatter(tree, ax, color='r', spline_samples=50, point_size=0.5):
    if tree is None:
        return
    if hasattr(tree.data, 'get') and "r" in tree.data and tree.data["r"] is not None:
        coeffs = tree.data["r"]
        if isinstance(coeffs, (list, np.ndarray)) and len(coeffs) == 36:
            pts = sample_spline(coeffs, n_samples=spline_samples)
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, marker='.', s=point_size)
    draw_tree_splines_scatter(tree.left, ax, color=color,
                              spline_samples=spline_samples, point_size=point_size)
    draw_tree_splines_scatter(tree.right, ax, color=color,
                              spline_samples=spline_samples, point_size=point_size)


def draw_tree_centerlines(tree, ax, color='gray', linewidth=1.0, alpha=0.6,
                          parent_pos=None):
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


def _sync_3d_axes(ax1, ax2):
    all_lims = []
    for ax in [ax1, ax2]:
        all_lims.extend([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    lo = min(l[0] for l in all_lims)
    hi = max(l[1] for l in all_lims)
    m = (hi - lo) * 0.05
    for ax in [ax1, ax2]:
        ax.set_xlim3d(lo - m, hi + m)
        ax.set_ylim3d(lo - m, hi + m)
        ax.set_zlim3d(lo - m, hi + m)


def visualize_trees(gen_list, gt_list=None, names=None, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_gt = gt_list is not None and len(gt_list) > 0
    for i, gen_data in enumerate(gen_list):
        gen_tree = data_to_tree(gen_data)
        name = names[i] if names and i < len(names) else f"tree_{i:04d}"

        if has_gt and i < len(gt_list):
            gt_tree = data_to_tree(gt_list[i])
            fig = plt.figure(figsize=(14, 6))
            fig.suptitle(f"{name}  —  {len(gen_data)} nodes", fontsize=12)

            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title("Ground Truth")
            draw_tree_centerlines(gt_tree, ax1, color='darkblue', linewidth=1.2, alpha=0.5)
            draw_tree_splines_scatter(gt_tree, ax1, color='blue', spline_samples=60)

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("Generated (AR Flow-Matching)")
            draw_tree_centerlines(gen_tree, ax2, color='darkred', linewidth=1.2, alpha=0.5)
            draw_tree_splines_scatter(gen_tree, ax2, color='red', spline_samples=60)
            _sync_3d_axes(ax1, ax2)
        else:
            fig = plt.figure(figsize=(8, 7))
            fig.suptitle(f"{name}  —  {len(gen_data)} nodes", fontsize=12)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Generated (AR Flow-Matching)")
            draw_tree_centerlines(gen_tree, ax, color='darkred', linewidth=1.2, alpha=0.5)
            draw_tree_splines_scatter(gen_tree, ax, color='red', spline_samples=60)

        plt.tight_layout()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig_path = Path(output_dir) / f"{name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(gen_abs, gt_abs):
    n = min(len(gen_abs), len(gt_abs))
    pos_gen = gen_abs[:n, 1:4]
    pos_gt = gt_abs[:n, 1:4]
    cp_gen = gen_abs[:n, 4:28]
    cp_gt = gt_abs[:n, 4:28]

    pos_mae = np.abs(pos_gen - pos_gt).mean()
    cp_mae = np.abs(cp_gen - cp_gt).mean()
    knot_mae = np.abs(gen_abs[:n, 28:40] - gt_abs[:n, 28:40]).mean()

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
        "gen_cp_radius": float(gen_radius),
        "gt_cp_radius": float(gt_radius),
        "radius_ratio": float(gen_radius / max(gt_radius, 1e-8)),
    }


# ── Postprocessing (Fixes 1–4) ──────────────────────────────────────────────

def postprocess_tree(gen_40, k_counts, rp_damp=0.75, abs_pos_mode=False):
    """Apply Fixes 1–4 in-place on gen_40 [N, 40]."""
    n_nodes = gen_40.shape[0]
    parents_np = preorder_kcount_parent_indices(k_counts)

    # Fix 1: Knot monotonicity
    for ni in range(n_nodes):
        gen_40[ni, 28:40] = np.sort(gen_40[ni, 28:40])

    # Fix 2: CP planarity
    for ni in range(n_nodes):
        cp_x = gen_40[ni, 4:12]
        cp_y = gen_40[ni, 12:20]
        cp_z = gen_40[ni, 20:28]
        cps = np.column_stack([cp_x, cp_y, cp_z])
        centroid = cps.mean(axis=0)
        centered = cps - centroid
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        projected = U[:, :2] @ np.diag(S[:2]) @ Vh[:2, :]
        cps_planar = projected + centroid
        gen_40[ni, 4:12] = cps_planar[:, 0]
        gen_40[ni, 12:20] = cps_planar[:, 1]
        gen_40[ni, 20:28] = cps_planar[:, 2]

    # Fix 3: CS normal → branch direction
    for ni in range(n_nodes):
        if abs_pos_mode:
            pi = parents_np[ni]
            if pi < 0:
                continue
            rp = gen_40[ni, 1:4] - gen_40[pi, 1:4]
        else:
            rp = gen_40[ni, 1:4]
        rp_len = np.linalg.norm(rp)
        if rp_len < 1e-8:
            continue
        target = rp / rp_len
        cps = np.column_stack([gen_40[ni, 4:12],
                               gen_40[ni, 12:20],
                               gen_40[ni, 20:28]])
        centroid = cps.mean(axis=0)
        centered = cps - centroid
        _, _, Vh = np.linalg.svd(centered, full_matrices=False)
        normal = Vh[2]
        if np.dot(normal, target) < 0:
            normal = -normal
        c = np.dot(normal, target)
        if abs(c - 1.0) < 1e-6:
            continue
        if abs(c + 1.0) < 1e-6:
            axis = Vh[0]
            R = 2.0 * np.outer(axis, axis) - np.eye(3)
        else:
            v = np.cross(normal, target)
            s = np.linalg.norm(v)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
        rotated = (R @ centered.T).T + centroid
        gen_40[ni, 4:12] = rotated[:, 0]
        gen_40[ni, 12:20] = rotated[:, 1]
        gen_40[ni, 20:28] = rotated[:, 2]

    # Fix 4: Periodic B-spline refit + radius preservation + aspect correction
    _target_aspect = 0.786
    _aspect_alpha = 0.35
    for ni in range(n_nodes):
        cp_x = gen_40[ni, 4:12].copy()
        cp_y = gen_40[ni, 12:20].copy()
        cp_z = gen_40[ni, 20:28].copy()
        knots = gen_40[ni, 28:40].copy()
        tck_old = (knots, [cp_x, cp_y, cp_z], 3)
        try:
            u_samp = np.linspace(0, 1, 64)
            pts = np.column_stack(splev(u_samp, tck_old))
        except Exception:
            continue
        old_cps = np.column_stack([cp_x, cp_y, cp_z])
        old_cent = old_cps.mean(axis=0)
        old_cp_rms = np.sqrt(np.mean(np.sum((old_cps - old_cent)**2, axis=1)))
        center = pts.mean(0)
        r = np.median(np.linalg.norm(pts - center, axis=1))
        s_val = max((0.01 * r) ** 2 * len(pts), 1e-12)
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", category=RuntimeWarning)
                tck_new, _ = splprep(
                    [pts[:, 0], pts[:, 1], pts[:, 2]],
                    k=3, per=True, s=s_val, nest=12,
                )
        except Exception:
            continue
        t_new, c_new, _ = tck_new
        if len(c_new[0]) >= 8 and len(t_new) >= 12:
            new_cp = np.column_stack([c_new[0][:8], c_new[1][:8], c_new[2][:8]])
            old_cp = np.column_stack([cp_x, cp_y, cp_z])
            new_norm = np.linalg.norm(new_cp - new_cp.mean(0))
            old_norm = np.linalg.norm(old_cp - old_cp.mean(0))
            if old_norm > 1e-10 and new_norm / old_norm > 3.0:
                continue
            new_cent = new_cp.mean(axis=0)
            new_cp_rms = np.sqrt(np.mean(np.sum((new_cp - new_cent)**2, axis=1)))
            if new_cp_rms > 1e-10 and old_cp_rms > 1e-10:
                r_scale = 1.0 + rp_damp * (old_cp_rms / new_cp_rms - 1.0)
                new_cp = (new_cp - new_cent) * r_scale + new_cent
            gen_40[ni, 4:12]  = new_cp[:, 0]
            gen_40[ni, 12:20] = new_cp[:, 1]
            gen_40[ni, 20:28] = new_cp[:, 2]
            gen_40[ni, 28:40] = t_new[:12]
        # Aspect ratio correction
        cps = np.column_stack([gen_40[ni, 4:12],
                               gen_40[ni, 12:20],
                               gen_40[ni, 20:28]])
        centroid = cps.mean(axis=0)
        centered = cps - centroid
        try:
            U_cp, S_cp, Vh_cp = np.linalg.svd(centered, full_matrices=False)
        except:
            continue
        if S_cp[0] > 1e-10:
            old_area = S_cp[0] * S_cp[1]
            target_s1 = _target_aspect * S_cp[0]
            S_cp_new = S_cp.copy()
            S_cp_new[1] = S_cp[1] + _aspect_alpha * (target_s1 - S_cp[1])
            new_area = S_cp_new[0] * S_cp_new[1]
            if new_area > 1e-20:
                scale = np.sqrt(old_area / new_area)
                S_cp_new[0] *= scale
                S_cp_new[1] *= scale
            cps_corrected = U_cp @ np.diag(S_cp_new) @ Vh_cp + centroid
            gen_40[ni, 4:12]  = cps_corrected[:, 0]
            gen_40[ni, 12:20] = cps_corrected[:, 1]
            gen_40[ni, 20:28] = cps_corrected[:, 2]

    # Fix 5: Re-center CPs at origin in node-local frame (kill drift bias)
    from tree_functions import recenter_node_local_cps
    gen_40 = recenter_node_local_cps(gen_40, copy=False)

    return gen_40


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(cfg):
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})

    device = resolve_device(params.get("device", 0))
    seed_all(int(params.get("seed", 42)))

    n_samples = int(params.get("n_samples", 20))
    n_steps = int(params.get("n_steps", 50))
    output_dir = Path(params.get("output_dir", "Stage2_AutoregressiveFM/generated/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "npy").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Load model
    ckpt_path = paths["checkpoint"]
    print(f"Loading AutoregressiveTreeFlowNet from {ckpt_path}")

    model = AutoregressiveTreeFlowNet(
        geom_dim=int(model_cfg.get("geom_dim", 39)),
        k_classes=int(model_cfg.get("k_classes", 3)),
        max_depth=int(model_cfg.get("max_depth", 128)),
        d_model=int(model_cfg.get("d_model", 384)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 10)),
        d_ff=int(model_cfg.get("d_ff", 1536)),
        max_nodes=int(model_cfg.get("max_nodes", 256)),
        dropout=0.0,
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_ema = bool(params.get("use_ema", True))
    if use_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        print("  Using EMA weights")
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters")

    # Feature stats
    feature_stats_path = paths.get("feature_stats", None)
    feature_mean, feature_std = None, None
    if feature_stats_path and os.path.exists(feature_stats_path):
        feature_mean, feature_std = load_feature_stats(feature_stats_path, device)
        print(f"Feature denormalization: loaded from {feature_stats_path}")
    else:
        print("Feature denormalization: DISABLED")

    # Validation data
    val_dir = paths["val_dir"]
    print(f"Loading validation trees from {val_dir}")
    gt_trees_raw, names = load_val_trees(val_dir, n_samples=n_samples, shuffle=True)
    print(f"  Loaded {len(gt_trees_raw)} trees")

    abs_pos_mode = bool(cfg.get("data", {}).get("absolute_positions", False))
    if abs_pos_mode:
        print("Position mode: ABSOLUTE (model predicts world coordinates)")
    else:
        print("Position mode: RELATIVE (model predicts parent-relative offsets)")

    rp_damp = float(params.get("rp_damp", 0.75))
    print(f"Postprocessing: rp_damp={rp_damp}")

    # Generate
    print(f"\n=== Generating (n_steps={n_steps} per level) ===")
    gen_data_list, gt_data_list, all_metrics = [], [], []

    for i, (gt_raw, name) in enumerate(zip(gt_trees_raw, names)):
        k_counts = np.clip(np.rint(gt_raw[:, 0]), 0, 2).astype(np.int64)
        parents = preorder_kcount_parent_indices(k_counts)
        depths, child_slots = compute_depths_and_child_slots(parents)
        n_nodes = len(k_counts)

        k_t = torch.from_numpy(k_counts).long().unsqueeze(0).to(device)
        d_t = torch.from_numpy(depths).long().unsqueeze(0).to(device)
        cs_t = torch.from_numpy(child_slots).long().unsqueeze(0).to(device)
        nm_t = torch.ones(1, n_nodes, dtype=torch.bool, device=device)

        max_depth = int(depths.max())

        with torch.no_grad():
            pred_local = model.sample(
                k_t, d_t, cs_t, nm_t,
                n_steps=n_steps,
                velocity_scale=float(params.get("velocity_scale", 1.0)),
            )

        pred_39 = pred_local[0].cpu().numpy()

        # Denormalize
        if feature_mean is not None and feature_std is not None:
            fm = feature_mean.cpu().numpy()
            fs = feature_std.cpu().numpy()
            pred_39 = pred_39 * fs + fm

        # Reconstruct [N, 40]
        gen_40 = np.zeros((n_nodes, 40), dtype=np.float32)
        gen_40[:, 0] = k_counts.astype(np.float32)
        gen_40[:, 1:] = pred_39

        # Postprocess
        gen_40 = postprocess_tree(
            gen_40,
            k_counts,
            rp_damp=rp_damp,
            abs_pos_mode=abs_pos_mode,
        )

        # Convert to absolute
        gen_abs = local_geometry_tree_to_absolute(
            gen_40,
            position_slice=(1, 4),
            control_point_slices=((4, 12), (12, 20), (20, 28)),
            relative_positions=(not abs_pos_mode),
            node_local_control_points=True,
            copy=True,
        )
        gt_abs = local_geometry_tree_to_absolute(
            gt_raw.copy(),
            position_slice=(1, 4),
            control_point_slices=((4, 12), (12, 20), (20, 28)),
            relative_positions=True,
            node_local_control_points=True,
            copy=False,
        )

        gen_data_list.append(gen_abs)
        gt_data_list.append(gt_abs)

        m = compute_metrics(gen_abs, gt_abs)
        m["name"] = name
        m["n_nodes"] = n_nodes
        m["max_depth"] = max_depth
        all_metrics.append(m)

        np.save(output_dir / "npy" / f"{name}.npy", gen_abs)
        np.save(output_dir / "npy" / f"{name}_gt.npy", gt_abs)
        np.save(output_dir / "npy" / f"{name}_local.npy", gen_40)
        np.save(output_dir / "npy" / f"{name}_gt_local.npy", gt_raw)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(gt_trees_raw)}] {name}: {n_nodes} nodes (d={max_depth}), "
                  f"pos_mae={m['pos_mae']:.4f}, radius={m['radius_ratio']:.3f}")

    # Aggregate
    avg_pos_mae = np.mean([m["pos_mae"] for m in all_metrics])
    avg_cp_mae = np.mean([m["cp_mae"] for m in all_metrics])
    avg_knot_mae = np.mean([m["knot_mae"] for m in all_metrics])
    avg_radius = np.mean([m["radius_ratio"] for m in all_metrics])
    print(f"\n=== Metrics (n={len(all_metrics)}) ===")
    print(f"  avg pos_mae:      {avg_pos_mae:.5f}")
    print(f"  avg cp_mae:       {avg_cp_mae:.5f}")
    print(f"  avg knot_mae:     {avg_knot_mae:.5f}")
    print(f"  avg radius_ratio: {avg_radius:.3f}x GT")
    print(f"  avg |radius-1|:   {abs(avg_radius - 1.0):.3f}")

    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n=== Visualizing ===")
    visualize_trees(gen_data_list, gt_data_list, names,
                    output_dir=str(output_dir / "images"))

    meta = {
        "n_samples": len(gen_data_list),
        "n_steps_per_level": n_steps,
        "avg_pos_mae": avg_pos_mae,
        "avg_cp_mae": avg_cp_mae,
        "avg_knot_mae": avg_knot_mae,
        "avg_radius_ratio": avg_radius,
    }
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(gen_data_list)} trees -> {output_dir}")
    return gen_data_list, gt_data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg.setdefault("params", {})["n_samples"] = args.n_samples
    if args.n_steps is not None:
        cfg.setdefault("params", {})["n_steps"] = args.n_steps
    if args.seed is not None:
        cfg.setdefault("params", {})["seed"] = args.seed

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
