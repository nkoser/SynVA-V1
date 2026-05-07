"""
Generate vessel trees using the trained Flow Matching velocity model.

Pipeline:
  1. Load trained FlowMatchingVelocityModel
  2. Obtain topologies from validation .npy files
  3. Euler-integrate from t=0 (noise) to t=1 (clean data)  →  [N, 39] local geom
  4. Convert relative → absolute coordinates
  5. Visualize side-by-side with GT

Usage:
    python Stage2_FlowMatching/generate_trees.py \
        --config Stage2_FlowMatching/configs/generate_flow_v1.yaml
"""

import argparse
import csv
import json
import math
import os
import random
import sys
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

from Stage2_FlowMatching.model import FlowMatchingVelocityModel
from tree_functions import (
    Tree,
    preorder_kcount_parent_indices,
    local_geometry_tree_to_absolute,
    parent_relative_positions_to_absolute,
    deserialize_pre_order_kcount,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def load_feature_stats(path, device="cpu"):
    """Load per-feature mean/std from a .npz file for denormalization."""
    data = np.load(path)
    mean = torch.from_numpy(data["mean"]).float().to(device)  # [39]
    std = torch.from_numpy(data["std"]).float().to(device)    # [39]
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
    """Load validation .npy files, return list of [N, 40] arrays + names."""
    files = sorted(
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir)
        if f.endswith(".npy") and not f.startswith(".")
    )
    if shuffle:
        random.shuffle(files)
    if n_samples and n_samples < len(files):
        files = files[:n_samples]

    trees = []
    names = []
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


# ── Visualization (reusable) ────────────────────────────────────────────────

def data_to_tree(data):
    """Convert [N, 40] absolute data to Tree structure."""
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
    from scipy.interpolate import splev
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
    """Draw the centerline (edges from parent to child node positions)."""
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


def visualize_trees(gen_list, gt_list=None, names=None, output_dir=None, mode="splines"):
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
            ax1.scatter(gt_tree.data["x"], gt_tree.data["y"], gt_tree.data["z"],
                        c='green', marker='*', s=50, zorder=10)

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("Generated (Flow Matching)")
            draw_tree_centerlines(gen_tree, ax2, color='darkred', linewidth=1.2, alpha=0.5)
            draw_tree_splines_scatter(gen_tree, ax2, color='red', spline_samples=60)
            ax2.scatter(gen_tree.data["x"], gen_tree.data["y"], gen_tree.data["z"],
                        c='green', marker='*', s=50, zorder=10)
            _sync_3d_axes(ax1, ax2)
        else:
            fig = plt.figure(figsize=(8, 7))
            fig.suptitle(f"{name}  —  {len(gen_data)} nodes", fontsize=12)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Generated (Flow Matching)")
            draw_tree_centerlines(gen_tree, ax, color='darkred', linewidth=1.2, alpha=0.5)
            draw_tree_splines_scatter(gen_tree, ax, color='red', spline_samples=60)
            ax.scatter(gen_tree.data["x"], gen_tree.data["y"], gen_tree.data["z"],
                       c='green', marker='*', s=50, zorder=10)

        plt.tight_layout()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig_path = Path(output_dir) / f"{name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {fig_path}")
        plt.close(fig)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(gen_abs, gt_abs):
    """Compute per-tree metrics between generated and GT absolute data."""
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

    # CP radius: true cross-section radius (distance from node position)
    # CPs in gen_abs are ABSOLUTE world coords → subtract node position
    cp_x_gen = cp_gen[:, 0:8] - pos_gen[:, 0:1]    # [n, 8]
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
        "gen_cp_radius": float(gen_radius),
        "gt_cp_radius": float(gt_radius),
        "radius_ratio": float(gen_radius / max(gt_radius, 1e-8)),
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(cfg):
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})

    device = resolve_device(params.get("device", 0))
    seed_all(int(params.get("seed", 42)))

    n_samples = int(params.get("n_samples", 20))
    n_steps = int(params.get("n_steps", 50))
    output_dir = Path(params.get("output_dir", "Stage2_FlowMatching/generated/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "npy").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # ── Load model ──
    ckpt_path = paths["checkpoint"]
    print(f"Loading model from {ckpt_path}")

    model = FlowMatchingVelocityModel(
        geom_dim=int(model_cfg.get("geom_dim", 39)),
        k_classes=int(model_cfg.get("k_classes", 3)),
        max_depth=int(model_cfg.get("max_depth", 128)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_ff=int(model_cfg.get("d_ff", 1024)),
        max_nodes=int(model_cfg.get("max_nodes", 256)),
        dropout=0.0,
        tree_attn_hops=int(model_cfg.get("tree_attn_hops", 0)),
        input_clamp_value=model_cfg.get("input_clamp_value", None),
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
        cfg_dropout=float(model_cfg.get("cfg_dropout", 0.0)),
        depth_in_geometry=bool(model_cfg.get("depth_in_geometry", False)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA weights if available
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

    # ── Load feature stats for denormalization ──
    feature_stats_path = paths.get("feature_stats", None)
    feature_mean, feature_std = None, None
    if feature_stats_path and os.path.exists(feature_stats_path):
        feature_mean, feature_std = load_feature_stats(feature_stats_path, device)
        print(f"Feature denormalization: loaded from {feature_stats_path}")
    else:
        # Check if checkpoint has the path
        if isinstance(ckpt, dict) and ckpt.get("feature_stats_path"):
            fsp = ckpt["feature_stats_path"]
            if os.path.exists(fsp):
                feature_mean, feature_std = load_feature_stats(fsp, device)
                print(f"Feature denormalization: loaded from checkpoint ref {fsp}")
        if feature_mean is None:
            print("Feature denormalization: DISABLED (no feature_stats path)")

    # ── Load validation data ──
    val_dir = paths["val_dir"]
    print(f"Loading validation trees from {val_dir}")
    gt_trees_raw, names = load_val_trees(val_dir, n_samples=n_samples, shuffle=True)
    print(f"  Loaded {len(gt_trees_raw)} trees")

    # ── Absolute positions mode ──
    abs_pos_mode = bool(cfg.get("data", {}).get("absolute_positions", False))
    if abs_pos_mode:
        print("Position mode: ABSOLUTE (model predicts world coordinates)")
    else:
        print("Position mode: RELATIVE (model predicts parent-relative offsets)")

    # ── Sampling mode: normal vs guided ──
    sampling_mode = str(params.get("sampling_mode", "normal"))
    physio_fn = None
    physio_kwargs = {}
    guidance_cfg = cfg.get("guidance", {})

    if sampling_mode == "guided":
        from Stage2_FlowMatching_Physio.physio_losses import physiological_loss
        physio_fn = physiological_loss
        physio_kwargs = {
            "murray_weight": float(guidance_cfg.get("murray_weight", 1.0)),
            "murray_gamma": float(guidance_cfg.get("murray_gamma", 3.0)),
            "bifurcation_angle_weight": float(guidance_cfg.get("bifurcation_angle_weight", 0.0)),
            "target_angle_deg": float(guidance_cfg.get("target_angle_deg", 70.0)),
            "angle_margin_deg": float(guidance_cfg.get("angle_margin_deg", 20.0)),
            "tapering_weight": float(guidance_cfg.get("tapering_weight", 0.0)),
            "symmetry_weight": float(guidance_cfg.get("symmetry_weight", 0.0)),
            "symmetry_target": float(guidance_cfg.get("symmetry_target", 0.8)),
            "symmetry_margin": float(guidance_cfg.get("symmetry_margin", 0.2)),
            "sibling_cosine_weight": float(guidance_cfg.get("sibling_cosine_weight", 0.0)),
            "sibling_cosine_target": float(guidance_cfg.get("sibling_cosine_target", -0.5)),
            "depth_radius_weight": float(guidance_cfg.get("depth_radius_weight", 0.0)),
            "depth_radius_target_decay": float(guidance_cfg.get("depth_radius_target_decay", 0.95)),
        }
        print(f"Sampling mode: GUIDED  (λ={guidance_cfg.get('strength', 0.1)}, "
              f"t∈[{guidance_cfg.get('t_min', 0.3)}, {guidance_cfg.get('t_max', 0.95)}], "
              f"schedule={guidance_cfg.get('schedule', 'linear')})")
        print(f"  Physio weights: {physio_kwargs}")
    else:
        print(f"Sampling mode: NORMAL (no physio guidance)")

    # ── Generate ──
    print(f"\n=== Generating (n_steps={n_steps}, mode={sampling_mode}) ===")
    gen_data_list = []
    gt_data_list = []
    all_metrics = []

    for i, (gt_raw, name) in enumerate(zip(gt_trees_raw, names)):
        # Extract topology from GT
        k_counts = np.clip(np.rint(gt_raw[:, 0]), 0, 2).astype(np.int64)
        parents = preorder_kcount_parent_indices(k_counts)
        depths, child_slots = compute_depths_and_child_slots(parents)
        n_nodes = len(k_counts)

        # Tensors for generation
        k_t = torch.from_numpy(k_counts).long().unsqueeze(0).to(device)
        d_t = torch.from_numpy(depths).long().unsqueeze(0).to(device)
        cs_t = torch.from_numpy(child_slots).long().unsqueeze(0).to(device)
        nm_t = torch.ones(1, n_nodes, dtype=torch.bool, device=device)
        par_t = torch.from_numpy(parents).long().unsqueeze(0).to(device)

        # Generate via Euler integration
        # Parse per-group velocity scales if configured
        vsg_raw = params.get("velocity_scale_per_group", None)
        vsg = None
        if vsg_raw:
            vsg = {}
            for key, val in vsg_raw.items():
                parts = key.split(":")
                vsg[(int(parts[0]), int(parts[1]))] = float(val)

        common_kwargs = dict(
            node_mask=nm_t, parents=par_t,
            n_steps=n_steps,
            velocity_scale=float(params.get("velocity_scale", 1.0)),
            velocity_scale_per_group=vsg,
            solver=str(params.get("solver", "euler")),
            time_schedule=str(params.get("time_schedule", "linear")),
            guidance_scale=float(params.get("guidance_scale", 1.0)),
        )

        if sampling_mode == "guided" and physio_fn is not None:
            # Physio-Guided Sampling — needs gradients
            pred_local = model.sample_guided(
                k_t, d_t, cs_t,
                **common_kwargs,
                physio_fn=physio_fn,
                physio_kwargs=physio_kwargs,
                guidance_strength=float(guidance_cfg.get("strength", 0.1)),
                guidance_t_min=float(guidance_cfg.get("t_min", 0.3)),
                guidance_t_max=float(guidance_cfg.get("t_max", 0.95)),
                guidance_schedule=str(guidance_cfg.get("schedule", "linear")),
                guidance_grad_clip=float(guidance_cfg.get("grad_clip", 1.0)),
            )  # [1, N, 39]
        else:
            # Standard sampling — no gradients
            with torch.no_grad():
                pred_local = model.sample(
                    k_t, d_t, cs_t,
                    **common_kwargs,
                )  # [1, N, 39]

        pred_39 = pred_local[0].cpu().numpy()  # [N, 39]

        # ── Denormalize if feature stats were used during training ────
        if feature_mean is not None and feature_std is not None:
            fm = feature_mean.cpu().numpy()  # [39]
            fs = feature_std.cpu().numpy()   # [39]

            # Optional: post-hoc output scale correction
            # The model may produce features at a different scale than expected.
            # If output_scale_correction is a dict, apply per-group correction
            # factors computed from training data analysis.
            osc = params.get("output_scale_correction", False)
            if isinstance(osc, dict):
                for key, factor in osc.items():
                    parts = key.split(":")
                    a, b = int(parts[0]), int(parts[1])
                    pred_39[:, a:b] *= float(factor)
            elif osc:
                # Automatic per-tree normalization (divide by actual std)
                if pred_39.shape[0] > 1:
                    for (a, b) in [(0, 27), (27, 39)]:
                        cur_std = pred_39[:, a:b].std()
                        if cur_std > 1e-6:
                            pred_39[:, a:b] *= (1.0 / cur_std)

            pred_39 = pred_39 * fs + fm

        # Reconstruct [N, 40]
        gen_40 = np.zeros((n_nodes, 40), dtype=np.float32)
        gen_40[:, 0] = k_counts.astype(np.float32)
        gen_40[:, 1:] = pred_39

        # ── Fix 1: Knot vectors — enforce monotonicity (sort) ─────────
        for ni in range(n_nodes):
            knots = gen_40[ni, 28:40]
            gen_40[ni, 28:40] = np.sort(knots)

        # ── Fix 2: Control points — enforce planarity ─────────────────
        # GT cross-sections are exactly planar (all 8 CPs lie on a plane).
        # Project each node's 8 CPs onto their best-fit plane via SVD.
        for ni in range(n_nodes):
            cp_x = gen_40[ni, 4:12]    # 8 values
            cp_y = gen_40[ni, 12:20]   # 8 values
            cp_z = gen_40[ni, 20:28]   # 8 values
            cps = np.column_stack([cp_x, cp_y, cp_z])  # [8, 3]
            centroid = cps.mean(axis=0)
            centered = cps - centroid
            U, S, Vh = np.linalg.svd(centered, full_matrices=False)
            # Keep only the first 2 principal components (zero out 3rd)
            projected = U[:, :2] @ np.diag(S[:2]) @ Vh[:2, :]
            cps_planar = projected + centroid
            gen_40[ni, 4:12] = cps_planar[:, 0]
            gen_40[ni, 12:20] = cps_planar[:, 1]
            gen_40[ni, 20:28] = cps_planar[:, 2]

        # ── Fix 3: Align CS normal to branch direction ───────────────
        # In GT, 95% of cross-section normals are strongly aligned with
        # the branch direction.  Rotate each CS so its plane normal
        # matches the branch direction (Rodrigues' rotation).
        # In abs-pos mode, direction = pos[node] - pos[parent].
        # In rel-pos mode, direction = rel_pos[node] directly.
        parents_np = preorder_kcount_parent_indices(k_counts)
        for ni in range(n_nodes):
            if abs_pos_mode:
                pi = parents_np[ni]
                if pi < 0:
                    continue  # root node — no parent direction
                rp = gen_40[ni, 1:4] - gen_40[pi, 1:4]
            else:
                rp = gen_40[ni, 1:4]
            rp_len = np.linalg.norm(rp)
            if rp_len < 1e-8:
                continue  # no direction defined
            target = rp / rp_len

            cps = np.column_stack([gen_40[ni, 4:12],
                                   gen_40[ni, 12:20],
                                   gen_40[ni, 20:28]])  # [8, 3]
            centroid = cps.mean(axis=0)
            centered = cps - centroid
            _, _, Vh = np.linalg.svd(centered, full_matrices=False)
            normal = Vh[2]  # current plane normal

            # Pick sign closest to target direction
            if np.dot(normal, target) < 0:
                normal = -normal

            c = np.dot(normal, target)
            if abs(c - 1.0) < 1e-6:
                continue  # already aligned

            if abs(c + 1.0) < 1e-6:
                # 180° — rotate around the first principal axis
                axis = Vh[0]
                R = 2.0 * np.outer(axis, axis) - np.eye(3)
            else:
                v = np.cross(normal, target)
                s = np.linalg.norm(v)
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

            rotated = (R @ centered.T).T + centroid
            gen_40[ni, 4:12] = rotated[:, 0]
            gen_40[ni, 12:20] = rotated[:, 1]
            gen_40[ni, 20:28] = rotated[:, 2]

        # ── Fix 4: Periodic B-spline refit + aspect ratio correction ──
        # GT splines use splprep(per=True): all cross-sections are
        # perfectly closed (P(0)=P(1)) with median aspect ratio ~0.786.
        # Steps per node:
        #   a) Evaluate the model's spline at 64 points
        #   b) Re-fit through splprep(per=True) to guarantee closure
        #   c) Aspect ratio correction on the REFIT CPs (after splprep,
        #      which tends to push toward circular)
        import warnings as _w
        _target_aspect = 0.786   # GT median aspect ratio (S2/S1)
        _aspect_alpha = 0.35     # 0=no change, 1=force to target
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
                continue  # keep original if evaluation fails

            # Remember pre-refit CP RMS for radius compensation
            old_cps = np.column_stack([cp_x, cp_y, cp_z])
            old_cent = old_cps.mean(axis=0)
            old_cp_rms = np.sqrt(np.mean(np.sum((old_cps - old_cent)**2, axis=1)))

            # ── (b) Periodic refit ──
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
                continue  # keep original if refit fails

            t_new, c_new, _ = tck_new
            # Only accept if the refit produced exactly 8 CPs / 12 knots
            # AND the new CPs don't explode (sanity check)
            if len(c_new[0]) >= 8 and len(t_new) >= 12:
                new_cp = np.column_stack([c_new[0][:8], c_new[1][:8], c_new[2][:8]])
                old_cp = np.column_stack([cp_x, cp_y, cp_z])
                new_norm = np.linalg.norm(new_cp - new_cp.mean(0))
                old_norm = np.linalg.norm(old_cp - old_cp.mean(0))
                # Reject if refit CPs are >3× larger than original
                if old_norm > 1e-10 and new_norm / old_norm > 3.0:
                    continue
                # Radius-preserving: scale refit CPs so CP RMS matches pre-refit
                # Use damping (0.75) because splprep also shifts CPs slightly
                _rp_damp = 0.75
                new_cent = new_cp.mean(axis=0)
                new_cp_rms = np.sqrt(np.mean(np.sum((new_cp - new_cent)**2, axis=1)))
                if new_cp_rms > 1e-10 and old_cp_rms > 1e-10:
                    r_scale = 1.0 + _rp_damp * (old_cp_rms / new_cp_rms - 1.0)
                    new_cp = (new_cp - new_cent) * r_scale + new_cent
                gen_40[ni, 4:12]  = new_cp[:, 0]
                gen_40[ni, 12:20] = new_cp[:, 1]
                gen_40[ni, 20:28] = new_cp[:, 2]
                gen_40[ni, 28:40] = t_new[:12]

            # ── (c) Aspect ratio correction (AFTER refit) ──
            # The splprep refit tends to push shapes toward circular.
            # Correct the aspect ratio of the final 8 CPs via SVD.
            # Area-preserving: scale so that S[0]*S[1] stays constant.
            cps = np.column_stack([gen_40[ni, 4:12],
                                   gen_40[ni, 12:20],
                                   gen_40[ni, 20:28]])  # [8, 3]
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
                # Area-preserving scale: maintain S[0]*S[1]
                new_area = S_cp_new[0] * S_cp_new[1]
                if new_area > 1e-20:
                    scale = np.sqrt(old_area / new_area)
                    S_cp_new[0] *= scale
                    S_cp_new[1] *= scale
                cps_corrected = U_cp @ np.diag(S_cp_new) @ Vh_cp + centroid
                gen_40[ni, 4:12]  = cps_corrected[:, 0]
                gen_40[ni, 12:20] = cps_corrected[:, 1]
                gen_40[ni, 20:28] = cps_corrected[:, 2]

        # ── Fix 5: Re-center CPs at origin in node-local frame ──
        # GT cross-section CPs always have centroid ≈ 0 in node-local frame.
        # The model regresses each CP component independently, so small per-
        # component errors accumulate into a systematic centroid offset
        # (~7σ in z-score units) → cross-section sits beside the centerline.
        from tree_functions import recenter_node_local_cps
        gen_40 = recenter_node_local_cps(gen_40, copy=False)

        # Convert to absolute
        # abs_pos_mode: positions are already absolute, only need to add pos→CPs
        # rel_pos_mode: need both relative→absolute pos AND node-local→absolute CPs
        gen_abs = local_geometry_tree_to_absolute(
            gen_40,
            position_slice=(1, 4),
            control_point_slices=((4, 12), (12, 20), (20, 28)),
            relative_positions=(not abs_pos_mode),
            node_local_control_points=True,
            copy=True,
        )

        # GT absolute  (raw data is always in local relative format from disk)
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

        # Metrics
        m = compute_metrics(gen_abs, gt_abs)
        m["name"] = name
        m["n_nodes"] = n_nodes
        all_metrics.append(m)

        np.save(output_dir / "npy" / f"{name}.npy", gen_abs)
        np.save(output_dir / "npy" / f"{name}_gt.npy", gt_abs)
        np.save(output_dir / "npy" / f"{name}_local.npy", gen_40)
        np.save(output_dir / "npy" / f"{name}_gt_local.npy", gt_raw)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(gt_trees_raw)}] {name}: {n_nodes} nodes, "
                  f"pos_mae={m['pos_mae']:.4f}, radius_ratio={m['radius_ratio']:.3f}")

    # ── Aggregate metrics ──
    avg_pos_mae = np.mean([m["pos_mae"] for m in all_metrics])
    avg_cp_mae = np.mean([m["cp_mae"] for m in all_metrics])
    avg_radius = np.mean([m["radius_ratio"] for m in all_metrics])
    print(f"\n=== Metrics (n={len(all_metrics)}) ===")
    print(f"  avg pos_mae:     {avg_pos_mae:.5f}")
    print(f"  avg cp_mae:      {avg_cp_mae:.5f}")
    print(f"  avg radius_ratio: {avg_radius:.3f}× GT")

    # Save metrics
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    # ── Visualize ──
    print(f"\n=== Visualizing ===")
    visualize_trees(
        gen_data_list, gt_data_list, names,
        output_dir=str(output_dir / "images"),
    )

    meta = {
        "n_samples": len(gen_data_list),
        "n_steps": n_steps,
        "avg_pos_mae": avg_pos_mae,
        "avg_cp_mae": avg_cp_mae,
        "avg_radius_ratio": avg_radius,
    }
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Done. {len(gen_data_list)} trees → {output_dir}")
    return gen_data_list, gt_data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--time_schedule", type=str, default=None, choices=["linear", "quadratic"])
    parser.add_argument("--velocity_scale", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--sampling_mode", type=str, default=None,
                        choices=["normal", "guided"],
                        help="Sampling mode: 'normal' or 'guided' (physio-guided)")
    parser.add_argument("--guidance_strength", type=float, default=None,
                        help="Physio guidance strength λ (only for --sampling_mode=guided)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg.setdefault("params", {})["n_samples"] = args.n_samples
    if args.n_steps is not None:
        cfg.setdefault("params", {})["n_steps"] = args.n_steps
    if args.seed is not None:
        cfg.setdefault("params", {})["seed"] = args.seed
    if args.solver is not None:
        cfg.setdefault("params", {})["solver"] = args.solver
    if args.time_schedule is not None:
        cfg.setdefault("params", {})["time_schedule"] = args.time_schedule
    if args.velocity_scale is not None:
        cfg.setdefault("params", {})["velocity_scale"] = args.velocity_scale
    if args.guidance_scale is not None:
        cfg.setdefault("params", {})["guidance_scale"] = args.guidance_scale
    if args.sampling_mode is not None:
        cfg.setdefault("params", {})["sampling_mode"] = args.sampling_mode
    if args.guidance_strength is not None:
        cfg.setdefault("guidance", {})["strength"] = args.guidance_strength

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
