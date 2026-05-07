"""
Train the Flow Matching velocity model for vessel tree geometry generation.

Flow Matching (Optimal-Transport Conditional Flow Matching / OT-CFM):
    x_t = (1 - t) * ε  +  t * x_0          ε ~ N(0, I),  t ~ U(0, 1)
    target velocity  u  = x_0 - ε           (straight-line path)
    loss = || v_θ(x_t, t | topo) - u ||²   (per-node, masked)

Usage:
    python Stage2_FlowMatching/train_flow_matching.py \
        --config Stage2_FlowMatching/configs/flow_matching_v1.yaml
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    raise RuntimeError("pip install pyyaml")

import copy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stage2_FlowMatching.model import FlowMatchingVelocityModel
from tree_functions import preorder_kcount_parent_indices, parent_relative_positions_to_absolute


# ── EMA (Exponential Moving Average) ────────────────────────────────────────

class EMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        # Also update buffers (running stats, etc.)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

# ── Dataset ──────────────────────────────────────────────────────────────────

GEOM_START, GEOM_END = 1, 40


def load_feature_stats(path, device="cpu"):
    """Load per-feature mean/std from a .npz file for z-score normalization."""
    data = np.load(path)
    mean = torch.from_numpy(data["mean"]).float().to(device)  # [39]
    std = torch.from_numpy(data["std"]).float().to(device)    # [39]
    return mean, std


def list_npy_files(folder, limit=None):
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".npy") and not f.startswith(".")
    )
    return files[:limit] if limit else files


def compute_depths_and_child_slots(parents):
    parents = np.asarray(parents, dtype=np.int64).ravel()
    depths = np.zeros_like(parents)
    child_slots = np.zeros_like(parents)
    child_counts = {}
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            child_slots[i] = 0
            continue
        depths[i] = depths[p] + 1
        slot = child_counts.get(p, 0)
        child_slots[i] = 1 if slot == 0 else 2
        child_counts[p] = slot + 1
    return depths, child_slots


class TreeGeometryDataset(torch.utils.data.Dataset):
    """Load [N, 40] .npy files, return geometry [N, 39] + topology.
    
    If absolute_positions=True, convert parent-relative positions to absolute
    world coordinates before returning.  CPs remain node-local.
    This eliminates error accumulation along tree branches during generation.
    """

    def __init__(self, folder, limit=None, absolute_positions=False):
        self.files = list_npy_files(folder, limit)
        self.absolute_positions = absolute_positions

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        # strip trailing all-zero rows
        valid = ~(np.all(np.abs(arr[:, 1:]) < 1e-8, axis=1))
        arr = arr[valid]

        # Optionally convert parent-relative positions → absolute
        if self.absolute_positions:
            arr = parent_relative_positions_to_absolute(
                arr, position_slice=(1, 4), copy=True
            )

        k = np.clip(np.rint(arr[:, 0]), 0, 2).astype(np.int64)
        parents = preorder_kcount_parent_indices(k)
        depths, child_slots = compute_depths_and_child_slots(parents)

        return {
            "geometry": torch.from_numpy(arr[:, GEOM_START:GEOM_END]),
            "k_counts": torch.from_numpy(k),
            "parents": torch.from_numpy(parents.astype(np.int64)),
            "depths": torch.from_numpy(depths.astype(np.int64)),
            "child_slots": torch.from_numpy(child_slots.astype(np.int64)),
        }


def collate_fn(batch):
    B = len(batch)
    mx = max(b["k_counts"].shape[0] for b in batch)
    G = batch[0]["geometry"].shape[1]

    geometry = torch.zeros(B, mx, G)
    k_counts = torch.zeros(B, mx, dtype=torch.long)
    parents = torch.full((B, mx), -1, dtype=torch.long)
    depths = torch.zeros(B, mx, dtype=torch.long)
    child_slots = torch.zeros(B, mx, dtype=torch.long)
    node_mask = torch.zeros(B, mx, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["k_counts"].shape[0]
        geometry[i, :n] = b["geometry"]
        k_counts[i, :n] = b["k_counts"]
        parents[i, :n] = b["parents"]
        depths[i, :n] = b["depths"]
        child_slots[i, :n] = b["child_slots"]
        node_mask[i, :n] = True

    return {
        "geometry": geometry,
        "k_counts": k_counts,
        "parents": parents,
        "depths": depths,
        "child_slots": child_slots,
        "node_mask": node_mask,
    }


# ── Loss ─────────────────────────────────────────────────────────────────────

def sample_logit_normal(B, device, mean=0.0, std=1.0):
    """Sample t from logit-normal distribution (more density near 0 and 1)."""
    z = torch.randn(B, device=device) * std + mean
    t = torch.sigmoid(z)
    # Clamp to avoid numerical issues at boundaries
    return t.clamp(1e-5, 1.0 - 1e-5)


def flow_matching_loss(
    model, batch, device,
    pos_weight=1.0, cp_weight=1.0, knots_weight=1.0,
    time_sampling="uniform",
    self_cond_prob=0.0,
    knot_mono_weight=0.0,
    planarity_weight=0.0,
    radius_weight=0.0,
    eccentricity_weight=0.0,
    feature_mean=None,
    feature_std=None,
):
    """
    Compute OT-CFM loss for one batch.

    x_t = (1-t) * noise + t * x_0
    target = x_0 - noise
    loss   = masked MSE( v_θ(x_t, t | topo),  target )

    Self-conditioning: with prob self_cond_prob, first predict x̂_1 without
    self-cond, then feed detached x̂_1 as conditioning for the actual loss.
    
    knot_mono_weight: If > 0, adds a soft penalty for non-monotonic knots
    in the estimated clean data x̂_1 = x_t + (1-t)*v.

    planarity_weight: If > 0, penalizes the 3rd singular value of the
    8 cross-section control points in x̂_1 (should be ~0 for planar CPs).

    radius_weight: If > 0, adds a log-space radius loss that penalizes
    discrepancies in cross-section RMS radius between x̂_1 and x_0.
    Uses log-space MSE for scale-invariant penalty (equal weight on
    small and large vessels).  Addresses the systematic radius-shrinkage
    problem where large vessel radii get compressed to ~0.4× GT.

    eccentricity_weight: If > 0, adds a Gram-matrix loss on the shape of
    the 8 cross-section CPs: ||C_hat^T C_hat - C_gt^T C_gt||_F^2.
    This captures both aspect ratio and orientation WITHOUT using SVD,
    avoiding gradient instability when singular values are degenerate
    (S0 ≈ S1, i.e. near-circular cross-sections).  Everywhere smooth
    and differentiable.
    """
    x_0 = batch["geometry"].to(device)   # [B, N, 39]
    k = batch["k_counts"].to(device)
    d = batch["depths"].to(device)
    cs = batch["child_slots"].to(device)
    nm = batch["node_mask"].to(device)
    par = batch["parents"].to(device)

    B, N, G = x_0.shape
    mask_f = nm.float().unsqueeze(-1)  # [B, N, 1]

    # ── Z-score normalization ─────────────────────────────────────────
    # If feature_mean/std are provided, normalize x_0 so all features
    # are ~N(0,1).  This makes the noise→data flow path well-conditioned
    # across all feature groups (geometry std≈0.04, knots std≈0.83).
    if feature_mean is not None and feature_std is not None:
        x_0 = (x_0 - feature_mean.view(1, 1, G)) / feature_std.view(1, 1, G)

    # Sample t  and  noise ~ N(0, I)
    if time_sampling == "logit_normal":
        t = sample_logit_normal(B, device, mean=0.0, std=1.0)
    else:
        t = torch.rand(B, device=device)
    noise = torch.randn_like(x_0)

    # Interpolate
    t_expand = t.view(B, 1, 1)
    x_t = (1.0 - t_expand) * noise + t_expand * x_0

    # Target velocity (straight path)
    target = x_0 - noise  # [B, N, 39]

    # Self-conditioning
    x_self_cond = None
    if model.self_conditioning and self_cond_prob > 0.0 and random.random() < self_cond_prob:
        with torch.no_grad():
            v_sc = model(k, d, cs, x_t, t, node_mask=nm, parents=par, x_self_cond=None)
            remaining = (1.0 - t_expand).clamp(min=1e-4)
            x_self_cond = (x_t + remaining * v_sc).detach() * mask_f

    # Predict
    v_pred = model(k, d, cs, x_t, t, node_mask=nm, parents=par, x_self_cond=x_self_cond)

    # Masked weighted MSE
    diff2 = (v_pred - target) ** 2     # [B, N, 39]

    # Position [0:3], CPs [3:27], Knots [27:39]
    pos_loss = (diff2[..., 0:3] * mask_f).sum() / mask_f.sum().clamp(min=1) / 3
    cp_loss = (diff2[..., 3:27] * mask_f).sum() / mask_f.sum().clamp(min=1) / 24
    knot_loss = (diff2[..., 27:39] * mask_f).sum() / mask_f.sum().clamp(min=1) / 12

    total = pos_weight * pos_loss + cp_weight * cp_loss + knots_weight * knot_loss

    # ── Regularization losses on estimated clean data x̂_1 ────────────
    mono_loss = torch.tensor(0.0, device=device)
    planar_loss = torch.tensor(0.0, device=device)
    radius_loss = torch.tensor(0.0, device=device)
    ecc_loss = torch.tensor(0.0, device=device)

    need_x_hat = (knot_mono_weight > 0.0) or (planarity_weight > 0.0) or (radius_weight > 0.0) or (eccentricity_weight > 0.0)
    if need_x_hat:
        remaining = (1.0 - t_expand).clamp(min=1e-4)
        x_hat_1 = x_t + remaining * v_pred  # [B, N, 39]

    # Knot monotonicity regularization
    if knot_mono_weight > 0.0:
        knots_hat = x_hat_1[..., 27:39]  # [B, N, 12]
        diffs = knots_hat[..., 1:] - knots_hat[..., :-1]  # [B, N, 11]
        violations = torch.relu(-diffs)
        mono_loss = (violations * mask_f).sum() / mask_f.sum().clamp(min=1) / 11
        total = total + knot_mono_weight * mono_loss

    # Planarity regularization: penalize 3rd singular value of CPs
    if planarity_weight > 0.0:
        cps_hat = x_hat_1[..., 3:27]  # [B, N, 24]  (8 CPs × 3 axes)
        cps_3d = cps_hat.reshape(B, N, 8, 3)
        centroid = cps_3d.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
        centered = cps_3d - centroid  # [B, N, 8, 3]
        centered_flat = centered.reshape(B * N, 8, 3)
        # Batched SVD — S has shape [B*N, 3]
        _, S_vals, _ = torch.linalg.svd(centered_flat, full_matrices=False)
        s3 = S_vals[:, 2].reshape(B, N)  # 3rd singular value
        nm_f = nm.float()  # [B, N]
        planar_loss = (s3 * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + planarity_weight * planar_loss

    # Radius loss: log-space MSE on RMS cross-section radius
    if radius_weight > 0.0:
        # Correctly reshape CPs: [cp_x(8), cp_y(8), cp_z(8)] → [8, 3]
        cp_x_hat = x_hat_1[..., 3:11]    # [B, N, 8]
        cp_y_hat = x_hat_1[..., 11:19]   # [B, N, 8]
        cp_z_hat = x_hat_1[..., 19:27]   # [B, N, 8]
        cps_hat_3d = torch.stack([cp_x_hat, cp_y_hat, cp_z_hat], dim=-1)  # [B, N, 8, 3]

        cp_x_0 = x_0[..., 3:11]
        cp_y_0 = x_0[..., 11:19]
        cp_z_0 = x_0[..., 19:27]
        cps_0_3d = torch.stack([cp_x_0, cp_y_0, cp_z_0], dim=-1)  # [B, N, 8, 3]

        # Subtract centroid for correct radius (distance from cross-section center)
        cent_hat = cps_hat_3d.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
        cent_0 = cps_0_3d.mean(dim=2, keepdim=True)
        centered_hat = cps_hat_3d - cent_hat  # [B, N, 8, 3]
        centered_0 = cps_0_3d - cent_0

        # RMS radius per node
        eps_r = 1e-8
        r_hat = (centered_hat.pow(2).sum(dim=-1).mean(dim=-1) + eps_r).sqrt()   # [B, N]
        r_0 = (centered_0.pow(2).sum(dim=-1).mean(dim=-1) + eps_r).sqrt()       # [B, N]

        # Log-space MSE: scale-invariant penalty
        log_r_hat = torch.log(r_hat + 1e-6)
        log_r_0 = torch.log(r_0 + 1e-6)
        nm_f = nm.float()
        radius_loss = ((log_r_hat - log_r_0).pow(2) * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + radius_weight * radius_loss

    # Gram-matrix loss: ||C_hat^T C_hat - C_gt^T C_gt||_F^2
    # Penalizes shape (aspect ratio + orientation) mismatch without SVD.
    # Everywhere smooth — no gradient singularity when S0 ≈ S1.
    if eccentricity_weight > 0.0:
        # Reshape CPs: [cp_x(8), cp_y(8), cp_z(8)] → [B, N, 8, 3]
        cp_x_h = x_hat_1[..., 3:11]
        cp_y_h = x_hat_1[..., 11:19]
        cp_z_h = x_hat_1[..., 19:27]
        cps_h = torch.stack([cp_x_h, cp_y_h, cp_z_h], dim=-1)  # [B, N, 8, 3]

        cp_x_g = x_0[..., 3:11]
        cp_y_g = x_0[..., 11:19]
        cp_z_g = x_0[..., 19:27]
        cps_g = torch.stack([cp_x_g, cp_y_g, cp_z_g], dim=-1)  # [B, N, 8, 3]

        # Center CPs (remove translation)
        cps_h_c = cps_h - cps_h.mean(dim=2, keepdim=True)  # [B, N, 8, 3]
        cps_g_c = cps_g - cps_g.mean(dim=2, keepdim=True)

        # Gram matrices: C^T C → [B, N, 3, 3]
        G_h = torch.matmul(cps_h_c.transpose(-2, -1), cps_h_c)  # [B, N, 3, 3]
        G_g = torch.matmul(cps_g_c.transpose(-2, -1), cps_g_c)  # [B, N, 3, 3]

        # Frobenius norm of difference, per node
        diff = G_h - G_g  # [B, N, 3, 3]
        gram_frob = diff.pow(2).sum(dim=(-2, -1))  # [B, N]

        # Normalize by GT Gram norm to make scale-invariant
        g_norm = G_g.pow(2).sum(dim=(-2, -1)).clamp(min=1e-12)  # [B, N]
        gram_rel = gram_frob / g_norm  # [B, N]

        nm_f_ecc = nm.float()
        ecc_loss = (gram_rel * nm_f_ecc).sum() / nm_f_ecc.sum().clamp(min=1)
        total = total + eccentricity_weight * ecc_loss

    return {
        "total": total,
        "pos": pos_loss,
        "cp": cp_loss,
        "knots": knot_loss,
        "mono": mono_loss,
        "planar": planar_loss,
        "radius": radius_loss,
        "ecc": ecc_loss,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, loss_cfg, ema=None,
                    feature_mean=None, feature_std=None):
    model.train()
    accum = {"total": 0.0, "pos": 0.0, "cp": 0.0, "knots": 0.0, "mono": 0.0, "planar": 0.0, "radius": 0.0, "ecc": 0.0}
    n = 0
    for batch in loader:
        losses = flow_matching_loss(model, batch, device,
                                    feature_mean=feature_mean,
                                    feature_std=feature_std,
                                    **loss_cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update per optimization step (critical: NOT per epoch!)
        if ema is not None:
            ema.update(model)

        bs = batch["node_mask"].shape[0]
        for k_name in accum:
            accum[k_name] += losses[k_name].item() * bs
        n += bs
    return {k_name: v / max(n, 1) for k_name, v in accum.items()}


@torch.no_grad()
def val_one_epoch(model, loader, device, loss_cfg,
                  feature_mean=None, feature_std=None):
    model.eval()
    accum = {"total": 0.0, "pos": 0.0, "cp": 0.0, "knots": 0.0, "mono": 0.0, "planar": 0.0, "radius": 0.0, "ecc": 0.0}
    n = 0
    # Remove self_cond_prob for validation — always use basic forward
    # Also remove regularization weights for clean validation
    val_loss_cfg = {k: v for k, v in loss_cfg.items()
                    if k not in ("self_cond_prob", "knot_mono_weight", "planarity_weight", "radius_weight", "eccentricity_weight")}
    val_loss_cfg["self_cond_prob"] = 0.0
    val_loss_cfg["knot_mono_weight"] = 0.0
    val_loss_cfg["planarity_weight"] = 0.0
    val_loss_cfg["radius_weight"] = 0.0
    val_loss_cfg["eccentricity_weight"] = 0.0
    for batch in loader:
        losses = flow_matching_loss(model, batch, device,
                                    feature_mean=feature_mean,
                                    feature_std=feature_std,
                                    **val_loss_cfg)
        bs = batch["node_mask"].shape[0]
        for k_name in accum:
            accum[k_name] += losses[k_name].item() * bs
        n += bs
    return {k_name: v / max(n, 1) for k_name, v in accum.items()}


@torch.no_grad()
def val_rollout(model, loader, device, n_steps=50, n_samples=4,
                feature_mean=None, feature_std=None):
    """Run full generation and compare to GT (first n_samples from loader)."""
    model.eval()
    total_mse, count = 0.0, 0

    for batch in loader:
        B = batch["node_mask"].shape[0]
        for i in range(min(B, n_samples - count)):
            n_nodes = batch["node_mask"][i].sum().item()
            k = batch["k_counts"][i:i+1, :n_nodes].to(device)
            d = batch["depths"][i:i+1, :n_nodes].to(device)
            cs = batch["child_slots"][i:i+1, :n_nodes].to(device)
            nm = batch["node_mask"][i:i+1, :n_nodes].to(device)
            par = batch["parents"][i:i+1, :n_nodes].to(device)
            gt = batch["geometry"][i:i+1, :n_nodes].to(device)

            pred = model.sample(k, d, cs, node_mask=nm, parents=par, n_steps=n_steps)

            # Denormalize prediction and normalize GT for comparison
            if feature_mean is not None and feature_std is not None:
                # pred is in normalized space — denormalize
                G = pred.shape[-1]
                pred_denorm = pred * feature_std.view(1, 1, G) + feature_mean.view(1, 1, G)
                mse = ((pred_denorm - gt) ** 2 * nm.float().unsqueeze(-1)).sum() / (nm.sum() * 39)
            else:
                mse = ((pred - gt) ** 2 * nm.float().unsqueeze(-1)).sum() / (nm.sum() * 39)

            total_mse += mse.item()
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    return total_mse / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})
    loss_cfg_raw = cfg.get("loss", {})

    device = resolve_device(params.get("device", 0))
    seed_all(int(params.get("seed", 42)))

    output_dir = Path(paths.get("output_dir", "Stage2_FlowMatching/output/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    abs_pos = bool(cfg.get("data", {}).get("absolute_positions", False))
    if abs_pos:
        print("Data mode: ABSOLUTE positions (no error accumulation)")
    else:
        print("Data mode: RELATIVE positions (parent-relative, legacy)")
    train_ds = TreeGeometryDataset(paths["train_dir"], limit=params.get("train_limit"),
                                   absolute_positions=abs_pos)
    val_ds = TreeGeometryDataset(paths["val_dir"], limit=params.get("val_limit"),
                                 absolute_positions=abs_pos)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(params.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(params.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(params.get("batch_size_val", 32)),
        shuffle=False,
        num_workers=int(params.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = FlowMatchingVelocityModel(
        geom_dim=int(model_cfg.get("geom_dim", 39)),
        k_classes=int(model_cfg.get("k_classes", 3)),
        max_depth=int(model_cfg.get("max_depth", 128)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_ff=int(model_cfg.get("d_ff", 1024)),
        max_nodes=int(model_cfg.get("max_nodes", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        tree_attn_hops=int(model_cfg.get("tree_attn_hops", 0)),
        input_clamp_value=model_cfg.get("input_clamp_value", None),
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
        cfg_dropout=float(model_cfg.get("cfg_dropout", 0.0)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M parameters")
    print(f"  d_model={model_cfg.get('d_model', 256)}, "
          f"n_layers={model_cfg.get('n_layers', 8)}, "
          f"n_heads={model_cfg.get('n_heads', 8)}, "
          f"d_ff={model_cfg.get('d_ff', 1024)}")

    # Optimizer + Scheduler
    lr = float(params.get("lr", 2e-4))
    wd = float(params.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(params.get("epochs", 300))
    warmup_steps = int(params.get("warmup_steps", 500))
    total_steps = epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_cfg = {
        "pos_weight": float(loss_cfg_raw.get("pos_weight", 1.0)),
        "cp_weight": float(loss_cfg_raw.get("cp_weight", 1.0)),
        "knots_weight": float(loss_cfg_raw.get("knots_weight", 1.0)),
        "time_sampling": str(loss_cfg_raw.get("time_sampling", "uniform")),
        "self_cond_prob": float(loss_cfg_raw.get("self_cond_prob", 0.0)),
        "knot_mono_weight": float(loss_cfg_raw.get("knot_mono_weight", 0.0)),
        "planarity_weight": float(loss_cfg_raw.get("planarity_weight", 0.0)),
        "radius_weight": float(loss_cfg_raw.get("radius_weight", 0.0)),
        "eccentricity_weight": float(loss_cfg_raw.get("eccentricity_weight", 0.0)),
    }

    # ── Feature normalization (z-score) ──────────────────────────────
    feature_stats_path = paths.get("feature_stats", None)
    feature_mean, feature_std = None, None
    if feature_stats_path and os.path.exists(feature_stats_path):
        feature_mean, feature_std = load_feature_stats(feature_stats_path, device)
        print(f"Feature normalization: loaded from {feature_stats_path}")
        print(f"  mean range: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
        print(f"  std  range: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
    else:
        print("Feature normalization: DISABLED (no feature_stats path in config)")

    # EMA
    ema_decay = float(params.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0.0:
        ema = EMA(model, decay=ema_decay)
        print(f"EMA enabled: decay={ema_decay}")

    # History
    patience = int(params.get("patience", 60))
    best_val = float("inf")
    no_improve = 0
    rollout_interval = int(params.get("rollout_interval", 25))
    rollout_steps = int(params.get("rollout_steps", 50))
    rollout_samples = int(params.get("rollout_samples", 8))

    history_path = output_dir / "history.csv"
    history_fields = [
        "epoch", "lr", "train_total", "train_pos", "train_cp", "train_knots",
        "train_radius", "train_ecc",
        "val_total", "val_pos", "val_cp", "val_knots",
        "best_val", "no_improve", "rollout_mse",
    ]

    # ── Resume from checkpoint if available ────────────────────────────
    start_epoch = 0
    resume_path = output_dir / "best_model.pt"
    if resume_path.exists() and params.get("resume", True):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        # Advance scheduler to the right position
        steps_done = start_epoch * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1} (best_val={best_val:.5f})")
    else:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writeheader()

    print(f"\n{'='*70}")
    print(f"Training Flow Matching  |  epochs={epochs}  lr={lr}  patience={patience}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_cfg,
                                     ema=ema, feature_mean=feature_mean, feature_std=feature_std)
        val_loss = val_one_epoch(model, val_loader, device, loss_cfg,
                                 feature_mean=feature_mean, feature_std=feature_std)
        current_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler (once per epoch)
        for _ in range(len(train_loader)):
            scheduler.step()

        # Rollout evaluation (use EMA model if available)
        rollout_mse = None
        if (epoch + 1) % rollout_interval == 0 or epoch == 0:
            eval_model = ema.shadow if ema is not None else model
            rollout_mse = val_rollout(
                eval_model, val_loader, device,
                n_steps=rollout_steps, n_samples=rollout_samples,
                feature_mean=feature_mean, feature_std=feature_std,
            )

        # Checkpointing
        improved = val_loss["total"] < best_val
        if improved:
            best_val = val_loss["total"]
            no_improve = 0
            save_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": cfg,
                "feature_stats_path": feature_stats_path,
                "absolute_positions": abs_pos,
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, output_dir / "best_model.pt")
        else:
            no_improve += 1

        # Log
        rollout_str = f"  rollout_mse={rollout_mse:.6f}" if rollout_mse is not None else ""
        marker = " *" if improved else ""
        radius_str = f" r={train_loss['radius']:.5f}" if loss_cfg.get('radius_weight', 0) > 0 else ""
        ecc_str = f" e={train_loss['ecc']:.5f}" if loss_cfg.get('eccentricity_weight', 0) > 0 else ""
        print(
            f"Ep {epoch:3d} | "
            f"Train {train_loss['total']:.5f} (p={train_loss['pos']:.5f} c={train_loss['cp']:.5f} k={train_loss['knots']:.5f}{radius_str}{ecc_str}) | "
            f"Val {val_loss['total']:.5f} (p={val_loss['pos']:.5f} c={val_loss['cp']:.5f} k={val_loss['knots']:.5f}) | "
            f"NI={no_improve}/{patience} lr={current_lr:.6f}{rollout_str}{marker}"
        )

        row = {
            "epoch": epoch, "lr": current_lr,
            "train_total": train_loss["total"], "train_pos": train_loss["pos"],
            "train_cp": train_loss["cp"], "train_knots": train_loss["knots"],
            "train_radius": train_loss["radius"], "train_ecc": train_loss["ecc"],
            "val_total": val_loss["total"], "val_pos": val_loss["pos"],
            "val_cp": val_loss["cp"], "val_knots": val_loss["knots"],
            "best_val": best_val, "no_improve": no_improve,
            "rollout_mse": rollout_mse if rollout_mse is not None else "",
        }
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writerow(row)

        # Save periodic checkpoint
        save_every = int(params.get("save_every", 50))
        if (epoch + 1) % save_every == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                output_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience {patience}).")
            break

    # Save final
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch},
        output_dir / "last_model.pt",
    )
    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
