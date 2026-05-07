"""
Train Flow Matching + Physiological Constraint Losses.

Extends the v8 Transformer Flow Matching training with soft physiological
regularizers:
    - Murray's Law:       r_p^γ = r_c1^γ + r_c2^γ  at bifurcations
    - Bifurcation angles: penalize non-physiological branching angles
    - Radius tapering:    r_child ≤ r_parent
    - Symmetry ratio:     r_minor / r_major in physiological range

All physio losses are computed on the estimated clean data x̂_1 = x_t+(1-t)*v
(same mechanism as existing radius / planarity / eccentricity losses in v8).

Usage:
    python Stage2_FlowMatching_Physio/train.py \
        --config Stage2_FlowMatching_Physio/configs/physio_v1.yaml
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

# Reuse v8 model (same architecture, just extra loss terms)
from Stage2_FlowMatching.model import FlowMatchingVelocityModel
from Stage2_FlowMatching_Physio.physio_losses import physiological_loss
from tree_functions import preorder_kcount_parent_indices, parent_relative_positions_to_absolute


# ── EMA ──────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(m.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ── Dataset ──────────────────────────────────────────────────────────────────

GEOM_START, GEOM_END = 1, 40


def load_feature_stats(path, device="cpu"):
    data = np.load(path)
    mean = torch.from_numpy(data["mean"]).float().to(device)
    std = torch.from_numpy(data["std"]).float().to(device)
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
    def __init__(self, folder, limit=None, absolute_positions=False):
        self.files = list_npy_files(folder, limit)
        self.absolute_positions = absolute_positions

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        valid = ~(np.all(np.abs(arr[:, 1:]) < 1e-8, axis=1))
        arr = arr[valid]
        if self.absolute_positions:
            arr = parent_relative_positions_to_absolute(arr, position_slice=(1, 4), copy=True)
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
    z = torch.randn(B, device=device) * std + mean
    return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)


def flow_matching_loss_with_physio(
    model, batch, device,
    # Standard v8 loss weights
    pos_weight=1.0, cp_weight=1.0, knots_weight=1.0,
    time_sampling="uniform",
    time_logit_normal_mean=0.0,
    time_logit_normal_std=1.0,
    self_cond_prob=0.0,
    knot_mono_weight=0.0,
    planarity_weight=0.0,
    radius_weight=0.0,
    eccentricity_weight=0.0,
    # v6 continuity losses (cross-section circle / plane orientation)
    circularity_weight=0.0,
    normal_tangent_weight=0.0,
    normal_smooth_weight=0.0,
    feature_mean=None, feature_std=None,
    # Physiological constraint weights
    murray_weight=0.0,
    murray_gamma=3.0,
    bifurcation_angle_weight=0.0,
    target_angle_deg=70.0,
    angle_margin_deg=20.0,
    tapering_weight=0.0,
    symmetry_weight=0.0,
    symmetry_target=0.8,
    symmetry_margin=0.2,
    sibling_cosine_weight=0.0,
    sibling_cosine_target=-0.5,
    depth_radius_weight=0.0,
    depth_radius_target_decay=0.95,
):
    """
    OT-CFM loss + existing v8 regularizers + physiological constraints.
    All regularizers computed on x̂_1 = x_t + (1-t)*v_pred.
    """
    x_0 = batch["geometry"].to(device)
    k = batch["k_counts"].to(device)
    d = batch["depths"].to(device)
    cs = batch["child_slots"].to(device)
    nm = batch["node_mask"].to(device)
    par = batch["parents"].to(device)

    B, N, G = x_0.shape
    mask_f = nm.float().unsqueeze(-1)

    # Z-score normalize
    if feature_mean is not None and feature_std is not None:
        x_0 = (x_0 - feature_mean.view(1, 1, G)) / feature_std.view(1, 1, G)

    # Sample t and noise
    if time_sampling == "logit_normal":
        t = sample_logit_normal(
            B, device,
            mean=time_logit_normal_mean,
            std=time_logit_normal_std,
        )
    else:
        t = torch.rand(B, device=device)
    noise = torch.randn_like(x_0)

    t_expand = t.view(B, 1, 1)
    x_t = (1.0 - t_expand) * noise + t_expand * x_0
    target = x_0 - noise

    # Self-conditioning
    x_self_cond = None
    if model.self_conditioning and self_cond_prob > 0.0 and random.random() < self_cond_prob:
        with torch.no_grad():
            v_sc = model(k, d, cs, x_t, t, node_mask=nm, parents=par, x_self_cond=None)
            remaining = (1.0 - t_expand).clamp(min=1e-4)
            x_self_cond = (x_t + remaining * v_sc).detach() * mask_f

    v_pred = model(k, d, cs, x_t, t, node_mask=nm, parents=par, x_self_cond=x_self_cond)

    # ── Base velocity MSE ────────────────────────────────────────────
    diff2 = (v_pred - target) ** 2
    pos_loss = (diff2[..., 0:3] * mask_f).sum() / mask_f.sum().clamp(min=1) / 3
    cp_loss = (diff2[..., 3:27] * mask_f).sum() / mask_f.sum().clamp(min=1) / 24
    knot_loss = (diff2[..., 27:39] * mask_f).sum() / mask_f.sum().clamp(min=1) / 12

    total = pos_weight * pos_loss + cp_weight * cp_loss + knots_weight * knot_loss

    # ── x̂_1 (estimated clean data) ──────────────────────────────────
    mono_loss = torch.tensor(0.0, device=device)
    planar_loss = torch.tensor(0.0, device=device)
    rad_loss = torch.tensor(0.0, device=device)
    ecc_loss = torch.tensor(0.0, device=device)
    circ_loss = torch.tensor(0.0, device=device)
    nt_loss = torch.tensor(0.0, device=device)
    ns_loss = torch.tensor(0.0, device=device)

    any_reg = (knot_mono_weight > 0 or planarity_weight > 0 or
               radius_weight > 0 or eccentricity_weight > 0 or
               circularity_weight > 0 or normal_tangent_weight > 0 or
               normal_smooth_weight > 0 or
               murray_weight > 0 or bifurcation_angle_weight > 0 or
               tapering_weight > 0 or symmetry_weight > 0 or
               sibling_cosine_weight > 0 or depth_radius_weight > 0)

    if any_reg:
        remaining = (1.0 - t_expand).clamp(min=1e-4)
        x_hat_1 = x_t + remaining * v_pred  # [B, N, 39]

    # ── Knot monotonicity ────────────────────────────────────────────
    if knot_mono_weight > 0:
        knots_hat = x_hat_1[..., 27:39]
        diffs = knots_hat[..., 1:] - knots_hat[..., :-1]
        violations = torch.relu(-diffs)
        mono_loss = (violations * mask_f).sum() / mask_f.sum().clamp(min=1) / 11
        total = total + knot_mono_weight * mono_loss

    # ── Planarity (SVD S3) ───────────────────────────────────────────
    if planarity_weight > 0:
        cps_3d = x_hat_1[..., 3:27].reshape(B, N, 8, 3)
        centered = cps_3d - cps_3d.mean(dim=2, keepdim=True)
        _, S_vals, _ = torch.linalg.svd(centered.reshape(B * N, 8, 3), full_matrices=False)
        s3 = S_vals[:, 2].reshape(B, N)
        nm_f = nm.float()
        planar_loss = (s3 * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + planarity_weight * planar_loss

    # ── Radius (log-space RMS) ───────────────────────────────────────
    if radius_weight > 0:
        cp_x_h = x_hat_1[..., 3:11]
        cp_y_h = x_hat_1[..., 11:19]
        cp_z_h = x_hat_1[..., 19:27]
        cps_h = torch.stack([cp_x_h, cp_y_h, cp_z_h], dim=-1)
        cent_h = cps_h.mean(dim=2, keepdim=True)
        c_h = cps_h - cent_h
        eps_r = 1e-8
        r_hat = (c_h.pow(2).sum(-1).mean(-1) + eps_r).sqrt()

        cp_x_0 = x_0[..., 3:11]
        cp_y_0 = x_0[..., 11:19]
        cp_z_0 = x_0[..., 19:27]
        cps_0 = torch.stack([cp_x_0, cp_y_0, cp_z_0], dim=-1)
        cent_0 = cps_0.mean(dim=2, keepdim=True)
        c_0 = cps_0 - cent_0
        r_0 = (c_0.pow(2).sum(-1).mean(-1) + eps_r).sqrt()

        nm_f = nm.float()
        rad_loss = ((torch.log(r_hat + 1e-6) - torch.log(r_0 + 1e-6)).pow(2) * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + radius_weight * rad_loss

    # ── Eccentricity (Gram-matrix) ───────────────────────────────────
    if eccentricity_weight > 0:
        cp_x_h = x_hat_1[..., 3:11]
        cp_y_h = x_hat_1[..., 11:19]
        cp_z_h = x_hat_1[..., 19:27]
        cps_h = torch.stack([cp_x_h, cp_y_h, cp_z_h], dim=-1)
        cps_h_c = cps_h - cps_h.mean(dim=2, keepdim=True)

        cp_x_g = x_0[..., 3:11]
        cp_y_g = x_0[..., 11:19]
        cp_z_g = x_0[..., 19:27]
        cps_g = torch.stack([cp_x_g, cp_y_g, cp_z_g], dim=-1)
        cps_g_c = cps_g - cps_g.mean(dim=2, keepdim=True)

        G_h = torch.matmul(cps_h_c.transpose(-2, -1), cps_h_c)
        G_g = torch.matmul(cps_g_c.transpose(-2, -1), cps_g_c)
        diff_gram = (G_h - G_g).pow(2).sum(dim=(-2, -1))
        g_norm = G_g.pow(2).sum(dim=(-2, -1)).clamp(min=1e-12)
        gram_rel = diff_gram / g_norm

        nm_f = nm.float()
        ecc_loss = (gram_rel * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + eccentricity_weight * ecc_loss

    # ── v6: Cross-section continuity ─────────────────────────────────
    need_cps_v6 = (circularity_weight > 0 or normal_tangent_weight > 0 or
                   normal_smooth_weight > 0)
    need_normal = (normal_tangent_weight > 0 or normal_smooth_weight > 0)

    if need_cps_v6:
        cps_v6 = x_hat_1[..., 3:27].reshape(B, N, 8, 3)
        cent_v6 = cps_v6.mean(dim=2, keepdim=True)               # (B,N,1,3)
        diff_v6 = cps_v6 - cent_v6                                # (B,N,8,3)
        nm_f = nm.float()

    # Circularity: all 8 CP distances should equal node radius (perfect circle)
    if circularity_weight > 0:
        radii = diff_v6.norm(dim=-1)                              # (B,N,8)
        r_mean = radii.mean(dim=-1, keepdim=True)                 # (B,N,1)
        # normalize by r_mean so weight is scale-free
        circ_rel = ((radii - r_mean) / r_mean.clamp(min=1e-6)).pow(2).mean(-1)  # (B,N)
        circ_loss = (circ_rel * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + circularity_weight * circ_loss

    if need_normal:
        # Plane normal via SVD (smallest right-singular vector)
        try:
            _, _, V_v6 = torch.linalg.svd(diff_v6.reshape(B * N, 8, 3),
                                          full_matrices=False)
            normal_v6 = V_v6[:, 2, :].reshape(B, N, 3)            # (B,N,3)
        except Exception:
            normal_v6 = torch.zeros(B, N, 3, device=device)
            normal_v6[..., 2] = 1.0

        cent_pn = cent_v6.squeeze(2)                              # (B,N,3)
        par_clamp = par.clamp(min=0)                              # (B,N)
        has_par = (par >= 0).float() * nm_f.squeeze(-1)            # (B,N)
        n_par_idx = par_clamp.unsqueeze(-1).expand(-1, -1, 3)

    # Normal ⊥ tangent: cross-section plane should be perpendicular to centerline
    if normal_tangent_weight > 0:
        cent_par = cent_pn.gather(1, n_par_idx)                   # (B,N,3)
        tang = cent_pn - cent_par
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        align_nt = (normal_v6 * tang).sum(-1).abs()               # |n·t|, want 1
        nt_pen = (1.0 - align_nt).pow(2)
        nt_loss = (nt_pen * has_par).sum() / has_par.sum().clamp(min=1)
        total = total + normal_tangent_weight * nt_loss

    # Normal smoothness: parent and child plane normals should align (no flip/twist)
    if normal_smooth_weight > 0:
        n_par_v = normal_v6.gather(1, n_par_idx)
        align_pc = (normal_v6 * n_par_v).sum(-1).abs()
        ns_pen = (1.0 - align_pc).pow(2)
        ns_loss = (ns_pen * has_par).sum() / has_par.sum().clamp(min=1)
        total = total + normal_smooth_weight * ns_loss

    # ── Physiological losses ─────────────────────────────────────────
    physio_losses = {
        "murray": torch.tensor(0.0, device=device),
        "bif_angle": torch.tensor(0.0, device=device),
        "tapering": torch.tensor(0.0, device=device),
        "symmetry": torch.tensor(0.0, device=device),
        "sib_cosine": torch.tensor(0.0, device=device),
        "depth_radius": torch.tensor(0.0, device=device),
        "physio_total": torch.tensor(0.0, device=device),
    }

    any_physio = (murray_weight > 0 or bifurcation_angle_weight > 0 or
                  tapering_weight > 0 or symmetry_weight > 0 or
                  sibling_cosine_weight > 0 or depth_radius_weight > 0)

    if any_physio:
        physio_losses = physiological_loss(
            x_hat_1, par, k, nm,
            depths=d,
            murray_weight=murray_weight,
            murray_gamma=murray_gamma,
            bifurcation_angle_weight=bifurcation_angle_weight,
            target_angle_deg=target_angle_deg,
            angle_margin_deg=angle_margin_deg,
            tapering_weight=tapering_weight,
            symmetry_weight=symmetry_weight,
            symmetry_target=symmetry_target,
            symmetry_margin=symmetry_margin,
            sibling_cosine_weight=sibling_cosine_weight,
            sibling_cosine_target=sibling_cosine_target,
            depth_radius_weight=depth_radius_weight,
            depth_radius_target_decay=depth_radius_target_decay,
        )
        total = total + physio_losses["physio_total"]

    return {
        "total": total,
        "pos": pos_loss,
        "cp": cp_loss,
        "knots": knot_loss,
        "mono": mono_loss,
        "planar": planar_loss,
        "radius": rad_loss,
        "ecc": ecc_loss,
        "circ": circ_loss,
        "nt": nt_loss,
        "ns": ns_loss,
        "murray": physio_losses["murray"],
        "bif_angle": physio_losses["bif_angle"],
        "tapering": physio_losses["tapering"],
        "symmetry": physio_losses["symmetry"],
        "sib_cosine": physio_losses["sib_cosine"],
        "depth_radius": physio_losses["depth_radius"],
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

LOSS_KEYS = ["total", "pos", "cp", "knots", "mono", "planar", "radius", "ecc",
             "circ", "nt", "ns",
             "murray", "bif_angle", "tapering", "symmetry", "sib_cosine", "depth_radius"]


def train_one_epoch(model, loader, optimizer, device, loss_cfg, ema=None,
                    feature_mean=None, feature_std=None):
    model.train()
    accum = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    for batch in loader:
        losses = flow_matching_loss_with_physio(
            model, batch, device,
            feature_mean=feature_mean, feature_std=feature_std,
            **loss_cfg,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
    """Validation: evaluate base loss only (no regularization, no physio)."""
    model.eval()
    accum = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    # Clean validation: disable physio regularization (target-based) but
    # keep the cross-section continuity terms (ci/nt/ns) active so val_total
    # reflects them and early-stopping/patience can trigger on improvements.
    val_cfg = {k: v for k, v in loss_cfg.items()}
    for drop_key in ("self_cond_prob",
                     "knot_mono_weight", "planarity_weight",
                     "radius_weight", "eccentricity_weight",
                     "murray_weight", "bifurcation_angle_weight",
                     "tapering_weight", "symmetry_weight",
                     "sibling_cosine_weight", "depth_radius_weight"):
        val_cfg[drop_key] = 0.0

    for batch in loader:
        losses = flow_matching_loss_with_physio(
            model, batch, device,
            feature_mean=feature_mean, feature_std=feature_std,
            **val_cfg,
        )
        bs = batch["node_mask"].shape[0]
        for k_name in accum:
            accum[k_name] += losses[k_name].item() * bs
        n += bs
    return {k_name: v / max(n, 1) for k_name, v in accum.items()}


@torch.no_grad()
def val_rollout(model, loader, device, n_steps=50, n_samples=4,
                feature_mean=None, feature_std=None):
    """Run full generation on a few val samples and compute MSE."""
    model.eval()
    total_mse, count = 0.0, 0

    for batch in loader:
        B = batch["node_mask"].shape[0]
        for i in range(min(B, n_samples - count)):
            n_nodes = batch["node_mask"][i].sum().item()
            k = batch["k_counts"][i:i+1, :n_nodes].to(device)
            d = batch["depths"][i:i+1, :n_nodes].to(device)
            cs = batch["child_slots"][i:i+1, :n_nodes].to(device)
            nm_i = batch["node_mask"][i:i+1, :n_nodes].to(device)
            par = batch["parents"][i:i+1, :n_nodes].to(device)
            gt = batch["geometry"][i:i+1, :n_nodes].to(device)

            pred = model.sample(k, d, cs, node_mask=nm_i, parents=par, n_steps=n_steps)

            if feature_mean is not None and feature_std is not None:
                _G = pred.shape[-1]
                pred_denorm = pred * feature_std.view(1, 1, _G) + feature_mean.view(1, 1, _G)
                mse = ((pred_denorm - gt) ** 2 * nm_i.float().unsqueeze(-1)).sum() / (nm_i.sum() * 39)
            else:
                mse = ((pred - gt) ** 2 * nm_i.float().unsqueeze(-1)).sum() / (nm_i.sum() * 39)

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
    physio_cfg = cfg.get("physio", {})

    device = resolve_device(params.get("device", 0))
    seed_all(int(params.get("seed", 42)))

    output_dir = Path(paths.get("output_dir", "Stage2_FlowMatching_Physio/output/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    abs_pos = bool(cfg.get("data", {}).get("absolute_positions", False))
    print(f"Data mode: {'ABSOLUTE' if abs_pos else 'RELATIVE'} positions")
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

    # Model (same v8 Transformer)
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
        depth_in_geometry=bool(model_cfg.get("depth_in_geometry", False)),
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

    # Build combined loss config (v8 base + physio)
    loss_cfg = {
        "pos_weight": float(loss_cfg_raw.get("pos_weight", 1.0)),
        "cp_weight": float(loss_cfg_raw.get("cp_weight", 1.0)),
        "knots_weight": float(loss_cfg_raw.get("knots_weight", 1.0)),
        "time_sampling": str(loss_cfg_raw.get("time_sampling", "uniform")),
        "time_logit_normal_mean": float(loss_cfg_raw.get("time_logit_normal_mean", 0.0)),
        "time_logit_normal_std": float(loss_cfg_raw.get("time_logit_normal_std", 1.0)),
        "self_cond_prob": float(loss_cfg_raw.get("self_cond_prob", 0.0)),
        "knot_mono_weight": float(loss_cfg_raw.get("knot_mono_weight", 0.0)),
        "planarity_weight": float(loss_cfg_raw.get("planarity_weight", 0.0)),
        "radius_weight": float(loss_cfg_raw.get("radius_weight", 0.0)),
        "eccentricity_weight": float(loss_cfg_raw.get("eccentricity_weight", 0.0)),
        "circularity_weight": float(loss_cfg_raw.get("circularity_weight", 0.0)),
        "normal_tangent_weight": float(loss_cfg_raw.get("normal_tangent_weight", 0.0)),
        "normal_smooth_weight": float(loss_cfg_raw.get("normal_smooth_weight", 0.0)),
        # Physio
        "murray_weight": float(physio_cfg.get("murray_weight", 0.0)),
        "murray_gamma": float(physio_cfg.get("murray_gamma", 3.0)),
        "bifurcation_angle_weight": float(physio_cfg.get("bifurcation_angle_weight", 0.0)),
        "target_angle_deg": float(physio_cfg.get("target_angle_deg", 70.0)),
        "angle_margin_deg": float(physio_cfg.get("angle_margin_deg", 20.0)),
        "tapering_weight": float(physio_cfg.get("tapering_weight", 0.0)),
        "symmetry_weight": float(physio_cfg.get("symmetry_weight", 0.0)),
        "symmetry_target": float(physio_cfg.get("symmetry_target", 0.8)),
        "symmetry_margin": float(physio_cfg.get("symmetry_margin", 0.2)),
        "sibling_cosine_weight": float(physio_cfg.get("sibling_cosine_weight", 0.0)),
        "sibling_cosine_target": float(physio_cfg.get("sibling_cosine_target", -0.5)),
        "depth_radius_weight": float(physio_cfg.get("depth_radius_weight", 0.0)),
        "depth_radius_target_decay": float(physio_cfg.get("depth_radius_target_decay", 0.95)),
    }

    # Feature normalization
    feature_stats_path = paths.get("feature_stats", None)
    feature_mean, feature_std = None, None
    if feature_stats_path and os.path.exists(feature_stats_path):
        feature_mean, feature_std = load_feature_stats(feature_stats_path, device)
        print(f"Feature normalization: loaded from {feature_stats_path}")
    else:
        print("Feature normalization: DISABLED")

    # EMA
    ema_decay = float(params.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay)
        print(f"EMA enabled: decay={ema_decay}")

    # Training state
    patience = int(params.get("patience", 60))
    best_val = float("inf")
    no_improve = 0
    rollout_interval = int(params.get("rollout_interval", 25))
    rollout_steps = int(params.get("rollout_steps", 50))
    rollout_samples = int(params.get("rollout_samples", 8))

    history_path = output_dir / "history.csv"
    history_fields = [
        "epoch", "lr",
        "train_total", "train_pos", "train_cp", "train_knots",
        "train_radius", "train_ecc",
        "train_circ", "train_nt", "train_ns",
        "train_murray", "train_bif_angle", "train_tapering", "train_symmetry",
        "train_sib_cosine", "train_depth_radius",
        "val_total", "val_pos", "val_cp", "val_knots",
        "best_val", "no_improve", "rollout_mse",
    ]

    # Print time-sampling + physio config
    if loss_cfg["time_sampling"] == "logit_normal":
        print(f"\nTime sampling:  logit_normal "
              f"(mean={loss_cfg['time_logit_normal_mean']}, "
              f"std={loss_cfg['time_logit_normal_std']})")
    else:
        print(f"\nTime sampling:  uniform")

    print(f"\nPhysiological losses:")
    print(f"  Murray's Law:  weight={loss_cfg['murray_weight']}, gamma={loss_cfg['murray_gamma']}")
    print(f"  Bif. angles:   weight={loss_cfg['bifurcation_angle_weight']}, "
          f"target={loss_cfg['target_angle_deg']}°±{loss_cfg['angle_margin_deg']}°")
    print(f"  Tapering:      weight={loss_cfg['tapering_weight']}")
    print(f"  Symmetry:      weight={loss_cfg['symmetry_weight']}, "
          f"target={loss_cfg['symmetry_target']}±{loss_cfg['symmetry_margin']}")
    print(f"  Sib. cosine:   weight={loss_cfg['sibling_cosine_weight']}, "
          f"target={loss_cfg['sibling_cosine_target']}")
    print(f"  Depth-radius:  weight={loss_cfg['depth_radius_weight']}, "
          f"decay={loss_cfg['depth_radius_target_decay']}")

    # Init from another checkpoint (warm-start, no optimizer/epoch resume)
    init_from_path = paths.get("init_from", None)
    init_strict = bool(paths.get("init_strict", True))
    if init_from_path and os.path.exists(init_from_path):
        ckpt = torch.load(init_from_path, map_location=device, weights_only=False)
        if init_strict:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Partial load: skip mismatched layers (e.g. changed geom_proj input dim)
            state = ckpt["model_state_dict"]
            model_state = model.state_dict()
            filtered = {}
            skipped = []
            for k, v in state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            model.load_state_dict(filtered, strict=False)
            if skipped:
                print(f"  Skipped {len(skipped)} mismatched keys: {skipped}")
        if ema is not None and "ema_state_dict" in ckpt:
            if init_strict:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema_sd = ckpt["ema_state_dict"]
                shadow_state = ema.shadow.state_dict()
                filtered_ema = {k: v for k, v in ema_sd.items()
                                if k in shadow_state and v.shape == shadow_state[k].shape}
                ema.shadow.load_state_dict(filtered_ema, strict=False)
        print(f"Initialized weights from {init_from_path} (strict={init_strict})")

    # Resume
    start_epoch = 0
    resume_path = output_dir / "best_model.pt"
    if resume_path.exists() and params.get("resume", True):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        steps_done = start_epoch * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1} (best_val={best_val:.5f})")
    else:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writeheader()

    print(f"\n{'='*80}")
    print(f"Training Flow Matching + Physio  |  epochs={epochs}  lr={lr}  patience={patience}")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_cfg,
            ema=ema, feature_mean=feature_mean, feature_std=feature_std,
        )
        val_loss = val_one_epoch(
            model, val_loader, device, loss_cfg,
            feature_mean=feature_mean, feature_std=feature_std,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        for _ in range(len(train_loader)):
            scheduler.step()

        # Rollout
        rollout_mse = None
        if (epoch + 1) % rollout_interval == 0 or epoch == 0:
            eval_model = ema.shadow if ema is not None else model
            rollout_mse = val_rollout(
                eval_model, val_loader, device,
                n_steps=rollout_steps, n_samples=rollout_samples,
                feature_mean=feature_mean, feature_std=feature_std,
            )

        # Checkpoint
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
        rollout_str = f"  roll={rollout_mse:.6f}" if rollout_mse is not None else ""
        marker = " *" if improved else ""
        # Physio loss strings
        physio_str = ""
        if loss_cfg["murray_weight"] > 0:
            physio_str += f" mu={train_loss['murray']:.5f}"
        if loss_cfg["bifurcation_angle_weight"] > 0:
            physio_str += f" ba={train_loss['bif_angle']:.5f}"
        if loss_cfg["tapering_weight"] > 0:
            physio_str += f" tp={train_loss['tapering']:.5f}"
        if loss_cfg["symmetry_weight"] > 0:
            physio_str += f" sy={train_loss['symmetry']:.5f}"
        if loss_cfg["sibling_cosine_weight"] > 0:
            physio_str += f" sc={train_loss['sib_cosine']:.5f}"
        if loss_cfg["depth_radius_weight"] > 0:
            physio_str += f" dr={train_loss['depth_radius']:.5f}"
        if loss_cfg.get("circularity_weight", 0) > 0:
            physio_str += f" ci={train_loss['circ']:.5f}"
        if loss_cfg.get("normal_tangent_weight", 0) > 0:
            physio_str += f" nt={train_loss['nt']:.5f}"
        if loss_cfg.get("normal_smooth_weight", 0) > 0:
            physio_str += f" ns={train_loss['ns']:.5f}"

        # Val cross-section continuity (mirrors training-time cfg)
        val_cont_str = ""
        if loss_cfg.get("circularity_weight", 0) > 0:
            val_cont_str += f" ci={val_loss['circ']:.5f}"
        if loss_cfg.get("normal_tangent_weight", 0) > 0:
            val_cont_str += f" nt={val_loss['nt']:.5f}"
        if loss_cfg.get("normal_smooth_weight", 0) > 0:
            val_cont_str += f" ns={val_loss['ns']:.5f}"

        print(
            f"Ep {epoch:3d} | "
            f"Train {train_loss['total']:.5f} (p={train_loss['pos']:.5f} c={train_loss['cp']:.5f} k={train_loss['knots']:.5f}"
            f" r={train_loss['radius']:.5f}{physio_str}) | "
            f"Val {val_loss['total']:.5f} (p={val_loss['pos']:.5f} c={val_loss['cp']:.5f} k={val_loss['knots']:.5f}{val_cont_str}) | "
            f"NI={no_improve}/{patience} lr={current_lr:.6f}{rollout_str}{marker}"
        )

        row = {
            "epoch": epoch, "lr": current_lr,
            "train_total": train_loss["total"], "train_pos": train_loss["pos"],
            "train_cp": train_loss["cp"], "train_knots": train_loss["knots"],
            "train_radius": train_loss["radius"], "train_ecc": train_loss["ecc"],
            "train_circ": train_loss["circ"], "train_nt": train_loss["nt"],
            "train_ns": train_loss["ns"],
            "train_murray": train_loss["murray"],
            "train_bif_angle": train_loss["bif_angle"],
            "train_tapering": train_loss["tapering"],
            "train_symmetry": train_loss["symmetry"],
            "train_sib_cosine": train_loss["sib_cosine"],
            "train_depth_radius": train_loss["depth_radius"],
            "val_total": val_loss["total"], "val_pos": val_loss["pos"],
            "val_cp": val_loss["cp"], "val_knots": val_loss["knots"],
            "best_val": best_val, "no_improve": no_improve,
            "rollout_mse": rollout_mse if rollout_mse is not None else "",
        }
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writerow(row)

        save_every = int(params.get("save_every", 50))
        if (epoch + 1) % save_every == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                output_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience {patience}).")
            break

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch},
        output_dir / "last_model.pt",
    )
    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
