"""
Train the Tree-GNN Flow Matching velocity model for vessel tree geometry.

Same OT-CFM loss as the Transformer variant, but using TreeFlowNet (GNN backbone).

Usage:
    python Stage2_FlowMatching_TreeGNN/train.py \
        --config Stage2_FlowMatching_TreeGNN/configs/treegnn_v1.yaml
"""

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stage2_FlowMatching_TreeGNN_v2.model import TreeFlowNet
from Stage2_FlowMatching_Physio.physio_losses import physiological_loss
from tree_functions import preorder_kcount_parent_indices, parent_relative_positions_to_absolute


# ── EMA ──────────────────────────────────────────────────────────────────────

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
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

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
    """Load [N, 40] .npy files, return geometry [N, 39] + topology."""

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
            arr = parent_relative_positions_to_absolute(
                arr, position_slice=(1, 4), copy=True,
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
    z = torch.randn(B, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(1e-5, 1.0 - 1e-5)


def flow_matching_loss(
    model, batch, device,
    pos_weight=1.0, cp_weight=1.0, knots_weight=1.0,
    time_sampling="uniform",
    time_logit_normal_mean=0.0,
    time_logit_normal_std=1.0,
    self_cond_prob=0.0,
    knot_mono_weight=0.0,
    planarity_weight=0.0,
    radius_weight=0.0,
    # v6 cross-section continuity (ported from Physio v6)
    circularity_weight=0.0,
    normal_tangent_weight=0.0,
    normal_smooth_weight=0.0,
    # Physio constraints (ported from Physio v11; Murray off by default)
    murray_weight=0.0,
    murray_gamma=3.0,
    bifurcation_angle_weight=0.0,
    target_angle_deg=120.0,
    angle_margin_deg=30.0,
    tapering_weight=0.0,
    symmetry_weight=0.0,
    symmetry_target=0.78,
    symmetry_margin=0.22,
    sibling_cosine_weight=0.0,
    sibling_cosine_target=-0.5,
    depth_radius_weight=0.0,
    depth_radius_target_decay=1.05,
    # Focal pos / cp-bif (ported from TreeGNN v7-v9)
    pos_focal_lambda=0.0,
    pos_focal_sigma=1.0,
    cp_bif_focal_lambda=0.0,
    # Sibling separation hinge (ported from TreeGNN v12)
    sibling_separation_weight=0.0,
    sibling_separation_eps=0.5,
    feature_mean=None,
    feature_std=None,
):
    """
    OT-CFM loss: same as Transformer variant.
    x_t = (1-t)*noise + t*x_0,  target = x_0 - noise
    """
    x_0 = batch["geometry"].to(device)
    k = batch["k_counts"].to(device)
    d = batch["depths"].to(device)
    cs = batch["child_slots"].to(device)
    nm = batch["node_mask"].to(device)
    par = batch["parents"].to(device)

    B, N, G = x_0.shape
    mask_f = nm.float().unsqueeze(-1)

    # Z-score normalization
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

    # Depth-warped time: per-node effective time if model uses it
    if hasattr(model, 'depth_warp_alpha') and model.depth_warp_alpha > 0.0:
        t_per_node = model.warp_time(t, d, nm)  # [B, N]
        t_expand = t_per_node.unsqueeze(-1)      # [B, N, 1]
    else:
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

    # Predict velocity
    v_pred = model(k, d, cs, x_t, t, node_mask=nm, parents=par, x_self_cond=x_self_cond)

    # Weighted MSE
    diff2 = (v_pred - target) ** 2

    # Optional focal weighting on the position channel: upweights nodes whose
    # GT rel_pos magnitude is small (continuation splines rel_pos≈0).
    if pos_focal_lambda > 0.0:
        gt_rel_pos_sq = (x_0[..., 0:3] ** 2).sum(-1, keepdim=True)
        focal_w = 1.0 + pos_focal_lambda * torch.exp(
            -gt_rel_pos_sq / (2.0 * (pos_focal_sigma ** 2))
        )
        w_pos = focal_w * mask_f
        pos_loss = (diff2[..., 0:3] * w_pos).sum() / w_pos.sum().clamp(min=1) / 3
    else:
        pos_loss = (diff2[..., 0:3] * mask_f).sum() / mask_f.sum().clamp(min=1) / 3

    if cp_bif_focal_lambda > 0.0:
        is_bif = (k == 2).float().unsqueeze(-1)
        w_cp = (1.0 + cp_bif_focal_lambda * is_bif) * mask_f
        cp_loss = (diff2[..., 3:27] * w_cp).sum() / w_cp.sum().clamp(min=1) / 24
    else:
        cp_loss = (diff2[..., 3:27] * mask_f).sum() / mask_f.sum().clamp(min=1) / 24
    knot_loss = (diff2[..., 27:39] * mask_f).sum() / mask_f.sum().clamp(min=1) / 12

    total = pos_weight * pos_loss + cp_weight * cp_loss + knots_weight * knot_loss

    # Regularization losses
    mono_loss = torch.tensor(0.0, device=device)
    planar_loss = torch.tensor(0.0, device=device)
    radius_loss = torch.tensor(0.0, device=device)

    x_hat_1 = None
    need_x_hat = (knot_mono_weight > 0.0) or (planarity_weight > 0.0) or (radius_weight > 0.0)
    if need_x_hat:
        remaining = (1.0 - t_expand).clamp(min=1e-4)
        x_hat_1 = x_t + remaining * v_pred

    if knot_mono_weight > 0.0:
        knots_hat = x_hat_1[..., 27:39]
        diffs = knots_hat[..., 1:] - knots_hat[..., :-1]
        violations = torch.relu(-diffs)
        mono_loss = (violations * mask_f).sum() / mask_f.sum().clamp(min=1) / 11
        total = total + knot_mono_weight * mono_loss

    if planarity_weight > 0.0:
        cps_hat = x_hat_1[..., 3:27]
        cps_3d = cps_hat.reshape(B, N, 8, 3)
        centroid = cps_3d.mean(dim=2, keepdim=True)
        centered = cps_3d - centroid
        centered_flat = centered.reshape(B * N, 8, 3)
        _, S_vals, _ = torch.linalg.svd(centered_flat, full_matrices=False)
        s3 = S_vals[:, 2].reshape(B, N)
        nm_f = nm.float()
        planar_loss = (s3 * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + planarity_weight * planar_loss

    if radius_weight > 0.0:
        cp_x_hat = x_hat_1[..., 3:11]
        cp_y_hat = x_hat_1[..., 11:19]
        cp_z_hat = x_hat_1[..., 19:27]
        cps_hat_3d = torch.stack([cp_x_hat, cp_y_hat, cp_z_hat], dim=-1)

        cp_x_0 = x_0[..., 3:11]
        cp_y_0 = x_0[..., 11:19]
        cp_z_0 = x_0[..., 19:27]
        cps_0_3d = torch.stack([cp_x_0, cp_y_0, cp_z_0], dim=-1)

        cent_hat = cps_hat_3d.mean(dim=2, keepdim=True)
        cent_0 = cps_0_3d.mean(dim=2, keepdim=True)
        centered_hat = cps_hat_3d - cent_hat
        centered_0 = cps_0_3d - cent_0

        eps_r = 1e-8
        r_hat = (centered_hat.pow(2).sum(dim=-1).mean(dim=-1) + eps_r).sqrt()
        r_0 = (centered_0.pow(2).sum(dim=-1).mean(dim=-1) + eps_r).sqrt()

        log_r_hat = torch.log(r_hat + 1e-6)
        log_r_0 = torch.log(r_0 + 1e-6)
        nm_f = nm.float()
        radius_loss = ((log_r_hat - log_r_0).pow(2) * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + radius_weight * radius_loss

    # ── v6 cross-section continuity losses (ported from Physio train.py) ─────
    circ_loss = torch.tensor(0.0, device=device)
    nt_loss = torch.tensor(0.0, device=device)
    ns_loss = torch.tensor(0.0, device=device)

    need_cps_v6 = (circularity_weight > 0 or normal_tangent_weight > 0
                   or normal_smooth_weight > 0)
    need_normal = (normal_tangent_weight > 0 or normal_smooth_weight > 0)
    nm_f = nm.float()
    nm_f_unsq = nm_f.unsqueeze(-1)
    if need_x_hat is False and need_cps_v6:
        # x_hat_1 not yet computed — produce it now.
        remaining_v6 = (1.0 - t_expand).clamp(min=1e-4)
        x_hat_1 = x_t + remaining_v6 * v_pred

    if need_cps_v6:
        cps_v6 = torch.stack(
            [x_hat_1[..., 3:11], x_hat_1[..., 11:19], x_hat_1[..., 19:27]],
            dim=-1,
        )                                                          # (B,N,8,3)
        cent_v6 = cps_v6.mean(dim=2, keepdim=True)                  # (B,N,1,3)
        diff_v6 = cps_v6 - cent_v6                                  # (B,N,8,3)
        r_v6 = diff_v6.pow(2).sum(dim=-1).clamp(min=1e-12).sqrt()   # (B,N,8)
        r_mean_v6 = r_v6.mean(dim=-1).clamp(min=1e-6)               # (B,N)

    if circularity_weight > 0:
        circ_dev = (r_v6 - r_mean_v6.unsqueeze(-1)).pow(2).mean(dim=-1)
        circ_rel = circ_dev / r_mean_v6.pow(2)
        circ_loss = (circ_rel * nm_f).sum() / nm_f.sum().clamp(min=1)
        total = total + circularity_weight * circ_loss

    if need_normal:
        try:
            _, _, V_v6 = torch.linalg.svd(diff_v6.reshape(B * N, 8, 3),
                                          full_matrices=False)
            normal_v6 = V_v6[:, 2, :].reshape(B, N, 3)
        except Exception:
            normal_v6 = torch.zeros(B, N, 3, device=device)
            normal_v6[..., 2] = 1.0

        cent_pn = cent_v6.squeeze(2)
        par_clamp = par.clamp(min=0)
        has_par = (par >= 0).float() * nm_f
        n_par_idx = par_clamp.unsqueeze(-1).expand(-1, -1, 3)

    if normal_tangent_weight > 0:
        cent_par = cent_pn.gather(1, n_par_idx)
        tang = cent_pn - cent_par
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        align_nt = (normal_v6 * tang).sum(-1).abs()
        nt_pen = (1.0 - align_nt).pow(2)
        nt_loss = (nt_pen * has_par).sum() / has_par.sum().clamp(min=1)
        total = total + normal_tangent_weight * nt_loss

    if normal_smooth_weight > 0:
        n_par_v = normal_v6.gather(1, n_par_idx)
        align_pc = (normal_v6 * n_par_v).sum(-1).abs()
        ns_pen = (1.0 - align_pc).pow(2)
        ns_loss = (ns_pen * has_par).sum() / has_par.sum().clamp(min=1)
        total = total + normal_smooth_weight * ns_loss

    # ── Physio constraints (ported from Physio train.py) ─────────────────────
    physio_losses = {
        "murray": torch.tensor(0.0, device=device),
        "bif_angle": torch.tensor(0.0, device=device),
        "tapering": torch.tensor(0.0, device=device),
        "symmetry": torch.tensor(0.0, device=device),
        "sib_cosine": torch.tensor(0.0, device=device),
        "depth_radius": torch.tensor(0.0, device=device),
        "physio_total": torch.tensor(0.0, device=device),
    }
    any_physio = (murray_weight > 0 or bifurcation_angle_weight > 0
                  or tapering_weight > 0 or symmetry_weight > 0
                  or sibling_cosine_weight > 0 or depth_radius_weight > 0)
    if any_physio:
        if not need_x_hat and not need_cps_v6:
            remaining_p = (1.0 - t_expand).clamp(min=1e-4)
            x_hat_1 = x_t + remaining_p * v_pred
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

    # Sibling-separation hinge loss (ported from TreeGNN v12). Hinge on the
    # L2 distance between sibling rel_pos vectors. Computed on x_hat_1.
    sib_sep_loss = torch.tensor(0.0, device=device)
    if sibling_separation_weight > 0.0:
        if x_hat_1 is None:
            remaining_v_sib = (1.0 - t_expand).clamp(min=1e-4)
            x_hat_1 = x_t + remaining_v_sib * v_pred
        rel_pos_pred = x_hat_1[..., 0:3]
        par_unsq_a = par.unsqueeze(2)
        par_unsq_b = par.unsqueeze(1)
        same_parent = (par_unsq_a == par_unsq_b) & (par_unsq_a >= 0)
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        not_self = ~eye
        nm_a = nm.unsqueeze(2)
        nm_b = nm.unsqueeze(1)
        sib_mask = same_parent & not_self & nm_a & nm_b
        if sib_mask.any():
            diff = rel_pos_pred.unsqueeze(2) - rel_pos_pred.unsqueeze(1)
            sep = diff.pow(2).sum(-1).clamp(min=1e-12).sqrt()
            hinge = (sibling_separation_eps - sep).clamp(min=0).pow(2)
            sib_w = sib_mask.float()
            denom = sib_w.sum().clamp(min=1)
            sib_sep_loss = (hinge * sib_w).sum() / denom
            total = total + sibling_separation_weight * sib_sep_loss

    return {
        "total": total,
        "pos": pos_loss,
        "cp": cp_loss,
        "knots": knot_loss,
        "mono": mono_loss,
        "planar": planar_loss,
        "radius": radius_loss,
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

LOSS_KEYS = [
    "total", "pos", "cp", "knots",
    "mono", "planar", "radius",
    "circ", "nt", "ns",
    "murray", "bif_angle", "tapering", "symmetry", "sib_cosine", "depth_radius",
]


def train_one_epoch(model, loader, optimizer, device, loss_cfg, ema=None,
                    feature_mean=None, feature_std=None):
    model.train()
    accum = {k: 0.0 for k in LOSS_KEYS}
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
    """Validation: keep cross-section continuity terms (circ/nt/ns) in val_total
    but zero out target-based regularizers and physio constraints — same
    convention as Stage2_FlowMatching_Physio.train.val_one_epoch."""
    model.eval()
    accum = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    val_loss_cfg = {k: v for k, v in loss_cfg.items()}
    for drop_key in (
        "self_cond_prob",
        "knot_mono_weight", "planarity_weight", "radius_weight",
        "murray_weight", "bifurcation_angle_weight",
        "tapering_weight", "symmetry_weight",
        "sibling_cosine_weight", "depth_radius_weight",
    ):
        val_loss_cfg[drop_key] = 0.0
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
    """Generate a few trees and compare MSE against GT."""
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

            if feature_mean is not None and feature_std is not None:
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

    output_dir = Path(paths.get("output_dir", "Stage2_FlowMatching_TreeGNN/output/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    abs_pos = bool(cfg.get("data", {}).get("absolute_positions", False))
    if abs_pos:
        print("Data mode: ABSOLUTE positions")
    else:
        print("Data mode: RELATIVE positions (parent-relative)")

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
    model = TreeFlowNet(
        geom_dim=int(model_cfg.get("geom_dim", 39)),
        k_classes=int(model_cfg.get("k_classes", 3)),
        max_depth=int(model_cfg.get("max_depth", 128)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_ff=int(model_cfg.get("d_ff", 1024)),
        max_nodes=int(model_cfg.get("max_nodes", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
        global_attn_every=int(model_cfg.get("global_attn_every", 0)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TreeFlowNet — {n_params / 1e6:.2f}M parameters")
    print(f"  d_model={model_cfg.get('d_model', 256)}, "
          f"n_layers={model_cfg.get('n_layers', 8)}, "
          f"global_attn_every={model_cfg.get('global_attn_every', 0)}")

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

    physio_cfg_raw = cfg.get("physio", {})

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
        # v6 cross-section continuity
        "circularity_weight": float(loss_cfg_raw.get("circularity_weight", 0.0)),
        "normal_tangent_weight": float(loss_cfg_raw.get("normal_tangent_weight", 0.0)),
        "normal_smooth_weight": float(loss_cfg_raw.get("normal_smooth_weight", 0.0)),
        # Physio
        "murray_weight": float(physio_cfg_raw.get("murray_weight", 0.0)),
        "murray_gamma": float(physio_cfg_raw.get("murray_gamma", 3.0)),
        "bifurcation_angle_weight": float(physio_cfg_raw.get("bifurcation_angle_weight", 0.0)),
        "target_angle_deg": float(physio_cfg_raw.get("target_angle_deg", 120.0)),
        "angle_margin_deg": float(physio_cfg_raw.get("angle_margin_deg", 30.0)),
        "tapering_weight": float(physio_cfg_raw.get("tapering_weight", 0.0)),
        "symmetry_weight": float(physio_cfg_raw.get("symmetry_weight", 0.0)),
        "symmetry_target": float(physio_cfg_raw.get("symmetry_target", 0.78)),
        "symmetry_margin": float(physio_cfg_raw.get("symmetry_margin", 0.22)),
        "sibling_cosine_weight": float(physio_cfg_raw.get("sibling_cosine_weight", 0.0)),
        "sibling_cosine_target": float(physio_cfg_raw.get("sibling_cosine_target", -0.5)),
        "depth_radius_weight": float(physio_cfg_raw.get("depth_radius_weight", 0.0)),
        "depth_radius_target_decay": float(physio_cfg_raw.get("depth_radius_target_decay", 1.05)),
        "pos_focal_lambda": float(loss_cfg_raw.get("pos_focal_lambda", 0.0)),
        "pos_focal_sigma": float(loss_cfg_raw.get("pos_focal_sigma", 1.0)),
        "cp_bif_focal_lambda": float(loss_cfg_raw.get("cp_bif_focal_lambda", 0.0)),
        "sibling_separation_weight": float(loss_cfg_raw.get("sibling_separation_weight", 0.0)),
        "sibling_separation_eps": float(loss_cfg_raw.get("sibling_separation_eps", 0.5)),
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
    history_fields = (
        ["epoch", "lr"]
        + [f"train_{k}" for k in LOSS_KEYS]
        + [f"val_{k}" for k in LOSS_KEYS]
        + ["best_val", "no_improve", "rollout_mse"]
    )

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
    print(f"\nCross-section continuity (v6):")
    print(f"  Circularity:   weight={loss_cfg['circularity_weight']}")
    print(f"  Normal-tang.:  weight={loss_cfg['normal_tangent_weight']}")
    print(f"  Normal-smooth: weight={loss_cfg['normal_smooth_weight']}")

    print(f"\n{'='*70}")
    print(f"Training TreeGNN Flow Matching  |  epochs={epochs}  lr={lr}  patience={patience}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_cfg,
                                     ema=ema, feature_mean=feature_mean, feature_std=feature_std)
        val_loss = val_one_epoch(model, val_loader, device, loss_cfg,
                                 feature_mean=feature_mean, feature_std=feature_std)
        current_lr = optimizer.param_groups[0]["lr"]

        for _ in range(len(train_loader)):
            scheduler.step()

        # Rollout evaluation
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
                "model_type": "TreeFlowNet",
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, output_dir / "best_model.pt")
        else:
            no_improve += 1

        marker = " *" if improved else ""
        aux_str = (
            f" r={train_loss['radius']:.4f}"
            f" ci={train_loss['circ']:.4f}"
            f" nt={train_loss['nt']:.4f}"
            f" ns={train_loss['ns']:.4f}"
            f" tp={train_loss['tapering']:.4f}"
            f" sy={train_loss['symmetry']:.4f}"
            f" sc={train_loss['sib_cosine']:.4f}"
            f" dr={train_loss['depth_radius']:.4f}"
        )
        print(
            f"Ep {epoch:3d} | "
            f"Train {train_loss['total']:.5f} "
            f"(p={train_loss['pos']:.4f} c={train_loss['cp']:.4f} k={train_loss['knots']:.4f}{aux_str}) | "
            f"Val {val_loss['total']:.5f} "
            f"(p={val_loss['pos']:.4f} c={val_loss['cp']:.4f} k={val_loss['knots']:.4f} "
            f"ci={val_loss['circ']:.4f} nt={val_loss['nt']:.4f} ns={val_loss['ns']:.4f}) | "
            f"NI={no_improve}/{patience} lr={current_lr:.6f}"
            f"{f'  rollout_mse={rollout_mse:.6f}' if rollout_mse is not None else ''}"
            f"{marker}"
        )

        row = {"epoch": epoch, "lr": current_lr}
        for k_name in LOSS_KEYS:
            row[f"train_{k_name}"] = train_loss[k_name]
            row[f"val_{k_name}"] = val_loss[k_name]
        row["best_val"] = best_val
        row["no_improve"] = no_improve
        row["rollout_mse"] = rollout_mse if rollout_mse is not None else ""
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
