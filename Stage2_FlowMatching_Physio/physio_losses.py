"""
Physiological constraint losses for vessel tree generation.

These losses encode known biophysical properties of vascular networks and are
computed on the estimated clean data x̂_1 = x_t + (1-t)*v during training.
They serve as soft regularizers that bias the Flow Matching model towards
generating physiologically plausible vessel trees.

Feature layout (z-score normalized, 39 features):
    [0:3]   = relative position (dx, dy, dz)
    [3:11]  = cp_x (8 control points, x-coord)
    [11:19] = cp_y (8 control points, y-coord)
    [19:27] = cp_z (8 control points, z-coord)
    [27:39] = knots (12 knot values)

References:
    Murray, C.D. (1926). "The Physiological Principle of Minimum Work
    Applied to the Angle of Branching of Arteries."
    
    Zamir, M. (1999). "On fractal properties of arterial trees."
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_node_radii(x, eps=1e-8):
    """
    Compute RMS cross-section radius for each node from 8 control points.

    Args:
        x: [B, N, 39] features (z-score normalized or raw)
    Returns:
        radii: [B, N] RMS radius per node
    """
    cp_x = x[..., 3:11]    # [B, N, 8]
    cp_y = x[..., 11:19]
    cp_z = x[..., 19:27]
    cps = torch.stack([cp_x, cp_y, cp_z], dim=-1)  # [B, N, 8, 3]

    centroid = cps.mean(dim=-2, keepdim=True)  # [B, N, 1, 3]
    centered = cps - centroid                   # [B, N, 8, 3]

    # RMS radius = sqrt(mean(||cp_i - centroid||^2))
    r2 = centered.pow(2).sum(dim=-1).mean(dim=-1)  # [B, N]
    return (r2 + eps).sqrt()


def get_children_mask(parents, k_counts, node_mask):
    """
    For each bifurcation node (k_count == 2), find its two children.

    Args:
        parents:   [B, N]  parent indices (-1 for root)
        k_counts:  [B, N]  number of children (0, 1, 2)
        node_mask: [B, N]  bool

    Returns:
        bif_mask:  [B, N]  bool — which nodes are bifurcations
        child1:    [B, N]  long — index of first child (0 if not bifurcation)
        child2:    [B, N]  long — index of second child (0 if not bifurcation)
    """
    B, N = parents.shape
    device = parents.device

    bif_mask = (k_counts == 2) & node_mask  # [B, N]

    # For each parent, collect children
    child1 = torch.zeros(B, N, dtype=torch.long, device=device)
    child2 = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        child_count = {}
        for i in range(N):
            if not node_mask[b, i]:
                continue
            p = parents[b, i].item()
            if p < 0:
                continue
            if p not in child_count:
                child_count[p] = []
            child_count[p].append(i)
        for p, children in child_count.items():
            if len(children) >= 2 and bif_mask[b, p]:
                child1[b, p] = children[0]
                child2[b, p] = children[1]

    return bif_mask, child1, child2


# ──────────────────────────────────────────────────────────────────────────────
# Murray's Law Loss
# ──────────────────────────────────────────────────────────────────────────────

def murrays_law_loss(x_hat, parents, k_counts, node_mask, gamma=3.0):
    """
    Murray's Law: At a bifurcation, the parent vessel radius raised to
    power gamma equals the sum of child radii raised to power gamma.

        r_parent^γ = r_child1^γ + r_child2^γ

    Classical Murray's Law uses γ=3 (minimum energy for laminar flow).
    Some studies suggest γ≈2.55 for intracranial arteries.

    The loss penalizes the relative deviation:
        L = | r_p^γ - (r_c1^γ + r_c2^γ) | / r_p^γ

    Args:
        x_hat:     [B, N, 39]  estimated clean features
        parents:   [B, N]      parent indices
        k_counts:  [B, N]      children count per node
        node_mask: [B, N]      bool
        gamma:     float       Murray's Law exponent (default 3.0)

    Returns:
        loss: scalar, average Murray violation over all bifurcations
    """
    radii = compute_node_radii(x_hat)  # [B, N]
    bif_mask, child1, child2 = get_children_mask(parents, k_counts, node_mask)

    n_bifs = bif_mask.float().sum()
    if n_bifs < 1:
        return torch.tensor(0.0, device=x_hat.device)

    B, N = parents.shape

    # Gather radii for parent, child1, child2
    r_parent = radii  # [B, N] — we only use entries where bif_mask=True
    r_c1 = torch.gather(radii, 1, child1)  # [B, N]
    r_c2 = torch.gather(radii, 1, child2)  # [B, N]

    # Murray's law: r_p^γ = r_c1^γ + r_c2^γ
    eps = 1e-8
    rp_g = r_parent.pow(gamma) + eps
    rc_sum = r_c1.pow(gamma) + r_c2.pow(gamma) + eps

    # Relative deviation (scale-invariant)
    murray_violation = ((rp_g - rc_sum).abs() / rp_g) * bif_mask.float()

    loss = murray_violation.sum() / n_bifs.clamp(min=1)
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Bifurcation Angle Loss
# ──────────────────────────────────────────────────────────────────────────────

def bifurcation_angle_loss(
    x_hat, parents, k_counts, node_mask,
    target_angle_deg=70.0,
    angle_margin_deg=20.0,
):
    """
    Penalize bifurcation angles that deviate too much from physiological range.

    At each bifurcation, the angle between the two child branch directions
    should be within a physiological range. Intracranial arteries typically
    have bifurcation angles of 50°–90° (Zamir 1999).

    Uses a soft hinge: loss = max(0, |θ - target| - margin)²

    The child branch direction is given by the relative position vector
    of the child node (features [0:3]).

    Args:
        x_hat:            [B, N, 39]
        parents:          [B, N]
        k_counts:         [B, N]
        node_mask:        [B, N]
        target_angle_deg: float, target bifurcation angle in degrees
        angle_margin_deg: float, margin before penalty kicks in

    Returns:
        loss: scalar
    """
    bif_mask, child1, child2 = get_children_mask(parents, k_counts, node_mask)

    n_bifs = bif_mask.float().sum()
    if n_bifs < 1:
        return torch.tensor(0.0, device=x_hat.device)

    B, N = parents.shape

    # Child relative positions (branch directions)
    rel_pos = x_hat[..., 0:3]  # [B, N, 3]

    # Gather child directions
    child1_exp = child1.unsqueeze(-1).expand(-1, -1, 3)  # [B, N, 3]
    child2_exp = child2.unsqueeze(-1).expand(-1, -1, 3)

    dir_c1 = torch.gather(rel_pos, 1, child1_exp)  # [B, N, 3]
    dir_c2 = torch.gather(rel_pos, 1, child2_exp)  # [B, N, 3]

    # Compute angle via atan2(|cross|, dot) — numerically stable, no acos
    eps = 1e-8
    cross = torch.cross(dir_c1, dir_c2, dim=-1)     # [B, N, 3]
    cross_norm = cross.norm(dim=-1)                   # [B, N]
    dot = (dir_c1 * dir_c2).sum(dim=-1)               # [B, N]
    angle_rad = torch.atan2(cross_norm + eps, dot)     # [B, N], in [0, π]

    # Soft hinge loss in RADIANS for proper gradient scaling.
    # (Previous version computed in degrees then divided by 180² ≈ 32400,
    #  which made the angle loss effectively invisible at ~0.3% of total.)
    target_rad = torch.tensor(target_angle_deg * 3.14159265 / 180.0, device=x_hat.device)
    margin_rad = torch.tensor(angle_margin_deg * 3.14159265 / 180.0, device=x_hat.device)
    deviation = (angle_rad - target_rad).abs() - margin_rad
    penalty = torch.relu(deviation).pow(2) * bif_mask.float()

    loss = penalty.sum() / n_bifs.clamp(min=1)

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Radius Tapering (Child ≤ Parent) Loss
# ──────────────────────────────────────────────────────────────────────────────

def radius_tapering_loss(x_hat, parents, node_mask):
    """
    In vascular networks, child vessels are typically thinner than or
    equal to their parent. This loss penalizes cases where a child's
    cross-section radius exceeds its parent's radius.

        L = mean( max(0, r_child - r_parent) / r_parent )

    Only applied to nodes with a valid parent (not the root).

    Args:
        x_hat:     [B, N, 39]
        parents:   [B, N]
        node_mask: [B, N]

    Returns:
        loss: scalar
    """
    radii = compute_node_radii(x_hat)  # [B, N]
    B, N = parents.shape
    device = x_hat.device

    # Mask: nodes with valid parents
    has_parent = (parents >= 0) & node_mask  # [B, N]
    n_valid = has_parent.float().sum()
    if n_valid < 1:
        return torch.tensor(0.0, device=device)

    # Gather parent radii
    safe_parents = parents.clamp(min=0)  # avoid negative index
    r_parent = torch.gather(radii, 1, safe_parents)  # [B, N]
    r_child = radii  # [B, N]

    # Penalize r_child > r_parent (relative violation)
    eps = 1e-8
    violation = torch.relu(r_child - r_parent) / (r_parent + eps)
    loss = (violation * has_parent.float()).sum() / n_valid.clamp(min=1)

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Depth–Radius Decay Loss
# ──────────────────────────────────────────────────────────────────────────────

def depth_radius_loss(x_hat, depths, parents, node_mask, target_decay=0.95):
    """
    Enforce that cross-section radius *actively decreases* with tree depth.

    Unlike the basic tapering loss (which only penalizes r_child > r_parent
    with a linear penalty), this loss:
      1. Targets actual decay: r_child should be ≤ target_decay × r_parent
      2. Uses squared penalty (stronger gradient for violators)
      3. Adds a global depth–spread correlation term that penalizes
         the overall trend of radius increasing with depth

    Combined loss:
        L_local  = mean( relu(r_child - target_decay * r_parent)² )
        L_global = relu(corr(depth, radius))²   (penalize positive corr)
        L = L_local + L_global

    Args:
        x_hat:        [B, N, 39]
        depths:       [B, N]  integer depth per node
        parents:      [B, N]  parent indices (-1 for root)
        node_mask:    [B, N]  bool
        target_decay: float   target ratio r_child / r_parent (< 1 = decay)

    Returns:
        loss: scalar
    """
    radii = compute_node_radii(x_hat)  # [B, N]
    B, N = parents.shape
    device = x_hat.device

    # ── Local: penalise r_child > target_decay * r_parent ────────────
    has_parent = (parents >= 0) & node_mask  # [B, N]
    n_valid = has_parent.float().sum()
    if n_valid < 1:
        return torch.tensor(0.0, device=device)

    safe_parents = parents.clamp(min=0)
    r_parent = torch.gather(radii, 1, safe_parents)  # [B, N]
    r_child = radii

    # Squared penalty for exceeding target_decay * r_parent
    target = target_decay * r_parent
    local_viol = torch.relu(r_child - target).pow(2)
    local_loss = (local_viol * has_parent.float()).sum() / n_valid.clamp(min=1)

    # ── Global: penalise positive Pearson correlation(depth, radius) ─
    valid = node_mask.float()  # [B, N]
    n_per_batch = valid.sum(dim=1, keepdim=True).clamp(min=2)  # [B, 1]

    d = depths.float() * valid                          # [B, N]
    r = radii * valid                                    # [B, N]

    d_mean = d.sum(dim=1, keepdim=True) / n_per_batch   # [B, 1]
    r_mean = r.sum(dim=1, keepdim=True) / n_per_batch

    d_c = (d - d_mean) * valid                           # centered
    r_c = (r - r_mean) * valid

    cov_dr = (d_c * r_c).sum(dim=1)                     # [B]
    std_d = (d_c.pow(2).sum(dim=1) + 1e-8).sqrt()
    std_r = (r_c.pow(2).sum(dim=1) + 1e-8).sqrt()

    corr = cov_dr / (std_d * std_r + 1e-8)              # Pearson r, [B]
    # Penalize positive correlation (radius grows with depth)
    global_loss = torch.relu(corr).pow(2).mean()

    return local_loss + global_loss


# ──────────────────────────────────────────────────────────────────────────────
# Murray Symmetry Ratio Loss
# ──────────────────────────────────────────────────────────────────────────────

def symmetry_ratio_loss(x_hat, parents, k_counts, node_mask,
                        target_ratio=0.8, margin=0.2):
    """
    At a bifurcation, the ratio of the smaller to larger child radius
    (symmetry ratio α = r_minor / r_major) often lies in a characteristic
    range. For intracranial arteries, α ≈ 0.6–1.0.

    Penalize when α deviates too much from the target:
        L = max(0, |α - target| - margin)²

    Args:
        x_hat:        [B, N, 39]
        parents:      [B, N]
        k_counts:     [B, N]
        node_mask:    [B, N]
        target_ratio: float, target symmetry ratio
        margin:       float, margin before penalty

    Returns:
        loss: scalar
    """
    radii = compute_node_radii(x_hat)
    bif_mask, child1, child2 = get_children_mask(parents, k_counts, node_mask)

    n_bifs = bif_mask.float().sum()
    if n_bifs < 1:
        return torch.tensor(0.0, device=x_hat.device)

    r_c1 = torch.gather(radii, 1, child1)
    r_c2 = torch.gather(radii, 1, child2)

    eps = 1e-8
    r_max = torch.max(r_c1, r_c2) + eps
    r_min = torch.min(r_c1, r_c2)
    alpha = r_min / r_max  # [B, N]

    deviation = (alpha - target_ratio).abs() - margin
    penalty = torch.relu(deviation).pow(2) * bif_mask.float()

    loss = penalty.sum() / n_bifs.clamp(min=1)
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Sibling Cosine Similarity Loss
# ──────────────────────────────────────────────────────────────────────────────

def sibling_cosine_loss(
    x_hat, parents, k_counts, node_mask,
    target_cosine=-0.5,
):
    """
    Penalize sibling branches that point in similar directions.

    At a bifurcation, the two child branch direction vectors should diverge.
    In ground-truth intracranial vessels, the mean cosine similarity between
    sibling direction vectors is approximately -0.49 (≈120° apart).
    Generated models typically produce +0.65 (≈50° apart — nearly parallel).

    This loss directly penalizes the cosine similarity between sibling
    direction vectors, pulling them towards the physiological target.

    Advantages over angle-based loss:
        - Smooth, well-conditioned gradient everywhere (no atan2 or acos)
        - loss ∈ [0, 4] — naturally balanced, no normalization issues
        - Gradient directly rotates child position vectors apart

    Args:
        x_hat:         [B, N, 39] estimated clean features
        parents:       [B, N]     parent indices
        k_counts:      [B, N]     children count per node
        node_mask:     [B, N]     bool
        target_cosine: float      target cosine similarity
                                  (cos(120°) ≈ -0.5, cos(90°) = 0)

    Returns:
        loss: scalar, mean squared deviation from target cosine
    """
    bif_mask, child1, child2 = get_children_mask(parents, k_counts, node_mask)

    n_bifs = bif_mask.float().sum()
    if n_bifs < 1:
        return torch.tensor(0.0, device=x_hat.device)

    B, N = parents.shape

    # Child relative positions (branch directions)
    rel_pos = x_hat[..., 0:3]  # [B, N, 3]

    child1_exp = child1.unsqueeze(-1).expand(-1, -1, 3)
    child2_exp = child2.unsqueeze(-1).expand(-1, -1, 3)

    dir_c1 = torch.gather(rel_pos, 1, child1_exp)  # [B, N, 3]
    dir_c2 = torch.gather(rel_pos, 1, child2_exp)  # [B, N, 3]

    # Normalize directions
    eps = 1e-8
    dir_c1_n = dir_c1 / (dir_c1.norm(dim=-1, keepdim=True) + eps)
    dir_c2_n = dir_c2 / (dir_c2.norm(dim=-1, keepdim=True) + eps)

    # Cosine similarity between siblings
    cos_sim = (dir_c1_n * dir_c2_n).sum(dim=-1)  # [B, N]

    # Squared deviation from target
    target = torch.tensor(target_cosine, device=x_hat.device)
    penalty = (cos_sim - target).pow(2) * bif_mask.float()

    loss = penalty.sum() / n_bifs.clamp(min=1)
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Combined Physiological Loss
# ──────────────────────────────────────────────────────────────────────────────

def physiological_loss(
    x_hat, parents, k_counts, node_mask,
    depths=None,
    murray_weight=1.0,
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
    Compute combined physiological constraint losses.

    Returns dict with individual and total losses.
    """
    device = x_hat.device

    murray = torch.tensor(0.0, device=device)
    angle = torch.tensor(0.0, device=device)
    taper = torch.tensor(0.0, device=device)
    sym = torch.tensor(0.0, device=device)
    sib_cos = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    if murray_weight > 0:
        murray = murrays_law_loss(x_hat, parents, k_counts, node_mask,
                                  gamma=murray_gamma)
        total = total + murray_weight * murray

    if bifurcation_angle_weight > 0:
        angle = bifurcation_angle_loss(
            x_hat, parents, k_counts, node_mask,
            target_angle_deg=target_angle_deg,
            angle_margin_deg=angle_margin_deg,
        )
        total = total + bifurcation_angle_weight * angle

    if tapering_weight > 0:
        taper = radius_tapering_loss(x_hat, parents, node_mask)
        total = total + tapering_weight * taper

    if symmetry_weight > 0:
        sym = symmetry_ratio_loss(
            x_hat, parents, k_counts, node_mask,
            target_ratio=symmetry_target,
            margin=symmetry_margin,
        )
        total = total + symmetry_weight * sym

    if sibling_cosine_weight > 0:
        sib_cos = sibling_cosine_loss(
            x_hat, parents, k_counts, node_mask,
            target_cosine=sibling_cosine_target,
        )
        total = total + sibling_cosine_weight * sib_cos

    depth_rad = torch.tensor(0.0, device=device)
    if depth_radius_weight > 0 and depths is not None:
        depth_rad = depth_radius_loss(
            x_hat, depths, parents, node_mask,
            target_decay=depth_radius_target_decay,
        )
        total = total + depth_radius_weight * depth_rad

    return {
        "physio_total": total,
        "murray": murray,
        "bif_angle": angle,
        "tapering": taper,
        "symmetry": sym,
        "sib_cosine": sib_cos,
        "depth_radius": depth_rad,
    }
