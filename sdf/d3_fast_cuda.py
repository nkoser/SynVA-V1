"""
CUDA-accelerated SDF functions for vessel reconstruction.

Replaces the scipy KDTree + numpy loops in d3_fast.py with PyTorch GPU ops.
The vessel SDF closures f(P) accept numpy arrays and return numpy arrays,
but internally do all heavy computation on GPU (NVIDIA B200 in this setup).

Usage:
    sdf_variant: cuda   # in reconstruct_mesh_config.yaml
"""

import numpy as np
import torch
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree

# Re-export the non-GPU primitives unchanged so get_backend() works
from .d3_fast import (
    capped_cone,
    elliptical_tapered_capsule,
    _angle_batch,
    _polyval_batch,
    _compute_rmf_frames,
)
from .d3 import sdf3, sample_spline, smooth_union

# ─── GPU utility helpers ──────────────────────────────────────────────────────

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _angle_batch_torch(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Vectorized angle computation on GPU.
    All tensors shape (N, 3), returns angle (N,) in [0, 1].
    """
    ba = b - a
    ba_len = ba.norm(dim=1, keepdim=True).clamp(min=1e-12)
    ba_norm = ba / ba_len

    pa = p - a
    dot = (pa * ba_norm).sum(dim=1, keepdim=True)
    pa_proj = pa - dot * ba_norm

    # Build a local perpendicular frame (u, v) per point
    arbitrary = torch.zeros_like(ba_norm)
    mask_z = ba_norm[:, 2].abs() < 0.9
    arbitrary[mask_z]  = torch.tensor([0., 0., 1.], dtype=a.dtype, device=a.device)
    arbitrary[~mask_z] = torch.tensor([1., 0., 0.], dtype=a.dtype, device=a.device)

    u = torch.linalg.cross(ba_norm, arbitrary)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-12)
    v = torch.linalg.cross(ba_norm, u)

    x_proj = (pa_proj * u).sum(dim=1)
    y_proj = (pa_proj * v).sum(dim=1)

    angle = torch.atan2(y_proj, x_proj)
    angle = (angle + torch.pi) / (2.0 * torch.pi)

    # Degenerate directions → angle 0
    mask_bad = ~ba_norm.norm(dim=1).gt(1e-12)
    angle[mask_bad] = 0.0
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)
    return angle


def _polyval_batch_torch(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Horner's method: coeffs (N, deg+1), x (N,) → y (N,)
    """
    y = coeffs[:, 0].clone()
    for i in range(1, coeffs.shape[1]):
        y = y * x + coeffs[:, i]
    return y


def _table_lookup_batch_torch(tables: torch.Tensor, angles: torch.Tensor, n_angle_bins: int) -> torch.Tensor:
    """
    Bilinear lookup into angle-bin tables.
    tables: (N, n_angle_bins) float32
    angles: (N,) float32 in [0, 1)
    returns (N,) float32
    """
    pos = angles * n_angle_bins
    i0 = pos.floor().long() % n_angle_bins
    i1 = (i0 + 1) % n_angle_bins
    w  = pos - pos.floor()

    N = tables.shape[0]
    row = torch.arange(N, device=tables.device)
    v0 = tables[row, i0]
    v1 = tables[row, i1]
    return (1.0 - w) * v0 + w * v1


# ─── GPU_BATCH: process query in GPU-sized chunks ─────────────────────────────

def _gpu_eval_chunks(fn_gpu, P_np: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """
    Call fn_gpu(P_tensor) on chunks of size `chunk` to avoid OOM.
    fn_gpu: callable (N, 3) tensor → (N,) tensor
    Returns numpy float32 array.
    """
    results = []
    for start in range(0, len(P_np), chunk):
        end = min(start + chunk, len(P_np))
        P_chunk = torch.tensor(P_np[start:end], dtype=torch.float32, device=_get_device())
        with torch.no_grad():
            out = fn_gpu(P_chunk)
        results.append(out.cpu().numpy())
    return np.concatenate(results, axis=0).astype(np.float32)


# ─── vessel3 CUDA ─────────────────────────────────────────────────────────────

@sdf3
def vessel3(
    tree_points,
    points,
    splines,
    tck=None,
    sampled_spline=None,
    t_values=None,
    centerline_t_mode="kdtree",
):
    """
    CUDA-accelerated version of vessel3 (original legacy poly-fit radius).
    Builds poly coefficients on CPU exactly as before, but inner f(P) runs on GPU.
    """
    device = _get_device()

    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    if sampled_spline is None:
        n_samples = 200          # denser than CPU version (200 vs 100) for accuracy
        t_values = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        n_samples = sampled_spline.shape[0]
        if t_values is None:
            t_values = np.linspace(0, 1, n_samples)

    # Pre-compute a_pts and b_pts along the centerline (avoid splev inside closure)
    dt = 1.0 / n_samples
    t_next = np.where(t_values + dt <= 1.0, t_values + dt, t_values - dt)
    a_pts_np = sampled_spline                               # (n_samples, 3)
    b_pts_np = np.array(splev(t_next, tck)).T              # (n_samples, 3)

    # Build per-segment polynomial coefficients (on CPU, cheap)
    kdtree = KDTree(sampled_spline)
    coeffs = []
    for i in range(len(points)):
        center = points[i]
        sp = sample_spline(splines[i], n_samples=50)
        if sp is None:
            sp = (sample_spline(splines[i - 1], n_samples=50) if i > 0
                  else np.repeat(center[None], 50, axis=0))
        if sp is None:
            sp = np.repeat(center[None], 50, axis=0)
        distances = np.linalg.norm(sp - center, axis=1)
        xs = np.linspace(0, 1, 50)
        coeff = np.polyfit(xs, distances, 5)
        _, idx = kdtree.query(center)
        t = float(t_values[idx])
        if i == len(points) - 1:
            t = 1.0
        coeffs.append((t, coeff))

    ts_np      = np.array([c[0] for c in coeffs], dtype=np.float32)
    coeff_np   = np.array([c[1] for c in coeffs], dtype=np.float32)

    # Move everything to GPU
    sampled_t  = torch.tensor(sampled_spline, dtype=torch.float32, device=device)
    a_pts_t    = torch.tensor(a_pts_np,       dtype=torch.float32, device=device)
    b_pts_t    = torch.tensor(b_pts_np,       dtype=torch.float32, device=device)
    t_values_t = torch.tensor(t_values,       dtype=torch.float32, device=device)
    ts_t       = torch.tensor(ts_np,          dtype=torch.float32, device=device)
    coeff_t    = torch.tensor(coeff_np,       dtype=torch.float32, device=device)

    single_seg = len(ts_np) == 1

    def _gpu_fn(P: torch.Tensor) -> torch.Tensor:
        # Nearest centerline point via squared L2 distance
        dists2 = torch.cdist(P, sampled_t)      # (N, n_samples)
        nearest_idx = dists2.argmin(dim=1)       # (N,)

        c_pts     = sampled_t[nearest_idx]       # (N, 3)
        min_dist  = (P - c_pts).norm(dim=1)      # (N,)
        t         = t_values_t[nearest_idx]      # (N,)

        a = a_pts_t[nearest_idx]                 # (N, 3)
        b = b_pts_t[nearest_idx]                 # (N, 3)
        angles = _angle_batch_torch(a, b, P)     # (N,) in [0,1]

        if single_seg:
            coeff0 = coeff_t[0:1].expand(len(P), -1)
            radius = _polyval_batch_torch(coeff0, angles)
        else:
            idx      = torch.searchsorted(ts_t.contiguous(), t.contiguous()) - 1
            idx      = idx.clamp(0, len(ts_t) - 2)
            idx_next = idx + 1

            t0    = ts_t[idx]
            t1    = ts_t[idx_next]
            denom = t1 - t0
            alpha = torch.where(denom > 1e-12, (t - t0) / denom,
                                torch.zeros_like(t)).clamp(0.0, 1.0)

            r0 = _polyval_batch_torch(coeff_t[idx],      angles)
            r1 = _polyval_batch_torch(coeff_t[idx_next], angles)
            radius = (1.0 - alpha) * r0 + alpha * r1

        return min_dist - radius

    def f(P_np):
        if P_np.ndim == 1:
            P_np = P_np[None, :]
        return _gpu_eval_chunks(_gpu_fn, P_np)

    return f


# ─── vessel3_robust CUDA ──────────────────────────────────────────────────────

@sdf3
def vessel3_robust(
    tree_points,
    points,
    splines,
    tck=None,
    sampled_spline=None,
    t_values=None,
    centerline_t_mode="kdtree",
    fallback_radius=0.02,
    min_radius=0.005,
    radius_cap=None,
    sanity_percentile=95,
    sanity_threshold=None,
    debug=False,
    debug_scalar_threshold=10.0,
):
    """
    CUDA-accelerated version of vessel3_robust (angle-bin radius tables).
    Angle-bin tables are built identically on CPU; inner f(P) runs on GPU.
    """
    device = _get_device()
    n_profile_samples = 80
    n_angle_bins      = 256
    radius_percentile = 90.0

    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    if sampled_spline is None:
        n_centerline_samples = 200
        t_values = np.linspace(0, 1, n_centerline_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        if t_values is None:
            t_values = np.linspace(0, 1, sampled_spline.shape[0])

    dt     = 1.0 / len(t_values)
    t_next = np.where(t_values + dt <= 1.0, t_values + dt, t_values - dt)
    a_pts_np = sampled_spline
    b_pts_np = np.array(splev(t_next, tck)).T

    # ── Rotation-Minimizing Frames for the centerline ──────────────────────
    # Compute RMF (parallel-transport) reference directions once from the
    # densely-sampled centerline.  Using these instead of recomputing
    # cross(tangent, [0,0,1]) per node eliminates the angular drift that
    # causes spiral artefacts on curved branches.
    _raw_tang = (b_pts_np - a_pts_np).astype(np.float64)
    _tang_norms = np.linalg.norm(_raw_tang, axis=1, keepdims=True).clip(min=1e-12)
    _tangents_norm = _raw_tang / _tang_norms
    u_pts_np, v_pts_np = _compute_rmf_frames(_tangents_norm)

    kdtree = KDTree(sampled_spline)

    # ── Build radius tables using RMF frame ────────────────────────────────
    def build_radius_table(center, t_node, spline_points, u_ref, v_ref):
        pts = np.asarray(spline_points, dtype=np.float64)
        pa = pts - np.asarray(center, dtype=np.float64)
        # Project ring points onto the RMF (u, v) plane
        x_proj = pa @ u_ref.astype(np.float64)
        y_proj = pa @ v_ref.astype(np.float64)
        angles = (np.arctan2(y_proj, x_proj) + np.pi) / (2.0 * np.pi)
        angles = angles % 1.0
        # Use 2D in-plane radial distance rather than full 3D norm.
        # The 3D norm inflates the radius when the ring plane is tilted relative
        # to the centerline tangent (e.g. at curved endpoints), which creates a
        # directional spike in the table and a visible fin artefact.
        r = np.sqrt(x_proj**2 + y_proj**2)
        mask = np.isfinite(r)
        angles = angles[mask]
        radii  = r[mask]

        if radii.size == 0:
            table  = np.full(n_angle_bins, float(fallback_radius), dtype=np.float32)
            scalar = float(fallback_radius)
            return table, scalar

        if sanity_threshold is not None:
            try:
                sp = np.percentile(radii, sanity_percentile)
                if np.isfinite(sp) and sp > float(sanity_threshold):
                    table  = np.full(n_angle_bins, float(fallback_radius), dtype=np.float32)
                    scalar = float(fallback_radius)
                    return table, scalar
            except Exception:
                pass

        radii  = np.array(radii, dtype=np.float32)
        scalar = float(np.percentile(radii, radius_percentile))
        if not np.isfinite(scalar) or scalar < min_radius:
            scalar = float(max(fallback_radius, min_radius))
        if radius_cap is not None and np.isfinite(radius_cap):
            scalar = min(scalar, float(radius_cap))

        table = np.full(n_angle_bins, np.nan, dtype=np.float32)
        bins  = (np.array(angles) * n_angle_bins).astype(int) % n_angle_bins
        for bin_idx in range(n_angle_bins):
            m = bins == bin_idx
            if np.any(m):
                table[bin_idx] = np.median(radii[m])

        valid = np.isfinite(table)
        if valid.sum() >= 2:
            x_v   = np.where(valid)[0]
            y_v   = table[valid]
            x_ext = np.concatenate([x_v, [x_v[0] + n_angle_bins]])
            y_ext = np.concatenate([y_v, [y_v[0]]])
            xi    = np.arange(n_angle_bins)
            table = np.interp(xi, x_ext, y_ext).astype(np.float32)
        else:
            table[:] = scalar

        if radius_cap is not None and np.isfinite(radius_cap):
            table = np.minimum(table, float(radius_cap)).astype(np.float32)
        return table, scalar

    profiles   = []
    last_profile = None
    for i in range(len(points)):
        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=n_profile_samples)
        if spline_points is None or not np.all(np.isfinite(spline_points)):
            if last_profile is not None:
                profiles.append(last_profile)
                continue
            spline_points = np.repeat(center[None, :], n_profile_samples, axis=0)

        _, idx    = kdtree.query(center)
        t_node    = float(t_values[idx])
        if i == len(points) - 1:
            t_node = 1.0

        table, scalar = build_radius_table(center, t_node, spline_points,
                                           u_pts_np[idx], v_pts_np[idx])
        if debug and scalar > float(debug_scalar_threshold):
            print(f"HUGE scalar {scalar} center {center} t {t_node}")
        profile = (float(t_node), table, float(scalar))
        profiles.append(profile)
        last_profile = profile

    if len(profiles) == 0:
        def empty(p):
            return np.ones((p.shape[0],), dtype=np.float32)
        return empty

    ts_np     = np.array([p[0] for p in profiles], dtype=np.float32)
    tables_np = np.stack([
        p[1] if p[1] is not None else np.full(n_angle_bins, p[2], dtype=np.float32)
        for p in profiles
    ])  # (n_nodes, n_angle_bins)
    # scalars not needed at inference (tables cover all)

    # Move to GPU
    sampled_t  = torch.tensor(sampled_spline, dtype=torch.float32, device=device)
    u_pts_t    = torch.tensor(u_pts_np,       dtype=torch.float32, device=device)
    v_pts_t    = torch.tensor(v_pts_np,       dtype=torch.float32, device=device)
    t_values_t = torch.tensor(t_values,       dtype=torch.float32, device=device)
    ts_t       = torch.tensor(ts_np,          dtype=torch.float32, device=device)
    tables_t   = torch.tensor(tables_np,      dtype=torch.float32, device=device)  # (n_nodes, n_bins)

    single_seg  = len(ts_np) == 1

    def _gpu_fn(P: torch.Tensor) -> torch.Tensor:
        dists2      = torch.cdist(P, sampled_t)
        nearest_idx = dists2.argmin(dim=1)

        c_pts    = sampled_t[nearest_idx]
        radial   = (P - c_pts).norm(dim=1)
        t        = t_values_t[nearest_idx]

        # RMF-consistent angle: project onto pre-computed (u, v) frame
        u_sel  = u_pts_t[nearest_idx]          # (N, 3)
        v_sel  = v_pts_t[nearest_idx]          # (N, 3)
        pa     = P - c_pts
        x_proj = (pa * u_sel).sum(dim=1)
        y_proj = (pa * v_sel).sum(dim=1)
        angles = ((torch.atan2(y_proj, x_proj) + torch.pi) / (2.0 * torch.pi)) % 1.0

        if single_seg:
            tables_sel = tables_t[0:1].expand(len(P), -1)
            radius = _table_lookup_batch_torch(tables_sel, angles, n_angle_bins)
        else:
            idx      = torch.searchsorted(ts_t.contiguous(), t.contiguous()) - 1
            idx      = idx.clamp(0, len(ts_t) - 2)
            idx_next = idx + 1

            t0    = ts_t[idx]
            t1    = ts_t[idx_next]
            denom = t1 - t0
            alpha = torch.where(denom > 1e-12, (t - t0) / denom,
                                torch.zeros_like(t)).clamp(0.0, 1.0)

            r0 = _table_lookup_batch_torch(tables_t[idx],      angles, n_angle_bins)
            r1 = _table_lookup_batch_torch(tables_t[idx_next], angles, n_angle_bins)
            radius = (1.0 - alpha) * r0 + alpha * r1

        return radial - radius

    def f(P_np):
        if P_np.ndim == 1:
            P_np = P_np[None, :]
        return _gpu_eval_chunks(_gpu_fn, P_np)

    return f


# ─── vessel3_stable CUDA ──────────────────────────────────────────────────────

@sdf3
def vessel3_stable(
    tree_points,
    points,
    splines,
    radius_mode="median",
    radius_percentile=90,
    radius_cap=None,
    center_mode="node",
    fallback_radius=0.0,
    tck=None,
    sampled_spline=None,
    t_values=None,
    centerline_t_mode="kdtree",
):
    """
    CUDA-accelerated version of vessel3_stable (scalar radius per segment).
    """
    device = _get_device()

    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    if sampled_spline is None:
        n_samples = 200
        t_values  = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        if t_values is None:
            t_values = np.linspace(0, 1, sampled_spline.shape[0])

    kdtree = KDTree(sampled_spline)

    coeffs = []
    for i in range(len(points)):
        center = points[i]
        sp = sample_spline(splines[i], n_samples=50)
        if sp is None:
            sp = (sample_spline(splines[i - 1], n_samples=50)
                  if i > 0 else None)
        if sp is None:
            radius_scalar = float(fallback_radius)
        else:
            c4r = center if center_mode == "node" else np.mean(sp, axis=0)
            dists = np.linalg.norm(sp - c4r, axis=1)
            if radius_mode == "max":
                radius_scalar = float(np.max(dists))
            elif radius_mode == "mean":
                radius_scalar = float(np.mean(dists))
            elif radius_mode == "percentile":
                radius_scalar = float(np.percentile(dists, radius_percentile))
            else:
                radius_scalar = float(np.median(dists))
            if radius_cap is not None:
                radius_scalar = min(radius_scalar, float(radius_cap))

        _, idx = kdtree.query(center)
        t_node = float(t_values[idx])
        if i == len(points) - 1:
            t_node = 1.0
        coeffs.append((t_node, radius_scalar))

    ts_np = np.array([c[0] for c in coeffs], dtype=np.float32)
    rs_np = np.array([c[1] for c in coeffs], dtype=np.float32)

    sampled_t  = torch.tensor(sampled_spline, dtype=torch.float32, device=device)
    t_values_t = torch.tensor(t_values,       dtype=torch.float32, device=device)
    ts_t       = torch.tensor(ts_np,          dtype=torch.float32, device=device)
    rs_t       = torch.tensor(rs_np,          dtype=torch.float32, device=device)

    single_seg = len(ts_np) == 1

    def _gpu_fn(P: torch.Tensor) -> torch.Tensor:
        dists2      = torch.cdist(P, sampled_t)
        nearest_idx = dists2.argmin(dim=1)
        c_pts       = sampled_t[nearest_idx]
        min_dist    = (P - c_pts).norm(dim=1)
        t           = t_values_t[nearest_idx]

        if single_seg:
            radius = rs_t[0].expand(len(P))
        else:
            idx      = torch.searchsorted(ts_t.contiguous(), t.contiguous()) - 1
            idx      = idx.clamp(0, len(ts_t) - 2)
            idx_next = idx + 1

            t0    = ts_t[idx]
            t1    = ts_t[idx_next]
            denom = t1 - t0
            alpha = torch.where(denom > 1e-12, (t - t0) / denom,
                                torch.zeros_like(t)).clamp(0.0, 1.0)
            radius = (1.0 - alpha) * rs_t[idx] + alpha * rs_t[idx_next]

        return min_dist - radius

    def f(P_np):
        if P_np.ndim == 1:
            P_np = P_np[None, :]
        return _gpu_eval_chunks(_gpu_fn, P_np)

    return f
