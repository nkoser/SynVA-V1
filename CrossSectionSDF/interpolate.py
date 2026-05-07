"""Dense interpolation of centerlines, local frames, and angular radius tables.

Pipeline per segment:
  1. Evaluate B-spline rings at original nodes (3-D world coords).
  2. Align ring vertex ordering (minimize twist via cyclic shift).
  3. PCHIP-interpolate ring offsets in **3-D** from sparse to dense stations.
  4. Compute parallel-transport frames at dense stations.
  5. At each dense station, project the interpolated 3-D ring into the
     station's *own* (N, B) frame and build the angular radius table R(θ).

This avoids frame-alignment artefacts: each R(θ) table is guaranteed to
be consistent with the frame used for SDF query projection.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy.interpolate import PchipInterpolator, splev


N_RING = 64     # points sampled on each B-spline ring
N_ANGULAR = 128 # resolution of the angular radius table
N_CP = 8        # control points per axis in the B-spline
MIN_RADIUS = 0.001  # clamp to avoid zero-radius gaps


@dataclass
class DenseSegment:
    """Densely interpolated vessel segment ready for SDF evaluation."""
    centers: np.ndarray      # (S, 3) station centerline positions
    tangents: np.ndarray     # (S, 3) unit tangent vectors
    normals: np.ndarray      # (S, 3) unit normal vectors
    binormals: np.ndarray    # (S, 3) unit binormal vectors
    radii_table: np.ndarray  # (S, M) angular radius R(θ) at M evenly-spaced angles


# ── B-spline ring evaluation ────────────────────────────────────────────


def evaluate_bspline_ring(coeffs: np.ndarray, n_pts: int = N_RING) -> Optional[np.ndarray]:
    """Sample a closed B-spline cross-section curve.

    Parameters
    ----------
    coeffs : (36,) array
        [cx(8), cy(8), cz(8), knots(12)] — absolute coordinates.

    Returns
    -------
    (n_pts, 3) array of ring positions, or None if degenerate.
    """
    n_knot = N_CP + 4  # 12
    cx = coeffs[0:N_CP]
    cy = coeffs[N_CP:2 * N_CP]
    cz = coeffs[2 * N_CP:3 * N_CP]
    t = coeffs[3 * N_CP:3 * N_CP + n_knot]

    if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))
            and np.all(np.isfinite(cz)) and np.all(np.isfinite(t))):
        return None

    # Clamp near-one knots (common fp noise in preprocessed data)
    t = np.where(np.abs(t - 1.0) < 0.01, 1.0, t)

    if t[-1] - t[0] < 1e-6:
        return None

    try:
        tck = (t, [cx, cy, cz], 3)
        u = np.linspace(0.0, 1.0, n_pts, endpoint=False)
        x, y, z = splev(u, tck)
        ring = np.column_stack((x, y, z))
        if not np.all(np.isfinite(ring)):
            return None
        return ring
    except Exception:
        return None


# ── Parallel-transport frame ────────────────────────────────────────────


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """Arbitrary unit vector perpendicular to v."""
    v = v / (np.linalg.norm(v) + 1e-15)
    if abs(v[0]) < 0.9:
        p = np.cross(v, np.array([1.0, 0.0, 0.0]))
    else:
        p = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return p / (np.linalg.norm(p) + 1e-15)


def _transport(v: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Parallel-transport vector v from tangent t1 to t2 (Rodrigues)."""
    axis = np.cross(t1, t2)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(t1, t2))
    if sin_a < 1e-10:
        return v.copy() if cos_a > 0 else -v
    axis /= sin_a
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


def compute_frames(centers: np.ndarray):
    """Compute rotation-minimizing (T, N, B) frames along a centerline.

    Parameters
    ----------
    centers : (M, 3) ordered centerline points.

    Returns
    -------
    tangents, normals, binormals — each (M, 3), unit vectors.
    """
    M = len(centers)
    T = np.zeros((M, 3))
    N = np.zeros((M, 3))
    B = np.zeros((M, 3))

    # Tangents via central differences
    for i in range(M):
        if i == 0:
            d = centers[min(1, M - 1)] - centers[0]
        elif i == M - 1:
            d = centers[-1] - centers[max(0, M - 2)]
        else:
            d = centers[i + 1] - centers[i - 1]
        n = np.linalg.norm(d)
        T[i] = d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Initial frame
    N[0] = _perpendicular(T[0])
    B[0] = np.cross(T[0], N[0])
    B[0] /= np.linalg.norm(B[0]) + 1e-15

    # Propagate via parallel transport
    for i in range(1, M):
        ni = _transport(N[i - 1], T[i - 1], T[i])
        ni -= np.dot(ni, T[i]) * T[i]  # re-orthogonalise
        nn = np.linalg.norm(ni)
        N[i] = ni / nn if nn > 1e-10 else _perpendicular(T[i])
        bi = np.cross(T[i], N[i])
        B[i] = bi / (np.linalg.norm(bi) + 1e-15)

    return T, N, B


# ── Angular radius table ───────────────────────────────────────────────


def ring_to_radii_table(ring_2d: np.ndarray, n_angular: int = N_ANGULAR) -> np.ndarray:
    """Convert a 2-D ring (centered at origin) to an angular radius table.

    Assumes the cross-section is star-convex from the center.

    Parameters
    ----------
    ring_2d : (K, 2)  — ring vertices in the local (N, B) plane.

    Returns
    -------
    (n_angular,) radii at θ = linspace(0, 2π, n_angular, endpoint=False).
    """
    angles = np.arctan2(ring_2d[:, 1], ring_2d[:, 0]) % (2 * np.pi)
    radii = np.linalg.norm(ring_2d, axis=1)

    order = np.argsort(angles)
    a_sorted = angles[order]
    r_sorted = radii[order]

    # Make periodic by wrapping
    a_ext = np.concatenate([a_sorted - 2 * np.pi, a_sorted, a_sorted + 2 * np.pi])
    r_ext = np.concatenate([r_sorted, r_sorted, r_sorted])

    target = np.linspace(0.0, 2 * np.pi, n_angular, endpoint=False)
    return np.interp(target, a_ext, r_ext)


# ── Ring alignment helpers ──────────────────────────────────────────────


def _make_fallback_ring(centers: np.ndarray, idx: int, K: int,
                        radius: float = 0.001) -> np.ndarray:
    """Small circle perpendicular to estimated tangent at node *idx*."""
    N = len(centers)
    if N >= 2:
        if idx == 0:
            t = centers[min(1, N - 1)] - centers[0]
        elif idx == N - 1:
            t = centers[-1] - centers[max(0, N - 2)]
        else:
            t = centers[idx + 1] - centers[idx - 1]
    else:
        t = np.array([0.0, 0.0, 1.0])
    t = t / (np.linalg.norm(t) + 1e-15)
    n = _perpendicular(t)
    b = np.cross(t, n)
    b /= np.linalg.norm(b) + 1e-15
    theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return centers[idx] + radius * (np.outer(np.cos(theta), n)
                                    + np.outer(np.sin(theta), b))


def _align_rings_sequential(rings: list) -> list:
    """Align ring vertex ordering to minimize twist between neighbours.

    For each consecutive pair of rings, find the cyclic shift of the
    current ring that minimises the sum-of-squared distances to the
    previous ring.  This ensures PCHIP can interpolate corresponding
    vertices coherently.
    """
    if not rings:
        return rings
    aligned = [rings[0].copy()]
    for i in range(1, len(rings)):
        prev = aligned[-1]
        curr = rings[i]
        K = len(curr)
        best_shift, best_cost = 0, np.inf
        for s in range(K):
            shifted = np.roll(curr, s, axis=0)
            cost = float(np.sum((shifted - prev) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_shift = s
        aligned.append(np.roll(curr, best_shift, axis=0))
    return aligned


# ── Segment interpolation ──────────────────────────────────────────────


def interpolate_segment(
    centers_orig: np.ndarray,
    coeffs_orig: np.ndarray,
    target_spacing: float = 0.003,
    n_angular: int = N_ANGULAR,
    global_radius_cap: float = None,
) -> Optional[DenseSegment]:
    """Densely interpolate one segment into stations with angular radius tables.

    Uses **3-D vertex interpolation**: B-spline rings are evaluated at the
    original (sparse) nodes, aligned for vertex correspondence, then
    PCHIP-interpolated in world space.  At each dense station the
    interpolated ring is projected into that station's own (N, B) frame
    to build R(θ).  This avoids frame-alignment artefacts that occur when
    angular tables are interpolated directly.

    Parameters
    ----------
    centers_orig : (N, 3)    — original node positions.
    coeffs_orig  : (N, 36)   — B-spline coefficients per node.
    target_spacing : float    — arc-length between dense stations.
    n_angular : int           — angular resolution for radius table.
    global_radius_cap : float or None
        Hard cap on ring radius (from global tree statistics).

    Returns
    -------
    DenseSegment or None if the segment is degenerate.
    """
    N = len(centers_orig)
    if N < 2:
        return None

    # ── 1.  Arc-length parameterisation ──────────────────────────────
    diffs = np.linalg.norm(np.diff(centers_orig, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total_len = arc[-1]
    if total_len < 1e-8:
        return None

    # Remove duplicate arc positions
    mask = np.concatenate([[True], diffs > 1e-10])
    if mask.sum() < 2:
        return None
    arc_u = arc[mask]
    centers_u = centers_orig[mask]
    coeffs_u = coeffs_orig[mask]
    N_u = len(arc_u)

    # ── 2.  Evaluate B-spline rings at original nodes (3-D) ─────────
    K = N_RING
    orig_rings = []
    for i in range(N_u):
        ring = evaluate_bspline_ring(coeffs_u[i], K)
        if ring is None:
            ring = _make_fallback_ring(centers_u, i, K)
        orig_rings.append(ring)

    # ── 3.  Radius sanity check (two-phase clamp) ─────────────────
    #   Phase A: global cap — prevents bifurcation cross-sections
    #   (which capture both branch openings) from bloating the mesh.
    #   Phase B: rolling-window clamp — catches local outliers.
    radii = np.array([np.max(np.linalg.norm(r - c, axis=1))
                      for r, c in zip(orig_rings, centers_u)])

    if global_radius_cap is not None:
        cap = global_radius_cap
    else:
        cap = max(float(np.median(radii)) * 2.0, 0.02)

    for i in range(N_u):
        offsets = orig_rings[i] - centers_u[i]
        dists = np.linalg.norm(offsets, axis=1)
        if dists.max() > cap:
            scale = cap / (dists.max() + 1e-12)
            orig_rings[i] = centers_u[i] + offsets * scale

    # Recompute radii after global cap
    radii = np.array([np.max(np.linalg.norm(r - c, axis=1))
                      for r, c in zip(orig_rings, centers_u)])
    roll_hw = max(3, N_u // 8)
    for i in range(N_u):
        lo, hi = max(0, i - roll_hw), min(N_u, i + roll_hw + 1)
        allowed = max(float(np.median(radii[lo:hi])) * 1.5, 0.02)
        offsets = orig_rings[i] - centers_u[i]
        dists = np.linalg.norm(offsets, axis=1)
        if dists.max() > allowed:
            scale = allowed / (dists.max() + 1e-12)
            orig_rings[i] = centers_u[i] + offsets * scale

    # ── 4.  Align ring vertex ordering (minimize twist) ─────────────
    aligned = _align_rings_sequential(orig_rings)

    # ── 5.  Center rings (3-D offsets from centerline) ──────────────
    ring_offsets = np.array([r - c for r, c in zip(aligned, centers_u)])  # (N_u, K, 3)

    # ── 6.  Dense stations ──────────────────────────────────────────
    n_dense = max(int(total_len / target_spacing) + 1, N_u, 4)
    n_dense = min(n_dense, 8000)
    arc_dense = np.linspace(0.0, total_len, n_dense)

    # ── 7.  PCHIP interpolate centerline ────────────────────────────
    centers_dense = PchipInterpolator(arc_u, centers_u)(arc_dense)   # (S, 3)

    # ── 8.  PCHIP interpolate ring offsets in 3-D ──────────────────
    off_flat = ring_offsets.reshape(N_u, -1)                         # (N_u, K*3)
    off_dense = PchipInterpolator(arc_u, off_flat)(arc_dense)        # (S, K*3)
    off_dense = off_dense.reshape(n_dense, K, 3)                     # (S, K, 3)

    # ── 9.  Compute parallel-transport frames at dense stations ─────
    tangents, normals, binormals = compute_frames(centers_dense)

    # ── 10. Project interpolated rings → angular radius tables ──────
    #   Each table is computed in that station's own (N, B) frame,
    #   so R(θ) and the SDF query projection are always consistent.
    ring_n = np.einsum('skd,sd->sk', off_dense, normals)            # (S, K)
    ring_b = np.einsum('skd,sd->sk', off_dense, binormals)          # (S, K)

    target_angles = np.linspace(0.0, 2 * np.pi, n_angular, endpoint=False)
    radii_table = np.full((n_dense, n_angular), MIN_RADIUS)

    for i in range(n_dense):
        rn, rb = ring_n[i], ring_b[i]
        angles = np.arctan2(rb, rn) % (2 * np.pi)
        r = np.sqrt(rn * rn + rb * rb)
        order = np.argsort(angles)
        a_s, r_s = angles[order], r[order]
        # Periodic extension for wrapping at 0/2π
        a_ext = np.concatenate([a_s - 2 * np.pi, a_s, a_s + 2 * np.pi])
        r_ext = np.concatenate([r_s, r_s, r_s])
        radii_table[i] = np.maximum(np.interp(target_angles, a_ext, r_ext),
                                    MIN_RADIUS)

    return DenseSegment(
        centers=centers_dense,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        radii_table=radii_table,
    )
