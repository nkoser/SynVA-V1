"""
interpolation.py — Dense centerline & cross-section interpolation.

Given tree segments (sequences of nodes with B-spline cross-section
parameters), this module:

1. Computes arc-length parameterisation along the centerline.
2. Interpolates both centerline positions **and** B-spline coefficients
   (control points + knots) at a user-defined spacing so that the gap
   between consecutive cross-sections is small enough for high-quality
   mesh reconstruction.
3. Evaluates the (interpolated) B-splines to produce 3-D cross-section
   rings at every interpolated location.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator, splev


# ── B-spline evaluation ─────────────────────────────────────────────────


def evaluate_bspline_ring(coeffs: np.ndarray, n_pts: int = 64, n_cp: int = 8) -> np.ndarray | None:
    """Sample *n_pts* 3-D points from a B-spline cross-section.

    Parameters
    ----------
    coeffs : (3*n_cp + n_cp+4,) array
        [cx, cy, cz, knots]  (absolute coordinates).
    n_pts : int
        Number of points to sample on the closed curve.
    n_cp : int
        Number of control points per axis (default 8).

    Returns
    -------
    (n_pts, 3) array or *None* if the spline is degenerate.
    """
    n_knot = n_cp + 4
    expected = 3 * n_cp + n_knot
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape[0] < expected:
        return None

    cx = coeffs[0:n_cp]
    cy = coeffs[n_cp:2*n_cp]
    cz = coeffs[2*n_cp:3*n_cp]
    t = coeffs[3*n_cp:3*n_cp+n_knot]

    if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))
            and np.all(np.isfinite(cz)) and np.all(np.isfinite(t))):
        return None

    # Clamp knots near 1.0 (common fp noise in preprocessed data)
    t = np.where(np.abs(t - 1.0) < 0.01, 1.0, t)
    tck = (t, [cx, cy, cz], 3)

    u = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    try:
        x, y, z = splev(u, tck)
    except Exception:
        return None

    ring = np.column_stack((x, y, z))
    if not np.all(np.isfinite(ring)):
        return None
    return ring


# ── Parallel-transport frame for twist-free ring alignment ───────────────


def _perpendicular_unit(v: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to *v*."""
    v = v / (np.linalg.norm(v) + 1e-12)
    if abs(v[0]) < 0.9:
        perp = np.cross(v, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return perp / (np.linalg.norm(perp) + 1e-12)


def _parallel_transport(n: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Parallel-transport *n* from tangent *t1* to tangent *t2* (Rodrigues)."""
    axis = np.cross(t1, t2)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(t1, t2))
    if sin_a < 1e-10:
        return n.copy() if cos_a > 0 else -n
    axis /= sin_a
    return n * cos_a + np.cross(axis, n) * sin_a + axis * np.dot(axis, n) * (1 - cos_a)


def compute_transport_frames(centers: np.ndarray):
    """Compute parallel-transport (tangent, normal, binormal) at each center.

    Parameters
    ----------
    centers : (M, 3) array — ordered centerline points.

    Returns
    -------
    tangents, normals, binormals — each (M, 3).
    """
    M = len(centers)
    tangents = np.zeros((M, 3))
    normals = np.zeros((M, 3))
    binormals = np.zeros((M, 3))

    # Tangent estimation (central differences, forward/backward at ends)
    for i in range(M):
        if i == 0:
            t = centers[min(1, M - 1)] - centers[0]
        elif i == M - 1:
            t = centers[-1] - centers[max(0, M - 2)]
        else:
            t = centers[i + 1] - centers[i - 1]
        tn = np.linalg.norm(t)
        tangents[i] = t / tn if tn > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Normal / binormal via parallel transport
    normals[0] = _perpendicular_unit(tangents[0])
    binormals[0] = np.cross(tangents[0], normals[0])
    binormals[0] /= np.linalg.norm(binormals[0]) + 1e-12

    for i in range(1, M):
        n = _parallel_transport(normals[i - 1], tangents[i - 1], tangents[i])
        n = n - np.dot(n, tangents[i]) * tangents[i]
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            n = _perpendicular_unit(tangents[i])
        else:
            n /= nn
        normals[i] = n
        b = np.cross(tangents[i], n)
        binormals[i] = b / (np.linalg.norm(b) + 1e-12)

    return tangents, normals, binormals


# ── Dense interpolation along a single segment ──────────────────────────


def _arc_lengths(nodes: np.ndarray) -> np.ndarray:
    """Cumulative arc length along *nodes* (N, 3).  Returns (N,)."""
    dists = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(dists)])


def interpolate_segment(
    nodes: np.ndarray,
    spline_coeffs: np.ndarray,
    target_spacing: float = 0.005,
    n_ring_pts: int = 64,
    min_interp_nodes: int = 4,
    n_cp: int = 8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Densely interpolate centerline + cross-sections along one segment.

    Uses **vertex-based PCHIP** interpolation: evaluate B-spline rings at the
    original tree nodes, align them for proper vertex correspondence, then
    PCHIP-interpolate the 3-D ring vertices directly.  This avoids
    artifacts from coefficient-space interpolation (nonlinear B-spline
    evaluation on interpolated knots/CPs can cause oscillating shapes).

    Parameters
    ----------
    nodes : (N, 3)
        Centerline positions (absolute).
    spline_coeffs : (N, 3*n_cp + n_cp+4)
        B-spline coefficients per node (absolute).
    target_spacing : float
        Desired arc-length between interpolated cross-sections.
    n_ring_pts : int
        Points per cross-section ring.
    min_interp_nodes : int
        Minimum number of output stations (even if segment is short).
    n_cp : int
        Number of control points per axis.

    Returns
    -------
    centers : (M, 3)   — interpolated centerline positions.
    rings   : list of M (n_ring_pts, 3) arrays — cross-section rings.
    """
    N = len(nodes)
    n_knot = n_cp + 4
    n_coeff = 3 * n_cp + n_knot
    if spline_coeffs.shape != (N, n_coeff):
        raise ValueError(f"spline_coeffs shape mismatch: expected ({N}, {n_coeff}), got {spline_coeffs.shape}")

    # ── 1.  Arc-length parameterisation ──────────────────────────────
    arc = _arc_lengths(nodes)
    total_len = arc[-1]

    if total_len < 1e-8 or N < 2:
        centers = nodes.copy()
        rings = []
        for i in range(N):
            ring = evaluate_bspline_ring(spline_coeffs[i], n_ring_pts, n_cp=n_cp)
            if ring is None:
                ring = _fallback_ring(nodes[i], 0.01, n_ring_pts)
            rings.append(ring)
        return centers, rings

    # Remove duplicate arc-length positions (nodes at exact same spot)
    mask = np.concatenate([[True], np.diff(arc) > 1e-10])
    if mask.sum() < 2:
        centers = nodes.copy()
        rings = []
        for i in range(N):
            ring = evaluate_bspline_ring(spline_coeffs[i], n_ring_pts, n_cp=n_cp)
            if ring is None:
                ring = _fallback_ring(nodes[i], 0.01, n_ring_pts)
            rings.append(ring)
        return centers, rings

    arc_u = arc[mask]
    nodes_u = nodes[mask]
    coeffs_u = spline_coeffs[mask]
    N_u = len(arc_u)

    # ── 2.  Evaluate rings at ORIGINAL nodes ─────────────────────────
    orig_rings = []
    orig_radii = []
    for i in range(N_u):
        ring = evaluate_bspline_ring(coeffs_u[i], n_ring_pts, n_cp=n_cp)
        if ring is None:
            # Estimate local tangent for a properly oriented fallback ring
            if N_u >= 2:
                if i == 0:
                    tang = nodes_u[1] - nodes_u[0]
                elif i == N_u - 1:
                    tang = nodes_u[-1] - nodes_u[-2]
                else:
                    tang = nodes_u[i + 1] - nodes_u[i - 1]
            else:
                tang = None
            ring = _fallback_ring(nodes_u[i], 0.01, n_ring_pts, tangent=tang)
        orig_rings.append(ring)
        orig_radii.append(np.max(np.linalg.norm(ring - nodes_u[i], axis=1)))

    orig_radii_arr = np.array(orig_radii) if orig_radii else np.array([0.01])

    # Adaptive per-node clamping using a rolling median.  The window
    # scales with segment length so that short segments use a narrow
    # window (preserving aneurysm bulges whose neighbours raise the
    # local median), while long segments use a wider window that
    # dilutes clusters of fitting-artefact rings.
    _roll_hw = max(10, N_u // 6)
    for i in range(N_u):
        lo = max(0, i - _roll_hw)
        hi = min(N_u, i + _roll_hw + 1)
        local_median = float(np.median(orig_radii_arr[lo:hi]))
        local_allowed = max(local_median * 2.5, 0.03)
        offsets = orig_rings[i] - nodes_u[i]
        dists = np.linalg.norm(offsets, axis=1)
        if dists.max() > local_allowed:
            scale = local_allowed / (dists.max() + 1e-12)
            orig_rings[i] = nodes_u[i] + offsets * scale

    # ── 3.  Align original rings (vertex correspondence) ─────────────
    aligned_orig = align_rings_along_segment(nodes_u, orig_rings)

    # ── 4.  Determine interpolation stations ─────────────────────────
    n_interp = max(int(total_len / target_spacing) + 1, min_interp_nodes, N_u)
    if n_interp > 4000:
        import warnings
        warnings.warn(
            f"Segment requires {n_interp} stations (length={total_len:.3f}, "
            f"spacing={target_spacing}), capping at 4000.",
            stacklevel=2,
        )
        n_interp = 4000
    arc_new = np.linspace(0.0, total_len, n_interp)

    # ── 5.  PCHIP interpolate centerline ─────────────────────────────
    if N_u >= 2:
        centers = PchipInterpolator(arc_u, nodes_u)(arc_new)
    else:
        centers = np.tile(nodes_u[0], (n_interp, 1))

    # ── 6.  PCHIP interpolate ring VERTICES directly ─────────────────
    #   Center rings around their node positions first (separates
    #   centerline motion from shape evolution).
    rings_stack = np.array(aligned_orig)               # (N_u, K, 3)
    centered_rings = rings_stack - nodes_u[:, None, :]  # (N_u, K, 3)

    K = rings_stack.shape[1]
    if N_u >= 2:
        flat = centered_rings.reshape(N_u, -1)             # (N_u, K*3)
        interp_flat = PchipInterpolator(arc_u, flat)(arc_new)  # (M, K*3)
        interp_centered = interp_flat.reshape(n_interp, K, 3)
    else:
        interp_centered = np.tile(centered_rings[0], (n_interp, 1, 1))

    rings_interp = interp_centered + centers[:, None, :]   # (M, K, 3)

    # ── 7.  Final radius clamping (rolling median, matches step 2) ───
    node_allowed = np.array([
        max(float(np.median(orig_radii_arr[max(0, j - _roll_hw):min(N_u, j + _roll_hw + 1)])) * 2.5, 0.03)
        for j in range(N_u)
    ])
    if N_u >= 2:
        allowed_interp = np.interp(arc_new, arc_u, node_allowed)
    else:
        allowed_interp = np.full(n_interp, 0.03)

    rings = []
    for i in range(n_interp):
        ring = rings_interp[i]
        offsets = ring - centers[i]
        dists = np.linalg.norm(offsets, axis=1)
        if dists.max() > allowed_interp[i]:
            scale = allowed_interp[i] / (dists.max() + 1e-12)
            ring = centers[i] + offsets * scale
        rings.append(ring)

    return centers, rings


def _fallback_ring(center: np.ndarray, radius: float, n_pts: int,
                   tangent: np.ndarray | None = None) -> np.ndarray:
    """Create a simple circular ring as fallback.

    If *tangent* is given the ring lies in the plane perpendicular to it;
    otherwise an arbitrary orientation (XY plane) is used.
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_pts)])
    if tangent is not None:
        t = tangent / (np.linalg.norm(tangent) + 1e-12)
        # Build an orthonormal frame
        if abs(t[0]) < 0.9:
            perp = np.cross(t, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(t, np.array([0.0, 1.0, 0.0]))
        n = perp / (np.linalg.norm(perp) + 1e-12)
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-12
        # Map xy circle into n-b plane
        circle = circle[:, 0:1] * n + circle[:, 1:2] * b
    ring = center + radius * circle
    return ring


# ── Align rings to reduce twist ─────────────────────────────────────────


def align_rings_along_segment(
    centers: np.ndarray,
    rings: list[np.ndarray],
) -> list[np.ndarray]:
    """Optimally roll each ring to minimize twist relative to its predecessor.

    Also projects each ring's center so it is consistent with the
    interpolated centerline (prevents drift from spline evaluation noise).

    Parameters
    ----------
    centers : (M, 3)
    rings : list of M (K, 3) arrays

    Returns
    -------
    List of aligned (K, 3) ring arrays.
    """
    if len(rings) == 0:
        return rings

    n_pts = rings[0].shape[0]

    # ── Compute parallel-transport frames ────────────────────────────
    tangents, normals, binormals = compute_transport_frames(centers)

    aligned = []
    for i, ring in enumerate(rings):
        rc = ring.mean(axis=0)
        local = ring - rc  # shape (K, 3)

        # Project local offsets into the normal-binormal plane
        x = local @ normals[i]
        y = local @ binormals[i]

        # Sort by angle to canonicalise winding
        angles = np.arctan2(y, x)
        order = np.argsort(angles)
        ring_sorted = ring[order]

        # Re-center on the interpolated centerline point
        ring_sorted = ring_sorted - ring_sorted.mean(axis=0) + centers[i]
        aligned.append(ring_sorted)

    # ── Minimise twist by cyclically shifting each ring ──────────────
    for i in range(1, len(aligned)):
        best_shift = 0
        best_cost = float("inf")
        for s in range(n_pts):
            rolled = np.roll(aligned[i], s, axis=0)
            cost = np.sum(np.linalg.norm(rolled - aligned[i - 1], axis=1))
            if cost < best_cost:
                best_cost = cost
                best_shift = s
        if best_shift != 0:
            aligned[i] = np.roll(aligned[i], best_shift, axis=0)

    return aligned


# ── Process a full tree (all segments) ───────────────────────────────────


def interpolate_tree_segments(
    segments: list[np.ndarray],
    target_spacing: float = 0.005,
    n_ring_pts: int = 64,
    n_cp: int = 8,
) -> list[dict]:
    """Interpolate all segments of a vessel tree.

    Parameters
    ----------
    segments : list of (N_i, C) arrays
        Each segment from ``get_segments(tree, k)``.  Columns 0-2 = xyz,
        remaining columns = spline coefficients (3*n_cp + n_cp+4).
    target_spacing : float
        Desired distance between consecutive cross-sections.
    n_ring_pts : int
        Points per ring.
    n_cp : int
        Number of B-spline control points per axis.

    Returns
    -------
    List of dicts, each with keys ``"centers"`` (M, 3) and
    ``"rings"`` (list of M (n_ring_pts, 3) arrays).
    """
    n_knot = n_cp + 4
    n_coeff = 3 * n_cp + n_knot  # 36 for n_cp=8, 68 for n_cp=16
    col_end = 3 + n_coeff
    results = []
    for seg in segments:
        if seg.shape[0] < 1:
            continue
        nodes = seg[:, :3].astype(np.float64)
        spline_coeffs = seg[:, 3:col_end].astype(np.float64)
        if spline_coeffs.shape[1] < n_coeff:
            import warnings
            warnings.warn(
                f"Segment has {spline_coeffs.shape[1]} coefficients, expected {n_coeff}. "
                f"Zero-padding applied — check upstream data format.",
                stacklevel=2,
            )
            pad = np.zeros((spline_coeffs.shape[0], n_coeff - spline_coeffs.shape[1]))
            spline_coeffs = np.hstack([spline_coeffs, pad])

        centers, rings = interpolate_segment(
            nodes, spline_coeffs,
            target_spacing=target_spacing,
            n_ring_pts=n_ring_pts,
            n_cp=n_cp,
        )

        results.append({"centers": centers, "rings": rings})
    return results
