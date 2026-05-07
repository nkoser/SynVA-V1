import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree

from .d3 import sdf3, sample_spline, calculate_angle, smooth_union


@sdf3
def capped_cone(a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)

    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = np.sum((p - a) * (p - a), axis=1)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(np.maximum(papa - paba * paba * baba, 0))
        cax = np.maximum(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(
            np.minimum(
                cax * cax + cay * cay * baba,
                cbx * cbx + cby * cby * baba,
            )
        )

    return f


@sdf3
def elliptical_tapered_capsule(p0, p1, a0, b0, a1, b1, n_vec, b_vec):
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    n_vec = np.array(n_vec, dtype=np.float64)
    b_vec = np.array(b_vec, dtype=np.float64)
    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        a0 = float(max(a0, 1e-6))
        b0 = float(max(b0, 1e-6))

        def f(p):
            d = p - p0
            x = d @ n_vec
            y = d @ b_vec
            q = (x / a0) ** 2 + (y / b0) ** 2
            return (q - 1.0) * min(a0, b0)

        return f

    def f(p):
        pa = p - p0
        s = np.clip((pa @ v) / vv, 0.0, 1.0)
        c = p0 + s[:, None] * v
        d = p - c
        x = d @ n_vec
        y = d @ b_vec
        a = (1.0 - s) * float(a0) + s * float(a1)
        b = (1.0 - s) * float(b0) + s * float(b1)
        a = np.maximum(a, 1e-6)
        b = np.maximum(b, 1e-6)
        q = (x / a) ** 2 + (y / b) ** 2
        return (q - 1.0) * np.minimum(a, b)

    return f


def _compute_rmf_frames(tangents: np.ndarray):
    """Compute Rotation-Minimizing Frames via parallel transport.

    Returns u_pts (N, 3) and v_pts (N, 3) perpendicular to each tangent.
    Adjacent frames are as close as possible (no artificial twist).
    """
    N = len(tangents)
    u_pts = np.empty((N, 3), dtype=np.float64)
    v_pts = np.empty((N, 3), dtype=np.float64)
    t0 = tangents[0]
    arb = np.array([0., 0., 1.]) if abs(t0[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(t0, arb)
    nm = np.linalg.norm(u)
    u = u / nm if nm > 1e-12 else arb
    for i in range(N):
        t = tangents[i]
        u = u - np.dot(u, t) * t
        nm = np.linalg.norm(u)
        if nm < 1e-12:
            arb = np.array([0., 0., 1.]) if abs(t[2]) < 0.9 else np.array([1., 0., 0.])
            u = np.cross(t, arb)
            nm = np.linalg.norm(u)
            u = u / nm if nm > 1e-12 else arb
        else:
            u /= nm
        v = np.cross(t, u)
        u_pts[i] = u
        v_pts[i] = v
    return u_pts.astype(np.float32), v_pts.astype(np.float32)


def _angle_batch(a, b, p):
    # Vectorized version of calculate_angle for arrays (N,3)
    ba = b - a
    ba_len = np.linalg.norm(ba, axis=1)
    ba_norm = np.zeros_like(ba)
    mask = ba_len > 1e-12
    ba_norm[mask] = ba[mask] / ba_len[mask][:, None]

    pa = p - a
    dot = np.einsum("ij,ij->i", pa, ba_norm)
    pa_proj = pa - dot[:, None] * ba_norm

    arbitrary = np.zeros_like(ba_norm)
    mask_z = np.abs(ba_norm[:, 2]) < 0.9
    arbitrary[mask_z] = np.array([0.0, 0.0, 1.0])
    arbitrary[~mask_z] = np.array([1.0, 0.0, 0.0])

    u = np.cross(ba_norm, arbitrary)
    u_len = np.linalg.norm(u, axis=1)
    mask_u = u_len > 1e-12
    u[mask_u] = u[mask_u] / u_len[mask_u][:, None]

    v = np.cross(ba_norm, u)
    x_proj = np.einsum("ij,ij->i", pa_proj, u)
    y_proj = np.einsum("ij,ij->i", pa_proj, v)

    angle = np.arctan2(y_proj, x_proj)
    angle = (angle + np.pi) / (2 * np.pi)
    angle[~mask] = 0.0
    angle[~np.isfinite(angle)] = 0.0
    return angle


def _polyval_batch(coeffs, x):
    # coeffs: (N, deg+1), x: (N,)
    y = coeffs[:, 0]
    for c in coeffs[:, 1:].T:
        y = y * x + c
    return y


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
    # Original legacy algorithm, but faster t lookup + vectorized SDF
    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    if sampled_spline is None:
        n_samples = 100
        t_values = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        n_samples = sampled_spline.shape[0]
        if t_values is None:
            t_values = np.linspace(0, 1, n_samples)

    kdtree = KDTree(sampled_spline)

    coeffs = []
    for i in range(len(points)):
        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=50)
        if spline_points is None:
            if len(coeffs) > 0:
                spline_points = sample_spline(splines[i - 1], n_samples=50)
                center = points[i - 1]
            else:
                spline_points = np.repeat(center[np.newaxis, :], 50, axis=0)

        distances = np.linalg.norm(spline_points - center, axis=1)
        xs = np.linspace(0, 1, 50)
        coeff = np.polyfit(xs, distances, 5)

        if centerline_t_mode == "kdtree":
            _, idx = kdtree.query(center)
            t = float(t_values[idx])
        else:
            t = minimize_scalar(lambda tt: np.linalg.norm(np.array(splev(tt, tck)) - center),
                                bounds=(0, 1), method="bounded").x
        if i == len(points) - 1:
            t = 1.0
        coeffs.append((t, coeff))

    ts = np.array([c[0] for c in coeffs], dtype=np.float32)
    coeff_arr = np.array([c[1] for c in coeffs], dtype=np.float32)

    def f(P):
        if P.ndim == 1:
            P = P[np.newaxis, :]

        _, nearest_idx = kdtree.query(P)
        nearest_idx = np.array(nearest_idx, dtype=int)
        c_pts = sampled_spline[nearest_idx]
        t = t_values[nearest_idx]

        t2 = np.where(t + 1.0 / n_samples <= 1.0, t + 1.0 / n_samples, t - 1.0 / n_samples)
        a = np.column_stack(splev(t, tck))
        b = np.column_stack(splev(t2, tck))
        angles = _angle_batch(a, b, P)

        # segment indices
        if len(ts) == 1:
            coeff0 = np.repeat(coeff_arr[[0]], len(P), axis=0)
            radius = _polyval_batch(coeff0, angles)
        else:
            idx = np.searchsorted(ts, t, side="right") - 1
            idx = np.clip(idx, 0, len(ts) - 2)
            idx_next = idx + 1

            t0 = ts[idx]
            t1 = ts[idx_next]
            denom = t1 - t0
            alpha = np.where(denom > 1e-12, (t - t0) / denom, 0.0)
            alpha = np.clip(alpha, 0.0, 1.0)

            coeff0 = coeff_arr[idx]
            coeff1 = coeff_arr[idx_next]
            r0 = _polyval_batch(coeff0, angles)
            r1 = _polyval_batch(coeff1, angles)
            radius = (1.0 - alpha) * r0 + alpha * r1

        min_dist = np.linalg.norm(P - c_pts, axis=1)
        sdf_values = min_dist - radius
        return sdf_values.astype(np.float32)

    return f


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
    # Robust radius tables + vectorized SDF evaluation
    n_profile_samples = 80
    n_angle_bins = 256
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

    tangents = np.array(splev(t_values, tck, der=1)).T
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-12] = 1.0
    tangents = tangents / tangent_norm

    # Rotation-Minimizing Frames (parallel transport) — eliminates per-node
    # angular drift that was causing spiral artefacts on curved branches.
    u_pts_np, v_pts_np = _compute_rmf_frames(tangents)

    kdtree = KDTree(sampled_spline)

    def build_radius_table(center, t_node, spline_points, u_ref, v_ref):
        pts = np.asarray(spline_points, dtype=np.float64)
        pa = pts - np.asarray(center, dtype=np.float64)
        x_proj = pa @ u_ref.astype(np.float64)
        y_proj = pa @ v_ref.astype(np.float64)
        angles = (np.arctan2(y_proj, x_proj) + np.pi) / (2.0 * np.pi)
        angles = angles % 1.0
        # 2D in-plane radial distance: avoids fin artefact from tilted ring planes
        r = np.sqrt(x_proj**2 + y_proj**2)
        mask = np.isfinite(r)
        angles = angles[mask]
        radii = r[mask]
        if radii.size == 0:
            table = np.full(n_angle_bins, float(fallback_radius), dtype=np.float32)
            scalar = float(fallback_radius)
            return table, scalar
        if sanity_threshold is not None:
            try:
                sp = np.percentile(radii, sanity_percentile)
                if np.isfinite(sp) and sp > float(sanity_threshold):
                    table = np.full(n_angle_bins, float(fallback_radius), dtype=np.float32)
                    scalar = float(fallback_radius)
                    return table, scalar
            except Exception:
                pass
        radii = np.array(radii, dtype=np.float32)
        scalar = float(np.percentile(radii, radius_percentile))
        if not np.isfinite(scalar) or scalar < min_radius:
            scalar = float(max(fallback_radius, min_radius))
        if radius_cap is not None and np.isfinite(radius_cap):
            scalar = min(scalar, float(radius_cap))
        table = np.full(n_angle_bins, np.nan, dtype=np.float32)
        bins = (np.array(angles) * n_angle_bins).astype(int) % n_angle_bins
        for bin_idx in range(n_angle_bins):
            mask = bins == bin_idx
            if np.any(mask):
                table[bin_idx] = np.median(radii[mask])
        valid = np.isfinite(table)
        if valid.sum() >= 2:
            x = np.where(valid)[0]
            y = table[valid]
            x_ext = np.concatenate([x, [x[0] + n_angle_bins]])
            y_ext = np.concatenate([y, [y[0]]])
            xi = np.arange(n_angle_bins)
            table = np.interp(xi, x_ext, y_ext).astype(np.float32)
        else:
            table[:] = scalar
        if radius_cap is not None and np.isfinite(radius_cap):
            table = np.minimum(table, float(radius_cap)).astype(np.float32)
        return table, scalar

    profiles = []
    last_profile = None
    for i in range(len(points)):
        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=n_profile_samples)
        if spline_points is None or not np.all(np.isfinite(spline_points)):
            if last_profile is not None:
                profiles.append(last_profile)
                continue
            spline_points = np.repeat(center[np.newaxis, :], n_profile_samples, axis=0)

        # Always compute nearest sample index (needed for RMF frame)
        _, idx = kdtree.query(center)
        if centerline_t_mode == "kdtree":
            t_node = float(t_values[idx])
        else:
            t_node = minimize_scalar(lambda tt: np.linalg.norm(np.array(splev(tt, tck)) - center),
                                     bounds=(0, 1), method="bounded").x
        if i == len(points) - 1:
            t_node = 1.0

        table, scalar = build_radius_table(center, t_node, spline_points,
                                           u_pts_np[idx], v_pts_np[idx])
        if debug and scalar > float(debug_scalar_threshold):
            print(
                "HUGE scalar", scalar,
                "center", center,
                "t", t_node,
                "sp min/max", np.nanmin(spline_points, axis=0), np.nanmax(spline_points, axis=0),
            )
        profile = (float(t_node), table, float(scalar))
        profiles.append(profile)
        last_profile = profile

    if len(profiles) == 0:
        def empty(p):
            return np.ones((p.shape[0],), dtype=np.float32)
        return empty

    ts = np.array([p[0] for p in profiles], dtype=np.float32)
    tables = np.stack([p[1] if p[1] is not None else np.full(n_angle_bins, p[2], dtype=np.float32) for p in profiles])
    scalars = np.array([p[2] for p in profiles], dtype=np.float32)

    def table_lookup_batch(tables_sel, angles):
        pos = angles * n_angle_bins
        i0 = np.floor(pos).astype(int) % n_angle_bins
        i1 = (i0 + 1) % n_angle_bins
        w = pos - np.floor(pos)
        v0 = tables_sel[np.arange(len(angles)), i0]
        v1 = tables_sel[np.arange(len(angles)), i1]
        return (1.0 - w) * v0 + w * v1

    def f(P):
        if P.ndim == 1:
            P = P[np.newaxis, :]

        _, nearest_idx = kdtree.query(P)
        nearest_idx = np.array(nearest_idx, dtype=int)

        c_pts = sampled_spline[nearest_idx]

        t = t_values[nearest_idx]
        # RMF-consistent angle + 2D in-plane radial distance
        u_sel = u_pts_np[nearest_idx]   # (N, 3)
        v_sel = v_pts_np[nearest_idx]   # (N, 3)
        pa_vals = P - c_pts
        x_proj = np.einsum('ij,ij->i', pa_vals, u_sel.astype(np.float64))
        y_proj = np.einsum('ij,ij->i', pa_vals, v_sel.astype(np.float64))
        radial_dist = np.linalg.norm(pa_vals, axis=1)   # 3D: correct endcap distance
        angles = (np.arctan2(y_proj, x_proj) + np.pi) / (2.0 * np.pi) % 1.0

        if len(ts) == 1:
            radius = table_lookup_batch(tables[[0] * len(P)], angles)
        else:
            idx = np.searchsorted(ts, t, side="right") - 1
            idx = np.clip(idx, 0, len(ts) - 2)
            idx_next = idx + 1

            t0 = ts[idx]
            t1 = ts[idx_next]
            denom = t1 - t0
            alpha = np.where(denom > 1e-12, (t - t0) / denom, 0.0)
            alpha = np.clip(alpha, 0.0, 1.0)

            r0 = table_lookup_batch(tables[idx], angles)
            r1 = table_lookup_batch(tables[idx_next], angles)
            radius = (1.0 - alpha) * r0 + alpha * r1

        sdf_values = radial_dist - radius
        return sdf_values.astype(np.float32)

    return f


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
    # Stable = scalar radius per node; vectorized SDF
    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    if sampled_spline is None:
        n_samples = 100
        t_values = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        if t_values is None:
            t_values = np.linspace(0, 1, sampled_spline.shape[0])

    kdtree = KDTree(sampled_spline)

    coeffs = []
    for i in range(len(points)):
        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=50)
        if spline_points is None:
            if len(coeffs) > 0:
                spline_points = sample_spline(splines[i - 1], n_samples=50)
                center = points[i - 1]
            else:
                spline_points = None

        if spline_points is None:
            radius_scalar = float(fallback_radius)
        else:
            center_for_radius = center
            if center_mode == "centroid":
                center_for_radius = np.mean(spline_points, axis=0)
            distances = np.linalg.norm(spline_points - center_for_radius, axis=1)
            if radius_mode == "max":
                radius_scalar = float(np.max(distances))
            elif radius_mode == "mean":
                radius_scalar = float(np.mean(distances))
            elif radius_mode == "percentile":
                radius_scalar = float(np.percentile(distances, radius_percentile))
            else:
                radius_scalar = float(np.median(distances))
            if radius_cap is not None:
                radius_scalar = min(radius_scalar, float(radius_cap))

        if centerline_t_mode == "kdtree":
            _, idx = kdtree.query(center)
            t_node = float(t_values[idx])
        else:
            t_node = minimize_scalar(lambda tt: np.linalg.norm(np.array(splev(tt, tck)) - center),
                                     bounds=(0, 1), method="bounded").x
        if i == len(points) - 1:
            t_node = 1.0
        coeffs.append((t_node, radius_scalar))

    ts = np.array([c[0] for c in coeffs], dtype=np.float32)
    rs = np.array([c[1] for c in coeffs], dtype=np.float32)

    def f(P):
        if P.ndim == 1:
            P = P[np.newaxis, :]

        _, nearest_idx = kdtree.query(P)
        nearest_idx = np.array(nearest_idx, dtype=int)

        c_pts = sampled_spline[nearest_idx]
        min_dist = np.linalg.norm(P - c_pts, axis=1)
        t = t_values[nearest_idx]

        if len(ts) == 1:
            radius = np.full_like(min_dist, rs[0])
        else:
            idx = np.searchsorted(ts, t, side="right") - 1
            idx = np.clip(idx, 0, len(ts) - 2)
            idx_next = idx + 1

            t0 = ts[idx]
            t1 = ts[idx_next]
            denom = t1 - t0
            alpha = np.where(denom > 1e-12, (t - t0) / denom, 0.0)
            alpha = np.clip(alpha, 0.0, 1.0)
            radius = (1.0 - alpha) * rs[idx] + alpha * rs[idx_next]

        sdf_values = min_dist - radius
        return sdf_values.astype(np.float32)

    return f
