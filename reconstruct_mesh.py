import argparse
import inspect
import importlib
import os
import sys
from glob import glob

import numpy as np

try:
    import pyvista as pv
except Exception:
    pv = None

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splprep, splev, PchipInterpolator, UnivariateSpline
except Exception as exc:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize
from sdf import union, blend
from sdf import progress as sdf_progress

_BACKEND_CACHE = {}


def get_backend(params):
    recon_mode = (params.get("recon_mode", "legacy") or "legacy").lower()
    sdf_variant = (params.get("sdf_variant", "original") or "original").lower()

    if recon_mode in {"sweep", "sweep_ellipse", "sweep_fast", "sweep_ellipse_fast"}:
        module_name = "sdf.d3_fast"
    elif sdf_variant == "cuda":
        module_name = "sdf.d3_fast_cuda"
    elif sdf_variant == "fast":
        module_name = "sdf.d3_fast"
    else:
        module_name = "sdf.d3"

    cache_key = module_name
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    mod = importlib.import_module(module_name)

    # Pick the marching-cubes core: cuda variant uses core_cuda for full-grid eval
    if sdf_variant == "cuda" and recon_mode not in {"sweep", "sweep_ellipse", "sweep_fast", "sweep_ellipse_fast"}:
        core_mod = importlib.import_module("sdf.core_cuda")
    else:
        from sdf import core as core_mod  # type: ignore

    backend = {
        "mod": mod,
        "core": core_mod,
        "sdf3": getattr(mod, "sdf3", None),
        "smooth_union": getattr(mod, "smooth_union", None),
        "vessel3": getattr(mod, "vessel3", None),
        "vessel3_robust": getattr(mod, "vessel3_robust", None),
        "vessel3_stable": getattr(mod, "vessel3_stable", None),
        "capped_cone": getattr(mod, "capped_cone", None),
        "elliptical_tapered_capsule": getattr(mod, "elliptical_tapered_capsule", None),
    }
    _BACKEND_CACHE[cache_key] = backend
    return backend


def get_sdf_fns(params):
    b = get_backend(params)
    return (b["vessel3"], b["vessel3_robust"], b["vessel3_stable"])


def get_sweep_primitives_and_union(params):
    b = get_backend(params)
    if b["capped_cone"] is None or b["elliptical_tapered_capsule"] is None:
        raise RuntimeError("Sweep requires backend with sweep primitives (expected in sdf.d3_fast).")
    return b["capped_cone"], b["elliptical_tapered_capsule"], b["smooth_union"], b["sdf3"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_inputs(path, pattern):
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, pattern)))
    return [path]


def _coerce_row(row, k):
    row = list(row)
    head = row[:3]
    tail = row[3:]
    needed = k - 3
    if len(tail) < needed:
        tail = tail + [0.0] * (needed - len(tail))
    return head + tail[:needed]


def get_branches(tree, k):
    if tree is None:
        return []

    branches = []

    def dfs(node, path):
        if node is None:
            return
        row = [node.data["x"], node.data["y"], node.data["z"]] + list(node.data.get("r", []))
        path.append(_coerce_row(row, k))
        if node.left is None and node.right is None:
            branches.append(path[:])
        else:
            if node.left:
                dfs(node.left, path)
            if node.right:
                dfs(node.right, path)
        path.pop()

    dfs(tree, [])
    return [np.array(branch, dtype=np.float32) for branch in branches]


def get_segments(tree, k):
    """Split tree into non-overlapping segments between bifurcation/leaf/root points.

    Unlike *get_branches* (which returns root-to-leaf paths and thus
    duplicates shared trunk nodes), this function returns one segment per
    edge run so that every tree node appears in exactly one segment (plus
    an overlap of 1 at junction nodes for continuity).
    """
    if tree is None:
        return []

    segments = []

    def _row(node):
        row = [node.data["x"], node.data["y"], node.data["z"]] + list(node.data.get("r", []))
        return _coerce_row(row, k)

    def _dfs(node, seg):
        if node is None:
            return
        seg.append(_row(node))
        children = [c for c in (node.left, node.right) if c is not None]
        if len(children) == 0:
            # leaf – flush current segment
            segments.append(np.array(seg, dtype=np.float64))
        elif len(children) == 1:
            # continuation – keep building current segment
            _dfs(children[0], seg)
        else:
            # bifurcation – flush current segment (keeping this node as end),
            # then start new segments for each child (including this node as start for continuity)
            segments.append(np.array(seg, dtype=np.float64))
            for child in children:
                _dfs(child, [_row(node)])  # overlap by 1 node

    _dfs(tree, [])
    return segments


def _parallel_transport(v, t1, t2):
    """Parallel-transport vector *v* from tangent *t1* to tangent *t2* via Rodrigues."""
    axis = np.cross(t1, t2)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(t1, t2))
    if sin_a < 1e-10:
        return v.copy() if cos_a > 0 else -v
    axis = axis / sin_a
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


def _compute_global_rings(tree, spline_samples, canonicalize_fn, align_fn,
                          ring_mode="circular", circle_points=64):
    """Walk tree once, compute one ring per node with parallel-transport frame.

    *ring_mode*:
        ``"circular"`` – replace each cross-section with a perfect circle
        whose radius equals the mean radius of the sampled spline.  Combined
        with parallel transport this guarantees zero twist artefacts.

        ``"spline"`` – keep the original B-spline cross-section shape,
        canonicalise in the PT frame, and refine with ``align_ring``.  This
        preserves shape detail but may show residual twist on near-circular
        rings.

    Returns *rings_map* (node.id → ring array or None), *nodes_map*
    (node.id → [x,y,z]), and *bif_info* (list of (position, radius)).
    """
    rings_map = {}
    nodes_map = {}
    bif_info = []

    def _walk(node, parent_ring=None, prev_tangent=None, prev_n=None):
        if node is None:
            return
        d = node.data
        pos = np.array([d["x"], d["y"], d["z"]], dtype=np.float64)
        coeffs = np.array(d.get("r", []), dtype=np.float64)
        nodes_map[node.id] = pos

        pts = sample_spline_coeffs(coeffs, spline_samples)
        if pts is None or not np.all(np.isfinite(pts)):
            rings_map[node.id] = None
            return

        children = [c for c in (node.left, node.right) if c is not None]

        # --- tangent ---
        if len(children) > 0:
            cpos = np.array([children[0].data["x"], children[0].data["y"],
                             children[0].data["z"]], dtype=np.float64)
            tangent = cpos - pos
        elif prev_tangent is not None:
            tangent = prev_tangent.copy()
        else:
            tangent = np.array([0.0, 0.0, 1.0])
        tn = np.linalg.norm(tangent)
        if tn < 1e-8:
            tangent = np.array([0.0, 0.0, 1.0])
        else:
            tangent = tangent / tn

        # --- reference direction via parallel transport ---
        if prev_n is None or prev_tangent is None:
            n = _perpendicular_unit(tangent)
        else:
            n = _parallel_transport(prev_n, prev_tangent / np.linalg.norm(prev_tangent), tangent)
            # Re-orthogonalise (numerical drift)
            n = n - np.dot(n, tangent) * tangent
            nn = np.linalg.norm(n)
            if nn < 1e-10:
                n = _perpendicular_unit(tangent)
            else:
                n = n / nn

        b = np.cross(tangent, n)
        bn = np.linalg.norm(b)
        if bn < 1e-10:
            n = _perpendicular_unit(tangent)
            b = np.cross(tangent, n)
            b = b / (np.linalg.norm(b) + 1e-12)
        else:
            b = b / bn

        # --- build ring ---
        if ring_mode == "circular":
            # Replace cross-section with a perfect circle (no twist possible)
            center = np.mean(pts, axis=0)
            radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
            n_pts = circle_points
            theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
            pts = pos + radius * (np.outer(np.cos(theta), n) + np.outer(np.sin(theta), b))
        else:
            # Keep original spline shape, canonicalise in PT frame
            center = pos
            dp = pts - center
            x = dp @ n
            y = dp @ b
            ang = np.arctan2(y, x)
            order = np.argsort(ang)
            pts = pts[order]
            x_sorted = x[order]
            start = int(np.argmax(x_sorted))
            pts = np.roll(pts, -start, axis=0)
            # Fine-tune with cyclic-shift alignment to parent ring
            if parent_ring is not None:
                pts = align_ring(parent_ring, pts, allow_flip=True)

        rings_map[node.id] = pts

        if len(children) == 2:
            r = float(np.mean(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1)))
            bif_info.append((pos, r))

        for c in children:
            _walk(c, pts, tangent, n)

    _walk(tree)
    return rings_map, nodes_map, bif_info


def _build_segments_from_global_rings(tree, rings_map):
    """Build segment ring lists from pre-computed global rings, splitting at bifurcations."""
    segments = []

    def _dfs(node, seg):
        if node is None:
            return
        ring = rings_map.get(node.id)
        if ring is None:
            return
        seg.append(ring)
        children = [c for c in (node.left, node.right) if c is not None]
        if len(children) == 0:
            segments.append(seg)
        elif len(children) == 1:
            _dfs(children[0], seg)
        else:
            segments.append(seg)
            for child in children:
                _dfs(child, [ring])  # overlap by 1 ring at junction

    _dfs(tree, [])
    return segments


def get_edges(tree):
    edges = []

    def dfs(node):
        if node is None:
            return
        if node.left is not None:
            edges.append((node, node.left))
            dfs(node.left)
        if node.right is not None:
            edges.append((node, node.right))
            dfs(node.right)

    dfs(tree)
    return edges


def create_3d_spline(points, smooth):
    points = np.asarray(points, dtype=np.float64)
    # drop NaN/Inf
    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    # drop duplicates (approx)
    if len(points) > 1:
        _, idx = np.unique(np.round(points, 6), axis=0, return_index=True)
        points = points[np.sort(idx)]
    if len(points) < 2:
        return None
    k = 3
    if len(points) <= 3:
        k = max(1, len(points) - 1)
    try:
        tck, _ = splprep(points.T, s=smooth, k=k)
        return tck
    except Exception:
        return None


def sample_spline(tck, num_samples):
    t = np.linspace(0, 1, num_samples)
    x, y, z = splev(t, tck)
    return np.vstack((x, y, z)).T


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    if len(coeffs) < 36:
        return None
    t = np.array(coeffs[24:36], dtype=np.float64)
    if t.shape[0] != 12:
        return None
    if not np.all(np.isfinite(t)):
        return None
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8 : (i + 1) * 8], dtype=np.float64) for i in range(3)]
    if not all(np.all(np.isfinite(ci)) for ci in c):
        return None
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples, endpoint=False)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def align_ring(prev, curr, allow_flip=True):
    if prev is None or curr is None:
        return curr
    if prev.shape != curr.shape:
        return curr

    def best_shift_score(reference, ring):
        n = reference.shape[0]
        best_shift = 0
        best_score = None
        for shift in range(n):
            rolled = np.roll(ring, -shift, axis=0)
            score = np.sum((reference - rolled) ** 2)
            if best_score is None or score < best_score:
                best_score = score
                best_shift = shift
        return np.roll(ring, -best_shift, axis=0), best_score

    aligned, score = best_shift_score(prev, curr)
    if allow_flip:
        flipped = curr[::-1].copy()
        aligned_flip, score_flip = best_shift_score(prev, flipped)
        if score_flip is not None and (score is None or score_flip < score):
            return aligned_flip
    return aligned


def canonicalize_ring(ring, center, tangent):
    ring = np.asarray(ring, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    t = np.asarray(tangent, dtype=np.float64)
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        t = t / tn

    n = _perpendicular_unit(t)
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-12)

    d = ring - center
    x = d @ n
    y = d @ b
    ang = np.arctan2(y, x)
    order = np.argsort(ang)
    ring = ring[order]

    x_sorted = x[order]
    start = int(np.argmax(x_sorted))
    ring = np.roll(ring, -start, axis=0)
    return ring


def ring_radius(points):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return float(np.median(distances))


def _resample_rings(rings, factor=1, mode="pchip", smooth_s=0.0):
    rings = np.asarray(rings, dtype=np.float64)
    if rings.ndim != 3:
        return rings
    num_rings, num_pts, _ = rings.shape
    if factor is None or factor <= 1 or num_rings < 2:
        return rings
    t = np.linspace(0.0, 1.0, num_rings)
    t_dense = np.linspace(0.0, 1.0, (num_rings - 1) * int(factor) + 1)
    out = np.zeros((len(t_dense), num_pts, 3), dtype=np.float64)
    for j in range(num_pts):
        for dim in range(3):
            y = rings[:, j, dim]
            if mode == "spline":
                f = UnivariateSpline(t, y, s=float(smooth_s))
                out[:, j, dim] = f(t_dense)
            elif mode == "pchip":
                f = PchipInterpolator(t, y, extrapolate=True)
                out[:, j, dim] = f(t_dense)
            else:
                out[:, j, dim] = np.interp(t_dense, t, y)
    return out


def _branch_arc_t(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    if len(nodes) == 0:
        return np.array([], dtype=np.float32)
    if len(nodes) == 1:
        return np.array([0.0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 1e-12:
        return np.zeros(len(nodes), dtype=np.float32)
    return (cum / total).astype(np.float32)


def _dedupe_t_points(t_vals, points, r_vals):
    t_vals = np.asarray(t_vals, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    r_vals = np.asarray(r_vals, dtype=np.float64)
    order = np.argsort(t_vals)
    t_vals = t_vals[order]
    points = points[order]
    r_vals = r_vals[order]
    unique_t = []
    unique_p = []
    unique_r = []
    i = 0
    while i < len(t_vals):
        j = i + 1
        while j < len(t_vals) and np.isclose(t_vals[j], t_vals[i], atol=1e-6):
            j += 1
        unique_t.append(float(t_vals[i]))
        unique_p.append(np.mean(points[i:j], axis=0))
        unique_r.append(float(np.mean(r_vals[i:j])))
        i = j
    return (
        np.array(unique_t, dtype=np.float64),
        np.array(unique_p, dtype=np.float64),
        np.array(unique_r, dtype=np.float64),
    )


def _dedupe_t_vectors(t_vals, vectors):
    t_vals = np.asarray(t_vals, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    order = np.argsort(t_vals)
    t_vals = t_vals[order]
    vectors = vectors[order]
    unique_t = []
    unique_v = []
    i = 0
    last_v = None
    while i < len(t_vals):
        j = i + 1
        while j < len(t_vals) and np.isclose(t_vals[j], t_vals[i], atol=1e-6):
            j += 1
        ref = vectors[i]
        block = vectors[i:j].copy()
        for k in range(block.shape[0]):
            if np.dot(block[k], ref) < 0:
                block[k] *= -1.0
        v = np.mean(block, axis=0)
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = last_v if last_v is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            v = v / n
        unique_t.append(float(t_vals[i]))
        unique_v.append(v)
        last_v = v
        i = j
    return np.array(unique_t, dtype=np.float64), np.array(unique_v, dtype=np.float64)


def _map_vectors_to_t(t_target, t_src, vectors_src):
    t_target = np.asarray(t_target, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    vectors_src = np.asarray(vectors_src, dtype=np.float64)
    if len(t_src) == 0:
        return np.zeros((len(t_target), 3), dtype=np.float64)
    idx = np.searchsorted(t_src, t_target, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return vectors_src[idx]


def _map_vectors_to_t_linear(t_target, t_src, vectors_src):
    t_target = np.asarray(t_target, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    v = np.asarray(vectors_src, dtype=np.float64)
    if len(t_src) == 0:
        return np.zeros((len(t_target), 3), dtype=np.float64)
    if len(t_src) == 1:
        vv = v[0] / (np.linalg.norm(v[0]) + 1e-12)
        return np.repeat(vv[None, :], len(t_target), axis=0)
    i = np.searchsorted(t_src, t_target, side="right") - 1
    i = np.clip(i, 0, len(t_src) - 2)
    j = i + 1
    t0 = t_src[i]
    t1 = t_src[j]
    w = np.where((t1 - t0) > 1e-12, (t_target - t0) / (t1 - t0), 0.0)
    w = w[:, None]
    vi = v[i]
    vj = v[j]
    flip = (np.sum(vi * vj, axis=1) < 0.0)[:, None]
    vj = np.where(flip, -vj, vj)
    out = (1.0 - w) * vi + w * vj
    out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out


def _orthonormalize_axes(axis1, axis2):
    axis1 = np.asarray(axis1, dtype=np.float64)
    axis2 = np.asarray(axis2, dtype=np.float64)
    axis1 = axis1 / (np.linalg.norm(axis1, axis=1, keepdims=True) + 1e-12)
    axis2 = axis2 - np.sum(axis2 * axis1, axis=1, keepdims=True) * axis1
    axis2 = axis2 / (np.linalg.norm(axis2, axis=1, keepdims=True) + 1e-12)
    return axis1, axis2


def _project_axes_to_segments(axis1, axis2, seg_vecs):
    axis1 = np.asarray(axis1, dtype=np.float64)
    axis2 = np.asarray(axis2, dtype=np.float64)
    seg_vecs = np.asarray(seg_vecs, dtype=np.float64)
    out1 = np.zeros_like(axis1)
    out2 = np.zeros_like(axis2)
    for i in range(seg_vecs.shape[0]):
        v = seg_vecs[i]
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            out1[i] = axis1[i]
            out2[i] = axis2[i]
            continue
        vhat = v / vn
        a1 = axis1[i] - np.dot(axis1[i], vhat) * vhat
        n1 = np.linalg.norm(a1)
        if n1 < 1e-8:
            a1 = _perpendicular_unit(vhat)
        else:
            a1 = a1 / n1
        a2 = np.cross(vhat, a1)
        n2 = np.linalg.norm(a2)
        if n2 < 1e-8:
            a2 = _perpendicular_unit(vhat)
        else:
            a2 = a2 / n2
        a1 = np.cross(a2, vhat)
        a1 = a1 / (np.linalg.norm(a1) + 1e-12)
        out1[i] = a1
        out2[i] = a2
    return out1, out2


def _project_axes_to_segment_pair(axis1, axis2, seg_vec):
    a1, a2 = _project_axes_to_segments(
        np.asarray(axis1, dtype=np.float64)[None, :],
        np.asarray(axis2, dtype=np.float64)[None, :],
        np.asarray(seg_vec, dtype=np.float64)[None, :],
    )
    return a1[0], a2[0]


def _ellipse_sdf_batch(x, y, a, b, n_iter=3):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    x0 = np.abs(x)
    y0 = np.abs(y)
    a = np.maximum(a, 1e-6)
    b = np.maximum(b, 1e-6)

    t = np.arctan2(y0 * b, x0 * a)
    for _ in range(max(1, int(n_iter))):
        ct = np.cos(t)
        st = np.sin(t)
        u = a * ct - x0
        v = b * st - y0
        f = -a * st * u + b * ct * v
        fp = -a * ct * u + (a * a) * st * st - b * st * v + (b * b) * ct * ct
        t = t - f / (fp + 1e-12)

    ct = np.cos(t)
    st = np.sin(t)
    dx = a * ct - x0
    dy = b * st - y0
    dist = np.sqrt(dx * dx + dy * dy)
    inside = (x0 / a) ** 2 + (y0 / b) ** 2 - 1.0
    return np.where(inside < 0.0, -dist, dist)


def _compute_radius_for_node(points, center, radius_mode, radius_percentile):
    distances = np.linalg.norm(points - center, axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return None
    if radius_mode == "max":
        return float(np.max(distances))
    if radius_mode == "mean":
        return float(np.mean(distances))
    if radius_mode == "percentile":
        return float(np.percentile(distances, radius_percentile))
    return float(np.median(distances))


def _perpendicular_unit(v):
    v = np.asarray(v, dtype=np.float64)
    if np.linalg.norm(v) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(v[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(v, other)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return u / n


def _node_tangent(nodes, i):
    nodes = np.asarray(nodes, dtype=np.float64)
    if len(nodes) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if len(nodes) == 1:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if i == 0:
        t = nodes[1] - nodes[0]
    elif i == len(nodes) - 1:
        t = nodes[-1] - nodes[-2]
    else:
        t = nodes[i + 1] - nodes[i - 1]
    n = np.linalg.norm(t)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return t / n


def _parallel_transport_frames(points):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    # tangents
    T = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if i == 0:
            tangent = points[1] - points[0] if n > 1 else np.array([0.0, 0.0, 1.0])
        elif i == n - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            tangent = T[i - 1] if i > 0 else np.array([0.0, 0.0, 1.0])
            norm = np.linalg.norm(tangent)
        T[i] = tangent / (norm if norm > 0 else 1.0)

    N = np.zeros_like(T)
    B = np.zeros_like(T)
    N[0] = _perpendicular_unit(T[0])
    B[0] = np.cross(T[0], N[0])
    for i in range(1, n):
        t_prev = T[i - 1]
        t_cur = T[i]
        v = np.cross(t_prev, t_cur)
        s = np.linalg.norm(v)
        if s < 1e-10:
            n_cur = N[i - 1].copy()
        else:
            k = v / s
            c = np.clip(np.dot(t_prev, t_cur), -1.0, 1.0)
            theta = np.arctan2(s, c)
            n_cur = (
                N[i - 1] * np.cos(theta)
                + np.cross(k, N[i - 1]) * np.sin(theta)
                + k * np.dot(k, N[i - 1]) * (1.0 - np.cos(theta))
            )
        n_cur = n_cur - np.dot(n_cur, t_cur) * t_cur
        nn = np.linalg.norm(n_cur)
        if nn < 1e-12:
            n_cur = _perpendicular_unit(t_cur)
        else:
            n_cur = n_cur / nn
        N[i] = n_cur
        B[i] = np.cross(t_cur, n_cur)
    return T, N, B


def _ellipse_axes_from_ring(points, center, n_vec, b_vec, percentile, use_pca=True):
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    d = points - center
    u = d @ n_vec
    v = d @ b_vec
    uv = np.stack([u, v], axis=1)
    if use_pca and uv.shape[0] >= 3:
        cov = np.cov(uv, rowvar=False)
        try:
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evecs = evecs[:, order]
        except Exception:
            evecs = np.eye(2)
    else:
        evecs = np.eye(2)
    # ensure right-handed
    if np.linalg.det(evecs) < 0:
        evecs[:, 1] *= -1.0

    uv_rot = uv @ evecs
    a = float(np.percentile(np.abs(uv_rot[:, 0]), percentile))
    b = float(np.percentile(np.abs(uv_rot[:, 1]), percentile))
    # rotated axes in 3D
    axis1 = n_vec * evecs[0, 0] + b_vec * evecs[1, 0]
    axis2 = n_vec * evecs[0, 1] + b_vec * evecs[1, 1]
    # normalize
    axis1 = axis1 / (np.linalg.norm(axis1) + 1e-12)
    axis2 = axis2 / (np.linalg.norm(axis2) + 1e-12)
    return a, b, axis1, axis2




def _smooth_radii(t_vals, r_vals, mode, smooth_s):
    if len(t_vals) < 2:
        return None
    if mode == "pchip":
        return PchipInterpolator(t_vals, r_vals, extrapolate=True)
    if mode == "spline":
        return UnivariateSpline(t_vals, r_vals, s=float(smooth_s))
    return None


def _reduce_balanced(items, combine):
    items = list(items)
    if not items:
        return None
    while len(items) > 1:
        next_items = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                next_items.append(combine(items[i], items[i + 1]))
            else:
                next_items.append(items[i])
        items = next_items
    return items[0]


def _sample_centerline(nodes, t_nodes, t_samples, smooth, mode):
    nodes = np.asarray(nodes, dtype=np.float64)
    t_nodes = np.asarray(t_nodes, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64)
    if mode == "spline":
        tck = create_3d_spline(nodes, smooth)
        if tck is not None:
            pts = np.column_stack(splev(t_samples, tck))
            return pts
    # fallback: linear in xyz over arc-length t
    xs = np.interp(t_samples, t_nodes, nodes[:, 0])
    ys = np.interp(t_samples, t_nodes, nodes[:, 1])
    zs = np.interp(t_samples, t_nodes, nodes[:, 2])
    return np.column_stack([xs, ys, zs])


# ---------------------------------------------------------------------------
# loft_ring mode: per-branch lofting with actual cross-section spline shapes
# ---------------------------------------------------------------------------

def _loft_ring_branch(nodes, splines, n_pts=32, target_spacing=0.005,
                      interpolate=True):
    """Build a watertight tube mesh for a single branch.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Node positions along the branch.
    splines : ndarray (N, 36)
        Per-node cross-section spline coefficients.
    n_pts : int
        Number of points per ring.
    target_spacing : float
        Desired spacing between interpolated rings along the centerline.
    interpolate : bool
        If True, add cubic-spline interpolated rings between nodes for
        smoother geometry.  If False, use only the original node rings.

    Returns
    -------
    trimesh.Trimesh or None
    """
    import trimesh as _tm
    from scipy.interpolate import CubicSpline as _CS, interp1d as _interp1d

    # ---- deduplicate zero-length segments --------------------------------
    keep = [0]
    for i in range(1, len(nodes)):
        if np.linalg.norm(nodes[i] - nodes[keep[-1]]) > 1e-6:
            keep.append(i)
    if len(keep) < 2:
        return None
    nodes_c = nodes[keep]
    splines_c = splines[keep]

    # ---- sample cross-section rings at node positions --------------------
    node_rings = []
    for i in range(len(nodes_c)):
        ring = sample_spline_coeffs(splines_c[i], n_pts)
        if ring is None:
            t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            ring = nodes_c[i] + 0.005 * np.column_stack(
                [np.cos(t), np.sin(t), np.zeros(n_pts)])
        if len(ring) != n_pts:
            t_old = np.linspace(0, 1, len(ring))
            t_new = np.linspace(0, 1, n_pts)
            ring = _interp1d(t_old, ring, axis=0)(t_new)
        node_rings.append(ring)

    # ---- align consecutive rings (minimize twist) ------------------------
    for i in range(1, len(node_rings)):
        best_shift, best_dist = 0, float('inf')
        for s in range(n_pts):
            rolled = np.roll(node_rings[i], s, axis=0)
            d = np.sum(np.linalg.norm(rolled - node_rings[i - 1], axis=1))
            if d < best_dist:
                best_dist = d
                best_shift = s
        node_rings[i] = np.roll(node_rings[i], best_shift, axis=0)

    # ---- optional: interpolate between nodes for smoother tubes ----------
    if interpolate and len(node_rings) >= 2:
        # Use NODE positions for arc-length (robust to degenerate splines
        # where ring centers can be far from the node position).
        centers = nodes_c.copy()
        dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(dists)])

        # Ensure strictly increasing arc (remove coincident centers)
        mask = np.concatenate([[True], dists > 1e-10])
        if mask.sum() < 2:
            # Fall back to non-interpolated
            all_rings = node_rings
        else:
            arc = arc[mask]
            centers = centers[mask]
            kept_rings = [node_rings[i] for i, m in enumerate(mask) if m]
            ring_centers = np.array([r.mean(0) for r in kept_rings])
            local_rings = [kept_rings[i] - ring_centers[i]
                           for i in range(len(kept_rings))]

            total_len = arc[-1]
            if total_len < 1e-8:
                return None

            n_interp = max(int(total_len / target_spacing), len(kept_rings))
            # Safety cap: prevent degenerate cases from consuming all memory
            MAX_RINGS = 2000
            n_interp = min(n_interp, MAX_RINGS)
            arc_new = np.linspace(0, total_len, n_interp)

            # Interpolate node positions (centerline)
            if len(centers) >= 4:
                cs_center = _CS(arc, centers, bc_type='natural')
            else:
                cs_center = PchipInterpolator(arc, centers)
            centers_interp = cs_center(arc_new)

            # Interpolate ring centers along the arc as well
            if len(ring_centers) >= 4:
                cs_rc = _CS(arc, ring_centers, bc_type='natural')
            else:
                cs_rc = PchipInterpolator(arc, ring_centers)
            ring_centers_interp = cs_rc(arc_new)

            local_arr = np.array(local_rings)  # (n_nodes, n_pts, 3)
            rings_interp = []
            for j in range(n_pts):
                offsets_j = local_arr[:, j, :]
                if len(offsets_j) >= 4:
                    cs_j = _CS(arc, offsets_j, bc_type='natural')
                else:
                    cs_j = PchipInterpolator(arc, offsets_j)
                rings_interp.append(cs_j(arc_new))

            all_rings = []
            for i in range(n_interp):
                ring_i = np.array([rings_interp[j][i] for j in range(n_pts)])
                all_rings.append(ring_i + ring_centers_interp[i])
    else:
        all_rings = node_rings

    # ---- triangulate ------------------------------------------------------
    n_rings = len(all_rings)
    verts = np.vstack(all_rings)

    faces = []
    for i in range(n_rings - 1):
        bc = i * n_pts
        bn = (i + 1) * n_pts
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            faces.append([bc + j, bn + j, bn + j1])
            faces.append([bc + j, bn + j1, bc + j1])

    # start cap
    c0 = all_rings[0].mean(0)
    c0_idx = len(verts)
    verts = np.vstack([verts, c0.reshape(1, 3)])
    for j in range(n_pts):
        j1 = (j + 1) % n_pts
        faces.append([c0_idx, j1, j])

    # end cap
    c1 = all_rings[-1].mean(0)
    c1_idx = len(verts)
    verts = np.vstack([verts, c1.reshape(1, 3)])
    base_last = (n_rings - 1) * n_pts
    for j in range(n_pts):
        j1 = (j + 1) % n_pts
        faces.append([c1_idx, base_last + j, base_last + j1])

    mesh = _tm.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    # Skip fix_normals – the lofting already guarantees consistent winding and
    # trimesh's fix_normals builds a NetworkX face-adjacency graph that is
    # extremely slow / memory-hungry for large meshes.
    return mesh


def build_loft_ring_mesh(tree, k, params):
    """Build a combined mesh by lofting per-branch cross-section rings.

    Each branch becomes a separate watertight tube; they overlap naturally
    at bifurcation nodes.  Returns a VTK vtkPolyData or None on failure.

    Relevant params keys:
        loft_ring_pts        – points per ring (default 32)
        loft_ring_spacing    – target arc-length between interpolated rings
                               (default 0.005)
        loft_ring_interpolate – whether to add interpolated intermediate
                                rings (default True)
    """
    import trimesh as _tm

    n_pts = int(params.get("loft_ring_pts", 32))
    spacing = float(params.get("loft_ring_spacing", 0.005))
    do_interp = bool(params.get("loft_ring_interpolate", True))

    branches = get_branches(tree, k)
    meshes = []
    for branch in branches:
        m = _loft_ring_branch(branch[:, :3], branch[:, 3:],
                              n_pts=n_pts, target_spacing=spacing,
                              interpolate=do_interp)
        if m is not None:
            meshes.append(m)

    if not meshes:
        return None

    combined = _tm.util.concatenate(meshes)

    # Convert to VTK polydata using numpy arrays (fast, no Python loop)
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        verts_f = np.ascontiguousarray(combined.vertices, dtype=np.float64)
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(verts_f, deep=True))

        nf = len(combined.faces)
        # VTK cell array: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        faces_i = np.ascontiguousarray(combined.faces, dtype=np.int64)
        conn = np.column_stack([
            np.full(nf, 3, dtype=np.int64),
            faces_i
        ]).ravel()
        cells = vtk.vtkCellArray()
        vtk_id_arr = numpy_to_vtkIdTypeArray(conn, deep=True)
        cells.SetCells(nf, vtk_id_arr)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(pd)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        return normals.GetOutput()
    except Exception:
        return None


def build_loft_polydata(rings, add_caps):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for loft reconstruction.") from exc

    if len(rings) < 2:
        return None

    n_points = rings[0].shape[0]
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    for ring in rings:
        if ring.shape[0] != n_points:
            return None
        for point in ring:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))

    for r in range(len(rings) - 1):
        for i in range(n_points):
            i0 = r * n_points + i
            i1 = r * n_points + (i + 1) % n_points
            i2 = (r + 1) * n_points + (i + 1) % n_points
            i3 = (r + 1) * n_points + i
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i0)
            quad.GetPointIds().SetId(1, i1)
            quad.GetPointIds().SetId(2, i2)
            quad.GetPointIds().SetId(3, i3)
            polys.InsertNextCell(quad)

    if add_caps:
        start_center = np.mean(rings[0], axis=0)
        end_center = np.mean(rings[-1], axis=0)
        start_id = points.InsertNextPoint(float(start_center[0]), float(start_center[1]), float(start_center[2]))
        end_id = points.InsertNextPoint(float(end_center[0]), float(end_center[1]), float(end_center[2]))

        for i in range(n_points):
            i0 = i
            i1 = (i + 1) % n_points
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, start_id)
            tri.GetPointIds().SetId(1, i1)
            tri.GetPointIds().SetId(2, i0)
            polys.InsertNextCell(tri)

        base = (len(rings) - 1) * n_points
        for i in range(n_points):
            i0 = base + i
            i1 = base + (i + 1) % n_points
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, end_id)
            tri.GetPointIds().SetId(1, i0)
            tri.GetPointIds().SetId(2, i1)
            polys.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


def build_loft_mesh(tree, k, params):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for loft reconstruction.") from exc

    loft_branch_mode = params.get("loft_branch_mode", "branches")  # branches | segments | smooth_segments

    spline_samples = int(params.get("spline_samples", 50))
    loft_align = bool(params.get("loft_align", True))
    loft_allow_flip = bool(params.get("loft_allow_flip", True))
    loft_caps = bool(params.get("loft_caps", False))
    loft_clean = bool(params.get("loft_clean", True))
    loft_clean_tol = float(params.get("loft_clean_tol", 0.0))
    loft_min_radius = float(params.get("loft_min_radius", 0.0))
    loft_resample_factor = int(params.get("loft_resample_factor", 1))
    loft_vertex_smooth = params.get("loft_vertex_smooth", "pchip")
    loft_vertex_spline_s = float(params.get("loft_vertex_spline_s", 0.0))
    loft_canonicalize = bool(params.get("loft_canonicalize", True))
    loft_center_mode = params.get("loft_center_mode", "node")
    loft_invalid_mode = params.get("loft_invalid_mode", "skip")
    loft_normals = bool(params.get("loft_normals", False))
    loft_smooth = bool(params.get("loft_smooth", False))
    loft_smooth_iters = int(params.get("loft_smooth_iters", 30))
    loft_subdiv_levels = int(params.get("loft_subdiv_levels", 0))
    loft_fill_holes = bool(params.get("loft_fill_holes", False))
    loft_fill_hole_size = float(params.get("loft_fill_hole_size", 0.5))
    loft_ring_mode = str(params.get("loft_ring_mode", "circular"))   # "circular" | "spline"

    # -----------------------------------------------------------------
    # smooth_segments path: global rings → per-segment loft → per-segment
    # Loop subdivision → append → clean → fill holes → normals
    # -----------------------------------------------------------------
    if loft_branch_mode == "smooth_segments":
        can_fn = canonicalize_ring if loft_canonicalize else (lambda pts, c, t: pts)
        aln_fn = (lambda prev, pts, allow_flip=True: align_ring(prev, pts, allow_flip=allow_flip)) if loft_align else (lambda prev, pts, **kw: pts)
        rings_map, nodes_map, bif_info = _compute_global_rings(
            tree, spline_samples, can_fn, aln_fn,
            ring_mode=loft_ring_mode, circle_points=spline_samples,
        )

        seg_ring_lists = _build_segments_from_global_rings(tree, rings_map)
        if not seg_ring_lists:
            return None

        append = vtk.vtkAppendPolyData()
        # Collect per-segment centerlines for radial normal computation
        all_seg_centerlines = []
        for seg_rings in seg_ring_lists:
            if len(seg_rings) < 2:
                continue
            seg_cl = np.array([r.mean(axis=0) for r in seg_rings])
            resampled = _resample_rings(
                seg_rings,
                factor=loft_resample_factor,
                mode=loft_vertex_smooth,
                smooth_s=loft_vertex_spline_s,
            )
            # Centerline from resampled rings (higher resolution)
            resampled_cl = np.array([r.mean(axis=0) for r in resampled])
            all_seg_centerlines.append(resampled_cl)
            poly = build_loft_polydata(resampled, False)
            if poly is None:
                all_seg_centerlines.pop()
                continue
            # Triangulate before subdivision (Loop requires triangles)
            tri_seg = vtk.vtkTriangleFilter()
            tri_seg.SetInputData(poly)
            tri_seg.Update()
            seg_poly = tri_seg.GetOutput()
            # Per-segment Loop subdivision (works because each segment is manifold)
            if loft_subdiv_levels > 0 and seg_poly.GetNumberOfCells() >= 4:
                subdiv = vtk.vtkLoopSubdivisionFilter()
                subdiv.SetInputData(seg_poly)
                subdiv.SetNumberOfSubdivisions(loft_subdiv_levels)
                subdiv.Update()
                result = subdiv.GetOutput()
                if result is not None and result.GetNumberOfCells() > 0:
                    seg_poly = result
            append.AddInputData(seg_poly)

        append.Update()
        polydata = append.GetOutput()

        # Clean merged mesh
        if loft_clean:
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(polydata)
            if loft_clean_tol > 0:
                cleaner.SetTolerance(loft_clean_tol)
            cleaner.Update()
            polydata = cleaner.GetOutput()

        # Fill holes at junctions
        if loft_fill_holes:
            filler = vtk.vtkFillHolesFilter()
            filler.SetInputData(polydata)
            filler.SetHoleSize(loft_fill_hole_size)
            filler.Update()
            polydata = filler.GetOutput()

        # Final normals: compute outward normals from tube centerlines.
        # For each vertex, find nearest centerline point across all
        # segments, then normal = normalize(vertex - nearest_cl_point).
        if loft_normals:
            # Build a single dense centerline array from all segments
            cl_all = np.vstack(all_seg_centerlines) if all_seg_centerlines else np.zeros((1, 3))
            n_pts = polydata.GetNumberOfPoints()
            # Extract all vertex positions into numpy array
            vtk_pts = polydata.GetPoints()
            verts = np.array([vtk_pts.GetPoint(i) for i in range(n_pts)], dtype=np.float64)
            # Use scipy KDTree for fast nearest-centerline lookup
            try:
                from scipy.spatial import cKDTree
                tree_kd = cKDTree(cl_all)
                _, idx = tree_kd.query(verts, k=1)
                nearest = cl_all[idx]
            except ImportError:
                # Fallback: brute-force (slower but works)
                # Process in chunks to limit memory
                chunk = 2000
                nearest = np.empty_like(verts)
                for s in range(0, n_pts, chunk):
                    e = min(s + chunk, n_pts)
                    d = np.sum((verts[s:e, None, :] - cl_all[None, :, :]) ** 2, axis=2)
                    nearest[s:e] = cl_all[np.argmin(d, axis=1)]
            radial = verts - nearest
            rn = np.linalg.norm(radial, axis=1, keepdims=True)
            rn[rn < 1e-12] = 1.0
            radial = radial / rn
            normals_arr = vtk.vtkFloatArray()
            normals_arr.SetNumberOfComponents(3)
            normals_arr.SetNumberOfTuples(n_pts)
            normals_arr.SetName("Normals")
            for vi in range(n_pts):
                normals_arr.SetTuple3(vi, radial[vi, 0], radial[vi, 1], radial[vi, 2])
            polydata.GetPointData().SetNormals(normals_arr)

        # Optional global smooth
        if loft_smooth:
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(polydata)
            smooth.SetNumberOfIterations(loft_smooth_iters)
            smooth.BoundarySmoothingOff()
            smooth.FeatureEdgeSmoothingOff()
            smooth.NonManifoldSmoothingOn()
            smooth.NormalizeCoordinatesOn()
            smooth.Update()
            polydata = smooth.GetOutput()

        return polydata

    # -----------------------------------------------------------------
    # loft_v2 path: global rings → per-segment PCHIP loft (NO per-segment
    # subdivision) → append → clean → strip degenerate cells → triangulate
    # → optional Taubin smooth → normals.
    # This avoids the connectivity breakage caused by per-segment Loop
    # subdivision (which independently moves shared boundary vertices at
    # bifurcations, breaking coincident-point merging).
    # -----------------------------------------------------------------
    if loft_branch_mode == "loft_v2":
        can_fn = canonicalize_ring if loft_canonicalize else (lambda pts, c, t: pts)
        aln_fn = (
            (lambda prev, pts, allow_flip=True: align_ring(prev, pts, allow_flip=allow_flip))
            if loft_align
            else (lambda prev, pts, **kw: pts)
        )
        rings_map, nodes_map, bif_info = _compute_global_rings(
            tree, spline_samples, can_fn, aln_fn,
            ring_mode=loft_ring_mode, circle_points=spline_samples,
        )

        seg_ring_lists = _build_segments_from_global_rings(tree, rings_map)
        if not seg_ring_lists:
            return None

        append = vtk.vtkAppendPolyData()
        all_seg_centerlines = []
        for seg_rings in seg_ring_lists:
            if len(seg_rings) < 2:
                continue
            resampled = _resample_rings(
                seg_rings,
                factor=loft_resample_factor,
                mode=loft_vertex_smooth,
                smooth_s=loft_vertex_spline_s,
            )
            resampled_cl = np.array([r.mean(axis=0) for r in resampled])
            all_seg_centerlines.append(resampled_cl)
            poly = build_loft_polydata(resampled, False)
            if poly is None:
                all_seg_centerlines.pop()
                continue
            append.AddInputData(poly)

        append.Update()
        polydata = append.GetOutput()

        # Merge coincident points at shared bifurcation rings.
        # Use absolute tolerance to avoid bounding-box-relative merging
        # that can either miss exact duplicates (tol=0.0 relative) or
        # incorrectly merge nearby vertices from PCHIP interpolation.
        if loft_clean:
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(polydata)
            cleaner.ToleranceIsAbsoluteOn()
            cleaner.SetAbsoluteTolerance(
                loft_clean_tol if loft_clean_tol > 0 else 1e-12
            )
            cleaner.PointMergingOn()
            cleaner.Update()
            polydata = cleaner.GetOutput()

        # Strip degenerate line cells produced when quads at bifurcation
        # boundaries have vertices merged to the same point
        polys = polydata.GetPolys()
        pts = polydata.GetPoints()
        stripped = vtk.vtkPolyData()
        stripped.SetPoints(pts)
        stripped.SetPolys(polys)
        polydata = stripped

        # Triangulate
        tri_f = vtk.vtkTriangleFilter()
        tri_f.SetInputData(polydata)
        tri_f.Update()
        polydata = tri_f.GetOutput()

        # Optional global Taubin smooth (preserves volume better than Laplacian)
        if loft_smooth:
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(polydata)
            smooth.SetNumberOfIterations(loft_smooth_iters)
            smooth.SetPassBand(0.1)
            smooth.BoundarySmoothingOff()
            smooth.FeatureEdgeSmoothingOff()
            smooth.NonManifoldSmoothingOn()
            smooth.NormalizeCoordinatesOn()
            smooth.Update()
            polydata = smooth.GetOutput()

        # Compute outward normals from tube centerlines
        if loft_normals and all_seg_centerlines:
            cl_all = np.vstack(all_seg_centerlines)
            n_pts = polydata.GetNumberOfPoints()
            vtk_pts = polydata.GetPoints()
            verts = np.array(
                [vtk_pts.GetPoint(i) for i in range(n_pts)], dtype=np.float64
            )
            try:
                from scipy.spatial import cKDTree
                tree_kd = cKDTree(cl_all)
                _, idx = tree_kd.query(verts, k=1)
                nearest = cl_all[idx]
            except ImportError:
                chunk = 2000
                nearest = np.empty_like(verts)
                for s in range(0, n_pts, chunk):
                    e = min(s + chunk, n_pts)
                    d = np.sum(
                        (verts[s:e, None, :] - cl_all[None, :, :]) ** 2, axis=2
                    )
                    nearest[s:e] = cl_all[np.argmin(d, axis=1)]
            radial = verts - nearest
            rn = np.linalg.norm(radial, axis=1, keepdims=True)
            rn[rn < 1e-12] = 1.0
            radial = radial / rn
            normals_arr = vtk.vtkFloatArray()
            normals_arr.SetNumberOfComponents(3)
            normals_arr.SetNumberOfTuples(n_pts)
            normals_arr.SetName("Normals")
            for vi in range(n_pts):
                normals_arr.SetTuple3(
                    vi, radial[vi, 0], radial[vi, 1], radial[vi, 2]
                )
            polydata.GetPointData().SetNormals(normals_arr)

        # Fill open holes at junctions (optional, usually not needed)
        if loft_fill_holes:
            filler = vtk.vtkFillHolesFilter()
            filler.SetInputData(polydata)
            filler.SetHoleSize(loft_fill_hole_size)
            filler.Update()
            polydata = filler.GetOutput()

        return polydata

    # -----------------------------------------------------------------
    # Original branches / segments path
    # -----------------------------------------------------------------
    if loft_branch_mode == "segments":
        branches = get_segments(tree, k)
    else:
        branches = get_branches(tree, k)
    if not branches:
        return None

    append = vtk.vtkAppendPolyData()

    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        rings = []
        prev = None
        for idx, coeffs in enumerate(splines):
            points = sample_spline_coeffs(coeffs, spline_samples)
            if points is None:
                if loft_invalid_mode == "copy" and prev is not None:
                    points = prev.copy()
                else:
                    continue
            if not np.all(np.isfinite(points)):
                if loft_invalid_mode == "copy" and prev is not None:
                    points = prev.copy()
                else:
                    continue
            # Sanity: reject rings with extreme coordinates (exploded splines)
            _ring_max = np.max(np.abs(points))
            _node_max = np.max(np.abs(nodes)) if len(nodes) > 0 else 1.0
            _ring_limit = max(10.0 * _node_max, 10.0)
            if _ring_max > _ring_limit:
                if loft_invalid_mode == "copy" and prev is not None:
                    points = prev.copy()
                else:
                    continue
            # Sanity: reject rings far from their node center
            if idx < len(nodes):
                _ring_center = np.mean(points, axis=0)
                _dist_to_node = np.linalg.norm(_ring_center - nodes[idx])
                if _dist_to_node > 2.0 * _node_max and _dist_to_node > 1.0:
                    if loft_invalid_mode == "copy" and prev is not None:
                        points = prev.copy()
                    else:
                        continue
            if loft_min_radius > 0 and ring_radius(points) < loft_min_radius:
                if loft_invalid_mode == "copy" and prev is not None:
                    points = prev.copy()
                else:
                    continue

            if loft_center_mode == "node" and idx < len(nodes):
                center = nodes[idx]
            else:
                center = np.mean(points, axis=0)

            if loft_canonicalize:
                if idx == 0:
                    tangent = nodes[1] - nodes[0] if len(nodes) > 1 else np.array([0.0, 0.0, 1.0])
                elif idx == len(nodes) - 1:
                    tangent = nodes[-1] - nodes[-2]
                else:
                    tangent = nodes[idx + 1] - nodes[idx - 1]
                points = canonicalize_ring(points, center, tangent)

            if loft_align:
                points = align_ring(prev, points, allow_flip=loft_allow_flip)
            rings.append(points)
            prev = points

        if len(rings) < 2:
            continue

        rings = _resample_rings(
            rings,
            factor=loft_resample_factor,
            mode=loft_vertex_smooth,
            smooth_s=loft_vertex_spline_s,
        )

        poly = build_loft_polydata(rings, loft_caps)
        if poly is None:
            continue
        append.AddInputData(poly)

    append.Update()
    polydata = append.GetOutput()
    if loft_clean:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        if loft_clean_tol > 0:
            cleaner.SetTolerance(loft_clean_tol)
        cleaner.Update()
        polydata = cleaner.GetOutput()

    if loft_normals:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.AutoOrientNormalsOn()
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        polydata = normals.GetOutput()

    if loft_smooth:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(polydata)
        smooth.SetNumberOfIterations(loft_smooth_iters)
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        polydata = smooth.GetOutput()

    # Fill holes (also available in original path)
    if loft_fill_holes:
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(polydata)
        filler.SetHoleSize(loft_fill_hole_size)
        filler.Update()
        polydata = filler.GetOutput()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    return tri.GetOutput()


def build_sweep_sdf(tree, k, params):
    branches = get_branches(tree, k)
    if not branches:
        return None

    radius_mode = params.get("radius_mode", "median")
    radius_percentile = params.get("radius_percentile", 90)
    radius_cap = params.get("radius_cap", None)
    center_mode = params.get("center_mode", "node")
    spline_samples = int(params.get("spline_samples", 50))
    fallback_radius = float(params.get("sweep_fallback_radius", params.get("fallback_radius", 0.02)))
    min_radius = float(params.get("sweep_min_radius", 0.0))
    smooth_mode = params.get("sweep_radius_smooth", "pchip")
    smooth_s = float(params.get("sweep_spline_s", 0.0))
    sweep_samples = int(params.get("sweep_samples", 400))
    sweep_centerline_mode = params.get("sweep_centerline_mode", "spline")
    sweep_centerline_smooth = float(params.get("sweep_centerline_smooth", params.get("centerline_smooth", 0.0)))
    sweep_min_segment_length = float(params.get("sweep_min_segment_length", 0.0))
    sweep_radius_sanity_factor = params.get("sweep_radius_sanity_factor", None)
    branch_smooth_k = params.get("sweep_branch_smooth_k", 0.0)
    try:
        branch_smooth_k = float(branch_smooth_k) if branch_smooth_k is not None else 0.0
    except Exception:
        branch_smooth_k = 0.0

    cone_fn, _, sweep_union, _ = get_sweep_primitives_and_union(params)

    vessels = []
    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        if nodes.shape[0] < 2:
            continue

        t_nodes = _branch_arc_t(nodes)
        radii = []
        last_radius = None
        for coeffs, node in zip(splines, nodes):
            points = sample_spline_coeffs(coeffs, spline_samples)
            if points is None or not np.all(np.isfinite(points)):
                radius = last_radius if last_radius is not None else fallback_radius
                radii.append(float(radius))
                last_radius = float(radius)
                continue

            center = node
            if center_mode == "centroid":
                center = np.mean(points, axis=0)
            radius = _compute_radius_for_node(points, center, radius_mode, radius_percentile)
            if radius is None or not np.isfinite(radius):
                radius = last_radius if last_radius is not None else fallback_radius
            if radius_cap is not None:
                try:
                    radius = min(radius, float(radius_cap))
                except Exception:
                    pass
            if min_radius > 0:
                radius = max(radius, min_radius)
            radii.append(float(radius))
            last_radius = float(radius)

        if sweep_radius_sanity_factor is not None:
            try:
                factor = float(sweep_radius_sanity_factor)
            except Exception:
                factor = None
            if factor is not None and factor > 0 and len(radii) > 0:
                r_med = float(np.nanmedian(radii))
                if np.isfinite(r_med) and r_med > 0:
                    cap_val = r_med * factor
                    radii = [min(r, cap_val) if np.isfinite(r) else r for r in radii]

        t_u, nodes_u, r_u = _dedupe_t_points(t_nodes, nodes, radii)
        if len(t_u) < 2:
            continue

        r_fn = _smooth_radii(t_u, r_u, smooth_mode, smooth_s)
        samples = max(2, sweep_samples)
        t_samples = np.linspace(0.0, 1.0, samples)

        # resample centerline along arc-length
        centers = _sample_centerline(nodes_u, t_u, t_samples, sweep_centerline_smooth, sweep_centerline_mode)

        if r_fn is None:
            r_samples = np.interp(t_samples, t_u, r_u)
        else:
            r_samples = r_fn(t_samples)
        r_samples = np.asarray(r_samples, dtype=np.float64)
        if radius_cap is not None:
            try:
                r_samples = np.minimum(r_samples, float(radius_cap))
            except Exception:
                pass
        if min_radius > 0:
            r_samples = np.maximum(r_samples, min_radius)
        r_samples = np.where(np.isfinite(r_samples), r_samples, fallback_radius)

        segments = []
        for j in range(len(centers) - 1):
            p0 = centers[j]
            p1 = centers[j + 1]
            r0 = float(r_samples[j])
            r1 = float(r_samples[j + 1])
            if r0 <= 0 and r1 <= 0:
                continue
            if sweep_min_segment_length > 0:
                if np.linalg.norm(p1 - p0) < sweep_min_segment_length:
                    continue
            segments.append(cone_fn(p0, p1, r0, r1))

        if segments:
            if branch_smooth_k > 0:
                branch_sdf = _reduce_balanced(
                    segments,
                    lambda a, b: sweep_union(a, b, k=branch_smooth_k),
                )
            else:
                branch_sdf = union(*segments)
            if branch_sdf is not None:
                vessels.append(branch_sdf)

    if not vessels:
        return None

    k = params.get("smooth_union_k")
    try:
        k = float(k) if k is not None else 0.0
    except Exception:
        k = 0.0

    if k > 0:
        return _reduce_balanced(vessels, lambda a, b: sweep_union(a, b, k=k))
    return union(*vessels)


def build_sdf(tree, k, centerline_samples, centerline_smooth, params):
    branches = get_branches(tree, k)
    if not branches:
        return None

    vessel3, vessel3_robust, vessel3_stable = get_sdf_fns(params)
    smooth_union_fn = get_backend(params).get("smooth_union")

    vessels = []
    debug_sdf = bool(params.get("debug_sdf", False))
    recon_mode = params.get("recon_mode", "legacy")
    branch_local = False
    if recon_mode in {"legacy_branch", "legacy_branch_local"}:
        branch_local = True
        recon_mode = "legacy"
    if recon_mode not in {"legacy", "stable", "sdf_offset"}:
        recon_mode = "legacy"
    legacy_variant = params.get("legacy_variant", "original")  # original | robust

    radius_mode = params.get("radius_mode", "median")
    radius_percentile = params.get("radius_percentile", 90)
    radius_cap = params.get("radius_cap")
    center_mode = params.get("center_mode", "node")
    fallback_radius = params.get("fallback_radius", 0.0)
    spline_samples = int(params.get("spline_samples", 50))
    centerline_t_mode = params.get("centerline_t_mode", "optimize")
    leaf_taper = bool(params.get("leaf_taper", False))

    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]

        # ── leaf-end endcap reduction ────────────────────────────────────
        # Reduce the last node's ring radius in-place so the endcap hemisphere
        # is smaller than the vessel body.  No new nodes → no topology change
        # → no watertight issues.  The endcap sphere radius becomes taper_r
        # (if leaf ring radius > taper_r); the vessel tapers visually at its tip.
        if leaf_taper and len(nodes) >= 2:
            try:
                leaf_spts = sample_spline_coeffs(list(splines[-1]), 64)
                if leaf_spts is not None and len(leaf_spts) > 0:
                    leaf_mean_r = float(np.linalg.norm(
                        leaf_spts - leaf_spts.mean(0), axis=1).mean())
                else:
                    leaf_mean_r = 0.0
            except Exception:
                leaf_mean_r = 0.0

            taper_min_r = float(params.get("leaf_taper_min_r", 0.018))
            taper_r     = min(taper_min_r, leaf_mean_r * 0.4)

            # Only shrink large-radius leaves that have noticeable endcap bumps.
            # Skip if taper_r is too small (leaf is already thin) or if the
            # leaf ring radius is already ≤ taper_min_r (nothing to gain).
            if taper_r >= 0.010 and leaf_mean_r > taper_r * 1.1:
                scale = taper_r / max(leaf_mean_r, 1e-9)
                # Scale ctrl pts relative to the ring centroid (≈ leaf node xyz)
                leaf_ctr = nodes[-1].astype(np.float64)
                reduced  = splines[-1].astype(np.float64).copy()
                cx = reduced[0:8]   - leaf_ctr[0]
                cy = reduced[8:16]  - leaf_ctr[1]
                cz = reduced[16:24] - leaf_ctr[2]
                reduced[0:8]   = leaf_ctr[0] + cx * scale
                reduced[8:16]  = leaf_ctr[1] + cy * scale
                reduced[16:24] = leaf_ctr[2] + cz * scale
                # Overwrite last row of splines (in-place replace)
                splines = np.vstack([splines[:-1], reduced[np.newaxis].astype(np.float32)])

        if debug_sdf:
            print("NODES min/max:", np.nanmin(nodes, axis=0), np.nanmax(nodes, axis=0))
            print("NODES finite:", np.all(np.isfinite(nodes)))
        centerline_nodes = nodes
        if recon_mode == "sdf_offset":
            centers = []
            last_center = None
            for coeffs, node in zip(splines, nodes):
                points = sample_spline_coeffs(coeffs, spline_samples)
                if points is None:
                    center = last_center if last_center is not None else node
                else:
                    center = np.mean(points, axis=0)
                centers.append(center)
                last_center = center
            centerline_nodes = np.array(centers, dtype=np.float32)
        tck = create_3d_spline(centerline_nodes, centerline_smooth)
        if tck is None:
            continue
        if branch_local:
            sampled = np.array(centerline_nodes, dtype=np.float64)
            t_values = _branch_arc_t(sampled)
            local_t_mode = "kdtree"
        else:
            sampled = sample_spline(tck, centerline_samples)
            t_values = np.linspace(0, 1, sampled.shape[0])
            local_t_mode = centerline_t_mode
        if recon_mode in {"stable", "sdf_offset"}:
            centers_for_radius = centerline_nodes if recon_mode == "sdf_offset" else nodes
            stable_kwargs = dict(
                radius_mode=radius_mode,
                radius_percentile=radius_percentile,
                radius_cap=radius_cap,
                center_mode=center_mode,
                fallback_radius=fallback_radius,
            )
            if "tck" in inspect.signature(vessel3_stable).parameters:
                stable_kwargs.update(tck=tck, sampled_spline=sampled, t_values=t_values)
            if "centerline_t_mode" in inspect.signature(vessel3_stable).parameters:
                stable_kwargs.update(centerline_t_mode=local_t_mode)
            vessels.append(vessel3_stable(sampled, centers_for_radius, splines, **stable_kwargs))
        else:
            if legacy_variant == "robust":
                robust_kwargs = {}
                if "tck" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(tck=tck, sampled_spline=sampled, t_values=t_values)
                if "centerline_t_mode" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(centerline_t_mode=local_t_mode)
                if "fallback_radius" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(fallback_radius=params.get("robust_fallback_radius", 0.02))
                if "min_radius" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(min_radius=params.get("robust_min_radius", 0.005))
                if "radius_cap" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(radius_cap=params.get("radius_cap", None))
                if "sanity_percentile" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(
                        sanity_percentile=params.get("robust_sanity_percentile", 95),
                        sanity_threshold=params.get("robust_sanity_threshold", None),
                    )
                if "debug" in inspect.signature(vessel3_robust).parameters:
                    robust_kwargs.update(
                        debug=debug_sdf,
                        debug_scalar_threshold=params.get("robust_debug_scalar_threshold", 10.0),
                    )
                vessels.append(vessel3_robust(sampled, nodes, splines, **robust_kwargs))
            else:
                legacy_kwargs = {}
                if "tck" in inspect.signature(vessel3).parameters:
                    legacy_kwargs.update(tck=tck, sampled_spline=sampled, t_values=t_values)
                if "centerline_t_mode" in inspect.signature(vessel3).parameters:
                    legacy_kwargs.update(centerline_t_mode=local_t_mode)
                vessels.append(vessel3(sampled, nodes, splines, **legacy_kwargs))

    if not vessels:
        return None
    if len(vessels) == 1:
        return vessels[0]

    k = params.get("smooth_union_k")
    try:
        k = float(k) if k is not None else 0.0
    except Exception:
        k = 0.0

    out = vessels[0]
    for v in vessels[1:]:
        if k > 0 and smooth_union_fn is not None:
            out = smooth_union_fn(out, v, k=k)
        else:
            out = union(out, v)
    return out


def build_sdf_segments(tree, k, centerline_samples, centerline_smooth, params):
    """Segment-based SDF construction — avoids trunk node duplication.

    Unlike ``build_sdf`` (which uses ``get_branches`` and duplicates every
    shared trunk node once per child branch, causing ~2× overlap and junction
    bulging), this function uses ``get_segments`` to split the tree into
    non-overlapping segments with only 1-node overlap at bifurcations.

    This produces cleaner junctions and more accurate geometry overall.
    """
    segments = get_segments(tree, k)
    if not segments:
        return None

    vessel3, vessel3_robust, vessel3_stable = get_sdf_fns(params)
    smooth_union_fn = get_backend(params).get("smooth_union")

    vessels = []
    recon_mode = params.get("recon_mode", "legacy")
    if recon_mode not in {"legacy", "stable", "sdf_offset"}:
        recon_mode = "legacy"
    legacy_variant = params.get("legacy_variant", "original")
    spline_samples = int(params.get("spline_samples", 128))
    centerline_t_mode = params.get("centerline_t_mode", "optimize")
    leaf_taper = bool(params.get("leaf_taper", False))

    for seg in segments:
        if len(seg) < 2:
            continue  # skip single-node segments (bifurcation points)

        nodes = seg[:, :3]
        splines = seg[:, 3:]

        # ── leaf-end endcap taper (same as build_sdf) ────────────────────
        if leaf_taper and len(nodes) >= 2:
            try:
                from reconstruct_mesh import sample_spline_coeffs
                leaf_spts = sample_spline_coeffs(list(splines[-1]), 64)
                if leaf_spts is not None and len(leaf_spts) > 0:
                    leaf_mean_r = float(np.linalg.norm(
                        leaf_spts - leaf_spts.mean(0), axis=1).mean())
                else:
                    leaf_mean_r = 0.0
            except Exception:
                leaf_mean_r = 0.0
            taper_min_r = float(params.get("leaf_taper_min_r", 0.018))
            taper_r = min(taper_min_r, leaf_mean_r * 0.4)
            if taper_r >= 0.010 and leaf_mean_r > taper_r * 1.1:
                scale = taper_r / max(leaf_mean_r, 1e-9)
                leaf_ctr = nodes[-1].astype(np.float64)
                reduced = splines[-1].astype(np.float64).copy()
                cx = reduced[0:8] - leaf_ctr[0]
                cy = reduced[8:16] - leaf_ctr[1]
                cz = reduced[16:24] - leaf_ctr[2]
                reduced[0:8] = leaf_ctr[0] + cx * scale
                reduced[8:16] = leaf_ctr[1] + cy * scale
                reduced[16:24] = leaf_ctr[2] + cz * scale
                splines = np.vstack([splines[:-1], reduced[np.newaxis].astype(np.float32)])

        tck = create_3d_spline(nodes, centerline_smooth)
        if tck is None:
            continue
        sampled = sample_spline(tck, centerline_samples)
        t_values = np.linspace(0, 1, sampled.shape[0])

        if legacy_variant == "robust":
            robust_kwargs = {}
            if "tck" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(tck=tck, sampled_spline=sampled, t_values=t_values)
            if "centerline_t_mode" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(centerline_t_mode=centerline_t_mode)
            if "fallback_radius" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(fallback_radius=params.get("robust_fallback_radius", 0.02))
            if "min_radius" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(min_radius=params.get("robust_min_radius", 0.005))
            if "radius_cap" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(radius_cap=params.get("radius_cap", None))
            if "sanity_percentile" in inspect.signature(vessel3_robust).parameters:
                robust_kwargs.update(
                    sanity_percentile=params.get("robust_sanity_percentile", 95),
                    sanity_threshold=params.get("robust_sanity_threshold", None),
                )
            vessels.append(vessel3_robust(sampled, nodes, splines, **robust_kwargs))
        else:
            legacy_kwargs = {}
            if "tck" in inspect.signature(vessel3).parameters:
                legacy_kwargs.update(tck=tck, sampled_spline=sampled, t_values=t_values)
            if "centerline_t_mode" in inspect.signature(vessel3).parameters:
                legacy_kwargs.update(centerline_t_mode=centerline_t_mode)
            vessels.append(vessel3(sampled, nodes, splines, **legacy_kwargs))

    if not vessels:
        return None
    if len(vessels) == 1:
        return vessels[0]

    su_k = params.get("smooth_union_k")
    try:
        su_k = float(su_k) if su_k is not None else 0.0
    except Exception:
        su_k = 0.0

    out = vessels[0]
    for v in vessels[1:]:
        if su_k > 0 and smooth_union_fn is not None:
            out = smooth_union_fn(out, v, k=su_k)
        else:
            out = union(out, v)
    return out


def build_sweep_ellipse_sdf(tree, k, params):
    branches = get_branches(tree, k)
    if not branches:
        return None

    radius_percentile = float(params.get("radius_percentile", 90))
    radius_cap = params.get("radius_cap", None)
    center_mode = params.get("center_mode", "node")
    spline_samples = int(params.get("spline_samples", 50))
    fallback_radius = float(params.get("sweep_fallback_radius", params.get("fallback_radius", 0.02)))
    min_radius = float(params.get("sweep_min_radius", 0.0))
    smooth_mode = params.get("sweep_radius_smooth", "pchip")
    smooth_s = float(params.get("sweep_spline_s", 0.0))
    sweep_samples = int(params.get("sweep_samples", 400))
    sweep_centerline_mode = params.get("sweep_centerline_mode", "spline")
    sweep_centerline_smooth = float(params.get("sweep_centerline_smooth", params.get("centerline_smooth", 0.0)))
    sweep_min_segment_length = float(params.get("sweep_min_segment_length", 0.0))
    sweep_radius_sanity_factor = params.get("sweep_radius_sanity_factor", None)
    branch_smooth_k = params.get("sweep_branch_smooth_k", 0.0)
    use_pca = bool(params.get("sweep_ellipse_use_pca", True))
    try:
        branch_smooth_k = float(branch_smooth_k) if branch_smooth_k is not None else 0.0
    except Exception:
        branch_smooth_k = 0.0

    _, ellipse_fn, sweep_union, _ = get_sweep_primitives_and_union(params)

    vessels = []
    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        if nodes.shape[0] < 2:
            continue
        # drop zero-length steps
        if nodes.shape[0] > 2:
            seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
            keep = np.concatenate([[True], seg > 1e-6])
            nodes = nodes[keep]
            splines = splines[keep]
        if nodes.shape[0] < 2:
            continue

        t_nodes = _branch_arc_t(nodes)

        a_list = []
        b_list = []
        axis1_list = []
        axis2_list = []
        last_a = None
        last_b = None
        for i, (coeffs, node) in enumerate(zip(splines, nodes)):
            ring = sample_spline_coeffs(coeffs, spline_samples)
            if ring is None or not np.all(np.isfinite(ring)):
                a = last_a if last_a is not None else fallback_radius
                b = last_b if last_b is not None else fallback_radius
                axis1 = N[i]
                axis2 = B[i]
            else:
                center = node
                if center_mode == "centroid":
                    center = np.mean(ring, axis=0)
                a, b, axis1, axis2 = _ellipse_axes_from_ring(
                    ring, center, N[i], B[i], radius_percentile, use_pca=use_pca
                )
            if radius_cap is not None:
                try:
                    cap = float(radius_cap)
                    a = min(a, cap)
                    b = min(b, cap)
                except Exception:
                    pass
            if min_radius > 0:
                a = max(a, min_radius)
                b = max(b, min_radius)
            a_list.append(float(a))
            b_list.append(float(b))
            axis1_list.append(axis1)
            axis2_list.append(axis2)
            last_a, last_b = float(a), float(b)

        if len(a_list) < 2:
            continue

        if sweep_radius_sanity_factor is not None:
            try:
                factor = float(sweep_radius_sanity_factor)
            except Exception:
                factor = None
            if factor is not None and factor > 0:
                a_med = float(np.nanmedian(a_list))
                b_med = float(np.nanmedian(b_list))
                if np.isfinite(a_med) and a_med > 0:
                    a_cap = a_med * factor
                    a_list = [min(a, a_cap) if np.isfinite(a) else a for a in a_list]
                if np.isfinite(b_med) and b_med > 0:
                    b_cap = b_med * factor
                    b_list = [min(b, b_cap) if np.isfinite(b) else b for b in b_list]

        # smooth a(t), b(t)
        t_u, nodes_u, a_u = _dedupe_t_points(t_nodes, nodes, a_list)
        t_u, _, b_u = _dedupe_t_points(t_nodes, nodes, b_list)
        t_axis1, axis1_u = _dedupe_t_vectors(t_nodes, axis1_list)
        t_axis2, axis2_u = _dedupe_t_vectors(t_nodes, axis2_list)
        axis1_u = _map_vectors_to_t_linear(t_u, t_axis1, axis1_u)
        axis2_u = _map_vectors_to_t_linear(t_u, t_axis2, axis2_u)
        axis1_u, axis2_u = _orthonormalize_axes(axis1_u, axis2_u)
        if len(t_u) < 2:
            continue

        a_fn = _smooth_radii(t_u, a_u, smooth_mode, smooth_s)
        b_fn = _smooth_radii(t_u, b_u, smooth_mode, smooth_s)

        samples = max(2, sweep_samples)
        t_samples = np.linspace(0.0, 1.0, samples)

        centers = _sample_centerline(nodes_u, t_u, t_samples, sweep_centerline_smooth, sweep_centerline_mode)

        if a_fn is None:
            a_samples = np.interp(t_samples, t_u, a_u)
        else:
            a_samples = a_fn(t_samples)
        if b_fn is None:
            b_samples = np.interp(t_samples, t_u, b_u)
        else:
            b_samples = b_fn(t_samples)
        a_samples = np.asarray(a_samples, dtype=np.float64)
        b_samples = np.asarray(b_samples, dtype=np.float64)
        if radius_cap is not None:
            try:
                cap = float(radius_cap)
                a_samples = np.minimum(a_samples, cap)
                b_samples = np.minimum(b_samples, cap)
            except Exception:
                pass
        if min_radius > 0:
            a_samples = np.maximum(a_samples, min_radius)
            b_samples = np.maximum(b_samples, min_radius)
        a_samples = np.where(np.isfinite(a_samples), a_samples, fallback_radius)
        b_samples = np.where(np.isfinite(b_samples), b_samples, fallback_radius)

        # orientation per segment from nearest node
        seg_idx = np.searchsorted(t_u, t_samples[:-1], side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(axis1_u) - 1)

        segments = []
        for j in range(len(centers) - 1):
            p0 = centers[j]
            p1 = centers[j + 1]
            a0 = float(a_samples[j])
            b0 = float(b_samples[j])
            a1 = float(a_samples[j + 1])
            b1 = float(b_samples[j + 1])
            if a0 <= 0 and b0 <= 0 and a1 <= 0 and b1 <= 0:
                continue
            if sweep_min_segment_length > 0:
                if np.linalg.norm(p1 - p0) < sweep_min_segment_length:
                    continue
            axis1 = axis1_u[seg_idx[j]]
            axis2 = axis2_u[seg_idx[j]]
            axis1, axis2 = _project_axes_to_segment_pair(axis1, axis2, p1 - p0)
            segments.append(ellipse_fn(p0, p1, a0, b0, a1, b1, axis1, axis2))

        if segments:
            if branch_smooth_k > 0:
                branch_sdf = _reduce_balanced(
                    segments, lambda a, b: sweep_union(a, b, k=branch_smooth_k)
                )
            else:
                branch_sdf = union(*segments)
            if branch_sdf is not None:
                vessels.append(branch_sdf)

    if not vessels:
        return None

    k = params.get("smooth_union_k")
    try:
        k = float(k) if k is not None else 0.0
    except Exception:
        k = 0.0
    if k > 0:
        return _reduce_balanced(vessels, lambda a, b: sweep_union(a, b, k=k))
    return union(*vessels)


def build_sweep_fast_sdf(tree, k, params, ellipse=False):
    branches = get_branches(tree, k)
    if not branches:
        return None

    # sweep params
    radius_mode = params.get("radius_mode", "median")
    radius_percentile = float(params.get("radius_percentile", 90))
    radius_cap = params.get("radius_cap", None)
    center_mode = params.get("center_mode", "node")
    spline_samples = int(params.get("spline_samples", 50))
    fallback_radius = float(params.get("sweep_fallback_radius", params.get("fallback_radius", 0.02)))
    min_radius = float(params.get("sweep_min_radius", 0.0))
    sweep_samples = int(params.get("sweep_samples", 400))
    sweep_centerline_mode = params.get("sweep_centerline_mode", "spline")
    sweep_centerline_smooth = float(params.get("sweep_centerline_smooth", params.get("centerline_smooth", 0.0)))
    sweep_min_segment_length = float(params.get("sweep_min_segment_length", 0.0))
    sweep_radius_sanity_factor = params.get("sweep_radius_sanity_factor", None)
    smooth_mode = params.get("sweep_radius_smooth", "pchip")
    smooth_s = float(params.get("sweep_spline_s", 0.0))
    sweep_sdf_clip = params.get("sweep_sdf_clip", None)
    sweep_ellipse_sdf_iters = int(params.get("sweep_ellipse_sdf_iters", 3))
    debug_ellipse_fast = bool(params.get("debug_ellipse_fast", False))
    use_pca = bool(params.get("sweep_ellipse_use_pca", True))
    seg_batch = int(params.get("sweep_segment_batch", 256))
    _, _, _, sdf3_backend = get_sweep_primitives_and_union(params)
    if sdf3_backend is None:
        raise RuntimeError("Backend missing sdf3 decorator.")

    # collect segment data across all branches
    P0 = []
    P1 = []
    R0 = []
    R1 = []
    A0 = []
    B0 = []
    A1 = []
    B1 = []
    N0 = []
    B0v = []

    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        if nodes.shape[0] < 2:
            continue

        # drop zero-length steps
        if nodes.shape[0] > 2:
            seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
            keep = np.concatenate([[True], seg > 1e-6])
            nodes = nodes[keep]
            splines = splines[keep]
        if nodes.shape[0] < 2:
            continue

        t_nodes = _branch_arc_t(nodes)
        T, N, B = _parallel_transport_frames(nodes)

        if ellipse:
            a_list = []
            b_list = []
            axis1_list = []
            axis2_list = []
            last_a = None
            last_b = None
            for i, (coeffs, node) in enumerate(zip(splines, nodes)):
                tvec = _node_tangent(nodes, i)
                n0 = _perpendicular_unit(tvec)
                b0 = np.cross(tvec, n0)
                b0 = b0 / (np.linalg.norm(b0) + 1e-12)
                ring = sample_spline_coeffs(coeffs, spline_samples)
                if ring is None or not np.all(np.isfinite(ring)):
                    a = last_a if last_a is not None else fallback_radius
                    b = last_b if last_b is not None else fallback_radius
                    axis1 = n0
                    axis2 = b0
                elif np.linalg.norm(ring.max(0) - ring.min(0)) < 1e-4:
                    # Degenerate ring (all points collapsed to same location)
                    a = last_a if last_a is not None else fallback_radius
                    b = last_b if last_b is not None else fallback_radius
                    axis1 = n0
                    axis2 = b0
                else:
                    center = node
                    if center_mode == "centroid":
                        center = np.mean(ring, axis=0)
                    a, b, axis1, axis2 = _ellipse_axes_from_ring(
                        ring, center, n0, b0, radius_percentile, use_pca=use_pca
                    )
                if radius_cap is not None:
                    try:
                        cap = float(radius_cap)
                        a = min(a, cap)
                        b = min(b, cap)
                    except Exception:
                        pass
                if min_radius > 0:
                    a = max(a, min_radius)
                    b = max(b, min_radius)
                a_list.append(float(a))
                b_list.append(float(b))
                axis1_list.append(axis1)
                axis2_list.append(axis2)
                last_a, last_b = float(a), float(b)

            if len(a_list) < 2:
                continue

            if sweep_radius_sanity_factor is not None:
                try:
                    factor = float(sweep_radius_sanity_factor)
                except Exception:
                    factor = None
                if factor is not None and factor > 0:
                    a_med = float(np.nanmedian(a_list))
                    b_med = float(np.nanmedian(b_list))
                    if np.isfinite(a_med) and a_med > 0:
                        a_cap = a_med * factor
                        a_list = [min(a, a_cap) if np.isfinite(a) else a for a in a_list]
                    if np.isfinite(b_med) and b_med > 0:
                        b_cap = b_med * factor
                        b_list = [min(b, b_cap) if np.isfinite(b) else b for b in b_list]

            t_u, nodes_u, a_u = _dedupe_t_points(t_nodes, nodes, a_list)
            t_u, _, b_u = _dedupe_t_points(t_nodes, nodes, b_list)
            t_axis1, axis1_u = _dedupe_t_vectors(t_nodes, axis1_list)
            t_axis2, axis2_u = _dedupe_t_vectors(t_nodes, axis2_list)
            axis1_u = _map_vectors_to_t_linear(t_u, t_axis1, axis1_u)
            axis2_u = _map_vectors_to_t_linear(t_u, t_axis2, axis2_u)
            axis1_u, axis2_u = _orthonormalize_axes(axis1_u, axis2_u)
            if len(t_u) < 2:
                continue

            a_fn = _smooth_radii(t_u, a_u, smooth_mode, smooth_s)
            b_fn = _smooth_radii(t_u, b_u, smooth_mode, smooth_s)

            t_samples = np.linspace(0.0, 1.0, max(2, sweep_samples))
            centers = _sample_centerline(nodes_u, t_u, t_samples, sweep_centerline_smooth, sweep_centerline_mode)

            a_samples = np.interp(t_samples, t_u, a_u) if a_fn is None else a_fn(t_samples)
            b_samples = np.interp(t_samples, t_u, b_u) if b_fn is None else b_fn(t_samples)
            a_samples = np.asarray(a_samples, dtype=np.float64)
            b_samples = np.asarray(b_samples, dtype=np.float64)
            if radius_cap is not None:
                try:
                    cap = float(radius_cap)
                    a_samples = np.minimum(a_samples, cap)
                    b_samples = np.minimum(b_samples, cap)
                except Exception:
                    pass
            if min_radius > 0:
                a_samples = np.maximum(a_samples, min_radius)
                b_samples = np.maximum(b_samples, min_radius)
            a_samples = np.where(np.isfinite(a_samples), a_samples, fallback_radius)
            b_samples = np.where(np.isfinite(b_samples), b_samples, fallback_radius)

            seg_idx = np.searchsorted(t_u, t_samples[:-1], side="right") - 1
            seg_idx = np.clip(seg_idx, 0, len(axis1_u) - 1)
            for j in range(len(centers) - 1):
                if sweep_min_segment_length > 0:
                    if np.linalg.norm(centers[j + 1] - centers[j]) < sweep_min_segment_length:
                        continue
                axis1 = axis1_u[seg_idx[j]]
                axis2 = axis2_u[seg_idx[j]]
                if not (np.all(np.isfinite(axis1)) and np.all(np.isfinite(axis2))):
                    continue
                P0.append(centers[j])
                P1.append(centers[j + 1])
                A0.append(float(a_samples[j]))
                B0.append(float(b_samples[j]))
                A1.append(float(a_samples[j + 1]))
                B1.append(float(b_samples[j + 1]))
                N0.append(axis1)
                B0v.append(axis2)

        else:
            radii = []
            last_radius = None
            for coeffs, node in zip(splines, nodes):
                points = sample_spline_coeffs(coeffs, spline_samples)
                if points is None or not np.all(np.isfinite(points)):
                    radius = last_radius if last_radius is not None else fallback_radius
                    radii.append(float(radius))
                    last_radius = float(radius)
                    continue

                # Detect degenerate ring (all points collapsed to same location)
                ring_spread = np.linalg.norm(points.max(0) - points.min(0))
                if ring_spread < 1e-4:
                    radius = last_radius if last_radius is not None else fallback_radius
                    radii.append(float(radius))
                    last_radius = float(radius)
                    continue

                center = node
                if center_mode == "centroid":
                    center = np.mean(points, axis=0)
                radius = _compute_radius_for_node(points, center, radius_mode, radius_percentile)
                if radius is None or not np.isfinite(radius) or radius < 1e-6:
                    radius = last_radius if last_radius is not None else fallback_radius
                if radius_cap is not None:
                    try:
                        radius = min(radius, float(radius_cap))
                    except Exception:
                        pass
                if min_radius > 0:
                    radius = max(radius, min_radius)
                radii.append(float(radius))
                last_radius = float(radius)

            if sweep_radius_sanity_factor is not None:
                try:
                    factor = float(sweep_radius_sanity_factor)
                except Exception:
                    factor = None
                if factor is not None and factor > 0 and len(radii) > 0:
                    r_med = float(np.nanmedian(radii))
                    if np.isfinite(r_med) and r_med > 0:
                        cap_val = r_med * factor
                        radii = [min(r, cap_val) if np.isfinite(r) else r for r in radii]

            t_u, nodes_u, r_u = _dedupe_t_points(t_nodes, nodes, radii)
            if len(t_u) < 2:
                continue
            r_fn = _smooth_radii(t_u, r_u, smooth_mode, smooth_s)
            t_samples = np.linspace(0.0, 1.0, max(2, sweep_samples))
            centers = _sample_centerline(nodes_u, t_u, t_samples, sweep_centerline_smooth, sweep_centerline_mode)
            r_samples = np.interp(t_samples, t_u, r_u) if r_fn is None else r_fn(t_samples)
            r_samples = np.asarray(r_samples, dtype=np.float64)
            if radius_cap is not None:
                try:
                    r_samples = np.minimum(r_samples, float(radius_cap))
                except Exception:
                    pass
            if min_radius > 0:
                r_samples = np.maximum(r_samples, min_radius)

            for j in range(len(centers) - 1):
                if sweep_min_segment_length > 0:
                    if np.linalg.norm(centers[j + 1] - centers[j]) < sweep_min_segment_length:
                        continue
                P0.append(centers[j])
                P1.append(centers[j + 1])
                R0.append(float(r_samples[j]))
                R1.append(float(r_samples[j + 1]))

    if not P0:
        return None

    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    seg_v = P1 - P0
    seg_vv = np.sum(seg_v * seg_v, axis=1)
    seg_vv = np.where(seg_vv < 1e-12, 1e-12, seg_vv)

    if ellipse:
        A0 = np.asarray(A0, dtype=np.float64)
        B0 = np.asarray(B0, dtype=np.float64)
        A1 = np.asarray(A1, dtype=np.float64)
        B1 = np.asarray(B1, dtype=np.float64)
        N0 = np.asarray(N0, dtype=np.float64)
        B0v = np.asarray(B0v, dtype=np.float64)
        N0, B0v = _project_axes_to_segments(N0, B0v, seg_v)
        if debug_ellipse_fast:
            a_all = np.concatenate([A0, A1]) if A0.size and A1.size else A0
            b_all = np.concatenate([B0, B1]) if B0.size and B1.size else B0
            centers_all = np.vstack([P0, P1]) if len(P0) and len(P1) else P0
            try:
                print("ellipse_fast: a_samples min/max", np.nanmin(a_all), np.nanmax(a_all))
                print("ellipse_fast: b_samples min/max", np.nanmin(b_all), np.nanmax(b_all))
                print("ellipse_fast: centers min/max", np.nanmin(centers_all, axis=0), np.nanmax(centers_all, axis=0))
                print("ellipse_fast: seg_v min/max", np.nanmin(seg_v, axis=0), np.nanmax(seg_v, axis=0))
            except Exception:
                pass
    else:
        R0 = np.asarray(R0, dtype=np.float64)
        R1 = np.asarray(R1, dtype=np.float64)

    def _fast_sweep():
        def f(P):
            if P.ndim == 1:
                P = P[np.newaxis, :]
            Np = P.shape[0]
            best = np.full((Np,), np.inf, dtype=np.float64)
            for i0 in range(0, len(P0), seg_batch):
                i1 = min(i0 + seg_batch, len(P0))
                p0 = P0[i0:i1]
                v = seg_v[i0:i1]
                vv = seg_vv[i0:i1]
                pa = P[:, None, :] - p0[None, :, :]
                s = np.sum(pa * v[None, :, :], axis=2) / vv[None, :]
                s = np.clip(s, 0.0, 1.0)
                c = p0[None, :, :] + s[:, :, None] * v[None, :, :]
                d = P[:, None, :] - c

                if ellipse:
                    a0 = A0[i0:i1][None, :]
                    b0 = B0[i0:i1][None, :]
                    a1 = A1[i0:i1][None, :]
                    b1 = B1[i0:i1][None, :]
                    n = N0[i0:i1][None, :, :]
                    bb = B0v[i0:i1][None, :, :]
                    a = (1.0 - s) * a0 + s * a1
                    b = (1.0 - s) * b0 + s * b1
                    eps = 1e-6
                    a = np.maximum(a, eps)
                    b = np.maximum(b, eps)
                    a = np.where(np.isfinite(a), a, fallback_radius)
                    b = np.where(np.isfinite(b), b, fallback_radius)
                    x = np.sum(d * n, axis=2)
                    y = np.sum(d * bb, axis=2)
                    q = (x / a) ** 2 + (y / b) ** 2
                    sdf = (q - 1.0) * np.minimum(a, b)
                else:
                    r0 = R0[i0:i1][None, :]
                    r1 = R1[i0:i1][None, :]
                    r = (1.0 - s) * r0 + s * r1
                    dist = np.linalg.norm(d, axis=2)
                    sdf = dist - r
                best = np.minimum(best, np.min(sdf, axis=1))
            best = np.where(np.isfinite(best), best, 1e6)
            if sweep_sdf_clip is not None:
                try:
                    clip = float(sweep_sdf_clip)
                    if clip > 0:
                        best = np.clip(best, -clip, clip)
                except Exception:
                    pass
            return best.astype(np.float32)

        return f

    return sdf3_backend(_fast_sweep)()


def compute_centerline_bounds(tree, k, pad_ratio=0.1, pad_abs=None):
    branches = get_branches(tree, k)
    if not branches:
        return None
    nodes = np.vstack([b[:, :3] for b in branches])
    mins = np.nanmin(nodes, axis=0)
    maxs = np.nanmax(nodes, axis=0)
    size = maxs - mins
    if pad_abs is not None:
        pad = np.array([float(pad_abs)] * 3, dtype=np.float32)
    else:
        pad = size * float(pad_ratio)
    return (mins - pad, maxs + pad)


def parse_int(value, default):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return default


def _smooth_sdf_mesh(path, iterations, relaxation=0.5, verbose=False):
    """Apply Taubin (non-shrinking) smoothing to a mesh file in-place."""
    import trimesh
    mesh = trimesh.load(path, force='mesh', process=False)
    if verbose:
        print(f"Smoothing: {mesh.vertices.shape[0]} verts, {iterations} iters (Taubin λ={relaxation})")
    trimesh.smoothing.filter_taubin(mesh, lamb=relaxation, nu=-0.53, iterations=iterations)
    # Re-export in same format
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        mesh.export(path, file_type="obj")
    elif ext == ".ply":
        mesh.export(path, file_type="ply")
    elif ext == ".stl":
        mesh.export(path, file_type="stl")
    else:
        mesh.export(path)
    if verbose:
        print(f"Smoothing done: {mesh.vertices.shape[0]} verts")


def _keep_largest_components(path, min_face_ratio=0.01, verbose=False):
    """Keep only connected components with at least min_face_ratio of total faces."""
    import trimesh
    mesh = trimesh.load(path, force='mesh', process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        if verbose:
            print(f"Component filter: single component, no change")
        return
    total_faces = mesh.faces.shape[0]
    threshold = max(1, int(total_faces * min_face_ratio))
    kept = [p for p in parts if p.faces.shape[0] >= threshold]
    if not kept:
        kept = [max(parts, key=lambda p: p.faces.shape[0])]
    if len(kept) == len(parts):
        if verbose:
            print(f"Component filter: all {len(parts)} components above threshold")
        return
    combined = trimesh.util.concatenate(kept)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        combined.export(path, file_type="obj")
    elif ext == ".ply":
        combined.export(path, file_type="ply")
    elif ext == ".stl":
        combined.export(path, file_type="stl")
    else:
        combined.export(path)
    if verbose:
        print(f"Component filter: kept {len(kept)}/{len(parts)} components "
              f"({combined.faces.shape[0]}/{total_faces} faces)")


def _endpoint_radius(node_row):
    """Compute the mean cross-section radius for one tree node (1-D feature row)."""
    pts = sample_spline_coeffs(node_row[3:], 64)
    if pts is not None and np.all(np.isfinite(pts)):
        spread = np.linalg.norm(pts.max(0) - pts.min(0))
        if spread > 1e-4:
            centroid = pts.mean(0)
            return float(np.linalg.norm(pts - centroid, axis=1).mean())
    return 0.02  # fallback


def _get_tree_endpoints(tree, k):
    """Return [(position, outward_tangent, local_radius), ...] for root and all leaf endpoints."""
    branches = get_branches(tree, k)
    if not branches:
        return []

    endpoints = []

    # Root endpoint (first node of first branch, tangent pointing away from tree)
    root_pos = branches[0][0, :3].copy()
    if branches[0].shape[0] > 1:
        tangent = root_pos - branches[0][1, :3]
    else:
        tangent = np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(tangent)
    if norm > 1e-12:
        tangent = tangent / norm
    else:
        tangent = np.array([0.0, 0.0, 1.0])
    root_r = _endpoint_radius(branches[0][0])
    endpoints.append((root_pos, tangent, root_r))

    # Leaf endpoints (last node of each branch, tangent pointing outward)
    seen = set()
    for br in branches:
        leaf_pos = br[-1, :3].copy()
        key = tuple(np.round(leaf_pos, 5))
        if key in seen:
            continue
        seen.add(key)

        if br.shape[0] > 1:
            tangent = br[-1, :3] - br[-2, :3]
        else:
            tangent = np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(tangent)
        if norm > 1e-12:
            tangent = tangent / norm
        else:
            tangent = np.array([0.0, 0.0, 1.0])
        leaf_r = _endpoint_radius(br[-1])
        endpoints.append((leaf_pos, tangent, leaf_r))

    return endpoints


def _clip_mesh_endpoints(path, endpoints, clip_offset=0.0,
                         clip_radius_factor=1.5, clip_cone_cos=0.5,
                         verbose=False):
    """Remove hemispherical caps at vessel endpoints using *localised* face removal.

    For each endpoint we identify faces whose centroid is:
      1. On the outward (cap) side of the cutting plane, AND
      2. Within ``clip_radius_factor * local_radius`` of the endpoint, AND
      3. Predominantly in the outward direction (within a cone of
         half-angle ``arccos(clip_cone_cos)``).

    Conditions (2) and (3) prevent the removal from accidentally
    severing connections to neighbouring branches at bifurcations.

    Parameters
    ----------
    clip_offset : float
        Distance to push the cutting plane outward along the tangent
        before testing.  0.0 = cut at the node position.
    clip_radius_factor : float
        Multiplier on the local vessel radius that defines the ball
        around the endpoint inside which faces may be removed (default 1.5).
    clip_cone_cos : float
        Cosine of the half-angle of the directional cone.  Only faces
        whose centroid-to-endpoint direction has ``cos(angle) > clip_cone_cos``
        with the outward tangent are removed (default 0.5 ≈ 60° cone).
    """
    import trimesh

    mesh = trimesh.load(path, force='mesh', process=False)
    n_before = mesh.faces.shape[0]

    verts = mesh.vertices
    faces = mesh.faces
    centroids = verts[faces].mean(axis=1)  # (F, 3)

    faces_to_remove = np.zeros(len(faces), dtype=bool)
    n_clipped = 0

    for pos, outward, local_r in endpoints:
        plane_origin = pos + clip_offset * outward
        clip_radius = clip_radius_factor * max(local_r, 0.005)

        diff = centroids - plane_origin  # (F, 3)
        proj = diff @ outward             # projection along outward direction
        dist = np.linalg.norm(diff, axis=1)  # distance from endpoint

        # Cap faces: on the outward side of the plane AND within the ball
        cap_mask = (proj > 0) & (dist < clip_radius)

        # Cone constraint: only faces predominantly in the outward direction
        if clip_cone_cos > 0:
            cos_angle = proj / np.maximum(dist, 1e-12)
            cap_mask &= (cos_angle > clip_cone_cos)

        if cap_mask.any():
            faces_to_remove |= cap_mask
            n_clipped += 1

    n_removed = int(faces_to_remove.sum())
    if n_removed == 0:
        if verbose:
            print("Endpoint clipping: nothing to clip")
        return
    if n_removed >= n_before:
        if verbose:
            print("Endpoint clipping would remove entire mesh — skipping")
        return

    mesh.update_faces(~faces_to_remove)
    mesh.remove_unreferenced_vertices()

    ext = os.path.splitext(path)[1].lower()
    ft = ext.lstrip('.')
    if ft not in ('obj', 'ply', 'stl'):
        ft = 'obj'
    mesh.export(path, file_type=ft)
    if verbose:
        print(f"Endpoint clipping: {n_before} → {mesh.faces.shape[0]} faces "
              f"(removed {n_removed}, {n_clipped}/{len(endpoints)} endpoints clipped)")


def process_file(path, output_dir, params):
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    output_ext = params.get("output_ext", ".stl")
    overwrite = bool(params.get("overwrite", False))
    centerline_samples = int(params.get("centerline_samples", 100))
    centerline_smooth = float(params.get("centerline_smooth", 0.0))
    centerline_t_mode = params.get("centerline_t_mode", "optimize")
    samples_coarse = parse_int(params.get("samples_coarse"), 32)
    samples_fine = parse_int(params.get("samples_fine"), None)
    _step_raw = params.get("sdf_step")
    sdf_step = float(_step_raw) if _step_raw is not None else None
    smooth_iterations = int(params.get("smooth_iterations", 0))
    smooth_relaxation = float(params.get("smooth_relaxation", 0.5))
    sparse = bool(params.get("sparse", True))
    use_bounds = bool(params.get("use_bounds", True))
    sdf_verbose = bool(params.get("sdf_verbose", True))
    bounds_mode = params.get("bounds_mode", "auto")  # auto | centerline
    bounds_pad_ratio = params.get("bounds_pad_ratio", 0.1)
    bounds_pad_abs = params.get("bounds_pad_abs")
    save_centerline = bool(params.get("save_centerline", False))
    centerline_out_dir = params.get("centerline_output_dir") or output_dir
    centerline_suffix = params.get("centerline_suffix", "_centerline.npy")
    centerline_format = params.get("centerline_format", "npy").lower()
    centerline_tube_radius = float(params.get("centerline_tube_radius", 0.0))
    centerline_tube_sides = int(params.get("centerline_tube_sides", 8))
    centerline_mode = params.get("centerline_mode", "branches")
    centerline_edge_samples = int(params.get("centerline_edge_samples", 2))

    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    recon_mode = params.get("recon_mode", "legacy")
    if recon_mode not in {"legacy", "legacy_branch", "legacy_branch_local", "stable", "sdf_offset", "loft", "loft_v2", "loft_ring", "sweep", "sweep_ellipse", "sweep_fast", "sweep_ellipse_fast"}:
        recon_mode = "legacy"

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, base + output_ext)
    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path

    os.makedirs(output_dir, exist_ok=True)

    # Optional: save resampled centerlines
    if save_centerline:
        centerline_points = []
        centerline_branches = []
        if centerline_mode == "edges":
            edges = get_edges(tree)
            samples = max(2, centerline_edge_samples)
            for e_idx, (parent, child) in enumerate(edges):
                p0 = np.array([parent.data["x"], parent.data["y"], parent.data["z"]], dtype=np.float32)
                p1 = np.array([child.data["x"], child.data["y"], child.data["z"]], dtype=np.float32)
                t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
                sampled = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
                branch_col = np.full((sampled.shape[0], 1), e_idx, dtype=np.int32)
                arr = np.hstack((sampled.astype(np.float32), branch_col))
                centerline_points.append(arr)
                centerline_branches.append(sampled.astype(np.float32))
        else:
            branches = get_branches(tree, k)
            for b_idx, branch in enumerate(branches):
                nodes = branch[:, :3]
                tck = create_3d_spline(nodes, centerline_smooth)
                if tck is None:
                    continue
                sampled = sample_spline(tck, centerline_samples)
                if sampled is None or len(sampled) == 0:
                    continue
                branch_col = np.full((sampled.shape[0], 1), b_idx, dtype=np.int32)
                arr = np.hstack((sampled.astype(np.float32), branch_col))
                centerline_points.append(arr)
                centerline_branches.append(sampled.astype(np.float32))
        if centerline_points:
            os.makedirs(centerline_out_dir, exist_ok=True)
            if centerline_format in {"stl", "obj", "vtp"}:
                try:
                    import vtk
                except Exception as exc:
                    raise RuntimeError("VTK is required for centerline_format=stl/obj/vtp.") from exc
                append = vtk.vtkAppendPolyData()
                for b_idx, pts in enumerate(centerline_branches):
                    if pts.shape[0] < 2:
                        continue
                    vtk_pts = vtk.vtkPoints()
                    for p in pts:
                        vtk_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                    polyline = vtk.vtkPolyLine()
                    polyline.GetPointIds().SetNumberOfIds(pts.shape[0])
                    for i in range(pts.shape[0]):
                        polyline.GetPointIds().SetId(i, i)
                    cells = vtk.vtkCellArray()
                    cells.InsertNextCell(polyline)
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(vtk_pts)
                    polydata.SetLines(cells)
                    append.AddInputData(polydata)
                append.Update()
                cl_poly = append.GetOutput()

                # STL/OBJ require surface triangles; convert lines -> tube if requested
                if centerline_format in {"stl", "obj"}:
                    tube = vtk.vtkTubeFilter()
                    tube.SetInputData(cl_poly)
                    tube.SetNumberOfSides(max(3, centerline_tube_sides))
                    if centerline_tube_radius > 0:
                        tube.SetRadius(centerline_tube_radius)
                    else:
                        tube.SetRadius(0.001)
                    tube.CappingOn()
                    tube.Update()
                    cl_poly = tube.GetOutput()

                if centerline_format == "stl":
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".stl") else ".stl"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkSTLWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
                elif centerline_format == "obj":
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".obj") else ".obj"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkOBJWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
                else:
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".vtp") else ".vtp"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
            else:
                cl_path = os.path.join(centerline_out_dir, base + centerline_suffix)
                np.save(cl_path, np.vstack(centerline_points))

    if recon_mode in ("loft", "loft_v2"):
        try:
            import vtk
        except Exception as exc:
            raise RuntimeError("VTK is required for loft reconstruction.") from exc
        # For loft_v2, set branch mode automatically
        if recon_mode == "loft_v2":
            params = dict(params)  # copy to avoid mutating caller
            params["loft_branch_mode"] = "loft_v2"
        mesh = build_loft_mesh(tree, k, params)
        if mesh is None:
            return "skip", None
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".stl":
            writer = vtk.vtkSTLWriter()
        elif ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        elif ext == ".ply":
            writer = vtk.vtkPLYWriter()
        elif ext == ".obj":
            writer = vtk.vtkOBJWriter()
        else:
            raise RuntimeError(f"Unsupported loft output extension: {ext}")
        writer.SetFileName(out_path)
        writer.SetInputData(mesh)
        writer.Write()
    elif recon_mode == "loft_ring":
        try:
            import vtk
        except Exception as exc:
            raise RuntimeError("VTK is required for loft_ring reconstruction.") from exc
        mesh = build_loft_ring_mesh(tree, k, params)
        if mesh is None:
            return "skip", None
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".stl":
            writer = vtk.vtkSTLWriter()
        elif ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        elif ext == ".ply":
            writer = vtk.vtkPLYWriter()
        elif ext == ".obj":
            writer = vtk.vtkOBJWriter()
        else:
            raise RuntimeError(f"Unsupported loft_ring output extension: {ext}")
        writer.SetFileName(out_path)
        writer.SetInputData(mesh)
        writer.Write()
    else:
        if recon_mode == "sweep":
            sdf_obj = build_sweep_sdf(tree, k, params)
        elif recon_mode == "sweep_ellipse":
            sdf_obj = build_sweep_ellipse_sdf(tree, k, params)
        elif recon_mode == "sweep_fast":
            sdf_obj = build_sweep_fast_sdf(tree, k, params, ellipse=False)
        elif recon_mode == "sweep_ellipse_fast":
            sdf_obj = build_sweep_fast_sdf(tree, k, params, ellipse=True)
        else:
            sdf_obj = build_sdf(tree, k, centerline_samples, centerline_smooth, params)
        if sdf_obj is None:
            return "skip", None
        bounds = None
        if bounds_mode == "centerline":
            bounds = compute_centerline_bounds(tree, k, pad_ratio=bounds_pad_ratio, pad_abs=bounds_pad_abs)

        # Build kwargs for sdf generate/save
        sdf_kw = dict(sparse=sparse, verbose=sdf_verbose)
        if bounds is not None:
            sdf_kw["bounds"] = bounds
        if sdf_step is not None:
            sdf_kw["step"] = sdf_step
            sdf_kw["samples"] = None  # step overrides samples

        # Route to GPU core when sdf_variant==cuda, else use sdf_obj.save() (core.py)
        _backend_core = get_backend(params).get("core", None)
        def _save(path, **kw):
            if _backend_core is not None and hasattr(_backend_core, "save"):
                return _backend_core.save(path, sdf_obj, **kw)
            return sdf_obj.save(path, **kw)

        try:
            if sdf_step is not None:
                # Single pass with explicit step
                if sdf_verbose:
                    print(f"SDF pass: step={sdf_step}")
                _save(out_path, **sdf_kw)
            else:
                if sdf_verbose:
                    print("SDF pass: coarse")
                sdf_kw_coarse = {**sdf_kw, "samples": samples_coarse}
                bounds = _save(out_path, **sdf_kw_coarse)
                if samples_fine:
                    sdf_kw_fine = {**sdf_kw, "samples": samples_fine, "bounds": bounds}
                    if sdf_verbose:
                        print("SDF pass: fine")
                    _save(out_path, **sdf_kw_fine)
        except Exception as exc:
            print(f"SDF generation failed: {exc}")
            return "skip", None

        # Optional Taubin smoothing on output mesh
        if smooth_iterations > 0 and os.path.exists(out_path):
            try:
                _smooth_sdf_mesh(out_path, smooth_iterations, smooth_relaxation, sdf_verbose)
            except Exception as exc:
                print(f"Smoothing failed (keeping raw mesh): {exc}")

        # Optional: remove small disconnected components
        min_component_ratio = params.get("min_component_ratio")
        if min_component_ratio is not None and os.path.exists(out_path):
            try:
                _keep_largest_components(out_path, float(min_component_ratio), sdf_verbose)
            except Exception as exc:
                print(f"Component filter failed (keeping mesh as-is): {exc}")

        # Optional: clip hemispherical caps at branch endpoints to create open ends
        clip_endpoints = bool(params.get("clip_endpoints", False))
        if clip_endpoints and os.path.exists(out_path):
            try:
                clip_offset = float(params.get("clip_endpoint_offset", 0.002))
                clip_radius_factor = float(params.get("clip_radius_factor", 1.5))
                clip_cone_cos = float(params.get("clip_cone_cos", 0.5))
                endpoints = _get_tree_endpoints(tree, k)
                _clip_mesh_endpoints(out_path, endpoints, clip_offset,
                                     clip_radius_factor, clip_cone_cos,
                                     sdf_verbose)
            except Exception as exc:
                print(f"Endpoint clipping failed (keeping mesh as-is): {exc}")

    return "ok", out_path


def preview_mesh(mesh_path, params):
    if pv is None:
        print("preview skipped: pyvista not installed")
        return
    color = params.get("preview_color", "red")
    off_screen = bool(params.get("preview_offscreen", True))
    image_dir = params.get("preview_image_dir")
    plotter = pv.Plotter(off_screen=off_screen)
    mesh = pv.read(mesh_path)
    plotter.add_mesh(mesh, color=color, show_edges=False)
    plotter.add_axes()
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(
            image_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".png"
        )
        plotter.screenshot(image_path)
        plotter.close()
        print(f"preview saved {image_path}")
    else:
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Reconstruct mesh from tree splines via SDF.")
    parser.add_argument("--config", default="reconstruct_mesh_config.yaml", help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Input .npy file or directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for meshes")
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    input_path = args.input or paths.get("input")
    output_dir = args.output_dir or paths.get("output_dir")
    pattern = params.get("pattern", "*.npy")
    max_files = params.get("max_files")
    preview = bool(params.get("preview", False)) 
    preview_max = params.get("preview_max")
    file_progress = bool(params.get("file_progress", True))

    if not input_path or not output_dir:
        raise SystemExit("Error: --input and --output-dir are required (or provide them in config).")

    files = iter_inputs(input_path, pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    total = len(files)
    written = 0
    bar = sdf_progress.Bar(total, enabled=file_progress) if file_progress else None
    for idx, path in enumerate(files, start=1):
        status, out_path = process_file(path, output_dir, params)
        if status == "ok":
            written += 1
            print(f"[{idx}/{total}] saved {out_path}")
            if preview:
                if preview_max is None or written <= int(preview_max):
                    preview_mesh(out_path, params)
        else:
            print(f"[{idx}/{total}] skipped {os.path.basename(path)}")
        if bar:
            bar.increment(1)

    if bar:
        bar.done()
    print(f"done: {written}/{total} written")


if __name__ == "__main__":
    main()
