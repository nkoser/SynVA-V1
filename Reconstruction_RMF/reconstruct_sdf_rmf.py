"""
SDF + Marching-Cubes vessel reconstruction.

Wraps the proven `build_sdf()` pipeline from reconstruct_mesh.py and adds
GPU-accelerated marching cubes (core_cuda) + trimesh post-processing.

Pipeline
--------
1. Deserialize .npy → tree
2. build_sdf(tree, …)  [legacy-robust variant: smooth B-spline per branch,
   then vessel3_robust angle-bin radius tables]
3. core_cuda.generate(sdf_fn, step)  → triangle soup (GPU)
4. Deduplicate → trimesh → component filter → Taubin smooth → export

Usage
-----
    python Reconstruction_RMF/reconstruct_sdf_rmf.py \\
        --config Reconstruction_RMF/reconstruct_sdf_rmf_config.yaml

    python Reconstruction_RMF/reconstruct_sdf_rmf.py \\
        --input path/to/file.npy --output_dir /out/dir
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import yaml

# ── repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.interpolate import splev
from tree_functions import deserialize
from reconstruct_mesh import build_sdf, build_sdf_segments, compute_centerline_bounds, get_branches


# ── phantom-node sanitizer ───────────────────────────────────────────────────

def _is_phantom(r_vec):
    """Return True if the B-spline has a degenerate knot vector (all knots ≈ 1.0).

    Such nodes cause ``splev`` to produce a ring at the world origin [0,0,0]
    regardless of the control-point positions.  The check mirrors the logic in
    ``check_degenerate_splines.py`` (knot_mean ≈ 1.0 within tol=0.1).
    """
    if len(r_vec) < 36:
        return False
    knots = np.asarray(list(r_vec)[24:36], dtype=np.float64)
    return bool(np.abs(np.mean(knots) - 1.0) < 0.1)


def _ring_shape_metrics(r_vec, node_xyz) -> tuple:
    """Return (centroid_offset, span_ratio) for a ring B-spline.

    ``span_ratio`` = bbox_diagonal / mean_radius.  A circle gives ~2.83.
    A self-intersecting or loopy ring gives >> 4.  Returns (0, 0) if the
    ring is phantom or cannot be evaluated.
    """
    if len(r_vec) < 36 or _is_phantom(r_vec):
        return 0.0, 0.0
    try:
        t_raw = np.asarray(r_vec[24:36], dtype=np.float64)
        t = np.where(np.abs(t_raw - 1) < 0.01, 1.0, t_raw)
        c = [np.array(r_vec[i * 8:(i + 1) * 8], dtype=np.float64)
             for i in range(3)]
        x, y, z = splev(np.linspace(0, 1, 64), (t, c, 3))
        pts = np.column_stack([x, y, z])
        centroid = pts.mean(0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).mean())
        offset = float(np.linalg.norm(centroid - np.asarray(node_xyz)))
        span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        ratio = span / (radius + 1e-9)
        return offset, ratio
    except Exception:
        return 0.0, 0.0


def _ring_centroid_offset(r_vec, node_xyz) -> float:
    return _ring_shape_metrics(r_vec, node_xyz)[0]


def _recenter_ring(r_vec, node_xyz):
    """Return a copy of r_vec with control points shifted so the ring centroid
    is at ``node_xyz``.

    A displaced leaf ring creates a sinusoidal spike in the angle-bin radius
    table: on the side toward the centroid the effective radius is r+d, on
    the opposite side r-d.  Recentering removes this without changing the
    ring's actual cross-sectional shape or radius, so the vessel diameter at
    the endpoint is preserved.  Unlike ring *replacement* (which substitutes
    the parent's ring and may change the radius), recentering is always safe.
    """
    if len(r_vec) < 36 or _is_phantom(r_vec):
        return list(r_vec)
    try:
        t_raw = np.asarray(r_vec[24:36], dtype=np.float64)
        t = np.where(np.abs(t_raw - 1) < 0.01, 1.0, t_raw)
        c = [np.array(r_vec[i * 8:(i + 1) * 8], dtype=np.float64) for i in range(3)]
        x, y, z = splev(np.linspace(0, 1, 64), (t, c, 3))
        centroid = np.array([x.mean(), y.mean(), z.mean()])
        d = centroid - np.asarray(node_xyz, dtype=np.float64)
        new_r = list(r_vec)
        for i in range(8):        # ctrl_x
            new_r[i]      = float(r_vec[i])      - d[0]
        for i in range(8):        # ctrl_y
            new_r[8  + i] = float(r_vec[8  + i]) - d[1]
        for i in range(8):        # ctrl_z
            new_r[16 + i] = float(r_vec[16 + i]) - d[2]
        return new_r
    except Exception:
        return list(r_vec)


# Rings whose centroid is displaced by more than _SANITIZE_DISPLACEMENT OR
# whose bbox span > _SANITIZE_SPAN_RATIO * mean_radius (loopy/self-intersecting)
# are sanitized:
#   • Non-leaf displaced/loopy rings → replaced with parent ring.
#   • Leaf displaced rings → recentered (ctrl pts shifted, radius unchanged);
#     this avoids the endcap fin without the radius-mismatch risk of replacement.
_SANITIZE_DISPLACEMENT      = 0.05   # non-leaf nodes
_SANITIZE_DISPLACEMENT_LEAF = 0.015  # leaf (terminal) nodes — tighter threshold
                                      # because a displaced leaf ring creates a
                                      # directional spike → visible fin artifact.
_SANITIZE_SPAN_RATIO        = 3.2    # circle ≈ 2.83; loopy/spiraling rings > 3.2


def _sanitize_phantom_tree(tree, k: int) -> int:
    """Walk the tree BFS-style and sanitize phantom or badly-displaced
    B-spline coefficients.

    Actions taken:
      1. Phantom rings (knot mean ≈ 1.0) on non-leaf nodes → replace with
         parent ring.  Leaf phantoms are left alone (natural taper is OK).
      2. Non-leaf rings displaced > _SANITIZE_DISPLACEMENT (0.05) → replace.
      3. Leaf rings displaced > _SANITIZE_DISPLACEMENT_LEAF (0.015):
         → RECENTER (shift control points) rather than replace.  Keeps the
         ring's actual radius so no endcap size mismatch.
      4. Any ring with span_ratio > _SANITIZE_SPAN_RATIO (loopy) → replace.

    Returns the number of nodes that were repaired.
    """
    if tree is None:
        return 0
    repaired = 0
    from collections import deque
    r_key = "r"
    queue = deque()
    root_r = list(tree.data.get(r_key, []))
    queue.append((tree, root_r))
    while queue:
        node, parent_r = queue.popleft()
        node_r = list(node.data.get(r_key, []))
        is_leaf = (node.left is None and node.right is None)
        node_xyz = [node.data.get("x", 0.0), node.data.get("y", 0.0),
                    node.data.get("z", 0.0)]
        offset, span_ratio = _ring_shape_metrics(node_r, node_xyz)
        is_phantom_nonleaf = _is_phantom(node_r) and not is_leaf
        is_loopy           = span_ratio > _SANITIZE_SPAN_RATIO
        # Separate displacement thresholds for leaf vs non-leaf
        disp_thresh  = _SANITIZE_DISPLACEMENT_LEAF if is_leaf else _SANITIZE_DISPLACEMENT
        is_displaced = offset > disp_thresh

        if is_leaf and is_displaced and not is_loopy:
            # Leaf endpoint: RECENTER (preserve radius, eliminate fin)
            new_r = _recenter_ring(node_r, node_xyz)
            node.data[r_key] = new_r
            repaired += 1
            propagate_r = new_r
        elif (is_phantom_nonleaf or is_displaced or is_loopy) and not _is_phantom(parent_r):
            # Non-leaf or loopy: REPLACE with parent
            node.data[r_key] = list(parent_r)
            repaired += 1
            propagate_r = parent_r
        else:
            propagate_r = node_r
        if node.left  is not None: queue.append((node.left,  propagate_r))
        if node.right is not None: queue.append((node.right, propagate_r))
    return repaired


# ── displaced-ring recentering ────────────────────────────────────────────────

def _recenter_displaced_rings(tree, k: int) -> int:
    """No-op: displaced rings (both leaf and non-leaf) are now fully handled
    by ``_sanitize_phantom_tree``, which replaces them with the parent ring.
    Kept for API compatibility.
    """
    return 0


# ── backtracking-centerline straightener ──────────────────────────────────────

def _straighten_backtracking_centerlines(tree, k: int) -> int:
    """Detect and repair branches whose centerline path self-intersects.

    A branch truly loops when two non-adjacent nodes (separation > 2) are
    closer together than the local inter-node spacing.  U-shaped or helical
    vessels never satisfy this criterion because their paths don't revisit
    the same spatial location.

    Fix: for each self-intersecting node, replace its xyz with the midpoint
    of its immediate neighbours (one iteration).  Repeat until no more
    self-intersections or max 10 passes.

    Returns the number of nodes corrected.
    """
    if tree is None:
        return 0

    total_fixed = 0

    def dfs(node, path_nodes):
        nonlocal total_fixed
        path_nodes = path_nodes + [node]
        if node.left is None and node.right is None:
            pts = np.array([[n.data.get("x", 0.), n.data.get("y", 0.),
                              n.data.get("z", 0.)] for n in path_nodes])
            n_pts = len(pts)
            if n_pts < 5:
                return
            # Local inter-node spacing
            dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            min_spacing = np.percentile(dists, 25)  # robust to outliers
            threshold = min_spacing * 0.6

            for _pass in range(10):
                fixed_this_pass = False
                for i in range(1, n_pts - 3):
                    for j in range(i + 3, min(i + 12, n_pts - 1)):
                        if np.linalg.norm(pts[i] - pts[j]) < threshold:
                            # Node i is too close to non-adjacent node j → fix i
                            new_xyz = (pts[i - 1] + pts[i + 1]) / 2.0
                            path_nodes[i].data["x"] = float(new_xyz[0])
                            path_nodes[i].data["y"] = float(new_xyz[1])
                            path_nodes[i].data["z"] = float(new_xyz[2])
                            pts[i] = new_xyz
                            total_fixed += 1
                            fixed_this_pass = True
                if not fixed_this_pass:
                    break
        else:
            if node.left  is not None: dfs(node.left,  path_nodes)
            if node.right is not None: dfs(node.right, path_nodes)

    dfs(tree, [])
    return total_fixed


# ── bounds from actual ring extents ──────────────────────────────────────────

def _ring_aware_bounds(tree, k: int, pad_abs: float = 0.02):
    """Compute bounding box from the actual 3-D ring points at every node.

    Unlike ``compute_centerline_bounds`` (which only looks at node xyz), this
    samples each cross-section B-spline and takes the true geometric extent.
    A small absolute margin ``pad_abs`` is then added on all sides.

    For terminal nodes (branch start/end), the SDF endcap hemisphere can
    extend ~ring_radius beyond the ring contour.  We therefore add 6
    axis-aligned "endcap sphere" probe points around each terminal node so
    the MC grid always contains the full endcap surface.

    Phantom-ring filter: if a ring has near-zero radius (<0.003) AND its
    centroid is far from the node (>0.03), the B-spline coefficients are
    corrupted (Preprocessing bug).  Such nodes contribute only their node xyz
    to the bounds and get no endcap sphere — avoiding massive bounds inflation.
    """
    _PHANTOM_RADIUS = 0.003   # rings smaller than this …
    _PHANTOM_OFFSET = 0.030   # … whose centroid is this far from the node

    branches = get_branches(tree, k)
    all_pts = []
    for branch in branches:
        n = len(branch)
        for j, row in enumerate(branch):
            is_terminal = (j == 0 or j == n - 1)
            coeffs = list(row[3:3 + 36])
            node_xyz = row[:3].reshape(1, 3)
            if len(coeffs) < 36:
                all_pts.append(node_xyz)
                continue
            try:
                t = np.array(coeffs[24:])
                t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
                c = [np.array(coeffs[i * 8:(i + 1) * 8]) for i in range(3)]
                u = np.linspace(0, 1, 64)
                x, y, z = splev(u, (t, c, 3))
                ring_pts = np.column_stack([x, y, z])

                ring_radius = float(np.linalg.norm(
                    ring_pts - ring_pts.mean(0), axis=1).mean())
                ring_offset = float(np.linalg.norm(
                    ring_pts.mean(0) - node_xyz.flatten()))

                # Phantom rings: corrupted coefficients → use node xyz only
                if ring_radius < _PHANTOM_RADIUS and ring_offset > _PHANTOM_OFFSET:
                    all_pts.append(node_xyz)
                    continue

                all_pts.append(ring_pts)
                # For terminal nodes, also add an axis-aligned bounding sphere
                # of radius = ring_ext around the node so the SDF endcap
                # hemisphere (which extends ~ring_ext beyond the ring contour)
                # stays inside the MC grid.
                if is_terminal:
                    ring_ext = float(np.abs(ring_pts - node_xyz).max())
                    offsets = ring_ext * np.eye(3, dtype=np.float32)
                    sphere_pts = np.vstack([node_xyz + offsets,
                                            node_xyz - offsets])
                    all_pts.append(sphere_pts)
            except Exception:
                all_pts.append(node_xyz)
    if not all_pts:
        return compute_centerline_bounds(tree, k, pad_ratio=0.15)
    pts = np.vstack(all_pts)
    mn = pts.min(axis=0).astype(np.float32) - pad_abs
    mx = pts.max(axis=0).astype(np.float32) + pad_abs
    return (mn, mx)


# ── single-file reconstruction ────────────────────────────────────────────────

def reconstruct_file(npy_path: str,
                     output_dir: str,
                     params: dict) -> tuple[str, Optional[str]]:
    """
    Reconstruct one .npy file and save the result.
    Returns (status, output_path)  where status is 'ok' | 'skip' | 'error'.
    """
    import trimesh

    k    = int(params.get("k", 39))
    mode = params.get("mode", "pre_order_kcount")

    stem     = os.path.splitext(os.path.basename(npy_path))[0]
    ext      = params.get("output_ext", ".obj")
    out_path = os.path.join(output_dir, stem + ext)

    if not params.get("overwrite", True) and os.path.exists(out_path):
        return "skip", out_path

    # ── load tree ──────────────────────────────────────────────────────────
    try:
        data = np.load(npy_path)
        if data.ndim == 1:
            data = data.reshape((-1, k))
        tree = deserialize(list(data.flatten()), mode=mode, k=k)
    except Exception as e:
        return "error", f"load failed: {e}"

    # ── choose SDF + marching-cubes backend ────────────────────────────────
    use_cuda  = params.get("use_cuda", True)
    core_name = "sdf.core_cuda" if use_cuda else "sdf.core"
    try:
        import importlib
        core_mod = importlib.import_module(core_name)
    except Exception as e:
        return "error", f"SDF backend import failed: {e}"

    # ── build the SDF via build_sdf() (proven legacy-robust pipeline) ──────
    # Mirror the params used by reconstruct_mesh_legacy_cuda_gt.yaml exactly
    sdf_params = {
        "recon_mode":               params.get("recon_mode", "legacy"),
        "legacy_variant":           params.get("legacy_variant", "robust"),
        # sdf_variant controls which backend module get_backend() selects:
        #   "cuda" → sdf.d3_fast_cuda   "fast" → sdf.d3_fast   else → sdf.d3
        "sdf_variant":              "cuda" if use_cuda else "fast",
        # radius / robust settings — match legacy_cuda defaults
        "robust_fallback_radius":   float(params.get("fallback_radius", 0.01)),
        "robust_min_radius":        float(params.get("min_radius", 0.005)),
        "radius_cap":               params.get("radius_cap", 0.05),
        "robust_sanity_percentile": int(params.get("sanity_percentile", 90)),
        "robust_sanity_threshold":  params.get("sanity_threshold", 0.15),
        "smooth_union_k":           float(params.get("smooth_union_k", 0.005)),
        # leaf endcap taper — adds a small circular ring past each
        # leaf endpoint, tapering the endcap to a small sphere
        "leaf_taper":               bool(params.get("leaf_taper", False)),
        "leaf_taper_dist":          float(params.get("leaf_taper_dist", 0.0)),
        "leaf_taper_min_r":         float(params.get("leaf_taper_min_r", 0.018)),
        # centerline / spline quality — match legacy_cuda defaults
        "spline_samples":           int(params.get("spline_samples", 128)),
        "centerline_t_mode":        params.get("centerline_t_mode", "optimize"),
    }
    centerline_samples = int(params.get("centerline_samples", 1500))
    centerline_smooth  = float(params.get("centerline_smooth", 0.0))

    # ── sanitize phantom nodes (zero ctrl-pts → SDF blob at world origin) ──
    n_repaired = _sanitize_phantom_tree(tree, k)
    if n_repaired and params.get("verbose", False):
        print(f"           [sanitize] repaired {n_repaired} phantom nodes")

    # ── straighten backtracking centerlines (spiral/coil artifacts) ───────
    n_straightened = _straighten_backtracking_centerlines(tree, k)
    if n_straightened and params.get("verbose", False):
        print(f"           [straighten] fixed {n_straightened} backtracking nodes")

    # ── recenter misregistered rings (centroid displaced from node_xyz) ───
    n_recentered = _recenter_displaced_rings(tree, k)
    if n_recentered and params.get("verbose", False):
        print(f"           [recenter] recentered {n_recentered} displaced rings")

    try:
        use_segments = bool(params.get("use_segments", False))
        if use_segments:
            sdf_fn = build_sdf_segments(tree, k, centerline_samples,
                                        centerline_smooth, sdf_params)
        else:
            sdf_fn = build_sdf(tree, k, centerline_samples, centerline_smooth,
                               sdf_params)
    except Exception as e:
        return "error", f"build_sdf failed: {e}"

    if sdf_fn is None:
        return "error", "build_sdf returned None (no branches/segments)"

    # ── marching cubes ────────────────────────────────────────────────────
    step = float(params.get("step", 0.004))
    # Pre-compute bounds from node positions so tiny-coordinate trees don't
    # trip up _estimate_bounds (which starts coarse at ±1e9 and may miss them)
    bounds_pad_abs = float(params.get("bounds_pad_abs", 0.02))
    mc_bounds = _ring_aware_bounds(tree, k, pad_abs=bounds_pad_abs)
    try:
        points, bounds = core_mod.generate(sdf_fn, step=step, bounds=mc_bounds)
    except Exception as e:
        return "error", f"marching cubes failed: {e}"

    if not points:
        return "error", "empty mesh from marching cubes"

    # ── triangle soup → shared-vertex trimesh ─────────────────────────────
    try:
        pts_arr = np.array(points, dtype=np.float32)   # (N*3, 3)
        pts_u, inv = np.unique(pts_arr, axis=0, return_inverse=True)
        faces = inv.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=pts_u, faces=faces, process=True)
    except Exception as e:
        return "error", f"mesh conversion failed: {e}"

    # ── keep largest components ───────────────────────────────────────────
    min_ratio = float(params.get("min_component_ratio", 0.03))
    comps = mesh.split(only_watertight=False)
    if comps:
        largest = max(comps, key=lambda m: len(m.faces))
        threshold = int(len(largest.faces) * min_ratio)
        big = [c for c in comps if len(c.faces) >= threshold]
        if big:
            mesh = trimesh.util.concatenate(big)

    # ── Taubin smooth ─────────────────────────────────────────────────────
    n_smooth = int(params.get("smooth_iterations", 10))
    if n_smooth > 0:
        try:
            import trimesh.smoothing as _sm
            _sm.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=n_smooth)
        except Exception:
            pass

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    try:
        mesh.export(out_path)
    except Exception as e:
        return "error", f"export failed: {e}"

    if params.get("verbose"):
        wt = getattr(mesh, "is_watertight", False)
        print(f"  {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces,"
              f" watertight={wt}")

    return "ok", out_path


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dir  = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    params     = cfg.get("params", {})

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))

    print(f"SDF-RMF reconstruction: {len(files)} files → {output_dir}")
    ok = skip = err = 0
    t0_total = time.time()

    for i, fname in enumerate(files, 1):
        npy_path = os.path.join(input_dir, fname)
        t0 = time.time()
        status, result = reconstruct_file(npy_path, output_dir, params)
        elapsed = time.time() - t0

        sym = {"ok": "ok", "skip": "skip", "error": "ERR"}[status]
        print(f"  [{i:3d}/{len(files)}] {sym} ({elapsed:.1f}s)  {fname}")
        if status == "error":
            print(f"           → {result}")

        if status == "ok":     ok   += 1
        elif status == "skip": skip += 1
        else:                  err  += 1

    total = time.time() - t0_total
    print(f"\ndone: {ok} ok, {skip} skipped, {err} errors  ({total:.1f}s total)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SDF + Marching-Cubes vessel mesh reconstruction")
    parser.add_argument("--config",     help="YAML config file")
    parser.add_argument("--input",      help="Single .npy input file")
    parser.add_argument("--output_dir", help="Output directory (single-file mode)")
    args = parser.parse_args()

    if args.config:
        run_batch(args.config)
    elif args.input and args.output_dir:
        params = {"verbose": True, "overwrite": True}
        status, result = reconstruct_file(args.input, args.output_dir, params)
        print(f"status={status}  result={result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
