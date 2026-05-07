"""
RMF-based vessel mesh reconstruction from centerline cross-sections.

Algorithm overview
------------------
1.  Tree traversal via get_segments() → non-overlapping branch segments
2.  Per-segment: extract node centers + B-spline rings, then:
    a. Compute Rotation-Minimizing Frames (Bishop / parallel transport) along the centerline
    b. Canonically align each ring to the RMF frame: shift so the point
       most aligned with N is at index 0  → no drift, globally consistent
    c. Optionally densify between nodes (cubic spline interpolation of ring shapes)
    d. Stitch consecutive rings into a quad-strip tube mesh
3.  At each bifurcation: split junction ring along the bisector plane, connect
    each half-arc to the respective child tube, close the seam with fan triangles
4.  Combine all segment meshes into a single output mesh

Reference frame
---------------
Rotation-Minimizing Frames (Bishop frames) were described in:
  Wang et al., "Computation of Rotation Minimizing Frames", ACM Trans. Graph. 2008
  Bergou et al., "Discrete Elastic Rods", SIGGRAPH 2008

The key property vs Frenet-Serret: no singularities at inflection points,
and the frame rotates by the minimum amount necessary to keep up with the
tangent direction. This avoids the "twisting" artefacts common in naive
lofting.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import splev, splprep


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp(v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to *v*."""
    v = _unit(v)
    if abs(v[0]) < 0.9:
        w = np.array([1.0, 0.0, 0.0])
    else:
        w = np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, w))


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector *v* around *axis* (unit) by *angle* (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


# ─── Rotation-Minimizing Frames (Bishop / parallel-transport) ──────────────

def compute_rmf(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rotation-minimizing frames along a polyline.

    Parameters
    ----------
    points : (N, 3) array of 3-D points

    Returns
    -------
    T : (N, 3)  unit tangent vectors
    N : (N, 3)  normal (parallel-transported)
    B : (N, 3)  binormal = T × N
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)

    # ---- tangents (central differences at interior) ----------------------
    T = np.empty((n, 3))
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0] if n > 1 else np.array([0., 0., 1.])
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        T[i] = _unit(t)

    # ---- initial frame ---------------------------------------------------
    N = np.empty_like(T)
    B = np.empty_like(T)
    N[0] = _perp(T[0])
    B[0] = _unit(np.cross(T[0], N[0]))

    # ---- parallel transport (Bishop 1975, Wang et al. 2008) --------------
    for i in range(1, n):
        t0, t1 = T[i - 1], T[i]
        axis = np.cross(t0, t1)
        sin_a = np.linalg.norm(axis)
        if sin_a < 1e-10:
            # Nearly parallel tangents: keep frame as-is
            N[i] = N[i - 1] - np.dot(N[i - 1], t1) * t1
        else:
            axis /= sin_a
            cos_a = np.clip(np.dot(t0, t1), -1.0, 1.0)
            angle = np.arctan2(sin_a, cos_a)
            N[i] = _rodrigues(N[i - 1], axis, angle)
            N[i] -= np.dot(N[i], t1) * t1  # project out tangent component
        N[i] = _unit(N[i])
        B[i] = _unit(np.cross(t1, N[i]))

    return T, N, B


# ─── Cross-section ring sampling ──────────────────────────────────────────────

def sample_cross_section(coeffs, n_pts: int = 32) -> Optional[np.ndarray]:
    """
    Sample a periodic cubic B-spline cross-section stored as a flat coefficient vector.

    The encoding (from reconstruct_mesh.py / sample_spline_coeffs):
        coeffs[0:8]   = x B-spline control points
        coeffs[8:16]  = y control points
        coeffs[16:24] = z control points
        coeffs[24:36] = knot vector (12 knots)

    Returns (n_pts, 3) or None if coefficients are degenerate.
    """
    coeffs = list(coeffs)
    if len(coeffs) < 36:
        return None
    t = np.array(coeffs[24:36], dtype=np.float64)
    if not np.all(np.isfinite(t)):
        return None
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8:(i + 1) * 8], dtype=np.float64) for i in range(3)]
    if not all(np.all(np.isfinite(ci)) for ci in c):
        return None
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_pts, endpoint=False)
    try:
        x, y, z = splev(u, tck)
        return np.column_stack([x, y, z])
    except Exception:
        return None


# ─── Ring alignment helpers ──────────────────────────────────────────────────

def _fine_align_consecutive(ring_prev: np.ndarray,
                             ring_curr: np.ndarray) -> np.ndarray:
    """
    Cyclically shift *ring_curr* to minimize summed vertex-to-vertex L2 distance
    to *ring_prev*.  Also tries the reversed ring to catch winding flips.
    """
    n = len(ring_curr)
    best_shift, best_cost, best_flip = 0, float('inf'), False
    ring_rev = ring_curr[::-1].copy()
    for flip, candidate in [(False, ring_curr), (True, ring_rev)]:
        for s in range(n):
            rolled = np.roll(candidate, s, axis=0)
            d = np.linalg.norm(rolled - ring_prev, axis=1).sum()
            if d < best_cost:
                best_cost = d
                best_shift = s
                best_flip = flip
    base = ring_rev if best_flip else ring_curr
    return np.roll(base, best_shift, axis=0)


# ─── Ring densification ───────────────────────────────────────────────────────

def densify_rings(centers: np.ndarray, rings: List[np.ndarray],
                  target_spacing: float = 0.003,
                  max_rings: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Densely re-sample a segment using a two-pass strategy:

    (a) Smooth cubic B-spline through the centerline points — the tube follows
        the actual vessel curve instead of piecewise straight lines, eliminating
        the faceted / blocky look between sparse nodes.

    (b) Linear blend of cross-section shapes (in the local frame of each node)
        — avoids the radius overshoot that higher-order shape interpolation
        produces at nodes with very different cross-section sizes.

    Returns densified (centers, rings).
    """
    k = len(centers)
    if k < 2:
        return centers, rings

    n_pts = rings[0].shape[0]

    # ── arc-length parameter for input nodes ─────────────────────────────
    dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    dists[dists < 1e-8] = 1e-8
    arc   = np.concatenate([[0.0], np.cumsum(dists)])
    total = arc[-1]

    n_out = min(max_rings, max(k, int(total / target_spacing) + 1))
    u_in  = arc / total            # normalized [0, 1]
    u_out = np.linspace(0.0, 1.0, n_out)

    # ── (a) smooth centerline via cubic parametric spline ─────────────────
    # s=0 → interpolating spline (passes through all nodes); a tiny s is used
    # only when many nodes are present to tolerate near-duplicate points.
    s = 0.0 if k <= 6 else min(k * 1e-7, 1e-5)
    try:
        tck_c, _ = splprep(centers.T.tolist(), u=u_in, s=s, k=min(3, k - 1))
        centers_out = np.column_stack(splev(u_out, tck_c))   # (n_out, 3)
    except Exception:
        # Fallback to linear interpolation if spline fitting fails
        centers_out = np.column_stack([
            np.interp(u_out, u_in, centers[:, d]) for d in range(3)
        ])

    # ── (b) linear blend of ring shapes in local frame ────────────────────
    ring_arr = np.array(rings)                  # (k, n_pts, 3)
    local    = ring_arr - centers[:, None, :]   # (k, n_pts, 3)  — remove translation

    local_out = np.zeros((n_out, n_pts, 3))
    for j in range(n_pts):
        for d in range(3):
            local_out[:, j, d] = np.interp(u_out, u_in, local[:, j, d])

    rings_out = [local_out[i] + centers_out[i] for i in range(n_out)]
    return centers_out, rings_out


# ─── Tube mesh ────────────────────────────────────────────────────────────────

def build_tube_mesh(rings: List[np.ndarray],
                    open_start: bool = False,
                    open_end: bool = False,
                    flip_normals: bool = False,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch a sequence of aligned rings into a quad-strip (triangulated) mesh.

    Parameters
    ----------
    rings       : list of (M, 3) arrays — consecutive cross-section rings,
                  already aligned so that vertex j corresponds across rings.
    open_start  : do not add an end-cap at the first ring.
    open_end    : do not add an end-cap at the last ring.
    flip_normals: reverse face winding (outward vs inward normal).

    Returns
    -------
    verts : (V, 3) float64
    faces : (F, 3) int32
    """
    n_rings = len(rings)
    if n_rings < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    n_pts = rings[0].shape[0]
    verts = np.vstack(rings)            # (n_rings * n_pts, 3)
    faces = []

    # Lateral quad strip
    for i in range(n_rings - 1):
        base_a = i * n_pts
        base_b = (i + 1) * n_pts
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            a0, a1 = base_a + j, base_a + j1
            b0, b1 = base_b + j, base_b + j1
            if flip_normals:
                faces += [[a0, b0, b1], [a0, b1, a1]]
            else:
                faces += [[a0, b0, a1], [b0, b1, a1]]

    # Start cap (fan triangulation)
    if not open_start:
        c0 = rings[0].mean(0)
        ci = len(verts)
        verts = np.vstack([verts, c0[None, :]])
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            if flip_normals:
                faces.append([ci, j, j1])
            else:
                faces.append([ci, j1, j])

    # End cap (fan triangulation)
    if not open_end:
        c1 = rings[-1].mean(0)
        ci = len(verts)
        verts = np.vstack([verts, c1[None, :]])
        base = (n_rings - 1) * n_pts
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            if flip_normals:
                faces.append([ci, base + j1, base + j])
            else:
                faces.append([ci, base + j, base + j1])

    return verts, np.array(faces, dtype=np.int32)


# ─── Bifurcation patch ────────────────────────────────────────────────────────

def _ring_split(ring: np.ndarray, center: np.ndarray,
                dir1: np.ndarray, dir2: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Split a ring into two arcs based on which child branch each vertex
    faces.  The split is done along the bisector plane between dir1 and dir2.

    Parameters
    ----------
    ring    : (M, 3) ring vertices
    center  : (3,) ring centroid
    dir1    : (3,) unit direction toward child 1
    dir2    : (3,) unit direction toward child 2

    Returns
    -------
    half1   : vertex indices facing child 1
    half2   : vertex indices facing child 2
    seam_a  : first seam vertex index
    seam_b  : second seam vertex index
    """
    M = len(ring)
    # Split plane normal: bisector = d1 - d2 (points "toward d1 side")
    split_normal = _unit(dir1 - dir2)
    d = ring - center
    side = d @ split_normal     # (M,) positive = child1 side

    # Sort into two arcs, preserving cyclic order
    half1 = np.where(side >= 0)[0]
    half2 = np.where(side < 0)[0]

    # Find seam vertices: the boundary between the two arcs
    # → index with largest side value closest to 0 from each side
    def _seam_candidates(indices, sign=1):
        if len(indices) == 0:
            return 0
        vals = side[indices] * sign
        best = indices[np.argmin(vals)]
        return best

    seam_a = _seam_candidates(half1, sign=1)   # last positive, closest to 0
    seam_b = _seam_candidates(half2, sign=-1)  # first negative, closest to 0

    return half1, half2, seam_a, seam_b


def build_bifurcation_patch(
    junc_ring: np.ndarray,          # (M, 3) junction ring (parent's last ring)
    child1_ring: np.ndarray,        # (M, 3) child 1's first ring, already aligned
    child2_ring: np.ndarray,        # (M, 3) child 2's first ring, already aligned
    junc_center: np.ndarray,        # (3,)
    child1_center: np.ndarray,      # (3,)
    child2_center: np.ndarray,      # (3,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a triangulated bifurcation junction mesh.

    Strategy
    --------
    - Determine which half of junc_ring faces each child (via bisector plane).
    - Build a partial loft from each half of junc_ring to the closest points
      on the corresponding child's first ring.
    - Cap the two seam edges with fan triangles connecting the junction center.

    Returns (verts, faces) ready to be concatenated with the tube meshes.
    """
    M = len(junc_ring)
    d1 = _unit(child1_center - junc_center)
    d2 = _unit(child2_center - junc_center)

    half1_idx, half2_idx, seam_a, seam_b = _ring_split(
        junc_ring, junc_center, d1, d2)

    all_verts = [junc_ring, child1_ring, child2_ring]
    # offsets into combined vertex buffer
    off_junc = 0
    off_c1   = M
    off_c2   = 2 * M

    faces = []

    def _match_to_child(half_idx: np.ndarray, child_ring: np.ndarray,
                        child_off: int):
        """For each junc vertex in half_idx, find the closest child vertex."""
        for vi in half_idx:
            jv = junc_ring[vi]
            dists = np.linalg.norm(child_ring - jv, axis=1)
            ci = int(np.argmin(dists))
            ci_next = (ci + 1) % M
            vi_next = (vi + 1) % M if vi != seam_b else None

            if vi_next is not None and vi_next in half_idx:
                faces.append([off_junc + vi, child_off + ci, off_junc + vi_next])
                faces.append([child_off + ci, child_off + ci_next, off_junc + vi_next])

    _match_to_child(half1_idx, child1_ring, off_c1)
    _match_to_child(half2_idx, child2_ring, off_c2)

    # Seam cap: connect the two seam edges to the junction center
    jc_idx = 3 * M   # extra vertex: junction center
    all_verts.append(junc_center[None, :])
    faces.append([off_junc + seam_a, off_c1 + int(np.argmin(
        np.linalg.norm(child1_ring - junc_ring[seam_a], axis=1))), jc_idx])
    faces.append([off_junc + seam_b, off_c2 + int(np.argmin(
        np.linalg.norm(child2_ring - junc_ring[seam_b], axis=1))), jc_idx])

    verts = np.vstack(all_verts)
    return verts, np.array(faces, dtype=np.int32)


# ─── Per-segment tube builder ─────────────────────────────────────────────────

def build_segment_tube(
    nodes: np.ndarray,                # (N, 3) centerline node positions
    splines: np.ndarray,              # (N, 36) B-spline cross-section coefficients
    n_pts: int = 32,
    target_spacing: float = 0.003,
    use_densify: bool = True,
    max_rings: int = 800,
    open_start: bool = False,
    open_end: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a tube mesh for one branch segment using RMF alignment.

    Returns (verts, faces) or None if the segment is degenerate.
    """
    # ---- deduplicate coincident nodes ------------------------------------
    keep = [0]
    for i in range(1, len(nodes)):
        if np.linalg.norm(nodes[i] - nodes[keep[-1]]) > 1e-6:
            keep.append(i)
    if len(keep) < 2:
        return None
    nodes   = nodes[keep]
    splines = splines[keep]

    # ---- sample cross-section rings at each node -------------------------
    rings_raw: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i in range(len(nodes)):
        ring = sample_cross_section(splines[i], n_pts)
        if ring is None or not np.all(np.isfinite(ring)):
            # Fallback: tiny circle in the node's local plane
            ring = _fallback_ring(nodes[i], n_pts, radius=0.005)
        rings_raw.append(ring)
        valid_idx.append(i)

    if len(rings_raw) < 2:
        return None

    # ---- greedy sequential alignment (cyclic shift + flip check) --------
    # The rings are already in world space; we just need a consistent cyclic
    # ordering.  We do NOT use canonical_align with RMF here because the RMF
    # normal is not guaranteed to align with the B-spline parameterisation,
    # which would introduce artificial twist.
    rings_aligned: List[np.ndarray] = [rings_raw[0]]
    for i in range(1, len(rings_raw)):
        rings_aligned.append(
            _fine_align_consecutive(rings_aligned[-1], rings_raw[i])
        )

    # ---- densify between nodes for smoother geometry --------------------
    if use_densify and len(rings_aligned) >= 2:
        centers_d, rings_d = densify_rings(
            nodes, rings_aligned,
            target_spacing=target_spacing,
            max_rings=max_rings,
        )
        # Alignment is preserved by linear interpolation — no further shift needed
    else:
        rings_d = rings_aligned

    # ---- stitch into mesh -----------------------------------------------
    return build_tube_mesh(rings_d, open_start=open_start, open_end=open_end)


def _fallback_ring(center: np.ndarray, n_pts: int, radius: float = 0.005) -> np.ndarray:
    """Tiny circle as fallback when the B-spline coefficients are invalid."""
    perp1 = _perp(np.array([0., 0., 1.]))
    perp2 = _unit(np.cross(np.array([0., 0., 1.]), perp1))
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return center + radius * (np.cos(t)[:, None] * perp1 + np.sin(t)[:, None] * perp2)


# ─── Full tree reconstruction ─────────────────────────────────────────────────

def reconstruct_tree(
    tree,
    k: int,
    n_pts: int = 32,
    target_spacing: float = 0.003,
    use_densify: bool = True,
    max_rings_per_segment: int = 800,
    bifurcation_patches: bool = True,
    open_ends: bool = False,
) -> Optional["trimesh.Trimesh"]:
    """
    Reconstruct a watertight vessel mesh from a VesselGPT tree node.

    Parameters
    ----------
    tree                    : root node of the deserialized binary tree
    k                       : feature dimension (number of cols per node row)
    n_pts                   : number of vertices per cross-section ring
    target_spacing          : target arc-length spacing between densified rings
    use_densify             : whether to densify rings between nodes
    max_rings_per_segment   : safety cap on ring count per segment
    bifurcation_patches     : add explicit junction meshes at bifurcations
    open_ends               : leave branch ends open (no cap discs)

    Returns
    -------
    trimesh.Trimesh or None
    """
    import trimesh

    # ---- get non-overlapping segments (1-node overlap at junctions) ------
    # We use get_segments from the parent module to avoid re-implementing
    # tree traversal logic.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reconstruct_mesh import get_segments, _coerce_row

    segments = get_segments(tree, k)
    if not segments:
        return None

    all_meshes = []

    # ---- identify terminal ends (leaf + root) ----------------------------
    # Each segment has 1-node overlap at bifurcations.  A segment end is
    # terminal if its endpoint does NOT appear as the start of any other
    # segment.  We use node position (rounded) as the key.
    def _key(pt):
        return tuple(np.round(pt, 6))

    seg_starts = set(_key(seg[0, :3]) for seg in segments)

    for seg in segments:
        nodes   = seg[:, :3].astype(np.float64)
        splines = seg[:, 3:]

        # Is this segment's START the tree root? (no other segment ends here)
        seg_ends_elsewhere = any(
            _key(s[-1, :3]) == _key(nodes[0]) for s in segments if s is not seg
        )
        root_end = not seg_ends_elsewhere

        # Is this segment's END a leaf? (no other segment starts here)
        leaf_end = _key(nodes[-1]) not in seg_starts or \
                   _key(nodes[-1]) == _key(nodes[0])  # single-node degenerate

        result = build_segment_tube(
            nodes, splines,
            n_pts=n_pts,
            target_spacing=target_spacing,
            use_densify=use_densify,
            max_rings=max_rings_per_segment,
            open_start=not (root_end and not open_ends),
            open_end=not (leaf_end and not open_ends),
        )
        if result is None:
            continue
        verts, faces = result
        if len(faces) == 0:
            continue
        all_meshes.append(trimesh.Trimesh(
            vertices=verts, faces=faces, process=False))

    if not all_meshes:
        return None

    # ---- combine all segment meshes + bifurcation patches ---------------
    if len(all_meshes) == 1:
        combined = all_meshes[0]
    else:
        combined = trimesh.util.concatenate(all_meshes)

    # Minimal cleanup: merge vertices within numerical precision
    combined.merge_vertices()

    return combined


def _add_end_caps(mesh_list, tree, k, n_pts):
    """Add flat cap discs at root and leaf nodes (terminal ends)."""
    import trimesh
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reconstruct_mesh import get_segments, sample_spline_coeffs

    segments = get_segments(tree, k)
    if not segments:
        return

    for seg in segments:
        nodes   = seg[:, :3].astype(np.float64)
        splines = seg[:, 3:]

        for end_idx, is_open in [(0, True), (-1, True)]:
            # sample the ring at this end
            ring = sample_cross_section(splines[end_idx], n_pts)
            if ring is None:
                continue
            center = ring.mean(0)
            n = len(ring)
            verts = np.vstack([ring, center[None, :]])
            ci = n
            # outward cap: for end_idx=0 wind inward, for end_idx=-1 wind outward
            if end_idx == 0:
                faces = [[ci, (j + 1) % n, j] for j in range(n)]
            else:
                faces = [[ci, j, (j + 1) % n] for j in range(n)]
            mesh_list.append(trimesh.Trimesh(
                vertices=verts,
                faces=np.array(faces, dtype=np.int32),
                process=False,
            ))
