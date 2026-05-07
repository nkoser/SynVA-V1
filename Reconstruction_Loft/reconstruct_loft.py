"""
Reconstruction_Loft/reconstruct_loft.py
=======================================
Direct tube-lofting reconstruction from VesselGPT tree (.npy) files.

Instead of SDF + marching cubes (reconstruct_sdf_rmf.py), this module builds
triangle meshes by directly sweeping the stored cross-section B-splines along
the vessel centerline:

  1. **Global RMF walk**: parallel-transport Rotation-Minimizing Frames (RMF)
     through the entire tree in DFS order.  Every node gets a unique (u, v)
     frame; frame is continuous across bifurcations → no inter-segment twist.

  2. **Ring sampling**: each stored ring B-spline is sampled at ``n_angular``
     evenly-spaced angles in the node's RMF frame.  The ring is projected onto
     the tangent plane (off-plane scatter eliminated) before sampling.

  3. **Ring interpolation**: between each pair of adjacent stored rings, we
     insert ``n_interp`` linearly-blended intermediate rings, giving a smooth
     tube even for coarsely-placed nodes.

  4. **Segment tubes**: ``get_segments()`` produces non-overlapping tree
     segments (each internal node appears at exactly one segment's end and its
     children's segment starts).  Each segment is lofted into a quad-strip
     triangle mesh.

  5. **Leaf caps**: closed fan-triangulation at every tree leaf.

  6. **Vertex welding**: junction nodes are shared between parent and child
     segments.  After all segment tubes are built, matching ring vertices
     (same position, consistent angular alignment) are fused into a single
     vertex list so the final mesh is topologically manifold.

  7. **OBJ export**: standard Wavefront .obj output (vertices + triangulated
     faces).

Advantages over SDF approach
-----------------------------
* No marching-cubes noise or grid-alignment artefacts.
* Cross-section shapes are reproduced exactly (each tube cross-section is the
  stored B-spline ring, projected into the tangent plane).
* Deterministic topology — mesh resolution is fully controlled by ``n_angular``
  and ``n_interp``.
* Faster: no 3-D grid evaluation.

Limitations
-----------
* At a bifurcation, parent and children share a ring row.  The resulting inner
  "T-notch" is topologically closed but may look slightly angular.  Use a
  larger ``n_interp`` to soften this.
* No smooth-union blending at junctions (the SDF approach blends volumes).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import warnings as _warnings
from scipy.interpolate import splev

# ── repo root ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "Reconstruction_RMF"))

import yaml
from tree_functions import deserialize
from reconstruct_sdf_rmf import _sanitize_phantom_tree

# ── constants ────────────────────────────────────────────────────────────────
_K = 39           # default token width: 3 xyz + 36 ring (8+8+8+12)
_N_ANGULAR = 64   # angular samples per ring (mesh circumferential resolution)
_N_INTERP  = 3    # extra interpolated rings between each stored-ring pair
_WELD_TOL  = 1e-5 # vertex welding tolerance


# ══════════════════════════════════════════════════════════════════════════════
# Low-level math helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parallel_transport(v: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Rotate vector *v* from frame aligned with *t1* to frame aligned with *t2*
    using double-reflection (axis = cross(t1, t2))."""
    axis = np.cross(t1, t2)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(t1, t2))
    if sin_a < 1e-10:
        return v.copy() if cos_a > 0 else -v.copy()
    axis /= sin_a
    return v * cos_a + np.cross(axis, v) * sin_a + axis * float(np.dot(axis, v)) * (1.0 - cos_a)


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else fallback.copy()


def _perp_to(tang: np.ndarray) -> np.ndarray:
    """Return any unit vector perpendicular to *tang*."""
    tang = _safe_normalize(tang, np.array([0., 0., 1.]))
    u = np.array([1., 0., 0.])
    if abs(np.dot(u, tang)) > 0.9:
        u = np.array([0., 1., 0.])
    u = u - np.dot(u, tang) * tang
    return _safe_normalize(u, np.array([0., 1., 0.]))


# ══════════════════════════════════════════════════════════════════════════════
# Global RMF computation (parallel transport through the whole tree)
# ══════════════════════════════════════════════════════════════════════════════

NodeInfo = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (position_xyz, u_vec, v_vec, tangent_vec)

def compute_global_rmf(tree) -> Dict[int, NodeInfo]:
    """DFS through *tree*, parallel-transporting the RMF frame at each edge.

    Returns a dict mapping ``id(node)`` to ``(pos, u, v, tang)`` where
    u ⊥ v ⊥ tang are all unit vectors.  Frames are consistent across
    bifurcations: child segments inherit the frame at their shared junction.
    """
    frames: Dict[int, NodeInfo] = {}

    def _dfs(node, parent_pos: Optional[np.ndarray], u_prev: np.ndarray):
        if node is None:
            return
        pos = np.array([node.data["x"], node.data["y"], node.data["z"]], dtype=np.float64)

        if parent_pos is None:
            # Root: compute tangent toward first defined child
            child = node.left or node.right
            if child is not None:
                cp = np.array([child.data["x"], child.data["y"], child.data["z"]])
                tang = _safe_normalize(cp - pos, np.array([0., 0., 1.]))
            else:
                tang = np.array([0., 0., 1.])
            u = _perp_to(tang)
        else:
            tang = _safe_normalize(pos - parent_pos, np.array([0., 0., 1.]))
            # Parallel-transport u from parent→current tangent
            prev_tang = _safe_normalize(pos - parent_pos, tang)
            u = _parallel_transport(u_prev, prev_tang, tang)
            # Re-orthogonalise (numerical drift)
            u = u - np.dot(u, tang) * tang
            u = _safe_normalize(u, _perp_to(tang))

        v = np.cross(tang, u)
        v = _safe_normalize(v, _perp_to(tang))
        # store
        frames[id(node)] = (pos.copy(), u.copy(), v.copy(), tang.copy())
        # propagate to children — both inherit the same (u, v) at this node
        _dfs(node.left,  pos, u)
        _dfs(node.right, pos, u)

    root_u = _perp_to(np.array([0., 0., 1.]))
    _dfs(tree, None, root_u)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Ring → N angular samples in RMF frame
# ══════════════════════════════════════════════════════════════════════════════

_DENSE_SAMPLES = 512  # dense ring evaluation for angle-bin lookup


def _eval_ring_dense(coeffs: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate B-spline ring at ``_DENSE_SAMPLES`` points.  Returns None on failure."""
    if len(coeffs) < 36 or not np.all(np.isfinite(coeffs[:36])):
        return None
    t = np.asarray(coeffs[24:36], dtype=np.float64)
    t = np.where(np.abs(t - 1.0) < 0.01, 1.0, t)
    if np.ptp(t) < 1e-8 or np.any(np.diff(t) < -1e-8):
        return None
    c = [np.asarray(coeffs[i * 8:(i + 1) * 8], dtype=np.float64) for i in range(3)]
    try:
        x, y, z = splev(np.linspace(0., 1., _DENSE_SAMPLES, endpoint=False), (t, c, 3))
        pts = np.column_stack([x, y, z])
        return pts if np.all(np.isfinite(pts)) else None
    except Exception:
        return None


def sample_ring(coeffs: np.ndarray,
                center: np.ndarray,
                tangent: np.ndarray,
                u_ref: np.ndarray,
                v_ref: np.ndarray,
                n_angular: int) -> np.ndarray:
    """Sample a ring B-spline at *n_angular* evenly-spaced angles in the RMF frame.

    The ring is first projected onto the plane perpendicular to *tangent*
    (eliminating off-plane scatter due to slightly non-planar cross-sections).
    For each target angle θ_k = 2π·k/n_angular, the radius is interpolated from
    the dense angle-bin table.  Returns an (n_angular, 3) array of 3-D points.

    Falls back to a circle of radius ``fallback_r`` if the ring cannot be evaluated.
    """
    pts = _eval_ring_dense(np.asarray(coeffs, dtype=np.float64))

    if pts is not None:
        # ── project onto tangent plane ────────────────────────────────────────
        pa = pts - center
        axial = (pa @ tangent)[:, None] * tangent
        pa_perp = pa - axial  # in-plane displacement vectors

        x_proj = pa_perp @ u_ref   # (N,)
        y_proj = pa_perp @ v_ref   # (N,)
        angles_dense = np.arctan2(y_proj, x_proj)   # -π…π
        radii_dense  = np.sqrt(x_proj ** 2 + y_proj ** 2)

        if not np.any(radii_dense > 1e-6):
            pts = None  # degenerate ring → fall back
        else:
            target_angles = np.linspace(-np.pi, np.pi, n_angular, endpoint=False)
            ring_pts = np.empty((n_angular, 3), dtype=np.float64)
            for k, theta in enumerate(target_angles):
                diff = angles_dense - theta
                # Wrap to [-π, π]
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                idx  = int(np.argmin(diff ** 2))
                r    = float(radii_dense[idx])
                ring_pts[k] = (center
                               + r * np.cos(theta) * u_ref
                               + r * np.sin(theta) * v_ref)
            return ring_pts

    # ── fall back: small circle ───────────────────────────────────────────────
    fallback_r = 0.005
    target_angles = np.linspace(-np.pi, np.pi, n_angular, endpoint=False)
    ring_pts = np.empty((n_angular, 3), dtype=np.float64)
    for k, theta in enumerate(target_angles):
        ring_pts[k] = (center
                       + fallback_r * np.cos(theta) * u_ref
                       + fallback_r * np.sin(theta) * v_ref)
    return ring_pts


# ══════════════════════════════════════════════════════════════════════════════
# Radius table (for interpolation)
# ══════════════════════════════════════════════════════════════════════════════

def _ring_to_radtable(coeffs: np.ndarray,
                      center: np.ndarray,
                      tangent: np.ndarray,
                      u_ref: np.ndarray,
                      v_ref: np.ndarray,
                      n_angular: int) -> np.ndarray:
    """Return (n_angular,) array of radii (in-plane).  Used for interpolation."""
    pts = _eval_ring_dense(np.asarray(coeffs, dtype=np.float64))
    if pts is None:
        return np.full(n_angular, 0.005)
    pa = pts - center
    pa_perp = pa - (pa @ tangent)[:, None] * tangent
    x_proj = pa_perp @ u_ref
    y_proj = pa_perp @ v_ref
    angles_dense = np.arctan2(y_proj, x_proj)
    radii_dense  = np.sqrt(x_proj ** 2 + y_proj ** 2)
    target_angles = np.linspace(-np.pi, np.pi, n_angular, endpoint=False)
    table = np.empty(n_angular, dtype=np.float64)
    for k, theta in enumerate(target_angles):
        diff = (angles_dense - theta + np.pi) % (2.0 * np.pi) - np.pi
        table[k] = float(radii_dense[int(np.argmin(diff ** 2))])
    return table


def interpolate_ring(center_a: np.ndarray, radtab_a: np.ndarray,
                     center_b: np.ndarray, radtab_b: np.ndarray,
                     u_ref: np.ndarray, v_ref: np.ndarray,
                     alpha: float, n_angular: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly blend two rings at parameter *alpha* ∈ [0, 1].

    Returns ``(ring_pts, center)`` where ring_pts is (n_angular, 3).
    """
    center = (1.0 - alpha) * center_a + alpha * center_b
    radtab = (1.0 - alpha) * radtab_a + alpha * radtab_b
    target_angles = np.linspace(-np.pi, np.pi, n_angular, endpoint=False)
    ring_pts = np.empty((n_angular, 3), dtype=np.float64)
    for k, theta in enumerate(target_angles):
        r = float(radtab[k])
        ring_pts[k] = center + r * np.cos(theta) * u_ref + r * np.sin(theta) * v_ref
    return ring_pts, center


# ══════════════════════════════════════════════════════════════════════════════
# Mesh building helpers
# ══════════════════════════════════════════════════════════════════════════════

def _loft_strip(ring_a: np.ndarray, ring_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create a quad-strip (2 triangles per quad) between two N-point rings.

    Returns ``(verts, faces)`` where verts = (2*N, 3) and faces = (2*N, 3).
    Winding is outward-facing (CCW when looking from outside the tube).
    """
    n = len(ring_a)
    verts = np.vstack([ring_a, ring_b])   # indices 0…n-1 = ring_a, n…2n-1 = ring_b
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # quad: (a_i, b_i, b_j) + (a_i, b_j, a_j)  → outward normals
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
    return verts, np.array(faces, dtype=np.int32)


def _fan_cap(ring_pts: np.ndarray, cap_center: np.ndarray,
             inward: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fan triangulation: connect *ring_pts* to *cap_center*.

    *inward*=True for root caps (normals pointing inward = toward the inside),
    False for leaf caps (normals pointing outward = away from vessel).
    """
    n = len(ring_pts)
    c_idx = n          # cap center is the last vertex
    verts = np.vstack([ring_pts, cap_center[None, :]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        if inward:
            faces.append([c_idx, j, i])
        else:
            faces.append([c_idx, i, j])
    return verts, np.array(faces, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Segment → tube mesh
# ══════════════════════════════════════════════════════════════════════════════

def build_segment_tube(segment: np.ndarray,
                       frames: Dict[int, NodeInfo],
                       n_angular: int,
                       n_interp: int,
                       seg_nodes: list,
                       cap_start: bool = True,
                       cap_end: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Build a tube mesh for one segment.

    When *cap_start* and *cap_end* are both True the result is a closed
    (watertight) manifold.  For junction-connected segments, pass
    ``cap_start=False`` or ``cap_end=False`` to leave that end open; the caller
    is then responsible for closing the topology (e.g., via Y-patch or boolean
    union of an inner closing disk).

    Parameters
    ----------
    segment :
        (M, k) array where columns 0:3 are xyz and 3:k are ring coefficients.
    frames :
        Pre-computed global RMF frames, keyed by ``id(node)``.
    n_angular :
        Number of angular samples per ring cross-section.
    n_interp :
        Number of interpolated rings to insert between each pair of stored rings.
    seg_nodes :
        Ordered list of tree node objects corresponding to rows of *segment*.
    cap_start :
        If True, close the start of the tube with a fan cap.
    cap_end :
        If True, close the end of the tube with a fan cap.

    Returns
    -------
    verts : (V, 3) float64
    faces : (F, 3) int32
    """
    M     = len(segment)
    nodes = segment[:, :3].astype(np.float64)
    spls  = segment[:, 3:].astype(np.float64)

    # ── build per-stored-node ring pts and radius tables ──────────────────────
    # We also need u, v, tang at each stored node.
    # Frames keyed by id(node) from the DFS.
    stored_frames = []
    stored_rings  = []  # (n_angular, 3) sampled ring points
    stored_rtabs  = []  # (n_angular,) radius tables for interpolation

    for i, node in enumerate(seg_nodes):
        nid = id(node)
        if nid in frames:
            pos, u, v, tang = frames[nid]
        else:
            # Fallback: compute tangent from segment geometry
            pos = nodes[i]
            if i + 1 < M:
                tang = _safe_normalize(nodes[i + 1] - nodes[i], np.array([0., 0., 1.]))
            elif i > 0:
                tang = _safe_normalize(nodes[i] - nodes[i - 1], np.array([0., 0., 1.]))
            else:
                tang = np.array([0., 0., 1.])
            u = _perp_to(tang)
            v = np.cross(tang, u)

        ring_pts = sample_ring(spls[i], pos, tang, u, v, n_angular)
        rtab     = _ring_to_radtable(spls[i], pos, tang, u, v, n_angular)
        stored_frames.append((pos, u, v, tang))
        stored_rings.append(ring_pts)
        stored_rtabs.append(rtab)

    # ── build stack of rings (stored + interpolated) ──────────────────────────
    # ring_stack : list of (n_angular, 3) arrays
    ring_stack = []
    interp_alphas = [k / (n_interp + 1) for k in range(1, n_interp + 1)]

    for i in range(M):
        ring_stack.append(stored_rings[i])
        if i < M - 1:
            pos_a, u_a, v_a, tang_a = stored_frames[i]
            pos_b, u_b, v_b, tang_b = stored_frames[i + 1]
            # Use the u, v from ring A for all interpolated rings in this span
            # (consistent with the RMF at node A; no extra frame interpolation needed
            # for short spans because the frame barely rotates between adjacent nodes)
            rtab_a = stored_rtabs[i]
            rtab_b = stored_rtabs[i + 1]
            for alpha in interp_alphas:
                r_pts, _c = interpolate_ring(pos_a, rtab_a, pos_b, rtab_b, u_a, v_a, alpha, n_angular)
                ring_stack.append(r_pts)

    # ── loft strip between consecutive ring rows ──────────────────────────────
    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    vert_offset = 0

    first_row_indices = None  # vertex indices of the very first ring row
    last_row_indices  = None  # vertex indices of the very last ring row

    # Each ring row owns n_angular vertices.
    # ring_stack has M + (M-1)*n_interp entries = M*(n_interp+1) - n_interp rows.
    ring_start_indices: List[np.ndarray] = []

    for r_idx, ring_r in enumerate(ring_stack):
        v_idx = np.arange(vert_offset, vert_offset + n_angular, dtype=np.int32)
        ring_start_indices.append(v_idx)
        all_verts.append(ring_r)
        vert_offset += n_angular

    first_row_indices = ring_start_indices[0]
    last_row_indices  = ring_start_indices[-1]

    # Connect consecutive rings
    for r_idx in range(len(ring_stack) - 1):
        a_global = ring_start_indices[r_idx]
        b_global = ring_start_indices[r_idx + 1]
        for i in range(n_angular):
            j = (i + 1) % n_angular
            ai = int(a_global[i]); aj = int(a_global[j])
            bi = int(b_global[i]); bj = int(b_global[j])
            all_faces.append([ai, bi, bj])
            all_faces.append([ai, bj, aj])

    # ── caps ──────────────────────────────────────────────────────────────────
    # cap_eps offsets each cap slightly *inside* the tube so that caps from
    # adjacent segments at a bifurcation form a proper 3D overlap (not
    # co-planar disks).  A 3D overlapping solid is much cheaper for boolean
    # union than near-coincident 2D surfaces.

    if cap_start:
        first_pos  = stored_frames[0][0]
        first_tang = stored_frames[0][3]
        # Radius of the first ring, used to scale the cap offset
        first_ring = ring_stack[0]  # (n_angular, 3)
        first_r    = float(np.mean(np.linalg.norm(first_ring - first_pos, axis=1)))
        cap_eps0   = max(first_r * 0.05, 5e-4)
        start_cap_ctr = first_pos - first_tang * cap_eps0
        cap_center_idx = vert_offset
        all_verts.append(start_cap_ctr[None, :])
        vert_offset += 1
        for i in range(n_angular):
            j = (i + 1) % n_angular
            all_faces.append([cap_center_idx,
                               int(first_row_indices[j]),
                               int(first_row_indices[i])])

    if cap_end:
        last_tang  = stored_frames[-1][3]
        last_pos   = stored_frames[-1][0]
        last_ring  = ring_stack[-1]
        last_r     = float(np.mean(np.linalg.norm(last_ring - last_pos, axis=1)))
        cap_epsL   = max(last_r * 0.05, 5e-4)
        end_cap_ctr = last_pos + last_tang * cap_epsL
        cap_center_idx = vert_offset
        all_verts.append(end_cap_ctr[None, :])
        vert_offset += 1
        for i in range(n_angular):
            j = (i + 1) % n_angular
            all_faces.append([cap_center_idx,
                               int(last_row_indices[i]),
                               int(last_row_indices[j])])

    # ── assemble ─────────────────────────────────────────────────────────────
    verts = np.vstack(all_verts).astype(np.float64)
    faces = np.array(all_faces, dtype=np.int32)

    return verts, faces


# ══════════════════════════════════════════════════════════════════════════════
# Tree → lofted mesh (boolean union)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_tree_nodes(tree) -> Dict[int, object]:
    """Return dict id(node) → node for every node in the tree."""
    result = {}
    def _dfs(n):
        if n is None:
            return
        result[id(n)] = n
        _dfs(n.left)
        _dfs(n.right)
    _dfs(tree)
    return result


def _segment_node_list(tree,
                       min_step: float = 8e-4) -> List[List[object]]:
    """Return one ordered list of tree-node objects per segment.

    Mirrors ``get_segments()`` ordering exactly: junction nodes appear at the
    END of the parent segment AND at the START of each child segment (1-node
    overlap for continuity).

    Phantom-padding nodes (those with near-zero displacement from their parent)
    are removed BEFORE building the segment list.  In the ``relpos_v*`` datasets,
    trailing padding nodes all occupy the same position as the preceding real
    node (relative offset = 0), and must be excluded to avoid degenerate tubes.

    Parameters
    ----------
    min_step :
        Minimum Euclidean distance between consecutive nodes (in normalized
        coordinate units) to consider a node real.  Nodes closer than this to
        their parent are considered phantom padding and are dropped.
    """
    seg_nodes: List[List[object]] = []

    def _pos(n) -> np.ndarray:
        return np.array([n.data["x"], n.data["y"], n.data["z"]], dtype=np.float64)

    def _dfs(node, seg, parent_pos):
        if node is None:
            return
        pos = _pos(node)
        if parent_pos is not None and np.linalg.norm(pos - parent_pos) < min_step:
            # phantom / duplicate-position node — skip but recurse into children
            _dfs(node.left,  seg, pos)
            _dfs(node.right, seg, pos)
            return

        seg.append(node)
        children = [c for c in (node.left, node.right) if c is not None]
        if len(children) == 0:
            if len(seg) >= 1:
                seg_nodes.append(seg[:])
        elif len(children) == 1:
            _dfs(children[0], seg, pos)
        else:
            if len(seg) >= 1:
                seg_nodes.append(seg[:])
            for child in children:
                _dfs(child, [node], pos)   # child shares junction node at start
        seg.pop()

    _dfs(tree, [], None)
    return seg_nodes


def _nodes_to_segment(node_list: list, k: int) -> np.ndarray:
    """Convert a list of tree-node objects into a (M, k) float32 segment array.

    Columns 0:3 = xyz, 3:k = ring B-spline coefficients from ``node.data["r"]``.
    """
    needed = k - 3
    rows = []
    for node in node_list:
        xyz = [node.data["x"], node.data["y"], node.data["z"]]
        r   = list(node.data.get("r", []))
        if len(r) < needed:
            r = r + [0.0] * (needed - len(r))
        rows.append(xyz + r[:needed])
    return np.array(rows, dtype=np.float32)


def build_loft_mesh(tree,
                    k: int = _K,
                    n_angular: int = _N_ANGULAR,
                    n_interp: int = _N_INTERP,
                    min_step: float = 8e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a watertight triangle mesh from a VesselGPT tree.

    Each segment is lofted into a **closed** tube (both ends capped).  The tubes
    are combined via ``trimesh.boolean.union`` with the *manifold* engine, which
    correctly merges the overlapping cap geometry at bifurcation junctions and
    produces a single watertight mesh.

    Parameters
    ----------
    tree :
        Deserialized tree (from ``tree_functions.deserialize``).
    k :
        Token width (default 39: 3 xyz + 36 ring values).
    n_angular :
        Circumferential ring resolution (number of angular samples).
    n_interp :
        Interpolated rings inserted between each pair of stored rings.
    min_step :
        Minimum distance between consecutive nodes to keep (phantom-filter).

    Returns
    -------
    verts : (V, 3) float64
    faces : (F, 3) int32
    """
    import trimesh

    # ── 1. pre-compute global RMF frames ─────────────────────────────────────
    frames = compute_global_rmf(tree)

    # ── 2. get segment node lists (phantom nodes filtered out) ────────────────
    seg_nodes = _segment_node_list(tree, min_step=min_step)
    segments  = [_nodes_to_segment(sn, k) for sn in seg_nodes]

    if not segments:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    # ── 3. identify cap flags ─────────────────────────────────────────────────
    # Every segment is capped at both ends so it's a closed, watertight volume
    # that can be fed into the boolean union.  The caps use a tiny epsilon
    # (1e-5) to minimize the intersection area at bifurcation junctions where
    # caps from adjacent segments overlap.

    n_segs = len(segments)
    cap_start_flags: List[bool] = [True] * n_segs
    cap_end_flags:   List[bool] = [True] * n_segs

    # ── 4. build one closed tube per segment ──────────────────────────────────
    import trimesh as _trimesh_mod
    trimesh_list: List = []
    for s_idx, (seg, sn) in enumerate(zip(segments, seg_nodes)):
        cs = cap_start_flags[s_idx]
        ce = cap_end_flags[s_idx]
        verts_s, faces_s = build_segment_tube(
            seg, frames, n_angular, n_interp, sn,
            cap_start=cs, cap_end=ce,
        )
        if len(faces_s) == 0:
            continue
        tm = _trimesh_mod.Trimesh(vertices=verts_s, faces=faces_s, process=True)
        tm.fix_normals()
        if not tm.is_empty and tm.is_watertight:
            trimesh_list.append(tm)

    if not trimesh_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    if len(trimesh_list) == 1:
        m = trimesh_list[0]
        return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)

    # ── 5. boolean union of all closed segment tubes ──────────────────────────
    try:
        from trimesh.boolean import union as trimesh_union
        merged = trimesh_union(trimesh_list, engine="manifold")
        merged.fix_normals()
        verts = np.asarray(merged.vertices, dtype=np.float64)
        faces = np.asarray(merged.faces,    dtype=np.int32)

        # Boolean union with manifold3d densely re-tessellates intersection
        # curves.  Decimate back down to the original tube resolution using
        # Open3D's quadric decimation (shape-preserving, watertight-safe).
        # Target face count = 2 × n_angular × (total inter-ring gaps), which
        # is the theoretical face count of the open tube walls alone.
        n_rings_total = sum(
            len(sn) * (n_interp + 1) - n_interp
            for sn in seg_nodes
            if len(sn) > 1
        )
        target_faces = max(int(2 * n_angular * n_rings_total * 1.5), 10000)

        if len(faces) > target_faces * 2:
            try:
                import open3d as o3d
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
                o3d_mesh.remove_degenerate_triangles()
                o3d_mesh.remove_unreferenced_vertices()
                verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
                faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
            except Exception:
                pass  # keep un-decimated mesh if open3d fails

        return verts, faces
    except Exception as exc:
        # Fallback: just concatenate without boolean (non-manifold at junctions)
        import warnings
        warnings.warn(f"boolean union failed ({exc}); falling back to simple concatenation")
        all_v: List[np.ndarray] = []
        all_f: List[np.ndarray] = []
        offset = 0
        for tm in trimesh_list:
            all_v.append(np.asarray(tm.vertices, dtype=np.float64))
            all_f.append(np.asarray(tm.faces, dtype=np.int32) + offset)
            offset += len(tm.vertices)
        verts = np.vstack(all_v)
        faces = np.vstack(all_f)
        return verts, faces


# ══════════════════════════════════════════════════════════════════════════════
# OBJ export
# ══════════════════════════════════════════════════════════════════════════════

def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    """Write a triangle mesh to Wavefront .obj format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# VesselGPT loft mesh — {len(verts)} verts, {len(faces)} faces\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Watertight check (trimesh / pyvista, optional)
# ══════════════════════════════════════════════════════════════════════════════

def _check_watertight(verts: np.ndarray, faces: np.ndarray) -> bool:
    """Return True if the mesh appears watertight.  Requires trimesh or pyvista."""
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return bool(m.is_watertight)
    except Exception:
        pass
    try:
        import pyvista as pv
        surf = pv.PolyData(verts.astype(np.float32),
                           np.c_[np.full(len(faces), 3, dtype=int), faces].ravel())
        surf = surf.triangulate().clean().fill_holes(hole_size=100)
        return bool(surf.n_open_edges == 0)
    except Exception:
        pass
    return False  # unknown


# ══════════════════════════════════════════════════════════════════════════════
# Single-file reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_file(input_path: str,
                     output_path: str,
                     params: dict) -> Tuple[str, str]:
    """Reconstruct one .npy tree file → .obj mesh.

    Returns ``("ok", timing_str)`` or ``("error", message)``.
    """
    t0 = time.time()
    k    = int(params.get("k", _K))
    mode = str(params.get("mode", "pre_order_kcount"))
    n_angular = int(params.get("n_angular", _N_ANGULAR))
    n_interp  = int(params.get("n_interp",  _N_INTERP))
    min_step  = float(params.get("min_step",  8e-4))
    overwrite = bool(params.get("overwrite", True))

    if os.path.exists(output_path) and not overwrite:
        return "skipped", "exists"

    try:
        npy  = np.load(input_path)
        node_dim = (k + 1) if "kcount" in mode else k
        if npy.ndim == 1:
            npy = npy.reshape(-1, node_dim)
        serial = list(npy.flatten())
        tree   = deserialize(serial, mode=mode, k=k)
        if tree is None:
            return "error", "deserialize returned None"
    except Exception as exc:
        return "error", f"load/deserialize: {exc}"

    # Sanitize phantom nodes (same as SDF pipeline)
    n_repaired = _sanitize_phantom_tree(tree, k)

    try:
        verts, faces = build_loft_mesh(tree, k=k, n_angular=n_angular, n_interp=n_interp, min_step=min_step)
    except Exception as exc:
        import traceback
        return "error", f"build_loft_mesh: {exc}\n{traceback.format_exc()}"

    if len(faces) == 0:
        return "error", "empty mesh"

    try:
        write_obj(output_path, verts, faces)
    except Exception as exc:
        return "error", f"write_obj: {exc}"

    wt = _check_watertight(verts, faces)
    dt = time.time() - t0
    extra = f"phantom={n_repaired} " if n_repaired else ""
    return "ok", f"{extra}{len(verts)} verts, {len(faces)} faces, watertight={wt}  ({dt:.1f}s)"


# ══════════════════════════════════════════════════════════════════════════════
# Batch runner (CLI)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct VesselGPT tree .npy files → triangle meshes via tube lofting."
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--input",  default=None, help="Single .npy input (overrides config input_dir)")
    parser.add_argument("--output_dir", default=None, help="Output directory override")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    params     = cfg.get("params", {})
    input_dir  = args.input or cfg.get("input_dir")
    output_dir = args.output_dir or cfg.get("output_dir")
    pattern    = params.get("pattern", "*.npy")
    ext        = params.get("output_ext", ".obj")
    verbose    = bool(params.get("verbose", True))

    if not input_dir:
        parser.error("input_dir must be set in config or via --input")
    if not output_dir:
        parser.error("output_dir must be set in config or via --output_dir")

    os.makedirs(output_dir, exist_ok=True)

    from glob import glob
    if os.path.isdir(input_dir):
        files = sorted(glob(os.path.join(input_dir, pattern)))
    else:
        files = [input_dir]

    total = len(files)
    n_ok = n_skip = n_err = 0

    for idx, fpath in enumerate(files, start=1):
        stem     = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(output_dir, stem + ext)
        if verbose:
            print(f"  [{idx:3d}/{total}] {stem}...", end=" ", flush=True)

        status, msg = reconstruct_file(fpath, out_path, params)
        if status == "ok":
            n_ok += 1
            if verbose:
                print(msg)
        elif status == "skipped":
            n_skip += 1
            if verbose:
                print(f"skip ({msg})")
        else:
            n_err += 1
            if verbose:
                print(f"ERROR: {msg}")

    print(f"\ndone: {n_ok} ok, {n_skip} skipped, {n_err} errors  (total={total})")


if __name__ == "__main__":
    main()
