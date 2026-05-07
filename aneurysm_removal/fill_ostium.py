"""
Core algorithm: remove aneurysm from vessel meshes.

Strategy: Cylindrical patch fill.
  - Removes all faces touching aneurysm/border vertices
  - Identifies ALL new boundary loops (holes) created by the removal
  - Fills each hole with constrained Delaunay triangulation
  - Maps interior fill vertices onto the local tube surface
  - Applies Taubin smoothing to blend the patches into the vessel
"""

import numpy as np
import trimesh
import triangle as tri_lib
from collections import Counter, defaultdict
from typing import List, Set, Tuple
from scipy.spatial import cKDTree


# -- Boundary loops --

def get_boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    edges = mesh.edges_sorted
    edge_tuples = [tuple(e) for e in edges]
    edge_counts = Counter(edge_tuples)
    boundary_edges = set(e for e, c in edge_counts.items() if c == 1)
    adj: dict = defaultdict(set)
    for e in boundary_edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    visited: set = set()
    loops: list = []
    for start in adj:
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbors = adj[current] - ({prev} if prev is not None else set())
            if not neighbors:
                break
            prev = current
            current = neighbors.pop()
        loops.append(loop)
    return loops


def find_ostium_loop(mesh, ostium_centroid):
    loops = get_boundary_loops(mesh)
    best_i, best_d = -1, float("inf")
    for i, loop in enumerate(loops):
        c = mesh.vertices[loop].mean(axis=0)
        d = float(np.linalg.norm(c - ostium_centroid))
        if d < best_d:
            best_d = d
            best_i = i
    return loops[best_i], best_i


def _loops_signature(loops: List[List[int]]) -> List[int]:
    return sorted(len(l) for l in loops)


def _fill_one_loop(
    loop_vidx: List[int],
    all_verts: list,
    vertices_3d: np.ndarray,
    labels: np.ndarray,
) -> Tuple[list, set]:
    """
    Fill a single boundary loop with constrained Delaunay triangulation
    projected onto the local tube surface.

    Returns (new_faces, new_vert_indices).
    """
    n_bnd = len(loop_vidx)
    bnd_pts_3d = vertices_3d[loop_vidx]
    loop_center = bnd_pts_3d.mean(0)

    # Tiny loops (3 vertices): fill with a single triangle
    if n_bnd == 3:
        return [[loop_vidx[0], loop_vidx[1], loop_vidx[2]]], set()
    if n_bnd < 3:
        return [], set()

    # Compute local plane normal from the loop (cross product of consecutive edges)
    normals = []
    for i in range(n_bnd):
        e1 = bnd_pts_3d[(i + 1) % n_bnd] - bnd_pts_3d[i]
        e2 = bnd_pts_3d[(i + 2) % n_bnd] - bnd_pts_3d[(i + 1) % n_bnd]
        n = np.cross(e1, e2)
        nl = np.linalg.norm(n)
        if nl > 1e-12:
            normals.append(n / nl)
    if normals:
        plane_normal = np.mean(normals, axis=0)
        plane_normal /= np.linalg.norm(plane_normal) + 1e-30
    else:
        plane_normal = np.array([0.0, 0.0, 1.0])

    # Local 2D frame
    u_ax = np.cross(plane_normal, [1, 0, 0])
    if np.linalg.norm(u_ax) < 0.1:
        u_ax = np.cross(plane_normal, [0, 1, 0])
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(plane_normal, u_ax)
    v_ax /= np.linalg.norm(v_ax)

    diff = bnd_pts_3d - loop_center
    bnd_2d = np.column_stack([diff @ u_ax, diff @ v_ax])

    # Check if 2D projection is self-intersecting; if so, fallback to PCA
    from shapely.geometry import LinearRing
    if n_bnd >= 4:
        ring = LinearRing(bnd_2d)
        if not ring.is_simple:
            _, _, Vt = np.linalg.svd(diff)
            u_ax = Vt[0]
            v_ax = Vt[1]
            bnd_2d = np.column_stack([diff @ u_ax, diff @ v_ax])
            ring = LinearRing(bnd_2d)
            if not ring.is_simple:
                return [], set()

    # Constrained Delaunay triangulation
    segments = np.array([[i, (i + 1) % n_bnd] for i in range(n_bnd)])
    med_edge = np.median(np.linalg.norm(np.diff(bnd_pts_3d, axis=0), axis=1))
    target_area = (np.sqrt(3) / 4) * med_edge ** 2

    try:
        tri_out = tri_lib.triangulate(
            {"vertices": bnd_2d, "segments": segments},
            f"pYq20a{target_area:.8f}",
        )
    except Exception:
        return [], set()

    fill_verts_2d = tri_out["vertices"]
    fill_tri = tri_out["triangles"]
    n_new = len(fill_verts_2d) - n_bnd
    new_pts_2d = fill_verts_2d[n_bnd:]

    # Map interior vertices onto tube surface
    # Fit local tube from nearby vessel vertices
    v_idx = np.where(labels == 0)[0]
    v_pts = vertices_3d[v_idx]
    max_bnd_dist = np.linalg.norm(bnd_pts_3d - loop_center, axis=1).max()
    near_mask = np.linalg.norm(v_pts - loop_center, axis=1) < max_bnd_dist * 3.0
    near_pts = v_pts[near_mask]

    if len(near_pts) < 10:
        # Not enough context: place on flat plane
        if n_new > 0:
            interior_3d = loop_center + new_pts_2d[:, 0:1] * u_ax + new_pts_2d[:, 1:2] * v_ax
        else:
            interior_3d = np.zeros((0, 3))
    else:
        tube_ctr = near_pts.mean(0)
        _, _, Vt = np.linalg.svd(near_pts - tube_ctr)
        tube_axis = Vt[0]

        dp_bnd = bnd_pts_3d - tube_ctr
        bnd_along = np.einsum("ij,j->i", dp_bnd, tube_axis)
        bnd_radial = dp_bnd - bnd_along[:, np.newaxis] * tube_axis[np.newaxis, :]
        bnd_rdist = np.linalg.norm(bnd_radial, axis=1)

        if n_new > 0:
            P_flat = loop_center + new_pts_2d[:, 0:1] * u_ax + new_pts_2d[:, 1:2] * v_ax
            dp = P_flat - tube_ctr
            along_comp = np.einsum("ij,j->i", dp, tube_axis)
            radial_vec = dp - along_comp[:, np.newaxis] * tube_axis[np.newaxis, :]
            r_actual = np.linalg.norm(radial_vec, axis=1, keepdims=True)
            r_actual = np.maximum(r_actual, 1e-12)

            bnd_tree_2d = cKDTree(bnd_2d)
            dd, ii = bnd_tree_2d.query(new_pts_2d, k=min(6, n_bnd))
            ww = 1.0 / (dd + 1e-10)
            ww /= ww.sum(1, keepdims=True)
            target_r = (bnd_rdist[ii] * ww).sum(axis=1, keepdims=True)

            interior_3d = (
                tube_ctr
                + along_comp[:, np.newaxis] * tube_axis[np.newaxis, :]
                + radial_vec * (target_r / r_actual)
            )
        else:
            interior_3d = np.zeros((0, 3))

    # Build index mapping and faces
    vidx_map = {}
    for i, vid in enumerate(loop_vidx):
        vidx_map[i] = vid

    new_vert_indices = set()
    for i, pt in enumerate(interior_3d):
        new_idx = len(all_verts)
        vidx_map[n_bnd + i] = new_idx
        all_verts.append(pt)
        new_vert_indices.add(new_idx)

    new_faces = []
    for face in fill_tri:
        new_faces.append([vidx_map[int(v)] for v in face])

    return new_faces, new_vert_indices


# -- Main --

def fill_ostium(
    full_mesh: trimesh.Trimesh,
    labels: np.ndarray,
    ostium_centroid: np.ndarray,
    normal_vec: np.ndarray,
    n_smooth_iters: int = 60,
    smooth_rings: int = 4,
    **kwargs,
) -> trimesh.Trimesh:
    """
    Remove the aneurysm by filling ALL holes with tube-conforming patches.

    Handles cases with multiple aneurysm openings (multiple boundary loops
    created by removing aneurysm+border faces).
    """
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    # 1. Remove aneurysm + border faces
    remove_mask = (labels == 1) | (labels == 2)
    remove_verts = set(np.where(remove_mask)[0].tolist())
    face_has_removed = remove_mask[full_mesh.faces].any(axis=1)
    vessel_faces = full_mesh.faces[~face_has_removed]

    # 1b. Remove peninsula faces (faces with 2+ boundary edges) near the removal site
    # Only remove peninsulas whose vertices are neighbors of removed vertices
    # Build adjacency using vectorized operations
    faces_np = full_mesh.faces.astype(np.int64)
    _near_removed_arr = np.zeros(len(full_mesh.vertices), dtype=bool)
    # Mark vertices that neighbor removed vertices (1-ring)
    for f in faces_np:
        a, b, c = f
        if remove_mask[a] or remove_mask[b] or remove_mask[c]:
            if not remove_mask[a]:
                _near_removed_arr[a] = True
            if not remove_mask[b]:
                _near_removed_arr[b] = True
            if not remove_mask[c]:
                _near_removed_arr[c] = True
    # Expand to 2-ring
    _near_2ring = _near_removed_arr.copy()
    for f in faces_np:
        a, b, c = f
        if _near_removed_arr[a] or _near_removed_arr[b] or _near_removed_arr[c]:
            _near_2ring[a] = True
            _near_2ring[b] = True
            _near_2ring[c] = True

    for _cleanup in range(5):
        # Compute boundary edges using numpy
        vf = vessel_faces.astype(np.int64)
        e1 = np.sort(vf[:, :2], axis=1)
        e2 = np.sort(vf[:, 1:], axis=1)
        e3 = np.sort(vf[:, [0, 2]], axis=1)
        all_edges = np.vstack([e1, e2, e3])
        # Count edge occurrences
        edge_keys = all_edges[:, 0].astype(np.int64) * len(full_mesh.vertices) + all_edges[:, 1]
        unique_keys, counts = np.unique(edge_keys, return_counts=True)
        bnd_key_set = set(unique_keys[counts == 1].tolist())

        nf = len(vf)
        # For each face, count how many of its 3 edges are boundary
        ek1 = e1[:, 0] * len(full_mesh.vertices) + e1[:, 1]  # edge keys per face
        ek2 = e2[:, 0] * len(full_mesh.vertices) + e2[:, 1]
        ek3 = e3[:, 0] * len(full_mesh.vertices) + e3[:, 1]

        # Check if face is near removed
        face_near = _near_2ring[vf[:, 0]] | _near_2ring[vf[:, 1]] | _near_2ring[vf[:, 2]]
        bnd1 = np.array([k in bnd_key_set for k in ek1.tolist()])
        bnd2 = np.array([k in bnd_key_set for k in ek2.tolist()])
        bnd3 = np.array([k in bnd_key_set for k in ek3.tolist()])
        n_bnd_per_face = bnd1.astype(int) + bnd2.astype(int) + bnd3.astype(int)

        thin_mask = face_near & (n_bnd_per_face >= 2)
        if not thin_mask.any():
            break
        vessel_faces = vessel_faces[~thin_mask]

    vessel_mesh = trimesh.Trimesh(
        vertices=full_mesh.vertices, faces=vessel_faces, process=False
    )

    # 2. Find NEW boundary loops (holes created by aneurysm removal)
    # A loop is "new" if any of its vertices are direct neighbors of removed vertices
    neighbor_of_removed = set(np.where(_near_removed_arr)[0].tolist())

    all_loops = get_boundary_loops(vessel_mesh)

    # Build boundary adjacency to check if loops are closed
    _edge_counts = Counter(tuple(e) for e in vessel_mesh.edges_sorted)
    _bnd_adj: dict = defaultdict(set)
    for e, cnt in _edge_counts.items():
        if cnt == 1:
            _bnd_adj[e[0]].add(e[1])
            _bnd_adj[e[1]].add(e[0])

    holes_to_fill = []
    for loop in all_loops:
        if not (set(loop) & neighbor_of_removed):
            continue
        # Only fill closed loops (real holes), skip open paths (peninsulas)
        if loop[-1] not in _bnd_adj.get(loop[0], set()):
            continue
        holes_to_fill.append(loop)

    if not holes_to_fill:
        return vessel_mesh

    # 4. Fill each hole
    all_verts = list(full_mesh.vertices)
    all_faces = [[int(f[0]), int(f[1]), int(f[2])] for f in vessel_faces]
    all_fill_verts = set()
    all_anchor_verts = set()

    for loop in holes_to_fill:
        new_faces, new_vert_ids = _fill_one_loop(
            loop, all_verts, full_mesh.vertices, labels
        )
        all_faces.extend(new_faces)
        all_fill_verts.update(new_vert_ids)
        all_anchor_verts.update(loop)

    verts_arr = np.array(all_verts, dtype=np.float64)
    faces_arr = np.array(all_faces, dtype=np.int64)

    if not all_fill_verts:
        return trimesh.Trimesh(vertices=verts_arr, faces=faces_arr, process=False)

    # 5. Taubin smoothing on all fill regions
    result_adj = defaultdict(set)
    for f in faces_arr:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        result_adj[a].update([b, c])
        result_adj[b].update([a, c])
        result_adj[c].update([a, b])

    smooth_set = set(all_fill_verts)
    for _ in range(smooth_rings):
        expand = set()
        for v in smooth_set:
            for nb in result_adj[v]:
                if nb not in smooth_set and nb not in all_anchor_verts:
                    expand.add(nb)
        smooth_set.update(expand)

    lam = 0.5
    mu = -0.53
    for it in range(n_smooth_iters):
        w = lam if it % 2 == 0 else mu
        nv = verts_arr.copy()
        for vi in smooth_set:
            nbs = list(result_adj[vi])
            if nbs:
                nv[vi] = verts_arr[vi] + w * (verts_arr[nbs].mean(0) - verts_arr[vi])
        verts_arr = nv

    return trimesh.Trimesh(vertices=verts_arr, faces=faces_arr, process=False)
