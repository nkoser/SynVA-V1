"""
mesh_generation.py — SDF + Marching Cubes mesh reconstruction from
densely interpolated vessel cross-sections.

Pipeline
--------
1. Collect all cross-section stations (center, frame, 2-D ring polygon).
2. Build a KD-tree on centerline points.
3. Create a regular 3-D grid covering the vessel bounding box.
4. For each grid point find K nearest cross-section stations.
5. Compute polygon SDF in each station's local 2-D frame.
6. Combine with a smooth-minimum for bifurcation blending.
7. Optional Gaussian smoothing of the SDF grid for surface quality.
8. Extract the isosurface with Marching Cubes.
9. Optional: Taubin smoothing, mesh decimation.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════════════
# Vectorised 2-D polygon signed distance
# ═══════════════════════════════════════════════════════════════════════════


def _batch_polygon_sdf_2d(
    queries: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Signed distance from Q query points to a closed 2-D polygon.

    Fully vectorised over Q.  Negative = inside, positive = outside.

    Parameters
    ----------
    queries : (Q, 2)
    polygon : (P, 2)  — vertices in order (closed, i.e. edge P-1→0 implicit).

    Returns
    -------
    (Q,) float64 signed distances.
    """
    Q = len(queries)
    P = len(polygon)
    if P < 3 or Q == 0:
        return np.ones(Q, dtype=np.float64)

    # Edge endpoints: a → b
    a = polygon                              # (P, 2)
    b = np.roll(polygon, -1, axis=0)         # (P, 2)
    ex = b[:, 0] - a[:, 0]                   # (P,)
    ey = b[:, 1] - a[:, 1]                   # (P,)

    px = queries[:, 0]                        # (Q,)
    py = queries[:, 1]                        # (Q,)
    ax = a[:, 0]                              # (P,)
    ay = a[:, 1]                              # (P,)
    bx = b[:, 0]
    by = b[:, 1]

    # ── Minimum distance to polygon edges ────────────────────────
    # f = query - edge_start   → (Q, P)  via broadcasting
    fx = px[:, None] - ax[None, :]            # (Q, P)
    fy = py[:, None] - ay[None, :]            # (Q, P)

    e_dot = ex * ex + ey * ey                 # (P,)
    t = (fx * ex[None, :] + fy * ey[None, :]) / (e_dot[None, :] + 1e-30)
    t = np.clip(t, 0.0, 1.0)                 # (Q, P)

    dx = fx - t * ex[None, :]                 # (Q, P)
    dy = fy - t * ey[None, :]
    dist_sq = dx * dx + dy * dy               # (Q, P)
    min_dist = np.sqrt(np.min(dist_sq, axis=1))  # (Q,)

    # ── Inside / outside via ray-casting ─────────────────────────
    # Count how many edges cross a horizontal ray going right.
    crosses_up   = (ay[None, :] <= py[:, None]) & (by[None, :] > py[:, None])
    crosses_down = (by[None, :] <= py[:, None]) & (ay[None, :] > py[:, None])
    crosses = crosses_up | crosses_down       # (Q, P)

    # x coordinate of the ray–edge intersection
    dy_edge = by - ay                          # (P,)
    t_inter = (py[:, None] - ay[None, :]) / (dy_edge[None, :] + 1e-30)
    x_inter = ax[None, :] + t_inter * ex[None, :]

    right = crosses & (x_inter > px[:, None])  # (Q, P)
    inside = (np.sum(right, axis=1) % 2 == 1)  # (Q,)

    sdf = np.where(inside, -min_dist, min_dist)
    return sdf


# ═══════════════════════════════════════════════════════════════════════════
# Station collection
# ═══════════════════════════════════════════════════════════════════════════


def _collect_stations(segment_data: list[dict]) -> dict | None:
    """Flatten all interpolated cross-section data into arrays.

    Returns a dict with arrays:
        centers   (S, 3)
        normals   (S, 3)   — N axis of each cross-section frame
        binormals (S, 3)   — B axis
        rings_2d  list[S] of (P, 2)  — ring vertices in local 2-D
        rings_3d  list[S] of (P, 3)  — ring vertices in world 3-D
        edges     (E, 2) int — pairs of consecutive station indices within each segment
    """
    from .interpolation import compute_transport_frames

    centers_list = []
    normals_list = []
    binormals_list = []
    rings_2d_list = []
    rings_3d_list = []
    edges_list = []

    offset = 0  # running station index offset

    for seg in segment_data:
        c = seg["centers"]
        rr = seg["rings"]
        M = len(rr)
        if M < 1:
            continue

        T, N, B = compute_transport_frames(c)

        for i in range(M):
            ring = rr[i]               # (P, 3) world coords
            center = c[i]
            n_vec = N[i]
            b_vec = B[i]

            diff = ring - center       # (P, 3)
            u = diff @ n_vec           # (P,) projection onto N
            v = diff @ b_vec           # (P,) projection onto B
            ring_2d = np.column_stack([u, v])

            centers_list.append(center)
            normals_list.append(n_vec)
            binormals_list.append(b_vec)
            rings_2d_list.append(ring_2d)
            rings_3d_list.append(ring)

        # Build edges for this segment (consecutive pairs)
        for i in range(M - 1):
            edges_list.append((offset + i, offset + i + 1))
        offset += M

    if not centers_list:
        return None

    return {
        "centers":   np.array(centers_list, dtype=np.float64),
        "normals":   np.array(normals_list, dtype=np.float64),
        "binormals": np.array(binormals_list, dtype=np.float64),
        "rings_2d":  rings_2d_list,
        "rings_3d":  rings_3d_list,
        "edges":     np.array(edges_list, dtype=np.int64) if edges_list else np.empty((0, 2), dtype=np.int64),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Smooth minimum
# ═══════════════════════════════════════════════════════════════════════════


def _smooth_min_poly(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-min (standard SDF modelling blend)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


# ═══════════════════════════════════════════════════════════════════════════
# Main SDF + Marching-Cubes reconstruction
# ═══════════════════════════════════════════════════════════════════════════


def sdf_reconstruction(
    segment_data: list[dict],
    grid_resolution: int = 256,
    padding: float = 0.02,
    n_neighbors: int = 3,
    smooth_k: float = 0.01,
    level: float = 0.0,
    narrow_band_factor: float = 3.0,
    chunk_size: int = 200_000,
    sdf_smooth_sigma: float = 0.0,
    sdf_erosion: float = 0.0,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Build an SDF from cross-section rings via sweep-based tube primitives
    and extract the isosurface with Marching Cubes.

    Instead of projecting each grid point onto the nearest station's plane
    (which creates a "beaded" look), this version projects onto the nearest
    centerline *edge* and linearly interpolates the cross-section shape
    along the edge.  This produces smooth, continuous tube surfaces.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``.
    grid_resolution : voxels per longest bbox axis.
    padding : extra space around the bounding box (world units).
    n_neighbors : number of nearest edges to query per grid point.
    smooth_k : polynomial smooth-min blending radius (0 = hard min).
    level : isosurface level for marching cubes.
    narrow_band_factor : only compute exact SDF within this × max_ring_radius.
    chunk_size : grid points per batch (memory ↔ speed trade-off).
    sdf_smooth_sigma : Gaussian smoothing sigma in voxel units (0 = off).
    sdf_erosion : positive offset added to SDF after smoothing to compensate
        for inflation from smooth-min blending.  In world-coordinate units.
        Typical values: 0.001-0.004.  (0 = off).
    verbose : print progress.

    Returns
    -------
    trimesh.Trimesh or None.
    """
    from skimage.measure import marching_cubes

    # ── 1. Collect stations + edges ──────────────────────────────────
    stations = _collect_stations(segment_data)
    if stations is None:
        return None

    centers   = stations["centers"]    # (S, 3)
    normals   = stations["normals"]    # (S, 3)
    binormals = stations["binormals"]  # (S, 3)
    rings_2d  = stations["rings_2d"]   # list[S] of (P, 2)
    rings_3d  = stations["rings_3d"]   # list[S] of (P, 3)
    edges     = stations["edges"]      # (E, 2) int
    S = len(centers)
    E = len(edges)

    if verbose:
        print(f"    Stations: {S}, edges: {E}, ring pts: {rings_2d[0].shape[0]}")

    # ── 2. Per-station max radius (for narrow band) ──────────────────
    max_radii = np.array([
        np.max(np.linalg.norm(r2d, axis=1)) for r2d in rings_2d
    ], dtype=np.float64)

    median_r = np.median(max_radii[max_radii > 1e-6]) if np.any(max_radii > 1e-6) else 0.01
    radius_cap = max(median_r * 10.0, 0.5)
    n_clamped = (max_radii > radius_cap).sum()
    if n_clamped > 0 and verbose:
        print(f"    Clamping {n_clamped} degenerate radii (>{radius_cap:.4f})")
    max_radii = np.clip(max_radii, 0, radius_cap)

    global_max_r = float(np.max(max_radii))
    narrow_band = narrow_band_factor * global_max_r

    if verbose:
        print(f"    Radius range: [{max_radii.min():.4f}, {global_max_r:.4f}]")
        print(f"    Narrow band: {narrow_band:.4f}")

    # ── 3. Edge midpoints + KD-tree ──────────────────────────────────
    if E > 0:
        edge_mids = 0.5 * (centers[edges[:, 0]] + centers[edges[:, 1]])
    else:
        # Fallback: no edges (single-node segments only)
        edge_mids = centers.copy()
    kd_edges = cKDTree(edge_mids)

    # Also keep a station KD-tree for narrow-band test
    kd_stations = cKDTree(centers)

    # ── 4. Bounding box & grid ───────────────────────────────────────
    all_ring_pts = np.vstack(rings_3d)
    bbox_min = np.minimum(centers.min(0), all_ring_pts.min(0)) - padding
    bbox_max = np.maximum(centers.max(0), all_ring_pts.max(0)) + padding
    extent = bbox_max - bbox_min
    longest = extent.max()

    voxel_size = longest / (grid_resolution - 1)
    grid_dims = np.ceil(extent / voxel_size).astype(int) + 1
    spacing = tuple([voxel_size] * 3)

    axes = [
        np.linspace(bbox_min[d], bbox_min[d] + (grid_dims[d] - 1) * voxel_size,
                     grid_dims[d])
        for d in range(3)
    ]

    if verbose:
        print(f"    Grid: {grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]}  "
              f"voxel={voxel_size:.5f}  bbox=[{bbox_min}, {bbox_max}]")

    # ── 5. Flat grid & narrow-band mask ──────────────────────────────
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    N_total = len(grid_pts)

    dist_cl, _ = kd_stations.query(grid_pts, k=1)
    near_mask = dist_cl < narrow_band
    near_indices = np.where(near_mask)[0]
    n_near = len(near_indices)

    if verbose:
        far_frac = 1.0 - n_near / N_total
        print(f"    Near-band: {n_near:,} / {N_total:,} points "
              f"({100 * (1 - far_frac):.1f}% computed, "
              f"{100 * far_frac:.1f}% skipped)")

    # ── 6. Init SDF ─────────────────────────────────────────────────
    sdf = dist_cl - global_max_r
    sdf = np.maximum(sdf, 0.001)

    # ── 7. Sweep-based SDF for near-band ─────────────────────────────
    if E == 0:
        # No edges — fall back to single-station SDF
        sdf_near = np.full(n_near, 1e6)
        pts_near = grid_pts[near_indices]
        _, nn_idx = kd_stations.query(pts_near, k=1)
        for sid in np.unique(nn_idx):
            mask_s = (nn_idx == sid)
            diff = pts_near[mask_s] - centers[sid]
            u = diff @ normals[sid]
            v = diff @ binormals[sid]
            q2d = np.column_stack([u, v])
            sdf_near[mask_s] = _batch_polygon_sdf_2d(q2d, rings_2d[sid])
        sdf[near_indices] = sdf_near
    else:
        # Query K nearest edges per grid point
        K_e = min(n_neighbors, E)
        pts_near = grid_pts[near_indices]
        _, nn_edge_idx = kd_edges.query(pts_near, k=K_e)
        if K_e == 1:
            nn_edge_idx = nn_edge_idx[:, None]

        sdf_near = np.full(n_near, 1e6, dtype=np.float64)

        for k_i in range(K_e):
            edge_ids = nn_edge_idx[:, k_i]
            sdf_k = _sweep_sdf_batch(
                pts_near, edge_ids, edges, centers, normals, binormals,
                rings_2d, chunk_size,
            )
            sdf_near = _smooth_min_poly(sdf_near, sdf_k, smooth_k)

        sdf[near_indices] = sdf_near

    # ── 10. Reshape & optional Gaussian smoothing ────────────────────
    sdf_grid = sdf.reshape(grid_dims[0], grid_dims[1], grid_dims[2])

    if sdf_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        sdf_grid = gaussian_filter(sdf_grid, sigma=sdf_smooth_sigma)
        if verbose:
            print(f"    Applied Gaussian smoothing (sigma={sdf_smooth_sigma:.1f} voxels)")

    # Erosion: shift SDF outward to compensate for smooth-min inflation
    if sdf_erosion > 0:
        erosion_voxels = sdf_erosion / voxel_size
        sdf_grid = sdf_grid + sdf_erosion
        if verbose:
            print(f"    Applied SDF erosion: {sdf_erosion:.4f} ({erosion_voxels:.2f} voxels)")

    # Force SDF > 0 at grid boundaries
    boundary_val = max(0.001, voxel_size)
    sdf_grid[ 0, :, :] = np.maximum(sdf_grid[ 0, :, :], boundary_val)
    sdf_grid[-1, :, :] = np.maximum(sdf_grid[-1, :, :], boundary_val)
    sdf_grid[:,  0, :] = np.maximum(sdf_grid[:,  0, :], boundary_val)
    sdf_grid[:, -1, :] = np.maximum(sdf_grid[:, -1, :], boundary_val)
    sdf_grid[:, :,  0] = np.maximum(sdf_grid[:, :,  0], boundary_val)
    sdf_grid[:, :, -1] = np.maximum(sdf_grid[:, :, -1], boundary_val)

    if verbose:
        n_inside = (sdf_grid < level).sum()
        print(f"    SDF: inside={n_inside:,} voxels  "
              f"(min={sdf_grid.min():.4f}, max={sdf_grid.max():.4f})")

    # ── 11. Marching cubes ───────────────────────────────────────────
    try:
        verts_mc, faces_mc, normals_mc, _ = marching_cubes(
            sdf_grid, level=level, spacing=spacing,
        )
    except Exception as e:
        if verbose:
            print(f"    Marching cubes failed: {e}")
        return None

    verts_mc = verts_mc + bbox_min

    if len(verts_mc) == 0 or len(faces_mc) == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts_mc, faces=faces_mc, process=False)

    # Remove tiny MC artefact components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components.sort(key=lambda c: len(c.faces), reverse=True)
        mesh = components[0]
        if verbose:
            print(f"    Removed {len(components) - 1} tiny artefact components")

    if verbose:
        print(f"    MC mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"    Watertight: {mesh.is_watertight}")

    return mesh


def _sweep_sdf_batch(
    pts: np.ndarray,
    edge_ids: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    rings_2d: list,
    chunk_size: int,
) -> np.ndarray:
    """Compute swept-tube SDF for a batch of points against their assigned edges.

    For each point, the SDF is computed by:
    1. Projecting the point onto the centerline segment (edge).
    2. Clamping the parameter t to [0, 1].
    3. Linearly interpolating the cross-section ring at parameter t.
    4. Projecting the point into the interpolated local frame.
    5. Computing 2-D polygon SDF against the interpolated ring.

    Parameters
    ----------
    pts : (Q, 3)
    edge_ids : (Q,) int — index into *edges* for each query point.
    edges : (E, 2) int — station index pairs.
    centers, normals, binormals : station data.
    rings_2d : list of (P, 2) per station.
    chunk_size : batch size for polygon SDF.

    Returns
    -------
    (Q,) SDF values.
    """
    Q = len(pts)
    sdf_out = np.full(Q, 1e6, dtype=np.float64)

    unique_eids = np.unique(edge_ids)
    for eid in unique_eids:
        mask = (edge_ids == eid)
        q_pts = pts[mask]  # (Qe, 3)

        ia, ib = edges[eid]
        ca, cb = centers[ia], centers[ib]
        na, nb = normals[ia], normals[ib]
        ba, bb = binormals[ia], binormals[ib]
        ra, rb = rings_2d[ia], rings_2d[ib]

        # Edge direction
        edge_vec = cb - ca
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            # Degenerate edge — use station a
            diff = q_pts - ca
            u = diff @ na
            v = diff @ ba
            sdf_out[mask] = _batch_polygon_sdf_2d(
                np.column_stack([u, v]), ra
            )
            continue

        edge_dir = edge_vec / edge_len

        # Project onto edge → parameter t ∈ [0, 1]
        diff = q_pts - ca  # (Qe, 3)
        t_proj = diff @ edge_dir  # (Qe,)
        t_norm = t_proj / edge_len
        t_clamped = np.clip(t_norm, 0.0, 1.0)  # (Qe,)

        # Interpolated center
        center_t = ca + t_clamped[:, None] * edge_vec  # (Qe, 3)

        # Interpolate frame vectors (linear + re-orthogonalise)
        tc = t_clamped[:, None]
        n_t = (1.0 - tc) * na + tc * nb  # (Qe, 3)
        b_t = (1.0 - tc) * ba + tc * bb  # (Qe, 3)
        # Re-orthogonalise (Gram-Schmidt wrt tangent isn't needed —
        # just normalize and ensure orthogonality of n/b)
        n_len = np.linalg.norm(n_t, axis=1, keepdims=True)
        n_t = n_t / np.maximum(n_len, 1e-12)
        b_t = b_t - np.sum(b_t * n_t, axis=1, keepdims=True) * n_t
        b_len = np.linalg.norm(b_t, axis=1, keepdims=True)
        b_t = b_t / np.maximum(b_len, 1e-12)

        # Project point into interpolated local frame
        diff_from_center = q_pts - center_t  # (Qe, 3)
        u = np.sum(diff_from_center * n_t, axis=1)  # (Qe,)
        v = np.sum(diff_from_center * b_t, axis=1)  # (Qe,)

        # Interpolated 2D ring:  (1-t)*ring_a + t*ring_b
        # Group by quantised t-bins for efficient polygon SDF
        N_BINS = 8
        t_bin = np.clip((t_clamped * N_BINS).astype(int), 0, N_BINS - 1)

        sdf_edge = np.full(q_pts.shape[0], 1e6, dtype=np.float64)
        for bi in range(N_BINS):
            bmask = (t_bin == bi)
            if not np.any(bmask):
                continue
            t_mid = (bi + 0.5) / N_BINS
            ring_interp = (1.0 - t_mid) * ra + t_mid * rb  # (P, 2)
            q2d = np.column_stack([u[bmask], v[bmask]])
            if len(q2d) <= chunk_size:
                sdf_edge[bmask] = _batch_polygon_sdf_2d(q2d, ring_interp)
            else:
                vals = np.empty(len(q2d), dtype=np.float64)
                for sc in range(0, len(q2d), chunk_size):
                    se = min(sc + chunk_size, len(q2d))
                    vals[sc:se] = _batch_polygon_sdf_2d(q2d[sc:se], ring_interp)
                sdf_edge[bmask] = vals

        sdf_out[mask] = sdf_edge

    return sdf_out


# ═══════════════════════════════════════════════════════════════════════════
# Fast SDF + Marching-Cubes (numba-accelerated)
# ═══════════════════════════════════════════════════════════════════════════


def sdf_fast_reconstruction(
    segment_data: list[dict],
    grid_resolution: int = 256,
    padding: float = 0.02,
    n_neighbors: int = 3,
    smooth_k: float = 0.005,
    level: float = 0.0,
    narrow_band_factor: float = 3.0,
    sdf_smooth_sigma: float = 1.5,
    sdf_erosion: float = 0.003,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Numba-accelerated SDF + Marching Cubes mesh reconstruction.

    Same algorithm as ``sdf_reconstruction`` but uses a JIT-compiled kernel
    for the sweep-based SDF evaluation, giving ~100× speed-up.
    Default post-processing (Gaussian smoothing + erosion) is tuned to
    compensate for smooth-min inflation while preserving aneurysm geometry.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``.
    grid_resolution : voxels per longest bbox axis.
    padding : extra space around the bounding box (world units).
    n_neighbors : number of nearest edges to query per grid point.
    smooth_k : polynomial smooth-min blending radius (0 = hard min).
    level : isosurface level for marching cubes.
    narrow_band_factor : only compute exact SDF within this × max_ring_radius.
    sdf_smooth_sigma : Gaussian smoothing sigma in voxel units (0 = off).
    sdf_erosion : positive offset added to SDF after smoothing to compensate
        for inflation from smooth-min blending (world units, 0 = off).
    verbose : print progress.
    """
    from skimage.measure import marching_cubes
    from .sdf_numba import compute_sdf_batch

    # ── 1. Collect stations + edges ──
    stations = _collect_stations(segment_data)
    if stations is None:
        return None

    centers   = stations["centers"]
    normals   = stations["normals"]
    binormals = stations["binormals"]
    rings_2d  = stations["rings_2d"]   # list[S] of (P, 2)
    rings_3d  = stations["rings_3d"]
    edges     = stations["edges"]
    S = len(centers)
    E = len(edges)

    if verbose:
        print(f"    Stations: {S}, edges: {E}, ring pts: {rings_2d[0].shape[0]}")

    # ── 2. Stack rings into 3-D array for numba ──
    rings_2d_arr = np.array(rings_2d, dtype=np.float64)  # (S, P, 2)

    # ── 3. Per-station max radius ──
    max_radii = np.array([
        np.max(np.linalg.norm(r2d, axis=1)) for r2d in rings_2d
    ], dtype=np.float64)
    median_r = np.median(max_radii[max_radii > 1e-6]) if np.any(max_radii > 1e-6) else 0.01
    radius_cap = max(median_r * 10.0, 0.5)
    max_radii = np.clip(max_radii, 0, radius_cap)
    global_max_r = float(np.max(max_radii))
    narrow_band = narrow_band_factor * global_max_r

    if verbose:
        print(f"    Radius range: [{max_radii.min():.4f}, {global_max_r:.4f}]")
        print(f"    Narrow band: {narrow_band:.4f}")

    # ── 4. KD-trees ──
    if E > 0:
        edge_mids = 0.5 * (centers[edges[:, 0]] + centers[edges[:, 1]])
    else:
        edge_mids = centers.copy()
    kd_edges = cKDTree(edge_mids)
    kd_stations = cKDTree(centers)

    # ── 5. Bounding box & grid ──
    all_ring_pts = np.vstack(rings_3d)
    bbox_min = np.minimum(centers.min(0), all_ring_pts.min(0)) - padding
    bbox_max = np.maximum(centers.max(0), all_ring_pts.max(0)) + padding
    extent = bbox_max - bbox_min
    voxel_size = extent.max() / (grid_resolution - 1)
    grid_dims = np.ceil(extent / voxel_size).astype(int) + 1

    axes = [
        np.linspace(bbox_min[d], bbox_min[d] + (grid_dims[d] - 1) * voxel_size,
                     grid_dims[d])
        for d in range(3)
    ]

    if verbose:
        print(f"    Grid: {grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]}  "
              f"voxel={voxel_size:.5f}")

    # ── 6. Flat grid & narrow-band ──
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    dist_cl, _ = kd_stations.query(grid_pts, k=1)
    near_indices = np.where(dist_cl < narrow_band)[0]
    n_near = len(near_indices)

    if verbose:
        print(f"    Near-band: {n_near:,} / {len(grid_pts):,} pts "
              f"({100 * n_near / len(grid_pts):.1f}%)")

    # ── 7. Init SDF ──
    sdf = dist_cl - global_max_r
    sdf = np.maximum(sdf, 0.001)

    # ── 8. Numba sweep SDF ──
    if E > 0:
        import time as _t
        pts_near = grid_pts[near_indices]
        K_e = min(n_neighbors, E)
        _, nn_edge_idx = kd_edges.query(pts_near, k=K_e)
        nn_edge_idx = nn_edge_idx.astype(np.int64)
        if K_e == 1:
            nn_edge_idx = nn_edge_idx.reshape(-1, 1)

        t0 = _t.time()
        sdf_near = compute_sdf_batch(
            pts_near, nn_edge_idx, K_e,
            edges, centers, normals, binormals,
            rings_2d_arr, smooth_k,
        )
        if verbose:
            print(f"    Numba SDF: {_t.time() - t0:.1f}s")

        sdf[near_indices] = sdf_near

    # ── 9. Reshape & post-process ──
    sdf_grid = sdf.reshape(grid_dims)

    if sdf_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        sdf_grid = gaussian_filter(sdf_grid, sigma=sdf_smooth_sigma)
        if verbose:
            print(f"    Gaussian smooth: sigma={sdf_smooth_sigma:.1f}")

    if sdf_erosion > 0:
        sdf_grid = sdf_grid + sdf_erosion
        if verbose:
            print(f"    SDF erosion: {sdf_erosion:.4f}")

    # Force positive at boundaries
    bv = max(0.001, voxel_size)
    sdf_grid[ 0, :, :] = np.maximum(sdf_grid[ 0, :, :], bv)
    sdf_grid[-1, :, :] = np.maximum(sdf_grid[-1, :, :], bv)
    sdf_grid[:,  0, :] = np.maximum(sdf_grid[:,  0, :], bv)
    sdf_grid[:, -1, :] = np.maximum(sdf_grid[:, -1, :], bv)
    sdf_grid[:, :,  0] = np.maximum(sdf_grid[:, :,  0], bv)
    sdf_grid[:, :, -1] = np.maximum(sdf_grid[:, :, -1], bv)

    n_inside = (sdf_grid < level).sum()
    if verbose:
        print(f"    Inside voxels: {n_inside:,}")

    if n_inside < 10:
        if verbose:
            print("    No interior — returning None")
        return None

    # ── 10. Marching cubes ──
    try:
        verts, faces, _, _ = marching_cubes(
            sdf_grid, level=level, spacing=(voxel_size,) * 3,
        )
    except Exception as e:
        if verbose:
            print(f"    MC failed: {e}")
        return None

    verts = verts + bbox_min
    if len(verts) == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Remove artefact components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components.sort(key=lambda c: len(c.faces), reverse=True)
        mesh = components[0]
        if verbose:
            print(f"    Removed {len(components) - 1} artefact components")

    if verbose:
        print(f"    Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, "
              f"watertight={mesh.is_watertight}")

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Screened Poisson Reconstruction (with exact ring normals)
# ═══════════════════════════════════════════════════════════════════════════


def poisson_reconstruction(
    segment_data: list[dict],
    depth: int = 8,
    linear_fit: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Screened Poisson Surface Reconstruction from cross-section ring points.

    Uses exact outward normals (center → ring vertex) rather than estimated
    normals, giving significantly better reconstruction quality.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``.
    depth : octree depth for Poisson solver (8–10 typical).
    linear_fit : use linear interpolation in the Poisson solver.
    verbose : print progress.
    """
    import open3d as o3d
    from .interpolation import compute_transport_frames

    # ── 1. Build oriented point cloud from ring vertices ──
    points_list = []
    normals_list = []
    for seg in segment_data:
        centers = seg["centers"]
        rings = seg["rings"]
        for i in range(len(rings)):
            ring = rings[i]          # (P, 3) world coords
            center = centers[i]      # (3,)
            diff = ring - center     # (P, 3)
            dists = np.linalg.norm(diff, axis=1, keepdims=True)
            outward = diff / np.maximum(dists, 1e-10)
            points_list.append(ring)
            normals_list.append(outward)

    pts = np.vstack(points_list).astype(np.float64)
    nrm = np.vstack(normals_list).astype(np.float64)

    if verbose:
        print(f"    Point cloud: {len(pts):,} points")

    # ── 2. Open3D Screened Poisson ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)

    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=linear_fit,
    )

    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    if len(verts) == 0 or len(faces) == 0:
        if verbose:
            print("    Poisson returned empty mesh")
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Remove small artefact components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components.sort(key=lambda c: len(c.faces), reverse=True)
        mesh = components[0]
        if verbose:
            print(f"    Removed {len(components) - 1} artefact components")

    if verbose:
        print(f"    Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, "
              f"watertight={mesh.is_watertight}")

    return mesh


def poisson_perseg_reconstruction(
    segment_data: list[dict],
    depth: int = 8,
    n_cap_radial: int = 8,
    repoisson: bool = True,
    repoisson_depth: int = 9,
    repoisson_samples: int = 120_000,
    iso_remesh: bool = False,
    iso_remesh_factor: float = 2.0,
    iso_remesh_iter: int = 3,
    iso_taubin_iter: int = 30,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Per-segment Screened Poisson + Boolean Union + optional Re-Poisson.

    Each vessel segment is reconstructed independently with capped ends,
    then merged via manifold3d Boolean Union.  An optional final Re-Poisson
    pass smooths away the Boolean-Union seam artefacts.

    Caps are only placed at leaf / root endpoints — **not** at bifurcation
    junctions where segments naturally overlap.  This prevents the "cap bump"
    artefacts visible at branch points.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``.
    depth : octree depth for per-segment Poisson solvers.
    n_cap_radial : number of concentric cap-disk rings at tube ends.
    repoisson : if True, run a second Poisson pass on the union surface
        to smooth away Boolean-Union seam artefacts.
    repoisson_depth : octree depth for the smoothing Poisson pass.
    repoisson_samples : number of surface samples for the Re-Poisson pass.
    verbose : print progress.
    """
    import open3d as o3d
    import manifold3d

    # ── Detect bifurcation endpoints ─────────────────────────────────
    # Collect all segment start/end centers; any center shared by ≥2
    # segments is a bifurcation junction → no cap needed there.
    _all_endpoints = []  # (seg_idx, "start"|"end", center)
    for si, seg in enumerate(segment_data):
        c = seg["centers"]
        if len(c) >= 1:
            _all_endpoints.append((si, "start", c[0]))
            _all_endpoints.append((si, "end", c[-1]))

    _ep_centers = np.array([e[2] for e in _all_endpoints])  # (2*S, 3)
    _bif_set = set()  # (seg_idx, "start"|"end") pairs at bifurcations
    if len(_ep_centers) > 1:
        from scipy.spatial import cKDTree as _cKD
        _ep_tree = _cKD(_ep_centers)
        # find all pairs within a tight tolerance
        pairs = _ep_tree.query_pairs(r=1e-4)
        for i, j in pairs:
            _bif_set.add((_all_endpoints[i][0], _all_endpoints[i][1]))
            _bif_set.add((_all_endpoints[j][0], _all_endpoints[j][1]))

    def _make_cap(ring, center, outward_normal, n_radial):
        """Build a cap disk to close a tube end for watertight Poisson."""
        pts = [center.reshape(1, 3)]
        nrm = [(-outward_normal).reshape(1, 3)]
        for t in np.linspace(0.3, 0.9, n_radial):
            cap_ring = center + t * (ring - center)
            pts.append(cap_ring)
            nrm.append(np.full_like(cap_ring, -outward_normal))
        return np.vstack(pts), np.vstack(nrm)

    def _poisson_segment(seg, cap_start=True, cap_end=True):
        """Reconstruct one segment with Screened Poisson.
        Caps are conditionally added — skipped at bifurcation endpoints
        for cleaner junction geometry."""
        centers, rings = seg["centers"], seg["rings"]
        n = len(centers)
        if n < 3:
            return None
        pts_l, nrm_l = [], []
        for i in range(n):
            diff = rings[i] - centers[i]
            d = np.linalg.norm(diff, axis=1, keepdims=True)
            pts_l.append(rings[i])
            nrm_l.append(diff / np.maximum(d, 1e-10))
        # Start cap (skip at bifurcation endpoints)
        if cap_start:
            t0 = centers[1] - centers[0]
            t0 /= np.linalg.norm(t0) + 1e-10
            cp, cn = _make_cap(rings[0], centers[0], t0, n_cap_radial)
            pts_l.append(cp); nrm_l.append(cn)
        # End cap (skip at bifurcation endpoints)
        if cap_end:
            t1 = centers[-1] - centers[-2]
            t1 /= np.linalg.norm(t1) + 1e-10
            cp, cn = _make_cap(rings[-1], centers[-1], -t1, n_cap_radial)
            pts_l.append(cp); nrm_l.append(cn)

        pts = np.vstack(pts_l).astype(np.float64)
        nrm = np.vstack(nrm_l).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(nrm)
        try:
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth)
        except Exception:
            return None
        v = np.asarray(mesh_o3d.vertices)
        f = np.asarray(mesh_o3d.triangles)
        if len(f) == 0:
            return None
        m = trimesh.Trimesh(vertices=v, faces=f, process=True)
        comps = m.split(only_watertight=False)
        if len(comps) > 1:
            comps.sort(key=lambda c: len(c.faces), reverse=True)
            m = comps[0]
        return m if m.is_watertight else None

    # ── 1. Reconstruct each segment ──
    seg_meshes = []
    n_caps_skipped = 0
    n_caps_fallback = 0
    n_degenerate = 0
    for si, seg in enumerate(segment_data):
        # Skip degenerate segments (too few stations or zero centerline length)
        centers = seg["centers"]
        if len(centers) < 3:
            n_degenerate += 1
            if verbose:
                print(f"    [SKIP] Seg {si}: only {len(centers)} stations")
            continue
        cl_diffs = np.diff(centers, axis=0)
        cl_len = np.linalg.norm(cl_diffs, axis=1).sum()
        if cl_len < 1e-5:
            n_degenerate += 1
            if verbose:
                print(f"    [SKIP] Seg {si}: zero-length centerline ({cl_len:.2e})")
            continue
        cap_s = (si, "start") not in _bif_set
        cap_e = (si, "end") not in _bif_set
        if not cap_s:
            n_caps_skipped += 1
        if not cap_e:
            n_caps_skipped += 1
        m = _poisson_segment(seg, cap_start=cap_s, cap_end=cap_e)
        # Fallback: if capless mesh isn't watertight, retry with caps
        if m is None and (not cap_s or not cap_e):
            m = _poisson_segment(seg, cap_start=True, cap_end=True)
            if m is not None:
                n_caps_fallback += 1
        if m is not None:
            seg_meshes.append(m)

    if verbose:
        print(f"    Per-segment Poisson: {len(seg_meshes)}/{len(segment_data)} watertight tubes"
              f" ({n_caps_skipped} bifurcation caps skipped, {n_caps_fallback} fallbacks, "
              f"{n_degenerate} degenerate skipped)")

    if len(seg_meshes) < 1:
        return None

    # ── 2. Boolean Union via manifold3d ──
    def _to_manifold(mesh):
        return manifold3d.Manifold(
            mesh=manifold3d.Mesh(
                vert_properties=np.array(mesh.vertices, dtype=np.float32),
                tri_verts=np.array(mesh.faces, dtype=np.uint32),
            ))

    n_union_fail = 0
    combined = _to_manifold(seg_meshes[0])
    for si, m in enumerate(seg_meshes[1:], 1):
        try:
            combined = combined + _to_manifold(m)
        except Exception as e:
            n_union_fail += 1
            if verbose:
                print(f"    [WARN] Boolean union failed for segment {si}: {e}")

    if n_union_fail > 0 and verbose:
        print(f"    Union: {n_union_fail} segment(s) could not be merged")

    out = combined.to_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.array(out.vert_properties[:, :3]),
        faces=np.array(out.tri_verts),
        process=True,
    )

    comps = mesh.split(only_watertight=False)
    n_kept = 1
    if len(comps) > 1:
        comps.sort(key=lambda c: len(c.faces), reverse=True)
        # Keep all components with at least 10% of the largest one's face count
        max_faces = len(comps[0].faces)
        keep = [c for c in comps if len(c.faces) >= max_faces * 0.1]
        n_kept = len(keep)
        if len(keep) == 1:
            mesh = keep[0]
        else:
            mesh = trimesh.util.concatenate(keep)

    if verbose:
        print(f"    Union: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, "
              f"components kept={n_kept}/{len(comps)}, "
              f"watertight={mesh.is_watertight}")

    # ── 3. Optional Re-Poisson to smooth Boolean-Union seam artefacts ──
    if repoisson:
        try:
            # --- Localized Laplacian smoothing at bifurcation junctions ---
            # Target only vertices near bifurcation centers for aggressive
            # smoothing without affecting the rest of the mesh.
            if len(_bif_set) > 0:
                from collections import defaultdict as _ddict
                from scipy.spatial import cKDTree as _cKD2

                # Collect unique bifurcation centers + local radii
                _bif_info = []  # (center, radius)
                for si, end_type in _bif_set:
                    seg = segment_data[si]
                    c = seg["centers"]
                    r = seg["rings"]
                    if len(c) == 0:
                        continue
                    if end_type == "start":
                        bc, br = c[0], np.linalg.norm(r[0] - c[0], axis=1).mean()
                    else:
                        bc, br = c[-1], np.linalg.norm(r[-1] - c[-1], axis=1).mean()
                    _bif_info.append((bc, br))

                # Deduplicate close centers
                if _bif_info:
                    _bc_arr = np.array([b[0] for b in _bif_info])
                    _bt = _cKD2(_bc_arr)
                    _used = set()
                    _unique_bif = []
                    for idx in range(len(_bif_info)):
                        if idx not in _used:
                            bc, br = _bif_info[idx]
                            _unique_bif.append((bc, br))
                            for jdx in _bt.query_ball_point(bc, r=1e-3):
                                _used.add(jdx)

                    # Build vertex adjacency
                    verts = np.array(mesh.vertices, dtype=np.float64)
                    faces = np.array(mesh.faces)
                    adj = _ddict(set)
                    for face in faces:
                        for a, b in [(face[0], face[1]), (face[1], face[2]),
                                     (face[0], face[2])]:
                            adj[a].add(b)
                            adj[b].add(a)

                    # Localized Laplacian at each bifurcation
                    _vtree = _cKD2(verts)
                    _n_loc_iter = 25
                    for bc, br in _unique_bif:
                        _sr = br * 3.0  # smooth within 3× vessel radius
                        nearby = _vtree.query_ball_point(bc, _sr)
                        if not nearby:
                            continue
                        for _ in range(_n_loc_iter):
                            new_v = verts.copy()
                            for vi in nearby:
                                nbrs = adj[vi]
                                if not nbrs:
                                    continue
                                d_bif = np.linalg.norm(verts[vi] - bc)
                                w = max(0.0, 1.0 - (d_bif / _sr) ** 2)
                                avg = np.mean([verts[n] for n in nbrs], axis=0)
                                new_v[vi] = verts[vi] + w * 0.5 * (avg - verts[vi])
                            verts = new_v
                    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces,
                                           process=False)
                    if verbose:
                        print(f"    Localized bifurcation smoothing "
                              f"({len(_unique_bif)} junctions, {_n_loc_iter} iter)")

            # Global Taubin pre-smooth
            if len(_bif_set) > 0:
                m_o3d = o3d.geometry.TriangleMesh()
                m_o3d.vertices = o3d.utility.Vector3dVector(
                    np.asarray(mesh.vertices, dtype=np.float64))
                m_o3d.triangles = o3d.utility.Vector3iVector(
                    np.asarray(mesh.faces, dtype=np.int32))
                m_o3d = m_o3d.filter_smooth_taubin(
                    number_of_iterations=20,
                    lambda_filter=0.5, mu=-0.53,
                )
                mesh = trimesh.Trimesh(
                    vertices=np.asarray(m_o3d.vertices),
                    faces=np.asarray(m_o3d.triangles),
                    process=False,
                )
                if verbose:
                    print(f"    Global pre-smooth (Taubin 20 iter) before Re-Poisson")

            mesh.fix_normals()
            n_s = min(repoisson_samples, len(mesh.faces) * 2)
            sampled = trimesh.sample.sample_surface(mesh, n_s)
            sp = sampled[0].astype(np.float64)
            sn = mesh.face_normals[sampled[1]].astype(np.float64)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sp)
            pcd.normals = o3d.utility.Vector3dVector(sn)

            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=repoisson_depth)
            v = np.asarray(mesh_o3d.vertices)
            f = np.asarray(mesh_o3d.triangles)
            if len(f) > 0:
                mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
                comps = mesh.split(only_watertight=False)
                if len(comps) > 1:
                    comps.sort(key=lambda c: len(c.faces), reverse=True)
                    max_faces = len(comps[0].faces)
                    keep = [c for c in comps if len(c.faces) >= max_faces * 0.1]
                    if len(keep) == 1:
                        mesh = keep[0]
                    else:
                        mesh = trimesh.util.concatenate(keep)
        except Exception as e:
            if verbose:
                print(f"    [WARN] Re-Poisson failed, keeping union mesh: {e}")

        if verbose:
            print(f"    Re-Poisson (d={repoisson_depth}): {len(mesh.vertices):,} verts, "
                  f"{len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    # ── 4. Optional isotropic remesh + Taubin to remove ring-banding ──
    if iso_remesh:
        import pymeshlab, tempfile, os
        med_edge = float(np.median(mesh.edges_unique_length))
        target_len = med_edge * iso_remesh_factor
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            tmp_path = f.name
        try:
            mesh.export(tmp_path)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(tmp_path)
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PureValue(target_len),
                iterations=iso_remesh_iter,
                adaptive=False,
            )
            ms.save_current_mesh(tmp_path)
            if iso_taubin_iter > 0:
                m_o3d = o3d.io.read_triangle_mesh(tmp_path)
                m_o3d = m_o3d.filter_smooth_taubin(
                    number_of_iterations=iso_taubin_iter,
                    lambda_filter=0.5, mu=-0.53,
                )
                m_o3d.compute_vertex_normals()
                o3d.io.write_triangle_mesh(tmp_path, m_o3d)
            mesh = trimesh.load(tmp_path, process=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        if verbose:
            print(f"    IsoRemesh (f={iso_remesh_factor}, T={iso_taubin_iter}): "
                  f"{len(mesh.vertices):,} verts, watertight={mesh.is_watertight}")

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 10,
    lamb: float = 0.5,
    mu: float = -0.53,
) -> trimesh.Trimesh:
    """Taubin smoothing via Open3D."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    m = m.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=lamb,
        mu=mu,
    )
    return trimesh.Trimesh(
        vertices=np.asarray(m.vertices),
        faces=np.asarray(m.triangles),
        process=False,
    )


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int | None = None,
    ratio: float | None = None,
) -> trimesh.Trimesh:
    """Quadric-error decimation via Open3D."""
    import open3d as o3d

    n_faces = len(mesh.faces)
    if target_faces is None and ratio is not None:
        target_faces = max(100, int(n_faces * ratio))
    if target_faces is None or target_faces >= n_faces:
        return mesh

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return trimesh.Trimesh(
        vertices=np.asarray(m.vertices),
        faces=np.asarray(m.triangles),
        process=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Direct Lofting — triangle mesh from cross-section rings
# ═══════════════════════════════════════════════════════════════════════════


def _find_bifurcation_centers(
    segment_data: list[dict],
    tol: float = 1e-4,
    nearby_rings: int = 5,
) -> list[dict]:
    """Detect bifurcation points from segment endpoint topology.

    At a bifurcation the parent segment ends at the same node where ≥2
    child segments begin.  We cluster all segment start/end centers by
    proximity and report groups that involve ≥3 segments.

    For radius estimation, we scan up to *nearby_rings* rings along each
    segment from the bifurcation endpoint and take the **maximum** radius
    found.  This ensures aneurysm bulges (which may have very large
    cross-sections a few rings away from the junction) are captured.

    Returns list of dicts with keys:
        center : (3,)
        radius : float  — max ring radius near the junction
        segments : list of (seg_idx, "start"|"end") — which segments meet here
    """
    # Collect endpoints with their segment index and position in segment
    endpoints = []
    for seg_idx, seg in enumerate(segment_data):
        c = seg["centers"]
        r = seg["rings"]
        M = len(c)
        if M < 1:
            continue
        # Start of segment
        radii_start = []
        for k in range(min(nearby_rings, M)):
            radii_start.append(float(np.max(np.linalg.norm(r[k] - c[k], axis=1))))
        endpoints.append({
            "pos": c[0],
            "max_radius": max(radii_start),
            "seg_idx": seg_idx,
            "end_type": "start",
        })
        # End of segment
        radii_end = []
        for k in range(max(0, M - nearby_rings), M):
            radii_end.append(float(np.max(np.linalg.norm(r[k] - c[k], axis=1))))
        endpoints.append({
            "pos": c[-1],
            "max_radius": max(radii_end),
            "seg_idx": seg_idx,
            "end_type": "end",
        })
    if not endpoints:
        return []

    positions = np.array([e["pos"] for e in endpoints])
    tree = cKDTree(positions)

    visited = set()
    bifurcations = []
    for i in range(len(endpoints)):
        if i in visited:
            continue
        neighbors = tree.query_ball_point(positions[i], tol)
        if len(neighbors) >= 3:
            visited.update(neighbors)
            center = np.mean(positions[neighbors], axis=0)
            # Use the MAXIMUM radius across all meeting segment ends
            # This ensures aneurysm bulges are fully covered
            max_radius = max(endpoints[j]["max_radius"] for j in neighbors)
            seg_info = [(endpoints[j]["seg_idx"], endpoints[j]["end_type"])
                        for j in neighbors]
            bifurcations.append({
                "center": center,
                "radius": max_radius,
                "segments": seg_info,
            })
        else:
            visited.add(i)

    return bifurcations


def _local_sdf_junction(
    segment_data: list[dict],
    bif: dict,
    rings_range: int = 25,
    grid_resolution: int = 128,
    smooth_k: float = 0.015,
    padding: float = 0.01,
    verbose: bool = False,
) -> trimesh.Trimesh | None:
    """Build a smooth SDF mesh for one bifurcation region.

    Collects nearby rings from all segments meeting at the bifurcation,
    builds a local SDF grid with smooth-min blending, and extracts
    the isosurface with Marching Cubes.  The result is a smooth hull
    that naturally envelopes the overlapping tube geometry.
    """
    # Build virtual segment_data containing only rings near the bifurcation
    virtual_segments = []
    for seg_idx, end_type in bif["segments"]:
        seg = segment_data[seg_idx]
        c = seg["centers"]
        r = seg["rings"]
        M = len(c)
        if M < 2:
            continue

        n = min(rings_range, M)
        if end_type == "start":
            virtual_segments.append({
                "centers": c[:n],
                "rings": r[:n],
            })
        else:  # "end"
            virtual_segments.append({
                "centers": c[-n:],
                "rings": r[-n:],
            })

    if len(virtual_segments) < 2:
        return None

    # Reuse existing sdf_reconstruction on the local segments
    patch = sdf_reconstruction(
        virtual_segments,
        grid_resolution=grid_resolution,
        padding=padding,
        n_neighbors=3,
        smooth_k=smooth_k,
        level=0.0,
        narrow_band_factor=3.0,
        chunk_size=200_000,
        sdf_smooth_sigma=0.5,
        sdf_erosion=0.0,
        verbose=verbose,
    )
    return patch


def _smooth_bifurcation_regions(
    mesh: trimesh.Trimesh,
    bifurcation_centers: list[dict],
    radius_factor: float = 2.5,
    iterations: int = 30,
    lamb: float = 0.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Laplacian smoothing localised to bifurcation regions.

    Vertices far from any bifurcation center are *not moved at all*,
    preserving exact ring geometry on tube surfaces.  A cosine falloff
    blends smoothly between the fully-smoothed junction zone and the
    untouched tube body.
    """
    if not bifurcation_centers:
        return mesh

    from scipy.sparse import csr_matrix

    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    N = len(verts)

    # ── Per-vertex blend weight (cosine falloff) ─────────────────
    weight = np.zeros(N, dtype=np.float64)
    for bif in bifurcation_centers:
        c = bif["center"]
        R = bif["radius"] * radius_factor
        dist = np.linalg.norm(verts - c, axis=1)
        t = np.clip(dist / R, 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(np.pi * t))  # 1 at center, 0 at boundary
        weight = np.maximum(weight, w)

    n_affected = int(np.sum(weight > 1e-6))
    if n_affected == 0:
        return mesh
    if verbose:
        print(f"    Bifurcation smoothing: {len(bifurcation_centers)} junctions, "
              f"{n_affected}/{N} affected verts, {iterations} iters")

    # ── Sparse uniform-Laplacian matrix ──────────────────────────
    row = np.concatenate([faces[:, 0], faces[:, 0],
                          faces[:, 1], faces[:, 1],
                          faces[:, 2], faces[:, 2]])
    col = np.concatenate([faces[:, 1], faces[:, 2],
                          faces[:, 0], faces[:, 2],
                          faces[:, 0], faces[:, 1]])
    data = np.ones(len(row), dtype=np.float64)
    adj = csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj.minimum(1.0)

    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    D_inv = csr_matrix((1.0 / degree, (np.arange(N), np.arange(N))), shape=(N, N))
    L = D_inv @ adj  # row-stochastic averaging matrix

    # ── Iterative weighted smoothing ─────────────────────────────
    w = weight[:, None]  # (N, 1)
    for _ in range(iterations):
        smoothed = L @ verts
        verts = verts + lamb * w * (smoothed - verts)

    result = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
    result.fix_normals()
    return result


def _remesh_bifurcation_regions(
    mesh: trimesh.Trimesh,
    bifurcation_centers: list[dict],
    radius_factor: float = 3.0,
    target_edge_pct: float = 1.5,
    remesh_iterations: int = 3,
    smooth_iterations: int = 40,
    smooth_lambda: float = 0.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Isotropic remeshing of the *entire* mesh followed by localised
    Laplacian smoothing at bifurcation regions.

    Remeshing the whole mesh avoids the seam/watertightness problem that
    arises when only a sub-mesh is extracted and re-stitched.  The isotropic
    remeshing rebuilds the triangle connectivity uniformly, eliminating the
    sharp "seam" topology that Boolean Union creates.  Subsequent localised
    smoothing then works on uniform triangles and produces genuinely smooth
    surfaces.
    """
    if not bifurcation_centers:
        return mesh

    import pymeshlab

    # ── Compute target edge length from current mesh ─────────────
    edges = mesh.edges_unique
    edge_vecs = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    median_edge = float(np.median(edge_lengths))
    target_len = median_edge * target_edge_pct

    if verbose:
        print(f"    Remeshing whole mesh: {len(mesh.vertices)} verts, "
              f"{len(mesh.faces)} faces, target_edge={target_len:.5f}")

    # ── pymeshlab isotropic remeshing on the whole mesh ──────────
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(m)
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_len),
        iterations=remesh_iterations,
        adaptive=True,
    )

    # ── Repair non-manifold edges/vertices and close holes ────────
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_close_holes(maxholesize=100)

    remeshed = ms.current_mesh()
    result = trimesh.Trimesh(
        vertices=remeshed.vertex_matrix(),
        faces=remeshed.face_matrix(),
        process=True,
    )

    if verbose:
        print(f"    Remeshed: {len(result.vertices)} verts, "
              f"{len(result.faces)} faces, watertight={result.is_watertight}")

    # ── Localised Laplacian smoothing on bifurcation regions ─────
    if smooth_iterations > 0:
        result = _smooth_bifurcation_regions(
            result,
            bifurcation_centers,
            radius_factor=radius_factor,
            iterations=smooth_iterations,
            lamb=smooth_lambda,
            verbose=verbose,
        )

    return result


def _loft_single_segment(
    centers: np.ndarray,
    rings: list[np.ndarray],
) -> trimesh.Trimesh | None:
    """Loft one segment into a watertight capped tube mesh."""
    M = len(rings)
    if M < 2:
        return None
    K = rings[0].shape[0]

    ring_verts = np.array(rings, dtype=np.float64).reshape(-1, 3)  # (M*K, 3)
    extra_verts = [ring_verts]
    faces = []

    # Loft between consecutive rings
    for i in range(M - 1):
        base_a = i * K
        base_b = (i + 1) * K
        for j in range(K):
            j1 = (j + 1) % K
            faces.append([base_a + j, base_b + j, base_b + j1])
            faces.append([base_a + j, base_b + j1, base_a + j1])

    # Start cap
    cap_start_idx = M * K
    extra_verts.append(centers[0].reshape(1, 3))
    for j in range(K):
        j1 = (j + 1) % K
        faces.append([cap_start_idx, j1, j])

    # End cap
    cap_end_idx = M * K + 1
    extra_verts.append(centers[-1].reshape(1, 3))
    last = (M - 1) * K
    for j in range(K):
        j1 = (j + 1) % K
        faces.append([cap_end_idx, last + j, last + j1])

    verts = np.vstack(extra_verts)
    faces = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def loft_reconstruction(
    segment_data: list[dict],
    cap_ends: bool = True,
    bif_smooth_iterations: int = 30,
    bif_smooth_radius_factor: float = 2.5,
    bif_smooth_lambda: float = 0.5,
    complex_threshold: float = 1.3,
    complex_iterations: int = 120,
    complex_radius_factor: float = 4.0,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Build a triangle mesh by lofting each segment into a capped tube,
    then performing Boolean union to merge overlapping bifurcation areas.

    After the union, adaptive localised Laplacian smoothing is applied:
    - **Complex** bifurcations (aneurysms, large radius variation) get
      aggressive smoothing (many iterations, large zone).
    - **Simple** bifurcations get lighter smoothing.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``
        Each dict has ``"centers"`` (M, 3) and ``"rings"`` list[M] of (K, 3).
    cap_ends : bool
        Unused (kept for API compat).
    bif_smooth_iterations : int
        Laplacian smoothing iterations for simple bifurcations. 0 = off.
    bif_smooth_radius_factor : float
        Region radius = ring_radius × this factor (simple bifs).
    bif_smooth_lambda : float
        Laplacian smoothing step size.
    complex_threshold : float
        A bifurcation is "complex" when max_nearby_radius / min_endpoint_radius
        exceeds this value. Typical: 1.3.
    complex_iterations : int
        Smoothing iterations for complex bifurcations (aneurysms).
    complex_radius_factor : float
        Smoothing zone radius factor for complex bifurcations.
    verbose : bool
        Print progress.

    Returns
    -------
    trimesh.Trimesh or None.
    """
    total_rings = sum(len(s["rings"]) for s in segment_data)
    if total_rings < 2:
        return None

    # ── Build one watertight tube per segment ───────────────────────
    tubes = []
    for seg_idx, seg in enumerate(segment_data):
        tube = _loft_single_segment(seg["centers"], seg["rings"])
        if tube is not None and len(tube.faces) > 0:
            tubes.append(tube)

    if not tubes:
        return None

    if verbose:
        print(f"    Built {len(tubes)} tube segments, {total_rings} total rings")

    # ── Boolean union via manifold3d (keeps Manifold obj for smoothing) ──
    import manifold3d

    def _tri_to_manifold(tri_mesh):
        m3d_mesh = manifold3d.Mesh(
            vert_properties=np.asarray(tri_mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(tri_mesh.faces, dtype=np.uint32),
        )
        return manifold3d.Manifold(m3d_mesh)

    if len(tubes) == 1:
        manifold_obj = _tri_to_manifold(tubes[0])
    else:
        manifold_obj = _tri_to_manifold(tubes[0])
        fail_count = 0
        for i in range(1, len(tubes)):
            try:
                tube_m = _tri_to_manifold(tubes[i])
                manifold_obj = manifold_obj + tube_m  # Boolean union
            except Exception as e:
                if verbose and fail_count < 3:
                    print(f"    Boolean union failed for segment {i}: {e}")
                fail_count += 1
        if verbose and fail_count > 0:
            print(f"    {fail_count}/{len(tubes)-1} boolean unions failed")

    # ── Classify bifurcations ────────────────────────────────────────
    all_bifs = _find_bifurcation_centers(segment_data, nearby_rings=15)

    complex_bifs = []
    simple_bifs = []
    for bif in all_bifs:
        min_ep_r = float("inf")
        max_bulge_r = 0.0
        for seg_idx, end_type in bif["segments"]:
            seg = segment_data[seg_idx]
            c, r = seg["centers"], seg["rings"]
            M = len(c)
            scan = min(25, M)
            if end_type == "start":
                indices = range(scan)
            else:
                indices = range(max(0, M - scan), M)
            seg_radii = [float(np.max(np.linalg.norm(r[j] - c[j], axis=1)))
                         for j in indices]
            max_bulge_r = max(max_bulge_r, max(seg_radii))
            ep_idx = 0 if end_type == "start" else -1
            ep_r = float(np.max(np.linalg.norm(r[ep_idx] - c[ep_idx], axis=1)))
            min_ep_r = min(min_ep_r, ep_r)

        ratio = max_bulge_r / (min_ep_r + 1e-8)
        bif["complexity_ratio"] = ratio
        bif["max_bulge_radius"] = max_bulge_r

        if ratio > complex_threshold:
            complex_bifs.append(bif)
        else:
            simple_bifs.append(bif)

    if verbose:
        print(f"    Bifurcations: {len(all_bifs)} total, "
              f"{len(complex_bifs)} complex, {len(simple_bifs)} simple")

    # ── manifold3d → trimesh ────────────────────────────────────────
    # Convert directly — no global smooth_out (it distorts tube geometry).
    m3d_mesh = manifold_obj.to_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(m3d_mesh.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(m3d_mesh.tri_verts, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals()

    if verbose:
        print(f"    Union mesh: {len(mesh.vertices):,} verts, "
              f"{len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    # ── Localised Laplacian smoothing ────────────────────────────────
    # Light smoothing on simple bifurcations
    if bif_smooth_iterations > 0 and simple_bifs:
        mesh = _smooth_bifurcation_regions(
            mesh,
            simple_bifs,
            radius_factor=bif_smooth_radius_factor,
            iterations=bif_smooth_iterations,
            lamb=bif_smooth_lambda,
            verbose=verbose,
        )
    # Additional localized smoothing on complex bifurcations
    if complex_iterations > 0 and complex_bifs:
        for bif in complex_bifs:
            bif["radius"] = bif["max_bulge_radius"]
        mesh = _smooth_bifurcation_regions(
            mesh,
            complex_bifs,
            radius_factor=complex_radius_factor,
            iterations=complex_iterations,
            lamb=bif_smooth_lambda,
            verbose=verbose,
        )

    if verbose:
        print(f"    Loft mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"    Watertight: {mesh.is_watertight}")

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Loft + SDF reconstruction
# ═══════════════════════════════════════════════════════════════════════════


def hybrid_reconstruction(
    segment_data: list[dict],
    complexity_threshold: float = 2.0,
    sdf_rings_range: int = 25,
    sdf_grid_resolution: int = 128,
    sdf_smooth_k: float = 0.015,
    bif_smooth_iterations: int = 30,
    bif_smooth_radius_factor: float = 2.5,
    bif_smooth_lambda: float = 0.5,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Hybrid reconstruction: lofted tubes + local SDF patches at complex
    bifurcations (e.g. aneurysms).

    Strategy
    --------
    1. Detect all bifurcations and classify them:
       - **complex**: max_radius / min_radius > complexity_threshold
         → build a local SDF patch with smooth-min blending
       - **simple**: all radii similar
         → handle via Boolean Union + local Laplacian smoothing
    2. Loft all segments into capped tubes.
    3. For complex bifurcations, generate a smooth SDF hull mesh.
    4. Boolean Union: all tubes + all SDF patches.
    5. Local Laplacian smoothing at remaining simple bifurcations.

    This gives SDF-quality smooth blending at aneurysms while preserving
    exact ring geometry on normal tube surfaces.

    Parameters
    ----------
    segment_data : output of ``interpolate_tree_segments``.
    complexity_threshold : bifurcation is "complex" when the max/min
        radius ratio exceeds this value.
    sdf_rings_range : how many rings along each segment to include
        in the local SDF patch.
    sdf_grid_resolution : voxels per axis for local SDF patches.
    sdf_smooth_k : smooth-min blending radius for local SDF patches.
    bif_smooth_iterations : Laplacian smoothing iterations for simple
        bifurcations (after Boolean Union).
    bif_smooth_radius_factor : smoothing zone radius factor for simple bifs.
    bif_smooth_lambda : smoothing step size.
    verbose : print progress.
    """
    total_rings = sum(len(s["rings"]) for s in segment_data)
    if total_rings < 2:
        return None

    # ── 1. Detect & classify bifurcations ────────────────────────────
    all_bifs = _find_bifurcation_centers(segment_data, nearby_rings=10)

    complex_bifs = []
    simple_bifs = []
    for bif in all_bifs:
        # Scan nearby rings (not just the endpoint) to detect aneurysm
        # bulges that sit a few rings away from the junction.
        max_radii_per_seg = []
        min_endpoint_radius = float("inf")
        for seg_idx, end_type in bif["segments"]:
            seg = segment_data[seg_idx]
            c, r = seg["centers"], seg["rings"]
            M = len(c)
            # Check up to sdf_rings_range rings from the bifurcation end
            scan = min(sdf_rings_range, M)
            if end_type == "start":
                indices = range(scan)
            else:
                indices = range(max(0, M - scan), M)
            seg_radii = [float(np.max(np.linalg.norm(r[j] - c[j], axis=1)))
                         for j in indices]
            max_radii_per_seg.append(max(seg_radii))
            # Endpoint radius (at the actual bifurcation point)
            ep_idx = 0 if end_type == "start" else -1
            ep_r = float(np.max(np.linalg.norm(r[ep_idx] - c[ep_idx], axis=1)))
            min_endpoint_radius = min(min_endpoint_radius, ep_r)

        # Complex if ANY segment has a bulge much larger than the endpoint
        global_max = max(max_radii_per_seg) if max_radii_per_seg else 0
        ratio = global_max / (min_endpoint_radius + 1e-8)

        if verbose:
            print(f"      Bif at [{bif['center'][0]:.3f},{bif['center'][1]:.3f},{bif['center'][2]:.3f}]: "
                  f"ep_min_r={min_endpoint_radius:.4f}, max_bulge_r={global_max:.4f}, "
                  f"ratio={ratio:.2f} → {'COMPLEX' if ratio > complexity_threshold else 'simple'}")

        if ratio > complexity_threshold:
            complex_bifs.append(bif)
        else:
            simple_bifs.append(bif)

    if verbose:
        print(f"    Bifurcations: {len(all_bifs)} total, "
              f"{len(complex_bifs)} complex (SDF), "
              f"{len(simple_bifs)} simple (smooth)")

    # ── 2. Build lofted tubes ────────────────────────────────────────
    tubes = []
    for seg_idx, seg in enumerate(segment_data):
        tube = _loft_single_segment(seg["centers"], seg["rings"])
        if tube is not None and len(tube.faces) > 0:
            tubes.append(tube)

    if not tubes:
        return None

    if verbose:
        print(f"    Built {len(tubes)} tube segments, {total_rings} total rings")

    # ── 3. Build local SDF patches for complex bifurcations ──────────
    sdf_patches = []
    for bi, bif in enumerate(complex_bifs):
        if verbose:
            print(f"    Building SDF patch for complex bif {bi+1}/{len(complex_bifs)} "
                  f"(r={bif['radius']:.4f}, {len(bif['segments'])} segs)")
        patch = _local_sdf_junction(
            segment_data,
            bif,
            rings_range=sdf_rings_range,
            grid_resolution=sdf_grid_resolution,
            smooth_k=sdf_smooth_k,
            verbose=False,
        )
        if patch is not None and len(patch.faces) > 0:
            sdf_patches.append(patch)
            if verbose:
                print(f"      → {len(patch.vertices):,} verts, {len(patch.faces):,} faces")

    # ── 4. Boolean Union: all tubes + SDF patches ────────────────────
    all_parts = tubes + sdf_patches
    if verbose and sdf_patches:
        print(f"    Merging {len(tubes)} tubes + {len(sdf_patches)} SDF patches")

    mesh = all_parts[0]
    fail_count = 0
    for i in range(1, len(all_parts)):
        try:
            result = mesh.union(all_parts[i], engine="manifold")
            if result is not None and len(result.faces) > 0:
                mesh = result
            else:
                mesh = trimesh.util.concatenate([mesh, all_parts[i]])
                fail_count += 1
        except Exception as e:
            if verbose and fail_count < 3:
                print(f"    Boolean union failed for part {i}: {e}")
            mesh = trimesh.util.concatenate([mesh, all_parts[i]])
            fail_count += 1
    if verbose and fail_count > 0:
        print(f"    {fail_count}/{len(all_parts)-1} boolean unions failed")

    # ── 5. Local smoothing at simple bifurcations ────────────────────
    if bif_smooth_iterations > 0 and simple_bifs:
        mesh = _smooth_bifurcation_regions(
            mesh,
            simple_bifs,
            radius_factor=bif_smooth_radius_factor,
            iterations=bif_smooth_iterations,
            lamb=bif_smooth_lambda,
            verbose=verbose,
        )

    if verbose:
        print(f"    Hybrid mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"    Watertight: {mesh.is_watertight}")

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# VTK export helper
# ═══════════════════════════════════════════════════════════════════════════


def trimesh_to_vtk(mesh: trimesh.Trimesh):
    """Convert a trimesh.Trimesh to a VTK vtkPolyData."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh.faces, dtype=np.int64)

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(verts, deep=True))

    nf = len(faces_np)
    conn = np.column_stack([np.full(nf, 3, dtype=np.int64), faces_np]).ravel()
    cells = vtk.vtkCellArray()
    vtk_conn = numpy_to_vtkIdTypeArray(conn, deep=True)
    cells.SetCells(nf, vtk_conn)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetPolys(cells)
    return polydata
