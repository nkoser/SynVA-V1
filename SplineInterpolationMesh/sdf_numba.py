"""Numba-accelerated SDF computation for vessel mesh reconstruction."""

import numpy as np
import numba as nb


@nb.njit
def _polygon_sdf_single(qx, qy, poly):
    """Signed distance from a single 2D point to a polygon."""
    P = len(poly)
    min_dist_sq = 1e30
    inside_count = 0
    for i in range(P):
        j = (i + 1) % P
        ax, ay = poly[i, 0], poly[i, 1]
        bx, by = poly[j, 0], poly[j, 1]
        ex, ey = bx - ax, by - ay
        fx, fy = qx - ax, qy - ay
        e_dot = ex * ex + ey * ey
        t = (fx * ex + fy * ey) / (e_dot + 1e-30)
        t = max(0.0, min(1.0, t))
        dx = fx - t * ex
        dy = fy - t * ey
        d_sq = dx * dx + dy * dy
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
        if (ay <= qy and by > qy) or (by <= qy and ay > qy):
            t_inter = (qy - ay) / (by - ay + 1e-30)
            x_inter = ax + t_inter * ex
            if x_inter > qx:
                inside_count += 1
    dist = min_dist_sq ** 0.5
    if inside_count % 2 == 1:
        return -dist
    return dist


@nb.njit(parallel=True)
def compute_sdf_batch(pts, nn_edges, K_e, edges, centers, normals,
                      binormals, rings_2d, smooth_k):
    """Compute sweep-based SDF for a batch of 3D points.

    Parameters
    ----------
    pts : (N, 3)  query points
    nn_edges : (N, K_e)  nearest edge indices per point
    K_e : int  number of neighbor edges
    edges : (E, 2)  station index pairs
    centers, normals, binormals : (S, 3)  station data
    rings_2d : (S, P, 2)  2D ring polygons per station
    smooth_k : float  smooth-min blending radius

    Returns
    -------
    (N,) SDF values (negative = inside)
    """
    N = len(pts)
    sdf = np.full(N, 1e6, dtype=np.float64)
    P_ring = rings_2d.shape[1]

    for i in nb.prange(N):
        s_val = 1e6
        for ki in range(K_e):
            eid = nn_edges[i, ki]
            ia = edges[eid, 0]
            ib = edges[eid, 1]

            ex = centers[ib, 0] - centers[ia, 0]
            ey = centers[ib, 1] - centers[ia, 1]
            ez = centers[ib, 2] - centers[ia, 2]
            el = (ex * ex + ey * ey + ez * ez) ** 0.5
            if el < 1e-12:
                continue

            edx, edy, edz = ex / el, ey / el, ez / el
            dx = pts[i, 0] - centers[ia, 0]
            dy = pts[i, 1] - centers[ia, 1]
            dz = pts[i, 2] - centers[ia, 2]
            t_proj = dx * edx + dy * edy + dz * edz
            t = max(0.0, min(1.0, t_proj / el))

            cx = centers[ia, 0] + t * ex
            cy = centers[ia, 1] + t * ey
            cz = centers[ia, 2] + t * ez

            nx = (1 - t) * normals[ia, 0] + t * normals[ib, 0]
            ny = (1 - t) * normals[ia, 1] + t * normals[ib, 1]
            nz = (1 - t) * normals[ia, 2] + t * normals[ib, 2]
            n_len = (nx * nx + ny * ny + nz * nz) ** 0.5
            if n_len > 1e-12:
                nx /= n_len
                ny /= n_len
                nz /= n_len

            bx = (1 - t) * binormals[ia, 0] + t * binormals[ib, 0]
            by = (1 - t) * binormals[ia, 1] + t * binormals[ib, 1]
            bz = (1 - t) * binormals[ia, 2] + t * binormals[ib, 2]
            dot_nb = bx * nx + by * ny + bz * nz
            bx -= dot_nb * nx
            by -= dot_nb * ny
            bz -= dot_nb * nz
            b_len = (bx * bx + by * by + bz * bz) ** 0.5
            if b_len > 1e-12:
                bx /= b_len
                by /= b_len
                bz /= b_len

            px = pts[i, 0] - cx
            py = pts[i, 1] - cy
            pz = pts[i, 2] - cz
            u = px * nx + py * ny + pz * nz
            v = px * bx + py * by + pz * bz

            ring = np.empty((P_ring, 2), dtype=np.float64)
            for p in range(P_ring):
                ring[p, 0] = (1 - t) * rings_2d[ia, p, 0] + t * rings_2d[ib, p, 0]
                ring[p, 1] = (1 - t) * rings_2d[ia, p, 1] + t * rings_2d[ib, p, 1]

            s = _polygon_sdf_single(u, v, ring)

            if smooth_k > 0:
                h = max(0.0, min(1.0, 0.5 + 0.5 * (s - s_val) / smooth_k))
                s_val = s * (1 - h) + s_val * h - smooth_k * h * (1 - h)
            else:
                s_val = min(s_val, s)

        sdf[i] = s_val
    return sdf
