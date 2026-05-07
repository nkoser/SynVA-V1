"""Cross-section-aware signed distance field evaluation.

For each query point:
  1. Find K nearest centerline stations via KD-tree.
  2. Project the point into each station's local (N, B) plane.
  3. Convert to polar (r, θ) and look up the cross-section radius R(θ).
  4. Local SDF = r − R(θ)  (positive outside, negative inside).
  5. Combine all K candidates with smooth-min for bifurcation blending.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List
from .interpolate import DenseSegment


def _smooth_min_poly(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-min (Inigo Quilez).  k controls blend radius."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


class CrossSectionSDF:
    """Evaluates the SDF of a vessel tree using swept cross-sections.

    Parameters
    ----------
    segments : list of DenseSegment
        Pre-interpolated vessel segments.
    smooth_k : float
        Smooth-min blending parameter.  0 = hard min (sharp intersections).
        0.005 is a good default for ~mm-scale vessels.
    k_neighbors : int
        Number of nearest centerline stations to check per query point.
    """

    def __init__(
        self,
        segments: List[DenseSegment],
        smooth_k: float = 0.005,
        k_neighbors: int = 5,
    ):
        self.smooth_k = smooth_k
        self.k_neighbors = k_neighbors

        # Concatenate all station data into flat arrays
        all_c, all_t, all_n, all_b, all_r = [], [], [], [], []
        for seg in segments:
            all_c.append(seg.centers)
            all_t.append(seg.tangents)
            all_n.append(seg.normals)
            all_b.append(seg.binormals)
            all_r.append(seg.radii_table)

        self.centers = np.vstack(all_c)      # (T, 3)
        self.tangents = np.vstack(all_t)     # (T, 3)
        self.normals = np.vstack(all_n)      # (T, 3)
        self.binormals = np.vstack(all_b)    # (T, 3)
        self.radii = np.vstack(all_r)        # (T, M)
        self.n_angular = self.radii.shape[1]
        self.max_radius = float(np.max(self.radii))

        # KD-tree over all station centers
        self.tree = cKDTree(self.centers)
        self._n_stations = len(self.centers)

    def bounds(self, padding: float = 0.05):
        """Axis-aligned bounding box (lo, hi) with padding."""
        pad = self.max_radius + padding
        lo = self.centers.min(axis=0) - pad
        hi = self.centers.max(axis=0) + pad
        return lo, hi

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate SDF at query points.

        Parameters
        ----------
        points : (Q, 3) query positions.

        Returns
        -------
        (Q,) signed distance (negative inside vessel).
        """
        Q = len(points)
        K = min(self.k_neighbors, self._n_stations)

        # Find K nearest stations
        _, indices = self.tree.query(points, k=K, workers=-1)
        if K == 1:
            indices = indices[:, None]

        # Evaluate SDF for each of the K neighbors and combine
        sdf = np.full(Q, 1e10, dtype=np.float64)
        qi = np.arange(Q)

        for ki in range(K):
            idx = indices[:, ki]  # (Q,) station indices

            # Gather station data
            C = self.centers[idx]     # (Q, 3)
            N = self.normals[idx]     # (Q, 3)
            B = self.binormals[idx]   # (Q, 3)
            R = self.radii[idx]       # (Q, M)

            # Project into local cross-section plane
            offset = points - C
            d_n = np.einsum("ij,ij->i", offset, N)  # (Q,)
            d_b = np.einsum("ij,ij->i", offset, B)  # (Q,)

            # Polar coordinates
            r = np.sqrt(d_n * d_n + d_b * d_b)
            theta = np.arctan2(d_b, d_n) % (2 * np.pi)

            # Angular radius lookup with linear interpolation
            M = self.n_angular
            theta_frac = theta * (M / (2 * np.pi))
            idx_lo = np.floor(theta_frac).astype(np.intp) % M
            idx_hi = (idx_lo + 1) % M
            frac = theta_frac - np.floor(theta_frac)

            R_theta = R[qi, idx_lo] * (1.0 - frac) + R[qi, idx_hi] * frac

            # Local SDF: positive outside, negative inside
            sdf_local = r - R_theta

            # Combine via smooth-min
            if self.smooth_k > 0:
                sdf = _smooth_min_poly(sdf, sdf_local, self.smooth_k)
            else:
                sdf = np.minimum(sdf, sdf_local)

        return sdf

    def evaluate_grid(self, step: float = 0.004, batch_slices: int = 40, verbose: bool = True):
        """Evaluate SDF on a regular 3-D grid with sparse acceleration.

        Only evaluates points within `max_radius + margin` of the nearest
        centerline station (typically ~5-10% of the grid).

        Parameters
        ----------
        step : float
            Grid spacing.
        batch_slices : int
            Number of X-slices to process per batch.
        verbose : bool
            Print progress info.

        Returns
        -------
        volume : (Nx, Ny, Nz) float32 array.
        origin : (3,) float64 — world position of grid[0,0,0].
        """
        lo, hi = self.bounds()
        shape = tuple(np.ceil((hi - lo) / step).astype(int) + 1)
        Nx, Ny, Nz = shape
        total_pts = Nx * Ny * Nz

        gx = lo[0] + np.arange(Nx) * step
        gy = lo[1] + np.arange(Ny) * step
        gz = lo[2] + np.arange(Nz) * step

        volume = np.ones(shape, dtype=np.float32)
        search_radius = self.max_radius + step * 3
        total_evaluated = 0

        if verbose:
            print(f"  Grid {Nx}×{Ny}×{Nz} = {total_pts:,} pts, step={step}")

        for ix0 in range(0, Nx, batch_slices):
            ix1 = min(ix0 + batch_slices, Nx)
            gx_batch = gx[ix0:ix1]

            # Build coordinates for this slab
            xx, yy, zz = np.meshgrid(gx_batch, gy, gz, indexing="ij")
            pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

            # Sparse filter: only evaluate near the vessel
            dists_nn, _ = self.tree.query(pts, k=1, workers=-1)
            mask = dists_nn < search_radius
            n_active = int(mask.sum())

            if n_active > 0:
                sdf_vals = np.ones(len(pts), dtype=np.float32)
                sdf_vals[mask] = self.evaluate(pts[mask]).astype(np.float32)
                volume[ix0:ix1] = sdf_vals.reshape(ix1 - ix0, Ny, Nz)
                total_evaluated += n_active

            if verbose:
                pct = 100 * ix1 / Nx
                print(f"\r  [{ix1}/{Nx}] {pct:.0f}%  ({total_evaluated:,} active)", end="", flush=True)

        if verbose:
            pct_active = 100 * total_evaluated / total_pts if total_pts > 0 else 0
            print(f"\n  Evaluated {total_evaluated:,} / {total_pts:,} pts ({pct_active:.1f}% active)")

        return volume, lo
