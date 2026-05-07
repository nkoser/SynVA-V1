"""Repair tilted ring planes in the dense_v2 dataset.

Background
----------
Each row in the .npy is a 3D B-spline ring (8 CPs forming a closed
cross-section). Visualizing C0003 (and ~50% of all trees) showed regions
where consecutive rings are tilted by 60-90° relative to each other —
artifacts from the spline-fit step at sharp curvature / bifurcations,
not real anatomy.

This script walks each tree in the dataset and, for any ring whose
plane normal is more than `tilt_threshold` off from its parent ring's
normal, RE-PROJECTS the 8 ring CPs onto a plane interpolated between
the parent's plane and the local centerline tangent. Centroid and mean
radius of the original ring are preserved.

Inputs : the prepared dataset (e.g. ..._biffilter_v2_norm/{train,val,test})
Outputs: a new dataset with `_aligned` suffix, identical layout.

Usage
-----
    python Preprocessing_modular_v2/align_ring_planes.py --config <yaml>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml


# Layout: cols 4-11 = cx[0..7], 12-19 = cy[0..7], 20-27 = cz[0..7]
def cps_3d(row):
    return np.column_stack([row[4:12], row[12:20], row[20:28]]).astype(np.float64)


def write_cps(row, cps):
    row = row.copy()
    row[4:12]  = cps[:, 0]
    row[12:20] = cps[:, 1]
    row[20:28] = cps[:, 2]
    return row


def ring_plane(cps):
    """Return centroid (3,), unit normal (3,), mean radius (scalar), planarity."""
    centroid = cps.mean(axis=0)
    centered = cps - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    radius = float(np.linalg.norm(centered, axis=1).mean())
    # Planarity: ratio of smallest to median singular value (0=perfect plane)
    planarity = float(s[-1] / max(s[1], 1e-12))
    return centroid, normal, radius, planarity


def align_normal_sign(parent_normal, candidate_normal):
    """SVD normals have arbitrary sign. Flip if dot < 0 with parent."""
    if np.dot(parent_normal, candidate_normal) < 0:
        return -candidate_normal
    return candidate_normal


def project_ring_to_plane(cps, target_centroid, target_normal):
    """Take the original cps, re-orient them onto a new plane through
    target_centroid with target_normal, preserving the in-plane shape and
    mean radius."""
    target_normal = target_normal / max(np.linalg.norm(target_normal), 1e-12)
    cps_centered = cps - cps.mean(axis=0)

    # Build orthonormal basis for the original ring plane (use first two
    # singular vectors as in-plane axes)
    _, _, Vt_old = np.linalg.svd(cps_centered, full_matrices=False)
    old_u = Vt_old[0]
    old_v = Vt_old[1]
    # Compute in-plane (u, v) coordinates of each CP
    uv = np.column_stack([cps_centered @ old_u, cps_centered @ old_v])  # [8, 2]

    # Build new in-plane basis (u', v') orthogonal to target_normal,
    # trying to keep angular alignment with the old basis where possible
    # so we don't introduce a twist.
    # Pick u' = projection of old_u onto plane(target_normal), normalized.
    proj = old_u - np.dot(old_u, target_normal) * target_normal
    nproj = np.linalg.norm(proj)
    if nproj < 1e-8:
        # old_u was almost parallel to target_normal → fall back to old_v
        proj = old_v - np.dot(old_v, target_normal) * target_normal
        nproj = np.linalg.norm(proj)
        if nproj < 1e-8:
            # Degenerate case: pick an arbitrary basis
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, target_normal)) > 0.95:
                tmp = np.array([0.0, 1.0, 0.0])
            proj = tmp - np.dot(tmp, target_normal) * target_normal
            nproj = np.linalg.norm(proj)
    new_u = proj / nproj
    new_v = np.cross(target_normal, new_u)

    # Reconstruct cps in 3D using same (u, v) coords but in the new basis
    new_cps = target_centroid + uv[:, 0:1] * new_u + uv[:, 1:2] * new_v
    return new_cps


def smooth_tree_rings(arr, parents, tilt_cos_threshold=0.5, tangent_blend=0.5):
    """Walk preorder; for each ring whose normal is too tilted vs its parent's,
    re-project into an interpolated plane between parent normal and local
    centerline tangent. Returns aligned arr (copy)."""
    out = arr.copy()
    N = arr.shape[0]
    centroids = np.zeros((N, 3))
    normals = np.zeros((N, 3))
    radii = np.zeros(N)

    # First pass: compute current planes
    for i in range(N):
        c, n, r, _ = ring_plane(cps_3d(arr[i]))
        centroids[i] = c
        normals[i] = n
        radii[i] = r

    # Second pass: walk topologically (parents already provided), align signs,
    # and re-project tilted rings.
    n_aligned = 0
    aligned_indices = []
    for i in range(N):
        p = int(parents[i]) if parents is not None else (i - 1 if i > 0 else -1)
        if p < 0 or p >= N:
            continue
        # Align sign first (SVD ambiguity)
        normals[i] = align_normal_sign(normals[p], normals[i])
        cos = float(np.dot(normals[p], normals[i]))
        if cos >= tilt_cos_threshold:
            continue

        # Compute target normal: blend between parent's normal and local
        # tangent (child centroid - parent centroid)
        tangent = centroids[i] - centroids[p]
        ntg = np.linalg.norm(tangent)
        if ntg < 1e-8:
            target = normals[p].copy()
        else:
            tangent = tangent / ntg
            tangent = align_normal_sign(normals[p], tangent)
            target = (1.0 - tangent_blend) * normals[p] + tangent_blend * tangent
            ntn = np.linalg.norm(target)
            if ntn < 1e-8:
                target = normals[p].copy()
            else:
                target = target / ntn

        # Re-project the ring onto the new plane (centroid stays the same,
        # in-plane shape stays the same, plane orientation changes)
        new_cps = project_ring_to_plane(cps_3d(arr[i]), centroids[i], target)
        # Preserve original mean radius (project_ring_to_plane should do
        # this since it re-uses (u, v) coords, but enforce explicitly)
        cur_r = float(np.linalg.norm(new_cps - new_cps.mean(0), axis=1).mean())
        if cur_r > 1e-8:
            new_cps = (new_cps - new_cps.mean(0)) * (radii[i] / cur_r) + centroids[i]

        out[i] = write_cps(arr[i], new_cps)
        normals[i] = target
        n_aligned += 1
        aligned_indices.append(i)

    return out, n_aligned, aligned_indices


def derive_parents_preorder(arr):
    """Without explicit parent indices, infer them from the preorder layout
    using the k_count column (col 0): each k=2 has 2 children, k=1 has 1,
    k=0 has 0. Standard preorder traversal."""
    k = arr[:, 0].astype(int)
    parents = np.full(len(arr), -1, dtype=int)
    stack = []  # (node_idx, remaining_children)
    for i in range(len(arr)):
        if stack:
            parent_idx, rem = stack[-1]
            parents[i] = parent_idx
            stack[-1] = (parent_idx, rem - 1)
            if stack[-1][1] == 0:
                stack.pop()
        ki = k[i]
        if ki == 1:
            stack.append((i, 1))
        elif ki == 2:
            stack.append((i, 2))
    return parents


def process_split(in_dir: Path, out_dir: Path, tilt_threshold: float, blend: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npy"))
    total_rings = 0
    total_aligned = 0
    trees_aligned = 0
    for fp in files:
        arr = np.load(fp, allow_pickle=True)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 40)
        parents = derive_parents_preorder(arr)
        out_arr, n_aligned, _ = smooth_tree_rings(
            arr, parents,
            tilt_cos_threshold=tilt_threshold,
            tangent_blend=blend,
        )
        np.save(out_dir / fp.name, out_arr.astype(np.float32))
        total_rings += len(arr)
        total_aligned += n_aligned
        if n_aligned > 0:
            trees_aligned += 1
    return {
        "n_files": len(files),
        "total_rings": total_rings,
        "total_aligned": total_aligned,
        "trees_aligned": trees_aligned,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    params = cfg.get("params", {})
    tilt = float(params.get("tilt_cos_threshold", 0.5))
    blend = float(params.get("tangent_blend", 0.5))
    in_train = Path(paths["train_dir"])
    in_val = Path(paths["val_dir"])
    in_test = Path(paths["test_dir"])
    out_root = Path(paths["output_root"])

    print(f"tilt_cos_threshold = {tilt}  (rings with |cos| < this trigger alignment)")
    print(f"tangent_blend      = {blend}  (0=use parent normal, 1=use local tangent)")
    print()
    for name, in_dir in (("train", in_train), ("val", in_val), ("test", in_test)):
        out_dir = out_root / name
        stats = process_split(in_dir, out_dir, tilt, blend)
        print(f"  [{name:5s}] files={stats['n_files']}  rings={stats['total_rings']:>10d}  "
              f"aligned={stats['total_aligned']:>8d} ({100*stats['total_aligned']/max(stats['total_rings'],1):.2f}%)  "
              f"trees_with_alignment={stats['trees_aligned']}")


if __name__ == "__main__":
    main()
