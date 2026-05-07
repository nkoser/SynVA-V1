#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
from scipy.interpolate import splev

# Canonical decoders for the relpos / nodecp encoding
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tree_functions import (  # noqa: E402
    local_geometry_tree_to_absolute,
    preorder_kcount_parent_indices,
)


def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)


class SimpleNode:
    def __init__(self, row, meta):
        self.row = row
        ps = meta['position_slice']
        cps = meta['control_point_slices']
        self.k_count = int(round(float(row[0])))
        self.abs_pos = np.array(row[ps[0]:ps[1]], dtype=np.float32)
        self.cp_x = np.array(row[cps[0][0]:cps[0][1]], dtype=np.float32)
        self.cp_y = np.array(row[cps[1][0]:cps[1][1]], dtype=np.float32)
        self.cp_z = np.array(row[cps[2][0]:cps[2][1]], dtype=np.float32)
        self.knots = np.array(row[cps[2][1]:], dtype=np.float32)
        self.left = None
        self.right = None


def decode_to_absolute(arr, meta):
    """Convert a stored tree array (which may use parent-relative positions
    and/or node-local control points) into a fully absolute representation."""
    return local_geometry_tree_to_absolute(
        arr,
        position_slice=tuple(meta.get('position_slice', (1, 4))),
        control_point_slices=tuple(tuple(s) for s in meta.get('control_point_slices', ((4, 12), (12, 20), (20, 28)))),
        relative_positions=bool(meta.get('relative_positions', False)),
        node_local_control_points=bool(meta.get('node_local_control_points', False)),
        copy=True,
    )


def build_tree_from_array(arr, meta):
    """Build a SimpleNode tree using pre-order k-count serialization.

    Column 0 stores the number of children (0/1/2). The first row is the root,
    children follow in pre-order. left child comes before right child.
    """
    parents = preorder_kcount_parent_indices(arr[:, 0])
    nodes = [SimpleNode(arr[i], meta) for i in range(arr.shape[0])]
    for idx, parent in enumerate(parents.tolist()):
        if parent < 0:
            continue
        p = nodes[parent]
        if p.left is None:
            p.left = nodes[idx]
        else:
            p.right = nodes[idx]
    root = nodes[0] if nodes else None
    return root


def collect_nodes(root):
    out = []

    def dfs(n):
        if n is None:
            return
        out.append(n)
        dfs(n.left)
        dfs(n.right)

    dfs(root)
    return out


def resample_closed_curve(pts, n):
    pts = np.asarray(pts)
    if pts.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    # compute segment lengths around closed loop
    N = pts.shape[0]
    segs = np.linalg.norm(pts[(np.arange(N) + 1) % N] - pts[np.arange(N)], axis=1)
    cum = np.concatenate(([0.0], np.cumsum(segs)))
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1))
    targets = np.linspace(0.0, total, n + 1)[:-1]
    new_pts = np.zeros((n, 3), dtype=np.float32)
    for i, t in enumerate(targets):
        j = np.searchsorted(cum, t) - 1
        if j < 0:
            j = 0
        t0 = cum[j]
        t1 = cum[j + 1]
        p0 = pts[j]
        p1 = pts[(j + 1) % N]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        new_pts[i] = p0 * (1.0 - alpha) + p1 * alpha
    return new_pts


def sample_spline(cp_x, cp_y, cp_z, knots, k=3, samples=32):
    try:
        t = np.asarray(knots, dtype=np.float64)
        cps = [np.asarray(cp_x, dtype=np.float64), np.asarray(cp_y, dtype=np.float64), np.asarray(cp_z, dtype=np.float64)]
        u_lo = float(t[k])
        u_hi = float(t[-k - 1])
        if not np.isfinite(u_lo) or not np.isfinite(u_hi) or u_hi <= u_lo:
            pts = np.vstack([cp_x, cp_y, cp_z]).T
            return resample_closed_curve(pts, samples)
        u = np.linspace(u_lo, u_hi, samples, endpoint=False)
        pts = np.array(splev(u, (t, cps, k))).T
        if pts.shape[0] != samples:
            pts = resample_closed_curve(pts, samples)
        return pts.astype(np.float32)
    except Exception:
        pts = np.vstack([cp_x, cp_y, cp_z]).T
        return resample_closed_curve(pts, samples)


def build_mesh_from_tree(root, samples_per_ring=32):
    nodes = collect_nodes(root)
    vertices = []
    ring_starts = {}
    ns = samples_per_ring

    # create ring vertices per node (CPs are already absolute after decoding)
    for i, n in enumerate(nodes):
        ring = sample_spline(n.cp_x, n.cp_y, n.cp_z, n.knots, samples=ns)
        start_idx = len(vertices) + 1
        for p in ring:
            vertices.append(p)
        ring_starts[id(n)] = (start_idx, ns)

    faces = []
    visited_edges = set()

    def connect_rings(n1, n2):
        key = tuple(sorted((id(n1), id(n2))))
        if key in visited_edges:
            return
        visited_edges.add(key)
        s1, m1 = ring_starts[id(n1)]
        s2, m2 = ring_starts[id(n2)]
        # if samples differ, we assume same m1==m2
        m = min(m1, m2)
        for k in range(m):
            a = s1 + k
            b = s1 + (k + 1) % m
            c = s2 + (k + 1) % m
            d = s2 + k
            faces.append((a, b, c))
            faces.append((a, c, d))

    # traverse tree and connect parent-child
    for n in nodes:
        if n.left is not None:
            connect_rings(n, n.left)
        if n.right is not None:
            connect_rings(n, n.right)

    return np.array(vertices, dtype=np.float32), faces


def write_obj_mesh(vertices, faces, out_path):
    with open(out_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write('f {} {} {}\n'.format(*face))


def process_folder(input_dir, meta_path, output_dir, samples_per_ring=32, subset=None):
    meta = load_metadata(meta_path)
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    if subset:
        files = [f for f in files if f in subset]
    for idx, fname in enumerate(files, 1):
        path = os.path.join(input_dir, fname)
        arr = np.load(path, allow_pickle=True)
        # Drop legacy all-zero padding rows if present
        nz_mask = ~np.all(arr == 0.0, axis=1)
        arr = arr[nz_mask]
        abs_arr = decode_to_absolute(arr, meta)
        root = build_tree_from_array(abs_arr, meta)
        verts, faces = build_mesh_from_tree(root, samples_per_ring=samples_per_ring)
        base = os.path.splitext(fname)[0]
        out_mesh = os.path.join(output_dir, base + '_mesh.obj')
        write_obj_mesh(verts, faces, out_mesh)
        print(f'Wrote mesh: {out_mesh} ({idx}/{len(files)})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='derived_data/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1/test')
    p.add_argument('--metadata', default='derived_data/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1/local_geometry_metadata.json')
    p.add_argument('--output_dir', default='derived_data_visualizations/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1_meshes')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--cases', nargs='*')
    args = p.parse_args()
    subset = None
    if args.cases:
        subset = args.cases
    process_folder(args.input_dir, args.metadata, args.output_dir, samples_per_ring=args.samples, subset=subset)


if __name__ == '__main__':
    main()
