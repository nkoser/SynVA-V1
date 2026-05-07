#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from scipy.interpolate import splev


def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)


class SimpleNode:
    def __init__(self, row, meta):
        self.row = row
        self.meta = meta
        ps = meta['position_slice']
        cps = meta['control_point_slices']
        self.flag = float(row[0])
        self.local_pos = np.array(row[ps[0]:ps[1]], dtype=np.float32)
        # control points: assume three slices for x,y,z
        self.cp_x = np.array(row[cps[0][0]:cps[0][1]], dtype=np.float32)
        self.cp_y = np.array(row[cps[1][0]:cps[1][1]], dtype=np.float32)
        self.cp_z = np.array(row[cps[2][0]:cps[2][1]], dtype=np.float32)
        self.knots = np.array(row[cps[2][1]:], dtype=np.float32)
        self.left = None
        self.right = None
        self.abs_pos = None


def deserialize_from_array(arr, meta):
    # arr: (N, C) numpy array, post-order with zero-rows as None markers
    idx = arr.shape[0] - 1

    def helper():
        nonlocal idx
        if idx < 0:
            return None
        row = arr[idx]
        idx -= 1
        if np.allclose(row, 0.0, atol=1e-9):
            return None
        node = SimpleNode(row, meta)
        # post-order: node, then right, then left were pushed; so pop right then left
        node.right = helper()
        node.left = helper()
        return node

    root = helper()
    return root


def compute_absolute_positions(root, relative=True, parent_pos=None):
    if root is None:
        return
    if parent_pos is None:
        root.abs_pos = root.local_pos.copy()
    else:
        if relative:
            root.abs_pos = parent_pos + root.local_pos
        else:
            root.abs_pos = root.local_pos.copy()
    compute_absolute_positions(root.left, relative, root.abs_pos)
    compute_absolute_positions(root.right, relative, root.abs_pos)


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


def sample_spline(cp_x, cp_y, cp_z, knots, k=3, samples=64):
    try:
        # knots should be an array-like of length n_cp + k + 1 (canonical periodic)
        t = np.asarray(knots, dtype=np.float64)
        cps = [np.asarray(cp_x, dtype=np.float64), np.asarray(cp_y, dtype=np.float64), np.asarray(cp_z, dtype=np.float64)]
        # valid u domain
        u_lo = float(t[k])
        u_hi = float(t[-k - 1])
        if not np.isfinite(u_lo) or not np.isfinite(u_hi) or u_hi <= u_lo:
            # degenerate, fallback to control points directly
            pts = np.vstack([cp_x, cp_y, cp_z]).T
            return pts
        u = np.linspace(u_lo, u_hi, samples, endpoint=False)
        pts = np.array(splev(u, (t, cps, k))).T
        return pts
    except Exception:
        # fallback: polygon through control points
        pts = np.vstack([cp_x, cp_y, cp_z]).T
        return pts


def write_obj_splines(nodes, out_path, samples_per_spline=64):
    verts = []
    lines = []
    for n in nodes:
        pts = sample_spline(n.cp_x, n.cp_y, n.cp_z, n.knots, samples=samples_per_spline)
        # assume cp points are local offsets -> add node.abs_pos
        pts_world = pts + n.abs_pos.reshape(1, 3)
        start_idx = len(verts) + 1
        for p in pts_world:
            verts.append(p)
        # closed loop line
        lines.append(list(range(start_idx, start_idx + pts_world.shape[0])) + [start_idx])

    with open(out_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # write spline loops
        for ln in lines:
            f.write('l ' + ' '.join(str(i) for i in ln) + '\n')


def write_obj_centerline(root, out_path):
    # collect unique node positions and edges
    nodes = collect_nodes(root)
    pos_to_idx = {}
    verts = []
    edges = []
    for n in nodes:
        key = tuple(np.round(n.abs_pos, 8))
        if key not in pos_to_idx:
            pos_to_idx[key] = len(verts) + 1
            verts.append(n.abs_pos)
    # edges: parent-child pairs
    def traverse(n):
        if n is None:
            return
        if n.left is not None:
            edges.append((pos_to_idx[tuple(np.round(n.abs_pos, 8))], pos_to_idx[tuple(np.round(n.left.abs_pos, 8))]))
            traverse(n.left)
        if n.right is not None:
            edges.append((pos_to_idx[tuple(np.round(n.abs_pos, 8))], pos_to_idx[tuple(np.round(n.right.abs_pos, 8))]))
            traverse(n.right)
    traverse(root)

    with open(out_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b in edges:
            f.write(f"l {a} {b}\n")


def process_folder(input_dir, meta_path, output_dir, subset=None):
    meta = load_metadata(meta_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    if subset:
        files = [f for f in files if f in subset]

    for fidx, fname in enumerate(files, 1):
        path = os.path.join(input_dir, fname)
        arr = np.load(path, allow_pickle=True)
        root = deserialize_from_array(arr, meta)
        compute_absolute_positions(root, relative=meta.get('relative_positions', True))
        nodes = collect_nodes(root)

        base = os.path.splitext(fname)[0]
        out_splines = os.path.join(output_dir, base + '_splines.obj')
        out_center = os.path.join(output_dir, base + '_centerline.obj')

        write_obj_splines(nodes, out_splines, samples_per_spline=64)
        write_obj_centerline(root, out_center)

        print(f'Wrote {out_splines} and {out_center} ({fidx}/{len(files)})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='derived_data/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1/test')
    p.add_argument('--metadata', default='derived_data/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1/local_geometry_metadata.json')
    p.add_argument('--output_dir', default='derived_data_visualizations/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1_objs')
    p.add_argument('--cases', nargs='*', help='optional list of filenames to process (basename.npy)')
    args = p.parse_args()

    subset = None
    if args.cases:
        subset = args.cases

    process_folder(args.input_dir, args.metadata, args.output_dir, subset=subset)


if __name__ == '__main__':
    main()
