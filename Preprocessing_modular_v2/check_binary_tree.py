import argparse
import os
from glob import glob

import numpy as np

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize


def iter_files(input_path, pattern):
    if os.path.isdir(input_path):
        return sorted(glob(os.path.join(input_path, pattern)))
    return [input_path]


def pick_file(files, file_index=None, file_name=None):
    if file_name:
        for f in files:
            if os.path.basename(f) == file_name:
                return f
        raise FileNotFoundError(f"file_name not found: {file_name}")
    if file_index is None:
        return files[0]
    if file_index < 0 or file_index >= len(files):
        raise IndexError("file_index out of range")
    return files[file_index]


def collect_nodes_edges(node, nodes, edges):
    if node is None:
        return None

    idx = len(nodes)
    nodes.append(node)

    left_idx = collect_nodes_edges(node.left, nodes, edges)
    if left_idx is not None:
        edges.append((idx, left_idx))

    right_idx = collect_nodes_edges(node.right, nodes, edges)
    if right_idx is not None:
        edges.append((idx, right_idx))

    return idx


def set_equal_aspect(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def count_children(node):
    count = 0
    if node.left is not None:
        count += 1
    if node.right is not None:
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Visualize tree and highlight non-binary nodes.")
    parser.add_argument("--input", required=True, help="Input folder or file")
    parser.add_argument("--k", type=int, default=39, help="Feature dimension (default: 39)")
    parser.add_argument("--mode", default="pre_order", choices=["pre_order", "post_order", "pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"])
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--file_name", default=None)
    parser.add_argument("--file_index", type=int, default=None)
    parser.add_argument("--output_image", default=None)
    args = parser.parse_args()

    files = iter_files(args.input, args.pattern)
    if not files:
        raise FileNotFoundError("No files found for input")
    file_path = pick_file(files, args.file_index, args.file_name)

    data = np.load(file_path)
    node_dim = args.k + 1 if args.mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else args.k
    if data.ndim == 1:
        data = data.reshape((-1, node_dim))

    serial = list(data.flatten())
    tree = deserialize(serial, mode=args.mode, k=args.k)

    nodes = []
    edges = []
    collect_nodes_edges(tree, nodes, edges)

    xyz = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in nodes], dtype=np.float32)

    child_counts = [count_children(n) for n in nodes]
    is_bad = np.array([c > 2 for c in child_counts])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in edges:
        p1 = xyz[i]
        p2 = xyz[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#444444", alpha=0.6)

    ax.scatter(xyz[~is_bad, 0], xyz[~is_bad, 1], xyz[~is_bad, 2], s=8, c="#1f77b4", alpha=0.7)
    if np.any(is_bad):
        ax.scatter(xyz[is_bad, 0], xyz[is_bad, 1], xyz[is_bad, 2], s=30, c="red", alpha=0.9)

    set_equal_aspect(ax, xyz)
    ax.set_title(os.path.basename(file_path))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if np.any(is_bad):
        bad_count = int(np.sum(is_bad))
        print(f"Non-binary nodes found: {bad_count}")
    else:
        print("Tree appears binary (max 2 children per node).")

    if args.output_image:
        fig.savefig(args.output_image, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
