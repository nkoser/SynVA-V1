import argparse
import json
import os

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splev
except Exception:
    splev = None

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import (
    deserialize,
    deserialize_pre_order_kcount,
    deserialize_pre_order_kdir,
    local_geometry_tree_to_absolute,
)


K_MODES = {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}


def _k_upper_bound(mode):
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        return 3
    if mode in {"pre_order_kcount", "pre_order_k"}:
        return 2
    return None


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_normalization_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    center = np.asarray(raw.get("center"), dtype=np.float32)
    scale = np.asarray(raw.get("scale"), dtype=np.float32)
    if center.ndim != 1 or scale.ndim != 1 or center.size == 0 or center.size != scale.size:
        raise ValueError("Invalid normalization stats: center/scale must be 1D arrays with equal non-zero length.")
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return {
        "feature_start": int(raw.get("feature_start", 1)),
        "feature_end": raw.get("feature_end", None),
        "center": center,
        "scale": scale,
    }


def _normalize_mask_len(mask, n_rows):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape[0] == n_rows:
        return mask
    fixed = np.zeros((n_rows,), dtype=bool)
    shared = min(n_rows, mask.shape[0])
    fixed[:shared] = mask[:shared]
    return fixed


def denormalize_rows(data, stats, k, mode):
    out = data.astype(np.float32, copy=True)
    center = stats["center"]
    scale = stats["scale"]

    fs = int(stats.get("feature_start", 1 if mode in K_MODES else 0))
    fe = stats.get("feature_end", None)
    if fe is None:
        fe = fs + center.size
    fe = int(fe)

    if fe <= out.shape[1] and (fe - fs) == center.size:
        sl = slice(fs, fe)
    else:
        fallback_start = 1 if mode in K_MODES else 0
        fallback_end = fallback_start + center.size
        if fallback_end > out.shape[1]:
            raise ValueError(
                f"Cannot denormalize {out.shape[1]} columns with center size {center.size}."
            )
        sl = slice(fallback_start, fallback_end)

    out[:, sl] = out[:, sl] * scale + center
    return out


def restore_local_geometry(data, mode, relative_positions=False, node_local_control_points=False):
    if mode not in K_MODES:
        return data
    if not (bool(relative_positions) or bool(node_local_control_points)):
        return data
    return local_geometry_tree_to_absolute(
        data,
        relative_positions=bool(relative_positions),
        node_local_control_points=bool(node_local_control_points),
        copy=True,
    )


def load_tree(
    file_path,
    k,
    mode,
    threshold,
    zero_mask=None,
    denorm_stats=None,
    relative_positions=False,
    node_local_control_points=False,
):
    data = np.load(file_path)
    node_dim = k + 1 if mode in K_MODES else k
    if data.ndim == 1:
        data = data.reshape((-1, node_dim))

    if denorm_stats is not None:
        data = denormalize_rows(data, denorm_stats, k=k, mode=mode)
    data = restore_local_geometry(
        data,
        mode=mode,
        relative_positions=relative_positions,
        node_local_control_points=node_local_control_points,
    )
    zero_mask = _normalize_mask_len(zero_mask, data.shape[0])
    if zero_mask is not None:
        data = data.copy()
        data[zero_mask] = 0
    if threshold is not None:
        data = data.copy()
        data[np.abs(data) < threshold] = 0
    serial = list(data.flatten())
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        tree, ret = deserialize_pre_order_kdir(serial, k=k)
    elif mode in {"pre_order_kcount", "pre_order_k"}:
        tree, ret = deserialize_pre_order_kcount(serial, k=k)
    else:
        tree = deserialize(serial, mode=mode, k=k)
        ret = []
    if tree is None:
        raise ValueError(f"Tree is empty: {file_path}")

    k_tokens = None
    if mode in K_MODES:
        k_tokens = np.rint(data[:, 0]).astype(np.int64)
        upper = _k_upper_bound(mode)
        if upper is not None:
            k_tokens = np.clip(k_tokens, 0, upper)

    meta = {
        "rows": int(data.shape[0]),
        "ret_len": int(len(ret)),
        "k_tokens": k_tokens,
    }
    return tree, meta


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


def build_arrays(tree):
    nodes = []
    edges = []
    collect_nodes_edges(tree, nodes, edges)
    xyz = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in nodes], dtype=np.float32)
    root = np.array([tree.data["x"], tree.data["y"], tree.data["z"]], dtype=np.float32)
    return xyz, edges, root, nodes


def _normalize_repaired_knots(t):
    t = np.asarray(t, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        return None
    t = np.clip(t, 0.0, 1.0)
    t = np.sort(t)
    if np.ptp(t) < 1e-8:
        return None
    t = (t - t[0]) / (t[-1] - t[0])
    return t


def sample_spline_coeffs(coeffs, n_samples, repair_knots=False):
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size < 36:
        return None
    # Ignore appended structure channels when present (k > 39 attributes).
    coeffs = coeffs[:36]

    # Try both common control-point layouts:
    # 1) block: [x1..x8, y1..y8, z1..z8]
    # 2) interleaved: [x1,y1,z1, x2,y2,z2, ...]
    candidates = []
    c_block = [np.array(coeffs[i * 8 : (i * 8) + 8], dtype=np.float64) for i in range(3)]
    candidates.append(c_block)
    cp_inter = np.array(coeffs[:24], dtype=np.float64).reshape(8, 3)
    c_inter = [cp_inter[:, 0], cp_inter[:, 1], cp_inter[:, 2]]
    candidates.append(c_inter)

    # Keep behavior consistent with view_trees: collapsed control points are valid
    # degenerate fallback splines, even if knots are all-ones markers.
    for c in candidates:
        ctrl = np.column_stack(c)
        if not np.all(np.isfinite(ctrl)):
            continue
        if np.allclose(ctrl, ctrl[0], atol=1e-6):
            return np.repeat(ctrl[0:1], int(n_samples), axis=0)

    t_raw = np.array(coeffs[24:36], dtype=np.float64)
    t_raw = np.where(np.abs(t_raw - 1) < 0.01, 1.0, t_raw)
    if repair_knots:
        t = _normalize_repaired_knots(t_raw)
        if t is None:
            return None
    else:
        t = t_raw
        if not np.all(np.isfinite(t)):
            return None
        if np.ptp(t) < 1e-8:
            return None
        if np.any(np.diff(t) < -1e-8):
            return None

    u = np.linspace(0, 1, n_samples)
    for c in candidates:
        ctrl = np.column_stack(c)
        if not np.all(np.isfinite(ctrl)):
            continue
        tck = (t, c, 3)
        try:
            x, y, z = splev(u, tck)
        except Exception:
            continue
        points = np.column_stack((x, y, z))
        if np.all(np.isfinite(points)):
            return points
    return None


def control_points_fallback(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size < 24:
        return None
    block = np.column_stack(
        [np.array(coeffs[i * 8 : (i * 8) + 8], dtype=np.float64) for i in range(3)]
    )
    inter = np.array(coeffs[:24], dtype=np.float64).reshape(8, 3)
    for ctrl in (block, inter):
        if np.all(np.isfinite(ctrl)) and np.max(np.std(ctrl, axis=0)) > 1e-8:
            return ctrl
    return None


def draw_splines(
    ax,
    nodes,
    root_pos,
    n_samples,
    color,
    alpha,
    size,
    center_root,
    repair_knots=True,
    reuse_last_valid=True,
):
    if splev is None:
        raise RuntimeError("scipy is required for spline visualization.")
    candidates = 0
    sampled = 0
    repaired = 0
    reused_prev = 0
    ctrl_fallback = 0
    skipped = 0
    last_valid = None
    for node in nodes:
        coeffs = node.data.get("r", [])
        if not isinstance(coeffs, (list, tuple, np.ndarray)) or len(coeffs) < 36:
            continue
        candidates += 1
        try:
            points = sample_spline_coeffs(coeffs, n_samples)
        except Exception:
            points = None
        if points is None and repair_knots:
            try:
                points = sample_spline_coeffs(coeffs, n_samples, repair_knots=True)
            except Exception:
                points = None
            if points is not None:
                repaired += 1
        if points is None and reuse_last_valid and last_valid is not None:
            points = last_valid.copy()
            node_xyz = np.array(
                [
                    float(node.data.get("x", 0.0)),
                    float(node.data.get("y", 0.0)),
                    float(node.data.get("z", 0.0)),
                ],
                dtype=np.float64,
            )
            if np.all(np.isfinite(node_xyz)):
                points = points + (node_xyz - points[0])
            reused_prev += 1
        if points is not None and points.shape[0] > 0:
            points_to_draw = points.copy()
            if center_root:
                points_to_draw = points_to_draw - root_pos
            ax.scatter(
                points_to_draw[:, 0],
                points_to_draw[:, 1],
                points_to_draw[:, 2],
                c=color,
                marker=".",
                s=size,
                alpha=alpha,
            )
            last_valid = points.copy()
            sampled += 1
            continue

        ctrl = control_points_fallback(coeffs)
        if ctrl is None:
            skipped += 1
            continue
        if center_root:
            ctrl = ctrl - root_pos
        ax.plot(
            ctrl[:, 0],
            ctrl[:, 1],
            ctrl[:, 2],
            color=color,
            alpha=max(0.35, alpha),
            linewidth=max(0.6, size * 1.8),
        )
        ctrl_fallback += 1
    print(
        f"[debug] spline render: candidates={candidates}, sampled={sampled}, repaired={repaired}, "
        f"reuse_last={reused_prev}, ctrl_fallback={ctrl_fallback}, skipped={skipped}"
    )


def set_equal_aspect(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_edges(ax, xyz, edges, color, alpha, width):
    for i, j in edges:
        p1 = xyz[i]
        p2 = xyz[j]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            alpha=alpha,
            linewidth=width,
        )


def main():
    parser = argparse.ArgumentParser(description="Overlay two trees for visual comparison.")
    parser.add_argument("--config", default="compare_trees_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    tree_a_path = paths.get("tree_a")
    tree_b_path = paths.get("tree_b")
    if not tree_a_path or not tree_b_path:
        raise ValueError("paths.tree_a and paths.tree_b are required")

    k = int(params.get("k", 39))
    mode_a = params.get("mode_a", params.get("mode", "pre_order"))
    mode_b = params.get("mode_b", params.get("mode", "pre_order"))
    threshold_a = params.get("threshold_a")
    threshold_b = params.get("threshold_b")
    threshold_a = float(threshold_a) if threshold_a is not None else None
    threshold_b = float(threshold_b) if threshold_b is not None else None
    align_root = bool(params.get("align_root", True))
    mask_from = paths.get("mask_from")
    mask_threshold = params.get("mask_threshold")
    mask_threshold = float(mask_threshold) if mask_threshold is not None else None
    edges_source = params.get("edges_source", "each")
    denorm_a = bool(params.get("denormalize_a", False))
    denorm_b = bool(params.get("denormalize_b", False))
    relative_positions_a = bool(params.get("relative_positions_a", params.get("relative_positions", False)))
    relative_positions_b = bool(params.get("relative_positions_b", params.get("relative_positions", False)))
    node_local_cp_a = bool(params.get("node_local_control_points_a", params.get("node_local_control_points", False)))
    node_local_cp_b = bool(params.get("node_local_control_points_b", params.get("node_local_control_points", False)))
    norm_stats_path = paths.get("normalization_stats", paths.get("norm_stats_path"))
    norm_stats = None
    if denorm_a or denorm_b:
        if not norm_stats_path:
            raise ValueError(
                "denormalize_a/denormalize_b is true, but paths.normalization_stats is missing."
            )
        norm_stats = load_normalization_stats(norm_stats_path)

    zero_mask = None
    if mask_from:
        mask_data = np.load(mask_from)
        mask_node_dim = k + 1 if mode_b in K_MODES else k
        if mask_data.ndim == 1:
            mask_data = mask_data.reshape((-1, mask_node_dim))
        if mask_threshold is None:
            mask_threshold = 0.0
        zero_mask = np.all(np.abs(mask_data) <= mask_threshold, axis=1)

    tree_a, meta_a = load_tree(
        tree_a_path,
        k,
        mode_a,
        threshold_a,
        denorm_stats=(norm_stats if denorm_a else None),
        relative_positions=relative_positions_a,
        node_local_control_points=node_local_cp_a,
    )
    tree_b, meta_b = load_tree(
        tree_b_path,
        k,
        mode_b,
        threshold_b,
        zero_mask=zero_mask,
        denorm_stats=(norm_stats if denorm_b else None),
        relative_positions=relative_positions_b,
        node_local_control_points=node_local_cp_b,
    )

    xyz_a, edges_a, root_a, tree_a_nodes = build_arrays(tree_a)
    xyz_b, edges_b, root_b, tree_b_nodes = build_arrays(tree_b)

    # Debug report for topology parsing and K-token consistency.
    print(f"[debug] A rows={meta_a['rows']} nodes={len(tree_a_nodes)} ret_len={meta_a['ret_len']}")
    print(f"[debug] B rows={meta_b['rows']} nodes={len(tree_b_nodes)} ret_len={meta_b['ret_len']}")
    if mode_a in K_MODES and mode_b in K_MODES and meta_a["k_tokens"] is not None and meta_b["k_tokens"] is not None:
        k_a = meta_a["k_tokens"]
        k_b = meta_b["k_tokens"]
        shared = min(k_a.shape[0], k_b.shape[0])
        mismatch_idx = np.flatnonzero(k_a[:shared] != k_b[:shared])
        if mismatch_idx.size > 0:
            first_idx = int(mismatch_idx[0])
            print(f"[debug] First K mismatch at row={first_idx}: A={int(k_a[first_idx])} B={int(k_b[first_idx])}")
        else:
            print("[debug] First K mismatch: none in shared prefix")

    # Preserve original roots for spline centering; plotting roots may be shifted.
    spline_root_a = root_a.copy()
    spline_root_b = root_b.copy()
    plot_root_a = root_a.copy()
    plot_root_b = root_b.copy()

    if align_root:
        xyz_a = xyz_a - spline_root_a
        xyz_b = xyz_b - spline_root_b
        plot_root_a = np.zeros(3, dtype=np.float32)
        plot_root_b = np.zeros(3, dtype=np.float32)

    fig = plt.figure(figsize=tuple(params.get("figsize", [10, 10])))
    ax = fig.add_subplot(111, projection="3d")

    edge_alpha = float(params.get("edge_alpha", 0.4))
    edge_width = float(params.get("edge_width", 1.0))
    node_size = float(params.get("node_size", 8.0)) 

    color_a = params.get("color_a", "#1f77b4")
    color_b = params.get("color_b", "#ff7f0e")
    label_a = params.get("label_a", "Original")
    label_b = params.get("label_b", "Reconstruction")

    draw_edges_flag = bool(params.get("draw_edges", True))
    if draw_edges_flag:
        draw_edges(ax, xyz_a, edges_a, color=color_a, alpha=edge_alpha, width=edge_width)
        if edges_source == "a":
            draw_edges(ax, xyz_b, edges_a, color=color_b, alpha=edge_alpha, width=edge_width)
        else:
            draw_edges(ax, xyz_b, edges_b, color=color_b, alpha=edge_alpha, width=edge_width)

    ax.scatter(xyz_a[:, 0], xyz_a[:, 1], xyz_a[:, 2], s=node_size, c=color_a, alpha=0.8, label=label_a)
    ax.scatter(xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2], s=node_size, c=color_b, alpha=0.8, label=label_b)

    if bool(params.get("draw_splines", False)):
        n_samples = int(params.get("spline_samples", 50))
        alpha = float(params.get("spline_alpha", 0.4))
        size = float(params.get("spline_size", 0.4))
        center_root = bool(params.get("spline_center_root", True))
        repair_knots = bool(params.get("spline_repair_knots", True))
        reuse_last_valid = bool(params.get("spline_reuse_last_valid", True))
        draw_splines(
            ax,
            tree_a_nodes,
            spline_root_a,
            n_samples,
            params.get("spline_color_a", color_a),
            alpha,
            size,
            center_root,
            repair_knots=repair_knots,
            reuse_last_valid=reuse_last_valid,
        )
        draw_splines(
            ax,
            tree_b_nodes,
            spline_root_b,
            n_samples,
            params.get("spline_color_b", color_b),
            alpha,
            size,
            center_root,
            repair_knots=repair_knots,
            reuse_last_valid=reuse_last_valid,
        )

    if bool(params.get("draw_root", True)):
        root_marker = params.get("root_marker", "x")
        root_size = float(params.get("root_size", 60.0))
        ax.scatter([plot_root_a[0]], [plot_root_a[1]], [plot_root_a[2]], c=color_a, marker=root_marker, s=root_size)
        ax.scatter([plot_root_b[0]], [plot_root_b[1]], [plot_root_b[2]], c=color_b, marker=root_marker, s=root_size)

    all_xyz = np.vstack((xyz_a, xyz_b))
    set_equal_aspect(ax, all_xyz)

    ax.set_title(params.get("title", "Tree overlay"))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    output_image = paths.get("output_image")
    if output_image:
        fig.savefig(output_image, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()

# %%
