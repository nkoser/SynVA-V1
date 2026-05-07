import argparse
import os
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

import matplotlib.pyplot as plt

try:
    from scipy.interpolate import splev, splprep
except Exception:
    splev = None
    splprep = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


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


def collect_branches(node, path, branches):
    if node is None:
        return
    path.append(node)
    if node.left is None and node.right is None:
        branches.append(path[:])
    else:
        if node.left:
            collect_branches(node.left, path, branches)
        if node.right:
            collect_branches(node.right, path, branches)
    path.pop()


def radius_from_node(node, k, radius_mode, radius_fixed):
    if k == 4:
        return float(node.data["r"])

    r = node.data.get("r", [])
    if not isinstance(r, (list, tuple, np.ndarray)):
        return float(r)

    r = np.asarray(r, dtype=np.float32)
    if radius_mode == "first":
        return float(r[0]) if len(r) > 0 else 0.0
    if radius_mode == "norm3":
        return float(np.linalg.norm(r[:3])) if len(r) >= 3 else 0.0
    if radius_mode == "mean":
        return float(np.mean(np.abs(r))) if len(r) > 0 else 0.0
    if radius_mode == "fixed":
        return float(radius_fixed)

    raise ValueError("Unsupported radius_mode")


def set_equal_aspect(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_sphere(ax, center, radius, color, alpha, resolution):
    if radius <= 0:
        return
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)


def compute_edge_radii(edge_lengths, radius_scale):
    radii = []
    for lengths in edge_lengths:
        if lengths:
            base = min(lengths)
        else:
            base = 0.0
        radii.append(base * radius_scale)
    return np.array(radii, dtype=np.float32)


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


def check_root_zero(root_pos, tol):
    is_zero = np.all(np.abs(root_pos) <= tol)
    if is_zero:
        print(f"root check: OK (|root| <= {tol})")
    else:
        print(f"root check: NOT zero (root={root_pos.tolist()}, tol={tol})")


def sample_spline_coeffs(coeffs, n_samples):
    if splev is None:
        raise RuntimeError("scipy is required for spline visualization.")
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size < 36:
        return None
    # Keep spline visualization stable when extra structure channels are appended.
    coeffs = coeffs[:36]
    c = [np.array(coeffs[i * 8 : (i * 8) + 8], dtype=np.float64) for i in range(3)]
    ctrl = np.column_stack(c)
    if not np.all(np.isfinite(ctrl)):
        return None
    # Degenerate fallback splines are encoded as collapsed control points.
    if np.allclose(ctrl, ctrl[0], atol=1e-6):
        return np.repeat(ctrl[0:1], int(n_samples), axis=0)

    t = np.array(coeffs[24:36], dtype=np.float64)
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    if not np.all(np.isfinite(t)):
        return None
    # Invalid knot vectors (e.g., all-ones markers) should be skipped.
    if np.ptp(t) < 1e-8:
        return None
    if np.any(np.diff(t) < -1e-8):
        return None

    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    try:
        x, y, z = splev(u, tck)
    except Exception:
        return None
    points = np.column_stack((x, y, z))
    if not np.all(np.isfinite(points)):
        return None
    return points


def draw_splines(ax, nodes, root_pos, params, normalize_xyz):
    if splev is None:
        raise RuntimeError("scipy is required for spline visualization.")
    n_samples = int(params.get("spline_samples", 50))
    color = params.get("spline_color", "#cc0000")
    line_color = params.get("spline_line_color", color)
    alpha = float(params.get("spline_alpha", 0.6))
    size = float(params.get("spline_size", 0.5))
    center_root = bool(params.get("spline_center_root", True))
    render_mode = params.get("spline_render", "points")
    line_width = float(params.get("spline_line_width", 1.0))

    for node in nodes:
        coeffs = node.data.get("r", [])
        if not isinstance(coeffs, (list, tuple, np.ndarray)) or len(coeffs) < 36:
            continue
        points = sample_spline_coeffs(coeffs, n_samples)
        if points is None or points.shape[0] == 0:
            continue
        if center_root:
            points = points - root_pos
        if normalize_xyz:
            max_abs = np.max(np.abs(points))
            if max_abs > 0:
                points = points / max_abs
        if render_mode in ("line", "both"):
            loop = np.vstack([points, points[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=line_color, alpha=alpha, linewidth=line_width)
        if render_mode in ("points", "both"):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=".", s=size, alpha=alpha)


def draw_centerline_splines(ax, branches, root_pos, params, normalize_xyz):
    if splprep is None or splev is None:
        raise RuntimeError("scipy is required for centerline spline visualization.")
    samples = int(params.get("centerline_spline_samples", 200))
    smooth = float(params.get("centerline_spline_smooth", 0.0))
    color = params.get("centerline_spline_color", "#0066cc")
    alpha = float(params.get("centerline_spline_alpha", 0.8))
    width = float(params.get("centerline_spline_width", 1.5))
    center_root = bool(params.get("centerline_spline_center_root", False))

    for branch in branches:
        pts = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in branch], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        if center_root:
            pts = pts - root_pos
        if normalize_xyz:
            max_abs = np.max(np.abs(pts))
            if max_abs > 0:
                pts = pts / max_abs
        k = 3 if pts.shape[0] > 3 else max(1, pts.shape[0] - 1)
        try:
            tck, _ = splprep(pts.T, s=smooth, k=k)
        except Exception:
            continue
        t = np.linspace(0, 1, samples)
        x, y, z = splev(t, tck)
        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=width)


def _build_polylines(points_list):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for VTP export.") from exc

    vtk_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for pts in points_list:
        if pts is None or len(pts) < 2:
            continue
        start_idx = vtk_points.GetNumberOfPoints()
        for p in pts:
            vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(pts))
        for i in range(len(pts)):
            polyline.GetPointIds().SetId(i, start_idx + i)
        lines.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    return polydata


def export_vtp(path, branches, nodes, root_pos, params, normalize_xyz, include_centerline=None, include_splines=None):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for VTP export.") from exc

    if path:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    if include_centerline is None:
        include_centerline = bool(params.get("vtp_include_centerline", True))
    if include_splines is None:
        include_splines = bool(params.get("vtp_include_splines", True))
    use_centerline_spline = bool(params.get("draw_centerline_spline", False))

    center_root = bool(params.get("spline_center_root", True))
    spline_samples = int(params.get("spline_samples", 50))

    polydatas = []

    if include_centerline:
        centerline_pts = []
        if use_centerline_spline:
            if splprep is None or splev is None:
                raise RuntimeError("scipy is required for centerline spline export.")
            samples = int(params.get("centerline_spline_samples", 200))
            smooth = float(params.get("centerline_spline_smooth", 0.0))
            for branch in branches:
                pts = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in branch], dtype=np.float32)
                if pts.shape[0] < 2:
                    continue
                if center_root:
                    pts = pts - root_pos
                if normalize_xyz:
                    max_abs = np.max(np.abs(pts))
                    if max_abs > 0:
                        pts = pts / max_abs
                k = 3 if pts.shape[0] > 3 else max(1, pts.shape[0] - 1)
                try:
                    tck, _ = splprep(pts.T, s=smooth, k=k)
                except Exception:
                    continue
                t = np.linspace(0, 1, samples)
                x, y, z = splev(t, tck)
                centerline_pts.append(np.column_stack((x, y, z)))
        else:
            for branch in branches:
                pts = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in branch], dtype=np.float32)
                if pts.shape[0] < 2:
                    continue
                if center_root:
                    pts = pts - root_pos
                if normalize_xyz:
                    max_abs = np.max(np.abs(pts))
                    if max_abs > 0:
                        pts = pts / max_abs
                centerline_pts.append(pts)

        if centerline_pts:
            polydatas.append(_build_polylines(centerline_pts))

    if include_splines:
        spline_pts = []
        if splev is None:
            raise RuntimeError("scipy is required for spline export.")
        for node in nodes:
            coeffs = node.data.get("r", [])
            if not isinstance(coeffs, (list, tuple, np.ndarray)) or len(coeffs) < 36:
                continue
            pts = sample_spline_coeffs(coeffs, spline_samples)
            if pts is None or pts.shape[0] == 0:
                continue
            if center_root:
                pts = pts - root_pos
            if normalize_xyz:
                max_abs = np.max(np.abs(pts))
                if max_abs > 0:
                    pts = pts / max_abs
            loop = np.vstack([pts, pts[0]])
            spline_pts.append(loop)

        if spline_pts:
            polydatas.append(_build_polylines(spline_pts))

    if not polydatas:
        raise RuntimeError("No geometry to export.")

    append = vtk.vtkAppendPolyData()
    for pd in polydatas:
        append.AddInputData(pd)
    append.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(append.GetOutput())
    writer.Write()


def render_custom_plot(
    title,
    xyz,
    edges,
    radii,
    root_pos,
    nodes,
    params,
    normalize_xyz,
    draw_edges_flag,
    draw_spheres_flag,
    draw_splines_flag,
    draw_centerline_spline_flag,
    figsize,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if draw_edges_flag:
        draw_edges(
            ax,
            xyz,
            edges,
            color=params.get("edge_color", "#444444"),
            alpha=float(params.get("edge_alpha", 0.5)),
            width=float(params.get("edge_width", 1.0)),
        )

    scatter_size = float(params.get("scatter_size", 8.0))
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=scatter_size, c="tab:blue", alpha=0.7)

    draw_root = bool(params.get("draw_root", True))
    if draw_root:
        root_color = params.get("root_color", "green")
        root_marker = params.get("root_marker", "*")
        root_size = float(params.get("root_size", 80.0))
        ax.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]], c=root_color, marker=root_marker, s=root_size)

    if draw_splines_flag:
        draw_splines(ax, nodes, root_pos, params, normalize_xyz)
    if draw_centerline_spline_flag:
        branches = []
        collect_branches(nodes[0], [], branches)
        draw_centerline_splines(ax, branches, root_pos, params, normalize_xyz)

    if draw_spheres_flag:
        max_spheres = int(params.get("max_spheres", 250))
        resolution = int(params.get("sphere_resolution", 12))
        alpha = float(params.get("sphere_alpha", 0.2))

        order = np.argsort(-radii)
        for idx in order[:max_spheres]:
            plot_sphere(ax, xyz[idx], radii[idx], color="tab:orange", alpha=alpha, resolution=resolution)

    set_equal_aspect(ax, xyz)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def main():
    parser = argparse.ArgumentParser(description="Visualize tree nodes with radius envelopes.")
    parser.add_argument("--config", default="tree_viewer_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = paths.get("input")
    if not input_path:
        raise ValueError("paths.input is required")

    files = iter_files(input_path, params.get("pattern", "*.npy"))
    if not files:
        raise FileNotFoundError("No files found for input")

    file_path = pick_file(files, params.get("file_index"), params.get("file_name"))

    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    node_dim = k + 1 if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else k

    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape((-1, node_dim))

    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)

    viewer = params.get("viewer", "custom")
    figsize = params.get("figsize", [8, 7])
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        figsize = (float(figsize[0]), float(figsize[1]))
    else:
        figsize = (8.0, 7.0)
    nodes = []
    edges = []
    collect_nodes_edges(tree, nodes, edges)
    branches = []
    collect_branches(tree, [], branches)

    xyz = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in nodes], dtype=np.float32)
    root_pos = np.array([tree.data["x"], tree.data["y"], tree.data["z"]], dtype=np.float32)
    if bool(params.get("check_root_zero", False)):
        tol = float(params.get("root_zero_tol", 1e-6))
        check_root_zero(root_pos, tol)

    normalize_xyz = bool(params.get("normalize_xyz", False))
    if normalize_xyz:
        max_abs = np.max(np.abs(xyz))
        if max_abs > 0:
            xyz = xyz / max_abs

    radius_mode = params.get("radius_mode", "edge")
    radius_fixed = float(params.get("radius_fixed", 0.01))
    radius_scale = float(params.get("radius_scale", 0.1))
    radius_min = float(params.get("radius_min", 0.0))
    radius_max = params.get("radius_max")
    radius_max = float(radius_max) if radius_max is not None else None

    if radius_mode == "edge":
        edge_lengths = [[] for _ in range(len(nodes))]
        for i, j in edges:
            dist = float(np.linalg.norm(xyz[i] - xyz[j]))
            edge_lengths[i].append(dist)
            edge_lengths[j].append(dist)
        radii = compute_edge_radii(edge_lengths, radius_scale)
    else:
        radii = []
        for n in nodes:
            r = radius_from_node(n, k, radius_mode, radius_fixed)
            r = abs(r) * radius_scale
            radii.append(r)
        radii = np.array(radii, dtype=np.float32)

    if radius_max is not None:
        radii = np.minimum(radii, radius_max)
    radii = np.maximum(radii, radius_min)

    draw_edges_flag = bool(params.get("draw_edges", True))
    draw_splines_flag = bool(params.get("draw_splines", False))
    draw_centerline_spline_flag = bool(params.get("draw_centerline_spline", False))
    draw_spheres_flag = bool(params.get("draw_spheres", False))

    if viewer == "custom":
        render_custom_plot(
            os.path.basename(file_path),
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            draw_edges_flag,
            draw_spheres_flag,
            draw_splines_flag,
            draw_centerline_spline_flag,
            figsize,
        )
    elif viewer == "legacy_splines":
        render_custom_plot(
            os.path.basename(file_path) + " (splines)",
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            False,
            False,
            True,
            draw_centerline_spline_flag,
            figsize,
        )
    elif viewer == "combined":
        render_custom_plot(
            os.path.basename(file_path) + " (combined)",
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            True,
            False,
            True,
            draw_centerline_spline_flag,
            figsize,
        )
    elif viewer == "all":
        render_custom_plot(
            os.path.basename(file_path) + " (custom)",
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            draw_edges_flag,
            draw_spheres_flag,
            False,
            draw_centerline_spline_flag,
            figsize,
        )
        render_custom_plot(
            os.path.basename(file_path) + " (legacy_splines)",
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            False,
            False,
            True,
            draw_centerline_spline_flag,
            figsize,
        )
        render_custom_plot(
            os.path.basename(file_path) + " (combined)",
            xyz,
            edges,
            radii,
            root_pos,
            nodes,
            params,
            normalize_xyz,
            True,
            False,
            True,
            draw_centerline_spline_flag,
            figsize,
        )
    else:
        raise ValueError("Unsupported viewer mode. Use custom, legacy_splines, combined, or all.")

    output_vtp = paths.get("output_vtp") or params.get("output_vtp")
    output_vtp_centerline = paths.get("output_vtp_centerline") or params.get("output_vtp_centerline")
    output_vtp_splines = paths.get("output_vtp_splines") or params.get("output_vtp_splines")
    if output_vtp:
        export_vtp(output_vtp, branches, nodes, root_pos, params, normalize_xyz)
    if output_vtp_centerline:
        export_vtp(output_vtp_centerline, branches, nodes, root_pos, params, normalize_xyz, include_centerline=True, include_splines=False)
    if output_vtp_splines:
        export_vtp(output_vtp_splines, branches, nodes, root_pos, params, normalize_xyz, include_centerline=False, include_splines=True)

    output_image = paths.get("output_image")
    if output_image:
        plt.savefig(output_image, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
