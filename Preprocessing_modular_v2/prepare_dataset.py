import argparse
import os
import random
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splev, splprep
except Exception as exc:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize, serialize_pre_order_k

K_MODE_SET = {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rotation_matrix(angle_degrees, axis):
    angle = np.radians(angle_degrees)
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def zero_root(data, mode):
    if mode not in {"pre_order", "post_order", "pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
        raise ValueError("mode must be pre_order, post_order, or pre_order_kcount")
    root = data[0, :3] if mode == "pre_order" else data[-1, :3]
    not_zero_mask = np.mean(data, axis=1) != 0
    data[not_zero_mask, :3] = data[not_zero_mask, :3] - root 
    return data, root, not_zero_mask


def all_elements_equal(values):
    return all(np.allclose(x, values[0], atol=1e-4) for x in values)


def limpiarRadiosSplines(tck):
    """Flatten a tck into 36 features (8 cp_x, 8 cp_y, 8 cp_z, 12 knots).

    Refits non-canonical periodic splines to the canonical 8-CP / 12-knot form
    while preserving the wrap (last 3 CPs = first 3 CPs).
    """
    from Preprocessing_v2.splines import _refit_to_canonical_periodic_8cp

    knots_in = np.asarray(tck[0], dtype=np.float32)
    cps_in   = list(tck[1])
    n_cp = 8
    n_knot = 12
    cleaned = []

    n_existing = len(np.asarray(cps_in[0]))
    n_existing_knots = len(knots_in)

    # Degenerate marker (collapsed knots all equal) -> pass through
    if n_existing_knots == n_knot and np.allclose(knots_in, knots_in[0]):
        for arr in cps_in:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < n_cp:
                arr = np.pad(arr, (0, n_cp - len(arr)), mode="edge")
            cleaned.extend(arr[:n_cp])
        cleaned.extend(knots_in[:n_knot])
        return cleaned

    try:
        knots_can, cps_can = _refit_to_canonical_periodic_8cp(
            cps_in, knots_in, k=3, n_cp=n_cp, n_samples=64
        )
    except Exception:
        knots_can = np.pad(
            knots_in, (0, max(0, n_knot - len(knots_in))), mode="edge"
        )[:n_knot]
        cps_can = []
        for arr in cps_in:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < n_cp:
                arr = np.pad(arr, (0, n_cp - len(arr)), mode="edge")
            cps_can.append(arr[:n_cp])

    for arr in cps_can:
        cleaned.extend(np.asarray(arr, dtype=np.float32)[:n_cp])
    cleaned.extend(np.asarray(knots_can, dtype=np.float32)[:n_knot])
    return cleaned


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    if len(coeffs) < 36:
        return None
    coeffs = np.asarray(coeffs[:36], dtype=np.float64)
    c = [np.array(coeffs[i * 8: (i * 8) + 8], dtype=np.float64) for i in range(3)]
    ctrl = np.column_stack(c)
    if not np.all(np.isfinite(ctrl)):
        return None
    if np.allclose(ctrl, ctrl[0], atol=1e-6):
        return np.repeat(ctrl[0:1], int(n_samples), axis=0)

    t = np.array(coeffs[24:36], dtype=np.float64)
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    if not np.all(np.isfinite(t)):
        return None
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


def _coerce_len(values, target_len):
    values = np.asarray(values, dtype=np.float32)
    if len(values) >= target_len:
        return values[:target_len]
    return np.pad(values, (0, target_len - len(values)), mode="edge")


def pack_spline_coeffs(tck, target_ctrl=8, target_knot=12):
    t, c, _k = tck
    cx = _coerce_len(c[0], target_ctrl)
    cy = _coerce_len(c[1], target_ctrl)
    cz = _coerce_len(c[2], target_ctrl)
    tt = _coerce_len(t, target_knot)
    return np.concatenate((cx, cy, cz, tt))


def _transform_spline_coeffs(coeffs, root, rot, scale):
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.size < 36:
        return coeffs
    ctrl = coeffs[:24].reshape(3, 8).T
    if root is not None:
        ctrl = ctrl - root
    if rot is not None:
        ctrl = ctrl @ rot.T
    if scale is not None:
        ctrl = ctrl / scale
    return np.concatenate((ctrl.T.reshape(24), coeffs[24:36]))


def _decode_child_count(k_value, mode):
    k_int = int(np.rint(float(k_value)))
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        if k_int <= 0:
            return 0
        if k_int == 3:
            return 2
        return 1
    return int(np.clip(k_int, 0, 2))


def _build_structure_channels(
    kcol,
    xyz,
    mode,
    add_depth_norm,
    add_parent_delta_xyz,
    add_pathlen_root_norm,
    depth_cap,
    pathlen_cap,
    delta_scale,
):
    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return np.zeros((n, 0), dtype=np.float32)
    if kcol is None or kcol.shape[0] != n:
        return np.zeros((n, 0), dtype=np.float32)

    parents = np.full(n, -1, dtype=np.int32)
    depth = np.zeros(n, dtype=np.float32)
    pathlen = np.zeros(n, dtype=np.float32)
    stack = []

    for idx in range(n):
        while stack and stack[-1][1] <= 0:
            stack.pop()
        if stack:
            parent_idx = stack[-1][0]
            parents[idx] = parent_idx
            depth[idx] = depth[parent_idx] + 1.0
            delta = xyz[idx, :3] - xyz[parent_idx, :3]
            pathlen[idx] = pathlen[parent_idx] + float(np.linalg.norm(delta))
            stack[-1][1] -= 1
        child_count = _decode_child_count(kcol[idx, 0], mode)
        if child_count > 0:
            stack.append([idx, child_count])

    channels = []

    if add_depth_norm:
        depth_cap_val = float(depth_cap)
        if (not np.isfinite(depth_cap_val)) or depth_cap_val <= 0:
            depth_cap_val = max(float(np.max(depth)), 1.0)
        channels.append((depth / depth_cap_val).reshape(-1, 1))

    if add_parent_delta_xyz:
        parent_delta = np.zeros((n, 3), dtype=np.float32)
        valid = parents >= 0
        if np.any(valid):
            parent_delta[valid] = xyz[valid, :3] - xyz[parents[valid], :3]
        delta_scale_val = float(delta_scale)
        if (not np.isfinite(delta_scale_val)) or abs(delta_scale_val) <= 1e-8:
            delta_scale_val = 1.0
        channels.append(parent_delta / delta_scale_val)

    if add_pathlen_root_norm:
        pathlen_cap_val = float(pathlen_cap)
        if (not np.isfinite(pathlen_cap_val)) or pathlen_cap_val <= 0:
            pathlen_cap_val = max(float(np.max(pathlen)), 1.0)
        channels.append((pathlen / pathlen_cap_val).reshape(-1, 1))

    if not channels:
        return np.zeros((n, 0), dtype=np.float32)
    return np.hstack(channels).astype(np.float32, copy=False)


def _parent_indices_from_kcol(kcol, mode):
    n = int(kcol.shape[0])
    parents = np.full(n, -1, dtype=np.int32)
    stack = []
    for idx in range(n):
        while stack and stack[-1][1] <= 0:
            stack.pop()
        if idx > 0:
            if stack:
                parents[idx] = int(stack[-1][0])
                stack[-1][1] -= 1
            else:
                parents[idx] = idx - 1
        child_count = _decode_child_count(kcol[idx, 0], mode)
        if child_count > 0:
            stack.append([idx, child_count])
    return parents


def _to_parent_delta_xyz(attrs, kcol, mode, root_zero=True):
    if attrs.ndim != 2 or attrs.shape[1] < 3:
        return attrs
    if kcol is None or kcol.shape[0] != attrs.shape[0]:
        return attrs
    out = attrs.copy()
    xyz = attrs[:, :3]
    parents = _parent_indices_from_kcol(kcol, mode=mode)
    delta = xyz.copy()
    for idx in range(xyz.shape[0]):
        parent_idx = int(parents[idx])
        if parent_idx >= 0:
            delta[idx] = xyz[idx] - xyz[parent_idx]
        elif root_zero:
            delta[idx] = 0.0
    out[:, :3] = delta
    return out.astype(np.float32, copy=False)


def build_spline_dataset(
    data,
    mode,
    n_rotations,
    n_samples,
    smooth,
    enable_rotation,
    enable_scaling,
    refit_splines,
):
    root = data[-1, :3] if mode == "post_order" else data[0, :3]
    not_zero_mask = np.mean(data, axis=1) != 0

    outputs = []
    for r in range(n_rotations):
        # rotation disabled by default to match datasets.ipynb
        if enable_rotation and r != 0:
            angle = random.randint(10, 350)
            axis = np.random.rand(3)
            rot = rotation_matrix(angle, axis)
        else:
            rot = None

        spline_points = np.zeros((len(data) * n_samples, 3), dtype=np.float32)
        j = 0
        for _, datum in enumerate(data):
            if np.any(datum):
                if all_elements_equal(datum[3:11]):
                    points = np.hstack(
                        (
                            np.full(n_samples, datum[0]).reshape(-1, 1),
                            np.full(n_samples, datum[1]).reshape(-1, 1),
                            np.full(n_samples, datum[2]).reshape(-1, 1),
                        )
                    ) - root
                else:
                    sampled = sample_spline_coeffs(datum[3:], n_samples=n_samples)
                    if sampled is None:
                        points = np.hstack(
                            (
                                np.full(n_samples, datum[0]).reshape(-1, 1),
                                np.full(n_samples, datum[1]).reshape(-1, 1),
                                np.full(n_samples, datum[2]).reshape(-1, 1),
                            )
                        ) - root
                    else:
                        points = sampled - root
                spline_points[j * n_samples: (j + 1) * n_samples] = points
            else:
                spline_points[j * n_samples: (j + 1) * n_samples] = np.zeros((n_samples, 3))
            j += 1

        # zero root and rotations
        data_xyz = data[:, :3].copy()
        data_xyz[not_zero_mask, :] = data_xyz[not_zero_mask, :] - root

        if rot is not None:
            data_xyz = data_xyz @ rot.T
            spline_points = spline_points @ rot.T

        scale = None
        if enable_scaling:
            all_data = np.vstack((data_xyz, spline_points))
            abs_max = abs(all_data).max()
            if abs_max > 0:
                all_data = all_data / abs_max
                scale = abs_max
            data_xyz = all_data[: len(data_xyz), :]
            spline_points = all_data[len(data_xyz):, :]

        if refit_splines:
            data_splines = []
            for i in range(0, len(data) * n_samples, n_samples):
                segment = spline_points[i: i + n_samples]
                if np.any(segment):
                    xs = segment[:, 0].flatten()
                    ys = segment[:, 1].flatten()
                    zs = segment[:, 2].flatten()

                    if all_elements_equal(xs):
                        t = np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0],
                            dtype=np.float32,
                        )
                        c = [xs[:8], ys[:8], zs[:8]]
                        tck = (t, c, 3)
                    else:
                        try:
                            tck, _ = splprep([xs, ys, zs], s=smooth, per=True, nest=12, k=3)
                        except Exception:
                            tck = None

                    if tck is None:
                        datum = data[i // n_samples]
                        if np.any(datum):
                            new_row = _transform_spline_coeffs(datum[3:], root, rot, scale)
                        else:
                            new_row = np.zeros(36, dtype=np.float32)
                    else:
                        new_row = limpiarRadiosSplines(tck)
                    data_splines.append(new_row)
                else:
                    data_splines.append(np.zeros(36))

            data_splines = np.array(data_splines, dtype=np.float32)
        else:
            data_splines = []
            for datum in data:
                if np.any(datum):
                    data_splines.append(
                        _transform_spline_coeffs(datum[3:], root, rot, scale)
                    )
                else:
                    data_splines.append(np.zeros(36))
            data_splines = np.array(data_splines, dtype=np.float32)
        new_data = np.hstack((data_xyz, data_splines))
        outputs.append(new_data)

    return outputs


def iter_files(input_dir, pattern):
    return sorted(glob(os.path.join(input_dir, pattern)))


def erase_all_files(folder_path):
    if not os.path.isdir(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def process_file(
    file_path,
    output_dir,
    k,
    mode,
    n_rotations,
    overwrite,
    n_samples,
    smooth,
    enable_rotation,
    enable_scaling,
    refit_splines,
    add_structure_features,
    structure_add_depth_norm,
    structure_add_parent_delta_xyz,
    structure_add_pathlen_root_norm,
    structure_depth_cap,
    structure_pathlen_cap,
    structure_delta_scale,
    delta_xyz_parent,
    delta_xyz_root_zero,
):
    data = np.load(file_path)
    mode_is_kcount = mode in K_MODE_SET
    node_dim = k + 1 if mode_is_kcount else k
    if data.ndim == 1:
        data = data.reshape((-1, node_dim))
    base = np.array(data, dtype=np.float32).reshape((-1, node_dim))
    kcol = None
    if mode_is_kcount and base.shape[1] == k + 1:
        kcol = base[:, :1]
        base = base[:, 1:]
    mode_for_attrs = "pre_order" if mode_is_kcount else mode

    if k == 39:
        outputs = build_spline_dataset(
            base,
            mode_for_attrs,
            n_rotations,
            n_samples,
            smooth,
            enable_rotation,
            enable_scaling,
            refit_splines,
        )
    else:
        base, _root, not_zero_mask = zero_root(base, mode_for_attrs)
        outputs = [base]
        for i in range(n_rotations):
            angle = random.randint(10, 350)
            axis = np.random.rand(3)
            rot = rotation_matrix(angle, axis)
            rotated = base.copy()
            rotated[not_zero_mask, :3] = rotated[not_zero_mask, :3] @ rot.T
            outputs.append(rotated)

    if add_structure_features and kcol is not None:
        augmented = []
        for arr in outputs:
            struct = _build_structure_channels(
                kcol=kcol,
                xyz=arr[:, :3],
                mode=mode,
                add_depth_norm=structure_add_depth_norm,
                add_parent_delta_xyz=structure_add_parent_delta_xyz,
                add_pathlen_root_norm=structure_add_pathlen_root_norm,
                depth_cap=structure_depth_cap,
                pathlen_cap=structure_pathlen_cap,
                delta_scale=structure_delta_scale,
            )
            if struct.shape[1] > 0:
                arr = np.hstack((arr, struct))
            augmented.append(arr.astype(np.float32, copy=False))
        outputs = augmented

    if delta_xyz_parent and kcol is not None:
        outputs = [
            _to_parent_delta_xyz(arr, kcol=kcol, mode=mode, root_zero=delta_xyz_root_zero)
            for arr in outputs
        ]

    if kcol is not None:
        outputs = [np.hstack((kcol, arr)).astype(np.float32, copy=False) for arr in outputs]

    written = 0
    skipped = 0
    failures_total = 0
    for idx, arr in enumerate(outputs):
        name = os.path.basename(file_path)
        if idx > 0:
            name = f"rot{idx}-" + name
        out_path = os.path.join(output_dir, name)
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue
        np.save(out_path, arr)
        written += 1

    return written, skipped, failures_total


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: depth cut, zero-root, rotations, spline refit (k=39)."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Input folder with .npy trees")
    parser.add_argument("--output", default=None, help="Output folder")
    parser.add_argument("--k", type=int, default=39, help="Feature dimension (default: 39)")
    parser.add_argument(
        "--mode",
        default="pre_order",
        choices=["pre_order", "post_order", "pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"],
    )
    parser.add_argument("--n-rotations", type=int, default=0)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--spline-samples", type=int, default=50)
    parser.add_argument("--spline-smooth", type=float, default=0.0000001)
    parser.add_argument("--enable-rotation", action="store_true")
    parser.add_argument("--enable-scaling", action="store_true")
    parser.add_argument("--refit-splines", dest="refit_splines", action="store_true", default=True)
    parser.add_argument("--no-refit-splines", dest="refit_splines", action="store_false")
    parser.add_argument("--add-structure-features", dest="add_structure_features", action="store_true")
    parser.add_argument("--no-add-structure-features", dest="add_structure_features", action="store_false")
    parser.add_argument("--structure-add-depth-norm", dest="structure_add_depth_norm", action="store_true")
    parser.add_argument("--no-structure-add-depth-norm", dest="structure_add_depth_norm", action="store_false")
    parser.add_argument("--structure-add-parent-delta-xyz", dest="structure_add_parent_delta_xyz", action="store_true")
    parser.add_argument("--no-structure-add-parent-delta-xyz", dest="structure_add_parent_delta_xyz", action="store_false")
    parser.add_argument("--structure-add-pathlen-root-norm", dest="structure_add_pathlen_root_norm", action="store_true")
    parser.add_argument("--no-structure-add-pathlen-root-norm", dest="structure_add_pathlen_root_norm", action="store_false")
    parser.add_argument("--structure-depth-cap", type=float, default=150.0)
    parser.add_argument("--structure-pathlen-cap", type=float, default=0.0)
    parser.add_argument("--structure-delta-scale", type=float, default=1.0)
    parser.add_argument("--delta-xyz-parent", dest="delta_xyz_parent", action="store_true")
    parser.add_argument("--no-delta-xyz-parent", dest="delta_xyz_parent", action="store_false")
    parser.add_argument("--delta-xyz-root-zero", dest="delta_xyz_root_zero", action="store_true")
    parser.add_argument("--no-delta-xyz-root-zero", dest="delta_xyz_root_zero", action="store_false")
    parser.add_argument("--erase-output", action="store_true")
    parser.set_defaults(
        add_structure_features=False,
        structure_add_depth_norm=True,
        structure_add_parent_delta_xyz=True,
        structure_add_pathlen_root_norm=True,
        delta_xyz_parent=False,
        delta_xyz_root_zero=True,
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        default_path = os.path.join(os.path.dirname(__file__), "prepare_dataset_config.yaml")
        if os.path.exists(default_path):
            config_path = default_path

    if config_path:
        cfg = load_config(config_path)
        paths = cfg.get("paths", {})
        params = cfg.get("params", {})
        args.input = paths.get("input", args.input)
        args.output = paths.get("output", args.output)
        args.k = int(params.get("k", args.k))
        args.mode = params.get("mode", args.mode)
        args.n_rotations = int(params.get("n_rotations", args.n_rotations))
        args.pattern = params.get("pattern", args.pattern)
        args.overwrite = bool(params.get("overwrite", args.overwrite))
        if "seed" in params:
            args.seed = params.get("seed")
        if "spline_samples" in params:
            args.spline_samples = int(params.get("spline_samples"))
        if "spline_smooth" in params:
            args.spline_smooth = float(params.get("spline_smooth"))
        if "enable_rotation" in params:
            args.enable_rotation = bool(params.get("enable_rotation"))
        if "enable_scaling" in params:
            args.enable_scaling = bool(params.get("enable_scaling"))
        if "refit_splines" in params:
            args.refit_splines = bool(params.get("refit_splines"))
        if "erase_output" in params:
            args.erase_output = bool(params.get("erase_output"))
        if "add_structure_features" in params:
            args.add_structure_features = bool(params.get("add_structure_features"))
        if "structure_add_depth_norm" in params:
            args.structure_add_depth_norm = bool(params.get("structure_add_depth_norm"))
        if "structure_add_parent_delta_xyz" in params:
            args.structure_add_parent_delta_xyz = bool(params.get("structure_add_parent_delta_xyz"))
        if "structure_add_pathlen_root_norm" in params:
            args.structure_add_pathlen_root_norm = bool(params.get("structure_add_pathlen_root_norm"))
        if "structure_depth_cap" in params:
            args.structure_depth_cap = float(params.get("structure_depth_cap"))
        if "structure_pathlen_cap" in params:
            args.structure_pathlen_cap = float(params.get("structure_pathlen_cap"))
        if "structure_delta_scale" in params:
            args.structure_delta_scale = float(params.get("structure_delta_scale"))
        if "delta_xyz_parent" in params:
            args.delta_xyz_parent = bool(params.get("delta_xyz_parent"))
        if "delta_xyz_root_zero" in params:
            args.delta_xyz_root_zero = bool(params.get("delta_xyz_root_zero"))

    if not args.input or not args.output:
        raise SystemExit("Error: --input and --output are required (or provide them in a config).")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    if args.erase_output:
        erase_all_files(args.output)

    files = iter_files(args.input, args.pattern)
    total_written = 0
    total_skipped = 0
    total_failures = 0

    if args.n_rotations <= 0:
        raise SystemExit("Error: n_rotations must be >= 1 to match datasets.ipynb behavior.")
    if args.add_structure_features and args.mode not in K_MODE_SET:
        print("Warning: add_structure_features is enabled but mode is not a K-mode; no structure channels will be added.")
    if args.delta_xyz_parent and args.mode not in K_MODE_SET:
        print("Warning: delta_xyz_parent is enabled but mode is not a K-mode; xyz deltas will not be applied.")

    for idx, file_path in enumerate(files, start=1):
        written, skipped, failures = process_file(
            file_path,
            args.output,
            args.k,
            args.mode,
            args.n_rotations,
            args.overwrite,
            args.spline_samples,
            args.spline_smooth,
            args.enable_rotation,
            args.enable_scaling,
            args.refit_splines,
            args.add_structure_features,
            args.structure_add_depth_norm,
            args.structure_add_parent_delta_xyz,
            args.structure_add_pathlen_root_norm,
            args.structure_depth_cap,
            args.structure_pathlen_cap,
            args.structure_delta_scale,
            args.delta_xyz_parent,
            args.delta_xyz_root_zero,
        )
        total_written += written
        total_skipped += skipped
        total_failures += failures
        print(f"[{idx}/{len(files)}] {os.path.basename(file_path)} -> +{written} (-{skipped})")

    print(f"done: {total_written} written, {total_skipped} skipped, spline failures {total_failures}")


if __name__ == "__main__":
    main()
