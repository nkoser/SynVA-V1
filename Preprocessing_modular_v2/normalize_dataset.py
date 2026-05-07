import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


@dataclass
class Stats:
    feature_start: int
    feature_end: int
    center: np.ndarray
    scale: np.ndarray
    clip_value: float
    method: str
    quantile: float


SPATIAL_X_IDX = {1, 4, 5, 6, 7, 8, 9, 10, 11}
SPATIAL_Y_IDX = {2, 12, 13, 14, 15, 16, 17, 18, 19}
SPATIAL_Z_IDX = {3, 20, 21, 22, 23, 24, 25, 26, 27}
KNOT_IDX = set(range(28, 40))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_files(folder: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))


def _reshape_tree(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        if arr.size % 40 == 0:
            return arr.reshape((-1, 40))
        if arr.size % 39 == 0:
            return arr.reshape((-1, 39))
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _resolve_feature_end(feature_end: Optional[int], dim: int) -> int:
    if feature_end is None:
        return int(dim)
    end = int(feature_end)
    if end <= 0:
        end = int(dim + end)
    return max(0, min(int(dim), end))


def _collect_train_matrix(train_files: list[str], feature_start: int, feature_end: Optional[int]) -> tuple[np.ndarray, int]:
    rows = []
    final_end = None
    for fp in train_files:
        arr = _reshape_tree(np.load(fp, mmap_mode="r"))
        if final_end is None:
            final_end = _resolve_feature_end(feature_end, arr.shape[1])
        if final_end <= feature_start:
            raise ValueError(f"Invalid feature range: start={feature_start}, end={final_end}")
        if arr.shape[1] < final_end:
            raise ValueError(
                f"File {fp} has dim {arr.shape[1]} but requires at least {final_end}."
            )
        part = np.asarray(arr[:, feature_start:final_end], dtype=np.float32)
        if part.size > 0:
            rows.append(part)
    if not rows:
        raise ValueError("No train rows found to compute normalization stats.")
    return np.concatenate(rows, axis=0), int(final_end)


def compute_stats(
    train_files: list[str],
    feature_start: int,
    feature_end: Optional[int],
    method: str,
    quantile: float,
    clip_value: float,
    eps: float,
    geometry_isotropic: bool = True,
    geometry_normalize_knots: bool = False,
    geometry_knots_quantile: Optional[float] = None,
) -> Stats:
    train_mat, final_end = _collect_train_matrix(train_files, feature_start, feature_end)

    method = str(method).lower().strip()
    if method == "robust_clip":
        center = np.median(train_mat, axis=0)
        abs_dev = np.abs(train_mat - center)
        scale = np.percentile(abs_dev, quantile, axis=0)
    elif method == "maxabs_clip":
        center = np.zeros((train_mat.shape[1],), dtype=np.float32)
        scale = np.max(np.abs(train_mat), axis=0)
    elif method == "zscore_clip":
        center = np.mean(train_mat, axis=0)
        scale = np.std(train_mat, axis=0)
    elif method == "geometry_clip":
        # Start from robust per-feature stats for non-spatial channels.
        center = np.median(train_mat, axis=0).astype(np.float32)
        abs_dev = np.abs(train_mat - center)
        scale = np.percentile(abs_dev, quantile, axis=0).astype(np.float32)

        x_vals, y_vals, z_vals = [], [], []
        knot_vals = []
        for fp in train_files:
            arr = _reshape_tree(np.load(fp, mmap_mode="r"))
            if arr.shape[1] < 40:
                continue
            x_vals.append(np.asarray(arr[:, 1], dtype=np.float32))
            y_vals.append(np.asarray(arr[:, 2], dtype=np.float32))
            z_vals.append(np.asarray(arr[:, 3], dtype=np.float32))
            if geometry_normalize_knots:
                knot_vals.append(np.asarray(arr[:, 28:40], dtype=np.float32))

        if not x_vals or not y_vals or not z_vals:
            raise ValueError("geometry_clip requires 40D inputs with xyz/spline channels.")

        cx = float(np.median(np.concatenate(x_vals)))
        cy = float(np.median(np.concatenate(y_vals)))
        cz = float(np.median(np.concatenate(z_vals)))

        ax_chunks, ay_chunks, az_chunks = [], [], []
        for fp in train_files:
            arr = _reshape_tree(np.load(fp, mmap_mode="r"))
            if arr.shape[1] < 40:
                continue
            ax = np.concatenate(
                (
                    np.abs(np.asarray(arr[:, 1], dtype=np.float32) - cx),
                    np.abs(np.asarray(arr[:, 4:12], dtype=np.float32).reshape(-1) - cx),
                ),
                axis=0,
            )
            ay = np.concatenate(
                (
                    np.abs(np.asarray(arr[:, 2], dtype=np.float32) - cy),
                    np.abs(np.asarray(arr[:, 12:20], dtype=np.float32).reshape(-1) - cy),
                ),
                axis=0,
            )
            az = np.concatenate(
                (
                    np.abs(np.asarray(arr[:, 3], dtype=np.float32) - cz),
                    np.abs(np.asarray(arr[:, 20:28], dtype=np.float32).reshape(-1) - cz),
                ),
                axis=0,
            )
            ax_chunks.append(ax)
            ay_chunks.append(ay)
            az_chunks.append(az)

        if geometry_isotropic:
            all_abs = np.concatenate(ax_chunks + ay_chunks + az_chunks, axis=0)
            s_xyz = float(np.percentile(all_abs, quantile))
            sx = sy = sz = s_xyz
        else:
            sx = float(np.percentile(np.concatenate(ax_chunks), quantile))
            sy = float(np.percentile(np.concatenate(ay_chunks), quantile))
            sz = float(np.percentile(np.concatenate(az_chunks), quantile))

        sx = 1.0 if abs(sx) < float(eps) else sx
        sy = 1.0 if abs(sy) < float(eps) else sy
        sz = 1.0 if abs(sz) < float(eps) else sz

        knot_center = None
        knot_scale = None
        if geometry_normalize_knots and knot_vals:
            knot_mat = np.concatenate(knot_vals, axis=0)
            kq = float(geometry_knots_quantile) if geometry_knots_quantile is not None else float(quantile)
            knot_center = np.median(knot_mat, axis=0).astype(np.float32)
            knot_abs = np.abs(knot_mat - knot_center)
            knot_scale = np.percentile(knot_abs, kq, axis=0).astype(np.float32)
            knot_scale = np.where(np.abs(knot_scale) < float(eps), np.float32(1.0), knot_scale)

        for global_idx in range(feature_start, final_end):
            local_idx = global_idx - feature_start
            if global_idx in SPATIAL_X_IDX:
                center[local_idx] = np.float32(cx)
                scale[local_idx] = np.float32(sx)
            elif global_idx in SPATIAL_Y_IDX:
                center[local_idx] = np.float32(cy)
                scale[local_idx] = np.float32(sy)
            elif global_idx in SPATIAL_Z_IDX:
                center[local_idx] = np.float32(cz)
                scale[local_idx] = np.float32(sz)
            elif global_idx in KNOT_IDX and not geometry_normalize_knots:
                center[local_idx] = np.float32(0.0)
                scale[local_idx] = np.float32(1.0)

        if knot_center is not None and knot_scale is not None:
            for global_idx in sorted(KNOT_IDX):
                if not (feature_start <= global_idx < final_end):
                    continue
                local_idx = global_idx - feature_start
                knot_local = global_idx - 28
                center[local_idx] = knot_center[knot_local]
                scale[local_idx] = knot_scale[knot_local]
    else:
        raise ValueError(
            "Unsupported params.method. Use one of: robust_clip, maxabs_clip, zscore_clip, geometry_clip"
        )

    scale = np.asarray(scale, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    scale = np.where(np.abs(scale) < float(eps), np.float32(1.0), scale)

    return Stats(
        feature_start=int(feature_start),
        feature_end=int(final_end),
        center=center,
        scale=scale,
        clip_value=float(clip_value),
        method=method,
        quantile=float(quantile),
    )


def apply_stats(
    arr: np.ndarray,
    stats: Stats,
    skip_clip_global_indices: Optional[set[int]] = None,
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    part = out[:, stats.feature_start:stats.feature_end]
    part = (part - stats.center) / stats.scale
    if skip_clip_global_indices:
        clip_mask = np.ones((part.shape[1],), dtype=bool)
        for global_idx in skip_clip_global_indices:
            if stats.feature_start <= int(global_idx) < stats.feature_end:
                clip_mask[int(global_idx) - stats.feature_start] = False
        if clip_mask.any():
            part[:, clip_mask] = np.clip(part[:, clip_mask], -stats.clip_value, stats.clip_value)
    else:
        part = np.clip(part, -stats.clip_value, stats.clip_value)
    out[:, stats.feature_start:stats.feature_end] = part
    return out


def enforce_root_zero_xyz(
    raw_arr: np.ndarray,
    norm_arr: np.ndarray,
    root_xyz_indices: tuple[int, int, int],
    root_position: str,
    root_eps: float,
) -> np.ndarray:
    root_position = str(root_position).lower().strip()
    root_row = -1 if root_position in {"last", "post_order"} else 0
    if raw_arr.shape[0] == 0:
        return norm_arr
    root_row = root_row if root_row >= 0 else raw_arr.shape[0] + root_row
    if root_row < 0 or root_row >= raw_arr.shape[0]:
        return norm_arr
    for idx in root_xyz_indices:
        if idx < 0 or idx >= raw_arr.shape[1]:
            continue
        # Only enforce hard zero when the raw root coordinate was already zero-centered.
        if abs(float(raw_arr[root_row, idx])) <= float(root_eps):
            norm_arr[root_row, idx] = np.float32(0.0)
    return norm_arr


def summarize_split(files: list[str], feature_start: int, feature_end: Optional[int]) -> dict:
    values = []
    final_end = None
    for fp in files:
        arr = _reshape_tree(np.load(fp, mmap_mode="r"))
        if final_end is None:
            final_end = _resolve_feature_end(feature_end, arr.shape[1])
        if final_end <= feature_start:
            raise ValueError(f"Invalid feature range: start={feature_start}, end={final_end}")
        part = np.asarray(arr[:, feature_start:final_end], dtype=np.float32)
        values.append(part.reshape(-1))
    if not values:
        return {"count": 0}
    x = np.concatenate(values)
    ax = np.abs(x)
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "abs_q99": float(np.percentile(ax, 99.0)),
        "abs_q99_9": float(np.percentile(ax, 99.9)),
        "abs_q99_99": float(np.percentile(ax, 99.99)),
        "abs_max": float(ax.max()),
    }


def normalize_split(
    input_dir: str,
    output_dir: str,
    pattern: str,
    stats: Stats,
    overwrite: bool,
    preserve_root_zero_xyz: bool = False,
    root_xyz_indices: tuple[int, int, int] = (1, 2, 3),
    root_position: str = "first",
    root_eps: float = 1e-8,
    skip_clip_global_indices: Optional[set[int]] = None,
) -> tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    files = iter_files(input_dir, pattern)
    written = 0
    skipped = 0
    for src in files:
        dst = os.path.join(output_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            skipped += 1
            continue
        arr = _reshape_tree(np.load(src))
        if arr.shape[1] < stats.feature_end:
            raise ValueError(
                f"File {src} has dim {arr.shape[1]} but requires at least {stats.feature_end}."
            )
        out = apply_stats(arr, stats, skip_clip_global_indices=skip_clip_global_indices)
        if preserve_root_zero_xyz:
            out = enforce_root_zero_xyz(
                raw_arr=arr,
                norm_arr=out,
                root_xyz_indices=root_xyz_indices,
                root_position=root_position,
                root_eps=root_eps,
            )
        np.save(dst, out.astype(np.float32, copy=False))
        written += 1
    return written, skipped


def write_stats(path: str, stats: Stats, extra: dict) -> None:
    payload = {
        "method": stats.method,
        "quantile": stats.quantile,
        "clip_value": stats.clip_value,
        "feature_start": stats.feature_start,
        "feature_end": stats.feature_end,
        "center": [float(v) for v in stats.center.tolist()],
        "scale": [float(v) for v in stats.scale.tolist()],
        "meta": extra,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_output_dir(input_dir: str, output_root: Optional[str], in_place: bool) -> str:
    if in_place:
        return input_dir
    if not output_root:
        raise ValueError("paths.output_root is required when params.in_place is false.")
    split_name = os.path.basename(os.path.normpath(input_dir))
    return os.path.join(output_root, split_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize tree feature datasets using train-derived statistics."
    )
    parser.add_argument("--config", default="normalize_dataset_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    train_dir = paths.get("train_dir")
    val_dir = paths.get("val_dir")
    test_dir = paths.get("test_dir")
    output_root = paths.get("output_root")
    stats_path = paths.get("stats_path")

    if not train_dir or not val_dir or not test_dir:
        raise ValueError("paths.train_dir, paths.val_dir, paths.test_dir are required.")

    pattern = str(params.get("pattern", "*.npy"))
    method = str(params.get("method", "robust_clip"))
    quantile = float(params.get("quantile", 99.9))
    clip_value = float(params.get("clip_value", 1.0))
    eps = float(params.get("eps", 1e-6))
    overwrite = bool(params.get("overwrite", False))
    in_place = bool(params.get("in_place", False))
    feature_start = int(params.get("feature_start", 1))
    feature_end = params.get("feature_end", None)
    geometry_isotropic = bool(params.get("geometry_isotropic", True))
    geometry_normalize_knots = bool(params.get("geometry_normalize_knots", False))
    geometry_knots_quantile = params.get("geometry_knots_quantile", None)
    preserve_root_zero_xyz = bool(params.get("preserve_root_zero_xyz", False))
    root_position = str(params.get("root_position", "first"))
    root_xyz_indices_cfg = params.get("root_xyz_indices", [1, 2, 3])
    root_eps = float(params.get("root_eps", 1e-8))
    if not isinstance(root_xyz_indices_cfg, (list, tuple)) or len(root_xyz_indices_cfg) != 3:
        raise ValueError("params.root_xyz_indices must be a list/tuple with exactly 3 indices.")
    root_xyz_indices = tuple(int(v) for v in root_xyz_indices_cfg)
    if geometry_knots_quantile is not None:
        geometry_knots_quantile = float(geometry_knots_quantile)
    if feature_end is not None:
        feature_end = int(feature_end)
    skip_clip_global_indices = None
    if method.lower().strip() == "geometry_clip" and not geometry_normalize_knots:
        # Keep spline knot channels unchanged when knot normalization is disabled.
        skip_clip_global_indices = set(KNOT_IDX)

    train_files = iter_files(train_dir, pattern)
    val_files = iter_files(val_dir, pattern)
    test_files = iter_files(test_dir, pattern)

    if not train_files:
        raise FileNotFoundError(f"No train files found in {train_dir} with pattern {pattern}")

    print("Computing stats from train split ...")
    before_train = summarize_split(train_files, feature_start=feature_start, feature_end=feature_end)
    before_val = summarize_split(val_files, feature_start=feature_start, feature_end=feature_end)
    before_test = summarize_split(test_files, feature_start=feature_start, feature_end=feature_end)

    stats = compute_stats(
        train_files=train_files,
        feature_start=feature_start,
        feature_end=feature_end,
        method=method,
        quantile=quantile,
        clip_value=clip_value,
        eps=eps,
        geometry_isotropic=geometry_isotropic,
        geometry_normalize_knots=geometry_normalize_knots,
        geometry_knots_quantile=geometry_knots_quantile,
    )

    out_train = _resolve_output_dir(train_dir, output_root, in_place)
    out_val = _resolve_output_dir(val_dir, output_root, in_place)
    out_test = _resolve_output_dir(test_dir, output_root, in_place)

    print(f"Applying normalization ({method}) ...")
    w_train, s_train = normalize_split(
        train_dir,
        out_train,
        pattern,
        stats,
        overwrite,
        preserve_root_zero_xyz=preserve_root_zero_xyz,
        root_xyz_indices=root_xyz_indices,
        root_position=root_position,
        root_eps=root_eps,
        skip_clip_global_indices=skip_clip_global_indices,
    )
    w_val, s_val = normalize_split(
        val_dir,
        out_val,
        pattern,
        stats,
        overwrite,
        preserve_root_zero_xyz=preserve_root_zero_xyz,
        root_xyz_indices=root_xyz_indices,
        root_position=root_position,
        root_eps=root_eps,
        skip_clip_global_indices=skip_clip_global_indices,
    )
    w_test, s_test = normalize_split(
        test_dir,
        out_test,
        pattern,
        stats,
        overwrite,
        preserve_root_zero_xyz=preserve_root_zero_xyz,
        root_xyz_indices=root_xyz_indices,
        root_position=root_position,
        root_eps=root_eps,
        skip_clip_global_indices=skip_clip_global_indices,
    )

    after_train = summarize_split(iter_files(out_train, pattern), stats.feature_start, stats.feature_end)
    after_val = summarize_split(iter_files(out_val, pattern), stats.feature_start, stats.feature_end)
    after_test = summarize_split(iter_files(out_test, pattern), stats.feature_start, stats.feature_end)

    if not stats_path:
        stats_root = output_root if output_root else os.path.dirname(os.path.normpath(train_dir))
        stats_path = os.path.join(stats_root, "normalization_stats.json")

    write_stats(
        stats_path,
        stats,
        extra={
            "config_path": os.path.abspath(args.config),
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
            "output_train_dir": out_train,
            "output_val_dir": out_val,
            "output_test_dir": out_test,
            "pattern": pattern,
            "preserve_root_zero_xyz": preserve_root_zero_xyz,
            "root_xyz_indices": list(root_xyz_indices),
            "root_position": root_position,
            "root_eps": root_eps,
            "skip_clip_global_indices": (
                sorted(int(v) for v in skip_clip_global_indices)
                if skip_clip_global_indices is not None
                else []
            ),
            "written": {"train": w_train, "val": w_val, "test": w_test},
            "skipped": {"train": s_train, "val": s_val, "test": s_test},
            "before": {"train": before_train, "val": before_val, "test": before_test},
            "after": {"train": after_train, "val": after_val, "test": after_test},
        },
    )

    print("Done.")
    print(f"Output train: {out_train} | written={w_train} skipped={s_train}")
    print(f"Output val:   {out_val} | written={w_val} skipped={s_val}")
    print(f"Output test:  {out_test} | written={w_test} skipped={s_test}")
    print(f"Stats saved to: {stats_path}")
    print("Before/After abs ranges (selected features):")
    print("  train before:", before_train)
    print("  train after: ", after_train)
    print("  val   before:", before_val)
    print("  val   after: ", after_val)
    print("  test  before:", before_test)
    print("  test  after: ", after_test)


if __name__ == "__main__":
    main()
