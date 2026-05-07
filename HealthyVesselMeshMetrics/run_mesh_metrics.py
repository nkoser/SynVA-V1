#!/usr/bin/env python3
"""Mesh comparison metrics for Healthy vessel GT vs generated reconstructions.

SSIM, PSNR, and BRISK are image-based metrics. This script therefore renders
each mesh pair into deterministic multi-view raster images and computes those
metrics on the rendered occupancy/depth images. It also reports native sampled
surface metrics such as Chamfer, Hausdorff, F-score, area ratio, and volume
ratio.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "healthy_vessel_mesh_metrics",
        "output_dir": "HealthyVesselMeshMetrics/output/healthy_vessel_mesh_metrics",
    },
    "data": {
        "gt_mesh_roots": [
            "/data/healthy_vessel_decapped",
            "/data/healthy_vessel",
        ],
        "generated_mesh_roots": {
            # Fill once generated mesh reconstructions exist.
            # "physio_v5": "/data/Stage2_FlowMatching_Physio/generated/physio_v5/meshes_sweep",
        },
        "file_extensions": [".obj", ".ply", ".stl", ".off"],
        "include_regex": None,
        "exclude_regex_gt": None,
        "exclude_regex_generated": r"(_gt|_gt_local|_local)$",
        "max_cases": None,
    },
    "surface": {
        "points_per_mesh": 20000,
        "seed": 123,
        "fscore_thresholds": [0.005, 0.01, 0.02],
    },
    "render": {
        "enabled": True,
        "points_per_mesh": 60000,
        "resolution": 256,
        "views": ["xy", "xz", "yz"],
        "bbox_mode": "union",  # union or gt
        "padding_fraction": 0.06,
        "depth_mode": "front",  # front or back
        "save_debug_images": 0,
        "debug_dir": "debug_renders",
    },
    "image_metrics": {
        "ssim": True,
        "psnr": True,
        "brisk": True,
        "brisk_image": "occupancy",  # occupancy or depth
        "brisk_lowe_ratio": 0.75,
    },
}


KNOWN_CASE_SUFFIXES = (
    "_vessel_submesh_closed",
    "_gt_local",
    "_gt",
    "_local",
    "_generated",
    "_gen",
)


@dataclasses.dataclass(frozen=True)
class MeshRecord:
    label: str
    case_id: str
    path: Path
    kind: str


@dataclasses.dataclass(frozen=True)
class PairRecord:
    label: str
    case_id: str
    gt_path: Path
    generated_path: Path

    @property
    def pair_id(self) -> str:
        return f"{self.label}:{self.case_id}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: Any, base: Optional[Path] = None) -> Path:
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return p
    return (base or repo_root()) / p


def deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        user_cfg = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Use JSON or install pyyaml.")
        user_cfg = yaml.safe_load(text) or {}
    return deep_update(cfg, user_cfg)


def parse_label_path(values: Optional[Sequence[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Expected LABEL=PATH, got: {value}")
        parsed[label] = path
    return parsed


def strip_known_suffixes(name: str) -> str:
    out = name
    changed = True
    while changed:
        changed = False
        for suffix in KNOWN_CASE_SUFFIXES:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def case_id_from_path(path: Path) -> str:
    stem = strip_known_suffixes(path.stem)
    if stem.lower() in {"mesh", "model", "surface", "vessel"}:
        parent = path.parent.name
        stem = path.parent.parent.name if parent == "01_mesh" else parent
    return strip_known_suffixes(stem)


def mesh_sort_key(path: Path, ext_priority: Mapping[str, int]) -> Tuple[int, int, str]:
    return (ext_priority.get(path.suffix.lower(), 10000), len(path.parts), path.as_posix())


def discover_meshes(
    root: Path,
    label: str,
    kind: str,
    extensions: Sequence[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    warnings_out: List[str],
) -> List[MeshRecord]:
    if not root.exists():
        warnings_out.append(f"[missing] {kind}:{label} root not found: {root}")
        return []

    exts = [e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in extensions]
    ext_priority = {ext: i for i, ext in enumerate(exts)}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_priority]
    files.sort(key=lambda p: mesh_sort_key(p, ext_priority))

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    records: List[MeshRecord] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        case_id = case_id_from_path(path)
        match_text = f"{path.stem} {rel} {case_id}"
        if include and not include.search(match_text):
            continue
        if exclude and exclude.search(match_text):
            continue
        records.append(MeshRecord(label=label, case_id=case_id, path=path, kind=kind))
    return records


def index_by_case(records: Iterable[MeshRecord], warnings_out: List[str]) -> Dict[str, MeshRecord]:
    out: Dict[str, MeshRecord] = {}
    for record in records:
        if record.case_id in out:
            previous = out[record.case_id]
            warnings_out.append(
                "[duplicate] "
                f"{record.kind}:{record.label}:{record.case_id} keeps {previous.path}, ignores {record.path}"
            )
            continue
        out[record.case_id] = record
    return out


def build_pairs(
    gt_records: Sequence[MeshRecord],
    generated_records: Mapping[str, Sequence[MeshRecord]],
    max_cases: Optional[int],
    warnings_out: List[str],
) -> Tuple[List[PairRecord], List[Dict[str, str]]]:
    gt_by_case = index_by_case(gt_records, warnings_out)
    missing_rows: List[Dict[str, str]] = []
    pairs: List[PairRecord] = []

    if not generated_records:
        warnings_out.append("[info] No generated mesh roots configured yet; pair manifest is empty.")
        return pairs, missing_rows

    for label, records in generated_records.items():
        gen_by_case = index_by_case(records, warnings_out)
        common = sorted(set(gt_by_case).intersection(gen_by_case))
        if max_cases is not None:
            common = common[: int(max_cases)]
        for case_id in common:
            pairs.append(
                PairRecord(
                    label=label,
                    case_id=case_id,
                    gt_path=gt_by_case[case_id].path,
                    generated_path=gen_by_case[case_id].path,
                )
            )
        for case_id in sorted(set(gen_by_case) - set(gt_by_case))[:200]:
            missing_rows.append({"label": label, "case_id": case_id, "reason": "missing_gt"})
        for case_id in sorted(set(gt_by_case) - set(gen_by_case))[:200]:
            missing_rows.append({"label": label, "case_id": case_id, "reason": "missing_generated"})

    return pairs, missing_rows


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if hasattr(g, "vertices")]
        if not parts:
            raise ValueError(f"Scene contains no mesh geometry: {path}")
        mesh = trimesh.util.concatenate(parts)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return mesh


def sample_mesh_points(mesh: Any, n_points: int, seed: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            pts = np.asarray(mesh.sample(int(n_points)), dtype=np.float32)
        finally:
            np.random.set_state(state)
        return pts
    rng = np.random.default_rng(seed)
    replace = len(vertices) < int(n_points)
    idx = rng.choice(len(vertices), size=int(n_points), replace=replace)
    return vertices[idx].astype(np.float32)


def chamfer_and_hausdorff(points_a: np.ndarray, points_b: np.ndarray) -> Dict[str, float]:
    from scipy.spatial import cKDTree

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_a_to_b = tree_b.query(points_a, workers=-1)[0]
    d_b_to_a = tree_a.query(points_b, workers=-1)[0]
    return {
        "chamfer_mean": float(0.5 * (d_a_to_b.mean() + d_b_to_a.mean())),
        "chamfer_a_to_b": float(d_a_to_b.mean()),
        "chamfer_b_to_a": float(d_b_to_a.mean()),
        "hausdorff": float(max(d_a_to_b.max(), d_b_to_a.max())),
        "hausdorff_p95": float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95))),
        "_d_a_to_b": d_a_to_b,
        "_d_b_to_a": d_b_to_a,
    }


def fscore_from_distances(d_a_to_b: np.ndarray, d_b_to_a: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for threshold in thresholds:
        t = float(threshold)
        precision = float(np.mean(d_b_to_a <= t))
        recall = float(np.mean(d_a_to_b <= t))
        fscore = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
        suffix = str(t).replace(".", "p")
        out[f"fscore_{suffix}"] = float(fscore)
        out[f"precision_{suffix}"] = float(precision)
        out[f"recall_{suffix}"] = float(recall)
    return out


def safe_float(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def mesh_scalar_metrics(gt_mesh: Any, gen_mesh: Any) -> Dict[str, float]:
    gt_area = safe_float(getattr(gt_mesh, "area", np.nan))
    gen_area = safe_float(getattr(gen_mesh, "area", np.nan))
    gt_volume = safe_float(getattr(gt_mesh, "volume", np.nan))
    gen_volume = safe_float(getattr(gen_mesh, "volume", np.nan))
    return {
        "gt_vertices": float(len(gt_mesh.vertices)),
        "gen_vertices": float(len(gen_mesh.vertices)),
        "gt_faces": float(len(gt_mesh.faces)) if getattr(gt_mesh, "faces", None) is not None else 0.0,
        "gen_faces": float(len(gen_mesh.faces)) if getattr(gen_mesh, "faces", None) is not None else 0.0,
        "gt_area": gt_area,
        "gen_area": gen_area,
        "area_ratio_gen_gt": gen_area / gt_area if gt_area > 1e-12 else float("nan"),
        "gt_volume": gt_volume,
        "gen_volume": gen_volume,
        "volume_ratio_gen_gt": gen_volume / gt_volume if abs(gt_volume) > 1e-12 else float("nan"),
        "gt_watertight": float(bool(getattr(gt_mesh, "is_watertight", False))),
        "gen_watertight": float(bool(getattr(gen_mesh, "is_watertight", False))),
    }


def view_axes(view: str) -> Tuple[int, int, int]:
    view = view.lower()
    if view == "xy":
        return 0, 1, 2
    if view == "xz":
        return 0, 2, 1
    if view == "yz":
        return 1, 2, 0
    raise ValueError(f"Unsupported render view: {view}. Use xy, xz, yz.")


def render_points(
    points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    view: str,
    resolution: int,
    depth_mode: str,
) -> Dict[str, np.ndarray]:
    ax_u, ax_v, ax_d = view_axes(view)
    lo = bounds_min.astype(np.float32)
    hi = bounds_max.astype(np.float32)
    extent = np.maximum(hi - lo, 1e-8)

    u = (points[:, ax_u] - lo[ax_u]) / extent[ax_u]
    v = (points[:, ax_v] - lo[ax_v]) / extent[ax_v]
    d = (points[:, ax_d] - lo[ax_d]) / extent[ax_d]
    valid = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0) & (d >= 0.0) & (d <= 1.0)
    u = u[valid]
    v = v[valid]
    d = d[valid]

    width = int(resolution)
    x = np.clip((u * (width - 1)).astype(np.int32), 0, width - 1)
    y = np.clip(((1.0 - v) * (width - 1)).astype(np.int32), 0, width - 1)
    flat = y * width + x

    occupancy = np.zeros((width, width), dtype=np.float32)
    depth = np.zeros((width, width), dtype=np.float32)
    if len(flat) == 0:
        return {"occupancy": occupancy, "depth": depth}

    occupancy.reshape(-1)[flat] = 1.0
    if depth_mode == "front":
        values = 1.0 - d
        init = np.zeros(width * width, dtype=np.float32)
        np.maximum.at(init, flat, values.astype(np.float32))
    elif depth_mode == "back":
        values = d
        init = np.zeros(width * width, dtype=np.float32)
        np.maximum.at(init, flat, values.astype(np.float32))
    else:
        raise ValueError(f"Unsupported depth_mode: {depth_mode}")
    depth = init.reshape(width, width)
    return {"occupancy": occupancy, "depth": depth}


def pair_bounds(gt_points: np.ndarray, gen_points: np.ndarray, mode: str, padding_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "gt":
        all_pts = gt_points
    elif mode == "union":
        all_pts = np.concatenate([gt_points, gen_points], axis=0)
    else:
        raise ValueError(f"Unsupported bbox_mode: {mode}")
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    extent = hi - lo
    pad = max(float(np.max(extent)) * float(padding_fraction), 1e-6)
    return lo - pad, hi + pad


def image_metrics(gt_img: np.ndarray, gen_img: np.ndarray, prefix: str, enabled: Mapping[str, Any]) -> Dict[str, float]:
    from skimage.metrics import structural_similarity

    out: Dict[str, float] = {}
    if enabled.get("ssim", True):
        out[f"{prefix}_ssim"] = float(structural_similarity(gt_img, gen_img, data_range=1.0))
    if enabled.get("psnr", True):
        mse = float(np.mean((gt_img - gen_img) ** 2))
        out[f"{prefix}_psnr"] = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    diff = gt_img - gen_img
    out[f"{prefix}_mae"] = float(np.mean(np.abs(diff)))
    out[f"{prefix}_rmse"] = float(np.sqrt(np.mean(diff * diff)))
    return out


def brisk_metrics(gt_img: np.ndarray, gen_img: np.ndarray, prefix: str, lowe_ratio: float) -> Dict[str, float]:
    try:
        import cv2
    except Exception:
        return {
            f"{prefix}_brisk_available": 0.0,
            f"{prefix}_brisk_kp_gt": float("nan"),
            f"{prefix}_brisk_kp_gen": float("nan"),
            f"{prefix}_brisk_matches": float("nan"),
            f"{prefix}_brisk_good_matches": float("nan"),
            f"{prefix}_brisk_good_match_ratio": float("nan"),
            f"{prefix}_brisk_mean_distance": float("nan"),
        }

    gt_u8 = np.clip(gt_img * 255.0, 0, 255).astype(np.uint8)
    gen_u8 = np.clip(gen_img * 255.0, 0, 255).astype(np.uint8)
    detector = cv2.BRISK_create()
    kp_gt, des_gt = detector.detectAndCompute(gt_u8, None)
    kp_gen, des_gen = detector.detectAndCompute(gen_u8, None)
    if des_gt is None or des_gen is None or len(kp_gt) == 0 or len(kp_gen) == 0:
        return {
            f"{prefix}_brisk_available": 1.0,
            f"{prefix}_brisk_kp_gt": float(len(kp_gt)),
            f"{prefix}_brisk_kp_gen": float(len(kp_gen)),
            f"{prefix}_brisk_matches": 0.0,
            f"{prefix}_brisk_good_matches": 0.0,
            f"{prefix}_brisk_good_match_ratio": 0.0,
            f"{prefix}_brisk_mean_distance": float("nan"),
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_gt, des_gen, k=2)
    good = []
    all_distances = []
    for pair in knn:
        if not pair:
            continue
        best = pair[0]
        all_distances.append(float(best.distance))
        if len(pair) == 1 or best.distance < float(lowe_ratio) * pair[1].distance:
            good.append(best)
    return {
        f"{prefix}_brisk_available": 1.0,
        f"{prefix}_brisk_kp_gt": float(len(kp_gt)),
        f"{prefix}_brisk_kp_gen": float(len(kp_gen)),
        f"{prefix}_brisk_matches": float(len(knn)),
        f"{prefix}_brisk_good_matches": float(len(good)),
        f"{prefix}_brisk_good_match_ratio": float(len(good) / max(1, min(len(kp_gt), len(kp_gen)))),
        f"{prefix}_brisk_mean_distance": float(np.mean(all_distances)) if all_distances else float("nan"),
    }


def save_debug_render(
    debug_dir: Path,
    pair: PairRecord,
    view_images: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)

    mpl_cache = debug_dir / ".mplconfig"
    xdg_cache = debug_dir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache.as_posix())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = list(view_images)
    fig, axes = plt.subplots(len(views), 4, figsize=(10, 2.4 * len(views)), squeeze=False)
    for row, view in enumerate(views):
        imgs = view_images[view]
        panels = [
            ("GT occ", imgs["gt_occupancy"]),
            ("Gen occ", imgs["gen_occupancy"]),
            ("GT depth", imgs["gt_depth"]),
            ("Gen depth", imgs["gen_depth"]),
        ]
        for col, (title, img) in enumerate(panels):
            axes[row, col].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_title(f"{view} {title}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle(pair.pair_id, fontsize=10)
    fig.tight_layout()
    out_path = debug_dir / f"{safe_filename(pair.pair_id)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_pair_metrics(
    pair: PairRecord,
    gt_points: np.ndarray,
    gen_points: np.ndarray,
    cfg: Mapping[str, Any],
    out_dir: Path,
    pair_index: int,
) -> Dict[str, float]:
    render_cfg = cfg["render"]
    img_cfg = cfg["image_metrics"]
    resolution = int(render_cfg.get("resolution", 256))
    views = list(render_cfg.get("views", ["xy", "xz", "yz"]))
    bounds_min, bounds_max = pair_bounds(
        gt_points,
        gen_points,
        str(render_cfg.get("bbox_mode", "union")),
        float(render_cfg.get("padding_fraction", 0.06)),
    )

    metrics: Dict[str, float] = {}
    view_images: Dict[str, Dict[str, np.ndarray]] = {}
    for view in views:
        gt_render = render_points(gt_points, bounds_min, bounds_max, view, resolution, str(render_cfg.get("depth_mode", "front")))
        gen_render = render_points(gen_points, bounds_min, bounds_max, view, resolution, str(render_cfg.get("depth_mode", "front")))
        view_images[view] = {
            "gt_occupancy": gt_render["occupancy"],
            "gen_occupancy": gen_render["occupancy"],
            "gt_depth": gt_render["depth"],
            "gen_depth": gen_render["depth"],
        }

        for image_type in ("occupancy", "depth"):
            prefix = f"{view}_{image_type}"
            metrics.update(image_metrics(gt_render[image_type], gen_render[image_type], prefix, img_cfg))

        if bool(img_cfg.get("brisk", True)):
            brisk_image = str(img_cfg.get("brisk_image", "occupancy"))
            prefix = f"{view}_{brisk_image}"
            metrics.update(
                brisk_metrics(
                    gt_render[brisk_image],
                    gen_render[brisk_image],
                    prefix,
                    float(img_cfg.get("brisk_lowe_ratio", 0.75)),
                )
            )

    max_debug = int(render_cfg.get("save_debug_images", 0))
    if pair_index < max_debug:
        save_debug_render(out_dir / str(render_cfg.get("debug_dir", "debug_renders")), pair, view_images)
    return metrics


def aggregate_numeric(rows: Sequence[Mapping[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)

    summary: List[Dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        out: Dict[str, Any] = {"label": label, "n_pairs": len(group)}
        keys = sorted({k for row in group for k in row if k not in {"label", "case_id", "gt_path", "generated_path", "error"}})
        for key in keys:
            vals = []
            for row in group:
                try:
                    val = float(row.get(key, np.nan))
                except Exception:
                    val = np.nan
                if np.isfinite(val):
                    vals.append(val)
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[f"{key}_mean"] = float(arr.mean())
                out[f"{key}_median"] = float(np.median(arr))
                out[f"{key}_std"] = float(arr.std())
        summary.append(out)
    return summary


def evaluate_pair(pair: PairRecord, cfg: Mapping[str, Any], out_dir: Path, pair_index: int) -> Dict[str, Any]:
    surface_cfg = cfg["surface"]
    render_cfg = cfg["render"]
    seed_base = int(surface_cfg.get("seed", 123))
    gt_mesh = load_mesh(pair.gt_path)
    gen_mesh = load_mesh(pair.generated_path)

    row: Dict[str, Any] = {
        "label": pair.label,
        "case_id": pair.case_id,
        "gt_path": pair.gt_path.as_posix(),
        "generated_path": pair.generated_path.as_posix(),
    }
    row.update(mesh_scalar_metrics(gt_mesh, gen_mesh))

    n_surface = int(surface_cfg.get("points_per_mesh", 20000))
    # Same seed for both surfaces makes identical-mesh smoke tests deterministic.
    surface_seed = stable_seed(seed_base, pair.pair_id)
    gt_surface = sample_mesh_points(gt_mesh, n_surface, surface_seed)
    gen_surface = sample_mesh_points(gen_mesh, n_surface, surface_seed)
    cd = chamfer_and_hausdorff(gt_surface, gen_surface)
    d_a_to_b = cd.pop("_d_a_to_b")
    d_b_to_a = cd.pop("_d_b_to_a")
    row.update(cd)
    row.update(fscore_from_distances(d_a_to_b, d_b_to_a, surface_cfg.get("fscore_thresholds", [])))

    if bool(render_cfg.get("enabled", True)):
        n_render = int(render_cfg.get("points_per_mesh", 60000))
        render_seed = stable_seed(seed_base, pair.pair_id + ":render")
        gt_render_points = sample_mesh_points(gt_mesh, n_render, render_seed)
        gen_render_points = sample_mesh_points(gen_mesh, n_render, render_seed)
        row.update(render_pair_metrics(pair, gt_render_points, gen_render_points, cfg, out_dir, pair_index))
    return row


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "sample"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_manifest(path: Path, pairs: Sequence[PairRecord]) -> None:
    write_csv(
        path,
        [
            {
                "label": p.label,
                "case_id": p.case_id,
                "gt_path": p.gt_path.as_posix(),
                "generated_path": p.generated_path.as_posix(),
            }
            for p in pairs
        ],
    )


def write_summary_md(
    out_dir: Path,
    cfg: Mapping[str, Any],
    pairs: Sequence[PairRecord],
    rows: Sequence[Mapping[str, Any]],
    warnings_out: Sequence[str],
    errors: Sequence[Mapping[str, Any]],
) -> None:
    counts: Dict[str, int] = {}
    for pair in pairs:
        counts[pair.label] = counts.get(pair.label, 0) + 1

    lines = [
        "# Healthy Vessel Mesh Metrics Run Summary",
        "",
        f"- Experiment: `{cfg['experiment'].get('name')}`",
        f"- Output dir: `{out_dir}`",
        f"- Pairs in manifest: `{len(pairs)}`",
        f"- Successful metric rows: `{len(rows)}`",
        f"- Errors: `{len(errors)}`",
        "",
        "## Metrics",
        "",
        "- Native mesh/surface: Chamfer, one-sided Chamfer, Hausdorff, p95 Hausdorff, F-score, area/volume ratios.",
        "- Rendered image: multi-view occupancy/depth SSIM, PSNR, MAE, RMSE.",
        "- BRISK: optional feature matching on rendered images; requires `cv2`/OpenCV.",
        "",
        "## Pair Counts",
        "",
    ]
    if counts:
        lines.extend(f"- `{label}`: {count}" for label, count in sorted(counts.items()))
    else:
        lines.append("- No generated/GT mesh pairs found yet.")

    if warnings_out:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {w}" for w in warnings_out[:120])
        if len(warnings_out) > 120:
            lines.append(f"- ... {len(warnings_out) - 120} more")

    if errors:
        lines += ["", "## First Errors", ""]
        for row in errors[:30]:
            lines.append(f"- `{row.get('label')}:{row.get('case_id')}`: {row.get('error')}")

    (out_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.gt_root:
        cfg["data"]["gt_mesh_roots"] = list(args.gt_root)
    generated = parse_label_path(args.generated)
    if generated:
        cfg["data"]["generated_mesh_roots"] = generated
    if args.max_cases is not None:
        cfg["data"]["max_cases"] = int(args.max_cases)
    if args.no_render:
        cfg["render"]["enabled"] = False
    if args.debug_images is not None:
        cfg["render"]["save_debug_images"] = int(args.debug_images)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("HealthyVesselMeshMetrics/config.yaml"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gt_root", action="append", default=None, help="GT mesh root. Can be passed multiple times.")
    parser.add_argument("--generated", action="append", default=None, metavar="LABEL=PATH")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_render", action="store_true", help="Only compute native surface metrics.")
    parser.add_argument("--debug_images", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = apply_cli_overrides(load_config(args.config), args)
    out_dir = resolve_path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings_out: List[str] = []
    data_cfg = cfg["data"]
    extensions = list(data_cfg.get("file_extensions", [".obj", ".ply", ".stl", ".off"]))

    gt_records: List[MeshRecord] = []
    for root in data_cfg.get("gt_mesh_roots", []):
        gt_records.extend(
            discover_meshes(
                resolve_path(root),
                label="GT",
                kind="gt",
                extensions=extensions,
                include_regex=data_cfg.get("include_regex"),
                exclude_regex=data_cfg.get("exclude_regex_gt"),
                warnings_out=warnings_out,
            )
        )

    generated_records: Dict[str, List[MeshRecord]] = {}
    for label, root in dict(data_cfg.get("generated_mesh_roots", {})).items():
        generated_records[str(label)] = discover_meshes(
            resolve_path(root),
            label=str(label),
            kind="generated",
            extensions=extensions,
            include_regex=data_cfg.get("include_regex"),
            exclude_regex=data_cfg.get("exclude_regex_generated"),
            warnings_out=warnings_out,
        )

    pairs, missing_rows = build_pairs(gt_records, generated_records, data_cfg.get("max_cases"), warnings_out)
    write_pair_manifest(out_dir / "mesh_pair_manifest.csv", pairs)
    write_csv(out_dir / "missing_cases.csv", missing_rows)

    if args.dry_run or not pairs:
        write_summary_md(out_dir, cfg, pairs, [], warnings_out, [])
        print(f"Discovery complete. Pairs: {len(pairs)}. Summary: {out_dir / 'run_summary.md'}")
        return 0

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        try:
            rows.append(evaluate_pair(pair, cfg, out_dir, i))
        except Exception as exc:
            errors.append(
                {
                    "label": pair.label,
                    "case_id": pair.case_id,
                    "gt_path": pair.gt_path.as_posix(),
                    "generated_path": pair.generated_path.as_posix(),
                    "error": str(exc),
                }
            )
            print(f"[WARN] {pair.pair_id}: {exc}", file=sys.stderr)

    write_csv(out_dir / "per_pair_metrics.csv", rows)
    write_csv(out_dir / "metric_errors.csv", errors)
    write_csv(out_dir / "summary_by_model.csv", aggregate_numeric(rows, "label"))
    write_summary_md(out_dir, cfg, pairs, rows, warnings_out, errors)
    print(f"Done. Metrics: {out_dir / 'per_pair_metrics.csv'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
