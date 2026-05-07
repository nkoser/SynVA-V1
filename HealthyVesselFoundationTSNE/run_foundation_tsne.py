#!/usr/bin/env python3
"""Prepare point-cloud foundation embeddings and t-SNE plots for Healthy vessels.

The script is intentionally model-agnostic. It can run a lightweight geometric
baseline for smoke tests, consume precomputed embeddings, or call a project-local
adapter for Uni3D, OpenShape, Michelangelo, or another point-cloud foundation
model.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - yaml is expected in this project env
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "healthy_vessel_foundation_tsne",
        "output_dir": "HealthyVesselFoundationTSNE/output/healthy_vessel_foundation_tsne",
    },
    "data": {
        "gt_mesh_roots": [
            "/data/healthy_vessel_decapped",
            "/data/healthy_vessel",
        ],
        "generated_mesh_roots": {
            # Fill these as soon as the reconstructed generated meshes exist.
            # "physio_v5": "/data/Stage2_FlowMatching_Physio/generated/physio_v5/meshes_sweep",
        },
        "file_extensions": [".obj", ".ply", ".stl", ".off"],
        "include_regex": None,
        "include_regex_gt": None,
        "include_regex_generated": None,
        "include_regex_generated_by_label": {},
        "gt_label_by_regex": {},
        "exclude_regex_gt": None,
        "exclude_regex_generated": r"(_gt|_gt_local|_local)$",
        "split_csv": None,
        "split_csv_delimiter": ";",
        "split_csv_uid_col": "uid",
        "split_csv_split_col": "split",
        "split_keep": None,
        "max_gt_records": None,
        "max_generated_records": None,
        "paired_only": True,
        "paired_require_all_generated": False,
        "max_cases": None,
    },
    "pointcloud": {
        "points_per_mesh": 4096,
        "sample_method": "surface",  # surface or vertices
        "normalize": "unit_sphere",  # unit_sphere, bbox, center, none
        "seed": 42,
        "cache_dir": "pointclouds",
        "overwrite_cache": False,
    },
    "embedding": {
        "backend": "geometric_baseline",
        "batch_size": 16,
        "device": "cuda",
        "cache_file": "embeddings.npz",
        "overwrite_cache": False,
        "l2_normalize": True,
        "external_npz": None,
        "custom_callable": None,
        "custom_repo_path": None,
        "openshape": {
            "repo_path": None,
            "callable": "HealthyVesselFoundationTSNE.backbones.openshape_adapter:embed_point_clouds",
            "checkpoint": None,
            "model_name": "openshape-pointbert-vitg14-rgb",
            "num_points": 10000,
            "rgb": [0.4, 0.4, 0.4],
        },
        "uni3d": {
            "repo_path": None,
            "callable": "HealthyVesselFoundationTSNE.backbones.uni3d_adapter:embed_point_clouds",
            "checkpoint": None,
            "model_name": "create_uni3d",
            "num_points": 10000,
            "scale": "giant",
            "rgb": [0.4, 0.4, 0.4],
            "args": {},
        },
        "michelangelo": {
            "repo_path": None,
            "callable": "HealthyVesselFoundationTSNE.backbones.michelangelo_adapter:embed_point_clouds",
            "checkpoint": None,
            "model_name": None,
            "num_points": 8192,
            "input_channels": 6,
            "normal_fill": [0.0, 0.0, 1.0],
            "factory_callable": None,
            "config_path": None,
            "model_module": None,
            "model_class": None,
            "model_kwargs": {},
            "encode_methods": ["encode_shape_embed", "encode", "encode_pc", "forward"],
        },
    },
    "tsne": {
        "perplexity": 30.0,
        "init": "pca",
        "learning_rate": "auto",
        "metric": "euclidean",
        "random_state": 42,
        "max_iter": 1000,
        "standardize": False,
    },
    "plot": {
        "filename": "healthy_vessel_foundation_tsne.png",
        "show_title": True,
        "figsize": [9, 7],
        "dpi": 220,
        "point_size": 34,
        "draw_pair_lines": True,
        "show_axis_labels": True,
        "show_legend": True,
        "marker_by_kind": {"gt": "o", "generated": "^"},
        "color_by_kind": {},
        "color_by_label": {},
        "label_alias": {},
    },
}


KNOWN_CASE_SUFFIXES = (
    "_vessel_submesh_closed",
    "_gt_local",
    "_gt",
    "_local",
    "_generated",
    "_gen",
    "_generated_aneurysm_world",
    "_vessel_with_generated_aneurysm_stitched",
    "_vessel_with_generated_aneurysm_unstitched",
)


@dataclasses.dataclass(frozen=True)
class MeshRecord:
    label: str
    case_id: str
    path: Path
    kind: str


@dataclasses.dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    label: str
    case_id: str
    kind: str
    mesh_path: Path
    pointcloud_path: Optional[Path] = None


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


def normalize_case_id(name: str) -> str:
    out = strip_known_suffixes(name)
    if out.startswith("aneux_"):
        out = out[len("aneux_") :]
    return strip_known_suffixes(out)


def case_id_from_path(path: Path) -> str:
    stem = strip_known_suffixes(path.stem)
    if stem.lower() in {"mesh", "model", "surface", "vessel", "vessel_submesh", "aneurysm_submesh"}:
        parent = path.parent.name
        if parent in {"01_mesh", "05_submeshes"} and path.parent.parent != path.parent:
            stem = path.parent.parent.name
        else:
            stem = parent
    return normalize_case_id(stem)


def mesh_sort_key(path: Path, ext_priority: Mapping[str, int]) -> Tuple[int, int, str]:
    return (
        ext_priority.get(path.suffix.lower(), 10_000),
        len(path.parts),
        path.as_posix(),
    )


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
        raw_text = f"{path.stem} {rel}"
        case_id = case_id_from_path(path)
        match_text = f"{raw_text} {case_id}"
        if include and not include.search(match_text):
            continue
        if exclude and exclude.search(match_text):
            continue
        records.append(MeshRecord(label=label, case_id=case_id, path=path, kind=kind))
    return records


def load_split_case_ids(
    path: Path,
    delimiter: str = ";",
    uid_col: str = "uid",
    split_col: str = "split",
    keep_splits: Optional[Sequence[str]] = None,
) -> set[str]:
    keep = {str(s).strip().lower() for s in (keep_splits or []) if str(s).strip()}
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        if uid_col not in fields or split_col not in fields:
            raise ValueError(f"CSV split file must contain columns {uid_col!r} and {split_col!r}: {path}")
        for row in reader:
            uid = str(row.get(uid_col, "")).strip()
            split = str(row.get(split_col, "")).strip().lower()
            if not uid:
                continue
            if keep and split not in keep:
                continue
            out.add(normalize_case_id(uid))
    return out


def filter_records_by_case_ids(records: Sequence[MeshRecord], allowed: set[str]) -> List[MeshRecord]:
    return [record for record in records if record.case_id in allowed]


def relabel_records_by_regex(
    records: Sequence[MeshRecord],
    rules: Mapping[str, str],
) -> List[MeshRecord]:
    if not rules:
        return list(records)
    compiled = [(re.compile(pattern), str(label)) for pattern, label in rules.items()]
    out: List[MeshRecord] = []
    for record in records:
        text = f"{record.path.as_posix()} {record.case_id} {record.label}"
        label = record.label
        for pattern, replacement in compiled:
            if pattern.search(text):
                label = replacement
                break
        out.append(dataclasses.replace(record, label=label))
    return out


def limit_records(
    records: Sequence[MeshRecord],
    max_records: Optional[int],
    label: str,
    warnings_out: List[str],
) -> List[MeshRecord]:
    if max_records is None:
        return list(records)
    limit = int(max_records)
    if limit < 0:
        raise ValueError(f"max record limit must be >= 0 for {label}: {limit}")
    limited = list(records)[:limit]
    warnings_out.append(f"[limit] kept {len(limited)}/{len(records)} meshes for {label}")
    return limited


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


def build_sample_plan(
    gt_records: Sequence[MeshRecord],
    generated_records: Mapping[str, Sequence[MeshRecord]],
    paired_only: bool,
    paired_require_all_generated: bool,
    max_cases: Optional[int],
    warnings_out: List[str],
) -> Tuple[List[SampleRecord], List[Dict[str, str]]]:
    gt_by_case = index_by_case(gt_records, warnings_out)
    gen_by_label = {
        label: index_by_case(records, warnings_out)
        for label, records in generated_records.items()
    }

    missing: List[Dict[str, str]] = []
    samples: List[SampleRecord] = []
    added_gt: set[str] = set()

    if paired_only and gen_by_label:
        if paired_require_all_generated:
            cases = set(gt_by_case)
            for gen_index in gen_by_label.values():
                cases.intersection_update(gen_index)
        else:
            cases: set[str] = set()
            for gen_index in gen_by_label.values():
                cases.update(set(gt_by_case).intersection(gen_index))
    elif paired_only:
        cases = set()
        warnings_out.append(
            "[info] No generated meshes discovered yet; paired manifest is empty. "
            "Use --all_available for an explicit GT-only smoke test."
        )
    else:
        cases = set(gt_by_case)
        for gen_index in gen_by_label.values():
            cases.update(gen_index)

    ordered_cases = sorted(cases)
    if max_cases is not None:
        ordered_cases = ordered_cases[: int(max_cases)]

    for case_id in ordered_cases:
        gt_record = gt_by_case.get(case_id)
        if gt_record is not None:
            samples.append(
                SampleRecord(
                    sample_id=f"GT:{case_id}",
                    label=gt_record.label,
                    case_id=case_id,
                    kind="gt",
                    mesh_path=gt_record.path,
                )
            )
            added_gt.add(case_id)
        elif paired_only:
            missing.append({"label": "GT", "case_id": case_id, "reason": "missing_gt"})

        for label, gen_index in gen_by_label.items():
            gen_record = gen_index.get(case_id)
            if gen_record is not None:
                samples.append(
                    SampleRecord(
                        sample_id=f"{label}:{case_id}",
                        label=label,
                        case_id=case_id,
                        kind="generated",
                        mesh_path=gen_record.path,
                    )
                )
            elif paired_only and gt_record is not None:
                missing.append({"label": label, "case_id": case_id, "reason": "missing_generated"})

    return samples, missing


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_mesh_points(mesh_path: Path, n_points: int, sample_method: str, seed: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if hasattr(g, "vertices")]
        if not parts:
            raise ValueError(f"Scene contains no mesh geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(parts)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    rng = np.random.default_rng(seed)
    use_surface = sample_method == "surface" and getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0
    if use_surface:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            pts = np.asarray(mesh.sample(int(n_points)), dtype=np.float32)
        finally:
            np.random.set_state(state)
    else:
        replace = len(vertices) < int(n_points)
        idx = rng.choice(len(vertices), size=int(n_points), replace=replace)
        pts = vertices[idx]
    return pts.astype(np.float32, copy=False)


def normalize_points(points: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    mode = str(mode or "none").lower()
    if mode == "none":
        return pts, {"mode": "none", "center": [0.0, 0.0, 0.0], "scale": 1.0}

    if mode == "center":
        center = pts.mean(axis=0)
        scale = 1.0
    elif mode == "bbox":
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) * 0.5
        scale = float(np.max(hi - lo))
    elif mode == "unit_sphere":
        center = pts.mean(axis=0)
        scale = float(np.max(np.linalg.norm(pts - center, axis=1)))
    else:
        raise ValueError(f"Unknown pointcloud.normalize mode: {mode}")

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    out = (pts - center) / scale
    return out.astype(np.float32), {
        "mode": mode,
        "center": center.astype(float).tolist(),
        "scale": float(scale),
    }


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "sample"


def materialize_pointclouds(
    samples: Sequence[SampleRecord],
    cfg: Mapping[str, Any],
    out_dir: Path,
) -> Tuple[List[SampleRecord], List[Dict[str, str]]]:
    pc_cfg = cfg["pointcloud"]
    pc_dir = out_dir / str(pc_cfg.get("cache_dir", "pointclouds"))
    pc_dir.mkdir(parents=True, exist_ok=True)
    transform_rows: List[Dict[str, str]] = []
    valid: List[SampleRecord] = []
    errors: List[Dict[str, str]] = []

    n_points = int(pc_cfg.get("points_per_mesh", 4096))
    sample_method = str(pc_cfg.get("sample_method", "surface")).lower()
    normalize_mode = str(pc_cfg.get("normalize", "unit_sphere"))
    seed = int(pc_cfg.get("seed", 42))
    overwrite = bool(pc_cfg.get("overwrite_cache", False))

    for sample in samples:
        pc_path = pc_dir / f"{safe_filename(sample.sample_id)}.npy"
        transform_path = pc_dir / f"{safe_filename(sample.sample_id)}.json"
        try:
            if pc_path.exists() and not overwrite:
                pts = np.load(pc_path)
                transform = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.exists() else {}
            else:
                sample_seed = stable_seed(seed, sample.sample_id)
                pts = load_mesh_points(sample.mesh_path, n_points, sample_method, sample_seed)
                pts, transform = normalize_points(pts, normalize_mode)
                np.save(pc_path, pts.astype(np.float32))
                transform_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")
            valid.append(dataclasses.replace(sample, pointcloud_path=pc_path))
            transform_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "case_id": sample.case_id,
                    "kind": sample.kind,
                    "mesh_path": sample.mesh_path.as_posix(),
                    "pointcloud_path": pc_path.as_posix(),
                    "normalize_mode": str(transform.get("mode", "")),
                    "scale": str(transform.get("scale", "")),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "case_id": sample.case_id,
                    "reason": f"pointcloud_failed: {exc}",
                }
            )

    write_csv(out_dir / "pointcloud_manifest.csv", transform_rows)
    return valid, errors


def geometric_baseline_embeddings(points_batch: np.ndarray) -> np.ndarray:
    """Small deterministic descriptor for smoke tests, not a foundation model."""
    features: List[np.ndarray] = []
    quantiles = np.linspace(0.0, 1.0, 11)
    hist_bins = np.linspace(-1.25, 1.25, 17)
    radial_bins = np.linspace(0.0, 1.5, 17)

    for pts in points_batch:
        pts = np.asarray(pts, dtype=np.float32)
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = hi - lo
        centered = pts - mean
        cov = np.cov(centered.T)
        eig = np.linalg.eigvalsh(cov).astype(np.float32)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        eig_norm = eig / max(float(eig.sum()), 1e-8)
        radii = np.linalg.norm(centered, axis=1)
        r_quant = np.quantile(radii, quantiles).astype(np.float32)
        r_hist = np.histogram(radii, bins=radial_bins, density=True)[0].astype(np.float32)
        x_hist = np.histogram(pts[:, 0], bins=hist_bins, density=True)[0].astype(np.float32)
        y_hist = np.histogram(pts[:, 1], bins=hist_bins, density=True)[0].astype(np.float32)
        z_hist = np.histogram(pts[:, 2], bins=hist_bins, density=True)[0].astype(np.float32)
        feat = np.concatenate(
            [
                mean,
                std,
                lo,
                hi,
                extent,
                eig,
                eig_norm,
                r_quant,
                r_hist,
                x_hist,
                y_hist,
                z_hist,
            ]
        )
        features.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(features, axis=0).astype(np.float32)


def add_repo_path(repo_path: Optional[str]) -> None:
    if not repo_path:
        return
    p = resolve_path(repo_path)
    if not p.exists():
        raise FileNotFoundError(f"Adapter repo_path does not exist: {p}")
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())


def resolve_callable(spec: str) -> Callable[..., Any]:
    root = repo_root()
    if root.as_posix() not in sys.path:
        sys.path.insert(0, root.as_posix())

    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    else:
        module_name, attr = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Resolved object is not callable: {spec}")
    return obj


def call_embedding_callable(
    fn: Callable[..., Any],
    points_batch: np.ndarray,
    backend_cfg: Mapping[str, Any],
    device: str,
) -> np.ndarray:
    try:
        out = fn(points_batch, backend_cfg, device)
    except TypeError:
        try:
            out = fn(points_batch, backend_cfg)
        except TypeError:
            out = fn(points_batch)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


def load_external_npz_embeddings(
    npz_path: Path,
    samples: Sequence[SampleRecord],
) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    emb_key = next((k for k in ("embeddings", "features", "embeds", "x") if k in data), None)
    id_key = next((k for k in ("sample_ids", "ids", "keys", "names") if k in data), None)
    if emb_key is None or id_key is None:
        raise KeyError(
            f"{npz_path} needs embedding key one of embeddings/features/embeds/x "
            "and id key one of sample_ids/ids/keys/names"
        )
    embeddings = np.asarray(data[emb_key], dtype=np.float32)
    ids = [str(x) for x in data[id_key].tolist()]
    index = {sample_id: embeddings[i] for i, sample_id in enumerate(ids)}

    out: List[np.ndarray] = []
    missing: List[str] = []
    for sample in samples:
        candidates = (
            sample.sample_id,
            f"{sample.label}:{sample.case_id}",
            f"{sample.kind}:{sample.case_id}",
            sample.case_id,
        )
        vec = next((index[k] for k in candidates if k in index), None)
        if vec is None:
            missing.append(sample.sample_id)
        else:
            out.append(vec)
    if missing:
        raise KeyError(f"External embeddings missing {len(missing)} samples, first: {missing[:5]}")
    return np.stack(out, axis=0).astype(np.float32)


def compute_embeddings(
    samples: Sequence[SampleRecord],
    cfg: Mapping[str, Any],
    out_dir: Path,
) -> np.ndarray:
    emb_cfg = cfg["embedding"]
    cache_file = out_dir / str(emb_cfg.get("cache_file", "embeddings.npz"))
    backend = str(emb_cfg.get("backend", "geometric_baseline")).lower()
    batch_size = int(emb_cfg.get("batch_size", 16))
    device = str(emb_cfg.get("device", "cuda"))
    embedding_signature = json.dumps(emb_cfg, sort_keys=True, default=str)

    if cache_file.exists() and not bool(emb_cfg.get("overwrite_cache", False)):
        cached = np.load(cache_file, allow_pickle=True)
        cached_ids = [str(x) for x in cached["sample_ids"].tolist()]
        expected_ids = [s.sample_id for s in samples]
        cached_backend = str(cached["backend"].item()) if "backend" in cached else ""
        cached_signature = str(cached["embedding_config_json"].item()) if "embedding_config_json" in cached else ""
        if cached_ids == expected_ids and cached_backend == backend and cached_signature == embedding_signature:
            return np.asarray(cached["embeddings"], dtype=np.float32)

    if backend == "external_npz":
        external = emb_cfg.get("external_npz")
        if not external:
            raise ValueError("embedding.backend=external_npz requires embedding.external_npz")
        embeddings = load_external_npz_embeddings(resolve_path(external), samples)
    else:
        pointclouds = [np.load(s.pointcloud_path).astype(np.float32) for s in samples if s.pointcloud_path]
        if len(pointclouds) != len(samples):
            raise ValueError("All samples need pointcloud_path before embedding.")

        embeddings_list: List[np.ndarray] = []
        if backend == "geometric_baseline":
            for start in range(0, len(pointclouds), batch_size):
                batch = np.stack(pointclouds[start : start + batch_size], axis=0)
                embeddings_list.append(geometric_baseline_embeddings(batch))
        else:
            backend_cfg = dict(emb_cfg.get(backend, {}))
            callable_spec = backend_cfg.get("callable") or emb_cfg.get("custom_callable")
            repo_path = backend_cfg.get("repo_path") or emb_cfg.get("custom_repo_path")
            if not callable_spec:
                raise ValueError(
                    f"embedding.backend={backend!r} needs either "
                    f"embedding.{backend}.callable or embedding.custom_callable. "
                    "Use foundation_adapter_template.py as the expected interface."
                )
            add_repo_path(repo_path)
            fn = resolve_callable(str(callable_spec))
            for start in range(0, len(pointclouds), batch_size):
                batch = np.stack(pointclouds[start : start + batch_size], axis=0)
                embeddings_list.append(call_embedding_callable(fn, batch, backend_cfg, device))
        embeddings = np.concatenate(embeddings_list, axis=0)

    if bool(emb_cfg.get("l2_normalize", True)):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norm, 1e-12)

    np.savez_compressed(
        cache_file,
        embeddings=embeddings.astype(np.float32),
        sample_ids=np.array([s.sample_id for s in samples], dtype=object),
        labels=np.array([s.label for s in samples], dtype=object),
        case_ids=np.array([s.case_id for s in samples], dtype=object),
        kinds=np.array([s.kind for s in samples], dtype=object),
        mesh_paths=np.array([s.mesh_path.as_posix() for s in samples], dtype=object),
        backend=np.array(backend, dtype=object),
        embedding_config_json=np.array(embedding_signature, dtype=object),
    )
    return embeddings.astype(np.float32)


def run_tsne(embeddings: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    if len(embeddings) < 3:
        raise ValueError("t-SNE needs at least 3 samples.")

    x = np.asarray(embeddings, dtype=np.float32)
    tsne_cfg = cfg["tsne"]
    if bool(tsne_cfg.get("standardize", False)):
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(x)

    from sklearn.manifold import TSNE

    requested = float(tsne_cfg.get("perplexity", 30.0))
    max_perplexity = max(1.0, min(requested, float(len(x) - 1), float(max(1, (len(x) - 1) // 3))))
    perplexity = max_perplexity
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=str(tsne_cfg.get("init", "pca")),
        learning_rate=tsne_cfg.get("learning_rate", "auto"),
        metric=str(tsne_cfg.get("metric", "euclidean")),
        random_state=int(tsne_cfg.get("random_state", 42)),
        max_iter=int(tsne_cfg.get("max_iter", 1000)),
    )
    return tsne.fit_transform(x).astype(np.float32)


def plot_tsne(coords: np.ndarray, samples: Sequence[SampleRecord], cfg: Mapping[str, Any], out_dir: Path) -> Path:
    mpl_cache = out_dir / ".mplconfig"
    xdg_cache = out_dir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache.as_posix())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cfg = cfg["plot"]
    out_path = out_dir / str(plot_cfg.get("filename", "healthy_vessel_foundation_tsne.png"))
    labels = list(dict.fromkeys(s.label for s in samples))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(labels), 1)))
    color_by_label = {label: colors[i] for i, label in enumerate(labels)}
    color_by_label.update(dict(plot_cfg.get("color_by_label", {}) or {}))
    color_by_kind = dict(plot_cfg.get("color_by_kind", {}) or {})
    label_alias = dict(plot_cfg.get("label_alias", {}) or {})
    marker_by_kind = {"gt": "o", "generated": "^"}
    marker_by_kind.update(dict(plot_cfg.get("marker_by_kind", {}) or {}))
    draw_pair_lines = bool(plot_cfg.get("draw_pair_lines", True))

    figsize = plot_cfg.get("figsize", [9, 7])
    fig, ax = plt.subplots(figsize=(float(figsize[0]), float(figsize[1])))
    if draw_pair_lines:
        gt_coord_by_case = {
            sample.case_id: coords[i]
            for i, sample in enumerate(samples)
            if sample.kind == "gt"
        }
        for i, sample in enumerate(samples):
            if sample.kind != "generated" or sample.case_id not in gt_coord_by_case:
                continue
            p0 = gt_coord_by_case[sample.case_id]
            p1 = coords[i]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.78", linewidth=0.7, alpha=0.45, zorder=0)

    for label in labels:
        idx = [i for i, s in enumerate(samples) if s.label == label]
        if not idx:
            continue
        kind = samples[idx[0]].kind
        color = color_by_label.get(label)
        if kind in color_by_kind:
            color = color_by_kind[kind]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=float(plot_cfg.get("point_size", 34)),
            marker=marker_by_kind.get(kind, "o"),
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
            label=str(label_alias.get(label, label)),
        )

    default_title = (
        "Healthy vessel point-cloud foundation embedding distributions"
        if not draw_pair_lines
        else "Healthy vessel GT vs generated point-cloud foundation embeddings"
    )
    if bool(plot_cfg.get("show_title", True)):
        ax.set_title(str(plot_cfg.get("title") or default_title))
    if bool(plot_cfg.get("show_axis_labels", True)):
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.grid(bool(plot_cfg.get("show_grid", True)), linewidth=0.4, alpha=0.25)
    if bool(plot_cfg.get("show_legend", True)):
        ax.legend(
            loc=str(plot_cfg.get("legend_loc", "best")),
            fontsize=float(plot_cfg.get("legend_fontsize", 9)),
            frameon=True,
            ncol=int(plot_cfg.get("legend_columns", 1)),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(plot_cfg.get("dpi", 220)))
    plt.close(fig)
    return out_path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, samples: Sequence[SampleRecord]) -> None:
    write_csv(
        path,
        [
            {
                "sample_id": s.sample_id,
                "label": s.label,
                "case_id": s.case_id,
                "kind": s.kind,
                "mesh_path": s.mesh_path.as_posix(),
                "pointcloud_path": s.pointcloud_path.as_posix() if s.pointcloud_path else "",
            }
            for s in samples
        ],
    )


def write_tsne_csv(path: Path, coords: np.ndarray, samples: Sequence[SampleRecord]) -> None:
    write_csv(
        path,
        [
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "case_id": sample.case_id,
                "kind": sample.kind,
                "tsne_x": float(coords[i, 0]),
                "tsne_y": float(coords[i, 1]),
                "mesh_path": sample.mesh_path.as_posix(),
            }
            for i, sample in enumerate(samples)
        ],
    )


def write_summary(
    out_dir: Path,
    cfg: Mapping[str, Any],
    samples: Sequence[SampleRecord],
    warnings_out: Sequence[str],
    missing_rows: Sequence[Mapping[str, str]],
    pointcloud_errors: Sequence[Mapping[str, str]],
    outputs: Mapping[str, Optional[Path]],
) -> None:
    labels: Dict[str, int] = {}
    for sample in samples:
        labels[sample.label] = labels.get(sample.label, 0) + 1

    lines = [
        "# Healthy Vessel Foundation t-SNE Run Summary",
        "",
        f"- Experiment: `{cfg['experiment'].get('name')}`",
        f"- Output dir: `{out_dir}`",
        f"- Embedding backend: `{cfg['embedding'].get('backend')}`",
        f"- Samples in manifest: `{len(samples)}`",
        f"- Missing rows: `{len(missing_rows)}`",
        f"- Point-cloud errors: `{len(pointcloud_errors)}`",
        "",
        "## Counts",
        "",
    ]
    if labels:
        for label, count in sorted(labels.items()):
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("- No samples discovered yet.")

    lines += ["", "## Outputs", ""]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{path}`" if path else f"- `{name}`: not written")

    if warnings_out:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {w}" for w in warnings_out[:100])
        if len(warnings_out) > 100:
            lines.append(f"- ... {len(warnings_out) - 100} more")

    if pointcloud_errors:
        lines += ["", "## First Point-Cloud Errors", ""]
        for row in pointcloud_errors[:20]:
            lines.append(f"- `{row.get('sample_id')}`: {row.get('reason')}")

    (out_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.gt_root:
        cfg["data"]["gt_mesh_roots"] = list(args.gt_root)
    generated = parse_label_path(args.generated)
    if generated:
        cfg["data"]["generated_mesh_roots"] = generated
    if args.backend:
        cfg["embedding"]["backend"] = args.backend
    if args.points is not None:
        cfg["pointcloud"]["points_per_mesh"] = int(args.points)
    if args.batch_size is not None:
        cfg["embedding"]["batch_size"] = int(args.batch_size)
    if args.max_cases is not None:
        cfg["data"]["max_cases"] = int(args.max_cases)
    if args.all_available:
        cfg["data"]["paired_only"] = False
    if args.no_pair_lines:
        cfg.setdefault("plot", {})["draw_pair_lines"] = False
    if args.plot_title:
        cfg.setdefault("plot", {})["title"] = args.plot_title
    if args.plot_filename:
        cfg.setdefault("plot", {})["filename"] = args.plot_filename
    if args.tsne_perplexity is not None:
        cfg.setdefault("tsne", {})["perplexity"] = float(args.tsne_perplexity)
    if args.tsne_init:
        cfg.setdefault("tsne", {})["init"] = args.tsne_init
    if args.tsne_metric:
        cfg.setdefault("tsne", {})["metric"] = args.tsne_metric
    if args.tsne_random_state is not None:
        cfg.setdefault("tsne", {})["random_state"] = int(args.tsne_random_state)
    if args.tsne_max_iter is not None:
        cfg.setdefault("tsne", {})["max_iter"] = int(args.tsne_max_iter)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("HealthyVesselFoundationTSNE/config.yaml"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gt_root", action="append", default=None, help="GT mesh root. Can be passed multiple times.")
    parser.add_argument(
        "--generated",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Generated mesh root. Can be passed multiple times.",
    )
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--all_available", action="store_true", help="Use all discovered meshes, not only paired cases.")
    parser.add_argument("--no_pair_lines", action="store_true", help="Do not draw GT-to-generated pair lines in the plot.")
    parser.add_argument("--plot_title", type=str, default=None, help="Override the plot title.")
    parser.add_argument("--plot_filename", type=str, default=None, help="Override the output plot filename.")
    parser.add_argument("--tsne_perplexity", type=float, default=None, help="Override t-SNE perplexity.")
    parser.add_argument("--tsne_init", type=str, default=None, choices=["pca", "random"], help="Override t-SNE init.")
    parser.add_argument("--tsne_metric", type=str, default=None, help="Override t-SNE metric, e.g. euclidean or cosine.")
    parser.add_argument("--tsne_random_state", type=int, default=None, help="Override t-SNE random seed.")
    parser.add_argument("--tsne_max_iter", type=int, default=None, help="Override t-SNE iterations.")
    parser.add_argument("--dry_run", action="store_true", help="Only discover meshes and write manifests.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = apply_cli_overrides(load_config(args.config), args)
    out_dir = resolve_path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings_out: List[str] = []
    data_cfg = cfg["data"]
    extensions = list(data_cfg.get("file_extensions", [".obj", ".ply", ".stl", ".off"]))
    include_regex = data_cfg.get("include_regex")
    include_regex_gt = data_cfg.get("include_regex_gt") or include_regex
    include_regex_generated = data_cfg.get("include_regex_generated") or include_regex

    gt_records: List[MeshRecord] = []
    for root in data_cfg.get("gt_mesh_roots", []):
        gt_records.extend(
            discover_meshes(
                resolve_path(root),
                label="GT",
                kind="gt",
                extensions=extensions,
                include_regex=include_regex_gt,
                exclude_regex=data_cfg.get("exclude_regex_gt"),
                warnings_out=warnings_out,
            )
        )
    gt_records = relabel_records_by_regex(
        gt_records,
        dict(data_cfg.get("gt_label_by_regex", {}) or {}),
    )

    generated_records: Dict[str, List[MeshRecord]] = {}
    include_regex_generated_by_label = dict(data_cfg.get("include_regex_generated_by_label", {}) or {})
    for label, root in dict(data_cfg.get("generated_mesh_roots", {})).items():
        label_include_regex = include_regex_generated_by_label.get(str(label), include_regex_generated)
        generated_records[label] = discover_meshes(
            resolve_path(root),
            label=str(label),
            kind="generated",
            extensions=extensions,
            include_regex=label_include_regex,
            exclude_regex=data_cfg.get("exclude_regex_generated"),
            warnings_out=warnings_out,
        )

    if data_cfg.get("split_csv"):
        keep_splits = data_cfg.get("split_keep") or ["test"]
        if isinstance(keep_splits, str):
            keep_splits = [keep_splits]
        split_ids = load_split_case_ids(
            resolve_path(data_cfg["split_csv"]),
            delimiter=str(data_cfg.get("split_csv_delimiter", ";")),
            uid_col=str(data_cfg.get("split_csv_uid_col", "uid")),
            split_col=str(data_cfg.get("split_csv_split_col", "split")),
            keep_splits=keep_splits,
        )
        before_gt = len(gt_records)
        gt_records = filter_records_by_case_ids(gt_records, split_ids)
        warnings_out.append(
            f"[split_csv] kept {len(gt_records)}/{before_gt} GT meshes for splits {list(keep_splits)}"
        )
        for label, records in list(generated_records.items()):
            before_gen = len(records)
            generated_records[label] = filter_records_by_case_ids(records, split_ids)
            warnings_out.append(
                f"[split_csv] kept {len(generated_records[label])}/{before_gen} generated meshes "
                f"for {label} and splits {list(keep_splits)}"
            )

    gt_records = limit_records(
        gt_records,
        data_cfg.get("max_gt_records"),
        "GT",
        warnings_out,
    )
    generated_limit = data_cfg.get("max_generated_records")
    for label, records in list(generated_records.items()):
        label_limit = generated_limit
        if isinstance(generated_limit, Mapping):
            label_limit = generated_limit.get(label)
        generated_records[label] = limit_records(
            records,
            label_limit,
            str(label),
            warnings_out,
        )

    samples, missing_rows = build_sample_plan(
        gt_records,
        generated_records,
        paired_only=bool(data_cfg.get("paired_only", True)),
        paired_require_all_generated=bool(data_cfg.get("paired_require_all_generated", False)),
        max_cases=data_cfg.get("max_cases"),
        warnings_out=warnings_out,
    )
    write_manifest(out_dir / "mesh_manifest.csv", samples)
    write_csv(out_dir / "missing_cases.csv", missing_rows)

    outputs: Dict[str, Optional[Path]] = {
        "mesh_manifest": out_dir / "mesh_manifest.csv",
        "missing_cases": out_dir / "missing_cases.csv",
        "pointcloud_manifest": None,
        "embeddings": None,
        "tsne_coordinates": None,
        "plot": None,
    }

    if args.dry_run or not samples:
        write_summary(out_dir, cfg, samples, warnings_out, missing_rows, [], outputs)
        print(f"Discovery complete. Samples: {len(samples)}. Summary: {out_dir / 'run_summary.md'}")
        return 0

    pointcloud_errors: List[Dict[str, str]] = []
    backend = str(cfg["embedding"].get("backend", "geometric_baseline")).lower()
    if backend == "external_npz":
        samples_for_embedding = samples
        warnings_out.append("[info] embedding.backend=external_npz; skipped point-cloud materialization.")
    else:
        samples_for_embedding, pointcloud_errors = materialize_pointclouds(samples, cfg, out_dir)
        outputs["pointcloud_manifest"] = out_dir / "pointcloud_manifest.csv"
        write_csv(out_dir / "pointcloud_errors.csv", pointcloud_errors)

    if len(samples_for_embedding) < 3:
        warnings_out.append("[info] Fewer than 3 samples are available; t-SNE was skipped.")
        write_summary(out_dir, cfg, samples_for_embedding, warnings_out, missing_rows, pointcloud_errors, outputs)
        print(f"Prepared {len(samples_for_embedding)} samples. Not enough for t-SNE.")
        return 0

    embeddings = compute_embeddings(samples_for_embedding, cfg, out_dir)
    outputs["embeddings"] = out_dir / str(cfg["embedding"].get("cache_file", "embeddings.npz"))
    coords = run_tsne(embeddings, cfg)
    tsne_csv = out_dir / "tsne_coordinates.csv"
    write_tsne_csv(tsne_csv, coords, samples_for_embedding)
    outputs["tsne_coordinates"] = tsne_csv
    outputs["plot"] = plot_tsne(coords, samples_for_embedding, cfg, out_dir)

    write_summary(out_dir, cfg, samples_for_embedding, warnings_out, missing_rows, pointcloud_errors, outputs)
    print(f"Done. Plot: {outputs['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
