from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage2_BranchTopology.representation import (
    compress_k_counts_to_branch_skeleton,
    compute_depths,
    normalize_k_counts,
    preorder_kcount_parent_indices,
    trim_valid_rows,
)


def strip_rotation_prefix(name: str) -> str:
    return re.sub(r"^rot\d+-", "", str(name))


def is_valid_preorder_kcount(k_counts: Sequence[int]) -> bool:
    slots = 1
    for value in normalize_k_counts(k_counts):
        slots -= 1
        if slots < 0:
            return False
        slots += int(value)
    return slots == 0


def summarize_values(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_max": float(arr.max()),
    }


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.sort(np.asarray(list(a), dtype=np.float64))
    y = np.sort(np.asarray(list(b), dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, max(x.size, y.size))
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def ks_distance(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.sort(np.asarray(list(a), dtype=np.float64))
    y = np.sort(np.asarray(list(b), dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, vals, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, vals, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def probability_histogram(a: Sequence[float], b: Sequence[float], eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if np.allclose(x, np.round(x)) and np.allclose(y, np.round(y)):
        start = math.floor(lo) - 0.5
        stop = math.ceil(hi) + 1.5
        bins = np.arange(start, stop, 1.0)
    else:
        bins = np.histogram_bin_edges(np.concatenate([x, y]), bins="auto")
        if bins.size < 2:
            bins = np.asarray([lo - 0.5, hi + 0.5], dtype=np.float64)
    px, _ = np.histogram(x, bins=bins)
    py, _ = np.histogram(y, bins=bins)
    px = px.astype(np.float64) + float(eps)
    py = py.astype(np.float64) + float(eps)
    px /= px.sum()
    py /= py.sum()
    return px, py


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def compare_value_distribution(
    name: str,
    gt_values: Sequence[float],
    gen_values: Sequence[float],
    source: str,
) -> Dict[str, float]:
    gt = [float(v) for v in gt_values]
    gen = [float(v) for v in gen_values]
    gt_mean = float(np.mean(gt)) if gt else float("nan")
    gen_mean = float(np.mean(gen)) if gen else float("nan")
    p, q = probability_histogram(gt, gen)
    return {
        "metric": name,
        "source": source,
        "gt_n_values": int(len(gt)),
        "gen_n_values": int(len(gen)),
        "gt_mean": gt_mean,
        "gen_mean": gen_mean,
        "mean_delta": gen_mean - gt_mean if math.isfinite(gt_mean) and math.isfinite(gen_mean) else float("nan"),
        "gt_std": float(np.std(gt)) if gt else float("nan"),
        "gen_std": float(np.std(gen)) if gen else float("nan"),
        "kl_gt_to_gen": kl_divergence(p, q),
        "kl_gen_to_gt": kl_divergence(q, p),
        "jsd": js_divergence(p, q),
        "wasserstein": wasserstein_1d(gt, gen),
        "ks": ks_distance(gt, gen),
    }


def graph_degrees_from_parents(parents: Sequence[int]) -> List[int]:
    degrees = [0] * len(parents)
    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0:
            degrees[idx] += 1
            degrees[parent] += 1
    return [int(v) for v in degrees]


def laplacian_spectrum_from_parents(parents: Sequence[int]) -> List[float]:
    n = int(len(parents))
    if n == 0:
        return []
    adjacency = np.zeros((n, n), dtype=np.float64)
    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0:
            adjacency[idx, parent] = 1.0
            adjacency[parent, idx] = 1.0
    degree = np.diag(adjacency.sum(axis=1))
    eigenvalues = np.linalg.eigvalsh(degree - adjacency)
    return [float(v) for v in eigenvalues]


def betti_numbers(parents: Sequence[int]) -> tuple[int, int]:
    n = int(len(parents))
    if n == 0:
        return 0, 0
    edges = int(sum(1 for parent in parents if int(parent) >= 0))
    roots = [idx for idx, parent in enumerate(parents) if int(parent) < 0]
    uf = list(range(n))

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            uf[rb] = ra

    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0:
            union(idx, parent)

    beta0 = int(len({find(idx) for idx in range(n)}))
    beta1 = int(max(0, edges - n + beta0))
    if not roots:
        beta0 = max(beta0, 1)
    return beta0, beta1


def metrics_from_k_counts(name: str, k_counts: Sequence[int]) -> Dict[str, float]:
    k = normalize_k_counts(k_counts)
    parents = preorder_kcount_parent_indices(k)
    depths = compute_depths(parents)
    valid = is_valid_preorder_kcount(k)
    skeleton = compress_k_counts_to_branch_skeleton(k)
    incoming = [int(v) for v in skeleton["incoming_lengths"]]
    nonroot_incoming = incoming[1:]
    degrees = [int(v) for v in skeleton["degrees"]]
    event_depths = compute_depths(skeleton["event_parents"])
    dense_edges = int(sum(1 for parent in parents if int(parent) >= 0))
    event_edges = int(sum(1 for parent in skeleton["event_parents"] if int(parent) >= 0))
    dense_beta0, dense_beta1 = betti_numbers(parents)
    event_beta0, event_beta1 = betti_numbers(skeleton["event_parents"])
    dense_graph_degrees = graph_degrees_from_parents(parents)
    event_graph_degrees = graph_degrees_from_parents(skeleton["event_parents"])
    dense_spectrum = laplacian_spectrum_from_parents(parents)
    event_spectrum = laplacian_spectrum_from_parents(skeleton["event_parents"])
    dense_algebraic_connectivity = dense_spectrum[1] if len(dense_spectrum) > 1 else 0.0
    event_algebraic_connectivity = event_spectrum[1] if len(event_spectrum) > 1 else 0.0

    return {
        "name": name,
        "valid": int(valid),
        "dense_nodes": int(len(k)),
        "dense_edges": dense_edges,
        "dense_leaves": int(sum(1 for v in k if v == 0)),
        "dense_unary": int(sum(1 for v in k if v == 1)),
        "dense_bifurcations": int(sum(1 for v in k if v == 2)),
        "dense_beta0": int(dense_beta0),
        "dense_beta1": int(dense_beta1),
        "dense_laplacian_lambda_max": float(max(dense_spectrum) if dense_spectrum else 0.0),
        "dense_algebraic_connectivity": float(dense_algebraic_connectivity),
        "dense_max_depth": int(max(depths) if depths else 0),
        "dense_mean_depth": float(np.mean(depths) if depths else 0.0),
        "event_nodes": int(len(degrees)),
        "event_edges": event_edges,
        "event_leaves": int(sum(1 for v in degrees if v == 0)),
        "event_unary": int(sum(1 for v in degrees if v == 1)),
        "event_bifurcations": int(sum(1 for v in degrees if v == 2)),
        "event_beta0": int(event_beta0),
        "event_beta1": int(event_beta1),
        "event_laplacian_lambda_max": float(max(event_spectrum) if event_spectrum else 0.0),
        "event_algebraic_connectivity": float(event_algebraic_connectivity),
        "event_max_depth": int(max(event_depths) if event_depths else 0),
        "event_mean_depth": float(np.mean(event_depths) if event_depths else 0.0),
        "incoming_total_unary": int(sum(nonroot_incoming)),
        "incoming_mean": float(np.mean(nonroot_incoming) if nonroot_incoming else 0.0),
        "incoming_max": int(max(nonroot_incoming) if nonroot_incoming else 0),
        "_dense_child_counts": [int(v) for v in k],
        "_dense_graph_degrees": dense_graph_degrees,
        "_dense_depth_values": [int(v) for v in depths],
        "_dense_laplacian_eigenvalues": dense_spectrum,
        "_event_child_counts": degrees,
        "_event_graph_degrees": event_graph_degrees,
        "_event_depth_values": [int(v) for v in event_depths],
        "_event_laplacian_eigenvalues": event_spectrum,
        "_incoming_lengths": nonroot_incoming,
    }


def load_gt_metrics(gt_dir: str) -> List[Dict[str, float]]:
    paths = sorted(Path(gt_dir).glob("*.npy"))
    seen = set()
    rows = []
    for path in paths:
        name = path.stem
        key = strip_rotation_prefix(name)
        if key in seen:
            continue
        seen.add(key)
        arr = np.load(path)
        valid_rows = trim_valid_rows(arr)
        if not valid_rows:
            continue
        k_counts = [row[0] for row in valid_rows]
        rows.append(metrics_from_k_counts(key, k_counts))
    return rows


def load_generated_metrics(sample_dir: str) -> List[Dict[str, float]]:
    rows = []
    for path in sorted(Path(sample_dir).glob("sample_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload.get("valid", False):
            rows.append({"name": path.stem, "valid": 0})
            continue
        rows.append(metrics_from_k_counts(path.stem, payload["dense_k_counts"]))
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys() if not key.startswith("_")})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def aggregate(rows: Sequence[Dict[str, float]], label: str) -> Dict[str, float]:
    out: Dict[str, float] = {"label": label, "n": int(len(rows))}
    numeric_keys = sorted(
        key
        for key in {k for row in rows for k in row.keys()}
        if key != "name" and all(isinstance(row.get(key), (int, float)) for row in rows if key in row)
    )
    for key in numeric_keys:
        out.update(summarize_values([float(row[key]) for row in rows if key in row], key))
    return out


def compare_distributions(gt_rows: Sequence[Dict[str, float]], gen_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    keys = [
        "dense_nodes",
        "dense_edges",
        "dense_leaves",
        "dense_unary",
        "dense_bifurcations",
        "dense_beta0",
        "dense_beta1",
        "dense_laplacian_lambda_max",
        "dense_algebraic_connectivity",
        "dense_max_depth",
        "dense_mean_depth",
        "event_nodes",
        "event_edges",
        "event_leaves",
        "event_bifurcations",
        "event_beta0",
        "event_beta1",
        "event_laplacian_lambda_max",
        "event_algebraic_connectivity",
        "event_max_depth",
        "incoming_total_unary",
        "incoming_mean",
        "incoming_max",
    ]
    rows = []
    valid_gen = [row for row in gen_rows if int(row.get("valid", 0)) == 1]
    for key in keys:
        gt_values = [float(row[key]) for row in gt_rows if key in row]
        gen_values = [float(row[key]) for row in valid_gen if key in row]
        gt_mean = float(np.mean(gt_values)) if gt_values else float("nan")
        gen_mean = float(np.mean(gen_values)) if gen_values else float("nan")
        rows.append(
            {
                "metric": key,
                "gt_mean": gt_mean,
                "gen_mean": gen_mean,
                "mean_delta": gen_mean - gt_mean if math.isfinite(gt_mean) and math.isfinite(gen_mean) else float("nan"),
                "gt_std": float(np.std(gt_values)) if gt_values else float("nan"),
                "gen_std": float(np.std(gen_values)) if gen_values else float("nan"),
                "wasserstein": wasserstein_1d(gt_values, gen_values),
                "ks": ks_distance(gt_values, gen_values),
            }
        )
    return rows


def collect_private_values(rows: Sequence[Dict[str, float]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        for value in row.get(key, []):
            values.append(float(value))
    return values


def compare_literature_topology_distributions(
    gt_rows: Sequence[Dict[str, float]], gen_rows: Sequence[Dict[str, float]]
) -> List[Dict[str, float]]:
    valid_gen = [row for row in gen_rows if int(row.get("valid", 0)) == 1]
    scalar_specs = [
        ("|V| dense nodes", "dense_nodes", "per-tree graph statistic"),
        ("|E| dense edges", "dense_edges", "per-tree graph statistic"),
        ("terminal branches / leaves", "dense_leaves", "per-tree graph statistic"),
        ("bifurcations", "dense_bifurcations", "per-tree graph statistic"),
        ("tree depth", "dense_max_depth", "per-tree graph statistic"),
        ("beta0 connected components", "dense_beta0", "Betti topology"),
        ("beta1 cycles", "dense_beta1", "Betti topology"),
        ("Laplacian lambda max", "dense_laplacian_lambda_max", "graph spectrum"),
        ("algebraic connectivity", "dense_algebraic_connectivity", "graph spectrum"),
        ("event graph |V|", "event_nodes", "compressed branch topology"),
        ("event graph |E|", "event_edges", "compressed branch topology"),
        ("event graph depth", "event_max_depth", "compressed branch topology"),
        ("event Laplacian lambda max", "event_laplacian_lambda_max", "compressed graph spectrum"),
        ("event algebraic connectivity", "event_algebraic_connectivity", "compressed graph spectrum"),
        ("incoming branch length", "incoming_mean", "compressed branch geometry/topology"),
    ]
    private_specs = [
        ("node degree distribution", "_dense_graph_degrees", "node-level graph statistic"),
        ("child-count distribution", "_dense_child_counts", "rooted tree degree statistic"),
        ("node depth distribution", "_dense_depth_values", "node-level graph statistic"),
        ("Laplacian spectrum distribution", "_dense_laplacian_eigenvalues", "graph spectrum"),
        ("event node degree distribution", "_event_graph_degrees", "compressed branch topology"),
        ("event child-count distribution", "_event_child_counts", "compressed branch topology"),
        ("event depth distribution", "_event_depth_values", "compressed branch topology"),
        ("event Laplacian spectrum distribution", "_event_laplacian_eigenvalues", "compressed graph spectrum"),
        ("incoming branch-length distribution", "_incoming_lengths", "compressed branch topology"),
    ]

    rows: List[Dict[str, float]] = []
    for label, key, source in scalar_specs:
        rows.append(
            compare_value_distribution(
                label,
                [float(row[key]) for row in gt_rows if key in row],
                [float(row[key]) for row in valid_gen if key in row],
                source,
            )
        )
    for label, key, source in private_specs:
        rows.append(
            compare_value_distribution(
                label,
                collect_private_values(gt_rows, key),
                collect_private_values(valid_gen, key),
                source,
            )
        )
    return rows


def topology_feature_matrix(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        if all(key in row for key in keys):
            matrix.append([float(row[key]) for key in keys])
    if not matrix:
        return np.zeros((0, len(keys)), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def one_nearest_neighbor_accuracy(gt_features: np.ndarray, gen_features: np.ndarray) -> float:
    features = np.concatenate([gt_features, gen_features], axis=0)
    labels = np.concatenate(
        [
            np.zeros(gt_features.shape[0], dtype=np.int64),
            np.ones(gen_features.shape[0], dtype=np.int64),
        ]
    )
    if features.shape[0] <= 1:
        return float("nan")
    distances = pairwise_l2(features, features)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(labels[nearest] == labels))


def mean_pairwise_distance(features: np.ndarray) -> float:
    if features.shape[0] <= 1:
        return float("nan")
    distances = pairwise_l2(features, features)
    tri = np.triu_indices(features.shape[0], k=1)
    return float(np.mean(distances[tri]))


def compare_topology_coverage(gt_rows: Sequence[Dict[str, float]], gen_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    valid_gen = [row for row in gen_rows if int(row.get("valid", 0)) == 1]
    feature_keys = [
        "dense_nodes",
        "dense_leaves",
        "dense_bifurcations",
        "dense_max_depth",
        "dense_mean_depth",
        "event_nodes",
        "event_max_depth",
        "incoming_mean",
        "incoming_max",
        "dense_laplacian_lambda_max",
        "dense_algebraic_connectivity",
    ]
    gt_features = topology_feature_matrix(gt_rows, feature_keys)
    gen_features = topology_feature_matrix(valid_gen, feature_keys)
    if gt_features.size == 0 or gen_features.size == 0:
        return {
            "feature_keys": list(feature_keys),
            "mmd_topology": float("nan"),
            "coverage_topology": float("nan"),
            "one_nna_topology": float("nan"),
            "gen_pairwise_diversity": float("nan"),
            "gt_pairwise_diversity": float("nan"),
            "relative_pairwise_diversity": float("nan"),
            "exact_generated_unique_fraction": float("nan"),
            "exact_generated_duplicates": 0,
            "exact_gt_covered_fraction": float("nan"),
            "exact_generated_seen_in_gt_fraction": float("nan"),
        }

    center = np.mean(gt_features, axis=0, keepdims=True)
    scale = np.std(gt_features, axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    gt_norm = (gt_features - center) / scale
    gen_norm = (gen_features - center) / scale
    gt_to_gen = pairwise_l2(gt_norm, gen_norm)

    nearest_gen_per_gt = np.min(gt_to_gen, axis=1)
    nearest_gt_per_gen = np.argmin(gt_to_gen, axis=0)
    covered_gt = set(int(v) for v in nearest_gt_per_gen.tolist())
    gt_pairwise = mean_pairwise_distance(gt_norm)
    gen_pairwise = mean_pairwise_distance(gen_norm)

    gt_exact = {tuple(int(v) for v in row.get("_dense_child_counts", [])) for row in gt_rows}
    gen_exact = [tuple(int(v) for v in row.get("_dense_child_counts", [])) for row in valid_gen]
    gen_unique = set(gen_exact)
    exact_hits = gen_unique.intersection(gt_exact)

    return {
        "feature_keys": list(feature_keys),
        "mmd_topology": float(np.mean(nearest_gen_per_gt)),
        "coverage_topology": float(len(covered_gt) / gt_norm.shape[0]),
        "one_nna_topology": one_nearest_neighbor_accuracy(gt_norm, gen_norm),
        "gen_pairwise_diversity": gen_pairwise,
        "gt_pairwise_diversity": gt_pairwise,
        "relative_pairwise_diversity": float(gen_pairwise / gt_pairwise) if gt_pairwise and math.isfinite(gt_pairwise) else float("nan"),
        "exact_generated_unique_fraction": float(len(gen_unique) / len(gen_exact)) if gen_exact else float("nan"),
        "exact_generated_duplicates": int(len(gen_exact) - len(gen_unique)),
        "exact_gt_covered_fraction": float(len(exact_hits) / len(gt_exact)) if gt_exact else float("nan"),
        "exact_generated_seen_in_gt_fraction": float(len(exact_hits) / len(gen_unique)) if gen_unique else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated branch topologies against GT test topology metrics.")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gt_rows = load_gt_metrics(args.gt_dir)
    gen_rows = load_generated_metrics(args.generated_dir)
    valid_gen = [row for row in gen_rows if int(row.get("valid", 0)) == 1]
    comparison = compare_distributions(gt_rows, gen_rows)
    literature_comparison = compare_literature_topology_distributions(gt_rows, gen_rows)
    coverage_comparison = compare_topology_coverage(gt_rows, gen_rows)

    write_csv(os.path.join(args.output_dir, "gt_metrics.csv"), gt_rows)
    write_csv(os.path.join(args.output_dir, "generated_metrics.csv"), gen_rows)
    write_csv(os.path.join(args.output_dir, "metric_comparison.csv"), comparison)
    write_csv(os.path.join(args.output_dir, "literature_topology_comparison.csv"), literature_comparison)

    summary = {
        "gt": aggregate(gt_rows, "gt"),
        "generated": aggregate(gen_rows, "generated"),
        "generated_valid": aggregate(valid_gen, "generated_valid"),
        "generated_valid_fraction": float(len(valid_gen) / len(gen_rows)) if gen_rows else 0.0,
        "comparison": comparison,
        "literature_topology_comparison": literature_comparison,
        "topology_coverage": coverage_comparison,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(args.output_dir, "topology_coverage.json"), "w", encoding="utf-8") as handle:
        json.dump(coverage_comparison, handle, indent=2)

    print(f"GT samples: {len(gt_rows)}")
    print(f"Generated samples: {len(gen_rows)} valid={len(valid_gen)} fraction={summary['generated_valid_fraction']:.3f}")
    print(f"Saved topology metrics to: {args.output_dir}")
    print("Key comparison:")
    for row in comparison:
        print(
            f"  {row['metric']}: gt_mean={row['gt_mean']:.3f} "
            f"gen_mean={row['gen_mean']:.3f} W1={row['wasserstein']:.3f} KS={row['ks']:.3f}"
        )
    print("Literature-style topology comparison:")
    for row in literature_comparison:
        print(
            f"  {row['metric']}: gt_mean={row['gt_mean']:.3f} gen_mean={row['gen_mean']:.3f} "
            f"KL(gt||gen)={row['kl_gt_to_gen']:.4f} JSD={row['jsd']:.4f} W1={row['wasserstein']:.3f}"
        )
    print("Topology coverage/diversity:")
    print(f"  MMD-topology={coverage_comparison['mmd_topology']:.4f}")
    print(f"  COV-topology={coverage_comparison['coverage_topology']:.4f}")
    print(f"  1-NNA-topology={coverage_comparison['one_nna_topology']:.4f}")
    print(
        f"  pairwise diversity gen={coverage_comparison['gen_pairwise_diversity']:.4f} "
        f"gt={coverage_comparison['gt_pairwise_diversity']:.4f} "
        f"relative={coverage_comparison['relative_pairwise_diversity']:.4f}"
    )
    print(
        f"  exact unique generated={coverage_comparison['exact_generated_unique_fraction']:.4f} "
        f"duplicates={coverage_comparison['exact_generated_duplicates']}"
    )


if __name__ == "__main__":
    main()
