from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage2_BranchTopology.representation import (
    compress_k_counts_to_branch_skeleton,
    normalize_k_counts,
    trim_valid_rows,
)
from Stage2_BranchTopology.vocab import build_branch_skeleton_vocab, encode_branch_skeleton


def load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg or {}


def list_npy_files(folder_path: str, limit: Optional[int] = None) -> List[str]:
    files = [
        os.path.join(folder_path, name)
        for name in sorted(os.listdir(folder_path))
        if name.endswith(".npy") and not name.startswith(".")
    ]
    if limit is not None:
        files = files[: int(limit)]
    return files


def strip_rotation_prefix(name: str) -> str:
    return re.sub(r"^rot\d+-", "", str(name))


def deduplicate_samples(samples, deduplicate_exact_sequences=False):
    current = list(samples)

    case_removed = 0
    unique = []
    seen = set()
    for file_path, skeleton in current:
        base = os.path.splitext(os.path.basename(file_path))[0]
        key = strip_rotation_prefix(base)
        if key in seen:
            case_removed += 1
            continue
        seen.add(key)
        unique.append((file_path, skeleton))
    current = unique

    sequence_removed = 0
    if deduplicate_exact_sequences:
        unique = []
        seen = set()
        for file_path, skeleton in current:
            key = (
                tuple(int(v) for v in skeleton["incoming_lengths"]),
                tuple(int(v) for v in skeleton["degrees"]),
            )
            if key in seen:
                sequence_removed += 1
                continue
            seen.add(key)
            unique.append((file_path, skeleton))
        current = unique

    return current, int(case_removed), int(sequence_removed)


def scan_split(
    file_paths: List[str],
    deduplicate_exact_sequences: bool = False,
) -> Dict[str, object]:
    samples = []
    max_incoming_length = 0
    original_nodes_total = 0
    event_nodes_total = 0
    for file_path in file_paths:
        arr = np.load(file_path)
        arr = trim_valid_rows(arr)
        if not arr:
            continue
        k_counts = normalize_k_counts(row[0] for row in arr)
        skeleton = compress_k_counts_to_branch_skeleton(k_counts)
        max_incoming_length = max(max_incoming_length, max(skeleton["incoming_lengths"], default=0))
        original_nodes_total += len(k_counts)
        event_nodes_total += len(skeleton["degrees"])
        samples.append((file_path, skeleton))

    raw_num_samples = len(samples)
    samples, case_removed, sequence_removed = deduplicate_samples(
        samples,
        deduplicate_exact_sequences=deduplicate_exact_sequences,
    )

    dedup_original_nodes_total = sum(len(skeleton["dense_k_counts"]) for _, skeleton in samples)
    dedup_event_nodes_total = sum(len(skeleton["degrees"]) for _, skeleton in samples)
    max_incoming_length = max(
        [max_incoming_length] + [max(skeleton["incoming_lengths"], default=0) for _, skeleton in samples]
    )

    return {
        "samples": samples,
        "raw_num_samples": int(raw_num_samples),
        "num_samples": int(len(samples)),
        "max_incoming_length": int(max_incoming_length),
        "original_nodes_total": int(dedup_original_nodes_total),
        "event_nodes_total": int(dedup_event_nodes_total),
        "raw_original_nodes_total": int(original_nodes_total),
        "raw_event_nodes_total": int(event_nodes_total),
        "dedup_case_rotations_removed": int(case_removed),
        "dedup_exact_sequences_removed": int(sequence_removed),
    }


def save_split(samples, output_dir: str, vocab: Dict[str, object]) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    n_samples = 0
    original_nodes_total = 0
    event_nodes_total = 0
    max_seq_len = 0
    max_event_nodes = 0
    max_original_nodes = 0

    for file_path, skeleton in samples:
        base = os.path.splitext(os.path.basename(file_path))[0]
        tokens = encode_branch_skeleton(
            skeleton["incoming_lengths"],
            skeleton["degrees"],
            vocab,
        )
        tensor = torch.tensor(tokens, dtype=torch.long)
        torch.save(tensor, os.path.join(output_dir, base + ".tok"))

        n_samples += 1
        original_nodes = len(skeleton["dense_k_counts"])
        event_nodes = len(skeleton["degrees"])
        original_nodes_total += original_nodes
        event_nodes_total += event_nodes
        max_seq_len = max(max_seq_len, int(tensor.numel()))
        max_event_nodes = max(max_event_nodes, event_nodes)
        max_original_nodes = max(max_original_nodes, original_nodes)

    ratio = float(event_nodes_total) / float(original_nodes_total) if original_nodes_total > 0 else 0.0
    return {
        "num_samples": int(n_samples),
        "original_nodes_total": int(original_nodes_total),
        "event_nodes_total": int(event_nodes_total),
        "compression_ratio": float(ratio),
        "max_sequence_length": int(max_seq_len),
        "max_event_nodes": int(max_event_nodes),
        "max_original_nodes": int(max_original_nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dense vessel trees into branch-skeleton topology tokens.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    train_dir = paths.get("train_dir")
    val_dir = paths.get("val_dir")
    output_dir = paths.get("output_dir")
    if not train_dir or not output_dir:
        raise ValueError("paths.train_dir and paths.output_dir are required.")

    train_files = list_npy_files(train_dir, limit=params.get("train_limit"))
    val_files = list_npy_files(val_dir, limit=params.get("val_limit")) if val_dir else []
    deduplicate_exact_sequences = bool(params.get("deduplicate_exact_sequences", False))

    scanned_train = scan_split(
        train_files,
        deduplicate_exact_sequences=deduplicate_exact_sequences,
    )
    scanned_val = scan_split(
        val_files,
        deduplicate_exact_sequences=deduplicate_exact_sequences,
    )
    max_incoming_length = max(
        int(scanned_train["max_incoming_length"]),
        int(scanned_val["max_incoming_length"]),
    )
    vocab = build_branch_skeleton_vocab(max_incoming_length=max_incoming_length)

    train_out = os.path.join(output_dir, "train")
    val_out = os.path.join(output_dir, "val")
    train_stats = save_split(scanned_train["samples"], train_out, vocab)
    val_stats = save_split(scanned_val["samples"], val_out, vocab) if val_dir else {}

    metadata = {
        "representation": vocab["representation"],
        "vocab": vocab,
        "paths": {
            "source_train_dir": os.path.abspath(train_dir),
            "source_val_dir": os.path.abspath(val_dir) if val_dir else None,
            "output_dir": os.path.abspath(output_dir),
            "train_token_dir": os.path.abspath(train_out),
            "val_token_dir": os.path.abspath(val_out) if val_dir else None,
        },
        "stats": {
            "train": train_stats,
            "val": val_stats,
        },
        "dedup": {
            "deduplicate_exact_sequences": deduplicate_exact_sequences,
            "train_raw_num_samples": int(scanned_train["raw_num_samples"]),
            "train_case_removed": int(scanned_train["dedup_case_rotations_removed"]),
            "train_sequence_removed": int(scanned_train["dedup_exact_sequences_removed"]),
            "val_raw_num_samples": int(scanned_val["raw_num_samples"]),
            "val_case_removed": int(scanned_val["dedup_case_rotations_removed"]),
            "val_sequence_removed": int(scanned_val["dedup_exact_sequences_removed"]),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "branch_topology_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved branch-skeleton tokens to: {output_dir}")
    print(f"Metadata: {meta_path}")
    print(
        "Train stats: "
        f"samples={train_stats.get('num_samples', 0)} "
        f"compression_ratio={train_stats.get('compression_ratio', 0.0):.4f} "
        f"max_seq_len={train_stats.get('max_sequence_length', 0)} "
        f"max_incoming_length={vocab['max_incoming_length']}"
    )
    print(
        "Train dedup: "
        f"raw={scanned_train['raw_num_samples']} "
        f"removed_case={scanned_train['dedup_case_rotations_removed']} "
        f"removed_seq={scanned_train['dedup_exact_sequences_removed']} "
        f"kept={train_stats.get('num_samples', 0)}"
    )
    if val_stats:
        print(
            "Val stats: "
            f"samples={val_stats.get('num_samples', 0)} "
            f"compression_ratio={val_stats.get('compression_ratio', 0.0):.4f} "
            f"max_seq_len={val_stats.get('max_sequence_length', 0)}"
        )
        print(
            "Val dedup: "
            f"raw={scanned_val['raw_num_samples']} "
            f"removed_case={scanned_val['dedup_case_rotations_removed']} "
            f"removed_seq={scanned_val['dedup_exact_sequences_removed']} "
            f"kept={val_stats.get('num_samples', 0)}"
        )


if __name__ == "__main__":
    main()
