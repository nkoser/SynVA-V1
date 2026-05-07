import argparse
import csv
import os
import random
import shutil
from glob import glob

import numpy as np
import yaml


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_counts(n_total, train_ratio, val_ratio):
    train_n = int(round(n_total * train_ratio))
    val_n = int(round(n_total * val_ratio))
    test_n = n_total - train_n - val_n
    if test_n < 0:
        test_n = 0
    return train_n, val_n, test_n


def infer_length(file_path):
    data = np.load(file_path, mmap_mode="r")
    if data.ndim <= 1:
        return int(data.size)
    return int(data.shape[0])


def stratified_split_by_length(files, train_ratio, val_ratio, seed, n_bins):
    lengths = {f: infer_length(f) for f in files}
    values = np.array([lengths[f] for f in files], dtype=np.float64)

    if len(files) < 3 or np.all(values == values[0]):
        shuffled = files[:]
        random.Random(seed).shuffle(shuffled)
        train_n, val_n, test_n = split_counts(len(shuffled), train_ratio, val_ratio)
        return (
            shuffled[:train_n],
            shuffled[train_n : train_n + val_n],
            shuffled[train_n + val_n : train_n + val_n + test_n],
            lengths,
            False,
        )

    quantiles = np.linspace(0.0, 1.0, int(max(1, n_bins)) + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        shuffled = files[:]
        random.Random(seed).shuffle(shuffled)
        train_n, val_n, test_n = split_counts(len(shuffled), train_ratio, val_ratio)
        return (
            shuffled[:train_n],
            shuffled[train_n : train_n + val_n],
            shuffled[train_n + val_n : train_n + val_n + test_n],
            lengths,
            False,
        )

    bin_ids = np.digitize(values, edges[1:-1], right=True)
    by_bin = {}
    for file_path, bin_id in zip(files, bin_ids):
        by_bin.setdefault(int(bin_id), []).append(file_path)

    rng = random.Random(seed)
    train_files, val_files, test_files = [], [], []

    for bin_id in sorted(by_bin.keys()):
        group = by_bin[bin_id]
        rng.shuffle(group)
        train_n, val_n, _ = split_counts(len(group), train_ratio, val_ratio)
        train_files.extend(group[:train_n])
        val_files.extend(group[train_n : train_n + val_n])
        test_files.extend(group[train_n + val_n :])

    target_train, target_val, target_test = split_counts(len(files), train_ratio, val_ratio)
    _rebalance_to_targets(
        train_files,
        val_files,
        test_files,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        seed=seed + 17,
    )

    return train_files, val_files, test_files, lengths, True


def _rebalance_to_targets(train_files, val_files, test_files, target_train, target_val, target_test, seed):
    rng = random.Random(seed)
    splits = {"train": train_files, "val": val_files, "test": test_files}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    while True:
        over = [name for name in splits if len(splits[name]) > targets[name]]
        under = [name for name in splits if len(splits[name]) < targets[name]]
        if not over or not under:
            break

        src = max(over, key=lambda name: len(splits[name]) - targets[name])
        dst = max(under, key=lambda name: targets[name] - len(splits[name]))

        idx = rng.randrange(len(splits[src]))
        item = splits[src].pop(idx)
        splits[dst].append(item)


def _print_length_stats(name, files, lengths):
    if not files:
        print(f"{name}: n=0")
        return
    vals = [lengths[f] for f in files]
    mean = sum(vals) / len(vals)
    print(
        f"{name}: n={len(files)} len[min/mean/max]={min(vals)}/{mean:.2f}/{max(vals)}"
    )


def _load_split_csv(path, delimiter=";", uid_col="uid", split_col="split"):
    mapping = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if uid_col not in (reader.fieldnames or []) or split_col not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV split file must contain columns {uid_col!r} and {split_col!r}: {path}"
            )
        for row in reader:
            uid = str(row.get(uid_col, "")).strip()
            split = str(row.get(split_col, "")).strip().lower()
            if not uid:
                continue
            if split not in {"train", "val", "valid", "validation", "test"}:
                continue
            if split in {"valid", "validation"}:
                split = "val"
            mapping[uid] = split
    return mapping


def csv_guided_split(files, split_csv, seed, train_ratio, val_ratio, delimiter=";",
                     uid_col="uid", split_col="split", unmatched="train",
                     stratify_val_by_length=True, stratify_bins=10):
    mapping = _load_split_csv(
        split_csv, delimiter=delimiter, uid_col=uid_col, split_col=split_col
    )
    lengths = {f: infer_length(f) for f in files}
    pools = {"train": [], "val": [], "test": []}
    missing = []

    for file_path in files:
        uid = os.path.splitext(os.path.basename(file_path))[0]
        split = mapping.get(uid)
        if split is None:
            missing.append(uid)
            if unmatched == "error":
                continue
            split = unmatched
        if split not in pools:
            split = "train"
        pools[split].append(file_path)

    if missing and unmatched == "error":
        preview = ", ".join(missing[:20])
        raise ValueError(f"{len(missing)} files are missing in {split_csv}: {preview}")

    # data_split_real.csv currently has train/test but no val. Keep test fixed
    # from the CSV and carve val reproducibly from the CSV train pool.
    if not pools["val"] and val_ratio > 0 and pools["train"]:
        val_from_train_ratio = val_ratio / max(train_ratio + val_ratio, 1e-8)
        if stratify_val_by_length:
            keep_train, val_files, _unused_test, _train_pool_lengths, _ = stratified_split_by_length(
                files=pools["train"],
                train_ratio=1.0 - val_from_train_ratio,
                val_ratio=val_from_train_ratio,
                seed=seed,
                n_bins=stratify_bins,
            )
        else:
            shuffled = pools["train"][:]
            random.Random(seed).shuffle(shuffled)
            val_n = int(round(len(shuffled) * val_from_train_ratio))
            val_files = shuffled[:val_n]
            keep_train = shuffled[val_n:]
        pools["train"] = keep_train
        pools["val"] = val_files

    return pools["train"], pools["val"], pools["test"], lengths, len(mapping), missing


def erase_split_dirs(output_dir):
    for name in ("train", "val", "test"):
        folder = os.path.join(output_dir, name)
        if not os.path.isdir(folder):
            continue
        for entry in os.listdir(folder):
            path = os.path.join(folder, entry)
            if os.path.isfile(path):
                os.remove(path)


def main():
    parser = argparse.ArgumentParser(description="Split tree dataset into train/val/test.")
    parser.add_argument("--config", default="split_dataset_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise SystemExit(f"Error: config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    input_dir = paths.get("input")
    output_dir = paths.get("output")
    train_ratio = float(params.get("train", 0.8))
    val_ratio = float(params.get("val", 0.1))
    test_ratio = float(params.get("test", 0.1))
    seed = int(params.get("seed", 42))
    pattern = params.get("pattern", "*.npy")
    move_files = bool(params.get("move", False))
    stratify_by_length = bool(params.get("stratify_by_length", True))
    stratify_bins = int(params.get("stratify_bins", 10))
    split_csv = params.get("split_csv")
    split_csv_delimiter = params.get("split_csv_delimiter", ";")
    split_csv_uid_col = params.get("split_csv_uid_col", "uid")
    split_csv_split_col = params.get("split_csv_split_col", "split")
    split_csv_unmatched = params.get("split_csv_unmatched", "train")
    erase_existing = bool(params.get("erase_existing", False))

    if not input_dir or not output_dir:
        raise SystemExit("Error: paths.input and paths.output are required in split_dataset_config.yaml.")

    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("Ratios must sum to a positive number")

    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    files = sorted(glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError("No files found in input")

    if split_csv:
        train_files, val_files, test_files, lengths, n_csv, missing_csv = csv_guided_split(
            files=files,
            split_csv=split_csv,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            delimiter=split_csv_delimiter,
            uid_col=split_csv_uid_col,
            split_col=split_csv_split_col,
            unmatched=split_csv_unmatched,
            stratify_val_by_length=stratify_by_length,
            stratify_bins=stratify_bins,
        )
        was_stratified = False
    elif stratify_by_length:
        train_files, val_files, test_files, lengths, was_stratified = stratified_split_by_length(
            files=files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            n_bins=stratify_bins,
        )
    else:
        random.seed(seed)
        random.shuffle(files)
        train_n, val_n, test_n = split_counts(len(files), train_ratio, val_ratio)
        train_files = files[:train_n]
        val_files = files[train_n : train_n + val_n]
        test_files = files[train_n + val_n : train_n + val_n + test_n]
        lengths = {f: infer_length(f) for f in files}
        was_stratified = False

    out_train = os.path.join(output_dir, "train")
    out_val = os.path.join(output_dir, "val")
    out_test = os.path.join(output_dir, "test")

    ensure_dir(out_train)
    ensure_dir(out_val)
    ensure_dir(out_test)

    if erase_existing:
        erase_split_dirs(output_dir)

    op = shutil.move if move_files else shutil.copy2

    for src in train_files:
        op(src, os.path.join(out_train, os.path.basename(src)))
    for src in val_files:
        op(src, os.path.join(out_val, os.path.basename(src)))
    for src in test_files:
        op(src, os.path.join(out_test, os.path.basename(src)))

    mode = "csv-guided" if split_csv else ("stratified-by-length" if was_stratified else "random")
    print(f"done ({mode}): train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    if split_csv:
        print(f"csv: path={split_csv} rows={n_csv} missing={len(missing_csv)} unmatched_policy={split_csv_unmatched}")
        if missing_csv:
            print("csv missing first:", ", ".join(missing_csv[:20]))
    _print_length_stats("train", train_files, lengths)
    _print_length_stats("val", val_files, lengths)
    _print_length_stats("test", test_files, lengths)


if __name__ == "__main__":
    main()
