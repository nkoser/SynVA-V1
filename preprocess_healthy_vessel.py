"""End-to-end preprocessing for /data/healthy_vessel meshes.

Mirrors the v4 → relpos_nodecp_v1 chain used to build
``derived_data/TreesSplines_k_count_100depth_prepared_norm_v4_relpos_nodecp_v1``
but uses split assignments from ``/data/data_split_real.csv`` and writes
outputs under the ``healthy_vessel`` naming scheme.

Pipeline:
  1. Build staging layout under /data/healthy_vessel_staging/<uid>/01_mesh/<uid>.obj
     via symlinks (uid = folder name with trailing "_vessel_submesh_closed" stripped).
     Only uids that exist BOTH in /data/healthy_vessel and /data/data_split_real.csv
     are included.
  2. Run the modular pipeline (Preprocessing_modular_v2) end-to-end up to
     TreesSplines.
  3. Trim trees to depth 100 (cortar_arboles).
  4. Split TreesSplines_k_count_100depth into train/val/test using
     /data/data_split_real.csv (10% of train deterministically held out as val).
  5. Prepare each split (10 rotations for train/val, 1 for test).
  6. Normalize (geometry_clip, knots untouched) -> *_prepared_norm_healthy_vessel.
  7. Convert to parent-relative + node-local CP -> derived_data/...relpos_nodecp_v1.

Usage:
    python preprocess_healthy_vessel.py             # run all steps
    python preprocess_healthy_vessel.py --skip stage,pipeline   # skip those steps
    python preprocess_healthy_vessel.py --only split,prepare    # only run those
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PMV2 = REPO / "Preprocessing_modular_v2"
CFG = PMV2 / "config" / "healthy_vessel"

PYTHON = sys.executable

HEALTHY_DIR = Path("/data/healthy_vessel_decapped")
STAGING_DIR = Path("/data/healthy_vessel_staging")
SPLIT_CSV = Path("/data/data_split_real.csv")
SUFFIX = ""  # decapped layout: <uid>/<uid>.obj (no suffix)

OUTPUT_ROOT = Path("/data/Output_healthy_vessel")
TREES_SPLINES_DIR = OUTPUT_ROOT / "TreesSplines"
TREES_100D_DIR    = OUTPUT_ROOT / "TreesSplines_k_count_100depth"
SPLIT_DIR         = OUTPUT_ROOT / "TreesSplines_k_count_100depth_split"

VAL_HOLDOUT_FRAC = 0.10   # fraction of train cases to move to val
VAL_SEED = "healthy_vessel_v1"  # deterministic salt for the hash split


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[healthy_vessel] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(f"command failed (rc={res.returncode}): {' '.join(cmd)}")


def load_csv_splits() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(SPLIT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            uid, split = row[0].strip(), row[1].strip().lower()
            if uid and split:
                out[uid] = split
    return out


def discover_uids() -> list[str]:
    uids: list[str] = []
    for entry in sorted(os.listdir(HEALTHY_DIR)):
        full = HEALTHY_DIR / entry
        if not full.is_dir():
            continue
        uid = entry  # decapped layout: folder name == uid
        obj = full / f"{uid}.obj"
        if not obj.is_file():
            log(f"WARN: no .obj inside {full}, skipping")
            continue
        uids.append(uid)
    return uids


def deterministic_hash01(uid: str) -> float:
    h = hashlib.sha256(f"{VAL_SEED}:{uid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# steps                                                                       #
# --------------------------------------------------------------------------- #
def step_stage(splits: dict[str, str], uids: list[str]) -> list[str]:
    """Symlink obj files into staging layout. Returns uids actually staged."""
    selected = sorted(set(splits) & set(uids))
    log(f"folders={len(uids)} csv={len(splits)} intersection={len(selected)}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for uid in selected:
        case_dir = STAGING_DIR / uid / "01_mesh"
        case_dir.mkdir(parents=True, exist_ok=True)
        src = HEALTHY_DIR / uid / f"{uid}.obj"
        dst = case_dir / f"{uid}.obj"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)
        staged.append(uid)
    log(f"staged {len(staged)} cases under {STAGING_DIR}")
    return staged


def step_pipeline() -> None:
    cfg = CFG / "pipeline_full.yaml"
    run([PYTHON, "run_preprocessing.py", "--config", str(cfg)], cwd=PMV2)


def step_cortar() -> None:
    run([PYTHON, "cortar_arboles.py", "--config",
         str(CFG / "cortar_arboles.yaml")], cwd=PMV2)


def step_split(splits: dict[str, str], uids_staged: list[str]) -> None:
    """CSV-driven split of TreesSplines_k_count_100depth into train/val/test."""
    if not TREES_100D_DIR.is_dir():
        raise SystemExit(f"missing {TREES_100D_DIR}; run cortar first")

    available = {p.stem: p for p in TREES_100D_DIR.glob("*.npy")}
    log(f"depth-100 files available: {len(available)}")

    out_train = SPLIT_DIR / "train"
    out_val   = SPLIT_DIR / "val"
    out_test  = SPLIT_DIR / "test"
    for d in (out_train, out_val, out_test):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_train = n_val = n_test = n_missing = 0
    for uid in uids_staged:
        src = available.get(uid)
        if src is None:
            n_missing += 1
            continue
        csv_split = splits.get(uid, "train")
        if csv_split == "test":
            dst_dir = out_test
            n_test += 1
        else:  # 'train' (or anything else) → maybe holdout to val
            if deterministic_hash01(uid) < VAL_HOLDOUT_FRAC:
                dst_dir = out_val
                n_val += 1
            else:
                dst_dir = out_train
                n_train += 1
        # copy (cheap; .npy files are small) so prepare can erase output safely
        shutil.copy2(src, dst_dir / src.name)

    log(f"split → train={n_train} val={n_val} test={n_test} (missing={n_missing})")


def step_prepare() -> None:
    for split in ("train", "val", "test"):
        run([PYTHON, "prepare_dataset.py", "--config",
             str(CFG / f"prepare_{split}.yaml")], cwd=PMV2)


def step_normalize() -> None:
    run([PYTHON, "normalize_dataset.py", "--config",
         str(CFG / "normalize.yaml")], cwd=PMV2)


def step_relpos() -> None:
    run([PYTHON, "convert_dataset_parent_relative.py", "--config",
         str(CFG / "relpos_nodecp.yaml")], cwd=PMV2)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
ALL_STEPS = ["stage", "pipeline", "cortar", "split", "prepare",
             "normalize", "relpos"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="",
                    help=f"comma-separated subset of steps to run "
                         f"(default: all). Steps: {','.join(ALL_STEPS)}")
    ap.add_argument("--skip", default="",
                    help="comma-separated steps to skip")
    args = ap.parse_args()

    only  = {s.strip() for s in args.only.split(",")  if s.strip()}
    skip  = {s.strip() for s in args.skip.split(",")  if s.strip()}
    todo  = [s for s in ALL_STEPS if (not only or s in only) and s not in skip]
    log(f"running steps: {todo}")

    splits = load_csv_splits()
    uids   = discover_uids()
    staged = sorted(set(splits) & set(uids))   # used by split step too

    if "stage" in todo:
        staged = step_stage(splits, uids)
    if "pipeline" in todo:
        step_pipeline()
    if "cortar" in todo:
        step_cortar()
    if "split" in todo:
        step_split(splits, staged)
    if "prepare" in todo:
        step_prepare()
    if "normalize" in todo:
        step_normalize()
    if "relpos" in todo:
        step_relpos()
    log("DONE")


if __name__ == "__main__":
    main()
