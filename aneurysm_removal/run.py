"""
Batch processing: remove aneurysms from all cases in prepared_meshes_3.
Creates healthy vessel meshes in a new output directory.

Usage:
    python -m aneurysm_removal.run --config aneurysm_removal/config.yaml
"""

import argparse
import os
import shutil
import yaml
import numpy as np
import trimesh
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .fill_ostium import fill_ostium


def process_case(
    case_name: str,
    input_dir: str,
    output_dir: str,
    fill_cfg: dict,
    out_cfg: dict,
) -> dict:
    """Process one case: tube-cut-and-bridge, save healthy mesh."""
    result = {"case": case_name, "status": "ok", "msg": ""}
    try:
        case_in = os.path.join(input_dir, case_name)
        case_out = os.path.join(output_dir, case_name)

        mesh_path = os.path.join(case_in, "01_mesh", "mesh.obj")
        labels_path = os.path.join(case_in, "02_labels", "labels.npy")
        ostium_path = os.path.join(case_in, "07_other", "centroid_ostium.npy")
        normal_path = os.path.join(case_in, "07_other", "normal_vector.npy")

        if not os.path.exists(mesh_path):
            return {**result, "status": "skip", "msg": "no mesh.obj"}
        if not os.path.exists(labels_path):
            return {**result, "status": "skip", "msg": "no labels.npy"}
        if not os.path.exists(ostium_path):
            return {**result, "status": "skip", "msg": "no centroid_ostium.npy"}
        if not os.path.exists(normal_path):
            return {**result, "status": "skip", "msg": "no normal_vector.npy"}

        # Load
        full_mesh = trimesh.load(mesh_path, process=False)
        labels = np.load(labels_path)
        ostium_centroid = np.load(ostium_path)
        normal_vec = np.load(normal_path)

        # Tube-cut-and-bridge
        healthy = fill_ostium(
            full_mesh,
            labels,
            ostium_centroid,
            normal_vec,
            n_intermediate=fill_cfg.get("n_intermediate", 5),
            max_grow_rings=fill_cfg.get("max_grow_rings", 50),
        )

        # Copy directory structure if requested
        if out_cfg.get("copy_structure", True):
            if not os.path.exists(case_out):
                shutil.copytree(case_in, case_out)

        # Save healthy mesh
        out_filename = out_cfg.get("filename", "vessel_healthy.obj")
        mesh_out = os.path.join(case_out, "01_mesh", out_filename)
        os.makedirs(os.path.dirname(mesh_out), exist_ok=True)
        healthy.export(mesh_out)

        result["msg"] = f"{len(full_mesh.vertices)}v -> {len(healthy.vertices)}v"

    except Exception as e:
        result["status"] = "error"
        result["msg"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Remove aneurysms from vessel meshes")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--cases", nargs="*", help="Process only these cases")
    parser.add_argument("--dry-run", action="store_true", help="List cases without processing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    fill_cfg = cfg.get("fill", {})
    out_cfg = cfg.get("output", {})
    num_workers = cfg.get("num_workers", 0)
    filter_prefixes = cfg.get("filter_prefixes", [])

    # Discover cases
    if args.cases:
        cases = args.cases
    else:
        cases = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ])

    # Filter by prefix
    if filter_prefixes:
        cases = [c for c in cases if any(c.startswith(p) for p in filter_prefixes)]

    # Skip already-processed cases (resume support)
    out_filename = out_cfg.get("filename", "vessel_healthy.obj")
    if not args.dry_run:
        already_done = [
            c for c in cases
            if os.path.exists(os.path.join(output_dir, c, "01_mesh", out_filename))
        ]
        if already_done:
            print(f"Skipping {len(already_done)} already-processed cases")
            done_set = set(already_done)
            cases = [c for c in cases if c not in done_set]

    print(f"Found {len(cases)} cases to process in {input_dir}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        for c in cases:
            print(f"  {c}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Process
    stats = {"ok": 0, "skip": 0, "error": 0}

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(process_case, c, input_dir, output_dir, fill_cfg, out_cfg): c
                for c in cases
            }
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                stats[r["status"]] += 1
                status_char = "✓" if r["status"] == "ok" else ("⊘" if r["status"] == "skip" else "✗")
                print(f"[{i+1}/{len(cases)}] {status_char} {r['case']}: {r['msg']}")
    else:
        for i, c in enumerate(cases):
            r = process_case(c, input_dir, output_dir, fill_cfg, out_cfg)
            stats[r["status"]] += 1
            status_char = "✓" if r["status"] == "ok" else ("⊘" if r["status"] == "skip" else "✗")
            print(f"[{i+1}/{len(cases)}] {status_char} {r['case']}: {r['msg']}")

    print(f"\nDone: {stats['ok']} ok, {stats['skip']} skipped, {stats['error']} errors")


if __name__ == "__main__":
    main()
