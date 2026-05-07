#!/usr/bin/env python3
"""
Build `generate_*_v2_clean.yaml` configs from `generate_*_healthy*.yaml`
templates and the actual training `*_v2_clean.yaml` configs.

Strategy:
  1. Load the existing healthy generate config (PyYAML).
  2. Load the corresponding training v2_clean config to get the real
     `paths.output_dir` for the checkpoint(s).
  3. Patch:
       - `paths.val_dir`               -> biffilter_v1 test split
       - `paths.checkpoint`            -> <train_output_dir>/best_model.pt
       - `paths.checkpoint_a/b`        -> from train_a / train_b configs
       - `paths.ae_checkpoint`         -> from ae train config
       - `paths.fm_checkpoint`         -> from latent_fm train config (best_fm_model.pt)
       - `params.output_dir`           -> generated/<basename>_v2_clean
     and leave everything else (feature_stats, aneu_label_dir, model defs)
     identical to the healthy template.
  4. Write to <pkg>/configs/generate_<NAME>_v2_clean.yaml.

Usage:
    python make_v2_clean_generate_configs.py
    python make_v2_clean_generate_configs.py --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent

OLD_TEST = "derived_data/TreesSplines_k_count_100depth_prepared_norm_healthy_vessel_relpos_nodecp_v1/test"
NEW_TEST = "derived_data/TreesSplines_k_count_100depth_prepared_biffilter_v1_norm_relpos_nodecp_v1/test"


# Each entry describes one generate-config to build.
TASKS = [
    # --- FlowMatching ---
    dict(name="flow_v11",
         src="Stage2_FlowMatching/configs/generate_flow_v11b_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching/configs/flow_matching_v11_v2_clean.yaml"}),
    dict(name="flow_v12_abspos",
         src="Stage2_FlowMatching/configs/generate_flow_v12_abspos_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching/configs/flow_matching_v12_abspos_v2_clean.yaml"}),

    # --- TreeGNN ---
    dict(name="treegnn_v1",
         src="Stage2_FlowMatching_TreeGNN/configs/generate_treegnn_v1_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching_TreeGNN/configs/treegnn_v1_v2_clean.yaml"}),
    dict(name="treegnn_v2_abspos",
         src="Stage2_FlowMatching_TreeGNN/configs/generate_treegnn_v2_abspos_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching_TreeGNN/configs/treegnn_v2_abspos_v2_clean.yaml"}),

    # --- Physio ---
    dict(name="physio_v5",
         src="Stage2_FlowMatching_Physio/configs/generate_physio_v5_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching_Physio/configs/physio_v5_v2_clean.yaml"}),
    dict(name="physio_v6",
         src="Stage2_FlowMatching_Physio/configs/generate_physio_v6_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching_Physio/configs/physio_v6_v2_clean.yaml"}),
    dict(name="physio_v8_abspos",
         src="Stage2_FlowMatching_Physio/configs/generate_physio_v8_abspos_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_FlowMatching_Physio/configs/physio_v8_abspos_v2_clean.yaml"}),

    # --- AutoregressiveFM ---
    dict(name="ar_fm_v2",
         src="Stage2_AutoregressiveFM/configs/generate_ar_fm_v2_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_AutoregressiveFM/configs/ar_fm_v2_v2_clean.yaml"}),
    dict(name="ar_fm_v3_abspos",
         src="Stage2_AutoregressiveFM/configs/generate_ar_fm_v3_abspos_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_AutoregressiveFM/configs/ar_fm_v3_abspos_v2_clean.yaml"}),

    # --- HierarchicalFM ---
    dict(name="hier_v1",
         src="Stage2_HierarchicalFM/configs/generate_hier_v1_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_HierarchicalFM/configs/hier_v1_v2_clean.yaml"}),

    # --- Wavefront ---
    dict(name="wfm_v1",
         src="Stage2_WavefrontFM/configs/generate_wfm_v1_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_WavefrontFM/configs/wfm_v1_v2_clean.yaml"}),

    # --- AneuCond / Branch ---
    dict(name="aneucond_v3_abspos",
         src="Stage2_AneuCondFM/configs/generate_aneucond_v3_abspos_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_AneuCondFM/configs/aneucond_v3_abspos_v2_clean.yaml"}),
    dict(name="branch_v2_abspos",
         src="Stage2_BranchFM/configs/generate_branch_v2_abspos_healthy.yaml",
         train_cfgs={"checkpoint": "Stage2_BranchFM/configs/branch_v2_abspos_v2_clean.yaml"}),

    # --- FlowMatching_TwoStage (chained-FM) ---
    dict(name="twostage_v2",
         src="Stage2_FlowMatching_TwoStage/configs/generate_twostage_v2_healthy.yaml",
         train_cfgs={
             "checkpoint_a": "Stage2_FlowMatching_TwoStage/configs/stage_a_v2_clean.yaml",
             "checkpoint_b": "Stage2_FlowMatching_TwoStage/configs/stage_b_v2_clean.yaml",
         }),

    # --- TwoStageFM ---
    dict(name="twostage_v1",
         src="Stage2_TwoStageFM/configs/generate_twostage_v1_healthy.yaml",
         train_cfgs={
             "checkpoint_a": "Stage2_TwoStageFM/configs/twostage_v1_a_v2_clean.yaml",
             "checkpoint_b": "Stage2_TwoStageFM/configs/twostage_v1_b_v2_clean.yaml",
         }),

    # --- HierFM_Strahler v2..v7 ---
    dict(name="hier_strahler_v2",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v2_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v2_v2_clean.yaml"}),
    dict(name="hier_strahler_v3_bifdepth_relpos",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v3_bifdepth_relpos_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v3_bifdepth_relpos_v2_clean.yaml"}),
    dict(name="hier_strahler_v4_bifdepth_abspos",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v4_bifdepth_abspos_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v4_bifdepth_abspos_v2_clean.yaml"}),
    dict(name="hier_strahler_v5_flat_relpos",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v5_flat_bifdepth_relpos_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v5_flat_bifdepth_relpos_v2_clean.yaml"}),
    dict(name="hier_strahler_v6_flat_abspos",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v6_flat_bifdepth_abspos_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v6_flat_bifdepth_abspos_v2_clean.yaml"}),
    dict(name="hier_strahler_v7_flat_abspos_nosc",
         src="Stage2_HierFM_Strahler/configs/generate_hier_v7_flat_bifdepth_abspos_nosc_healthy_test.yaml",
         train_cfgs={"checkpoint": "Stage2_HierFM_Strahler/configs/hier_v7_flat_bifdepth_abspos_nosc_v2_clean.yaml"}),

    # --- Latent Tree Diffusion ---
    dict(name="latent_v1",
         src="Stage2_LatentTreeDiffusion/configs/generate_latent_v1_healthy.yaml",
         train_cfgs={
             "ae_checkpoint": "Stage2_LatentTreeDiffusion/configs/ae_v1_v2_clean.yaml",
             "fm_checkpoint": "Stage2_LatentTreeDiffusion/configs/latent_fm_v1_v2_clean.yaml",
         }),
]


def load_yaml(p: Path) -> dict:
    with p.open() as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, data: dict, header: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def get_train_output_dir(train_cfg_path: Path) -> str:
    cfg = load_yaml(train_cfg_path)
    return cfg["paths"]["output_dir"]


def derive_pkg_dir(src_yaml: Path) -> Path:
    """Stage2_FlowMatching/configs/foo.yaml -> Stage2_FlowMatching/."""
    return src_yaml.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing *_v2_clean.yaml files.")
    args = ap.parse_args()

    n_made = 0
    n_skip = 0
    n_miss = 0
    for task in TASKS:
        name = task["name"]
        src = ROOT / task["src"]
        if not src.exists():
            print(f"  [MISS-src] {task['src']}")
            n_miss += 1
            continue

        train_paths = {k: ROOT / v for k, v in task["train_cfgs"].items()}
        missing_train = [str(v) for v in train_paths.values() if not v.exists()]
        if missing_train:
            for m in missing_train:
                print(f"  [MISS-train] {m}")
            n_miss += 1
            continue

        cfg = load_yaml(src)
        if "paths" not in cfg:
            cfg["paths"] = {}

        cfg["paths"]["val_dir"] = NEW_TEST

        # Track first 'model:' section from a training cfg — used to override
        # the healthy template's `model:` (which may target a different
        # architecture, e.g. v11b was a smaller variant than v11).
        train_model_section = None

        for ckpt_key, train_yaml in train_paths.items():
            train_cfg = load_yaml(train_yaml)
            train_out = train_cfg["paths"]["output_dir"]
            if ckpt_key == "fm_checkpoint":
                cfg["paths"][ckpt_key] = f"{train_out}/best_fm_model.pt"
            else:
                cfg["paths"][ckpt_key] = f"{train_out}/best_model.pt"
            # Use the *first* training cfg's model section. For two-stage
            # configs the generate template typically has model_a/model_b
            # which we leave alone; a single 'model:' from stage_b is wrong.
            if train_model_section is None and "model" in train_cfg:
                train_model_section = train_cfg["model"]
            # For two-stage configs: also patch model_a / model_b sections.
            if ckpt_key == "checkpoint_a" and "model_a" in cfg and "model" in train_cfg:
                cfg["model_a"] = train_cfg["model"]
            if ckpt_key == "checkpoint_b" and "model_b" in cfg and "model" in train_cfg:
                cfg["model_b"] = train_cfg["model"]

        # Override single 'model:' section with training cfg values. Preserve
        # generate-only knobs (depth_in_geometry, cfg_dropout) from template
        # only if NOT in the training cfg (training cfg is authoritative).
        if train_model_section is not None and "model" in cfg and "model_a" not in cfg:
            old_model = cfg.get("model", {}) or {}
            new_model = dict(train_model_section)
            for k in ("depth_in_geometry",):
                if k in old_model and k not in new_model:
                    new_model[k] = old_model[k]
            cfg["model"] = new_model

        pkg = derive_pkg_dir(src)
        out_dir = f"{pkg.name}/generated/{name}_v2_clean"
        if "params" not in cfg:
            cfg["params"] = {}
        cfg["params"]["output_dir"] = out_dir

        dst = pkg / "configs" / f"generate_{name}_v2_clean.yaml"
        if dst.exists() and not args.force:
            print(f"  [skip] {dst}")
            n_skip += 1
            continue

        header = (
            f"# Auto-generated from {src.name} by make_v2_clean_generate_configs.py\n"
            f"# v2_clean dataset variant (biffilter_v1).\n"
            f"# Checkpoints come from the corresponding *_v2_clean training runs.\n"
            f"# feature_stats / aneu_label_dir intentionally left at the original\n"
            f"# healthy paths — the v2_clean training used them as-is.\n"
        )
        write_yaml(dst, cfg, header)
        print(f"  [WROTE] {dst.relative_to(ROOT)}")
        n_made += 1

    print(f"\nDone. wrote={n_made} skip={n_skip} missing={n_miss}")


if __name__ == "__main__":
    main()
