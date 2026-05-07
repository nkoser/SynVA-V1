"""
Run evaluate_all on the *_v2_clean Stage2 models against the biffilter_v1
TreesSplines healthy_vessel TEST split (paired GT).

Includes every model that has a generate_*_v2_clean.yaml config (and thus a
matching generated/<name>_v2_clean/npy/ directory).

Usage:
    python evaluate_healthy_v2_clean.py [--output_dir evaluation_results_healthy_all_v2_clean]
"""
import argparse
import sys

import evaluate_all


evaluate_all.MODELS = {
    # --- Flow Matching baselines ---
    "FM v11 v2_clean":                "Stage2_FlowMatching/generated/flow_v11_v2_clean/npy",
    "FM v12 abspos v2_clean":         "Stage2_FlowMatching/generated/flow_v12_abspos_v2_clean/npy",

    # --- Physio FM ---
    "Physio v5 v2_clean":             "Stage2_FlowMatching_Physio/generated/physio_v5_v2_clean/npy",
    "Physio v6 v2_clean":             "Stage2_FlowMatching_Physio/generated/physio_v6_v2_clean/npy",
    "Physio v8 abspos v2_clean":      "Stage2_FlowMatching_Physio/generated/physio_v8_abspos_v2_clean/npy",
    # --- Physio target-realignment series (no Murray, GT-aligned constraints) ---
    "Physio v9 v2_clean":             "Stage2_FlowMatching_Physio/generated/physio_v9_v2_clean/npy",
    "Physio v10 v2_clean":            "Stage2_FlowMatching_Physio/generated/physio_v10_v2_clean/npy",
    "Physio v11 v2_clean":            "Stage2_FlowMatching_Physio/generated/physio_v11_v2_clean/npy",

    # --- TreeGNN FM ---
    "TreeGNN v1 v2_clean":            "Stage2_FlowMatching_TreeGNN/generated/treegnn_v1_v2_clean/npy",
    "TreeGNN v2 abspos v2_clean":     "Stage2_FlowMatching_TreeGNN/generated/treegnn_v2_abspos_v2_clean/npy",
    "TreeGNN v3 physio v2_clean":     "Stage2_FlowMatching_TreeGNN_v2/generated/treegnn_v3_physio_v2_clean/npy",
    "TreeGNN v4 physio_widt v2_clean":"Stage2_FlowMatching_TreeGNN_v2/generated/treegnn_v4_physio_v2_clean/npy",

    # --- TwoStage FM (chained-FM) ---
    "FM-TwoStage v2 v2_clean":        "Stage2_FlowMatching_TwoStage/generated/twostage_v2_v2_clean/npy",

    # --- TwoStageFM (separate stage-A + stage-B) ---
    "TwoStageFM v1 v2_clean":         "Stage2_TwoStageFM/generated/twostage_v1_v2_clean/npy",

    # --- Autoregressive FM ---
    "AR FM v2 v2_clean":              "Stage2_AutoregressiveFM/generated/ar_fm_v2_v2_clean/npy",
    "AR FM v3 abspos v2_clean":       "Stage2_AutoregressiveFM/generated/ar_fm_v3_abspos_v2_clean/npy",

    # --- Hierarchical FM ---
    "HierarchicalFM v1 v2_clean":     "Stage2_HierarchicalFM/generated/hier_v1_v2_clean/npy",

    # --- Wavefront FM ---
    "WFM v1 v2_clean":                "Stage2_WavefrontFM/generated/wfm_v1_v2_clean/npy",

    # --- Latent Tree Diffusion ---
    "Latent v1 v2_clean":             "Stage2_LatentTreeDiffusion/generated/latent_v1_v2_clean/npy",

    # --- Conditional / specialized FMs ---
    "AneuCond v3 abspos v2_clean":    "Stage2_AneuCondFM/generated/aneucond_v3_abspos_v2_clean/npy",
    "Branch v2 abspos v2_clean":      "Stage2_BranchFM/generated/branch_v2_abspos_v2_clean/npy",

    # --- HierFM-Strahler ablations (v2..v7) ---
    "HierFM-Strahler v2 v2_clean":                "Stage2_HierFM_Strahler/generated/hier_strahler_v2_v2_clean/npy",
    "HierFM-Strahler v3 bifdepth+rel v2_clean":   "Stage2_HierFM_Strahler/generated/hier_strahler_v3_bifdepth_relpos_v2_clean/npy",
    "HierFM-Strahler v4 bifdepth+abs v2_clean":   "Stage2_HierFM_Strahler/generated/hier_strahler_v4_bifdepth_abspos_v2_clean/npy",
    "HierFM-Strahler v5 flat+rel v2_clean":       "Stage2_HierFM_Strahler/generated/hier_strahler_v5_flat_relpos_v2_clean/npy",
    "HierFM-Strahler v6 flat+abs v2_clean":       "Stage2_HierFM_Strahler/generated/hier_strahler_v6_flat_abspos_v2_clean/npy",
    "HierFM-Strahler v7 flat+abs+nosc v2_clean":  "Stage2_HierFM_Strahler/generated/hier_strahler_v7_flat_abspos_nosc_v2_clean/npy",
}


# Ground-truth: biffilter_v1 healthy test split (the same split that
# generation was paired against).
evaluate_all.DEFAULT_GT_VAL_DIR = (
    "derived_data/TreesSplines_k_count_100depth_prepared_biffilter_v1_norm_relpos_nodecp_v1/test"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results_healthy_all_v2_clean",
        help="Where to write the evaluation summary tables and per-tree CSVs.",
    )
    parser.add_argument("--max_gt", type=int, default=None)
    args = parser.parse_args()

    sys.argv = [
        "evaluate_all.py",
        "--gt_val_dir", evaluate_all.DEFAULT_GT_VAL_DIR,
        "--output_dir", args.output_dir,
    ]
    if args.max_gt is not None:
        sys.argv += ["--max_gt", str(args.max_gt)]

    evaluate_all.main()
