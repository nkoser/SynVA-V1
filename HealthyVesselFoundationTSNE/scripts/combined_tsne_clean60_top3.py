"""Build two combined-t-SNE plots on the clean-60 subset:

  Plot 1 (original-GT):  GT raw mesh + 3 best models
  Plot 2 (pipelinegt):   GT through our recon pipeline + same 3 best models

The three "best" models (TreeGNN v15 combined, TreeGNN v14_phys 50 deg,
Physio v14 50 deg) are picked from the condensed paper section.

Inputs are the per-model UNI3D embeddings caches in
HealthyVesselFoundationTSNE/output/recon_dense_v2_short_clean60_*_per_model/
<model>/embeddings.npz. Each contains 60 GT + 60 generated rows; we
take GT once (from the first model) and the generated rows from each
of the three.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = Path("/workspace/SynVA_V1/HealthyVesselFoundationTSNE/output")

MODELS = [
    ("treegnn_v3_healthy_decap_dense_v2_short", "SynVA-V1$_{\\mathrm{depth}}$",   "#d62728", "o"),
    ("treegnn_v10_focalpos_aligned_short",      "SynVA-V1$_{\\mathrm{aligned}}$", "#2ca02c", "s"),
    ("treegnn_v1_healthy_decap_dense_v2_short", "SynVA-V1$_{\\mathrm{base}}$",    "#9467bd", "^"),
]
GT_LABEL  = "GT"
GT_COLOR  = "#1f4e79"
GT_MARKER = "D"

TSNE_KW = dict(perplexity=30, init="pca", learning_rate="auto",
               metric="euclidean", random_state=42, max_iter=1500)


def load_embeddings(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return dict(emb=d["embeddings"], kinds=d["kinds"], cases=d["case_ids"], labels=d["labels"])


def build_one(setup_dir: str, plot_filename: str, plot_title: str):
    base = ROOT / setup_dir
    print(f"\n[{setup_dir}]")

    # GT: take from the first model's npz (identical mesh source for all
    # models; small UNI3D non-determinism is negligible vs t-SNE jitter).
    first = load_embeddings(base / MODELS[0][0] / "embeddings.npz")
    gt_mask = first["kinds"] == "gt"
    gt_emb   = first["emb"][gt_mask]
    gt_cases = first["cases"][gt_mask]
    print(f"  GT rows: {gt_emb.shape}")

    blocks = [(GT_LABEL, GT_COLOR, GT_MARKER, gt_emb, gt_cases)]
    for slug, pretty, color, marker in MODELS:
        d = load_embeddings(base / slug / "embeddings.npz")
        m = d["kinds"] == "generated"
        emb_m, cases_m = d["emb"][m], d["cases"][m]
        # keep only cases that are also in GT, sorted to match GT order
        case_index = {c: i for i, c in enumerate(cases_m)}
        rows = [emb_m[case_index[c]] for c in gt_cases if c in case_index]
        emb_m = np.stack(rows, axis=0)
        cases_m = np.array([c for c in gt_cases if c in case_index])
        print(f"  {slug:50s}: gen rows={emb_m.shape}")
        blocks.append((pretty, color, marker, emb_m, cases_m))

    feat = np.concatenate([b[3] for b in blocks], axis=0).astype(np.float32)
    print(f"  combined: {feat.shape}; running t-SNE...")
    coords = TSNE(**TSNE_KW).fit_transform(feat)

    out_dir = base.parent / f"combined_top3_{setup_dir.split('/')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tsne_coordinates.csv"
    with csv_path.open("w") as fh:
        fh.write("source,case_id,tsne_x,tsne_y\n")
        cur = 0
        for label, _, _, emb, cases in blocks:
            for i, c in enumerate(cases):
                x, y = coords[cur + i]
                fh.write(f"{label},{c},{x:.6f},{y:.6f}\n")
            cur += emb.shape[0]
    print(f"  saved {csv_path}")

    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=200)
    cur = 0
    for label, color, marker, emb, _ in blocks:
        n = emb.shape[0]
        sub = coords[cur:cur + n]
        ax.scatter(sub[:, 0], sub[:, 1],
                   s=44, c=color, marker=marker,
                   linewidths=0.6, edgecolors="white",
                   alpha=0.85, label=f"{label} (n={n})", zorder=2)
        cur += n
    ax.set_title(plot_title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, ls=":", alpha=0.4, zorder=0)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_png = out_dir / plot_filename
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    return out_png


if __name__ == "__main__":
    p1 = build_one(
        "recon_dense_v2_short_clean60_uni3d_per_model",
        "combined_tsne_clean60_origGT_top3.png",
        "UNI3D t-SNE on clean-60: original GT mesh vs. 3 best models",
    )
    p2 = build_one(
        "recon_dense_v2_short_clean60_pipelinegt_uni3d_per_model",
        "combined_tsne_clean60_pipelineGT_top3.png",
        "UNI3D t-SNE on clean-60: GT through our recon pipeline vs. 3 best models",
    )
    print("\nDONE")
    print("Plot 1:", p1)
    print("Plot 2:", p2)
