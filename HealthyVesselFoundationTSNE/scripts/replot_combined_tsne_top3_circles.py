"""Re-render the two combined-top3 t-SNE plots from the saved
tsne_coordinates.csv: circles only (no shape encoding), color per source,
and visible t-SNE axes with ticks/labels."""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/workspace/SynVA_V1/HealthyVesselFoundationTSNE/output")

# Match the colours used in combined_tsne_clean60_top3.py.
SOURCE_STYLE = {
    "GT":                            "#666666",
    "SynVA-V1$_{\\mathrm{depth}}$":   "#B85450",
    "SynVA-V1$_{\\mathrm{aligned}}$": "#82B366",
    "SynVA-V1$_{\\mathrm{base}}$":    "#D79B00",
}
DISPLAY_LABEL = {k: k for k in SOURCE_STYLE}
ORDER = list(SOURCE_STYLE.keys())

PLOTS = [
    ("combined_top3_recon_dense_v2_short_clean60_uni3d_per_model",
     "combined_tsne_clean60_origGT_top3.png",
     "UNI3D t-SNE on clean-60: original GT mesh vs. 3 best models"),
    ("combined_top3_recon_dense_v2_short_clean60_pipelinegt_uni3d_per_model",
     "combined_tsne_clean60_pipelineGT_top3.png",
     "UNI3D t-SNE on clean-60: GT through our recon pipeline vs. 3 best models"),
]


def load_csv(path: Path):
    pts = {k: [] for k in ORDER}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            src = r["source"]
            if src not in pts:
                # tolerate minor differences in label formatting
                continue
            pts[src].append((float(r["tsne_x"]), float(r["tsne_y"])))
    return {k: np.asarray(v) for k, v in pts.items() if v}


def replot(setup_dir: str, fname: str, title: str):
    csv_path = ROOT / setup_dir / "tsne_coordinates.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} missing")
        return
    pts = load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=200)
    for label in ORDER:
        if label not in pts:
            continue
        xy = pts[label]
        ax.scatter(xy[:, 0], xy[:, 1],
                   s=44, c=SOURCE_STYLE[label], marker="o",
                   linewidths=0.6, edgecolors="white",
                   alpha=0.85, label=DISPLAY_LABEL[label], zorder=2)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, ls=":", alpha=0.4, zorder=0)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_png = ROOT / setup_dir / fname
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


for setup, fname, title in PLOTS:
    replot(setup, fname, title)
