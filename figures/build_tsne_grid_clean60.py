"""Compose all UNI3D t-SNE per-model plots (clean-60) into one grid figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

ROOT = Path("/workspace/SynVA_V1")
TSNE_BASE = ROOT / "HealthyVesselFoundationTSNE/output/recon_dense_v2_short_clean60_uni3d_per_model"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Stable display order: plain TreeGNN baselines, focal-pos, aligned, anticurl,
# physio (120°), then the three new models last.
ORDER = [
    ("treegnn_v1_healthy_decap_dense_v2_short", "TreeGNN v1"),
    ("treegnn_v3_healthy_decap_dense_v2_short", "TreeGNN v3"),
    ("treegnn_v4_healthy_decap_dense_v2_short", "TreeGNN v4"),
    ("treegnn_v7_focalpos_short", "TreeGNN v7 (focalpos λ=2)"),
    ("treegnn_v8_focalpos_aggressive_short", "TreeGNN v8 (focalpos λ=4)"),
    ("treegnn_v9_focalpos_cpbif_short", "TreeGNN v9 (focalpos+cpbif)"),
    ("treegnn_v10_focalpos_aligned_short", "TreeGNN v10 (aligned60°)"),
    ("treegnn_v11_focalpos_aligned2_short", "TreeGNN v11 (aligned45°)"),
    ("treegnn_v12_anticurlback_short", "TreeGNN v12 (anticurlback)"),
    ("treegnn_v5_physio_healthy_decap_dense_v2_short", "TreeGNN v5 phys (120°)"),
    ("treegnn_v6_physio_nowarp_healthy_decap_dense_v2_short", "TreeGNN v6 phys nowarp (120°)"),
    ("treegnn_v14_phys_targetangle50_short", "TreeGNN v14 phys (50°)"),
    ("treegnn_v15_combined_short", "TreeGNN v15 combined"),
    ("physio_v14_targetangle50_short", "Physio v14 (50°)"),
]

panels = []
for slug, label in ORDER:
    img_path = TSNE_BASE / slug / "uni3d_tsne_clean60.png"
    if not img_path.exists():
        print(f"[WARN] missing {img_path}")
        continue
    panels.append((label, img_path))

n = len(panels)
ncols = 4
nrows = math.ceil(n / ncols)
print(f"composing {n} panels into {nrows}x{ncols} grid")

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.ravel() if nrows * ncols > 1 else [axes]

for i, (label, path) in enumerate(panels):
    ax = axes[i]
    ax.imshow(mpimg.imread(path))
    ax.set_title(label, fontsize=11)
    ax.axis("off")
for j in range(n, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "UNI3D t-SNE: clean-60 GT (blue) vs generated (orange) — per model",
    fontsize=14, y=1.0,
)
plt.tight_layout(rect=(0, 0, 1, 0.99))

png_out = OUT_DIR / "tsne_grid_clean60.png"
pdf_out = OUT_DIR / "tsne_grid_clean60.pdf"
plt.savefig(png_out, dpi=180, bbox_inches="tight")
plt.savefig(pdf_out, bbox_inches="tight")
print(f"wrote {png_out}")
print(f"wrote {pdf_out}")
