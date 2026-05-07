"""Compose the pipeline-GT UNI3D t-SNE per-model plots into one grid.

Pipeline-GT means: the GT meshes have been reconstructed through the
same spline->Poisson pipeline as the predictions, so the embedding does
not pick up a recon-style bias.
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path("/workspace/SynVA_V1")
TSNE_BASE = ROOT / "HealthyVesselFoundationTSNE/output/recon_dense_v2_short_clean60_pipelinegt_uni3d_per_model"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

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
    p = TSNE_BASE / slug / "uni3d_tsne_clean60_pipelinegt.png"
    if not p.exists():
        print(f"[WARN] missing {p}")
        continue
    panels.append((label, p))

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
    "UNI3D t-SNE (pipeline-GT): clean-60 pipeline-recon GT (blue) vs generated (orange)",
    fontsize=14, y=1.0,
)
plt.tight_layout(rect=(0, 0, 1, 0.99))

png = OUT_DIR / "tsne_grid_clean60_pipelinegt.png"
pdf = OUT_DIR / "tsne_grid_clean60_pipelinegt.pdf"
plt.savefig(png, dpi=180, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
print(f"wrote {png}")
print(f"wrote {pdf}")
