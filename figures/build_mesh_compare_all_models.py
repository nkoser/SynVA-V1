"""Comprehensive mesh-comparison figure showing ALL 14 models on the
cleanest trees from the clean-60 subset.

Layout: rows = example trees (sorted by avg chamfer over the 14 models
ascending = "cleanest looking"), columns = GT + 14 models.
"""
from pathlib import Path
import numpy as np
import csv

import trimesh
import matplotlib.pyplot as plt

ROOT = Path("/workspace/SynVA_V1")
RECON_BASE = ROOT / "output/recon_healthy_decap_dense_v2_short"
GT_PATHS = [
    Path("/data/healthy_vessel_decapped"),
    Path("/data/healthy_vessel_decapped_staging_test_extra"),
]
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
MM_BASE = ROOT / "HealthyVesselMeshMetrics/output/mesh_metrics_clean60_per_model"

# Ordered display of all 14 models (GT first, then plain TreeGNN, focalpos,
# aligned, anticurl, phys-120, phys-50, combined, Physio Transformer).
MODELS = [
    ("v1",                 "treegnn_v1_healthy_decap_dense_v2_short"),
    ("v3",                 "treegnn_v3_healthy_decap_dense_v2_short"),
    ("v4",                 "treegnn_v4_healthy_decap_dense_v2_short"),
    ("v7 fp$\\lambda{=}2$", "treegnn_v7_focalpos_short"),
    ("v8 fp$\\lambda{=}4$", "treegnn_v8_focalpos_aggressive_short"),
    ("v9 fp+cpbif",        "treegnn_v9_focalpos_cpbif_short"),
    ("v10 al60°",          "treegnn_v10_focalpos_aligned_short"),
    ("v11 al45°",          "treegnn_v11_focalpos_aligned2_short"),
    ("v12 anticurl",       "treegnn_v12_anticurlback_short"),
    ("v5 phys 120°",       "treegnn_v5_physio_healthy_decap_dense_v2_short"),
    ("v6 phys nw 120°",    "treegnn_v6_physio_nowarp_healthy_decap_dense_v2_short"),
    ("v14_phys 50°",       "treegnn_v14_phys_targetangle50_short"),
    ("v15 combined",       "treegnn_v15_combined_short"),
    ("Physio v14 50°",     "physio_v14_targetangle50_short"),
]

N_POINTS = 25000
ELEV_DEG = 22.0
AZIM_DEG = 35.0
N_EXAMPLES = 8        # number of cleanest trees to show
PANEL_SIZE = 2.0      # inches per panel side


def find_gt(uid):
    for root in GT_PATHS:
        for cand in (root / uid / f"{uid}.obj", root / uid / "01_mesh" / f"{uid}.obj"):
            if cand.exists():
                return cand
    return None


def find_pred(uid, slug):
    p = RECON_BASE / slug / f"{uid}.obj"
    return p if p.exists() else None


def rot_mat(elev_deg, azim_deg):
    e, a = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rx @ Rz


def render_obj(path, base_color):
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception:
        return None
    if mesh.is_empty or mesh.area == 0:
        return None
    pts, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS)
    normals = mesh.face_normals[face_idx]
    R = rot_mat(ELEV_DEG, AZIM_DEG)
    pts_r = pts @ R.T
    normals_r = normals @ R.T
    light_dir = np.array([0.2, 0.2, 1.0])
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.clip(normals_r @ light_dir, 0, None)
    shade = 0.25 + 0.75 * diffuse
    rgb = np.clip(np.outer(shade, np.array(base_color)), 0, 1)
    z = pts_r[:, 2]
    order = np.argsort(z)
    return pts_r[order, :2], rgb[order]


def draw_panel(ax, payload):
    if payload is None:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center", fontsize=8)
    else:
        xy, rgb = payload
        ax.scatter(xy[:, 0], xy[:, 1], c=rgb, s=1.0, marker=".", edgecolors="none", rasterized=True)
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---- rank trees by avg chamfer across all 14 models ----
clean60 = {l.strip() for l in (ROOT / "clean_test_subset_60.txt").read_text().splitlines() if l.strip()}
chamfer = {}  # uid -> {slug: chamfer}
for label, slug in MODELS:
    pair_csv = MM_BASE / slug / "per_pair_metrics.csv"
    if not pair_csv.exists():
        continue
    with open(pair_csv) as f:
        for r in csv.DictReader(f):
            uid = r.get("case_id")
            if uid not in clean60:
                continue
            try:
                chamfer.setdefault(uid, {})[slug] = float(r["chamfer_mean"])
            except (ValueError, TypeError):
                pass

ranked = []
slugs_all = {s for _, s in MODELS}
for uid, vals in chamfer.items():
    if slugs_all.issubset(vals.keys()):  # all 14 present
        ranked.append((uid, sum(vals.values()) / len(vals)))
ranked.sort(key=lambda x: x[1])
EXAMPLES = [u for u, _ in ranked[:N_EXAMPLES]]
print("Selected trees (lowest avg chamfer over all 14 models):")
for u, ch in ranked[:N_EXAMPLES]:
    print(f"  {ch:.4f}  {u}")

# ---- render ----
nrows = len(EXAMPLES)
ncols = 1 + len(MODELS)
print(f"\nrendering {nrows * ncols} panels ...")
panels = {}
for i, uid in enumerate(EXAMPLES):
    gt = find_gt(uid)
    panels[(i, 0)] = render_obj(gt, base_color=(0.32, 0.5, 0.7)) if gt else None
    for j, (label, slug) in enumerate(MODELS):
        p = find_pred(uid, slug)
        panels[(i, j + 1)] = render_obj(p, base_color=(0.95, 0.55, 0.18)) if p else None
    print(f"  row {i}: {uid} done")

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(PANEL_SIZE * ncols, PANEL_SIZE * nrows),
)
if nrows == 1:
    axes = np.atleast_2d(axes)

col_titles = ["GT"] + [m[0] for m in MODELS]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=9)
for i, uid in enumerate(EXAMPLES):
    axes[i, 0].set_ylabel(uid, fontsize=7, rotation=90, labelpad=8)
    for j in range(ncols):
        draw_panel(axes[i, j], panels.get((i, j)))

plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)

png = OUT_DIR / "mesh_compare_clean60_all_models.png"
pdf = OUT_DIR / "mesh_compare_clean60_all_models.pdf"
plt.savefig(png, dpi=180, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {png}")
print(f"wrote {pdf}")
