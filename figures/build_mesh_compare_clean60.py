"""Render paper-style mesh-comparison figures.

Generates several variants on the clean-60 subset:
  - mesh_compare_clean60.png         — best small/intra examples (default)
  - mesh_compare_clean60_big.png     — four largest GT vessel structures
  - mesh_compare_clean60_mixed.png   — mix of large + medium

Each figure: rows = example trees, columns = (GT, Physio v14 50°,
v15 combined, v14_phys 50°, v3 short).  Pure-matplotlib rendering
(Lambert-shaded surface point cloud, no OpenGL/X required).
"""
from pathlib import Path
import numpy as np

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

MODELS = [
    ("Physio v14 50°", "physio_v14_targetangle50_short"),
    ("v15 combined", "treegnn_v15_combined_short"),
    ("v14_phys 50°", "treegnn_v14_phys_targetangle50_short"),
    ("v3 short", "treegnn_v3_healthy_decap_dense_v2_short"),
]

VARIANTS = {
    "default": [
        "intra_AN196-2",
        "intra_AN163-2",
        "aneux_UPF_P0156.00_ID1",
        "cmha_AHMU1218027",
    ],
    "big": [
        "aneux_USFD_0002",
        "aneux_USFD_0011",
        "aneux_USFD_0036",
        "aneux_p529_HRQcEhgWBw8ADQsaHBIGCAAW_LICA",
    ],
    "mixed": [
        "aneux_USFD_0002",
        "aneux_p514_EwAfEREGEx8SCBYOHAANBBgI_RICA",
        "aneux_p538_EgAAAxUSExUNBBQNFxsPERcZ_LICA",
        "aneux_UPF_P0019.00_ID1",
    ],
}

N_POINTS = 30000
ELEV_DEG = 22.0
AZIM_DEG = 35.0


def find_gt_obj(uid: str):
    for root in GT_PATHS:
        p1 = root / uid / f"{uid}.obj"
        if p1.exists():
            return p1
        p2 = root / uid / "01_mesh" / f"{uid}.obj"
        if p2.exists():
            return p2
    return None


def find_pred_obj(uid: str, slug: str):
    p = RECON_BASE / slug / f"{uid}.obj"
    return p if p.exists() else None


def rot_mat(elev_deg, azim_deg):
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rx @ Rz


def render_obj(path: Path, base_color):
    mesh = trimesh.load(str(path), force="mesh", process=False)
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
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    else:
        xy, rgb = payload
        ax.scatter(xy[:, 0], xy[:, 1], c=rgb, s=1.5, marker=".", edgecolors="none", rasterized=True)
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def build_figure(examples, out_stem):
    nrows = len(examples)
    ncols = 1 + len(MODELS)
    print(f"\n=== building {out_stem}: {nrows} examples × {ncols} cols ===")

    panels = {}
    for i, uid in enumerate(examples):
        gt = find_gt_obj(uid)
        if gt is None:
            print(f"  [warn] no GT for {uid}")
            panels[(i, 0)] = None
        else:
            print(f"  GT  {uid}  ({gt.stat().st_size//1024}KB)")
            panels[(i, 0)] = render_obj(gt, base_color=(0.32, 0.5, 0.7))
        for j, (label, slug) in enumerate(MODELS):
            p = find_pred_obj(uid, slug)
            if p is None:
                print(f"  [skip] {label} {uid}")
                panels[(i, j + 1)] = None
            else:
                panels[(i, j + 1)] = render_obj(p, base_color=(0.95, 0.55, 0.18))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    if nrows == 1:
        axes = np.atleast_2d(axes)
    col_titles = ["Ground truth"] + [m[0] for m in MODELS]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12)
    for i, uid in enumerate(examples):
        axes[i, 0].set_ylabel(uid, fontsize=9, rotation=90, labelpad=12)
        for j in range(ncols):
            draw_panel(axes[i, j], panels.get((i, j)))

    plt.tight_layout(pad=0.4)
    png = OUT_DIR / f"{out_stem}.png"
    pdf = OUT_DIR / f"{out_stem}.pdf"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


for name, ex in VARIANTS.items():
    suffix = "" if name == "default" else f"_{name}"
    build_figure(ex, f"mesh_compare_clean60{suffix}")
