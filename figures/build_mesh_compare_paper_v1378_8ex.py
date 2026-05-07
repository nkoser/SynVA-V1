"""Paper figure: GT + v1, v3, v7, v8 across 8 cleanest examples.

Examples are picked as the 8 trees with lowest avg chamfer (mean over
v1, v3, v7, v8) on the clean-60 subset.

Layout: 5 rows (GT, v1, v3, v7, v8), 8 cols (examples).
Colours: mesh #8b0000, wireframe (optional) #450000.
"""
from pathlib import Path
import csv
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path("/workspace/SynVA_V1")
RECON_BASE = ROOT / "output/recon_healthy_decap_dense_v2_short"
GT_PATHS = [
    Path("/data/healthy_vessel_decapped"),
    Path("/data/healthy_vessel_decapped_staging_test_extra"),
]
OUT_DIR = ROOT / "figures"
MM_BASE = ROOT / "HealthyVesselMeshMetrics/output/mesh_metrics_clean60_per_model"

ROWS = [
    ("GT", None),
    ("v1", "treegnn_v1_healthy_decap_dense_v2_short"),
    ("v3", "treegnn_v3_healthy_decap_dense_v2_short"),
    ("v7 fp$\\lambda{=}2$", "treegnn_v7_focalpos_short"),
    ("v8 fp$\\lambda{=}4$", "treegnn_v8_focalpos_aggressive_short"),
]

N_EXAMPLES = 8
ELEV_DEG = 22.0
AZIM_DEG = 35.0
PANEL_SIZE = 2.2
MESH_HEX = "#8b0000"
WIRE_HEX = "#450000"
WIRE_LW = 0.08
PAD_FRAC = 0.04


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


MESH_RGB = hex_to_rgb(MESH_HEX)


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


def render_mesh(path):
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as e:
        print(f"    [load fail] {path}: {e}")
        return None
    if mesh.is_empty or len(mesh.faces) == 0:
        return None
    R = rot_mat(ELEV_DEG, AZIM_DEG)
    verts_r = mesh.vertices @ R.T
    fnorm_r = mesh.face_normals @ R.T
    tri_z = verts_r[mesh.faces][..., 2].mean(axis=1)
    order = np.argsort(tri_z)
    light = np.array([0.2, 0.2, 1.0])
    light /= np.linalg.norm(light)
    diffuse = np.clip(fnorm_r @ light, 0, None)
    shade = 0.30 + 0.70 * diffuse
    face_rgb = np.clip(shade[:, None] * MESH_RGB[None, :], 0, 1)
    polys = verts_r[mesh.faces][..., :2][order]
    face_rgb = face_rgb[order]
    xmin, ymin = polys.reshape(-1, 2).min(axis=0)
    xmax, ymax = polys.reshape(-1, 2).max(axis=0)
    return polys, face_rgb, (xmin, xmax, ymin, ymax)


def draw_panel(ax, payload, *, wireframe):
    if payload is None:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color="gray")
    else:
        polys, face_rgb, (xmin, xmax, ymin, ymax) = payload
        ec = WIRE_HEX if wireframe else "none"
        lw = WIRE_LW if wireframe else 0.0
        pc = PolyCollection(polys, facecolors=face_rgb, edgecolors=ec,
                            linewidths=lw, antialiased=True, rasterized=True)
        ax.add_collection(pc)
        dx, dy = xmax - xmin, ymax - ymin
        pad = PAD_FRAC * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def short_label(uid):
    s = uid
    for p in ("aneux_", "cmha_", "intra_"):
        if s.startswith(p):
            s = s[len(p):]
            break
    if "_" in s and len(s) > 18:
        s = s.split("_")[0] + "…"
    return s


# ---- pick 8 cleanest UIDs by avg chamfer over the 4 selected models ----
clean60 = {l.strip() for l in (ROOT / "clean_test_subset_60.txt").read_text().splitlines() if l.strip()}
slugs = [slug for _, slug in ROWS if slug is not None]
chamfer = {}  # uid -> {slug: ch}
for slug in slugs:
    pair_csv = MM_BASE / slug / "per_pair_metrics.csv"
    if not pair_csv.exists():
        print(f"[warn] missing metrics for {slug}")
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
for uid, vals in chamfer.items():
    if set(slugs).issubset(vals.keys()):
        ranked.append((uid, sum(vals.values()) / len(vals)))
ranked.sort(key=lambda x: x[1])
EXAMPLES = [u for u, _ in ranked[:N_EXAMPLES]]
print(f"Picked {len(EXAMPLES)} cleanest trees (avg chamfer over v1/v3/v7/v8):")
for u, ch in ranked[:N_EXAMPLES]:
    print(f"  {ch:.4f}  {u}")


def build(wireframe):
    nrows, ncols = len(ROWS), len(EXAMPLES)
    print(f"\n=== rendering {'wf' if wireframe else 'solid'}: {nrows}r x {ncols}c ===")
    panels = {}
    for j, uid in enumerate(EXAMPLES):
        for i, (label, slug) in enumerate(ROWS):
            path = find_gt(uid) if slug is None else find_pred(uid, slug)
            panels[(i, j)] = render_mesh(path) if path else None
        print(f"  col {j+1}/{ncols}: {uid}")

    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_SIZE * ncols, PANEL_SIZE * nrows))
    for j, uid in enumerate(EXAMPLES):
        axes[0, j].set_title(short_label(uid), fontsize=9)
    for i, (label, _) in enumerate(ROWS):
        axes[i, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=8)
        for j in range(ncols):
            draw_panel(axes[i, j], panels.get((i, j)), wireframe=wireframe)

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    suffix = "_wf" if wireframe else ""
    stem = f"mesh_compare_paper_v1378_8ex{suffix}"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


for wf in (False, True):
    build(wireframe=wf)
