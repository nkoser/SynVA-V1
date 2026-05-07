"""Paper-style mesh comparison figures.

Three figures, each with up to 5 examples (columns):
  rows = GT, Physio v14 50°, TreeGNN v15 combined,
         TreeGNN v14_phys 50°, TreeGNN v3 short

Rendering: actual mesh triangles painted in matplotlib with Lambert
shading (mesh #8b0000) and an optional thin wireframe overlay
(#450000). Two variants written per set: solid and with-wireframe.

Outputs in figures/:
  mesh_compare_paper_setA[_wf].{png,pdf}
  mesh_compare_paper_setB[_wf].{png,pdf}
  mesh_compare_paper_setC[_wf].{png,pdf}
"""
from pathlib import Path
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
OUT_DIR.mkdir(exist_ok=True)

# Rows: GT first, then the four condensed models from the paper.
ROWS = [
    ("GT",                None),
    ("Physio v14 50°",    "physio_v14_targetangle50_short"),
    ("v15 combined",      "treegnn_v15_combined_short"),
    ("v14_phys 50°",      "treegnn_v14_phys_targetangle50_short"),
    ("v3",                "treegnn_v3_healthy_decap_dense_v2_short"),
]

SETS = {
    "A": [
        "intra_AN163-2",
        "aneux_p514_EwAfEREGEx8SCBYOHAANBBgI_RICA",
        "cmha_AHMU1218027",
        "cmha_AHMU1218064",
        "intra_AN182-1",
    ],
    "B": [
        "aneux_ANSYS_UNIGE_33_628",
        "aneux_UPF_P0156.00_ID1",
        "aneux_UPF_P0207.01_ID1",
        "aneux_USFD_0036",
        "aneux_USFD_0011",
    ],
    "C": [
        "aneux_UPF_P0165.00_ID1",
        "aneux_UPF_P0195.00_ID1",
        "intra_AN55",
        "aneux_p530_AAgcCR83HgQaEQMRGw8GCioU_LICA",
    ],
}

ELEV_DEG = 22.0
AZIM_DEG = 35.0
PANEL_SIZE = 2.4
MESH_HEX = "#8b0000"
WIRE_HEX = "#450000"
WIRE_LW = 0.08      # very thin
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
    """Return (polys (F,3,2), face_colors (F,3), bbox (xmin,xmax,ymin,ymax))."""
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

    # painter's algorithm: sort faces back-to-front by mean z
    tri_z = verts_r[mesh.faces][..., 2].mean(axis=1)
    order = np.argsort(tri_z)

    light = np.array([0.2, 0.2, 1.0])
    light /= np.linalg.norm(light)
    diffuse = np.clip(fnorm_r @ light, 0, None)
    shade = 0.30 + 0.70 * diffuse
    face_rgb = np.clip(shade[:, None] * MESH_RGB[None, :], 0, 1)

    polys = verts_r[mesh.faces][..., :2]
    polys = polys[order]
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
        dx = xmax - xmin
        dy = ymax - ymin
        pad = PAD_FRAC * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def short_label(uid):
    """Compact tree label for display."""
    s = uid
    for p in ("aneux_", "cmha_", "intra_"):
        if s.startswith(p):
            s = s[len(p):]
            break
    if "_" in s and len(s) > 20:
        s = s.split("_")[0] + "…"
    return s


def build_set(name, uids, wireframe):
    nrows = len(ROWS)
    ncols = len(uids)
    print(f"\n=== set {name} ({'wf' if wireframe else 'solid'}): "
          f"{nrows}r x {ncols}c ===")

    panels = {}
    for j, uid in enumerate(uids):
        for i, (label, slug) in enumerate(ROWS):
            path = find_gt(uid) if slug is None else find_pred(uid, slug)
            if path is None:
                print(f"  [miss] row={label} col={uid}")
                panels[(i, j)] = None
            else:
                panels[(i, j)] = render_mesh(path)
        print(f"  col {j+1}/{ncols}: {uid}")

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(PANEL_SIZE * ncols, PANEL_SIZE * nrows),
    )
    if ncols == 1:
        axes = axes[:, None]
    if nrows == 1:
        axes = axes[None, :]

    for j, uid in enumerate(uids):
        axes[0, j].set_title(short_label(uid), fontsize=9)
    for i, (label, _) in enumerate(ROWS):
        axes[i, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=8)
        for j in range(ncols):
            draw_panel(axes[i, j], panels.get((i, j)), wireframe=wireframe)

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    suffix = "_wf" if wireframe else ""
    stem = f"mesh_compare_paper_set{name}{suffix}"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


for name, uids in SETS.items():
    for wf in (False, True):
        build_set(name, uids, wireframe=wf)
