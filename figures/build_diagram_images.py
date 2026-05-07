"""Render the four images embedded in the TreeGNN pipeline draw.io figure.

Style:
  - Input mesh / output mesh: original .obj rendered as a solid Lambert
    surface in DARK_RED (#8b0000) with a thin DARKER_RED (#450000)
    wireframe overlay.
  - Spline-tree visualisations (encoding & decoding): use the EXACT
    helpers from Stage2_FlowMatching/generate_trees.py
    (draw_tree_centerlines + draw_tree_splines_scatter) so they match
    the per-tree PNGs that are emitted at generation time.
"""
import sys
from pathlib import Path
import numpy as np

import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path("/workspace/SynVA_V1")
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

UID = "aneux_C0003"

GT_MESH  = Path("/data/healthy_vessel_decapped") / UID / f"{UID}.obj"
GEN_MESH = ROOT / "output/recon_healthy_decap_dense_v2_short/treegnn_v15_combined_short" / f"{UID}.obj"
GT_NPY   = ROOT / "Stage2_FlowMatching_TreeGNN/generated/treegnn_v1_healthy_decap_dense_v2_short/npy" / f"{UID}_gt.npy"
GEN_NPY  = ROOT / "Stage2_FlowMatching_TreeGNN_v2/generated/treegnn_v15_combined_short/npy" / f"{UID}.npy"

ELEV_DEG = 22.0
AZIM_DEG = 35.0

DARK_RED   = "#8b0000"
DARKER_RED = "#450000"

# Allow importing draw helpers from Stage2_FlowMatching/generate_trees.py
sys.path.insert(0, str(ROOT))
from Stage2_FlowMatching.generate_trees import (
    data_to_tree, draw_tree_centerlines, draw_tree_splines_scatter,
)


def rot_mat(elev_deg, azim_deg):
    e, a = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rx @ Rz


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


# ─────────────────────────────────────────────────────────────────────
# Mesh — solid Lambert + thin wireframe
# ─────────────────────────────────────────────────────────────────────
def render_mesh_solid_with_wire(path, fill_hex=DARK_RED, wire_hex=DARKER_RED,
                                 wire_alpha=0.18, wire_lw=0.10, decimate_to=15000):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if decimate_to is not None and len(mesh.faces) > decimate_to:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=decimate_to)
        except Exception:
            pass

    R = rot_mat(ELEV_DEG, AZIM_DEG)
    verts = np.asarray(mesh.vertices) @ R.T
    faces = np.asarray(mesh.faces)
    fnormals = np.asarray(mesh.face_normals) @ R.T

    light = np.array([0.2, 0.2, 1.0]); light /= np.linalg.norm(light)
    diffuse = np.clip(fnormals @ light, 0, None)
    shade = 0.30 + 0.70 * diffuse
    base_rgb = np.array(_hex_to_rgb(fill_hex))
    face_rgb = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)

    fz = verts[faces].mean(axis=1)[:, 2]
    order = np.argsort(fz)

    fig, ax = plt.subplots(figsize=(5, 5))
    tri_xy = verts[faces][:, :, :2]
    pc = PolyCollection(tri_xy[order], facecolors=face_rgb[order],
                        edgecolors="none", linewidths=0, antialiased=True)
    ax.add_collection(pc)

    if wire_alpha > 0 and wire_lw > 0:
        edges = np.unique(np.sort(np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
        seg = verts[edges][:, :, :2]
        lc = LineCollection(seg, colors=wire_hex, linewidths=wire_lw,
                            alpha=wire_alpha, zorder=3)
        ax.add_collection(lc)

    pad = 0.04
    xy = verts[:, :2]
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    sx = (xmax - xmin) * pad; sy = (ymax - ymin) * pad
    ax.set_xlim(xmin - sx, xmax + sx); ax.set_ylim(ymin - sy, ymax + sy)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.patch.set_alpha(0.0)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Spline tree — same helpers as generate_trees.py
# ─────────────────────────────────────────────────────────────────────
def render_spline_tree_native(npy_path, line_color="darkblue", spline_color="blue",
                               linewidth=1.4, point_size=0.7, spline_samples=60,
                               show_axes=False):
    """Replicates the per-panel rendering used by visualize_trees in
    Stage2_FlowMatching/generate_trees.py."""
    data = np.load(npy_path).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 40)
    valid = ~(np.all(np.abs(data[:, 1:]) < 1e-8, axis=1))
    data = data[valid]
    tree = data_to_tree(data)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    draw_tree_centerlines(tree, ax, color=line_color, linewidth=linewidth, alpha=0.5)
    draw_tree_splines_scatter(tree, ax, color=spline_color, spline_samples=spline_samples,
                              point_size=point_size)
    ax.scatter(tree.data["x"], tree.data["y"], tree.data["z"],
               c="green", marker="*", s=40, zorder=10)
    ax.view_init(elev=ELEV_DEG, azim=AZIM_DEG)
    if not show_axes:
        ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────
print(f"[1/4] input mesh (DARK_RED + wireframe): {GT_MESH}")
fig = render_mesh_solid_with_wire(GT_MESH, fill_hex=DARK_RED, wire_hex=DARKER_RED,
                                   wire_alpha=0.18, wire_lw=0.10, decimate_to=15000)
fig.savefig(OUT_DIR / "diagram_input_mesh.png", dpi=200, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"[2/4] encoding tree (GT spline, blue): {GT_NPY}")
fig = render_spline_tree_native(GT_NPY, line_color="darkblue", spline_color="blue")
fig.savefig(OUT_DIR / "diagram_centerline_tree.png", dpi=200, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"[3/4] decoding tree (generated spline, red): {GEN_NPY}")
src_npy = GEN_NPY if GEN_NPY.exists() else GT_NPY
if not GEN_NPY.exists():
    print(f"  (using {src_npy} since {GEN_NPY} missing)")
fig = render_spline_tree_native(src_npy, line_color="darkred", spline_color="red")
fig.savefig(OUT_DIR / "diagram_spline_tree.png", dpi=200, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"[4/4] output mesh (DARK_RED + wireframe): {GEN_MESH}")
if GEN_MESH.exists():
    fig = render_mesh_solid_with_wire(GEN_MESH, fill_hex=DARK_RED, wire_hex=DARKER_RED,
                                       wire_alpha=0.18, wire_lw=0.10, decimate_to=15000)
    fig.savefig(OUT_DIR / "diagram_output_mesh.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)

print("\nGenerated:")
for n in ["diagram_input_mesh.png", "diagram_centerline_tree.png", "diagram_spline_tree.png", "diagram_output_mesh.png"]:
    p = OUT_DIR / n
    print(f"  {p}  ({p.stat().st_size if p.exists() else 'MISSING'} B)")
