"""Batch render loft mesh OBJ files -> PNG images using matplotlib 3D."""

import argparse
import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


def render_mesh_png(obj_path: str, out_path: str, max_faces: int = 6000,
                    dpi: int = 150, figsize=(8, 7), elev: float = 25., azim: float = -60.):
    mesh = trimesh.load(obj_path, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)

    # Subsample faces for rendering speed while keeping shape visible
    step = max(1, len(F) // max_faces)
    F2 = F[::step]
    tris = V[F2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, alpha=0.65, linewidths=0)
    coll.set_facecolor([0.55, 0.70, 0.90])
    ax.add_collection3d(coll)

    margin = 0.02 * (V.max() - V.min())
    ax.set_xlim(V[:, 0].min() - margin, V[:, 0].max() + margin)
    ax.set_ylim(V[:, 1].min() - margin, V[:, 1].max() + margin)
    ax.set_zlim(V[:, 2].min() - margin, V[:, 2].max() + margin)
    ax.set_box_aspect([
        V[:, 0].ptp(), V[:, 1].ptp(), V[:, 2].ptp()
    ])

    stem = os.path.splitext(os.path.basename(obj_path))[0]
    ax.set_title(f"{stem}\n{len(V)} verts  {len(F)} faces", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Render OBJ mesh files to PNG.")
    parser.add_argument("--input_dir",  required=True, help="Directory containing .obj files")
    parser.add_argument("--output_dir", default=None,  help="Output directory (default: <input_dir>/../mesh_images)")
    parser.add_argument("--pattern",    default="*.obj")
    parser.add_argument("--max_faces",  type=int, default=6000)
    parser.add_argument("--dpi",        type=int, default=150)
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    input_dir  = args.input_dir
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir.rstrip("/")), "mesh_images")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, args.pattern)))
    total = len(files)
    ok = err = skip = 0

    for idx, fpath in enumerate(files, 1):
        stem    = os.path.splitext(os.path.basename(fpath))[0]
        out     = os.path.join(output_dir, stem + ".png")
        print(f"[{idx:3d}/{total}] {stem}...", end=" ", flush=True)

        if os.path.exists(out) and not args.overwrite:
            print("skip"); skip += 1; continue

        try:
            render_mesh_png(fpath, out, max_faces=args.max_faces, dpi=args.dpi)
            print(f"ok -> {out}")
            ok += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            err += 1

    print(f"\ndone: {ok} ok, {err} errors, {skip} skipped  (output: {output_dir})")


if __name__ == "__main__":
    main()
