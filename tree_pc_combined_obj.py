"""Combine generated tree + conditioning PC into a single coloured OBJ.

Vertex colors use the MeshLab/Blender extension: `v x y z r g b`.
- PC: salmon (1.0, 0.4, 0.4)
- Tree: viridis-like green→yellow gradient by hierarchy level.

Usage:
    python tree_pc_combined_obj.py <run_label> [<run_label> ...] [--name STEM]
"""
import argparse
import sys
from pathlib import Path
import numpy as np


GT_DIR = Path("HierarchicalTreeGeometry/data/trees_v1/val")
DIRS = {
    "gt":              GT_DIR,
    "v23":             Path("HierarchicalTreeGeometryPC/output/fm_pc_v23/generated/npz"),
    "perceiver_v2":    Path("HierarchicalTreeGeometryPC_perceiver/output/fm_pc_perceiver_v2/generated/npz"),
    "pcnorm_v1":       Path("/data/HierarchicalTreeGeometryPCNorm/output/pcnorm_v1/generated/npz"),
    "pcnorm_v2_cfg1":  Path("/data/HierarchicalTreeGeometryPCNorm/output/pcnorm_v2/generated_cfg1p0/npz"),
    "pcnorm_v2_cfg2":  Path("/data/HierarchicalTreeGeometryPCNorm/output/pcnorm_v2/generated_cfg2p0/npz"),
    "pcnorm_v2_cfg4":  Path("/data/HierarchicalTreeGeometryPCNorm/output/pcnorm_v2/generated_cfg4p0/npz"),
    "htg_uncond":      Path("HierarchicalTreeGeometryPCGuided/output/htg_guided_a0p0/generated/npz"),
}
PC_DIR = Path("HierarchicalTreeGeometryPC/data/pc_cache_1024")
OUT = Path("HierarchicalTreeGeometryPC_perceiver/output/obj_combined")

PC_COLORS = {
    "salmon":     (1.0, 0.4, 0.4),
    "black":      (0.05, 0.05, 0.05),
    "gray":       (0.6, 0.6, 0.6),
    "cyan":       (0.0, 0.7, 0.9),
    "magenta":    (0.9, 0.2, 0.8),
    "light_blue": (0.5, 0.7, 1.0),
    "orange":     (1.0, 0.55, 0.0),
    "white":      (0.95, 0.95, 0.95),
}


def load_tree(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    sp = d["sp_absolute"]
    geom = d["geometry"]
    ep = d["ep_absolute"] if "ep_absolute" in d else sp + geom[:, :3]
    hl = d["hierarchy_levels"]
    mask = (
        np.any(np.abs(sp) > 1e-8, axis=-1)
        | np.any(np.abs(ep) > 1e-8, axis=-1)
    )
    return sp[mask], ep[mask], hl[mask]


def viridis(t):
    """Cheap 5-stop viridis approximation, t ∈ [0,1] → (r,g,b)."""
    stops = np.array([
        [0.267, 0.005, 0.329],
        [0.282, 0.140, 0.458],
        [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518],
        [0.267, 0.749, 0.441],
        [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150],
        [0.993, 0.906, 0.144],
    ])
    t = np.clip(t, 0.0, 1.0)
    f = t * (len(stops) - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, len(stops) - 1)
    a = f - i0
    return stops[i0] * (1 - a)[..., None] + stops[i1] * a[..., None]


def write_combined_obj(path: Path, sp, ep, hl, pc, pc_color):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_seg = len(sp)
    n_pc = len(pc) if pc is not None else 0
    max_hl = max(int(hl.max()) if len(hl) else 1, 1)
    rgb = viridis(hl / max_hl)  # [n_seg, 3]

    with open(path, "w") as f:
        f.write(f"# combined: {n_seg} tree segments + {n_pc} PC points\n")
        f.write("# vertex format: v x y z r g b  (MeshLab/Blender extension)\n")

        # Tree start points
        for p, c in zip(sp, rgb):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        # Tree end points
        for p, c in zip(ep, rgb):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        # PC points
        if pc is not None:
            r, g, b = pc_color
            for p in pc:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{r:.3f} {g:.3f} {b:.3f}\n")

        # Lines: sp[i] (1-based) → ep[i] (n+1-based)
        for i in range(n_seg):
            f.write(f"l {i + 1} {n_seg + i + 1}\n")
    print(f"  wrote {path}  (tree={n_seg}, pc={n_pc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+", help="run labels (e.g. gt pcnorm_v2_cfg2)")
    ap.add_argument("--name", default=None, help="sample stem (default: first common)")
    ap.add_argument(
        "--pc-color", default="cyan", choices=list(PC_COLORS),
        help="color of PC points (default: cyan)",
    )
    args = ap.parse_args()

    bad = [l for l in args.labels if l not in DIRS]
    if bad:
        sys.exit(f"unknown label(s): {bad}\navailable: {list(DIRS)}")

    if args.name is None:
        sets = [{f.stem for f in DIRS[l].glob("*.npz")} for l in args.labels]
        common = sorted(set.intersection(*sets))
        if not common:
            sys.exit("no common sample across labels")
        name = common[0]
    else:
        name = args.name
    print(f"sample: {name}")

    stem = name.replace("_skeleton_with_orders", "")
    pc_path = PC_DIR / f"{stem}.npy"
    pc = np.load(pc_path) if pc_path.exists() else None
    if pc is None:
        print(f"  WARNING: no PC at {pc_path}")

    OUT.mkdir(parents=True, exist_ok=True)
    pc_color = PC_COLORS[args.pc_color]
    print(f"  PC color: {args.pc_color} {pc_color}")
    for l in args.labels:
        sp, ep, hl = load_tree(DIRS[l] / f"{name}.npz")
        write_combined_obj(
            OUT / f"{name}__{l}_with_pc_{args.pc_color}.obj",
            sp, ep, hl, pc, pc_color,
        )


if __name__ == "__main__":
    main()
