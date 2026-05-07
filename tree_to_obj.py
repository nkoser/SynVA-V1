"""Convert generated tree NPZs to .obj (line skeleton + optional PC).

Each tree segment becomes two vertices and one OBJ "l" (line) entry.
Optionally writes the conditioning point cloud as a separate OBJ of points.

Usage:
    python tree_to_obj.py <run_label> [<run_label> ...] [--name STEM] [--pc]

Run labels = keys from make_topdown_perceiver.py "dirs" dict, plus "gt".
If no --name is given, the first common sample is used.
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
OUT = Path("HierarchicalTreeGeometryPC_perceiver/output/obj_export")


def load_tree(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    sp = d["sp_absolute"]
    geom = d["geometry"]
    ep = d["ep_absolute"] if "ep_absolute" in d else sp + geom[:, :3]
    mask = (
        np.any(np.abs(sp) > 1e-8, axis=-1)
        | np.any(np.abs(ep) > 1e-8, axis=-1)
    )
    return sp[mask], ep[mask]


def write_lines_obj(path: Path, sp, ep):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(sp)
    with open(path, "w") as f:
        f.write(f"# tree skeleton: {n} segments\n")
        for p in sp:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for p in ep:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        # OBJ is 1-indexed. sp at 1..n, ep at n+1..2n.
        for i in range(n):
            f.write(f"l {i + 1} {n + i + 1}\n")
    print(f"  wrote {path}  ({n} segments)")


def write_points_obj(path: Path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# point cloud: {len(pts)} points\n")
        for p in pts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"  wrote {path}  ({len(pts)} pts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+", help="run labels (e.g. gt pcnorm_v2_cfg2)")
    ap.add_argument("--name", default=None, help="sample stem (default: first common)")
    ap.add_argument("--pc", action="store_true", help="also write point cloud .obj")
    args = ap.parse_args()

    bad = [l for l in args.labels if l not in DIRS]
    if bad:
        sys.exit(f"unknown label(s): {bad}\navailable: {list(DIRS)}")

    if args.name is None:
        sets = []
        for l in args.labels:
            d = DIRS[l]
            if not d.exists():
                sys.exit(f"missing dir for {l}: {d}")
            sets.append({f.stem for f in d.glob("*.npz")})
        common = sorted(set.intersection(*sets))
        if not common:
            sys.exit("no common sample across labels")
        name = common[0]
    else:
        name = args.name
    print(f"sample: {name}")

    OUT.mkdir(parents=True, exist_ok=True)
    for l in args.labels:
        sp, ep = load_tree(DIRS[l] / f"{name}.npz")
        write_lines_obj(OUT / f"{name}__{l}.obj", sp, ep)

    if args.pc:
        stem = name.replace("_skeleton_with_orders", "")
        pc_path = PC_DIR / f"{stem}.npy"
        if pc_path.exists():
            pts = np.load(pc_path)
            write_points_obj(OUT / f"{name}__pc.obj", pts)
        else:
            print(f"  (no PC found at {pc_path})")


if __name__ == "__main__":
    main()
