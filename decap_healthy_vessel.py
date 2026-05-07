"""Build open meshes for healthy_vessel from smoothed closed meshes, while
using /data/prepared_meshes_3/<uid>/01_mesh/mesh.obj as opening reference.

Output keeps smoothed geometry from /data/healthy_vessel/*_closed and removes
cap faces using 01_mesh as geometric reference for valid openings.

Output: /data/healthy_vessel_decapped/<uid>/<uid>.obj (open, with smoothed
geometry from healthy_vessel preserved).
"""

import argparse
import os
import sys
import time

import numpy as np
import vtk
from scipy.spatial import cKDTree

HEALTHY_DIR  = "/data/healthy_vessel"
ORIGINAL_BASE = "/data/prepared_meshes_3"
OUT_DIR      = "/data/healthy_vessel_decapped"
SUFFIX       = "_vessel_submesh_closed"
VERTEX_TOL   = 1e-3   # vertex match tolerance (smoothed vs 01_mesh)
# Distance threshold (relative to mesh bbox diagonal) for deciding whether a
# cap component's boundary loop matches an 01_mesh opening. End-caps will
# match within a tiny fraction; aneurysm patch boundaries are far away.
CAP_LOOP_DIST_REL = 0.02
USE_ORIGINAL_OPEN_AS_OUTPUT = False


def _read_obj(path):
    r = vtk.vtkOBJReader(); r.SetFileName(path); r.Update()
    pd = vtk.vtkPolyData(); pd.DeepCopy(r.GetOutput())
    # merge duplicate vertices (Blender exports isolated triangles)
    c = vtk.vtkCleanPolyData()
    c.SetInputData(pd); c.SetTolerance(0.0); c.PointMergingOn()
    c.ConvertLinesToPointsOff(); c.ConvertPolysToLinesOff(); c.ConvertStripsToPolysOff()
    c.Update()
    out = vtk.vtkPolyData(); out.DeepCopy(c.GetOutput())
    return out


def _polydata_to_arrays(pd):
    n_pts = pd.GetNumberOfPoints()
    pts = np.empty((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        pts[i] = pd.GetPoint(i)
    polys = pd.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    tris = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() != 3:
            continue
        tris.append((id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)))
    tris = np.asarray(tris, dtype=np.int64)
    return pts, tris


def _triangle_signatures(pts, tris, decimals):
    """Return a sorted-tuple-of-rounded-coords signature per triangle."""
    rp = np.round(pts, decimals)
    sigs = []
    for a, b, c in tris:
        tri = [tuple(rp[a]), tuple(rp[b]), tuple(rp[c])]
        tri.sort()
        sigs.append((tri[0], tri[1], tri[2]))
    return sigs


def _write_obj(path, pts, tris):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# decapped from healthy_vessel\n")
        for p in pts:
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        for t in tris:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def _drop_unused_vertices(pts, tris):
    used = np.unique(tris.reshape(-1))
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_tris = remap[tris]
    new_pts = pts[used]
    return new_pts, new_tris


def decap_one(uid, verbose=False):
    closed_p = os.path.join(HEALTHY_DIR, uid + SUFFIX, uid + SUFFIX + ".obj")
    open_p   = os.path.join(ORIGINAL_BASE, uid, "01_mesh", "mesh.obj")

    if not os.path.isfile(closed_p):
        return False, f"missing closed: {closed_p}"
    if not os.path.isfile(open_p):
        return False, f"missing open: {open_p}"

    opened = _read_obj(open_p)

    if USE_ORIGINAL_OPEN_AS_OUTPUT:
        pts_o, tris_o = _polydata_to_arrays(opened)
        out_path = os.path.join(OUT_DIR, uid, uid + ".obj")
        _write_obj(out_path, pts_o, tris_o)

        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(opened)
        fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff()
        fe.Update()
        n_bnd = fe.GetOutput().GetNumberOfCells()
        msg = f"copied_original_open tris:{len(tris_o)} boundary_edges:{n_bnd}"
        if verbose:
            print(f"  {uid}: {msg}")
        return True, msg

    closed = _read_obj(closed_p)
    pts_c, tris_c = _polydata_to_arrays(closed)
    pts_o, tris_o = _polydata_to_arrays(opened)

    # ---- Step 1: identify cap-vertices = smoothed vertices NOT in 01_mesh.
    tree = cKDTree(pts_o)
    d, _ = tree.query(pts_c, k=1)
    is_cap_vert = d > VERTEX_TOL

    # ---- Step 2: 01_mesh boundary loop points and matching ring-vertices in smoothed.
    fe_o = vtk.vtkFeatureEdges()
    fe_o.SetInputData(opened)
    fe_o.BoundaryEdgesOn(); fe_o.FeatureEdgesOff(); fe_o.NonManifoldEdgesOff(); fe_o.ManifoldEdgesOff()
    fe_o.Update()
    bnd_pd = fe_o.GetOutput()
    n_bnd_o_pts = bnd_pd.GetNumberOfPoints()
    if n_bnd_o_pts == 0:
        return False, "01_mesh has no boundary edges"
    bnd_o_pts = np.array([bnd_pd.GetPoint(i) for i in range(n_bnd_o_pts)])
    # for each smoothed vertex: distance to nearest 01_mesh boundary point
    bnd_tree = cKDTree(bnd_o_pts)
    d_bnd, _ = bnd_tree.query(pts_c, k=1)

    bb = pts_c.max(axis=0) - pts_c.min(axis=0)
    diag = float(np.linalg.norm(bb))
    dist_thr = diag * CAP_LOOP_DIST_REL
    is_ring_vert = d_bnd < dist_thr  # smoothed vertex lies on a 01_mesh opening loop

    # ---- Step 3: cap triangles = ANY of its 3 vertices is a cap-vertex.
    # This catches end-cap fan tris (only the apex is new) AND aneurysm patch tris.
    tri_is_cap = is_cap_vert[tris_c].any(axis=1)
    n_total = len(tris_c)
    n_cap_tri = int(tri_is_cap.sum())

    if n_cap_tri == 0:
        out_path = os.path.join(OUT_DIR, uid, uid + ".obj")
        _write_obj(out_path, pts_c, tris_c)
        if verbose:
            print(f"  {uid}: no cap triangles found, copied closed as-is")
        return True, f"no_cap_tris closed:{n_total}"

    # ---- Step 4: connected components of cap triangles via shared edges.
    cap_tri_idx = np.where(tri_is_cap)[0]
    edge_to_tris = {}
    for ti in cap_tri_idx:
        a, b, c = tris_c[ti]
        for e in [(a, b), (b, c), (c, a)]:
            ek = (min(e), max(e))
            edge_to_tris.setdefault(ek, []).append(int(ti))
    adj = {int(ti): set() for ti in cap_tri_idx}
    for ek, lst in edge_to_tris.items():
        if len(lst) >= 2:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    adj[lst[i]].add(lst[j])
                    adj[lst[j]].add(lst[i])
    comp_id = {int(ti): -1 for ti in cap_tri_idx}
    components = []
    cur = 0
    for ti in cap_tri_idx:
        if comp_id[int(ti)] != -1:
            continue
        stack = [int(ti)]
        comp_tris = []
        while stack:
            u = stack.pop()
            if comp_id[u] != -1:
                continue
            comp_id[u] = cur
            comp_tris.append(u)
            for v in adj[u]:
                if comp_id[v] == -1:
                    stack.append(v)
        components.append(comp_tris)
        cur += 1

    # ---- Step 5: classify each component
    # End-cap component: contains ring vertices (vertices on a 01_mesh
    # boundary loop) AND its non-cap vertices are all on the ring.
    # Aneurysm component: its non-cap vertices are mid-wall (NOT on any ring).
    drop_tri = np.zeros(n_total, dtype=bool)
    comp_decisions = []
    for comp_tris in components:
        verts = set()
        non_cap_verts = set()
        for ti in comp_tris:
            for v in tris_c[ti]:
                vi = int(v)
                verts.add(vi)
                if not is_cap_vert[vi]:
                    non_cap_verts.add(vi)

        if not non_cap_verts:
            # fully isolated patch (rare); treat as keep to be safe
            comp_decisions.append(("KEEP_isolated", len(comp_tris), -1.0, -1.0))
            continue

        ncv = np.fromiter(non_cap_verts, dtype=np.int64)
        d2 = d_bnd[ncv]
        med_d = float(np.median(d2))
        max_d = float(np.max(d2))
        # End-cap if ALL non-cap (=interface) vertices lie on a 01_mesh boundary loop.
        if max_d < dist_thr:
            drop_tri[comp_tris] = True
            comp_decisions.append(("DROP_endcap", len(comp_tris), med_d, max_d))
        else:
            comp_decisions.append(("KEEP_aneurysm", len(comp_tris), med_d, max_d))

    keep_mask = ~drop_tri
    n_keep = int(keep_mask.sum())
    n_drop = n_total - n_keep
    n_capv  = int(is_cap_vert.sum())

    if n_keep == 0:
        return False, f"all triangles dropped (n_total={n_total})"

    kept_tris = tris_c[keep_mask]
    new_pts, new_tris = _drop_unused_vertices(pts_c, kept_tris)

    out_path = os.path.join(OUT_DIR, uid, uid + ".obj")
    _write_obj(out_path, new_pts, new_tris)

    # quick sanity: count boundary edges via VTK
    pd_new = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    for p in new_pts:
        vpts.InsertNextPoint(p.tolist())
    pd_new.SetPoints(vpts)
    cells = vtk.vtkCellArray()
    for t in new_tris:
        cells.InsertNextCell(3); cells.InsertCellPoint(int(t[0])); cells.InsertCellPoint(int(t[1])); cells.InsertCellPoint(int(t[2]))
    pd_new.SetPolys(cells)
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd_new)
    fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff()
    fe.Update()
    n_bnd = fe.GetOutput().GetNumberOfCells()

    decisions_short = ",".join(f"{d[0]}({d[1]},mx={d[3]:.3f})" for d in comp_decisions)
    msg = (f"closed:{n_total} 01_loops_pts:{n_bnd_o_pts} cap_verts:{n_capv} "
           f"comps:{len(components)} kept:{n_keep} dropped:{n_drop} "
           f"bnd_after:{n_bnd} thr:{dist_thr:.3f} [{decisions_short}]")
    if verbose:
        print(f"  {uid}: {msg}")
    return True, msg


def list_uids():
    return sorted([d[:-len(SUFFIX)] for d in os.listdir(HEALTHY_DIR)
                   if d.endswith(SUFFIX) and os.path.isdir(os.path.join(HEALTHY_DIR, d))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    uids = args.uids or list_uids()
    if args.limit:
        uids = uids[:args.limit]

    print(f"decapping {len(uids)} cases -> {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    n_ok = 0; n_fail = 0
    fails = []
    t0 = time.time()
    for i, uid in enumerate(uids, 1):
        ok, msg = decap_one(uid, verbose=not args.quiet)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            fails.append((uid, msg))
            print(f"  [FAIL] {uid}: {msg}")
        if i % 50 == 0 or i == len(uids):
            dt = time.time() - t0
            print(f"  progress {i}/{len(uids)} ok={n_ok} fail={n_fail} ({dt:.1f}s)")

    print(f"DONE ok={n_ok} fail={n_fail}")
    if fails and not args.quiet:
        print("failures:")
        for u, m in fails[:30]:
            print(f"  {u}: {m}")


if __name__ == "__main__":
    sys.exit(main())
