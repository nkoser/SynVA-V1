import numpy as np
import matplotlib.pyplot as plt
import vtk
import os

def rdp(points, epsilon):
    """
    Simplify a curve using the Ramer-Douglas-Peucker algorithm.

    Parameters:
    - points: List of (x, y) coordinates representing the curve.
    - epsilon: The maximum distance allowed between the original curve and the simplified curve.

    Returns:
    - List of simplified (x, y) coordinates.
    """
    if len(points) <= 2:
        return points

    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = np.linalg.norm(np.cross(
            np.array(points[end]) - np.array(points[0]),
            np.array(points[i]) - np.array(points[0])
        )) / np.linalg.norm(np.array(points[end]) - np.array(points[0]))

        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)

        # Combine the results
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end]]

    return result


def interpolarRDP (centerline, epsilon = 0.05):

    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_puntos = []
    lista_rama = []
    for j in range(centerline.GetNumberOfCells()):#iterar por ramas  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();# number of points of the branch
        for i in range(numberOfCellPoints):
            originalPointId = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(originalPointId)
            lista_puntos.append(list(point))
            
        #Ya tengo la lista de puntos de la rama entonces la resampleo y la agrego a la linea punto y polydata
        rama_resampleada = rdp(lista_puntos, epsilon)
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)
            polyline.GetPointIds().InsertNextId(newPointId)
        cellarray.InsertNextCell(polyline)
        lista_rama.append(lista_puntos)
        lista_puntos = []
    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    return resampleada

def interpolarRDP_conRadio(centerline, epsilon=0.05):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_rama = []

    # Prepare radius array output
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")

    # Get original radius array
    original_radius = centerline.GetPointData().GetArray("Radius")

    for j in range(centerline.GetNumberOfCells()):  # iterate over branches  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        puntos_rama = []
        radios_rama = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            radius = original_radius.GetValue(pid)
            puntos_rama.append(list(point))
            radios_rama.append(radius)

        # Apply RDP resampling to the branch
        rama_resampleada = rdp(puntos_rama, epsilon)

        # Reconstruct polyline and resample radius
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)

            # Find index of the point in original list to get corresponding radius
            index = puntos_rama.index(point)
            radius_array.InsertNextValue(radios_rama[index])

            polyline.GetPointIds().InsertNextId(newPointId)

        cellarray.InsertNextCell(polyline)
        lista_rama.append(puntos_rama)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    resampleada.GetPointData().AddArray(radius_array)

    return resampleada


def _clean_polyline(points, scalars=None, return_keep=False):
    if len(points) <= 1:
        if return_keep:
            return points, scalars, np.arange(len(points), dtype=int)
        return points, scalars
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-12:
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    points = points[keep]
    if scalars is not None:
        scalars = scalars[keep]
    if return_keep:
        return points, scalars, keep
    return points, scalars


def _resample_polyline_uniform(points, scalars, step):
    if len(points) <= 1:
        return points, scalars
    pts = np.asarray(points, dtype=np.float64)
    scal = None if scalars is None else np.asarray(scalars, dtype=np.float64)
    pts, scal = _clean_polyline(pts, scal)
    if len(pts) <= 1:
        return pts, scal

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if not np.isfinite(total) or total <= 1e-12:
        return pts, scal

    s = np.concatenate([[0.0], np.cumsum(seg)])
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        step = total

    s_new = np.arange(0.0, total, step)
    if s_new.size == 0 or s_new[-1] < total:
        s_new = np.concatenate([s_new, [total]])

    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    z = np.interp(s_new, s, pts[:, 2])
    new_pts = np.column_stack([x, y, z])

    new_scal = None
    if scal is not None:
        new_scal = np.interp(s_new, s, scal)
    return new_pts, new_scal


def _cap_samples(points, scalars, max_points):
    if max_points is None:
        return points, scalars
    max_points = int(max_points)
    if max_points <= 0 or len(points) <= max_points:
        return points, scalars
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    points = points[idx]
    if scalars is not None:
        scalars = scalars[idx]
    return points, scalars


def _compute_point_degrees(centerline):
    n = centerline.GetNumberOfPoints()
    deg = np.zeros(n, dtype=np.int32)
    lines = centerline.GetLines()
    lines.InitTraversal()
    idlist = vtk.vtkIdList()
    while lines.GetNextCell(idlist):
        m = idlist.GetNumberOfIds()
        for i in range(m - 1):
            a = idlist.GetId(i)
            b = idlist.GetId(i + 1)
            deg[a] += 1
            deg[b] += 1
    return deg


def _clean_centerline(polydata, tol=0.0):
    if tol is None or tol <= 0:
        return polydata
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.PointMergingOn()
    cleaner.SetTolerance(float(tol))
    cleaner.Update()
    return cleaner.GetOutput()


def _arc_length(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, float(s[-1])


def _interp_between(s0, p0, r0, s1, p1, r1, s):
    if s1 <= s0 + 1e-12:
        return p0, r0
    t = (s - s0) / (s1 - s0)
    p = p0 + t * (p1 - p0)
    r = r0 + t * (r1 - r0)
    return p, r


def _resample_polyline_radius_adaptive(
    points,
    scalars,
    step_min,
    step_max,
    radius_scale,
    radius_mode="inverse",
    base_step=None,
    curv_threshold=0.0,
    curv_boost=0.0,
    drds_threshold=0.0,
    drds_boost=1.0,
    junction_s=None,
    junction_window=0.0,
    junction_factor=1.0,
):
    if len(points) <= 1:
        return points, scalars
    pts = np.asarray(points, dtype=np.float64)
    scal = None if scalars is None else np.asarray(scalars, dtype=np.float64)
    pts, scal = _clean_polyline(pts, scal)
    if len(pts) <= 1:
        return pts, scal

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if not np.isfinite(total) or total <= 1e-12:
        return pts, scal

    s = np.concatenate([[0.0], np.cumsum(seg)])
    step_min = float(step_min)
    step_max = float(step_max)
    base_step = float(step_max if base_step is None else base_step)
    radius_scale = float(radius_scale)
    radius_mode = (radius_mode or "inverse").lower()
    curv_threshold = float(curv_threshold)
    curv_boost = float(curv_boost)
    drds_threshold = float(drds_threshold)
    drds_boost = float(drds_boost)
    junction_window = float(junction_window)
    junction_factor = float(junction_factor)
    if junction_s is None:
        junction_s = []
    junction_s = np.asarray(junction_s, dtype=np.float64)
    if junction_s.size > 0:
        junction_s = np.unique(np.clip(junction_s, 0.0, total))
        junction_s.sort()

    if scal is None:
        return _resample_polyline_uniform(pts, scal, step_max)

    if radius_mode == "drds":
        drds = np.gradient(scal, s)
        drds = np.abs(drds)
        if not np.all(np.isfinite(drds)):
            drds = np.where(np.isfinite(drds), drds, 0.0)
    else:
        drds = None

    if curv_boost > 0.0 and len(pts) >= 3:
        v_prev = pts[1:-1] - pts[:-2]
        v_next = pts[2:] - pts[1:-1]
        n1 = np.linalg.norm(v_prev, axis=1)
        n2 = np.linalg.norm(v_next, axis=1)
        denom = n1 * n2 + 1e-12
        cosang = np.sum(v_prev * v_next, axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = np.arccos(cosang)
        arclen = 0.5 * (n1 + n2) + 1e-12
        curv_mid = angle / arclen
        curv = np.zeros(len(pts), dtype=np.float64)
        curv[1:-1] = curv_mid
        curv[0] = curv[1]
        curv[-1] = curv[-2]
    else:
        curv = None

    def interp_with_ptr(x_val, xs, ys, ptr):
        n = len(xs)
        while ptr < n - 2 and x_val > xs[ptr + 1]:
            ptr += 1
        x0, x1 = xs[ptr], xs[ptr + 1]
        y0, y1 = ys[ptr], ys[ptr + 1]
        if x1 <= x0:
            return float(y0), ptr
        t = (x_val - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0)), ptr

    s_new = [0.0]
    cur = 0.0
    ptr_r = 0
    ptr_g = 0
    ptr_c = 0
    max_iter = int(np.ceil(total / max(step_min, 1e-12))) + 10
    junction_ptr = 0

    for _ in range(max_iter):
        step = base_step

        if radius_mode == "drds":
            g, ptr_g = interp_with_ptr(cur, s, drds, ptr_g)
            if np.isfinite(g) and g > drds_threshold:
                step = step / (1.0 + drds_boost * (g - drds_threshold))
        else:
            r, ptr_r = interp_with_ptr(cur, s, scal, ptr_r)
            if np.isfinite(r):
                r = max(r, 1e-12)
                if radius_mode == "direct":
                    step = min(step, radius_scale * r)
                else:
                    step = min(step, radius_scale / r)

        if curv is not None:
            c, ptr_c = interp_with_ptr(cur, s, curv, ptr_c)
            if np.isfinite(c) and c > curv_threshold:
                step = step / (1.0 + curv_boost * (c - curv_threshold))

        if junction_window > 0 and junction_s.size > 0:
            while junction_ptr < len(junction_s) - 1 and junction_s[junction_ptr] < cur:
                junction_ptr += 1
            d0 = abs(cur - junction_s[junction_ptr])
            d1 = abs(cur - junction_s[junction_ptr - 1]) if junction_ptr > 0 else d0
            if min(d0, d1) <= junction_window:
                step *= junction_factor

        if not np.isfinite(step) or step <= 0:
            step = base_step
        step = float(np.clip(step, step_min, base_step))
        cur += step
        if cur >= total:
            break
        s_new.append(cur)

    if s_new[-1] < total:
        s_new.append(total)
    s_new = np.asarray(s_new, dtype=np.float64)

    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    z = np.interp(s_new, s, pts[:, 2])
    new_pts = np.column_stack([x, y, z])
    new_scal = np.interp(s_new, s, scal)
    return new_pts, new_scal


def resample_centerline_step(centerline, step=0.003, use_radius=False, step_min=0.001, step_max=0.006,
                             radius_scale=0.3, radius_mode="inverse",
                             base_step=None, curv_threshold=0.0, curv_boost=0.0,
                             drds_threshold=0.0, drds_boost=1.0,
                             junction_window=0.0, junction_factor=1.0, junction_degree=3,
                             max_points_per_branch=None):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    original_radius = centerline.GetPointData().GetArray("Radius")
    deg = _compute_point_degrees(centerline)

    for j in range(centerline.GetNumberOfCells()):
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        pts = []
        rads = []
        pids = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            pts.append(point)
            pids.append(pid)
            if original_radius is not None:
                rads.append(float(original_radius.GetValue(pid)))

        pts = np.asarray(pts, dtype=np.float64)
        rads_arr = None if original_radius is None else np.asarray(rads, dtype=np.float64)
        pids = np.asarray(pids, dtype=np.int64)

        if use_radius:
            junction_s = []
            if junction_window > 0 and pids.size > 0:
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                s = np.concatenate([[0.0], np.cumsum(seg)])
                for idx, pid in enumerate(pids):
                    if deg[pid] >= junction_degree:
                        junction_s.append(float(s[idx]))
            new_pts, new_rads = _resample_polyline_radius_adaptive(
                pts, rads_arr, step_min, step_max, radius_scale, radius_mode,
                base_step=base_step, curv_threshold=curv_threshold, curv_boost=curv_boost,
                drds_threshold=drds_threshold, drds_boost=drds_boost,
                junction_s=junction_s, junction_window=junction_window,
                junction_factor=junction_factor
            )
        else:
            new_pts, new_rads = _resample_polyline_uniform(pts, rads_arr, step)

        new_pts, new_rads = _cap_samples(new_pts, new_rads, max_points_per_branch)

        polyline = vtk.vtkPolyLine()
        for idx in range(new_pts.shape[0]):
            newPointId = points.InsertNextPoint(new_pts[idx].tolist())
            polyline.GetPointIds().InsertNextId(newPointId)
            if new_rads is not None:
                radius_array.InsertNextValue(float(new_rads[idx]))
        cellarray.InsertNextCell(polyline)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    if radius_array.GetNumberOfTuples() > 0:
        resampleada.GetPointData().AddArray(radius_array)
    return resampleada


def resample_centerline_minimal_adaptive(
    centerline,
    geom_tol=0.02,
    rad_tol=0.04,
    w_rad=1.0,
    junction_degree=3,
    junction_keep_k=0,
    junction_keep_window=0.0,
    max_points_per_branch=60,
    clean_tol=0.0,
    minimal_prune_passes=4,
):
    centerline = _clean_centerline(centerline, tol=clean_tol)
    deg = _compute_point_degrees(centerline)

    original_radius = centerline.GetPointData().GetArray("Radius")

    out = vtk.vtkPolyData()
    out_pts = vtk.vtkPoints()
    out_lines = vtk.vtkCellArray()

    out_rad = vtk.vtkFloatArray()
    out_rad.SetName("Radius")

    geom_tol = float(geom_tol)
    rad_tol = float(rad_tol)
    w_rad = float(w_rad)
    max_points_per_branch = int(max_points_per_branch) if max_points_per_branch is not None else None
    minimal_prune_passes = max(0, int(minimal_prune_passes))

    for cid in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(cid)
        m = cell.GetNumberOfPoints()
        if m < 2:
            continue

        pts = np.zeros((m, 3), dtype=np.float64)
        pids = np.zeros(m, dtype=np.int64)
        rads = None
        if original_radius is not None:
            rads = np.zeros(m, dtype=np.float64)

        for i in range(m):
            pid = cell.GetPointId(i)
            pids[i] = pid
            pts[i] = centerline.GetPoints().GetPoint(pid)
            if rads is not None:
                rads[i] = float(original_radius.GetValue(pid))

        pts, rads, keep = _clean_polyline(pts, rads, return_keep=True)
        pids = pids[keep]
        if len(pts) <= 1:
            continue

        s, total = _arc_length(pts)
        if total <= 1e-12:
            continue

        # initial knots: endpoints + junctions
        knots = set([0, len(pts) - 1])
        junction_indices = []
        for i in range(len(pts)):
            if deg[pids[i]] >= junction_degree:
                knots.add(i)
                junction_indices.append(i)
        if junction_keep_k and junction_keep_k > 0:
            k = int(junction_keep_k)
            for i in junction_indices:
                for j in range(max(0, i - k), min(len(pts), i + k + 1)):
                    knots.add(j)
        if junction_keep_window and junction_keep_window > 0 and junction_indices:
            for i in junction_indices:
                s0 = s[i]
                for j in range(len(pts)):
                    if abs(s[j] - s0) <= junction_keep_window:
                        knots.add(j)
        knots = sorted(knots)
        mandatory_knots = set(knots)

        def segments_from_knots(kn):
            kn = sorted(kn)
            return [(kn[i], kn[i + 1]) for i in range(len(kn) - 1)]

        def worst_error_on_segment(a, b):
            if b <= a + 1:
                return 0.0, None
            s0, s1 = s[a], s[b]
            p0, p1 = pts[a], pts[b]
            r0 = 0.0 if rads is None else rads[a]
            r1 = 0.0 if rads is None else rads[b]
            worst_err_local = 0.0
            worst_idx_local = None
            for i in range(a + 1, b):
                p_hat, r_hat = _interp_between(s0, p0, r0, s1, p1, r1, s[i])
                e_geom = float(np.linalg.norm(pts[i] - p_hat))
                e_rad = 0.0
                if rads is not None:
                    e_rad = float(abs(rads[i] - r_hat))
                e_geom_norm = e_geom / max(geom_tol, 1e-12)
                e_rad_norm = (e_rad / max(rad_tol, 1e-12)) if rads is not None else 0.0
                e = max(e_geom_norm, w_rad * e_rad_norm)
                if e > worst_err_local:
                    worst_err_local = e
                    worst_idx_local = i
            return worst_err_local, worst_idx_local

        while True:
            if max_points_per_branch is not None and len(knots) >= max_points_per_branch:
                break

            worst_err = 0.0
            worst_idx = None

            for a, b in segments_from_knots(knots):
                e, i = worst_error_on_segment(a, b)
                if e > worst_err:
                    worst_err = e
                    worst_idx = i

            if worst_idx is None or worst_err <= 1.0:
                break

            knots.append(worst_idx)
            knots = sorted(set(knots))

        # Prune pass: remove any non-mandatory knot that is not needed to keep errors <= tolerances.
        for _ in range(minimal_prune_passes):
            removed_any = False
            i = 1
            while i < len(knots) - 1:
                k_idx = knots[i]
                if k_idx in mandatory_knots:
                    i += 1
                    continue
                a = knots[i - 1]
                b = knots[i + 1]
                e, _ = worst_error_on_segment(a, b)
                if e <= 1.0:
                    knots.pop(i)
                    removed_any = True
                    continue
                i += 1
            if not removed_any:
                break

        polyline = vtk.vtkPolyLine()
        for i in knots:
            pid_new = out_pts.InsertNextPoint(pts[i].tolist())
            polyline.GetPointIds().InsertNextId(pid_new)
            if rads is not None:
                out_rad.InsertNextValue(float(rads[i]))
        out_lines.InsertNextCell(polyline)

    out.SetPoints(out_pts)
    out.SetLines(out_lines)
    if out_rad.GetNumberOfTuples() > 0:
        out.GetPointData().AddArray(out_rad)
    return out


def resample_centerline_event(
    centerline,
    base_step=0.02,
    event_step=0.005,
    event_window=0.02,
    drds_threshold=0.0,
    curv_threshold=0.0,
    junction_window=0.02,
    junction_degree=3,
):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    original_radius = centerline.GetPointData().GetArray("Radius")
    deg = _compute_point_degrees(centerline)

    base_step = float(base_step)
    event_step = float(event_step)
    event_window = float(event_window)
    junction_window = float(junction_window)

    for j in range(centerline.GetNumberOfCells()):
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        pts = []
        rads = []
        pids = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            pts.append(point)
            pids.append(pid)
            if original_radius is not None:
                rads.append(float(original_radius.GetValue(pid)))

        pts = np.asarray(pts, dtype=np.float64)
        rads_arr = None if original_radius is None else np.asarray(rads, dtype=np.float64)
        pids = np.asarray(pids, dtype=np.int64)

        pts, rads_arr = _clean_polyline(pts, rads_arr)
        if len(pts) <= 1:
            continue

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if not np.isfinite(total) or total <= 1e-12:
            continue

        # event sources: dr/ds and curvature
        event_s = []
        if rads_arr is not None and drds_threshold > 0:
            drds = np.gradient(rads_arr, s)
            drds = np.abs(drds)
            if np.any(np.isfinite(drds)):
                idx = np.where(drds > drds_threshold)[0]
                event_s.extend(s[idx].tolist())

        if curv_threshold > 0 and len(pts) >= 3:
            v_prev = pts[1:-1] - pts[:-2]
            v_next = pts[2:] - pts[1:-1]
            n1 = np.linalg.norm(v_prev, axis=1)
            n2 = np.linalg.norm(v_next, axis=1)
            denom = n1 * n2 + 1e-12
            cosang = np.sum(v_prev * v_next, axis=1) / denom
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.arccos(cosang)
            arclen = 0.5 * (n1 + n2) + 1e-12
            curv_mid = angle / arclen
            curv = np.zeros(len(pts), dtype=np.float64)
            curv[1:-1] = curv_mid
            curv[0] = curv[1]
            curv[-1] = curv[-2]
            idx = np.where(curv > curv_threshold)[0]
            event_s.extend(s[idx].tolist())

        # junction events
        if junction_window > 0 and pids.size > 0:
            for idx, pid in enumerate(pids):
                if deg[pid] >= junction_degree:
                    event_s.append(float(s[idx]))

        # base sampling
        if not np.isfinite(base_step) or base_step <= 0:
            base_step = total
        s_new = np.arange(0.0, total, base_step).tolist()
        if len(s_new) == 0 or s_new[-1] < total:
            s_new.append(total)

        # event windows
        if event_s and event_step > 0 and event_window > 0:
            for s0 in event_s:
                a = max(0.0, s0 - event_window)
                b = min(total, s0 + event_window)
                dense = np.arange(a, b, event_step).tolist()
                if len(dense) == 0 or dense[-1] < b:
                    dense.append(b)
                s_new.extend(dense)

        s_new = np.unique(np.clip(np.asarray(s_new, dtype=np.float64), 0.0, total))
        s_new.sort()

        x = np.interp(s_new, s, pts[:, 0])
        y = np.interp(s_new, s, pts[:, 1])
        z = np.interp(s_new, s, pts[:, 2])
        new_pts = np.column_stack([x, y, z])
        new_rads = None
        if rads_arr is not None:
            new_rads = np.interp(s_new, s, rads_arr)

        polyline = vtk.vtkPolyLine()
        for idx in range(new_pts.shape[0]):
            newPointId = points.InsertNextPoint(new_pts[idx].tolist())
            polyline.GetPointIds().InsertNextId(newPointId)
            if new_rads is not None:
                radius_array.InsertNextValue(float(new_rads[idx]))
        cellarray.InsertNextCell(polyline)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    if radius_array.GetNumberOfTuples() > 0:
        resampleada.GetPointData().AddArray(radius_array)
    return resampleada


def resample_centerline_vmtk(centerline, step=0.003):
    try:
        from vmtk import vtkvmtk
    except Exception as exc:
        raise RuntimeError("vtkvmtk not available for centerline resampling") from exc

    f = vtkvmtk.vtkvmtkCenterlineResampling()
    f.SetInputData(centerline)
    f.SetResamplingStepLength(float(step))
    f.Update()
    return f.GetOutput()

def vtpToObj (file, folderin, folderout):
    polydata_reader = vtk.vtkXMLPolyDataReader()
    #polydata_reader.SetFileName("centerlines/"+file)
    polydata_reader.SetFileName(folderin + "/"+file)
   

    polydata_reader.Update()
        
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(polydata_reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Assign actor to the renderer
    ren.AddActor(actor)
    # Use splitext to strip ONLY the .vtp extension; do NOT split on the
    # first dot (UIDs like aneux_UPF_P0016.00_ID1.vtp have intermediate dots
    # that must be preserved or downstream graph/tree steps lose them).
    import os as _os
    base = _os.path.splitext(file)[0]
    print("writing file", folderout + "/" + base)
    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix(folderout + "/" + base);
    writer.SetInput(renWin);
    writer.Write()
