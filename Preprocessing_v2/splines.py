import warnings

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.interpolate import splprep, splev, make_splprep, BSpline
import matplotlib.pyplot as plt
import pickle
import os
#from parseObj import calcularMatriz
import traceback
import networkx as nx

from collections import Counter

from Preprocessing_v2.Arbol import Node


def get_points_by_line(centerline):
    points_array = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        points = cell.GetPoints()
        for j in range(points.GetNumberOfPoints()):
            point = points.GetPoint(j)#i me dice el numero de linea y j el de punto
            p = (point[0], point[1], point[2], i)
            points_array.append(p)
    return np.array(points_array)

# Step 1: Read the .vtp files
def read_vtp(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    output = reader.GetOutput()
    if not output:
        print(f"Error reading file: {file_path}")
    return reader.GetOutput()

def traversefeaturesSerializado(root, features, k):
    def post_order(root, features):
        if root:
            post_order(root.left, features)
            post_order(root.right, features)
            features.append(root.radius)
                
        else:
            features.append(k*[0.])          

    post_order(root, features)
    return features[:-1]  # remove last ,


def limpiarRadiosSplines2(radius):
    c= []
    r = []
    for x in radius:
        if isinstance(x, (np.float16, np.float32, np.float64)):
            c.append(float(x))
        elif(len(x))==3:
            for a in x:
                for num in a:
                    c.append(float(num))
        else:
            for a in x:
                c.append(float(a))
    return c

def _canonical_periodic_knots(n_cp: int = 8, k: int = 3) -> np.ndarray:
    """Uniform periodic clamped-knot vector for n_cp periodic CPs (last k = first k)."""
    n_indep = n_cp - k                     # 5 unique CPs
    h = 1.0 / n_indep                      # uniform step in [0,1]
    interior = np.arange(0.0, 1.0 + 0.5 * h, h)        # n_indep+1 knots
    pre  = np.arange(-k, 0) * h
    post = 1.0 + np.arange(1, k + 1) * h
    return np.concatenate([pre, interior, post]).astype(np.float64)


def _refit_to_canonical_periodic_8cp(cps_full, knots_full, k=3, n_cp=8, n_samples=64):
    """
    Convert a periodic B-spline tck of arbitrary CP/knot count into the canonical
    8-CP / 12-knot periodic form (last 3 CPs = first 3, uniform knots).

    Strategy:
      1. Sample the full tck on its valid u-domain [knots_full[k], knots_full[-k-1]].
      2. Fit canonical 5 independent CPs via least-squares against periodic basis.
      3. Return (knots_12, [cp_x_8, cp_y_8, cp_z_8]) with last 3 = first 3.
    """
    cps_full = [np.asarray(c, dtype=np.float64) for c in cps_full]
    knots_full = np.asarray(knots_full, dtype=np.float64)
    n_existing = len(cps_full[0])

    # Already canonical? (8 CPs, 12 knots, exact wrap) -> just normalize knots
    if n_existing == n_cp and len(knots_full) == n_cp + k + 1:
        cp_arr = np.column_stack(cps_full)        # (n_cp, 3)
        wrap_diff = float(np.max(np.abs(cp_arr[-k:] - cp_arr[:k])))
        if wrap_diff < 1e-9:
            return knots_full.astype(np.float32), [c.astype(np.float32) for c in cps_full]

    # Sample dense points on valid u domain
    u_lo = float(knots_full[k]); u_hi = float(knots_full[-k - 1])
    if not np.isfinite(u_lo) or not np.isfinite(u_hi) or u_hi <= u_lo:
        # Degenerate -- fall back to first-CP repeated
        c0 = [c[0] for c in cps_full]
        cps_out = [np.full(n_cp, c0[i], dtype=np.float32) for i in range(3)]
        knots_out = np.full(n_cp + k + 1, 1.0, dtype=np.float32)
        return knots_out, cps_out

    u = np.linspace(u_lo, u_hi, n_samples, endpoint=False)
    tck_full = (knots_full, cps_full, k)
    pts = np.array(splev(u, tck_full)).T                     # (n_samples, 3)

    # Build canonical knots and basis design matrix
    knots_can = _canonical_periodic_knots(n_cp=n_cp, k=k)    # (n_cp+k+1,) = 12
    n_indep = n_cp - k                                       # 5

    # Map u of full domain to canonical [0,1] linearly
    u_can = (u - u_lo) / (u_hi - u_lo)

    # Build (n_samples, n_indep) design matrix collapsing wrap columns onto first k
    A = np.zeros((n_samples, n_indep), dtype=np.float64)
    for i in range(n_cp):
        c_unit = np.zeros(n_cp); c_unit[i] = 1.0
        b_i = BSpline(knots_can, c_unit, k, extrapolate=False)(u_can)
        b_i = np.nan_to_num(b_i, nan=0.0)
        A[:, i % n_indep] += b_i

    # LSQ per axis
    coeffs_indep, *_ = np.linalg.lstsq(A, pts, rcond=None)   # (n_indep, 3)
    cps_full_canonical = np.vstack([coeffs_indep, coeffs_indep[:k]])  # (n_cp, 3)

    cps_out = [cps_full_canonical[:, i].astype(np.float32) for i in range(3)]
    return knots_can.astype(np.float32), cps_out


def limpiarRadiosSplines(radius, n_cp=8):
    """Flatten (xyz, [cp_x, cp_y, cp_z], knots) into a fixed-length feature vector.

    Refits non-8-CP / non-12-knot periodic splines to the canonical 8-CP form
    while preserving the periodic wrap (last 3 CPs = first 3 CPs).
    """
    n_knot = n_cp + 4
    k = 3
    cleaned = []

    # Step 1: Keep first 3 values (xyz center) as-is
    for i in range(3):
        cleaned.append(float(radius[i]))

    array_list = radius[3]
    knots_in = np.asarray(radius[4], dtype=np.float32)

    n_existing = len(np.asarray(array_list[0]))
    n_existing_knots = len(knots_in)

    # Detect collapsed/degenerate marker (all-1.0 knots from _degenerate_tck_at_point)
    if n_existing_knots == n_knot and np.allclose(knots_in, knots_in[0]):
        # Degenerate placeholder -- pass through untouched (truncate to be safe)
        for arr in array_list:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < n_cp:
                arr = np.pad(arr, (0, n_cp - len(arr)), mode='edge')
            cleaned.extend(arr[:n_cp])
        cleaned.extend(knots_in[:n_knot])
        return cleaned

    # Refit to canonical 8-CP / 12-knot periodic form
    try:
        knots_can, cps_can = _refit_to_canonical_periodic_8cp(
            array_list, knots_in, k=k, n_cp=n_cp, n_samples=64
        )
    except Exception:
        # Fallback: legacy edge-pad behavior
        knots_can = np.pad(knots_in, (0, max(0, n_knot - len(knots_in))), mode='edge')[:n_knot]
        cps_can = []
        for arr in array_list:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < n_cp:
                arr = np.pad(arr, (0, n_cp - len(arr)), mode='edge')
            cps_can.append(arr[:n_cp])

    for arr in cps_can:
        cleaned.extend(np.asarray(arr, dtype=np.float32)[:n_cp])
    cleaned.extend(np.asarray(knots_can, dtype=np.float32)[:n_knot])

    return cleaned

def binarizar(graph):
    for node in list(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 3:
            # Create a chain of intermediate nodes
            for i in range(num_neighbors - 2):
                new_node = f"{node}.{i}"
                radio_value = graph.nodes[node]['radio']  # Get the 'radio' attribute of the current node
                
                # Add the new node and edge
                graph.add_node(new_node, radio=radio_value)
                graph.add_edge(node, new_node)

            # Connect the last intermediate node to the original neighbors
            i=0 #cuenta intermedios
            c=0 #cuenta nodos
            #print("///////////")
            for vecino in neighbors:
                intermediate_node = f"{node}.{i}"
                #print("intermedio", intermediate_node)
                graph.add_edge(intermediate_node, vecino)
                c+=1
                if c>1:
                    i+=1
                    c=0
           
            # Remove the original edges
            graph.remove_edges_from([(node, neighbor) for neighbor in neighbors])


    for nodo in graph.nodes:
        if len(graph.edges(nodo))>3:
            print("bin", len(graph.edges(nodo)))
            break
    return graph


def calculate_splines(mesh, coef_folder, centerfolder, meshfolder, params=None):
    areas = []
    ratios = []
    params = params or {}
    fit_mode = params.get("spline_fit_mode", os.environ.get("VGP_SPLINE_MODE", "legacy"))
    if fit_mode not in {"legacy", "robust"}:
        fit_mode = "legacy"
    fallback_recenter = bool(params.get("spline_fallback_recenter", True))
    fill_missing_nearest = bool(params.get("spline_fill_missing_with_nearest_valid", False))
    max_center_drift = float(params.get("spline_max_center_drift", 2.0))
    # Use splitext: keep intermediate dots in UIDs like aneux_UPF_P0016.00_ID1
    f, _ext = os.path.splitext(mesh)
    str = f+".pkl"
    if _ext == ".vtp" and os.path.exists(centerfolder + f + "-network.vtp") and str not in os.listdir(coef_folder):

        centerline = read_vtp(centerfolder + f + "-network.vtp")
        mesh = read_vtp(meshfolder + f + ".vtp")

        # Slice the mesh and filter cross-sections
        n_total = 0
        for j in range(centerline.GetNumberOfCells()):
            n_total += centerline.GetCell(j).GetNumberOfPoints()
        splines = [None] * n_total
        knots = [None] * n_total
        areas = [np.nan] * n_total
        ratios = [np.nan] * n_total
        points_Acum = 0
        for j in range(centerline.GetNumberOfCells()):#calculate the radius by branch to avoid problems at the connections between branches

            numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()# number of points of the branch
            last_good_tck = None
            branch_start_idx = points_Acum
            branch_missing_idxs = []

            for i in range (numberOfCellPoints):
                idx = points_Acum
                point1 = centerline.GetPoint(points_Acum)

                tangent = np.zeros(3)

                weightSum = 0.0
                ##tangent line with the previous point (not calculated at the first point)
                if (i>0):
                    point0 = centerline.GetPoint(points_Acum-1)

                    distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1))
                    if distance > 1e-12:
                        ##vector between the two points divided by the distance
                        tangent[0] += (point1[0] - point0[0]) / distance
                        tangent[1] += (point1[1] - point0[1]) / distance
                        tangent[2] += (point1[2] - point0[2]) / distance
                        weightSum += 1.0

                ##tangent line with the next point (not calculated at the last one)
                if (i<numberOfCellPoints-1):
                    point0 = centerline.GetPoint(points_Acum+1)

                    distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1))
                    if distance > 1e-12:
                        tangent[0] += (point0[0] - point1[0]) / distance
                        tangent[1] += (point0[1] - point1[1]) / distance
                        tangent[2] += (point0[2] - point1[2]) / distance
                        weightSum += 1.0

                if weightSum > 0:
                    tangent[0] /= weightSum
                    tangent[1] /= weightSum
                    tangent[2] /= weightSum
                else:
                    tangent = np.array([0.0, 0.0, 1.0])
                plane = vtk.vtkPlane()
                plane.SetOrigin(point1)
                plane.SetNormal(tangent)
                tck = None
                perimeter = np.nan
                # Slice the mesh
                cutter = vtk.vtkCutter()
                cutter.SetCutFunction(plane)
                cutter.SetInputData(mesh)
                cutter.SetSortBy(1)
                cutter.Update()
                sliced_polydata = cutter.GetOutput()

                if sliced_polydata.GetNumberOfPoints() > 0:
                    # Filter to keep only the region closest to the centerline point
                    connectivityFilter = vtk.vtkConnectivityFilter()
                    connectivityFilter.SetInputData(sliced_polydata)
                    connectivityFilter.SetExtractionModeToClosestPointRegion()
                    connectivityFilter.SetClosestPoint(point1)  # Set the centerline point
                    connectivityFilter.Update()
                    filtered_polydata = connectivityFilter.GetOutput()

                    if filtered_polydata.GetNumberOfPoints() > 0:
                        # Extract filtered points and fit a spline
                        points = vtk_to_numpy(filtered_polydata.GetPoints().GetData())

                        # Triangulate the contour points to form a 2D surface
                        delaunay = vtk.vtkDelaunay2D()
                        delaunay.SetInputData(filtered_polydata)
                        delaunay.Update()

                        triangulated_surface = delaunay.GetOutput()
                        # Now calculate the surface area
                        mass = vtk.vtkMassProperties()
                        mass.SetInputData(triangulated_surface)
                        mass.Update()
                        area = mass.GetSurfaceArea()

                        #areas.append(area)
                        distancias =points-point1
                        normas = np.linalg.norm(distancias, axis=1)
                        ratio = np.min(normas)/np.max(normas)
                        ratios[idx] = ratio
                        if fit_mode == "robust":
                            if os.environ.get("VGP_SPLINE_DEBUG"):
                                if points.shape[0] < 8:
                                    print(
                                        f"[splines] {f}: slice {points_Acum - 1} has {points.shape[0]} raw contour points (<8)"
                                    )
                            # V2: Remove outlier contour points before fitting
                            points = _remove_contour_outliers(points, point1)
                            try:
                                n_resample = params.get("spline_fit_resample", 128)
                                if n_resample is None:
                                    n_resample = 128
                                min_points = params.get("spline_fit_min_points", 12)
                                if min_points is None:
                                    min_points = 12
                                max_retries = params.get("spline_fit_max_retries", 4)
                                if max_retries is None:
                                    max_retries = 4
                                nest = params.get("spline_fit_nest", 12)
                                if nest is None:
                                    nest = 12
                                s_scale = params.get("spline_fit_s_scale", 0.01)
                                if s_scale is None:
                                    s_scale = 0.01
                                retry_factor = params.get("spline_fit_retry_factor", 10.0)
                                if retry_factor is None:
                                    retry_factor = 10.0

                                tck = fit_splprep_token_fixed_8(
                                    contour_points=points,
                                    tangent_normal=tangent,
                                    s_initial=params.get("spline_fit_s"),
                                    n_resample=int(n_resample),
                                    nest=int(nest),
                                    max_retries=int(max_retries),
                                    min_points_for_resample=int(min_points),
                                    resample_only_if_needed=bool(
                                        params.get("spline_fit_resample_only_if_needed", True)
                                    ),
                                    canonical_start=bool(
                                        params.get("spline_fit_canonical_start", False)
                                    ),
                                    s_scale=float(s_scale),
                                    retry_factor=float(retry_factor),
                                    centerline_point=np.asarray(point1, dtype=np.float64),
                                    max_center_drift=float(max_center_drift),
                                )
                                # V2: Final validation
                                if not _validate_ring_tck(tck, point1, max_center_drift=max_center_drift):
                                    tck = None
                            except Exception:
                                tck = None
                        else:
                            x, y, z = points[:, 0], points[:, 1], points[:, 2]

                            centroid_x = np.mean(x)
                            centroid_y = np.mean(y)
                            centroid_z = np.mean(z)
                            angles = np.arctan2(y - centroid_y, x - centroid_x)

                            # Step 4: Sort the points by angle (angular order)
                            sorted_indices = np.argsort(angles)
                            x_sorted = x[sorted_indices]
                            y_sorted = y[sorted_indices]
                            z_sorted = z[sorted_indices]
                            points = np.vstack([x_sorted, y_sorted, z_sorted]).T
                            if os.environ.get("VGP_SPLINE_DEBUG"):
                                n_points = len(x_sorted)
                                if n_points < 8:
                                    print(
                                        f"[splines] {f}: slice {points_Acum - 1} has {n_points} contour points (<8)"
                                    )
                            try:
                                tck, u = splprep(
                                    [x_sorted, y_sorted, z_sorted], s=0.01, per=True, nest=12, k=3
                                )
                                # V2: Validate ring center for legacy mode too
                                if not _validate_ring_tck(tck, point1, max_center_drift=max_center_drift):
                                    tck = None
                            except Exception:
                                tck = None

                if tck is None:
                    if last_good_tck is not None:
                        if fallback_recenter:
                            tck = _translate_tck_to_point(last_good_tck, point1)
                        else:
                            tck = last_good_tck
                    elif not fill_missing_nearest:
                        tck = _degenerate_tck_at_point(point1)  # returns None near world origin

                if tck is None:
                    branch_missing_idxs.append(idx)
                else:
                    splines[idx] = tck[1]
                    knots[idx] = tck[0]
                    last_good_tck = tck
                    if not np.isfinite(perimeter):
                        perimeter = _estimate_spline_perimeter(tck, n_samples=200)
                    areas[idx] = perimeter

                points_Acum += 1

            if branch_missing_idxs:
                branch_end_idx = branch_start_idx + numberOfCellPoints
                valid_branch_idxs = [
                    branch_idx
                    for branch_idx in range(branch_start_idx, branch_end_idx)
                    if splines[branch_idx] is not None and knots[branch_idx] is not None
                ]

                for missing_idx in branch_missing_idxs:
                    target_point = np.asarray(centerline.GetPoint(missing_idx), dtype=np.float64)
                    if fill_missing_nearest and valid_branch_idxs:
                        nearest_idx = min(valid_branch_idxs, key=lambda valid_idx: abs(valid_idx - missing_idx))
                        nearest_tck = (
                            np.asarray(knots[nearest_idx], dtype=np.float64).copy(),
                            [np.asarray(arr, dtype=np.float64).copy() for arr in splines[nearest_idx]],
                            3,
                        )
                        if fallback_recenter:
                            filled_tck = _translate_tck_to_point(nearest_tck, target_point)
                        else:
                            filled_tck = nearest_tck
                        if filled_tck is None:
                            filled_tck = _degenerate_tck_at_point(target_point)
                    else:
                        filled_tck = _degenerate_tck_at_point(target_point)  # None near world origin

                    if filled_tck is not None:
                        splines[missing_idx] = filled_tck[1]
                        knots[missing_idx] = filled_tck[0]
                        areas[missing_idx] = _estimate_spline_perimeter(filled_tck, n_samples=200)
                    # else: leave None → treated as missing downstream

                            ######################################################################
        centerline_np = get_points_by_line(centerline)
        ##find centerline repeated points
        if centerline_np.ndim == 2 and centerline_np.shape[0] > 0 and centerline_np.shape[1] >= 4:
            try:
                splited = np.split(centerline_np, np.where(np.diff(centerline_np[:,3]))[0]+1)
                e = {}# to save every branch endpoint
                sum = 0
                for i in range(len(splited)):
                    rama = splited[i]
                    start = rama[0, :3]
                    e[sum] = tuple(start) #key is the point index, value coordinates
                    finish = rama[rama.shape[0]-1, :3]
                    sum += rama.shape[0]
                    e[sum-1] = tuple(finish)

                ##keep only the repeated endpoints
                b = np.array([key for key,  value in Counter(e.values()).items() if value > 1])


                ##list with the indexes of the repeated points
                key_list = []
                for element in b: #coordintaes of each repeated point
                    element = tuple(element)
                    for key,value in e.copy().items():
                        if element == value:#if the endpoint is on the repeated list I save the index
                            key_list.append(key)#key_list tiene los indices de los puntos repetidos

                k = {}
                ##dictionary with the indexes and coordinates of the repeated points
                for key in key_list:
                    k[key] = tuple(centerline_np[key,:3])

                ## join the points with the same coordinates, key are the coordinates and values list with the indexes
                res = {}
                for i, v in k.items():
                    res[v] = [i] if v not in res.keys() else res[v] + [i]


                for point in res:
                    pairs = [(i, areas[i]) for i in res[point] if i < len(areas) and np.isfinite(areas[i])]
                    if not pairs:
                        continue
                    min_i, _ = min(pairs, key=lambda x: x[1])
                    for index in res[point]:
                        if index < len(splines):
                            splines[index] = splines[min_i]#coordinates
                            knots[index] = knots[min_i]#np.full(12,0.)

                mean_area = np.nanmean(areas) if np.any(np.isfinite(areas)) else np.nan
                if np.isfinite(mean_area):
                    indices_greater_than_average = sorted(
                        [(i, element) for i, element in enumerate(areas) if np.isfinite(element) and element > 3 * mean_area],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                else:
                    indices_greater_than_average = []
                for index, area in indices_greater_than_average:
                    target_pt = centerline_np[index][:3]
                    # Find the nearest ring with a valid (non-blown-up) area and
                    # translate its shape to the current centerline point.
                    # This preserves a realistic cross-section instead of collapsing
                    # the ring to a degenerate point (which creates bead-chain artifacts).
                    upper = 3 * mean_area
                    valid_neighbors = [
                        i for i in range(len(splines))
                        if i != index
                        and splines[i] is not None
                        and knots[i] is not None
                        and np.isfinite(areas[i])
                        and areas[i] <= upper
                    ]
                    fixed_tck = None
                    if valid_neighbors:
                        nn = min(valid_neighbors, key=lambda i: abs(i - index))
                        src_tck = (
                            np.asarray(knots[nn], dtype=np.float64),
                            [np.asarray(a, dtype=np.float64) for a in splines[nn]],
                            3,
                        )
                        fixed_tck = _translate_tck_to_point(src_tck, target_pt)
                    if fixed_tck is not None:
                        splines[index] = fixed_tck[1]
                        knots[index] = fixed_tck[0]
                        areas[index] = _estimate_spline_perimeter(fixed_tck)
                    else:
                        # Last resort: degenerate at actual centerline point
                        fallback = _degenerate_tck_at_point(target_pt)
                        if fallback is not None:
                            splines[index] = fallback[1]
                            knots[index] = fallback[0]

            except Exception as e:
                print("EXCEPT")

                traceback.print_exc()
                pass
        elif os.environ.get("VGP_SPLINE_DEBUG"):
            print(f"[splines] {f}: skipping repeated-endpoint cleanup due to invalid centerline shape {centerline_np.shape}")

        knot_folder = coef_folder.replace("coeficientes", "knots")
        with open(knot_folder+ f +'.pkl', 'wb') as t:
            pickle.dump(knots, t)
        with open(coef_folder+ f +'.pkl', 'wb') as t:
            pickle.dump(splines, t)

def binarizar(graph):
    for node in list(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 3:
            # Create a chain of intermediate nodes
            for i in range(num_neighbors - 2):
                new_node = f"{node}.{i}"
                radio_value = graph.nodes[node]['radio']  # Get the 'radio' attribute of the current node
                
                # Add the new node and edge
                graph.add_node(new_node, radio=radio_value)
                graph.add_edge(node, new_node)

            # Connect the last intermediate node to the original neighbors
            i=0 #cuenta intermedios
            c=0 #cuenta nodos
            #print("///////////")
            for vecino in neighbors:
                intermediate_node = f"{node}.{i}"
                #print("intermedio", intermediate_node)
                graph.add_edge(intermediate_node, vecino)
                c+=1
                if c>1:
                    i+=1
                    c=0
           
            # Remove the original edges
            graph.remove_edges_from([(node, neighbor) for neighbor in neighbors])


    for nodo in graph.nodes:
        if len(graph.edges(nodo))>3:
            print("bin", len(graph.edges(nodo)))
            break
    return graph


def grafo2arbol(grafo):
    aRecorrer = []
    numeroNodoInicial = 1
    distancias = nx.floyd_warshall( grafo )

    parMaximo = (-1, -1)
    maxima = -1
                
    for nodoInicial in distancias.keys():
        for nodoFinal in distancias[nodoInicial]:
            if distancias[nodoInicial][nodoFinal] > maxima:
                maxima = distancias[nodoInicial] [nodoFinal]
                parMaximo = (nodoInicial, nodoFinal)
            
    for nodo in grafo.nodes:
        if distancias[parMaximo[0]][nodo] == int( maxima / 2):
            numeroNodoInicial = nodo
            if len(grafo.edges(numeroNodoInicial))>2:
                numeroNodoInicial = list(grafo.edges(numeroNodoInicial))[0][1]
            break
            
    rad = list(grafo.nodes[numeroNodoInicial]['radio'])
    nodoRaiz = Node( numeroNodoInicial, radius =  rad )
    for vecino in grafo.neighbors( numeroNodoInicial ):
        if vecino != numeroNodoInicial:
            aRecorrer.append( (vecino, numeroNodoInicial,nodoRaiz ) )
    while len(aRecorrer) != 0:
        nodoAAgregar, numeroNodoPadre,nodoPadre = aRecorrer.pop(0)
        radius = list(grafo.nodes[nodoAAgregar]['radio'])
    
        nodoActual = Node( nodoAAgregar, radius =  radius)
        nodoPadre.agregarHijo( nodoActual )
        for vecino in grafo.neighbors( nodoAAgregar ):
            if vecino != numeroNodoPadre:
                aRecorrer.append( (vecino, nodoAAgregar,nodoActual) )

    serial = nodoRaiz.serialize(nodoRaiz)
    
    return serial

def _remove_contour_outliers(points: np.ndarray, centerline_point: np.ndarray,
                              max_dist_factor: float = 3.0) -> np.ndarray:
    """Remove contour points that are too far from the centroid.

    Some VTK slices pick up geometry from adjacent branches.  We detect
    these outliers as points whose distance to the centroid exceeds
    *max_dist_factor* × the median radius, and also reject any point
    further from *centerline_point* than 2× the median distance.
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 6:
        return P

    c = P.mean(axis=0)
    dists_to_c = np.linalg.norm(P - c, axis=1)
    med_r = np.median(dists_to_c)
    if med_r < 1e-10:
        return P

    # 1) Remove points whose distance to centroid is > max_dist_factor × median
    mask = dists_to_c < max_dist_factor * med_r

    # 2) Also remove points far from centerline point to avoid cross-branch hits
    cl = np.asarray(centerline_point, dtype=np.float64).ravel()
    dists_to_cl = np.linalg.norm(P - cl, axis=1)
    med_cl = np.median(dists_to_cl)
    mask &= dists_to_cl < max(3.0 * med_cl, 3.0 * med_r)

    P_clean = P[mask]
    return P_clean if len(P_clean) >= 6 else P


def _validate_ring_tck(tck, centerline_point: np.ndarray,
                       max_center_drift: float = 2.0,
                       n_eval: int = 64) -> bool:
    """Check if a fitted B-spline ring is plausible.

    Returns False if the ring center has drifted too far from the
    centerline point (relative to the ring radius), which indicates
    the contour captured geometry from a neighbouring branch.
    """
    if tck is None:
        return False
    try:
        u = np.linspace(0.0, 1.0, n_eval)
        pts = np.column_stack(splev(u, tck))
        ring_center = pts.mean(axis=0)
        ring_r = np.median(np.linalg.norm(pts - ring_center, axis=1))
        cl = np.asarray(centerline_point, dtype=np.float64).ravel()
        drift = np.linalg.norm(ring_center - cl)
        # Allow drift up to max_center_drift × radius (or a small absolute min)
        return drift < max(max_center_drift * ring_r, 0.02)
    except Exception:
        return False


def _plane_basis_from_normal(n: np.ndarray):
    n = np.asarray(n, float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
        nn = 1.0
    n = n / nn

    # deterministisch: Normal Richtung stabilisieren
    if np.dot(n, np.array([0.0, 0.0, 1.0])) < 0:
        n = -n

    gx = np.array([1.0, 0.0, 0.0])
    u = gx - np.dot(gx, n) * n
    if np.linalg.norm(u) < 1e-8:
        gy = np.array([0.0, 1.0, 0.0])
        u = gy - np.dot(gy, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v, n

def _sort_contour_in_plane(points: np.ndarray, plane_normal: np.ndarray, canonical_start=True):
    P = np.asarray(points, float)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        return P

    # dedupe
    Pr = np.round(P, 6)
    _, idx = np.unique(Pr, axis=0, return_index=True)
    P = P[np.sort(idx)]
    if len(P) < 3:
        return P

    u, v, _ = _plane_basis_from_normal(plane_normal)
    c = P.mean(axis=0)
    D = P - c
    x2 = D @ u
    y2 = D @ v
    ang = np.arctan2(y2, x2)
    order = np.argsort(ang)
    P = P[order]

    # konsistente Orientierung (CCW)
    x2s, y2s = x2[order], y2[order]
    signed_area = 0.5 * np.sum(x2s * np.roll(y2s, -1) - y2s * np.roll(x2s, -1))
    if signed_area < 0:
        P = P[::-1]
        x2s = x2s[::-1]

    # canonical start point (for token consistency)
    if canonical_start:
        start = int(np.argmax(x2s))
        P = np.roll(P, -start, axis=0)
    return P

def _resample_closed_polyline(P: np.ndarray, n_resample: int = 128):
    # P is already sorted (Nx3); treat it as a closed contour
    if len(P) < 3:
        return P

    P_closed = np.vstack([P, P[0]])
    seg = np.linalg.norm(np.diff(P_closed, axis=0), axis=1)
    total = float(np.sum(seg))
    if not np.isfinite(total) or total < 1e-9:
        return P

    s = np.concatenate([[0.0], np.cumsum(seg)])  # (N+1,)
    s_target = np.linspace(0.0, total, n_resample, endpoint=False)

    x = np.interp(s_target, s, P_closed[:, 0])
    y = np.interp(s_target, s, P_closed[:, 1])
    z = np.interp(s_target, s, P_closed[:, 2])
    return np.column_stack([x, y, z])


def _degenerate_tck_at_point(p: np.ndarray, n_cp=8):
    """Build a collapsed (degenerate) ring B-spline at point *p*.

    Returns ``None`` when *p* is at or extremely close to the world origin
    (‖p‖ < 1e-6), because that almost certainly means the upstream
    centerline extractor returned an invalid/uninitialised point.  Returning
    ``None`` lets the caller fall back to the nearest-valid-ring strategy
    instead of injecting a phantom blob at [0, 0, 0].
    """
    n_knot = n_cp + 4
    p = np.asarray(p, dtype=np.float64).ravel()
    if np.linalg.norm(p) < 1e-6:
        return None
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    coeffs = [
        np.full(n_cp, x),
        np.full(n_cp, y),
        np.full(n_cp, z),
    ]
    knots = np.full(n_knot, 1.0)
    return (knots, coeffs, 3)


def _translate_tck_to_point(tck, target_point, n_samples=64):
    """
    Shift an existing spline (tck) so its contour center matches target_point.
    Shape and knots stay unchanged (no refit).
    """
    if tck is None:
        return None
    try:
        knots, coeffs, deg = tck
        c = [np.asarray(arr, dtype=np.float64).copy() for arr in coeffs]
        if len(c) < 3:
            return tck

        u = np.linspace(0.0, 1.0, int(max(8, n_samples)))
        x, y, z = splev(u, (np.asarray(knots, dtype=np.float64), c, int(deg)))
        center = np.array([np.mean(x), np.mean(y), np.mean(z)], dtype=np.float64)
        target = np.asarray(target_point, dtype=np.float64).reshape(3)
        delta = target - center

        c[0] += delta[0]
        c[1] += delta[1]
        c[2] += delta[2]
        return (np.asarray(knots, dtype=np.float64).copy(), c, int(deg))
    except Exception:
        return tck

def _estimate_spline_perimeter(tck, n_samples=200):
    try:
        u_fine = np.linspace(0.0, 1.0, int(max(8, n_samples)))
        spline_points = np.array(splev(u_fine, tck))
        points = np.column_stack(spline_points)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return float(np.sum(distances))
    except Exception:
        return np.nan

def fit_splprep_token_fixed_8(
    contour_points: np.ndarray,
    tangent_normal: np.ndarray,
    s_initial: float | None = None,
    n_resample: int = 128,
    nest: int = 12,
    max_retries: int = 4,
    min_points_for_resample: int = 12,
    resample_only_if_needed: bool = True,
    canonical_start: bool = False,
    s_scale: float = 0.01,
    retry_factor: float = 10.0,
    centerline_point: np.ndarray | None = None,
    max_center_drift: float = 2.0,
):
    """
    Robust path: sort + resample the contour, then splprep with k=3,
    per=True, nest=12. On warning/error, s is increased automatically
    (more smoothing).

    V2 improvements:
    - Outlier removal from contour before fitting
    - Post-fit validation: reject ring if center drifts too far from centerline
    """
    # V2: Remove outlier points before sorting/fitting
    if centerline_point is not None:
        contour_points = _remove_contour_outliers(contour_points, centerline_point)

    P = _sort_contour_in_plane(contour_points, tangent_normal, canonical_start=canonical_start)
    if resample_only_if_needed:
        if len(P) < min_points_for_resample:
            P = _resample_closed_polyline(P, n_resample=n_resample)
    else:
        P = _resample_closed_polyline(P, n_resample=n_resample)

    if len(P) < 8:
        raise ValueError(f"Zu wenige Konturpunkte nach Resampling: {len(P)} (<8)")

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    # Scale-aware start for s if not provided.
    if s_initial is None:
        c = P.mean(axis=0)
        r = np.median(np.linalg.norm(P - c, axis=1))
        s = (s_scale * r) ** 2 * len(P)
    else:
        s = float(s_initial)

    # Retry strategy: increase s when splprep fails.
    last_err = None
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                tck, u = splprep([x, y, z], k=3, per=True, s=s, nest=nest)

            # V2: Post-fit validation — reject if ring center drifted
            if centerline_point is not None:
                if not _validate_ring_tck(tck, centerline_point,
                                         max_center_drift=max_center_drift):
                    # Try with tighter smoothing to get a better fit
                    s *= retry_factor
                    last_err = ValueError("Ring center drifted too far from centerline")
                    continue

            return tck  # tck = (knots, coeffs, k)
        except Exception as e:
            last_err = e
            s *= retry_factor
            continue

    raise RuntimeError(f"splprep failed even after retries. Last error: {last_err}")
