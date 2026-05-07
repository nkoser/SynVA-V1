from functools import partial
from multiprocessing.pool import ThreadPool
from skimage import measure

import multiprocessing
import itertools
import numpy as np
import time

from . import progress, stl



import itertools
from multiprocessing import Pool  # Changed from ThreadPool

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 8#32

def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts[faces].reshape((-1, 3))

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def _skip(sdf, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    r = abs(sdf(np.array([(x, y, z)])).reshape(-1)[0])
    d = np.linalg.norm(np.array((x-x0, y-y0, z-z0)))
    if r <= d:
        return False
    corners = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    values = sdf(corners).reshape(-1)
    same = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
    return same

def _worker(sdf, job, step, sparse):
    X, Y, Z = job
    if sparse and _skip(sdf, job):
        return None
        # return _debug_triangles(X, Y, Z)
    P = _cartesian_product(X, Y, Z)
    shape = (len(X), len(Y), len(Z))
    volume = sdf(P).reshape(shape)
    try:
        points = _marching_cubes(volume)
    except Exception:
        return []
        # return _debug_triangles(X, Y, Z)
    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    return points * scale + offset

def _estimate_bounds(sdf):
    # TODO: raise exception if bound estimation fails
    s = 16
    x0 = y0 = z0 = -1e9
    x1 = y1 = z1 = 1e9
    prev = None
    for i in range(32):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        Z = np.linspace(z0, z1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = _cartesian_product(X, Y, Z)
        volume = sdf(P).reshape((len(X), len(Y), len(Z)))
        where = np.argwhere(np.abs(volume) <= threshold)
        x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
        x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
    return ((x0, y0, z0), (x1, y1, z1))

def generate(
        sdf,
        step=None, bounds=None, samples=SAMPLES,
        workers=WORKERS, batch_size=BATCH_SIZE,
        verbose=True, sparse=True):

    #version martin
    start = time.time()

    #calculate bounds if not provided
    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    #If no step is given, it computes it from the total volume divided by desired number of samples.
    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print('min %g, %g, %g' % (x0, y0, z0))
        print('max %g, %g, %g' % (x1, y1, z1))
        print('step %g, %g, %g' % (dx, dy, dz))

    #grid creation
    #Creates 1D arrays of coordinates for each axis using the step size.
    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    #Break each axis into chunks of batch_size, forming smaller 3D grids.
    s = batch_size
    Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
    Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

    #Creates all combinations of Xs, Ys, Zs to form batches of 3D cubes.
    batches = list(itertools.product(Xs, Ys, Zs))
    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs)
        for xs, ys, zs in batches)

    if verbose:
        print('%d samples in %d batches with %d workers' %
            (num_samples, num_batches, workers))

    points = []
    skipped = empty = nonempty = 0
    bar = progress.Bar(num_batches, enabled=verbose)

    #A thread pool runs _worker() on each batch, applying the SDF and filtering/sampling points based on distance.
    #The results are collected and counted for skipped, empty, and nonempty batches.
    pool = ThreadPool(workers)
    f = partial(_worker, sdf, step=(dx, dy, dz), sparse=sparse)
    for result in pool.imap(f, batches): #Gathers all valid points (likely near the surface, i.e., where SDF ≈ 0).
        bar.increment(1)
        if result is None:
            skipped += 1
        elif len(result) == 0:
            empty += 1
        else:
            nonempty += 1
            points.extend(result)
    bar.done()

    if verbose:
        print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))
        triangles = len(points) // 3
        seconds = time.time() - start
        print('%d triangles in %g seconds' % (triangles, seconds))

    return points, bounds #points: All sampled points that met the criteria. bounds: The 3D boundaries used.

def save(path, *args, **kwargs):
    points, bounds = generate(*args, **kwargs)
    lower = path.lower()
    if lower.endswith('.stl'):
        stl.write_binary_stl(path, points)
    elif lower.endswith('.vtp'):
        _write_vtp(path, points)
    else:
        mesh = _mesh(points)
        mesh.write(path)
    return bounds

def _mesh(points):
    import meshio
    points, cells = np.unique(points, axis=0, return_inverse=True)
    cells = [('triangle', cells.reshape((-1, 3)))]
    return meshio.Mesh(points, cells)


def _write_vtp(path, points):
    import vtk
    pts, cells = np.unique(points, axis=0, return_inverse=True)
    cells = cells.reshape((-1, 3))

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(len(pts))
    for i, p in enumerate(pts):
        vtk_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    triangles = vtk.vtkCellArray()
    for a, b, c in cells:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(a))
        tri.GetPointIds().SetId(1, int(b))
        tri.GetPointIds().SetId(2, int(c))
        triangles.InsertNextCell(tri)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(triangles)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.Write()

def _debug_triangles(X, Y, Z):
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]

    p = 0.25
    x0, x1 = x0 + (x1 - x0) * p, x1 - (x1 - x0) * p
    y0, y1 = y0 + (y1 - y0) * p, y1 - (y1 - y0) * p
    z0, z1 = z0 + (z1 - z0) * p, z1 - (z1 - z0) * p

    v = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return [
        v[3], v[5], v[7],
        v[5], v[3], v[1],
        v[0], v[6], v[4],
        v[6], v[0], v[2],
        v[0], v[5], v[1],
        v[5], v[0], v[4],
        v[5], v[6], v[7],
        v[6], v[5], v[4],
        v[6], v[3], v[7],
        v[3], v[6], v[2],
        v[0], v[3], v[2],
        v[3], v[0], v[1],
    ]



def generate2(
        sdf,
        step=None, bounds=None, samples=SAMPLES,
        workers=WORKERS, batch_size=BATCH_SIZE,
        verbose=True, sparse=True):
    #version gpt
    start = time.time()

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print(f'min {x0:.3f}, {y0:.3f}, {z0:.3f}')
        print(f'max {x1:.3f}, {y1:.3f}, {z1:.3f}')
        print(f'step {dx:.3f}, {dy:.3f}, {dz:.3f}')

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    # ⬇️ Batching: coarser granularity (optional, adjust s for perf/memory)
    s = batch_size
    Xs = [X[i:i + s + 1] for i in range(0, len(X), s)]
    Ys = [Y[i:i + s + 1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i + s + 1] for i in range(0, len(Z), s)]

    # ⬇️ Use generator instead of list() to avoid memory + CPU waste
    batches = itertools.product(Xs, Ys, Zs)

    # ⬇️ Estimate number of batches (for verbose or bar)
    num_batches = (len(Xs) * len(Ys) * len(Zs))
    num_samples = sum(len(xs) * len(ys) * len(zs) for xs in Xs for ys in Ys for zs in Zs)

    if verbose:
        print(f'{num_samples} samples in ~{num_batches} batches with {workers} workers')

    points = []
    skipped = empty = nonempty = 0

    # ⬇️ Only create progress bar if verbose
    bar = progress.Bar(num_batches, enabled=verbose) if verbose else None

    # ⬇️ Use multiprocessing Pool instead of ThreadPool
    f = partial(_worker, sdf, step=(dx, dy, dz), sparse=sparse)
    with Pool(processes=workers) as pool:
        for result in pool.imap(f, batches):
            if bar:
                bar.increment(1)
            if result is None:
                skipped += 1
            elif len(result) == 0:
                empty += 1
            else:
                nonempty += 1
                points.extend(result)

    if bar:
        bar.done()

    if verbose:
        print(f'{skipped} skipped, {empty} empty, {nonempty} nonempty')
        triangles = len(points) // 3
        seconds = time.time() - start
        print(f'{triangles} triangles in {seconds:.2f} seconds')

    return points, bounds


def sample_slice(
        sdf, w=1024, h=1024,
        x=None, y=None, z=None, bounds=None):

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if x is not None:
        X = np.array([x])
        Y = np.linspace(y0, y1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], Y[0], Y[-1])
        axes = 'ZY'
    elif y is not None:
        Y = np.array([y])
        X = np.linspace(x0, x1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], X[0], X[-1])
        axes = 'ZX'
    elif z is not None:
        Z = np.array([z])
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
        extent = (Y[0], Y[-1], X[0], X[-1])
        axes = 'YX'
    else:
        raise Exception('x, y, or z position must be specified')

    P = _cartesian_product(X, Y, Z)
    return sdf(P).reshape((w, h)), extent, axes

def show_slice(*args, **kwargs):
    import matplotlib.pyplot as plt
    show_abs = kwargs.pop('abs', False)
    a, extent, axes = sample_slice(*args, **kwargs)
    if show_abs:
        a = np.abs(a)
    im = plt.imshow(a, extent=extent, origin='lower')
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.colorbar(im)
    plt.show()
