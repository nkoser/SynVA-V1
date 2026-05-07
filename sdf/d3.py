import functools
import numpy as np
import math
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree, cKDTree
from numba import njit

# import operator

from . import core, dn, d2, ease

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}


class SDF3:
    def __init__(self, f):
        self.f = f

    def __call__(self, p):
        return self.f(p).reshape((-1, 1))

    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        return getattr(self.f, name)
        raise AttributeError

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def k(self, k=None):
        self._k = k
        return self

    def generate(self, *args, **kwargs):
        return core.generate(self, *args, **kwargs)

    def save(self, path, *args, **kwargs):
        return core.save(path, self, *args, **kwargs)

    def show_slice(self, *args, **kwargs):
        return core.show_slice(self, *args, **kwargs)


def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    return wrapper


def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


@op3
def smooth_union(a, b, k=0.05):
    def f(p):
        d1 = a(p).reshape(-1)
        d2 = b(p).reshape(-1)
        k_ = float(k)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / k_, 0.0, 1.0)
        d = d2 * (1.0 - h) + d1 * h - k_ * h * (1.0 - h)
        return d.reshape((-1, 1))
    return f


# Helpers

def _length(a):
    return np.linalg.norm(a, axis=1)


def _normalize(a):
    return a / np.linalg.norm(a)


def _dot(a, b):
    return np.sum(a * b, axis=1)


def _vec(*arrs):
    return np.stack(arrs, axis=-1)


def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


_min = np.minimum
_max = np.maximum


# Primitives

@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - center) - radius

    return f


@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize(normal)

    def f(p):
        return np.dot(point - p, normal)

    return f


@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)


@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)

    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

    return f


@sdf3
def rounded_box(size, radius):
    size = np.array(size)

    def f(p):
        q = np.abs(p) - size / 2 + radius
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - radius

    return f


@sdf3
def wireframe_box(size, thickness):
    size = np.array(size)

    def g(a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)

    def f(p):
        p = np.abs(p) - size / 2 - thickness / 2
        q = np.abs(p + thickness / 2) - thickness / 2
        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))

    return f


@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[:, [0, 1]]
        z = p[:, 2]
        a = _length(xy) - r1
        b = _length(_vec(a, z)) - r2
        return b

    return f


@sdf3
def capsule(a, b, radius):
    a = np.array(a)
    b = np.array(b)

    def f(p):
        pa = p - a
        ba = b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - radius

    return f


@sdf3
def cylinder(radius):
    def f(p):
        return _length(p[:, [0, 1]]) - radius;

    return f


@sdf3
def capped_cylinder(a, b, radius):
    a = np.array(a)
    b = np.array(b)

    def f(p):
        ba = b - a
        pa = p - a
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

    return f


def calculate_angle(a, b, p):
    ##version martin submission
    ba = b - a
    pa = p - a
    ba_len = np.linalg.norm(ba)
    if ba_len == 0:
        return None
    ba_norm = ba / ba_len

    pa_proj = pa - (np.dot(pa, ba_norm).reshape((-1, 1)) * ba_norm)

    arbitrary = np.array([0, 0, 1]) if abs(ba_norm[2]) < 0.9 else np.array([1, 0, 0])

    u = np.cross(ba_norm, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(ba_norm, u)

    x_proj = np.dot(pa_proj, u)
    y_proj = np.dot(pa_proj, v)
    angle = np.arctan2(y_proj, x_proj)
    if not np.isfinite(angle).all():
        return None

    # Normalize angle to [0, 1]
    normalized_angle = (angle + np.pi) / (2 * np.pi)

    return normalized_angle


def poly(x, coefficients):
    return np.polyval(coefficients, x)


def find_closest_t(tck, P):
    """
    Finds parameter t in [0, 1] for the closest point on the spline to P.
    tck: spline representation
    P: point in space (3,)
    """

    def objective(t):
        C = np.array(splev(t, tck))
        return np.sum((P - C) ** 2)

    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x  # t* that minimizes distance


def sample_spline(coeffs, n_samples):
    coeffs = list(coeffs)

    t = np.array(coeffs[24:])
    c = [np.array(coeffs[i * 8:(i * 8) + 8]) for i in range(3)]

    if np.abs(np.mean(t) - 1) < 1e-1: return None

    tck = (t, c, 3)

    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)

    return np.column_stack((x, y, z))


@sdf3
def vessel3(tree_points, points, splines, tck=None, sampled_spline=None, t_values=None, centerline_t_mode="optimize"):
    """
    Original legacy implementation (kept for compatibility).
    Uses polyfit on sampled contours and radius interpolation.
    """

    # Fit spline to the provided points (unless provided)
    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    def distance_to_spline(t, p):
        """
        Computes Euclidean distance between point p and spline at parameter t.
        """
        spline_point = np.array(splev(t, tck))
        return np.linalg.norm(p - spline_point)

    def find_minmax(t, l):
        # version martin
        min_t = l[0][0]
        min_i = 0

        for i in range(len(l)):

            if l[i][0] >= min_t and t >= l[i][0]:
                min_t = l[i][0]
                min_i = i

        return min_t, min_i

    # Sample the spline at `n_samples` points (unless provided)
    if sampled_spline is None:
        n_samples = 100
        t_values = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T  # (n_samples, 3)
    else:
        n_samples = sampled_spline.shape[0]
        if t_values is None:
            t_values = np.linspace(0, 1, n_samples)

    # Build a KDTree for fast nearest neighbor search
    kdtree = KDTree(sampled_spline)

    # Radius splines
    splines_sampled = []
    coeffs = []

    for i in range(len(points)):

        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=50)

        if type(spline_points) == type(None):

            if len(splines_sampled) > 0:
                spline_points = splines_sampled[-1].copy()
                center = points[i - 1]
            else:
                assert ("/")

        distances = np.linalg.norm(spline_points - center, axis=1)
        # print(np.mean(distances))

        splines_sampled.append(spline_points.copy())

        xs = np.linspace(0, 1, 50)
        coeff = np.polyfit(xs, distances, 5)

        if centerline_t_mode == "kdtree":
            _, idx = kdtree.query(center)
            t = float(t_values[idx])
        else:
            t = minimize_scalar(distance_to_spline, bounds=(0, 1), args=(center,),
                                method='bounded').x
        # _, nearest_idx = kdtree.query(center)
        # Compute the corresponding parameter t
        # t = nearest_idx / len(sampled_spline)

        if i == len(points) - 1: t = 1.0

        coeffs.append((t, coeff))

    def f(P):
        """
        Computes signed distance and closest t for a batch of points P.
        P: np.array of shape (N, 3)
        Returns:
            sdf_values: np.array of shape (N,)
        """
        if P.ndim == 1:
            P = P[np.newaxis, :]  # Handle single point input

        sdf_values = []

        for p in P:

            _, nearest_idx = kdtree.query(p)
            nearest_points = sampled_spline[nearest_idx]

            # Minimize distance_to_spline over t in [0,1]

            # res = minimize_scalar(distance_to_spline, bounds=(0, 1), args=(p,), method='bounded')
            # t = res.x
            # min_dist = res.fun

            min_dist = np.linalg.norm(p - nearest_points)
            t = nearest_idx / n_samples
            t2 = t

            if t * 1.05 > 1.0:
                t2 = t * 0.95
            else:
                t2 = t * 1.05

            a = np.array(splev(t, tck))
            b = np.array(splev(t2, tck))

            angle = calculate_angle(a, b, p)

            if type(angle) == type(None):
                angle = np.array([0.])

            poly_t, poly_i = find_minmax(t, coeffs)

            size = coeffs[poly_i + 1][0] - coeffs[poly_i][0]
            t_delta = (t - poly_t) / size

            radius_start = 0
            radius_end = 0

            radius_start = poly(angle, coeffs[poly_i][1])
            radius_end = poly(angle, coeffs[poly_i + 1][1])

            radius = (1 - t_delta) * radius_start + t_delta * radius_end

            # Apply tube radius for signed distance
            sdf = min_dist - radius
            sdf_values.append(sdf)

        return np.array(sdf_values)

    return f


@sdf3
def vessel3_robust(tree_points, points, splines, tck=None, sampled_spline=None, t_values=None, centerline_t_mode="optimize"):
    """
    Safer variant of legacy:
    - angle-binned radius tables (avoids polyfit overshoot)
    - radial distance w.r.t. centerline (not full point distance)
    - fallbacks when splines are missing/degenerate
    """

    # Tunable constants (kept internal to avoid changing callers)
    n_centerline_samples = 200
    n_profile_samples = 80
    n_angle_bins = 256
    radius_percentile = 90.0  # robust scalar fallback
    delta_t = 1.0 / n_centerline_samples

    # Fit spline to the provided points (unless provided)
    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    def distance_to_spline(t, p):
        """
        Computes Euclidean distance between point p and spline at parameter t.
        """
        spline_point = np.array(splev(t, tck))
        return np.linalg.norm(p - spline_point)

    def centerline_frame(t):
        """Return centerline point and a nearby point to define a local frame."""
        t_clamped = min(max(t, 0.0), 1.0)
        t2 = min(1.0, max(0.0, t_clamped + delta_t))
        a = np.array(splev(t_clamped, tck))
        b = np.array(splev(t2, tck))
        return a, b

    # Sample the spline at `n_centerline_samples` points (unless provided)
    if sampled_spline is None:
        t_values = np.linspace(0, 1, n_centerline_samples)
        sampled_spline = np.array(splev(t_values, tck)).T  # (M, 3)
    else:
        if t_values is None:
            t_values = np.linspace(0, 1, sampled_spline.shape[0])
        n_centerline_samples = sampled_spline.shape[0]
    tangents = np.array(splev(t_values, tck, der=1)).T
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-12] = 1.0
    tangents = tangents / tangent_norm

    # Build a KDTree for fast nearest neighbor search
    kdtree = KDTree(sampled_spline)

    def build_radius_table(center, t_node, spline_points):
        """Create angle->radius table (median per bin, interpolated)."""
        a, b = centerline_frame(t_node)
        angles = []
        radii = []
        for pt in spline_points:
            angle = calculate_angle(a, b, pt)
            if angle is None:
                continue
            r = np.linalg.norm(pt - center)
            if np.isfinite(r):
                angles.append(float(angle))
                radii.append(float(r))

        if len(radii) == 0:
            return None, 0.0

        radii = np.array(radii, dtype=np.float32)
        scalar = float(np.percentile(radii, radius_percentile))

        table = np.full(n_angle_bins, np.nan, dtype=np.float32)
        bins = (np.array(angles) * n_angle_bins).astype(int) % n_angle_bins
        for bin_idx in range(n_angle_bins):
            mask = bins == bin_idx
            if np.any(mask):
                table[bin_idx] = np.median(radii[mask])

        valid = np.isfinite(table)
        if valid.sum() >= 2:
            x = np.where(valid)[0]
            y = table[valid]
            x_ext = np.concatenate([x, [x[0] + n_angle_bins]])
            y_ext = np.concatenate([y, [y[0]]])
            xi = np.arange(n_angle_bins)
            table = np.interp(xi, x_ext, y_ext).astype(np.float32)
        else:
            table[:] = scalar
        return table, scalar

    def table_lookup(table, angle01, default_radius):
        if table is None:
            return default_radius
        a = float(angle01) % 1.0
        pos = a * len(table)
        i0 = int(np.floor(pos)) % len(table)
        i1 = (i0 + 1) % len(table)
        w = pos - np.floor(pos)
        val = (1.0 - w) * float(table[i0]) + w * float(table[i1])
        if not np.isfinite(val):
            return default_radius
        return val

    # Radius profiles per node: (t_node, table, scalar_radius)
    profiles = []
    last_profile = None

    for i in range(len(points)):

        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=n_profile_samples)

        if spline_points is None or not np.all(np.isfinite(spline_points)):
            if last_profile is not None:
                profiles.append(last_profile)
                continue
            else:
                # degenerate root: create a tiny circle
                spline_points = np.repeat(center[np.newaxis, :], n_profile_samples, axis=0)

        if centerline_t_mode == "kdtree":
            _, idx = kdtree.query(center)
            t = float(t_values[idx])
        else:
            t = minimize_scalar(distance_to_spline, bounds=(0, 1), args=(center,), method="bounded").x
        if i == len(points) - 1:
            t = 1.0

        table, scalar = build_radius_table(center, t, spline_points)
        profile = (float(t), table, float(scalar))
        profiles.append(profile)
        last_profile = profile

    if len(profiles) == 0:
        # no profiles, return empty SDF
        def empty(p):
            return np.ones((p.shape[0],), dtype=np.float32)

        return empty

    def f(P):
        """
        Computes signed distance and closest t for a batch of points P.
        P: np.array of shape (N, 3)
        Returns:
            sdf_values: np.array of shape (N,)
        """
        if P.ndim == 1:
            P = P[np.newaxis, :]  # Handle single point input

        sdf_values = np.zeros(P.shape[0], dtype=np.float32)
        ts = np.array([p[0] for p in profiles], dtype=np.float32)
        single_profile = len(profiles) == 1

        # nearest centerline sample for all points (vectorized)
        _, nearest_idx = kdtree.query(P)
        nearest_idx = np.array(nearest_idx, dtype=int)

        for k, p in enumerate(P):
            idx = nearest_idx[k]
            c_pt = sampled_spline[idx]
            # use euclidean distance to avoid tangential "ghost" tubes
            radial_dist = np.linalg.norm(p - c_pt)

            t = float(t_values[idx])
            t2 = min(1.0, t + delta_t) if t + delta_t <= 1.0 else max(0.0, t - delta_t)
            a = np.array(splev(t, tck))
            b = np.array(splev(t2, tck))

            angle = calculate_angle(a, b, p)
            if angle is None:
                angle = 0.0

            if single_profile:
                t0, tab0, s0 = profiles[0]
                r0 = table_lookup(tab0, angle, s0)
                radius = r0
            else:
                # choose segment
                i = int(np.searchsorted(ts, t, side="right") - 1)
                i = max(0, min(i, len(profiles) - 2))
                t0, tab0, s0 = profiles[i]
                t1, tab1, s1 = profiles[i + 1]
                denom = (t1 - t0)
                alpha = 0.0 if denom <= 1e-12 else (t - t0) / denom
                alpha = max(0.0, min(1.0, alpha))

                r0 = table_lookup(tab0, angle, s0)
                r1 = table_lookup(tab1, angle, s1)
                radius = (1.0 - alpha) * r0 + alpha * r1

            sdf_values[k] = radial_dist - radius

        return sdf_values

    return f


@sdf3
def vessel3_stable(
    tree_points,
    points,
    splines,
    radius_mode="median",
    radius_percentile=90,
    radius_cap=None,
    center_mode="node",
    fallback_radius=0.0,
    tck=None,
    sampled_spline=None,
    t_values=None,
    centerline_t_mode="optimize",
):
    """
    Stable variant of vessel3:
    - Uses robust radius statistics (median/percentile/mean/max).
    - Optional centroid-based radius computation.
    - Avoids polynomial overfitting on small branches.
    """
    if tck is None:
        tck, _ = splprep(tree_points.T, s=0)

    def distance_to_spline(t, p):
        spline_point = np.array(splev(t, tck))
        return np.linalg.norm(p - spline_point)

    def find_minmax(t, l):
        min_t = l[0][0]
        min_i = 0
        for i in range(len(l)):
            if l[i][0] >= min_t and t >= l[i][0]:
                min_t = l[i][0]
                min_i = i
        return min_t, min_i

    n_samples = 100
    if sampled_spline is None:
        t_values = np.linspace(0, 1, n_samples)
        sampled_spline = np.array(splev(t_values, tck)).T
    else:
        n_samples = sampled_spline.shape[0]
        if t_values is None:
            t_values = np.linspace(0, 1, n_samples)
    kdtree = KDTree(sampled_spline)

    splines_sampled = []
    coeffs = []

    for i in range(len(points)):
        center = points[i]
        spline_points = sample_spline(splines[i], n_samples=50)

        if type(spline_points) == type(None):
            if len(splines_sampled) > 0:
                spline_points = splines_sampled[-1].copy()
                center = points[i - 1]
            else:
                spline_points = None

        if spline_points is None:
            radius_scalar = float(fallback_radius)
        else:
            center_for_radius = center
            if center_mode == "centroid":
                center_for_radius = np.mean(spline_points, axis=0)
            distances = np.linalg.norm(spline_points - center_for_radius, axis=1)
            if radius_mode == "max":
                radius_scalar = float(np.max(distances))
            elif radius_mode == "mean":
                radius_scalar = float(np.mean(distances))
            elif radius_mode == "percentile":
                radius_scalar = float(np.percentile(distances, radius_percentile))
            else:
                radius_scalar = float(np.median(distances))
            if radius_cap is not None:
                radius_scalar = min(radius_scalar, float(radius_cap))
        coeff = np.array([radius_scalar], dtype=np.float32)

        splines_sampled.append(spline_points.copy() if spline_points is not None else None)

        if centerline_t_mode == "kdtree":
            _, idx = kdtree.query(center)
            t = float(t_values[idx])
        else:
            t = minimize_scalar(distance_to_spline, bounds=(0, 1), args=(center,), method="bounded").x
        if i == len(points) - 1:
            t = 1.0
        coeffs.append((t, coeff))

    def f(P):
        if P.ndim == 1:
            P = P[np.newaxis, :]

        sdf_values = []
        for p in P:
            _, nearest_idx = kdtree.query(p)
            nearest_points = sampled_spline[nearest_idx]
            min_dist = np.linalg.norm(p - nearest_points)
            t = nearest_idx / n_samples
            t2 = t * 1.05 if t * 1.05 <= 1.0 else t * 0.95

            a = np.array(splev(t, tck))
            b = np.array(splev(t2, tck))

            angle = calculate_angle(a, b, p)
            if type(angle) == type(None):
                angle = np.array([0.0])

            poly_t, poly_i = find_minmax(t, coeffs)
            size = coeffs[poly_i + 1][0] - coeffs[poly_i][0]
            t_delta = (t - poly_t) / size if size > 0 else 0

            radius_start = poly(angle, coeffs[poly_i][1])
            radius_end = poly(angle, coeffs[poly_i + 1][1])
            radius = (1 - t_delta) * radius_start + t_delta * radius_end

            sdf = min_dist - radius
            sdf_values.append(sdf)

        return np.array(sdf_values)

    return f


class VesselSDF:
    def __init__(self, tree_points, points, splines, n_samples=50):
        # Fit spline to the tree points
        self.tck, _ = splprep(tree_points.T, s=0)
        self.n_samples = n_samples

        # Sample the main spline
        self.t_values = np.linspace(0, 1, self.n_samples)
        self.sampled_spline = np.array(splev(self.t_values, self.tck)).T
        self.kdtree = KDTree(self.sampled_spline)

        # Process radii splines
        self.coeffs = self.compute_radius_coeffs(points, splines)

        # Precompute spline values for interpolation
        self.spline_values = np.array([splev(t, self.tck) for t in self.t_values])

    def compute_radius_coeffs(self, points, splines):
        coeffs = []
        splines_sampled = []

        for i in range(len(points)):
            center = points[i]
            spline_points = sample_spline(splines[i], n_samples=50)

            if spline_points is None:
                if len(splines_sampled) > 0:
                    spline_points = splines_sampled[-1].copy()
                    center = points[i - 1]
                else:
                    raise ValueError("Invalid root bifurcation with null spline")

            distances = np.linalg.norm(spline_points - center, axis=1)
            splines_sampled.append(spline_points.copy())

            xs = np.linspace(0, 1, 50)
            coeff = np.polyfit(xs, distances, 5)

            t = minimize_scalar(
                lambda t: np.linalg.norm(center - np.array(splev(t, self.tck))),
                bounds=(0, 1), method='bounded'
            ).x

            if i == len(points) - 1:
                t = 1.0

            coeffs.append((t, coeff))

        return coeffs

    def find_minmax(self, t):
        min_t = self.coeffs[0][0]
        min_i = 0
        for i in range(len(self.coeffs)):
            if self.coeffs[i][0] >= min_t and t >= self.coeffs[i][0]:
                min_t = self.coeffs[i][0]
                min_i = i
        return min_t, min_i

    def __call__(self, P):
        if P.ndim == 1:
            P = P[np.newaxis, :]

        N = P.shape[0]
        sdf_values = np.zeros(N)
        dist, nearest_idx = self.kdtree.query(P)
        nearest_points = self.sampled_spline[nearest_idx]

        for i in range(N):
            p = P[i]
            nearest_point = nearest_points[i]
            min_dist = np.linalg.norm(p - nearest_point)
            t = nearest_idx[i] / self.n_samples
            t2 = t * 1.05 if t * 1.05 <= 1.0 else t * 0.95

            a = self.spline_values[int(t * (self.n_samples - 1))]
            b = self.spline_values[int(t2 * (self.n_samples - 1))]

            angle = calculate_angle(a, b, p)
            if angle is None:
                angle = np.array([0.])

            poly_t, poly_i = self.find_minmax(t)

            # Prevent overflow
            if poly_i + 1 >= len(self.coeffs):
                poly_i = max(poly_i - 1, 0)

            size = self.coeffs[poly_i + 1][0] - self.coeffs[poly_i][0]
            t_delta = (t - poly_t) / size if size > 0 else 0

            radius_start = poly(angle, self.coeffs[poly_i][1])
            radius_end = poly(angle, self.coeffs[poly_i + 1][1])
            radius = (1 - t_delta) * radius_start + t_delta * radius_end

            sdf_values[i] = min_dist - radius

        return sdf_values


@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d = _vec(
            _length(p[:, [0, 1]]) - ra + rb,
            np.abs(p[:, 2]) - h / 2 + rb)
        return (
                _min(_max(d[:, 0], d[:, 1]), 0) +
                _length(_max(d, 0)) - rb)

    return f


@sdf3
def capped_cone(a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)

    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = _dot(p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))

    return f


@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[:, [0, 1]]), p[:, 2])
        b = (r1 - r2) / h
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - r1
        c2 = _length(q - _vec(0, h)) - r2
        c3 = np.dot(q, _vec(a, b)) - r1
        return np.where(k < 0, c1, np.where(k > a * h, c2, c3))

    return f


@sdf3
def ellipsoid(size):
    size = np.array(size)

    def f(p):
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        return k0 * (k0 - 1) / k1

    return f


@sdf3
def pyramid(h):
    def f(p):
        a = np.abs(p[:, [0, 1]]) - 0.5
        w = a[:, 1] > a[:, 0]
        a[w] = a[:, [1, 0]][w]
        px = a[:, 0]
        py = p[:, 2]
        pz = a[:, 1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))

    return f


# Platonic Solids

@sdf3
def tetrahedron(r):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - r) / np.sqrt(3)

    return f


@sdf3
def octahedron(r):
    def f(p):
        return (np.sum(np.abs(p), axis=1) - r) * np.tan(np.radians(30))

    return f


@sdf3
def dodecahedron(r):
    x, y, z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))

    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        q = (_max(_max(a, b), c) - x) * r
        return q

    return f


@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
    w = np.sqrt(3) / 3

    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        d = np.dot(p, (w, w, w)) - x
        return _max(_max(_max(a, b), c) - x, d) * r

    return f


# Positioning

@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)

    return f


@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))

    def f(p):
        return other(p / s) * m

    return f


@op3
def rotate(other, angle, vector=Z):
    x, y, z = _normalize(vector)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m * x * x + c, m * x * y + z * s, m * z * x - y * s],
        [m * x * y - z * s, m * y * y + c, m * y * z + x * s],
        [m * z * x + y * s, m * y * z - x * s, m * z * z + c],
    ]).T

    def f(p):
        return other(np.dot(p, matrix))

    return f


@op3
def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)


@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)


@op3
def circular_array(other, count, offset=0):
    other = other.translate(X * offset)
    da = 2 * np.pi / count

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        d1 = other(_vec(np.cos(a - da) * d, np.sin(a - da) * d, z))
        d2 = other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)

    return f


# Alterations

@op3
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:, 0].reshape((-1, 1))
        y = q[:, 1].reshape((-1, 1))
        z = q[:, 2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w

    return f


@op3
def twist(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def bend(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = -np.array(v)
    ab = p1 - p0

    def f(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)

    return f


@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))

    return f


@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    ab = p1 - p0

    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = np.hypot(p[:, 0], p[:, 1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0 = X * x0
    p1 = X * x1
    v = -Y
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * np.pi)

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:, 2] = z
        return other(q)

    return f


# 3D => 2D Operations

@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s

    def f(p):
        p = _vec(p[:, 0], p[:, 1], np.zeros(len(p)))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A

    return f


# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)
