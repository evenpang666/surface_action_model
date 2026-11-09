"""Independent VLSA and VLPSA numerical safety filters.

VLSA: arXiv:2512.11891v2, Sec. IV-C (end-effector ellipsoid + virtual plane).
VLPSA: arXiv:2609.22462v1, Sec. II/III (buffered Poisson field, full-body
samples, held-object samples, and Boolean OR of two resolutions).
The simulator adapter supplies kinematics; neither filter queries ground truth.
Controls are joint *increments* over dt, not joint velocities.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse.linalg import cg

from .ellipsoid_geometry import Ellipsoid, separation, separation_gradient, vlsa_barrier
from .oa_vla_actions import ObservationHold


def project_qp(reference, matrix, rhs, lower, upper, *, weights=None, clf=None):
    """Strict convex projection, with a soft CLF and hard CBFs and limits.

    SLSQP solves the small control-space QP; all constraints are independently
    checked after solving. A solver failure never silently executes the nominal.
    """
    reference = np.asarray(reference, float)
    n = len(reference)
    matrix = np.asarray(matrix, float).reshape(-1, n)
    rhs = np.asarray(rhs, float)
    weights = np.ones(n) if weights is None else np.asarray(weights, float)
    lower, upper = np.broadcast_to(lower, (n,)), np.broadcast_to(upper, (n,))
    if (not all(np.isfinite(x).all() for x in (reference, matrix, rhs, weights, lower, upper))
            or np.any(lower > upper) or np.any(weights <= 0) or rhs.shape != (len(matrix),)):
        raise ValueError("Invalid QP coefficients")
    if clf is not None:
        row, bound, penalty = clf
        row = np.asarray(row, float)
        if row.shape != (n,) or not np.isfinite(row).all() or not np.isfinite([bound, penalty]).all() or penalty <= 0:
            raise ValueError("Invalid CLF")
        matrix = np.column_stack((matrix, np.zeros(len(matrix))))
        matrix = np.vstack((matrix, np.r_[-row, 1.]))
        rhs = np.r_[rhs, -bound]
        reference = np.r_[reference, 0.]
        weights = np.r_[weights, penalty]
        lower, upper = np.r_[lower, 0.], np.r_[upper, np.inf]
    constraints = [] if not len(matrix) else [LinearConstraint(matrix, rhs, np.inf)]
    result = minimize(lambda x: .5*np.sum(weights*(x-reference)**2),
                      np.clip(reference, lower, upper), jac=lambda x: weights*(x-reference),
                      method="SLSQP", bounds=Bounds(lower, upper), constraints=constraints,
                      options={"ftol": 1e-11, "maxiter": 200})
    residual = float(np.min(matrix @ result.x-rhs)) if len(matrix) else 0.
    if (not result.success or not np.isfinite(result.x).all() or residual < -1e-7
            or np.any(result.x < lower-1e-8) or np.any(result.x > upper+1e-8)):
        raise ObservationHold(f"Safety QP failed: {result.message}; residual={residual:g}")
    return result.x[:n], dict(solver="scipy_slsqp_convex_qp", minimum_constraint_residual=residual,
                              clf_slack=float(result.x[-1]) if clf is not None else 0.)


class VLSAFilter:
    """End-effector-only Eq. (9), one auxiliary unit plane per obstacle.

    A joint-space Jacobian implements the full pose correction. There is no
    full-arm constraint, geodesic guidance, collision slack, or OA-VLA planner.
    Multiple supplied obstacles are an explicit extension of single-obstacle
    AEGIS; use one obstacle for that experiment protocol.
    """
    def __init__(self, *, margin=.01, alpha=10., virtual_step=.25):
        if margin < 0 or alpha <= 0 or virtual_step <= 0:
            raise ValueError("Invalid VLSA settings")
        self.margin, self.alpha, self.virtual_step = margin, alpha, virtual_step
        self.normals = {}

    def filter(self, reference, geometry, obstacles, lower, upper, dt, **unused):
        body, dc, ds = geometry.eef_differential()
        n, k = len(reference), len(obstacles)
        rows, rhs, values = [], [], []
        for i, obstacle in enumerate(obstacles):
            key = obstacle.key or str(i)
            normal = self.normals.get(key)
            if normal is None:
                normal = separation(body, obstacle)[1]
            value = vlsa_barrier(body, obstacle, normal)
            row = np.zeros(n+2*k)
            row[:n] = separation_gradient(body, value.normal, dc, ds)
            row[n+2*i:n+2*i+2] = value.direction_gradient
            rows.append(row)
            rhs.append(-min(self.alpha*dt, 1.)*(value.value-self.margin))
            values.append((key, value))
        delta, detail = project_qp(np.r_[reference, np.zeros(2*k)], rows, rhs,
            np.r_[lower, np.full(2*k, -self.virtual_step)],
            np.r_[upper, np.full(2*k, self.virtual_step)],
            weights=np.r_[np.ones(n), np.full(2*k, .01)])
        body_after = geometry.eef_body(delta[:n])
        after = []
        pending = {}
        for i, ((key, value), obstacle) in enumerate(zip(values, obstacles)):
            normal = value.normal+value.tangent_basis @ delta[n+2*i:n+2*i+2]
            normal /= np.linalg.norm(normal)
            h = vlsa_barrier(body_after, obstacle, normal).value-self.margin
            # Verify the finite control, rather than trusting a linearization.
            if h < min(0., value.value-self.margin)-1e-6:
                raise ObservationHold("VLSA nonlinear barrier verification failed")
            pending[key] = normal
            after.append(h)
        self.normals.update(pending)
        return delta[:n], dict(method="vlsa", constraint_scope="end_effector", **detail,
            virtual_variables=2*k, minimum_barrier_after=min(after, default=None))


def box_surface(half, spacing):
    """A box-face grid with covering radius <= spacing/sqrt(2)."""
    half = np.asarray(half, float)
    if half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all() or spacing <= 0:
        raise ValueError("Invalid sampling box")
    axes = [np.linspace(-h, h, int(np.ceil(2*h/spacing))+1) for h in half]
    faces = []
    for fixed in range(3):
        other = [j for j in range(3) if j != fixed]
        grid = np.array(list(product(axes[other[0]], axes[other[1]])))
        for sign in (-1, 1):
            points = np.zeros((len(grid), 3))
            points[:, fixed] = sign*half[fixed]
            points[:, other] = grid
            faces.append(points)
    return np.unique(np.concatenate(faces), axis=0)


@dataclass
class PoissonField:
    origin: np.ndarray
    spacing: float
    values: np.ndarray
    residual: float

    @classmethod
    def solve(cls, occupied, origin, spacing, *, buffer=0., tolerance=1e-7, maxiter=2000):
        """-Laplacian h=1 on free nodes, h=0 on obstacles and grid walls.

        Occupied interiors get a negative distance extension to avoid treating
        deep penetration as zero-gradient free space. CG convergence is checked.
        """
        occupied = np.asarray(occupied, bool).copy()
        if occupied.ndim != 3 or min(occupied.shape) < 3 or spacing <= 0 or buffer < 0:
            raise ValueError("Invalid Poisson grid")
        if buffer and occupied.any():
            occupied = ndimage.distance_transform_edt(~occupied, sampling=spacing) <= buffer
        for axis in range(3):
            index = [slice(None)]*3
            for edge in (0, -1):
                index[axis] = edge
                occupied[tuple(index)] = True
        free = ~occupied
        ids = np.full(free.shape, -1, dtype=np.int64)
        ids[free] = np.arange(free.sum())
        rows, cols = [np.arange(free.sum())], [np.arange(free.sum())]
        coefficients = [np.full(free.sum(), 6.)]
        for axis in range(3):
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            ia, ib = ids[tuple(a)], ids[tuple(b)]
            valid = (ia >= 0) & (ib >= 0)
            for source, target in ((ia, ib), (ib, ia)):
                rows.append(source[valid]); cols.append(target[valid])
                coefficients.append(-np.ones(valid.sum()))
        values = np.zeros(occupied.shape, float)
        residual = 0.
        if free.any():
            matrix = sparse.coo_matrix((np.concatenate(coefficients),
                (np.concatenate(rows), np.concatenate(cols))), shape=(free.sum(), free.sum())).tocsr()
            rhs = np.full(free.sum(), spacing**2)
            solution, status = cg(matrix, rhs, rtol=tolerance, atol=0., maxiter=maxiter)
            residual = float(np.linalg.norm(matrix @ solution-rhs)/np.linalg.norm(rhs))
            if status != 0 or residual > tolerance*1.1 or not np.isfinite(solution).all():
                raise ObservationHold(f"Poisson solve did not converge: status={status}, residual={residual:g}")
            values[free] = solution
        # Keep interface obstacle nodes at zero, give occupied interiors a slope.
        depth = ndimage.distance_transform_edt(occupied)*spacing
        values[occupied] = -np.maximum(depth[occupied]-spacing, 0.)*spacing
        for axis in range(3):
            index = [slice(None)]*3
            for edge in (0, -1):
                index[axis] = edge
                values[tuple(index)] = 0.
        origin = np.asarray(origin, float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid field origin")
        return cls(origin, spacing, values, residual)

    def query(self, points):
        """Trilinear value and its exact spatial derivative, plus domain mask."""
        points = np.asarray(points, float).reshape(-1, 3)
        coords = (points-self.origin)/self.spacing
        valid = np.isfinite(coords).all(1) & (coords >= 0).all(1) & (coords <= np.array(self.values.shape)-1).all(1)
        clipped = np.clip(coords, 0, np.array(self.values.shape)-1)
        low = np.minimum(np.floor(clipped).astype(int), np.array(self.values.shape)-2)
        frac = clipped-low
        value, gradient = np.zeros(len(points)), np.zeros((len(points), 3))
        for corner in product((0, 1), repeat=3):
            bits = np.array(corner)
            weights = np.where(bits, frac, 1-frac)
            v = self.values[tuple((low+bits).T)]
            value += v*weights.prod(1)
            for axis in range(3):
                gradient[:, axis] += v*(1 if bits[axis] else -1)*np.prod(np.delete(weights, axis, axis=1), axis=1)/self.spacing
        value[~valid], gradient[~valid] = np.nan, np.nan
        return value, gradient, valid


class VLPSAFilter:
    """PSF-CBF + soft tracking CLF on full-body and held-object samples."""
    def __init__(self, *, alpha=10., clf_rate=2., clf_weight=1000.):
        if min(alpha, clf_rate, clf_weight) <= 0:
            raise ValueError("Invalid VLPSA settings")
        self.alpha, self.clf_rate, self.clf_weight = alpha, clf_rate, clf_weight
        self.previous = None

    def filter(self, reference, geometry, obstacles, lower, upper, dt, *, fields, timestamp, **unused):
        points, jacobian = geometry.sample_differential()
        samples = [field.query(points) for field in fields]
        if not samples or not samples[0][2].all():
            raise ObservationHold("Full-body samples outside coarse Poisson workspace")
        values = np.array([np.where(valid, value, -np.inf) for value, _, valid in samples])
        maximum = values.max(axis=0)
        rows, rhs = [], []
        for level, (value, gradient, valid) in enumerate(samples):
            temporal = np.zeros(len(points))
            if self.previous is not None:
                old_time, old_fields = self.previous
                elapsed = timestamp-old_time
                if elapsed <= 0:
                    raise ValueError("Poisson timestamps must increase")
                old_value, _, old_valid = old_fields[level].query(points)
                overlap = valid & old_valid
                temporal[overlap] = (value[overlap]-old_value[overlap])/elapsed
            # Paper Eq. (7) Boolean OR: BOTH level constraints, each relaxed by
            # |max(hc,hf)-hi|. Never switch only to the winning gradient.
            relaxation = maximum[valid]+np.abs(maximum[valid]-value[valid])
            rows.extend(np.einsum('pi,pij->pj', gradient[valid], jacobian[valid]))
            rhs.extend(-self.alpha*dt*relaxation-temporal[valid]*dt)
        # Quadratic task-space tracking CLF, with explicit (non-safety) slack.
        error, task_jacobian = geometry.tracking_error()
        clf = (-error @ task_jacobian, -self.clf_rate*dt*.5*float(error @ error), self.clf_weight)
        delta, detail = project_qp(reference, rows, rhs, lower, upper, clf=clf)
        after_points = geometry.sample_points(delta)
        after_queries = [field.query(after_points) for field in fields]
        if not after_queries[0][2].all():
            raise ObservationHold("VLPSA predicted samples leave coarse workspace")
        after_values = np.array([np.where(valid, value, -np.inf) for value, _, valid in after_queries])
        after = after_values.max(axis=0)
        if not np.isfinite(after).all() or np.any(after < np.minimum(maximum, 0.)-1e-7):
            raise ObservationHold("VLPSA nonlinear barrier verification failed")
        self.previous = timestamp, fields
        return delta, dict(method="vlpsa", constraint_scope="full_body_and_held_object", **detail,
            sample_count=len(points), held_sample_count=geometry.held_sample_count,
            poisson_levels=len(fields), poisson_residuals=[f.residual for f in fields],
            cbf_constraints=len(rows), minimum_barrier_after=float(after.min()))


def obstacle_grid(obstacles, bounds, spacing, *, max_cells=2_000_000):
    """Conservative OBB voxelization: any intersected voxel is occupied."""
    bounds = np.asarray(bounds, float).reshape(3, 2)
    shape = np.ceil((bounds[:, 1]-bounds[:, 0])/spacing).astype(int)+1
    if spacing <= 0 or min(shape) < 3 or np.prod(shape) > max_cells:
        raise ValueError("Invalid/excessive Poisson grid size")
    occupied = np.zeros(shape, bool)
    # Nodes represent cubic cells; inflate in box coordinates by projected
    # voxel half-extents, not merely rounding the obstacle centroid.
    for box in obstacles:
        radius = np.abs(box.axes) @ box.half_sizes+spacing
        low = np.maximum(0, np.floor((box.center-radius-bounds[:, 0])/spacing).astype(int))
        high = np.minimum(shape-1, np.ceil((box.center+radius-bounds[:, 0])/spacing).astype(int))
        if np.any(high < low):
            continue
        indices = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(low, high)), indexing="ij"), -1)
        points = bounds[:, 0]+indices*spacing
        half = box.half_sizes+.5*spacing*np.abs(box.axes).sum(axis=0)
        inside = (np.abs((points-box.center) @ box.axes) <= half).all(-1)
        occupied[tuple(indices[inside].T)] = True
    return occupied, bounds[:, 0]
