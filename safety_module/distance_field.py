"""World-frame ESDF interpolation shared by native maps and snapshot readers."""

import numpy as np


def query_esdf_grid(grid, distances, observed_cells, active, maximum, points, *, interpolate=True):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("ESDF queries require finite world points (N,3)")
    local = (points-grid.origin) @ grid.axes / grid.resolution_m
    distance = np.full(len(points), np.nan)
    gradient = np.full((len(points), 3), np.nan)
    observed = np.zeros(len(points), bool)
    saturated = np.zeros(len(points), bool)
    gradient_valid = np.zeros(len(points), bool)
    if interpolate:
        base = np.floor(local-.5).astype(np.int64)
        offsets = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        cells = base[:, None, :]+offsets
    else:
        cells = np.floor(local).astype(np.int64)[:, None, :]
    inside = np.all((cells >= 0) & (cells < grid.shape), axis=(1, 2))
    rows = np.flatnonzero(inside)
    ids = cells[inside] @ grid.strides
    valid = np.all(active[ids] & observed_cells[ids], axis=1)
    rows, ids = rows[valid], ids[valid]
    values = distances[ids].astype(float)
    observed[rows] = True
    saturated[rows] = np.any(np.abs(values) >= maximum-1e-6, axis=1)
    if interpolate:
        fraction = local[rows]-.5-base[rows]
        factors = np.where(offsets[None, :, :], fraction[:, None, :], 1-fraction[:, None, :])
        distance[rows] = np.sum(values*np.prod(factors, axis=2), axis=1)
        derivative = np.empty((len(rows), 3))
        for dimension in range(3):
            other = [axis for axis in range(3) if axis != dimension]
            derivative[:, dimension] = np.sum(
                values*(2*offsets[:, dimension]-1)*np.prod(factors[:, :, other], axis=2), axis=1)
        gradient[rows] = derivative @ grid.axes.T / grid.resolution_m
        gradient_valid[rows] = ~saturated[rows]
        gradient[~gradient_valid] = np.nan
    else:
        distance[rows] = values[:, 0]
    return {"distance_m": distance, "observed": observed, "gradient": gradient,
            "gradient_valid": gradient_valid, "saturated": saturated}
