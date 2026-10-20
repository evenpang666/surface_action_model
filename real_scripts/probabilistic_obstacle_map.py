"""Build the fixed tabletop/workspace domain and integrate raw RGB-D rays."""

from itertools import product
import time

import numpy as np

from real_scripts.camera_geometry import pixel_rays
from safety_module.occupancy import LocalOccupancyGrid


def workspace_occupancy_grid(normal, offset, regions, *, resolution_m=.02,
                             min_height_m=.02, max_height_m=None,
                             grid_class=LocalOccupancyGrid, **settings):
    """Anchor grid axes to the first table prior; never move accumulated evidence."""
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Occupancy resolution must be finite and positive")
    regions = tuple(regions)
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    seed = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(seed, normal)
    first /= np.linalg.norm(first)
    axes = np.column_stack((first, np.cross(normal, first), normal))
    table_origin = -float(offset) * normal
    corners = []
    for bounds, transform in regions:
        bounds, transform = np.asarray(bounds, float), np.asarray(transform, float)
        local = np.asarray(list(product(*zip(bounds[::2], bounds[1::2]))))
        corners.append(local @ transform[:3, :3].T + transform[:3, 3])
    projected = (np.concatenate(corners) - table_origin) @ axes
    low = np.floor(projected.min(axis=0) / resolution_m) * resolution_m
    high = projected.max(axis=0)
    low[2] = min_height_m
    if max_height_m is not None:
        high[2] = min(high[2], max_height_m)
    if np.any(high <= low):
        raise ValueError("Table/height limits leave an empty occupancy workspace")
    shape = np.ceil((high - low) / resolution_m).astype(np.int64)
    origin = table_origin + low @ axes.T
    grid = grid_class(origin, axes, shape, resolution_m, **settings)
    centers = grid.centers(np.arange(grid.size))
    active = np.zeros(grid.size, bool)
    # Expanded center tests conservatively include cells that intersect either
    # transformed arm workspace; the gap between the two workspaces stays outside.
    padding = np.sqrt(3) * resolution_m / 2
    for bounds, transform in regions:
        inverse = np.linalg.inv(transform)
        local = centers @ inverse[:3, :3].T + inverse[:3, 3]
        bounds = np.asarray(bounds, float)
        active |= np.all((local >= bounds[::2] - padding) & (local <= bounds[1::2] + padding), axis=1)
    if not active.any():
        raise ValueError("Occupancy workspace has no active voxels")
    grid.active = active
    return grid


def integrate_depth_views(grid, views, raw_hits, completion_hits, *, timestamp_ns,
                          ray_stride=6, free_margin_m=.015, max_depth_m=2.5, uncertain_hits=()):
    """Raw-only clearing. Robot masks stop rays; inferred depth supplies hits only.

    Each view is (calibration, raw depth, accepted raw mask, rendered robot depth).
    An inconsistent raw return behind a nearer rendered robot cannot ray-clear
    through that robot. Rejected/invalid depth never establishes free space.
    Uncertain endpoints also veto another view's free ray or inferred hit.
    """
    if ray_stride < 1 or not np.isfinite([free_margin_m, max_depth_m]).all() or free_margin_m < 0 or max_depth_m <= 0:
        raise ValueError("Invalid occupancy ray settings")
    views = tuple(views)
    ray_start = time.perf_counter()
    free = []
    for camera, depth, keep, rendered in views:
        depth, keep, rendered = np.asarray(depth), np.asarray(keep, bool), np.asarray(rendered)
        if depth.ndim != 2 or depth.shape != keep.shape or depth.shape != rendered.shape:
            raise ValueError("Occupancy depth/mask/render dimensions disagree")
        # Sample before conversion and masking; free rays use only this lattice.
        # Full-resolution occupied/uncertain endpoints are passed separately.
        depth = np.asarray(depth[::ray_stride, ::ray_stride], float)
        keep = keep[::ray_stride, ::ray_stride]
        rendered = rendered[::ray_stride, ::ray_stride]
        valid = keep & np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(valid)
        if not len(u):
            continue
        z = np.minimum(depth[v, u], max_depth_m)
        robot_z = rendered[v, u]
        has_robot = np.isfinite(robot_z) & (robot_z > 0)
        z = np.where(has_robot, np.minimum(z, robot_z), z) - free_margin_m
        allowed = z > .01
        v, u, z = v[allowed], u[allowed], z[allowed]
        rays = pixel_rays(np.c_[u * ray_stride, v * ray_stride], camera.intrinsics, camera.distortion, camera.distortion_model)
        end = (rays * z[:, None]) @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
        ids, _ = grid.ray_cells(camera.camera_to_world[:3, 3][None], end,
                                exclude_endpoint=True, unique_only=True)
        free.append(ids)
    evidence = dict(timestamp_ns=timestamp_ns, raw_hits=raw_hits, completion_hits=completion_hits,
                    free_ids=np.unique(np.concatenate(free)) if free else np.empty(0, np.int64),
                    uncertain_hits=uncertain_hits)
    ray_seconds = time.perf_counter() - ray_start
    if hasattr(grid, "integrate_rgbd_views"):
        grid.integrate_rgbd_views(views, ray_stride=ray_stride, free_margin_m=free_margin_m,
                                  max_depth_m=max_depth_m, **evidence)
    else:
        grid.update(**evidence)
    grid.diagnostics["free_ray_traversal_seconds"] = ray_seconds
