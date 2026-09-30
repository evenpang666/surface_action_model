"""Measured RGB-D robot surface and local FK association for CBF derivatives."""

import numpy as np
from scipy.spatial import cKDTree


def associate_surface(surface_points, mesh_points, max_distance_m=.04):
    surface = np.asarray(surface_points, dtype=np.float32)
    mesh = np.asarray(mesh_points, dtype=np.float32)
    if surface.ndim != 2 or surface.shape[-1] != 3 or not np.isfinite(surface).all():
        raise ValueError("Measured surface must be finite [K,3]")
    distances, indices = cKDTree(mesh.reshape(-1, 3)).query(surface)
    valid = distances <= max_distance_m
    return indices // mesh.shape[1], valid


class MeasuredSurfaceSampler:
    """Transport measured points with their associated rigid FK link.

    Coordinates remain measured; the local derivative comes from the link's
    rigid transform, recovered from its fixed-identity mesh samples.
    """

    def __init__(self, sampler, qpos, gripper_position, surface_points, max_distance_m=.04):
        self.sampler = sampler
        self.reference = sampler.link_points(qpos, gripper_position)
        self.surface_points = np.asarray(surface_points, dtype=np.float32)
        self.link_ids, valid = associate_surface(self.surface_points, self.reference, max_distance_m)
        if not valid.all():
            raise ValueError("Measured surface cannot be associated with the current robot links")

    def link_points(self, qpos, gripper_position=None):
        moved = self.sampler.link_points(qpos, gripper_position)
        result = np.empty_like(self.surface_points)
        for link in np.unique(self.link_ids):
            before, after = self.reference[link].astype(float), moved[link].astype(float)
            center0, center1 = before.mean(0), after.mean(0)
            u, _, vt = np.linalg.svd((before - center0).T @ (after - center1))
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            mask = self.link_ids == link
            result[mask] = (self.surface_points[mask] - center0) @ rotation + center1
        return result
