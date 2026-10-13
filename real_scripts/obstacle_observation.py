"""Measured evidence and bounded tabletop estimation for fixed RGB-D cameras."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter

from real_scripts.camera_geometry import pixel_rays, project_pixels
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.reconstruct_realsense_pointcloud import estimate_dominant_plane


class TablePlanePrior:
    """Anchor the table once, then fit only points near the established plane.

    A supplied plane is expressed as n.x + offset = 0 in left_base. Without
    one, startup requires a sufficiently exposed table. Low support freezes
    the prior; large new planes never replace it.
    """

    def __init__(self, *, threshold_m: float = .012, plane=None,
                 min_points: int = 80, min_fraction: float = .35,
                 max_angle_deg: float = 3., max_offset_m: float = .01,
                 update_weight: float = .1) -> None:
        values = (threshold_m, min_fraction, max_angle_deg, max_offset_m, update_weight)
        if (not np.isfinite(values).all() or threshold_m <= 0 or min_points < 3
                or not 0 < min_fraction <= 1 or not 0 < max_angle_deg < 90
                or max_offset_m <= 0 or not 0 < update_weight <= 1):
            raise ValueError("Invalid table prior limits")
        self.threshold_m, self.min_points = float(threshold_m), int(min_points)
        self.min_fraction = float(min_fraction)
        self.max_angle_deg, self.max_offset_m = float(max_angle_deg), float(max_offset_m)
        self.update_weight = float(update_weight)
        self.normal = self.offset = self.anchor_normal = self.anchor_offset = None
        self.anchor_center = None
        self.diagnostics = {}
        if plane is not None:
            plane = np.asarray(plane, dtype=float)
            if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
                raise ValueError("Table plane must contain a finite nonzero normal and offset")
            length = np.linalg.norm(plane[:3])
            self._anchor(plane[:3] / length, float(plane[3] / length),
                         -plane[3] / length * plane[:3] / length)

    def _anchor(self, normal, offset, center):
        self.normal, self.offset = normal.copy(), float(offset)
        self.anchor_normal, self.anchor_offset = normal.copy(), float(offset)
        self.anchor_center = np.asarray(center, dtype=float).copy()

    def update(self, points: np.ndarray, camera_position: np.ndarray) -> tuple[np.ndarray, float]:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if self.normal is None:
            if len(points) < self.min_points:
                raise RuntimeError("Not enough measured points to initialize tabletop prior")
            normal, offset, inliers = estimate_dominant_plane(
                points, threshold=self.threshold_m, ransac_iterations=300)
            count = int(inliers.sum())
            if count < self.min_points or count / len(points) < self.min_fraction:
                raise RuntimeError("Measured tabletop plane has insufficient support")
            spread = np.linalg.svd(points[inliers] - points[inliers].mean(axis=0), compute_uv=False)
            if spread[1] / np.sqrt(count) < .03:
                raise RuntimeError("Measured tabletop plane has insufficient spatial coverage")
            if float(np.asarray(camera_position) @ normal + offset) < 0:
                normal, offset = -normal, -offset
            self._anchor(normal, offset, points[inliers].mean(axis=0))
            state = "initialized"
        else:
            if float(np.asarray(camera_position) @ self.normal + self.offset) < 0:
                self.normal, self.offset = -self.normal, -self.offset
                self.anchor_normal, self.anchor_offset = -self.anchor_normal, -self.anchor_offset
            # Only the established plane's neighbourhood is eligible for fitting.
            inliers = np.abs(points @ self.normal + self.offset) <= self.threshold_m
            count = int(inliers.sum())
            state = "retained_prior_low_support"
            if count >= self.min_points and count / max(len(points), 1) >= self.min_fraction:
                near = points[inliers]
                center = near.mean(axis=0)
                _, _, vectors = np.linalg.svd(near - center, full_matrices=False)
                candidate = vectors[-1]
                if candidate @ self.anchor_normal < 0:
                    candidate = -candidate
                offset = -float(candidate @ center)
                angle = np.degrees(np.arccos(np.clip(candidate @ self.anchor_normal, -1., 1.)))
                # Compare displacement at the same table location, not coordinate
                # offsets of planes whose normals differ in a non-world frame.
                shift = abs(float(candidate @ self.anchor_center + offset))
                state = "retained_prior_rejected_change"
                spread = np.linalg.svd(near - center, compute_uv=False)
                if (angle <= self.max_angle_deg and shift <= self.max_offset_m
                        and spread[1] / np.sqrt(count) >= .03):
                    weight = self.update_weight
                    normal = (1 - weight) * self.normal + weight * candidate
                    length = np.linalg.norm(normal)
                    normal = normal / length
                    offset = ((1 - weight) * self.offset + weight * offset) / length
                    if abs(float(normal @ self.anchor_center + offset)) <= self.max_offset_m:
                        self.normal, self.offset = normal, offset
                        state = "updated"
        self.diagnostics = {"state": state, "measured_inliers": count,
                            "measured_inlier_fraction": count / max(len(points), 1)}
        return self.normal.copy(), float(self.offset)


def measured_view_support(world, other_views, *, absolute_tolerance_m, relative_tolerance):
    """Measured agreement and free-space contradiction at world positions."""
    world = np.asarray(world, float).reshape(-1, 3)
    if (not np.isfinite([absolute_tolerance_m, relative_tolerance]).all()
            or absolute_tolerance_m < 0 or relative_tolerance < 0):
        raise ValueError("Invalid measured view evidence thresholds")
    supported, contradicted = np.zeros(len(world), bool), np.zeros(len(world), bool)
    for other, depth in other_views:
        depth = np.asarray(depth, dtype=float)
        transform = np.linalg.inv(other.camera_to_world)
        points = world @ transform[:3, :3].T + transform[:3, 3]
        indices = np.flatnonzero(points[:, 2] > .01)
        pixels = project_pixels(points[indices], other.intrinsics, other.distortion, other.distortion_model)
        finite = np.isfinite(pixels).all(axis=1)
        indices, pixels = indices[finite], pixels[finite]
        x, y = np.rint(pixels).astype(int).T
        inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        indices, x, y = indices[inside], x[inside], y[inside]
        observed = depth[y, x]
        valid = np.isfinite(observed) & (observed > 0)
        tol = absolute_tolerance_m + relative_tolerance * points[indices, 2]
        residual = observed - points[indices, 2]
        supported[indices] |= valid & (np.abs(residual) <= tol)
        # A closer surface is occlusion; a farther one contradicts this hit.
        contradicted[indices] |= valid & (residual > tol)
    return supported, contradicted


def verified_completion_depth(raw_depth: np.ndarray, refined_depth: np.ndarray,
                              raw_keep: np.ndarray, refined_keep: np.ndarray,
                              camera: CameraCalibration,
                              other_views: list[tuple[CameraCalibration, np.ndarray]],
                              *, support_pixels: int = 2,
                              absolute_tolerance_m: float = .015,
                              relative_tolerance: float = .015) -> tuple[np.ndarray, dict]:
    """Return only supported holes; original measured depth remains authoritative.

    Reject holes at depth discontinuities and require a measured return from
    another view at the inferred 3D position. Other views must already exclude
    robot pixels. Completion is never a source of clearing evidence.
    """
    raw = np.asarray(raw_depth, dtype=np.float32)
    refined = np.asarray(refined_depth, dtype=np.float32)
    keep = np.asarray(raw_keep, dtype=bool)
    refined_keep = np.asarray(refined_keep, dtype=bool)
    if raw.ndim != 2 or any(value.shape != raw.shape for value in (refined, keep, refined_keep)):
        raise ValueError("Raw/refined depth and keep masks must have matching HxW shapes")
    if (support_pixels < 0 or not np.isfinite([absolute_tolerance_m, relative_tolerance]).all()
            or absolute_tolerance_m < 0 or relative_tolerance < 0):
        raise ValueError("Invalid completion evidence thresholds")
    valid_raw = np.isfinite(raw) & (raw > 0)
    measured = valid_raw & keep
    valid_refined = np.isfinite(refined) & (refined > 0) & refined_keep & keep
    tolerance = absolute_tolerance_m + relative_tolerance * np.where(valid_raw, raw, 0)
    measured_disagreement = measured & valid_refined & (np.abs(refined - raw) > tolerance)
    result = np.zeros_like(raw)
    diagnostics = {"measured_pixels": int(measured.sum()),
                   "refined_measured_disagreement_pixels": int(measured_disagreement.sum()),
                   "accepted_completion_pixels": 0, "rejected_completion_pixels": 0}
    holes = valid_refined & ~valid_raw
    if support_pixels == 0:
        diagnostics["rejected_completion_pixels"] = int(holes.sum())
        return result, diagnostics
    size = 2 * support_pixels + 1
    supported = binary_dilation(measured, structure=np.ones((3, 3), bool), iterations=support_pixels)
    low = minimum_filter(np.where(measured, raw, np.inf), size=size, mode="constant", cval=np.inf)
    high = maximum_filter(np.where(measured, raw, -np.inf), size=size, mode="constant", cval=-np.inf)
    local_tolerance = absolute_tolerance_m + relative_tolerance * np.where(np.isfinite(high), high, 0)
    candidates = (holes & supported & np.isfinite(low) & np.isfinite(high)
                  & (high - low <= 2 * local_tolerance)
                  & (refined >= low - local_tolerance) & (refined <= high + local_tolerance))
    v, u = np.nonzero(candidates)
    if len(u):
        rays = pixel_rays(np.c_[u, v], camera.intrinsics, camera.distortion, camera.distortion_model)
        camera_points = rays * refined[v, u, None]
        world = camera_points @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
        supported_other, contradicted = measured_view_support(
            world, other_views, absolute_tolerance_m=absolute_tolerance_m,
            relative_tolerance=relative_tolerance)
        accepted = supported_other & ~contradicted
        result[v[accepted], u[accepted]] = refined[v[accepted], u[accepted]]
        diagnostics["accepted_completion_pixels"] = int(accepted.sum())
    diagnostics["rejected_completion_pixels"] = int(holes.sum()) - diagnostics["accepted_completion_pixels"]
    return result, diagnostics
