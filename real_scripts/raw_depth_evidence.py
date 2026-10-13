"""Reject unsupported RGB-D returns without inventing free space."""

from dataclasses import dataclass

import numpy as np

from real_scripts.camera_geometry import pixel_rays
from real_scripts.obstacle_observation import measured_view_support


@dataclass
class RawDepthEvidence:
    accepted: np.ndarray
    uncertain: np.ndarray
    candidates: np.ndarray
    diagnostics: dict


class RawDepthEvidenceFilter:
    """Local surface support, with bounded history for weakly supported returns.

    Never smooth or replace measured depth. Two neighbouring returns normally
    suffice, including for a one-pixel-wide line. A weak return with at least
    one neighbour can be confirmed over consecutive captures. Completely
    isolated returns require another camera's independently supported surface;
    repeating an isolated speckle alone never confirms it.
    """

    def __init__(self, *, radius_pixels=1, min_neighbors=2,
                 absolute_tolerance_m=.008, relative_tolerance=.005,
                 neighbor_distance_m=.020, confirm_frames=3, max_gap_s=2.):
        if (not isinstance(radius_pixels, int) or not 1 <= radius_pixels <= 3
                or not isinstance(min_neighbors, int)
                or not 2 <= min_neighbors <= (2 * radius_pixels + 1) ** 2 - 1
                or not isinstance(confirm_frames, int) or not 2 <= confirm_frames <= 65535
                or not np.isfinite([absolute_tolerance_m, relative_tolerance,
                                    neighbor_distance_m, max_gap_s]).all()
                or absolute_tolerance_m < 0 or relative_tolerance < 0
                or neighbor_distance_m <= 0 or max_gap_s <= 0):
            raise ValueError("Invalid raw depth evidence thresholds")
        self.radius_pixels, self.min_neighbors = radius_pixels, min_neighbors
        self.absolute_tolerance_m, self.relative_tolerance = absolute_tolerance_m, relative_tolerance
        self.neighbor_distance_m = neighbor_distance_m
        self.confirm_frames, self.max_gap_s = confirm_frames, max_gap_s
        self.timestamp_ns = 0
        self._rays = self._camera_key = self._previous = self._streak = None

    def update(self, depth, robot_keep, rendered, camera, *, timestamp_ns,
               robot_absolute_tolerance_m=.015, robot_relative_tolerance=.015,
               max_depth_m=2.5):
        if not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns <= self.timestamp_ns:
            raise ValueError("Depth evidence requires increasing capture timestamps")
        depth = np.asarray(depth, np.float32)
        keep, rendered = np.asarray(robot_keep, bool), np.asarray(rendered, np.float32)
        if depth.ndim != 2 or keep.shape != depth.shape or rendered.shape != depth.shape:
            raise ValueError("Depth evidence dimensions disagree")
        if (not np.isfinite([robot_absolute_tolerance_m, robot_relative_tolerance, max_depth_m]).all()
                or robot_absolute_tolerance_m < 0 or robot_relative_tolerance < 0 or max_depth_m <= 0):
            raise ValueError("Invalid robot depth evidence thresholds")
        # Cache distortion-aware rays; reset pixel history when calibration or
        # resolution changes. History never crosses cameras or time gaps.
        key = (depth.shape, np.asarray(camera.intrinsics).tobytes(),
               np.asarray(camera.camera_to_world).tobytes(),
               np.asarray(camera.distortion).tobytes(), camera.distortion_model)
        if key != self._camera_key:
            y, x = np.indices(depth.shape)
            self._rays = pixel_rays(np.c_[x.ravel(), y.ravel()], camera.intrinsics,
                                   camera.distortion, camera.distortion_model).astype(np.float32).reshape(*depth.shape, 3)
            self._previous = np.zeros_like(depth)
            self._streak = np.zeros(depth.shape, np.uint16)
            self._camera_key = key
        valid = keep & np.isfinite(depth) & (depth > 0) & (depth <= max_depth_m)
        visible_robot = np.isfinite(rendered) & (rendered > 0)
        robot_z = np.where(visible_robot, rendered, 0.)
        occluded = valid & visible_robot & (
            depth > robot_z + robot_absolute_tolerance_m + robot_relative_tolerance * robot_z)
        usable = valid & ~occluded
        clean = np.where(usable, depth, 0.)
        points = clean[..., None] * self._rays
        neighbors = np.zeros(depth.shape, np.uint8)
        height, width = depth.shape
        radius = self.radius_pixels
        # Support is symmetric: evaluate each pair once and credit both pixels.
        for dy in range(radius + 1):
            for dx in range(-radius, radius + 1):
                if (dy == 0 and dx <= 0) or dy >= height or abs(dx) >= width:
                    continue
                a = (slice(max(0, -dy), min(height, height-dy)),
                     slice(max(0, -dx), min(width, width-dx)))
                b = (slice(max(0, dy), min(height, height+dy)),
                     slice(max(0, dx), min(width, width+dx)))
                tolerance = self.absolute_tolerance_m + self.relative_tolerance * np.minimum(clean[a], clean[b])
                supported = (usable[a] & usable[b] & (np.abs(clean[a]-clean[b]) <= tolerance)
                             & (np.sum((points[a]-points[b]) ** 2, axis=-1) <= self.neighbor_distance_m ** 2))
                neighbors[a] += supported
                neighbors[b] += supported
        strong = usable & (neighbors >= self.min_neighbors)
        trackable = usable & (neighbors >= 1)
        continuous = (self.timestamp_ns > 0 and
                      (timestamp_ns-self.timestamp_ns) / 1e9 <= self.max_gap_s)
        stable = (continuous & (self._streak > 0)
                  & (np.abs(clean-self._previous) <= self.absolute_tolerance_m + self.relative_tolerance * clean))
        self._streak = np.where(trackable,
                                np.where(stable, np.minimum(self._streak.astype(np.uint32)+1, 65535), 1),
                                0).astype(np.uint16)
        temporal = trackable & ~strong & (self._streak >= self.confirm_frames)
        accepted = strong | temporal
        self._previous, self.timestamp_ns = clean, int(timestamp_ns)
        return RawDepthEvidence(
            accepted, valid & ~accepted, usable & ~accepted,
            {"valid_nonrobot_pixels": int(valid.sum()), "spatial_supported_pixels": int(strong.sum()),
             "temporal_confirmed_pixels": int(temporal.sum()), "multiview_confirmed_pixels": 0,
             "robot_occluded_pixels": int(occluded.sum()), "isolated_pixels": int((usable & (neighbors == 0)).sum()),
             "accepted_raw_pixels": int(accepted.sum()), "uncertain_raw_pixels": int((valid & ~accepted).sum())})

    def world_points(self, depth, mask, camera):
        """Lift all selected endpoints using the cached distortion-aware rays.

        Evidence must not depend on the viewer's point-stride lattice: even a
        small confirmed surface between sampled pixels must reach the map.
        """
        depth, mask = np.asarray(depth, np.float32), np.asarray(mask, bool)
        if self._rays is None or depth.shape != self._rays.shape[:2] or mask.shape != depth.shape:
            raise ValueError("Depth evidence must be updated before lifting endpoints")
        points = self._rays[mask] * depth[mask, None]
        transform = np.asarray(camera.camera_to_world, np.float32)
        return points @ transform[:3, :3].T + transform[:3, 3]


def confirm_raw_depth_views(views, *, absolute_tolerance_m=.008, relative_tolerance=.005):
    """Rescue thin returns using independently accepted raw depth in another view.

    views maps names to (calibration, depth, RawDepthEvidence). Capture the
    supporting images before promotion to avoid circular or order-dependent
    confirmation between two unsupported speckles. Robot-occluded pixels are
    ineligible. Neither inferred depth nor rejected raw returns can support it.
    """
    trusted = {name: np.where(evidence.accepted, depth, 0.)
               for name, (_, depth, evidence) in views.items()}
    for name, (camera, depth, evidence) in views.items():
        y, x = np.nonzero(evidence.candidates)
        if not len(x):
            continue
        rays = pixel_rays(np.c_[x, y], camera.intrinsics, camera.distortion, camera.distortion_model)
        points = rays * np.asarray(depth)[y, x, None]
        world = points @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
        supported, contradicted = measured_view_support(
            world, [(view, trusted[other]) for other, (view, _, _) in views.items() if other != name],
            absolute_tolerance_m=absolute_tolerance_m, relative_tolerance=relative_tolerance)
        good_y, good_x = y[supported & ~contradicted], x[supported & ~contradicted]
        evidence.accepted[good_y, good_x] = True
        evidence.uncertain[good_y, good_x] = False
        evidence.diagnostics.update(multiview_confirmed_pixels=len(good_x),
                                    accepted_raw_pixels=int(evidence.accepted.sum()),
                                    uncertain_raw_pixels=int(evidence.uncertain.sum()))
