#!/usr/bin/env python3
"""Real-robot adapters and UR7e geometry helpers for SafetyModule collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from scipy.ndimage import minimum_filter


UR7E_LINK_NAMES = (
    "base_shoulder",
    "shoulder_upper",
    "upper_forearm",
    "forearm_wrist1",
    "wrist1_wrist2",
    "wrist2_wrist3",
    "gripper_width",
)
UR7E_DH_PARAMETERS = (
    (0.0, np.pi / 2.0, 0.1625),
    (-0.425, 0.0, 0.0),
    (-0.3922, 0.0, 0.0),
    (0.0, np.pi / 2.0, 0.1333),
    (0.0, -np.pi / 2.0, 0.0997),
    (0.0, 0.0, 0.0996),
)
DEFAULT_RGBD_CAMERA_NAMES = ("front", "side", "wrist")
DEFAULT_SCENE_RGBD_CAMERA_NAMES = ("front", "side")


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    distortion: np.ndarray | None = None
    distortion_model: str = "brown_conrady"

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
        if camera_to_world.shape != (4, 4):
            raise ValueError(f"camera_to_world must have shape (4, 4), got {camera_to_world.shape}")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)
        if self.distortion is not None:
            distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
            if not np.isfinite(distortion).all():
                raise ValueError("Non-finite camera lens distortion")
            object.__setattr__(self, "distortion", distortion)


@dataclass(frozen=True)
class CameraCalibrationSession:
    """Calibrations plus the point-cloud mode declared by their JSON file."""

    calibrations: dict[str, CameraCalibration]
    camera_names: tuple[str, ...]
    fusion_enabled: bool
    camera_serials: dict[str, str | None]
    camera_streams: dict[str, tuple[int, int, int] | None]


@dataclass(frozen=True)
class RGBDFrame:
    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    # These fields describe the acquisition, rather than the processing time.
    # They are optional so existing offline callers can construct RGBDFrame from
    # an image/depth pair, but real-hardware collection should populate them.
    host_timestamp_ns: int | None = None
    device_timestamp_ms: float | None = None
    frame_number: int | None = None
    timestamp_domain: str | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        depth_m = np.asarray(self.depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(f"depth_m shape {depth_m.shape} must match rgb height/width {rgb.shape[:2]}")
        if self.host_timestamp_ns is not None and int(self.host_timestamp_ns) < 0:
            raise ValueError("host_timestamp_ns must be non-negative")
        if self.device_timestamp_ms is not None and not np.isfinite(float(self.device_timestamp_ms)):
            raise ValueError("device_timestamp_ms must be finite when supplied")
        if self.frame_number is not None and int(self.frame_number) < 0:
            raise ValueError("frame_number must be non-negative")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth_m)
        if self.host_timestamp_ns is not None:
            object.__setattr__(self, "host_timestamp_ns", int(self.host_timestamp_ns))
        if self.device_timestamp_ms is not None:
            object.__setattr__(self, "device_timestamp_ms", float(self.device_timestamp_ms))
        if self.frame_number is not None:
            object.__setattr__(self, "frame_number", int(self.frame_number))
        if self.timestamp_domain is not None:
            object.__setattr__(self, "timestamp_domain", str(self.timestamp_domain))

    def __getitem__(self, index: int) -> np.ndarray:
        """Compatibility with legacy ``source.read()[name][0/1]`` callers."""
        if index == 0:
            return self.rgb
        if index == 1:
            return self.depth_m
        raise IndexError(index)


@dataclass(frozen=True)
class FusedPointCloud:
    scene_points: np.ndarray
    scene_colors: np.ndarray
    environment_points: np.ndarray
    environment_colors: np.ndarray


class RealRobotAdapter(Protocol):
    """Interface expected by the online collector.

    Projects should implement this protocol for their robot SDK. The collector
    intentionally keeps this boundary small: observations for PI05, qpos for
    UR FK, three D435i RGB-D frames by default, and action execution.
    """

    def reset(self) -> None: ...

    def get_observation(self) -> dict: ...

    def get_rgbd_frames(self) -> list[RGBDFrame]: ...

    def execute_action(self, action: np.ndarray) -> None: ...

    def is_done(self) -> bool: ...

    def close(self) -> None: ...


def transform_from_dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    return np.asarray(
        [
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0.0, s_alpha, c_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class UR7ELinkPointSampler:
    """Fixed-topology UR7e link-point sampler based on official UR7e DH FK."""

    def __init__(
        self,
        *,
        points_per_link: int,
        base_to_world: np.ndarray | None = None,
        gripper_width: float = 0.085,
    ):
        if int(points_per_link) < 2:
            raise ValueError("points_per_link must be >= 2")
        if float(gripper_width) <= 0.0:
            raise ValueError("gripper_width must be > 0")
        self.points_per_link = int(points_per_link)
        self.base_to_world = np.eye(4, dtype=np.float64) if base_to_world is None else np.asarray(base_to_world, dtype=np.float64)
        if self.base_to_world.shape != (4, 4):
            raise ValueError(f"base_to_world must have shape (4, 4), got {self.base_to_world.shape}")
        self.gripper_width = float(gripper_width)
        self.link_names = UR7E_LINK_NAMES

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size < 6:
            raise ValueError(f"UR7e qpos must contain at least 6 joints, got {q.size}")
        q = q[:6]
        dh_rows = tuple((*params, theta) for params, theta in zip(UR7E_DH_PARAMETERS, q))

        transforms = [np.asarray(self.base_to_world, dtype=np.float64)]
        current = np.asarray(self.base_to_world, dtype=np.float64).copy()
        for a, alpha, d, theta in dh_rows:
            current = current @ transform_from_dh(a, alpha, d, theta)
            transforms.append(current.copy())
        anchors = np.stack([transform[:3, 3] for transform in transforms], axis=0)
        return anchors.astype(np.float64), transforms[-1][:3, :3].astype(np.float64)

    def link_segments(self, qpos: np.ndarray) -> np.ndarray:
        anchors, eef_rotation = self.forward_kinematics(qpos)
        segments = np.empty((7, 2, 3), dtype=np.float64)
        for link_idx in range(6):
            segments[link_idx, 0] = anchors[link_idx]
            segments[link_idx, 1] = anchors[link_idx + 1]

        gripper_axis = eef_rotation[:, 0]
        norm = float(np.linalg.norm(gripper_axis))
        if norm <= 1e-8:
            raise RuntimeError("UR7e gripper x axis has near-zero norm")
        gripper_axis = gripper_axis / norm
        half_width = 0.5 * self.gripper_width
        eef = anchors[-1]
        segments[6, 0] = eef - half_width * gripper_axis
        segments[6, 1] = eef + half_width * gripper_axis
        return segments.astype(np.float32)

    def link_points(self, qpos: np.ndarray) -> np.ndarray:
        segments = self.link_segments(qpos).astype(np.float64)
        u = np.linspace(0.0, 1.0, self.points_per_link, dtype=np.float64)
        start = segments[:, 0, :]
        end = segments[:, 1, :]
        points = (1.0 - u[None, :, None]) * start[:, None, :]
        points += u[None, :, None] * end[:, None, :]
        return points.astype(np.float32)


def depth_to_world_points(
    frame: RGBDFrame,
    calibration: CameraCalibration,
    *,
    stride: int = 1,
    max_depth: float | None = None,
    keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.camera_name != calibration.name:
        raise ValueError(f"frame camera {frame.camera_name!r} does not match calibration {calibration.name!r}")
    stride = max(int(stride), 1)
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    if keep_mask is None:
        mask = np.isfinite(depth) & (depth > 0.0)
    else:
        mask = np.asarray(keep_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if max_depth is not None:
        mask &= depth <= float(max_depth)
    mask[::stride, ::stride] &= True
    if stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask

    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth[v, u]
    fx = float(calibration.intrinsics[0, 0])
    fy = float(calibration.intrinsics[1, 1])
    cx = float(calibration.intrinsics[0, 2])
    cy = float(calibration.intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("camera intrinsics fx/fy must be non-zero")

    from real_scripts.camera_geometry import pixel_rays
    camera_points = pixel_rays(np.column_stack((u, v)), calibration.intrinsics,
                               calibration.distortion, calibration.distortion_model)*z[:, None]
    homogeneous = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float64)], axis=1)
    world_points = (calibration.camera_to_world @ homogeneous.T).T[:, :3]
    colors = rgb[v, u]
    return world_points.astype(np.float32), colors.astype(np.uint8)


def robot_depth_keep_mask(
    measured_depth_m: np.ndarray,
    rendered_robot_depth_m: np.ndarray,
    *,
    absolute_tolerance_m: float = 0.008,
    relative_tolerance: float = 0.01,
    dilation_pixels: int = 1,
) -> np.ndarray:
    """Return pixels not explained by a rendered robot surface.

    ``rendered_robot_depth_m`` is a z-buffer render of *only* the UR7e and its
    mounted tool in the same colour-camera frame as ``measured_depth_m``.  A
    pixel is removed only when its sensor depth agrees with the visible robot
    surface.  In particular, an object closer to the camera than the robot is
    retained instead of being erased by a silhouette-only mask.

    Rendered depth is dilated conservatively using the closest neighbouring
    robot surface.  This absorbs small mesh/extrinsic errors without treating
    every pixel behind the robot as robot geometry.
    """
    measured = np.asarray(measured_depth_m, dtype=np.float32)
    rendered = np.asarray(rendered_robot_depth_m, dtype=np.float32)
    if measured.shape != rendered.shape:
        raise ValueError(
            "measured_depth_m and rendered_robot_depth_m must have the same shape, "
            f"got {measured.shape} and {rendered.shape}"
        )
    if measured.ndim != 2:
        raise ValueError(f"depth inputs must be HxW, got {measured.shape}")
    if float(absolute_tolerance_m) < 0.0 or float(relative_tolerance) < 0.0:
        raise ValueError("depth tolerances must be non-negative")
    if int(dilation_pixels) < 0:
        raise ValueError("dilation_pixels must be non-negative")

    expanded = np.where(np.isfinite(rendered) & (rendered > 0.0), rendered, np.inf)
    if dilation_pixels:
        expanded = minimum_filter(expanded, size=2 * int(dilation_pixels) + 1, mode="constant", cval=np.inf)

    robot_visible = np.isfinite(expanded)
    measured_valid = np.isfinite(measured) & (measured > 0.0)
    tolerance = float(absolute_tolerance_m) + float(relative_tolerance) * expanded
    matches_robot_surface = robot_visible & measured_valid & (np.abs(measured - expanded) <= tolerance)
    return ~matches_robot_surface


def crop_workspace(
    points: np.ndarray,
    colors: np.ndarray,
    bounds: Iterable[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if bounds is None or points.size == 0:
        return points, colors
    xmin, xmax, ymin, ymax, zmin, zmax = [float(item) for item in bounds]
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    return points[mask], colors[mask]


def filter_robot_points(
    scene_points: np.ndarray,
    robot_link_points: np.ndarray,
    *,
    radius: float,
    chunk_size: int = 16384,
) -> np.ndarray:
    scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    robot_points = np.asarray(robot_link_points, dtype=np.float32).reshape(-1, 3)
    if scene_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if robot_points.shape[0] == 0 or float(radius) <= 0.0:
        return np.ones((scene_points.shape[0],), dtype=bool)

    keep = np.ones((scene_points.shape[0],), dtype=bool)
    radius_sq = float(radius) ** 2
    for start in range(0, scene_points.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scene_points.shape[0])
        diff = scene_points[start:stop, None, :] - robot_points[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=-1), axis=1)
        keep[start:stop] = min_dist_sq > radius_sq
    return keep


def filter_robot_capsules(
    scene_points: np.ndarray,
    robot_link_segments: np.ndarray,
    *,
    radii: float | Iterable[float],
    margin: float = 0.0,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Keep points outside the union of robot-link capsules.

    This is a continuous distance-to-segment test, unlike sampling sparse link
    centerline points.  It is therefore suitable as the dependency-free online
    fallback when a URDF mesh/SDF renderer is unavailable.  ``radii`` must be
    measured/tuned for the actual arm, gripper, and mounted tooling.
    """
    points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    segments = np.asarray(robot_link_segments, dtype=np.float32).reshape(-1, 2, 3)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if segments.shape[0] == 0:
        return np.ones((points.shape[0],), dtype=bool)
    radius_array = np.asarray(radii, dtype=np.float32).reshape(-1)
    if radius_array.size == 1:
        radius_array = np.full((segments.shape[0],), float(radius_array[0]), dtype=np.float32)
    if radius_array.shape != (segments.shape[0],):
        raise ValueError(f"radii must contain 1 or {segments.shape[0]} values, got {radius_array.shape}")
    if np.any(radius_array < 0.0) or float(margin) < 0.0:
        raise ValueError("capsule radii and margin must be non-negative")

    start = segments[:, 0, :]
    direction = segments[:, 1, :] - start
    direction_norm_sq = np.sum(direction * direction, axis=1)
    effective_radius_sq = (radius_array + float(margin)) ** 2
    keep = np.ones((points.shape[0],), dtype=bool)
    for offset in range(0, points.shape[0], int(chunk_size)):
        stop = min(offset + int(chunk_size), points.shape[0])
        delta = points[offset:stop, None, :] - start[None, :, :]
        projection = np.sum(delta * direction[None, :, :], axis=-1)
        t = np.divide(projection, direction_norm_sq[None, :], out=np.zeros_like(projection), where=direction_norm_sq[None, :] > 1e-12)
        t = np.clip(t, 0.0, 1.0)
        closest = start[None, :, :] + t[..., None] * direction[None, :, :]
        distance_sq = np.sum((points[offset:stop, None, :] - closest) ** 2, axis=-1)
        keep[offset:stop] = ~np.any(distance_sq <= effective_radius_sq[None, :], axis=1)
    return keep


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate/overlapping multi-view points into metric voxels."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must contain the same number of rows")
    if points.shape[0] == 0 or float(voxel_size) <= 0.0:
        return points, colors
    grid = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    point_sums = np.zeros((count, 3), dtype=np.float64)
    color_sums = np.zeros((count, 3), dtype=np.float64)
    point_counts = np.bincount(inverse, minlength=count).astype(np.float64)
    np.add.at(point_sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    return (point_sums / point_counts[:, None]).astype(np.float32), np.rint(color_sums / point_counts[:, None]).clip(0, 255).astype(np.uint8)


def fuse_rgbd_frames(
    frames: list[RGBDFrame],
    calibrations: dict[str, CameraCalibration],
    *,
    robot_link_points: np.ndarray,
    robot_link_segments: np.ndarray | None = None,
    robot_link_radii: float | Iterable[float] | None = None,
    robot_filter_margin: float = 0.0,
    camera_names: Iterable[str] | None = None,
    stride: int = 1,
    max_depth: float | None = None,
    robot_filter_radius: float = 0.04,
    workspace_bounds: Iterable[float] | None = None,
    voxel_size: float = 0.0,
    rendered_robot_depths: dict[str, np.ndarray] | None = None,
    rendered_robot_absolute_tolerance_m: float = 0.008,
    rendered_robot_relative_tolerance: float = 0.01,
    rendered_robot_dilation_pixels: int = 1,
) -> FusedPointCloud:
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    requested_names = None if camera_names is None else tuple(str(name) for name in camera_names)
    if requested_names is not None and len(requested_names) != len(set(requested_names)):
        raise ValueError(f"camera_names must be unique, got {requested_names}")
    frames_by_name = {frame.camera_name: frame for frame in frames}
    if requested_names is not None:
        missing = [name for name in requested_names if name not in frames_by_name]
        if missing:
            raise KeyError(f"Requested scene camera frames are missing: {missing}")
        selected_frames = [frames_by_name[name] for name in requested_names]
    else:
        selected_frames = frames
    for frame in selected_frames:
        if frame.camera_name not in calibrations:
            raise KeyError(f"Missing calibration for camera {frame.camera_name!r}")
        keep_mask = None
        if rendered_robot_depths is not None and frame.camera_name in rendered_robot_depths:
            keep_mask = robot_depth_keep_mask(
                frame.depth_m,
                rendered_robot_depths[frame.camera_name],
                absolute_tolerance_m=rendered_robot_absolute_tolerance_m,
                relative_tolerance=rendered_robot_relative_tolerance,
                dilation_pixels=rendered_robot_dilation_pixels,
            )
        points, colors = depth_to_world_points(
            frame,
            calibrations[frame.camera_name],
            stride=stride,
            max_depth=max_depth,
            keep_mask=keep_mask,
        )
        if len(points) > 0:
            point_sets.append(points)
            color_sets.append(colors)

    if not point_sets:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return FusedPointCloud(empty_points, empty_colors, empty_points, empty_colors)

    scene_points = np.concatenate(point_sets, axis=0).astype(np.float32)
    scene_colors = np.concatenate(color_sets, axis=0).astype(np.uint8)
    scene_points, scene_colors = crop_workspace(scene_points, scene_colors, workspace_bounds)
    scene_points, scene_colors = voxel_downsample_points(scene_points, scene_colors, voxel_size=voxel_size)
    if robot_link_segments is not None and robot_link_radii is not None:
        keep = filter_robot_capsules(
            scene_points,
            robot_link_segments,
            radii=robot_link_radii,
            margin=robot_filter_margin,
        )
    else:
        keep = filter_robot_points(scene_points, robot_link_points, radius=robot_filter_radius)
    return FusedPointCloud(
        scene_points=scene_points,
        scene_colors=scene_colors,
        environment_points=scene_points[keep].astype(np.float32),
        environment_colors=scene_colors[keep].astype(np.uint8),
    )


def load_camera_calibrations(path: Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    camera_payloads = payload["cameras"] if isinstance(payload, dict) and "cameras" in payload else payload
    calibrations = {}
    for name, item in camera_payloads.items():
        calibrations[str(name)] = CameraCalibration(
            name=str(name),
            intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
            camera_to_world=np.asarray(item["camera_to_world"], dtype=np.float64),
        )
    return calibrations


def load_camera_calibration_session(path: Path) -> CameraCalibrationSession:
    """Load calibration JSON and infer whether cloud fusion is required.

    New integrated-calibration files carry ``fusion.enabled``.  Older files
    have no such field, for which camera count is the backwards-compatible
    declaration: one camera means direct point-cloud generation; multiple
    calibrated cameras mean voxel fusion in their shared world frame.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if not isinstance(payload, dict) or "cameras" not in payload:
        raise ValueError(f"Calibration file {path} must contain a top-level 'cameras' object")
    calibrations = load_camera_calibrations(path)
    camera_names = tuple(calibrations)
    if not camera_names:
        raise ValueError(f"Calibration file {path} contains no cameras")
    fusion = payload.get("fusion")
    if fusion is not None and not isinstance(fusion, dict):
        raise ValueError("Calibration field 'fusion' must be an object when present")
    declared = None if fusion is None else fusion.get("enabled")
    if declared is not None and not isinstance(declared, bool):
        raise ValueError("Calibration field 'fusion.enabled' must be boolean when present")
    required = len(camera_names) > 1
    if declared is not None and declared != required:
        raise ValueError(
            f"Calibration fusion.enabled={declared} conflicts with its {len(camera_names)} calibrated camera(s)"
        )
    camera_payloads = payload["cameras"]
    serials = {
        name: (str(camera_payloads[name]["serial"]) if camera_payloads[name].get("serial") not in (None, "") else None)
        for name in camera_names
    }
    streams: dict[str, tuple[int, int, int] | None] = {}
    for name in camera_names:
        item = camera_payloads[name]
        values = (item.get("width"), item.get("height"), item.get("fps"))
        if all(value is None for value in values):
            streams[name] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Camera {name!r} must record width, height, and fps together")
        width, height, fps = (int(value) for value in values)
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Camera {name!r} stream width, height, and fps must be positive")
        streams[name] = (width, height, fps)
    return CameraCalibrationSession(calibrations, camera_names, required, serials, streams)


class ReplayJsonlAdapter:
    """Offline adapter for testing collector wiring with recorded JSONL frames.

    Each line must contain qpos plus image paths or inline lists. This adapter
    is intentionally simple; production deployments should implement
    RealRobotAdapter against the robot and camera SDKs directly.
    """

    def __init__(self, path: Path, camera_names: Sequence[str] = DEFAULT_RGBD_CAMERA_NAMES):
        if not camera_names:
            raise ValueError("At least one replay camera name is required")
        self.path = Path(path)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Replay file has no records: {self.path}")
        self.index = 0
        self.last_action = None

    def reset(self) -> None:
        self.index = 0
        self.last_action = None

    def get_observation(self) -> dict:
        record = self.records[min(self.index, len(self.records) - 1)]
        observation = {
            "qpos": np.asarray(record["qpos"], dtype=np.float32),
            "gripper": np.asarray(record.get("gripper", [0.0]), dtype=np.float32),
        }
        for name in self.camera_names:
            rgb_key = f"{name}_rgb"
            if rgb_key in record:
                rgb = record[rgb_key]
            elif name == "wrist":
                rgb = record["front_rgb"]
            else:
                raise KeyError(f"Missing replay RGB field {rgb_key!r}")
            observation[rgb_key] = np.asarray(rgb, dtype=np.uint8)
        return observation

    def get_rgbd_frames(self) -> list[RGBDFrame]:
        record = self.records[min(self.index, len(self.records) - 1)]
        frames = []
        for name in self.camera_names:
            depth_key = f"{name}_depth_m"
            if depth_key not in record:
                raise KeyError(f"Missing replay depth field {depth_key!r}")
            rgb_key = f"{name}_rgb"
            if rgb_key in record:
                rgb = record[rgb_key]
            elif name == "wrist":
                rgb = record["front_rgb"]
            else:
                raise KeyError(f"Missing replay RGB field {rgb_key!r}")
            frames.append(
                RGBDFrame(
                    name,
                    np.asarray(rgb, dtype=np.uint8),
                    np.asarray(record[depth_key], dtype=np.float32),
                )
            )
        return frames

    def execute_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32)
        self.index = min(self.index + 1, len(self.records))

    def is_done(self) -> bool:
        return self.index >= len(self.records)

    def close(self) -> None:
        return None
