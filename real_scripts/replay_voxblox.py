"""Read-only, selected-region TSDF acquisition owned by one replay process."""
from __future__ import annotations

from contextlib import ExitStack
from itertools import product
import threading
import time

import numpy as np

from real_scripts.camera_geometry import pixel_rays
from real_scripts.raw_depth_evidence import RawDepthEvidenceFilter, confirm_raw_depth_views
from real_scripts.voxblox_obstacle_map import VoxbloxOccupancyGrid, require_voxblox
from safety_module.occupancy import GridGeometry, LocalOccupancyGrid


def annotation_reference_grid(scene, resolution_m, *, max_cells=None):
    """Bound annotation frusta by the two configured arm workspaces, without a saved map."""
    corners = []
    for bounds, transform in scene.workspace_regions:
        points = np.asarray(list(product(*zip(bounds[::2], bounds[1::2]))))
        corners.append(points @ transform[:3, :3].T + transform[:3, 3])
    corners = np.concatenate(corners)
    origin = np.floor(corners.min(0)/resolution_m)*resolution_m
    shape = np.ceil((corners.max(0)-origin)/resolution_m).astype(np.int64)
    if max_cells is not None and int(np.prod(shape)) > max_cells:
        raise ValueError("Workspace grid exceeds cell budget; increase resolution or shrink bounds")
    grid = LocalOccupancyGrid(origin, np.eye(3), shape, resolution_m)
    centers = grid.centers(np.arange(grid.size))
    active = np.zeros(grid.size, bool)
    for bounds, transform in scene.workspace_regions:
        bounds = np.asarray(bounds)
        local = (centers-transform[:3, 3]) @ transform[:3, :3]
        active |= np.all((local >= bounds[::2]) & (local <= bounds[1::2]), axis=1)
    grid.active = active
    return grid


def selected_grid(manual_filter):
    """Keep the captured lattice and exact cell scope, with TSDF context padding."""
    reference = manual_filter.reference
    cells = np.column_stack(np.unravel_index(manual_filter.cell_scope.ids, tuple(reference.shape)))
    low = np.maximum(cells.min(0)-4, 0)
    high = np.minimum(cells.max(0)+5, reference.shape)
    origin = reference.origin + (low*reference.resolution_m) @ reference.axes.T
    geometry = GridGeometry(origin, reference.axes, high-low, reference.resolution_m)
    ids, inside = geometry.indices(manual_filter.selected_centers)
    if not inside.all():
        raise ValueError("Selected cells exceed the cropped TSDF grid")
    active = np.zeros(geometry.size, bool)
    active[ids] = True
    return VoxbloxOccupancyGrid(origin, reference.axes, high-low, reference.resolution_m,
                               active=active, esdf=False)


def roi_ray_mask(camera, shape, grid, *, max_depth_m=2.5):
    """Select camera rays intersecting the ROI, including endpoints beyond it.

    Test actual distorted pixel directions, not a projected corner rectangle.
    Ray parameter is optical z (the depth camera's measurement convention).
    """
    y, x = np.indices(shape)
    rays = pixel_rays(np.c_[x.ravel(), y.ravel()], camera.intrinsics,
                      camera.distortion, camera.distortion_model)
    direction = rays @ camera.camera_to_world[:3, :3].T @ grid.axes
    origin = (camera.camera_to_world[:3, 3]-grid.origin) @ grid.axes
    parallel = np.abs(direction) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        first = -origin/direction
        last = (grid.shape*grid.resolution_m-origin)/direction
    entry = np.maximum(np.where(parallel, -np.inf, np.minimum(first, last)).max(1), .01)
    exit_ = np.minimum(np.where(parallel, np.inf, np.maximum(first, last)).min(1), max_depth_m)
    valid = (exit_ >= entry) & ~np.any(parallel & (
        (origin < 0) | (origin >= grid.shape*grid.resolution_m)), axis=1)
    return valid.reshape(shape)


class SelectedTSDFPipeline:
    """No plane fitting, completion model, ESDF, display clouds or native exports."""

    def __init__(self, manual_filter, calibrations, *, grid=None):
        self.manual_filter = manual_filter
        self.calibrations = calibrations
        self.grid = selected_grid(manual_filter) if grid is None else grid
        self.filters = {name: RawDepthEvidenceFilter() for name in calibrations}
        self.roi_masks = {}
        self.support_masks = {}

    def update(self, frames, *, rendered_views, keep_views, qpos, grippers, calibration,
               active_tcp_checks=None):
        import cv2

        began = time.perf_counter()
        timestamp_ns = min(frame.host_timestamp_ns for frame in frames.values())
        evidence = {}
        for name, camera in self.calibrations.items():
            depth = frames[name].depth_m
            if not np.any(np.isfinite(depth) & (depth > 0) & (depth <= 2.5)):
                raise RuntimeError(f"{name}: no valid measured depth; cannot publish replay occupancy")
            if name not in self.roi_masks or self.roi_masks[name].shape != depth.shape:
                roi = roi_ray_mask(camera, depth.shape, self.grid)
                self.roi_masks[name] = roi
                # Retain neighbours around the ROI silhouette for raw evidence.
                self.support_masks[name] = cv2.dilate(roi.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
            evidence[name] = self.filters[name].update(
                depth, keep_views[name] & self.support_masks[name], rendered_views[name], camera,
                timestamp_ns=timestamp_ns)
        confirm_raw_depth_views({name: (self.calibrations[name], frames[name].depth_m, item)
                                 for name, item in evidence.items()})
        hits, uncertain, views = [], [], []
        for name, item in evidence.items():
            camera, depth, roi = self.calibrations[name], frames[name].depth_m, self.roi_masks[name]
            accepted = item.accepted & roi
            hits.append(self.filters[name].world_points(depth, accepted, camera))
            uncertain.append(self.filters[name].world_points(depth, item.uncertain & roi, camera))
            views.append((camera, depth, accepted, rendered_views[name]))
        # Native Voxblox already traverses measured rays. Avoid the generic
        # adapter's second NumPy free-ray traversal (compatibility metadata).
        self.grid.integrate_rgbd_views(
            views, timestamp_ns=timestamp_ns, raw_hits=np.concatenate(hits),
            uncertain_hits=np.concatenate(uncertain), free_ids=np.empty(0, np.int64), ray_stride=6)
        # Keep only the occupancy contract consumed by the planner. In
        # particular, do not encode dense native TSDF/weight/ESDF arrays.
        payload = LocalOccupancyGrid.snapshot(self.grid)
        payload.update(backend="voxblox", field_kind="TSDF", available_fields=["TSDF"],
                       voxblox={"esdf": {"enabled": False}})
        snapshot = self.manual_filter({
            "schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
            "frame": self.grid.frame, "host_timestamp_ns": timestamp_ns, "status": "ok",
            "obstacle_backend": "voxblox", "scene_observation_state": "observed",
            "unknown_obstacles": [], "obbs": [], "occupancy_map": payload,
            "calibration": calibration, "qpos": qpos, "gripper_position": grippers,
            "diagnostics": {"active_tcp_checks": active_tcp_checks or {}, "occupancy": self.grid.diagnostics,
                            "replay_voxblox": {"grid_voxels": self.grid.size,
                                               "field_kind": "TSDF", "ellipsoid_scope": self.manual_filter.scope_policy,
                                               "selected_voxels": int(self.grid.active.sum()),
                                               "roi_pixels": {n: int(m.sum()) for n, m in self.roi_masks.items()},
                                               "esdf_enabled": False, "viewer_enabled": False}}})
        snapshot["diagnostics"]["replay_voxblox"]["processing_seconds"] = time.perf_counter()-began
        return snapshot


class ReplayVoxbloxSource:
    """One worker owns cameras/read-only sockets; consumers get immutable frames.

    Failures take precedence over cached frames, including under latest-age
    replay. Creating the source validates dependencies/calibration without
    opening hardware; the first read starts acquisition.
    """

    def __init__(self, project_config, manual_filter, *, startup_timeout_s=30., robot_mask=False):
        from real_scripts import live_dual_ur7e_obstacle_model as live

        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("Voxblox startup timeout must be positive")
        require_voxblox()
        self.args = live.parse_args(["--project-config", str(project_config),
                                     "--robot-mask" if robot_mask else "--no-robot-mask", "--no-browser",
                                     "--occupancy-backend", "voxblox", "--no-voxblox-esdf"])
        self.scene = live.DualObstacleScene(self.args)
        if self.scene.report.get("sha256") != manual_filter.calibration_digest:
            raise ValueError("Manual selection calibration differs from replay Voxblox")
        stream_size = getattr(manual_filter, "stream_size", None)
        if stream_size is not None:
            self.args.width, self.args.height = stream_size
        self.manual_filter = manual_filter
        self.startup_timeout_s = startup_timeout_s
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.thread = None
        self.latest = None
        self.error = None
        self.started_at = None

    def read(self, *, max_wait_s=0., after_timestamp_ns=None):
        with self.condition:
            if self.stop.is_set():
                raise RuntimeError("Replay Voxblox source is closed")
            if self.thread is None:
                self.started_at = time.monotonic()
                self.thread = threading.Thread(target=self._worker, name="replay-voxblox", daemon=True)
                self.thread.start()
            deadline = time.monotonic()+max_wait_s
            while True:
                if self.error is not None:
                    raise RuntimeError(f"Replay Voxblox acquisition failed: {self.error}") from self.error
                if self.stop.is_set():
                    raise RuntimeError("Replay Voxblox source is closed")
                if self.latest is not None:
                    if after_timestamp_ns is None or self.latest["host_timestamp_ns"] > after_timestamp_ns:
                        return self.latest
                    remaining = deadline-time.monotonic()
                else:
                    remaining = self.started_at+self.startup_timeout_s-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Replay Voxblox has no requested frame; check camera ownership and acquisition")
                self.condition.wait(remaining)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=10.)
            if self.thread.is_alive():
                raise RuntimeError("Replay Voxblox worker did not finish releasing hardware")

    def _worker(self):
        try:
            self._acquire()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def _acquire(self):
        from rtde_receive import RTDEReceiveInterface
        from real_scripts import live_dual_ur7e_obstacle_model as live
        from real_scripts.real_robot_adapter import CameraCalibration, robot_depth_keep_mask
        from real_scripts.rtde_state import read_rtde_tcp_state
        from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource

        scene, args = self.scene, self.args
        with ExitStack() as cleanup:
            receivers = {}
            for arm in scene.arms:
                receiver = RTDEReceiveInterface(arm.ip)
                cleanup.callback(receiver.disconnect)
                receivers[arm.name] = receiver
            grippers = scene.open_grippers(args)
            for gripper in grippers.values():
                cleanup.callback(gripper.close)
            source = RealSenseD435iSource(
                cameras=tuple(D435iCameraConfig(n, scene.report["camera_serials"][n]) for n in scene.calibrations),
                width=args.width, height=args.height, fps=args.fps, wait_timeout_ms=1000, read_retries=1)
            cleanup.callback(source.stop)
            source.start()
            calibrations = dict(scene.calibrations)
            color_info = source.get_color_calibrations()
            for name, info in color_info.items():
                calibrations[name] = CameraCalibration(
                    name, np.asarray(info["intrinsics"]), calibrations[name].camera_to_world,
                    info.get("distortion", calibrations[name].distortion), info.get("distortion_model", "brown_conrady"))
            if hasattr(self.manual_filter, "prepare_reference"):
                self.manual_filter.prepare_reference(
                    annotation_reference_grid(scene, args.occupancy_resolution_m), color_info)
            pipeline = SelectedTSDFPipeline(self.manual_filter, calibrations)
            samples = live.collision_surface_samples(samples_per_face=args.samples_per_face) if args.robot_mask else None
            for _ in range(max(1, args.warmup_frames)):
                if self.stop.is_set():
                    return
                source.read()
            while not self.stop.is_set():
                began = time.perf_counter()
                q_before = ({a.name: np.asarray(receivers[a.name].getActualQ()) for a in scene.arms}
                            if args.robot_mask else {})
                frames = source.read()
                if set(frames) != set(calibrations):
                    raise RuntimeError("Replay Voxblox requires both calibrated camera views")
                stamps = [f.host_timestamp_ns for f in frames.values()]
                if (max(stamps)-min(stamps))/1e9 > scene.project["safety"]["max_camera_skew_s"]:
                    raise RuntimeError("Front/side capture skew exceeded the configured limit")
                states = {a.name: read_rtde_tcp_state(receivers[a.name]) for a in scene.arms}
                positions, _ = scene.read_grippers(grippers)
                feedback_done = time.perf_counter()
                robots, checks = [], {}
                for arm in scene.arms:
                    state = states[arm.name]
                    mm, degrees = live.validate_active_tcp(
                        state.q, state.tcp, scene.flange_to_tcp[arm.name], arm_name=f"{arm.name}_arm",
                        controller_timestamp_s=state.timestamp_s)
                    checks[f"{arm.name}_arm"] = {"position_error_mm": mm, "rotation_error_deg": degrees}
                    if args.robot_mask:
                        q = (q_before[arm.name]+state.q)*.5
                        robots.extend((live.arm_surface_points(q, arm.base_to_left, samples),
                                       scene.tool_points(arm, q, positions[arm.name])))
                robot = np.concatenate(robots) if args.robot_mask else None
                geometry_done = time.perf_counter()
                rendered, keeps = {}, {}
                for name, frame in frames.items():
                    if not args.robot_mask:
                        rendered[name] = np.zeros_like(frame.depth_m)
                        keeps[name] = np.ones(frame.depth_m.shape, bool)
                        continue
                    camera = calibrations[name]
                    rendered[name] = live.render_surface_points_depth(
                        robot, camera.camera_to_world, camera.intrinsics,
                        width=frame.depth_m.shape[1], height=frame.depth_m.shape[0], splat_radius_pixels=3,
                        distortion=camera.distortion, distortion_model=camera.distortion_model)
                    keeps[name] = robot_depth_keep_mask(
                        frame.depth_m, rendered[name], absolute_tolerance_m=args.absolute_tolerance_m,
                        relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels)
                mask_done = time.perf_counter()
                snapshot = pipeline.update(
                    frames, rendered_views=rendered, keep_views=keeps,
                    qpos={a.name: states[a.name].q.tolist() for a in scene.arms},
                    grippers=positions, calibration=scene.report,
                    active_tcp_checks=checks)
                snapshot["diagnostics"]["replay_voxblox"].update(
                    capture_and_feedback_seconds=feedback_done-began,
                    robot_geometry_seconds=geometry_done-feedback_done,
                    robot_mask_seconds=mask_done-geometry_done,
                    robot_mask_enabled=args.robot_mask,
                    total_frame_seconds=time.perf_counter()-began)
                with self.condition:
                    self.latest = snapshot
                    self.condition.notify_all()
