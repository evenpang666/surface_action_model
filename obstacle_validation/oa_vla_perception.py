"""Dynamic paired-image scopes over capture-aligned, full-workspace Voxblox."""

from dataclasses import dataclass
from pathlib import Path
import copy
import time
import numpy as np

from obstacle_validation.manual_annotation_filter import ManualAnnotationFilter
from obstacle_validation.oa_vla_actions import ObservationHold
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.replay_voxblox import SelectedTSDFPipeline, annotation_reference_grid
from real_scripts.voxblox_obstacle_map import VoxbloxOccupancyGrid
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED, OUTSIDE
from obstacle_validation.paired_rgb_geometry import PairedRegionError


@dataclass(frozen=True)
class MappedObservation:
    bundle: object
    snapshot: dict
    calibrations: dict


class WorkspaceScope:
    scope_policy = "full_workspace_before_vla_filter"

    def __call__(self, snapshot):
        return snapshot


class WorkspaceTSDF:
    """One mapping worker owns the volume; network/control never mutate it."""

    def __init__(self, scene, camera_info, args):
        from real_scripts import live_dual_ur7e_obstacle_model as live
        self.scene, self.args, self.live = scene, args, live
        self.calibrations = {}
        for name in ("front", "side"):
            info = camera_info[name]
            original = scene.calibrations[name]
            self.calibrations[name] = CameraCalibration(name, info["intrinsics"], original.camera_to_world,
                                                       info.get("distortion"), info.get("distortion_model", "brown_conrady"))
        reference = annotation_reference_grid(scene, args.voxel_resolution, max_cells=args.max_grid_cells)
        if reference.size > args.max_grid_cells:
            raise ValueError("Workspace voxel grid exceeds --max-grid-cells; increase resolution or shrink workspace")
        grid = VoxbloxOccupancyGrid(reference.origin, reference.axes, reference.shape, reference.resolution_m,
                                   active=reference.active, esdf=False)
        self.pipeline = SelectedTSDFPipeline(WorkspaceScope(), self.calibrations, grid=grid)
        self.samples = live.collision_surface_samples(samples_per_face=16) if args.robot_mask else None

    def update(self, bundle):
        frames = {name: bundle.frames[name] for name in ("front", "side")}
        robot = None
        if self.samples is not None:
            chunks = []
            for index, arm in enumerate(self.scene.arms):
                q = bundle.feedback.q[index * 6:(index + 1) * 6]
                chunks.extend((self.live.arm_surface_points(q, arm.base_to_left, self.samples),
                               self.scene.tool_points(arm, q, bundle.feedback.grip[index])))
            robot = np.concatenate(chunks)
        rendered, keep = {}, {}
        for name, frame in frames.items():
            camera = self.calibrations[name]
            if robot is None:
                rendered[name], keep[name] = np.zeros_like(frame.depth_m), np.ones(frame.depth_m.shape, bool)
            else:
                from real_scripts.real_robot_adapter import robot_depth_keep_mask
                rendered[name] = self.live.render_surface_points_depth(
                    robot, camera.camera_to_world, camera.intrinsics, width=frame.rgb.shape[1], height=frame.rgb.shape[0],
                    splat_radius_pixels=3, distortion=camera.distortion, distortion_model=camera.distortion_model)
                keep[name] = robot_depth_keep_mask(frame.depth_m, rendered[name], absolute_tolerance_m=.015,
                                                   relative_tolerance=.015, dilation_pixels=2)
        snapshot = self.pipeline.update(frames, rendered_views=rendered, keep_views=keep,
            qpos={name: bundle.feedback.q[index * 6:(index + 1) * 6].tolist() for index, name in enumerate(("left", "right"))},
            grippers=dict(zip(("left", "right"), bundle.feedback.grip.tolist())), calibration=self.scene.report,
            active_tcp_checks=bundle.feedback.tcp_checks)
        snapshot["vla_frame_id"] = bundle.frame_id
        # Native TSDF history is retained, but each published occupancy payload
        # is frozen at this packet. Never apply a returned OBB to a newer map.
        return MappedObservation(bundle, copy.deepcopy(snapshot), self.calibrations)


class DynamicOBBFilter(ManualAnnotationFilter):
    """Reuse replay projection/voxel packing without annotation files/camera reads."""
    scope_policy = "vla_paired_rgb_intersection"
    geometry_source = "vla_selected_live_voxels"
    ignore_inconsistent_pairs = True

    def __init__(self, mapped, prediction, *, padding_pixels=2., minimum_score=.5):
        bundle = mapped.bundle
        if (prediction.get("model_type") != "oa-vla" or prediction.get("detection_status") != "ok"
                or prediction.get("detection_box_format") != "cxcywh_angle_deg_original_pixels"
                or prediction.get("pair_id_scope") != "current_prediction"):
            raise ObservationHold("No trained, valid paired OBB prediction")
        objects = prediction.get("objects")
        if not isinstance(objects, list) or len(objects) > 1024:
            raise ValueError("Invalid object-pair list")
        if not objects:
            raise ObservationHold("No obstacle pairs returned; cannot certify an empty scene from a filtered prediction")
        self.path = Path(f"socket_frame_{bundle.frame_id}")
        self.calibration_digest = bundle.calibration_id
        self.cameras = mapped.calibrations
        self.sizes = {name: (bundle.frames[name].rgb.shape[1], bundle.frames[name].rgb.shape[0]) for name in self.cameras}
        self.annotations, self.pair_ids = {name: [] for name in self.cameras}, []
        for pair in objects:
            identity, score = pair.get("pair_id"), pair.get("score")
            if (type(identity) is not int or identity < 0 or identity in self.pair_ids
                    or not isinstance(score, (float, int)) or not np.isfinite(score) or not minimum_score <= score <= 1):
                raise ObservationHold("Invalid or insufficient-confidence object pair")
            self.pair_ids.append(identity)
            for name in ("front", "side"):
                entry = pair.get(name)
                if not isinstance(entry, dict):
                    raise ObservationHold("An object is invisible in one view; stop and reobserve instead of dropping it")
                values = np.asarray(entry.get("obb"), float)
                width, height = self.sizes[name]
                if (values.shape != (5,) or not np.isfinite(values).all() or min(values[2:4]) <= 0
                        or not 0 <= values[0] < width or not 0 <= values[1] < height):
                    raise ObservationHold("Predicted OBB is invalid or centered outside the RGB image")
                self.annotations[name].append(dict(object_id=str(identity), cx=float(values[0]), cy=float(values[1]),
                    width=float(values[2] + 2 * padding_pixels), height=float(values[3] + 2 * padding_pixels), angle_deg=float(values[4])))
        self.geometry_key = self.scope = self.packer = None
        self.last_diagnostics = None
        self.pair_occupied = {identity: 0 for identity in self.pair_ids}
        self.frame_id = bundle.frame_id
        self.expected_capture_ns = min(bundle.frames[name].host_timestamp_ns for name in self.cameras)

    def __call__(self, snapshot):
        if snapshot.get("vla_frame_id") != self.frame_id or snapshot.get("host_timestamp_ns") != self.expected_capture_ns:
            raise ValueError("OBB response and RGB-D/TSDF snapshot belong to different frames")
        # Both replay and inference use the same transparent-region completion.
        try:
            result = super().__call__(snapshot)
        except PairedRegionError as exc:
            raise ObservationHold(str(exc)) from exc
        self.pair_occupied = {identity: self.object_occupied[str(identity)] for identity in self.pair_ids}
        result["obstacle_scope"] = "vla_paired_rgb_v1"
        result["diagnostics"]["vla_pair_voxels"] = {str(key): count for key, count in self.pair_occupied.items()}
        if result["diagnostics"]["paired_rgb_obstacles"]["ignored_objects"]:
            result = self._retain_measured_workspace(snapshot, result)
        return result


    def _retain_measured_workspace(self, snapshot, result):
        """Unreliable RGB pairs may not erase stronger measured evidence."""
        live = ObservationGrid(snapshot["occupancy_map"])
        packer = LocalOccupancyGrid(live.origin, live.axes, live.shape, live.resolution_m)
        occupied = np.flatnonzero(live.state == OCCUPIED)
        measured = [dict(center=box.center.tolist(), axes=box.rotation.tolist(),
                         half_sizes=(box.extents / 2).tolist(), voxel_count=box.point_count,
                         geometry_source="vla_fallback_full_workspace_live_voxels")
                    for box in packer.boxes(occupied_ids=occupied)]
        # Keep other valid transparent pairs, but discard all previously packed
        # filtered cuboids so measured cells occur only once.
        transparent = [box for box in result["obbs"] if box.get("ellipsoid_shape") is not None]
        result.update(obbs=measured + transparent, occupancy_map=snapshot["occupancy_map"],
                      obstacle_scope="vla_full_voxels_inconsistent_pairs_v1",
                      unknown_obstacles=snapshot.get("unknown_obstacles", []),
                      scene_observation_state=snapshot.get("scene_observation_state", "observed"),
                      status=snapshot.get("status", ""))
        diag = result["diagnostics"]["paired_rgb_obstacles"]
        diag.update(measured_voxel_scope="full_workspace_on_inconsistent_pairs",
                    measured_occupied_voxels=len(occupied))
        result["diagnostics"]["manual_selection"] = {
            **result["diagnostics"]["manual_selection"], "scope_policy": diag["measured_voxel_scope"],
            "scope_voxels": int(np.count_nonzero(live.state != OUTSIDE)), "occupied_voxels": len(occupied)}
        self.last_diagnostics = result["diagnostics"]["manual_selection"]
        return result


def inconsistent_pairs_allow_empty_geometry(snapshot, prediction):
    """Only explicitly rejected, two-view pairs may admit an empty scene here."""
    diag = snapshot.get("diagnostics", {}).get("paired_rgb_obstacles", {})
    pairs = prediction.get("objects", [])
    ignored = diag.get("ignored_objects", [])
    return (snapshot.get("obstacle_scope") == "vla_full_voxels_inconsistent_pairs_v1"
            and bool(pairs) and len(ignored) == len(pairs)
            and set(ignored) == {str(pair.get("pair_id")) for pair in pairs}
            and all(isinstance(pair.get(name), dict) for pair in pairs for name in ("front", "side"))
            and diag.get("measured_voxel_scope") == "full_workspace_on_inconsistent_pairs"
            and diag.get("measured_occupied_voxels") == 0
            and diag.get("transparent_count") == 0)


def mapping_loop(workers, frames, output, factory):
    version, pipeline = 0, None
    while not workers.stop.is_set():
        try:
            version, bundle = frames.get(after=version, timeout=.25)
        except TimeoutError:
            continue
        pipeline = factory() if pipeline is None else pipeline
        output.publish(pipeline.update(bundle))
