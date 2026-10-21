"""Paired RGB voxel filtering and depth-free transparent obstacle completion."""

import json
from pathlib import Path

import numpy as np

from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
from obstacle_validation.paired_rgb_geometry import TRANSPARENT_SOURCE, PairedIntersectionEmpty, transparent_obstacle
from real_scripts.batch_obstacle_annotation import validate_boxes
from real_scripts.camera_geometry import project_pixels
from real_scripts.manual_obstacle_selection import pixels_in_obb
from real_scripts.real_robot_adapter import CameraCalibration
from safety_module.dual_camera_calibration import calibration_from_project
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED, OUTSIDE
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config
from safety_module.voxel_selection import VoxelSelectionScope


class ManualAnnotationFilter(ManualVoxelFilter):
    """Keep live cells whose centers project into matching front/side RGB OBBs.

    No saved selection/map/depth is read. A legacy capture.json supplies RGB
    provenance only; standalone annotations already include that provenance.
    """

    scope_policy = "dual_rgb_annotation_intersection"
    geometry_source = "manual_annotated_live_voxels"
    ignore_inconsistent_pairs = False  # Reviewed replay labels remain strict.

    def __init__(self, annotation_path, project_config=DEFAULT_PROJECT_CONFIG):
        self.path = Path(annotation_path).expanduser().resolve()
        row = json.loads(self.path.read_text(encoding="utf-8"))
        legacy = "annotations" in row
        capture = row
        if legacy:
            capture = json.loads((self.path.parent / "capture.json").read_text(encoding="utf-8"))
            if row.get("capture_id") != capture.get("capture_id"):
                raise ValueError("Annotation and capture IDs differ")
        if (row.get("annotation_image") != "recorded_rgb"
                or capture.get("annotation_image") != "recorded_rgb"
                or row.get("camera_names") != ["front", "side"]
                or row.get("coordinate_frame", "left_base") != "left_base"
                or capture.get("coordinate_frame", "left_base") != "left_base"
                or row.get("fusion_rule", "dual_view_intersection") != "dual_view_intersection"):
            raise ValueError("Replay annotations require front/side recorded RGB in left_base")
        if not legacy and (row.get("reviewed") is not True
                           or row.get("schema") not in ("dual_rgb_obb_annotation_v2", "dual_rgb_paired_obb_v3")):
            raise ValueError("Standalone annotations must be saved, reviewed RGB OBB v2/v3 labels")
        if row.get("status", "objects") != "objects":
            raise ValueError("Annotation avoidance requires at least one obstacle in both views")
        self.metadata = capture.get("cameras", {})
        self.sizes = {}
        dimensions = {}
        for name in ("front", "side"):
            info = self.metadata.get(name, {})
            width, height = info.get("rgb_width"), info.get("rgb_height")
            if type(width) is not int or type(height) is not int or min(width, height) < 1:
                raise ValueError(f"{name}: annotation requires recorded RGB width/height")
            self.sizes[name] = (width, height)
            dimensions[name] = {"rgb_width": width, "rgb_height": height}
        annotations = row.get("annotations" if legacy else "boxes")
        if row.get("schema") == "dual_rgb_paired_obb_v3":
            flags = row.get("visibility")
            validate_boxes(annotations, "objects", dimensions, flags)
            if flags is None or any(not all(item.values()) for item in flags.values()):
                raise ValueError("Dual-view voxel intersection requires every object visible in both views; "
                                 "single-view visibility labels are supported by OA-VLA training")
        self.annotations = validate_boxes(annotations, "objects", dimensions)
        project, _ = load_project_config(project_config)
        self.shared = calibration_from_project(project)
        if self.shared is None:
            raise ValueError("RGB annotation replay requires shared dual-camera calibration")
        self.calibration_digest = self.shared.digest
        captured_calibration = capture.get("calibration") or {}
        if captured_calibration.get("sha256", self.calibration_digest) != self.calibration_digest:
            raise ValueError("Annotation calibration differs from the replay project")
        for name in self.sizes:
            serial = ((self.metadata[name].get("stream") or {}).get("serial")
                      or captured_calibration.get("camera_serials", {}).get(name))
            if serial != self.shared.serials[name]:
                raise ValueError(f"{name}: annotation camera serial {serial!r} differs from calibrated {self.shared.serials[name]!r}")
        self.cameras = {}
        for name, info in self.metadata.items():
            saved = (info.get("stream") or {}).get("color_calibration")
            if saved is None and "intrinsics" in info:
                saved = info
            if saved is not None:
                self.cameras[name] = self._camera(name, saved)
        self.geometry_key = self.scope = self.packer = None
        self.last_diagnostics = None

    @property
    def stream_size(self):
        sizes = set(self.sizes.values())
        if len(sizes) != 1:
            raise ValueError("Embedded annotation replay requires matching front/side RGB sizes")
        return next(iter(sizes))

    def _camera(self, name, info):
        K = np.asarray(info["intrinsics"], float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError(f"{name}: invalid annotation RGB intrinsics")
        return CameraCalibration(name, K, self.shared.camera_to_left_base[name],
                                 info.get("distortion"), info.get("distortion_model", "brown_conrady"))

    def bind_color_calibrations(self, color_info):
        """Older standalone labels recover exact intrinsics from the same SDK profile."""
        for name in self.sizes:
            if name in self.cameras:
                continue
            info = color_info.get(name, {})
            if (info.get("width"), info.get("height")) != self.sizes[name]:
                width, height = self.sizes[name]
                raise ValueError(f"{name}: annotation has no saved RGB intrinsics; start detection with "
                                 f"--width {width} --height {height} and current RGB calibration metadata")
            self.cameras[name] = self._camera(name, info)

    def _object_masks(self, grid, active=None):
        """Yield bounded projection batches, preserving each object's identity."""
        if set(self.cameras) != {"front", "side"}:
            raise ValueError("Annotation RGB intrinsics have not been initialized")
        candidates = np.arange(grid.size) if active is None else np.flatnonzero(active)
        object_ids = sorted({box["object_id"] for box in self.annotations["front"]})
        for start in range(0, len(candidates), 65536):
            ids = candidates[start:start+65536]
            centers = grid.centers(ids)
            projected = {}
            for name, camera in self.cameras.items():
                local = (centers-camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
                visible = (local[:, 2] > .01) & (local[:, 2] <= 2.5)
                pixels = np.full((len(ids), 2), np.nan)
                pixels[visible] = project_pixels(local[visible], camera.intrinsics,
                                                  camera.distortion, camera.distortion_model)
                width, height = self.sizes[name]
                visible &= (np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
                projected[name] = pixels, visible
            masks = {}
            for object_id in object_ids:
                joint = np.ones(len(ids), bool)
                for name in self.cameras:
                    pixels, visible = projected[name]
                    matched = np.zeros(len(ids), bool)
                    for box in self.annotations[name]:
                        if box["object_id"] == object_id:
                            matched |= pixels_in_obb(pixels, box)
                    joint &= visible & matched
                masks[object_id] = joint
            yield ids, masks

    def scope_mask(self, grid, active=None):
        """Union per-object intersections; different object IDs never cross-match."""
        mask = np.zeros(grid.size, bool)
        for ids, masks in self._object_masks(grid, active):
            mask[ids] = np.logical_or.reduce(list(masks.values()))
        return mask

    def prepare_reference(self, grid, color_info):
        """Build an ROI from annotations before the embedded TSDF starts integrating."""
        self.bind_color_calibrations(color_info)
        scope = np.zeros(grid.size, bool)
        seen = {box["object_id"]: False for box in self.annotations["front"]}
        for candidates, masks in self._object_masks(grid, grid.active):
            for identity, mask in masks.items():
                scope[candidates] |= mask
                seen[identity] |= bool(mask.any())
        # A thin transparent object may intersect no voxel centre. Still retain
        # TSDF context around its continuous region instead of failing startup
        # or cropping it away because another object's ROI is nonempty.
        missing = [identity for identity, present in seen.items() if not present]
        if missing:
            centers = grid.centers(np.flatnonzero(grid.active))
            active_ids = np.flatnonzero(grid.active)
            for identity in missing:
                paired = {name: next(box for box in self.annotations[name] if box["object_id"] == identity)
                          for name in ("front", "side")}
                geometry = transparent_obstacle(identity, paired, self.cameras, self.sizes, grid)
                offsets = (centers - geometry["center"]) @ np.asarray(geometry["axes"])
                support = np.all(np.abs(offsets) <= np.asarray(geometry["half_sizes"])
                                 + np.sqrt(3) * grid.resolution_m / 2, axis=1)
                scope[active_ids[support]] = True
        ids = np.flatnonzero(scope)
        if not len(ids):
            raise ValueError("Front/side annotation intersection has no cells inside the replay workspace")
        self.reference = grid
        self.cell_scope = VoxelSelectionScope(grid, ids)
        self.selected_centers = self.cell_scope.centers

    def _prepare_geometry(self, live):
        key = VoxelSelectionScope.geometry_key(live)
        if key != self.geometry_key:
            self.scope = self.scope_mask(live, live.state != OUTSIDE)
            self.packer = LocalOccupancyGrid(live.origin, live.axes, live.shape, live.resolution_m)
            self.geometry_key = key
            self.transparent_cache = {}

    def _complete_transparent_objects(self, snapshot, result):
        """No occupied support + a valid paired volume => transparent hypothesis.

        Recount every capture: the projection scope is cached, occupancy is not.
        Never insert visual hypotheses into the measured occupancy payload.
        """
        live = ObservationGrid(snapshot["occupancy_map"])
        counts = {box["object_id"]: 0 for box in self.annotations["front"]}
        for _, masks in self._object_masks(live, live.state == OCCUPIED):
            for identity, mask in masks.items():
                counts[identity] += int(mask.sum())
        objects = {}
        transparent_ids, ignored_ids = [], []
        geometry_grid = getattr(self, "reference", live)
        for identity, count in counts.items():
            objects[identity] = dict(occupied_voxels=count,
                                     geometry_source=self.geometry_source if count else TRANSPARENT_SOURCE)
            if count:
                continue
            if identity not in self.transparent_cache:
                paired = {name: next(box for box in self.annotations[name] if box["object_id"] == identity)
                          for name in ("front", "side")}
                try:
                    self.transparent_cache[identity] = transparent_obstacle(
                        identity, paired, self.cameras, self.sizes, geometry_grid)
                except PairedIntersectionEmpty as exc:
                    if not self.ignore_inconsistent_pairs:
                        raise
                    objects[identity].update(geometry_source="ignored_inconsistent_rgb_pair",
                                             reason="no_3d_intersection", detail=str(exc))
                    ignored_ids.append(identity)
                    continue
            result["obbs"].append(self.transparent_cache[identity])
            transparent_ids.append(identity)
        result["diagnostics"]["paired_rgb_obstacles"] = dict(
            objects=objects, transparent_objects=transparent_ids,
            transparent_count=len(transparent_ids), ignored_objects=ignored_ids,
            ignored_count=len(ignored_ids),
            occupancy_contains_visual_hypotheses=False)
        self.object_occupied = counts
        return result

    def __call__(self, snapshot):
        if isinstance(snapshot.get("occupancy_map"), dict):
            self.bind_color_calibrations(snapshot.get("camera_color_calibrations", {}))
        result = super().__call__(snapshot)
        result = self._complete_transparent_objects(snapshot, result)
        result["obstacle_scope"] = "manual_rgb_annotations_v1"
        return result
