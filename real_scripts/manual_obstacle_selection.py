"""Frozen RGB-D OBB annotations and saved voxel selection for explicit replay."""

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from real_scripts.camera_geometry import pixel_rays, project_pixels
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED
from safety_module.voxel_selection import VoxelSelectionScope


def latest_annotation_path(output_dir):
    """Newest completed annotation capture, supporting timestamp and legacy IDs."""
    candidates = [path for path in Path(output_dir).glob("*/annotations.json") if path.is_file()]
    return max(candidates, key=lambda path: (path.parent.stat().st_mtime_ns, path.parent.name), default=None)


def timestamp_capture_directory(output_dir):
    """Reserve a readable local capture time, including timezone and microseconds."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        capture_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
        directory = root / capture_id
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return capture_id, directory


@dataclass(frozen=True)
class ManualFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    accepted: np.ndarray
    calibration: object
    timestamp_ns: int
    raw_depth_units: np.ndarray | None = None
    depth_scale_m: float | None = None
    rgb_calibration: object | None = None
    depth_to_color: np.ndarray | None = None


def rgb_manual_frame(frame, rgb_calibration, accepted=None):
    """Use SDK-aligned color-Z depth in the recorded RGB pixel grid."""
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    if depth.shape != frame.rgb.shape[:2]:
        raise ValueError("RGB annotation requires depth aligned to the RGB dimensions")
    valid = np.isfinite(depth) & (depth > 0)
    if accepted is not None:
        if np.shape(accepted) != depth.shape:
            raise ValueError("Aligned depth evidence and RGB dimensions differ")
        valid &= np.asarray(accepted, dtype=bool)
    native = frame.native_depth
    return ManualFrame(frame.rgb.copy(), depth.copy(), valid, rgb_calibration,
                       int(frame.host_timestamp_ns),
                       native.depth_units.copy() if native is not None else None,
                       native.depth_scale_m if native is not None else None,
                       rgb_calibration,
                       native.depth_to_color.copy() if native is not None else None)


def depth_display_range(frames):
    """Shared color scale from aligned returns within the detector's 2.5 m range."""
    samples = []
    for frame in frames.values():
        if frame.depth_m.ndim != 2 or frame.depth_m.shape != frame.accepted.shape:
            raise ValueError("Depth and accepted mask must have matching HxW dimensions")
        if frame.rgb.ndim != 3 or frame.rgb.shape[-1] != 3:
            raise ValueError("RGB reference must have HxWx3 dimensions")
        valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0) & (frame.depth_m <= 2.5)
        samples.append(frame.depth_m[valid])
    values = np.concatenate(samples)
    if not len(values):
        return .1, 2.5
    low, high = np.percentile(values, [2, 98])
    if high-low < .1:
        midpoint = (low+high)/2
        low, high = max(0., midpoint-.05), min(2.5, midpoint+.05)
    return float(low), float(high)


def colorize_depth(frame, low_m, high_m):
    """Colorize aligned measurements for display without mask overlays."""
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
    normalized = np.zeros(frame.depth_m.shape, np.uint8)
    normalized[valid] = np.rint(255 * (1-np.clip(
        (frame.depth_m[valid]-low_m)/(high_m-low_m), 0., 1.))).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def obb_mask(shape, box):
    """Rasterize an oriented rectangle at integer pixel centers; angle is degrees."""
    try:
        cx, cy, width, height, angle = [float(box[k]) for k in
                                      ("cx", "cy", "width", "height", "angle_deg")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OBB requires cx, cy, width, height, angle_deg") from exc
    if (not np.isfinite([cx, cy, width, height, angle]).all()
            or width < 1 or height < 1 or width > 2 * shape[1] or height > 2 * shape[0]
            or not 0 <= cx < shape[1] or not 0 <= cy < shape[0]):
        raise ValueError("OBB must be finite and within the captured image")
    v, u = np.indices(shape)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    x, y = u - cx, v - cy
    return (np.abs(c*x+s*y) <= width/2) & (np.abs(-s*x+c*y) <= height/2)


def fitted_obb(points, resolution_m):
    """PCA envelope of observed points, padded to enclose their voxel cells."""
    center = points.mean(axis=0)
    covariance = (points-center).T @ (points-center)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1].copy()
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (points-center) @ axes
    # A rotated voxel's half diagonal bounds it along every OBB axis.
    padding = np.sqrt(3) * resolution_m / 2
    low, high = local.min(axis=0)-padding, local.max(axis=0)+padding
    center = center + axes @ ((low+high)/2)
    half = (high-low)/2
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    return {"center": center.tolist(), "axes": axes.tolist(), "half_sizes": half.tolist(),
            "corners": (center + (signs*half) @ axes.T).tolist()}


def selection_rule(camera_names):
    names = tuple(camera_names)
    if names == ("front",):
        return "single_view_depth"
    if names == ("front", "side"):
        return "dual_view_intersection"
    raise ValueError("Manual selection supports front-only or front/side cameras")


def select_voxels(frames, map_payload, annotations, mode="depth", *, camera_names=("front", "side")):
    """Lift accepted depth endpoints, then enforce all configured view OBBs.

    A missing side view never downgrades the default dual-view contract.
    Front-only selection must be explicitly configured by the single-arm scene.
    """
    camera_names = tuple(camera_names)
    fusion_rule = selection_rule(camera_names)
    if mode not in ("depth", "obb"):
        raise ValueError("Selection mode must be depth or obb")
    if not isinstance(annotations, dict) or set(annotations) - set(frames):
        raise ValueError("Annotations must use the captured camera names")
    if set(frames) != set(camera_names):
        raise ValueError(f"Selection requires all configured RGB-D views: {list(camera_names)}")
    grid = ObservationGrid(map_payload)
    object_points, diagnostics, grouped = {}, {}, {}
    total_boxes = 0
    for name, frame in frames.items():
        if frame.depth_m.shape != frame.rgb.shape[:2]:
            raise ValueError(f"{name} depth must be aligned to its RGB annotation image")
        boxes = annotations.get(name, [])
        if not isinstance(boxes, list) or len(boxes) > 32:
            raise ValueError("Each view accepts at most 32 OBBs")
        view_counts = {}
        grouped[name] = {}
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError("Each OBB must be an object")
            object_id = box.get("object_id", "1")
            if not isinstance(object_id, str) or not object_id or len(object_id) > 64:
                raise ValueError("object_id must be a short nonempty string")
            mask = obb_mask(frame.depth_m.shape, box)
            grouped[name].setdefault(object_id, []).append(box)
            mask &= frame.accepted & np.isfinite(frame.depth_m) & (frame.depth_m > 0)
            v, u = np.nonzero(mask)
            camera = frame.calibration
            rays = pixel_rays(np.c_[u, v], camera.intrinsics, camera.distortion, camera.distortion_model)
            points = (rays * frame.depth_m[v, u, None]) @ camera.camera_to_world[:3, :3].T
            points += camera.camera_to_world[:3, 3]
            ids, inside = grid.indices(points)
            valid = np.flatnonzero(inside)
            valid = valid[grid.state[ids[valid]] == OCCUPIED]
            object_points.setdefault(object_id, []).append(points[valid])
            view_counts[object_id] = view_counts.get(object_id, 0) + len(valid)
            total_boxes += 1
        diagnostics[name] = view_counts
    if not total_boxes:
        raise ValueError("Draw at least one OBB before selecting voxels")
    if any(set(grouped[name]) != set(grouped["front"]) for name in camera_names):
        raise ValueError("Each obstacle ID must have an OBB in BOTH front and side RGB images")
    occupied = np.flatnonzero(grid.state == OCCUPIED)
    occupied_centers = grid.centers(occupied)
    projected = {name: project_rgb_centers(occupied_centers, frame) for name, frame in frames.items()}
    objects, all_ids = [], []
    for object_id, chunks in object_points.items():
        points = np.concatenate(chunks)
        if not len(points):
            objects.append({"object_id": object_id, "voxel_count": 0, "obb": None})
            continue
        ids, _ = grid.indices(points)
        ids = np.unique(ids)
        joint = np.ones(len(occupied), bool)
        for name in frames:
            pixels, visible = projected[name]
            in_boxes = np.zeros(len(occupied), bool)
            for annotated_box in grouped[name][object_id]:
                in_boxes |= pixels_in_obb(pixels, annotated_box)
            joint &= visible & in_boxes
        eligible = occupied[joint]
        ids = np.intersect1d(ids, eligible)
        if not len(ids):
            objects.append({"object_id": object_id, "voxel_count": 0, "obb": None})
            continue
        # Fit from selected voxel centers so the envelope includes complete cells.
        box = fitted_obb(grid.centers(ids), grid.resolution_m)
        if mode == "obb":
            local = (occupied_centers-np.asarray(box["center"])) @ np.asarray(box["axes"])
            ids = occupied[joint & np.all(np.abs(local) <= np.asarray(box["half_sizes"])+1e-9, axis=1)]
        all_ids.append(ids)
        objects.append({"object_id": object_id, "voxel_count": len(ids), "obb": box})
    ids = np.unique(np.concatenate(all_ids)) if all_ids else np.empty(0, np.int64)
    return {"mode": mode, "voxel_count": len(ids), "voxel_ids": ids.tolist(),
            "centers": grid.centers(ids).tolist(), "resolution_m": grid.resolution_m,
            "axes": grid.axes.tolist(), "objects": objects, "accepted_points": diagnostics,
            "fusion_rule": fusion_rule, "camera_names": list(camera_names),
            "annotation_image": "recorded_rgb"}


def pixels_in_obb(pixels, box):
    """Continuous projection test, so no resizing or rounded-pixel membership."""
    dx, dy = pixels[:, 0]-float(box["cx"]), pixels[:, 1]-float(box["cy"])
    c, s = np.cos(np.deg2rad(box["angle_deg"])), np.sin(np.deg2rad(box["angle_deg"]))
    return ((np.abs(c*dx+s*dy) <= float(box["width"])/2+1e-9)
            & (np.abs(-s*dx+c*dy) <= float(box["height"])/2+1e-9))


def project_rgb_centers(centers, frame):
    camera = frame.calibration
    local = (centers-camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
    visible = np.isfinite(local).all(axis=1) & (local[:, 2] > 1e-5)
    pixels = np.full((len(centers), 2), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels[visible] = project_pixels(local[visible], camera.intrinsics,
                                         camera.distortion, camera.distortion_model)
    height, width = frame.depth_m.shape
    visible &= (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    indices = np.flatnonzero(visible)
    if len(indices):
        rays = pixel_rays(pixels[indices], camera.intrinsics, camera.distortion, camera.distortion_model)
        errors = np.linalg.norm(rays[:, :2]-local[indices, :2]/local[indices, 2, None], axis=1)
        visible[indices] &= np.isfinite(errors) & (errors <= .5/max(camera.intrinsics[0, 0], camera.intrinsics[1, 1]))
    return pixels, visible


class ManualSelectionStore:
    """One immutable capture and its annotations; HTTP work stays off the detector."""

    def __init__(self, output_dir, *, camera_names=("front", "side"), coordinate_frame="left_base"):
        self.output_dir = Path(output_dir)
        self.camera_names = tuple(camera_names)
        self.fusion_rule = selection_rule(self.camera_names)
        expected_frame = "ur_base" if self.camera_names == ("front",) else "left_base"
        if coordinate_frame != expected_frame:
            raise ValueError(f"Manual selection for {self.camera_names} requires {expected_frame}")
        self.coordinate_frame = coordinate_frame
        self.lock = threading.Lock()
        self.capture_id = None
        self.frames = self.snapshot = self.scene = self.result = None
        self.directory = None
        self.metadata = None
        self.annotations = None
        self.cell_scope = None
        self.revision = 0
        self.annotation_path = None
        self.annotation_filter = None
        self.annotation_error = None

    def load_latest_annotations(self, project_config):
        """Restore RGB filtering for the live viewer without loading frozen maps."""
        if self.camera_names != ("front", "side") or self.coordinate_frame != "left_base":
            return None
        path = latest_annotation_path(self.output_dir)
        if path is None:
            return None
        # Lazy import: the replay filter also uses the RGB geometry in this module.
        from obstacle_validation.manual_annotation_filter import ManualAnnotationFilter
        with self.lock:
            self.annotation_path = path
            try:
                self.annotation_filter = ManualAnnotationFilter(path, project_config)
                self.annotation_error = None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.annotation_filter = None
                self.annotation_error = str(exc)
            self.revision += 1
        return path

    def configuration(self):
        with self.lock:
            return {"camera_names": list(self.camera_names), "coordinate_frame": self.coordinate_frame,
                    "fusion_rule": self.fusion_rule, "annotation_image": "recorded_rgb",
                    "viewer_filter_default": False,
                    "annotation_path": str(self.annotation_path) if self.annotation_path else None,
                    "annotation_error": self.annotation_error}

    def capture(self, snapshot, scene):
        frames = snapshot.get("_manual_frames")
        if not frames or "occupancy_map" not in snapshot:
            raise ValueError("Wait for a configured RGB-D frame with --occupancy-map enabled; restart and recapture")
        if set(frames) != set(self.camera_names) or any(frame.depth_m.shape != frame.rgb.shape[:2] for frame in frames.values()):
            raise ValueError(f"Manual RGB annotation requires aligned depth for {list(self.camera_names)}")
        if snapshot.get("coordinate_frame", self.coordinate_frame) != self.coordinate_frame:
            raise ValueError("Manual snapshot coordinate frame differs from the configured scene")
        low_m, high_m = depth_display_range(frames)
        capture_id, directory = timestamp_capture_directory(self.output_dir)
        metadata = {"capture_id": capture_id, "frame": snapshot["frame"],
                    "coordinate_frame": self.coordinate_frame, "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "host_timestamp_ns": snapshot["host_timestamp_ns"],
                    "annotation_image": "recorded_rgb",
                    "fusion_rule": self.fusion_rule, "camera_names": list(self.camera_names),
                    "depth_display": {"min_m": low_m, "max_m": high_m, "colormap": "turbo",
                                      "invalid_pixels": "black", "mask_overlay": False},
                    "calibration": snapshot.get("calibration"), "cameras": {}}
        arrays = {}
        images = {}
        depth_images = {}
        for name, frame in frames.items():
            ok, encoded = cv2.imencode(".png", frame.rgb[..., ::-1])
            if not ok:
                raise RuntimeError(f"Cannot encode {name} RGB")
            (directory / f"{name}_rgb.png").write_bytes(encoded.tobytes())
            images[name] = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
            ok, encoded_depth = cv2.imencode(".png", colorize_depth(frame, low_m, high_m))
            if not ok:
                raise RuntimeError(f"Cannot encode {name} depth preview")
            (directory / f"{name}_depth_preview.png").write_bytes(encoded_depth.tobytes())
            depth_images[name] = "data:image/png;base64," + base64.b64encode(encoded_depth).decode("ascii")
            camera = frame.calibration
            metadata["cameras"][name] = {
                "width": frame.depth_m.shape[1], "height": frame.depth_m.shape[0],
                "rgb_width": frame.rgb.shape[1], "rgb_height": frame.rgb.shape[0],
                "depth_coordinates": "rgb_color_optical_z_m",
                "host_timestamp_ns": frame.timestamp_ns, "intrinsics": camera.intrinsics.tolist(),
                "camera_to_world": camera.camera_to_world.tolist(),
                "distortion": None if camera.distortion is None else np.asarray(camera.distortion).tolist(),
                "distortion_model": camera.distortion_model}
            if frame.depth_to_color is not None:
                metadata["cameras"][name]["depth_to_color"] = frame.depth_to_color.tolist()
            if frame.rgb_calibration is not None:
                metadata["cameras"][name]["rgb_intrinsics"] = frame.rgb_calibration.intrinsics.tolist()
                metadata["cameras"][name]["rgb_camera_to_world"] = frame.rgb_calibration.camera_to_world.tolist()
            arrays[f"{name}_depth_m"] = frame.depth_m
            arrays[f"{name}_accepted"] = frame.accepted
            if frame.raw_depth_units is not None:
                metadata["cameras"][name]["native_depth_width"] = frame.raw_depth_units.shape[1]
                metadata["cameras"][name]["native_depth_height"] = frame.raw_depth_units.shape[0]
                arrays[f"{name}_native_depth_units"] = frame.raw_depth_units
                arrays[f"{name}_depth_scale_m"] = frame.depth_scale_m
                ok, encoded_units = cv2.imencode(".png", frame.raw_depth_units)
                if not ok:
                    raise RuntimeError(f"Cannot encode {name} native Z16 buffer")
                (directory / f"{name}_native_depth_units.png").write_bytes(encoded_units.tobytes())
                metadata["cameras"][name]["depth_scale_m"] = frame.depth_scale_m
        np.savez_compressed(directory / "depth_evidence.npz", **arrays)
        (directory / "capture.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (directory / "occupancy_map.json").write_text(json.dumps(snapshot["occupancy_map"]), encoding="utf-8")
        (directory / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        metadata.update(images=images, depth_images=depth_images, output_dir=str(directory.resolve()))
        with self.lock:
            self.capture_id, self.frames, self.snapshot, self.scene = capture_id, frames, snapshot, scene
            self.directory, self.result = directory, None
            self.metadata = metadata
            self.annotations = None
            self.cell_scope = None
            self.annotation_path = self.annotation_filter = self.annotation_error = None
            self.revision += 1
        return metadata

    def capture_payload(self):
        with self.lock:
            return self.metadata

    def select(self, capture_id, annotations, mode):
        annotations = json.loads(json.dumps(annotations))
        with self.lock:
            if not capture_id or capture_id != self.capture_id:
                raise ValueError("Capture changed; recapture and annotate the displayed images")
            result = select_voxels(self.frames, self.snapshot["occupancy_map"], annotations, mode,
                                   camera_names=self.camera_names)
            result.update(capture_id=capture_id, frame=self.snapshot["frame"],
                          coordinate_frame=self.coordinate_frame,
                          host_timestamp_ns=self.snapshot["host_timestamp_ns"],
                          output_dir=str(self.directory.resolve()))
            (self.directory / "annotations.json").write_text(
                json.dumps({"capture_id": capture_id, "mode": mode,
                            "annotation_image": "recorded_rgb", "fusion_rule": self.fusion_rule,
                            "coordinate_frame": self.coordinate_frame, "camera_names": list(self.camera_names),
                            "annotations": annotations}, indent=2), encoding="utf-8")
            (self.directory / "selection.json").write_text(json.dumps(result), encoding="utf-8")
            np.savez_compressed(self.directory / "selected_voxels.npz",
                                voxel_ids=np.asarray(result["voxel_ids"], dtype=np.int64),
                                centers=np.asarray(result["centers"]).reshape(-1, 3),
                                axes=np.asarray(result["axes"]), resolution_m=result["resolution_m"])
            self.result = result
            self.annotations = annotations
            self.cell_scope = VoxelSelectionScope(ObservationGrid(self.snapshot["occupancy_map"]),
                                                  result["voxel_ids"])
            self.revision += 1
            return result

    def viewer_filter(self, snapshot, *, tile_voxels=4):
        """Display live occupancy within saved RGB boxes or this session's selected cells.

        Do this only on a viewer request, outside the acquisition loop. Saved
        selections remain frozen; snapshot safety geometry is never replaced.
        """
        with self.lock:
            if self.annotation_path is not None:
                base = {"obbs": [], "voxel_count": 0, "capture_id": self.annotation_path.parent.name,
                        "revision": self.revision, "annotation_path": str(self.annotation_path)}
                if self.annotation_filter is None:
                    return {**base, "state": "annotation_invalid", "error": self.annotation_error}
                if "occupancy_map" not in snapshot:
                    return {**base, "state": "map_unavailable"}
                try:
                    selected = self.annotation_filter(snapshot)
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    return {**base, "state": "annotation_invalid", "error": str(exc)}
                signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
                boxes = [{**box, "corners": (np.asarray(box["center"]) +
                          (signs*np.asarray(box["half_sizes"])) @ np.asarray(box["axes"]).T).tolist()}
                         for box in selected["obbs"]]
                return {**base, "state": "applied", "frame": snapshot["frame"],
                        "annotation_frame": self.annotation_path.parent.name,
                        "scope_policy": self.annotation_filter.scope_policy,
                        "voxel_count": selected["occupancy_map"]["occupied_voxels"], "obbs": boxes}
            annotations = self.annotations
            capture_id = self.capture_id
            captured = self.snapshot
            revision = self.revision
            cell_scope = self.cell_scope
        base = {"obbs": [], "voxel_count": 0, "capture_id": capture_id, "revision": revision}
        frames = snapshot.get("_manual_frames")
        if "occupancy_map" not in snapshot or not frames:
            return {**base, "state": "map_unavailable"}
        if annotations is None:
            return {**base, "state": "missing_annotations"}
        if (snapshot.get("coordinate_frame") != self.coordinate_frame
                or set(frames) != set(self.camera_names)
                or snapshot.get("calibration") != captured.get("calibration")):
            return {**base, "state": "scene_changed"}
        for name, frame in frames.items():
            reference = captured["_manual_frames"][name]
            if (frame.depth_m.shape != reference.depth_m.shape
                    or not np.array_equal(frame.calibration.intrinsics, reference.calibration.intrinsics)
                    or not np.array_equal(frame.calibration.camera_to_world, reference.calibration.camera_to_world)
                    or frame.calibration.distortion_model != reference.calibration.distortion_model
                    or not np.array_equal(frame.calibration.distortion, reference.calibration.distortion)):
                return {**base, "state": "scene_changed"}
        grid = ObservationGrid(snapshot["occupancy_map"])
        try:
            scope = cell_scope.mask_for(grid)
        except ValueError:
            return {**base, "state": "scene_changed"}
        ids = np.flatnonzero((grid.state == OCCUPIED) & scope)
        packing = LocalOccupancyGrid(grid.origin, grid.axes, grid.shape, grid.resolution_m)
        boxes = packing.boxes(occupied_ids=ids, tile_voxels=tile_voxels)
        return {**base, "state": "applied", "frame": snapshot["frame"], "voxel_count": len(ids),
                "scope_policy": "selected_voxel_cells",
                "annotation_frame": captured["frame"],
                "obbs": [{"center": box.center.tolist(), "axes": box.rotation.tolist(),
                          "half_sizes": (box.extents / 2).tolist(), "corners": box.corners.tolist(),
                          "voxel_count": box.point_count} for box in boxes]}

    def scene_payload(self):
        with self.lock:
            if self.scene is None:
                raise ValueError("Capture a frame on the manual annotation page first")
            return {**self.scene, "manual_selection": self.result,
                    "manual_revision": self.revision,
                    "status": f"人工标注冻结帧 {self.snapshot['frame']}（非实时）"}
