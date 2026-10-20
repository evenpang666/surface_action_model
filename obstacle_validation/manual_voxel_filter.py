"""Explicit, fixed selected-cell scope for model-free live voxel replay."""

import json
from pathlib import Path

import numpy as np

from safety_module.occupancy import (FREE, OCCUPIED, OUTSIDE,
                                     LocalOccupancyGrid, ObservationGrid, _pack)
from safety_module.voxel_selection import VoxelSelectionScope


class ManualVoxelFilter:
    """Constrain only live occupancy inside the reviewed capture's selected cells."""

    scope_policy = "selected_voxel_cells"
    geometry_source = "manual_selected_live_voxels"

    def __init__(self, selection_path):
        self.path = Path(selection_path).resolve()
        selection = json.loads(self.path.read_text(encoding="utf-8"))
        capture = json.loads((self.path.parent / "capture.json").read_text(encoding="utf-8"))
        saved_map = json.loads((self.path.parent / "occupancy_map.json").read_text(encoding="utf-8"))
        self.reference = ObservationGrid(saved_map)
        if (selection.get("capture_id") != capture.get("capture_id")
                or selection.get("frame") != capture.get("frame")
                or selection.get("host_timestamp_ns") != capture.get("host_timestamp_ns")
                or saved_map.get("capture_timestamp_ns") != capture.get("host_timestamp_ns")
                or capture.get("coordinate_frame") != "left_base"
                or capture.get("annotation_image") != "recorded_rgb"
                or selection.get("fusion_rule") != "dual_view_intersection"):
            raise ValueError("Manual selection and frozen RGB-D capture do not match")
        ids = np.asarray(selection.get("voxel_ids"), dtype=np.int64)
        if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= self.reference.size)
                or np.any(self.reference.state[ids] != OCCUPIED)
                or selection.get("voxel_count") != len(ids)):
            raise ValueError("Manual selection must contain nonempty occupied capture voxels")
        self.boxes = []
        for item in selection.get("objects", []):
            if item.get("voxel_count", 0) <= 0:
                continue
            box = item.get("obb")
            if not isinstance(box, dict):
                raise ValueError("Every selected obstacle needs a saved 3D OBB")
            center = np.asarray(box.get("center"), float)
            axes = np.asarray(box.get("axes"), float)
            half = np.asarray(box.get("half_sizes"), float)
            if (center.shape != (3,) or axes.shape != (3, 3) or half.shape != (3,)
                    or not np.isfinite(center).all() or not np.isfinite(axes).all()
                    or not np.isfinite(half).all() or np.any(half <= 0)
                    or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4)
                    or not np.isclose(np.linalg.det(axes), 1., atol=1e-4)):
                raise ValueError("Invalid saved manual 3D OBB")
            self.boxes.append((center, axes, half))
        if not self.boxes:
            raise ValueError("Manual selection has no occupied 3D OBB")
        self.cell_scope = VoxelSelectionScope(self.reference, ids)
        self.selected_centers = self.cell_scope.centers
        captured_scope = np.zeros(len(ids), bool)
        for center, axes, half in self.boxes:
            captured_scope |= np.all(np.abs((self.selected_centers-center) @ axes) <= half+1e-9, axis=1)
        if not np.all(captured_scope):
            raise ValueError("Saved manual OBBs do not enclose their selected voxels")
        self.calibration_digest = (capture.get("calibration") or {}).get("sha256")
        if not self.calibration_digest:
            raise ValueError("Manual capture requires a calibration digest")
        self.geometry_key = None
        self.scope = None
        self.packer = None
        self.last_diagnostics = None

    def _prepare_geometry(self, live):
        """Cache selected cells without expanding to their fitted OBBs."""
        key = self.cell_scope.geometry_key(live)
        scope = self.cell_scope.mask_for(live)
        if key == self.geometry_key:
            return
        self.packer = LocalOccupancyGrid(live.origin, live.axes, live.shape, live.resolution_m)
        self.scope, self.geometry_key = scope, key

    def __call__(self, snapshot):
        status = str(snapshot.get("status", ""))
        # Failed acquisition/calibration snapshots have no map or backend. Report
        # that original failure before checking selection compatibility.
        if (status == "initializing" or (status.startswith("error:")
                and status != "error: unknown obstacle space"
                and not status.startswith("error: unstable obstacle point clusters"))):
            raise RuntimeError(f"Live obstacle detector: {status}")
        if snapshot.get("coordinate_frame") != "left_base":
            raise ValueError("Dual-arm manual replay requires the live left_base coordinate frame")
        payload = snapshot.get("occupancy_map")
        backend = snapshot.get("obstacle_backend") or (payload.get("backend") if isinstance(payload, dict) else None)
        if backend not in ("probabilistic_voxels", "numpy", "octomap", "voxblox"):
            raise ValueError(f"Manual selection requires a live voxel map (NumPy, OctoMap or Voxblox); "
                             f"received backend={backend!r}, status={status!r}. "
                             "Start detection with --occupancy-map --occupancy-backend voxblox")
        if (snapshot.get("calibration") or {}).get("sha256") != self.calibration_digest:
            raise ValueError("Manual selection calibration differs from the live detector")
        if not isinstance(payload, dict) or payload.get("capture_timestamp_ns") != snapshot.get("host_timestamp_ns"):
            raise ValueError("Live occupancy map is missing or not capture aligned")
        live = ObservationGrid(payload)
        self._prepare_geometry(live)
        other_unknown = [item for item in snapshot.get("unknown_obstacles", [])
                         if item.get("reason") != "unobserved_occupied_voxels"]
        state = live.state.copy()
        state[~self.scope] = OUTSIDE
        selected_ids = np.flatnonzero(state == OCCUPIED)
        boxes = self.packer.boxes(occupied_ids=selected_ids)
        # Rebuild exact occupied-cell geometry even when an older detector
        # publishes inflated OBBs. Fitted selection OBBs never broaden scope.
        result_boxes = []
        for box in boxes:
            half = box.extents / 2
            result_boxes.append({"center": box.center.tolist(), "axes": box.rotation.tolist(),
                                 "half_sizes": half.tolist(), "voxel_count": box.point_count,
                                 "geometry_source": self.geometry_source})
        counts = np.bincount(state, minlength=4)
        filtered_map = {**payload, "state_data": _pack(state),
                        "unknown_voxels": 0, "free_voxels": int(counts[FREE]),
                        "occupied_voxels": int(counts[OCCUPIED]), "outside_voxels": int(counts[OUTSIDE]),
                        "historical_unknown_voxels": 0}
        # The global ESDF contains distances to unselected obstacles. Cropping
        # its sample mask cannot recompute those distances for our new scope.
        if payload.get("backend") == "voxblox" and (payload.get("voxblox") or {}).get("esdf"):
            native = payload["voxblox"]
            filtered_map["voxblox"] = {**native, "esdf": {
                **native["esdf"], "guidance_usable": False,
                "guidance_disabled_reason": "manual_obstacle_scope_not_recomputed"}}
        diagnostics = {**snapshot.get("diagnostics", {}), "box_margin_m": 0.,
                       "manual_selection": {"path": str(self.path),
                                            "scope_policy": self.scope_policy,
                                            "scope_voxels": int(self.scope.sum()),
                                            "occupied_voxels": len(selected_ids),
                                            "detector_occupied_voxels": int(payload["occupied_voxels"])}}
        self.last_diagnostics = diagnostics["manual_selection"]
        status = snapshot.get("status", "")
        if not other_unknown and status == "error: unknown obstacle space":
            status = "manual selected voxel scope"
        return {**snapshot, "obbs": result_boxes, "occupancy_map": filtered_map,
                "obstacle_scope": "manual_selected_voxels_v1",
                "unknown_obstacles": other_unknown,
                "scene_observation_state": "unknown" if other_unknown else "observed",
                "status": status, "diagnostics": diagnostics}
