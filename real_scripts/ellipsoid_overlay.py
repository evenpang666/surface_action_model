"""On-demand display of the replay controller's volume models.

This runs in viewer HTTP requests, never in camera acquisition or control.
The scene supplies its actual calibrated transforms; no hardware is opened.
"""
import threading
from types import SimpleNamespace

import numpy as np


def encode_ellipsoid(body):
    lengths, axes = np.linalg.eigh(body.shape)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1  # Preserve outward mesh winding in the browser.
    return {"center": body.center.tolist(), "axes": axes.tolist(),
            "radii": np.sqrt(lengths).tolist(), "key": body.key,
            "arm": body.arm, "link": body.link}


class EllipsoidOverlay:
    def __init__(self, scene):
        self.scene = scene
        self.lock = threading.Lock()
        self.robot = None
        self.obstacle_key = None
        self.obstacles = []

    def build(self, snapshot, *, groups, frozen=False, filtered=False):
        from obstacle_validation.ellipsoid_geometry import RobotEllipsoids, fit_obstacles
        from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBFConfig
        from real_scripts.real_cbf_qp import OrientedBox

        scope = (("frozen_selection" if filtered else "frozen_all_cells") if frozen
                 else "selected_cells" if filtered else "all_cells")
        result = {"scope": scope, "frame": snapshot.get("frame"),
                  "robot": [], "obstacles": [], "messages": [],
                  "model": "replay_collision_envelopes", "margin_included": False}
        with self.lock:
            if "obstacles" in groups:
                if frozen and filtered:
                    selection = snapshot.get("manual_selection")
                    if selection is None:
                        items = []
                        result["messages"].append("尚无冻结筛选体素")
                    else:
                        half = np.full(3, selection["resolution_m"] / 2)
                        items = [{"center": c, "axes": selection["axes"], "half_sizes": half}
                                 for c in selection["centers"]]
                elif filtered:
                    selected = snapshot.get("viewer_filter", {})
                    items = selected.get("obbs", []) if selected.get("state") == "applied" else []
                    if selected.get("state") != "applied":
                        result["messages"].append("实时筛选不可用，请先完成人工体素筛选")
                elif isinstance(snapshot.get("occupancy_map"), dict):
                    # Fit exact current occupied cells, independent of the
                    # detector's padded OBBs and any saved annotation scope.
                    from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED
                    grid = ObservationGrid(snapshot["occupancy_map"])
                    packer = LocalOccupancyGrid(grid.origin, grid.axes, grid.shape, grid.resolution_m)
                    items = [{"center": b.center, "axes": b.rotation, "half_sizes": b.extents / 2}
                             for b in packer.boxes(occupied_ids=np.flatnonzero(grid.state == OCCUPIED))]
                else:
                    items = snapshot.get("obbs", [])
                boxes = [OrientedBox(np.asarray(b["center"], float), np.asarray(b["axes"], float),
                                     np.asarray(b["half_sizes"], float),
                                     None if b.get("ellipsoid_shape") is None else np.asarray(b["ellipsoid_shape"], float),
                                     str(b.get("object_id", ""))) for b in items]
                config = EllipsoidMCBFConfig(**self.scene.project.get("safety", {}).get("ellipsoid_mcbf", {}))
                key = (config.obstacle_max_parts, config.obstacle_max_span_m,
                       tuple((np.r_[b.center, b.axes.ravel(), b.half_sizes].tobytes(),
                              None if b.ellipsoid_shape is None else b.ellipsoid_shape.tobytes(), b.object_id)
                             for b in boxes))
                if key != self.obstacle_key:
                    models = (fit_obstacles(boxes, max_parts=config.obstacle_max_parts,
                                            max_span=config.obstacle_max_span_m) if boxes else [])
                    self.obstacles = [encode_ellipsoid(b) for b in models]
                    self.obstacle_key = key
                result["obstacles"] = self.obstacles
                result["source_boxes"] = len(boxes)
            if "robot" in groups:
                qpos, grip = snapshot.get("qpos", {}), snapshot.get("gripper_position", {})
                if (snapshot.get("coordinate_frame") != "left_base"
                        or not all(k in qpos and k in grip for k in ("left", "right"))):
                    result["messages"].append("连杆椭球需要双 UR7e 实时关节与夹爪状态")
                else:
                    if self.robot is None:
                        transforms = self.scene.flange_to_tcp
                        sampler = SimpleNamespace(
                            flange_to_tcp={f"{a}_arm": transforms[a] for a in ("left", "right")},
                            right_base_to_left_base=next(a.base_to_left for a in self.scene.arms if a.name == "right"))
                        self.robot = RobotEllipsoids(sampler)
                    q = np.asarray([*qpos["left"], *qpos["right"]], float)
                    g = np.asarray([grip["left"], grip["right"]], float)
                    if q.shape != (12,) or not np.isfinite(q).all() or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
                        raise ValueError("Invalid joint or gripper state for ellipsoid overlay")
                    result["robot"] = [encode_ellipsoid(b) for b in self.robot.bodies(q, g)]
        return result

    def attach(self, snapshot, **options):
        # A display failure must not break the existing viewer or safety data.
        try:
            overlay = self.build(snapshot, **options)
        except Exception as exc:
            overlay = {"robot": [], "obstacles": [], "error": f"{type(exc).__name__}: {exc}"}
        return {**snapshot, "ellipsoid_view": overlay}
