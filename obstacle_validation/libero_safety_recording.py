"""Explicit simulator-only training export; segmentation is never a policy input."""
import json
from pathlib import Path

import numpy as np

from obstacle_validation.libero_safety_geometry import STATE_SEMANTICS, IMAGE_CONVENTION

ACTION_LAYOUT = ("osc_dx", "osc_dy", "osc_dz", "osc_drx", "osc_dry", "osc_drz", "gripper_signed_command")


class TrainingRecorder:
    def __init__(self, env, camera_names, obstacle_names, horizon):
        self.env, self.cameras, self.names, self.horizon = env, camera_names, list(obstacle_names), horizon
        self.rows = []
        if not self.names or len(self.names) != len(set(self.names)) or len(self.names) > 64:
            raise ValueError("Training labels require 1..64 unique obstacle instance names")
        self.geoms = {}
        for name in self.names:
            ids = [i for i in range(env.sim.model.ngeom)
                   if (env.sim.model.geom_id2name(i) or "").startswith(name+"_")]
            if not ids:
                raise ValueError(f"No simulator geometry for label obstacle {name!r}")
            self.geoms[name] = ids

    def add(self, bundle):
        import cv2
        from oa_vla.geometry import pixel_obb_to_normalized
        boxes = np.zeros((2, len(self.names), 6), np.float32)
        valid = np.zeros((2, len(self.names)), bool)
        for vi, view in enumerate(("front", "side")):
            h, w = bundle.frames[view].rgb.shape[:2]
            seg = np.asarray(self.env.sim.render(camera_name=self.cameras[view], width=w, height=h,
                                                 segmentation=True))[::-1]
            for oi, name in enumerate(self.names):
                y, x = np.where((seg[..., 0] == 5) & np.isin(seg[..., 1], self.geoms[name]))
                if len(x) < 3:
                    continue
                (cx, cy), (bw, bh), angle = cv2.minAreaRect(np.c_[x, y].astype(np.float32))
                boxes[vi, oi] = pixel_obb_to_normalized((cx, cy, max(bw, 1), max(bh, 1), angle), h, w)
                valid[vi, oi] = True
        self.rows.append(dict(rgb={k: f.rgb.copy() for k, f in bundle.frames.items()},
                              state=bundle.feedback.q.copy(), boxes=boxes, valid=valid, action=None))

    def set_action(self, action):
        self.rows[-1]["action"] = np.asarray(action, np.float32).copy()

    def save(self, path, task_text):
        if not self.rows or any(row["action"] is None for row in self.rows):
            raise ValueError("Incomplete training episode")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        count = len(self.rows)
        actions = np.array([row["action"] for row in self.rows])
        indices = np.arange(count)[:, None]+np.arange(self.horizon)[None]
        mask = indices < count
        chunks = np.where(mask[..., None], actions[np.minimum(indices, count-1)], 0)
        profile = dict(action_semantics="libero_osc_pose", state_semantics=STATE_SEMANTICS,
                       image_convention=IMAGE_CONVENTION, camera_names=self.cameras)
        arrays = dict(task_text=np.asarray(task_text), qpos=np.array([r["state"] for r in self.rows]),
                      action_chunks=chunks, sample_frame_indices=np.arange(count), action_step_mask=mask,
                      action_dt_s=np.full(mask.shape, .05), action_layout=np.array(ACTION_LAYOUT),
                      action_source=np.asarray("executed"), runtime_profile_json=np.asarray(json.dumps(profile)),
                      detection_boxes=np.array([r["boxes"] for r in self.rows]),
                      detection_valid=np.array([r["valid"] for r in self.rows]),
                      detection_reviewed=np.ones((count, 2), bool),
                      detection_object_ids=np.broadcast_to(np.array(self.names, dtype="U64"), (count, 2, len(self.names))),
                      detection_box_format=np.asarray("cxcywh_cos2sin2_normalized_padded"),
                      detection_views=np.array(["front", "side"]))
        for view in self.cameras:
            arrays[f"rgb_{view}"] = np.array([r["rgb"][view] for r in self.rows])
        for name, array in arrays.items():
            np.save(path/f"{name}.npy", array, allow_pickle=False)
        (path/"provenance.json").write_text(json.dumps(dict(label_source="simulator_visible_segmentation",
            obstacle_names=self.names, policy_input="RGB only; no obstacle segmentation/poses",
            actions="commands actually passed to env.step; not measured joint deltas"), indent=2)+"\n")

