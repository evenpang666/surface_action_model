"""Seven-joint Panda adapter and explicit simulator-label perception protocol.

The shared-label protocol uses RGB-D + visible instance segmentation for every
method. It does not use obstacle poses/meshes for safety filtering. Collision
ground truth is used solely by the scorer and grasp contact sensor.
"""
from contextlib import contextmanager
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .ellipsoid_geometry import fit_enclosing
from .libero_safety_geometry import PandaGeometry
from .oa_vla_actions import ObservationHold
from .safety_filters import box_surface
from real_scripts.real_cbf_qp import OrientedBox


def label_geoms(env, names):
    model = env.sim.model
    result = {}
    for name in names:
        ids = [i for i in range(model.ngeom)
               if (model.geom_id2name(i) or "").startswith(name+"_")]
        if not ids:
            raise ValueError(f"No simulator geometry for instance {name!r}")
        result[name] = set(ids)
    return result


class LabelDepthPerception:
    """Visible label masks select depth, identical across all three methods.

    Bounds cover observed surfaces only. Padding is a configured uncertainty
    allowance, not a proof about occluded geometry. Persistent boxes are replaced
    on observation and expire; never fall back to simulator obstacle geometry.
    """
    def __init__(self, env, cameras, names, *, padding=.015, max_age=.5):
        self.env, self.cameras = env, cameras
        self.geoms = label_geoms(env, names)
        self.padding, self.max_age = padding, max_age
        self.cache = {}

    def update(self, bundle, calibrations, *, retained_names=()):
        clouds = {name: [] for name in self.geoms}
        for slot in ("front", "side"):
            frame = bundle.frames[slot]
            h, w = frame.depth_m.shape
            from robosuite.utils.camera_utils import get_real_depth_map
            # Render mask and depth TOGETHER. MuJoCo's segmentation mode can
            # differ in occlusion from the RGB pass (e.g. translucent geoms).
            # Combining a label from that pass with RGB-pass depth assigns
            # another surface's depth to the labelled object.
            seg, raw_depth = self.env.sim.render(camera_name=self.cameras[slot], width=w,
                                                height=h, segmentation=True, depth=True)
            seg = np.asarray(seg)[::-1]
            depth_map = get_real_depth_map(self.env.sim, np.asarray(raw_depth)[::-1])
            calibration = calibrations[slot]
            for name, ids in self.geoms.items():
                y, x = np.where((seg[..., 0] == 5) & np.isin(seg[..., 1], list(ids))
                               & np.isfinite(depth_map) & (depth_map > 0))
                if len(x) < 3:
                    continue
                depth = depth_map[y, x]
                camera = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(calibration.intrinsics).T*depth[:, None]
                pose = calibration.camera_to_world
                clouds[name].append(camera @ pose[:3, :3].T+pose[:3, 3])
        now = float(self.env.sim.data.time)
        for name, parts in clouds.items():
            if parts:
                points = np.concatenate(parts)
                low, high = points.min(0)-self.padding, points.max(0)+self.padding
                self.cache[name] = now, OrientedBox((low+high)/2, np.eye(3), (high-low)/2, object_id=name)
        missing = [name for name in self.geoms if name not in retained_names
                   and (name not in self.cache or now-self.cache[name][0] > self.max_age)]
        if missing:
            raise ObservationHold(f"No recent visible RGB-D bounds for {missing}")
        return {name: value[1] for name, value in self.cache.items()}


class JointPandaGeometry(PandaGeometry):
    """Full seven-DOF increments executed by JOINT_VELOCITY, same for all methods."""
    def __init__(self, env, *, spacing=.04):
        super().__init__(env)
        self.spacing = spacing
        self.gripper_gids = {env.sim.model.geom_name2id(n) for n in self.robot.gripper.contact_geoms}
        self.samples = []
        for gid, local in self.local:
            if gid in self.gripper_gids:
                continue  # use a fixed open/closed sweep envelope below
            # Box encloses the collision envelope, with a rigorous face-grid
            # covering bound. It is conservative relative to curved geometry.
            half = np.sqrt(np.diag(local.shape))
            self.samples.append((gid, box_surface(half, spacing)+local.center))
        self.held = {}
        self.eef_envelope = None

    @property
    def held_sample_count(self):
        return sum(len(points) for points in self.held.values())

    def bind(self):
        controller = self.robot.controller
        if controller.name != "JOINT_VELOCITY":
            raise ValueError("Comparison protocol requires JOINT_VELOCITY for every method")
        controller.update(force=True)
        self.base = self.sim.data.qpos.copy()
        self.grip = np.array([np.clip(1-np.mean(np.abs(self.base[self.gidx]))/.04, 0, 1)])
        self.jacobian = np.vstack((controller.J_pos, controller.J_ori))
        self.target_position, self.target_rotation = self.eef_pose()
        if self.eef_envelope is None:
            points = []
            p, r = self.eef_pose()
            for gid, local in self.local:
                if gid not in self.gripper_gids:
                    continue
                half = np.sqrt(np.diag(local.shape))
                # Include the fully open/closed finger sweep in one rigid TCP
                # envelope, independent of the current gripper command.
                corners = np.array(list(product((-1, 1), repeat=3)))*half+local.center
                world = corners @ self.sim.data.geom_xmat[gid].reshape(3, 3).T+self.sim.data.geom_xpos[gid]
                points.append((world-p) @ r)
            if not points:
                raise ValueError("No gripper collision geometry")
            points = np.concatenate(points)
            points = np.concatenate([points+np.array([0, d, 0]) for d in (-.04, .04)])
            self.eef_envelope = fit_enclosing(points, key="eef")
            self.eef_samples = box_surface(np.sqrt(np.diag(self.eef_envelope.shape)), self.spacing)+self.eef_envelope.center

    def eef_pose(self):
        sid = self.robot.eef_site_id
        return self.sim.data.site_xpos[sid].copy(), self.sim.data.site_xmat[sid].reshape(3, 3).copy()

    def set_reference(self, action, elapsed):
        # LeRobot/OA-VLA LIBERO action units: normalized .05m/.5rad over 50ms.
        # Tracking is re-evaluated at each low-level tick towards this waypoint.
        p, r = self.eef_pose()
        self.target_position = p+np.asarray(action[:3])*.05*elapsed/.05
        self.target_rotation = Rotation.from_rotvec(np.asarray(action[3:6])*.5*elapsed/.05).as_matrix() @ r

    def tracking_error(self):
        p, r = self.eef_pose()
        error = np.r_[self.target_position-p, Rotation.from_matrix(self.target_rotation @ r.T).as_rotvec()]
        return error, self.jacobian

    def nominal(self, dt, gain=8.):
        error, jacobian = self.tracking_error()
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+np.eye(6)*1e-4, gain*error*dt)

    def predicted_q(self, controls):
        return self.base[self.qidx]+np.asarray(controls)

    @contextmanager
    def at(self, controls):
        state = self.sim.get_state()
        try:
            self.sim.data.qpos[self.qidx] = self.predicted_q(controls)
            self.sim.forward()
            yield
        finally:
            self.sim.set_state(state)
            self.sim.forward()

    def bodies(self, controls, grip):
        with self.at(controls):
            result = []
            for gid, local in self.local:
                pose = np.eye(4)
                pose[:3, :3] = self.sim.data.geom_xmat[gid].reshape(3, 3)
                pose[:3, 3] = self.sim.data.geom_xpos[gid]
                result.append(local.transformed(pose, arm=0))
            return result

    def eef_body(self, controls):
        with self.at(controls):
            p, r = self.eef_pose()
            pose = np.eye(4); pose[:3, :3], pose[:3, 3] = r, p
            return self.eef_envelope.transformed(pose)

    def differential(self, controls, grip):
        bodies = self.bodies(controls, grip)
        dc, ds = np.empty((len(bodies), 3, 7)), np.empty((len(bodies), 3, 3, 7))
        for j in range(7):
            d = np.eye(7)[j]*1e-5
            plus, minus = self.bodies(controls+d, grip), self.bodies(controls-d, grip)
            dc[..., j] = np.array([a.center-b.center for a, b in zip(plus, minus)])/2e-5
            ds[..., j] = np.array([a.shape-b.shape for a, b in zip(plus, minus)])/2e-5
        return bodies, dc, ds

    def eef_differential(self):
        dc, ds = np.empty((3, 7)), np.empty((3, 3, 7))
        for j in range(7):
            delta = np.eye(7)[j]*1e-5
            plus, minus = self.eef_body(delta), self.eef_body(-delta)
            dc[:, j], ds[..., j] = (plus.center-minus.center)/2e-5, (plus.shape-minus.shape)/2e-5
        return self.eef_body(np.zeros(7)), dc, ds

    def sample_points(self, controls):
        with self.at(controls):
            points = [local @ self.sim.data.geom_xmat[gid].reshape(3, 3).T+self.sim.data.geom_xpos[gid]
                      for gid, local in self.samples]
            p, r = self.eef_pose()
            # The gripper is exogenous to the QP. Protect its full sweep rather
            # than only its currently observed opening.
            points.append(self.eef_samples @ r.T+p)
            points.extend(local @ r.T+p for local in self.held.values())
            return np.concatenate(points)

    def sample_differential(self):
        points = self.sample_points(np.zeros(7))
        jacobian = np.empty((len(points), 3, 7))
        for j in range(7):
            delta = np.eye(7)[j]*1e-5
            jacobian[..., j] = (self.sample_points(delta)-self.sample_points(-delta))/2e-5
        return points, jacobian

    def update_grasp(self, command, object_boxes, object_geoms, *, padding=.02):
        if command < 0:
            self.held.clear()
            return
        # Two-sided contact is a simulator grasp-sensor proxy; no pose oracle.
        important = self.robot.gripper.important_geoms
        groups = [important.get(key, []) for key in ("left_fingerpad", "right_fingerpad")]
        if not all(groups):
            groups = [important.get(key, []) for key in ("left_finger", "right_finger")]
        group_ids = [{self.sim.model.geom_name2id(n) for n in group} for group in groups]
        if not all(group_ids):
            raise ValueError("Two-sided gripper contact groups required")
        contacts = [set((int(c.geom1), int(c.geom2))) for c in self.sim.data.contact[:self.sim.data.ncon]]
        for name, ids in object_geoms.items():
            if name in self.held or name not in object_boxes:
                continue
            if all(any(pair & finger and pair & ids for pair in contacts) for finger in group_ids):
                box = object_boxes[name]
                local = box_surface(box.half_sizes+padding, self.spacing) @ box.axes.T+box.center
                p, r = self.eef_pose()
                self.held[name] = (local-p) @ r

    def control_bounds(self, dt, max_velocity):
        model = self.sim.model
        jids = [model.joint_name2id(name) for name in self.robot.robot_model.joints]
        ranges = model.jnt_range[jids]
        q = self.base[self.qidx]
        return np.maximum(-max_velocity*dt, ranges[:, 0]-q), np.minimum(max_velocity*dt, ranges[:, 1]-q)

    def velocity_action(self, delta, dt, gripper):
        controller = self.robot.controller
        zero, scale = controller.scale_action(np.zeros(7)), controller.scale_action(np.ones(7))
        if not np.allclose(zero, 0) or np.any(scale <= 0):
            raise ValueError("Comparison requires centered positive velocity scaling")
        action = np.asarray(delta)/dt/scale
        if np.any(np.abs(action) > 1+1e-6):
            raise ObservationHold("Joint velocity exceeds normalized controller range")
        return np.r_[np.clip(action, -1, 1), gripper]
