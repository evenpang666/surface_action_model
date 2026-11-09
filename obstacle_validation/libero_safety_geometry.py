"""Panda adapters for the same paired-RGB/Voxblox/ellipsoid MCBF used on hardware.

Only robot kinematics and self masks use simulator geometry. Environmental
obstacles are reconstructed from depth and selected by predicted image OBBs.
OSC prediction uses a damped Jacobian approximation, not a dynamics guarantee.
"""

from itertools import product
from types import SimpleNamespace
import copy
import hashlib
import time

import numpy as np

from obstacle_validation.ellipsoid_geometry import Ellipsoid, fit_enclosing
from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame


CAMERAS = dict(front="agentview", side="sideview", right_wrist="robot0_eye_in_hand", left_wrist="frontview")
STATE_SEMANTICS = "libero_eef_axis_angle_gripper"
IMAGE_CONVENTION = "opencv_top_left"


def libero_state(obs):
    from robosuite.utils.transform_utils import quat2axisangle
    # Match the dataset exporter, including quaternions whose angle exceeds pi.
    state = np.r_[obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                  obs["robot0_gripper_qpos"]].astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected LIBERO EEF xyz/axis-angle and two finger positions (8-D)")
    return state


def camera_calibration(sim, name, width, height):
    from robosuite.utils import camera_utils
    return CameraCalibration(name, camera_utils.get_camera_intrinsic_matrix(sim, name, height, width),
                             camera_utils.get_camera_extrinsic_matrix(sim, name),
                             distortion_model="none")


def capture(env, obs, camera_names, frame_id):
    """One physics state, no stepping between images; calibrated top-left images."""
    from robosuite.utils import camera_utils
    frames, calibrations = {}, {}
    stamp = time.monotonic_ns()
    for slot, name in camera_names.items():
        # Observables retain the benchmark's configured image perturbations.
        rgb = np.ascontiguousarray(obs[f"{name}_image"][::-1], dtype=np.uint8)
        h, w = rgb.shape[:2]
        depth = camera_utils.get_real_depth_map(env.sim, np.asarray(obs[f"{name}_depth"])[::-1].squeeze())
        frames[slot] = RGBDFrame(slot, rgb, np.asarray(depth, np.float32), host_timestamp_ns=stamp,
                                frame_number=frame_id)
        if slot in ("front", "side"):
            c = camera_calibration(env.sim, name, w, h)
            calibrations[slot] = CameraCalibration(slot, c.intrinsics, c.camera_to_world,
                                                  distortion_model="none")
    calibration_id = hashlib.sha256(b"".join(np.r_[c.intrinsics.ravel(), c.camera_to_world.ravel()].tobytes()
                                             for c in calibrations.values())).hexdigest()
    state = libero_state(obs)
    context = dict(frame_id=frame_id, capture_ns=stamp, feedback_ns=stamp, calibration_id=calibration_id,
                   views={slot: dict(timestamp_ns=stamp, frame_number=frame_id,
                                     height=f.rgb.shape[0], width=f.rgb.shape[1]) for slot, f in frames.items()})
    bundle = SimpleNamespace(frames=frames, frame_id=frame_id, capture_ns=stamp,
                             calibration_id=calibration_id, feedback=SimpleNamespace(q=state), context=lambda: context)
    return bundle, calibrations


class PandaGeometry:
    """Collision mesh envelopes and finite FK derivatives in normalized OSC inputs."""

    def __init__(self, env):
        self.env, self.sim = env, env.sim
        robot = env.robots[0]
        self.robot = robot
        self.qidx = np.asarray(robot._ref_joint_pos_indexes)
        self.vidx = np.asarray(robot._ref_joint_vel_indexes)
        self.gidx = np.asarray(robot._ref_gripper_joint_pos_indexes)
        self.local = []
        model = self.sim.model
        names = list(robot.robot_model.contact_geoms) + list(robot.gripper.contact_geoms)
        for name in dict.fromkeys(names):
            gid = model.geom_name2id(name)
            kind, size = int(model.geom_type[gid]), np.asarray(model.geom_size[gid])
            if kind == 7:  # compiled mesh vertices are already scaled / recentered
                mid = int(model.geom_dataid[gid])
                start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                points = np.asarray(model.mesh_vert[start:start+count])
                local = fit_enclosing(points, key=name)
            elif kind in (2, 4):  # sphere / ellipsoid
                axes = np.full(3, size[0]) if kind == 2 else size[:3]
                local = Ellipsoid(np.zeros(3), np.diag(axes**2), name)
            else:
                half = size[:3].copy()
                if kind in (3, 5):  # capsule / cylinder
                    half = np.array([size[0], size[0], size[1] + (size[0] if kind == 3 else 0)])
                if kind not in (3, 5, 6):
                    raise ValueError(f"Unsupported Panda collision geometry type {kind}: {name}")
                local = fit_enclosing(np.array(list(product((-1, 1), repeat=3))) * half, key=name)
            self.local.append((gid, local))
        if not self.local:
            raise ValueError("Panda collision geometry is empty")

    def bind(self):
        controller = self.robot.controller
        if (getattr(controller, "name", None) != "OSC_POSE" or not controller.use_delta
                or controller.impedance_mode != "fixed"):
            raise ValueError("Only fixed-impedance, delta OSC_POSE is supported")
        controller.update(force=True)
        jacobian = np.vstack((controller.J_pos, controller.J_ori))
        zero = controller.scale_action(np.zeros(6))
        if not np.allclose(zero, 0):
            raise ValueError("OSC scaling must be centered on zero")
        scale = controller.scale_action(np.ones(6))
        self.mapping = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-4, np.diag(scale))
        self.base = self.sim.data.qpos.copy()
        self.grip = np.array([np.clip(1-np.mean(np.abs(self.base[self.gidx]))/.04, 0, 1)])

    def gripper_target(self, command):
        # Panda uses a signed incremental gripper command, NOT an absolute target.
        current = np.asarray(self.robot.gripper.current_action).copy()
        target = np.clip(current + np.array([-1., 1.])*self.robot.gripper.speed*np.sign(command), -1, 1)
        return float((1-target[0])/2)

    def predicted_q(self, action):
        return self.base[self.qidx] + self.mapping @ np.asarray(action)

    def bodies(self, controls, grip):
        saved = self.sim.get_state()
        try:
            self.sim.data.qpos[self.qidx] = self.predicted_q(controls)
            # Mirrored Panda joints: +.04/-.04=open, zero=closed.
            self.sim.data.qpos[self.gidx] = np.array([.04, -.04])*(1-float(grip[0]))
            self.sim.forward()
            result = []
            for gid, local in self.local:
                pose = np.eye(4)
                pose[:3, :3] = self.sim.data.geom_xmat[gid].reshape(3, 3)
                pose[:3, 3] = self.sim.data.geom_xpos[gid]
                result.append(local.transformed(pose, arm=0))
            return result
        finally:
            self.sim.set_state(saved)
            self.sim.forward()

    def differential(self, controls, grip):
        bodies = self.bodies(controls, grip)
        dc, ds = np.empty((len(bodies), 3, 6)), np.empty((len(bodies), 3, 3, 6))
        for j in range(6):
            d = np.eye(6)[j]*1e-4
            plus, minus = self.bodies(controls+d, grip), self.bodies(controls-d, grip)
            dc[..., j] = np.array([a.center-b.center for a, b in zip(plus, minus)])/2e-4
            ds[..., j] = np.array([a.shape-b.shape for a, b in zip(plus, minus)])/2e-4
        return bodies, dc, ds

    @staticmethod
    def inter_arm_collision_pair(a, b):
        return False

    def robot_mask(self, name, width, height):
        segmentation = np.asarray(self.sim.render(camera_name=name, width=width, height=height,
                                                  segmentation=True))[::-1]
        # MuJoCo object type 5 = geom. Geometry is used ONLY for the robot mask.
        all_robot = {i for i in range(self.sim.model.ngeom)
                     if (self.sim.model.body_id2name(int(self.sim.model.geom_bodyid[i])) or "").startswith(
                         ("robot0_", "gripper0_"))}
        return (segmentation[..., 0] == 5) & np.isin(segmentation[..., 1], list(all_robot))


class SimWorkspaceTSDF:
    def __init__(self, calibrations, bounds, resolution, max_cells):
        from obstacle_validation.oa_vla_perception import WorkspaceScope
        from real_scripts.voxblox_obstacle_map import VoxbloxOccupancyGrid
        from real_scripts.replay_voxblox import SelectedTSDFPipeline
        bounds = np.asarray(bounds).reshape(3, 2)
        origin = bounds[:, 0]
        shape = np.ceil((bounds[:, 1]-origin)/resolution).astype(int)
        if int(np.prod(shape)) > max_cells:
            raise ValueError("Simulation workspace exceeds --max-grid-cells")
        grid = VoxbloxOccupancyGrid(origin, np.eye(3), shape, resolution, esdf=False)
        self.pipeline = SelectedTSDFPipeline(WorkspaceScope(), calibrations, grid=grid)
        self.signature = None

    def update(self, bundle, calibrations, robot_geometry, camera_names):
        from obstacle_validation.oa_vla_perception import MappedObservation
        # Extrinsics can be randomized on reset. Never reuse calibrations/ROI masks.
        self.pipeline.calibrations = calibrations
        self.pipeline.roi_masks.clear()
        self.pipeline.support_masks.clear()
        frames = {key: bundle.frames[key] for key in ("front", "side")}
        rendered, keep = {}, {}
        for key, frame in frames.items():
            h, w = frame.rgb.shape[:2]
            mask = robot_geometry.robot_mask(camera_names[key], w, h)
            rendered[key] = np.where(mask, frame.depth_m, 0)
            keep[key] = ~mask
        snapshot = self.pipeline.update(frames, rendered_views=rendered, keep_views=keep,
                                        qpos={"panda": robot_geometry.base[robot_geometry.qidx].tolist()},
                                        grippers={"panda": float(robot_geometry.grip[0])},
                                        calibration={"sha256": bundle.calibration_id})
        # Shared occupancy schema uses the legacy label left_base. In this adapter
        # every coordinate in that schema is MuJoCo world; no UR transform is used.
        snapshot["vla_frame_id"] = bundle.frame_id
        return MappedObservation(bundle, copy.deepcopy(snapshot), calibrations)
