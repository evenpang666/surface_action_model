"""Convert Quest3 episodes to state/action LeRobot data and replay like a VLA queue.

The action provider is the recorded dataset, not a learned policy.  Joint
actions are absolute 12-joint targets; the two gripper values are closing
targets.  Physical output is disabled unless the explicit execute flag and
acknowledgement are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import time

import h5py
import numpy as np
import pyarrow.parquet as parquet
import yaml

from obstacle_validation.cbf import RecordedPathCBF
from obstacle_validation.core import (CollisionChecker, PathFollower, smooth_joint_path,
                                      segments_within_box_margin, vector)
from obstacle_validation.quest3_schema import NAMES, _joint_order
from obstacle_validation.runtime import (EXECUTE_ACK, DualCommander,
                                         boxes_from_json, read_snapshot, snapshot_state)
from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.real_cbf_qp import validate_fk_rollout
from safety_module.dual_camera_calibration import validate_calibration_identity
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path


SCHEMA = "quest3_dual_ur7e_lerobot_absolute_action_v1"
STATE_NAMES = [*NAMES, "left_gripper", "right_gripper"]
ACTION_NAMES = [*NAMES, "left_gripper_target", "right_gripper_target"]


def _episode_arrays(filename: Path, start_frame: int, end_frame: int | None):
    with h5py.File(filename, "r") as source:
        if source.attrs.get("complete") != True or source.attrs.get("mode") != "dual":
            raise ValueError("Expected a complete dual-arm Quest3 HDF5 episode")
        names, q_order = _joint_order(source, "qpos_joint_names")
        _, action_order = _joint_order(source, "action_joint_names")
        q = np.asarray(source["observations/qpos"], dtype=float)
        action = np.asarray(source["action"], dtype=float)
        executed = np.asarray(source["executed_action"], dtype=float)
        grip = np.asarray(source["observations/gripper_position"], dtype=float)
        grip_action = np.asarray(source["gripper_action"], dtype=float)
        valid = np.asarray(source["action_valid"], dtype=bool)
        stamps = np.asarray(source["timestamps_ns"])
        n = len(q)
        if (q.shape != (n, 12) or n < 2 or action.shape != q.shape or executed.shape != q.shape
                or grip.shape != (n, 2) or grip_action.shape != grip.shape
                or valid.shape != (n,) or stamps.shape != (n,) or stamps.dtype.kind not in "iu"):
            raise ValueError("Misaligned Quest3 state/action/gripper/timestamp arrays")
        if any(not np.isfinite(x).all() for x in (q, action, executed, grip, grip_action)):
            raise ValueError("Non-finite Quest3 state or action")
        if np.any((grip < 0) | (grip > 1)) or np.any((grip_action < 0) | (grip_action > 1)):
            raise ValueError("Gripper position and action must use 0=open, 1=closed")
        if np.any(stamps[1:] <= stamps[:-1]):
            raise ValueError("Quest3 timestamps must strictly increase")
        if source.attrs.get("action_definition") != "commanded target_q - actual_q (radians)":
            raise ValueError("Unknown Quest3 joint action definition")
        if source.attrs.get("executed_action_definition") != "qpos[t+1] - qpos[t] (radians)":
            raise ValueError("Unknown Quest3 executed action definition")
        q, action, executed = q[:, q_order], action[:, action_order], executed[:, q_order]
        if not np.allclose(executed[:-1], np.diff(q, axis=0), atol=2e-6, rtol=0):
            raise ValueError("Executed joint increments disagree with measured qpos")
        arm_order = list(dict.fromkeys(name.split("/")[0] for name in names))
        if any(name.split("/")[0] != arm_order[i//6] for i, name in enumerate(names)):
            raise ValueError("Recorded arm blocks must be contiguous")
        gripper_order = [arm_order.index("left_arm"), arm_order.index("right_arm")]
        grip, grip_action = grip[:, gripper_order], grip_action[:, gripper_order]
        end_frame = n-1 if end_frame is None else int(end_frame)
        if not 0 <= start_frame < end_frame < n or not valid[start_frame:end_frame].all():
            raise ValueError("Selected episode range must contain valid actions and two states")
        task = str(source.attrs.get("task_description", "Quest3 dual-arm action replay")).strip()
        return (q[start_frame:end_frame+1], action[start_frame:end_frame+1],
                grip[start_frame:end_frame+1], grip_action[start_frame:end_frame+1],
                stamps[start_frame:end_frame+1], task or "Quest3 dual-arm action replay", end_frame)


def convert_hdf5(filename: Path, output: Path, *, fps: int = 20, time_scale: float = 1.,
                 start_frame: int = 0, end_frame: int | None = None) -> dict:
    """Write a real LeRobot episode with 14-D state and absolute 14-D action."""
    if fps < 1 or not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("fps and time-scale must be positive")
    filename, output = Path(filename).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset output already exists: {output}")
    q, commanded, grip, grip_action, stamps, task, end_frame = _episode_arrays(filename, start_frame, end_frame)
    elapsed = (stamps-stamps[0]).astype(np.float64)/1e9*time_scale
    duration = float(elapsed[-1])
    period = 1./fps
    count = max(1, math.ceil(duration/period))
    sample_at = np.arange(count, dtype=float)*period
    target_at = np.minimum(sample_at+period, duration)
    # The HDF5 action is relative to its own contemporaneous observation.
    # VLA-style absolute target = recorded qpos + recorded action.
    absolute_targets = q + commanded
    absolute_targets[-1] = q[-1]  # terminal action is invalid
    state_q = np.column_stack([np.interp(sample_at, elapsed, q[:, j]) for j in range(12)])
    state_g = np.column_stack([np.interp(sample_at, elapsed, grip[:, j]) for j in range(2)])
    target_q = np.column_stack([np.interp(target_at, elapsed, absolute_targets[:, j]) for j in range(12)])
    frame_ids = np.minimum(np.searchsorted(elapsed, sample_at, side="right")-1, len(q)-2)
    target_g = grip_action[frame_ids]
    states = np.column_stack((state_q, state_g)).astype(np.float32)
    actions = np.column_stack((target_q, target_g)).astype(np.float32)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Resampled LeRobot data is non-finite")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": ACTION_NAMES},
        "source_elapsed_s": {"dtype": "float64", "shape": (1,), "names": ["source_elapsed_s"]},
        "source_frame_index": {"dtype": "int64", "shape": (1,), "names": ["source_frame_index"]},
    }
    dataset = LeRobotDataset.create(repo_id="local/quest3_dual_ur7e_replay", root=output,
                                    robot_type="dual_ur7e_robotiq_2f85", fps=fps,
                                    features=features, use_videos=False)
    for index in range(count):
        dataset.add_frame({"observation.state": states[index], "action": actions[index],
                           "source_elapsed_s": np.asarray([sample_at[index]], dtype=np.float64),
                           "source_frame_index": np.asarray([start_frame+frame_ids[index]], dtype=np.int64),
                           "task": task})
    dataset.save_episode()
    metadata = {"schema": SCHEMA, "source_hdf5": str(filename), "start_frame": start_frame,
                "end_frame_inclusive": end_frame, "recorded_frames": len(q), "recorded_duration_s": duration/time_scale,
                "resampled_duration_s": count*period, "fps": fps, "time_scale": time_scale,
                "action_layout": ACTION_NAMES, "action_semantics": "absolute_joint_target_plus_gripper_closing_target",
                "initial_state": np.r_[q[0], grip[0]].tolist(),
                "final_joint_state": q[-1].tolist(), "task": task,
                "max_target_step_rad": float(np.max(np.abs(np.diff(np.vstack((q[0], target_q)), axis=0))))}
    (output/"replay_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


class DatasetActionPolicy:
    """Dataset-backed VLA action provider: infer returns the next action chunk."""

    def __init__(self, root: Path):
        root = Path(root).resolve()
        self.metadata = json.loads((root/"replay_metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("Not a Quest3 dual-arm LeRobot action replay dataset")
        data = parquet.read_table(root/"data/chunk-000/episode_000000.parquet",
                                  columns=["observation.state", "action", "source_elapsed_s", "source_frame_index"])
        self.states = np.asarray(data["observation.state"].to_pylist(), dtype=float)
        self.actions = np.asarray(data["action"].to_pylist(), dtype=float)
        self.source_elapsed = np.asarray(data["source_elapsed_s"].to_pylist(), dtype=float).reshape(-1)
        self.source_frames = np.asarray(data["source_frame_index"].to_pylist(), dtype=int).reshape(-1)
        if (self.states.shape != self.actions.shape or self.actions.ndim != 2 or self.actions.shape[1] != 14
                or len(self.actions) < 1 or not np.isfinite(self.states).all() or not np.isfinite(self.actions).all()
                or np.any((self.actions[:, 12:] < 0) | (self.actions[:, 12:] > 1))):
            raise ValueError("Malformed LeRobot state/action episode")
        self.period = 1./int(self.metadata["fps"])

    def infer(self, start: int, horizon: int = 32) -> np.ndarray:
        if not 0 <= start < len(self.actions) or horizon < 1:
            raise ValueError("Invalid action chunk request")
        return self.actions[start:min(len(self.actions), start+horizon)].copy()


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0., 1.))
    return alpha*alpha*(3.-2.*alpha)


class RobotiqTargetWriter:
    """Command an already active 2F gripper without resetting its activation."""

    def __init__(self, host: str, *, speed: int, force: int):
        self.socket = socket.create_connection((host, 63352), timeout=1.)
        self.socket.settimeout(1.)
        self.buffer = b""
        self.speed, self.force = speed, force
        self.last_target = None
        try:
            if self._query("GET STA") != "STA 3":
                raise RuntimeError(f"Robotiq gripper at {host} is not activated")
        except BaseException:
            self.close()
            raise

    def _readline(self) -> str:
        while b"\n" not in self.buffer:
            chunk = self.socket.recv(128)
            if not chunk:
                raise ConnectionError("Robotiq closed command connection")
            self.buffer += chunk
            if len(self.buffer) > 256:
                raise RuntimeError("Robotiq response exceeds 256 bytes")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()

    def _query(self, line: str) -> str:
        self.socket.sendall((line+"\n").encode("ascii"))
        return self._readline()

    def send_if_changed(self, closing_fraction: float) -> bool:
        target = float(closing_fraction)
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Gripper target must be in [0,1]")
        if self.last_target is not None and abs(target-self.last_target) < .01:
            return False
        # The installed URCap does not reliably reply to every SET line.
        # Match the working Quest3 collector: write one complete command and
        # verify actual motion through the separate receive-only GET POS reader.
        payload = (f"SET POS {round(target*255)}\n"
                   f"SET SPE {self.speed}\n"
                   f"SET FOR {self.force}\n"
                   "SET GTO 1\n")
        self.socket.sendall(payload.encode("ascii"))
        self.last_target = target
        return True

    def close(self):
        connection, self.socket = self.socket, None
        if connection is not None:
            connection.close()


class DualGripperCommander:
    """Two explicit, non-resetting Robotiq outputs."""

    def __init__(self, left_ip: str, right_ip: str, *, speed: int = 40, force: int = 25):
        if not 0 <= speed <= 255 or not 0 <= force <= 255:
            raise ValueError("Robotiq speed and force must be in [0,255]")
        self.outputs = []
        try:
            for host in (left_ip, right_ip):
                self.outputs.append(RobotiqTargetWriter(host, speed=speed, force=force))
        except BaseException:
            self.close()
            raise

    def send_if_changed(self, targets: np.ndarray) -> list[bool]:
        values = vector(targets, 2, "gripper targets")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Gripper targets must be in [0,1]")
        return [out.send_if_changed(float(target)) for out, target in zip(self.outputs, values, strict=True)]

    def close(self):
        outputs, self.outputs = self.outputs, []
        for out in outputs:
            out.close()


def _obstacle_point_distances(points, box):
    local = (points-box.center) @ box.axes
    delta = np.abs(local)-box.half_sizes
    return np.linalg.norm(np.maximum(delta, 0.), axis=-1)+np.minimum(delta.max(-1), 0.)


def _recovery_motion_safe(sampler, q, grip, delta, target_grip, boxes, safety):
    """Accept an unsafe start only when future points clear without worsening."""
    margin = safety["collision_margin_m"]
    arm_margin = safety["inter_arm_margin_m"]
    previous = np.asarray(sampler.link_points(q, grip)).reshape(-1, 3)
    left_count = len(sampler.left_link_names)*sampler.points_per_link
    if not np.isfinite(previous).all() or not 0 < left_count < len(previous):
        return False
    initial_box = [_obstacle_point_distances(previous, box) for box in boxes]
    previous_box = [values.copy() for values in initial_box]
    initial_pairs = np.linalg.norm(previous[:left_count, None]-previous[None, left_count:], axis=-1)
    previous_pairs = initial_pairs.copy()
    for step in range(1, 9):
        fraction = step/8.
        points = np.asarray(sampler.link_points(q+delta*fraction,
                            grip+(target_grip-grip)*fraction)).reshape(-1, 3)
        if points.shape != previous.shape or not np.isfinite(points).all():
            return False
        for index, box in enumerate(boxes):
            distances = _obstacle_point_distances(points, box)
            initially_safe = initial_box[index] > margin
            if (np.any(distances[initially_safe] <= margin)
                    or np.any(distances[~initially_safe] < previous_box[index][~initially_safe]-1e-6)
                    or np.any(segments_within_box_margin(previous, points, box, margin)[initially_safe])):
                return False
            previous_box[index] = distances
        first = previous[:left_count, None]-previous[None, left_count:]
        last = points[:left_count, None]-points[None, left_count:]
        change = last-first
        nearest = np.clip(-(first*change).sum(-1)
                          / np.maximum((change*change).sum(-1), 1e-15), 0., 1.)
        segment_distances = np.linalg.norm(first+nearest[..., None]*change, axis=-1)
        pair_distances = np.linalg.norm(last, axis=-1)
        initially_safe = initial_pairs > arm_margin
        if (np.any(segment_distances[initially_safe] <= arm_margin)
                or np.any(pair_distances[~initially_safe] < previous_pairs[~initially_safe]-1e-6)):
            return False
        previous_pairs = pair_distances
        previous = points
    endpoint = CollisionChecker(sampler, target_grip, boxes, margin=margin,
                                inter_arm_margin=arm_margin)
    obstacle_clearance, arm_clearance = endpoint.clearance(q+delta)
    initial_obstacle = min((float(values.min()) for values in initial_box), default=float("inf"))
    initial_arm = float(initial_pairs.min())
    # A single bounded control tick need not recover the entire clearance.
    # Continue only if every deficient clearance makes measurable progress;
    # otherwise a zero-length or tangential action could stall indefinitely.
    obstacle_progress = (obstacle_clearance >= margin-1e-6
                         or (initial_obstacle < margin
                             and obstacle_clearance >= initial_obstacle+1e-5))
    arm_progress = (arm_clearance >= arm_margin-1e-6
                    or (initial_arm < arm_margin and arm_clearance >= initial_arm+1e-5))
    return obstacle_progress and arm_progress


def _motion_safe(sampler, q, grip, delta, target_grip, boxes, safety, allow_unsafe_start=False):
    current = CollisionChecker(sampler, grip, boxes, margin=safety["collision_margin_m"],
                               inter_arm_margin=safety["inter_arm_margin_m"])
    if not current.state_safe(q):
        return (allow_unsafe_start and _recovery_motion_safe(
            sampler, q, grip, delta, target_grip, boxes, safety))
    points = sampler.link_points(q, grip).reshape(-1, 3)
    left_count = len(sampler.left_link_names)*sampler.points_per_link
    conservative_safe = validate_fk_rollout(
        sampler, q, grip, np.asarray([delta]), boxes,
        point_indices=np.arange(len(points)), left_mask=np.arange(len(points)) < left_count,
        gripper_chunk=np.asarray([target_grip]), collision_margin_m=safety["collision_margin_m"],
        inter_arm_margin_m=safety["inter_arm_margin_m"], substeps=4)
    if conservative_safe:
        return True
    # validate_fk_rollout expands every OBB axis by the margin. Near a box
    # corner that is stricter than the Euclidean clearance used by the CBF's
    # CollisionChecker. Recheck rejected motion with the exact rounded margin.
    previous = points
    for step in range(1, 5):
        fraction = step/4.
        current_points = np.asarray(sampler.link_points(q+delta*fraction,
                                   grip+(target_grip-grip)*fraction)).reshape(-1, 3)
        if current_points.shape != previous.shape or not np.isfinite(current_points).all():
            return False
        if any(segments_within_box_margin(previous, current_points, box,
                                          safety["collision_margin_m"]).any() for box in boxes):
            return False
        first = previous[:left_count, None]-previous[None, left_count:]
        last = current_points[:left_count, None]-current_points[None, left_count:]
        change = last-first
        fraction_nearest = np.clip(-(first*change).sum(-1)/np.maximum((change*change).sum(-1), 1e-15), 0, 1)
        if np.any(np.linalg.norm(first+fraction_nearest[..., None]*change, axis=-1)
                  <= safety["inter_arm_margin_m"]):
            return False
        previous = current_points
    return True


def _motion_failure_context(sampler, q, grip, delta, target_grip, boxes, safety, *, action, cbf_info,
                            allow_unsafe_start=False):
    checker = CollisionChecker(sampler, grip, boxes, margin=safety["collision_margin_m"],
                               inter_arm_margin=safety["inter_arm_margin_m"])
    current_obstacle, current_arm = checker.clearance(q)
    endpoint_checker = CollisionChecker(sampler, target_grip, boxes, margin=safety["collision_margin_m"],
                                        inter_arm_margin=safety["inter_arm_margin_m"])
    endpoint_obstacle, endpoint_arm = endpoint_checker.clearance(q+delta)
    joint_safe = _motion_safe(sampler, q, grip, delta, grip, boxes, safety, allow_unsafe_start)
    gripper_safe = _motion_safe(sampler, q, grip, np.zeros(12), target_grip, boxes, safety,
                                allow_unsafe_start)
    if not checker.state_safe(q) and allow_unsafe_start:
        reason = "future_recovery_sweep_blocked"
    elif not checker.state_safe(q):
        reason = "current_state_inside_clearance"
    elif not joint_safe and gripper_safe:
        reason = "joint_sweep_blocked"
    elif joint_safe and not gripper_safe:
        reason = "gripper_sweep_blocked"
    elif not joint_safe and not gripper_safe:
        reason = "joint_and_gripper_sweeps_blocked"
    else:
        reason = "combined_joint_gripper_sweep_blocked"

    def finite(value):
        return float(value) if np.isfinite(value) else None

    return {"reason": reason, "qpos": np.asarray(q).tolist(), "gripper_position": np.asarray(grip).tolist(),
            "joint_delta": np.asarray(delta).tolist(), "checked_gripper_target": np.asarray(target_grip).tolist(),
            "dataset_gripper_target": np.asarray(action[12:]).tolist(), "cbf": cbf_info,
            "current_obstacle_clearance_m": finite(current_obstacle),
            "current_inter_arm_clearance_m": finite(current_arm),
            "endpoint_obstacle_clearance_m": finite(endpoint_obstacle),
            "endpoint_inter_arm_clearance_m": finite(endpoint_arm),
            "collision_margin_m": safety["collision_margin_m"],
            "inter_arm_margin_m": safety["inter_arm_margin_m"],
            "obbs": [{"center": b.center.tolist(), "axes": b.axes.tolist(),
                      "half_sizes": b.half_sizes.tolist()} for b in boxes]}


class LiveSceneReader:
    """Use fresh tracked and candidate OBBs while perception is changing."""

    def __init__(self, url, age):
        self.url, self.age = url, age
        self.unstable_scene_reads = 0

    def read(self, *, max_wait_s=0.):
        snapshot = read_snapshot(self.url, self.age, max_wait_s=max_wait_s,
                                 allow_tracked_obbs=True)
        if str(snapshot.get("status", "")).startswith("error: unstable obstacle point clusters"):
            self.unstable_scene_reads += 1
        return snapshot


def _read_checked(commander, *, live, url, age, calibration_digest, snapshot_reader=None):
    snapshot = (snapshot_reader() if snapshot_reader is not None else read_snapshot(url, age)) if live else None
    if snapshot is not None:
        validate_calibration_identity(snapshot, calibration_digest)
    q, grip = commander.read_state()
    boxes = boxes_from_json(snapshot) if snapshot is not None else []
    return q, grip, boxes, snapshot


def run_replay(args) -> dict:
    config, config_path = load_project_config(args.project_config)
    safety = config["safety"]
    if (not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.acceleration)
            or args.acceleration <= 0 or not np.isfinite(args.gripper_speed_fraction_s)
            or args.gripper_speed_fraction_s <= 0 or args.action_horizon < 1
            or not np.isfinite(args.restore_settle_timeout) or args.restore_settle_timeout <= 0
            or not np.isfinite(args.restore_timeout) or args.restore_timeout <= 0
            or not np.isfinite(args.restore_planning_timeout) or args.restore_planning_timeout <= 0
            or not np.isfinite(args.restore_detour_room) or args.restore_detour_room <= 0
            or (args.mode == "baseline" and
                (not np.isfinite(args.action_timeout) or args.action_timeout <= 0))
            or not np.isfinite(args.arrival_tolerance) or args.arrival_tolerance <= 0
            or not np.isfinite(args.max_tracking_error) or args.max_tracking_error <= 0):
        raise ValueError("Speed, acceleration, tolerances and timeouts must be positive")
    if args.execute and args.initial_state:
        raise ValueError("--initial-state is only available for virtual replay")
    if args.execute and args.execute_ack != EXECUTE_ACK:
        raise ValueError(f"Physical execution requires --execute-ack {EXECUTE_ACK}")
    if args.execute and args.mode == "avoid" and not args.live:
        raise ValueError("Physical CBF replay requires --live obstacle detection")
    if args.live and not safety.get("camera_calibration_valid", False):
        raise RuntimeError("Live replay requires validated dual-view calibration")
    if args.hdf5 is not None:
        dataset_path = args.output/"lerobot_dataset"
        if not dataset_path.exists():
            convert_hdf5(args.hdf5, dataset_path, fps=args.fps, time_scale=args.time_scale,
                         start_frame=args.start_frame, end_frame=args.end_frame)
    else:
        dataset_path = args.dataset
    policy = DatasetActionPolicy(dataset_path)
    if args.hdf5 is not None:
        requested = Path(args.hdf5).resolve()
        source = policy.metadata
        if (source["source_hdf5"] != str(requested) or source["fps"] != args.fps
                or source["time_scale"] != args.time_scale or source["start_frame"] != args.start_frame
                or (args.end_frame is not None and source["end_frame_inclusive"] != args.end_frame)):
            raise ValueError("Existing LeRobot dataset does not match requested HDF5 conversion")
    sampler = DualUR7eSurfacePointSampler(points_per_link=args.points_per_link, project_config=config_path)
    if args.execute and (not all(sampler.tcp_transform_verified.values()) or not sampler.calibration_digest):
        raise RuntimeError("Physical replay requires verified dual TCP and shared calibration")
    if args.live and not sampler.calibration_digest:
        raise RuntimeError("Live replay requires shared dual-camera calibration")
    url = args.obstacle_url or safety["obstacle_url"]
    age = safety["stale_timeout_s"]
    scene_reader = LiveSceneReader(url, age)
    first = scene_reader.read() if args.live else None
    if first is not None:
        validate_calibration_identity(first, sampler.calibration_digest)
    q0, g0 = vector(policy.metadata["initial_state"][:12], 12, "recorded initial joints"), vector(
        policy.metadata["initial_state"][12:], 2, "recorded initial grippers")
    if args.execute:
        hardware = yaml.safe_load(resolve_repo_path(config["quest3"]["hardware_config"]).read_text())
        left_ip = str(hardware["arms"]["left_arm"]["robot_ip"])
        right_ip = str(hardware["arms"]["right_arm"]["robot_ip"])
        commander = DualCommander(left_ip, right_ip, policy.period)
        gripper_commander = None
        try:
            gripper_commander = DualGripperCommander(left_ip, right_ip, speed=args.gripper_speed,
                                                     force=args.gripper_force)
        except BaseException:
            commander.close()
            raise
        try:
            q, grip = commander.read_state()
        except BaseException:
            try: gripper_commander.close()
            finally: commander.close()
            raise
    else:
        commander = gripper_commander = None
        if args.initial_state:
            payload = json.loads(Path(args.initial_state).read_text(encoding="utf-8"))
            q = vector(payload["qpos"], 12, "initial joints")
            grip = vector(payload["gripper_position"], 2, "initial grippers")
        elif first is not None:
            q, grip = snapshot_state(first)
        else:
            q, grip = q0.copy(), g0.copy()
    cbf = (RecordedPathCBF(sampler, grip, collision_margin_m=safety["collision_margin_m"],
                           inter_arm_margin_m=safety["inter_arm_margin_m"],
                           trigger_margin_m=safety["trigger_margin_m"], alpha=safety["cbf_alpha"],
                           iterations=safety["cbf_iterations"]) if args.mode == "avoid" else None)
    boxes = boxes_from_json(first) if first is not None else []
    failure_context = None

    def plan_dataset_step(measured_q, measured_grip, boxes, action, previous_delta):
        nonlocal failure_context
        desired_q, desired_grip = action[:12], action[12:]
        delta = np.clip(desired_q-measured_q, -args.speed*policy.period, args.speed*policy.period)
        info = None
        if cbf is not None:
            scene = CollisionChecker(sampler, measured_grip, boxes, margin=safety["collision_margin_m"],
                                     inter_arm_margin=safety["inter_arm_margin_m"])
            cbf.gripper = measured_grip.copy()
            nominal_delta = delta.copy()
            joint_lower = np.minimum(policy.states[:, :12].min(0), q0)-.15
            joint_upper = np.maximum(policy.states[:, :12].max(0), q0)+.15
            delta, info = cbf.project(measured_q, delta, scene,
                                      lower=joint_lower, upper=joint_upper,
                                      max_step=args.speed*policy.period,
                                      previous_delta=previous_delta,
                                      acceleration_step=args.acceleration*policy.period**2,
                                      target_gripper=desired_grip,
                                      swept_safe=lambda step: _motion_safe(
                                          sampler, measured_q, measured_grip, step,
                                          desired_grip, boxes, safety, True))
            if not info["success"]:
                obstacle_clearance, arm_clearance = scene.clearance(measured_q)
                failure_context = {
                    "reason": "cbf_projection_failed", "action_index": action_index,
                    "source_frame": int(policy.source_frames[action_index]),
                    "qpos": measured_q.tolist(), "gripper_position": measured_grip.tolist(),
                    "dataset_action": action.tolist(), "nominal_delta": nominal_delta.tolist(),
                    "projected_delta": delta.tolist(), "previous_delta": previous_delta.tolist(),
                    "joint_lower": joint_lower.tolist(), "joint_upper": joint_upper.tolist(),
                    "current_obstacle_clearance_m": (float(obstacle_clearance)
                                                     if np.isfinite(obstacle_clearance) else None),
                    "current_inter_arm_clearance_m": float(arm_clearance),
                    "collision_margin_m": safety["collision_margin_m"],
                    "inter_arm_margin_m": safety["inter_arm_margin_m"], "cbf": info,
                    "obbs": [{"center": b.center.tolist(), "axes": b.axes.tolist(),
                              "half_sizes": b.half_sizes.tolist()} for b in boxes]}
                raise RuntimeError(f"CBF could not project dataset action: {info}")
            if info.get("recovery_step"):
                current_clearance = scene.clearance(measured_q)
                future_scene = CollisionChecker(
                    sampler, desired_grip, boxes, margin=safety["collision_margin_m"],
                    inter_arm_margin=safety["inter_arm_margin_m"])
                future_clearance = future_scene.clearance(measured_q+delta)
                info.update(current_obstacle_clearance_m=(float(current_clearance[0])
                                                         if np.isfinite(current_clearance[0]) else None),
                            future_obstacle_clearance_m=(float(future_clearance[0])
                                                        if np.isfinite(future_clearance[0]) else None),
                            current_inter_arm_clearance_m=float(current_clearance[1]),
                            future_inter_arm_clearance_m=float(future_clearance[1]))
        if not _motion_safe(sampler, measured_q, measured_grip, delta, desired_grip, boxes,
                            safety, cbf is not None):
            failure_context = _motion_failure_context(
                sampler, measured_q, measured_grip, delta, desired_grip, boxes, safety,
                action=action, cbf_info=info, allow_unsafe_start=cbf is not None)
            failure_context.update(action_index=action_index,
                                   source_frame=int(policy.source_frames[action_index]))
            raise RuntimeError(f"Dataset joint/gripper action is blocked: {failure_context['reason']}")
        return delta, desired_grip, info

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start_time = time.monotonic()
    restore_start = start_time
    restore_replans = 0
    status = "step_limit"
    gripper_hold_on_error = None
    gripper_commands_sent = np.zeros(2, dtype=int)
    def active_time():
        return time.monotonic()

    def read_checked():
        return _read_checked(commander, live=args.live, url=url, age=age,
                             calibration_digest=sampler.calibration_digest,
                             snapshot_reader=scene_reader.read)

    try:
        # Baseline matches the VLA runtime's joint/gripper interpolation.
        # Avoid mode follows a separately checked joint path, then restores
        # grippers in checked increments before any dataset action is sent.
        distance = max(float(np.max(np.abs(q0-q)))/args.speed,
                       float(np.max(np.abs(g0-grip)))/args.gripper_speed_fraction_s)
        restore_steps = max(1, math.ceil(1.5*distance/policy.period))
        start_q, start_grip = q.copy(), grip.copy()
        with (args.output/"steps.jsonl").open("w", encoding="utf-8") as log:
            if cbf is not None:
                # Restore with fixed gripper geometry. Every nominal joint
                # action aims at the recorded start and is projected by the
                # future-surface QP-CBF. RRT is only a fallback when local CBF
                # cannot advance toward a geometrically safe final pose.
                room = args.restore_detour_room
                restore_path = smooth_joint_path(
                    np.stack((start_q, q0)), start_grip, dt=policy.period,
                    speed=args.speed, acceleration=args.acceleration,
                    lower=np.minimum(start_q, q0)-room,
                    upper=np.maximum(start_q, q0)+room,
                    source={"kind": "restore_start"})
                restore_follower = PathFollower(
                    restore_path, avoid=True, detour_lower=restore_path.lower,
                    detour_upper=restore_path.upper, lookahead=len(restore_path.q),
                    planning_timeout=args.restore_planning_timeout,
                    tolerance=args.restore_tolerance,
                    max_step=args.speed*policy.period, cbf_projector=cbf,
                    direct_goal_cbf=True)
                if commander:
                    restore_follower.before_plan = commander.stop
                    cbf.before_solve = commander.stop
                restore_deadline = active_time()+args.restore_timeout
                restore_steps = 0  # The checked follower replaces linear interpolation.
                restore_step = 0
                grip_first = False
                arms_stopped_for_grip = False

                def restoration_failure(reason, delta, scene, scene_boxes):
                    nonlocal failure_context
                    obstacle_clearance, arm_clearance = scene.clearance(q)
                    failure_context = {
                        "reason": reason, "phase": "restore", "restore_step": restore_step,
                        "qpos": q.tolist(), "gripper_position": grip.tolist(),
                        "target_qpos": q0.tolist(), "target_gripper": g0.tolist(),
                        "joint_delta": np.asarray(delta).tolist(),
                        "joint_lower": restore_path.lower.tolist(),
                        "joint_upper": restore_path.upper.tolist(),
                        "current_obstacle_clearance_m": (float(obstacle_clearance)
                                                         if np.isfinite(obstacle_clearance) else None),
                        "current_inter_arm_clearance_m": float(arm_clearance),
                        "cbf": restore_follower.last_cbf_info,
                        "restore_replans": restore_follower.replans,
                        "restore_cbf_interventions": restore_follower.cbf_interventions,
                        "restore_cbf_failures": restore_follower.cbf_failures,
                        "obbs": [{"center": b.center.tolist(), "axes": b.axes.tolist(),
                                  "half_sizes": b.half_sizes.tolist()} for b in scene_boxes]}

                while True:
                    tick = time.monotonic()
                    if active_time() >= restore_deadline:
                        scene = CollisionChecker(sampler, grip, boxes,
                                                 margin=safety["collision_margin_m"],
                                                 inter_arm_margin=safety["inter_arm_margin_m"])
                        restoration_failure("restore_timeout", np.zeros(12), scene, boxes)
                        raise RuntimeError(f"Obstacle-aware initial-state restoration exceeded "
                                           f"{args.restore_timeout:g}s")
                    if commander:
                        q, grip, boxes, _ = read_checked()
                    else:
                        boxes = boxes_from_json(scene_reader.read()) if args.live else []
                    scene = CollisionChecker(sampler, grip, boxes,
                                             margin=safety["collision_margin_m"],
                                             inter_arm_margin=safety["inter_arm_margin_m"])
                    if not scene.state_safe(q):
                        restoration_failure("restore_current_state_unsafe", np.zeros(12), scene, boxes)
                        raise RuntimeError("Current restoration state is inside obstacle/inter-arm clearance")
                    goal_scene = CollisionChecker(sampler, g0, boxes,
                                                  margin=safety["collision_margin_m"],
                                                  inter_arm_margin=safety["inter_arm_margin_m"])
                    if not goal_scene.state_safe(q0):
                        restoration_failure("restore_goal_unsafe", np.zeros(12), scene, boxes)
                        raise RuntimeError("Recorded initial joint/gripper state is blocked by the live scene")
                    joint_error = float(np.max(np.abs(q-q0)))
                    grip_error = float(np.max(np.abs(grip-g0)))
                    if joint_error <= args.restore_tolerance and grip_error <= args.gripper_tolerance:
                        break
                    if joint_error > args.restore_tolerance and not scene.state_safe(q0):
                        grip_first = True
                    changing_grip = grip_first or joint_error <= args.restore_tolerance
                    delta = np.zeros(12)
                    desired_grip = grip.copy()
                    state = "gripper" if changing_grip else "nominal"
                    if changing_grip:
                        if commander and not arms_stopped_for_grip:
                            commander.stop()
                            arms_stopped_for_grip = True
                        desired_grip = grip+np.clip(g0-grip,
                                                     -args.gripper_speed_fraction_s*policy.period,
                                                     args.gripper_speed_fraction_s*policy.period)
                    else:
                        arms_stopped_for_grip = False
                        cbf.gripper = grip.copy()
                        delta, state = restore_follower.next_delta(q, scene)
                        restore_replans = restore_follower.replans
                        if state in ("unsafe_current_state", "baseline_blocked", "no_safe_rejoin",
                                     "planning_failed", "step_blocked", "cbf_blocked", "completed"):
                            restoration_failure(f"restore_{state}", delta, scene, boxes)
                            raise RuntimeError(f"Obstacle-aware restoration cannot advance: {state}")
                    if not _motion_safe(sampler, q, grip, delta, desired_grip, boxes, safety):
                        restoration_failure("restore_motion_blocked", delta, scene, boxes)
                        raise RuntimeError("Obstacle-aware restoration step is blocked")
                    if active_time() >= restore_deadline:
                        restoration_failure("restore_timeout", delta, scene, boxes)
                        raise RuntimeError(f"Obstacle-aware initial-state restoration exceeded "
                                           f"{args.restore_timeout:g}s")
                    if commander and args.live:
                        fresh_q, fresh_grip, fresh_boxes, _ = read_checked()
                        drift = float(np.max(np.abs(fresh_q-q)))
                        grip_drift = float(np.max(np.abs(fresh_grip-grip)))
                        q, grip, boxes = fresh_q, fresh_grip, fresh_boxes
                        if drift > safety.get("max_state_drift_rad", .01) or grip_drift > .02:
                            restore_follower.detour = []
                            restore_follower.last_delta[:] = 0
                            continue
                        if changing_grip:
                            desired_grip = grip+np.clip(g0-grip,
                                                         -args.gripper_speed_fraction_s*policy.period,
                                                         args.gripper_speed_fraction_s*policy.period)
                        scene = CollisionChecker(sampler, grip, boxes,
                                                 margin=safety["collision_margin_m"],
                                                 inter_arm_margin=safety["inter_arm_margin_m"])
                        if (np.any(q+delta < restore_path.lower) or
                                np.any(q+delta > restore_path.upper) or
                                not _motion_safe(sampler, q, grip, delta, desired_grip, boxes, safety)):
                            restore_follower.detour = []
                            restore_follower.last_delta[:] = 0
                            continue
                    if active_time() >= restore_deadline:
                        restoration_failure("restore_timeout", delta, scene, boxes)
                        raise RuntimeError(f"Obstacle-aware initial-state restoration exceeded "
                                           f"{args.restore_timeout:g}s")
                    if commander:
                        if not changing_grip:
                            commander.send(q, delta)
                        else:
                            gripper_commands_sent += gripper_commander.send_if_changed(desired_grip)
                    else:
                        q += delta
                        grip = desired_grip.copy()
                    if grip_first and np.max(np.abs(grip-g0)) <= args.gripper_tolerance:
                        adjusted_goal = CollisionChecker(sampler, grip, boxes,
                                                         margin=safety["collision_margin_m"],
                                                         inter_arm_margin=safety["inter_arm_margin_m"])
                        if adjusted_goal.state_safe(q0):
                            grip_first = False
                    row = {"phase": "restore", "step": restore_step, "state": state,
                           "qpos": q.tolist(), "gripper_position": grip.tolist(),
                           "target_qpos": q0.tolist(), "target_gripper": desired_grip.tolist(),
                           "joint_delta": delta.tolist(), "cbf": restore_follower.last_cbf_info,
                           "restore_replans": restore_follower.replans,
                           "restore_cbf_interventions": restore_follower.cbf_interventions,
                           "cycle_compute_s": time.monotonic()-tick}
                    records.append(row)
                    log.write(json.dumps(row)+"\n")
                    restore_step += 1
                    if commander or args.live:
                        time.sleep(max(0., policy.period-(time.monotonic()-tick)))
                cbf.before_solve = None
            for index in range(restore_steps):
                tick = time.monotonic()
                fraction = smoothstep((index+1)/restore_steps)
                desired_q = start_q+fraction*(q0-start_q)
                desired_grip = start_grip+fraction*(g0-start_grip)
                if commander:
                    q, grip, boxes, _ = read_checked()
                else:
                    boxes = boxes_from_json(scene_reader.read()) if args.live else []
                delta = np.clip(desired_q-q, -args.speed*policy.period, args.speed*policy.period)
                if not _motion_safe(sampler, q, grip, delta, desired_grip, boxes, safety):
                    raise RuntimeError("Interpolated initial-state restoration is blocked")
                if commander:
                    # The arms and grippers can still move after the previous
                    # servoJ/SET. Use this measured state as the command base;
                    # a second read may differ even without external control.
                    if args.live:
                        q, grip, boxes, _ = read_checked()
                        delta = np.clip(desired_q-q, -args.speed*policy.period, args.speed*policy.period)
                        if not _motion_safe(sampler, q, grip, delta, desired_grip, boxes, safety):
                            raise RuntimeError("Fresh scene blocks initial-state restoration")
                    commander.send(q, delta)
                    sent = gripper_commander.send_if_changed(desired_grip)
                    gripper_commands_sent += sent
                else:
                    q += delta
                    grip = desired_grip.copy()
                row = {"phase": "restore", "step": index, "qpos": q.tolist(), "gripper_position": grip.tolist(),
                       "target_qpos": desired_q.tolist(), "target_gripper": desired_grip.tolist(),
                       "cycle_compute_s": time.monotonic()-tick}
                records.append(row)
                log.write(json.dumps(row)+"\n")
                if commander or args.live:
                    time.sleep(max(0., policy.period-(time.monotonic()-tick)))
            if commander and cbf is None:
                deadline = active_time()+args.restore_settle_timeout
                while True:
                    tick = time.monotonic()
                    q, grip, boxes, _ = read_checked()
                    if (np.max(np.abs(q-q0)) <= args.restore_tolerance
                            and np.max(np.abs(grip-g0)) <= args.gripper_tolerance):
                        break
                    if active_time() >= deadline:
                        raise RuntimeError("Initial-state restoration did not settle at recorded joints/grippers")
                    delta = np.clip(q0-q, -args.speed*policy.period, args.speed*policy.period)
                    if not _motion_safe(sampler, q, grip, delta, g0, boxes, safety):
                        raise RuntimeError("Initial-state settling is blocked")
                    if args.live:
                        q, grip, boxes, _ = read_checked()
                        delta = np.clip(q0-q, -args.speed*policy.period, args.speed*policy.period)
                        if not _motion_safe(sampler, q, grip, delta, g0, boxes, safety):
                            raise RuntimeError("Fresh scene blocks initial-state settling")
                    commander.send(q, delta)
                    gripper_commands_sent += gripper_commander.send_if_changed(g0)
                    time.sleep(max(0., policy.period-(time.monotonic()-tick)))
                commander.stop()
                q, grip = commander.read_state()
            elif commander:
                commander.stop()
                q, grip = commander.read_state()
            if np.max(np.abs(q-q0)) > args.restore_tolerance or np.max(np.abs(grip-g0)) > args.gripper_tolerance:
                raise RuntimeError("Initial-state restoration did not reach the recorded joint/gripper state")
            restore_elapsed = time.monotonic()-restore_start
            path_start = time.monotonic()
            action_index = 0
            action_attempt = 0
            action_started_at = active_time()
            hold_ticks = 0
            max_tracking_error = 0.
            previous_delta = np.zeros(12)
            while action_index < len(policy.actions):
                tick = time.monotonic()
                if commander:
                    q, grip, boxes, _ = read_checked()
                else:
                    boxes = boxes_from_json(scene_reader.read()) if args.live else []
                action = policy.infer(action_index, args.action_horizon)[0]
                error = float(np.max(np.abs(q-policy.states[action_index, :12])))
                max_tracking_error = max(max_tracking_error, error)
                if error > args.max_tracking_error:
                    raise RuntimeError(f"Dataset action replay lost its recorded path: {error:.4f} rad")
                if commander and args.live:
                    q, grip, boxes, _ = read_checked()
                    error = float(np.max(np.abs(q-policy.states[action_index, :12])))
                    max_tracking_error = max(max_tracking_error, error)
                    if error > args.max_tracking_error:
                        raise RuntimeError(f"Dataset action replay lost its recorded path: {error:.4f} rad")
                delta, desired_grip, info = plan_dataset_step(q, grip, boxes, action, previous_delta)
                if commander:
                    commander.send(q, delta)
                    sent = gripper_commander.send_if_changed(desired_grip)
                    gripper_commands_sent += sent
                else:
                    q += delta
                    grip = desired_grip.copy()
                target_error = float(np.max(np.abs(q-action[:12])))
                gripper_error = float(np.max(np.abs(grip-action[12:])))
                arrival_tolerance = min(args.arrival_tolerance, args.final_tolerance) if action_index == len(policy.actions)-1 else args.arrival_tolerance
                # A Robotiq SET POS 255 request may settle below 255 (for
                # example, after contacting an object). Dataset actions are
                # commands, not measured waypoints for the grippers. Their
                # live feedback still drives every FK/CBF check above.
                advanced = target_error <= arrival_tolerance
                row = {"phase": "path", "step": action_index, "attempt": action_attempt,
                       "action_advanced": advanced, "target_error_rad": target_error,
                       "gripper_error": gripper_error,
                       "source_frame": int(policy.source_frames[action_index]),
                       "qpos": q.tolist(), "gripper_position": grip.tolist(),
                       "action": action.tolist(), "sent_joint_delta": delta.tolist(),
                       "gripper_target": desired_grip.tolist(), "cbf": info,
                       "tracking_error_rad": error, "cycle_compute_s": time.monotonic()-tick}
                records.append(row)
                log.write(json.dumps(row)+"\n")
                previous_delta = delta.copy()
                if advanced:
                    action_index += 1
                    action_attempt = 0
                    action_started_at = active_time()
                else:
                    hold_ticks += 1
                    action_attempt += 1
                    if args.mode == "baseline" and active_time()-action_started_at > args.action_timeout:
                        failure_context = {"reason": "dataset_action_arrival_timeout",
                                           "action_index": action_index,
                                           "target_error_rad": target_error,
                                           "gripper_error": gripper_error,
                                           "target_gripper": desired_grip.tolist(),
                                           "measured_gripper": grip.tolist()}
                        raise RuntimeError(f"Dataset action {action_index} did not reach its "
                                           f"joint target within {args.action_timeout:g}s: "
                                           f"{target_error:.4f} rad, {gripper_error:.4f} grip")
                if commander or args.live:
                    time.sleep(max(0., policy.period-(time.monotonic()-tick)))
            path_elapsed = time.monotonic()-path_start
            if commander:
                commander.stop()
                q, grip = commander.read_state()
            final_error = float(np.max(np.abs(q-policy.metadata["final_joint_state"])))
            status = "completed" if final_error <= args.final_tolerance else "final_joint_error"
    except BaseException as exc:
        status = f"error: {type(exc).__name__}: {exc}"
        if commander and gripper_commander:
            try:
                _, measured_grip = commander.read_state()
                gripper_commander.send_if_changed(measured_grip)
                gripper_hold_on_error = "sent_current_measured_position"
            except BaseException as hold_exc:
                gripper_hold_on_error = f"failed: {type(hold_exc).__name__}: {hold_exc}"
        raise
    finally:
        try:
            if commander:
                commander.close()
        finally:
            if gripper_commander:
                gripper_commander.close()
            report = {"status": status, "completed": status == "completed", "execution": "physical" if args.execute else "virtual",
                      "inference_source": "lerobot_dataset_actions", "mode": args.mode,
                      "dataset": str(Path(dataset_path).resolve()), "source": policy.metadata,
                      "steps": len(records), "actions_executed": sum(r.get("action_advanced", False) for r in records),
                      "path_control_ticks": sum(r["phase"] == "path" for r in records),
                      "hold_ticks": locals().get("hold_ticks", 0),
                      "unstable_scene_reads": scene_reader.unstable_scene_reads,
                      "gripper_commands_sent": gripper_commands_sent.tolist(),
                      "gripper_hold_on_error": gripper_hold_on_error,
                      "failure_context": failure_context,
                      "mean_cycle_compute_s": (float(np.mean([r["cycle_compute_s"] for r in records])) if records else None),
                      "max_cycle_compute_s": (max(r["cycle_compute_s"] for r in records) if records else None),
                      "restore_elapsed_s": locals().get("restore_elapsed", time.monotonic()-restore_start),
                      "restore_replans": restore_replans,
                      "restore_cbf_interventions": (locals()["restore_follower"].cbf_interventions
                                                    if "restore_follower" in locals() else 0),
                      "restore_cbf_failures": (locals()["restore_follower"].cbf_failures
                                               if "restore_follower" in locals() else 0),
                      "path_elapsed_s": locals().get("path_elapsed", time.monotonic()-path_start if "path_start" in locals() else None),
                      "final_joint_error_rad": locals().get("final_error"),
                      "max_tracking_error_rad": locals().get("max_tracking_error"),
                      "elapsed_s": time.monotonic()-start_time}
            (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    commands = p.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="HDF5 to state/action LeRobot dataset")
    convert.add_argument("--hdf5", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    for command in (convert,):
        command.add_argument("--fps", type=int, default=20)
        command.add_argument("--time-scale", type=float, default=1.)
        command.add_argument("--start-frame", type=int, default=0)
        command.add_argument("--end-frame", type=int)
    replay = commands.add_parser("replay", help="Dataset action inference and dual-arm/dual-gripper replay")
    source = replay.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5", type=Path)
    source.add_argument("--dataset", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--mode", choices=("baseline", "avoid"), default="baseline")
    replay.add_argument("--live", action="store_true")
    replay.add_argument("--execute", action="store_true")
    replay.add_argument("--execute-ack", default="")
    replay.add_argument("--initial-state", type=Path)
    replay.add_argument("--obstacle-url")
    replay.add_argument("--fps", type=int, default=20)
    replay.add_argument("--time-scale", type=float, default=1.)
    replay.add_argument("--start-frame", type=int, default=0)
    replay.add_argument("--end-frame", type=int)
    replay.add_argument("--points-per-link", type=int, default=16)
    replay.add_argument("--action-horizon", type=int, default=32)
    replay.add_argument("--speed", type=float, default=.5)
    replay.add_argument("--acceleration", type=float, default=1.2)
    replay.add_argument("--gripper-speed-fraction-s", type=float, default=.5)
    replay.add_argument("--gripper-speed", type=int, default=40)
    replay.add_argument("--gripper-force", type=int, default=25)
    replay.add_argument("--restore-tolerance", type=float, default=.015)
    replay.add_argument("--gripper-tolerance", type=float, default=.06)
    replay.add_argument("--restore-settle-timeout", type=float, default=10.)
    replay.add_argument("--restore-timeout", type=float, default=90.,
                        help="Maximum total time for obstacle-aware restoration, seconds")
    replay.add_argument("--restore-planning-timeout", type=float, default=5.,
                        help="Maximum time for each restoration RRT plan, seconds")
    replay.add_argument("--restore-detour-room", type=float, default=.35,
                        help="Extra joint-space room around restoration endpoints, radians")
    replay.add_argument("--max-tracking-error", type=float, default=.12)
    replay.add_argument("--arrival-tolerance", type=float, default=.04)
    replay.add_argument("--action-timeout", type=float, default=5.,
                        help="Per-action joint arrival timeout in baseline mode only (seconds)")
    replay.add_argument("--final-tolerance", type=float, default=.03)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "convert":
        print(json.dumps(convert_hdf5(args.hdf5, args.output, fps=args.fps, time_scale=args.time_scale,
                                      start_frame=args.start_frame, end_frame=args.end_frame), ensure_ascii=False))
        return 0
    report = run_replay(args)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
