"""Convert Quest3 episodes to state/action LeRobot data and replay like a VLA queue.

The action provider is the recorded dataset, not a learned policy.  Joint
actions are absolute 12-joint targets; the two gripper values are closing
targets. Replay executes on hardware by default; use --no-execute for virtual
replay.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import math
from pathlib import Path
import socket
import sys
import time

import h5py
import numpy as np
import pyarrow.parquet as parquet
import yaml

from obstacle_validation.cbf import PreviewFKCache, RecordedPathCBF
from obstacle_validation.core import CollisionChecker, segments_within_box_margin, vector
from obstacle_validation.quest3_schema import NAMES, _joint_order
from obstacle_validation.runtime import (EXECUTE_ACK, CapturedFrameObservation, DualCommander, LatestSnapshotCache,
                                         boxes_from_json, read_snapshot, snapshot_state, timed_call,
                                         validate_obstacle_snapshot)
from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
from obstacle_validation.manual_annotation_filter import ManualAnnotationFilter
from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.manual_obstacle_selection import latest_annotation_path
from safety_module.geometry import swept_inter_arm_safe
from safety_module.dual_camera_calibration import validate_calibration_identity
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from safety_module.runtime_contract import SnapshotExpired


SCHEMA = "quest3_dual_ur7e_lerobot_absolute_action_v1"
STATE_NAMES = [*NAMES, "left_gripper", "right_gripper"]
ACTION_NAMES = [*NAMES, "left_gripper_target", "right_gripper_target"]


def latest_manual_annotations() -> Path:
    """Find the newest capture directory containing annotations, including legacy IDs."""
    root = resolve_repo_path("outputs/manual_obstacles")
    path = latest_annotation_path(root)
    if path is None:
        raise ValueError(f"No annotations.json found under {root}; save annotations first "
                         "or pass --manual-annotations /path/to/annotations.json")
    return path


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

    def __init__(self, left_ip: str, right_ip: str, *, speed: int = 255, force: int = 25):
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


def _motion_safe(sampler, q, grip, delta, target_grip, boxes, safety, allow_unsafe_start=False,
                 observation_now_ns=None):
    from safety_module.occupancy import observation_rollout_safe
    points = np.asarray(sampler.link_points(q, grip)).reshape(-1, 3)
    left_count = len(sampler.left_link_names)*sampler.points_per_link
    if not len(points) or not np.isfinite(points).all() or not 0 < left_count < len(points):
        raise ValueError("Invalid dual-arm FK points")
    initial_safe = (all(float(_obstacle_point_distances(points, box).min()) > safety["collision_margin_m"]
                        for box in boxes)
                    and swept_inter_arm_safe(points, points, left_count, safety["inter_arm_margin_m"]))
    if not initial_safe:
        return (allow_unsafe_start and observation_rollout_safe(
            sampler, q, grip, [delta], boxes, gripper_chunk=[target_grip],
            margin_m=safety["collision_margin_m"], now_ns=observation_now_ns,
            allow_margin_recovery=True) and _recovery_motion_safe(
                sampler, q, grip, delta, target_grip, boxes, safety))
    # Check observation, rounded obstacle margins and inter-arm separation in
    # the same FK pass. Each commanded sweep is still checked at four substeps.
    observation = getattr(boxes, "observation", None)
    previous = points
    for step in range(1, 5):
        fraction = step/4.
        current_points = np.asarray(sampler.link_points(q+delta*fraction,
                                   grip+(target_grip-grip)*fraction)).reshape(-1, 3)
        if current_points.shape != previous.shape or not np.isfinite(current_points).all():
            return False
        if observation is not None and not observation.segments_observed(
                previous, current_points, margin_m=safety["collision_margin_m"], now_ns=observation_now_ns):
            return False
        if any(segments_within_box_margin(previous, current_points, box,
                                          safety["collision_margin_m"]).any() for box in boxes):
            return False
        if not swept_inter_arm_safe(previous, current_points, left_count, safety["inter_arm_margin_m"]):
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
    from safety_module.occupancy import observation_rollout_safe
    if not observation_rollout_safe(sampler, q, grip, [delta], boxes,
                                    gripper_chunk=[target_grip], margin_m=safety["collision_margin_m"],
                                    allow_margin_recovery=allow_unsafe_start):
        reason = "occupied_voxel_sweep"

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
    """Read cached OBBs with an explicit latest-frame or strict age policy."""

    def __init__(self, url, age, manual_filter=None, *, reuse_expired=False, provider=None):
        self.url, self.age, self.manual_filter = url, age, manual_filter
        self.reuse_expired = reuse_expired
        self.stale_snapshot_reads = 0
        self.max_snapshot_age_s = 0.
        self.unstable_scene_reads = 0
        self.timings = {}
        self.cache = None
        self.provider = provider
        self._closed = False
        self._decoded_snapshot = self._decoded_boxes = None

    def start(self, initial):
        if self.provider is not None:
            return  # The embedded source already publishes from its own worker.
        self.cache = LatestSnapshotCache(self._read_direct, self._validate, initial,
                                         reuse_on_transport_error=self.reuse_expired)

    def close(self):
        if self._closed:
            return
        if self.cache is not None:
            self.cache.close()
        if self.provider is not None:
            self.provider.close()
        self._closed = True

    def _validate(self, snapshot):
        age = validate_obstacle_snapshot(snapshot, float("inf") if self.reuse_expired else self.age,
                                         allow_tracked_obbs=True)
        self.max_snapshot_age_s = max(self.max_snapshot_age_s, age)
        if age > self.age:
            self.stale_snapshot_reads += 1

    def boxes(self, snapshot):
        # Used only by the control thread. The worker publishes immutable
        # snapshots; mutable sweep diagnostics stay local to the controller.
        if snapshot is not self._decoded_snapshot:
            self._decoded_boxes = timed_call(self.timings, "control_decode", boxes_from_json, snapshot)
            if self.reuse_expired and getattr(self._decoded_boxes, "observation", None) is not None:
                self._decoded_boxes.observation = CapturedFrameObservation(self._decoded_boxes.observation)
            self._decoded_snapshot = snapshot
            if hasattr(self._decoded_boxes, "__dict__"):
                self._decoded_boxes.source_snapshot = snapshot
        return self._decoded_boxes

    def read(self, *, max_wait_s=0., after_timestamp_ns=None):
        if self.reuse_expired:
            max_wait_s, after_timestamp_ns = 0., None
        if self.cache is not None:
            return self.cache.read(max_wait_s=max_wait_s, after_timestamp_ns=after_timestamp_ns)
        return self._read_direct(max_wait_s=max_wait_s, after_timestamp_ns=after_timestamp_ns)

    def _read_direct(self, *, max_wait_s=0., after_timestamp_ns=None):
        if self.provider is not None:
            snapshot = timed_call(self.timings, "embedded_read", self.provider.read,
                                  max_wait_s=max_wait_s, after_timestamp_ns=after_timestamp_ns)
            self._validate(snapshot)
            return snapshot
        options = {"snapshot_transform": self.manual_filter} if self.manual_filter is not None else {}
        if after_timestamp_ns is not None:
            options["after_timestamp_ns"] = after_timestamp_ns
        if self.reuse_expired:
            max_wait_s = 0.
            options["http_timeout_s"] = self.age
        snapshot = read_snapshot(self.url, float("inf") if self.reuse_expired else self.age, max_wait_s=max_wait_s,
                                 allow_tracked_obbs=True, timing_stats=self.timings, **options)
        if str(snapshot.get("status", "")).startswith("error: unstable obstacle point clusters"):
            self.unstable_scene_reads += 1
        return snapshot


def _read_checked(commander, *, live, url, age, calibration_digest, snapshot_reader=None,
                  boxes_reader=boxes_from_json, timing_stats=None):
    snapshot = (snapshot_reader() if snapshot_reader is not None else read_snapshot(url, age)) if live else None
    if snapshot is not None:
        validate_calibration_identity(snapshot, calibration_digest)
    q, grip = timed_call(timing_stats, "robot_feedback", commander.read_state)
    boxes = timed_call(timing_stats, "snapshot_decode", boxes_reader, snapshot) if snapshot is not None else []
    return q, grip, boxes, snapshot


def run_replay(args) -> dict:
    # Acquisition can start before the main replay loop, including in virtual
    # mode. Release it on every initialization error as well as normal exit.
    with ExitStack() as cleanup:
        return _run_replay(args, cleanup)


def _run_replay(args, cleanup) -> dict:
    annotation_path = getattr(args, "manual_annotations", None)
    if args.mode == "avoid" and not (args.manual_selection or annotation_path):
        annotation_path = args.manual_annotations = latest_manual_annotations()
        print(f"[replay] using latest annotations: {annotation_path}", file=sys.stderr, flush=True)
    if args.mode == "baseline" and (args.manual_selection or annotation_path):
        raise ValueError("--manual-selection / --manual-annotations is only used with --mode avoid")
    # Mode determines acquisition: direct replay never starts perception;
    # avoidance always reads the selected live scene, even without --live.
    args.live = args.mode == "avoid"
    config, config_path = load_project_config(args.project_config)
    safety = config["safety"]
    # Use the full Robotiq hardware speed in both replay modes unless requested otherwise.
    gripper_speed = args.gripper_speed
    if not 0 <= gripper_speed <= 255 or not 0 <= args.gripper_force <= 255:
        raise ValueError("Robotiq gripper speed and force must be in [0,255]")
    rate_limits = (args.speed, args.acceleration)
    if (any(value is not None and (not np.isfinite(value) or value <= 0) for value in rate_limits)
            or not np.isfinite(args.restore_speed) or args.restore_speed <= 0
            or not np.isfinite(args.gripper_speed_fraction_s) or args.gripper_speed_fraction_s <= 0
            or args.action_horizon < 1 or args.execute_horizon < 1
            or not np.isfinite(args.restore_settle_timeout) or args.restore_settle_timeout <= 0
            or not np.isfinite(args.restore_timeout) or args.restore_timeout <= 0
            or (args.mode == "baseline" and
                (not np.isfinite(args.action_timeout) or args.action_timeout <= 0))
            or not np.isfinite(args.arrival_tolerance) or args.arrival_tolerance <= 0
            or not np.isfinite(args.max_tracking_error) or args.max_tracking_error <= 0):
        raise ValueError("Speed, acceleration, tolerances and timeouts must be positive")
    if args.execute and args.initial_state:
        raise ValueError("--initial-state is only available for virtual replay")
    if args.execute and args.execute_ack != EXECUTE_ACK:
        raise ValueError(f"Physical execution requires --execute-ack {EXECUTE_ACK}")
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
    max_joint_step = args.speed*policy.period if args.speed is not None else np.inf
    restore_max_joint_step = args.restore_speed*policy.period
    acceleration_step = args.acceleration*policy.period**2 if args.acceleration is not None else np.inf
    if args.hdf5 is not None:
        requested = Path(args.hdf5).resolve()
        source = policy.metadata
        if (source["source_hdf5"] != str(requested) or source["fps"] != args.fps
                or source["time_scale"] != args.time_scale or source["start_frame"] != args.start_frame
                or (args.end_frame is not None and source["end_frame_inclusive"] != args.end_frame)):
            raise ValueError("Existing LeRobot dataset does not match requested HDF5 conversion")
    sampler = (DualUR7eSurfacePointSampler(points_per_link=args.points_per_link, project_config=config_path)
               if args.mode == "avoid" else None)
    if args.execute and sampler is not None and (not all(sampler.tcp_transform_verified.values()) or not sampler.calibration_digest):
        raise RuntimeError("Physical replay requires verified dual TCP and shared calibration")
    if args.live and not sampler.calibration_digest:
        raise RuntimeError("Live replay requires shared dual-camera calibration")
    url = args.obstacle_url or safety["obstacle_url"]
    age = safety["stale_timeout_s"]
    if annotation_path:
        manual_filter = ManualAnnotationFilter(annotation_path, config_path)
    elif args.manual_selection:
        # Existing command lines may pass annotations.json through the old option.
        saved_path = Path(args.manual_selection)
        saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.is_file() else {}
        manual_filter = (ManualAnnotationFilter(args.manual_selection, config_path)
                         if "annotations" in saved or "boxes" in saved else ManualVoxelFilter(args.manual_selection))
    else:
        manual_filter = None
    reuse_expired = args.mode == "avoid" and args.obstacle_age_policy == "latest"
    embedded = (args.live and args.mode == "avoid" and manual_filter is not None
                and args.obstacle_source == "auto" and args.obstacle_url is None)
    provider = None
    if embedded:
        from real_scripts.replay_voxblox import ReplayVoxbloxSource
        provider = ReplayVoxbloxSource(config_path, manual_filter,
                                      startup_timeout_s=args.voxblox_startup_timeout,
                                      robot_mask=getattr(args, "robot_mask", False))
        scene_reader = LiveSceneReader(url, age, manual_filter, reuse_expired=reuse_expired, provider=provider)
        print(f"[replay] starting selected-region Voxblox TSDF (robot mask={getattr(args, 'robot_mask', False)}; ESDF/viewer off)",
              file=sys.stderr, flush=True)
    else:
        scene_reader = LiveSceneReader(url, age, manual_filter, reuse_expired=reuse_expired)
    cleanup.callback(scene_reader.close)
    # Physical restoration uses robot feedback only; defer live perception
    # until it finishes. Virtual live replay may read a scene for its initial pose.
    # Embedded acquisition must be ready before creating motion controllers.
    # The physical restore still uses only robot feedback.
    ready = scene_reader.read(max_wait_s=5.) if embedded else None
    if ready is not None:
        validate_calibration_identity(ready, sampler.calibration_digest)
    first = (ready if embedded else scene_reader.read(max_wait_s=5.)) if args.live and not args.execute else None
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
            gripper_commander = DualGripperCommander(left_ip, right_ip, speed=gripper_speed,
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
                           iterations=safety["cbf_iterations"],
                           modulation=args.avoidance_controller == "mcbf",
                           optimization_space=args.optimization_space) if args.mode == "avoid" else None)
    adaptive = None
    if cbf is not None and cbf.modulation:
        from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
        adaptive = EllipsoidMCBF(
            sampler, margin=safety["collision_margin_m"], alpha=safety["cbf_alpha"],
            inter_arm_margin=safety["inter_arm_margin_m"], period=policy.period,
            optimization_space=args.optimization_space,
            config=EllipsoidMCBFConfig(**safety.get("ellipsoid_mcbf", {})))
    scene_snapshot = first
    boxes = scene_reader.boxes(first) if first is not None else []
    failure_context = None
    last_cbf_failure = last_tracking_excursion = None
    action_joint_lower = np.minimum(policy.states[:, :12].min(0), q0)-.15 if cbf is not None else None
    action_joint_upper = np.maximum(policy.states[:, :12].max(0), q0)+.15 if cbf is not None else None
    if adaptive is not None:
        # The recorded tracking tube excludes useful detours. Use the UR
        # model's joint range, retaining the same angle branch as the episode.
        arm_limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        limits = np.tile(arm_limits, 2)
        action_joint_lower, action_joint_upper = -limits, limits
        if np.any(policy.actions[:, :12] < -limits) or np.any(policy.actions[:, :12] > limits):
            raise ValueError("Recorded targets exceed the UR model joint bounds")
    # Preserve waypoint precision when removing the speed limit: a looser
    # arrival tolerance would consume room reserved for avoidance detours.
    avoid_waypoint_tolerance = min(.0125, args.arrival_tolerance, max_joint_step/2,
                                  args.max_tracking_error/4)
    tracking_feedback_margin = min(float(safety.get("max_state_drift_rad", .01)),
                                   args.max_tracking_error/4)
    # A measured waypoint can be reached with residual error. Reserve that
    # tolerance and feedback motion inside the original tracking limit.
    planning_tracking_room = (args.max_tracking_error-avoid_waypoint_tolerance
                              -tracking_feedback_margin)*(1.-1e-6)
    planning_sampler = (PreviewFKCache(sampler, capacity=max(64, 2*args.action_horizon))
                        if cbf is not None else None)
    runtime_timings = {}

    def measure(name, function, *args, **kwargs):
        return timed_call(runtime_timings, name, function, *args, **kwargs)

    def plan_dataset_chunk(measured_q, measured_grip, boxes, actions, previous_delta):
        nonlocal last_cbf_failure
        if adaptive is not None:
            optimized, info = measure(
                "adaptive_mcbf_pipeline", adaptive.plan, measured_q, measured_grip, actions, boxes,
                lower=action_joint_lower, upper=action_joint_upper,
                max_step=max_joint_step, previous_delta=previous_delta, acceleration_step=acceleration_step)
            progress_log.write(json.dumps({"action_index": action_index, "stage": "adaptive_control",
                "elapsed_s": info["planning_seconds"], "adaptive_stage": info["adaptive_stage"],
                "solver_message": info["steps"][0]["solver_message"]})+"\n")
            progress_log.flush()
            return optimized, info
        def progress(event):
            row = {"action_index": action_index, "horizon": len(actions), **event}
            progress_log.write(json.dumps(row)+"\n")
            progress_log.flush()
            if event["stage"] in ("seed", "joint_qp", "joint_qp_complete", "guidance_fallback", "complete"):
                print(f"[replay] action {action_index}:{action_index+len(actions)} "
                      f"{event['stage']} {event['elapsed_s']:.2f}s"
                      +(f" space={event['optimization_space']}" if "optimization_space" in event else "")
                      +(f" {event['solver_message']}" if "solver_message" in event else "")
                      +(f" {event['reason']}" if event.get("reason") else ""), file=sys.stderr, flush=True)
        cbf.progress_callback = progress
        planning_now = time.monotonic_ns()
        scene = CollisionChecker(planning_sampler, measured_grip, boxes,
                                 margin=safety["collision_margin_m"],
                                 inter_arm_margin=safety["inter_arm_margin_m"],
                                 observation_now_ns=planning_now)
        indices = action_index+np.arange(len(actions))
        references = policy.states[indices, :12]
        following = policy.states[np.minimum(indices+1, len(policy.states)-1), :12]
        # Each corrected endpoint must remain valid when the executor advances
        # to the next sample; do not plan a detour that the tracking guard rejects.
        room = planning_tracking_room
        lower = np.maximum(action_joint_lower, np.maximum(references, following)-room)
        upper = np.minimum(action_joint_upper, np.minimum(references, following)+room)
        # The CBF requires a finite step bound. With no requested rate limit,
        # use the full joint/tracking envelope so no valid motion is clipped.
        planning_max_step = min(max_joint_step, float(np.max(
            np.maximum(upper.max(0), measured_q)-np.minimum(lower.min(0), measured_q))))
        optimized, info = measure(
            "chunk_cbf_pipeline", cbf.project_chunk, measured_q, measured_grip, actions, scene,
            lower=lower, upper=upper,
            max_step=planning_max_step, previous_delta=previous_delta,
            acceleration_step=acceleration_step, fk_sampler=planning_sampler)
        if not info["success"]:
            last_cbf_failure = {
                "reason": "cbf_projection_failed", "action_index": action_index,
                "failed_action_index": action_index+info["failed_action_index"],
                "qpos": measured_q.tolist(), "gripper_position": measured_grip.tolist(),
                "failed_planning_qpos": info["failed_qpos"],
                "dataset_action": actions[info["failed_action_index"]].tolist(),
                "nominal_delta": info["failed_nominal_delta"], "cbf": info["failed_step"],
                "dataset_action_chunk": actions.tolist(), "cbf_chunk": info,
                "joint_lower": info["failed_joint_lower"], "joint_upper": info["failed_joint_upper"],
                "obbs": [{"center": b.center.tolist(), "axes": b.axes.tolist(),
                          "half_sizes": b.half_sizes.tolist()} for b in boxes]}
        return optimized, info

    args.output.mkdir(parents=True, exist_ok=True)
    # A previous run's failure must not look like this run's live status while
    # a long optimization is still in progress. Final metrics replace this.
    (args.output/"report.json").write_text(json.dumps({
        "status": "running", "completed": False,
        "execution": "physical" if args.execute else "virtual", "mode": args.mode,
        "constraint_checks": cbf is not None, "action_optimization": cbf is not None,
        "dataset": str(Path(dataset_path).resolve()),
        "progress_file": "planning.jsonl" if cbf is not None else None}), encoding="utf-8")
    records = []
    start_time = time.monotonic()
    restore_start = start_time
    restore_replans = 0
    status = "step_limit"
    gripper_hold_on_error = None
    gripper_commands_sent = np.zeros(2, dtype=int)
    stale_scene_pauses = 0
    def active_time():
        return time.monotonic()

    def check_tracking(measured_q, index):
        nonlocal max_tracking_error, failure_context, last_tracking_excursion
        difference = np.abs(measured_q-policy.states[index, :12])
        joint = int(np.argmax(difference))
        error = float(difference[joint])
        max_tracking_error = max(max_tracking_error, error)
        if cbf is not None and error > args.max_tracking_error:
            context = {"reason": "dataset_tracking_error", "action_index": index,
                               "joint_index": joint, "joint_name": STATE_NAMES[joint],
                               "tracking_error_rad": error, "limit_rad": args.max_tracking_error,
                               "qpos": measured_q.tolist(),
                               "recorded_qpos": policy.states[index, :12].tolist()}
            # Continue optimizing the same action back into its tracking
            # envelope. An excursion does not consume or abort an action.
            last_tracking_excursion = context
        return error

    def pause_for_fresh_scene(after_timestamp_ns=None):
        nonlocal gripper_commands_sent, stale_scene_pauses
        if reuse_expired:
            raise SnapshotExpired("Cached frame is unusable at capture time; latest policy does not wait for refresh")
        measure("stop", commander.stop)
        _, held_grip = measure("robot_feedback", commander.read_state)
        gripper_commands_sent += measure("gripper_command", gripper_commander.send_if_changed, held_grip)
        stale_scene_pauses += 1
        fresh = measure("fresh_frame_wait", scene_reader.read, max_wait_s=3., after_timestamp_ns=after_timestamp_ns)
        validate_calibration_identity(fresh, sampler.calibration_digest)
        measured_q, measured_grip = measure("robot_feedback", commander.read_state)
        return measured_q, measured_grip, scene_reader.boxes(fresh), {**fresh, "_stale_scene_pause": True}

    def read_checked():
        try:
            return measure("scene_and_feedback", _read_checked, commander, live=args.live, url=url, age=age,
                                 calibration_digest=sampler.calibration_digest,
                                 snapshot_reader=scene_reader.read, boxes_reader=scene_reader.boxes,
                                 timing_stats=runtime_timings)
        except SnapshotExpired:
            if not (commander and args.mode == "avoid" and manual_filter is not None):
                raise
            return pause_for_fresh_scene()

    def read_restore_feedback():
        # Restoration intentionally does not consume perception or run FK.
        return measure("robot_feedback", commander.read_state)

    restore_deadline = active_time()+args.restore_timeout

    def check_restore_timeout():
        nonlocal failure_context
        if active_time() >= restore_deadline:
            failure_context = {"phase": "restore", "reason": "restore_timeout",
                               "qpos": q.tolist(), "gripper_position": grip.tolist(),
                               "target_qpos": q0.tolist(), "target_gripper": g0.tolist(),
                               "restore_controller": "interpolation", "collision_checks": False}
            raise RuntimeError(f"Initial-state restoration exceeded {args.restore_timeout:g}s")

    try:
        # Both modes restore with independent joint/gripper interpolation rates, without
        # obstacle, observation, inter-arm or CBF checks. Live scene consumption
        # starts only after physical restoration has settled and stopped.
        distance = max(float(np.max(np.abs(q0-q)))/args.restore_speed,
                       float(np.max(np.abs(g0-grip)))/args.gripper_speed_fraction_s)
        restore_steps = max(1, math.ceil(1.5*distance/policy.period))
        start_q, start_grip = q.copy(), grip.copy()
        with (args.output/"steps.jsonl").open("w", encoding="utf-8") as log, \
             (args.output/"chunks.jsonl").open("w", encoding="utf-8") as chunk_log, \
             (args.output/"planning.jsonl").open("w", encoding="utf-8") as progress_log:
            for index in range(restore_steps):
                tick = time.monotonic()
                check_restore_timeout()
                fraction = smoothstep((index+1)/restore_steps)
                desired_q = start_q+fraction*(q0-start_q)
                desired_grip = start_grip+fraction*(g0-start_grip)
                if commander:
                    q, grip = read_restore_feedback()
                delta = np.clip(desired_q-q, -restore_max_joint_step, restore_max_joint_step)
                if commander:
                    measure("joint_command", commander.send, q, delta)
                    gripper_commands_sent += measure("gripper_command", gripper_commander.send_if_changed, desired_grip)
                else:
                    q += delta
                    grip = desired_grip.copy()
                row = {"phase": "restore", "step": index, "qpos": q.tolist(), "gripper_position": grip.tolist(),
                       "target_qpos": desired_q.tolist(), "target_gripper": desired_grip.tolist(),
                       "joint_delta": delta.tolist(), "cbf": None,
                       "restore_controller": "interpolation", "collision_checks": False,
                       "cycle_compute_s": time.monotonic()-tick}
                records.append(row)
                log.write(json.dumps(row)+"\n")
                if commander or args.live:
                    time.sleep(max(0., policy.period-(time.monotonic()-tick)))
            if commander:
                settle_deadline = active_time()+args.restore_settle_timeout
                while True:
                    tick = time.monotonic()
                    q, grip = read_restore_feedback()
                    if (np.max(np.abs(q-q0)) <= args.restore_tolerance
                            and np.max(np.abs(grip-g0)) <= args.gripper_tolerance):
                        break
                    check_restore_timeout()
                    if active_time() >= settle_deadline:
                        failure_context = {"phase": "restore", "reason": "restore_settle_timeout",
                                           "qpos": q.tolist(), "gripper_position": grip.tolist(),
                                           "target_qpos": q0.tolist(), "target_gripper": g0.tolist(),
                                           "collision_checks": False}
                        raise RuntimeError("Initial-state restoration did not settle at recorded joints/grippers")
                    delta = np.clip(q0-q, -restore_max_joint_step, restore_max_joint_step)
                    measure("joint_command", commander.send, q, delta)
                    gripper_commands_sent += measure("gripper_command", gripper_commander.send_if_changed, g0)
                    time.sleep(max(0., policy.period-(time.monotonic()-tick)))
                measure("stop", commander.stop)
                q, grip = read_restore_feedback()
            if np.max(np.abs(q-q0)) > args.restore_tolerance or np.max(np.abs(grip-g0)) > args.gripper_tolerance:
                raise RuntimeError("Initial-state restoration did not reach the recorded joint/gripper state")
            restore_elapsed = time.monotonic()-restore_start
            if args.live:
                scene_snapshot = scene_reader.read(max_wait_s=5.)
                validate_calibration_identity(scene_snapshot, sampler.calibration_digest)
                boxes = scene_reader.boxes(scene_snapshot)
                scene_reader.start(scene_snapshot)
            path_start = time.monotonic()
            action_index = 0
            action_attempt = 0
            action_started_at = active_time()
            hold_ticks = 0
            detour_control_ticks = 0
            max_tracking_error = 0.
            previous_delta = np.zeros(12)
            chunk_plan = chunk_info = None
            chunk_start = chunk_end = 0
            chunk_id = -1
            chunk_replans = 0
            optimization_retries = 0
            waypoint_index = 0
            final_optimized_target = np.asarray(policy.metadata["final_joint_state"]).copy()
            while action_index < len(policy.actions):
                tick = time.monotonic()
                if commander:
                    if cbf is None:
                        q, grip = measure("robot_feedback", commander.read_state)
                    else:
                        q, grip, boxes, scene_snapshot = read_checked()
                    if scene_snapshot and scene_snapshot.get("_stale_scene_pause"):
                        previous_delta[:] = 0
                        chunk_plan = None
                        chunk_replans += 1
                else:
                    if args.live:
                        scene_snapshot = scene_reader.read()
                        boxes = scene_reader.boxes(scene_snapshot)
                    else:
                        boxes = []
                action = policy.actions[action_index]
                error = check_tracking(q, action_index)
                try:
                    if cbf is not None:
                        if chunk_plan is None or action_index >= chunk_end:
                            # The CBF chunk pipeline holds before planning;
                            # Adaptive MCBF computes the next feedback command.
                            if commander and adaptive is None:
                                measure("stop", commander.stop)
                                q, grip, boxes, scene_snapshot = read_checked()
                                previous_delta[:] = 0
                            if action_index >= chunk_end:
                                chunk_end = min(len(policy.actions), action_index+args.action_horizon,
                                                action_index+args.execute_horizon)
                            chunk_start = action_index
                            action_chunk = policy.infer(chunk_start, args.action_horizon)
                            execution_actions = action_chunk[:chunk_end-chunk_start]
                            chunk_plan, chunk_info = plan_dataset_chunk(
                                q, grip, boxes, execution_actions, previous_delta)
                            if chunk_info.get("adaptive_feedback"):
                                execution_actions = execution_actions[:1]
                                chunk_end = chunk_start+1
                            if not chunk_info["success"]:
                                optimization_retries += 1
                                chunk_replans += 1
                                hold_ticks += 1
                                action_attempt += 1
                                previous_delta[:] = 0
                                chunk_plan = None
                                row = {"phase": "optimization_retry", "step": action_index,
                                       "attempt": action_attempt, "action_advanced": False,
                                       "qpos": q.tolist(), "gripper_position": grip.tolist(),
                                       "reason": chunk_info.get("reason", "cbf_projection_failed"),
                                       "cbf": chunk_info["failed_step"],
                                       "tracking_error_rad": error,
                                       "cycle_compute_s": time.monotonic()-tick}
                                records.append(row)
                                log.write(json.dumps(row)+"\n")
                                log.flush()
                                chunk_log.write(json.dumps({"status": "retry",
                                    "optimization_retry": optimization_retries,
                                    **last_cbf_failure})+"\n")
                                chunk_log.flush()
                                # Retry indefinitely with new feedback/scene,
                                # retaining the action index and full step size.
                                time.sleep(max(0., policy.period-(time.monotonic()-tick)))
                                continue
                            waypoint_index = 0
                            chunk_id += 1
                            chunk_log.write(json.dumps({
                                "chunk_id": chunk_id, "start_action_index": chunk_start,
                                "end_action_index_exclusive": chunk_end,
                                "inferred_horizon": len(action_chunk),
                                "planned_horizon": len(execution_actions),
                                "dataset_actions": execution_actions.tolist(),
                                "optimized_actions": chunk_plan.tolist(),
                                "control_waypoints": chunk_info["waypoints"],
                                "waypoint_action_indices": chunk_info["waypoint_action_indices"],
                                "triggered_action_indices": chunk_info["triggered_action_indices"],
                                "modified_action_indices": chunk_info["modified_action_indices"],
                                "avoidance_controller": args.avoidance_controller,
                                "optimization_mode": chunk_info.get("mode"),
                                "optimization_space": chunk_info.get("optimization_space"),
                                "joint_variables": chunk_info.get("joint_variables"),
                                "ee_variables": chunk_info.get("ee_variables", 0),
                                "gripper_variables": chunk_info.get("gripper_variables", 0),
                                "ik_method": chunk_info.get("ik_method"),
                                "cartesian_objective_before": chunk_info.get("cartesian_objective_before"),
                                "cartesian_objective_after": chunk_info.get("cartesian_objective_after"),
                                "joint_qp_solve_calls": chunk_info.get("joint_qp_solve_calls"),
                                "solver": chunk_info.get("solver"),
                                "adaptive_stage": chunk_info.get("adaptive_stage"),
                                "velocity_source": chunk_info.get("velocity_source"),
                                "post_optimization_checks": chunk_info.get("post_optimization_checks"),
                                "avoidance_bypassed": chunk_info.get("avoidance_bypassed", False),
                                "passthrough_reason": (chunk_info.get("reason")
                                                       if chunk_info.get("avoidance_bypassed") else None),
                                "planning_seconds": chunk_info.get("planning_seconds"),
                                "guidance_fallback": chunk_info.get("guidance_fallback"),
                                "sweep_validation_enabled": chunk_info.get("sweep_validation_enabled", False),
                                "smoothness_weights": chunk_info.get("smoothness_weights"),
                                "seed_acceleration_squared": chunk_info.get("seed_acceleration_squared"),
                                "optimized_acceleration_squared": chunk_info.get("optimized_acceleration_squared"),
                                "target_repairs": chunk_info.get("target_repairs", [])})+"\n")
                            chunk_log.flush()
                        planned_action = np.asarray(chunk_info["waypoints"][waypoint_index])
                        optimized_action = chunk_plan[action_index-chunk_start]
                        desired_grip = planned_action[12:]
                        delta = np.clip(planned_action[:12]-q, -max_joint_step, max_joint_step)
                        info = chunk_info["waypoint_steps"][waypoint_index]
                    else:
                        planned_action = action
                        optimized_action = action
                        delta = np.clip(action[:12]-q, -max_joint_step, max_joint_step)
                        desired_grip, info = action[12:], None
                except SnapshotExpired:
                    if not args.live:
                        raise
                    observation = getattr(boxes, "observation", None)
                    after = observation.timestamp_ns if observation is not None else None
                    if commander:
                        q, grip, boxes, scene_snapshot = pause_for_fresh_scene(after)
                    else:
                        scene_snapshot = scene_reader.read(max_wait_s=3., after_timestamp_ns=after)
                        validate_calibration_identity(scene_snapshot, sampler.calibration_digest)
                        boxes = scene_reader.boxes(scene_snapshot)
                    previous_delta[:] = 0
                    chunk_plan = None
                    chunk_replans += 1
                    continue
                if commander and cbf is not None:
                    # Execute the solved QP targets with refreshed feedback.
                    # Avoid mode performs no swept-path validation, either
                    # during optimization or before sending commands.
                    q, grip = measure("robot_feedback", commander.read_state)
                    delta = np.clip(planned_action[:12]-q, -max_joint_step, max_joint_step)
                    error = check_tracking(q, action_index)
                if commander:
                    measure("joint_command", commander.send, q, delta)
                    sent = measure("gripper_command", gripper_commander.send_if_changed, desired_grip)
                    gripper_commands_sent += sent
                else:
                    q += delta
                    grip = desired_grip.copy()
                target_error = float(np.max(np.abs(q-optimized_action[:12])))
                waypoint_error = float(np.max(np.abs(q-planned_action[:12])))
                original_target_error = float(np.max(np.abs(q-action[:12])))
                gripper_error = float(np.max(np.abs(grip-action[12:])))
                arrival_tolerance = min(args.arrival_tolerance, args.final_tolerance) if action_index == len(policy.actions)-1 else args.arrival_tolerance
                # A Robotiq SET POS 255 request may settle below 255 (for
                # example, after contacting an object). Dataset actions are
                # commands, not measured waypoints for the grippers. Their
                # live feedback still drives every FK/CBF check above.
                waypoint_reached = waypoint_error <= min(arrival_tolerance, avoid_waypoint_tolerance)
                last_waypoint_for_action = (cbf is None or waypoint_index+1 == len(chunk_info["waypoints"])
                                           or chunk_info["waypoint_action_indices"][waypoint_index+1]
                                           != action_index-chunk_start)
                advanced = target_error <= arrival_tolerance and last_waypoint_for_action and (cbf is None or waypoint_reached)
                next_tracking_error = (float(np.max(np.abs(q-policy.states[action_index+1, :12])))
                                       if action_index+1 < len(policy.states) else None)
                if cbf is not None and adaptive is None and next_tracking_error is not None:
                    advanced = advanced and next_tracking_error <= args.max_tracking_error
                row = {"phase": "path", "step": action_index, "attempt": action_attempt,
                       "action_advanced": advanced, "target_error_rad": target_error,
                       "gripper_error": gripper_error,
                       "source_frame": int(policy.source_frames[action_index]),
                       "qpos": q.tolist(), "gripper_position": grip.tolist(),
                       "action": action.tolist(), "optimized_action": optimized_action.tolist(),
                       "control_target": planned_action.tolist(),
                       "original_target_error_rad": original_target_error,
                       "chunk_id": chunk_id if cbf is not None else None,
                       "chunk_step": action_index-chunk_start if cbf is not None else None,
                       "sent_joint_delta": delta.tolist(),
                       "gripper_target": desired_grip.tolist(), "cbf": info,
                       "tracking_error_rad": error, "next_tracking_error_rad": next_tracking_error,
                       "cycle_compute_s": time.monotonic()-tick}
                records.append(row)
                log.write(json.dumps(row)+"\n")
                previous_delta = delta.copy()
                if adaptive is not None:
                    # A detour waypoint is not the dataset goal. Rebuild the
                    # controller from feedback until that original goal is reached.
                    chunk_plan = None
                if cbf is not None:
                    cbf.detour_direction = np.asarray(info.get("detour_direction", np.zeros(12)), float)
                if cbf is not None and waypoint_reached and (not last_waypoint_for_action or advanced):
                    waypoint_index += 1
                if advanced:
                    final_optimized_target = optimized_action[:12].copy()
                    action_index += 1
                    action_attempt = 0
                    action_started_at = active_time()
                else:
                    if adaptive is None:
                        hold_ticks += 1
                    else:
                        detour_control_ticks += 1
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
                measure("stop", commander.stop)
                q, grip = measure("robot_feedback", commander.read_state)
            final_error = float(np.max(np.abs(q-policy.metadata["final_joint_state"])))
            final_optimized_error = float(np.max(np.abs(q-final_optimized_target)))
            completion_error = final_optimized_error if cbf is not None else final_error
            status = "completed" if completion_error <= args.final_tolerance else "final_joint_error"
    except BaseException as exc:
        status = f"error: {type(exc).__name__}: {exc}"
        if commander and gripper_commander:
            try:
                _, measured_grip = measure("robot_feedback", commander.read_state)
                measure("gripper_command", gripper_commander.send_if_changed, measured_grip)
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
            scene_reader.close()
            report = {"status": status, "completed": status == "completed", "execution": "physical" if args.execute else "virtual",
                      "inference_source": "lerobot_dataset_actions", "mode": args.mode,
                      "avoidance_controller": args.avoidance_controller if cbf is not None else None,
                      "optimization_space": args.optimization_space if cbf is not None else None,
                      "gripper_optimization": False,
                      "action_horizon": args.action_horizon,
                      "obstacle_age_policy": args.obstacle_age_policy if args.mode == "avoid" else "strict",
                      "stale_snapshot_reads": scene_reader.stale_snapshot_reads,
                      "max_snapshot_age_s": scene_reader.max_snapshot_age_s,
                      "execute_horizon": args.execute_horizon,
                      "effective_execute_horizon": (1 if adaptive is not None else min(args.action_horizon, args.execute_horizon)),
                      "cbf_chunk_mode": ("ellipsoid_mcbf_feedback" if adaptive is not None else (("cartesian_execution_prefix" if args.optimization_space == "ee"
                                          else "joint_execution_prefix") if cbf is not None else None)),
                      "progress_file": "planning.jsonl" if cbf is not None else None,
                      "chunk_plans": locals().get("chunk_id", -1)+1,
                      "chunk_replans": locals().get("chunk_replans", 0),
                      "optimization_retries": locals().get("optimization_retries", 0),
                      "optimization_retry_policy": ("penalized_distance_slack" if adaptive is not None
                                                    else "until_actions_exhausted" if cbf is not None else None),
                      "controller_variant": "ellipsoid_mcbf" if adaptive is not None else args.avoidance_controller if cbf is not None else None,
                      "adaptive_mcbf_config": vars(adaptive.config) if adaptive is not None else None,
                      "post_optimization_checks": True if adaptive is not None else None,
                      "constraint_checks": cbf is not None,
                      "action_optimization": cbf is not None,
                      "post_optimization_collision_checks": False,
                      "optimization_sweep_checks": False,
                      "last_cbf_failure": last_cbf_failure,
                      "last_tracking_excursion": last_tracking_excursion,
                      "restore_controller": "interpolation",
                      "restore_collision_checks": False,
                      "manual_selection": str(manual_filter.path) if manual_filter else None,
                      "manual_annotations": (str(manual_filter.path)
                                             if isinstance(manual_filter, ManualAnnotationFilter) else None),
                      "obstacle_source": "embedded_voxblox" if embedded else "external" if args.live else "disabled",
                      "replay_voxblox": ((scene_snapshot or {}).get("diagnostics", {}).get("replay_voxblox")
                                         if embedded else None),
                      "manual_selection_diagnostics": ((scene_snapshot or {}).get("diagnostics", {}).get(
                                                           "manual_selection")
                                                       or getattr(manual_filter, "last_diagnostics", None)
                                                       if manual_filter else None),
                      "dataset": str(Path(dataset_path).resolve()), "source": policy.metadata,
                      "steps": len(records), "actions_executed": sum(r.get("action_advanced", False) for r in records),
                      "path_control_ticks": sum(r["phase"] == "path" for r in records),
                      "hold_ticks": locals().get("hold_ticks", 0),
                      "detour_control_ticks": locals().get("detour_control_ticks", 0),
                      "unstable_scene_reads": scene_reader.unstable_scene_reads,
                      "stale_scene_pauses": stale_scene_pauses,
                      "runtime_timings": runtime_timings,
                      "snapshot_timings": scene_reader.timings,
                      "snapshot_cache": ({"reads": scene_reader.cache.reads,
                                          "refreshes": scene_reader.cache.refreshes,
                                          "refresh_errors": scene_reader.cache.refresh_errors,
                                          "last_refresh_error": scene_reader.cache.last_refresh_error,
                                          "poll_interval_s": scene_reader.cache.poll_interval_s}
                                         if scene_reader.cache is not None else None),
                      "cbf_runtime": adaptive.runtime_stats if adaptive is not None else getattr(cbf, "runtime_stats", None),
                      "planning_fk_cache": ({"hits": planning_sampler.hits, "misses": planning_sampler.misses}
                                            if planning_sampler is not None else None),
                      "watchdog_stops": getattr(commander, "watchdog_stops", 0),
                      "gripper_commands_sent": gripper_commands_sent.tolist(),
                      "gripper_speed": gripper_speed,
                      "joint_speed_limit_rad_s": args.speed,
                      "joint_acceleration_limit_rad_s2": args.acceleration,
                      "restore_joint_speed_limit_rad_s": args.restore_speed,
                      "restore_gripper_speed_limit_fraction_s": args.gripper_speed_fraction_s,
                      "gripper_force": args.gripper_force,
                      "gripper_hold_on_error": gripper_hold_on_error,
                      "failure_context": failure_context,
                      "mean_cycle_compute_s": (float(np.mean([r["cycle_compute_s"] for r in records])) if records else None),
                      "max_cycle_compute_s": (max(r["cycle_compute_s"] for r in records) if records else None),
                      "restore_elapsed_s": locals().get("restore_elapsed", time.monotonic()-restore_start),
                      "restore_replans": restore_replans,
                      "restore_cbf_interventions": 0,
                      "restore_cbf_failures": 0,
                      "restore_cbf_hold_ticks": 0,
                      "path_elapsed_s": locals().get("path_elapsed", time.monotonic()-path_start if "path_start" in locals() else None),
                      "final_joint_error_rad": locals().get("final_error"),
                      "final_optimized_joint_error_rad": locals().get("final_optimized_error"),
                      "max_tracking_error_rad": locals().get("max_tracking_error"),
                      "tracking_error_limit_rad": args.max_tracking_error if cbf is not None else None,
                      "tracking_error_policy": "diagnostic_only" if adaptive is not None else "tracking_envelope",
                      "planning_tracking_room_rad": planning_tracking_room if cbf is not None and adaptive is None else None,
                      "tracking_feedback_margin_rad": tracking_feedback_margin if cbf is not None else None,
                      "waypoint_tolerance_rad": avoid_waypoint_tolerance if cbf is not None else None,
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
    replay.add_argument("--output", type=Path,
                        default=Path("outputs/obstacle_validation") / f"annotation_replay_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f%z}",
                        help="Run output directory (default: outputs/obstacle_validation/annotation_replay_<timestamp>)")
    replay.add_argument("--mode", choices=("baseline", "avoid"), default="baseline",
                        help="baseline: direct recorded actions, no constraints or optimization; avoid: selected-voxel CBF/MCBF")
    replay.add_argument("--avoidance-controller", choices=("cbf", "mcbf"), default="mcbf",
                        help="Avoid mode: mcbf uses whole-body ellipsoid Adaptive MCBF-QP with stage selection, geodesics "
                             "and one-step feedback toward the original recorded target")
    replay.add_argument("--optimization-space", choices=("joint", "ee"), default="joint",
                        help="Avoid mode: ee optimizes active-TCP position/orientation increments via local IK; "
                             "joint retains the existing optimizer. Gripper commands are never optimized")
    replay.add_argument("--live", action="store_true",
                        help="Compatibility flag: avoid always uses live detection; baseline never uses it")
    replay.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                        help="Execute on hardware (default); --no-execute uses virtual replay")
    replay.add_argument("--execute-ack", default=EXECUTE_ACK,
                        help="Compatibility option; the execution acknowledgement is supplied by default")
    replay.add_argument("--initial-state", type=Path)
    replay.add_argument("--obstacle-url")
    replay.add_argument("--obstacle-source", choices=("auto", "external"), default="auto",
                        help="Auto starts selected-region Voxblox in avoid mode; explicit URL uses external detection")
    replay.add_argument("--voxblox-startup-timeout", type=float, default=30.,
                        help="Embedded camera/map startup deadline in seconds (default: 30)")
    replay.add_argument("--robot-mask", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to RGB-D robot masking; disabled by default for lower latency")
    manual = replay.add_mutually_exclusive_group()
    manual.add_argument("--manual-annotations", "--annotations", type=Path,
                        help="Avoid mode: annotations.json (default: newest capture directory under outputs/manual_obstacles); "
                             "filter current voxels by matching front/side RGB boxes")
    manual.add_argument("--manual-selection", type=Path,
                        help="Compatibility: selection.json for saved cell scope, or annotations.json for RGB box filtering")
    replay.add_argument("--fps", type=int, default=20)
    replay.add_argument("--time-scale", type=float, default=1.)
    replay.add_argument("--start-frame", type=int, default=0)
    replay.add_argument("--end-frame", type=int)
    replay.add_argument("--points-per-link", type=int, default=16)
    replay.add_argument("--action-horizon", type=int, default=32,
                        help="Dataset/VLA prediction chunk length; only its execution prefix is optimized")
    replay.add_argument("--execute-horizon", type=int, default=10,
                        help="Jointly optimize and execute this many actions from each prediction (default: 10)")
    replay.add_argument("--obstacle-age-policy", choices=("latest", "strict"), default="latest",
                        help="Avoid mode: latest reuses cached occupancy without age expiry or refresh waits; strict enforces expiry")
    replay.add_argument("--speed", type=float,
                        help="Optional dataset replay joint speed limit in rad/s; default: no software limit")
    replay.add_argument("--acceleration", type=float,
                        help="Optional CBF joint acceleration limit in rad/s²; default: no software limit")
    replay.add_argument("--restore-speed", type=float, default=.5,
                        help="Restoration joint interpolation speed in rad/s (default: 0.5); independent of replay speed")
    replay.add_argument("--gripper-speed-fraction-s", type=float, default=.5,
                        help="Restoration gripper interpolation rate in closing fraction/s (default: 0.5)")
    replay.add_argument("--gripper-speed", type=int, default=255,
                        help="Robotiq SPE register [0,255]; default: full hardware speed (255) in both modes")
    replay.add_argument("--gripper-force", type=int, default=25)
    replay.add_argument("--restore-tolerance", type=float, default=.015)
    replay.add_argument("--gripper-tolerance", type=float, default=.06)
    replay.add_argument("--restore-settle-timeout", type=float, default=10.)
    replay.add_argument("--restore-timeout", type=float, default=90.,
                        help="Maximum total time for initial-state interpolation and settling, seconds")
    replay.add_argument("--restore-planning-timeout", type=float, help=argparse.SUPPRESS)
    replay.add_argument("--restore-joint-room", "--restore-detour-room", dest="restore_joint_room",
                        type=float, default=.35,
                        help=argparse.SUPPRESS)
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
