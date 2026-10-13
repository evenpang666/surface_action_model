#!/usr/bin/env python3
"""Run the canonical dual-UR7e PI05 + obstacle OBB + CBF-QP loop.

The obstacle service is the sole owner of the front D455 and RTDE receive
streams. This executor consumes its timestamped snapshot, queries the jointly
trained PI05 action/point-flow service, and projects the two-arm joint delta
through one coupled 12-DOF CBF problem. It is dry-run by default.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT = REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler  # noqa: E402
from real_scripts.real_cbf_qp import (  # noqa: E402
    OrientedBox, boxes_from_snapshot, point_jacobian_fd, project_joint_delta_qp,
    select_inter_arm_constraints, select_point_flow_constraints, validate_fk_rollout,
)
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path  # noqa: E402
from safety_module.runtime_contract import validate_prediction, validate_snapshot
from safety_module.measured_surface import MeasuredSurfaceSampler, associate_surface
from safety_module.command_watchdog import CommandWatchdog
from safety_module.dual_camera_calibration import validate_calibration_identity


EXECUTE_ACK = "I_UNDERSTAND_DUAL_UR7E_WILL_MOVE"
LEGACY_LEFT_ACTION_ONLY_ACK = "I_UNDERSTAND_LEGACY_LEFT_POLICY_HAS_NO_VALID_CBF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--obstacle-url", default=None)
    parser.add_argument("--points-per-link", type=int, default=None)
    parser.add_argument("--poll-hz", type=float, default=None, help="Default is the checkpoint action period")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--dry-run-stale-timeout-s", type=float, default=None,
        help="Optional relaxed snapshot-age limit for non-executing diagnostics only; --execute always uses config safety.stale_timeout_s.",
    )
    parser.add_argument(
        "--allow-legacy-left-only-dry-run",
        action="store_true",
        help="Validation only: adapt a legacy 6-state/7-action/8-points-per-link left-only checkpoint. "
        "It pads the right action with zeros, omits right-arm CBF constraints, and is incompatible with --execute.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-grippers", action="store_true", help="With --execute, command both 2F-85 grippers only after the surface rollout check")
    parser.add_argument(
        "--execute-left-arm-only", action="store_true",
        help="With --execute, open an RTDE command channel only to the left UR7e and force all right-arm joint deltas to zero.",
    )
    parser.add_argument(
        "--execute-legacy-left-action-only", action="store_true",
        help="One-step, left-UR-only legacy-policy test without obstacle/inter-arm CBF. Requires the separate legacy acknowledgement.",
    )
    parser.add_argument(
        "--execute-left-gripper",
        action="store_true",
        help="With --execute, send the PI05 left 2F-85 target only when no point-flow CBF constraint is active. Right 2F-85 is never commanded.",
    )
    parser.add_argument("--left-gripper-speed", type=int, default=40)
    parser.add_argument("--left-gripper-force", type=int, default=25)
    parser.add_argument("--video-output", type=Path, default=None, help="Optional front-RGB MP4 with current and predicted robot surface points.")
    parser.add_argument("--video-fps", type=float, default=None, help="Defaults to --poll-hz when --video-output is used.")
    parser.add_argument(
        "--front-calibration",
        type=Path,
        default=REPO_ROOT.parent / "quest3_collect" / "config" / "calibration" / "left_base_to_front_camera.yaml",
        help="^left_baseT_front-camera calibration used by --video-output.",
    )
    parser.add_argument("--allow-unverified-right-tcp", action="store_true", help="Allow provisional 2F-85 FK in dry-run only.")
    parser.add_argument(
        "--execute-ack", default="",
        help=f"With --execute, must equal {EXECUTE_ACK!r}; legacy action-only mode requires {LEGACY_LEFT_ACTION_ONLY_ACK!r}.",
    )
    return parser.parse_args()


def _snapshot(url: str, timeout_s: float) -> dict:
    request = Request(url, headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout_s) as response:
        value = json.loads(response.read())
    if value.get("schema") != "dual_ur7e_obstacle_snapshot_v2" or value.get("coordinate_frame") != "left_base":
        raise ValueError("Obstacle endpoint does not expose the v2 left_base safety snapshot")
    validate_snapshot(value, timeout_s)
    return value


def _rgb(snapshot: dict) -> np.ndarray:
    payload = base64.b64decode(snapshot["rgb_front_jpeg_base64"], validate=True)
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Obstacle snapshot contains an invalid front RGB JPEG")
    return bgr[..., ::-1]


def _boxes(snapshot: dict) -> list[OrientedBox]:
    return boxes_from_snapshot(snapshot)


def _qpos(snapshot: dict) -> np.ndarray:
    q = np.asarray((*snapshot["qpos"]["left"], *snapshot["qpos"]["right"]), dtype=np.float32)
    if q.shape != (12,) or not np.isfinite(q).all():
        raise ValueError("Obstacle snapshot qpos is not a finite dual-arm vector")
    return q


def _gripper(snapshot: dict) -> np.ndarray:
    value = snapshot["gripper_position"]
    result = np.asarray((value["left"], value["right"]), dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError("Both measured gripper closing fractions must be in [0,1]")
    return result


def _joint_delta(action: np.ndarray) -> np.ndarray:
    if action.shape != (14,):
        raise ValueError(f"Dual-arm executor requires one 14-D action, got {action.shape}")
    return np.concatenate((action[:6], action[7:13])).astype(np.float32)


def _validate_policy_metadata(
    metadata: object, sampler: DualUR7eSurfacePointSampler, *, points_per_link: int, allow_legacy_left_only_dry_run: bool
) -> bool:
    """Fail before a live snapshot/request when a checkpoint is not deployable here."""
    if not isinstance(metadata, dict):
        raise RuntimeError("Policy server did not provide checkpoint metadata during the WebSocket handshake")
    joint_dim = metadata.get("joint_dim")
    action_dim = metadata.get("logical_action_dim")
    legacy_left_only = joint_dim == 6 and action_dim == 7 and metadata.get("points_per_link") == 8
    if legacy_left_only and allow_legacy_left_only_dry_run:
        return True
    errors: list[str] = []
    if joint_dim != 12:
        errors.append(f"joint_dim={joint_dim!r} (dual UR7e executor requires 12)")
    if action_dim != 14:
        errors.append(f"logical_action_dim={action_dim!r} (dual UR7e executor requires 14: 6+gripper per arm)")
    if metadata.get("points_per_link") != points_per_link:
        errors.append(
            f"points_per_link={metadata.get('points_per_link')!r} "
            f"(executor is configured for {points_per_link})"
        )
    surface_hash = str(metadata.get("surface_model_hash", ""))
    if surface_hash not in (sampler.model_hash(1), sampler.model_hash(2)):
        errors.append("surface_model_hash does not match the current UR7e + 2F-85/2F-85 mesh sampler")
    if errors:
        raise RuntimeError(
            "Incompatible policy checkpoint on the inference server: " + "; ".join(errors) + ". "
            "Do not use a legacy 6-DOF/7-action/8-points-per-link checkpoint for this dual-arm cell. "
            "Restart the server with a retrained 12-DOF, 14-action, 128-points-per-link checkpoint."
        )
    return False


def _recorded_intrinsics(calibration_path: Path, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Load calibration and reproduce the recorder's crop/resize transform."""
    payload = yaml.safe_load(calibration_path.expanduser().read_text(encoding="utf-8"))
    intr = payload["intrinsics"]
    source_width, source_height = int(intr["width"]), int(intr["height"])
    K = np.asarray(((intr["fx"], 0.0, intr["ppx"]), (0.0, intr["fy"], intr["ppy"]), (0.0, 0.0, 1.0)), dtype=np.float64)
    source_ratio, target_ratio = source_width / source_height, width / height
    if source_ratio > target_ratio:
        crop_width = source_height * target_ratio
        K[0, 2] -= (source_width - crop_width) * 0.5
        scale = width / crop_width
    else:
        crop_height = source_width / target_ratio
        K[1, 2] -= (source_height - crop_height) * 0.5
        scale = height / crop_height
    K[0] *= scale
    K[1] *= scale
    camera_to_left_base = np.asarray(payload["matrix"], dtype=np.float64)
    if camera_to_left_base.shape != (4, 4):
        raise ValueError(f"Invalid front calibration matrix: {calibration_path}")
    return K, camera_to_left_base


def _project_front(points: np.ndarray, K: np.ndarray, camera_to_left_base: np.ndarray, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera = (np.linalg.inv(camera_to_left_base) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    depth = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth > 1e-5)
    uv = np.empty((len(points), 2), dtype=np.int32)
    uv[:, 0] = np.rint(K[0, 0] * camera[:, 0] / np.maximum(depth, 1e-5) + K[0, 2]).astype(np.int32)
    uv[:, 1] = np.rint(K[1, 1] * camera[:, 1] / np.maximum(depth, 1e-5) + K[1, 2]).astype(np.int32)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid


def _draw_projected_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int, K: np.ndarray, camera_to_left_base: np.ndarray) -> None:
    uv, valid = _project_front(points, K, camera_to_left_base, width=image.shape[1], height=image.shape[0])
    for x, y in uv[valid]:
        cv2.circle(image, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _video_frame(rgb: np.ndarray, current: np.ndarray, prediction: np.ndarray, *, step: int, constraints: int, K: np.ndarray, camera_to_left_base: np.ndarray) -> np.ndarray:
    """RGB overlay. Gray=now; cyan=PI05's final-horizon surface prediction."""
    image = np.ascontiguousarray(np.asarray(rgb)[..., ::-1].copy())
    _draw_projected_points(image, current, (160, 160, 160), 2, K, camera_to_left_base)
    _draw_projected_points(image, prediction, (255, 230, 30), 3, K, camera_to_left_base)
    cv2.rectangle(image, (8, 8), (410, 62), (0, 0, 0), thickness=-1)
    cv2.putText(image, "gray=current robot surface", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(image, "cyan=PI05 final-horizon predicted surface", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"step={step}  active CBF constraints={constraints}", (12, image.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return image


class DualRTDECommander:
    """Explicitly armed, rate-limited dual servoJ writer; no gripper guessing."""

    def __init__(self, left_ip: str, right_ip: str, period_s: float) -> None:
        from rtde_control import RTDEControlInterface

        self.controls = (RTDEControlInterface(left_ip), RTDEControlInterface(right_ip))
        self.period_s = float(period_s)
        self.watchdog = CommandWatchdog(self.stop, max(.25, 2 * self.period_s))

    def stop(self):
        for control in self.controls:
            control.servoStop(1.)

    def send(self, qpos: np.ndarray, safe_delta: np.ndarray) -> None:
        targets = (qpos[:6] + safe_delta[:6], qpos[6:] + safe_delta[6:])

        def command(pair) -> bool:
            control, target = pair
            return bool(control.servoJ(target.tolist(), 0.10, 0.30, self.period_s, 0.10, 300))

        with self.watchdog.lock:
            self.watchdog.renew()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(command, zip(self.controls, targets, strict=True)))
        if not all(results):
            raise RuntimeError("At least one UR controller rejected servoJ")

    def close(self) -> None:
        self.watchdog.close()
        for control in self.controls:
            try:
                control.servoStop(1.0)
            finally:
                control.disconnect()


class LeftRTDECommander:
    """Explicit left-only servoJ writer; the right controller is never opened."""

    def __init__(self, left_ip: str, period_s: float) -> None:
        from rtde_control import RTDEControlInterface

        self.control = RTDEControlInterface(left_ip)
        self.period_s = float(period_s)
        self.watchdog = CommandWatchdog(lambda: self.control.servoStop(1.), max(.25, 2 * self.period_s))

    def send(self, qpos: np.ndarray, safe_delta: np.ndarray) -> None:
        target = qpos[:6] + safe_delta[:6]
        with self.watchdog.lock:
            self.watchdog.renew()
            if not bool(self.control.servoJ(target.tolist(), 0.10, 0.30, self.period_s, 0.10, 300)):
                raise RuntimeError("Left UR controller rejected servoJ")

    def close(self) -> None:
        self.watchdog.close()
        try:
            self.control.servoStop(1.0)
        finally:
            self.control.disconnect()


class LeftRobotiqCommander:
    """Explicit left 2F-85 writer, separate from the UR servoJ channels."""

    def __init__(self, host: str, *, speed: int, force: int) -> None:
        from real_scripts.ur7e_robotiq_d435i_collector.utils.robotiq_interface import RobotiqGripper

        self.gripper = RobotiqGripper(host, port=63352, timeout=2.0)
        self.gripper.connect()
        self.speed = int(np.clip(speed, 0, 255))
        self.force = int(np.clip(force, 0, 255))
        self.last_target: float | None = None

    def send_if_changed(self, closing_fraction: float) -> bool:
        target = float(np.clip(closing_fraction, 0.0, 1.0))
        if self.last_target is not None and abs(target - self.last_target) < 0.03:
            return False
        self.gripper.write_position(target, speed=self.speed, force=self.force)
        self.last_target = target
        return True

    def close(self) -> None:
        self.gripper.disconnect()


def run(args: argparse.Namespace) -> None:
    config, config_path = load_project_config(args.project_config)
    safety = config["safety"]
    if not safety.get("camera_calibration_valid", False):
        raise RuntimeError("Front/side extrinsics are invalid. Complete and validate a new dual-view calibration before live inference.")
    points_per_link = int(args.points_per_link or config["preprocess"]["points_per_link"])
    obstacle_url = args.obstacle_url or str(safety["obstacle_url"])
    snapshot_age_limit_s = float(args.dry_run_stale_timeout_s or safety["stale_timeout_s"])
    period_s = 1.0 / float(args.poll_hz or 5.)
    required_ack = LEGACY_LEFT_ACTION_ONLY_ACK if args.execute_legacy_left_action_only else EXECUTE_ACK
    if args.execute and args.execute_ack != required_ack:
        raise ValueError(f"Physical execution requires --execute-ack {required_ack}")
    if args.execute_left_gripper and not args.execute:
        raise ValueError("--execute-left-gripper requires --execute")
    if args.execute_grippers and not args.execute:
        raise ValueError("--execute-grippers requires --execute")
    if args.execute_left_arm_only and not args.execute:
        raise ValueError("--execute-left-arm-only requires --execute")
    if args.execute_legacy_left_action_only:
        raise ValueError("Legacy action-only execution is disabled in the Surface Action safety runtime; legacy checkpoints are offline replay only")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.dry_run_stale_timeout_s is not None and args.dry_run_stale_timeout_s <= 0:
        raise ValueError("--dry-run-stale-timeout-s must be positive")
    if args.execute and args.dry_run_stale_timeout_s is not None:
        raise ValueError("--dry-run-stale-timeout-s is not permitted with --execute")
    if args.execute and args.allow_legacy_left_only_dry_run and not args.execute_legacy_left_action_only:
        raise ValueError("--allow-legacy-left-only-dry-run is intentionally incompatible with --execute")
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.policy_host or safety.get("policy_host", "192.168.124.64"), port=args.policy_port, timeout_s=float(safety["stale_timeout_s"]))
    sampler = DualUR7eSurfacePointSampler(points_per_link=points_per_link, project_config=config_path)
    legacy_left_only = _validate_policy_metadata(
        policy.get_server_metadata(), sampler, points_per_link=points_per_link,
        allow_legacy_left_only_dry_run=args.allow_legacy_left_only_dry_run,
    )
    legacy_sampler = DualUR7eSurfacePointSampler(points_per_link=8, project_config=config_path) if legacy_left_only else None
    measured_surface = policy.get_server_metadata().get("point_target") == "visual"
    trained_period = policy.get_server_metadata().get("action_period_s")
    if trained_period is not None:
        if args.poll_hz is not None and not np.isclose(period_s, float(trained_period), rtol=.05):
            raise ValueError("Control period must match the checkpoint action period")
        period_s = float(trained_period)
    elif args.execute:
        raise ValueError("Checkpoint lacks an action period; rebuild data and retrain before execution")
    args.poll_hz = 1. / period_s
    if args.execute and policy.get_server_metadata().get("model_type") != "SurfaceActionModelPytorch":
        raise RuntimeError("Legacy regression checkpoints are replay-only; retrain SurfaceActionModelPytorch before execution")
    if legacy_left_only:
        mode = "legacy left action-only execute: obstacle/inter-arm CBF disabled" if args.execute_legacy_left_action_only else "legacy dry-run"
        print(f"[{mode}] Using legacy left-only 6-state/7-action checkpoint; right action is fixed to zero and right-arm CBF is disabled.")
    if not legacy_left_only and not sampler.tcp_transform_verified["right_arm"]:
        if args.execute or not args.allow_unverified_right_tcp:
            raise RuntimeError(
                "Right flange-to-active-TCP is not verified; physical execution is blocked. "
                "For geometry-only testing, pass --allow-unverified-right-tcp without --execute."
            )
    commander = None
    left_gripper_commander = None
    right_gripper_commander = None
    video_writer: cv2.VideoWriter | None = None
    video_K: np.ndarray | None = None
    video_camera_to_left_base: np.ndarray | None = None
    if args.execute:
        with resolve_repo_path(config["quest3"]["hardware_config"]).open(encoding="utf-8") as handle:
            import yaml
            hardware = yaml.safe_load(handle)
        if args.execute_left_arm_only:
            commander = LeftRTDECommander(str(hardware["arms"]["left_arm"]["robot_ip"]), period_s)
            print("[execute] left-only UR servoJ enabled; no RTDE control channel is opened to the right UR.")
        else:
            commander = DualRTDECommander(str(hardware["arms"]["left_arm"]["robot_ip"]), str(hardware["arms"]["right_arm"]["robot_ip"]), period_s)
        if args.execute_left_gripper or args.execute_grippers:
            left_gripper_commander = LeftRobotiqCommander(
                str(hardware["arms"]["left_arm"]["robot_ip"]), speed=args.left_gripper_speed, force=args.left_gripper_force
            )
            if args.execute_grippers and not args.execute_left_arm_only:
                right_gripper_commander = LeftRobotiqCommander(
                    str(hardware["arms"]["right_arm"]["robot_ip"]), speed=args.left_gripper_speed, force=args.left_gripper_force
                )
            print("[execute] left 2F-85 output enabled; it is held whenever a point-flow CBF constraint is active. Both grippers are held while CBF constraints are active.")
        else:
            print("[execute] arm servoJ enabled; both gripper outputs remain disabled")
    last_frame = -1
    try:
        for step in range(args.max_steps):
            began = time.monotonic()
            snap = _snapshot(obstacle_url, timeout_s=snapshot_age_limit_s)
            validate_calibration_identity(snap, sampler.calibration_digest)
            age_s = (time.monotonic_ns() - int(snap["host_timestamp_ns"])) / 1e9
            if age_s < -0.1 or age_s > snapshot_age_limit_s:
                raise RuntimeError(f"Obstacle snapshot is stale or from another clock domain: age={age_s:.3f}s")
            if int(snap["frame"]) == last_frame:
                time.sleep(max(0.0, period_s - (time.monotonic() - began)))
                continue
            last_frame = int(snap["frame"])
            qpos, grip = _qpos(snap), _gripper(snap)
            full_points = sampler.link_points(qpos, grip).reshape(-1, 3)
            rgb = _rgb(snap)
            # The old single-arm model was trained on a separate 8-points-per-link
            # left mesh.  Feed it exactly that historical contract; passing the
            # 12-DOF/current 128-points-per-link representation would silently
            # scramble its point identities.
            legacy_full_points = None
            if legacy_left_only:
                assert legacy_sampler is not None
                legacy_full_points = legacy_sampler.link_points(qpos, grip).reshape(-1, 3)[: len(legacy_sampler.left_link_names) * 8]
                request_qpos, request_points = qpos[:6], legacy_full_points
            else:
                request_qpos, request_points = qpos, full_points
            if measured_surface:
                request_points = np.asarray(snap["current_surface"], dtype=np.float32).reshape(-1, 3)
                _, valid = associate_surface(request_points, sampler.link_points(qpos, grip))
                request_points = request_points[valid]
            result = policy.infer({"prompt": args.prompt, "rgb_front": rgb, "qpos": request_qpos,
                                  "robot_points": request_points, "current_surface": request_points,
                                  "surface_source": snap.get("surface_source")})
            validate_snapshot(snap, snapshot_age_limit_s)
            actions = np.asarray(result["actions"], dtype=np.float32)
            prediction = np.asarray(result["predicted_robot_points"], dtype=np.float32)
            selected = np.asarray(result["selected_point_indices"])
            validate_prediction(actions, prediction, selected, action_dim=7 if legacy_left_only else 14,
                                point_count=len(request_points), horizon=int(policy.get_server_metadata()["action_horizon"]))
            checkpoint_hash = str(result.get("surface_model_hash", ""))
            left_only_flow = legacy_left_only or checkpoint_hash == sampler.model_hash(1)
            if not legacy_left_only and checkpoint_hash not in (sampler.model_hash(1), sampler.model_hash(2)):
                raise ValueError("Policy checkpoint surface mesh/hash differs from the live UR7e tool sampler")
            if not legacy_left_only and int(result.get("points_per_link", 0)) != points_per_link:
                raise ValueError("Policy checkpoint points_per_link differs from the executor")
            expected_action_dim = 7 if legacy_left_only else 14
            if actions.ndim != 2 or actions.shape[1] != expected_action_dim or prediction.ndim != 3 or prediction.shape[1:] != (len(selected), 3):
                raise ValueError(f"Policy contract mismatch: actions={actions.shape}, points={prediction.shape}")
            if legacy_left_only:
                assert legacy_full_points is not None
                current = legacy_full_points[selected]
                padded_actions = np.zeros((actions.shape[0], 14), dtype=np.float32)
                padded_actions[:, :7] = actions
                actions = padded_actions
            else:
                current = request_points[selected]
            # Left-only Quest3 episodes retain a 14-D action interface by
            # padding the inactive right arm, but supervise only left surface
            # flow.  Add a stationary right-arm geometry subset for safety;
            # its joint bounds below are fixed at zero, so this deployment can
            # never move the unmodelled right arm.
            safety_current, safety_prediction, safety_indices = current, prediction, selected
            left_full_count = len(sampler.left_link_names) * points_per_link
            if left_only_flow and measured_surface:
                raise ValueError("Live measured-surface control requires a genuinely dual-arm training checkpoint")
            if left_only_flow and not legacy_left_only:
                right_indices = np.linspace(left_full_count, len(full_points) - 1, min(len(selected), len(full_points) - left_full_count), dtype=np.int64)
                right_current = full_points[right_indices]
                safety_current = np.concatenate((current, right_current), axis=0)
                safety_prediction = np.concatenate((prediction, np.broadcast_to(right_current, (prediction.shape[0], *right_current.shape))), axis=1)
                safety_indices = np.concatenate((selected, right_indices))
            boxes = _boxes(snap)
            geometry_sampler = legacy_sampler if legacy_left_only else sampler
            left_mask = safety_indices < left_full_count
            if measured_surface:
                geometry_sampler = MeasuredSurfaceSampler(sampler, qpos, grip, safety_current)
                left_mask = geometry_sampler.link_ids < len(sampler.left_link_names)
                safety_indices = np.arange(len(safety_current))
            obstacle_constraints = [] if args.execute_legacy_left_action_only else select_point_flow_constraints(safety_current, safety_prediction, boxes, collision_margin_m=float(safety["collision_margin_m"]), trigger_margin_m=float(safety["trigger_margin_m"]))
            arm_constraints = [] if legacy_left_only or args.execute_legacy_left_action_only else select_inter_arm_constraints(safety_current, safety_prediction, left_mask, minimum_distance_m=float(safety["inter_arm_margin_m"]), trigger_margin_m=float(safety["trigger_margin_m"]))
            constraints = [*obstacle_constraints, *arm_constraints]
            nominal = _joint_delta(actions[0])
            if left_only_flow or args.execute_left_arm_only:
                nominal[6:] = 0.0
            limit = float(safety["max_joint_step_rad"])
            gripper_chunk = np.broadcast_to(grip, (len(actions), 2)).copy()
            if not constraints:
                if left_gripper_commander is not None: gripper_chunk[:, 0] = np.clip(actions[:, 6], 0, 1)
                if right_gripper_commander is not None: gripper_chunk[:, 1] = np.clip(actions[:, 13], 0, 1)
            if constraints:
                jacobian = point_jacobian_fd(geometry_sampler, qpos, gripper_position=grip, point_indices=safety_indices)
                lower, upper = np.full(12, -limit), np.full(12, limit)
                if left_only_flow or args.execute_left_arm_only:
                    lower[6:] = upper[6:] = 0.0
                safe_delta, info = project_joint_delta_qp(nominal, jacobian, constraints, lower_delta=lower, upper_delta=upper, alpha=float(safety["cbf_alpha"]), iterations=int(safety["cbf_iterations"]))
            else:
                safe_delta = np.clip(nominal, -limit, limit)
                info = {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
            if not bool(info["success"]):
                safe_delta = np.zeros(12, dtype=np.float32)
                corrected_chunk = np.zeros((len(actions), 12), dtype=np.float32)
            elif not args.execute_legacy_left_action_only:
                corrected_chunk = np.stack([_joint_delta(action) for action in actions])
                corrected_chunk[0] = safe_delta
                if left_only_flow or args.execute_left_arm_only:
                    corrected_chunk[:, 6:] = 0
                if not validate_fk_rollout(geometry_sampler, qpos, grip, corrected_chunk, boxes,
                                          point_indices=safety_indices, left_mask=left_mask,
                                          gripper_chunk=gripper_chunk,
                                          collision_margin_m=float(safety["collision_margin_m"]),
                                          inter_arm_margin_m=float(safety["inter_arm_margin_m"])):
                    safe_delta = np.zeros(12, dtype=np.float32)
                    corrected_chunk = np.zeros((len(actions), 12), dtype=np.float32)
                    info.update(success=False, reason="corrected_fk_rollout_collision")
            if args.video_output is not None:
                if video_writer is None:
                    K, camera_to_left_base = _recorded_intrinsics(args.front_calibration, width=rgb.shape[1], height=rgb.shape[0])
                    args.video_output.parent.mkdir(parents=True, exist_ok=True)
                    fps = float(args.video_fps or args.poll_hz)
                    video_writer = cv2.VideoWriter(str(args.video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rgb.shape[1], rgb.shape[0]))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Could not open MP4 writer: {args.video_output}")
                    video_K, video_camera_to_left_base = K, camera_to_left_base
                assert video_K is not None and video_camera_to_left_base is not None
                video_writer.write(_video_frame(rgb, safety_current, safety_prediction[-1], step=step, constraints=len(constraints), K=video_K, camera_to_left_base=video_camera_to_left_base))
            if commander is not None:
                # Include network inference, geometry and QP time in the deadline.
                validate_snapshot(snap, snapshot_age_limit_s)
                fresh = _snapshot(obstacle_url, timeout_s=snapshot_age_limit_s)
                validate_calibration_identity(fresh, sampler.calibration_digest)
                if np.max(np.abs(_qpos(fresh) - qpos)) > float(safety.get("max_state_drift_rad", 0.01)):
                    raise RuntimeError("Robot state changed during policy inference; discarding action")
                if not np.allclose(_gripper(fresh), grip, atol=.02):
                    raise RuntimeError("Gripper state changed during inference; replan before moving")
                if not validate_fk_rollout(geometry_sampler, _qpos(fresh), _gripper(fresh), corrected_chunk, _boxes(fresh),
                                          point_indices=safety_indices, left_mask=left_mask,
                                          gripper_chunk=gripper_chunk,
                                          collision_margin_m=float(safety["collision_margin_m"]),
                                          inter_arm_margin_m=float(safety["inter_arm_margin_m"])):
                    raise RuntimeError("Corrected chunk is unsafe against the latest obstacle snapshot")
                validate_snapshot(fresh, snapshot_age_limit_s)
                commander.send(_qpos(fresh), safe_delta)
            left_gripper_sent = False
            if left_gripper_commander is not None and not constraints and bool(info["success"]):
                # The present QP differentiates only 12 UR joints.  Holding
                # the gripper whenever any future point enters a safety active
                # set prevents a finger-only closure from bypassing that QP.
                left_gripper_sent = left_gripper_commander.send_if_changed(float(actions[0, 6]))
            if right_gripper_commander is not None and not constraints and bool(info["success"]):
                right_gripper_commander.send_if_changed(float(actions[0, 13]))
            print(json.dumps({
                "step": step, "source_frame": last_frame, "snapshot_age_s": round(age_s, 4), "obbs": len(boxes),
                "obstacle_constraints": len(obstacle_constraints), "inter_arm_constraints": len(arm_constraints), "cbf": info,
                "nominal_joint_delta": nominal.tolist(), "safe_joint_delta": safe_delta.tolist(),
                "left_gripper_target": float(actions[0, 6]), "right_gripper_target": float(actions[0, 13]),
                "left_gripper_command_sent": left_gripper_sent,
                "left_only_surface_flow": left_only_flow,
                "mode": "execute_left_arm_and_left_gripper" if left_gripper_commander is not None and args.execute_left_arm_only else ("execute_arms_and_left_gripper" if left_gripper_commander is not None else ("execute_left_arm_only" if commander is not None and args.execute_left_arm_only else ("execute_arms_only" if commander is not None else "dry_run"))),
            }, ensure_ascii=False))
            if args.once:
                break
            time.sleep(max(0.0, period_s - (time.monotonic() - began)))
    finally:
        if commander is not None:
            commander.close()
        if left_gripper_commander is not None:
            left_gripper_commander.close()
        if right_gripper_commander is not None:
            right_gripper_commander.close()
        if video_writer is not None:
            video_writer.release()
            print(f"[video] saved predicted-point overlay: {args.video_output}")


if __name__ == "__main__":
    run(parse_args())
