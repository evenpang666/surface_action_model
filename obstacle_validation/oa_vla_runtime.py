"""Local OA-VLA feedback loop: shared RGB-D, paired voxel scope, MCBF, UR outputs."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from obstacle_validation.oa_vla_actions import (ObservationHold, validate_server_metadata, first_action_target,
                                               validate_age_and_drift, check_optimized_step)
from obstacle_validation.oa_vla_perception import (DynamicOBBFilter, WorkspaceTSDF, mapping_loop,
                                                  inconsistent_pairs_allow_empty_geometry)
from obstacle_validation.shared_frames import Workers, feedback_loop, camera_loop
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path


@dataclass
class PredictionTicket:
    mapped: object
    prediction: dict | None
    snapshot: dict | None
    hold_reason: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    resume_after_ns: int = 0


def inference_loop(workers, mapped_frames, results, client, prompt, args):
    version, resume_after_ns = 0, 0
    while not workers.stop.is_set():
        try:
            version, mapped = mapped_frames.get(after=version, timeout=.25)
        except TimeoutError:
            continue
        bundle = mapped.bundle
        if bundle.capture_ns <= resume_after_ns or (time.monotonic_ns() - bundle.capture_ns) / 1e9 > args.max_result_age:
            continue
        rgb = {name: frame.rgb for name, frame in bundle.frames.items()}
        prediction = client.infer(rgb, bundle.feedback.q, prompt, context=bundle.context(), encoding=args.encoding)
        try:
            mask = DynamicOBBFilter(mapped, prediction, padding_pixels=args.obb_padding_pixels,
                                    minimum_score=args.minimum_object_score)
            snapshot = mask(mapped.snapshot)
            ticket = PredictionTicket(mapped, prediction, snapshot)
        except ObservationHold as exc:
            ticket = PredictionTicket(mapped, prediction, None, str(exc))
        results.publish(ticket)
        # No action backlog. A new inference must observe feedback/cameras AFTER
        # the previous bounded control step (or explicit hold) has completed.
        while not workers.stop.is_set() and not ticket.done.wait(.05):
            pass
        resume_after_ns = ticket.resume_after_ns


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


class LiveRuntime:
    """Dependency-injectable coordinator; background workers never send motion."""

    def __init__(self, args, client, driver, source, mapping_factory, planner, *, calibration_id):
        self.args, self.client, self.driver, self.source = args, client, driver, source
        self.mapping_factory, self.planner, self.calibration_id = mapping_factory, planner, calibration_id
        self.policy_period = validate_server_metadata(client.metadata)
        self.workers = Workers()
        self.feedback, self.frames, self.mapped, self.results = (self.workers.channel() for _ in range(4))
        self.lower = -np.tile([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi], 2)
        self.upper = -self.lower
        self.previous_delta = np.zeros(12)
        self.camera_info = {}
        self.report = dict(completed=False, execution="physical" if args.execute else "read_only_dry_run",
                           perception="shared_four_view_capture_full_workspace_voxblox",
                           controller="ellipsoid_mcbf", execution_prefix=1, restore_initial_pose=False,
                           observations=0, commands_sent=0, holds=0, same_arm_avoidance=False,
                           sweep_validation_enabled=False, calibration_id=calibration_id)

    def _read_feedback(self, timeout=.25):
        self.workers.check()
        return self.feedback.get(timeout=timeout)[1]

    def _check_live(self):
        self.workers.check()
        self.feedback.check()
        self.frames.check()
        self.mapped.check()
        state = self._read_feedback()
        if (time.monotonic_ns() - state.timestamp_ns) / 1e9 > self.args.max_feedback_age:
            raise RuntimeError("Live robot feedback stopped updating")

    def step(self, ticket):
        args = self.args
        if ticket.hold_reason:
            raise ObservationHold(ticket.hold_reason)
        bundle = ticket.mapped.bundle
        if bundle.calibration_id != self.calibration_id:
            raise ValueError("Camera packet calibration changed")
        state = self._read_feedback()
        validate_age_and_drift(bundle, state, max_age_s=args.max_result_age, max_drift_rad=args.max_state_drift,
                               max_feedback_age_s=args.max_feedback_age)
        from obstacle_validation.runtime import boxes_from_json, validate_obstacle_snapshot
        from safety_module.dual_camera_calibration import validate_calibration_identity
        validate_calibration_identity(ticket.snapshot, self.calibration_id)
        validate_obstacle_snapshot(ticket.snapshot, args.max_result_age)
        boxes = boxes_from_json(ticket.snapshot)
        if not boxes and not inconsistent_pairs_allow_empty_geometry(ticket.snapshot, ticket.prediction):
            raise ObservationHold("Paired image scopes contain no usable obstacle geometry")
        if hasattr(boxes, "__dict__"):
            boxes.source_snapshot = ticket.snapshot
        goal = first_action_target(ticket.prediction, bundle.feedback.q, self.client.metadata["action_semantics"],
                                   lower=self.lower, upper=self.upper)
        # Keep fingers as exogenous targets while limiting their per-tick travel.
        goal[12:] = np.clip(goal[12:], state.grip - args.gripper_fraction_speed * args.control_period,
                           state.grip + args.gripper_fraction_speed * args.control_period)
        max_step = min(args.max_joint_step, args.speed * args.control_period)
        _, info = self.planner.plan(state.q, state.grip, goal[None], boxes, lower=self.lower, upper=self.upper,
                                    max_step=max_step, previous_delta=self.previous_delta,
                                    acceleration_step=args.acceleration * args.control_period ** 2)
        target = check_optimized_step(info, state.q, info.get("waypoints", [[]])[0],
                                      max_step=max_step, max_slack_m=args.max_distance_slack)
        # Planning can take time. Recheck feedback/packet age immediately before
        # sending; never rebase a stale optimized trajectory onto moved joints.
        fresh = self._read_feedback()
        validate_age_and_drift(bundle, fresh, max_age_s=args.max_result_age, max_drift_rad=args.max_state_drift,
                               max_feedback_age_s=args.max_feedback_age)
        if np.max(np.abs(fresh.q - state.q)) > args.max_planning_drift:
            raise ObservationHold("Robot moved while optimizing; reobserve")
        target = check_optimized_step(info, fresh.q, target, max_step=max_step, max_slack_m=args.max_distance_slack)
        if np.any(target[:12] < self.lower) or np.any(target[:12] > self.upper):
            raise ObservationHold("Optimized target exceeds joint bounds")
        self.workers.check()
        began = time.monotonic()
        delta = target[:12] - fresh.q
        self.driver.send(fresh.q, delta, target[12:])
        self.previous_delta = delta.copy()
        # Observe the one control tick while acquisition/mapping stay alive.
        # Waiting for VLA never leaves a trajectory running on the hardware.
        try:
            while time.monotonic() - began < args.control_period:
                self.workers.stop.wait(min(.01, args.control_period - (time.monotonic() - began)))
                self._check_live()
        finally:
            self.driver.stop()
        if args.execute:
            self.report["commands_sent"] += 1
        return dict(status="executed" if args.execute else "planned_no_hardware_output", frame_id=bundle.frame_id,
                    observed_q=bundle.feedback.q, feedback_q=fresh.q, nominal_target=goal,
                    optimized_target=target, optimized_delta=delta,
                    obstacle_pairs=ticket.prediction["objects"], optimizer=info,
                    voxel_filter=ticket.snapshot.get("diagnostics", {}).get("vla_pair_voxels"),
                    paired_geometry=ticket.snapshot.get("diagnostics", {}).get("paired_rgb_obstacles"))

    def run(self):
        args, workers = self.args, self.workers
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        pending = None
        with (args.output / "steps.jsonl").open("w", encoding="utf-8") as log:
            try:
                workers.start("oa-feedback", lambda: feedback_loop(workers, self.feedback, self.driver.read,
                                                                    period=args.feedback_period))
                self.feedback.get(timeout=args.startup_timeout)
                workers.start("oa-cameras", lambda: camera_loop(workers, self.frames, self.feedback, self.source,
                    calibration_id=self.calibration_id, max_skew_s=args.max_camera_skew,
                    max_feedback_age_s=args.max_feedback_age, camera_info=self.camera_info, warmup_frames=args.warmup_frames))
                workers.start("oa-voxblox", lambda: mapping_loop(workers, self.frames, self.mapped,
                                                                 lambda: self.mapping_factory(self.camera_info)))
                workers.start("oa-inference", lambda: inference_loop(workers, self.mapped, self.results,
                                                                     self.client, args.prompt, args))
                version, last_result = 0, time.monotonic()
                while not workers.stop.is_set():
                    self._check_live()
                    if args.max_runtime and time.monotonic() - started >= args.max_runtime:
                        self.report["stop_reason"] = "runtime_limit"
                        break
                    try:
                        version, pending = self.results.get(after=version, timeout=.05)
                    except TimeoutError:
                        self.driver.stop()
                        if time.monotonic() - last_result > args.startup_timeout:
                            raise TimeoutError("No VLA/mapping result within startup/stall timeout")
                        continue
                    last_result = time.monotonic()
                    self.report["observations"] += 1
                    self.report["paired_geometry"] = (pending.snapshot or {}).get("diagnostics", {}).get("paired_rgb_obstacles")
                    try:
                        record = self.step(pending)
                    except ObservationHold as exc:
                        self.driver.stop()
                        self.previous_delta[:] = 0
                        self.report["holds"] += 1
                        record = dict(status="hold", frame_id=pending.mapped.bundle.frame_id, reason=str(exc),
                                      paired_geometry=self.report["paired_geometry"])
                    finally:
                        # Require the next captured packet to postdate execution.
                        pending.resume_after_ns = time.monotonic_ns()
                        pending.done.set()
                        pending = None
                    log.write(json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    print(f"[oa-vla] {record['status']} frame={record['frame_id']}" +
                          (f" reason={record['reason']}" if "reason" in record else ""), flush=True)
                    if args.max_cycles and self.report["observations"] >= args.max_cycles:
                        self.report["stop_reason"] = "observation_limit"
                        break
                self.report["completed"] = True
            except KeyboardInterrupt:
                self.report.update(completed=True, stop_reason="interrupted")
            except BaseException as exc:
                self.report.update(error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                workers.stop.set()
                if pending is not None:
                    pending.done.set()
                cleanup_errors = []
                for close in (self.driver.stop, self.client.close, workers.close, self.driver.close):
                    try:
                        close()
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
                if cleanup_errors:
                    self.report.update(completed=False, cleanup_errors=cleanup_errors)
                self.report["elapsed_s"] = time.monotonic() - started
                (args.output / "report.json").write_text(json.dumps(_json_value(self.report), ensure_ascii=False, indent=2) + "\n")
        return self.report


class HardwareIO:
    """Feedback SDK objects are used by one owner thread, outputs by main only."""

    def __init__(self, scene, args):
        from obstacle_validation.runtime import DualCommander, DualStateReader
        from obstacle_validation.gripper_output import DualGripperCommander
        self.scene, self.args = scene, args
        self.commander = self.grippers = self.reader = None
        self.moving = False
        left, right = (arm.ip for arm in scene.arms)
        try:
            if args.execute:
                self.commander = DualCommander(left, right, args.control_period)
                self.reader = self.commander.reader
                self.grippers = DualGripperCommander(left, right, speed=args.gripper_speed, force=args.gripper_force)
            else:
                self.reader = DualStateReader(left, right)
        except BaseException:
            self.close()
            raise

    def read(self):
        from real_scripts.rtde_state import read_rtde_tcp_state
        from real_scripts.live_dual_ur7e_obstacle_model import validate_active_tcp
        _, grip = self.reader.read()
        q, checks = [], {}
        for arm, receiver in zip(self.scene.arms, self.reader.receivers):
            state = read_rtde_tcp_state(receiver)
            mm, degrees = validate_active_tcp(state.q, state.tcp, self.scene.flange_to_tcp[arm.name],
                arm_name=f"{arm.name}_arm", controller_timestamp_s=state.timestamp_s)
            q.extend(state.q)
            checks[arm.name] = dict(position_error_mm=mm, rotation_error_deg=degrees)
        return np.asarray(q), grip, checks

    def send(self, q, delta, grip):
        if self.commander is not None:
            self.moving = True
            self.commander.send(q, delta)
            self.grippers.send_if_changed(grip)

    def stop(self):
        if self.commander is not None and self.moving:
            self.commander.stop()
            self.moving = False

    def close(self):
        errors = []
        for obj in (self.grippers, self.commander, None if self.commander else self.reader):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:
                    errors.append(str(exc))
        self.grippers = self.commander = self.reader = None
        if errors:
            raise RuntimeError(f"Hardware cleanup failed: {errors}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1", help="OA-VLA TCP host (default: local machine)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/oa_vla") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable hardware writes; default is live read-only acquisition and planning")
    parser.add_argument("--check-only", action="store_true", help="Validate configuration/dependencies; no camera/robot/socket I/O")
    parser.add_argument("--encoding", choices=("raw", "png", "jpeg"), default="raw")
    parser.add_argument("--socket-timeout", type=float, default=20.)
    parser.add_argument("--startup-timeout", type=float, default=30.)
    parser.add_argument("--control-period", type=float, default=.05)
    parser.add_argument("--feedback-period", type=float, default=.02)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--acceleration", type=float, default=1.)
    parser.add_argument("--max-joint-step", type=float, default=.015)
    parser.add_argument("--max-result-age", type=float, default=.75)
    parser.add_argument("--max-feedback-age", type=float, default=.15)
    parser.add_argument("--max-state-drift", type=float, default=.01)
    parser.add_argument("--max-planning-drift", type=float, default=.001)
    parser.add_argument("--max-camera-skew", type=float, default=.1)
    parser.add_argument("--max-distance-slack", type=float, default=1e-5)
    parser.add_argument("--minimum-object-score", type=float, default=.5)
    parser.add_argument("--obb-padding-pixels", type=float, default=2.)
    parser.add_argument("--voxel-resolution", type=float, default=.02)
    parser.add_argument("--max-grid-cells", type=int, default=2000000)
    parser.add_argument("--robot-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=25)
    parser.add_argument("--gripper-fraction-speed", type=float, default=.5)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-runtime", type=float)
    args = parser.parse_args(argv)
    positive = ("socket_timeout", "startup_timeout", "control_period", "feedback_period", "speed", "acceleration",
                "max_joint_step", "max_result_age", "max_feedback_age", "max_state_drift", "max_planning_drift",
                "max_camera_skew", "voxel_resolution", "gripper_fraction_speed")
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in ("max_distance_slack", "obb_padding_pixels"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if not 0 <= args.minimum_object_score <= 1 or not 1 <= args.port <= 65535 or not args.prompt.strip():
        parser.error("Invalid score, port or empty prompt")
    if args.control_period > args.max_result_age or args.feedback_period >= args.max_feedback_age:
        parser.error("Control/feedback periods exceed freshness budgets")
    for name in ("width", "height", "fps", "max_grid_cells"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.warmup_frames < 0 or not 0 <= args.gripper_speed <= 255 or not 0 <= args.gripper_force <= 255:
        parser.error("Invalid warmup or gripper setting")
    if (args.max_cycles is not None and args.max_cycles < 1
            or args.max_runtime is not None and (not math.isfinite(args.max_runtime) or args.max_runtime <= 0)):
        parser.error("Cycle/runtime limits must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    project, path = load_project_config(args.project_config)
    dependencies = ("oa_vla.socket_protocol", "msgpack", "osqp", "pyrealsense2", "rtde_receive", "real_scripts._voxblox_native")
    if args.execute:
        dependencies += ("rtde_control",)
    missing = []
    for name in dependencies:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except ModuleNotFoundError:
            missing.append(name)
    if args.check_only:
        print(json.dumps(dict(ready=not missing, missing=missing, execution=args.execute,
                              host=args.host, port=args.port,
                              cameras=["front", "side", "right_wrist", "left_wrist"]), indent=2))
        return 0 if not missing else 2
    if missing:
        raise RuntimeError(f"Missing runtime dependencies: {missing}; see docs/OA_VLA_SOCKET_RUNTIME.md")
    if args.output.exists():
        raise FileExistsError(f"Choose a new log directory: {args.output}")
    from oa_vla.socket_protocol import PolicyClient
    from real_scripts import live_dual_ur7e_obstacle_model as live
    from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource
    from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
    from real_scripts.voxblox_obstacle_map import require_voxblox
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
    import yaml

    require_voxblox()
    live_args = live.parse_args(["--project-config", str(path), "--no-browser", "--no-lingbot-depth",
                                 "--occupancy-backend", "voxblox", "--no-voxblox-esdf"])
    scene = live.DualObstacleScene(live_args)
    sampler = DualUR7eSurfacePointSampler(points_per_link=16, project_config=path)
    if (not project["safety"].get("camera_calibration_valid") or not sampler.calibration_digest
            or scene.report.get("sha256") != sampler.calibration_digest):
        raise RuntimeError("Validated camera/robot calibration required")
    if args.execute and not all(sampler.tcp_transform_verified.values()):
        raise RuntimeError("Verified active TCP transforms required for hardware execution")
    hardware = yaml.safe_load(resolve_repo_path(project["quest3"]["hardware_config"]).read_text())
    configs = []
    for name in ("front", "side", "right_wrist", "left_wrist"):
        serial = str(hardware["cameras"][name]["serial"])
        if name in scene.report["camera_serials"] and serial != scene.report["camera_serials"][name]:
            raise ValueError(f"{name} configured camera differs from calibrated serial")
        configs.append(D435iCameraConfig(name, serial))
    source = RealSenseD435iSource(cameras=tuple(configs), width=args.width, height=args.height, fps=args.fps,
                                 wait_timeout_ms=1000, read_retries=1, rgb_only_views=("right_wrist", "left_wrist"))
    planner = EllipsoidMCBF(sampler, margin=project["safety"]["collision_margin_m"],
        inter_arm_margin=project["safety"]["inter_arm_margin_m"], period=args.control_period,
        config=EllipsoidMCBFConfig(**project["safety"].get("ellipsoid_mcbf", {})))
    client = PolicyClient(args.host, args.port, timeout=args.socket_timeout)
    driver = None
    try:
        period = validate_server_metadata(client.metadata)
        if args.control_period > period + 1e-6:
            raise ValueError("Control period exceeds policy action period; choose a shorter --control-period")
        driver = HardwareIO(scene, args)
        runtime = LiveRuntime(args, client, driver, source,
                              lambda info: WorkspaceTSDF(scene, info, args), planner,
                              calibration_id=sampler.calibration_digest)
        return 0 if runtime.run()["completed"] else 1
    finally:
        client.close()
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
