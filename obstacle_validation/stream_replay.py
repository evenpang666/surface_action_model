"""Socket action replay: baseline without a selection, ellipsoid avoidance with one."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import select
import socket
import time

import numpy as np
import yaml

from obstacle_validation.action_stream import ProtocolError, receive_session
from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
from obstacle_validation.lerobot_replay import LiveSceneReader, DualGripperCommander
from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
from obstacle_validation.runtime import DualCommander, EXECUTE_ACK, snapshot_state
from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.replay_voxblox import ReplayVoxbloxSource
from safety_module.dual_camera_calibration import validate_calibration_identity
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path


class ReplayExecutor:
    def __init__(self, args, cleanup, connection):
        self.args, self.cleanup, self.connection = args, cleanup, connection
        self.commander = self.grippers = self.scene = self.controller = None
        self.restored = False
        self.report = {"completed": False, "status": "initializing", "actions_executed": 0,
                       "restore_completed": False,
                       "mode": "baseline" if self.baseline else "avoid",
                       "controller": "baseline" if self.baseline else "ellipsoid_mcbf", "gripper_optimization": False,
                       "perception_enabled": not self.baseline, "avoidance_enabled": not self.baseline,
                       "same_arm_avoidance": False,
                       "robot_mask_enabled": False if self.baseline else args.robot_mask, "inference_source": "socket",
                       "execution": "physical" if args.execute else "virtual",
                       "distance_relaxation_ticks": 0, "max_distance_slack_m": 0.,
                       "max_cycle_s": 0.}
        args.output.mkdir(parents=True, exist_ok=True)
        self.log = cleanup.enter_context((args.output/"steps.jsonl").open("w"))
        self.chunk_log = cleanup.enter_context((args.output/"chunks.jsonl").open("w"))
        self._write_report()

    @property
    def baseline(self):
        return self.args.manual_selection is None

    def _write_report(self):
        (self.args.output/"report.json").write_text(json.dumps(self.report, indent=2, ensure_ascii=False))

    def start(self, hello):
        args = self.args
        prefix = "baseline" if self.baseline else "avoidance"
        print(f"[{prefix}] preparing {'direct action replay' if self.baseline else 'selected Voxblox and link ellipsoids'}", flush=True)
        project, path = load_project_config(args.project_config)
        safety = project["safety"]
        self.period = hello["period_s"]
        self.total = hello["total_actions"]
        self.lower = -np.tile([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi], 2)
        self.upper = -self.lower
        initial = np.asarray(hello["initial_state"], float)
        if np.any(initial[:12] < self.lower) or np.any(initial[:12] > self.upper):
            raise ValueError("Recorded initial state exceeds UR joint bounds")
        self.initial = initial.copy()
        if args.execute and args.initial_state:
            raise ValueError("--initial-state is virtual-only")
        provider = manual = None
        if self.baseline:
            # No sampler, camera calibration, scene reader, TSDF or optimizer.
            # Virtual baseline starts at the recorded state; physical baseline
            # reads real joint/gripper feedback below before interpolation.
            self.q, self.grip = initial[:12].copy(), initial[12:].copy()
        else:
            sampler = DualUR7eSurfacePointSampler(points_per_link=16, project_config=path)
            if not safety.get("camera_calibration_valid") or not sampler.calibration_digest:
                raise RuntimeError("Replay requires validated shared camera calibration")
            if args.execute and not all(sampler.tcp_transform_verified.values()):
                raise RuntimeError("Physical replay requires verified active TCPs")
            self.digest = sampler.calibration_digest
            manual = ManualVoxelFilter(args.manual_selection)
            if args.obstacle_source == "auto" and args.obstacle_url is None:
                provider = ReplayVoxbloxSource(path, manual, startup_timeout_s=args.voxblox_startup_timeout,
                                              robot_mask=args.robot_mask)
            self.scene = LiveSceneReader(args.obstacle_url or safety["obstacle_url"],
                                         safety["stale_timeout_s"], manual,
                                         reuse_expired=args.obstacle_age_policy == "latest", provider=provider)
            self.cleanup.callback(self.scene.close)
            snapshot = self.scene.read(max_wait_s=5.)
            validate_calibration_identity(snapshot, self.digest)
            self.scene.start(snapshot)
            self.controller = EllipsoidMCBF(sampler, margin=safety["collision_margin_m"],
                                             inter_arm_margin=safety["inter_arm_margin_m"], period=self.period,
                                             config=EllipsoidMCBFConfig(**safety.get("ellipsoid_mcbf", {})))
            # Fit all rigid link models before opening motion controllers.
            self.q, self.grip = snapshot_state(snapshot)
        if args.initial_state:
            value = json.loads(args.initial_state.read_text())
            self.q, self.grip = np.asarray(value["qpos"], float), np.asarray(value["gripper_position"], float)
            if (self.q.shape != (12,) or self.grip.shape != (2,)
                    or not np.isfinite(self.q).all() or not np.isfinite(self.grip).all()
                    or np.any(self.q < self.lower) or np.any(self.q > self.upper)
                    or np.any((self.grip < 0) | (self.grip > 1))):
                raise ValueError("Invalid virtual initial joints or grippers")
        if args.execute:
            hardware = yaml.safe_load(resolve_repo_path(project["quest3"]["hardware_config"]).read_text())
            left, right = (str(hardware["arms"][name]["robot_ip"]) for name in ("left_arm", "right_arm"))
            self.commander = DualCommander(left, right, self.period)
            self.cleanup.callback(self.commander.close)
            self.grippers = DualGripperCommander(left, right, speed=args.gripper_speed, force=args.gripper_force)
            self.cleanup.callback(self.grippers.close)
            self.q, self.grip = self.commander.read_state()
        self.previous = np.zeros(12)
        self.step = args.speed*self.period if args.speed else np.inf
        self.acceleration = args.acceleration*self.period**2 if args.acceleration else np.inf
        self.report.update(status="ready", total_actions=self.total, period_s=self.period,
                           manual_selection=None if manual is None else str(manual.path), recorded_initial_state=initial.tolist(),
                           initial_qpos=self.q.tolist(), restore_policy="recorded_initial_state",
                           restore_controller="interpolation", restore_collision_checks=False,
                           obstacle_source="disabled" if self.baseline else "embedded_voxblox" if provider else "external")
        self._write_report()
        print(f"[{prefix}] ready: {self.total} actions, nominal period {self.period:g}s; "
              f"execution={'physical' if args.execute else 'virtual'}", flush=True)

    def _check_peer(self):
        readable, _, _ = select.select([self.connection], [], [], 0)
        if readable:
            data = self.connection.recv(1, socket.MSG_PEEK)
            if not data:
                raise EOFError("VLA sender disconnected during execution")
            raise ProtocolError("Sender transmitted before execution acknowledgement")

    def _restore(self, progress):
        """Restore recorded joints/grippers without perception or avoidance control."""
        args = self.args
        goal_q, goal_grip = self.initial[:12], self.initial[12:]
        start_q, start_grip = self.q.copy(), self.grip.copy()
        duration = max(float(np.max(np.abs(goal_q-start_q)))/args.restore_speed,
                       float(np.max(np.abs(goal_grip-start_grip)))/args.gripper_speed_fraction_s)
        steps = max(1, math.ceil(1.5*duration/self.period))
        started, last_progress, attempt = time.monotonic(), -np.inf, 0
        self.report.update(status="restoring")
        self._write_report()
        print("[restore] interpolating to recorded initial state; collision_checks=False", flush=True)
        while True:
            tick = time.monotonic()
            self._check_peer()
            if tick-started >= args.restore_timeout:
                raise TimeoutError("Initial-state restoration did not reach recorded joints/grippers")
            fraction = min((attempt+1)/steps, 1.)
            fraction = fraction*fraction*(3-2*fraction)
            desired_q = start_q+fraction*(goal_q-start_q)
            desired_grip = start_grip+fraction*(goal_grip-start_grip)
            if self.commander:
                self.q, self.grip = self.commander.read_state()
            delta = np.clip(desired_q-self.q, -args.restore_speed*self.period,
                            args.restore_speed*self.period)
            if self.commander:
                self.commander.send(self.q, delta)
                self.grippers.send_if_changed(desired_grip)
            else:
                self.q, self.grip = self.q+delta, desired_grip.copy()
            time.sleep(max(0., self.period-(time.monotonic()-tick)))
            if self.commander:
                self.q, self.grip = self.commander.read_state()
            error = float(np.max(np.abs(self.q-goal_q)))
            grip_error = float(np.max(np.abs(self.grip-goal_grip)))
            self.log.write(json.dumps({"phase": "restore", "step": attempt,
                "qpos": self.q.tolist(), "gripper_position": self.grip.tolist(),
                "target_qpos": desired_q.tolist(), "target_gripper": desired_grip.tolist(),
                "joint_delta": delta.tolist(), "target_error_rad": error,
                "gripper_error": grip_error, "restore_controller": "interpolation",
                "collision_checks": False, "cbf": None,
                "cycle_compute_s": time.monotonic()-tick})+"\n")
            self.log.flush()
            if time.monotonic()-last_progress >= 1.:
                progress(None, phase="restore", stage="restore", target_error_rad=error,
                         gripper_error=grip_error, collision_checks=False)
                last_progress = time.monotonic()
            attempt += 1
            if (attempt >= steps and error <= args.restore_tolerance
                    and grip_error <= args.gripper_tolerance):
                break
        if self.commander:
            self.commander.stop()
            self.q, self.grip = self.commander.read_state()
            if (np.max(np.abs(self.q-goal_q)) > args.restore_tolerance
                    or np.max(np.abs(self.grip-goal_grip)) > args.gripper_tolerance):
                raise RuntimeError("Initial-state restoration moved outside tolerance after stopping")
        self.previous = np.zeros(12)
        self.restored = True
        self.report.update(status="running", restore_completed=True,
                           restore_elapsed_s=time.monotonic()-started,
                           restored_qpos=self.q.tolist(), restored_gripper_position=self.grip.tolist())
        self._write_report()
        print(f"[restore] completed; starting {'baseline' if self.baseline else 'obstacle-aware'} action replay", flush=True)

    def _baseline_target(self, action):
        """Recorded joints/grippers, optionally bounded by explicit rate limits."""
        if not np.isfinite(self.step) and not np.isfinite(self.acceleration):
            return action.copy()
        low, high = np.maximum(self.lower-self.q, -self.step), np.minimum(self.upper-self.q, self.step)
        if np.isfinite(self.acceleration):
            low = np.maximum(low, self.previous-self.acceleration)
            high = np.minimum(high, self.previous+self.acceleration)
        if np.any(low > high):
            raise ValueError("Measured state and rate bounds have no feasible joint step")
        return np.r_[self.q+np.clip(action[:12]-self.q, low, high), action[12:]]

    def execute_chunk(self, actions, start, progress):
        # Validate the entire chunk before moving any joint.
        if np.any(actions[:, :12] < self.lower) or np.any(actions[:, :12] > self.upper):
            raise ValueError("Chunk target exceeds UR joint bounds")
        self.chunk_log.write(json.dumps({"start": start, "actions": actions.tolist(), "status": "received"})+"\n")
        self.chunk_log.flush()
        if not self.restored:
            self._restore(progress)
        last_progress = -np.inf
        for offset, action in enumerate(actions):
            index, attempt, started = start+offset, 0, time.monotonic()
            tolerance = self.args.final_tolerance if index == self.total-1 else self.args.arrival_tolerance
            while True:
                tick = time.monotonic()
                self._check_peer()
                if self.commander:
                    self.q, self.grip = self.commander.read_state()
                detail = None
                if self.baseline:
                    target = self._baseline_target(action)
                else:
                    snapshot = self.scene.read()
                    validate_calibration_identity(snapshot, self.digest)
                    boxes = self.scene.boxes(snapshot)
                    _, info = self.controller.plan(self.q, self.grip, action[None], boxes,
                                                    lower=self.lower, upper=self.upper, max_step=self.step,
                                                    previous_delta=self.previous, acceleration_step=self.acceleration)
                    target = np.asarray(info["waypoints"][0])
                    detail = info["steps"][0]
                delta = target[:12]-self.q
                if self.commander:
                    self.commander.send(self.q, delta)
                    self.grippers.send_if_changed(action[12:])
                else:
                    self.q, self.grip = target[:12].copy(), action[12:].copy()
                self.previous = delta
                time.sleep(max(0., self.period-(time.monotonic()-tick)))
                # ACK requires feedback AFTER execution, not a predicted target.
                if self.commander:
                    self.q, self.grip = self.commander.read_state()
                error = float(np.max(np.abs(self.q-action[:12])))
                self.report["max_cycle_s"] = max(self.report["max_cycle_s"], time.monotonic()-tick)
                if detail is not None:
                    self.report["distance_relaxation_ticks"] += int(detail["distance_constraints_relaxed"])
                    self.report["max_distance_slack_m"] = max(self.report["max_distance_slack_m"], detail["max_distance_slack_m"])
                    self.report["last_geometry"] = {key: detail[key] for key in
                                                    ("robot_ellipsoids", "obstacle_ellipsoids", "constraint_pairs")}
                    self.report["replay_voxblox"] = snapshot.get("diagnostics", {}).get("replay_voxblox")
                self.log.write(json.dumps({"phase": "path", "action_index": index, "attempt": attempt,
                    "qpos": self.q.tolist(), "gripper_position": self.grip.tolist(), "action": action.tolist(),
                    "control_target": target.tolist(), "target_error_rad": error,
                    "cycle_compute_s": time.monotonic()-tick, "cbf": detail})+"\n")
                self.log.flush()
                attempt += 1
                if time.monotonic()-last_progress >= 1.:
                    progress(index, phase="path", target_error_rad=error,
                             stage="baseline" if detail is None else info["adaptive_stage"]["stage"],
                             optimization_applied=False if detail is None else detail["optimization_applied"],
                             active_constraints=0 if detail is None else detail["active_constraints"],
                             path_progress_rad=[] if detail is None else detail["path_progress_rad"])
                    last_progress = time.monotonic()
                if error <= tolerance:
                    self.report["actions_executed"] = index+1
                    break
                if self.args.action_timeout and time.monotonic()-started > self.args.action_timeout:
                    raise TimeoutError(f"Action {index} did not arrive; error={error:.4f} rad")
        self.chunk_log.write(json.dumps({"start": start, "status": "executed", "count": len(actions)})+"\n")
        self.chunk_log.flush()
        self._write_report()
        print(f"[{'baseline' if self.baseline else 'avoidance'}] executed actions {start}:{start+len(actions)}", flush=True)
        # Between chunks no unacknowledged action is executed. The existing
        # servo watchdog handles a sender that stalls after receiving its ACK.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8877)
    p.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    p.add_argument("--manual-selection", type=Path,
                   help="With a selection: ellipsoid avoidance; omitted: baseline without perception or optimization")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--execute-ack", default="")
    p.add_argument("--initial-state", type=Path)
    p.add_argument("--obstacle-source", choices=("auto", "external"), default="auto")
    p.add_argument("--obstacle-url")
    p.add_argument("--obstacle-age-policy", choices=("latest", "strict"), default="latest")
    p.add_argument("--robot-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--voxblox-startup-timeout", type=float, default=30.)
    p.add_argument("--socket-timeout", type=float, default=120.)
    p.add_argument("--speed", type=float)
    p.add_argument("--acceleration", type=float)
    p.add_argument("--restore-speed", type=float, default=.5, help="Initial-state interpolation speed, rad/s")
    p.add_argument("--gripper-speed-fraction-s", type=float, default=.5)
    p.add_argument("--restore-tolerance", type=float, default=.015)
    p.add_argument("--gripper-tolerance", type=float, default=.06)
    p.add_argument("--restore-timeout", type=float, default=90.)
    p.add_argument("--arrival-tolerance", type=float, default=.025)
    p.add_argument("--final-tolerance", type=float, default=.015)
    p.add_argument("--action-timeout", type=float, default=0., help="0: continue seeking the original goal")
    p.add_argument("--gripper-speed", type=int, default=255)
    p.add_argument("--gripper-force", type=int, default=25)
    return p


def run(args):
    if args.execute and args.execute_ack != EXECUTE_ACK:
        raise ValueError(f"Physical execution requires --execute-ack {EXECUTE_ACK}")
    for name in ("speed", "acceleration", "arrival_tolerance", "final_tolerance",
                 "socket_timeout", "voxblox_startup_timeout", "restore_speed",
                 "gripper_speed_fraction_s", "restore_tolerance", "gripper_tolerance", "restore_timeout"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive")
    if (not np.isfinite(args.action_timeout) or args.action_timeout < 0
            or not 0 <= args.gripper_speed <= 255 or not 0 <= args.gripper_force <= 255):
        raise ValueError("Invalid action timeout or gripper setting")
    executor = None
    try:
        with ExitStack() as cleanup:
            connection = cleanup.enter_context(socket.create_connection((args.host, args.port), timeout=args.socket_timeout))
            connection.settimeout(args.socket_timeout)
            executor = ReplayExecutor(args, cleanup, connection)
            result = receive_session(connection, executor.execute_chunk, start_session=executor.start)
            executor.report.update(result, status="completed")
    except BaseException as exc:
        if executor is not None:
            executor.report.update(completed=False, status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if executor is not None:
            executor._write_report()
    return executor.report


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
