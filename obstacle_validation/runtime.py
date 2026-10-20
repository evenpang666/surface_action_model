"""Read-only obstacle snapshots and explicitly armed dual RTDE output."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np

from real_scripts.real_cbf_qp import boxes_from_snapshot
from safety_module.command_watchdog import CommandWatchdog
from safety_module.runtime_contract import SnapshotExpired, SnapshotUnknown, validate_snapshot
from obstacle_validation.core import vector


EXECUTE_ACK = "I_UNDERSTAND_DUAL_UR7E_WILL_MOVE"


def timed_call(stats, name, function, *args, **kwargs):
    if stats is None:
        return function(*args, **kwargs)
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        elapsed = time.perf_counter()-started
        item = stats.setdefault(name, {"calls": 0, "total_s": 0., "max_s": 0.})
        item["calls"] += 1
        item["total_s"] += elapsed
        item["max_s"] = max(item["max_s"], elapsed)


class LatestSnapshotCache:
    """Fetch in one worker; control reads reuse the latest accepted scene."""

    def __init__(self, fetch, validate, initial, *, poll_interval_s=.1, reuse_on_transport_error=False):
        self.fetch, self.validate = fetch, validate
        self.latest, self.error = initial, None
        self.poll_interval_s = poll_interval_s
        self.reuse_on_transport_error = reuse_on_transport_error
        self.refresh_errors = 0
        self.last_refresh_error = None
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.reads = self.refreshes = 0
        self.thread = threading.Thread(target=self._refresh, name="replay-obstacles", daemon=True)
        self.thread.start()

    def _refresh(self):
        while not self.stopping.is_set():
            retain = False
            try:
                value, error = self.fetch(), None
            except Exception as exc:
                value, error = None, exc
                retain = (self.reuse_on_transport_error
                          and isinstance(exc, (URLError, TimeoutError, ConnectionError))
                          and not isinstance(exc, HTTPError))
            with self.condition:
                if error is not None:
                    self.refresh_errors += 1
                    self.last_refresh_error = f"{type(error).__name__}: {error}"
                # A transport outage neither erases an accepted frame nor
                # clears a previously reported unsafe/invalid scene.
                if not (retain and (self.latest is not None or self.error is not None)):
                    self.latest, self.error = value, error
                self.refreshes += 1
                self.condition.notify_all()
            self.stopping.wait(self.poll_interval_s)

    def read(self, *, max_wait_s=0., after_timestamp_ns=None):
        if not np.isfinite(max_wait_s) or max_wait_s < 0:
            raise ValueError("Snapshot wait must be finite and nonnegative")
        if after_timestamp_ns is not None and (not isinstance(after_timestamp_ns, int)
                                              or after_timestamp_ns < 0):
            raise ValueError("Invalid previous capture timestamp")
        deadline = time.monotonic()+max_wait_s
        with self.condition:
            self.reads += 1
            while True:
                try:
                    if self.error is not None:
                        raise self.error
                    self.validate(self.latest)
                    if after_timestamp_ns is not None and self.latest["host_timestamp_ns"] <= after_timestamp_ns:
                        raise SnapshotExpired("Waiting for a newer capture after swept free-space evidence expired")
                    return self.latest
                except SnapshotExpired:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or self.stopping.is_set():
                        raise
                    self.condition.wait(remaining)

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3.)


class CapturedFrameObservation:
    """Reuse a frame's published occupancy belief without elapsed-time decay.

    Only replay's explicit latest-frame policy uses this adapter. The original
    capture time and cell ages remain intact; occupied cells still block.
    """

    def __init__(self, observation):
        self.observation = observation

    def __getattr__(self, name):
        return getattr(self.observation, name)

    def segments_observed(self, start, end, *, margin_m=0., now_ns=None,
                          allow_margin_recovery=False):
        recovery = {"allow_margin_recovery": True} if allow_margin_recovery else {}
        return self.observation.segments_observed(start, end, margin_m=margin_m,
                                                 now_ns=self.observation.timestamp_ns, **recovery)


class ObstacleSceneUnstable(RuntimeError):
    """A fresh perception frame whose obstacle clusters are still changing."""


def _tracked_obbs_cover_unstable_scene(value):
    if value.get("scene_observation_state") == "unknown" or value.get("unknown_obstacles"):
        return False
    status = str(value.get("status", ""))
    prefix = "error: unstable obstacle point clusters ("
    if not status.startswith(prefix) or not status.endswith(")"):
        return False
    reasons = {part.strip() for part in status[len(prefix):-1].split(",")}
    diagnostics = value.get("diagnostics", {})
    if (not reasons or not reasons <= {"new_cluster", "changed_cluster",
                                       "merged_clusters", "missing_cluster"}
            or set(diagnostics.get("obb_unstable_reasons", [])) != reasons
            or not isinstance(value.get("obbs"), list) or not value["obbs"]
            or diagnostics.get("published_obbs") != len(value["obbs"])):
        return False
    # Older detectors publish every confirmed historical box for merges and
    # missing clusters. New/changed clusters require a detector that also
    # publishes current candidate boxes, so no observed obstacle is omitted.
    return (reasons <= {"merged_clusters", "missing_cluster"}
            or diagnostics.get("safety_obbs_cover_candidates") is True)


def boxes_from_json(value):
    return boxes_from_snapshot(value)


def validate_obstacle_snapshot(value, max_age, *, allow_tracked_obbs=False):
    if allow_tracked_obbs and _tracked_obbs_cover_unstable_scene(value):
        value = {**value, "status": "tracked_obbs_in_use"}
    return validate_snapshot(value, max_age)


def snapshot_state(value):
    q = vector([*value["qpos"]["left"], *value["qpos"]["right"]], 12, "snapshot joints")
    grip = vector([value["gripper_position"]["left"], value["gripper_position"]["right"]], 2, "snapshot grippers")
    if np.any(grip < 0) or np.any(grip > 1):
        raise ValueError("Snapshot grippers must be in [0,1]")
    return q, grip


def read_snapshot(url, max_age, *, max_wait_s=0., http_timeout_s=None,
                  allow_tracked_obbs=False, snapshot_transform=None, after_timestamp_ns=None,
                  timing_stats=None):
    """Read the small safety payload; optionally wait before controllers connect.

    Finite-age execution reads remain fail-fast. With allow_tracked_obbs, a
    fresh unstable-cluster frame may be used when its published OBBs cover
    the tracked scene and every currently observed candidate.
    Older servers ignore compact=1 and still return a compatible full snapshot.
    An infinite max_age disables only the age cutoff and requires a finite
    http_timeout_s so an unresponsive detector cannot block indefinitely.
    A supplied transform runs before validation; the manual replay filter uses
    it to check occupied cells inside its explicitly marked voxel scope.
    ``after_timestamp_ns`` explicitly requires a newer capture, even if the
    snapshot's overall age is acceptable.
    """
    if not np.isfinite(max_wait_s) or max_wait_s < 0:
        raise ValueError("Snapshot wait must be finite and nonnegative")
    if after_timestamp_ns is not None and (not isinstance(after_timestamp_ns, int)
                                          or after_timestamp_ns < 0):
        raise ValueError("Invalid previous capture timestamp")
    timeout_s = max_age if http_timeout_s is None else float(http_timeout_s)
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Snapshot HTTP timeout must be finite and positive")
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "compact"]
    url = urlunsplit(parts._replace(query=urlencode([*query, ("compact", "1")])))
    deadline = time.monotonic() + max_wait_s
    def fetch():
        with urlopen(Request(url, headers={"Cache-Control": "no-cache"}), timeout=timeout_s) as response:
            return json.load(response)
    while True:
        value = timed_call(timing_stats, "fetch_and_json", fetch)
        try:
            if snapshot_transform is not None:
                value = timed_call(timing_stats, "manual_filter", snapshot_transform, value)
            validate_obstacle_snapshot(value, max_age, allow_tracked_obbs=allow_tracked_obbs)
            if after_timestamp_ns is not None and value["host_timestamp_ns"] <= after_timestamp_ns:
                raise SnapshotExpired("Waiting for a newer capture")
        except SnapshotExpired as exc:
            remaining = deadline - time.monotonic()
            if max_wait_s > 0 and remaining > 0:
                time.sleep(min(.05, remaining))
                continue
            diagnostics = value.get("diagnostics", {})
            hint = ("; restart detection with --no-lingbot-depth for model-free validation"
                    if diagnostics.get("depth_refinement") == "lingbot-depth" else "")
            capture_to_publish = diagnostics.get("capture_to_publish_seconds")
            if isinstance(capture_to_publish, (int, float)) and np.isfinite(capture_to_publish):
                hint += f"; capture-to-publish {capture_to_publish:.3f}s"
            raise SnapshotExpired(f"{exc}; detector status: {value.get('status', 'unknown')}{hint}") from exc
        except SnapshotUnknown:
            # Candidate coverage cannot turn a missing observation into free space.
            raise
        except RuntimeError as exc:
            # Strict callers may wait for cluster confirmation at startup.
            # Live replay accepts only the explicitly covered OBB fallback;
            # other unstable frames still fail immediately during motion.
            remaining = deadline - time.monotonic()
            if value.get("status") == "initializing" and max_wait_s > 0 and remaining > 0:
                time.sleep(min(.05, remaining))
                continue
            if str(value.get("status", "")).startswith("error: unstable obstacle point clusters"):
                if max_wait_s > 0 and remaining > 0:
                    time.sleep(min(.05, remaining))
                    continue
                raise ObstacleSceneUnstable(str(exc)) from exc
            raise
        snapshot_state(value)
        timed_call(timing_stats, "decode_validate", boxes_from_json, value)
        return value


class DualStateReader:
    """Receive-only joint/GET POS feedback, independent of perception cadence."""

    def __init__(self, left_ip, right_ip, *, gripper_port=63352):
        from rtde_receive import RTDEReceiveInterface
        from real_scripts.live_dual_ur7e_obstacle_model import ReadOnlyRobotiq2FPosition

        self.receivers, self.grippers = [], []
        self.timestamps, self.updated_at = [None, None], [None, None]
        try:
            for address in (left_ip, right_ip):
                self.receivers.append(RTDEReceiveInterface(address))
                self.grippers.append(ReadOnlyRobotiq2FPosition(address, port=gripper_port))
        except BaseException:
            try: self.close()
            except Exception: pass
            raise

    def read(self):
        began = time.monotonic()
        joints, grip = [], []
        for index, (receiver, gripper) in enumerate(zip(self.receivers, self.grippers)):
            name = ("left", "right")[index]
            if not receiver.isConnected():
                raise RuntimeError(f"{name} RTDE feedback disconnected")
            stamp, now = float(receiver.getTimestamp()), time.monotonic()
            previous = self.timestamps[index]
            if not np.isfinite(stamp) or (previous is not None and stamp < previous):
                raise RuntimeError(f"{name} RTDE clock invalid or restarted")
            if previous is None or stamp > previous:
                self.timestamps[index], self.updated_at[index] = stamp, now
            elif now-self.updated_at[index] > .25:
                raise RuntimeError(f"{name} RTDE feedback stopped updating for over 0.25s")
            joints.extend(vector(receiver.getActualQ(), 6, f"{name} RTDE joints"))
            grip.append(gripper.closing_fraction())
        grip = vector(grip, 2, "measured grippers")
        if np.any(grip < 0) or np.any(grip > 1):
            raise RuntimeError("Measured grippers must be in [0,1]")
        if time.monotonic()-began > .25:
            raise RuntimeError("Robot feedback read took over 0.25s")
        return vector(joints, 12, "measured joints"), grip

    def diagnostics(self):
        return {name: {"robot_mode": r.getRobotMode(), "runtime_state": r.getRuntimeState(),
                       "speed_scaling": r.getSpeedScaling(), "target_speed_fraction": r.getTargetSpeedFraction()}
                for name, r in zip(("left", "right"), self.receivers)}

    def close(self):
        receivers, grippers = self.receivers, self.grippers
        self.receivers, self.grippers = [], []
        errors = []
        for close in [*(r.disconnect for r in receivers), *(g.close for g in grippers)]:
            try: close()
            except Exception as exc: errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close read-only robot feedback") from errors[0]


class DualCommander:
    """Checked dual output with separate receive-only feedback; no policy."""

    def __init__(self, left_ip, right_ip, period):
        from rtde_control import RTDEControlInterface

        self.controls, self.period = [], period
        self.reader = DualStateReader(left_ip, right_ip)
        try:
            for address in (left_ip, right_ip):
                self.controls.append(RTDEControlInterface(address))
        except BaseException:
            for control in self.controls:
                try: control.servoStop(1.)
                except Exception: pass
                try: control.disconnect()
                except Exception: pass
            try: self.reader.close()
            except Exception: pass
            raise
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.commands_sent, self.watchdog_stops = 0, 0
        self.last_send_started_at = float("-inf")
        self.watchdog = CommandWatchdog(self._watchdog_stop, max(.25, 2*period))

    def _watchdog_stop(self):
        self.watchdog_stops += 1
        self.stop()

    def read_state(self):
        return self.reader.read()

    def controller_diagnostics(self):
        return self.reader.diagnostics()

    def stop(self):
        errors = []
        with self.watchdog.lock:
            self.watchdog.deadline = None
            for control in self.controls:
                try:
                    control.servoStop(1.)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("Failed to stop a UR controller") from errors[0]

    def send(self, q, delta):
        targets = (q[:6]+delta[:6], q[6:]+delta[6:])

        def send_one(pair):
            control, target = pair
            return bool(control.servoJ(target.tolist(), .10, .30, self.period, .10, 300))

        with self.watchdog.lock:
            self.last_send_started_at = time.monotonic()
            self.watchdog.renew()
            accepted = list(self.pool.map(send_one, zip(self.controls, targets)))
        if not all(accepted):
            self.stop()
            raise RuntimeError("A UR controller rejected the validation step")
        self.commands_sent += 1

    def close(self):
        self.watchdog.close()
        errors = []
        cleanup = [self.stop, self.pool.shutdown, self.reader.close,
                   *(control.disconnect for control in self.controls)]
        for close in cleanup:
            try: close()
            except Exception as exc: errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close dual UR output") from errors[0]


