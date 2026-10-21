"""Single camera owner and bounded immutable fan-out for inference and mapping."""

from dataclasses import dataclass
from types import MappingProxyType
import threading
import time

import numpy as np

from real_scripts.real_robot_adapter import RGBDFrame

POLICY_VIEWS = ("front", "side", "right_wrist", "left_wrist")


class LatestValue:
    """One replaceable item, no backlog. Worker errors supersede cached values."""

    def __init__(self):
        self.condition = threading.Condition()
        self.version, self.value, self.error, self.closed = 0, None, None, False

    def publish(self, value):
        with self.condition:
            if self.closed:
                return
            self.version += 1
            self.value = value
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"Background acquisition/inference failed: {self.error}") from self.error

    def get(self, *, after=0, timeout=1.):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                self.check()
                if self.closed:
                    raise EOFError("Frame channel closed")
                if self.value is not None and self.version > after:
                    return self.version, self.value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No fresh shared frame/result")
                self.condition.wait(remaining)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def immutable_array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Feedback:
    q: np.ndarray
    grip: np.ndarray
    timestamp_ns: int
    tcp_checks: dict

    def __post_init__(self):
        object.__setattr__(self, "q", immutable_array(self.q, (12,), "dual-arm joints"))
        grip = immutable_array(self.grip, (2,), "grippers")
        if np.any((grip < 0) | (grip > 1)) or type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("Invalid feedback timestamp/gripper range")
        object.__setattr__(self, "grip", grip)


def freeze_frame(frame):
    """Copy SDK-owned storage once, then share read-only NumPy arrays locally."""
    rgb = np.array(frame.rgb, dtype=np.uint8, copy=True)
    depth = np.array(frame.depth_m, dtype=np.float32, copy=True)
    rgb.setflags(write=False)
    depth.setflags(write=False)
    return RGBDFrame(frame.camera_name, rgb, depth, host_timestamp_ns=frame.host_timestamp_ns,
                     device_timestamp_ms=frame.device_timestamp_ms, frame_number=frame.frame_number,
                     timestamp_domain=frame.timestamp_domain)


@dataclass(frozen=True)
class FrameBundle:
    frame_id: int
    frames: dict
    feedback: Feedback
    calibration_id: str

    @property
    def capture_ns(self):
        return min(frame.host_timestamp_ns for frame in self.frames.values())

    def context(self):
        return dict(frame_id=self.frame_id, capture_ns=self.capture_ns, calibration_id=self.calibration_id,
                    feedback_ns=self.feedback.timestamp_ns,
                    views={name: dict(timestamp_ns=frame.host_timestamp_ns, frame_number=frame.frame_number,
                                      height=frame.rgb.shape[0], width=frame.rgb.shape[1])
                           for name, frame in self.frames.items()})


class Workers:
    """Workers own SDK/TSDF objects; shutdown wakes readers and joins owners."""

    def __init__(self):
        self.stop = threading.Event()
        self.threads, self.channels = [], []
        self.failures = LatestValue()

    def start(self, name, function):
        def run():
            try:
                function()
            except BaseException as exc:
                if not self.stop.is_set():
                    self.failures.fail(exc)
        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()

    def channel(self):
        result = LatestValue()
        self.channels.append(result)
        return result

    def check(self):
        self.failures.check()

    def close(self):
        self.stop.set()
        for channel in self.channels:
            channel.close()
        for thread in self.threads:
            thread.join(timeout=5.)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"Workers did not release resources: {alive}")


def feedback_loop(workers, channel, reader, *, period=.02):
    while not workers.stop.is_set():
        began = time.monotonic()
        q, grip, checks = reader()
        channel.publish(Feedback(q, grip, time.monotonic_ns(), checks))
        workers.stop.wait(max(0., period - (time.monotonic() - began)))


def camera_loop(workers, channel, feedback, source, *, calibration_id, max_skew_s=.1,
                max_feedback_age_s=.15, camera_info=None, warmup_frames=5):
    """This is the only code that starts/reads/stops physical camera streams."""
    try:
        source.start()
        if camera_info is not None:
            camera_info.update(source.get_color_calibrations())
        for _ in range(warmup_frames):
            if workers.stop.is_set():
                return
            source.read()
        frame_id = 0
        while not workers.stop.is_set():
            frames = source.read()
            if set(frames) != set(POLICY_VIEWS):
                raise ValueError("All four physical camera streams are required")
            stamps = [frame.host_timestamp_ns for frame in frames.values()]
            if any(type(stamp) is not int or stamp <= 0 for stamp in stamps):
                raise ValueError("Every RGB-D frame requires a host monotonic timestamp")
            if (max(stamps) - min(stamps)) / 1e9 > max_skew_s:
                raise RuntimeError("Four-view camera skew exceeds limit")
            _, state = feedback.get(timeout=1.)
            if max(abs(stamp - state.timestamp_ns) for stamp in stamps) / 1e9 > max_feedback_age_s:
                raise RuntimeError("Camera frame and robot feedback are not time aligned")
            frozen = MappingProxyType({name: freeze_frame(frame) for name, frame in frames.items()})
            channel.publish(FrameBundle(frame_id, frozen, state, calibration_id))
            frame_id += 1
    finally:
        source.stop()
