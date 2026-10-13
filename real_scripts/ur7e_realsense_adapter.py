#!/usr/bin/env python3
"""UR7e + three Intel RealSense D435i adapter for real SafetyModule scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
import warnings
from typing import Protocol, Sequence

import numpy as np

from real_scripts.real_robot_adapter import DEFAULT_RGBD_CAMERA_NAMES, NativeDepthFrame, RGBDFrame
from real_scripts.ur7e_controller import ROBOT_IP, UR7eVectorController


DEFAULT_D435I_CAMERA_NAMES = DEFAULT_RGBD_CAMERA_NAMES


class UR7eControllerLike(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def get_current_joints(self) -> Sequence[float]: ...

    def get_gripper_open_ratio(self) -> float: ...

    def send_ee_delta_vector(
        self,
        delta7,
        acceleration: float = 0.18,
        velocity: float = 0.04,
        wait_after_arm_s: float = 0.2,
    ): ...


class RGBDSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> dict[str, RGBDFrame | tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True)
class D435iCameraConfig:
    name: str
    serial: str | None = None


class RealSenseD435iSource:
    """Small pyrealsense2 wrapper returning timestamped RGB-D frames."""

    def __init__(
        self,
        *,
        cameras: Sequence[D435iCameraConfig],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_width: int | None = None,
        depth_height: int | None = None,
        wait_timeout_ms: int = 5000,
        read_retries: int = 3,
        retain_native_depth: bool = False,
    ) -> None:
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("At least one RealSense camera config is required")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        if (depth_width is None) != (depth_height is None):
            raise ValueError("Specify both depth_width and depth_height")
        self.depth_size = None if depth_width is None else (int(depth_width), int(depth_height))
        if min(self.width, self.height, self.fps, *(self.depth_size or (1, 1))) <= 0:
            raise ValueError("RealSense resolutions and fps must be positive")
        if len({c.name for c in self.cameras}) != len(self.cameras):
            raise ValueError("RealSense camera names must be unique")
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.read_retries = max(int(read_retries), 1)
        self.retain_native_depth = bool(retain_native_depth)
        self._rs = None
        self._pipelines = []
        self._aligns = []
        self._color_calibrations: dict[str, dict[str, object]] = {}
        self._stream_info: dict[str, dict[str, object]] = {}

    def _stream_requests(self, rs):
        devices = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
        requests, used = [], set()
        for camera in self.cameras:
            serial = str(camera.serial) if camera.serial else next((s for s in devices if s not in used), None)
            if serial not in devices:
                raise RuntimeError(f"RealSense camera {camera.name!r}, serial={serial}, is not connected; "
                                   f"available serials: {sorted(devices)}")
            if serial in used:
                raise ValueError(f"RealSense serial {serial} assigned to multiple views")
            used.add(serial)
            device = devices[serial]
            usb = device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown"
            profiles = {rs.stream.color: set(), rs.stream.depth: set()}
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    stream = profile.stream_type()
                    if stream not in profiles:
                        continue
                    wanted_format = rs.format.rgb8 if stream == rs.stream.color else rs.format.z16
                    if profile.format() == wanted_format:
                        video = profile.as_video_stream_profile()
                        profiles[stream].add((video.width(), video.height(), profile.fps()))
            requested = (self.width, self.height, self.fps)
            if requested not in profiles[rs.stream.color]:
                raise RuntimeError(f"RealSense {camera.name!r} serial={serial} USB={usb}: "
                                   f"unsupported RGB {requested}; supported: {sorted(profiles[rs.stream.color])}")
            if self.depth_size is not None:
                depth = (*self.depth_size, self.fps)
                if depth not in profiles[rs.stream.depth]:
                    raise RuntimeError(f"RealSense {camera.name!r} serial={serial} USB={usb}: "
                                       f"unsupported depth {depth}; supported: {sorted(profiles[rs.stream.depth])}")
            elif requested in profiles[rs.stream.depth]:
                depth = requested
            else:
                candidates = [p for p in profiles[rs.stream.depth] if p[2] == self.fps]
                if not candidates:
                    raise RuntimeError(f"RealSense {camera.name!r} serial={serial} USB={usb}: "
                                       f"no Z16 depth profile at {self.fps} fps; supported: {sorted(profiles[rs.stream.depth])}")
                depth = min(candidates, key=lambda p: (abs(p[0]*p[1]-self.width*self.height), p))
                warnings.warn(f"RealSense {camera.name!r} serial={serial} USB={usb}: depth {requested} "
                              f"unsupported; using {depth}, aligned to RGB {requested}", RuntimeWarning, stacklevel=2)
            info = {"serial": serial, "usb": usb, "color": list(requested), "depth": list(depth)}
            requests.append((camera.name, serial, depth, info))
        return requests

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("Missing dependency 'pyrealsense2' for Intel RealSense D435i cameras.") from exc

        self._rs = rs
        if self._pipelines:
            raise RuntimeError("RealSense source is already started")
        self._color_calibrations = {}
        self._stream_info = {}
        requests = self._stream_requests(rs)
        try:
            for name, serial, depth, info in requests:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
                config.enable_stream(rs.stream.depth, *depth[:2], rs.format.z16, depth[2])
                try:
                    profile = pipeline.start(config)
                except RuntimeError as exc:
                    raise RuntimeError(f"Failed to start RealSense {name!r} serial={serial} USB={info['usb']} "
                                       f"RGB={info['color']} depth={info['depth']}: {exc}. "
                                       "Check camera ownership and USB connection/bandwidth.") from exc
                self._pipelines.append((name, pipeline))
                color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self._color_calibrations[name] = {
                    "intrinsics": [
                        [float(color_intrinsics.fx), 0.0, float(color_intrinsics.ppx)],
                        [0.0, float(color_intrinsics.fy), float(color_intrinsics.ppy)],
                        [0.0, 0.0, 1.0],
                    ],
                    "distortion": [float(value) for value in color_intrinsics.coeffs],
                    "distortion_model": str(color_intrinsics.model).rsplit(".", 1)[-1],
                    "width": int(color_intrinsics.width),
                    "height": int(color_intrinsics.height),
                }
                self._stream_info[name] = info
                self._aligns.append(rs.align(rs.stream.color))
        except BaseException:
            try:
                self.stop()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        pipelines = self._pipelines
        self._pipelines = []
        self._aligns = []
        self._color_calibrations = {}
        self._stream_info = {}
        errors = []
        for name, pipeline in pipelines:
            try:
                pipeline.stop()
            except Exception as exc:
                errors.append((name, exc))
        if errors:
            raise RuntimeError(f"Failed to stop RealSense cameras: {[name for name, _ in errors]}") from errors[0][1]

    def get_stream_info(self) -> dict[str, dict[str, object]]:
        return {name: dict(info) for name, info in self._stream_info.items()}

    def get_color_calibrations(self) -> dict[str, dict[str, object]]:
        """Return SDK-reported colour intrinsics for the RGB-aligned depth stream."""
        if not self._color_calibrations:
            raise RuntimeError("RealSenseD435iSource is not started")
        return {name: dict(item) for name, item in self._color_calibrations.items()}

    def read(self) -> dict[str, RGBDFrame]:
        if self._rs is None or not self._pipelines:
            raise RuntimeError("RealSenseD435iSource is not started")
        frames: dict[str, RGBDFrame] = {}
        for (name, pipeline), align in zip(self._pipelines, self._aligns):
            aligned = None
            native_depth = None
            last_error: Exception | None = None
            for attempt in range(1, self.read_retries + 1):
                try:
                    sdk_frames = pipeline.wait_for_frames(self.wait_timeout_ms)
                    if self.retain_native_depth:
                        native = sdk_frames.get_depth_frame()
                        color = sdk_frames.get_color_frame()
                        if not native or not color:
                            raise RuntimeError(f"Missing native RGB-D frames for {name!r}")
                        depth_profile = native.get_profile().as_video_stream_profile()
                        intr = depth_profile.get_intrinsics()
                        extr = depth_profile.get_extrinsics_to(color.get_profile())
                        transform = np.eye(4)
                        # rs2_extrinsics.rotation is column-major, in metres.
                        transform[:3, :3] = np.asarray(extr.rotation).reshape(3, 3, order="F")
                        transform[:3, 3] = extr.translation
                        native_depth = NativeDepthFrame(
                            np.asarray(native.get_data(), dtype=np.uint16), float(native.get_units()),
                            np.array([[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]]),
                            transform, np.asarray(intr.coeffs), str(intr.model).rsplit(".", 1)[-1])
                    aligned = align.process(sdk_frames)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if attempt >= self.read_retries:
                        raise RuntimeError(
                            f"Timed out reading RealSense camera {name!r} after "
                            f"{self.read_retries} attempt(s) with wait_timeout_ms={self.wait_timeout_ms}. "
                            "Check USB bandwidth/power, camera serial env vars, and whether another process is using the camera."
                        ) from exc
            if aligned is None:
                raise RuntimeError(f"Failed to read RealSense camera {name!r}") from last_error
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"Failed to read synchronized RGB-D frame from camera {name!r}")
            rgb = np.asarray(color_frame.get_data(), dtype=np.uint8).copy()
            depth_scale = float(depth_frame.get_units())
            depth_m = np.asarray(depth_frame.get_data(), dtype=np.float32) * depth_scale
            # The camera timestamp is retained even though two independent
            # D435i devices may not share a clock.  host_timestamp_ns is the
            # common-clock arrival marker used to calculate/validate skew.
            timestamp_domain = str(depth_frame.get_frame_timestamp_domain())
            frames[name] = RGBDFrame(
                name,
                rgb,
                depth_m.astype(np.float32),
                host_timestamp_ns=time.monotonic_ns(),
                device_timestamp_ms=float(depth_frame.get_timestamp()),
                frame_number=int(depth_frame.get_frame_number()),
                timestamp_domain=timestamp_domain,
                native_depth=native_depth,
            )
        return frames


class UR7eRealSenseAdapter:
    """RealRobotAdapter implementation using UR7eVectorController and D435i RGB-D."""

    def __init__(
        self,
        *,
        controller: UR7eControllerLike,
        camera_source: RGBDSource,
        camera_names: Sequence[str] = DEFAULT_D435I_CAMERA_NAMES,
        acceleration: float = 0.18,
        velocity: float = 0.04,
        wait_after_arm_s: float = 0.2,
    ) -> None:
        if not camera_names:
            raise ValueError("At least one camera name is required")
        self.controller = controller
        self.camera_source = camera_source
        self.camera_names = tuple(str(name) for name in camera_names)
        self.acceleration = float(acceleration)
        self.velocity = float(velocity)
        self.wait_after_arm_s = float(wait_after_arm_s)
        self._closed = False
        self._latest_frames: dict[str, RGBDFrame] | None = None

    def reset(self) -> None:
        self.controller.connect()
        self.camera_source.start()
        self._closed = False
        self._latest_frames = None

    def _read_frames(self, *, refresh: bool = False) -> dict[str, RGBDFrame]:
        if refresh or self._latest_frames is None:
            raw_frames = self.camera_source.read()
            self._latest_frames = {
                name: value
                if isinstance(value, RGBDFrame)
                else RGBDFrame(name, np.asarray(value[0], dtype=np.uint8), np.asarray(value[1], dtype=np.float32))
                for name, value in raw_frames.items()
            }
        return self._latest_frames

    def get_observation(self) -> dict:
        frames = self._read_frames(refresh=True)
        qpos = np.asarray(self.controller.get_current_joints(), dtype=np.float32)
        if qpos.size != 6:
            raise RuntimeError(f"UR controller returned {qpos.size} joints; expected 6")
        gripper = np.asarray([self.controller.get_gripper_open_ratio()], dtype=np.float32)
        observation = {
            "qpos": qpos,
            "gripper": gripper,
        }
        for name in self.camera_names:
            if name not in frames:
                raise KeyError(f"Missing D435i camera frame {name!r}")
            observation[f"{name}_rgb"] = np.ascontiguousarray(frames[name].rgb)
        return observation

    def get_rgbd_frames(self) -> list[RGBDFrame]:
        frames = self._read_frames()
        rgbd_frames = []
        for name in self.camera_names:
            if name not in frames:
                raise KeyError(f"Missing D435i camera frame {name!r}")
            rgbd_frames.append(frames[name])
        return rgbd_frames

    def execute_action(self, action: np.ndarray) -> None:
        action7 = np.asarray(action, dtype=np.float32).reshape(-1)
        if action7.size != 7:
            raise ValueError(f"UR7e action must be 7D [dx_mm,dy_mm,dz_mm,droll,dpitch,dyaw,g], got {action7.size}")
        self.controller.send_ee_delta_vector(
            action7.tolist(),
            acceleration=self.acceleration,
            velocity=self.velocity,
            wait_after_arm_s=self.wait_after_arm_s,
        )
        self._latest_frames = None

    def is_done(self) -> bool:
        return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.camera_source.stop()
        finally:
            self.controller.close()
            self._closed = True


def _serials_from_env() -> dict[str, str | None]:
    return {
        "front": os.environ.get("REAL_SENSE_FRONT_SERIAL"),
        "side": os.environ.get("REAL_SENSE_SIDE_SERIAL"),
        "wrist": os.environ.get("REAL_SENSE_WRIST_SERIAL"),
    }


def create_adapter(camera_names: Sequence[str] | None = None) -> UR7eRealSenseAdapter:
    """Factory for --adapter real_scripts.ur7e_realsense_adapter:create_adapter."""
    selected_camera_names = tuple(DEFAULT_D435I_CAMERA_NAMES if camera_names is None else camera_names)
    serials = _serials_from_env()
    missing_serial_envs = [
        f"REAL_SENSE_{name.upper()}_SERIAL"
        for name in selected_camera_names
        if serials.get(name) is None
    ]
    if len(selected_camera_names) > 1 and missing_serial_envs:
        missing = ", ".join(missing_serial_envs)
        raise ValueError(f"Multiple RealSense cameras require explicit serials. Set: {missing}")
    cameras = [D435iCameraConfig(name=name, serial=serials.get(name)) for name in selected_camera_names]
    controller = UR7eVectorController(
        robot_ip=os.environ.get("UR_ROBOT_IP", ROBOT_IP),
        strict_gripper_connection=os.environ.get("UR_STRICT_GRIPPER", "1") not in {"0", "false", "False"},
        robotiq_urscript_defs_path=os.environ.get("ROBOTIQ_URSCRIPT_DEFS") or None,
    )
    camera_source = RealSenseD435iSource(
        cameras=cameras,
        width=int(os.environ.get("REAL_SENSE_WIDTH", "640")),
        height=int(os.environ.get("REAL_SENSE_HEIGHT", "480")),
        fps=int(os.environ.get("REAL_SENSE_FPS", "30")),
        wait_timeout_ms=int(os.environ.get("REAL_SENSE_WAIT_TIMEOUT_MS", "5000")),
        read_retries=int(os.environ.get("REAL_SENSE_READ_RETRIES", "3")),
    )
    return UR7eRealSenseAdapter(
        controller=controller,
        camera_source=camera_source,
        camera_names=selected_camera_names,
        acceleration=float(os.environ.get("UR_ACTION_ACCELERATION", "0.18")),
        velocity=float(os.environ.get("UR_ACTION_VELOCITY", "0.04")),
        wait_after_arm_s=float(os.environ.get("UR_WAIT_AFTER_ARM_S", "0.2")),
    )
