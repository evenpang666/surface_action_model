#!/usr/bin/env python3
"""Export front/side D455 RGB-D calibration without opening camera streams.

Select the native sensor profiles used during recording, before the Quest3
recorder's crop/resize. This reads factory calibration from the configured
devices; it does not estimate camera-to-base transforms or modify devices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from real_scripts.quest3_depth import _lens
from safety_module.dual_camera_calibration import rigid_matrix


def _profile_size(value: str) -> tuple[int, int, int]:
    try:
        profile = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Profile must be WIDTHxHEIGHTxFPS") from exc
    if len(profile) != 3 or min(profile) <= 0:
        raise argparse.ArgumentTypeError("Profile must be positive WIDTHxHEIGHTxFPS")
    return profile


def _select_profile(device, rs, stream, format_, requested):
    width, height, fps = requested
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != stream or profile.format() != format_ or profile.fps() != fps:
                continue
            video = profile.as_video_stream_profile()
            if (video.width(), video.height()) == (width, height):
                return video
    raise ValueError(f"Device lacks {stream}/{format_} profile {width}x{height}x{fps}; "
                     "specify the actual native recording profile")


def _intrinsics(profile):
    intr = profile.get_intrinsics()
    model = str(intr.model).rsplit(".", 1)[-1]
    spec = {"width": int(intr.width), "height": int(intr.height),
            "fx": float(intr.fx), "fy": float(intr.fy),
            "ppx": float(intr.ppx), "ppy": float(intr.ppy),
            "distortion": [float(value) for value in intr.coeffs],
            "distortion_model": model, "sdk_distortion_model": model,
            "image_transform": "center_crop_resize"}
    _lens(spec)
    return spec


def query_calibrations(rs, *, serials: dict[str, str], color_profiles: dict[str, tuple[int, int, int]],
                       depth_profile: tuple[int, int, int]) -> dict:
    context = rs.context()
    devices = {str(device.get_info(rs.camera_info.serial_number)): device
               for device in context.query_devices()}
    cameras = {}
    for view in ("front", "side"):
        serial = serials[view]
        if serial not in devices:
            raise RuntimeError(f"{view} D455 serial {serial} is not connected/accessible")
        device = devices[serial]
        name = str(device.get_info(rs.camera_info.name))
        if "D455" not in name:
            raise ValueError(f"{view} serial {serial} identifies {name}, expected D455")
        depth = _select_profile(device, rs, rs.stream.depth, rs.format.z16, depth_profile)
        color = _select_profile(device, rs, rs.stream.color, rs.format.rgb8, color_profiles[view])
        extr = depth.get_extrinsics_to(color)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(extr.rotation, dtype=float).reshape(3, 3, order="F")
        transform[:3, 3] = extr.translation
        rigid_matrix(transform, f"{view}.depth_to_color")
        scale = float(device.first_depth_sensor().get_depth_scale())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{view} has invalid depth scale")
        cameras[view] = {"camera_serial": serial, "camera_model": name,
                         "depth_intrinsics": _intrinsics(depth), "color_intrinsics": _intrinsics(color),
                         "depth_to_color": transform.tolist(), "depth_scale_m": scale,
                         "depth_profile": list(depth_profile), "color_profile": list(color_profiles[view])}
    return {"schema": "quest3_recorded_rgbd_calibration_v1",
            "queried_utc": datetime.now(timezone.utc).isoformat(),
            "source": "device_sdk_factory_calibration",
            "transform_convention": "P_color = depth_to_color @ P_depth; translation in metres",
            "cameras": cameras}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "configs/hardware/calibration/quest3_rgbd.yaml")
    parser.add_argument("--front-color-profile", type=_profile_size, default=(1280, 720, 30))
    parser.add_argument("--side-color-profile", type=_profile_size, default=(640, 480, 30))
    parser.add_argument("--depth-profile", type=_profile_size, default=(640, 480, 30))
    args = parser.parse_args()
    project, _ = load_project_config(args.project_config)
    hardware = yaml.safe_load(resolve_repo_path(project["quest3"]["hardware_config"]).read_text(encoding="utf-8"))
    serials = {view: str(hardware["cameras"][view]["serial"]) for view in ("front", "side")}
    import pyrealsense2 as rs
    payload = query_calibrations(rs, serials=serials,
                                 color_profiles={"front": args.front_color_profile, "side": args.side_color_profile},
                                 depth_profile=args.depth_profile)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve previously queried/recording-specific calibrations.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) if output.suffix == ".json" else yaml.safe_dump(payload, sort_keys=False))
    print(json.dumps({"output": str(output), "cameras": {view: {
        "serial": data["camera_serial"], "color_profile": data["color_profile"],
        "depth_profile": data["depth_profile"], "depth_scale_m": data["depth_scale_m"]}
        for view, data in payload["cameras"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
