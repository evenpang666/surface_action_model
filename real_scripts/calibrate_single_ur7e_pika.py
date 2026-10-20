#!/usr/bin/env python3
"""Calibrate only the single UR7e/Pika scene using receive-only RTDE."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_scripts.single_camera_charuco_calibration import CalibrationQualityError, run_calibration
from real_scripts.live_dual_ur7e_obstacle_model import rotvec_transform, validate_active_tcp
from real_scripts.read_dual_active_tcp_calibration import _mean_transform
from real_scripts.single_ur7e_pika_scene import DEFAULT_SINGLE_CONFIG, load_scene_artifact, load_single_config
from real_scripts.ur7e_collision_mesh import flange_transform
from safety_module.dual_camera_calibration import rigid_matrix
from safety_module.project_config import resolve_repo_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tcp", "camera"), help="Read configured TCP first, then calibrate the fixed front camera.")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_SINGLE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/single_pika/calibration")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--camera-samples", type=int, default=30)
    parser.add_argument("--touch-corner-ids", type=int, nargs=4, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume saved observation/contacts only while the board is still fixed.")
    parser.add_argument("--retouch-corner-ids", type=int, nargs="+", default=[])
    parser.add_argument("--headless", action="store_true", help="Save previews and lock from terminal instead of a live OpenCV window.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-s", type=float, default=.05)
    args = parser.parse_args(argv)
    if args.samples < 2 or not np.isfinite(args.interval_s) or args.interval_s < 0:
        parser.error("samples must be >=2 and interval-s finite and nonnegative")
    if args.camera_samples < 2 or min(args.width, args.height, args.fps) < 1:
        parser.error("camera-samples must be >=2 and stream dimensions/fps positive")
    if args.retouch_corner_ids and not args.resume:
        parser.error("--retouch-corner-ids requires --resume while the board remains fixed")
    return args


def read_active_tcp(robot_ip, *, samples, interval_s):
    from rtde_receive import RTDEReceiveInterface
    receiver = RTDEReceiveInterface(robot_ip)
    transforms, joints = [], []
    try:
        for _ in range(samples):
            q = np.asarray(receiver.getActualQ(), dtype=float)
            tcp = np.asarray(receiver.getActualTCPPose(), dtype=float)
            if q.shape != (6,) or tcp.shape != (6,) or not np.isfinite([q, tcp]).all():
                raise RuntimeError("Invalid receive-only UR7e state")
            joints.append(q)
            transforms.append(np.linalg.inv(flange_transform(q)) @ rotvec_transform(tcp))
            time.sleep(interval_s)
    finally:
        receiver.disconnect()
    if np.max(np.abs(np.stack(joints) - joints[0])) > .001:
        raise RuntimeError("Keep UR7e stationary while measuring the configured TCP")
    matrix, mm, deg = _mean_transform(np.stack(transforms))
    if mm > .5 or deg > .2:
        raise RuntimeError(f"Unstable TCP measurements: {mm:.3f} mm / {deg:.3f} deg")
    return {"matrix": matrix.tolist(), "sample_count": samples,
            "translation_max_spread_mm": mm, "rotation_max_spread_deg": deg}


def check_active_tcp(config, matrix):
    from rtde_receive import RTDEReceiveInterface
    receiver = RTDEReceiveInterface(config["robot"]["ip"])
    try:
        q = np.asarray(receiver.getActualQ(), dtype=float)
        tcp = np.asarray(receiver.getActualTCPPose(), dtype=float)
        if q.shape != (6,) or tcp.shape != (6,) or not np.isfinite([q, tcp]).all():
            raise RuntimeError("Invalid receive-only UR7e state")
        validate_active_tcp(q, tcp, matrix, arm_name="ur7e")
    finally:
        receiver.disconnect()


def write_artifact(path, payload):
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[done] {path}")


def main(argv=None):
    args = parse_args(argv)
    config = load_single_config(args.project_config)
    metadata = {"scene": config["scene"], "robot_ip": config["robot"]["ip"],
                "created_utc": datetime.now(timezone.utc).isoformat(), "verified": True}
    if args.stage == "tcp":
        result = read_active_tcp(config["robot"]["ip"], samples=args.samples, interval_s=args.interval_s)
        write_artifact(config["robot"]["active_tcp_calibration"],
                       {**metadata, **result, "method": "receive_only_inv_base_T_flange_times_base_T_active_tcp"})
        return
    tcp, tcp_bytes = load_scene_artifact(config["robot"]["active_tcp_calibration"], config)
    matrix = rigid_matrix(tcp["matrix"], "flange_T_active_tcp")
    check_active_tcp(config, matrix)
    board = config["calibration_board"]
    calibration_args = argparse.Namespace(
        serials=[config["cameras"]["front"]["serial"]], output_dir=args.output_dir, project_config=args.project_config,
        output=args.output_dir / "camera_calibration.json", robot_ip=config["robot"]["ip"],
        width=args.width, height=args.height, fps=args.fps, warmup_frames=30, wait_timeout_ms=5000,
        dictionary=board["dictionary"], squares_x=board["squares_x"], squares_y=board["squares_y"],
        square_length_m=board["square_length_m"], marker_length_m=board["marker_length_m"],
        min_charuco_corners=6, max_reprojection_rms_px=board["max_reprojection_rms_px"],
        max_probe_rms_m=board["max_probe_rms_m"], legacy_pattern=board["legacy_pattern"],
        camera_samples=args.camera_samples, verification_samples=10,
        max_camera_spread_mm=board["max_camera_spread_mm"],
        max_board_motion_mm=board["max_board_motion_mm"], max_board_motion_deg=board["max_board_motion_deg"],
        touch_corner_ids=args.touch_corner_ids, resume=args.resume, retouch_corner_ids=args.retouch_corner_ids,
        headless=args.headless, active_tcp_sha256=hashlib.sha256(tcp_bytes).hexdigest())
    result = run_calibration(calibration_args)
    check_active_tcp(config, matrix)
    if resolve_repo_path(config["robot"]["active_tcp_calibration"]).read_bytes() != tcp_bytes:
        raise RuntimeError("TCP artifact changed during camera calibration; repeat calibration")
    write_artifact(config["cameras"]["front"]["calibration"],
                   {**result, **metadata, "active_tcp_sha256": hashlib.sha256(tcp_bytes).hexdigest(),
                    "board": board})


def cli(argv=None):
    try:
        main(argv)
    except CalibrationQualityError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
