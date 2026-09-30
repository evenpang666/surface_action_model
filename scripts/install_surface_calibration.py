#!/usr/bin/env python3
"""Validate two new ChArUco sessions and install front/side deployment extrinsics.

The board and cameras must be fixed between left/right probing sessions.
This script reads saved measurements only and never connects to a robot.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safety_module.project_config import load_project_config, resolve_repo_path, DEFAULT_PROJECT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-session", type=Path, required=True)
    parser.add_argument("--right-session", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    args = parser.parse_args()
    config, config_path = load_project_config(args.project_config)
    hardware_path = resolve_repo_path(config["quest3"]["hardware_config"])
    hardware = yaml.safe_load(hardware_path.read_text())
    front, side = (str(hardware["cameras"][name]["serial"]) for name in ("front", "side"))
    left, right = (json.loads(p.read_text()) for p in (args.left_session, args.right_session))
    prepared = []
    for session, serial, base, key in ((left, front, "left_base", "front_calibration"),
                                      (left, side, "left_base", "side_calibration"),
                                      (right, front, "right_base", "right_front_calibration")):
        camera = session["cameras"][serial]
        rms = float(session["tcp_board_probe"]["rms_m"])
        if not np.isfinite(rms) or rms > .003:
            raise ValueError("Probe RMS must be <= 3mm")
        matrix = np.asarray(camera["camera_to_world"], float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(matrix[:3, :3]) < .99:
            raise ValueError("Invalid camera extrinsic")
        K = np.asarray(camera["intrinsics"], float)
        reprojection = float(camera["charuco_reprojection_rms_px"])
        if not np.isfinite(reprojection) or reprojection > .8:
            raise ValueError("Camera reprojection RMS must be <= .8px")
        payload = dict(parent_frame=base, child_frame=f"{serial}_color_optical_frame", matrix=matrix.tolist(), calibrated=True,
                       camera_serial=serial, touch_rms_mm=rms * 1000, quality_limits={"max_touch_rms_mm": 3.},
                       intrinsics=dict(fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2]),
                                       width=int(camera["width"]), height=int(camera["height"])))
        path = resolve_repo_path(config["safety"][key])
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite calibration: {path}")
        prepared.append((path, payload))
    for path, payload in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    hardware["arms"]["left_arm"]["front_calibration"] = str(prepared[0][0].relative_to(hardware_path.parent))
    hardware["arms"]["right_arm"]["front_calibration"] = str(prepared[2][0].relative_to(hardware_path.parent))
    hardware_path.write_text(yaml.safe_dump(hardware, sort_keys=False))
    config["safety"]["camera_calibration_valid"] = True
    config["safety"].pop("calibration_invalid_reason", None)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print("Installed new calibration. Rebuild geometry identities and validate checkpoint compatibility before deployment.")


if __name__ == "__main__":
    main()
