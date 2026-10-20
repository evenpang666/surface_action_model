"""Isolated, receive-only UR7e/Pika scene for the shared obstacle engine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import yaml

from real_scripts.live_dual_ur7e_obstacle_model import Arm, VIEWER_HTML, points_in_bounds, transform_points
from real_scripts.pika_gripper_live_reader import PikaOpeningReader
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.ur7e_collision_mesh import flange_transform, mesh_surface_samples
from safety_module.dual_camera_calibration import rigid_matrix
from safety_module.project_config import REPO_ROOT, resolve_repo_path

DEFAULT_SINGLE_CONFIG = REPO_ROOT / "configs/ur7e_single_pika.yaml"


def load_single_config(path=DEFAULT_SINGLE_CONFIG):
    config = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("schema_version") != 1
            or config.get("scene") != "ur7e_single_pika" or config.get("coordinate_frame") != "ur_base"):
        raise ValueError("Single Pika config must describe ur7e_single_pika in ur_base")
    if config.get("reconstruction_cameras") != ["front"]:
        raise ValueError("Single Pika reconstruction must use only the fixed front camera")
    serials = [config["cameras"][name]["serial"] for name in ("front", "wrist")]
    if not all(isinstance(value, str) and value for value in serials) or serials[0] == serials[1]:
        raise ValueError("Front/wrist serials must be distinct nonempty strings")
    if config["robot"].get("tool") != "pika_gripper":
        raise ValueError("Single Pika scene requires pika_gripper")
    return config


def load_scene_artifact(path, config, *, require_verified=True):
    """Reject another scene/controller's calibration even if its matrix is valid."""
    path = resolve_repo_path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing single Pika calibration: {path}. See docs/SINGLE_UR7E_PIKA.md")
    payload = path.read_bytes()
    data = json.loads(payload)
    if data.get("scene") != config["scene"] or data.get("robot_ip") != config["robot"]["ip"]:
        raise ValueError(f"{path}: calibration scene or robot IP differs")
    if require_verified and data.get("verified") is not True:
        raise ValueError(f"{path}: calibration has not been verified")
    return data, payload


def load_pika_geometry(config):
    """Accept measured geometry or the explicitly selected legacy installation.

    Inheritance preserves the old viewer's provisional assumptions. It must not
    turn them into a claim that the new mount/finger kinematics were measured.
    Camera and TCP artifacts still use the strict verified-artifact loader.
    """
    path = config["robot"]["pika_geometry"]
    data, payload = load_scene_artifact(path, config, require_verified=False)
    inherited = (
        config["robot"].get("pika_geometry_mode") == "legacy_inherited"
        and data.get("geometry_mode") == "legacy_inherited"
        and data.get("acceptance_basis") == "same_installation_as_legacy"
    )
    if inherited:
        reference = data.get("legacy_reference", {})
        if (not reference.get("source") or not reference.get("source_status")
                or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("source_sha256", "")))
                or data.get("moving_finger_model") != "legacy_parallel_step_x"):
            raise ValueError("Inherited Pika geometry must record its legacy source and finger model")
        mode = "legacy_inherited_parallel_fingers"
    else:
        if data.get("verified") is not True:
            raise ValueError(f"{path}: Pika installation has not been verified or explicitly inherited")
        if data.get("moving_finger_kinematics_verified") is not True:
            raise ValueError("Verify Pika STEP finger axes/reference opening before enabling robot masking")
        mode = "verified_parallel_fingers"
    rigid_matrix(data["flange_to_pika_step_frame"], "flange_T_pika_step")
    opening = float(data["mesh_reference_opening_mm"])
    if not np.isfinite(opening) or not 0 <= opening <= 95.:
        raise ValueError("Pika mesh reference opening must be in [0,95] mm")
    return data, payload, mode


def load_front_calibration(config):
    path = config["cameras"]["front"]["calibration"]
    data, payload = load_scene_artifact(path, config)
    serial = config["cameras"]["front"]["serial"]
    if data.get("coordinate_frame") != "ur_base" or set(data.get("cameras", {})) != {serial}:
        raise ValueError("Front calibration must contain only the configured front serial in ur_base")
    camera = data["cameras"][serial]
    if camera.get("serial") != serial or camera.get("depth_alignment") != "depth_to_color":
        raise ValueError("Front calibration serial/alignment differs")
    board = config["calibration_board"]
    rms = float(data["tcp_board_probe"]["rms_m"])
    reprojection = float(camera["charuco_reprojection_rms_px"])
    if (not np.isfinite([rms, reprojection]).all() or not 0 <= rms <= board["max_probe_rms_m"]
            or not 0 <= reprojection <= board["max_reprojection_rms_px"]):
        raise ValueError("Single Pika front calibration quality limits exceeded")
    if data.get("calibration_workflow") == "table_fixed_charuco_v2":
        if data.get("camera_quality_passed") is not True or data.get("camera_samples", 0) < 2:
            raise ValueError("Single Pika camera observation quality failed")
        if data.get("board", {}).get("legacy_pattern") is not board["legacy_pattern"]:
            raise ValueError("Front calibration board layout differs from scene config")
        for field, limit in (("camera_translation_max_spread_mm", "max_camera_spread_mm"),
                             ("final_board_motion_mm", "max_board_motion_mm"),
                             ("final_board_motion_deg", "max_board_motion_deg")):
            value = float(data[field])
            if not np.isfinite(value) or not 0 <= value <= board[limit]:
                raise ValueError(f"Single Pika front calibration quality limit exceeded: {field}")
    K = np.asarray(camera["intrinsics"], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("Invalid front camera intrinsics")
    transform = rigid_matrix(camera["camera_to_world"], "ur_base_T_front")
    return CameraCalibration("front", K, transform, camera.get("distortion")), data, payload


def pika_local_points(samples, opening_mm, reference_opening_mm):
    """Use the separate STEP body/finger meshes with verified parallel motion."""
    offset = .5 * (float(opening_mm) - float(reference_opening_mm)) / 1000.
    return np.concatenate((samples[0], samples[1] + [-offset, 0., 0.],
                           samples[2] + [offset, 0., 0.])).astype(np.float32)


def single_viewer_html():
    html = VIEWER_HTML.replace("Dual UR7e obstacle model", "UR7e Pika obstacle model")
    html = html.replace("双 UR7e", "单臂 UR7e + Pika").replace("双 Robotiq 2F-85 与双臂", "Pika 与 UR7e")
    html = html.replace("双臂", "单臂").replace("dual robots", "UR7e + Pika")
    html = html.replace("left_base", "ur_base")
    html = html.replace("const workspaces=Number.isFinite(left)&&Number.isFinite(right)?` · 左/右工作区 ${left}/${right} 点`:'';",
                        "const workspaces=Number.isFinite(left)?` · 单臂工作区 ${left} 点`:'';")
    return html


class SinglePikaObstacleScene:
    schema = "single_ur7e_pika_obstacle_snapshot_v1"
    coordinate_frame = "ur_base"
    manual_selection = True
    robot_label = "单臂"
    raw_depth_label = "raw-realsense"
    viewer_html = single_viewer_html()

    def __init__(self, args):
        config = self.project = load_single_config(args.project_config)
        if args.side_serial or args.right_workspace_bounds or args.left_tool_from_tcp or args.right_tool_from_tcp:
            raise ValueError("Single Pika scene does not accept dual-arm/side-camera overrides")
        serial = config["cameras"]["front"]["serial"]
        if args.front_serial and args.front_serial != serial:
            raise ValueError("Front serial differs from single Pika scene config")
        if args.lingbot_depth:
            raise ValueError("Front-only reconstruction requires --no-lingbot-depth: verified completion needs another measured view")
        calibration, front, camera_bytes = load_front_calibration(config)
        self.arms = (Arm("ur7e", config["robot"]["ip"], np.eye(4), "Pika gripper", np.eye(4)),)
        self.calibrations = {"front": calibration}
        points_in_bounds(np.empty((0, 3)), args.workspace_bounds)
        self.workspace_regions = [(args.workspace_bounds, np.eye(4))]
        self.report = {"scene": config["scene"], "camera_serial": serial, "camera_serials": {"front": serial},
                       "source": str(resolve_repo_path(config["cameras"]["front"]["calibration"])),
                       "transform_convention": "parent_T_child", "touch_rms_mm": front["tcp_board_probe"]["rms_m"] * 1000.,
                       "sha256": hashlib.sha256(camera_bytes).hexdigest()}
        self.flange_to_tcp = {}
        if args.no_robot:
            return
        tcp, tcp_bytes = load_scene_artifact(config["robot"]["active_tcp_calibration"], config)
        # The camera probe must use the same active TCP as live geometry.
        if front.get("active_tcp_sha256") != hashlib.sha256(tcp_bytes).hexdigest():
            raise ValueError("Active TCP changed since front calibration; recalibrate the camera")
        self.flange_to_tcp = {"ur7e": rigid_matrix(tcp["matrix"], "flange_T_active_tcp")}
        geometry, geometry_bytes, geometry_mode = load_pika_geometry(config)
        self.pika_mount = rigid_matrix(geometry["flange_to_pika_step_frame"], "flange_T_pika_step")
        self.reference_opening_mm = float(geometry["mesh_reference_opening_mm"])
        self.report.update({
            "sha256": hashlib.sha256(camera_bytes + tcp_bytes + geometry_bytes).hexdigest(),
            "pika_geometry": geometry_mode,
            "pika_geometry_verified": (geometry.get("verified") is True
                                       and geometry.get("moving_finger_kinematics_verified") is True),
            "pika_geometry_reference": geometry.get("legacy_reference"),
            "gripper_position_convention": "opening_ratio_0_closed_1_open"})
        import trimesh
        self.pika_samples = []
        mesh_hash = hashlib.sha256()
        for name in ("pika_gripper_body_collision.stl", "finger_a_candidate.stl", "finger_b_candidate.stl"):
            path = resolve_repo_path(config["robot"]["pika_collision_dir"]) / name
            mesh_hash.update(path.read_bytes())
            mesh = trimesh.load_mesh(path, process=False)
            mesh.vertices = np.asarray(mesh.vertices, dtype=float) * .001
            self.pika_samples.append(mesh_surface_samples(mesh, samples_per_face=args.samples_per_face))
        self.report["pika_mesh_sha256"] = mesh_hash.hexdigest()

    def workspace_masks(self, points):
        mask = points_in_bounds(points, self.workspace_regions[0][0])
        return mask, np.zeros(len(points), dtype=bool), mask

    def open_grippers(self, args):
        robot = self.project["robot"]
        reader = PikaOpeningReader(args.pika_gripper_port or os.environ.get("PIKA_GRIPPER_PORT") or robot["gripper_port"],
                                   max_state_age_s=robot["gripper_max_state_age_s"])
        try:
            reader.connect()
        except Exception:
            reader.close()
            raise
        return {"ur7e": reader}

    def read_grippers(self, grippers):
        opening = grippers["ur7e"].opening_mm()
        return {"ur7e": opening / 95.}, f"Pika opening {opening:.1f} mm (read-only)"

    def tool_points(self, arm, q, position):
        local = pika_local_points(self.pika_samples, float(position) * 95., self.reference_opening_mm)
        return transform_points(local, flange_transform(q) @ self.pika_mount)
