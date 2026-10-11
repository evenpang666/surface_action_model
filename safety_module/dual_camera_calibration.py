"""One validated calibration graph for dual-arm FK and front/side perception.

All transforms follow parent_T_child: multiply a child-frame point to obtain
its parent-frame coordinates. Names containing 'to' never override frames.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from safety_module.project_config import resolve_repo_path


def rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name}: invalid homogeneous matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError(f"{name}: rotation is not proper orthonormal")
    return matrix


def transform_error(first, second):
    """Translation difference in mm, rotation difference in degrees."""
    return (float(np.linalg.norm(first[:3, 3]-second[:3, 3])*1000),
            float(np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())))


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _bounded(value, limit, name):
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > limit:
        raise ValueError(f"{name}: {value} exceeds limit {limit}")


def load_camera_file(path, *, parent=None, child=None):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("calibrated") is not True:
        raise ValueError(f"Uncalibrated camera: {path}")
    if parent is not None and data.get("parent_frame") != parent:
        raise ValueError(f"{path}: expected parent_frame {parent}")
    if child is not None and data.get("child_frame") != child:
        raise ValueError(f"{path}: expected child_frame {child}")
    matrix = rigid_matrix(data["matrix"], str(path))
    limits = data["quality_limits"]
    _bounded(data["touch_rms_mm"], _positive(limits["max_touch_rms_mm"], "touch RMS limit"), f"{path}: touch RMS")
    # New Quest3 exports contain these checks; legacy installed files do not.
    if "camera_quality_passed" in data and data["camera_quality_passed"] is not True:
        raise ValueError(f"{path}: camera quality failed")
    for field, limit in (("camera_translation_max_spread_mm", "max_camera_spread_mm"),
                         ("final_board_motion_mm", "max_board_motion_mm"),
                         ("final_board_motion_deg", "max_board_motion_deg")):
        if field in data:
            _bounded(data[field], _positive(limits[limit], limit), f"{path}: {field}")
    intr = data["intrinsics"]
    fx, fy = (_positive(intr[key], key) for key in ("fx", "fy"))
    cx, cy = float(intr["ppx"]), float(intr["ppy"])
    if not np.isfinite([cx, cy]).all() or not data.get("camera_serial"):
        raise ValueError(f"{path}: invalid intrinsics/serial")
    K = np.asarray([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    return matrix, K, str(data["camera_serial"]), data


@dataclass(frozen=True)
class DualCameraCalibration:
    path: Path
    digest: str
    right_base_to_left_base: np.ndarray
    camera_to_left_base: dict[str, np.ndarray]
    intrinsics: dict[str, np.ndarray]
    serials: dict[str, str]
    source_paths: dict[str, Path]
    source_data: dict[str, dict]
    residuals: dict[str, dict]

    def report(self):
        return {"source": str(self.path), "sha256": self.digest,
                "transform_convention": "parent_T_child",
                "right_base_to_left_base": self.right_base_to_left_base.tolist(),
                "camera_to_left_base": {k: v.tolist() for k, v in self.camera_to_left_base.items()},
                "camera_serials": self.serials, "closure_residuals": self.residuals}


def load_dual_camera_calibration(path, *, expected_serials=None):
    path = Path(path).expanduser().resolve()
    payload = path.read_bytes()
    data = yaml.safe_load(payload)
    if data.get("calibrated") is not True:
        raise ValueError("Dual-arm camera calibration is not calibrated")
    if data.get("transform_convention") != "parent_T_child maps child coordinates into parent coordinates":
        raise ValueError("Unsupported dual-arm transform convention")
    limits = data["quality_limits"]
    translation_limit = _positive(limits["max_translation_mm"], "translation limit")
    rotation_limit = _positive(limits["max_rotation_deg"], "rotation limit")
    residuals = {}

    def check(name, first, second):
        mm, deg = transform_error(first, second)
        _bounded(mm, translation_limit, f"{name} translation mm")
        _bounded(deg, rotation_limit, f"{name} rotation deg")
        residuals[name] = {"translation_mm": mm, "rotation_deg": deg}

    def transform(key, parent, child):
        item = data[key]
        if item.get("parent_frame") != parent or item.get("child_frame") != child:
            raise ValueError(f"{key}: expected {parent}_T_{child}")
        return rigid_matrix(item["matrix"], key)

    base = transform("left_base_to_right_base", "left_base", "right_base")
    front_frame, side_frame = "front_camera_color_optical_frame", "side_camera_color_optical_frame"
    front_from_side = transform("front_camera_to_side_camera", front_frame, side_frame)
    source_paths, source_data, matrices, intrinsics, serials = {}, {}, {}, {}, {}
    for arm in ("left", "right"):
        for camera, frame in (("front", front_frame), ("side", side_frame)):
            key = f"{arm}/{camera}"
            reference = data["source_calibrations"][key]
            source = Path(reference["path"]).expanduser()
            if not source.is_absolute():
                source = path.parent/source
            source = source.resolve()
            actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual_hash != reference["sha256"]:
                raise ValueError(f"Source calibration SHA256 differs: {key}: {source}")
            matrix, K, serial, details = load_camera_file(source, parent=f"{arm}_base", child=frame)
            source_paths[key], source_data[key], matrices[key] = source, details, matrix
            if camera in serials and serials[camera] != serial:
                raise ValueError(f"The two arms reference different {camera} cameras")
            if expected_serials and str(expected_serials[camera]) != serial:
                raise ValueError(f"Hardware {camera} serial differs from calibration")
            serials[camera] = serial
            if arm == "left":
                intrinsics[camera] = K
    if serials["front"] == serials["side"]:
        raise ValueError("Front and side must be different cameras")
    via_front = matrices["left/front"] @ np.linalg.inv(matrices["right/front"])
    via_side = matrices["left/side"] @ np.linalg.inv(matrices["right/side"])
    camera_via_left = np.linalg.inv(matrices["left/front"]) @ matrices["left/side"]
    camera_via_right = np.linalg.inv(matrices["right/front"]) @ matrices["right/side"]
    for name in ("base_via_front_vs_side", "cameras_via_left_vs_right"):
        recorded = data["consistency"][name]
        if recorded.get("passed") is not True:
            raise ValueError(f"Recorded calibration consistency failed: {name}")
        _bounded(recorded["translation_error_mm"], translation_limit, name)
        _bounded(recorded["rotation_error_deg"], rotation_limit, name)
    check("base_via_front_vs_side", via_front, via_side)
    check("cameras_via_left_vs_right", camera_via_left, camera_via_right)
    check("aggregate_base_vs_front", base, via_front)
    check("aggregate_base_vs_side", base, via_side)
    check("aggregate_cameras_vs_left", front_from_side, camera_via_left)
    check("aggregate_cameras_vs_right", front_from_side, camera_via_right)
    for view, derived in (("front", via_front), ("side", via_side)):
        check(f"recorded_base_via_{view}", transform(f"left_base_to_right_base_via_{view}", "left_base", "right_base"), derived)
    # Anchor in measured left/front; use the aggregate relative transforms to
    # obtain an internally coherent camera/base graph for all runtime callers.
    camera_to_left = {"front": matrices["left/front"], "side": matrices["left/front"] @ front_from_side}
    for camera in ("front", "side"):
        check(f"canonical_{camera}_vs_left_source", camera_to_left[camera], matrices[f"left/{camera}"])
        check(f"canonical_{camera}_vs_right_source", camera_to_left[camera], base @ matrices[f"right/{camera}"])
    return DualCameraCalibration(path, hashlib.sha256(payload).hexdigest(), base, camera_to_left,
                                 intrinsics, serials, source_paths, source_data, residuals)


def calibration_from_project(project):
    """None only for explicitly legacy configs lacking an aggregate file."""
    configured = project["safety"].get("dual_arm_camera_calibration")
    if not configured:
        return None
    hardware = yaml.safe_load(resolve_repo_path(project["quest3"]["hardware_config"]).read_text())
    return load_dual_camera_calibration(resolve_repo_path(configured), expected_serials={
        name: hardware["cameras"][name]["serial"] for name in ("front", "side")})


def validate_calibration_identity(snapshot, expected_digest):
    if expected_digest is not None and snapshot.get("calibration", {}).get("sha256") != expected_digest:
        raise RuntimeError("Obstacle snapshot calibration differs from fixed-point FK; restart perception with the same project config")
