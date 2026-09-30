#!/usr/bin/env python3
"""Run RealSense intrinsic, ChArUco extrinsic, and TCP board-probe calibration.

The ChArUco board must stay rigidly fixed throughout this program.  The final
output follows the repository convention: ``camera_to_world`` is
``^ur_base T_camera``.  A single selected camera is valid without point-cloud
fusion; two or more cameras are calibrated into the same UR-base frame and the
output explicitly enables fusion.

This program reads the *configured TCP* pose from the UR controller while the
operator touches known ChArUco corners.  Consequently it calibrates the board
in the UR base frame (and therefore the cameras), not a physical tool-offset
on the UR controller.  Calibrate the tool offset in PolyScope before using a
TCP probe for this procedure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.calibrate_d435i_to_ur_base import (  # noqa: E402
    camera_to_base_from_board_pose,
    rigid_transform_from_correspondences,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource  # noqa: E402


class ReadOnlyTCPProbe:
    """Calibration must never create an RTDE control or gripper connection."""
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.receiver = None

    def connect(self):
        from rtde_receive import RTDEReceiveInterface
        self.receiver = RTDEReceiveInterface(self.robot_ip)

    def get_current_tcp_pose(self):
        value = np.asarray(self.receiver.getActualTCPPose(), dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError("Invalid measured active TCP pose")
        return value

    def close(self):
        if self.receiver is not None:
            self.receiver.disconnect()


@dataclass(frozen=True)
class ProbeCorner:
    """One visible, non-edge ChArUco corner used for the TCP probe."""

    charuco_id: int
    board_point_m: np.ndarray
    label: str


@dataclass(frozen=True)
class CharucoObservation:
    board_to_camera: np.ndarray
    reprojection_rms_px: float
    corner_count: int
    detected_ids: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serials", nargs="+", required=True, metavar="SERIAL", help="RealSense serial numbers to calibrate.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for all session artifacts.")
    parser.add_argument("--output", type=Path, default=None, help="Final UR-base calibration JSON (default: OUTPUT_DIR/camera_calibration.json).")
    parser.add_argument("--robot-ip", required=True, help="UR controller IP used only to read the configured TCP pose.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--wait-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary printed on the ChArUco board (default: DICT_4X4_50).",
    )
    parser.add_argument("--squares-x", type=int, required=True)
    parser.add_argument("--squares-y", type=int, required=True)
    parser.add_argument("--square-length-m", type=float, required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument("--min-charuco-corners", type=int, default=6, help="Minimum interpolated corners required in every camera.")
    parser.add_argument("--max-reprojection-rms-px", type=float, default=0.8)
    parser.add_argument("--max-probe-rms-m", type=float, default=0.003, help="Maximum RMS for the four TCP-to-board correspondences.")
    parser.add_argument("--max-probe-retries", type=int, default=3, help="Maximum automatic requests to re-probe a suspect point.")
    return parser.parse_args()


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("This calibration requires opencv-contrib-python with cv2.aruco.") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build lacks cv2.aruco; install opencv-contrib-python.")
    return cv2


def create_charuco_board(
    *, squares_x: int, squares_y: int, square_length_m: float, marker_length_m: float, dictionary_name: str
):
    cv2 = _require_cv2()
    normalized_dictionary_name = str(dictionary_name).upper()
    dictionary_id = getattr(cv2.aruco, normalized_dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(
            f"Unknown OpenCV ArUco dictionary {dictionary_name!r} "
            f"(normalized to {normalized_dictionary_name!r})"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length_m), float(marker_length_m), dictionary)
    return cv2.aruco.CharucoBoard_create(int(squares_x), int(squares_y), float(square_length_m), float(marker_length_m), dictionary)


def board_chessboard_corners(board: Any) -> np.ndarray:
    """Return OpenCV's ChArUco 3D-corner order for both supported APIs."""
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    return np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)


def select_inner_probe_corners(board: Any, *, squares_x: int, squares_y: int) -> tuple[ProbeCorner, ...]:
    """Choose four spatially separated internal ChArUco corners.

    IDs match the IDs returned by ChArUco detection, so the command-line
    instructions and the coordinates used in the fit can never drift apart.
    """
    cols, rows = int(squares_x) - 1, int(squares_y) - 1
    if cols < 4 or rows < 4:
        raise ValueError("Four inner probe corners require --squares-x and --squares-y to be at least 5")
    corners = board_chessboard_corners(board)
    locations = ((1, 1, "upper-left"), (cols - 2, 1, "upper-right"), (1, rows - 2, "lower-left"), (cols - 2, rows - 2, "lower-right"))
    result = []
    for col, row, label in locations:
        charuco_id = row * cols + col
        result.append(ProbeCorner(charuco_id, corners[charuco_id].copy(), f"{label} inner corner (ChArUco ID {charuco_id})"))
    return tuple(result)


def _detect_charuco(board: Any, rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    # A modest upscale makes fixed boards at the far end of the workspace more
    # reliable without altering the underlying geometry.
    scale = 2.0
    detection_image = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.005
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), parameters)
        corners, ids, _, _ = detector.detectBoard(detection_image)
        return corners, ids
    dictionary = board.getDictionary() if hasattr(board, "getDictionary") else board.dictionary
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(detection_image, dictionary, parameters=parameters)
    if marker_ids is None:
        return None, None
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, detection_image, board)
    return corners, ids


def observe_charuco(
    board: Any,
    rgb: np.ndarray,
    *,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> CharucoObservation:
    """Estimate ``^camera T_board`` and report ChArUco placement quality."""
    cv2 = _require_cv2()
    corners, ids = _detect_charuco(board, rgb)
    if corners is None or ids is None or len(ids) == 0:
        return CharucoObservation(np.eye(4), float("inf"), 0, frozenset())
    object_points, image_points = board.matchImagePoints(corners, ids)
    scale = 2.0
    camera_matrix = np.asarray(intrinsics, dtype=np.float64).copy()
    camera_matrix[:2, :] *= scale
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        np.asarray(distortion, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return CharucoObservation(np.eye(4), float("inf"), int(len(ids)), frozenset(int(v) for v in ids.reshape(-1)))
    reprojection, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, np.asarray(distortion, dtype=np.float64).reshape(-1, 1))
    rms_px = float(np.sqrt(np.mean(np.sum((reprojection.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    board_to_camera = np.eye(4, dtype=np.float64)
    board_to_camera[:3, :3] = rotation
    board_to_camera[:3, 3] = tvec.reshape(3)
    return CharucoObservation(board_to_camera, rms_px, int(len(ids)), frozenset(int(v) for v in ids.reshape(-1)))


def placement_problem(
    observation: CharucoObservation,
    *,
    required_ids: Iterable[int],
    min_corners: int,
    max_rms_px: float,
) -> str | None:
    if observation.corner_count < min_corners:
        return f"only {observation.corner_count} ChArUco corners detected; need at least {min_corners}"
    missing = sorted(set(required_ids) - observation.detected_ids)
    if missing:
        return f"the four TCP probe corners are not all visible; missing ChArUco IDs {missing}"
    if not np.isfinite(observation.reprojection_rms_px) or observation.reprojection_rms_px > max_rms_px:
        return f"reprojection RMS is {observation.reprojection_rms_px:.3f}px; limit is {max_rms_px:.3f}px"
    return None


def validate_tcp_correspondences(
    board_points_m: Sequence[Sequence[float]], base_points_m: Sequence[Sequence[float]], *, max_rms_m: float
) -> dict[str, Any]:
    """Fit ``^base T_board`` and identify a bad probe point if the fit fails."""
    board_points = np.asarray(board_points_m, dtype=np.float64).reshape(-1, 3)
    base_points = np.asarray(base_points_m, dtype=np.float64).reshape(-1, 3)
    if board_points.shape != base_points.shape or board_points.shape[0] != 4:
        raise ValueError("Exactly four paired board/base points are required")
    transform = rigid_transform_from_correspondences(board_points, base_points)
    fitted = (transform[:3, :3] @ board_points.T).T + transform[:3, 3]
    residuals = np.linalg.norm(base_points - fitted, axis=1)
    rms_m = float(np.sqrt(np.mean(residuals**2)))

    # With a single misplaced touch, the leave-one-out prediction error gives
    # a substantially clearer culprit than residuals of an all-points fit.
    loo_errors = []
    for index in range(4):
        keep = np.arange(4) != index
        leave_out_transform = rigid_transform_from_correspondences(board_points[keep], base_points[keep])
        prediction = leave_out_transform[:3, :3] @ board_points[index] + leave_out_transform[:3, 3]
        loo_errors.append(float(np.linalg.norm(base_points[index] - prediction)))
    suspect_index = int(np.argmax(loo_errors))
    return {
        "usable": rms_m <= float(max_rms_m),
        "board_to_base": transform,
        "rms_m": rms_m,
        "residuals_m": residuals,
        "leave_one_out_errors_m": np.asarray(loo_errors, dtype=np.float64),
        "suspect_index": suspect_index,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _intrinsics_payload(source: RealSenseD435iSource, camera_names: Sequence[str]) -> dict[str, Any]:
    values = source.get_color_calibrations()
    return {
        "camera_frame": "color",
        "depth_alignment": "depth_to_color",
        "cameras": {name: values[name] for name in camera_names},
    }


def _save_placement_preview(
    path: Path,
    *,
    board: Any,
    rgb: np.ndarray,
    camera_name: str,
    attempt: int,
    observation: CharucoObservation,
) -> None:
    """Save an RGB placement-check frame, with detected ChArUco corners overlaid."""
    cv2 = _require_cv2()
    preview = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    corners, ids = _detect_charuco(board, rgb)
    if corners is not None and ids is not None and len(ids):
        # Detection runs on a 2x upscaled image; map its coordinates back to
        # the native-resolution RGB frame before drawing.
        cv2.aruco.drawDetectedCornersCharuco(preview, corners / 2.0, ids)
    label = f"{camera_name} | attempt {attempt} | ChArUco corners: {observation.corner_count}"
    cv2.rectangle(preview, (0, 0), (min(preview.shape[1], 620), 32), (0, 0, 0), thickness=-1)
    cv2.putText(preview, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Failed to save ChArUco placement preview to {path}")


def _wait_for_valid_placement(
    source: RealSenseD435iSource,
    intrinsics_payload: dict[str, Any],
    board: Any,
    probe_corners: Sequence[ProbeCorner],
    *, min_corners: int, max_rms_px: float, preview_dir: Path,
) -> dict[str, CharucoObservation]:
    required_ids = [corner.charuco_id for corner in probe_corners]
    attempt = 0
    while True:
        attempt += 1
        frames = source.read()
        observations: dict[str, CharucoObservation] = {}
        problems: dict[str, str] = {}
        for name, item in intrinsics_payload["cameras"].items():
            observation = observe_charuco(
                board,
                frames[name].rgb,
                intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                distortion=np.asarray(item.get("distortion", np.zeros(5)), dtype=np.float64),
            )
            observations[name] = observation
            preview_path = preview_dir / f"attempt_{attempt:03d}_{name}_rgb.png"
            _save_placement_preview(
                preview_path,
                board=board,
                rgb=frames[name].rgb,
                camera_name=name,
                attempt=attempt,
                observation=observation,
            )
            print(f"[preview] saved RGB placement frame: {preview_path}")
            problem = placement_problem(observation, required_ids=required_ids, min_corners=min_corners, max_rms_px=max_rms_px)
            if problem:
                problems[name] = problem
        if not problems:
            print(f"[ok] all {len(observations)} camera(s) passed ChArUco placement check on attempt {attempt}.")
            return observations
        print("[retry] ChArUco board placement is not ready:")
        for name, problem in problems.items():
            print(f"  - {name}: {problem}")
        input("Adjust the board/cameras for visibility, then press Enter to test all cameras again (do not move it after it passes): ")


def _probe_board_with_tcp(
    controller: ReadOnlyTCPProbe,
    probe_corners: Sequence[ProbeCorner],
    *, max_rms_m: float, max_retries: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    board_points = np.asarray([corner.board_point_m for corner in probe_corners], dtype=np.float64)
    base_points = np.empty((4, 3), dtype=np.float64)
    print("\nTCP board probe: use the already calibrated probe tip. Keep the ChArUco board fixed.")
    for index, corner in enumerate(probe_corners):
        xyz_mm = corner.board_point_m * 1000.0
        input(f"Move the TCP tip to the {corner.label} at board [{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}] mm, then press Enter: ")
        pose = controller.get_current_tcp_pose()
        base_points[index] = np.asarray(pose[:3], dtype=np.float64)
        print(f"[captured] {corner.label}: base [{base_points[index, 0]:.6f}, {base_points[index, 1]:.6f}, {base_points[index, 2]:.6f}] m")

    report = validate_tcp_correspondences(board_points, base_points, max_rms_m=max_rms_m)
    for retry in range(max(0, int(max_retries))):
        if report["usable"]:
            break
        suspect = int(report["suspect_index"])
        corner = probe_corners[suspect]
        print(
            f"[retry] TCP probe RMS is {report['rms_m'] * 1000:.2f} mm (limit {max_rms_m * 1000:.2f} mm). "
            f"Please re-place the {corner.label}; leave-one-out error is "
            f"{report['leave_one_out_errors_m'][suspect] * 1000:.2f} mm."
        )
        input(f"Re-touch the {corner.label}, then press Enter to replace only this sample: ")
        base_points[suspect] = np.asarray(controller.get_current_tcp_pose()[:3], dtype=np.float64)
        report = validate_tcp_correspondences(board_points, base_points, max_rms_m=max_rms_m)
    if not report["usable"]:
        suspect = probe_corners[int(report["suspect_index"])]
        raise RuntimeError(
            f"TCP board probe is unusable after retries: RMS={report['rms_m'] * 1000:.2f} mm. "
            f"Re-place {suspect.label} and rerun the TCP probe stage."
        )
    return board_points, base_points, report


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    serials = tuple(str(serial).strip() for serial in args.serials)
    if not serials or any(not serial for serial in serials) or len(set(serials)) != len(serials):
        raise ValueError("--serials must contain one or more unique, non-empty RealSense serial numbers")
    if args.square_length_m <= 0.0 or args.marker_length_m <= 0.0 or args.marker_length_m >= args.square_length_m:
        raise ValueError("Require 0 < --marker-length-m < --square-length-m")
    if args.min_charuco_corners < 4 or args.max_reprojection_rms_px <= 0.0 or args.max_probe_rms_m <= 0.0:
        raise ValueError("ChArUco corner count and calibration thresholds must be positive")

    output_dir = Path(args.output_dir)
    output = Path(args.output) if args.output is not None else output_dir / "camera_calibration.json"
    board = create_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
    )
    probe_corners = select_inner_probe_corners(board, squares_x=args.squares_x, squares_y=args.squares_y)
    source = RealSenseD435iSource(
        cameras=[D435iCameraConfig(name=serial, serial=serial) for serial in serials],
        width=args.width,
        height=args.height,
        fps=args.fps,
        wait_timeout_ms=args.wait_timeout_ms,
    )

    source.start()
    try:
        for _ in range(max(1, int(args.warmup_frames))):
            source.read()
        intrinsics = _intrinsics_payload(source, serials)
        _write_json(output_dir / "realsense_color_intrinsics.json", intrinsics)
        print(f"[done] captured colour-stream intrinsics for {len(serials)} camera(s).")
        print("Place the rigid ChArUco board where every selected camera can see all four indicated inner corners.")
        input("Press Enter to start the all-camera ChArUco placement check: ")
        observations = _wait_for_valid_placement(
            source,
            intrinsics,
            board,
            probe_corners,
            min_corners=args.min_charuco_corners,
            max_rms_px=args.max_reprojection_rms_px,
            preview_dir=output_dir / "charuco_placement_previews",
        )
    finally:
        source.stop()

    board_extrinsics = {
        "coordinate_frame": "charuco_board",
        "fusion": {"enabled": len(serials) > 1, "camera_count": len(serials)},
        "probe_corners": [{"charuco_id": corner.charuco_id, "label": corner.label, "board_point_m": corner.board_point_m.tolist()} for corner in probe_corners],
        "cameras": {},
    }
    for name in serials:
        observation, item = observations[name], intrinsics["cameras"][name]
        board_extrinsics["cameras"][name] = {
            "intrinsics": item["intrinsics"],
            "distortion": item.get("distortion", []),
            "board_to_camera": observation.board_to_camera.tolist(),
            "camera_to_board": np.linalg.inv(observation.board_to_camera).tolist(),
            "charuco_corner_count": observation.corner_count,
            "charuco_reprojection_rms_px": observation.reprojection_rms_px,
        }
    _write_json(output_dir / "camera_extrinsics_charuco_board.json", board_extrinsics)
    print("[done] all camera extrinsics were calculated in the ChArUco-board frame.")

    controller = ReadOnlyTCPProbe(robot_ip=args.robot_ip)
    controller.connect()
    try:
        board_points, base_points, probe_report = _probe_board_with_tcp(
            controller, probe_corners, max_rms_m=args.max_probe_rms_m, max_retries=args.max_probe_retries
        )
    finally:
        controller.close()
    board_to_base = np.asarray(probe_report["board_to_base"], dtype=np.float64)
    _write_json(
        output_dir / "board_base_correspondences.json",
        {"board_points_m": board_points.tolist(), "base_points_m": base_points.tolist(), "fit_rms_m": probe_report["rms_m"]},
    )

    output_cameras: dict[str, Any] = {}
    for name in serials:
        observation, item = observations[name], intrinsics["cameras"][name]
        output_cameras[name] = {
            "model": "intel_realsense_d435i",
            "serial": name,
            "width": int(item["width"]),
            "height": int(item["height"]),
            "fps": int(args.fps),
            "frame": "ur_base",
            "depth_alignment": "depth_to_color",
            "intrinsics": item["intrinsics"],
            "distortion": item.get("distortion", []),
            "camera_to_world": camera_to_base_from_board_pose(board_to_base, observation.board_to_camera).tolist(),
            "charuco_corner_count": observation.corner_count,
            "charuco_reprojection_rms_px": observation.reprojection_rms_px,
        }
    result = {
        "coordinate_frame": "ur_base",
        "board_to_base": board_to_base.tolist(),
        "tcp_board_probe": {
            "board_points_m": board_points.tolist(),
            "base_points_m": base_points.tolist(),
            "rms_m": probe_report["rms_m"],
            "residuals_m": probe_report["residuals_m"].tolist(),
            "leave_one_out_errors_m": probe_report["leave_one_out_errors_m"].tolist(),
        },
        "fusion": {"enabled": len(serials) > 1, "camera_count": len(serials), "coordinate_frame": "ur_base"},
        "cameras": output_cameras,
    }
    _write_json(output, result)
    return result


def main() -> None:
    result = run_calibration(parse_args())
    print(
        f"[done] wrote {len(result['cameras'])} UR-base camera calibration(s); "
        f"fusion={'enabled' if result['fusion']['enabled'] else 'not needed'}; "
        f"TCP board-probe RMS={result['tcp_board_probe']['rms_m'] * 1000:.2f} mm."
    )


if __name__ == "__main__":
    main()
