"""Front-camera calibration using the established table-fixed dual-arm workflow.

Observe/lock the board, average frames, select visible spread corners, read TCP
contacts, and reobserve the board after probing. Supports modern and legacy
OpenCV APIs without changing the historical dual-arm calibration tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real_scripts.calibrate_ur7e_realsense_integrated import ReadOnlyTCPProbe
from real_scripts.calibrate_d435i_to_ur_base import rigid_transform_from_correspondences
from safety_module.dual_camera_calibration import rigid_matrix, transform_error


class CalibrationQualityError(RuntimeError):
    """Measured calibration failed a quality gate; retained for inspection."""


@dataclass(frozen=True)
class BoardObservation:
    board_to_camera: np.ndarray | None
    corners: np.ndarray | None
    ids: np.ndarray | None
    marker_count: int
    consistent_markers: int
    reprojection_rms_px: float

    @property
    def corner_count(self):
        return 0 if self.ids is None else len(self.ids)


def create_board(args):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dictionary))
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_length_m,
                                  args.marker_length_m, dictionary)
    else:
        board = aruco.CharucoBoard_create(args.squares_x, args.squares_y, args.square_length_m,
                                         args.marker_length_m, dictionary)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(args.legacy_pattern)
    elif not args.legacy_pattern:
        raise ValueError("This OpenCV version only supports legacy ChArUco boards")
    return board, dictionary


def board_points(board):
    return np.asarray(board.getChessboardCorners() if hasattr(board, "getChessboardCorners")
                      else board.chessboardCorners, dtype=float).reshape(-1, 3)


def consistent_markers(corners, ids, board):
    """Use the same homography outlier gate as the installed dual-arm workflow."""
    if ids is None or len(ids) < 6:
        return corners, ids
    marker_ids = board.getIds() if hasattr(board, "getIds") else board.ids
    marker_points = board.getObjPoints() if hasattr(board, "getObjPoints") else board.objPoints
    centers = {int(i): np.asarray(p).reshape(4, 3)[:, :2].mean(axis=0)
               for i, p in zip(np.asarray(marker_ids).reshape(-1), marker_points)}
    indices = [i for i, value in enumerate(ids.reshape(-1)) if int(value) in centers]
    if len(indices) < 6:
        return [], None
    expected = np.asarray([centers[int(ids[i, 0])] for i in indices], np.float32)
    observed = np.asarray([np.asarray(corners[i]).reshape(4, 2).mean(axis=0) for i in indices], np.float32)
    _, inliers = cv2.findHomography(expected, observed, cv2.RANSAC, 8.)
    if inliers is None or int(inliers.sum()) < max(6, int(np.ceil(.6 * len(indices)))):
        return [], None
    keep = np.asarray(indices)[inliers.reshape(-1).astype(bool)]
    return [corners[i] for i in keep], ids[keep]


def observe_board(image_bgr, board, dictionary, intrinsics, distortion):
    """Detect in native pixels with calibrated interpolation; no 2x upscale."""
    aruco = cv2.aruco
    parameters = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    if hasattr(aruco, "ArucoDetector"):
        markers, ids, _ = aruco.ArucoDetector(dictionary, parameters).detectMarkers(image_bgr)
    else:
        markers, ids, _ = aruco.detectMarkers(image_bgr, dictionary, parameters=parameters)
    marker_count = 0 if ids is None else len(ids)
    markers, ids = consistent_markers(markers, ids, board)
    consistent_count = 0 if ids is None else len(ids)
    corners, corner_ids, pose, rms = None, None, None, float("inf")
    K, dist = np.asarray(intrinsics, float), np.asarray(distortion, float)
    if consistent_count:
        if hasattr(aruco, "CharucoDetector"):
            settings = aruco.CharucoParameters()
            settings.cameraMatrix, settings.distCoeffs = K, dist
            corners, corner_ids, _, _ = aruco.CharucoDetector(board, settings, parameters).detectBoard(
                image_bgr, markerCorners=markers, markerIds=ids)
        else:
            _, corners, corner_ids = aruco.interpolateCornersCharuco(
                markers, ids, image_bgr, board, cameraMatrix=K, distCoeffs=dist)
        if corner_ids is not None and len(corner_ids) >= 6:
            obj = board_points(board)[corner_ids.reshape(-1)].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
                projection, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                rms = float(np.sqrt(np.mean(np.sum((projection.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    return BoardObservation(pose, corners, corner_ids, marker_count, consistent_count, rms)


def select_spread_corners(detected_ids, points):
    """Maximize total pairwise distance, then minimum distance, as in dual-arm.

    Vectorize bounded chunks instead of looping over ~1M quadrilaterals in Python.
    """
    ids = sorted(set(int(value) for value in detected_ids))
    if len(ids) < 4 or any(i < 0 or i >= len(points) for i in ids):
        raise ValueError("Need at least four valid detected board corner IDs")
    candidates, best, best_score = combinations(ids, 4), None, (-np.inf, -np.inf)
    while chunk := list(islice(candidates, 16384)):
        indices = np.asarray(chunk)
        p = points[indices, :2]
        a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
        valid = (np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-10) | (
            np.abs(a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0]) > 1e-10)
        if not valid.any():
            continue
        distances = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in combinations(range(4), 2)], axis=1)
        totals, minima = distances.sum(axis=1), distances.min(axis=1)
        totals[~valid] = -np.inf
        winner = np.lexsort((minima, totals))[-1]
        score = (totals[winner], minima[winner])
        if score > best_score:
            best_score, best = score, indices[winner].tolist()
    if best is None:
        raise ValueError("Detected board corners are collinear")
    return best


def validate_touch_ids(ids, points):
    if (len(ids) != 4 or len(set(ids)) != 4 or any(i < 0 or i >= len(points) for i in ids)
            or np.linalg.matrix_rank(points[ids] - points[ids].mean(axis=0)) < 2):
        raise ValueError(f"Need four unique non-collinear corner IDs in [0,{len(points) - 1}]")


def touch_label(corner_id, squares_x):
    row, col = divmod(corner_id, squares_x - 1)
    return f"板坐标原点起第 {row + 1} 行、第 {col + 1} 列内角（corner ID {corner_id}）"


def average_observations(observations):
    transforms = np.stack([obs.board_to_camera for obs in observations])
    result = np.eye(4)
    result[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(transforms[:, :3, :3]).mean().as_matrix()
    residuals = [transform_error(result, value) for value in transforms]
    return result, max(mm for mm, _ in residuals), max(deg for _, deg in residuals)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ColorSource:
    """Open only the selected camera's RGB stream, like dual-arm calibration."""

    def __init__(self, serial, width, height, fps):
        self.serial, self.requested = serial, (width, height, fps)
        self.pipeline = None

    def start(self):
        import pyrealsense2 as rs
        context = rs.context()
        device = next((d for d in context.query_devices()
                       if d.get_info(rs.camera_info.serial_number) == self.serial), None)
        if device is None:
            raise RuntimeError(f"RealSense not connected: {self.serial}")
        available = set()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    video = profile.as_video_stream_profile()
                    available.add((video.width(), video.height(), profile.fps()))
        preferred = [self.requested, (1280, 720, 30), (1280, 720, 15), (640, 480, 30), (640, 480, 15)]
        candidates = list(dict.fromkeys(p for p in preferred if p in available))
        candidates.extend(sorted(available - set(candidates), key=lambda p: (-p[0] * p[1], -p[2])))
        errors = []
        for width, height, fps in candidates:
            pipeline, cfg = rs.pipeline(context), rs.config()
            cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(cfg)
            except RuntimeError as exc:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                if "busy" in str(exc).lower():
                    raise RuntimeError(f"RealSense {self.serial} is busy; stop other camera processes") from exc
                errors.append(str(exc))
                continue
            self.pipeline = pipeline
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.asarray([[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]])
            self.distortion = np.asarray(intr.coeffs, float)
            self.info = {"intrinsics": self.K.tolist(), "distortion": self.distortion.tolist(),
                         "width": intr.width, "height": intr.height, "fps": fps,
                         "distortion_model": str(intr.model).rsplit(".", 1)[-1]}
            print(f"RGB ready: {self.serial}, {width}×{height}@{fps} BGR8（仅颜色流）")
            for _ in range(20):
                self.read()
            return
        raise RuntimeError(f"Cannot start color stream for {self.serial}: {errors}")

    def read(self):
        frame = self.pipeline.wait_for_frames(5000).get_color_frame()
        if not frame:
            raise RuntimeError("Missing RealSense color frame")
        return np.asanyarray(frame.get_data()).copy()

    def stop(self):
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            pipeline.stop()


def good_observation(obs, args):
    return (obs.board_to_camera is not None and obs.corner_count >= args.min_charuco_corners
            and np.isfinite(obs.reprojection_rms_px) and obs.reprojection_rms_px <= args.max_reprojection_rms_px)


def annotated_image(image, obs, args, *, progress="", touch_ids=()):
    image = image.copy()
    if obs.ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(image, obs.corners, obs.ids)
        pixels = dict(zip(obs.ids.reshape(-1).tolist(), obs.corners.reshape(-1, 2)))
        for value in touch_ids:
            if value in pixels:
                u, v = np.rint(pixels[value]).astype(int)
                cv2.circle(image, (u, v), 14, (0, 0, 255), 3)
                cv2.putText(image, f"P{value}", (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    text = (f"markers={obs.marker_count}/{obs.consistent_markers} corners={obs.corner_count} "
            f"rms={obs.reprojection_rms_px:.2f}px legacy={int(args.legacy_pattern)} {progress}")
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def save_preview(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save calibration preview: {path}")


def capture_board(args, board, dictionary, signature):
    path = args.output_dir / "camera_observation.json"
    if args.resume:
        if not path.is_file():
            raise RuntimeError("No camera observation checkpoint; start a new session without --resume")
        observation = json.loads(path.read_text(encoding="utf-8"))
        if observation.get("signature") != signature:
            raise ValueError("Saved observation differs from camera/board/TCP; start a new session without --resume")
        rigid_matrix(observation["board_to_camera"], "saved_camera_T_board")
        validate_touch_ids(observation["touch_corner_ids"], board_points(board))
        rms = float(observation["charuco_reprojection_rms_px"])
        spread = float(observation["camera_translation_max_spread_mm"])
        if (observation["camera_samples"] < args.camera_samples
                or observation["charuco_corner_count"] < args.min_charuco_corners
                or not np.isfinite([rms, spread]).all()
                or not 0 <= rms <= args.max_reprojection_rms_px
                or not 0 <= spread <= args.max_camera_spread_mm):
            raise ValueError("Saved camera observation fails current quality limits; start a new session")
        print(f"复用已保存的相机观测：{path}；标定板必须仍固定在原位置。")
        return observation
    source = ColorSource(args.serials[0], args.width, args.height, args.fps)
    observations, selected, locked, locked_pixels = [], None, None, None
    capturing, attempts = False, 0
    print("保持板和 frontcam 固定；检测稳定后按一次空格，自动采集稳定帧；q 取消。")
    if args.headless:
        print("无窗口模式：先保存检测预览，在终端回车锁定，再自动采样。")
    try:
        source.start()
        while len(observations) < args.camera_samples:
            image = source.read()
            obs = observe_board(image, board, dictionary, source.K, source.distortion)
            good = good_observation(obs, args)
            preview = annotated_image(image, obs, args, progress=f"auto={len(observations)}/{args.camera_samples}",
                                      touch_ids=selected or ())
            if args.headless and not capturing:
                attempts += 1
                preview_path = args.output_dir / "board_preview.png"
                save_preview(preview_path, preview)
                print(f"[preview] {preview_path}; markers={obs.marker_count}, consistent={obs.consistent_markers}, "
                      f"corners={obs.corner_count}, RMS={obs.reprojection_rms_px:.3f}px")
                answer = input("回车锁定当前预览（板必须固定），r 重新检测，q 取消：").strip().lower()
                if answer == "q":
                    raise SystemExit("Calibration cancelled")
                lock = not answer and good
                if not good:
                    print("当前观测未通过：检查板布局/参数和可见度；不会降低角点或误差门槛。")
            else:
                lock = False
                if not args.headless:
                    cv2.imshow("frontcam ChArUco calibration (SPACE lock, q cancel)", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise SystemExit("Calibration cancelled")
                    lock = key == 32 and good
            if lock and not capturing:
                selected = (list(args.touch_corner_ids) if args.touch_corner_ids else
                            select_spread_corners(obs.ids.reshape(-1).tolist(), board_points(board)))
                validate_touch_ids(selected, board_points(board))
                pixels = dict(zip(obs.ids.reshape(-1).tolist(), obs.corners.reshape(-1, 2)))
                if any(value not in pixels for value in selected):
                    print("指定触点未全部识别；请等待清晰画面后重新锁定。")
                    selected = None
                    continue
                locked, locked_pixels = annotated_image(image, obs, args, touch_ids=selected), pixels
                capturing = True
                print("已锁定标定板，正在自动采样。保持板和相机固定。")
            if capturing and good:
                observations.append(obs)
            elif capturing and args.headless:
                attempts += 1
                if attempts > args.camera_samples * 10:
                    raise CalibrationQualityError("Not enough stable board frames; restart observation")
        pose, spread_mm, spread_deg = average_observations(observations)
        selected_path = args.output_dir / "selected_points.png"
        save_preview(selected_path, locked)
        print(f"触点标注图：{selected_path}")
        for value in selected:
            u, v = locked_pixels[value]
            print(f"  P{value}: {touch_label(value, args.squares_x)}，像素 ({u:.1f}, {v:.1f})")
        if spread_mm > args.max_camera_spread_mm:
            raise CalibrationQualityError(f"Camera board-pose spread {spread_mm:.3f} mm exceeds {args.max_camera_spread_mm} mm")
        observation = {"signature": signature, "observation_id": uuid4().hex,
                       "board_to_camera": pose.tolist(), "touch_corner_ids": selected,
                       "detected_corner_ids": sorted(locked_pixels),
                       "detected_corner_pixels": {int(k): v.tolist() for k, v in locked_pixels.items()},
                       "camera_samples": len(observations), "camera_translation_max_spread_mm": spread_mm,
                       "camera_rotation_max_spread_deg": spread_deg,
                       "charuco_reprojection_rms_px": max(obs.reprojection_rms_px for obs in observations),
                       "charuco_corner_count": min(obs.corner_count for obs in observations),
                       "stream": source.info}
        write_json(path, observation)
        return observation
    finally:
        source.stop()
        if not args.headless:
            cv2.destroyAllWindows()


def touch_board(args, points, observation, signature):
    path = args.output_dir / "touches.json"
    touch_signature = {"session": signature, "observation_id": observation["observation_id"]}
    saved = json.loads(path.read_text(encoding="utf-8")) if args.resume and path.exists() else {}
    if saved and saved.get("signature") != touch_signature:
        raise ValueError("Touch checkpoint belongs to a different camera observation; start a new session")
    ids = list(saved.get("touch_corner_ids", observation["touch_corner_ids"]))
    validate_touch_ids(ids, points)
    touches = {int(k): np.asarray(v, float) for k, v in saved.get("touches", {}).items()}
    for value in args.retouch_corner_ids:
        if value not in ids:
            raise ValueError(f"Cannot retouch P{value}: not one of the saved touch IDs")
        touches.pop(value, None)
    print("确认示教器已激活已标定 TCP；仅记录位姿，不发送运动命令。")
    print(f"行列从 1 开始，仅数板内交点，每行 {args.squares_x - 1} 个；对照 selected_points.png。")
    probe = ReadOnlyTCPProbe(args.robot_ip)
    try:
        probe.connect()
        for index in range(4):
            value = ids[index]
            if value in touches:
                print(f"[{index + 1}/4] 复用 P{value} 触点。")
                continue
            while True:
                answer = input(f"[{index + 1}/4] 触碰{touch_label(value, args.squares_x)}后回车；id=N 换点，q 取消：").strip().lower()
                if answer == "q":
                    raise SystemExit("Calibration cancelled; saved contacts retained")
                if not answer:
                    break
                try:
                    if not answer.startswith("id="):
                        raise ValueError("请输入 Enter、id=N 或 q")
                    replacement = int(answer.split("=", 1)[1])
                    proposed = ids.copy()
                    proposed[index] = replacement
                    validate_touch_ids(proposed, points)
                except ValueError as exc:
                    print(exc)
                    continue
                value = replacement
                ids[index] = value
                if value not in observation["detected_corner_ids"]:
                    print("此内角未在锁定图中识别，请按板行列确认实物位置并保持板固定。")
                print(f"改为 {touch_label(value, args.squares_x)}；触碰后再次回车记录。")
            touches[value] = np.asarray(probe.get_current_tcp_pose()[:3], float)
            write_json(path, {"signature": touch_signature, "touch_corner_ids": ids,
                              "touches": {str(k): v.tolist() for k, v in touches.items()}})
    finally:
        probe.close()
    base = np.asarray([touches[i] for i in ids], float)
    if base.shape != (4, 3) or not np.isfinite(base).all():
        raise ValueError("Invalid measured TCP contact positions")
    matrix = rigid_transform_from_correspondences(points[ids], base)
    predicted = points[ids] @ matrix[:3, :3].T + matrix[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((predicted - base) ** 2, axis=1))))
    return ids, base, matrix, rms


def probe_diagnostics(ids, points, contacts, board_to_base):
    """Report fit residuals and rigid-motion-invariant pair lengths.

    The largest residual helps prioritize a remeasurement; with four contacts
    it does not uniquely identify a bad contact. Pair lengths also expose board
    scale, contact placement, or physical TCP errors that a rigid fit cannot fix.
    """
    points, contacts = np.asarray(points, float), np.asarray(contacts, float)
    predicted = points @ board_to_base[:3, :3].T + board_to_base[:3, 3]
    residuals = np.linalg.norm(predicted - contacts, axis=1)
    pairs = []
    for i, j in combinations(range(len(ids)), 2):
        expected = float(np.linalg.norm(points[i] - points[j]))
        measured = float(np.linalg.norm(contacts[i] - contacts[j]))
        pairs.append({"corner_ids": [int(ids[i]), int(ids[j])], "expected_m": expected,
                      "measured_m": measured, "difference_m": measured - expected})
    return {"residuals_m": residuals.tolist(), "distance_checks": pairs,
            "largest_residual_corner_id": int(ids[int(np.argmax(residuals))])}


def print_probe_diagnostics(ids, report):
    print("触点拟合残差：" + "，".join(f"P{i}={r * 1000:.3f} mm"
                                         for i, r in zip(ids, report["residuals_m"])))
    for pair in report["distance_checks"]:
        i, j = pair["corner_ids"]
        print(f"  P{i}–P{j} 距离：配置 {pair['expected_m'] * 1000:.3f} mm，"
              f"实测 {pair['measured_m'] * 1000:.3f} mm，差 {pair['difference_m'] * 1000:+.3f} mm")


def verify_board(args, board, dictionary, initial_pose):
    input("移开机械臂，保持板和 frontcam 固定，回车开始最终板位移复核：")
    source, observations = ColorSource(args.serials[0], args.width, args.height, args.fps), []
    try:
        source.start()
        for _ in range(300):
            image = source.read()
            obs = observe_board(image, board, dictionary, source.K, source.distortion)
            if good_observation(obs, args):
                observations.append(obs)
                if len(observations) >= args.verification_samples:
                    save_preview(args.output_dir / "board_verification.png", annotated_image(image, obs, args))
                    break
        if len(observations) < args.verification_samples:
            raise CalibrationQualityError("Final board-motion check failed: insufficient visible board corners")
        pose, _, _ = average_observations(observations)
        mm, deg = transform_error(initial_pose, pose)
        print(f"板位移复核：{mm:.3f} mm / {deg:.3f}°")
        return mm, deg
    finally:
        source.stop()


def run_calibration(args):
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    board, dictionary = create_board(args)
    points = board_points(board)
    signature = {"workflow": "table_fixed_charuco_v2", "robot_ip": args.robot_ip, "serial": args.serials[0],
                 "squares_x": args.squares_x, "squares_y": args.squares_y,
                 "square_length_m": args.square_length_m, "marker_length_m": args.marker_length_m,
                 "dictionary": args.dictionary, "legacy_pattern": args.legacy_pattern,
                 "active_tcp_sha256": args.active_tcp_sha256}
    print(f"标定：frontcam {args.serials[0]}，UR {args.robot_ip}；"
          f"{args.squares_x}×{args.squares_y}，{args.dictionary}，legacy_pattern={args.legacy_pattern}")
    observation = capture_board(args, board, dictionary, signature)
    initial_pose = rigid_matrix(observation["board_to_camera"], "camera_T_board")
    if args.resume:
        print("恢复前先检查板是否仍位于已锁定位置。")
        mm, deg = verify_board(args, board, dictionary, initial_pose)
        if mm > args.max_board_motion_mm or deg > args.max_board_motion_deg:
            raise CalibrationQualityError("Board moved since checkpoint; start a new session without --resume")
    ids, contacts, base_to_board, rms = touch_board(args, points, observation, signature)
    probe_report = probe_diagnostics(ids, points[ids], contacts, base_to_board)
    print_probe_diagnostics(ids, probe_report)
    mm, deg = verify_board(args, board, dictionary, initial_pose)
    passed = (rms <= args.max_probe_rms_m and mm <= args.max_board_motion_mm
              and deg <= args.max_board_motion_deg
              and observation["camera_translation_max_spread_mm"] <= args.max_camera_spread_mm)
    serial = args.serials[0]
    camera = {**observation["stream"], "serial": serial, "frame": "ur_base", "depth_alignment": "depth_to_color",
              "camera_to_world": (base_to_board @ np.linalg.inv(initial_pose)).tolist(),
              "charuco_corner_count": observation["charuco_corner_count"],
              "charuco_reprojection_rms_px": observation["charuco_reprojection_rms_px"]}
    result = {"coordinate_frame": "ur_base", "calibration_workflow": "table_fixed_charuco_v2",
              "verified": bool(passed), "camera_quality_passed": bool(passed), "board_to_base": base_to_board.tolist(),
              "touch_corner_ids": ids, "tcp_board_probe": {"board_points_m": points[ids].tolist(),
                  "base_points_m": contacts.tolist(), "rms_m": float(rms), **probe_report},
              "camera_samples": observation["camera_samples"],
              "camera_translation_max_spread_mm": observation["camera_translation_max_spread_mm"],
              "camera_rotation_max_spread_deg": observation["camera_rotation_max_spread_deg"],
              "final_board_motion_mm": mm, "final_board_motion_deg": deg,
              "quality_limits": {"max_probe_rms_m": args.max_probe_rms_m, "max_camera_spread_mm": args.max_camera_spread_mm,
                  "max_board_motion_mm": args.max_board_motion_mm, "max_board_motion_deg": args.max_board_motion_deg},
              "fusion": {"enabled": False, "camera_count": 1, "coordinate_frame": "ur_base"}, "cameras": {serial: camera}}
    write_json(args.output, result)
    print(f"探点 RMS：{rms * 1000:.3f} mm；{'PASS' if passed else 'FAIL'}；过程结果：{args.output}")
    if not passed:
        if rms > args.max_probe_rms_m:
            largest = probe_report["largest_residual_corner_id"]
            print(f"探点 RMS {rms * 1000:.3f} mm 超过 {args.max_probe_rms_m * 1000:.3f} mm。"
                  f"残差最大为 P{largest}，仅作为补测顺序参考，不能唯一定位错点。")
            print("请核对实际格距、触点位置和物理 TCP 尖端；若两条平行边同方向偏长/偏短，先检查尺度或共同接触误差。")
            print("先尺量板的实际跨度，并确认每次接触使用同一个已标定 TCP 实体点；"
                  "不根据最大残差自动指定补测点。确认后可用 --resume --retouch-corner-ids 补测一个或多个点。")
        failures = []
        if rms > args.max_probe_rms_m:
            failures.append(f"探点 RMS {rms * 1000:.3f} mm > {args.max_probe_rms_m * 1000:.3f} mm")
        if mm > args.max_board_motion_mm or deg > args.max_board_motion_deg:
            failures.append(f"板位移 {mm:.3f} mm / {deg:.3f}° 超限，需重新采集相机观测与触点")
        if observation["camera_translation_max_spread_mm"] > args.max_camera_spread_mm:
            failures.append("相机板位姿采样离散度超限")
        raise CalibrationQualityError("Calibration quality failed: " + "；".join(failures) + "。未安装部署外参。")
    return result
