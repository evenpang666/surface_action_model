"""Real image detection and the complete table-fixed calibration workflow."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_scripts import single_camera_charuco_calibration as calibration
from real_scripts import calibrate_single_ur7e_pika as entry
from real_scripts import single_ur7e_pika_scene as scene
from safety_module.dual_camera_calibration import transform_error


@pytest.fixture
def setup(tmp_path):
    cfg = scene.load_single_config()
    b = cfg["calibration_board"]
    args = SimpleNamespace(
        output_dir=tmp_path, output=tmp_path / "camera_calibration.json", serials=["405622074939"],
        robot_ip="192.168.124.73", squares_x=b["squares_x"], squares_y=b["squares_y"],
        square_length_m=b["square_length_m"], marker_length_m=b["marker_length_m"],
        dictionary=b["dictionary"], legacy_pattern=b["legacy_pattern"], width=1280, height=720, fps=15,
        min_charuco_corners=6, max_reprojection_rms_px=.8, max_probe_rms_m=.0025,
        camera_samples=3, verification_samples=2, max_camera_spread_mm=3., max_board_motion_mm=5.,
        max_board_motion_deg=2., resume=False, retouch_corner_ids=[], touch_corner_ids=None,
        headless=True, active_tcp_sha256="test-tcp-digest")
    board, dictionary = calibration.create_board(args)
    K = np.array([[906., 0., 640.], [0., 905., 360.], [0., 0., 1.]])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.25, 0., .03]).as_matrix()
    pose[:3, 3] = [-.38, -.25, 1.2]
    def render(shift=0.):
        printed = board.generateImage((1100, 800))
        corners = np.array([[0., 0., 0.], [.77, 0., 0.], [.77, .56, 0.], [0., .56, 0.]])
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        dest, _ = cv2.projectPoints(corners, rvec, pose[:3, 3] + [shift, 0., 0.], K, np.zeros(5))
        homography = cv2.getPerspectiveTransform(np.array([[0, 0], [1099, 0], [1099, 799], [0, 799]], np.float32), dest.reshape(4, 2).astype(np.float32))
        gray = cv2.warpPerspective(printed, homography, (1280, 720), borderValue=210)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return args, board, dictionary, K, pose, render


def test_legacy_print_detects_seventy_corners_and_recovers_correct_origin(setup):
    args, board, dictionary, K, expected, render = setup
    image = render()
    obs = calibration.observe_board(image, board, dictionary, K, np.zeros(5))
    assert obs.marker_count == obs.consistent_markers == 44
    assert obs.corner_count == 70 and obs.reprojection_rms_px < .8
    mm, deg = transform_error(expected, obs.board_to_camera)
    assert mm < 4. and deg < .5
    assert calibration.select_spread_corners(obs.ids.reshape(-1), calibration.board_points(board)) == [0, 9, 60, 69]
    # This is the previous detector's failure: default modern layout on this print.
    modern = cv2.aruco.CharucoBoard((11, 8), .07, .052, dictionary)
    assert cv2.aruco.CharucoDetector(modern).detectBoard(image)[1] is None


def test_selection_only_uses_detected_corners_and_can_replace_unreachable_point(setup):
    args, board, *_ = setup
    points = calibration.board_points(board)
    visible = [2, 4, 7, 13, 25, 37, 42, 55, 68]
    selected = calibration.select_spread_corners(visible, points)
    assert len(selected) == 4 and set(selected) <= set(visible)
    with pytest.raises(ValueError, match="collinear"):
        calibration.select_spread_corners([0, 1, 2, 3], points)
    assert "第 2 行、第 10 列" in calibration.touch_label(19, args.squares_x)


@pytest.mark.parametrize("fault", [None, "board_moved", "bad_contact"])
def test_complete_capture_touch_reobserve_and_quality_gate(monkeypatch, setup, fault):
    args, board, _, K, camera_pose, render = setup
    contacts, sources, closed, prompts = [], [], [], []
    base_to_board = np.eye(4)
    base_to_board[:3, :3] = Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    base_to_board[:3, 3] = [.4, .1, -.05]
    points = calibration.board_points(board)
    expected_ids = [1, 9, 60, 69]  # operator replaces the auto-selected P0 with P1
    answers = iter(["", "id=1", "", "", "", "", ""])
    def answer(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr("builtins.input", answer)
    class Source:
        def __init__(self, serial, width, height, fps):
            assert serial == "405622074939"
            self.index = len(sources)
            sources.append(serial)
            self.K, self.distortion = K, np.zeros(5)
            self.info = {"width": width, "height": height, "fps": fps, "intrinsics": K.tolist(),
                         "distortion": [0.] * 5}
        def start(self):
            pass
        def read(self):
            return render(.02 if fault == "board_moved" and self.index > 0 else 0.)
        def stop(self):
            closed.append("camera")
    class Probe:
        def __init__(self, ip):
            assert ip == "192.168.124.73"
        def connect(self):
            pass
        def get_current_tcp_pose(self):
            value = expected_ids[len(contacts) % 4]
            point = base_to_board[:3, :3] @ points[value] + base_to_board[:3, 3]
            if fault == "bad_contact" and value == 9:
                point += [.02, 0., 0.]
            contacts.append(value)
            return np.r_[point, [0., 0., 0.]]
        def close(self):
            closed.append("robot")
    monkeypatch.setattr(calibration, "ColorSource", Source)
    monkeypatch.setattr(calibration, "ReadOnlyTCPProbe", Probe)
    if fault:
        with pytest.raises(RuntimeError, match="quality failed"):
            calibration.run_calibration(args)
        result = json.loads(args.output.read_text())
        assert result["verified"] is False
        if fault == "board_moved":
            assert result["final_board_motion_mm"] > 5.
        else:
            assert result["tcp_board_probe"]["rms_m"] > .0025
    else:
        result = calibration.run_calibration(args)
        assert result["verified"] and result["camera_samples"] == 3
        expected = base_to_board @ np.linalg.inv(camera_pose)
        mm, deg = transform_error(expected, np.asarray(result["cameras"]["405622074939"]["camera_to_world"]))
        assert mm < 4. and deg < .5
        assert result["tcp_board_probe"]["rms_m"] < 1e-8
        assert result["final_board_motion_mm"] == pytest.approx(0.)
    assert contacts == expected_ids
    assert sources == ["405622074939"] * 2
    assert closed.count("camera") == 2 and closed.count("robot") == 1
    assert (args.output_dir / "selected_points.png").is_file()
    assert json.loads((args.output_dir / "touches.json").read_text())["touch_corner_ids"] == expected_ids
    if fault is None:
        saved_observation = json.loads((args.output_dir / "camera_observation.json").read_text())
        args.resume, args.retouch_corner_ids = True, [1]
        answers = iter(["", "", ""])  # verify saved board, replace P1, verify board again
        calibration.run_calibration(args)
        assert contacts == expected_ids + [1]
        assert len(sources) == 4  # two new verifications; no new initial capture
        assert json.loads((args.output_dir / "camera_observation.json").read_text())["observation_id"] == saved_observation["observation_id"]


def test_resume_rejects_another_tcp_and_moved_board(monkeypatch, setup):
    args, board, dictionary, *_ = setup
    args.resume = True
    signature = {"active_tcp_sha256": "new"}
    calibration.write_json(args.output_dir / "camera_observation.json", {"signature": {"active_tcp_sha256": "old"}})
    with pytest.raises(ValueError, match="differs"):
        calibration.capture_board(args, board, dictionary, signature)
    saved = {"board_to_camera": np.eye(4).tolist()}
    monkeypatch.setattr(calibration, "capture_board", lambda *_a: saved)
    monkeypatch.setattr(calibration, "verify_board", lambda *_a: (20., 0.))
    monkeypatch.setattr(calibration, "touch_board", lambda *_a: pytest.fail("Cannot probe after moving saved board"))
    with pytest.raises(RuntimeError, match="Board moved since checkpoint"):
        calibration.run_calibration(args)


def test_entry_uses_new_backend_and_retouch_requires_resume():
    assert entry.run_calibration is calibration.run_calibration
    assert entry.parse_args(["camera"]).camera_samples == 30
    assert not entry.parse_args(["camera"]).headless
    with pytest.raises(SystemExit):
        entry.parse_args(["camera", "--retouch-corner-ids", "19"])


def test_color_source_opens_front_rgb_only_using_advertised_profile(monkeypatch):
    import sys
    calls = []
    intr = SimpleNamespace(fx=900., fy=901., ppx=640., ppy=360., width=1280, height=720,
                           coeffs=[0.] * 5, model="distortion.brown_conrady")
    video = SimpleNamespace(width=lambda: 1280, height=lambda: 720, fps=lambda: 15,
                            get_intrinsics=lambda: intr)
    profile = SimpleNamespace(stream_type=lambda: "color", format=lambda: "bgr8", fps=lambda: 15, as_video_stream_profile=lambda: video)
    sensor = SimpleNamespace(get_stream_profiles=lambda: [profile])
    device = lambda serial: SimpleNamespace(get_info=lambda _key: serial, query_sensors=lambda: [sensor])
    context = SimpleNamespace(query_devices=lambda: [device("260522276066"), device("405622074939")])
    class Config:
        def enable_device(self, serial):
            calls.append(("device", serial))
        def enable_stream(self, *args):
            calls.append(("stream", *args))
    class Pipeline:
        def __init__(self, ctx):
            assert ctx is context
        def start(self, cfg):
            return SimpleNamespace(get_stream=lambda stream: profile)
        def wait_for_frames(self, timeout):
            assert timeout == 5000
            return SimpleNamespace(get_color_frame=lambda: SimpleNamespace(get_data=lambda: np.zeros((720, 1280, 3), np.uint8)))
        def stop(self):
            calls.append("stop")
    rs = SimpleNamespace(context=lambda: context, pipeline=Pipeline, config=Config,
                         camera_info=SimpleNamespace(serial_number="serial"), stream=SimpleNamespace(color="color"),
                         format=SimpleNamespace(bgr8="bgr8"))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    source = calibration.ColorSource("405622074939", 1280, 720, 30)
    source.start()
    np.testing.assert_allclose(source.K, [[900., 0., 640.], [0., 901., 360.], [0., 0., 1.]])
    assert source.info["fps"] == 15
    source.stop()
    assert calls == [("device", "405622074939"), ("stream", "color", 1280, 720, "bgr8", 15), "stop"]


@pytest.mark.parametrize("direction_scale", [1., 426. / 420.])
def test_probe_diagnostics_reports_rigid_invariant_pair_length_errors(direction_scale):
    ids = [0, 9, 60, 69]
    points = np.array([[.07, .07, 0.], [.70, .07, 0.], [.07, .49, 0.], [.70, .49, 0.]])
    scaled = points * [1., direction_scale, 1.]
    R = Rotation.from_rotvec([.3, -.1, .8]).as_matrix()
    contacts = scaled @ R.T + [.4, -.2, .1]
    from real_scripts.calibrate_d435i_to_ur_base import rigid_transform_from_correspondences
    fit = rigid_transform_from_correspondences(points, contacts)
    report = calibration.probe_diagnostics(ids, points, contacts, fit)
    pairs = {tuple(p["corner_ids"]): p for p in report["distance_checks"]}
    assert pairs[(0, 9)]["expected_m"] == pytest.approx(.63)
    assert pairs[(0, 9)]["measured_m"] == pytest.approx(.63)
    for edge in ((0, 60), (9, 69)):
        assert pairs[edge]["expected_m"] == pytest.approx(.42)
        assert pairs[edge]["measured_m"] == pytest.approx(.42 * direction_scale)
    expected_residual = .21 * (direction_scale - 1.)
    np.testing.assert_allclose(report["residuals_m"], expected_residual, atol=1e-12)


def test_cli_quality_rejection_is_clean_and_preserves_nonzero_status(monkeypatch, capsys):
    def reject(_argv):
        raise calibration.CalibrationQualityError("探点 RMS 2.520 mm > 2.500 mm。未安装部署外参。")
    monkeypatch.setattr(entry, "main", reject)
    assert entry.cli(["camera"]) == 2
    captured = capsys.readouterr()
    assert "2.520 mm > 2.500 mm" in captured.err
    assert "Traceback" not in captured.err


def test_cli_does_not_hide_unexpected_programming_errors(monkeypatch):
    def broken(_argv):
        raise TypeError("unexpected bug")
    monkeypatch.setattr(entry, "main", broken)
    with pytest.raises(TypeError, match="unexpected bug"):
        entry.cli(["camera"])
