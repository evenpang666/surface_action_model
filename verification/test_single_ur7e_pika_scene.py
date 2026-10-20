"""Scene isolation, calibration gates, and front-only runtime regression tests."""
import copy
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from real_scripts import live_dual_ur7e_obstacle_model as live
from real_scripts import single_ur7e_pika_scene as single
from real_scripts.real_robot_adapter import RGBDFrame, CameraCalibration
from real_scripts.ur7e_collision_mesh import flange_transform
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config


@pytest.fixture
def configured_scene(tmp_path):
    config = single.load_single_config()
    config = copy.deepcopy(config)
    config["robot"]["active_tcp_calibration"] = str(tmp_path / "tcp.json")
    config["robot"]["pika_geometry"] = str(tmp_path / "pika.json")
    config["cameras"]["front"]["calibration"] = str(tmp_path / "front.json")
    metadata = {"scene": config["scene"], "robot_ip": config["robot"]["ip"], "verified": True}
    tcp_bytes = json.dumps({**metadata, "matrix": np.eye(4).tolist()}).encode()
    (tmp_path / "tcp.json").write_bytes(tcp_bytes)
    geometry = {**metadata, "flange_to_pika_step_frame": np.eye(4).tolist(),
                "mesh_reference_opening_mm": 0., "moving_finger_kinematics_verified": True}
    (tmp_path / "pika.json").write_text(json.dumps(geometry))
    serial = config["cameras"]["front"]["serial"]
    front = {**metadata, "coordinate_frame": "ur_base", "active_tcp_sha256": hashlib.sha256(tcp_bytes).hexdigest(),
             "tcp_board_probe": {"rms_m": .001}, "cameras": {serial: {
                 "serial": serial, "depth_alignment": "depth_to_color",
                 "intrinsics": [[100., 0., 16.], [0., 100., 12.], [0., 0., 1.]],
                 "camera_to_world": np.eye(4).tolist(), "charuco_reprojection_rms_px": .3}}}
    (tmp_path / "front.json").write_text(json.dumps(front))
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


def small_scene(monkeypatch, configured_scene):
    path, _ = configured_scene
    # Mesh geometry is tested separately; keep end-to-end evidence tests small.
    monkeypatch.setattr(single, "mesh_surface_samples", lambda *_a, **_k: np.array([[0., 0., 0.]], np.float32))
    args = live.parse_args(["--scene", "single-pika", "--project-config", str(path)])
    return single.SinglePikaObstacleScene(args)


def test_scene_defaults_preserve_dual_and_isolate_single():
    dual = live.parse_args([])
    pika = live.parse_args(["--scene", "single-pika"])
    assert dual.project_config == DEFAULT_PROJECT_CONFIG and dual.port == 8766
    assert not dual.lingbot_depth and dual.lingbot_camera_names == ("front", "side")
    assert live.parse_args(["--lingbot-depth"]).lingbot_depth
    assert pika.project_config == single.DEFAULT_SINGLE_CONFIG and pika.port == 8767
    assert not pika.lingbot_depth and pika.lingbot_camera_names == ("front",)
    assert not dual.robot_mask and not pika.robot_mask
    assert not dual.no_robot and not pika.no_robot
    assert live.parse_args(["--no-robot"]).no_robot
    with pytest.raises(SystemExit):
        live.parse_args(["--no-robot", "--robot-mask"])
    assert live.parse_args(["--robot-mask"]).robot_mask
    assert not live.parse_args(["--no-robot-mask"]).robot_mask
    assert live.parse_args([], default_scene="single-pika").scene == "single-pika"
    with pytest.raises(ValueError, match="canonical"):
        load_project_config(pika.project_config)
    with pytest.raises(ValueError, match="Single Pika config"):
        single.load_single_config(dual.project_config)
    config = single.load_single_config()
    assert config["robot"]["ip"] == "192.168.124.73"
    assert config["cameras"]["front"]["serial"] == "405622074939"
    assert config["cameras"]["wrist"]["serial"] == "260522276066"
    assert config["reconstruction_cameras"] == ["front"]
    old_board = yaml.safe_load((single.REPO_ROOT / "configs/hardware/calibration/left_base_to_front_camera.yaml").read_text())["board"]
    assert config["calibration_board"]["squares_x"] == old_board["squares_x"]
    assert config["calibration_board"]["squares_y"] == old_board["squares_y"]
    assert config["calibration_board"]["square_length_m"] == old_board["square_size_m"]
    assert config["calibration_board"]["marker_length_m"] == old_board["marker_size_m"]
    assert config["calibration_board"]["dictionary"] == old_board["dictionary"]


@pytest.mark.parametrize("change", ["serial", "robot", "unverified", "quality", "rotation", "tcp_changed", "fingers"])
def test_rejects_foreign_invalid_or_unverified_calibration(configured_scene, change):
    path, config = configured_scene
    target = config["robot"]["pika_geometry"] if change == "fingers" else config["cameras"]["front"]["calibration"]
    from pathlib import Path
    target = Path(target)
    data = json.loads(target.read_text())
    serial = config["cameras"]["front"]["serial"]
    if change == "serial":
        data["cameras"] = {"260522276066": data["cameras"][serial]}
    elif change == "robot":
        data["robot_ip"] = "192.168.124.10"
    elif change == "unverified":
        data["verified"] = False
    elif change == "quality":
        data["tcp_board_probe"]["rms_m"] = float("nan")
    elif change == "rotation":
        data["cameras"][serial]["camera_to_world"][0][0] = 2.
    elif change == "tcp_changed":
        data["active_tcp_sha256"] = "different"
    else:
        data["moving_finger_kinematics_verified"] = False
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        single.SinglePikaObstacleScene(live.parse_args(["--scene", "single-pika", "--project-config", str(path)]))


def test_missing_calibration_fails_before_any_hardware(tmp_path, monkeypatch):
    config = single.load_single_config()
    config["cameras"]["front"]["calibration"] = str(tmp_path / "missing.json")
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(live, "RealSenseD435iSource", lambda **_kw: pytest.fail("camera opened before calibration"))
    with pytest.raises(RuntimeError, match="Missing single Pika calibration"):
        live.main(["--scene", "single-pika", "--project-config", str(path), "--once"])


def test_pika_fingers_follow_measured_opening_and_mount():
    body, finger_a, finger_b = (np.array([[x, 0., 0.]]) for x in (0., -.01, .01))
    result = single.pika_local_points([body, finger_a, finger_b], 40., 0.)
    np.testing.assert_allclose(result[:, 0], [0., -.03, .03])
    result = single.pika_local_points([body, finger_a, finger_b], 10., 10.)
    np.testing.assert_allclose(result, np.concatenate([body, finger_a, finger_b]))


def test_single_workspace_has_no_second_arm_and_viewer_names(monkeypatch, configured_scene):
    scene = small_scene(monkeypatch, configured_scene)
    first, second, union = scene.workspace_masks(np.array([[0., 0., .1], [2., 2., 2.]]))
    assert first.tolist() == union.tolist() == [True, False]
    assert not second.any()
    assert len(scene.arms) == len(scene.workspace_regions) == 1
    assert "Robotiq" not in scene.viewer_html and "left_base" not in scene.viewer_html
    assert 'href="/manual-obstacles"' in scene.viewer_html


@pytest.mark.parametrize("occupancy", [False, True])
@pytest.mark.parametrize("scene_name", ["single-pika", "dual"])
@pytest.mark.parametrize("robot_mask,no_robot,occupancy_backend", [
    (False, False, "numpy"), (True, False, "numpy"), (False, True, "numpy"),
    (False, True, "octomap"), (False, True, "voxblox")])
def test_complete_detection_loop_uses_only_selected_hardware(monkeypatch, configured_scene, occupancy, scene_name, robot_mask, no_robot, occupancy_backend):
    if occupancy_backend != "numpy":
        pytest.importorskip("octomap" if occupancy_backend == "octomap" else "real_scripts._voxblox_native")
        if not occupancy:
            pytest.skip("Native backends require occupancy mapping")
    monkeypatch.delenv("PIKA_GRIPPER_PORT", raising=False)
    scene = small_scene(monkeypatch, configured_scene)
    args = ["--scene", scene_name, "--once", "--no-browser", "--no-lingbot-depth",
            "--point-stride", "1", "--warmup-frames", "1", "--table-plane", "0", "0", "-1", "1",
            "--workspace-bounds", "-.2", ".2", "-.2", ".2", "0", "1.1",
            "--cluster-eps-m", ".025", "--min-cluster-points", "3", "--occupancy-resolution-m", ".04"]
    if not occupancy:
        args.append("--no-occupancy-map")
    if robot_mask:
        args.append("--robot-mask")
    if no_robot:
        args.append("--no-robot")
    if occupancy_backend != "numpy":
        args += ["--occupancy-backend", occupancy_backend]
    scene.workspace_regions = [((-.2, .2, -.2, .2, 0., 1.1), np.eye(4))]
    gripper_calls, camera_calls, robot_calls, closed = [], [], [], []
    if scene_name == "dual":
        # Exercise the real dual adapter methods and the same shared loop.
        dual = live.DualObstacleScene.__new__(live.DualObstacleScene)
        dual.arms = (live.Arm("left", "left-ip", np.eye(4), "Robotiq", np.eye(4)),
                     live.Arm("right", "right-ip", np.eye(4), "Robotiq", np.eye(4)))
        dual.project = scene.project
        dual.flange_to_tcp = {a.name: np.eye(4) for a in dual.arms}
        dual.calibrations = {name: CameraCalibration(name, scene.calibrations["front"].intrinsics, np.eye(4))
                             for name in ("front", "side")}
        dual.report = {"camera_serials": {"front": "dual-front", "side": "dual-side"}}
        dual.workspace_regions = scene.workspace_regions * 2
        scene = dual
        monkeypatch.setattr(live, "DualObstacleScene", lambda _a: scene)
    else:
        monkeypatch.setattr(single, "SinglePikaObstacleScene", lambda _a: scene)

    class Receiver:
        def __init__(self, ip):
            robot_calls.append(ip)
        def getTimestamp(self):
            return 1.
        def getActualQ(self):
            return np.zeros(6)
        def getActualTCPPose(self):
            matrix = flange_transform(np.zeros(6))
            return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
        def disconnect(self):
            closed.append("robot")

    class Pika:
        def __init__(self, port, **_kw):
            gripper_calls.append(port)
        def connect(self):
            pass
        def opening_mm(self):
            return 47.5
        def close(self):
            closed.append("pika")

    class Robotiq:
        def __init__(self, host, **_kw):
            gripper_calls.append(host)
        def closing_fraction(self):
            return .25
        def close(self):
            closed.append("robotiq")

    class Source:
        def __init__(self, *, cameras, **_kw):
            self.cameras = cameras
            camera_calls.extend((c.name, c.serial) for c in cameras)
        def start(self):
            pass
        def stop(self):
            closed.append("camera")
        def get_stream_info(self):
            return {}
        def get_color_calibrations(self):
            return {}
        def read(self):
            depth = np.ones((24, 32), np.float32)
            depth[7:17, 11:21] = .85
            timestamp = time.monotonic_ns()
            return {c.name: RGBDFrame(c.name, np.zeros((24, 32, 3), np.uint8), depth,
                                      host_timestamp_ns=timestamp) for c in self.cameras}

    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    monkeypatch.setattr(single, "PikaOpeningReader", Pika)
    monkeypatch.setattr(live, "ReadOnlyRobotiq2FPosition", Robotiq)
    monkeypatch.setattr(live, "RealSenseD435iSource", Source)
    monkeypatch.setattr(live, "collision_surface_samples", lambda **_kw: {})
    monkeypatch.setattr(live, "arm_surface_points", lambda *_a: np.empty((0, 3), np.float32))
    monkeypatch.setattr(live, "tool_surface_points", lambda *_a: np.zeros((1, 3), np.float32))
    renders = []
    def render(*_a, width, height, **_kw):
        assert robot_mask, "unmasked detection must skip robot rendering"
        renders.append(1)
        return np.zeros((height, width))
    monkeypatch.setattr(live, "render_surface_points_depth", render)
    original_keep_mask = live.robot_depth_keep_mask
    def keep_mask(*a, **kw):
        assert robot_mask, "unmasked detection must skip both raw/refined robot masks"
        return original_keep_mask(*a, **kw)
    monkeypatch.setattr(live, "robot_depth_keep_mask", keep_mask)
    if no_robot:
        # Missing RTDE must be harmless, and no robot/mesh code may run.
        monkeypatch.setitem(sys.modules, "rtde_receive", None)
        def forbidden(*_a, **_kw):
            pytest.fail("camera-only reconstruction touched robot hardware or geometry")
        monkeypatch.setattr(scene, "open_grippers", forbidden)
        monkeypatch.setattr(scene, "read_grippers", forbidden)
        monkeypatch.setattr(scene, "tool_points", forbidden)
        monkeypatch.setattr(live, "collision_surface_samples", forbidden)
        monkeypatch.setattr(live, "arm_surface_points", forbidden)
    # Exercise the refined branch too, without a model or GPU. The accepted
    # raw depth remains authoritative when the refiner returns the same frame.
    class Refiner:
        last_inference_seconds = 0.
        model_id, device, use_fp16 = "fake", "cpu", False
        camera_names = tuple(scene.calibrations)
        def refine(self, frames, _calibrations):
            return frames
    if scene_name == "dual" and occupancy_backend != "voxblox":
        monkeypatch.setattr(live, "create_lingbot_depth_refiner_from_args", lambda _args: Refiner())
    result = live.main(args)
    assert result["frame"] == 3 and result["obbs"]
    assert ("occupancy_map" in result) is occupancy
    if occupancy_backend == "octomap":
        assert result["obstacle_backend"] == "octomap"
        assert result["occupancy_map"]["backend"] == "octomap"
        assert result["occupancy_map"]["octomap"]["coordinate_frame"] == scene.coordinate_frame
    if occupancy_backend == "voxblox":
        assert result["obstacle_backend"] == "voxblox"
        assert result["occupancy_map"]["backend"] == "voxblox"
        assert result["diagnostics"]["occupancy"]["tsdf_observed_voxels"] > 0
    assert result["schema"] == scene.schema and result["coordinate_frame"] == scene.coordinate_frame
    assert result["camera_names"] == list(scene.calibrations)
    assert len(result["qpos"]) == (0 if no_robot else len(scene.arms))
    assert result["diagnostics"]["robot_connected"] is not no_robot
    assert result["diagnostics"]["robot_mask_enabled"] is robot_mask
    assert bool(renders) is robot_mask
    if not robot_mask:
        assert result["current_surface"] == []
        assert all(view["robot_occluded_pixels"] == 0 for view in result["diagnostics"]["depth_evidence"].values())
    if no_robot:
        assert not robot_calls and not gripper_calls
        assert result["mode"] == "reconstruction-only"
        assert result["robot"] == {"points": [], "colors": []}
        assert result["qpos"] == result["gripper_position"] == {}
        assert result["surface_source"] == "unavailable"
        assert closed == ["camera"]
        expected_cameras = ([("front", "405622074939")] if scene_name == "single-pika" else
                            [("front", "dual-front"), ("side", "dual-side")])
        assert camera_calls == expected_cameras
    elif scene_name == "single-pika":
        assert camera_calls == [("front", "405622074939")]
        assert robot_calls == ["192.168.124.73"]
        assert gripper_calls == [configured_scene[1]["robot"]["gripper_port"]]
        assert result["gripper_position"] == {"ur7e": .5}
        assert closed.count("pika") == closed.count("robot") == closed.count("camera") == 1
    else:
        assert camera_calls == [("front", "dual-front"), ("side", "dual-side")]
        assert robot_calls == gripper_calls == ["left-ip", "right-ip"]
        assert result["gripper_position"] == {"left": .25, "right": .25}
        assert closed.count("robotiq") == closed.count("robot") == 2


def test_pika_stale_state_is_not_a_live_opening():
    from real_scripts.pika_gripper_live_reader import PikaOpeningReader
    reader = PikaOpeningReader("fake", max_state_age_s=.25)
    reader._device = SimpleNamespace(get_gripper_distance=lambda: 30.)
    with pytest.raises(RuntimeError, match="missing or stale"):
        reader.opening_mm()
    reader._last_state_time = time.monotonic()
    assert reader.opening_mm() == 30.
    reader._last_state_time -= 1.
    with pytest.raises(RuntimeError, match="missing or stale"):
        reader.opening_mm()


def test_real_dual_scene_still_uses_installed_calibration_graph():
    scene = live.DualObstacleScene(live.parse_args([]))
    config, _ = load_project_config()
    from safety_module.dual_camera_calibration import calibration_from_project
    graph = calibration_from_project(config)
    assert scene.report["sha256"] == graph.digest
    assert scene.report["camera_serials"] == graph.serials
    assert list(scene.calibrations) == ["front", "side"]
    np.testing.assert_array_equal(scene.arms[1].base_to_left, graph.right_base_to_left_base)
    assert scene.schema == "dual_ur7e_obstacle_snapshot_v2" and scene.coordinate_frame == "left_base"


def test_no_robot_dual_skips_robot_geometry(monkeypatch):
    def forbidden(*_a, **_kw):
        pytest.fail("robot geometry loaded in camera-only mode")
    monkeypatch.setattr(live, "DualUR7eSurfacePointSampler", forbidden)
    scene = live.DualObstacleScene(live.parse_args(["--no-robot"]))
    assert list(scene.calibrations) == ["front", "side"]
    assert len(scene.workspace_regions) == 2 and scene.flange_to_tcp == {}


def test_no_robot_single_needs_only_camera_artifact(monkeypatch, configured_scene):
    from pathlib import Path
    path, config = configured_scene
    Path(config["robot"]["active_tcp_calibration"]).unlink()
    Path(config["robot"]["pika_geometry"]).unlink()
    def forbidden(*_a, **_kw):
        pytest.fail("Pika geometry loaded in camera-only mode")
    monkeypatch.setattr(single, "mesh_surface_samples", forbidden)
    scene = single.SinglePikaObstacleScene(live.parse_args([
        "--scene", "single-pika", "--project-config", str(path), "--no-robot"]))
    assert list(scene.calibrations) == ["front"] and scene.flange_to_tcp == {}
    assert len(scene.workspace_regions) == 1
    assert scene.report["sha256"] == hashlib.sha256(
        Path(config["cameras"]["front"]["calibration"]).read_bytes()).hexdigest()


def test_camera_calibration_routes_same_board_front_only_and_saves_tcp_identity(monkeypatch, configured_scene, tmp_path):
    from pathlib import Path
    from real_scripts import calibrate_single_ur7e_pika as calibration
    path, config = configured_scene
    calls, checks = [], []
    original = json.loads(Path(config["cameras"]["front"]["calibration"]).read_text())
    monkeypatch.setattr(calibration, "check_active_tcp", lambda cfg, matrix: checks.append((cfg["robot"]["ip"], matrix)))
    def run(args):
        calls.append(args)
        return original
    monkeypatch.setattr(calibration, "run_calibration", run)
    calibration.main(["camera", "--project-config", str(path), "--output-dir", str(tmp_path / "session")])
    assert len(calls) == 1 and len(checks) == 2
    args = calls[0]
    assert args.serials == ["405622074939"] and args.robot_ip == "192.168.124.73"
    assert (args.squares_x, args.squares_y, args.square_length_m, args.marker_length_m, args.dictionary) == (
        11, 8, .07, .052, "DICT_5X5_100")
    assert args.legacy_pattern is True and args.camera_samples == 30 and args.verification_samples == 10
    assert args.max_camera_spread_mm == 3. and args.max_board_motion_mm == 5.
    assert args.output == tmp_path / "session/camera_calibration.json"
    _, saved, _ = single.load_front_calibration(config)
    assert saved["verified"] and saved["active_tcp_sha256"] == original["active_tcp_sha256"]


@pytest.mark.parametrize("moving", [False, True])
def test_tcp_measurement_is_receive_only_and_disconnects(monkeypatch, moving):
    from real_scripts.calibrate_single_ur7e_pika import read_active_tcp
    calls = []
    class Receiver:
        def __init__(self, ip):
            calls.append(ip)
            self.count = 0
        def getActualQ(self):
            self.count += 1
            return np.full(6, .002 * self.count if moving else 0.)
        def getActualTCPPose(self):
            matrix = flange_transform(np.full(6, .002 * self.count if moving else 0.))
            return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
        def disconnect(self):
            calls.append("closed")
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    if moving:
        with pytest.raises(RuntimeError, match="stationary"):
            read_active_tcp("192.168.124.73", samples=3, interval_s=0.)
    else:
        result = read_active_tcp("192.168.124.73", samples=3, interval_s=0.)
        np.testing.assert_allclose(result["matrix"], np.eye(4), atol=1e-12)
    assert calls == ["192.168.124.73", "closed"]


def test_sdk_readback_requires_motor_messages_without_motor_commands(monkeypatch):
    from real_scripts.pika_gripper_live_reader import PikaOpeningReader
    calls = []
    class SDK:
        def __init__(self, port):
            calls.append(port)
        def connect(self):
            return True
        def _data_callback(self, data):
            self.position = data["motor"]["Position"]
        def get_gripper_distance(self):
            return self.position
        def disconnect(self):
            calls.append("closed")
    monkeypatch.setitem(sys.modules, "pika.gripper", SimpleNamespace(Gripper=SDK))
    reader = PikaOpeningReader("fake", max_state_age_s=.25)
    reader.connect()
    reader._device._data_callback({"motorstatus": {"Voltage": 24.}})
    with pytest.raises(RuntimeError, match="missing or stale"):
        reader.opening_mm()
    reader._device._data_callback({"motor": {"Position": 40.}})
    assert reader.opening_mm() == 40.
    reader._device._data_callback({"motor": {"Position": float("nan")}})
    assert reader.opening_mm() == 40.
    reader.close()
    assert calls == ["fake", "closed"]


@pytest.mark.parametrize("field,value", [("final_board_motion_mm", 8.),
                                         ("camera_translation_max_spread_mm", float("nan")),
                                         ("final_board_motion_deg", 3.)])
def test_runtime_rejects_failed_post_probe_camera_quality(configured_scene, field, value):
    from pathlib import Path
    _, config = configured_scene
    path = Path(config["cameras"]["front"]["calibration"])
    data = json.loads(path.read_text())
    data.update(calibration_workflow="table_fixed_charuco_v2", camera_quality_passed=True, camera_samples=30,
                board=config["calibration_board"], final_board_motion_mm=1., final_board_motion_deg=.1,
                camera_translation_max_spread_mm=.5)
    path.write_text(json.dumps(data))
    single.load_front_calibration(config)
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="quality limit exceeded"):
        single.load_front_calibration(config)


def inherited_geometry(config):
    return {"scene": config["scene"], "robot_ip": config["robot"]["ip"],
            "verified": False, "moving_finger_kinematics_verified": False,
            "geometry_mode": "legacy_inherited", "acceptance_basis": "same_installation_as_legacy",
            "flange_to_pika_step_frame": [[-1., 0., 0., -.01], [0., -1., 0., -.007],
                                          [0., 0., 1., -.073], [0., 0., 0., 1.]],
            "mesh_reference_opening_mm": 0., "moving_finger_model": "legacy_parallel_step_x",
            "legacy_reference": {"source": "legacy_mount.json", "source_sha256": "a" * 64,
                                 "source_status": "provisional_requires_visual_validation"}}


def test_explicit_legacy_geometry_retains_mount_and_provisional_status(monkeypatch, configured_scene):
    from pathlib import Path
    _, config = configured_scene
    geometry = inherited_geometry(config)
    Path(config["robot"]["pika_geometry"]).write_text(json.dumps(geometry))
    scene = small_scene(monkeypatch, configured_scene)
    np.testing.assert_array_equal(scene.pika_mount, geometry["flange_to_pika_step_frame"])
    assert scene.reference_opening_mm == 0.
    assert scene.report["pika_geometry"] == "legacy_inherited_parallel_fingers"
    assert scene.report["pika_geometry_verified"] is False
    assert scene.report["pika_geometry_reference"]["source_status"] == "provisional_requires_visual_validation"
    # Provenance does not change the old parallel-finger model or units.
    q = np.linspace(-.3, .3, 6)
    expected = single.transform_points(single.pika_local_points(scene.pika_samples, 47.5, 0.),
                                       flange_transform(q) @ scene.pika_mount)
    np.testing.assert_allclose(scene.tool_points(scene.arms[0], q, .5), expected)


@pytest.mark.parametrize("change", ["config_mode", "basis", "source_hash", "finger_model", "matrix", "robot_ip"])
def test_legacy_geometry_requires_explicit_valid_inheritance(configured_scene, change):
    from pathlib import Path
    _, config = configured_scene
    geometry = inherited_geometry(config)
    if change == "config_mode":
        config["robot"].pop("pika_geometry_mode")
    elif change == "basis":
        geometry["acceptance_basis"] = "unknown"
    elif change == "source_hash":
        geometry["legacy_reference"]["source_sha256"] = "missing"
    elif change == "finger_model":
        geometry["moving_finger_model"] = "unknown"
    elif change == "matrix":
        geometry["flange_to_pika_step_frame"][0][0] = 2.
    else:
        geometry["robot_ip"] = "other-controller"
    Path(config["robot"]["pika_geometry"]).write_text(json.dumps(geometry))
    with pytest.raises(ValueError):
        single.load_pika_geometry(config)


def test_legacy_mode_does_not_accept_unverified_tcp_artifact(configured_scene):
    from pathlib import Path
    _, config = configured_scene
    path = Path(config["robot"]["active_tcp_calibration"])
    data = json.loads(path.read_text())
    data.update(verified=False, geometry_mode="legacy_inherited", acceptance_basis="same_installation_as_legacy")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="has not been verified"):
        single.load_scene_artifact(path, config)


@pytest.mark.parametrize("cli_port,expected", [(None, "env-port"), ("cli-port", "cli-port")])
def test_pika_port_override_priority(monkeypatch, configured_scene, cli_port, expected):
    scene = small_scene(monkeypatch, configured_scene)
    calls = []
    class Reader:
        def __init__(self, port, **_kwargs):
            calls.append(port)
        def connect(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(single, "PikaOpeningReader", Reader)
    monkeypatch.setenv("PIKA_GRIPPER_PORT", "env-port")
    scene.open_grippers(SimpleNamespace(pika_gripper_port=cli_port))
    assert calls == [expected]
