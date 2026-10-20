"""Exercise the actual perception HTTP routes with synthetic receive-only inputs."""
import json
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_scripts import live_dual_ur7e_obstacle_model as live
from real_scripts import single_ur7e_pika_scene as single
from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame
from real_scripts.ur7e_collision_mesh import flange_transform


FLAGS = ["--no-browser", "--no-lingbot-depth", "--point-stride", "1", "--warmup-frames", "1",
         "--table-plane", "0", "0", "-1", "1", "--workspace-bounds", "-.2", ".2", "-.2", ".2", "0", "1.1",
         "--occupancy-resolution-m", ".04", "--cluster-eps-m", ".025", "--min-cluster-points", "3"]


def install_synthetic_inputs(monkeypatch, scene_name, *, image_scale=1):
    is_single = scene_name == "single-pika"
    names = ("front",) if is_single else ("front", "side")
    K = np.array([[100., 0., 16.], [0., 100., 12.], [0., 0., 1.]])
    K[:2] *= image_scale
    arms = (live.Arm("ur7e" if is_single else "left", "mock-arm", np.eye(4), "mock", np.eye(4)),)
    bounds = (-.2, .2, -.2, .2, 0., 1.1)
    def workspace(points):
        mask = live.points_in_bounds(points, bounds)
        return mask, np.zeros(len(points), bool), mask
    gripper = SimpleNamespace(close=lambda: None)
    scene = SimpleNamespace(
        schema="single_ur7e_pika_obstacle_snapshot_v1" if is_single else "dual_ur7e_obstacle_snapshot_v2",
        coordinate_frame="ur_base" if is_single else "left_base", manual_selection=True,
        robot_label="模拟臂", raw_depth_label="raw-realsense", arms=arms,
        calibrations={n: CameraCalibration(n, K, np.eye(4)) for n in names},
        project={"safety": {"max_camera_skew_s": .1}},
        report={"camera_serials": {n: f"mock-{n}" for n in names}, "sha256": "synthetic-calibration"},
        flange_to_tcp={a.name: np.eye(4) for a in arms}, workspace_regions=[(bounds, np.eye(4))],
        workspace_masks=workspace, viewer_html=single.single_viewer_html() if is_single else live.VIEWER_HTML,
        open_grippers=lambda _a: {arms[0].name: gripper}, read_grippers=lambda _g: ({arms[0].name: .5}, "模拟开度"),
        tool_points=lambda *_a: np.zeros((1, 3), np.float32))
    class Receiver:
        def __init__(self, _ip):
            pass
        def getTimestamp(self):
            return 1.
        def getActualQ(self):
            return np.zeros(6)
        def getActualTCPPose(self):
            matrix = flange_transform(np.zeros(6))
            return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
        def disconnect(self):
            pass
    class Source:
        def __init__(self, *, cameras, **_kw):
            self.cameras = cameras
        def start(self):
            pass
        def stop(self):
            pass
        def get_stream_info(self):
            return {}
        def get_color_calibrations(self):
            return {}
        def read(self):
            time.sleep(.03)
            depth = np.ones((24, 32), np.float32)
            depth[7:17, 11:21] = .85
            rgb = np.full((24, 32, 3), [65, 80, 90], np.uint8)
            rgb[7:17, 11:21] = [35, 175, 145]
            if image_scale != 1:
                depth = np.repeat(np.repeat(depth, image_scale, axis=0), image_scale, axis=1)
                rgb = np.repeat(np.repeat(rgb, image_scale, axis=0), image_scale, axis=1)
            stamp = time.monotonic_ns()
            return {c.name: RGBDFrame(c.name, rgb, depth, host_timestamp_ns=stamp) for c in self.cameras}
    monkeypatch.setattr(single, "SinglePikaObstacleScene", lambda _a: scene)
    monkeypatch.setattr(live, "DualObstacleScene", lambda _a: scene)
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    monkeypatch.setattr(live, "RealSenseD435iSource", Source)
    monkeypatch.setattr(live, "collision_surface_samples", lambda **_kw: {})
    monkeypatch.setattr(live, "arm_surface_points", lambda *_a: np.zeros((1, 3), np.float32))
    monkeypatch.setattr(live, "render_surface_points_depth", lambda *_a, width, height, **_kw: np.zeros((height, width)))
    return scene


def test_live_viewer_resumes_saved_annotations_without_a_new_capture(monkeypatch, tmp_path):
    from obstacle_validation import manual_annotation_filter as annotation
    scene = install_synthetic_inputs(monkeypatch, "dual")
    shared = SimpleNamespace(digest=scene.report["sha256"], serials=scene.report["camera_serials"],
                             camera_to_left_base={name: camera.camera_to_world
                                                  for name, camera in scene.calibrations.items()})
    monkeypatch.setattr(annotation, "calibration_from_project", lambda project: shared)
    directory = tmp_path / "latest"
    directory.mkdir()
    labels = {"schema": "dual_rgb_obb_annotation_v2", "annotation_image": "recorded_rgb",
              "camera_names": ["front", "side"], "coordinate_frame": "left_base",
              "reviewed": True, "status": "objects",
              "boxes": {name: [{"object_id": "1", "cx": 16., "cy": 12., "width": 32.,
                                "height": 24., "angle_deg": 0.}] for name in scene.calibrations},
              "cameras": {name: {"rgb_width": 32, "rgb_height": 24,
                                  "stream": {"serial": shared.serials[name], "color_calibration": {
                                      "intrinsics": camera.intrinsics.tolist(), "distortion_model": "none"}}}
                          for name, camera in scene.calibrations.items()}}
    (directory / "annotations.json").write_text(json.dumps(labels))
    checked = []
    class CheckedServer(live.ThreadingHTTPServer):
        def serve_forever(self):
            worker = threading.Thread(target=super().serve_forever, kwargs={"poll_interval": .01})
            worker.start()
            def request(path):
                with urlopen(f"http://127.0.0.1:{self.server_port}" + path, timeout=3) as response:
                    return json.load(response)
            try:
                config = request("/manual-obstacles/config.json")
                assert not config["viewer_filter_default"] and config["annotation_error"] is None
                assert config["annotation_path"] == str(directory / "annotations.json")
                for _ in range(100):
                    snapshot = request("/snapshot.json?viewer=1&render=1&manual_filter=1")
                    if snapshot["viewer_filter"]["state"] == "applied":
                        break
                    time.sleep(.02)
                else:
                    pytest.fail(f"Saved annotation filter never became available: {snapshot}")
                assert snapshot["viewer_filter"]["voxel_count"] > 0
                assert snapshot["viewer_filter"]["scope_policy"] == "dual_rgb_annotation_intersection"
                assert "viewer_filter" not in request("/snapshot.json?compact=1")
                checked.append(True)
            finally:
                self.shutdown()
                worker.join(timeout=5)
    monkeypatch.setattr(live, "ThreadingHTTPServer", CheckedServer)
    live.main(["--port", "0", "--manual-output-dir", str(tmp_path), *FLAGS])
    assert checked == [True]


def test_tcp_failure_snapshot_retains_raw_evidence_and_failed_worker(monkeypatch, tmp_path):
    install_synthetic_inputs(monkeypatch, "dual")
    class Receiver:
        def __init__(self, _ip): pass
        def getTimestamp(self): return 1.
        def getActualQ(self): return np.zeros(6)
        def getActualTCPPose(self):
            matrix = flange_transform(np.zeros(6))
            matrix[0, 3] += .006
            return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
        def disconnect(self): pass
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    class CheckedServer(live.ThreadingHTTPServer):
        def serve_forever(self):
            worker = threading.Thread(target=super().serve_forever, kwargs={"poll_interval": .01})
            worker.start()
            try:
                for _ in range(100):
                    with urlopen(f"http://127.0.0.1:{self.server_address[1]}/snapshot.json?compact=1", timeout=3) as r:
                        snapshot = json.load(r)
                    if snapshot["status"].startswith("error:"):
                        break
                    time.sleep(.02)
                else:
                    pytest.fail("Detector did not report the failed TCP check")
                assert "Active TCP differs" in snapshot["status"]
                assert "occupancy_map" not in snapshot
                assert snapshot["failed_timestamp_ns"] > 0
                assert snapshot["diagnostics"]["acquisition_worker_state"] == "failed"
                check = snapshot["diagnostics"]["active_tcp_checks"]["left_arm"]
                assert check["position_error_mm"] == pytest.approx(6.)
                assert check["controller_timestamp_s"] == 1.
                np.testing.assert_allclose(check["q_rad"], np.zeros(6))
                np.testing.assert_allclose(check["measured_tcp_pose"], Receiver("").getActualTCPPose())
            finally:
                self.shutdown()
                worker.join(timeout=5)
    monkeypatch.setattr(live, "ThreadingHTTPServer", CheckedServer)
    live.main(["--port", "0", "--manual-output-dir", str(tmp_path), *FLAGS])


@pytest.mark.parametrize("scene_name", ["single-pika", "dual"])
@pytest.mark.parametrize("occupancy", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "octomap", "voxblox"])
def test_manual_http_capture_select_and_scene_config(monkeypatch, tmp_path, scene_name, occupancy, backend):
    if backend != "numpy":
        pytest.importorskip("octomap" if backend == "octomap" else "real_scripts._voxblox_native")
        if not occupancy:
            pytest.skip("Native backends require occupancy mapping")
    scene = install_synthetic_inputs(monkeypatch, scene_name)
    original_server = live.ThreadingHTTPServer
    checked = []
    class CheckedServer(original_server):
        def serve_forever(self):
            server_thread = threading.Thread(target=super().serve_forever, kwargs={"poll_interval": .01})
            server_thread.start()
            base = f"http://127.0.0.1:{self.server_address[1]}"
            def request(path, body=None):
                data = None if body is None else json.dumps(body).encode()
                req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=5) as response:
                    return None if response.status == 204 else json.loads(response.read())
            try:
                with urlopen(base + "/", timeout=5) as response:
                    assert b'/octomap-viewer.js' in response.read()
                with urlopen(base + "/octomap-viewer.js", timeout=5) as response:
                    assert "javascript" in response.headers["Content-Type"]
                    assert b'class OctomapLeafViewer' in response.read()
                with urlopen(base + "/voxblox-viewer.js", timeout=5) as response:
                    assert b'class VoxbloxViewer' in response.read()
                with urlopen(base + "/ellipsoid-viewer.js", timeout=5) as response:
                    assert b'class EllipsoidViewer' in response.read()
                config = request("/manual-obstacles/config.json")
                assert config["camera_names"] == list(scene.calibrations)
                assert config["coordinate_frame"] == scene.coordinate_frame
                for _ in range(100):
                    snapshot = request("/snapshot.json?compact=1")
                    if snapshot.get("frame", 0) > 0:
                        break
                    time.sleep(.03)
                else:
                    pytest.fail("Synthetic detector did not publish a frame")
                if backend != "numpy":
                    assert snapshot["obstacle_backend"] == backend
                    assert snapshot["qpos"] == {} and snapshot["gripper_position"] == {}
                assert all(key not in snapshot for key in ("octomap_view", "_octomap_view", "voxblox_view", "_voxblox_view"))
                filtered = request("/snapshot.json?viewer=1&manual_filter=1")
                assert filtered["viewer_filter"]["state"] == ("missing_annotations" if occupancy else "map_unavailable")
                unselected = request("/snapshot.json?viewer=1&render=1&ellipsoids=obstacles")
                assert unselected["ellipsoid_view"]["scope"] == "all_cells"
                assert "viewer_filter" not in unselected
                if occupancy:
                    assert unselected["ellipsoid_view"]["obstacles"]
                unavailable = request("/snapshot.json?viewer=1&render=1&manual_filter=1&ellipsoids=obstacles")
                assert unavailable["ellipsoid_view"]["scope"] == "selected_cells"
                assert not unavailable["ellipsoid_view"]["obstacles"]
                assert "viewer_filter" not in request("/snapshot.json?compact=1&manual_filter=1")
                if not occupancy:
                    with pytest.raises(HTTPError) as error:
                        request("/manual-obstacles/capture", {})
                    assert error.value.code == 400
                    assert "--occupancy-map" in json.load(error.value)["error"]
                    checked.append(True)
                    return
                capture = request("/manual-obstacles/capture", {})
                assert set(capture["images"]) == set(scene.calibrations)
                assert capture["camera_names"] == list(scene.calibrations)
                annotations = {n: [{"object_id": "cube", "cx": 16, "cy": 12, "width": 12,
                                    "height": 12, "angle_deg": 15}] for n in scene.calibrations}
                for mode in ("depth", "obb"):
                    result = request("/manual-obstacles/select", {"capture_id": capture["capture_id"],
                                     "mode": mode, "annotations": annotations})
                    assert result["voxel_count"] > 0 and result["objects"][0]["obb"]
                    assert result["fusion_rule"] == config["fusion_rule"]
                    assert result["coordinate_frame"] == scene.coordinate_frame
                    preview = request("/manual-obstacles/scene.json")
                    assert preview["manual_selection"] == result
                    assert preview["frame"] == capture["frame"]
                    filtered = request("/snapshot.json?viewer=1&manual_filter=1")
                    assert filtered["viewer_filter"]["state"] == "applied"
                    assert filtered["viewer_filter"]["frame"] == filtered["frame"]
                    assert filtered["viewer_filter"]["voxel_count"] > 0
                    assert filtered["viewer_filter"]["voxel_count"] <= filtered["occupancy_map"]["occupied_voxels"]
                    assert filtered["obbs"]  # The unfiltered safety boxes remain present.
                    assert "viewer_filter" not in request("/snapshot.json?viewer=1")
                    rendered = request("/snapshot.json?viewer=1&render=1&manual_filter=1")
                    assert "occupancy_map" not in rendered and "calibration" not in rendered
                    assert rendered["environment"]["points"]
                    assert bool(rendered["robot"]["points"]) is (backend == "numpy")
                    assert all(set(box) == {"corners"} for box in rendered["obbs"])
                    if backend == "octomap":
                        leaves = rendered["octomap_view"]
                        assert leaves["frame"] == rendered["frame"]
                        assert leaves["capture_timestamp_ns"] == rendered["host_timestamp_ns"]
                        assert leaves["coordinate_frame"] == scene.coordinate_frame
                        assert leaves["occupied"]["shown"] > 0
                        assert leaves["free"]["shown"] > 0
                    elif backend == "voxblox":
                        field = rendered["voxblox_view"]
                        assert field["frame"] == rendered["frame"]
                        assert field["capture_timestamp_ns"] == rendered["host_timestamp_ns"]
                        assert field["surface"]["shown"] > 0 and field["field"]["shown"] > 0
                        assert field["esdf_enabled"]
                        assert field["esdf"]["surface"]["shown"] > 0
                        assert field["esdf"]["field"]["shown"] > 0
                    else:
                        assert "octomap_view" not in rendered
                    assert rendered["viewer_filter"]["state"] == "applied"
                    assert rendered["manual_revision"] == rendered["viewer_filter"]["revision"]
                    assert "ellipsoid_view" not in rendered
                    overlay = request("/snapshot.json?viewer=1&render=1&manual_filter=1&ellipsoids=obstacles&after="
                                      + quote(rendered["viewer_revision"], safe=""))
                    assert overlay is not None  # Toggling a layer invalidates the 204 token.
                    assert overlay["ellipsoid_view"]["scope"] == "selected_cells"
                    assert overlay["ellipsoid_view"]["obstacles"]
                    assert not overlay["ellipsoid_view"]["robot"]
                    assert overlay["ellipsoid_view"]["frame"] == overlay["frame"]
                    all_cells = request("/snapshot.json?viewer=1&render=1&ellipsoids=obstacles&after="
                                        + quote(overlay["viewer_revision"], safe=""))
                    assert all_cells is not None
                    assert all_cells["ellipsoid_view"]["scope"] == "all_cells"
                    assert all_cells["ellipsoid_view"]["obstacles"]
                    assert "viewer_filter" not in all_cells
                    frozen = request("/manual-obstacles/scene.json?manual_filter=1&ellipsoids=obstacles")
                    assert frozen["ellipsoid_view"]["scope"] == "frozen_selection"
                    assert frozen["ellipsoid_view"]["obstacles"]
                    assert "ellipsoid_view" not in request("/snapshot.json?compact=1&ellipsoids=obstacles")
                    token = rendered["viewer_revision"]
                    next_poll = request("/snapshot.json?viewer=1&render=1&manual_filter=1&after=" + quote(token, safe=""))
                    assert next_poll is None or next_poll["viewer_revision"] != token
                    assert "occupancy_map" in request("/snapshot.json?compact=1&render=1")
                invalid = {"front": annotations["front"]}
                if scene_name == "single-pika":
                    invalid["side"] = annotations["front"]
                with pytest.raises(HTTPError) as error:
                    request("/manual-obstacles/select", {"capture_id": capture["capture_id"], "annotations": invalid})
                assert error.value.code == 400
                checked.append(True)
            finally:
                self.shutdown()
                server_thread.join(timeout=5)
    monkeypatch.setattr(live, "ThreadingHTTPServer", CheckedServer)
    flags = ["--scene", scene_name, "--port", "0", "--manual-output-dir", str(tmp_path), *FLAGS]
    if not occupancy:
        flags.append("--no-occupancy-map")
    if backend == "octomap":
        flags += ["--no-robot", "--occupancy-backend", "octomap",
                  "--octomap-output", str(tmp_path / "map.ot")]
    elif backend == "voxblox":
        flags += ["--no-robot", "--occupancy-backend", "voxblox", "--voxblox-output", str(tmp_path / "field.npz")]
    live.main(flags)
    assert checked == [True]
    if backend == "octomap":
        assert (tmp_path / "map.ot").is_file()
        placement = json.loads((tmp_path / "map.ot.json").read_text())
        assert placement["coordinate_frame"] == scene.coordinate_frame
    elif backend == "voxblox":
        with np.load(tmp_path / "field.npz") as field:
            assert json.loads(str(field["metadata"]))["coordinate_frame"] == scene.coordinate_frame
            assert field["esdf_observed"].any()
            assert np.isnan(field["esdf_distance_m"][~field["esdf_observed"]]).all()
