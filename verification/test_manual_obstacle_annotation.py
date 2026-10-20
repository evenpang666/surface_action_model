"""Standalone annotation persistence, camera lifetime and HTTP contract."""

import argparse
import json
from pathlib import Path
import re
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np
import pytest

from real_scripts import manual_obstacle_annotation as annotation
from real_scripts.batch_obstacle_annotation import BatchAnnotations, GROUP, handler_for
from real_scripts.real_robot_adapter import RGBDFrame


def capture_frames(names):
    return {name: RGBDFrame(name, np.full((24, 32, 3), [20, 80, 160], np.uint8),
                           np.ones((24, 32)), host_timestamp_ns=123456789)
            for name in names}, {name: {"serial": f"mock-{name}"} for name in names}


def labels(names, **kwargs):
    return {"episode_id": 0, "frame_index": 0, "revision": 0, "status": "objects",
            "boxes": {name: [{"object_id": "cube", "cx": 16, "cy": 12,
                              "width": 8, "height": 6, "angle_deg": 30}] for name in names}, **kwargs}


def episode(path, names):
    with h5py.File(path, "w") as source:
        source.create_dataset("timestamps_ns", data=[1700000000000000000, 1700000000000000001])
        for name in names:
            source.create_dataset(f"observations/images/{name}/rgb", data=np.zeros((2, 24, 32, 3), np.uint8))
    return path


@pytest.mark.parametrize("names", [("front",), ("front", "side")])
def test_hdf5_file_selection_resume_and_exact_frame_alignment(tmp_path, names):
    selected = episode(tmp_path / "selected.hdf5", names)
    other = episode(tmp_path / "other.hdf5", names)
    untouched = other.read_bytes()
    dataset = BatchAnnotations(selected)
    assert dataset.paths == [selected]
    dataset.save(labels(names, frame_index=1))
    resumed = BatchAnnotations(selected)
    assert resumed.frame(0, 1)["boxes"] == labels(names)["boxes"]
    assert not resumed.frame(0, 0)["reviewed"]
    with h5py.File(selected) as source:
        saved = json.loads(source[GROUP + "/frames_json"][1])
        assert saved["timestamp_ns"] == "1700000000000000001"
        assert source[GROUP + "/reviewed"][:].tolist() == [False, True]
    assert other.read_bytes() == untouched


@pytest.mark.parametrize("names", [("front",), ("front", "side")])
def test_camera_frozen_capture_time_names_and_explicit_save(tmp_path, names):
    calls = []
    def capture():
        calls.append(True)
        return capture_frames(names)
    dataset = annotation.CameraAnnotations(tmp_path, capture, camera_names=names)
    assert dataset.manifest()["episodes"] == [] and not calls
    frame = dataset.capture()
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}[+-]\d{4}", frame["capture_id"])
    directory = Path(frame["output_dir"])
    assert (directory / "front_rgb.png").is_file()
    assert not (directory / "annotations.json").exists()
    result = dataset.save(labels(names))
    saved = json.loads(Path(result["output_path"]).read_text())
    assert saved["boxes"] == labels(names)["boxes"]
    assert saved["cameras"]["front"]["host_timestamp_ns"] == "123456789"
    assert dataset.frame(0, 0)["reviewed"] and len(calls) == 1
    with pytest.raises(ValueError, match="another window"):
        dataset.save(labels(names))
    dataset.save(labels(names, revision=1, status="no_obstacles", boxes={name: [] for name in names}))
    assert dataset.frame(0, 0)["boxes"] == {name: [] for name in names}
    second = dataset.capture()
    assert second["capture_id"] != frame["capture_id"]
    assert dataset.frame(0, 0)["revision"] == 2


@pytest.mark.parametrize("mode", ["hdf5", "camera"])
def test_standalone_http_read_capture_save_and_invalid_requests(tmp_path, mode):
    names = ("front",)
    dataset = (BatchAnnotations(episode(tmp_path / "test.h5", names)) if mode == "hdf5" else
               annotation.CameraAnnotations(tmp_path, lambda: capture_frames(names), camera_names=names))
    with annotation.ThreadingHTTPServer(("127.0.0.1", 0), handler_for(dataset)) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        def request(path, body=None, **headers):
            req = Request(url + path, data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **headers})
            with urlopen(req, timeout=5) as response:
                return json.load(response)
        try:
            with urlopen(url) as response:
                assert b'id="capture"' in response.read()
            if mode == "camera":
                assert request("/api/manifest")["total_frames"] == 0
                assert request("/api/capture", {})["camera_names"] == ["front"]
            else:
                with pytest.raises(HTTPError) as error:
                    request("/api/capture", {})
                assert error.value.code == 404
            assert set(request("/api/frame?episode_id=0&frame_index=0")["images"]) == {"front"}
            with pytest.raises(HTTPError) as error:
                request("/api/save", labels(names), Origin="http://example.com")
            assert error.value.code == 403
            assert request("/api/save", labels(names))["reviewed"]
            assert request("/api/manifest")["reviewed_frames"] == 1
            with pytest.raises(HTTPError) as error:
                request("/api/save", labels(names))
            assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("front_serial", ["a", None])
@pytest.mark.parametrize("side_serial", ["b", None])
def test_camera_warmup_and_release_even_on_failure(monkeypatch, fail, front_serial, side_serial):
    from real_scripts import ur7e_realsense_adapter as adapter
    calls = []
    class Source:
        def __init__(self, **kwargs):
            assert [camera.name for camera in kwargs["cameras"]] == ["front", "side"]
            assert {camera.name: camera.serial for camera in kwargs["cameras"]} == {"front": front_serial or "260922305390", "side": side_serial or "260922303231"}
        def start(self): calls.append("start")
        def read(self):
            calls.append("read")
            if fail:
                raise RuntimeError("camera unavailable")
            return capture_frames(("front", "side"))[0]
        def get_stream_info(self): return {name: {"serial": name} for name in ("front", "side")}
        def get_color_calibrations(self):
            return {name: {"width": 640, "height": 480, "intrinsics": np.eye(3).tolist()}
                    for name in ("front", "side")}
        def stop(self): calls.append("stop")
    monkeypatch.setattr(adapter, "RealSenseD435iSource", Source)
    args = argparse.Namespace(project_config=annotation.DEFAULT_PROJECT_CONFIG, front_serial=front_serial, side_serial=side_serial, width=640, height=480, fps=15, warmup_frames=2)
    if fail:
        with pytest.raises(RuntimeError, match="unavailable"):
            annotation.camera_capture(args)
    else:
        frames, streams = annotation.camera_capture(args)
        assert list(frames) == ["front", "side"]
        assert streams["front"]["color_calibration"]["height"] == 480
        assert calls.count("read") == 3
    assert calls[0] == "start" and calls[-1] == "stop"


def test_hdf5_inspect_never_touches_camera(monkeypatch, tmp_path, capsys):
    path = episode(tmp_path / "test.h5", ("front",))
    def unexpected(*_):
        pytest.fail("Offline inspection touched a camera")
    monkeypatch.setattr(annotation, "camera_capture", unexpected)
    annotation.main(["--hdf5", str(path), "--inspect"])
    assert json.loads(capsys.readouterr().out)["total_frames"] == 2


def test_camera_defaults_to_both_views_and_rejects_missing_side(tmp_path):
    dataset = annotation.CameraAnnotations(tmp_path, lambda: capture_frames(("front",)))
    assert dataset.manifest()["camera_names"] == ["front", "side"]
    with pytest.raises(ValueError, match="front, side"):
        dataset.capture()
    assert dataset.manifest()["total_frames"] == 0
    assert not list(tmp_path.iterdir())


def test_default_camera_serials_match_hardware_and_all_calibrations():
    import yaml
    from safety_module.project_config import load_project_config, resolve_repo_path
    args = argparse.Namespace(project_config=annotation.DEFAULT_PROJECT_CONFIG, front_serial=None, side_serial=None)
    serials = annotation.configured_camera_serials(args)
    assert serials == {"front": "260922305390", "side": "260922303231"}
    project, _ = load_project_config(args.project_config)
    for name in ("front", "side"):
        for key in (f"{name}_calibration", f"right_{name}_calibration"):
            with resolve_repo_path(project["safety"][key]).open() as handle:
                assert yaml.safe_load(handle)["camera_serial"] == serials[name]


def test_camera_serials_follow_selected_hardware_config_and_reject_duplicates(monkeypatch, tmp_path):
    import yaml
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(yaml.safe_dump({"cameras": {"front": {"serial": "custom-front"},
                                                  "side": {"serial": "custom-side"}}}))
    monkeypatch.setattr(annotation, "load_project_config", lambda path: ({"quest3": {"hardware_config": str(hardware)}}, path))
    args = argparse.Namespace(project_config=tmp_path / "project.yaml", front_serial=None, side_serial=None)
    assert annotation.configured_camera_serials(args) == {"front": "custom-front", "side": "custom-side"}
    args.front_serial = "custom-side"
    with pytest.raises(ValueError, match="different"):
        annotation.configured_camera_serials(args)
