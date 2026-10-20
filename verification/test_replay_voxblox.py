"""Embedded replay acquisition: exact scope, measured clearing, lifecycle, routing."""
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
from obstacle_validation import lerobot_replay as replay
from real_scripts import replay_voxblox as embedded
from real_scripts.real_robot_adapter import CameraCalibration
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, GridGeometry, OCCUPIED, OUTSIDE
from safety_module.runtime_contract import validate_snapshot


@pytest.fixture
def selection(tmp_path):
    pytest.importorskip("real_scripts._voxblox_native")
    grid = LocalOccupancyGrid([-1., -1., 0.], np.eye(3), [40, 40, 50], .05)
    centers = np.array([[.025, .025, .975], [.075, .025, .975]])
    stamp = time.monotonic_ns()-1_000_000
    grid.update(timestamp_ns=stamp, raw_hits=centers, free_ids=np.empty(0, np.int64))
    capture = {"capture_id": "test", "frame": 1, "host_timestamp_ns": stamp,
               "coordinate_frame": "left_base", "annotation_image": "recorded_rgb",
               "calibration": {"sha256": "digest"}}
    saved = {**{k: capture[k] for k in ("capture_id", "frame", "host_timestamp_ns")},
             "fusion_rule": "dual_view_intersection", "voxel_ids": grid.current_hits.tolist(), "voxel_count": 2,
             "objects": [{"voxel_count": 2, "obb": {"center": [.05, .025, .975],
                          "axes": np.eye(3).tolist(), "half_sizes": [.05, .025, .025]}}]}
    for name, value in (("selection.json", saved), ("capture.json", capture), ("occupancy_map.json", grid.snapshot())):
        (tmp_path/name).write_text(json.dumps(value))
    return ManualVoxelFilter(tmp_path/"selection.json")


def cameras():
    K = np.array([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]])
    return {n: CameraCalibration(n, K, np.eye(4)) for n in ("front", "side")}


def update(pipeline, depth=1., *, blocked=False):
    image = np.full((49, 65), depth, np.float32)
    frames = {n: SimpleNamespace(depth_m=image, host_timestamp_ns=time.monotonic_ns()) for n in cameras()}
    return pipeline.update(frames, rendered_views={n: np.zeros_like(image) for n in frames},
                           keep_views={n: np.full(image.shape, not blocked) for n in frames},
                           qpos={n: [0.]*6 for n in ("left", "right")},
                           grippers={"left": 0., "right": 0.}, calibration={"sha256": "digest"})


def test_cropped_grid_keeps_exact_lattice_and_scope(selection):
    grid = embedded.selected_grid(selection)
    assert grid.size < selection.reference.size/20
    assert not grid.esdf_enabled
    assert grid.active.sum() == 2
    np.testing.assert_allclose(grid.centers(np.flatnonzero(grid.active)), selection.selected_centers)
    pipeline = embedded.SelectedTSDFPipeline(selection, cameras())
    result = update(pipeline)
    assert result["occupancy_map"]["field_kind"] == "TSDF"
    assert result["occupancy_map"]["available_fields"] == ["TSDF"]
    assert result["diagnostics"]["replay_voxblox"]["field_kind"] == "TSDF"
    assert validate_snapshot(result, 2.) >= 0
    observed = ObservationGrid(result["occupancy_map"])
    assert (observed.state[~grid.active] == OUTSIDE).all()
    assert (observed.state[grid.active] == OCCUPIED).all()
    assert result["diagnostics"]["manual_selection"]["scope_voxels"] == 2
    assert "distance_m_data" not in result["occupancy_map"].get("voxblox", {})
    assert not observed.query_esdf(selection.selected_centers)
    assert "environment" not in result and "robot" not in result
    replay.snapshot_state(result)


def test_rays_beyond_roi_clear_moved_obstacle_but_robot_mask_does_not(selection):
    pipeline = embedded.SelectedTSDFPipeline(selection, cameras())
    first = update(pipeline)
    assert first["occupancy_map"]["occupied_voxels"] == 2
    for _ in range(3):
        blocked = update(pipeline, depth=1.4, blocked=True)
    assert blocked["occupancy_map"]["occupied_voxels"] == 2
    for _ in range(8):
        cleared = update(pipeline, depth=1.4)
    assert cleared["occupancy_map"]["occupied_voxels"] == 0
    assert cleared["obbs"] == []
    # Saved selection is a scope, never a source of permanent occupied cells.
    assert first["occupancy_map"]["occupied_voxels"] == 2
    assert cleared["diagnostics"]["occupancy"]["voxblox_integrated_rays"] > 0


def test_roi_intersection_handles_rotation_parallel_and_distortion():
    camera = cameras()["front"]
    grid = GridGeometry([-.1, -.1, .5], np.eye(3), [2, 2, 2], .1)
    mask = embedded.roi_ray_mask(camera, (49, 65), grid)
    assert mask[24, 32] and not mask[0, 0]
    behind = GridGeometry([-.1, -.1, -1.], np.eye(3), [2, 2, 2], .1)
    assert not embedded.roi_ray_mask(camera, (49, 65), behind).any()
    camera = CameraCalibration("front", camera.intrinsics, np.eye(4), np.array([.4, -.1, .01, .02, 0.]))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = GridGeometry([.1, -.1, .5], rotation, [2, 2, 2], .1)
    mask = embedded.roi_ray_mask(camera, (49, 65), rotated)
    assert mask[24, 32]
    # Compare every result against the grid's independent finite-ray traversal.
    from real_scripts.camera_geometry import pixel_rays
    y, x = np.indices(mask.shape)
    rays = pixel_rays(np.c_[x.ravel(), y.ravel()], camera.intrinsics, camera.distortion)
    _, ids = rotated.ray_cells(np.zeros((1, 3)), rays*2.5)
    expected = np.zeros(mask.size, bool)
    expected[ids] = True
    np.testing.assert_array_equal(mask, expected.reshape(mask.shape))


def test_thin_endpoints_between_sampled_rays_are_kept_and_blank_depth_rejected(selection):
    pipeline = embedded.SelectedTSDFPipeline(selection, cameras())
    image = np.zeros((49, 65), np.float32)
    # No hit is on the six-pixel TSDF sampling lattice.
    image[25, 33:36] = .975
    frames = {n: SimpleNamespace(depth_m=image, host_timestamp_ns=time.monotonic_ns()) for n in cameras()}
    result = pipeline.update(
        frames, rendered_views={n: np.zeros_like(image) for n in frames},
        keep_views={n: np.ones(image.shape, bool) for n in frames},
        qpos={n: [0.]*6 for n in ("left", "right")}, grippers={"left": 0., "right": 0.},
        calibration={"sha256": "digest"})
    assert result["occupancy_map"]["occupied_voxels"] == 1
    assert result["diagnostics"]["occupancy"]["voxblox_integrated_rays"] == 0
    assert result["diagnostics"]["occupancy"]["current_endpoint_only_voxels"] == 1
    with pytest.raises(RuntimeError, match="no valid measured depth"):
        update(pipeline, depth=0.)


def make_source(monkeypatch, selection, *, digest="digest", timeout=1.):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    monkeypatch.setattr(live, "DualObstacleScene", lambda args: SimpleNamespace(report={"sha256": digest}))
    return embedded.ReplayVoxbloxSource("configs/ur7e_dual_quest3.yaml", selection, startup_timeout_s=timeout)


def test_selection_calibration_rejected_before_acquisition(monkeypatch, selection):
    monkeypatch.setattr(embedded.ReplayVoxbloxSource, "_acquire", lambda self: pytest.fail("opened hardware"))
    with pytest.raises(ValueError, match="calibration differs"):
        make_source(monkeypatch, selection, digest="changed")


def test_source_failure_overrides_cached_latest_and_releases_worker(monkeypatch, selection):
    source = make_source(monkeypatch, selection)
    fail = threading.Event()
    def acquire():
        with source.condition:
            source.latest = {"host_timestamp_ns": 1}
            source.condition.notify_all()
        assert fail.wait(1.)
        raise RuntimeError("camera disconnected")
    monkeypatch.setattr(source, "_acquire", acquire)
    try:
        assert source.read()["host_timestamp_ns"] == 1
        fail.set()
        source.thread.join(1.)
        with pytest.raises(RuntimeError, match="camera disconnected"):
            source.read()
    finally:
        fail.set()
        source.close()
    assert not source.thread.is_alive()


def test_startup_timeout_and_fresh_frame_wait(monkeypatch, selection):
    source = make_source(monkeypatch, selection, timeout=.02)
    monkeypatch.setattr(source, "_acquire", lambda: source.stop.wait(1.))
    try:
        with pytest.raises(TimeoutError, match="camera ownership"):
            source.read()
        with source.condition:
            source.latest = {"host_timestamp_ns": 2}
        with pytest.raises(TimeoutError):
            source.read(after_timestamp_ns=2, max_wait_s=.01)
    finally:
        source.close()


def test_acquisition_initialization_error_closes_partial_hardware(monkeypatch, selection):
    from real_scripts import ur7e_realsense_adapter as adapter
    source = make_source(monkeypatch, selection)
    events = []
    class Receiver:
        def __init__(self, ip): events.append("receive_connect")
        def disconnect(self): events.append("receive_disconnect")
    class Camera:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError("camera busy")
        def stop(self): events.append("camera_stop")
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    monkeypatch.setattr(adapter, "RealSenseD435iSource", Camera)
    source.scene.arms = [SimpleNamespace(name="left", ip="fake")]
    source.scene.calibrations = {"front": None, "side": None}
    source.scene.report["camera_serials"] = {"front": "f", "side": "s"}
    source.scene.open_grippers = lambda args: {"left": SimpleNamespace(close=lambda: events.append("gripper_close"))}
    try:
        with pytest.raises(RuntimeError, match="camera busy"):
            source.read()
    finally:
        source.close()
    assert events == ["receive_connect", "camera_stop", "gripper_close", "receive_disconnect"]


@pytest.mark.parametrize("external", [False, True, "url"])
def test_replay_routes_source_and_cleans_up_on_initialization_error(tmp_path, monkeypatch, selection, external):
    events = []
    class Provider:
        def __init__(self, *args, **kwargs): events.append("provider_created")
        def read(self, **kwargs): events.append("embedded_read"); return {}
        def close(self): events.append("provider_closed")
    class Sampler:
        calibration_digest = "digest"
        def __init__(self, **kwargs): pass
    monkeypatch.setattr(embedded, "ReplayVoxbloxSource", Provider)
    monkeypatch.setattr(replay, "DatasetActionPolicy", lambda path: SimpleNamespace(period=.05))
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "validate_obstacle_snapshot", lambda *args, **kwargs: 0.)
    monkeypatch.setattr(replay, "read_snapshot", lambda *args, **kwargs: events.append("external_read") or {})
    def reject(*args): raise ValueError("test initialization failure")
    monkeypatch.setattr(replay, "validate_calibration_identity", reject)
    arguments = ["replay", "--no-execute", "--dataset", str(tmp_path/"dataset"), "--output", str(tmp_path/"out"),
                 "--mode", "avoid", "--live", "--manual-selection", str(selection.path)]
    if external == "url":
        arguments.extend(["--obstacle-url", "http://existing/snapshot"])
    elif external:
        arguments.extend(["--obstacle-source", "external"])
    with pytest.raises(ValueError, match="test initialization"):
        replay.run_replay(replay.parser().parse_args(arguments))
    if external:
        assert events == ["external_read"]
    else:
        assert events[:2] == ["provider_created", "embedded_read"]
        assert "provider_closed" in events


def test_virtual_replay_completes_with_embedded_map_and_reports_diagnostics(tmp_path, monkeypatch, selection):
    from verification.test_lerobot_action_replay import episode
    from verification.test_obstacle_validation import TwoPointSampler
    dataset = tmp_path/"dataset"
    episode(tmp_path/"episode.hdf5")
    replay.convert_hdf5(tmp_path/"episode.hdf5", dataset)
    snapshot = update(embedded.SelectedTSDFPipeline(selection, cameras()))
    snapshot["gripper_position"] = {"left": .2, "right": .7}
    events = []
    class Provider:
        def __init__(self, *args, **kwargs): events.append("created")
        def read(self, **kwargs): return snapshot
        def close(self): events.append("closed")
    class Sampler(TwoPointSampler):
        calibration_digest = "digest"
        def __init__(self, **kwargs): pass
    monkeypatch.setattr(embedded, "ReplayVoxbloxSource", Provider)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay, "read_snapshot", lambda *a, **k: pytest.fail("unexpected HTTP request"))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args([
        "replay", "--no-execute", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
        "--mode", "avoid", "--manual-selection", str(selection.path)])
    result = replay.run_replay(args)
    assert result["completed"] and result["actions_executed"] == 3
    assert result["obstacle_source"] == "embedded_voxblox"
    assert result["replay_voxblox"]["selected_voxels"] == 2
    assert result["manual_selection_diagnostics"]["scope_voxels"] == 2
    assert result["snapshot_cache"] is None
    assert events == ["created", "closed"]
