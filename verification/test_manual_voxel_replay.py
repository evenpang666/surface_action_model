"""Manual OBB scope must filter live cells while retaining occupied-cell collision checks."""

import json
import io
import time

import numpy as np
import pytest

from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
from obstacle_validation import runtime
from real_scripts.real_cbf_qp import boxes_from_snapshot
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OUTSIDE, FREE, OCCUPIED
from safety_module.runtime_contract import SnapshotUnknown, validate_snapshot


def prepared(tmp_path):
    grid = LocalOccupancyGrid([0., 0., 0.], np.eye(3), [6, 1, 1], .1)
    captured_at = time.monotonic_ns()-10_000_000
    grid.update(timestamp_ns=captured_at, raw_hits=[[.25, .05, .05], [.45, .05, .05]],
                free_ids=np.empty(0, np.int64))
    chosen = int(grid.point_ids([[.25, .05, .05]])[0])
    capture = {"capture_id": "capture", "frame": 1, "host_timestamp_ns": captured_at,
               "coordinate_frame": "left_base", "annotation_image": "recorded_rgb",
               "calibration": {"sha256": "digest"}}
    selection = {"capture_id": "capture", "frame": 1, "host_timestamp_ns": captured_at,
                 "fusion_rule": "dual_view_intersection", "voxel_ids": [chosen], "voxel_count": 1,
                 "objects": [{"voxel_count": 1, "obb": {"center": [.3, .05, .05],
                              "axes": np.eye(3).tolist(), "half_sizes": [.06, .05, .05]}}]}
    for name, value in (("capture.json", capture), ("selection.json", selection),
                        ("occupancy_map.json", grid.snapshot())):
        (tmp_path/name).write_text(json.dumps(value), encoding="utf-8")
    return grid, tmp_path/"selection.json"


def snapshot(grid):
    return {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
            "obstacle_backend": "probabilistic_voxels", "host_timestamp_ns": grid.timestamp_ns,
            "scene_observation_state": "observed", "unknown_obstacles": [],
            "status": "ok", "occupancy_map": grid.snapshot(),
            "calibration": {"sha256": "digest"}, "obbs": [], "diagnostics": {"box_margin_m": .01}}


@pytest.mark.parametrize("backend", ["probabilistic_voxels", "octomap", "voxblox"])
def test_manual_filter_uses_current_occupied_voxels_without_historical_blockage(tmp_path, backend):
    grid, path = prepared(tmp_path)
    manual = ManualVoxelFilter(path)
    grid.update(timestamp_ns=time.monotonic_ns()-3_000_000,
                raw_hits=[[.25, .05, .05], [.35, .05, .05], [.55, .05, .05]], free_ids=np.empty(0, np.int64))
    original = snapshot(grid)
    original["obstacle_backend"] = backend
    assert validate_snapshot(original, 1.) >= 0
    filtered = manual(original)
    assert validate_snapshot(filtered, 1.) >= 0
    assert filtered["diagnostics"]["manual_selection"]["occupied_voxels"] == 1
    assert filtered["diagnostics"]["manual_selection"]["detector_occupied_voxels"] == 3
    assert filtered["diagnostics"]["manual_selection"]["scope_voxels"] == 1
    assert len(filtered["obbs"]) == 1
    np.testing.assert_allclose(filtered["obbs"][0]["center"], [.25, .05, .05])
    np.testing.assert_allclose(filtered["obbs"][0]["half_sizes"], [.05, .05, .05])
    assert filtered["diagnostics"]["box_margin_m"] == 0.
    observed = ObservationGrid(filtered["occupancy_map"])
    assert observed.state[grid.point_ids([[.25, .05, .05]])[0]] == OCCUPIED
    # .35 is enclosed by the saved fitted OBB but was never selected.
    assert observed.state[grid.point_ids([[.35, .05, .05]])[0]] == OUTSIDE
    assert observed.state[grid.point_ids([[.45, .05, .05]])[0]] == OUTSIDE
    assert observed.state[grid.point_ids([[.55, .05, .05]])[0]] == OUTSIDE
    assert not observed.segments_observed([[.25, .05, .05]], [[.26, .05, .05]], now_ns=grid.timestamp_ns)
    assert observed.segments_observed([[.34, .05, .05]], [[.36, .05, .05]], now_ns=grid.timestamp_ns)
    assert observed.segments_observed([[.45, .05, .05]], [[.46, .05, .05]], now_ns=grid.timestamp_ns)
    boxes = boxes_from_snapshot(filtered)
    assert len(boxes) == 1
    # A path 5 mm beyond the occupied cell is free even if the upstream
    # detector still advertises its old 10 mm OBB inflation.
    from obstacle_validation.core import segments_within_box_margin
    start, end = np.array([[.305, .02, .05]]), np.array([[.305, .08, .05]])
    assert observed.segments_observed(start, end, now_ns=grid.timestamp_ns)
    assert not segments_within_box_margin(start, end, boxes[0], 0.).any()
    grid.update(timestamp_ns=grid.timestamp_ns+1_000_000, raw_hits=[[.35, .05, .05]],
                free_ids=np.empty(0, np.int64))
    cleared = manual(snapshot(grid))
    assert cleared["obbs"] == []
    assert ObservationGrid(cleared["occupancy_map"]).state[manual.cell_scope.ids[0]] == FREE


def test_manual_voxblox_scope_disables_unrecomputed_global_esdf_guidance(tmp_path):
    grid, path = prepared(tmp_path)
    original = snapshot(grid)
    original['obstacle_backend'] = 'voxblox'
    original['occupancy_map'].update(backend='voxblox', voxblox={
        'esdf': {'enabled': True, 'distance_dtype': 'float32-le'}})
    filtered = ManualVoxelFilter(path)(original)
    field = filtered['occupancy_map']['voxblox']['esdf']
    assert field['enabled'] and not field['guidance_usable']
    assert field['guidance_disabled_reason'] == 'manual_obstacle_scope_not_recomputed'
    assert 'guidance_usable' not in original['occupancy_map']['voxblox']['esdf']
    assert ObservationGrid(filtered['occupancy_map']).query_esdf([[.25, .05, .05]]) is None


def test_manual_filter_reprojects_selected_cells_onto_a_changed_live_grid(tmp_path):
    _, path = prepared(tmp_path)
    manual = ManualVoxelFilter(path)
    shifted = LocalOccupancyGrid([.01, 0., 0.], np.eye(3), [7, 1, 1], .1)
    shifted.update(timestamp_ns=time.monotonic_ns()-3_000_000,
                   raw_hits=[[.26, .05, .05], [.36, .05, .05]], free_ids=np.empty(0, np.int64))
    filtered = manual(snapshot(shifted))
    assert validate_snapshot(filtered, 1.) >= 0
    assert filtered["occupancy_map"]["origin"] == [.01, 0., 0.]
    assert filtered["occupancy_map"]["shape"] == [7, 1, 1]
    assert filtered["diagnostics"]["manual_selection"]["occupied_voxels"] == 1
    np.testing.assert_allclose(filtered["obbs"][0]["center"], [.26, .05, .05])
    angle = .02
    axes = np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = LocalOccupancyGrid([0., -.05, 0.], axes, [6, 2, 1], .1)
    hit = rotated.centers([4])[0]
    rotated.update(timestamp_ns=time.monotonic_ns()-2_000_000,
                   raw_hits=[hit], free_ids=np.empty(0, np.int64))
    rotated_result = manual(snapshot(rotated))
    assert rotated_result["diagnostics"]["manual_selection"]["occupied_voxels"] == 1
    np.testing.assert_allclose(rotated_result["obbs"][0]["axes"], axes)


def test_manual_filter_rejects_calibration_uncovered_workspace_and_other_unknown(tmp_path):
    grid, path = prepared(tmp_path)
    manual = ManualVoxelFilter(path)
    current = snapshot(grid)
    current["calibration"] = {"sha256": "changed"}
    with pytest.raises(ValueError, match="calibration"):
        manual(current)
    far = LocalOccupancyGrid([1., 0., 0.], np.eye(3), [6, 1, 1], .1)
    far.update(timestamp_ns=time.monotonic_ns()-3_000_000,
               raw_hits=[[1.25, .05, .05]], free_ids=np.empty(0, np.int64))
    with pytest.raises(ValueError, match="outside the live occupancy workspace"):
        manual(snapshot(far))
    current = snapshot(grid)
    current["unknown_obstacles"] = [{"reason": "insufficient_measured_workspace_depth"}]
    with pytest.raises(SnapshotUnknown):
        validate_snapshot(manual(current), 1.)


def test_manual_filter_requires_nonempty_saved_selection(tmp_path):
    _, path = prepared(tmp_path)
    selected = json.loads(path.read_text())
    selected["voxel_ids"] = []
    selected["voxel_count"] = 0
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="nonempty"):
        ManualVoxelFilter(path)


def test_live_reader_filters_before_snapshot_validation(tmp_path, monkeypatch):
    grid, path = prepared(tmp_path)
    grid.update(timestamp_ns=time.monotonic_ns()-3_000_000,
                raw_hits=[[.25, .05, .05]], free_ids=np.empty(0, np.int64))
    current = snapshot(grid)
    current["qpos"] = {"left": [0.] * 6, "right": [0.] * 6}
    current["gripper_position"] = {"left": 0., "right": 0.}
    wire = json.dumps(current).encode()
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: io.BytesIO(wire))
    assert runtime.read_snapshot("http://localhost/snapshot.json", 1.)["host_timestamp_ns"] == grid.timestamp_ns
    filtered = runtime.read_snapshot("http://localhost/snapshot.json", 1.,
                                     snapshot_transform=ManualVoxelFilter(path))
    assert len(filtered["obbs"]) == 1


def test_startup_wait_retries_expired_manual_frame(tmp_path, monkeypatch):
    grid, path = prepared(tmp_path)
    grid.update(timestamp_ns=time.monotonic_ns()-3_000_000,
                raw_hits=[[.25, .05, .05]], free_ids=np.empty(0, np.int64))
    fresh = snapshot(grid)
    fresh["qpos"] = {"left": [0.] * 6, "right": [0.] * 6}
    fresh["gripper_position"] = {"left": 0., "right": 0.}
    old = json.loads(json.dumps(fresh))
    old["host_timestamp_ns"] -= 10_000_000_000
    old["occupancy_map"]["capture_timestamp_ns"] = old["host_timestamp_ns"]
    replies = iter([old, fresh])
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(next(replies)).encode()))
    filtered = runtime.read_snapshot("http://localhost/snapshot.json", 1., max_wait_s=.5,
                                     snapshot_transform=ManualVoxelFilter(path))
    assert filtered["host_timestamp_ns"] == fresh["host_timestamp_ns"]


def test_manual_filter_reports_original_detector_failure_and_actual_backend(tmp_path):
    grid, path = prepared(tmp_path)
    manual = ManualVoxelFilter(path)
    failure = "error: RuntimeError: right_arm: Active TCP differs from fixed-surface FK: 7.198 mm / 1.129 deg"
    with pytest.raises(RuntimeError, match="Active TCP differs.*7.198 mm"):
        manual({"status": failure, "coordinate_frame": "left_base"})
    current = snapshot(grid)
    current["obstacle_backend"] = "clustered_obbs"
    with pytest.raises(ValueError, match="NumPy, OctoMap or Voxblox.*clustered_obbs"):
        manual(current)
    current.pop("obstacle_backend")
    current["occupancy_map"]["backend"] = "voxblox"
    assert manual(current)["obbs"]


def test_startup_wait_retries_initializing_before_manual_filter(tmp_path, monkeypatch):
    grid, path = prepared(tmp_path)
    fresh = snapshot(grid)
    fresh.update(qpos={"left": [0.] * 6, "right": [0.] * 6},
                 gripper_position={"left": 0., "right": 0.})
    replies = iter([{"status": "initializing"}, fresh])
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(next(replies)).encode()))
    filtered = runtime.read_snapshot("http://localhost/snapshot.json", 1., max_wait_s=.5,
                                     snapshot_transform=ManualVoxelFilter(path))
    assert filtered["obbs"]
