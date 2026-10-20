"""Native OcTree integration, temporal evidence, placement and export regressions."""

import json
import sys
import time

import numpy as np
import pytest

from real_scripts import live_dual_ur7e_obstacle_model as live
from real_scripts.octomap_obstacle_map import OctomapOccupancyGrid
from real_scripts.probabilistic_obstacle_map import integrate_depth_views, workspace_occupancy_grid
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_cbf_qp import boxes_from_snapshot
from safety_module.occupancy import FREE, OCCUPIED, ObservationGrid
from safety_module.runtime_contract import validate_snapshot


@pytest.fixture
def native():
    return pytest.importorskip("octomap")


@pytest.fixture
def grid(native):
    return OctomapOccupancyGrid(np.zeros(3), np.eye(3), (8, 4, 4), .1)


def update(grid, hits=(), free=(), completed=(), uncertain=()):
    grid.update(timestamp_ns=grid.timestamp_ns + 1_000_000,
                raw_hits=np.asarray(hits).reshape(-1, 3), free_ids=np.asarray(free, dtype=np.int64),
                completion_hits=np.asarray(completed).reshape(-1, 3),
                uncertain_hits=np.asarray(uncertain).reshape(-1, 3))


def test_cli_default_and_opt_in():
    assert live.parse_args([]).occupancy_backend == "numpy"
    assert live.parse_args([]).octomap_view_max_leaves == 6000
    assert live.parse_args(["--no-robot", "--occupancy-backend", "octomap"]).occupancy_backend == "octomap"
    for flags in (["--occupancy-backend", "octomap", "--no-occupancy-map"],
                  ["--octomap-output", "map.bt"],
                  ["--occupancy-backend", "octomap", "--octomap-output", "map.ply"],
                  ["--octomap-view-max-leaves", "1"]):
        with pytest.raises(SystemExit):
            live.parse_args(flags)


def test_missing_dependency_fails_before_scene_or_hardware(monkeypatch):
    monkeypatch.setitem(sys.modules, "octomap", None)
    monkeypatch.setattr(live, "DualObstacleScene", lambda _args: pytest.fail("scene opened before dependency check"))
    with pytest.raises(RuntimeError, match="pip install.*octomap-python"):
        live.main(["--no-robot", "--occupancy-backend", "octomap", "--once"])


def test_native_probability_deduplication_and_conflicts(grid):
    point = np.array([[.25, .15, .15]])
    ids = grid.point_ids(point)
    update(grid, hits=np.repeat(point, 200, axis=0), free=np.repeat(ids, 200))
    node = grid.tree.search(grid._local_centers(ids)[0])
    assert node.getLogOdds() == pytest.approx(np.log(.7 / .3))
    assert grid.log_odds[ids[0]] == node.getLogOdds()
    assert grid.tree.isNodeOccupied(node)
    assert grid.states()[ids[0]] == OCCUPIED
    assert grid.diagnostics["conflicting_voxels"] == 1


def test_history_requires_raw_free_rays_to_clear(grid):
    ids = grid.point_ids([[.25, .15, .15]])
    update(grid, hits=[[.25, .15, .15]])
    update(grid)  # No new observation must not delete an obstacle.
    assert grid.states()[ids[0]] == OCCUPIED
    assert grid.diagnostics["historical_occupied_voxels"] == 1
    update(grid, free=ids)
    assert grid.states()[ids[0]] == OCCUPIED
    update(grid, free=ids)
    update(grid, free=ids)
    assert grid.states()[ids[0]] == FREE
    assert not grid.tree.isNodeOccupied(grid.tree.search(grid._local_centers(ids)[0]))


def test_fresh_hit_overrides_strong_free_history_and_clamps(grid):
    point = [[.25, .15, .15]]
    ids = grid.point_ids(point)
    for _ in range(20):
        update(grid, free=ids)
    assert grid.log_odds[ids[0]] == pytest.approx(-2.)
    update(grid, hits=point)
    assert grid.states()[ids[0]] == OCCUPIED
    assert grid.log_odds[ids[0]] == pytest.approx(grid.hit_log)
    for _ in range(20):
        update(grid, hits=point)
    assert grid.log_odds[ids[0]] == pytest.approx(3.5)


def test_uncertain_returns_veto_clearing_and_completion_without_erasing_history(grid):
    point = [[.25, .15, .15]]
    ids = grid.point_ids(point)
    update(grid, completed=point, uncertain=point, free=ids)
    assert not grid.known.any() and not grid.current_hits.size
    update(grid, hits=point)
    before = grid.log_odds.copy()
    update(grid, uncertain=point, free=ids)
    np.testing.assert_array_equal(before, grid.log_odds)
    assert grid.states()[ids[0]] == OCCUPIED
    update(grid, completed=[[.35, .15, .15]])
    completed_id = grid.point_ids([[.35, .15, .15]])[0]
    assert grid.log_odds[completed_id] == pytest.approx(.405465)
    assert grid.sources[completed_id] == 4


def test_workspace_and_rotated_scene_snapshot_placement(native):
    normal = np.array([0., 1., 1.]) / np.sqrt(2)
    grid = workspace_occupancy_grid(normal, 0., [((-.3, .3, -.3, .3, 0., .5), np.eye(4))],
                                    resolution_m=.1, grid_class=OctomapOccupancyGrid)
    ids = np.flatnonzero(grid.active)[:4]
    update(grid, hits=grid.centers(ids))
    query = grid._local_centers(ids)
    np.testing.assert_array_equal(grid.tree.getLabels(query), np.ones(len(ids), np.int32))
    snapshot = grid.snapshot()
    view = ObservationGrid(snapshot)
    assert snapshot["backend"] == "octomap" and snapshot["occupied_voxels"] == len(ids)
    transform = np.asarray(snapshot["octomap"]["parent_T_octomap"])
    np.testing.assert_allclose(query @ transform[:3, :3].T + transform[:3, 3], grid.centers(ids))
    assert np.all(view.state[ids] == OCCUPIED)
    assert sum(b.point_count for b in grid.boxes()) == len(ids)


@pytest.mark.parametrize("depth,keep", [(0., True), (np.nan, True), (1., False)])
def test_invalid_depth_never_clears_native_history(native, depth, keep):
    grid = OctomapOccupancyGrid([-.05, -.05, 0.], np.eye(3), (1, 1, 12), .1)
    update(grid, hits=[[0., 0., .85]])
    before = grid.log_odds.copy()
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    integrate_depth_views(grid, [(camera, [[depth]], [[keep]], [[0.]])], [], [],
                          timestamp_ns=grid.timestamp_ns + 1, ray_stride=1)
    np.testing.assert_array_equal(before, grid.log_odds)
    assert grid.states()[8] == OCCUPIED


def test_active_domain_and_invalid_frame_are_enforced(grid):
    grid.active[0] = False
    update(grid, hits=[[.05, .05, .05], [99., 99., 99.]])
    assert grid.tree.size() == 0
    with pytest.raises(ValueError, match="increasing"):
        grid.update(timestamp_ns=grid.timestamp_ns, raw_hits=[], free_ids=np.array([], np.int64))
    with pytest.raises(ValueError, match="exceeds"):
        update(grid, free=[grid.size])


@pytest.mark.parametrize("extension", ["bt", "ot"])
def test_export_round_trip_preserves_live_probability_and_scene_frame(grid, native, tmp_path, extension):
    update(grid, hits=[[.25, .15, .15]])
    odds = grid.log_odds.copy()
    path = tmp_path / "scene" / f"map.{extension}"
    grid.export(path, coordinate_frame="ur_base")
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    assert metadata["coordinate_frame"] == "ur_base"
    assert metadata["capture_timestamp_ns"] == grid.timestamp_ns
    local = grid._local_centers(grid.current_hits)[0]
    restored = native.OcTree(.1)
    if extension == "bt":
        assert restored.readBinary(str(path).encode())
    else:
        restored = native.OcTree.read(path.read_bytes())
        assert restored.search(local).getLogOdds() == pytest.approx(grid.hit_log)
    assert restored.isNodeOccupied(restored.search(local))
    np.testing.assert_array_equal(grid.log_odds, odds)
    assert grid.tree.search(local).getLogOdds() == pytest.approx(grid.hit_log)
    update(grid, free=grid.current_hits)
    assert grid.tree.search(local).getLogOdds() == pytest.approx(grid.hit_log + grid.miss_log)


def test_octomap_snapshot_consumers_require_capture_aligned_map(grid):
    stamp = time.monotonic_ns()
    grid.update(timestamp_ns=stamp, raw_hits=[[.25, .15, .15]], free_ids=np.array([], np.int64))
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "status": "ok", "host_timestamp_ns": stamp, "obstacle_backend": "octomap",
                "occupancy_map": grid.snapshot(), "obbs": []}
    validate_snapshot(snapshot, 1.)
    assert boxes_from_snapshot(snapshot).observation is not None
    snapshot["host_timestamp_ns"] += 1
    with pytest.raises(ValueError, match="capture-aligned"):
        validate_snapshot(snapshot, 1.)
    with pytest.raises(ValueError, match="timestamps"):
        boxes_from_snapshot(snapshot)
    del snapshot["occupancy_map"]
    with pytest.raises(ValueError, match="occupancy map"):
        validate_snapshot(snapshot, 1.)
    with pytest.raises(ValueError, match="missing"):
        boxes_from_snapshot(snapshot)


def test_native_leaves_prune_equal_probabilities_and_preserve_real_extent(native):
    grid = OctomapOccupancyGrid([1., 2., 3.], np.eye(3), (2, 2, 2), .1)
    update(grid, hits=grid.centers(np.arange(8)))
    view = grid.viewer_payload()
    occupied = view["occupied"]
    assert occupied["count"] == occupied["shown"] == 1
    assert occupied["depths"] == [grid.tree.getTreeDepth() - 1]
    np.testing.assert_allclose(occupied["centers"], [[1.1, 2.1, 3.1]])
    assert occupied["sizes_m"] == [.2]
    assert view["free"]["count"] == 0
    assert occupied["probabilities"] == pytest.approx([.7])
    assert sum(box.point_count for box in grid.boxes()) == 8


def test_pruned_free_leaf_splits_on_hit_without_resetting_siblings(native):
    grid = OctomapOccupancyGrid(np.zeros(3), np.eye(3), (2, 2, 2), .1)
    update(grid, free=np.arange(8))
    assert grid.viewer_payload()["free"]["sizes_m"] == [.2]
    point = grid.centers([0])
    update(grid, hits=point)
    assert grid.tree.getNumLeafNodes() == 8
    assert grid.log_odds[0] == pytest.approx(grid.hit_log)
    np.testing.assert_allclose(grid.log_odds[1:], grid.miss_log)
    np.testing.assert_array_equal(grid.tree.getLabels(grid._local_centers(np.arange(8))), [1,0,0,0,0,0,0,0])
    assert grid.viewer_payload()["occupied"]["sizes_m"] == [.1]


def test_native_pruning_never_fills_unknown_gaps_or_merges_different_probabilities(native):
    grid = OctomapOccupancyGrid(np.zeros(3), np.eye(3), (2, 2, 2), .1)
    update(grid, hits=grid.centers(np.arange(7)))
    assert grid.viewer_payload()["occupied"]["sizes_m"] == [.1] * 7
    update(grid, hits=grid.centers(np.arange(8)))
    view = grid.viewer_payload()
    assert view["occupied"]["count"] == 8
    assert max(view["occupied"]["probabilities"]) > min(view["occupied"]["probabilities"])


def test_leaf_sampling_is_bounded_deterministic_and_display_only(grid):
    ids = np.arange(grid.size)
    # Unequal probabilities keep these leaves unpruned; native occupancy stays full.
    update(grid, hits=grid.centers(ids[::2]), free=ids[1::2])
    before, nodes = grid.log_odds.copy(), grid.tree.size()
    view = grid.viewer_payload(max_leaves=6)
    assert view == grid.viewer_payload(max_leaves=6)
    assert sum(view[name]["shown"] for name in ("occupied", "free")) <= 6
    assert sum(view[name]["count"] for name in ("occupied", "free")) == grid.tree.getNumLeafNodes()
    assert view["occupied"]["sampled"] and view["free"]["sampled"]
    for name in ("occupied", "free"):
        group = view[name]
        assert all(len(group[field]) == group["shown"] for field in ("centers", "sizes_m", "probabilities", "depths"))
    np.testing.assert_array_equal(before, grid.log_odds)
    assert nodes == grid.tree.size()


def test_viewer_leaf_centers_follow_rotated_grid_axes(native):
    axes = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    grid = OctomapOccupancyGrid([1., 2., 3.], axes, (2, 2, 2), .1)
    update(grid, hits=grid.centers(np.arange(8)))
    view = grid.viewer_payload()
    np.testing.assert_allclose(view["occupied"]["centers"], [[.9, 2.1, 3.1]])
    np.testing.assert_array_equal(view["axes"], axes)


def test_native_leaf_payload_is_only_in_viewer_responses(grid):
    update(grid, hits=[[.25, .15, .15]])
    snapshot = {"frame": 1, "host_timestamp_ns": grid.timestamp_ns,
                "environment": {"points": [], "colors": []}, "robot": {"points": [], "colors": []},
                "obbs": [], "_octomap_view": grid.viewer_payload()}
    for options in ({}, {"compact": True}, {"compact": True, "viewer": True}):
        payload = json.loads(live.snapshot_payload(snapshot, **options))
        assert "octomap_view" not in payload and "_octomap_view" not in payload
    for options in ({"viewer": True}, {"viewer": True, "render_only": True}):
        payload = json.loads(live.snapshot_payload(snapshot, **options))
        assert payload["octomap_view"]["capture_timestamp_ns"] == grid.timestamp_ns
        assert payload["octomap_view"]["occupied"]["count"] == 1
        assert "_octomap_view" not in payload
