"""Scenario checks for the map authority and all robot motion validation paths."""

import json
import time

import numpy as np
import pytest

from real_scripts.probabilistic_obstacle_map import integrate_depth_views, workspace_occupancy_grid
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_cbf_qp import boxes_from_snapshot, validate_fk_rollout
from safety_module.occupancy import (
    FREE, OCCUPIED, OUTSIDE, UNKNOWN, GridGeometry, LocalOccupancyGrid, ObservationGrid,
)
from safety_module.runtime_contract import SnapshotUnknown, validate_snapshot


def grid(shape=(8, 4, 4), **kwargs):
    return LocalOccupancyGrid(np.zeros(3), np.eye(3), shape, .1, **kwargs)


def update(target, hits=(), free=(), completed=(), stamp=None):
    target.update(timestamp_ns=target.timestamp_ns + 1_000_000 if stamp is None else stamp,
                  raw_hits=np.asarray(hits).reshape(-1, 3),
                  completion_hits=np.asarray(completed).reshape(-1, 3), free_ids=np.asarray(free, dtype=np.int64))


def test_endpoint_occupied_ray_before_it_free_and_behind_it_unknown():
    target = grid((1, 1, 12))
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    target.origin[:2] = -.05
    depth = np.array([[.85]])
    for _ in range(3):
        integrate_depth_views(target, [(camera, depth, np.ones((1, 1), bool), np.zeros((1, 1)))],
                              [[0., 0., .85]], [], timestamp_ns=target.timestamp_ns+1,
                              ray_stride=1, free_margin_m=.015)
    assert np.all(target.states()[:8] == FREE)
    assert target.states()[8] == OCCUPIED
    assert np.all(target.states()[9:] == UNKNOWN)
    assert target.sources[0] == 2 and target.sources[8] == 1
    assert 1 / (1 + np.exp(-target.log_odds[8])) > .7


@pytest.mark.parametrize("depth,keep", [(0., True), (np.nan, True), (1., False)])
def test_invalid_and_robot_masked_depth_do_not_clear(depth, keep):
    target = grid((1, 1, 12))
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    target.origin[:2] = -.05
    integrate_depth_views(target, [(camera, np.array([[depth]]), np.array([[keep]]), np.zeros((1, 1)))],
                          [], [], timestamp_ns=1, ray_stride=1)
    assert not target.last_observed_ns.any()
    assert np.all(target.states() == UNKNOWN)


def test_nearer_robot_render_truncates_inconsistent_background_ray():
    target = grid((1, 1, 12))
    target.origin[:2] = -.05
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    for _ in range(3):
        integrate_depth_views(target, [(camera, np.array([[1.1]]), np.ones((1, 1), bool), np.array([[.45]]))],
                              [], [], timestamp_ns=target.timestamp_ns+1, ray_stride=1)
    assert np.all(target.states()[:4] == FREE)
    assert np.all(target.states()[4:] == UNKNOWN)


def test_multiview_conflict_is_order_independent_and_voxel_updates_once():
    a, b = grid(), grid()
    point = np.array([[.25, .15, .15]])
    ids = a.point_ids(point)
    update(a, hits=np.repeat(point, 100, axis=0), free=np.repeat(ids, 500))
    update(b, hits=point, free=ids[::-1])
    np.testing.assert_array_equal(a.log_odds, b.log_odds)
    assert a.states()[ids[0]] == OCCUPIED
    assert a.diagnostics["conflicting_voxels"] == 1
    assert not a.free_streak.any()


def test_new_raw_hit_overrides_strong_free_history_without_confirmation_delay():
    target = grid()
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    for _ in range(20):
        update(target, free=ids)
    assert target.states()[ids[0]] == FREE
    update(target, hits=point)
    assert target.states()[ids[0]] == OCCUPIED
    assert target.log_odds[ids[0]] == pytest.approx(target.hit_log)
    assert target.boxes()


def test_lost_obstacle_becomes_unknown_without_box_and_never_expires():
    target = grid()
    update(target, hits=[[.25, .15, .15]])
    assert len(target.boxes()) == 1
    for _ in range(30):
        update(target)
        assert not target.boxes()
        assert target.unknown_obstacles[0]["voxel_count"] == 1
        assert not {"center", "axes", "half_sizes", "corners"} & target.unknown_obstacles[0].keys()
    previous = target.log_odds.copy()
    update(target, stamp=target.timestamp_ns+10_000_000_000)
    assert target.log_odds.sum() < previous.sum() / 100
    assert target.unknown_obstacles


def test_only_consecutive_raw_free_evidence_resolves_historical_unknown():
    target = grid()
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    update(target, hits=point)
    update(target, free=ids)
    update(target, free=ids)
    assert target.unknown_obstacles
    update(target)
    for _ in range(2):
        update(target, free=ids)
        assert target.unknown_obstacles
    # Belief and consecutive-frame thresholds both have to be satisfied.
    for _ in range(4):
        update(target, free=ids)
    assert not target.unknown_obstacles
    assert target.states()[ids[0]] == FREE
    update(target, hits=point)
    assert not target.unknown_obstacles  # Reobserved as occupied immediately.


def test_completion_cannot_clear_unknown_or_supply_free_rays():
    target = grid()
    point = [[.25, .15, .15]]
    update(target, hits=point)
    update(target)
    assert target.unknown_obstacles
    update(target, completed=point)
    assert target.sources[target.point_ids(point)[0]] == 4
    assert target.states()[target.point_ids(point)[0]] == OCCUPIED
    assert not (target.states() == FREE).any()
    update(target)
    assert target.unknown_obstacles


def test_free_space_ages_to_unknown_including_after_snapshot_publication():
    target = grid(free_ttl_s=.2)
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    for _ in range(3):
        update(target, free=ids)
    captured = target.timestamp_ns
    view = ObservationGrid(target.snapshot())
    assert view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=captured)
    assert not view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=captured+201_000_000)
    assert target.states(now_ns=captured+201_000_000)[ids[0]] == UNKNOWN
    assert not target.unknown_obstacles  # Expired free is not a lost obstacle.


def test_map_snapshot_round_trip_keeps_unobserved_cells_and_probability_history():
    target = grid()
    update(target, hits=[[.25, .15, .15]])
    payload = json.loads(json.dumps(target.snapshot()))
    view = ObservationGrid(payload)
    np.testing.assert_array_equal(view.state, target.states())
    assert payload["occupied_voxels"] == 1
    assert payload["unknown_voxels"] == target.size - 1
    assert payload["capture_timestamp_ns"] == target.last_observed_ns.max()
    assert target.log_odds.sum() > 0


def test_probability_decay_continues_at_consumer_without_refreshing_capture():
    target = grid(free_ttl_s=10.)
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    for _ in range(3):
        update(target, free=ids)
    view = ObservationGrid(target.snapshot())
    later = target.timestamp_ns+1_100_000_000
    assert target.states(now_ns=later)[ids[0]] == UNKNOWN
    assert not view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=later)


def test_small_tiles_pack_only_occupied_cuboids_preserving_gaps():
    target = grid((12, 4, 4))
    cells = np.array([[x, y, z] for x in (1, 2, 5, 6, 7, 8)
                      for y in (1, 2) for z in (1, 2)])
    ids = cells @ target.strides
    update(target, hits=target.centers(ids))
    boxes = target.boxes(tile_voxels=4)
    assert len(boxes) > 1
    assert all(np.max(box.extents) <= .4+1e-10 for box in boxes)
    covered = np.zeros(target.size, bool)
    centers = target.centers(np.arange(target.size))
    for box in boxes:
        covered |= np.all(np.abs((centers-box.center) @ box.rotation) < box.extents/2, axis=1)
    np.testing.assert_array_equal(np.flatnonzero(covered), np.sort(ids))


def test_table_frame_and_dual_workspace_union_are_fixed_and_do_not_fill_gap():
    transform = np.eye(4)
    transform[0, 3] = 1.
    regions = [([0., .3, 0., .3, 0., .6], np.eye(4)),
               ([0., .3, 0., .3, 0., .6], transform)]
    target = workspace_occupancy_grid([0., 0., 1.], 0., regions, resolution_m=.05)
    mid, inside = target.indices([[.65, .15, .3]])
    assert inside[0]
    assert target.states()[mid[0]] == OUTSIDE
    normal = np.array([0., 1., 1.]) / np.sqrt(2)
    tilted = workspace_occupancy_grid(normal, 0., regions, resolution_m=.05)
    np.testing.assert_allclose(tilted.axes[:, 2], normal)


@pytest.mark.parametrize("a,b,expected", [
    ([.05, .05, .05], [.35, .05, .05], [0, 1, 2, 3]),
    ([.35, .05, .05], [.05, .05, .05], [0, 1, 2, 3]),
    ([-.2, .05, .05], [.25, .05, .05], [0, 1, 2]),
    ([.05, .05, .05], [.3, .05, .05], [0, 1, 2, 3]),
])
def test_ray_traversal_covers_axial_clipping_reverse_and_boundary_endpoint(a, b, expected):
    geometry = GridGeometry([0., 0., 0.], np.eye(3), [5, 1, 1], .1)
    ids, _ = geometry.ray_cells([a], [b])
    assert np.unique(ids).tolist() == expected
    unique, _ = geometry.ray_cells([a], [b], unique_only=True)
    np.testing.assert_array_equal(unique, np.unique(ids))


class Sampler:
    left_link_names, points_per_link = ["left"], 1
    def link_points(self, q, grip):
        return np.array([[.15+q[0]+grip[0]*.4, .15, .15], [.15, .35, .15]])


def observed_boxes(*, unknown_cell=None):
    target = grid()
    free = np.arange(target.size)
    if unknown_cell is not None:
        free = free[free != np.asarray(unknown_cell) @ target.strides]
    stamp = time.monotonic_ns()-3_000_000
    for index in range(3):
        update(target, free=free, stamp=stamp+index*1_000_000)
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "status": "ok", "obstacle_backend": "probabilistic_voxels",
                "host_timestamp_ns": target.timestamp_ns, "occupancy_map": target.snapshot(), "obbs": []}
    return boxes_from_snapshot(snapshot), snapshot


def test_fk_rrt_and_replay_fallbacks_reject_unknown_interior_with_free_endpoints():
    from obstacle_validation.core import CollisionChecker
    from obstacle_validation.lerobot_replay import _motion_safe
    boxes, snapshot = observed_boxes(unknown_cell=[3, 1, 1])
    assert validate_snapshot(snapshot, 1.) >= 0
    q, grip, delta = np.zeros(12), np.zeros(2), np.zeros(12)
    delta[0] = .4
    checker = CollisionChecker(Sampler(), grip, boxes, margin=0., inter_arm_margin=.03)
    assert checker.state_safe(q) and checker.state_safe(q+delta)
    assert not checker.edge_safe(q, q+delta)
    assert checker.observation_rejections["unknown"] > 0
    assert checker.last_observation_blockage["reason"] == "unknown"
    assert not validate_fk_rollout(Sampler(), q, grip, [delta], boxes, point_indices=np.array([0, 1]),
                                  left_mask=np.array([True, False]), collision_margin_m=0., inter_arm_margin_m=.03)
    assert not _motion_safe(Sampler(), q, grip, delta, grip, boxes,
                            {"collision_margin_m": 0., "inter_arm_margin_m": .03}, True)
    safe_boxes, _ = observed_boxes()
    assert _motion_safe(Sampler(), q, grip, delta, grip, safe_boxes,
                        {"collision_margin_m": 0., "inter_arm_margin_m": .03})


def test_gripper_only_motion_and_subvoxel_creeping_cannot_bypass_unknown():
    boxes, _ = observed_boxes(unknown_cell=[3, 1, 1])
    assert not validate_fk_rollout(Sampler(), np.zeros(12), np.zeros(2), [np.zeros(12)], boxes,
                                  point_indices=np.array([0, 1]), left_mask=np.array([True, False]),
                                  collision_margin_m=0., inter_arm_margin_m=.03, gripper_chunk=[[.5, 0.]])
    target = grid()
    update(target)
    view = ObservationGrid(target.snapshot())
    assert not view.segments_observed([[.15, .15, .15]], [[.151, .15, .15]], now_ns=target.timestamp_ns)
    assert view.segments_observed([[.15, .15, .15]], [[.15, .15, .15]], now_ns=target.timestamp_ns)


def test_motion_margin_checks_unknown_at_the_local_map_boundary():
    target = grid()
    update(target)
    view = ObservationGrid(target.snapshot())
    assert view.segments_observed([[.81, .15, .15]], [[.815, .15, .15]], now_ns=target.timestamp_ns)
    assert not view.segments_observed([[.81, .15, .15]], [[.815, .15, .15]],
                                       margin_m=.01, now_ns=target.timestamp_ns)


@pytest.mark.parametrize("state", [UNKNOWN, OCCUPIED, FREE])
@pytest.mark.parametrize("angle", [0., .7])
def test_voxel_margin_uses_swept_distance_instead_of_full_neighbour_cells(state, angle):
    axes = np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    active = np.zeros(64, bool); active[21] = True  # cell (1,1,1)
    target = LocalOccupancyGrid([1., .2, -.3], axes, [4, 4, 4], .02, active=active)
    stamp = time.monotonic_ns()-3_000_000
    for index in range(3):
        update(target, hits=target.centers([21]) if state == OCCUPIED else (),
               free=[21] if state == FREE else (), stamp=stamp+index*1_000_000)
    view = ObservationGrid(target.snapshot())
    now = target.timestamp_ns+(3_000_000_000 if state == FREE else 0)

    def safe(a, b):
        return view.segments_observed(np.array([a]) @ axes.T+target.origin,
                                       np.array([b]) @ axes.T+target.origin,
                                       margin_m=.01, now_ns=now)

    # 19 mm separation is safe even in the adjacent 20 mm cell.
    assert safe([.059, .03, .03], [.059, .031, .03])
    assert view.last_blockage is None
    # The guard has rounded corners: sqrt(8²+8²) mm exceeds 10 mm.
    assert safe([.048, .048, .03], [.048, .049, .03])
    assert not safe([.049, .03, .03], [.049, .031, .03])
    assert view.last_blockage["reason"] == {
        UNKNOWN: "unknown", OCCUPIED: "occupied", FREE: "expired_free"}[state]
    assert not safe([.047, .047, .03], [.047, .048, .03])
    # Endpoints are safe but the segment passes within 8 mm of the cell.
    assert not safe([.01, .048, .03], [.05, .048, .03])


def test_restoration_can_advance_past_an_unknown_neighbour_outside_margin():
    from obstacle_validation.core import CollisionChecker, JointPath, PathFollower
    from safety_module.occupancy import ObservedBoxes

    active = np.zeros(64, bool); active[21] = True
    target = LocalOccupancyGrid(np.zeros(3), np.eye(3), [4, 4, 4], .02, active=active)
    update(target)

    class Points:
        left_link_names, points_per_link = ["left"], 1
        def link_points(self, q, grip):
            return np.array([[[q[0], .03, .03]], [[3., 3., 3.]]])

    q = np.zeros((2, 12)); q[:, 0] = [.059, .061]
    path = JointPath(q, np.zeros(2), np.full(12, -.1), np.full(12, .1),
                     .05, .5, 1.2, "test", 7)
    boxes = ObservedBoxes([], ObservationGrid(target.snapshot()))
    checker = CollisionChecker(Points(), np.zeros(2), boxes,
                                observation_now_ns=target.timestamp_ns)
    follower = PathFollower(path, avoid=True, detour_lower=path.lower, detour_upper=path.upper,
                            direct_goal_cbf=True, tolerance=1e-6)
    delta, state = follower.next_delta(q[0], checker)
    assert state == "nominal" and delta[0] > 0.
    assert follower.replans == 0 and checker.edge_safe(q[0], q[0]+delta)


def test_historical_unknown_still_stops_global_snapshot_and_missing_map_is_rejected():
    target = grid()
    update(target, hits=[[.25, .15, .15]])
    update(target)
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "status": "ok", "host_timestamp_ns": target.timestamp_ns,
                "obstacle_backend": "probabilistic_voxels", "occupancy_map": target.snapshot(),
                "unknown_obstacles": target.unknown_obstacles, "obbs": []}
    with pytest.raises(SnapshotUnknown):
        validate_snapshot(snapshot, 1.)
    snapshot.pop("unknown_obstacles")
    with pytest.raises(SnapshotUnknown):
        validate_snapshot(snapshot, 1.)
    snapshot.pop("occupancy_map")
    with pytest.raises(ValueError, match="occupancy map"):
        boxes_from_snapshot(snapshot)


@pytest.mark.parametrize("fault", ["shape", "encoding", "data", "count", "timestamp", "probability"])
def test_corrupted_map_contracts_fail_closed(fault):
    _, snapshot = observed_boxes()
    payload = snapshot["occupancy_map"]
    if fault == "shape": payload["shape"] = [2_000_000, 2, 2]
    if fault == "encoding": payload["encoding"] = "other"
    if fault == "data": payload["state_data"] = "invalid!"
    if fault == "count": payload["free_voxels"] += 1
    if fault == "timestamp": payload["capture_timestamp_ns"] += 1
    if fault == "probability": payload["log_odds_dtype"] = "float16"
    with pytest.raises(ValueError):
        boxes_from_snapshot(snapshot)
