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
from safety_module.runtime_contract import validate_snapshot


def grid(shape=(8, 4, 4), **kwargs):
    return LocalOccupancyGrid(np.zeros(3), np.eye(3), shape, .1, **kwargs)


def update(target, hits=(), free=(), completed=(), stamp=None):
    target.update(timestamp_ns=target.timestamp_ns + 1_000_000 if stamp is None else stamp,
                  raw_hits=np.asarray(hits).reshape(-1, 3),
                  completion_hits=np.asarray(completed).reshape(-1, 3), free_ids=np.asarray(free, dtype=np.int64))


def test_endpoint_occupied_and_all_other_cells_free():
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
    assert np.all(target.states()[9:] == FREE)
    assert target.sources[0] == 2 and target.sources[8] == 1
    assert 1 / (1 + np.exp(-target.log_odds[8])) > .7


@pytest.mark.parametrize("depth,keep", [(0., True), (np.nan, True), (1., False)])
def test_invalid_and_robot_masked_depth_do_not_supply_evidence(depth, keep):
    target = grid((1, 1, 12))
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    target.origin[:2] = -.05
    integrate_depth_views(target, [(camera, np.array([[depth]]), np.array([[keep]]), np.zeros((1, 1)))],
                          [], [], timestamp_ns=1, ray_stride=1)
    assert not target.last_observed_ns.any()
    assert np.all(target.states() == FREE)


def test_nearer_robot_render_truncates_inconsistent_background_ray():
    target = grid((1, 1, 12))
    target.origin[:2] = -.05
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    for _ in range(3):
        integrate_depth_views(target, [(camera, np.array([[1.1]]), np.ones((1, 1), bool), np.array([[.45]]))],
                              [], [], timestamp_ns=target.timestamp_ns+1, ray_stride=1)
    assert np.all(target.states()[:4] == FREE)
    assert np.all(target.states()[4:] == FREE)


def test_multiview_conflict_is_order_independent_and_voxel_updates_once():
    a, b = grid(), grid()
    point = np.array([[.25, .15, .15]])
    ids = a.point_ids(point)
    update(a, hits=np.repeat(point, 100, axis=0), free=np.repeat(ids, 500))
    update(b, hits=point, free=ids[::-1])
    np.testing.assert_array_equal(a.log_odds, b.log_odds)
    assert a.states()[ids[0]] == OCCUPIED
    assert a.diagnostics["conflicting_voxels"] == 1
    assert a.diagnostics["raw_free_voxels"] == 0


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


def test_lost_obstacle_is_free_without_historical_blockage():
    target = grid()
    point = [[.25, .15, .15]]
    update(target, hits=point)
    assert len(target.boxes()) == 1
    for _ in range(30):
        update(target)
        assert not target.boxes()
        assert np.all(target.states() == FREE)
        assert target.snapshot()["historical_unknown_voxels"] == 0
    update(target, hits=point)
    assert target.states()[target.point_ids(point)[0]] == OCCUPIED


def test_missing_hit_clears_occupancy_without_free_confirmation():
    target = grid(clear_frames=100)
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    update(target, hits=point)
    update(target, free=ids)
    assert target.states()[ids[0]] == FREE
    assert target.log_odds[ids[0]] > target.free_log
    assert not target.boxes()


def test_completion_supplies_current_occupancy_without_free_rays():
    target = grid()
    point = [[.25, .15, .15]]
    update(target, completed=point)
    assert target.sources[target.point_ids(point)[0]] == 4
    assert target.states()[target.point_ids(point)[0]] == OCCUPIED
    assert target.diagnostics["raw_free_voxels"] == 0
    update(target)
    assert np.all(target.states() == FREE)


def test_free_space_remains_traversable_after_evidence_expires():
    target = grid(free_ttl_s=.2)
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    for _ in range(3):
        update(target, free=ids)
    captured = target.timestamp_ns
    view = ObservationGrid(target.snapshot())
    assert view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=captured)
    assert view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=captured+201_000_000)
    assert target.states(now_ns=captured+201_000_000)[ids[0]] == FREE


def test_map_snapshot_round_trip_keeps_unobserved_cells_and_probability_history():
    target = grid()
    update(target, hits=[[.25, .15, .15]])
    payload = json.loads(json.dumps(target.snapshot()))
    view = ObservationGrid(payload)
    np.testing.assert_array_equal(view.state, target.states())
    assert payload["occupied_voxels"] == 1
    assert payload["unknown_voxels"] == 0
    assert payload["free_voxels"] == target.size - 1
    assert payload["capture_timestamp_ns"] == target.last_observed_ns.max()
    assert target.log_odds.sum() > 0


def test_probability_decay_does_not_block_free_cells():
    target = grid(free_ttl_s=10.)
    point = [[.25, .15, .15]]
    ids = target.point_ids(point)
    for _ in range(3):
        update(target, free=ids)
    view = ObservationGrid(target.snapshot())
    later = target.timestamp_ns+1_100_000_000
    assert target.states(now_ns=later)[ids[0]] == FREE
    assert view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=later)


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


@pytest.mark.parametrize("tile_size", [1, 2, 4, 7])
def test_tile_packing_exactly_covers_changed_occupancy_with_rotation(tile_size):
    # Irregular occupancy, partial edge tiles and changed patterns must preserve
    # every occupied cell, including when cached patterns are reused.
    target = grid((9, 6, 5))
    target.axes = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rng = np.random.default_rng(80)
    for _ in range(3):
        ids = np.flatnonzero(rng.random(target.size) < .4)
        update(target, hits=target.centers(ids))
        expected = np.zeros(target.size, np.int64)
        expected[ids] = 1
        for _ in range(2):
            actual = np.zeros(target.size, np.int64)
            for box in target.boxes(tile_voxels=tile_size):
                inside = np.all(np.abs((target.centers(np.arange(target.size)) - box.center) @ box.rotation)
                                < box.extents / 2 - 1e-10, axis=1)
                assert inside.sum() == box.point_count
                actual += inside
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("exclude_endpoint", [False, True])
def test_batched_ray_compaction_keeps_original_ray_identity(exclude_endpoint):
    target = grid((5, 2, 1))
    start = [[-.2, .05, .05], [.45, .15, .05], [.15, .05, .05], [-.1, .3, .05]]
    end = [[.25, .05, .05], [.15, .15, .05], [.15, .05, .05], [.4, .3, .05]]
    ids, rays = target.ray_cells(start, end, exclude_endpoint=exclude_endpoint)
    expected = ([0, 2], [5, 7, 9], [], []) if exclude_endpoint else ([0, 2, 4], [3, 5, 7, 9], [2], [])
    for ray, cells in enumerate(expected):
        assert np.unique(ids[rays == ray]).tolist() == cells
    unique, _ = target.ray_cells(start, end, exclude_endpoint=exclude_endpoint, unique_only=True)
    np.testing.assert_array_equal(unique, np.unique(ids))


@pytest.mark.parametrize('unique_only', [False, True])
def test_stationary_rays_check_occupied_endpoints_but_never_supply_free_evidence(unique_only):
    target = grid((5, 1, 1))
    points = np.array([[.15, .05, .05], [2., .05, .05]])
    ids, rays = target.ray_cells(points, points, unique_only=unique_only)
    np.testing.assert_array_equal(ids, [1])
    if not unique_only:
        np.testing.assert_array_equal(rays, [0])
    free, _ = target.ray_cells(points, points, unique_only=unique_only, exclude_endpoint=True)
    assert not len(free)
    update(target, hits=[points[0]])
    assert not ObservationGrid(target.snapshot()).segments_observed(points, points,
                                                                   now_ns=target.timestamp_ns)


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


def observed_boxes(*, occupied_cell=None):
    target = grid()
    free = np.arange(target.size)
    hits = () if occupied_cell is None else target.centers([np.asarray(occupied_cell) @ target.strides])
    stamp = time.monotonic_ns()-3_000_000
    for index in range(3):
        update(target, hits=hits, free=free, stamp=stamp+index*1_000_000)
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "status": "ok", "obstacle_backend": "probabilistic_voxels",
                "host_timestamp_ns": target.timestamp_ns, "occupancy_map": target.snapshot(), "obbs": []}
    return boxes_from_snapshot(snapshot), snapshot


def test_fk_rrt_and_replay_fallbacks_reject_occupied_interior_with_free_endpoints():
    from obstacle_validation.core import CollisionChecker
    from obstacle_validation.lerobot_replay import _motion_safe
    boxes, snapshot = observed_boxes(occupied_cell=[3, 1, 1])
    assert validate_snapshot(snapshot, 1.) >= 0
    q, grip, delta = np.zeros(12), np.zeros(2), np.zeros(12)
    delta[0] = .4
    checker = CollisionChecker(Sampler(), grip, boxes, margin=0., inter_arm_margin=.03)
    assert checker.state_safe(q) and checker.state_safe(q+delta)
    assert not checker.edge_safe(q, q+delta)
    assert checker.observation_rejections["occupied"] > 0
    assert checker.last_observation_blockage["reason"] == "occupied"
    assert not validate_fk_rollout(Sampler(), q, grip, [delta], boxes, point_indices=np.array([0, 1]),
                                  left_mask=np.array([True, False]), collision_margin_m=0., inter_arm_margin_m=.03)
    assert not _motion_safe(Sampler(), q, grip, delta, grip, boxes,
                            {"collision_margin_m": 0., "inter_arm_margin_m": .03}, True)
    safe_boxes, _ = observed_boxes()
    assert _motion_safe(Sampler(), q, grip, delta, grip, safe_boxes,
                        {"collision_margin_m": 0., "inter_arm_margin_m": .03})


def test_gripper_only_motion_cannot_bypass_occupancy_and_unobserved_motion_is_allowed():
    boxes, _ = observed_boxes(occupied_cell=[3, 1, 1])
    assert not validate_fk_rollout(Sampler(), np.zeros(12), np.zeros(2), [np.zeros(12)], boxes,
                                  point_indices=np.array([0, 1]), left_mask=np.array([True, False]),
                                  collision_margin_m=0., inter_arm_margin_m=.03, gripper_chunk=[[.5, 0.]])
    target = grid()
    update(target)
    view = ObservationGrid(target.snapshot())
    assert view.segments_observed([[.15, .15, .15]], [[.151, .15, .15]], now_ns=target.timestamp_ns)
    assert view.segments_observed([[.15, .15, .15]], [[.15, .15, .15]], now_ns=target.timestamp_ns)


def test_motion_margin_checks_occupied_at_the_local_map_boundary():
    target = grid()
    update(target, hits=[[.75, .15, .15]])
    view = ObservationGrid(target.snapshot())
    assert view.segments_observed([[.81, .15, .15]], [[.815, .15, .15]], now_ns=target.timestamp_ns)
    assert not view.segments_observed([[.81, .15, .15]], [[.815, .15, .15]],
                                       margin_m=.01, now_ns=target.timestamp_ns)


@pytest.mark.parametrize("state", [OCCUPIED, FREE])
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
    assert safe([.049, .03, .03], [.049, .031, .03]) == (state == FREE)
    if state == OCCUPIED:
        assert view.last_blockage["reason"] == "occupied"
    assert safe([.047, .047, .03], [.047, .048, .03]) == (state == FREE)
    # Endpoints are safe but the segment passes within 8 mm of the cell.
    assert safe([.01, .048, .03], [.05, .048, .03]) == (state == FREE)


def test_restoration_can_advance_past_an_occupied_neighbour_outside_margin():
    from obstacle_validation.core import CollisionChecker, JointPath, PathFollower
    from safety_module.occupancy import ObservedBoxes

    active = np.zeros(64, bool); active[21] = True
    target = LocalOccupancyGrid(np.zeros(3), np.eye(3), [4, 4, 4], .02, active=active)
    update(target, hits=target.centers([21]))

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


def test_lost_voxels_do_not_stop_snapshot_and_missing_map_is_rejected():
    target = grid()
    update(target, hits=[[.25, .15, .15]], stamp=time.monotonic_ns()-1_000_000)
    update(target)
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "status": "ok", "host_timestamp_ns": target.timestamp_ns,
                "obstacle_backend": "probabilistic_voxels", "occupancy_map": target.snapshot(),
                "unknown_obstacles": [], "obbs": []}
    assert validate_snapshot(snapshot, 1.) >= 0
    assert snapshot["occupancy_map"]["historical_unknown_voxels"] == 0
    snapshot.pop("occupancy_map")
    with pytest.raises(ValueError, match="occupancy map"):
        boxes_from_snapshot(snapshot)


def test_legacy_unknown_cells_are_read_as_free():
    from safety_module.occupancy import _pack
    target = grid()
    update(target, hits=[[.25, .15, .15]])
    payload = target.snapshot()
    legacy_state = target.states().copy()
    legacy_state[legacy_state == FREE] = UNKNOWN
    payload.update(state_data=_pack(legacy_state), unknown_voxels=target.size-1, free_voxels=0)
    view = ObservationGrid(payload)
    assert not np.any(view.state == UNKNOWN)
    assert view.segments_observed([[.1, .1, .1]], [[.15, .1, .1]], now_ns=target.timestamp_ns)
    assert not view.segments_observed([[.21, .15, .15]], [[.29, .15, .15]], now_ns=target.timestamp_ns)


@pytest.mark.parametrize("start,end,allowed", [
    ([.196, .05, .05], [.17, .05, .05], True),  # exit a deficient 5 mm margin
    ([.196, .05, .05], [.195, .05, .05], True),  # partial recovery
    ([.196, .04, .05], [.196, .06, .05], True),  # nonworsening tangent
    ([.196, .05, .05], [.197, .05, .05], False),  # move nearer
    ([.19, .05, .05], [.196, .05, .05], False),  # enter margin from safe start
    ([.21, .05, .05], [.17, .05, .05], False),  # start inside actual occupied cell
    ([.20, .05, .05], [.17, .05, .05], False),  # start on actual cell boundary
    ([.196, .05, .05], [.304, .05, .05], False),  # cross obstacle to far side
])
def test_voxel_margin_recovery_only_allows_nonworsening_clearance(start, end, allowed):
    target = grid()
    update(target, hits=[[.25, .05, .05]])
    view = ObservationGrid(target.snapshot())
    assert not view.segments_observed([start], [end], margin_m=.005,
                                      now_ns=target.timestamp_ns)
    assert view.segments_observed([start], [end], margin_m=.005,
                                 now_ns=target.timestamp_ns,
                                 allow_margin_recovery=True) == allowed


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
