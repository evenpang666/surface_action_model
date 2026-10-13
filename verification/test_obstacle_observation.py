"""Regression scenarios for missing geometry, table priors and depth evidence."""

import io
import json
import time

import numpy as np
import pytest

from real_scripts.live_dual_ur7e_obstacle_model import (
    TemporalOBBGate, merge_overlapping_obbs, obstacle_cluster_boxes,
    observed_free_space_fraction, snapshot_payload, tabletop_obb,
)
from real_scripts.obstacle_observation import TablePlanePrior, verified_completion_depth
from real_scripts.real_robot_adapter import CameraCalibration
from safety_module.runtime_contract import SnapshotUnknown, validate_snapshot


def cloud(z=.06):
    return np.array([[x, y, z + dz] for x in np.linspace(-.03, .03, 8)
                     for y in np.linspace(-.02, .02, 8) for dz in (0., .02)])


def camera(name="test", transform=None):
    return CameraCalibration(name, np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
                             np.eye(4) if transform is None else transform)


def test_unknown_has_no_box_and_resolves_on_reobservation_after_history_expires():
    points = cloud()
    original = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    gate.update([original], [points])
    for _ in range(25):
        assert gate.update([], []) == []
    unknown = gate.unknown_obstacles
    assert len(unknown) == 1
    assert not {"center", "axes", "corners", "half_sizes", "extents"} & unknown[0].keys()
    assert gate._tracks[0].box is None
    recovered = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    assert gate.update([recovered], [points]) == [recovered]
    assert not gate.unknown_obstacles


def test_nearby_different_object_cannot_resolve_unknown_by_center():
    points = cloud()
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    gate.update([tabletop_obb(points, np.array([0., 0., 1.]), 0.)], [points])
    gate.update([], [])
    other = points + [0., 0., .025]
    gate.update([tabletop_obb(other, np.array([0., 0., 1.]), 0.)], [other])
    assert len(gate.unknown_obstacles) == 1


@pytest.mark.parametrize("status", ["ok", "error: unstable obstacle point clusters (missing_cluster)"])
def test_unknown_survives_compact_payload_and_blocks_replay_bypass(monkeypatch, status):
    import obstacle_validation.runtime as runtime
    value = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
             "status": status, "host_timestamp_ns": time.monotonic_ns(),
             "scene_observation_state": "unknown", "unknown_obstacles": [{"track_id": 1, "state": "unknown"}],
             "obbs": [{"center": [8.] * 3, "axes": np.eye(3).tolist(), "half_sizes": [.1] * 3}],
             "qpos": {"left": [0.] * 6, "right": [0.] * 6},
             "gripper_position": {"left": 0., "right": 0.},
             "diagnostics": {"obb_unstable_reasons": ["missing_cluster"], "published_obbs": 1,
                             "safety_obbs_cover_candidates": True}}
    payload = snapshot_payload(value, compact=True).decode()
    assert json.loads(payload)["unknown_obstacles"] == value["unknown_obstacles"]
    with pytest.raises(SnapshotUnknown):
        validate_snapshot(json.loads(payload), 1.)
    monkeypatch.setattr(runtime, "urlopen", lambda *_args, **_kwargs: io.StringIO(payload))
    with pytest.raises(SnapshotUnknown):
        runtime.read_snapshot("http://unused", 1., allow_tracked_obbs=True)


def test_occupied_view_vetoes_free_view_for_unknown_clearing():
    points = cloud(1.)
    views = [(camera("free"), np.full((100, 100), 1.5)),
             (camera("occupied"), np.full((100, 100), 1.01))]
    assert observed_free_space_fraction(points, views) == 0.
    views[1] = (camera("occluded"), np.full((100, 100), .5))
    assert observed_free_space_fraction(points, views) == 1.


def test_cluster_backend_uncertain_returns_veto_free_without_retaining_geometry():
    points = cloud(1.)
    view = camera()
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    gate.update([tabletop_obb(points, np.array([0., 0., 1.]), -1.)], [points])
    free = [(view, np.full((100, 100), 1.5))]
    uncertain = [(view, np.full((100, 100), 1.01))]
    for _ in range(5):
        assert gate.update([], [], free_space_views=free, uncertain_space_views=uncertain) == []
        assert gate.unknown_obstacles and gate._tracks[0].box is None
    for _ in range(3):
        gate.update([], [], free_space_views=free)
    assert not gate.unknown_obstacles


@pytest.mark.parametrize("flags,expected", [([], True), (["--occupancy-map"], True),
                                           (["--no-occupancy-map"], False)])
def test_occupancy_map_cli_toggle(monkeypatch, flags, expected):
    from real_scripts.live_dual_ur7e_obstacle_model import parse_args
    monkeypatch.setattr("sys.argv", ["detector", *flags])
    assert parse_args().occupancy_map is expected


def test_disabled_map_does_not_validate_unused_voxel_settings(monkeypatch):
    from real_scripts.live_dual_ur7e_obstacle_model import parse_args
    monkeypatch.setattr("sys.argv", ["detector", "--no-occupancy-map", "--occupancy-resolution-m", "0",
                                     "--occupancy-box-tile-voxels", "0", "--occupancy-ray-stride", "0"])
    assert parse_args().occupancy_map is False


def test_partial_free_evidence_cannot_clear_remaining_unknown_points():
    points = cloud(1.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    gate.update([tabletop_obb(points, np.array([0., 0., 1.]), -1.)], [points])
    depth = np.full((100, 100), 1.5)
    depth[:, 49:] = .5
    for _ in range(15):
        gate.update([], [], free_space_views=[(camera(), depth)])
    assert len(gate.unknown_obstacles) == 1


def test_floating_and_tall_clusters_are_detected_without_filling_to_table():
    floating = cloud(.55)
    attached = cloud(.025) + [.3, 0., 0.]
    # Keep both sampled surface layers connected at the production 10 mm radius.
    floating[1::2, 2] = .555
    attached[1::2, 2] = .030
    boxes, clusters, diagnostics = obstacle_cluster_boxes(
        np.vstack((floating, attached)), np.array([0., 0., 1.]), 0.,
        min_height_m=.02, max_height_m=None, outlier_neighbour_m=.02,
        eps_m=.01, min_samples=6, min_cluster_points=80, attachment_distance_m=.05)
    assert len(boxes) == len(clusters) == 2
    assert diagnostics["floating_components"] == diagnostics["table_connected_components"] == 1
    floating_box = next(box for box in boxes if box.center[2] > .5)
    assert floating_box.corners[:, 2].min() == pytest.approx(.55)
    assert next(box for box in boxes if box.center[2] < .5).corners[:, 2].min() == pytest.approx(0.)
    second = tabletop_obb(floating + [.005, 0., 0.], np.array([0., 0., 1.]), 0., extend_to_table=False)
    merged = merge_overlapping_obbs([floating_box, second], np.array([0., 0., 1.]), 0.)
    assert len(merged) == 1
    assert merged[0].corners[:, 2].min() == pytest.approx(.55)


def table_points(z=0., count=20):
    return np.array([[x, y, z] for x in np.linspace(-.3, .3, count)
                     for y in np.linspace(-.2, .2, count)])


def test_table_prior_ignores_new_dominant_object_top_and_survives_occlusion():
    prior = TablePlanePrior()
    prior.update(table_points(), np.array([0., 0., 1.]))
    normal, offset = prior.update(np.vstack((table_points(count=10), table_points(.15, 30))),
                                 np.array([0., 0., 1.]))
    np.testing.assert_allclose(normal, [0., 0., 1.], atol=1e-10)
    assert offset == pytest.approx(0.)
    assert prior.diagnostics["state"] == "retained_prior_low_support"
    normal, offset = prior.update(np.empty((0, 3)), np.array([0., 0., 1.]))
    assert offset == pytest.approx(0.)


def test_table_prior_updates_are_bounded_by_startup_anchor():
    prior = TablePlanePrior(max_offset_m=.005)
    prior.update(table_points(), np.array([0., 0., 1.]))
    for z in np.linspace(.001, .025, 50):
        _, offset = prior.update(table_points(z), np.array([0., 0., 1.]))
        assert abs(offset) <= .005
    assert prior.diagnostics["state"].startswith("retained_prior")


def test_explicit_table_prior_is_normalized_and_bad_startup_support_is_rejected():
    prior = TablePlanePrior(plane=[0., 0., 2., -1.2])
    normal, offset = prior.update(np.empty((0, 3)), np.array([0., 0., 1.]))
    np.testing.assert_allclose(normal, [0., 0., 1.])
    assert offset == pytest.approx(-.6)
    with pytest.raises(RuntimeError, match="initialize tabletop"):
        TablePlanePrior().update(np.empty((0, 3)), np.array([0., 0., 1.]))


def completion_inputs():
    raw = np.ones((100, 100), np.float32)
    raw[50, 50] = 0.
    refined = np.ones_like(raw)
    keep = np.ones(raw.shape, bool)
    return raw, refined, keep


def test_completion_requires_measured_support_from_other_view_and_does_not_replace_raw():
    raw, refined, keep = completion_inputs()
    refined[20, 20] = 1.5
    result, evidence = verified_completion_depth(raw, refined, keep, keep, camera(),
                                                [(camera("other"), np.ones_like(raw))])
    assert result[50, 50] == 1.
    assert np.count_nonzero(result) == 1
    assert result[20, 20] == 0.
    assert evidence["refined_measured_disagreement_pixels"] == 1
    assert raw[20, 20] == 1.
    result, evidence = verified_completion_depth(raw, refined, keep, keep, camera(), [])
    assert not result.any()
    assert evidence["rejected_completion_pixels"] == 1


@pytest.mark.parametrize("other_depth", [0., .5, 1.5])
def test_completion_rejects_invalid_occluded_and_conflicting_other_view(other_depth):
    raw, refined, keep = completion_inputs()
    result, _ = verified_completion_depth(raw, refined, keep, keep, camera(),
                                          [(camera("other"), np.full_like(raw, other_depth))])
    assert not result.any()


def test_completion_rejects_discontinuities_large_holes_and_robot_pixels():
    raw, refined, keep = completion_inputs()
    raw[48:53, 48:50] = .5
    result, _ = verified_completion_depth(raw, refined, keep, keep, camera(),
                                          [(camera("other"), np.ones_like(raw))])
    assert not result.any()
    raw, refined, keep = completion_inputs()
    raw[40:60, 40:60] = 0.
    result, _ = verified_completion_depth(raw, refined, keep, keep, camera(),
                                          [(camera("other"), np.ones_like(raw))])
    assert result[50, 50] == 0.
    raw, refined, keep = completion_inputs()
    keep[50, 50] = False
    result, _ = verified_completion_depth(raw, refined, keep, keep, camera(),
                                          [(camera("other"), np.ones_like(raw))])
    assert not result.any()


def test_completion_projects_between_different_camera_poses():
    raw, refined, keep = completion_inputs()
    transform = np.eye(4)
    transform[0, 3] = .1
    other_depth = np.zeros_like(raw)
    other_depth[50, 40] = 1.
    result, _ = verified_completion_depth(raw, refined, keep, keep, camera(),
                                          [(camera("other", transform), other_depth)])
    assert result[50, 50] == 1.
