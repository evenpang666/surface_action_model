"""Noise rejection must not erase thin obstacles or manufacture free space."""

import numpy as np
import pytest

from real_scripts.obstacle_observation import verified_completion_depth
from real_scripts.probabilistic_obstacle_map import integrate_depth_views
from real_scripts.raw_depth_evidence import RawDepthEvidenceFilter, confirm_raw_depth_views
from real_scripts.real_robot_adapter import CameraCalibration
from safety_module.occupancy import FREE, OCCUPIED, UNKNOWN, LocalOccupancyGrid, ObservationGrid


def camera(name="front", focal=500., transform=None):
    return CameraCalibration(name, np.array([[focal, 0., 10.], [0., focal, 10.], [0., 0., 1.]]),
                             np.eye(4) if transform is None else transform)


def filtered(depth, *, processor=None, stamp=1, view=None, keep=None, rendered=None):
    return (processor or RawDepthEvidenceFilter()).update(
        depth, np.ones_like(depth, bool) if keep is None else keep,
        np.zeros_like(depth) if rendered is None else rendered,
        camera() if view is None else view, timestamp_ns=stamp)


def test_isolated_near_and_far_speckles_never_gain_geometry_by_repetition():
    depth = np.ones((21, 21), np.float32)
    depth[5, 5], depth[15, 15] = .7, 1.3
    processor = RawDepthEvidenceFilter()
    for stamp in range(1, 12):
        result = filtered(depth, processor=processor, stamp=stamp)
        assert not result.accepted[5, 5] and not result.accepted[15, 15]
        assert result.uncertain[5, 5] and result.uncertain[15, 15]
        assert result.diagnostics["isolated_pixels"] == 2
        assert result.accepted.sum() == depth.size - 2


def test_depth_discontinuities_preserve_foreground_and_background_surfaces():
    depth = np.ones((21, 21), np.float32)
    depth[5:16, 5:16] = .7
    result = filtered(depth)
    assert result.accepted.all()
    assert not result.uncertain.any()
    assert depth[5, 5] == pytest.approx(.7)  # No smoothing across the edge.


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_support_pairs_are_symmetric_at_image_borders(radius):
    # A corner has (radius+1)^2-1 neighbours, all at compatible depths.
    depth = np.full((21, 21), .7, np.float32)
    minimum = (radius+1)**2 - 1
    result = filtered(depth, processor=RawDepthEvidenceFilter(radius_pixels=radius, min_neighbors=minimum))
    assert result.accepted.all()
    stricter = filtered(depth, processor=RawDepthEvidenceFilter(radius_pixels=radius, min_neighbors=minimum+1))
    for y, x in ((0, 0), (0, 20), (20, 0), (20, 20)):
        assert stricter.uncertain[y, x]
    assert stricter.accepted[10, 10]


def test_one_pixel_line_has_immediate_interior_support_and_confirmed_endpoints():
    depth = np.ones((21, 21), np.float32)
    depth[3:18, 10] = .7
    processor = RawDepthEvidenceFilter()
    first = filtered(depth, processor=processor)
    assert first.accepted[4:17, 10].all()
    assert first.uncertain[3, 10] and first.uncertain[17, 10]
    filtered(depth, processor=processor, stamp=2)
    last = filtered(depth, processor=processor, stamp=3)
    assert last.accepted[3:18, 10].all()
    assert last.diagnostics["temporal_confirmed_pixels"] == 2


def test_three_pixel_object_is_accepted_without_a_cluster_size_gate():
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10:12] = .7
    depth[11, 10] = .7
    assert filtered(depth).accepted.sum() == 3


def test_full_pixel_endpoint_lifting_preserves_a_tiny_surface_with_distortion_and_extrinsics():
    from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10:12], depth[11, 10] = .7, .7
    transform = np.eye(4)
    transform[:3, 3] = [.1, -.2, .3]
    view = CameraCalibration("front", camera().intrinsics, transform, [.1, -.05, .002, -.001, .01])
    processor = RawDepthEvidenceFilter()
    evidence = filtered(depth, processor=processor, view=view)
    actual = processor.world_points(depth, evidence.accepted, view)
    frame = RGBDFrame("front", np.zeros((21, 21, 3), np.uint8), depth, host_timestamp_ns=1)
    expected, _ = depth_to_world_points(frame, view, stride=1, keep_mask=evidence.accepted)
    assert len(actual) == 3
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("reset", ["gap", "depth_change", "missing", "calibration", "resolution"])
def test_weak_pixel_confirmation_resets_on_missing_or_changed_measurements(reset):
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10:12] = .7
    processor = RawDepthEvidenceFilter(max_gap_s=.1)
    for stamp in (1, 2):
        assert not filtered(depth, processor=processor, stamp=stamp).accepted.any()
    view, stamp = camera(), 3
    if reset == "gap":
        stamp = 200_000_003
    elif reset == "depth_change":
        depth[10, 10:12] = .8
    elif reset == "missing":
        filtered(np.zeros_like(depth), processor=processor, stamp=3)
        stamp = 4
    elif reset == "calibration":
        transform = np.eye(4)
        transform[0, 3] = .01
        view = camera(transform=transform)
    else:
        depth = depth[:20]
    assert not filtered(depth, processor=processor, stamp=stamp, view=view).accepted.any()


def test_neighbor_support_is_a_3d_distance_and_requires_unmasked_valid_returns():
    depth = np.ones((21, 21), np.float32)
    # Adjacent image pixels at this focal length are physically too far apart.
    assert not filtered(depth, view=camera(focal=10.)).accepted.any()
    depth[:] = np.nan
    depth[10, 10] = 1.
    depth[10, 11] = np.inf
    depth[11, 10] = 3.
    result = filtered(depth)
    assert result.uncertain.sum() == 1
    assert not result.accepted.any()
    keep = np.zeros((21, 21), bool)
    keep[10, 10] = True
    assert not filtered(np.ones_like(depth), keep=keep).accepted.any()


def test_surface_behind_rendered_robot_is_unknown_but_foreground_is_retained():
    rendered = np.ones((21, 21), np.float32)
    behind = filtered(np.full_like(rendered, 1.2), rendered=rendered)
    assert not behind.accepted.any()
    assert behind.uncertain.all()
    assert not behind.candidates.any()  # Another view cannot rescue an occluded return.
    assert behind.diagnostics["robot_occluded_pixels"] == rendered.size
    foreground = filtered(np.full_like(rendered, .8), rendered=rendered)
    assert foreground.accepted.all()


def test_another_supported_camera_can_confirm_a_single_pixel_without_circular_support():
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10] = .7
    a, b = camera("front"), camera("side")
    result = filtered(depth, view=a)
    other = filtered(np.full_like(depth, .7), view=b)
    confirm_raw_depth_views({"front": (a, depth, result), "side": (b, np.full_like(depth, .7), other)})
    assert result.accepted[10, 10] and not result.uncertain[10, 10]
    assert result.diagnostics["multiview_confirmed_pixels"] == 1
    result, other = filtered(depth, view=a), filtered(depth, view=b)
    confirm_raw_depth_views({"side": (b, depth, other), "front": (a, depth, result)})
    assert not result.accepted.any() and not other.accepted.any()


def test_multiview_confirmation_accounts_for_extrinsics_and_farther_surface_veto():
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10] = .7
    transform = np.eye(4)
    transform[0, 3] = .7 / 500.
    a, b = camera("front"), camera("side", transform=transform)
    surface = np.zeros_like(depth)
    surface[9:12, 8:11] = .7  # Front's centre projects to x=9 in side.
    result, other = filtered(depth, view=a), filtered(surface, view=b)
    confirm_raw_depth_views({"front": (a, depth, result), "side": (b, surface, other)})
    assert result.accepted[10, 10]
    result = filtered(depth, view=a)
    c = camera("contradicting")
    far = np.full_like(depth, .9)
    third = filtered(far, view=c)
    confirm_raw_depth_views({"front": (a, depth, result), "side": (b, surface, other), "third": (c, far, third)})
    assert not result.accepted[10, 10]


def test_rejected_measurements_cannot_support_lingbot_holes_or_replace_raw_speckles():
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10] = .7
    evidence = filtered(depth)
    completion, _ = verified_completion_depth(
        depth, np.full_like(depth, .7), ~evidence.uncertain, np.ones_like(depth, bool), camera(),
        [(camera("side"), np.full_like(depth, .7))])
    assert not completion.any()


def test_noise_only_frame_supplies_neither_obbs_nor_free_rays():
    target = LocalOccupancyGrid([-.1, -.1, 0.], np.eye(3), [10, 10, 60], .02)
    depth = np.zeros((21, 21), np.float32)
    depth[10, 10] = .7
    evidence = filtered(depth)
    integrate_depth_views(target, [(camera(), depth, evidence.accepted, np.zeros_like(depth))], [], [],
                          uncertain_hits=[[0., 0., .7]], timestamp_ns=1, ray_stride=1)
    assert not target.boxes()
    assert np.all(target.states() == UNKNOWN)
    assert target.diagnostics["raw_free_voxels"] == 0
    assert target.diagnostics["uncertain_raw_voxels"] == 1
    assert not target.unknown_obstacles  # Never confirmed as an obstacle.


def test_uncertain_endpoint_revokes_free_belief_and_vetoes_free_rays_and_completion():
    target = LocalOccupancyGrid([0., 0., 0.], np.eye(3), [4, 4, 4], .1)
    point = [[.15, .15, .15]]
    ids = target.point_ids(point)
    for stamp in range(1, 8):
        target.update(timestamp_ns=stamp, raw_hits=[], free_ids=ids)
    assert target.states()[ids[0]] == FREE
    target.update(timestamp_ns=8, raw_hits=[], free_ids=ids, completion_hits=point, uncertain_hits=point)
    assert target.states()[ids[0]] == UNKNOWN
    assert target.sources[ids[0]] == 8
    assert target.free_streak[ids[0]] == 0
    assert not target.boxes() and not target.unknown_obstacles
    observation = ObservationGrid(target.snapshot())
    assert not observation.segments_observed([[.12, .15, .15]], [[.18, .15, .15]], now_ns=8)
    # A new valid raw hit is immediately authoritative over uncertainty.
    target.update(timestamp_ns=9, raw_hits=point, free_ids=ids, uncertain_hits=point)
    assert target.states()[ids[0]] == OCCUPIED
    assert len(target.boxes()) == 1
    # A confirmed obstacle that now has insufficient evidence keeps its hazard,
    # but never its geometry. Further speckles cannot clear the historical hazard.
    for stamp in range(10, 15):
        target.update(timestamp_ns=stamp, raw_hits=[], free_ids=ids, uncertain_hits=point)
        assert target.unknown_obstacles and not target.boxes()
    for stamp in range(15, 18):
        target.update(timestamp_ns=stamp, raw_hits=[], free_ids=ids)
    assert target.states()[ids[0]] == FREE and not target.unknown_obstacles


@pytest.mark.parametrize("kwargs", [{"neighbor_distance_m": 0.}, {"max_gap_s": np.nan},
                                    {"confirm_frames": 1}, {"radius_pixels": 0}, {"min_neighbors": 1}])
def test_invalid_filter_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RawDepthEvidenceFilter(**kwargs)
