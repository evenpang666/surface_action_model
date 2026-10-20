import json

import cv2
import numpy as np
import pytest

from real_scripts.manual_obstacle_selection import (
    ManualFrame, ManualSelectionStore, colorize_depth, depth_display_range, rgb_manual_frame,
    obb_mask, select_voxels,
)
from real_scripts.real_robot_adapter import CameraCalibration, NativeDepthFrame, RGBDFrame
from safety_module.occupancy import LocalOccupancyGrid


def box(cx, cy, width=1, height=1, angle=0, object_id="1"):
    return dict(cx=cx, cy=cy, width=width, height=height, angle_deg=angle, object_id=object_id)


def fixture():
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    depth = np.ones((3, 3))
    accepted = np.ones((3, 3), bool)
    accepted[1, 1] = False
    frame = ManualFrame(np.zeros((3, 3, 3), np.uint8), depth, accepted, camera, 100)
    grid = LocalOccupancyGrid([-.5, -.5, .5], np.eye(3), [4, 4, 2], 1.)
    grid.update(timestamp_ns=100, raw_hits=[[0, 0, 1], [1, 0, 1], [2, 0, 1], [1, 1, 1]],
                free_ids=np.empty(0, np.int64))
    return {"front": frame, "side": frame}, grid


def test_rotated_obb_pixel_membership_and_invalid_coordinates():
    mask = obb_mask((7, 7), box(3, 3, 4, 1, 90))
    assert mask[1:6, 3].all()
    assert not mask[3, 1] and not mask[3, 5]
    for invalid in (box(-1, 0), box(0, 0, angle=float("nan")), box(0, 0, width=0)):
        with pytest.raises(ValueError):
            obb_mask((7, 7), invalid)


def test_occupied_endpoints_must_also_satisfy_both_view_annotations():
    frames, grid = fixture()
    annotations = {"front": [box(1, 0, width=3), box(1, 1), box(2, 2)], "side": [box(2, 0)]}
    result = select_voxels(frames, grid.snapshot(), annotations)
    expected = grid.point_ids(np.array([[2, 0, 1]]))
    assert result["voxel_ids"] == expected.tolist()
    assert result["accepted_points"]["front"]["1"] == 3
    assert result["accepted_points"]["side"]["1"] == 1
    assert result["objects"][0]["voxel_count"] == 1
    assert result["fusion_rule"] == "dual_view_intersection"


def test_backprojection_uses_camera_extrinsic():
    frames, grid = fixture()
    transform = np.eye(4)
    transform[0, 3] = 2
    camera = CameraCalibration("side", np.eye(3), transform)
    frames["side"] = ManualFrame(frames["side"].rgb, frames["side"].depth_m,
                                 frames["side"].accepted, camera, 100)
    result = select_voxels(frames, grid.snapshot(), {"front": [box(2, 0)], "side": [box(0, 0)]})
    assert result["voxel_ids"] == grid.point_ids(np.array([[2, 0, 1]])).tolist()


def test_rgb_obb_uses_aligned_pixels_when_native_depth_has_another_size():
    _, grid = fixture()
    camera = CameraCalibration("front", np.eye(3), np.eye(4))
    native = NativeDepthFrame(np.ones((2, 2), np.uint16), .001, np.eye(3), np.eye(4))
    rgb = np.zeros((3, 3, 3), np.uint8)
    aligned = np.ones((3, 3), np.float32)
    frame = rgb_manual_frame(RGBDFrame("front", rgb, aligned, host_timestamp_ns=100,
                                      native_depth=native), camera)
    result = select_voxels({"front": frame, "side": frame}, grid.snapshot(),
                           {name: [box(2, 0)] for name in ("front", "side")})
    assert result["annotation_image"] == "recorded_rgb"
    assert result["voxel_ids"] == grid.point_ids(np.array([[2, 0, 1]])).tolist()


def test_3d_envelope_mode_can_include_other_occupied_cells():
    frames, grid = fixture()
    for frame in frames.values():
        frame.accepted[0, 1] = False
    annotations = {name: [box(1, 0, width=3)] for name in frames}
    exact = select_voxels(frames, grid.snapshot(), annotations, "depth")
    envelope = select_voxels(frames, grid.snapshot(), annotations, "obb")
    middle = grid.point_ids(np.array([[1, 0, 1]]))[0]
    assert middle not in exact["voxel_ids"] and middle in envelope["voxel_ids"]
    assert set(exact["voxel_ids"]) <= set(envelope["voxel_ids"])
    restricted = select_voxels(frames, grid.snapshot(), {
        "front": [box(0, 0), box(2, 0)], "side": [box(1, 0, width=3)]}, "obb")
    assert middle not in restricted["voxel_ids"]


def test_capture_persists_rgb_depth_map_and_rejects_wrong_frame(tmp_path):
    frames, grid = fixture()
    snapshot = {"frame": 4, "host_timestamp_ns": 100, "occupancy_map": grid.snapshot(),
                "_manual_frames": frames}
    scene = {"frame": 4, "obbs": [], "status": "ok"}
    store = ManualSelectionStore(tmp_path)
    capture = store.capture(snapshot, scene)
    directory = tmp_path / capture["capture_id"]
    assert (directory / "front_rgb.png").exists() and (directory / "side_rgb.png").exists()
    assert capture["annotation_image"] == "recorded_rgb"
    for name in frames:
        assert capture["depth_images"][name].startswith("data:image/png;base64,")
        preview = cv2.imread(str(directory / f"{name}_depth_preview.png"))
        assert preview.shape == frames[name].rgb.shape
        np.testing.assert_array_equal(preview[1, 1], preview[0, 0])
    depth = np.load(directory / "depth_evidence.npz")
    assert not depth["front_accepted"][1, 1]
    np.testing.assert_array_equal(depth["front_depth_m"], frames["front"].depth_m)
    with pytest.raises(ValueError, match="Capture changed"):
        store.select("old-capture", {"front": [box(0, 0)]}, "depth")
    result = store.select(capture["capture_id"], {n: [box(0, 0)] for n in frames}, "depth")
    assert result["voxel_count"] == 1
    assert json.loads((directory / "annotations.json").read_text())["capture_id"] == capture["capture_id"]
    assert json.loads((directory / "annotations.json").read_text())["annotation_image"] == "recorded_rgb"
    assert np.load(directory / "selected_voxels.npz")["voxel_ids"].size == 1
    assert store.scene_payload()["manual_selection"] == result
    assert snapshot["occupancy_map"] == grid.snapshot()
    new_capture = store.capture(snapshot, scene)
    assert new_capture["capture_id"] != capture["capture_id"]
    assert store.scene_payload()["manual_selection"] is None


def test_empty_selection_requires_boxes_but_allows_missing_depth():
    frames, grid = fixture()
    with pytest.raises(ValueError, match="at least one"):
        select_voxels(frames, grid.snapshot(), {})
    result = select_voxels(frames, grid.snapshot(), {n: [box(1, 1)] for n in frames})
    assert result["voxel_count"] == 0 and result["objects"][0]["obb"] is None


@pytest.mark.parametrize("single_view", [False, True])
def test_viewer_filters_current_occupancy_without_changing_frozen_or_safety_data(tmp_path, single_view):
    frames, grid = fixture()
    if single_view:
        frames = {"front": frames["front"]}
    coordinate_frame = "ur_base" if single_view else "left_base"
    store = ManualSelectionStore(tmp_path, camera_names=tuple(frames), coordinate_frame=coordinate_frame)
    snapshot = {"frame": 4, "host_timestamp_ns": 100, "occupancy_map": grid.snapshot(),
                "coordinate_frame": coordinate_frame, "_manual_frames": frames, "obbs": [{"original": True}]}
    assert store.viewer_filter(snapshot)["state"] == "missing_annotations"
    capture = store.capture(snapshot, {"frame": 4})
    annotations = {name: [box(.5, 0, width=2)] for name in frames}
    frozen = store.select(capture["capture_id"], annotations, "depth")
    assert frozen["voxel_count"] == 2
    # Mutating the request after applying it must not change the applied boxes.
    annotations["front"][0]["cx"] = 2.
    grid.update(timestamp_ns=200, raw_hits=[[1, 0, 1], [2, 0, 1]], free_ids=np.empty(0, np.int64))
    current = {**snapshot, "frame": 5, "host_timestamp_ns": 200, "occupancy_map": grid.snapshot()}
    before = json.dumps(current["occupancy_map"])
    result = store.viewer_filter(current)
    assert result["state"] == "applied" and result["frame"] == 5 and result["annotation_frame"] == 4
    assert result["voxel_count"] == 1
    np.testing.assert_allclose(result["obbs"][0]["center"], [1., 0., 1.])
    assert current["obbs"] == [{"original": True}]
    assert json.dumps(current["occupancy_map"]) == before
    assert store.scene_payload()["manual_selection"] == frozen
    assert "viewer_filter" not in current
    assert store.viewer_filter({**current, "coordinate_frame": "foreign"})["state"] == "scene_changed"
    assert store.viewer_filter({**current, "calibration": {"changed": True}})["state"] == "scene_changed"
    assert store.viewer_filter({k: v for k, v in current.items() if k != "occupancy_map"})["state"] == "map_unavailable"
    grid.update(timestamp_ns=300, raw_hits=[], free_ids=np.empty(0, np.int64))
    empty = store.viewer_filter({**current, "frame": 6, "occupancy_map": grid.snapshot()})
    assert empty["state"] == "applied" and empty["voxel_count"] == 0 and not empty["obbs"]
    store.capture(snapshot, {"frame": 4})
    assert store.viewer_filter(current)["state"] == "missing_annotations"


def test_viewer_filter_requires_matching_object_ids_in_both_views(tmp_path):
    frames, grid = fixture()
    snapshot = {"frame": 4, "host_timestamp_ns": 100, "occupancy_map": grid.snapshot(),
                "coordinate_frame": "left_base", "_manual_frames": frames}
    store = ManualSelectionStore(tmp_path)
    capture = store.capture(snapshot, {"frame": 4})
    annotations = {"front": [box(0, 0, object_id="a"), box(2, 0, object_id="b")],
                   "side": [box(2, 0, object_id="a"), box(0, 0, object_id="b")]}
    store.select(capture["capture_id"], annotations, "depth")
    result = store.viewer_filter(snapshot)
    assert result["state"] == "applied" and result["voxel_count"] == 0


def test_viewer_does_not_add_new_cells_inside_applied_rgb_boxes(tmp_path):
    frames, grid = fixture()
    # Both RGB boxes cover the currently missing (2, 0, 1) cell too.
    grid.update(timestamp_ns=200, raw_hits=[[0, 0, 1], [1, 0, 1]], free_ids=np.empty(0, np.int64))
    snapshot = {"frame": 1, "host_timestamp_ns": 200, "occupancy_map": grid.snapshot(),
                "coordinate_frame": "left_base", "_manual_frames": frames}
    store = ManualSelectionStore(tmp_path)
    capture = store.capture(snapshot, {"frame": 1})
    result = store.select(capture["capture_id"], {name: [box(1, 0, width=3)] for name in frames}, "depth")
    assert result["voxel_count"] == 2
    grid.update(timestamp_ns=300, raw_hits=[[2, 0, 1]], free_ids=np.empty(0, np.int64))
    filtered = store.viewer_filter({**snapshot, "frame": 2, "occupancy_map": grid.snapshot()})
    assert filtered["state"] == "applied"
    assert filtered["voxel_count"] == 0


def test_internal_annotation_arrays_never_enter_public_snapshots():
    from real_scripts.live_dual_ur7e_obstacle_model import snapshot_payload
    snapshot = {"frame": 1, "obbs": [], "_manual_frames": {"front": object()}}
    for options in ({}, {"compact": True}, {"viewer": True}):
        assert json.loads(snapshot_payload(snapshot, **options)) == {"frame": 1, "obbs": []}


def test_render_only_wire_format_preserves_safety_payload_and_source_arrays():
    from real_scripts.live_dual_ur7e_obstacle_model import snapshot_payload
    frames, grid = fixture()
    snapshot = {"frame": 1, "status": "ok", "occupancy_map": grid.snapshot(), "calibration": {"sha256": "example"},
                "environment": {"points": [[.123456789, 0., 1.]], "colors": [[10, 20, 30]]},
                "robot": {"points": [], "colors": []},
                "obbs": [{"center": [1, 2, 3], "corners": [[.123456789, 0, 1]]}],
                "viewer_filter": {"state": "applied", "obbs": [{"corners": [[.123456789, 0, 1]]}]},
                "_manual_frames": frames}
    before = json.dumps({k: v for k, v in snapshot.items() if k != "_manual_frames"})
    rendered = json.loads(snapshot_payload(snapshot, viewer=True, render_only=True))
    assert "occupancy_map" not in rendered and "calibration" not in rendered
    assert rendered["environment"]["points"] == [[.12346, 0., 1.]]
    assert rendered["obbs"] == [{"corners": [[.12346, 0., 1.]]}]
    assert rendered["viewer_filter"]["obbs"] == rendered["obbs"]
    for options in ({}, {"viewer": True}, {"compact": True}, {"compact": True, "viewer": True, "render_only": True}):
        original = json.loads(snapshot_payload(snapshot, **options))
        assert original["occupancy_map"] == snapshot["occupancy_map"]
        assert original["obbs"][0]["center"] == [1, 2, 3]
    assert json.dumps({k: v for k, v in snapshot.items() if k != "_manual_frames"}) == before


def test_native_depth_preview_marks_invalid_but_never_overlays_detector_mask():
    frames, _ = fixture()
    depth = np.array([[0., np.nan, np.inf], [.5, 1., 2.], [.5, 1., 2.]])
    accepted = np.ones((3, 3), bool)
    accepted[2] = False
    frames["front"] = ManualFrame(frames["front"].rgb, depth, accepted,
                                   frames["front"].calibration, 100)
    low, high = depth_display_range(frames)
    assert np.isfinite([low, high]).all() and high > low
    image = colorize_depth(frames["front"], low, high)
    assert not image[0].any()
    np.testing.assert_array_equal(image[2], image[1])
    side = colorize_depth(frames["side"], low, high)
    np.testing.assert_array_equal(image[1, 1], side[0, 0])
    assert not np.array_equal(image[1, 0], image[1, 2])
    assert np.isnan(depth[0, 1]) and depth[1, 0] == .5


def test_manual_capture_rejects_unaligned_depth_and_rgb_dimensions(tmp_path):
    frames, grid = fixture()
    frames["front"] = ManualFrame(np.zeros((6, 6, 3), np.uint8), frames["front"].depth_m,
                                   frames["front"].accepted, frames["front"].calibration, 100)
    snapshot = {"frame": 4, "host_timestamp_ns": 100, "occupancy_map": grid.snapshot(),
                "_manual_frames": frames}
    with pytest.raises(ValueError, match="requires aligned"):
        ManualSelectionStore(tmp_path).capture(snapshot, {})


def test_color_range_ignores_out_of_range_but_keeps_native_measurements():
    frames, _ = fixture()
    depth = np.array([[.7, .8, .9], [1., 1.1, 1.2], [4.3, 32.6, 65.535]])
    accepted = np.ones((3, 3), bool)
    accepted[0, 0] = False
    frames = {"front": ManualFrame(frames["front"].rgb, depth, accepted,
                                  frames["front"].calibration, 100)}
    low, high = depth_display_range(frames)
    assert .7 <= low < high <= 1.2
    accepted[:] = False
    low, high = depth_display_range(frames)
    assert .7 <= low < high <= 1.2


def test_missing_view_or_mismatched_ids_are_rejected():
    frames, grid = fixture()
    for annotations in ({"front": [box(0, 0)]},
                        {"front": [box(0, 0, object_id="a")], "side": [box(0, 0, object_id="b")]}):
        with pytest.raises(ValueError, match="BOTH"):
            select_voxels(frames, grid.snapshot(), annotations)


def test_disjoint_view_boxes_produce_no_selected_voxels():
    frames, grid = fixture()
    annotations = {"front": [box(0, 0)], "side": [box(2, 0)]}
    for mode in ("depth", "obb"):
        assert select_voxels(frames, grid.snapshot(), annotations, mode)["voxel_count"] == 0


def test_rgb_aligned_depth_and_native_z16_buffer_are_preserved(tmp_path):
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    depth_to_color = np.eye(4)
    depth_to_color[:3, :3] = rotation
    depth_to_color[:3, 3] = [.03, -.02, .01]
    color_to_world = np.eye(4)
    color_to_world[:3, 3] = [1., 2., 3.]
    raw = np.array([[0, 300, 900], [1600, 65535, 500]], np.uint16)
    native = NativeDepthFrame(raw, .001, np.eye(3), depth_to_color)
    rgb = np.zeros((4, 6, 3), np.uint8)
    sdk_frame = RGBDFrame("front", rgb, np.full((4, 6), 42., np.float32),
                          host_timestamp_ns=100, native_depth=native)
    calibration = CameraCalibration("front", np.eye(3), color_to_world)
    frame = rgb_manual_frame(sdk_frame, calibration)
    assert frame.depth_m.shape == (4, 6)
    np.testing.assert_array_equal(frame.raw_depth_units, raw)
    np.testing.assert_allclose(frame.depth_m, 42.)
    np.testing.assert_allclose(frame.calibration.camera_to_world, color_to_world)
    _, grid = fixture()
    store = ManualSelectionStore(tmp_path)
    capture = store.capture({"frame": 1, "host_timestamp_ns": 100, "occupancy_map": grid.snapshot(),
                             "_manual_frames": {"front": frame, "side": frame}}, {})
    directory = tmp_path / capture["capture_id"]
    np.testing.assert_array_equal(cv2.imread(str(directory / "front_native_depth_units.png"), cv2.IMREAD_UNCHANGED), raw)
    evidence = np.load(directory / "depth_evidence.npz")
    np.testing.assert_array_equal(evidence["front_native_depth_units"], raw)
    assert capture["cameras"]["front"]["depth_scale_m"] == .001
    without_native = rgb_manual_frame(RGBDFrame("front", rgb, np.zeros((4, 6)), host_timestamp_ns=100), calibration)
    assert without_native.raw_depth_units is None and not without_native.accepted.any()
    with pytest.raises(ValueError, match="must match rgb height/width"):
        rgb_manual_frame(RGBDFrame("front", rgb, np.zeros((2, 3)), host_timestamp_ns=100), calibration)


def test_nonparallel_views_resolve_front_ray_ambiguity():
    K = np.array([[3., 0., 3.], [0., 3., 3.], [0., 0., 1.]])
    side_to_world = np.eye(4)
    side_to_world[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    side_to_world[:3, 3] = [3, 0, 0]
    front_depth, side_depth = np.zeros((8, 8)), np.zeros((8, 8))
    front_depth[3, 3] = 1.
    side_depth[3, 4:6] = 3.
    frames = {"front": ManualFrame(np.zeros((8, 8, 3)), front_depth, front_depth > 0,
                                    CameraCalibration("front", K, np.eye(4)), 100),
              "side": ManualFrame(np.zeros((8, 8, 3)), side_depth, side_depth > 0,
                                   CameraCalibration("side", K, side_to_world), 100)}
    grid = LocalOccupancyGrid([-.05, -.05, .95], np.eye(3), [2, 2, 12], .1)
    grid.update(timestamp_ns=100, raw_hits=[[0, 0, 1], [0, 0, 2]], free_ids=np.empty(0, np.int64))
    for side_u, expected_z in ((4, 1), (5, 2)):
        result = select_voxels(frames, grid.snapshot(), {"front": [box(3, 3)], "side": [box(side_u, 3)]})
        assert result["voxel_ids"] == grid.point_ids(np.array([[0, 0, expected_z]])).tolist()


def test_front_only_selection_requires_explicit_mode_and_uses_accepted_occupied_depth():
    frames, grid = fixture()
    front = {"front": frames["front"]}
    annotations = {"front": [box(1, 0, width=3), box(1, 1), box(2, 2)]}
    with pytest.raises(ValueError, match="all configured"):
        select_voxels(front, grid.snapshot(), annotations)
    result = select_voxels(front, grid.snapshot(), annotations, camera_names=("front",))
    assert result["voxel_ids"] == sorted(grid.point_ids(np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]])).tolist())
    assert result["fusion_rule"] == "single_view_depth" and result["camera_names"] == ["front"]
    # Rejected center pixel and never-occupied returns are not turned into cells.
    assert result["accepted_points"]["front"]["1"] == 3
    with pytest.raises(ValueError, match="captured camera names"):
        select_voxels(front, grid.snapshot(), {**annotations, "side": []}, camera_names=("front",))


def test_front_only_3d_envelope_still_respects_rgb_boxes_and_occupied_cells():
    frames, grid = fixture()
    frame = frames["front"]
    frame.accepted[0, 1] = False
    front = {"front": frame}
    annotations = {"front": [box(1, 0, width=3)]}
    exact = select_voxels(front, grid.snapshot(), annotations, camera_names=("front",))
    envelope = select_voxels(front, grid.snapshot(), annotations, "obb", camera_names=("front",))
    middle = grid.point_ids(np.array([[1, 0, 1]]))[0]
    assert middle not in exact["voxel_ids"] and middle in envelope["voxel_ids"]
    restricted = select_voxels(front, grid.snapshot(), {"front": [box(0, 0), box(2, 0)]},
                               "obb", camera_names=("front",))
    assert middle not in restricted["voxel_ids"]


def test_front_only_capture_export_preview_and_dual_replay_isolation(tmp_path):
    from obstacle_validation.manual_voxel_filter import ManualVoxelFilter
    frames, grid = fixture()
    snapshot = {"frame": 8, "coordinate_frame": "ur_base", "host_timestamp_ns": 100,
                "occupancy_map": grid.snapshot(), "_manual_frames": {"front": frames["front"]},
                "calibration": {"sha256": "single-calibration"}}
    store = ManualSelectionStore(tmp_path, camera_names=("front",), coordinate_frame="ur_base")
    capture = store.capture(snapshot, {"frame": 8, "coordinate_frame": "ur_base"})
    result = store.select(capture["capture_id"], {"front": [box(2, 0)]}, "depth")
    directory = tmp_path / capture["capture_id"]
    assert capture["camera_names"] == ["front"] and set(capture["images"]) == {"front"}
    assert not (directory / "side_rgb.png").exists()
    for name in ("capture.json", "annotations.json", "selection.json"):
        data = json.loads((directory / name).read_text())
        assert data["coordinate_frame"] == "ur_base"
        assert data["fusion_rule"] == "single_view_depth"
        assert data["camera_names"] == ["front"]
    assert store.scene_payload()["manual_selection"] == result
    assert np.load(directory / "selected_voxels.npz")["centers"].shape == (1, 3)
    with pytest.raises(ValueError, match="do not match"):
        ManualVoxelFilter(directory / "selection.json")
    with pytest.raises(ValueError, match="requires aligned"):
        ManualSelectionStore(tmp_path / "dual").capture(snapshot, {})
    with pytest.raises(ValueError, match="coordinate frame"):
        store.capture({**snapshot, "coordinate_frame": "left_base"}, {})
    with pytest.raises(ValueError, match="--occupancy-map"):
        store.capture({k: v for k, v in snapshot.items() if k != "occupancy_map"}, {})
