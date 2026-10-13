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


def test_internal_annotation_arrays_never_enter_public_snapshots():
    from real_scripts.live_dual_ur7e_obstacle_model import snapshot_payload
    snapshot = {"frame": 1, "obbs": [], "_manual_frames": {"front": object()}}
    for options in ({}, {"compact": True}, {"viewer": True}):
        assert json.loads(snapshot_payload(snapshot, **options)) == {"frame": 1, "obbs": []}


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
