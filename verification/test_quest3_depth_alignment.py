"""Quest3 preprocessing consumes already RGB-aligned HDF5 depth unchanged."""

import json

import h5py
import numpy as np
import pytest
import yaml

from real_scripts.quest3_depth import Quest3DepthReader, recorded_intrinsics
from real_scripts.real_robot_adapter import CameraCalibration


def intrinsics(width=2, height=2, **kwargs):
    return dict(width=width, height=height, fx=1., fy=1., ppx=0., ppy=0., **kwargs)


def make_source(path, *, topic="/front/aligned_depth_to_color/image_raw", declaration=None,
                depth=None):
    with h5py.File(path, "w") as source:
        if topic:
            source.attrs["config_yaml"] = yaml.safe_dump({"cameras": {"front": {"depth_topic": topic}}})
        dataset = source.create_dataset(
            "observations/images/front/depth",
            data=np.ones((2, 2, 2), np.uint16) * 1000 if depth is None else depth)
        if declaration:
            dataset.attrs["coordinate_system"] = declaration


def reader(source, *, shape=(2, 2), scale=None):
    return Quest3DepthReader(source, view="front", rgb_shape=shape,
                             calibration=CameraCalibration("front", np.eye(3), np.eye(4)),
                             depth_unit_m=scale)


def test_aligned_reader_uses_recorded_pixels_without_reprojection(tmp_path):
    path = tmp_path / "episode.hdf5"
    depth = np.array([[[0, 1000], [2000, 3000]], [[4000, 0], [5000, 6000]]], np.uint16)
    make_source(path, depth=depth)
    with h5py.File(path, "r+") as source:
        source["observations/images/front/depth"].attrs["depth_scale_m"] = .002
        result = reader(source)
        assert result.metadata["input_mode"] == "recorded_aligned"
        assert result.metadata["rasterization"] == "none"
        np.testing.assert_array_equal(result.read(0), depth[0].astype(np.float32) * .002)
        np.testing.assert_array_equal(result.read(1), depth[1].astype(np.float32) * .002)
        assert reader(source, scale=.001).depth_unit_m == .001


@pytest.mark.parametrize("topic,declaration", [
    ("/front/depth/image_rect_raw", None),
    ("/front/depth/image_raw", "aligned"),
    ("/front/aligned_depth_to_color/image_raw", "native_raw_depth"),
    (None, None),
])
def test_reader_rejects_native_or_unknown_coordinates_even_when_dimensions_match(
        tmp_path, topic, declaration):
    path = tmp_path / "episode.hdf5"
    make_source(path, topic=topic, declaration=declaration)
    with h5py.File(path) as source:
        with pytest.raises(ValueError, match="Native depth|coordinates are unknown"):
            reader(source)


def test_aligned_reader_rejects_mismatched_dimensions_and_bad_scale(tmp_path):
    path = tmp_path / "episode.hdf5"
    make_source(path)
    with h5py.File(path) as source:
        with pytest.raises(ValueError, match="matching recorded RGB dimensions"):
            reader(source, shape=(2, 4))
        with pytest.raises(ValueError, match="Depth unit"):
            reader(source, scale=0.)


def test_recorded_color_intrinsics_keep_crop_and_resize(tmp_path):
    spec = dict(width=1280, height=720, fx=640., fy=640., ppx=640., ppy=360.)
    K = recorded_intrinsics(spec, width=640, height=480)
    np.testing.assert_allclose(K, [[640 * 2 / 3, 0, 320], [0, 640 * 2 / 3, 240], [0, 0, 1]])
    spec["image_transform"] = "none"
    with pytest.raises(ValueError, match="dimensions"):
        recorded_intrinsics(spec, width=640, height=480)


def test_aligned_extractor_passes_depth_directly_to_measured_supervision(tmp_path, monkeypatch):
    from real_scripts import preprocess_left_vr_hdf5_cotracker_observations as extractor

    points = np.array([[1., 0, 1], [2., 0, 1], [1., 1, 1], [2., 1, 1]], np.float32)
    geometry = tmp_path / "gate.npz"
    np.savez(geometry, fixed_link_points=np.broadcast_to(points, (2, 1, 4, 3)),
             qpos=np.zeros((2, 6)), link_names=["left_link"], arm_count=1)
    calibration_path = tmp_path / "world.yaml"
    calibration_path.write_text(yaml.safe_dump(dict(intrinsics=intrinsics(4, 2), matrix=np.eye(4).tolist())))
    monkeypatch.setattr(extractor, "_track_cotracker", lambda video, seeds, **kwargs: (
        np.broadcast_to(seeds, (len(video), len(seeds), 2)).copy(), np.ones((len(video), len(seeds)), bool)))
    monkeypatch.setattr(extractor, "_write_preview", lambda *args: None)
    monkeypatch.setattr(extractor, "_write_html", lambda *args, **kwargs: None)
    path = tmp_path / "aligned.hdf5"
    depth = np.broadcast_to(np.array([[0, 1000, 1000, 0], [0, 1000, 1000, 0]], np.uint16), (2, 2, 4))
    make_source(path, depth=depth)
    with h5py.File(path, "r+") as source:
        source["observations/images/front/rgb"] = np.zeros((2, 2, 4, 3), np.uint8)
        source["timestamps_ns"] = [100, 200]
    output = tmp_path / "tracks.npz"
    monkeypatch.setattr("sys.argv", ["extractor", "--episode", str(path), "--geometry-shard", str(geometry),
                        "--calibration", str(calibration_path), "--output", str(output),
                        "--preview", str(tmp_path / "preview.png"), "--visualization", str(tmp_path / "view.html"),
                        "--tracking-scale", "1", "--seed-stride", "1", "--render-splat-pixels", "0"])
    extractor.main()
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["observed_points"], np.broadcast_to(points, (2, 4, 3)))
        assert data["observed_visible"].all()
        assert json.loads(data["depth_alignment_metadata"].item())["rasterization"] == "none"


def test_training_merge_retains_recorded_depth_provenance(tmp_path):
    from scripts.preprocess_quest3_hdf5 import _merge_visual

    episode = tmp_path / "episode.hdf5"
    measured = tmp_path / "measured.npz"
    metadata = json.dumps({"input_mode": "recorded_aligned", "rasterization": "none"})
    np.savez(measured, observed_points=np.ones((2, 1, 3), np.float32),
             observed_visible=np.ones((2, 1), bool), source_episode=str(episode),
             depth_alignment_metadata=metadata, depth_unit_m=.001, intrinsics=np.eye(3))
    result = _merge_visual(measured, episode_path=episode, sample_indices=np.array([0]), horizon=1,
                           frame_count=2)
    assert result["visual_depth_alignment_metadata"].item() == metadata


def test_right_base_front_calibration_is_composed_into_the_shared_frame(tmp_path):
    from real_scripts.preprocess_left_vr_hdf5_cotracker_observations import _recorded_intrinsics

    left_right = np.eye(4)
    left_right[:3, 3] = [1., .2, .1]
    right_front = np.eye(4)
    right_front[:3, 3] = [.3, .4, .5]
    path = tmp_path / "right_base_to_front_camera.yaml"
    path.write_text(yaml.safe_dump(dict(parent_frame="right_base",
                                       child_frame="front_camera_color_optical_frame",
                                       intrinsics=intrinsics(), matrix=right_front.tolist())))
    _, actual = _recorded_intrinsics(path, width=2, height=2, right_base_to_left_base=left_right)
    np.testing.assert_allclose(actual, left_right @ right_front)
    with pytest.raises(ValueError, match="requires right_base_to_left_base"):
        _recorded_intrinsics(path, width=2, height=2)


@pytest.mark.parametrize("child", ["left_wrist_camera_color_optical_frame",
                                   "right_wrist_camera_color_optical_frame",
                                   "side_camera_color_optical_frame"])
def test_front_extractor_rejects_other_camera_extrinsics(tmp_path, child):
    from real_scripts.preprocess_left_vr_hdf5_cotracker_observations import _recorded_intrinsics

    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(dict(parent_frame="left_base", child_frame=child,
                                       intrinsics=intrinsics(), matrix=np.eye(4).tolist())))
    with pytest.raises(ValueError, match="requires a fixed front camera"):
        _recorded_intrinsics(path, width=2, height=2)
