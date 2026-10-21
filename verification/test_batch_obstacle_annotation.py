import base64
import hashlib
import json

import cv2
import h5py
import numpy as np
import pytest
import yaml

from real_scripts.batch_obstacle_annotation import BatchAnnotations, GROUP, SCHEMA, inspect_episode


def episode(path, n=3, *, native=True, jpeg=False, scale=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as source:
        source.attrs.update(complete=True, num_samples=n, task_description="test task")
        if not native:
            source.attrs["config_yaml"] = yaml.safe_dump({"cameras": {
                name: {"depth_topic": f"/{name}/camera/aligned_depth_to_color/image_raw"} for name in ("front", "side")}})
        source.create_dataset("timestamps_ns", data=np.arange(n, dtype=np.uint64)+1700000000000000000)
        source.create_dataset("action", data=np.arange(n*4, dtype=np.float32).reshape(n, 4))
        for offset, name in enumerate(("front", "side")):
            group = source.require_group(f"observations/images/{name}")
            depth = np.arange(n*4*5, dtype=np.uint16).reshape(n, 4, 5)+1000
            ds = group.create_dataset("native_depth_units" if native else "depth", data=depth)
            if scale:
                ds.attrs["depth_scale_m"] = .001
            rgb = np.full((n, 6, 8, 3), [20, 40, 60], np.uint8)
            if jpeg:
                images = group.create_dataset("rgb", (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for i in range(n):
                    images[i] = cv2.imencode(".jpg", rgb[i, ..., ::-1])[1].reshape(-1)
            else:
                group.create_dataset("rgb", data=rgb)
            source.create_dataset(f"source_timestamps_ns/{name}_depth",
                                  data=np.arange(n, dtype=np.uint64)+1700000000000000000+offset)
    return path


def boxes():
    return {name: [dict(object_id="tube", cx=2., cy=2., width=2., height=2., angle_deg=25.)]
            for name in ("front", "side")}


def save_body(index=0, revision=0, *, empty=False):
    return {"episode_id": 0, "frame_index": index, "revision": revision,
            "status": "no_obstacles" if empty else "objects",
            "boxes": {"front": [], "side": []} if empty else boxes()}


def test_inventory_and_lazy_rgb_frame_reads_cover_all_files_without_writes(tmp_path):
    paths = [episode(tmp_path/"episode_00.hdf5", n=2), episode(tmp_path/"nested/episode_01.h5", n=4, jpeg=True)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    dataset = BatchAnnotations(tmp_path)
    manifest = dataset.manifest()
    assert manifest["total_frames"] == 6 and manifest["reviewed_frames"] == 0
    for episode_id, count in enumerate((2, 4)):
        for i in range(count):
            frame = dataset.frame(episode_id, i)
            assert frame["frame_index"] == i and frame["revision"] == 0 and not frame["reviewed"]
            assert frame["timestamp_ns"] == str(1700000000000000000+i)
            assert frame["annotation_image"] == "recorded_rgb"
            rgb = cv2.imdecode(np.frombuffer(base64.b64decode(frame["images"]["side"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            assert rgb.shape == (6, 8, 3)
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]


def test_labels_write_into_exact_original_sample_and_leave_original_data_unchanged(tmp_path):
    path = episode(tmp_path/"episode.hdf5")
    with h5py.File(path, "r") as source:
        original_action = source["action"][:]
        original_depth = source["observations/images/front/native_depth_units"][:]
        root_attrs = dict(source.attrs)
    dataset = BatchAnnotations(tmp_path)
    result = dataset.save(save_body(1))
    assert result["revision"] == 1
    with h5py.File(path, "r") as source:
        group = source[GROUP]
        assert group.attrs["schema"] == SCHEMA
        assert group.attrs["annotation_image"] == "recorded_rgb"
        assert group["reviewed"][:].tolist() == [False, True, False]
        assert group["revision"][:].tolist() == [0, 1, 0]
        np.testing.assert_array_equal(group["frame_index"][:], [0, 1, 2])
        np.testing.assert_array_equal(group["timestamps_ns"][:], source["timestamps_ns"][:])
        saved = json.loads(group["frames_json"][1])
        assert saved["frame_index"] == 1 and saved["boxes"] == boxes()
        assert saved["timestamp_ns"] == str(1700000000000000001)
        assert saved["source_timestamps_ns"]["side"] == str(1700000000000000002)
        assert saved["depth_sources"]["front"]["depth_kind"] == "native_raw_depth"
        assert saved["depth_sources"]["front"]["effective_depth_scale_m"] == .001
        assert saved["annotation_image"] == "recorded_rgb"
        np.testing.assert_array_equal(source["action"][:], original_action)
        np.testing.assert_array_equal(source["observations/images/front/native_depth_units"][:], original_depth)
        assert dict(source.attrs) == root_attrs
    resumed = BatchAnnotations(tmp_path)
    assert resumed.manifest()["reviewed_frames"] == 1
    assert resumed.manifest()["episodes"][0]["first_unreviewed"] == 0
    assert resumed.frame(0, 1)["boxes"] == boxes()


def test_explicit_empty_labels_and_stale_edit_rejection(tmp_path):
    path = episode(tmp_path/"episode.hdf5", n=2)
    dataset = BatchAnnotations(tmp_path)
    dataset.save(save_body(0))
    with pytest.raises(ValueError, match="another window"):
        dataset.save(save_body(0))
    assert dataset.frame(0, 0)["boxes"] == boxes()
    dataset.save(save_body(0, revision=1, empty=True))
    frame = dataset.frame(0, 0)
    assert frame["reviewed"] and frame["status"] == "no_obstacles" and frame["boxes"] == {"front": [], "side": []}
    assert not dataset.frame(0, 1)["reviewed"]
    with h5py.File(path, "r") as source:
        assert source[GROUP+"/revision"][0] == 2


@pytest.mark.parametrize("fault", ["single_view", "wrong_id", "bad_coordinate", "bad_index", "implicit_empty"])
def test_invalid_labels_never_create_annotation_group(tmp_path, fault):
    path = episode(tmp_path/"episode.hdf5")
    dataset = BatchAnnotations(tmp_path)
    body = save_body()
    if fault == "single_view": body["boxes"]["side"] = []
    elif fault == "wrong_id": body["boxes"]["side"][0]["object_id"] = "other"
    elif fault == "bad_coordinate": body["boxes"]["front"][0]["cx"] = -1
    elif fault == "bad_index": body["frame_index"] = 3
    else: body["boxes"] = {"front": [], "side": []}
    with pytest.raises(ValueError):
        dataset.save(body)
    with h5py.File(path, "r") as source:
        assert GROUP not in source


def test_recorded_aligned_depth_can_still_be_annotated_in_rgb_pixels(tmp_path):
    path = episode(tmp_path/"episode.hdf5", native=False)
    dataset = BatchAnnotations(tmp_path)
    assert dataset.manifest()["episodes"][0]["cameras"]["front"]["depth_kind"] == "recorded_aligned_depth"
    dataset.save(save_body())
    with h5py.File(path, "r") as source:
        assert bool(source[GROUP+"/reviewed"][0])


def test_rgb_annotation_does_not_require_a_depth_dataset(tmp_path):
    path = episode(tmp_path/"episode.hdf5")
    with h5py.File(path, "r+") as source:
        for name in ("front", "side"):
            del source[f"observations/images/{name}/native_depth_units"]
    dataset = BatchAnnotations(tmp_path)
    assert dataset.manifest()["episodes"][0]["cameras"]["front"]["depth_kind"] == "absent"
    dataset.save(save_body())
    with h5py.File(path, "r") as source:
        assert source[GROUP].attrs["annotation_image"] == "recorded_rgb"


def test_rgb_bounds_are_independent_of_native_depth_dimensions(tmp_path):
    episode(tmp_path/"episode.hdf5")
    dataset = BatchAnnotations(tmp_path)
    body = save_body()
    body["boxes"]["front"][0]["cx"] = 7
    dataset.save(body)
    body["frame_index"], body["revision"] = 1, 0
    body["boxes"]["front"][0]["cx"] = 8
    with pytest.raises(ValueError, match="within the captured image"):
        dataset.save(body)


def test_legacy_depth_pixel_schema_is_not_silently_reinterpreted(tmp_path):
    path = episode(tmp_path/"episode.hdf5")
    with h5py.File(path, "r+") as source:
        source.require_group(GROUP).attrs["schema"] = "dual_depth_obb_annotation_v1"
    with pytest.raises(ValueError, match="RGB coordinates require a new annotation group"):
        BatchAnnotations(tmp_path)


def test_resume_rejects_changed_frame_timestamps_or_scale(tmp_path):
    path = episode(tmp_path/"episode.hdf5", scale=False)
    dataset = BatchAnnotations(tmp_path, depth_unit_m=.001)
    dataset.save(save_body())
    with pytest.raises(ValueError, match="original --depth-unit-m"):
        BatchAnnotations(tmp_path, depth_unit_m=.002)
    with h5py.File(path, "r+") as source:
        source["timestamps_ns"][0] = 999
    with pytest.raises(ValueError, match="timestamps no longer match"):
        inspect_episode(path)


def test_frame_count_mismatch_is_not_silently_truncated(tmp_path):
    path = episode(tmp_path/"episode.hdf5")
    with h5py.File(path, "r+") as source:
        source.attrs["num_samples"] = 2
    with pytest.raises(ValueError, match="num_samples"):
        BatchAnnotations(tmp_path)


def test_paired_annotations_preserve_ids_rotation_and_confirmed_visibility(tmp_path):
    from real_scripts.batch_obstacle_annotation import PAIRED_SCHEMA
    path = episode(tmp_path / "paired.h5", n=2)
    dataset = BatchAnnotations(path)
    body = save_body()
    first = dict(object_id="a", cx=2., cy=2., width=2., height=1., angle_deg=35.)
    second = dict(object_id="b", cx=5., cy=3., width=2., height=1., angle_deg=-20.)
    body["boxes"] = {"front": [first, second], "side": [second, first]}
    # A legacy fully-visible save remains readable after a v3 group upgrade.
    dataset.save(body)
    body.update(frame_index=1, boxes={"front": [], "side": [first]},
                visibility={"a": {"front": False, "side": True}})
    dataset.save(body)
    resumed = BatchAnnotations(path)
    assert resumed.frame(0, 1)["visibility"] == body["visibility"]
    assert resumed.manifest()["reviewed_frames"] == 2
    with h5py.File(path, "r") as source:
        assert source[GROUP].attrs["schema"] == PAIRED_SCHEMA
        assert json.loads(source[GROUP+"/frames_json"][0])["schema"] == SCHEMA
        row = json.loads(source[GROUP+"/frames_json"][1])
        assert row["schema"] == PAIRED_SCHEMA
        assert row["boxes"]["side"][0]["angle_deg"] == 35.


@pytest.mark.parametrize("fault", ["duplicate", "visibility_mismatch", "both_invisible"])
def test_invalid_pair_annotations_are_rejected_before_writing(tmp_path, fault):
    path = episode(tmp_path / "episode.h5")
    original = path.read_bytes()
    body = save_body()
    if fault == "duplicate":
        body["boxes"]["front"].append(dict(body["boxes"]["front"][0]))
    elif fault == "visibility_mismatch":
        body["visibility"] = {"tube": {"front": False, "side": True}}
    else:
        body["boxes"] = {"front": [], "side": []}
        body["visibility"] = {"tube": {"front": False, "side": False}}
    with pytest.raises(ValueError):
        BatchAnnotations(path).save(body)
    assert path.read_bytes() == original
