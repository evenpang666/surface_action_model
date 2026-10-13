import json

import h5py
import numpy as np
import pytest

from scripts.export_quest3_training_lerobot import export


class FakeDataset:
    def __init__(self, root, features):
        self.root, self.features, self.frames = root, features, []

    @classmethod
    def create(cls, **kwargs):
        obj = cls(kwargs["root"], kwargs["features"])
        obj.root.mkdir()
        cls.last = obj
        return obj

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        pass


def sources(tmp_path, *, mismatch=False):
    episode = tmp_path / "episode.hdf5"
    with h5py.File(episode, "w") as h:
        h.create_dataset("timestamps_ns", data=np.array([100, 200], np.int64))
        for name in ("front", "side", "left_wrist", "right_wrist"):
            h.create_dataset(f"observations/images/{name}/rgb", data=np.full((2, 4, 5, 3), 8, np.uint8))
        group = h.require_group("annotations/obstacle_obbs")
        group.attrs.update(schema="dual_rgb_obb_annotation_v2", annotation_image="recorded_rgb")
        group.create_dataset("reviewed", data=np.array([True, False]))
        group.create_dataset("timestamps_ns", data=np.array([100, 200], np.int64))
        entries = group.create_dataset("frames_json", shape=(2,), dtype=h5py.string_dtype())
        entries[0] = json.dumps({"schema": "dual_rgb_obb_annotation_v2", "annotation_image": "recorded_rgb",
                                 "frame_index": 0, "timestamp_ns": "100", "status": "objects", "boxes": {
                                     view: [{"object_id": "tube", "cx": 4., "cy": 2., "width": 2.,
                                             "height": 2., "angle_deg": 0.}]
                                     for view in ("front", "side")}})
    shard = tmp_path / "episode.npz"
    np.savez(shard,
             surface_source="measured_depth_cotracker", source_episode=str(episode), coordinate_frame="left_base",
             sample_frame_indices=np.arange(2), timestamps_ns=np.array([100, 201 if mismatch else 200], np.int64),
             qpos=np.zeros((2, 12), np.float32), gripper_position=np.zeros((2, 2), np.float32),
             actions=np.ones((2, 14), np.float32),
             visual_robot_current_points=np.ones((2, 3, 3), np.float32),
             current_surface_mask=np.ones((2, 3), bool),
             visual_robot_future_offsets=np.zeros((2, 2, 3, 3), np.float32),
             visual_robot_flow_supervision_mask=np.ones((2, 2, 3), bool),
             action_chunks=np.ones((2, 2, 14), np.float32),
             action_step_mask=np.array([[True, True], [True, False]]), task_text="pick up tube")
    return episode, shard


def test_export_preserves_four_rgb_views_surface_and_reviewed_obbs(tmp_path):
    episode, shard = sources(tmp_path)
    metadata = export(shard, episode, tmp_path / "lerobot", repo_id="local/test", dataset_class=FakeDataset)
    assert metadata["reviewed_frames"] == 1 and metadata["object_ids"] == {"tube": 1}
    assert len(FakeDataset.last.frames) == 2
    first, second = FakeDataset.last.frames
    assert first["task"] == "pick up tube"
    assert first["observation.images.front"].shape == (4, 5, 3)
    assert first["observation.images.right_wrist"].shape == (4, 5, 3)
    assert first["observation.robot_surface"].shape == (3, 3)
    assert first["annotation.obstacle_ids.front"][0] == 1
    assert first["annotation.obstacle_obbs.side"][0, 0] == 4
    assert first["annotation.reviewed"].tolist() == [1]
    assert second["annotation.reviewed"].tolist() == [0]
    assert not second["annotation.obstacle_ids.front"].any()
    assert json.loads((tmp_path / "lerobot/quest3_training_metadata.json").read_text())["annotation_image"] == "recorded_rgb"


def test_export_rejects_stale_shard_and_unreviewed_final_export(tmp_path):
    episode, shard = sources(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="timestamps no longer match"):
        export(shard, episode, tmp_path / "bad", repo_id="local/test", dataset_class=FakeDataset)
    episode, shard = sources(tmp_path)
    with pytest.raises(ValueError, match="unreviewed"):
        export(shard, episode, tmp_path / "bad", repo_id="local/test", require_reviewed=True,
               dataset_class=FakeDataset)
