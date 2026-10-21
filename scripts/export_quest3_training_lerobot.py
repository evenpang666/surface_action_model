#!/usr/bin/env python3
"""Export a measured Quest3 surface shard and four RGB streams as LeRobot.

The action remains the recorder's per-frame joint delta plus gripper target.
OBBs are image-plane RGB labels, not 3-D collision boxes. Their reviewed mask
distinguishes missing labels from explicitly empty scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_scripts.batch_obstacle_annotation import GROUP, SCHEMA, PAIRED_SCHEMA, rgb_image, validate_boxes
from safety_module.shards import open_shard

VIEWS = ("front", "side", "left_wrist", "right_wrist")
BOX_FIELDS = ("cx", "cy", "width", "height", "angle_deg")
MAX_BOXES = 32


def annotation_rows(episode: h5py.File, indices: np.ndarray, rgb_shapes: dict[str, tuple[int, int, int]]):
    """Return fixed-size RGB boxes; zero reviewed never means no obstacle."""
    rows = []
    if GROUP not in episode:
        return [None] * len(indices), {}
    labels = episode[GROUP]
    if labels.attrs.get("schema") not in (SCHEMA, PAIRED_SCHEMA) or labels.attrs.get("annotation_image") != "recorded_rgb":
        raise ValueError("Obstacle annotations must use the RGB-pixel v2/v3 schema")
    count = len(episode["timestamps_ns"])
    if (labels["reviewed"].shape != (count,) or labels["frames_json"].shape != (count,)
            or not np.array_equal(labels["timestamps_ns"][:], episode["timestamps_ns"][:])):
        raise ValueError("Obstacle annotations do not align with HDF5 frames")
    names = set()
    for index in indices:
        if not bool(labels["reviewed"][index]):
            rows.append(None)
            continue
        row = json.loads(labels["frames_json"][index])
        if (row.get("schema") not in (SCHEMA, PAIRED_SCHEMA) or row.get("annotation_image") != "recorded_rgb"
                or row.get("frame_index") != int(index)
                or row.get("timestamp_ns") != str(int(episode["timestamps_ns"][index]))):
            raise ValueError(f"Obstacle annotation for frame {index} is stale or in the wrong coordinates")
        boxes = row.get("boxes", {})
        if set(boxes) != {"front", "side"} or any(len(boxes[view]) > MAX_BOXES for view in ("front", "side")):
            raise ValueError(f"Obstacle annotation for frame {index} has invalid views or box count")
        dimensions = {view: {"rgb_height": rgb_shapes[view][0], "rgb_width": rgb_shapes[view][1]}
                      for view in ("front", "side")}
        if row["schema"] == PAIRED_SCHEMA and not isinstance(row.get("visibility"), dict):
            raise ValueError("Paired v3 annotations require explicit visibility")
        validate_boxes(boxes, row.get("status"), dimensions,
                       row.get("visibility") if row["schema"] == PAIRED_SCHEMA else None)
        names.update(box["object_id"] for view in ("front", "side") for box in boxes[view])
        rows.append(row)
    return rows, {name: number for number, name in enumerate(sorted(names), start=1)}


def packed_boxes(row, view: str, ids: dict[str, int]):
    boxes = np.zeros((MAX_BOXES, len(BOX_FIELDS)), np.float32)
    object_ids = np.zeros((MAX_BOXES,), np.int64)
    if row is not None:
        for slot, box in enumerate(row["boxes"][view]):
            boxes[slot] = [float(box[field]) for field in BOX_FIELDS]
            object_ids[slot] = ids[box["object_id"]]
    return boxes, object_ids


def export(shard_path: Path, episode_path: Path, output: Path, *, repo_id: str, fps: int = 20,
           require_reviewed: bool = False, dataset_class=None):
    shard_path, episode_path, output = map(lambda p: Path(p).expanduser().resolve(),
                                           (shard_path, episode_path, output))
    if output.exists():
        raise FileExistsError(f"LeRobot output already exists: {output}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dataset_class is None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset_class = LeRobotDataset
    with open_shard(shard_path) as shard, h5py.File(episode_path, "r") as episode:
        if str(np.asarray(shard["surface_source"]).item()) != "measured_depth_cotracker":
            raise ValueError("Training export requires a measured CoTracker shard, not geometry-only FK points")
        recorded_source = Path(str(np.asarray(shard["source_episode"]).item())).resolve()
        if recorded_source != episode_path:
            raise ValueError("Shard was built from a different HDF5 episode")
        if str(np.asarray(shard["coordinate_frame"]).item()) != "left_base":
            raise ValueError("Surface points must be in left_base")
        indices = np.asarray(shard["sample_frame_indices"], dtype=np.int64)
        timestamps = np.asarray(episode["timestamps_ns"], dtype=np.int64)
        if (not len(indices) or np.any(indices < 0) or np.any(indices >= len(timestamps))
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("Shard sample indices do not fit the episode")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Episode timestamps must be strictly increasing")
        if not np.array_equal(np.asarray(shard["timestamps_ns"], dtype=np.int64), timestamps):
            raise ValueError("Shard timestamps no longer match the episode")
        state = np.asarray(shard["qpos"], dtype=np.float32)
        grip = np.asarray(shard["gripper_position"], dtype=np.float32)
        actions = np.asarray(shard["actions"], dtype=np.float32)
        points = shard["visual_robot_current_points"]
        point_mask = shard["current_surface_mask"]
        offsets = shard["visual_robot_future_offsets"]
        flow_mask = shard["visual_robot_flow_supervision_mask"]
        chunks, step_mask = shard["action_chunks"], shard["action_step_mask"]
        n, point_count, horizon = len(indices), points.shape[1], chunks.shape[1]
        if (state.shape != (len(timestamps), 12) or grip.shape != (len(timestamps), 2)
                or actions.shape != (len(timestamps), 14) or points.shape != (n, point_count, 3)
                or point_mask.shape != (n, point_count) or offsets.shape != (n, horizon, point_count, 3)
                or flow_mask.shape != (n, horizon, point_count) or chunks.shape != (n, horizon, 14)
                or step_mask.shape != (n, horizon)):
            raise ValueError("Shard state/action/CoTracker arrays are not frame-aligned")
        image_sets = {}
        features = {
            "observation.state": {"dtype": "float32", "shape": (14,)},
            "action": {"dtype": "float32", "shape": (14,)},
            "action_chunk": {"dtype": "float32", "shape": (horizon, 14)},
            "action_step_mask": {"dtype": "int64", "shape": (horizon,)},
            "observation.robot_surface": {"dtype": "float32", "shape": (point_count, 3)},
            "observation.robot_surface_mask": {"dtype": "int64", "shape": (point_count,)},
            "target.robot_surface_offsets": {"dtype": "float32", "shape": (horizon, point_count, 3)},
            "target.robot_surface_mask": {"dtype": "int64", "shape": (horizon, point_count)},
            "source_frame_index": {"dtype": "int64", "shape": (1,)},
            "source_timestamp_ns": {"dtype": "int64", "shape": (1,)},
            "annotation.reviewed": {"dtype": "int64", "shape": (1,)},
        }
        for view in VIEWS:
            key = f"observations/images/{view}/rgb"
            if key not in episode or len(episode[key]) != len(timestamps):
                raise ValueError(f"Missing frame-aligned RGB view: {view}")
            image_sets[view] = episode[key]
            height, width = rgb_image(image_sets[view], int(indices[0])).shape[:2]
            features[f"observation.images.{view}"] = {"dtype": "image", "shape": (height, width, 3)}
        for view in ("front", "side"):
            features[f"annotation.obstacle_obbs.{view}"] = {"dtype": "float32", "shape": (MAX_BOXES, 5)}
            features[f"annotation.obstacle_ids.{view}"] = {"dtype": "int64", "shape": (MAX_BOXES,)}
        rows, object_ids = annotation_rows(
            episode, indices, {view: features[f"observation.images.{view}"]["shape"] for view in VIEWS})
        if require_reviewed and any(row is None for row in rows):
            raise ValueError("Some frames are unreviewed; annotate them before final training export")
        dataset = dataset_class.create(repo_id=repo_id, root=output, robot_type="dual_ur7e_robotiq_2f85",
                                       fps=fps, features=features, use_videos=False)
        for row_index, source_index in enumerate(indices):
            source_index = int(source_index)
            frame = {
                "observation.state": np.r_[state[source_index], grip[source_index]],
                "action": actions[source_index],
                "action_chunk": np.asarray(chunks[row_index], np.float32),
                "action_step_mask": np.asarray(step_mask[row_index], np.int64),
                "observation.robot_surface": np.asarray(points[row_index], np.float32),
                "observation.robot_surface_mask": np.asarray(point_mask[row_index], np.int64),
                "target.robot_surface_offsets": np.asarray(offsets[row_index], np.float32),
                "target.robot_surface_mask": np.asarray(flow_mask[row_index], np.int64),
                "source_frame_index": np.asarray([source_index], np.int64),
                "source_timestamp_ns": np.asarray([timestamps[source_index]], np.int64),
                "annotation.reviewed": np.asarray([rows[row_index] is not None], np.int64),
                "task": str(np.asarray(shard["task_text"]).item()),
            }
            for view, images in image_sets.items():
                image = rgb_image(images, source_index)
                if image.shape != features[f"observation.images.{view}"]["shape"]:
                    raise ValueError(f"RGB dimensions changed for {view} at frame {source_index}")
                frame[f"observation.images.{view}"] = image
            for view in ("front", "side"):
                frame[f"annotation.obstacle_obbs.{view}"], frame[f"annotation.obstacle_ids.{view}"] = packed_boxes(
                    rows[row_index], view, object_ids)
            dataset.add_frame(frame)
        dataset.save_episode()
        metadata = {"schema": "quest3_surface_training_lerobot_v1", "source_hdf5": str(episode_path),
                    "source_shard": str(shard_path), "annotation_schema": episode[GROUP].attrs["schema"] if GROUP in episode else None,
                    "annotation_image": "recorded_rgb", "object_ids": object_ids,
                    "action_semantics": "joint_delta_radians_plus_gripper_target",
                    "surface_coordinate_frame": "left_base", "rgb_views": list(VIEWS),
                    "frames": n, "reviewed_frames": sum(row is not None for row in rows),
                    "fps": fps, "source_timestamps_preserved": True}
        (output / "quest3_training_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/quest3_dual_surface_training")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.shard, args.episode, args.output, repo_id=args.repo_id,
                            fps=args.fps, require_reviewed=args.require_reviewed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
