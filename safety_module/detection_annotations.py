"""Load reviewed front/side RGB OBB labels aligned to surface shards."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "dual_rgb_obb_annotation_v2"
PAIRED_SCHEMA = "dual_rgb_paired_obb_v3"
GROUP = "annotations/obstacle_obbs"
VIEWS = ("front", "side")
MAX_BOXES = 32


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _source_path(shard_path: Path, shard) -> Path:
    original = Path(_text(np.asarray(shard["source_episode"]).item())).expanduser()
    preserved = shard_path.parent / _text(np.asarray(shard["source_episode_relative"]).item()) if "source_episode_relative" in shard else None
    for candidate in (original, preserved):
        if candidate is not None and candidate.is_file():
            with h5py.File(candidate, "r") as episode:
                if GROUP in episode:
                    return candidate
    if original.is_file():
        return original
    if preserved is not None and preserved.is_file():
        return preserved
    raise FileNotFoundError(f"No source HDF5 for detection labels in {shard_path}")


def load_detection_annotations(shard_path: Path, shard, sample_frames: np.ndarray,
                               rgb_shapes: dict[str, tuple[int, int, int]], *,
                               axis_aligned: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized [N,2,32,6] targets, valid [N,2,32], reviewed [N,2]."""
    if set(VIEWS) - set(rgb_shapes):
        raise ValueError("Detection training requires both front and side RGB")
    n = len(sample_frames)
    boxes = np.zeros((n, 2, MAX_BOXES, 4 if axis_aligned else 6), np.float32)
    valid = np.zeros((n, 2, MAX_BOXES), bool)
    reviewed = np.zeros((n, 2), bool)
    with h5py.File(_source_path(Path(shard_path), shard), "r") as episode:
        timestamps = np.asarray(episode["timestamps_ns"], dtype=np.int64)
        shard_timestamps = np.asarray(shard["timestamps_ns"], dtype=np.int64)
        if not np.array_equal(timestamps, shard_timestamps):
            raise ValueError(f"{shard_path}: source HDF5 timestamps differ from surface shard")
        if GROUP not in episode:
            return boxes, valid, reviewed
        labels = episode[GROUP]
        if _text(labels.attrs.get("schema", "")) not in (SCHEMA, PAIRED_SCHEMA) or _text(labels.attrs.get("annotation_image", "")) != "recorded_rgb":
            raise ValueError(f"{shard_path}: detection labels must use recorded RGB v2 coordinates")
        if (labels["reviewed"].shape != timestamps.shape or labels["frames_json"].shape != timestamps.shape
                or not np.array_equal(np.asarray(labels["timestamps_ns"], np.int64), timestamps)):
            raise ValueError(f"{shard_path}: detection labels are not aligned with the recorded frames")
        for sample, index in enumerate(sample_frames):
            index = int(index)
            if not bool(labels["reviewed"][index]):
                continue
            row = json.loads(_text(labels["frames_json"][index]))
            if (row.get("schema") not in (SCHEMA, PAIRED_SCHEMA) or row.get("annotation_image") != "recorded_rgb"
                    or row.get("frame_index") != index or row.get("timestamp_ns") != str(int(timestamps[index]))):
                raise ValueError(f"{shard_path}: stale detection label at frame {index}")
            entries = row.get("boxes", {})
            if set(entries) != set(VIEWS) or row.get("status") not in ("objects", "no_obstacles"):
                raise ValueError(f"{shard_path}: invalid detection status/views at frame {index}")
            ids = [{item["object_id"] for item in entries[view]} for view in VIEWS]
            if any(len(ids[view_index]) != len(entries[view]) for view_index, view in enumerate(VIEWS)):
                raise ValueError(f"{shard_path}: duplicate detection object ID at frame {index}")
            if ((row["status"] == "no_obstacles" and any(ids))
                    or (row["status"] == "objects" and (not ids[0] or ids[0] != ids[1]))):
                raise ValueError(f"{shard_path}: inconsistent detection IDs at frame {index}; single-view visibility requires the standalone OA-VLA paired trainer")
            reviewed[sample] = True
            for view_index, view in enumerate(VIEWS):
                height, width = rgb_shapes[view][:2]
                ratio = max(width / 224, height / 224)
                resized_h, resized_w = int(height / ratio), int(width / ratio)
                top, left = (224 - resized_h) // 2, (224 - resized_w) // 2
                if len(entries[view]) > MAX_BOXES:
                    raise ValueError(f"{shard_path}: more than {MAX_BOXES} boxes at frame {index}")
                for slot, item in enumerate(entries[view]):
                    values = np.asarray([item[key] for key in ("cx", "cy", "width", "height", "angle_deg")], np.float32)
                    if (not np.isfinite(values).all() or not 0 <= values[0] < width or not 0 <= values[1] < height
                            or values[2] < 1 or values[3] < 1):
                        raise ValueError(f"{shard_path}: invalid {view} OBB at frame {index}")
                    # (w,h,theta) and (h,w,theta+90°) describe the same OBB.
                    # Canonical labels avoid contradictory targets across frames.
                    if values[2] < values[3]:
                        values[2], values[3] = values[3], values[2]
                        values[4] += 90
                    angle = np.deg2rad(2 * values[4])
                    if axis_aligned:
                        # Enclose the annotated rotated rectangle in the
                        # original RGB plane, clip there, then letterbox it.
                        theta = np.deg2rad(values[4])
                        extent_x = (abs(np.cos(theta)) * values[2] + abs(np.sin(theta)) * values[3]) / 2
                        extent_y = (abs(np.sin(theta)) * values[2] + abs(np.cos(theta)) * values[3]) / 2
                        x0, x1 = np.clip([values[0] - extent_x, values[0] + extent_x], 0, width)
                        y0, y1 = np.clip([values[1] - extent_y, values[1] + extent_y], 0, height)
                        boxes[sample, view_index, slot] = [
                            ((x0 + x1) / 2 * resized_w / width + left) / 224,
                            ((y0 + y1) / 2 * resized_h / height + top) / 224,
                            (x1 - x0) * resized_w / width / 224,
                            (y1 - y0) * resized_h / height / 224,
                        ]
                    else:
                        boxes[sample, view_index, slot] = [
                            (values[0] * resized_w / width + left) / 224,
                            (values[1] * resized_h / height + top) / 224,
                            values[2] * resized_w / width / 224,
                            values[3] * resized_h / height / 224,
                            np.cos(angle), np.sin(angle),
                        ]
                    valid[sample, view_index, slot] = True
    return boxes, valid, reviewed
