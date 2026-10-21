"""Recorded RGB annotation validation and ID-aligned paired OBB loading."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SCHEMA = "dual_rgb_paired_obb_v3"
LEGACY_SCHEMA = "dual_rgb_obb_annotation_v2"
GROUP = "annotations/obstacle_obbs"
VIEWS = ("front", "side")
MAX_BOXES = 32


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_frame(boxes, status, rgb_shapes, visibility=None):
    """Validate complete reviewed frames; explicit false is confirmed invisibility.

    Omitting visibility accepts legacy v2 only: identical IDs must occur once in
    each view. A missing box is never silently interpreted as an invisible box.
    """
    if not isinstance(boxes, dict) or set(boxes) != set(VIEWS):
        raise ValueError("Boxes must contain exactly front and side")
    result, ids = {}, {}
    for view in VIEWS:
        entries = boxes[view]
        if not isinstance(entries, list) or len(entries) > MAX_BOXES:
            raise ValueError("At most 32 OBBs per view")
        result[view], ids[view] = [], set()
        for item in entries:
            name = item.get("object_id") if isinstance(item, dict) else None
            if not isinstance(name, str) or not name.strip() or len(name.strip()) > 64:
                raise ValueError("Each OBB requires a nonempty object_id of at most 64 characters")
            name = name.strip()
            if name in ids[view]:
                raise ValueError(f"Duplicate object_id {name!r} in {view}; use one OBB per object per view")
            try:
                values = [float(item[key]) for key in ("cx", "cy", "width", "height", "angle_deg")]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each OBB requires cx,cy,width,height,angle_deg") from exc
            height, width = rgb_shapes[view][:2]
            if (not np.isfinite(values).all() or not 0 <= values[0] < width or not 0 <= values[1] < height
                    or min(values[2:4]) < 1):
                raise ValueError(f"Invalid {view} OBB")
            result[view].append(dict(object_id=name, **dict(zip(("cx", "cy", "width", "height", "angle_deg"), values))))
            ids[view].add(name)
    objects = ids["front"] | ids["side"]
    if len(objects) > MAX_BOXES:
        raise ValueError("At most 32 object pairs per frame")
    if visibility is None:
        if ids["front"] != ids["side"]:
            raise ValueError("Use matching IDs in both views or explicitly declare per-view visibility")
        visibility = {name: {view: True for view in VIEWS} for name in objects}
    if not isinstance(visibility, dict) or set(visibility) != objects:
        raise ValueError("Visibility must describe every object with at least one visible OBB")
    for name, flags in visibility.items():
        if (not isinstance(flags, dict) or set(flags) != set(VIEWS)
                or any(type(flags[view]) is not bool for view in VIEWS)
                or not any(flags.values())
                or any(flags[view] != (name in ids[view]) for view in VIEWS)):
            raise ValueError(f"Visibility disagrees with visible OBBs for {name!r}")
    if status not in ("objects", "no_obstacles") or (status == "objects") != bool(objects):
        raise ValueError("Use objects for annotated objects or explicitly mark no_obstacles")
    return result, visibility


def _source_path(shard_path, shard):
    import h5py
    original = Path(_text(np.asarray(shard["source_episode"]).item())).expanduser()
    preserved = shard_path.parent / _text(np.asarray(shard["source_episode_relative"]).item()) if "source_episode_relative" in shard else None
    for candidate in (original, preserved):
        if candidate is not None and candidate.is_file():
            with h5py.File(candidate, "r") as episode:
                if GROUP in episode:
                    return candidate
    for candidate in (original, preserved):
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No source HDF5 for detection labels in {shard_path}")


def load_detection_annotations(shard_path, shard, sample_frames, rgb_shapes):
    """Return ID-aligned boxes[N,2,32,6], visible, reviewed, object_ids[N,32]."""
    import h5py
    from .geometry import pixel_obb_to_normalized
    if set(VIEWS) - set(rgb_shapes):
        raise ValueError("Detection training requires both front and side RGB")
    n = len(sample_frames)
    boxes = np.zeros((n, 2, MAX_BOXES, 6), np.float32)
    valid = np.zeros((n, 2, MAX_BOXES), bool)
    reviewed = np.zeros((n, 2), bool)
    object_ids = np.full((n, MAX_BOXES), "", dtype="U64")
    with h5py.File(_source_path(Path(shard_path), shard), "r") as episode:
        timestamps = np.asarray(episode["timestamps_ns"], np.int64)
        if not np.array_equal(timestamps, np.asarray(shard["timestamps_ns"], np.int64)):
            raise ValueError(f"{shard_path}: source HDF5 timestamps differ from RGB/action shard")
        if GROUP not in episode:
            return boxes, valid, reviewed, object_ids
        labels = episode[GROUP]
        if (_text(labels.attrs.get("schema", "")) not in (SCHEMA, LEGACY_SCHEMA)
                or _text(labels.attrs.get("annotation_image", "")) != "recorded_rgb"):
            raise ValueError(f"{shard_path}: labels must use recorded RGB v2/v3 coordinates")
        if (labels["reviewed"].shape != timestamps.shape or labels["frames_json"].shape != timestamps.shape
                or not np.array_equal(np.asarray(labels["timestamps_ns"], np.int64), timestamps)):
            raise ValueError(f"{shard_path}: labels are not aligned with recorded frames")
        for sample, index in enumerate(sample_frames):
            index = int(index)
            if not bool(labels["reviewed"][index]):
                continue
            row = json.loads(_text(labels["frames_json"][index]))
            if (row.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or row.get("annotation_image") != "recorded_rgb"
                    or row.get("frame_index") != index or row.get("timestamp_ns") != str(int(timestamps[index]))):
                raise ValueError(f"{shard_path}: stale detection label at frame {index}")
            if row["schema"] == SCHEMA and not isinstance(row.get("visibility"), dict):
                raise ValueError("Paired v3 labels require explicit visibility")
            entries, visibility = validate_frame(row.get("boxes"), row.get("status"), rgb_shapes,
                                                  row.get("visibility") if row["schema"] == SCHEMA else None)
            # Both views use ONE object ordering, independent of drawing order.
            names = sorted(visibility)
            slots = {name: slot for slot, name in enumerate(names)}
            object_ids[sample, :len(names)] = names
            reviewed[sample] = True
            for view_index, view in enumerate(VIEWS):
                for item in entries[view]:
                    slot = slots[item["object_id"]]
                    boxes[sample, view_index, slot] = pixel_obb_to_normalized(
                        [item[key] for key in ("cx", "cy", "width", "height", "angle_deg")], *rgb_shapes[view][:2])
                    valid[sample, view_index, slot] = True
    return boxes, valid, reviewed, object_ids
