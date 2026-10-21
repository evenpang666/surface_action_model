#!/usr/bin/env python3
"""Convert recorded four-view RGB and actions to SAM 3 VLA shards; no depth/FK/points."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_module.policy_contract import POLICY_VIEWS
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config
from scripts.preprocess_quest3_hdf5 import _decode_rgb, _pack_actions, _action_layout, _resolve_task_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .shard directory or .npz file")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--future-horizon", type=int)
    parser.add_argument("--action-source", choices=("commanded", "executed"))
    parser.add_argument("--task")
    args = parser.parse_args()
    config, _ = load_project_config(args.project_config)
    horizon = args.future_horizon if args.future_horizon is not None else int(config["preprocess"]["future_horizon"])
    if horizon < 1:
        raise ValueError("future-horizon must be positive")
    output, source_path = args.output.expanduser().resolve(), args.episode.expanduser().resolve()
    if output.suffix not in (".shard", ".npz") or output.exists():
        raise ValueError("Output must be a new .shard directory or .npz file")
    preserved = output.with_suffix(".source.hdf5")
    if preserved.exists() or preserved == source_path:
        raise FileExistsError(f"Preserved source already exists: {preserved}")
    action_source = args.action_source or config["preprocess"]["action_source"]
    with h5py.File(source_path, "r") as source:
        qpos = np.asarray(source["observations/qpos"], np.float32)
        joint_actions = np.asarray(source["action" if action_source == "commanded" else "executed_action"], np.float32)
        gripper = np.asarray(source["gripper_action"], np.float32)
        valid = np.asarray(source["action_valid"], bool)
        timestamps = np.asarray(source["timestamps_ns"], np.int64)
        if qpos.ndim != 2 or qpos.shape[1] not in (6, 12) or len(qpos) < 2:
            raise ValueError("qpos must have at least two rows and 6 or 12 joints")
        count, joint_dim = qpos.shape
        arms = joint_dim // 6
        if (joint_actions.shape != qpos.shape or gripper.shape != (count, arms)
                or valid.shape != (count,) or timestamps.shape != (count,)):
            raise ValueError("Robot state, actions and timestamps are not frame-aligned")
        if not np.isfinite(qpos).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("qpos must be finite and timestamps strictly increasing")
        missing = [view for view in POLICY_VIEWS if f"observations/images/{view}/rgb" not in source]
        if missing:
            raise ValueError(f"Recording lacks policy RGB views: {missing}")
        rgb = {view: _decode_rgb(source[f"observations/images/{view}/rgb"]) for view in POLICY_VIEWS}
        if any(value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3 for value in rgb.values()):
            raise ValueError("All RGB cameras must be aligned to the robot/timestamp frame axis")
        task, task_source = _resolve_task_text(args.task, source.attrs, config["preprocess"]["task_text"])
    actions = _pack_actions(joint_actions, gripper, arms)
    if not np.isfinite(actions[valid]).all():
        raise ValueError("Valid recorded actions must be finite")
    frames = np.arange(count, dtype=np.int64)
    ids = frames[:, None] + np.arange(horizon)[None]
    clamped = np.minimum(ids, count - 1)
    mask = (ids + 1 < count) & valid[clamped]
    payload = {
        "schema": np.asarray("quest3_pi05_sam3_rgb_action"), "format_version": np.asarray(5),
        "coordinate_frame": np.asarray("left_base"), "source_episode": np.asarray(str(source_path)),
        "source_episode_relative": np.asarray(preserved.name), "task_text": np.asarray(task),
        "task_text_source": np.asarray(task_source), "qpos": qpos, "timestamps_ns": timestamps,
        "sample_frame_indices": frames, "action_chunks": np.where(mask[..., None], actions[clamped], 0).astype(np.float32),
        "action_step_mask": mask, "action_layout": np.asarray(_action_layout(arms)),
        "action_source": np.asarray(action_source),
        "action_dt_s": ((timestamps[np.minimum(ids + 1, count - 1)] - timestamps[clamped]) / 1e9).astype(np.float32),
        **{f"rgb_{view}": value for view, value in rgb.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, preserved)
    if output.suffix == ".npz":
        np.savez_compressed(output, **payload)
    else:
        output.mkdir(exist_ok=False)
        for key, value in payload.items():
            np.save(output / f"{key}.npy", value, allow_pickle=False)
    print(json.dumps({"output": str(output), "frames": count, "action_horizon": horizon,
                      "policy_views": POLICY_VIEWS, "point_output": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
