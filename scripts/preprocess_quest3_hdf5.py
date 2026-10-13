#!/usr/bin/env python3
"""Convert Quest3 single/dual-arm HDF5 episodes into PI05 safety shards.

The default scene is the current left-base dual UR7e cell: a left Robotiq
2F-85 and a right Robotiq 2F-85.  Single-arm recordings emit only the recorded
left robot by default; ``--pad-inactive-right-arm`` gives them the deployed
12-D state / 14-D action interface while retaining supervision exclusively on
the genuinely observed left-arm surface.  Actions are stored per arm as six
joint deltas followed by one gripper target (7-D single / 14-D dual).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler  # noqa: E402
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Default: outputs/quest3_pi05/<episode>.npz")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument(
        "--task",
        default=None,
        help="Explicit language instruction override. Otherwise uses the episode task_description attribute, then project config.",
    )
    parser.add_argument("--points-per-link", type=int, default=None)
    parser.add_argument("--future-horizon", type=int, default=None)
    parser.add_argument("--action-source", choices=("commanded", "executed"), default=None)
    parser.add_argument(
        "--pad-inactive-right-arm",
        action="store_true",
        help=(
            "For a left-only episode, append an unobserved stationary right-arm "
            "state/action placeholder (zeros). Surface-flow labels remain left-only."
        ),
    )
    parser.add_argument("--rgb-view", action="append", default=[], help="RGB view to retain; repeat for multiple. Default follows project camera_to_model (front only).")
    parser.add_argument("--measured-flow-npz", type=Path, default=None, help="Optional CoTracker observation NPZ to merge as masked auxiliary targets.")
    parser.add_argument("--geometry-only", action="store_true", help="Create FK gate data for the CoTracker extraction stage; not measured training data.")
    parser.add_argument("--complete-windows-only", action="store_true", help="Legacy behavior: drop incomplete windows instead of padding with explicit masks.")
    parser.add_argument("--allow-unverified-right-tcp", action="store_true", help="Permit provisional right 2F-85 FK labels (never recommended for safety training).")
    parser.add_argument(
        "--left-gripper-position-convention",
        choices=("closing_fraction", "opening_fraction"),
        default="closing_fraction",
        help=(
            "Meaning of the recorded left observations/gripper_position. "
            "closing_fraction means 0=open,1=closed (current collector contract); "
            "opening_fraction means 1=open,0=closed (legacy recordings only)."
        ),
    )
    return parser.parse_args()


def _decode_rgb(dataset: h5py.Dataset) -> np.ndarray:
    if dataset.attrs.get("storage", "raw_lzf") != "jpeg":
        return np.asarray(dataset, dtype=np.uint8)
    shape = tuple(int(v) for v in dataset.attrs["original_shape"])
    frames = np.empty((len(dataset), *shape), dtype=np.uint8)
    for index, payload in enumerate(dataset):
        bgr = cv2.imdecode(np.asarray(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != shape:
            raise ValueError(f"Could not decode RGB frame {index} from {dataset.name}")
        frames[index] = bgr[..., ::-1]
    return frames


def _attribute_text(attrs: h5py.AttributeManager, key: str) -> str | None:
    """Return a non-empty HDF5 scalar attribute as normalized UTF-8 text."""
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    text = str(value).strip()
    return text or None


def _resolve_task_text(
    cli_task: str | None, attrs: h5py.AttributeManager, project_default: str
) -> tuple[str, str]:
    """Use explicit CLI text, then per-episode annotation, then legacy default."""
    if cli_task is not None:
        text = cli_task.strip()
        if not text:
            raise ValueError("--task must not be empty when supplied")
        return text, "cli_override"
    recorded = _attribute_text(attrs, "task_description")
    if recorded is not None:
        return recorded, "hdf5_root_attribute:task_description"
    fallback = project_default.strip()
    if not fallback:
        raise ValueError("Project preprocess.task_text must not be empty")
    return fallback, "project_config_fallback"


def _action_layout(arm_count: int) -> list[str]:
    joints = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    result: list[str] = []
    for arm in ("left", "right")[:arm_count]:
        result.extend(f"{arm}_{joint}_delta_rad" for joint in joints)
        result.append(f"{arm}_gripper_target")
    return result


def _pack_actions(joint_actions: np.ndarray, gripper_actions: np.ndarray, arm_count: int) -> np.ndarray:
    packed = np.empty((len(joint_actions), arm_count * 7), dtype=np.float32)
    for arm in range(arm_count):
        packed[:, arm * 7 : arm * 7 + 6] = joint_actions[:, arm * 6 : arm * 6 + 6]
        packed[:, arm * 7 + 6] = gripper_actions[:, arm]
    return packed


def _window_starts(valid: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray(
        [index for index in range(len(valid) - horizon) if bool(np.asarray(valid[index : index + horizon]).all())],
        dtype=np.int64,
    )


def _merge_visual(path: Path, *, episode_path: Path, sample_indices: np.ndarray, horizon: int, frame_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observed = np.asarray(data["observed_points"], dtype=np.float32)
        visible = np.asarray(data["observed_visible"], dtype=bool)
        source = str(np.asarray(data["source_episode"]).item())
        depth_provenance = {
            f"visual_{key}": np.asarray(data[key]).copy()
            for key in ("depth_alignment_metadata", "depth_unit_m", "intrinsics", "camera_to_world", "distortion", "distortion_model")
            if key in data
        }
    if observed.shape[:2] != visible.shape or observed.shape[0] != frame_count or observed.shape[-1] != 3:
        raise ValueError(f"Measured point flow {path} does not align with the HDF5 episode")
    if Path(source).resolve() != episode_path.resolve():
        raise ValueError(f"Measured point flow belongs to {source}, not {episode_path}")
    # Model inputs must be finite.  Forward-fill each stable tracker ID using
    # only current/past measurements; leading gaps remain zero and masked.
    filled = np.zeros_like(observed)
    last = np.zeros((observed.shape[1], 3), dtype=np.float32)
    have = np.zeros((observed.shape[1],), dtype=bool)
    for frame in range(frame_count):
        valid = visible[frame] & np.isfinite(observed[frame]).all(axis=-1)
        last[valid] = observed[frame, valid]
        have |= valid
        filled[frame, have] = last[have]
    current = filled[sample_indices]
    current_visible = visible[sample_indices] & np.isfinite(observed[sample_indices]).all(axis=-1)
    offsets = np.zeros((len(sample_indices), horizon, observed.shape[1], 3), dtype=np.float32)
    mask = np.zeros(offsets.shape[:-1], dtype=bool)
    for row, start in enumerate(sample_indices):
        future_ids = start + 1 + np.arange(horizon)
        inside = future_ids < frame_count
        future_ids = np.minimum(future_ids, frame_count - 1)
        future = observed[future_ids]
        valid = current_visible[row][None, :] & visible[future_ids] & inside[:, None]
        valid &= np.isfinite(future).all(axis=-1)
        offsets[row] = np.where(valid[..., None], future - current[row][None, ...], 0.0)
        mask[row] = valid
    return {
        **depth_provenance,
        "current_surface": current,
        "surface_chunk": current[:, None] + offsets,
        "current_surface_mask": current_visible,
        "visual_robot_current_points": current,
        "visual_robot_future_offsets": offsets,
        "visual_robot_flow_supervision_mask": mask,
        "visual_point_source": np.asarray("measured_depth_cotracker_masked"),
        "visual_source_episode": np.asarray(source),
    }


def main() -> None:
    args = parse_args()
    if args.measured_flow_npz is None and not args.geometry_only:
        raise ValueError("Surface Action training requires --measured-flow-npz; use --geometry-only to prepare the FK gate first")
    config, config_path = load_project_config(args.project_config)
    defaults = config["preprocess"]
    points_per_link = int(args.points_per_link or defaults["points_per_link"])
    horizon = int(args.future_horizon or defaults["future_horizon"])
    action_source = args.action_source or str(defaults["action_source"])
    requested_rgb = tuple(args.rgb_view or config["quest3"]["camera_to_model"].keys())
    if points_per_link < 2 or horizon < 1:
        raise ValueError("points-per-link must be >=2 and future-horizon must be positive")
    episode_path = args.episode.expanduser().resolve()
    output = args.output or (REPO_ROOT / "outputs" / "quest3_pi05" / f"{episode_path.stem}.shard")
    with h5py.File(episode_path, "r") as source:
        task, task_source = _resolve_task_text(args.task, source.attrs, str(defaults["task_text"]))
        required = ("observations/qpos", "observations/gripper_position", "action", "executed_action", "gripper_action", "action_valid", "timestamps_ns")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{episode_path} is missing {missing}")
        qpos = np.asarray(source["observations/qpos"], dtype=np.float32)
        gripper_position = np.asarray(source["observations/gripper_position"], dtype=np.float32)
        commanded = np.asarray(source["action"], dtype=np.float32)
        executed = np.asarray(source["executed_action"], dtype=np.float32)
        gripper_action = np.asarray(source["gripper_action"], dtype=np.float32)
        action_valid = np.asarray(source["action_valid"], dtype=bool)
        timestamps_ns = np.asarray(source["timestamps_ns"], dtype=np.int64)
        missing_rgb = [name for name in requested_rgb if f"observations/images/{name}/rgb" not in source]
        if missing_rgb:
            raise KeyError(f"Episode does not contain requested RGB views: {missing_rgb}")
        rgb = {name: _decode_rgb(source[f"observations/images/{name}/rgb"]) for name in requested_rgb}
        source_mode = str(source.attrs.get("mode", "unknown"))
        source_attrs = {
            str(key): str(value)
            for key, value in source.attrs.items()
            if key in ("mode", "created_utc", "action_definition", "executed_action_definition", "task_description")
        }
    frame_count, joint_dim = qpos.shape
    if joint_dim not in (6, 12):
        raise ValueError(f"Quest3 qpos must have 6 or 12 columns, got {qpos.shape}")
    recorded_arm_count = joint_dim // 6
    arm_count = recorded_arm_count
    expected = {
        "gripper_position": (frame_count, recorded_arm_count), "commanded action": (frame_count, joint_dim),
        "executed action": (frame_count, joint_dim), "gripper action": (frame_count, recorded_arm_count),
    }
    actual = {"gripper_position": gripper_position.shape, "commanded action": commanded.shape, "executed action": executed.shape, "gripper action": gripper_action.shape}
    bad = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if bad or action_valid.shape != (frame_count,) or timestamps_ns.shape != (frame_count,):
        raise ValueError(f"Misaligned Quest3 arrays: {bad}")
    if not rgb or any(frames.shape[0] != frame_count for frames in rgb.values()):
        raise ValueError("RGB arrays are absent or not aligned to robot state")
    joint_actions = commanded if action_source == "commanded" else executed
    if args.pad_inactive_right_arm and recorded_arm_count != 1:
        raise ValueError("--pad-inactive-right-arm is only valid for a left-only six-joint episode")
    inactive_right_arm = bool(args.pad_inactive_right_arm and recorded_arm_count == 1)
    if inactive_right_arm:
        # Do not fabricate collision points for this unknown physical pose.  The
        # placeholder solely makes state/action shapes compatible with dual-arm
        # deployment; its zero action represents a stationary, uncommanded arm.
        qpos = np.concatenate((qpos, np.zeros_like(qpos)), axis=1)
        gripper_position = np.concatenate((gripper_position, np.zeros_like(gripper_position)), axis=1)
        joint_actions = np.concatenate((joint_actions, np.zeros_like(joint_actions)), axis=1)
        gripper_action = np.concatenate((gripper_action, np.zeros_like(gripper_action)), axis=1)
        arm_count = 2
    actions = _pack_actions(joint_actions, gripper_action, arm_count)
    # The collision model always consumes a physical closing fraction: finger
    # angle 0 rad is open and +0.8 rad is closed in the vendor 2F-85 Xacro.
    # Keep the recorder value untouched in ``gripper_position`` for audit and
    # action replay, but explicitly convert only the mesh/FK input.  Some
    # historical Quest3 recordings predate the 0=open/1=closed contract and
    # stored the complementary opening fraction.
    surface_gripper_position = gripper_position.copy()
    if args.left_gripper_position_convention == "opening_fraction":
        surface_gripper_position[:, 0] = 1.0 - surface_gripper_position[:, 0]
    sampler = DualUR7eSurfacePointSampler(points_per_link=points_per_link, project_config=config_path)
    if arm_count == 2 and not sampler.tcp_transform_verified["right_arm"] and not args.allow_unverified_right_tcp:
        raise RuntimeError(
            "Right flange-to-active-TCP is not verified. Reconnect the right UR7e and record its read-only RTDE transform, "
            "or use --allow-unverified-right-tcp only for non-safety pipeline tests."
        )
    surface_arm_count = recorded_arm_count
    link_names = sampler.link_names(surface_arm_count)
    fixed = np.empty((frame_count, len(link_names), points_per_link, 3), dtype=np.float32)
    for frame in range(frame_count):
        fixed[frame] = sampler.link_points(
            qpos[frame, : recorded_arm_count * 6], surface_gripper_position[frame, :recorded_arm_count]
        )
    sample_indices = _window_starts(action_valid, horizon) if args.complete_windows_only else np.arange(frame_count)
    if not len(sample_indices):
        raise ValueError("Episode has no complete valid action/flow window")
    flat = fixed.reshape(frame_count, -1, 3)
    action_ids = sample_indices[:, None] + np.arange(horizon)[None, :]
    future_ids = np.minimum(action_ids + 1, frame_count - 1)
    step_mask = (action_ids + 1 < frame_count) & action_valid[np.minimum(action_ids, frame_count - 1)]
    chunks = np.where(step_mask[..., None], actions[np.minimum(action_ids, frame_count - 1)], 0.)
    offsets = flat[future_ids] - flat[sample_indices, None]
    dt = (timestamps_ns[future_ids] - timestamps_ns[np.minimum(action_ids, frame_count - 1)]) / 1e9
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(3, dtype=np.int32),
        "schema": np.asarray("quest3_pi05_surface_flow"),
        "project_config": np.asarray(str(config_path)),
        "source_episode": np.asarray(str(episode_path)),
        "source_mode": np.asarray(source_mode),
        "source_attrs": np.asarray(json.dumps(source_attrs, ensure_ascii=False)),
        "coordinate_frame": np.asarray("left_base"),
        "task_text": np.asarray(task),
        "task_text_source": np.asarray(task_source),
        "arm_count": np.asarray(arm_count, dtype=np.int32),
        "arm_names": np.asarray(("left_arm", "right_arm")[:arm_count]),
        "surface_arm_count": np.asarray(surface_arm_count, dtype=np.int32),
        "inactive_arm_names": np.asarray(("right_arm",) if inactive_right_arm else ()),
        "inactive_right_arm_placeholder": np.asarray(inactive_right_arm),
        "qpos": qpos,
        "gripper_position": gripper_position,
        "surface_gripper_position": surface_gripper_position,
        "left_gripper_position_convention": np.asarray(args.left_gripper_position_convention),
        "surface_gripper_position_semantics": np.asarray("closing_fraction_0_open_1_closed"),
        "actions": actions,
        "action_valid": action_valid,
        "action_source": np.asarray(action_source),
        "action_layout": np.asarray(_action_layout(arm_count)),
        "action_dim": np.asarray(arm_count * 7, dtype=np.int32),
        "timestamps_ns": timestamps_ns,
        "sample_frame_indices": sample_indices,
        "action_chunks": chunks.astype(np.float32),
        "action_chunk": chunks.astype(np.float32),
        "commanded_actions": commanded,
        "executed_actions": executed,
        "action_dt_s": dt.astype(np.float32),
        "action_step_mask": step_mask,
        "surface_source": np.asarray("mesh_gate_only" if args.geometry_only else "measured_depth_cotracker"),
        "fixed_link_points": fixed,
        "current_link_points": flat[sample_indices],
        "target_point_offsets": offsets.astype(np.float32),
        "target_point_mask": np.broadcast_to(step_mask[..., None], offsets.shape[:-1]),
        "link_names": np.asarray(link_names),
        "point_ids": sampler.point_ids(surface_arm_count),
        "points_per_link": np.asarray(points_per_link, dtype=np.int32),
        "point_identity_version": np.asarray(sampler.point_identity_version),
        "surface_model_hash": np.asarray(sampler.model_hash(surface_arm_count)),
        "right_base_to_left_base": sampler.right_base_to_left_base,
    }
    for name, frames in rgb.items():
        payload[f"rgb_{name}"] = frames
    if args.measured_flow_npz is not None:
        payload.update(_merge_visual(args.measured_flow_npz, episode_path=episode_path, sample_indices=sample_indices, horizon=horizon, frame_count=frame_count))
        payload["visual_robot_flow_supervision_mask"] &= step_mask[..., None]
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    original_copy = output.with_suffix(".source.hdf5")
    if original_copy.exists():
        raise FileExistsError(f"Preserved source already exists: {original_copy}; choose a new output path")
    shutil.copy2(episode_path, original_copy)
    payload["source_episode_relative"] = np.asarray(original_copy.name)
    if output.suffix == ".npz":
        np.savez_compressed(output, **payload)
    elif output.suffix == ".shard":
        output.mkdir(exist_ok=False)
        for name, value in payload.items():
            np.save(output / f"{name}.npy", value, allow_pickle=False)
    else:
        raise ValueError("Output must end in .shard (mmap) or .npz (legacy)")
    print(json.dumps({
        "output": str(output), "frames": frame_count, "samples": len(sample_indices), "arms": arm_count,
        "action_dim": arm_count * 7, "action_source": action_source, "links": len(link_names),
        "fixed_points": flat.shape[1], "rgb_views": sorted(rgb), "measured_flow": args.measured_flow_npz is not None,
        "task_text": task, "task_source": task_source, "inactive_right_arm_placeholder": inactive_right_arm,
        "left_gripper_position_convention": args.left_gripper_position_convention,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
