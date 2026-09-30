#!/usr/bin/env python3
"""Offline PI05 point-flow validation on a preprocessed Quest3 episode.

The video compares, in the left-base frame, the selected robot surface points
at the current instant (gray), their PI05 predicted future locations (cyan),
and the recorded FK future locations (red).  It is deliberately offline: it
never connects to either robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "openpi" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True, help="One preprocessed Quest3 .npz shard.")
    parser.add_argument("--output", type=Path, required=True, help="MP4 comparison video.")
    parser.add_argument("--predictions-output", type=Path, default=None, help="Optional .npz containing predictions and metrics.")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto (CUDA is strongly recommended).")
    parser.add_argument("--num-steps", type=int, default=10, help="Euler flow-sampling steps per context.")
    parser.add_argument("--max-contexts", type=int, default=24, help="Evenly spaced observation contexts to evaluate.")
    parser.add_argument("--start", type=int, default=0, help="First shard sample index eligible for evaluation.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0, help="Makes stochastic action-flow sampling repeatable.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow very slow CPU inference when CUDA is unavailable.")
    parser.add_argument(
        "--front-calibration",
        type=Path,
        default=REPO_ROOT.parent / "quest3_collect" / "config" / "calibration" / "left_base_to_front_camera.yaml",
        help="Calibrated ^left_baseT_front_camera YAML used to project points onto recorded front RGB.",
    )
    parser.add_argument(
        "--allow-surface-layout-mismatch",
        action="store_true",
        help="Adapt a legacy checkpoint to a newer mesh layout for visualization only; never treats its RMSE as a formal metric.",
    )
    return parser.parse_args()


def _choose_indices(count: int, start: int, maximum: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("--max-contexts must be positive")
    if not 0 <= start < count:
        raise ValueError(f"--start must be in [0, {count}), got {start}")
    return np.unique(np.linspace(start, count - 1, min(maximum, count - start), dtype=np.int64))


def _project_top_down(points: np.ndarray, *, center: np.ndarray, scale: float, origin: tuple[int, int]) -> np.ndarray:
    """Orthographic left-base XY projection; +X right, +Y up in the panel."""
    xy = np.asarray(points, dtype=np.float32)[..., :2]
    projected = np.empty((len(xy), 2), dtype=np.int32)
    projected[:, 0] = np.rint(origin[0] + (xy[:, 0] - center[0]) * scale).astype(np.int32)
    projected[:, 1] = np.rint(origin[1] - (xy[:, 1] - center[1]) * scale).astype(np.int32)
    return projected


def _draw_points(frame: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    height, width = frame.shape[:2]
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    for x, y in pixels[valid]:
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _recorded_intrinsics(calibration_path: Path, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return recorder-adjusted K and ^left_baseT_camera for an RGB frame."""
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Front-camera calibration is absent: {calibration_path}")
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    intrinsics = payload.get("intrinsics", {})
    source_width, source_height = int(intrinsics["width"]), int(intrinsics["height"])
    K = np.asarray(((intrinsics["fx"], 0.0, intrinsics["ppx"]), (0.0, intrinsics["fy"], intrinsics["ppy"]), (0.0, 0.0, 1.0)), dtype=np.float64)
    source_ratio, target_ratio = source_width / source_height, width / height
    if source_ratio > target_ratio:
        crop_width = source_height * target_ratio
        K[0, 2] -= (source_width - crop_width) * 0.5
        scale_x = scale_y = width / crop_width
    else:
        crop_height = source_width / target_ratio
        K[1, 2] -= (source_height - crop_height) * 0.5
        scale_x = scale_y = height / crop_height
    K[0] *= scale_x
    K[1] *= scale_y
    camera_to_left_base = np.asarray(payload["matrix"], dtype=np.float64)
    if camera_to_left_base.shape != (4, 4):
        raise ValueError(f"Invalid 4x4 calibration matrix in {calibration_path}")
    return K, camera_to_left_base


def _project_front(points: np.ndarray, *, K: np.ndarray, camera_to_left_base: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project left-base points to integer front-RGB pixels; invalid rows are -1."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hom = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (np.linalg.inv(camera_to_left_base) @ hom.T).T[:, :3]
    z = camera[:, 2]
    pixels = np.full((len(points), 2), -1, dtype=np.int32)
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-5)
    u = np.rint(K[0, 0] * camera[:, 0] / np.maximum(z, 1e-5) + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * camera[:, 1] / np.maximum(z, 1e-5) + K[1, 2]).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixels[valid] = np.stack((u[valid], v[valid]), axis=1)
    return pixels


def _overlay_front_points(rgb: np.ndarray, current: np.ndarray, predicted: np.ndarray, target: np.ndarray, *, K: np.ndarray, camera_to_left_base: np.ndarray) -> np.ndarray:
    """Draw current / recorded future / predicted future points on the RGB image."""
    image = np.ascontiguousarray(np.asarray(rgb)[..., ::-1].copy())
    height, width = image.shape[:2]
    for points, color, radius in (
        (current, (150, 150, 150), 2),       # gray: current points
        (target, (50, 50, 255), 3),          # red: recorded future points
        (predicted, (255, 230, 30), 3),      # cyan: PI05 predicted future points
    ):
        _draw_points(image, _project_front(points, K=K, camera_to_left_base=camera_to_left_base, width=width, height=height), color, radius)
    cv2.rectangle(image, (8, 8), (337, 54), (0, 0, 0), thickness=-1)
    cv2.putText(image, "gray=current  red=recorded future", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(image, "cyan=PI05 predicted future", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 230, 255), 1, cv2.LINE_AA)
    return image


def _render_frame(
    rgb: np.ndarray,
    current: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    context: int,
    source_frame: int,
    horizon_index: int,
    horizon: int,
    rmse_mm: float,
    task: str,
    compatibility_label: str,
    K: np.ndarray,
    camera_to_left_base: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    left_width = width // 2
    camera_overlay = _overlay_front_points(rgb, current, predicted, target, K=K, camera_to_left_base=camera_to_left_base)
    camera = cv2.resize(camera_overlay, (left_width, height - 58), interpolation=cv2.INTER_AREA)
    frame[58:, :left_width] = camera
    cv2.putText(frame, "Recorded front RGB", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

    panel_x0, panel_y0 = left_width, 58
    panel = frame[panel_y0:, panel_x0:]
    panel[:] = (15, 15, 15)
    all_points = np.concatenate((current, predicted, target), axis=0)
    low, high = np.quantile(all_points[:, :2], (0.01, 0.99), axis=0)
    center = (low + high) * 0.5
    span = max(float(np.max(high - low)), 0.30)
    scale = min((panel.shape[1] - 72) / span, (panel.shape[0] - 110) / span)
    origin = (panel.shape[1] // 2, panel.shape[0] // 2 + 30)
    # Coordinates and 10-cm scale marker make the geometric units visible.
    cv2.line(panel, (36, panel.shape[0] - 42), (136, panel.shape[0] - 42), (170, 170, 170), 2)
    cv2.putText(panel, "10 cm", (45, panel.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.arrowedLine(panel, (38, 35), (96, 35), (120, 120, 120), 1, tipLength=0.12)
    cv2.arrowedLine(panel, (38, 35), (38, 93), (120, 120, 120), 1, tipLength=0.12)
    cv2.putText(panel, "+X", (98, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "+Y", (20, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 170, 170), 1, cv2.LINE_AA)
    _draw_points(panel, _project_top_down(current, center=center, scale=scale, origin=origin), (120, 120, 120), 1)
    _draw_points(panel, _project_top_down(target, center=center, scale=scale, origin=origin), (70, 70, 255), 2)
    _draw_points(panel, _project_top_down(predicted, center=center, scale=scale, origin=origin), (255, 230, 30), 2)
    cv2.putText(frame, "Left-base top view: current / ground truth / PI05", (panel_x0 + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, f"context {context}  source frame {source_frame}  lookahead {horizon_index + 1}/{horizon}", (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, f"point RMSE: {rmse_mm:.2f} mm", (panel_x0 + 18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"task: {task}", (18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    if compatibility_label:
        cv2.putText(frame, compatibility_label, (panel_x0 + 18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 185, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    if args.num_steps < 1 or args.fps < 1 or args.width < 640 or args.height < 360:
        raise ValueError("num-steps/fps must be positive and video size must be at least 640x360")
    if not args.checkpoint.is_file() or not args.episode.exists():
        raise FileNotFoundError("--checkpoint and --episode must both be existing files")
    import torch
    from scripts.serve_quest3_pi05_safety import Quest3JointSafetyPolicy

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA was not selected/detected. PI05 CPU inference is extremely slow; pass --allow-cpu only for a tiny smoke evaluation.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but PyTorch cannot access a CUDA device")

    from safety_module.shards import open_shard
    policy = Quest3JointSafetyPolicy(args.checkpoint, device_name=device, num_steps=args.num_steps, tokenizer_model=None)
    with open_shard(args.episode) as data:
        required = ("qpos", "sample_frame_indices", "current_link_points", "target_point_offsets", "target_point_mask", "rgb_front", "task_text")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{args.episode} is missing {missing}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        sample_frames = np.asarray(data["sample_frame_indices"], dtype=np.int64)
        current_points = np.asarray(data["current_link_points"], dtype=np.float32)
        target_offsets = np.asarray(data["target_point_offsets"], dtype=np.float32)
        target_mask = np.asarray(data["target_point_mask"], dtype=bool)
        current_mask = np.ones(current_points.shape[:2], dtype=bool)
        if policy.measured_surface:
            current_points = np.asarray(data["visual_robot_current_points"], dtype=np.float32)
            target_offsets = np.asarray(data["visual_robot_future_offsets"], dtype=np.float32)
            target_mask = np.asarray(data["visual_robot_flow_supervision_mask"], dtype=bool)
            current_mask = np.asarray(data["current_surface_mask"], dtype=bool)
        rgb_front = np.asarray(data["rgb_front"], dtype=np.uint8)
        task = str(np.asarray(data["task_text"]).item())
    K, camera_to_left_base = _recorded_intrinsics(args.front_calibration, width=rgb_front.shape[2], height=rgb_front.shape[1])
    contexts = _choose_indices(len(current_points), args.start, args.max_contexts)
    shard_hash = ""
    with open_shard(args.episode) as metadata_data:
        shard_hash = str(np.asarray(metadata_data["surface_model_hash"]).item()) if "surface_model_hash" in metadata_data else ""
    checkpoint_hash = str(policy.info.get("surface_model_hash", ""))
    checkpoint_points = len(policy.point_indices)
    input_indices: np.ndarray | None = None
    compatibility_label = ""
    if (shard_hash and checkpoint_hash and shard_hash != checkpoint_hash) or (not policy.measured_surface and current_points.shape[1] <= int(policy.point_indices.max(initial=-1))):
        if not args.allow_surface_layout_mismatch:
            raise ValueError(
                "Checkpoint and shard use different robot-surface point identities. "
                "Use a matching shard/checkpoint for quantitative validation, or explicitly pass "
                "--allow-surface-layout-mismatch for a visualization-only compatibility preview."
            )
        source_points_per_link = int(current_points.shape[1] // 17)
        if current_points.shape[1] % 17 or checkpoint_points % 17:
            raise ValueError("Cannot safely adapt surface layouts: expected the 17-link left UR7e+2F85 topology")
        desired_points_per_link = checkpoint_points // 17
        if desired_points_per_link > source_points_per_link:
            raise ValueError("Legacy checkpoint requires more points per link than the shard provides")
        local = np.linspace(0, source_points_per_link - 1, desired_points_per_link, dtype=np.int64)
        input_indices = (np.arange(17, dtype=np.int64)[:, None] * source_points_per_link + local[None, :]).reshape(-1)
        compatibility_label = "LAYOUT-ADAPTED PREVIEW — NOT A FORMAL METRIC"
        print(
            f"[compatibility] adapting {source_points_per_link} -> {desired_points_per_link} points/link "
            f"(shard_hash={shard_hash[:12]}, checkpoint_hash={checkpoint_hash[:12]})",
            flush=True,
        )
    elif current_points.shape[1] <= int(policy.point_indices.max(initial=-1)):
        raise ValueError("Episode surface point layout is smaller than the checkpoint's selected-point indices")
    if target_offsets.shape[1] != int(policy.info["action_horizon"]):
        raise ValueError("Episode horizon differs from checkpoint action_horizon")

    selected_current, predicted, target, masks, predicted_actions = [], [], [], [], []
    for ordinal, index in enumerate(contexts):
        torch.manual_seed(args.seed + int(index))
        source_frame = int(sample_frames[index])
        qpos_input = qpos[source_frame]
        if qpos_input.shape[0] != policy.joint_dim:
            if qpos_input.shape[0] > policy.joint_dim and args.allow_surface_layout_mismatch:
                qpos_input = qpos_input[: policy.joint_dim]
            else:
                raise ValueError(f"Checkpoint expects {policy.joint_dim} qpos values but shard provides {qpos_input.shape[0]}")
        points_input = current_points[index] if input_indices is None else current_points[index][input_indices]
        valid_ids = np.flatnonzero(current_mask[index]) if policy.measured_surface else np.arange(len(points_input))
        if policy.measured_surface:
            if len(valid_ids) < checkpoint_points:
                raise ValueError(f"Frame {index} has insufficient measured surface points for this checkpoint")
            points_input = points_input[valid_ids]
        result = policy.infer({"prompt": task, "rgb_front": rgb_front[source_frame], "qpos": qpos_input,
                               "robot_points": points_input, "current_surface": points_input, "surface_source": "measured_depth_fk_gated"})
        output_ids = np.asarray(result["selected_point_indices"], dtype=np.int64)
        selected = points_input[output_ids]
        selected_current.append(selected)
        predicted.append(np.asarray(result["predicted_robot_points"], dtype=np.float32))
        target_indices = valid_ids[output_ids] if policy.measured_surface else (output_ids if input_indices is None else input_indices[output_ids])
        target.append(selected[None] + target_offsets[index][:, target_indices])
        masks.append(target_mask[index][:, target_indices])
        predicted_actions.append(np.asarray(result["actions"], dtype=np.float32))
        print(f"[infer] {ordinal + 1}/{len(contexts)} context={index} source_frame={source_frame}", flush=True)

    current = np.asarray(selected_current, dtype=np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=bool)
    error_squared = np.square(predicted_array - target_array).sum(axis=-1)
    counts = mask_array.sum(axis=(0, 2)).clip(1)
    per_horizon_rmse_m = np.sqrt((error_squared * mask_array).sum(axis=(0, 2)) / counts)
    overall_rmse_m = float(np.sqrt((error_squared * mask_array).sum() / mask_array.sum().clip(1)))
    metric_kind = "layout-adapted preview (non-formal)" if compatibility_label else "point-identity matched"
    print(f"[metrics] kind={metric_kind} contexts={len(contexts)} points={current.shape[1]} overall_3d_point_rmse_mm={overall_rmse_m * 1000:.3f}")
    print("[metrics] per_horizon_3d_point_rmse_mm=" + ", ".join(f"{value * 1000:.3f}" for value in per_horizon_rmse_m))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer; install an FFmpeg-enabled OpenCV build")
    try:
        for local_index, index in enumerate(contexts):
            source_frame = int(sample_frames[index])
            for horizon_index in range(predicted_array.shape[1]):
                valid = mask_array[local_index, horizon_index]
                if not np.any(valid):
                    continue
                frame = _render_frame(
                    rgb_front[source_frame], current[local_index][valid], predicted_array[local_index, horizon_index][valid], target_array[local_index, horizon_index][valid],
                    context=int(index), source_frame=source_frame, horizon_index=horizon_index, horizon=predicted_array.shape[1],
                    rmse_mm=float(np.sqrt(error_squared[local_index, horizon_index][valid].mean()) * 1000), task=task,
                    compatibility_label=compatibility_label, width=args.width, height=args.height,
                    K=K, camera_to_left_base=camera_to_left_base,
                )
                writer.write(frame)
    finally:
        writer.release()
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output, contexts=contexts, selected_current_points=current, predicted_robot_points=predicted_array,
            target_robot_points=target_array, target_mask=mask_array, predicted_actions=np.asarray(predicted_actions),
            per_horizon_3d_point_rmse_m=per_horizon_rmse_m, overall_3d_point_rmse_m=np.asarray(overall_rmse_m, dtype=np.float32),
            checkpoint=np.asarray(str(args.checkpoint.resolve())), episode=np.asarray(str(args.episode.resolve())), task_text=np.asarray(task),
            compatibility_label=np.asarray(compatibility_label),
        )
    print(f"[done] video={args.output}")


if __name__ == "__main__":
    main()
