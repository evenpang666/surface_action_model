#!/usr/bin/env python3
"""Jointly fine-tune PyTorch PI05 for UR7e actions and robot-surface flow.

Input shards are produced by ``scripts/preprocess_quest3_hdf5.py``.
Each sample uses task text, current RGB images, qpos, and current robot points;
the targets are a normalized Quest3 action chunk (7-D single arm or 14-D
dual arm) and its future point offsets.
Run inside the OpenPI Python environment, for example:

  uv run --project openpi ../scripts/train_pi05_surface_legacy.py \
    --dataset ../outputs/pi05_surface --output ../outputs/pi05_ur7e_joint.pt
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = REPO_ROOT / "openpi" / "src"
for path in (REPO_ROOT, OPENPI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from safety_module.tokenizer import MODEL_IMAGE_KEYS, _Tokenizer
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from safety_module.shards import open_shard, SampleArray
from safety_module.detection_annotations import load_detection_annotations
DEFAULT_PRETRAINED = REPO_ROOT / "outputs" / "pretrained" / "pi05_base_pytorch" / "model.safetensors"





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, help="Preprocessed .npz files or directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output PyTorch checkpoint (.pt).")
    parser.add_argument("--pretrained", type=Path, default=None, help="Override model.pretrained in project config.")
    parser.add_argument("--allow-random-init", action="store_true", help="Testing only: train without a pretrained PI05 backbone.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--point-loss-weight", type=float, default=10.0)
    parser.add_argument("--detection-stage", choices=("none", "joint", "action_point", "detection"), default="none",
                        help="Use the three-head checkpoint; detection stage freezes action/point and resumes their weights.")
    parser.add_argument("--detection-loss-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None, help="Load action/point or three-head checkpoint weights before this stage.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--action-horizon", type=int, default=None, help="Defaults to the horizon stored in the dataset.")
    parser.add_argument("--max-points", type=int, default=128, help="Stable evenly-spaced point subset; use 0 for all points.")
    parser.add_argument("--point-target", choices=("fixed", "visual"), default="visual")
    parser.add_argument(
        "--camera-map",
        action="append",
        default=[],
        metavar="RGB_NAME=MODEL_KEY",
        help="For example front=base_0_rgb. Unmapped model views are zero-filled and masked.",
    )
    parser.add_argument("--tokenizer-model", type=Path, default=None, help="Optional tokenizer override. Default uses Hugging Face cache/download.")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True, help="Freeze VLM only (default); action expert and point DiT remain trainable.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Optimizer steps after this many per-rank batches; effective global batch is world_size × batch_size × this value.")
    parser.add_argument("--distributed", action="store_true", help="Require torchrun distributed launch; one DDP process per CUDA device.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable activation checkpointing (normally unsuitable for 16 GB GPUs).")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer updates across all epochs; omitted means epochs alone determine training length.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap batches per epoch for a bounded smoke test; omitted means use the complete dataset.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true", help="Validate shards and print the resolved contract without loading PI05.")
    return parser.parse_args()


def _resolve_shards(paths: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for path in paths:
        if path.is_dir() and path.suffix == ".shard":
            shards.append(path)
        elif path.is_dir():
            shards.extend(sorted(path.glob("*.npz")))
            shards.extend(sorted(path.glob("*.shard")))
        elif path.suffix == ".npz":
            shards.append(path)
        else:
            raise ValueError(f"Dataset input must be an .npz or directory: {path}")
    if not shards:
        raise FileNotFoundError("No preprocessed .npz shards found")
    return shards


def _parse_camera_map(values: list[str], rgb_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or source not in rgb_names or target not in MODEL_IMAGE_KEYS:
            raise ValueError(f"Invalid --camera-map {value!r}; use RGB_NAME=one of {MODEL_IMAGE_KEYS}")
        if target in mapping.values():
            raise ValueError(f"Multiple cameras mapped to the same PI05 view: {target}")
        mapping[source] = target
    if not mapping:
        # The deployed obstacle service owns the one calibrated front D455.
        # Wrist views remain in the shard for later experiments, but the
        # default train/deploy contract is deliberately front-only.
        source = "front" if "front" in rgb_names else rgb_names[0]
        mapping[source] = "base_0_rgb"
    return mapping




@dataclass
class _Episode:
    path: Path
    task_text: str
    task_text_source: str
    qpos: np.ndarray
    actions: np.ndarray
    current_points: np.ndarray
    target_offsets: np.ndarray
    target_mask: np.ndarray
    rgb: dict[str, np.ndarray]
    sample_count: int
    action_dim: int
    joint_dim: int
    selected_point_indices: np.ndarray
    surface_model_hash: str
    point_identity_version: str
    points_per_link: int
    arm_count: int
    current_mask: np.ndarray
    action_step_mask: np.ndarray
    detection_boxes: np.ndarray | None = None
    detection_valid: np.ndarray | None = None
    detection_reviewed: np.ndarray | None = None


class Ur7eSurfaceDataset(Dataset):
    def __init__(
        self,
        shard_paths: list[Path],
        *,
        max_points: int,
        point_target: str,
        action_horizon: int | None,
        tokenizer: _Tokenizer,
        qpos_mean: np.ndarray,
        qpos_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        include_detection: bool = False,
        reviewed_only: bool = False,
    ):
        self.qpos_mean, self.qpos_std = qpos_mean, qpos_std
        self.action_mean, self.action_std = action_mean, action_std
        self.tokenizer = tokenizer
        self.include_detection = include_detection
        self.episodes: list[_Episode] = []
        self.index: list[tuple[int, int]] = []
        image_shape: tuple[int, int, int] | None = None
        reference_points: int | None = None
        for path in shard_paths:
            episode = self._load_episode(path, max_points=max_points, point_target=point_target, action_horizon=action_horizon)
            if reference_points is None:
                reference_points = episode.current_points.shape[1]
            elif reference_points != episode.current_points.shape[1]:
                raise ValueError("All shards must resolve to the same selected point count")
            for frames in episode.rgb.values():
                if image_shape is None:
                    image_shape = tuple(frames.shape[1:])
                elif image_shape != tuple(frames.shape[1:]):
                    raise ValueError("All RGB views must share a shape before batching")
            episode_index = len(self.episodes)
            self.episodes.append(episode)
            self.index.extend((episode_index, sample_index) for sample_index in range(episode.sample_count)
                              if episode.current_mask[sample_index].any() and episode.action_step_mask[sample_index].any()
                              and (not reviewed_only or bool(episode.detection_reviewed[sample_index, 0])))
        if not self.index or image_shape is None:
            raise ValueError("Dataset contains no samples or RGB frames")
        self.image_shape = image_shape
        self.point_count = int(reference_points)
        self.camera_map: dict[str, str] = {}  # assigned by the caller after CLI validation

    @staticmethod
    def _point_indices(point_count: int, max_points: int) -> np.ndarray:
        if max_points <= 0 or max_points >= point_count:
            return np.arange(point_count, dtype=np.int64)
        return np.linspace(0, point_count - 1, max_points, dtype=np.int64)

    def _load_episode(self, path: Path, *, max_points: int, point_target: str, action_horizon: int | None) -> _Episode:
        with open_shard(path) as data:
            required = ("task_text", "qpos", "action_chunks", "sample_frame_indices", "coordinate_frame")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path} is not a preprocessed PI05 surface shard; missing {missing}")
            if str(data["coordinate_frame"].item()) != "left_base":
                raise ValueError(f"{path}: surface coordinate frame must be left_base")
            if point_target == "visual":
                required_points = ("visual_robot_current_points", "visual_robot_future_offsets", "visual_robot_flow_supervision_mask")
                missing = [key for key in required_points if key not in data]
                if missing:
                    raise ValueError(f"{path} has no visual point target; missing {missing}")
                current = np.asarray(data["visual_robot_current_points"], dtype=np.float32)
                offsets = np.asarray(data["visual_robot_future_offsets"], dtype=np.float32)
                mask = np.asarray(data["visual_robot_flow_supervision_mask"], dtype=bool)
                current_mask = np.asarray(data["current_surface_mask"], dtype=bool) if "current_surface_mask" in data else mask.any(axis=1)
            else:
                current = np.asarray(data["current_link_points"], dtype=np.float32)
                offsets = np.asarray(data["target_point_offsets"], dtype=np.float32)
                mask = np.asarray(data["target_point_mask"], dtype=bool) if "target_point_mask" in data else np.ones(offsets.shape[:-1], dtype=bool)
                current_mask = np.ones(current.shape[:2], dtype=bool)
            actions = np.asarray(data["action_chunks"], dtype=np.float32)
            action_step_mask = np.asarray(data["action_step_mask"], dtype=bool) if "action_step_mask" in data else np.ones(actions.shape[:2], dtype=bool)
            sample_frames = np.asarray(data["sample_frame_indices"], dtype=np.int64)
            qpos = np.asarray(data["qpos"], dtype=np.float32)[sample_frames]
            rgb = {key[4:]: SampleArray(np.asarray(data[key], dtype=np.uint8), frames=sample_frames) for key in data.files if key.startswith("rgb_")}
            detection_boxes = detection_valid = detection_reviewed = None
            if self.include_detection:
                rgb_shapes = {name: tuple(array.shape[1:]) for name, array in rgb.items()}
                detection_boxes, detection_valid, detection_reviewed = load_detection_annotations(
                    path, data, sample_frames, rgb_shapes)
            task_text = str(np.asarray(data["task_text"]).item()).strip()
            task_text_source = (
                str(np.asarray(data["task_text_source"]).item())
                if "task_text_source" in data
                else "legacy_shard"
            )
            surface_model_hash = str(np.asarray(data["surface_model_hash"]).item())
            point_identity_version = str(np.asarray(data["point_identity_version"]).item())
            points_per_link = int(np.asarray(data["points_per_link"]).item())
            arm_count = int(np.asarray(data["arm_count"]).item())
        if not task_text:
            raise ValueError(f"{path} has an empty task_text")
        if not rgb:
            raise ValueError(f"{path} contains no rgb_* arrays")
        if actions.ndim != 3 or not 1 <= actions.shape[-1] <= 32:
            raise ValueError(f"{path} action_chunks must have shape [N,H,A] with 1<=A<=32, got {actions.shape}")
        if qpos.ndim != 2 or qpos.shape[1] not in (6, 12):
            raise ValueError(f"{path} qpos must resolve to [N,6] or [N,12], got {qpos.shape}")
        if action_horizon is not None and actions.shape[1] != action_horizon:
            raise ValueError(f"{path} horizon {actions.shape[1]} does not match --action-horizon={action_horizon}")
        if current.ndim != 3 or current.shape[0] != actions.shape[0] or current.shape[-1] != 3:
            raise ValueError(f"{path} current points must have shape [N,K,3], got {current.shape}")
        if offsets.shape[:3] != (actions.shape[0], actions.shape[1], current.shape[1]) or offsets.shape[-1] != 3:
            raise ValueError(f"{path} point target shapes do not align with action_chunks")
        if mask.shape != offsets.shape[:-1]:
            raise ValueError(f"{path} point target mask must have shape {offsets.shape[:-1]}, got {mask.shape}")
        ids = self._point_indices(current.shape[1], max_points)
        return _Episode(
            path=path,
            task_text=task_text,
            task_text_source=task_text_source,
            qpos=qpos,
            actions=actions,
            current_points=SampleArray(current, points=ids),
            target_offsets=SampleArray(offsets, points=ids, point_axis=2),
            target_mask=SampleArray(mask, points=ids, point_axis=2),
            rgb=rgb,
            sample_count=actions.shape[0],
            action_dim=actions.shape[-1],
            joint_dim=qpos.shape[-1],
            selected_point_indices=ids,
            surface_model_hash=surface_model_hash,
            point_identity_version=point_identity_version,
            points_per_link=points_per_link,
            arm_count=arm_count,
            current_mask=current_mask[:, ids],
            action_step_mask=action_step_mask,
            detection_boxes=detection_boxes,
            detection_valid=detection_valid,
            detection_reviewed=detection_reviewed,
        )

    def set_camera_map(self, mapping: dict[str, str]) -> None:
        for episode in self.episodes:
            if set(episode.rgb) != set(self.episodes[0].rgb):
                raise ValueError("Every shard must contain the same rgb_* camera names")
        self.camera_map = mapping

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        episode_index, sample_index = self.index[index]
        episode = self.episodes[episode_index]
        valid_ids = np.flatnonzero(episode.current_mask[sample_index])
        if not len(valid_ids):
            raise ValueError(f"No visible current surface at {episode.path}:{sample_index}")
        # A point ID is stable within the chunk; ordering across samples is not a model input.
        ids = valid_ids[np.linspace(0, len(valid_ids) - 1, self.point_count, dtype=np.int64)]
        qpos = (episode.qpos[sample_index] - self.qpos_mean) / self.qpos_std
        state = np.zeros((32,), dtype=np.float32)
        state[: len(qpos)] = qpos
        tokens, token_mask = self.tokenizer.tokenize(episode.task_text, state)
        images = {key: np.zeros(self.image_shape, dtype=np.uint8) for key in MODEL_IMAGE_KEYS}
        image_masks = {key: np.asarray(False) for key in MODEL_IMAGE_KEYS}
        for source, target in self.camera_map.items():
            images[target] = episode.rgb[source][sample_index]
            image_masks[target] = np.asarray(True)
        actions = np.zeros((episode.actions.shape[1], 32), dtype=np.float32)
        actions[:, : episode.action_dim] = (episode.actions[sample_index] - self.action_mean) / self.action_std
        result = {
            "images": images,
            "image_masks": image_masks,
            "state": state,
            "tokenized_prompt": tokens.astype(np.int64),
            "tokenized_prompt_mask": token_mask,
            "actions": actions,
            "robot_points": episode.current_points[sample_index, ids],
            "joint_positions": qpos,
            "target_point_offsets": episode.target_offsets[sample_index][:, ids],
            "target_point_mask": episode.target_mask[sample_index][:, ids],
            "action_mask": (np.arange(32) < episode.action_dim)[None] & episode.action_step_mask[sample_index, :, None],
        }
        if self.include_detection:
            result["detection_images"] = {view: episode.rgb[view][sample_index] for view in ("front", "side")}
            result["detection_targets"] = episode.detection_boxes[sample_index]
            result["detection_valid"] = episode.detection_valid[sample_index]
            result["detection_reviewed"] = episode.detection_reviewed[sample_index]
        return result


def _normalization_stats(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    qpos_values, action_values, horizon, action_dim, joint_dim = [], [], None, None, None
    for path in paths:
        with open_shard(path) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
            sample_frames = np.asarray(data["sample_frame_indices"], dtype=np.int64)
            actions = np.asarray(data["action_chunks"], dtype=np.float32)
            valid = np.asarray(data["action_step_mask"], dtype=bool) if "action_step_mask" in data else np.ones(actions.shape[:2], dtype=bool)
        if actions.ndim != 3 or not 1 <= actions.shape[-1] <= 32:
            raise ValueError(f"{path} must contain action_chunks[N,H,A], 1<=A<=32")
        if horizon is not None and horizon != actions.shape[1]:
            raise ValueError("All shards must have the same action horizon")
        if action_dim is not None and action_dim != actions.shape[2]:
            raise ValueError("Do not mix single-arm 7-D and dual-arm 14-D shards in one checkpoint")
        if joint_dim is not None and joint_dim != qpos.shape[1]:
            raise ValueError("Do not mix six-joint and twelve-joint shards in one checkpoint")
        horizon = actions.shape[1]
        action_dim, joint_dim = actions.shape[2], qpos.shape[1]
        qpos_values.append(qpos[sample_frames])
        action_values.append(actions[valid])
    qpos_all, action_all = np.concatenate(qpos_values), np.concatenate(action_values)
    return (
        qpos_all.mean(axis=0).astype(np.float32),
        qpos_all.std(axis=0).clip(1e-6).astype(np.float32),
        action_all.mean(axis=0).astype(np.float32),
        action_all.std(axis=0).clip(1e-6).astype(np.float32),
        int(horizon), int(action_dim), int(joint_dim),
    )


def _to_observation(batch: dict[str, Any], device: torch.device) -> SimpleNamespace:
    # We deliberately use a lightweight observation object to keep this loader
    # independent of JAX. Mirror Observation.from_dict's uint8 NHWC -> float
    # NCHW [-1, 1] conversion before PI05's PyTorch preprocessor sees it.
    images = {
        key: batch["images"][key].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        for key in MODEL_IMAGE_KEYS
    }
    return SimpleNamespace(
        images=images,
        image_masks={key: batch["image_masks"][key].to(device=device, dtype=torch.bool) for key in MODEL_IMAGE_KEYS},
        state=batch["state"].to(device=device, dtype=torch.float32),
        tokenized_prompt=batch["tokenized_prompt"].to(device=device, dtype=torch.long),
        tokenized_prompt_mask=batch["tokenized_prompt_mask"].to(device=device, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    three_head = bool(metadata.get("detection_stage"))
    torch.save({"model_type": "SurfaceActionDetectionModelPytorch" if three_head else "SurfaceActionModelPytorch",
                "schema_version": 4 if three_head else 3, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "metadata": metadata}, path)


def _distributed_context(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    """Initialize one process per GPU when launched through ``torchrun``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.distributed and world_size < 2:
        raise ValueError("--distributed requires torchrun with at least two processes")
    if world_size == 1:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        return device, 0, 1, False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed PI05 training requires CUDA and an NCCL-capable torch build")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}"), int(distributed.get_rank()), world_size, True


def main() -> None:
    args = parse_args()
    project, _ = load_project_config(args.project_config)
    if args.pretrained is None:
        args.pretrained = resolve_repo_path(project["model"]["pretrained"])
    if args.epochs < 1 or args.batch_size < 1 or args.max_points < 0:
        raise ValueError("epochs and batch-size must be positive; max-points must be >= 0")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max-train-batches must be positive when supplied")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive when supplied")
    if args.detection_loss_weight < 0:
        raise ValueError("--detection-loss-weight must be nonnegative")
    if args.detection_stage == "detection" and args.resume is None:
        raise ValueError("Detection-only stage requires --resume with trained action/point weights")
    if args.detection_stage == "none" and args.resume is not None:
        raise ValueError("--resume is supported only for the three-head training stages")
    if args.resume is not None and args.output.resolve() == args.resume.resolve():
        raise ValueError("Stage output must not overwrite its resume checkpoint")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, rank, world_size, is_distributed = (torch.device("cpu"), 0, 1, False) if args.validate_only else _distributed_context(args)
    is_primary = rank == 0
    if device.type == "cuda" and args.precision == "bfloat16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("bfloat16 requires Ampere-or-newer CUDA hardware. V100 requires --precision float16.")
    if device.type == "cuda" and not args.freeze_base and torch.cuda.get_device_properties(device).total_memory < 48 * 2**30:
        raise ValueError("Full PI05 AdamW fine-tuning needs roughly 48+ GiB per GPU. On V100 use --freeze-base; DDP replicates rather than shards the backbone.")
    shards = _resolve_shards(args.dataset)
    periods = []
    for path in shards:
        with open_shard(path) as shard:
            if "action_dt_s" in shard:
                dt = np.asarray(shard["action_dt_s"])
                valid = np.asarray(shard["action_step_mask"], bool)
                values = dt[valid]
                if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"{path}: invalid action timing")
                period = float(np.median(values))
                if np.max(np.abs(values - period)) > .2 * period:
                    raise ValueError(f"{path}: irregular sample intervals; resample the episode before fixed-period training")
                periods.append(period)
    if periods and max(periods) - min(periods) > .05 * np.median(periods):
        raise ValueError("Shards use different action periods")
    qpos_mean, qpos_std, action_mean, action_std, detected_horizon, logical_action_dim, joint_dim = _normalization_stats(shards)
    horizon = detected_horizon if args.action_horizon is None else args.action_horizon
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        if resume_payload.get("model_type") not in ("SurfaceActionModelPytorch", "SurfaceActionDetectionModelPytorch"):
            raise ValueError("--resume must be a surface action or three-head checkpoint")
        previous = resume_payload["metadata"]
        if (int(previous["action_horizon"]) != horizon or int(previous["logical_action_dim"]) != logical_action_dim
                or int(previous["joint_dim"]) != joint_dim or previous.get("point_target") != args.point_target):
            raise ValueError("Resume checkpoint action/point contract differs from the training dataset")
        if previous.get("precision") != args.precision:
            raise ValueError("Resume checkpoint precision differs from --precision")
        qpos_mean, qpos_std = np.asarray(previous["qpos_mean"], np.float32), np.asarray(previous["qpos_std"], np.float32)
        action_mean, action_std = np.asarray(previous["action_mean"], np.float32), np.asarray(previous["action_std"], np.float32)
    tokenizer = None if args.validate_only else _Tokenizer(max_len=200, model_path=args.tokenizer_model)
    dataset = Ur7eSurfaceDataset(
        shards,
        max_points=args.max_points,
        point_target=args.point_target,
        action_horizon=horizon,
        tokenizer=tokenizer,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        action_mean=action_mean,
        action_std=action_std,
        include_detection=args.detection_stage in ("joint", "detection"),
        reviewed_only=args.detection_stage == "detection",
    )
    rgb_names = sorted(dataset.episodes[0].rgb)
    camera_values = args.camera_map or [f"{source}={target}" for source, target in project["quest3"]["camera_to_model"].items()]
    dataset.set_camera_map(_parse_camera_map(camera_values, rgb_names))
    if resume_payload is not None and dataset.camera_map != resume_payload["metadata"]["camera_map"]:
        raise ValueError("Resume checkpoint camera map differs from the training dataset")
    if args.detection_stage in ("joint", "detection"):
        reviewed_count = sum(int(np.count_nonzero(episode.detection_reviewed[:, 0])) for episode in dataset.episodes)
        if not reviewed_count:
            raise ValueError("Detection training requires at least one reviewed front/side RGB frame")
    reference = dataset.episodes[0]
    if resume_payload is not None:
        if (reference.surface_model_hash != previous.get("surface_model_hash")
                or reference.point_identity_version != previous.get("point_identity_version")
                or dataset.point_count != int(previous.get("point_count", -1))
                or reference.selected_point_indices.tolist() != previous.get("selected_point_indices")):
            raise ValueError("Resume checkpoint surface point contract differs from the training dataset")
    for episode in dataset.episodes[1:]:
        if (episode.surface_model_hash, episode.point_identity_version, episode.points_per_link, episode.arm_count) != (
            reference.surface_model_hash, reference.point_identity_version, reference.points_per_link, reference.arm_count
        ):
            raise ValueError("All shards in one checkpoint must use the same arm/tool surface model and point identity")
    if args.validate_only:
        print({
            "shards": len(shards), "samples": len(dataset), "joint_dim": joint_dim,
            "logical_action_dim": logical_action_dim, "action_horizon": horizon,
            "point_count": dataset.point_count, "rgb_views": rgb_names,
            "camera_map": dataset.camera_map, "coordinate_frame": "left_base",
            "detection_stage": args.detection_stage,
            "reviewed_detection_frames": sum(int(np.count_nonzero(ep.detection_reviewed[:, 0])) for ep in dataset.episodes) if args.detection_stage in ("joint", "detection") else 0,
            "episode_task_texts": [episode.task_text for episode in dataset.episodes],
            "episode_task_text_sources": [episode.task_text_source for episode in dataset.episodes],
        })
        return
    pretrained = None if args.allow_random_init or resume_payload is not None else args.pretrained
    if pretrained is None and not args.allow_random_init and resume_payload is None:
        raise ValueError("Joint world-action training requires the default PI05 weights (or explicit testing-only --allow-random-init)")
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(
            f"PI05 weights were not found at {pretrained}. Run the OpenPI-to-PyTorch conversion or pass --pretrained."
        )
    if len(dataset) < args.batch_size:
        raise ValueError(f"Dataset has {len(dataset)} samples, smaller than --batch-size={args.batch_size}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers, drop_last=True)
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
        from openpi.models_pytorch.surface_detection_pytorch import SurfaceActionDetectionModelPytorch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenPI dependencies are unavailable. Run this script in the OpenPI environment, for example: "
            "cd openpi && uv run --project . ../scripts/train_pi05_surface_legacy.py ..."
        ) from exc
    config = Pi0Config(action_dim=32, action_horizon=horizon, pi05=True, dtype=args.precision)
    model_class = SurfaceActionModelPytorch if args.detection_stage == "none" else SurfaceActionDetectionModelPytorch
    model = model_class(config, joint_dim=joint_dim).to(device)
    if not args.freeze_base:
        raise ValueError("Surface Action Model freezes the VLM by design; omit --no-freeze-base")
    if resume_payload is not None:
        missing, unexpected = model.load_state_dict(resume_payload["model_state_dict"], strict=False)
        if unexpected or (resume_payload["model_type"] == "SurfaceActionDetectionModelPytorch" and missing) or any(
                not key.startswith("detection_expert.") for key in missing):
            raise RuntimeError(f"Resume weights do not fit the three-head model: missing={missing}, unexpected={unexpected}")
        if is_primary:
            print(f"[resume] loaded={args.resume}; initialized_detection_keys={len(missing)}")
        del resume_payload
    elif pretrained is not None:
        from safetensors.torch import load_model

        # load_model understands the tied-weight aliases recorded by
        # safetensors.save_model. load_file()+load_state_dict() silently leaves
        # PaliGemma's shared token embedding random.
        missing, unexpected = load_model(model, str(pretrained), strict=False, device=str(device))
        disallowed_missing = [key for key in missing if not key.startswith(("surface_", "detection_expert."))]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Pretrained PI05 backbone is incomplete or incompatible: "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
        if is_primary:
            print(f"[weights] PI05 backbone complete; new_surface_keys={len(missing)}")
    if args.freeze_base:
        trainable_prefixes = (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.", "action_out_proj.", "time_mlp_", "surface_", "detection_expert.",
        )
        for name, parameter in model.named_parameters():
            trainable = name.startswith(trainable_prefixes)
            if args.detection_stage == "detection":
                trainable = name.startswith("detection_expert.")
            elif args.detection_stage == "action_point":
                trainable = trainable and not name.startswith("detection_expert.")
            parameter.requires_grad_(trainable)
        if is_primary:
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in model.parameters())
            print(f"[freeze-base] stage={args.detection_stage}; trainable={trainable:,}/{total:,} parameters")
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # FP16 optimizer updates need FP32 master parameters and loss scaling.
    if args.precision == "float16":
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "float16")
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {
        "schema_version": 4 if args.detection_stage != "none" else 3,
        "detection_stage": args.detection_stage if args.detection_stage != "none" else None,
        "detection_views": ["front", "side"] if args.detection_stage != "none" else None,
        "detection_resolution": 224 if args.detection_stage != "none" else None,
        "detection_queries": 32 if args.detection_stage != "none" else None,
        "action_period_s": float(np.median(periods)) if periods else None,
        "dataset": [str(path) for path in shards], "point_target": args.point_target, "point_count": dataset.point_count,
        "camera_map": dataset.camera_map,
        "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
        "qpos_mean": qpos_mean.tolist(), "qpos_std": qpos_std.tolist(), "action_horizon": horizon,
        "logical_action_dim": logical_action_dim, "joint_dim": joint_dim,
        "model_action_dim": 32, "coordinate_frame": "left_base",
        "selected_point_indices": dataset.episodes[0].selected_point_indices.tolist(),
        "precision": args.precision,
        "surface_model_hash": reference.surface_model_hash,
        "point_identity_version": reference.point_identity_version,
        "points_per_link": reference.points_per_link,
        "arm_count": reference.arm_count,
        "episode_task_texts": [episode.task_text for episode in dataset.episodes],
        "episode_task_text_sources": [episode.task_text_source for episode in dataset.episodes],
        "max_train_batches": args.max_train_batches,
        "world_size": world_size,
        "per_rank_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": world_size * args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
    }
    completed_steps = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = total_action = total_point = total_detection = 0.0
        optimizer.zero_grad(set_to_none=True)
        requested_batches = min(len(loader), args.max_train_batches or len(loader))
        reached_step_limit = False
        for batch_index, batch in enumerate(loader, start=1):
            accumulation_start = ((batch_index - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps + 1
            accumulation_size = min(args.gradient_accumulation_steps, requested_batches - accumulation_start + 1)
            should_step = batch_index == accumulation_start + accumulation_size - 1
            sync_context = model.no_sync() if is_distributed and not should_step else nullcontext()
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" and args.precision == "float16" else nullcontext()
            with sync_context, amp_context:
                detection_kwargs = {}
                if args.detection_stage in ("joint", "detection"):
                    detection_kwargs = {
                        "detection_images": {view: batch["detection_images"][view].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0 for view in ("front", "side")},
                        "detection_targets": batch["detection_targets"].to(device=device, dtype=torch.float32),
                        "detection_valid": batch["detection_valid"].to(device=device, dtype=torch.bool),
                        "detection_reviewed": batch["detection_reviewed"].to(device=device, dtype=torch.bool),
                        "detection_loss_weight": args.detection_loss_weight,
                    }
                losses = model(
                    _to_observation(batch, device),
                batch["actions"].to(device=device, dtype=torch.float32),
                batch["robot_points"].to(device=device, dtype=torch.float32),
                batch["joint_positions"].to(device=device, dtype=torch.float32),
                batch["target_point_offsets"].to(device=device, dtype=torch.float32),
                batch["target_point_mask"].to(device=device, dtype=torch.bool),
                point_loss_weight=args.point_loss_weight,
                action_mask=batch["action_mask"].to(device=device),
                **({"train_stage": args.detection_stage, **detection_kwargs} if args.detection_stage != "none" else {}),
                )
                scaler.scale(losses["loss"] / accumulation_size).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                reached_step_limit = args.max_steps is not None and completed_steps >= args.max_steps
            total_loss += float(losses["loss"].detach())
            total_action += float(losses["action_loss"]) if "action_loss" in losses else 0.0
            total_point += float(losses["point_loss"]) if "point_loss" in losses else 0.0
            total_detection += float(losses["detection_loss"].detach()) if "detection_loss" in losses else 0.0
            if (args.max_train_batches is not None and batch_index >= args.max_train_batches) or reached_step_limit:
                break
        batches = batch_index
        aggregate = torch.tensor((total_loss, total_action, total_point, total_detection, float(batches)), dtype=torch.float64, device=device)
        if is_distributed:
            distributed.all_reduce(aggregate, op=distributed.ReduceOp.SUM)
        if is_primary:
            global_batches = aggregate[4].item()
            print(f"epoch={epoch:03d} step={completed_steps} loss={aggregate[0].item() / global_batches:.6f} action={aggregate[1].item() / global_batches:.6f} point={aggregate[2].item() / global_batches:.6f} detection={aggregate[3].item() / global_batches:.6f}")
            if epoch % args.save_every == 0 or epoch == args.epochs or reached_step_limit:
                _save_checkpoint(args.output, model.module if is_distributed else model, optimizer, epoch, metadata)
        if reached_step_limit:
            break
    if is_distributed:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
