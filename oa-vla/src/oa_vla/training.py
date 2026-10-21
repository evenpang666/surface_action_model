"""Dataset discovery, normalization and distributed helpers local to OA-VLA."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed

from .shards import open_shard


def resolve_shards(paths):
    result = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir() and path.suffix == ".shard" or path.is_file() and path.suffix == ".npz":
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.glob("*.npz")))
            result.extend(sorted(path.glob("*.shard")))
        else:
            raise ValueError(f"Expected an .npz, .shard directory or dataset directory: {path}")
    if not result:
        raise ValueError("No dataset shards found")
    return result


def normalization_stats(paths):
    states, actions, contract, periods = [], [], None, []
    layouts, sources = [], []
    for path in paths:
        with open_shard(path) as data:
            qpos = np.asarray(data["qpos"], np.float32)
            frames = np.asarray(data["sample_frame_indices"], np.int64)
            chunks = np.asarray(data["action_chunks"], np.float32)
            if qpos.ndim != 2 or chunks.ndim != 3 or frames.shape != chunks.shape[:1]:
                raise ValueError(f"{path}: invalid state/action sample shapes")
            if np.any(frames < 0) or np.any(frames >= len(qpos)):
                raise ValueError(f"{path}: invalid sample_frame_indices")
            layouts.append(tuple(str(x) for x in data["action_layout"].tolist()) if "action_layout" in data else None)
            sources.append(str(data["action_source"].item()) if "action_source" in data else None)
            current = (chunks.shape[1], chunks.shape[2], qpos.shape[1])
            if contract is not None and current != contract:
                raise ValueError("Dataset shards must share action horizon, action and state dimensions")
            contract = current
            mask = np.asarray(data["action_step_mask"], bool) if "action_step_mask" in data else np.ones(chunks.shape[:2], bool)
            if mask.shape != chunks.shape[:2] or not np.isfinite(chunks[mask]).all() or not np.isfinite(qpos[frames]).all():
                raise ValueError(f"{path}: invalid action mask or non-finite supervised values")
            states.append(qpos[frames])
            actions.append(chunks[mask])
            if "action_dt_s" in data:
                dt = np.asarray(data["action_dt_s"])
                if dt.shape != mask.shape:
                    raise ValueError(f"{path}: action timing shape mismatch")
                values = dt[mask]
                if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"{path}: invalid action timing")
                period = float(np.median(values))
                if np.max(np.abs(values - period)) > .2 * period:
                    raise ValueError(f"{path}: resample irregular action intervals before training")
                periods.append(period)
    states, actions = np.concatenate(states), np.concatenate(actions)
    if not len(states) or not len(actions):
        raise ValueError("Dataset has no valid state/action targets")
    if periods and max(periods) - min(periods) > .05 * np.median(periods):
        raise ValueError("Dataset shards use different action periods")
    known_layouts = {value for value in layouts if value is not None}
    known_sources = {value for value in sources if value is not None}
    if len(known_layouts) > 1 or len(known_sources) > 1:
        raise ValueError("Dataset shards use different action layouts/sources")
    semantics = None
    layout = next(iter(known_layouts)) if known_layouts and all(x is not None for x in layouts) else None
    if layout is not None:
        joints = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        expected = tuple(name for arm in ("left", "right")
                         for name in (*(f"{arm}_{joint}_delta_rad" for joint in joints), f"{arm}_gripper_target"))
        if layout == expected and contract[1:] == (14, 12):
            semantics = "interleaved_joint_delta_gripper_target"
    metadata = {"action_semantics": semantics, "action_layout": list(layout) if layout is not None else None,
                "action_source": next(iter(known_sources)) if known_sources and all(x is not None for x in sources) else None,
                "qpos_mean": states.mean(0).tolist(), "qpos_std": states.std(0).clip(1e-6).tolist(),
                "action_mean": actions.mean(0).tolist(), "action_std": actions.std(0).clip(1e-6).tolist(),
                "action_period_s": float(np.median(periods)) if periods else None}
    return metadata, contract


def distributed_context(device_name, *, require_distributed=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if require_distributed:
            raise ValueError("--distributed requires torchrun with at least two processes")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; explicitly choose --device cpu for CPU tests")
        return torch.device(device_name), 0, 1, False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA/NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl")
    return torch.device(f"cuda:{local_rank}"), distributed.get_rank(), world_size, True


def set_train_stage(model, stage, *, freeze_backbone=True):
    if stage not in ("joint", "action", "detection"):
        raise ValueError(f"Unknown training stage: {stage}")
    for name, parameter in model.named_parameters():
        detector = name.startswith("detection_expert.")
        backbone = name.startswith("paligemma_with_expert.paligemma.")
        parameter.requires_grad_((stage != "action" if detector else stage != "detection")
                                 and (not backbone or not freeze_backbone))
