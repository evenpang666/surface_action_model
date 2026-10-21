#!/usr/bin/env python3
"""Train PI0.5 actions + SAM 3 front/side boxes using four RGB views, no points."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "openpi" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from safety_module.policy_contract import CAMERA_TO_MODEL, DETECTION_VIEWS, MODEL_TYPE, SCHEMA_VERSION, BOX_FORMAT
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from safety_module.sam3_policy_data import Sam3VlaDataset, to_observation
from safety_module.shards import open_shard
from safety_module.tokenizer import _Tokenizer
from scripts.train_pi05_surface_legacy import _resolve_shards, _normalization_stats, _distributed_context


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, help="Base PI0.5 safetensors; defaults to project configuration")
    parser.add_argument("--sam3-checkpoint", type=Path, help="Initialize native SAM 3 decoder/scorer from official weights")
    parser.add_argument("--resume", type=Path, help="Continue from a schema-5 SAM 3 VLA checkpoint")
    parser.add_argument("--init-from", type=Path, help="Migrate only PI0.5 weights from an old surface checkpoint; discard point/detection heads")
    parser.add_argument("--allow-random-init", action="store_true", help="Testing only; skip pretrained PI0.5 weights")
    parser.add_argument("--detection-stage", choices=("joint", "action", "detection"), default="joint")
    parser.add_argument("--detection-loss-weight", type=float, default=1.)
    parser.add_argument("--detection-queries", type=int, default=200)
    parser.add_argument("--detection-layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def set_train_stage(model, stage, *, freeze_base=True):
    for name, parameter in model.named_parameters():
        detector = name.startswith("detection_expert.")
        backbone = name.startswith("paligemma_with_expert.paligemma.")
        parameter.requires_grad_((stage != "action" if detector else stage != "detection")
                                 and (not backbone or not freeze_base))


def save_checkpoint(path, model, optimizer, epoch, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"model_type": MODEL_TYPE, "schema_version": SCHEMA_VERSION,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "metadata": metadata}, temporary)
    temporary.replace(path)


def action_period(paths):
    periods = []
    for path in paths:
        with open_shard(path) as data:
            if "action_dt_s" not in data:
                continue
            dt = np.asarray(data["action_dt_s"])
            valid = np.asarray(data["action_step_mask"], bool) if "action_step_mask" in data else np.ones(dt.shape, bool)
            values = dt[valid]
            if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{path}: invalid action timing")
            period = float(np.median(values))
            if np.max(np.abs(values - period)) > .2 * period:
                raise ValueError(f"{path}: resample irregular action intervals before training")
            periods.append(period)
    if periods and max(periods) - min(periods) > .05 * np.median(periods):
        raise ValueError("Shards use different action periods")
    return float(np.median(periods)) if periods else None


def main():
    args = parse_args()
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "save_every", "detection_queries", "detection_layers"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_steps", "max_train_batches", "action_horizon"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.detection_loss_weight) or args.detection_loss_weight < 0:
        raise ValueError("--detection-loss-weight must be finite and nonnegative")
    if args.resume and (args.init_from or args.sam3_checkpoint or args.allow_random_init):
        raise ValueError("--resume cannot be combined with initialization options")
    if args.init_from and args.allow_random_init:
        raise ValueError("--init-from cannot be combined with --allow-random-init")
    for source in (args.resume, args.init_from, args.sam3_checkpoint, args.pretrained):
        if source is not None and source.resolve() == args.output.resolve():
            raise ValueError("Output must not overwrite an input checkpoint")
    project, _ = load_project_config(args.project_config)
    configured_cameras = project["model"].get("camera_to_model", CAMERA_TO_MODEL)
    if (configured_cameras != CAMERA_TO_MODEL
            or tuple(project["model"].get("detection_views", DETECTION_VIEWS)) != DETECTION_VIEWS
            or project["model"].get("point_output", False)):
        raise ValueError("Project model must use four policy cameras, front/side detection and no point output")
    pretrained = args.pretrained or resolve_repo_path(project["model"]["pretrained"])
    if args.output.resolve() == pretrained.resolve():
        raise ValueError("Output must not overwrite the pretrained model")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = _resolve_shards(args.dataset)
    qmean, qstd, amean, astd, horizon, action_dim, joint_dim = _normalization_stats(paths)
    if (joint_dim not in (6, 12) or action_dim not in (7, 14) or action_dim != joint_dim // 6 * 7
            or not all(np.isfinite(value).all() for value in (qmean, qstd, amean, astd))):
        raise ValueError("Expected finite UR7e normalization with 6/7 or 12/14 state/action dimensions")
    if args.action_horizon is not None and args.action_horizon != horizon:
        raise ValueError("--action-horizon differs from dataset")
    payload = None
    previous = {}
    if args.resume or args.init_from:
        payload = torch.load(args.resume or args.init_from, map_location="cpu", weights_only=True, mmap=True)
        if args.resume and (payload.get("model_type") != MODEL_TYPE or payload.get("schema_version") != SCHEMA_VERSION):
            raise ValueError("--resume requires schema-5 SAM 3 VLA; use --init-from to discard old point heads")
        if args.init_from and payload.get("model_type") not in (
                "PI05SafetyPytorch", "SurfaceActionModelPytorch", "SurfaceActionDetectionModelPytorch"):
            raise ValueError("--init-from requires an old PI0.5 surface checkpoint")
        previous = payload["metadata"]
        if (previous.get("action_horizon") != horizon or previous.get("logical_action_dim") != action_dim
                or previous.get("joint_dim") != joint_dim or previous.get("precision") != args.precision):
            raise ValueError("Input checkpoint action/precision contract differs from dataset/CLI")
        if args.resume and (previous.get("camera_map") != CAMERA_TO_MODEL
                            or previous.get("detection_queries") != args.detection_queries
                            or previous.get("detection_layers") != args.detection_layers):
            raise ValueError("Resume camera/decoder configuration differs from CLI")
        qmean, qstd = np.asarray(previous["qpos_mean"], np.float32), np.asarray(previous["qpos_std"], np.float32)
        amean, astd = np.asarray(previous["action_mean"], np.float32), np.asarray(previous["action_std"], np.float32)
    dataset = Sam3VlaDataset(paths, tokenizer=None if args.validate_only else _Tokenizer(200, args.tokenizer_model),
                            qpos_mean=qmean, qpos_std=qstd, action_mean=amean, action_std=astd,
                            action_horizon=horizon, reviewed_only=args.detection_stage == "detection",
                            include_detection=args.detection_stage != "action")
    reviewed_count = sum(bool(dataset.episodes[episode].reviewed[sample].any())
                         for episode, sample in dataset.index)
    if args.detection_stage != "action" and not reviewed_count:
        raise ValueError("Detection training requires at least one reviewed front/side frame")
    period = action_period(paths)
    if args.resume and previous.get("action_period_s") != period:
        raise ValueError("Resume checkpoint action period differs from dataset")
    detection_trained = bool(previous.get("detection_trained", False)) if args.resume else False
    metadata = {
        "schema_version": SCHEMA_VERSION, "camera_map": CAMERA_TO_MODEL,
        "detection_views": list(DETECTION_VIEWS), "detection_box_format": BOX_FORMAT,
        "detection_output_format": "xyxy_original_pixels", "point_output": False,
        "detection_stage": args.detection_stage, "detection_trained": detection_trained,
        "detection_queries": args.detection_queries, "detection_layers": args.detection_layers,
        "detection_width": 256, "detection_heads": 8, "detection_ffn_width": 2048,
        "detection_context": "separate_front_side_text_prefix",
        "sam3_checkpoint": str(args.sam3_checkpoint) if args.sam3_checkpoint else previous.get("sam3_checkpoint"),
        "sam3_source_revision": "7f473cfd983d5eb43040ff608f66bff0d7bdcb53",
        "precision": args.precision, "model_action_dim": 32,
        "qpos_mean": qmean.tolist(), "qpos_std": qstd.tolist(),
        "action_mean": amean.tolist(), "action_std": astd.tolist(),
        "joint_dim": joint_dim, "logical_action_dim": action_dim, "action_horizon": horizon,
        "action_period_s": period, "coordinate_frame": "left_base",
        "dataset": [str(path) for path in paths], "freeze_base": args.freeze_base,
        "episode_task_texts": [ep.task_text for ep in dataset.episodes],
        "world_size": 1, "per_rank_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if args.validate_only:
        print({"model_type": MODEL_TYPE, "shards": len(paths), "samples": len(dataset),
               "reviewed_detection_frames": reviewed_count, **metadata})
        return
    if not args.resume and not args.init_from and not args.allow_random_init and not pretrained.is_file():
        raise FileNotFoundError(f"PI0.5 weights not found: {pretrained}")
    device, rank, world_size, is_distributed = _distributed_context(args)
    if device.type == "cuda" and args.precision == "bfloat16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("BF16 requires Ampere or newer; use --precision float16 on V100")
    if len(dataset) < args.batch_size * world_size:
        raise ValueError("Not enough samples for one full global batch")
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.sam3_vla_pytorch import Sam3VlaModelPytorch

    model = Sam3VlaModelPytorch(Pi0Config(pi05=True, dtype=args.precision, action_horizon=horizon),
                               detection_queries=args.detection_queries, detection_layers=args.detection_layers).to(device)
    if payload is not None:
        state = payload["model_state_dict"]
        if args.init_from:
            state = {key: value for key, value in state.items() if not key.startswith(("surface_", "detection_expert."))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("detection_expert.") for key in missing) or (args.resume and missing):
            raise RuntimeError(f"Incompatible model weights: missing={missing}, unexpected={unexpected}")
    elif not args.allow_random_init:
        from safetensors.torch import load_model
        missing, unexpected = load_model(model, str(pretrained), strict=False, device=str(device))
        if unexpected or any(not key.startswith("detection_expert.") for key in missing):
            raise RuntimeError(f"Incomplete PI0.5 weights: missing={missing}, unexpected={unexpected}")
    if args.sam3_checkpoint:
        print({"sam3_loaded": model.detection_expert.load_sam3_weights(args.sam3_checkpoint)})
    elif not args.resume and rank == 0:
        print("[SAM 3] Decoder/scorer initialized randomly; adapters and detection head require training. Pass --sam3-checkpoint to initialize official detection weights.")
    set_train_stage(model, args.detection_stage, freeze_base=args.freeze_base)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.detection_expert.decoder.use_act_checkpoint = not args.no_gradient_checkpointing
    if args.precision == "float16":
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.seed) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, drop_last=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    same_stage = args.resume and previous.get("detection_stage") == args.detection_stage and previous.get("freeze_base") == args.freeze_base
    if same_stage:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    start_epoch = int(payload["epoch"]) + 1 if same_stage else 1
    del payload
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "float16")
    metadata.update(world_size=world_size, effective_global_batch_size=world_size * args.batch_size * args.gradient_accumulation_steps)
    completed_steps = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        totals = np.zeros(3)
        detection_supervised = False
        requested = min(len(loader), args.max_train_batches or len(loader))
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader, 1):
            start = (index - 1) // args.gradient_accumulation_steps * args.gradient_accumulation_steps + 1
            accumulation_size = min(args.gradient_accumulation_steps, requested - start + 1)
            should_step = index == start + accumulation_size - 1
            sync = model.no_sync() if is_distributed and not should_step else nullcontext()
            amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and args.precision == "float16" else nullcontext()
            with sync, amp:
                losses = model(to_observation(batch, device), batch["actions"].to(device),
                               action_mask=batch["action_mask"].to(device),
                               detection_targets=batch["detection_targets"].to(device),
                               detection_valid=batch["detection_valid"].to(device),
                               detection_reviewed=batch["detection_reviewed"].to(device),
                               detection_loss_weight=args.detection_loss_weight, train_stage=args.detection_stage)
                scaler.scale(losses["loss"] / accumulation_size).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
            totals += [float(losses["loss"].detach()), float(losses.get("action_loss", 0)),
                       float(losses.get("detection_loss", torch.tensor(0.)).detach())]
            detection_supervised |= (args.detection_stage != "action" and args.detection_loss_weight > 0
                                     and bool(batch["detection_reviewed"].any()))
            if index >= requested or (should_step and args.max_steps is not None and completed_steps >= args.max_steps):
                break
        aggregate = torch.tensor([*totals, index, float(detection_supervised)], dtype=torch.float64, device=device)
        if is_distributed:
            distributed.all_reduce(aggregate)
        detection_trained |= bool(aggregate[4] > 0)
        metadata["detection_trained"] = detection_trained
        if rank == 0:
            values = (aggregate[:3] / aggregate[3]).tolist()
            print(f"epoch={epoch} step={completed_steps} loss={values[0]:.6f} action={values[1]:.6f} detection={values[2]:.6f}")
            if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1 or (args.max_steps and completed_steps >= args.max_steps):
                save_checkpoint(args.output, model.module if is_distributed else model, optimizer, epoch, metadata)
        if args.max_steps and completed_steps >= args.max_steps:
            break
    if is_distributed:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
