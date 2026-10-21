"""Standalone training CLI: python -m oa_vla.train or oa-vla-train."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import OAVLAConfig
from .data import OAVlaDataset, to_observation
from .policy import OAVLAPolicy
from .tokenizer import Tokenizer, ByteTokenizer
from .training import resolve_shards, normalization_stats, distributed_context, set_train_stage


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Portable export directory plus training_state.pt")
    parser.add_argument("--config", type=Path, help="OA-VLA JSON config; otherwise infer dimensions from dataset")
    parser.add_argument("--tiny", action="store_true", help="Small offline CPU test architecture")
    parser.add_argument("--pretrained", type=Path, help="PI0.5-compatible backbone safetensors or .pt")
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="Existing OA-VLA export directory")
    parser.add_argument("--allow-random-init", action="store_true", help="Testing only")
    parser.add_argument("--stage", choices=("joint", "action", "detection"), default="joint")
    parser.add_argument("--detection-loss-weight", type=float, default=1.)
    parser.add_argument("--detection-queries", type=int)
    parser.add_argument("--detection-layers", type=int)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20, help="Additional epochs, including when resuming")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "save_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("max_steps", "max_train_batches", "detection_queries", "detection_layers"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("lr", "grad_clip_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(args.detection_loss_weight) or args.detection_loss_weight < 0:
        raise ValueError("detection_loss_weight must be finite and nonnegative")
    if args.resume and any((args.pretrained, args.sam3_checkpoint, args.allow_random_init, args.tiny)):
        raise ValueError("--resume cannot be combined with initialization options")
    if args.config and args.tiny:
        raise ValueError("Choose --config or --tiny")
    paths = resolve_shards(args.dataset)
    stats, (horizon, action_dim, state_dim) = normalization_stats(paths)
    previous, resume_state = {}, None
    if args.resume:
        config = OAVLAConfig.load(args.resume / "config.json")
        if args.config and OAVLAConfig.load(args.config) != config:
            raise ValueError("Resume model configuration differs from --config")
        previous = json.loads((args.resume / "metadata.json").read_text())
        for key in ("action_semantics", "action_source", "action_layout"):
            if previous.get(key) is not None and stats.get(key) is not None and previous[key] != stats[key]:
                raise ValueError(f"Resume {key} differs from dataset")
            if stats.get(key) is None:
                stats[key] = previous.get(key)
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
            stats[key] = previous[key]
        if previous.get("action_period_s") != stats["action_period_s"]:
            raise ValueError("Resume action period differs from dataset")
    elif args.config:
        config = OAVLAConfig.load(args.config)
    else:
        config = OAVLAConfig.tiny() if args.tiny else OAVLAConfig()
        config = replace(config, action_horizon=horizon, logical_action_dim=action_dim, state_dim=state_dim)
    if (config.action_horizon, config.logical_action_dim, config.state_dim) != (horizon, action_dim, state_dim):
        raise ValueError("Dataset horizon/state/action dimensions differ from configuration")
    overrides = {}
    for cli, field in ((args.precision, "dtype"), (args.detection_queries, "detection_queries"),
                       (args.detection_layers, "detection_layers"), (args.freeze_backbone, "freeze_backbone")):
        if cli is not None:
            if args.resume and field not in ("freeze_backbone",) and cli != getattr(config, field):
                raise ValueError(f"Cannot change {field} when resuming")
            overrides[field] = cli
    if args.no_gradient_checkpointing:
        overrides["gradient_checkpointing"] = False
    config = replace(config, **overrides)
    tokenizer = None
    if not args.validate_only:
        if args.tiny:
            if not args.allow_random_init:
                raise ValueError("--tiny requires explicit --allow-random-init")
            tokenizer = ByteTokenizer(config.max_token_len)
        elif args.resume and (args.resume / "processor.json").exists():
            settings = json.loads((args.resume / "processor.json").read_text())
            if settings["type"] == "byte_test_only":
                tokenizer = ByteTokenizer(config.max_token_len)
        if tokenizer is None:
            local_tokenizer = args.resume / "tokenizer.model" if args.resume else None
            tokenizer = Tokenizer(config.max_token_len, args.tokenizer_model or local_tokenizer)
    dataset = OAVlaDataset(paths, tokenizer=tokenizer, action_horizon=horizon,
                          **{name: stats[name] for name in ("qpos_mean", "qpos_std", "action_mean", "action_std")},
                          reviewed_only=args.stage == "detection", include_detection=args.stage != "action",
                          padded_action_dim=config.action_dim)
    reviewed_count = sum(bool(dataset.episodes[episode].reviewed[sample].any()) for episode, sample in dataset.index)
    if args.stage != "action" and not reviewed_count:
        raise ValueError("Detection training requires reviewed eligible front/side frames")
    if args.stage != "action" and any((ep.valid & ep.reviewed[..., None]).any(1).sum(-1).max(initial=0) > config.detection_queries
                                       for ep in dataset.episodes):
        raise ValueError("More annotated objects than detection queries")
    if args.validate_only:
        print(json.dumps({"model_type": "oa-vla", "samples": len(dataset), "reviewed_frames": reviewed_count,
                          "policy_views": ["front", "side", "right_wrist", "left_wrist"],
                          "detection_views": ["front", "side"], "detection_format": "paired_obb_v1", "point_output": False,
                          "config": config.to_dict()}, indent=2))
        return
    if not args.resume and not args.pretrained and not args.allow_random_init:
        raise ValueError("Pass --pretrained, --resume or explicit testing-only --allow-random-init")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Choose a new export directory or explicitly --resume the existing model")
    if args.resume and args.output.resolve() != args.resume.resolve() and args.output.exists():
        raise FileExistsError("New stage output already exists")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, rank, world_size, is_distributed = distributed_context(args.device, require_distributed=args.distributed)
    if device.type == "cuda" and config.dtype == "bfloat16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("BF16 requires Ampere or newer; use --precision float16 on V100")
    if len(dataset) < args.batch_size * world_size:
        raise ValueError("Not enough samples for a full global batch")
    from .model import OAVLA
    from .checkpoint import load_base_weights

    if args.resume:
        model = OAVLA.from_pretrained(args.resume, device=device)
        model.config = config
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.detection_expert.decoder.use_act_checkpoint = config.gradient_checkpointing
    else:
        model = OAVLA(config).to(device)
        if args.pretrained:
            print(load_base_weights(model, args.pretrained))
        if args.sam3_checkpoint:
            print({"sam3_loaded": model.detection_expert.load_sam3_weights(args.sam3_checkpoint)})
        elif rank == 0:
            print("[oa-vla] SAM 3 detection weights are randomly initialized and require training.")
    set_train_stage(model, args.stage, freeze_backbone=config.freeze_backbone)
    if config.dtype == "float16":
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.dtype == "float16")
    same_stage = args.resume and previous.get("training_stage") == args.stage and previous.get("freeze_backbone") == config.freeze_backbone
    start_epoch = 1
    if same_stage and (args.resume / "training_state.pt").exists():
        resume_state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        del resume_state
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.seed) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, drop_last=True)
    metadata = {**stats, "model_type": "oa-vla", "training_stage": args.stage,
                "freeze_backbone": config.freeze_backbone,
                "detection_trained": bool(previous.get("detection_trained", False)),
                "detection_views": ["front", "side"], "detection_format": "paired_obb_v1", "point_output": False,
                "sam3_checkpoint": str(args.sam3_checkpoint) if args.sam3_checkpoint else previous.get("sam3_checkpoint")}
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    completed_steps = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        totals, detection_supervised = np.zeros(3), False
        requested = min(len(loader), args.max_train_batches or len(loader))
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader, 1):
            start = (index - 1) // args.gradient_accumulation_steps * args.gradient_accumulation_steps + 1
            accumulation = min(args.gradient_accumulation_steps, requested - start + 1)
            should_step = index == start + accumulation - 1
            sync = model.no_sync() if is_distributed and not should_step else nullcontext()
            amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and config.dtype == "float16" else nullcontext()
            with sync, amp:
                losses = model(to_observation(batch, device), batch["actions"].to(device),
                               action_mask=batch["action_mask"].to(device),
                               detection_targets=batch["detection_targets"].to(device),
                               detection_valid=batch["detection_valid"].to(device),
                               detection_reviewed=batch["detection_reviewed"].to(device),
                               detection_loss_weight=args.detection_loss_weight, train_stage=args.stage)
                scaler.scale(losses["loss"] / accumulation).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
            totals += [float(losses["loss"].detach()), float(losses.get("action_loss", 0)),
                       float(losses.get("detection_loss", torch.tensor(0.)).detach())]
            detection_supervised |= args.stage != "action" and args.detection_loss_weight > 0 and bool(batch["detection_reviewed"].any())
            if index >= requested or should_step and args.max_steps and completed_steps >= args.max_steps:
                break
        aggregate = torch.tensor([*totals, index, float(detection_supervised)], device=device, dtype=torch.float64)
        if is_distributed:
            distributed.all_reduce(aggregate)
        metadata["detection_trained"] |= bool(aggregate[4] > 0)
        if rank == 0:
            means = (aggregate[:3] / aggregate[3]).tolist()
            print(f"epoch={epoch} updates={completed_steps} loss={means[0]:.6f} action={means[1]:.6f} detection={means[2]:.6f}")
            if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1 or args.max_steps and completed_steps >= args.max_steps:
                original = model.module if is_distributed else model
                OAVLAPolicy(original, tokenizer, metadata=metadata).save_pretrained(args.output)
                temporary = args.output / "training_state.pt.tmp"
                torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch}, temporary)
                temporary.replace(args.output / "training_state.pt")
        if args.max_steps and completed_steps >= args.max_steps:
            break
    if is_distributed:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
