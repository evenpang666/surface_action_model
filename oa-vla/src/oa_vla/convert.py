"""Migrate independent-BB weights to paired OBB initialization; retraining is required."""

import argparse
import json
from pathlib import Path
import torch

from .buffers import restore_nonpersistent_buffers
from .config import OAVLAConfig
from .model import OAVLA

NEW_PREFIXES = ("detection_expert.pair_fusion.", "detection_expert.pair_head.", "detection_expert.obb_head.")


def _reset_new_heads(model):
    head = model.detection_expert
    for root in (head.pair_fusion, head.pair_head, head.obb_head):
        for module in root.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            if isinstance(module, torch.nn.MultiheadAttention):
                module._reset_parameters()
    torch.nn.init.zeros_(head.obb_head.layers[-1].weight)
    torch.nn.init.zeros_(head.obb_head.layers[-1].bias)
    with torch.no_grad():
        head.obb_head.layers[-1].bias[4] = 1
        head.obb_head.layers[-1].bias[5] = .01


def convert_checkpoint(checkpoint, output, *, config=None, tokenizer_model=None):
    """Accept schema-5 .pt or OA-VLA format-1 directory, never claim paired training."""
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    source_export = checkpoint.is_dir()
    if source_export:
        raw = json.loads((checkpoint / "config.json").read_text())
        if raw.get("model_type") != "oa-vla" or raw.get("format_version") != 1:
            raise ValueError("Directory migration expects OA-VLA format version 1 (independent horizontal boxes)")
        raw["format_version"] = 2
        old_config = OAVLAConfig.from_dict(raw)
        if config is not None and config != old_config:
            raise ValueError("Requested configuration differs from source architecture")
        config = old_config
        info = json.loads((checkpoint / "metadata.json").read_text())
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("model_type") != "Sam3VlaModelPytorch" or payload.get("schema_version") != 5:
            raise ValueError("Expected schema-5 action/box checkpoint; point-flow checkpoints are incompatible")
        info = payload["metadata"]
        if set(info.get("camera_map", {})) != {"front", "side", "right_wrist", "left_wrist"} or info.get("detection_views") != ["front", "side"]:
            raise ValueError("Source checkpoint must use four RGB views and front/side detections")
        config = config or OAVLAConfig(
            action_horizon=int(info["action_horizon"]), logical_action_dim=int(info["logical_action_dim"]),
            state_dim=int(info["joint_dim"]), dtype=info["precision"],
            paligemma_variant=info.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=info.get("action_expert_variant", "gemma_300m"),
            detection_queries=int(info["detection_queries"]), detection_layers=int(info["detection_layers"]),
            detection_width=int(info.get("detection_width", 256)), detection_heads=int(info.get("detection_heads", 8)),
            detection_ffn_width=int(info.get("detection_ffn_width", 2048)))
        if (config.action_horizon, config.logical_action_dim, config.state_dim) != (
                info["action_horizon"], info["logical_action_dim"], info["joint_dim"]):
            raise ValueError("Source state/action dimensions differ from requested configuration")
    with torch.device("meta"):
        model = OAVLA(config)
    model.to_empty(device="cpu")
    model.paligemma_with_expert.paligemma.tie_weights()
    if source_export:
        from safetensors.torch import load_model
        missing, unexpected = load_model(model, str(checkpoint / "model.safetensors"), strict=False, device="cpu")
    else:
        missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    expected_new = {name for name in model.state_dict() if name.startswith(NEW_PREFIXES)}
    if set(missing) != expected_new or unexpected:
        raise ValueError(f"Incompatible independent-BB weights: missing={set(missing) - expected_new}, unexpected={unexpected}; "
                         "provide the exact architecture and an original independent-BB checkpoint")
    _reset_new_heads(model)
    restore_nonpersistent_buffers(model)
    keys = ("qpos_mean", "qpos_std", "action_mean", "action_std", "action_period_s", "action_semantics", "action_layout", "action_source")
    metadata = {key: info[key] for key in keys if key in info}
    metadata.update(model_type="oa-vla", source_checkpoint=str(checkpoint), detection_trained=False,
                    legacy_detection_trained=bool(info.get("detection_trained", False)),
                    detection_format="paired_obb_v1", detection_views=["front", "side"], point_output=False,
                    initialization_note="New pair fusion, pair scoring and OBB heads require supervised training")
    if source_export and (checkpoint / "processor.json").exists():
        import shutil
        # Export first; copy portable processor state, without loading/downloading a tokenizer.
        model.save_pretrained(output, metadata=metadata)
        for name in ("processor.json", "tokenizer.model"):
            if (checkpoint / name).exists():
                shutil.copy2(checkpoint / name, output / name)
    elif tokenizer_model:
        from .policy import OAVLAPolicy
        from .tokenizer import Tokenizer
        OAVLAPolicy(model, Tokenizer(config.max_token_len, tokenizer_model), metadata=metadata).save_pretrained(output)
    else:
        model.save_pretrained(output, metadata=metadata)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Old format-1 export directory or schema-5 .pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Exact architecture as format-2 configuration for schema-5 sources")
    parser.add_argument("--tokenizer-model", type=Path)
    args = parser.parse_args(argv)
    print(convert_checkpoint(args.checkpoint, args.output,
                             config=OAVLAConfig.load(args.config) if args.config else None, tokenizer_model=args.tokenizer_model))


if __name__ == "__main__":
    main()
