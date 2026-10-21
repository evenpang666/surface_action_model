"""Portable OA-VLA exports and explicit, strict initialization of pretrained weights."""

import json
from pathlib import Path

import torch

from .buffers import restore_nonpersistent_buffers
from .config import OAVLAConfig


def save_pretrained(model, directory, *, metadata=None):
    from safetensors.torch import save_model
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The safetensors helper records tied PaliGemma token/head aliases.
    temporary = directory / "model.safetensors.tmp"
    save_model(model, str(temporary))
    temporary.replace(directory / "model.safetensors")
    model.config.save(directory / "config.json")
    (directory / "metadata.json").write_text(json.dumps(metadata or {}, indent=2) + "\n", encoding="utf-8")
    return directory


def load_pretrained(directory, *, device="cpu"):
    from safetensors.torch import load_model
    from .model import OAVLA

    directory = Path(directory)
    config = OAVLAConfig.load(directory / "config.json")
    with torch.device("meta"):
        model = OAVLA(config)
    model.to_empty(device=device)
    # to_empty can split tied Parameter storage; re-establish it before loading.
    model.paligemma_with_expert.paligemma.tie_weights()
    load_model(model, str(directory / "model.safetensors"), strict=True, device=str(device))
    restore_nonpersistent_buffers(model)
    return model.eval()


def load_base_weights(model, path):
    """Initialize a PI0.5-compatible action backbone without importing its implementation."""
    from safetensors.torch import load_model

    path = Path(path)
    if path.suffix == ".safetensors":
        missing, unexpected = load_model(model, str(path), strict=False, device=str(next(model.parameters()).device))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
    invalid = [name for name in missing if not name.startswith("detection_expert.")]
    if invalid or unexpected:
        raise ValueError(f"Incompatible action backbone: missing={invalid}, unexpected={unexpected}")
    return {"loaded_backbone": True, "new_detection_tensors": len(missing)}
