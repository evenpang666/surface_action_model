"""Restore deterministic buffers omitted by torch state dictionaries."""

import torch


def restore_nonpersistent_buffers(model: torch.nn.Module) -> None:
    """Call after meta/to_empty/loading. Refuse unknown buffers, never guess."""
    for path, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            buffer = module._buffers.get(name)
            if buffer is None:
                continue
            device = buffer.device
            if name == "inv_freq" and hasattr(module, "rope_init_fn"):
                value, scaling = module.rope_init_fn(module.config, device)
                module.inv_freq = value
                module.original_inv_freq = value.clone()
                module.attention_scaling = scaling
            elif name == "position_ids" and buffer.ndim == 2:
                module.position_ids = torch.arange(buffer.shape[-1], device=device).expand(buffer.shape)
            else:
                raise RuntimeError(f"No deterministic buffer initializer for {path}.{name}")
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta" or not torch.isfinite(buffer).all():
            raise RuntimeError(f"Invalid model buffer after checkpoint load: {name}")
