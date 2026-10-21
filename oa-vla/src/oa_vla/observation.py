"""Low-level tensor inputs; public camera names are independent of any robot SDK."""

from dataclasses import dataclass, field
import torch


@dataclass
class Observation:
    images: dict[str, torch.Tensor]  # float32 NCHW in [-1, 1]
    state: torch.Tensor             # normalized and padded [B, action_dim]
    tokenized_prompt: torch.Tensor  # int64 [B,T], including discretized state
    tokenized_prompt_mask: torch.Tensor
    image_masks: dict[str, torch.Tensor] = field(default_factory=dict)
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

    def __post_init__(self):
        if self.state.ndim != 2 or self.tokenized_prompt.ndim != 2:
            raise ValueError("state and tokenized_prompt must be batch-first matrices")
        if self.tokenized_prompt.shape != self.tokenized_prompt_mask.shape:
            raise ValueError("Prompt tokens and mask must have the same shape")
        if self.state.shape[0] != self.tokenized_prompt.shape[0]:
            raise ValueError("Observation batch sizes differ")
        for name, image in self.images.items():
            if image.ndim != 4 or image.shape[:2] != (self.state.shape[0], 3):
                raise ValueError(f"{name} must be a normalized [B,3,H,W] RGB tensor")
            if name not in self.image_masks:
                self.image_masks[name] = torch.ones(self.state.shape[0], dtype=torch.bool, device=self.state.device)
