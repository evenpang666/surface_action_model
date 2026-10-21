"""Serializable model configuration with no framework or robotics dependencies."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class TextArchitecture:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_text_config(name):
    variants = {
        "gemma_2b": TextArchitecture(2048, 18, 16384, 8, 1, 256),
        "gemma_300m": TextArchitecture(1024, 18, 4096, 8, 1, 256),
        "dummy": TextArchitecture(64, 4, 128, 8, 1, 16),
        "tiny_vlm": TextArchitecture(64, 2, 128, 4, 1, 16),
        "tiny_action": TextArchitecture(32, 2, 64, 4, 1, 16),
    }
    try:
        return variants[name]
    except KeyError as exc:
        raise ValueError(f"Unknown text architecture {name!r}; choose from {tuple(variants)}") from exc


@dataclass(frozen=True)
class OAVLAConfig:
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    action_dim: int = 32
    logical_action_dim: int = 14
    state_dim: int = 12
    action_horizon: int = 8
    max_token_len: int = 200
    vocab_size: int = 257152
    image_size: int = 224
    vision_patch_size: int = 14
    vision_width: int = 1152
    vision_layers: int = 27
    vision_heads: int = 16
    vision_mlp_dim: int = 4304
    detection_width: int = 256
    detection_queries: int = 200
    detection_layers: int = 6
    detection_heads: int = 8
    detection_ffn_width: int = 2048
    freeze_backbone: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is int and value < 1:
                raise ValueError(f"{field.name} must be positive")
        if not 1 <= self.logical_action_dim <= self.action_dim or not 1 <= self.state_dim <= self.action_dim:
            raise ValueError("Logical action/state dimensions must fit the padded action dimension")
        if self.image_size != 224:
            raise ValueError("Current BB preprocessing uses 224-pixel letterboxed images")
        if self.image_size % self.vision_patch_size or self.vision_width % self.vision_heads:
            raise ValueError("Image/patch and vision width/head dimensions must divide evenly")
        if self.detection_width % 4 or self.detection_width % self.detection_heads:
            raise ValueError("Detection width must be divisible by four and the head count")
        first, second = get_text_config(self.paligemma_variant), get_text_config(self.action_expert_variant)
        if (first.depth, first.num_heads, first.num_kv_heads, first.head_dim) != (
                second.depth, second.num_heads, second.num_kv_heads, second.head_dim):
            raise ValueError("VLM and action expert must share attention layer/head dimensions")

    @property
    def pi05(self):
        return True

    def to_dict(self):
        return {"model_type": "oa-vla", "format_version": 2, **asdict(self)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("model_type", "oa-vla") != "oa-vla" or value.pop("format_version", 2) != 2:
            raise ValueError("Expected an oa-vla configuration, format version 2 (paired OBB); migrate old weights with oa-vla-convert")
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def tiny(cls, **overrides):
        """Small random model for CPU tests, never a substitute for pretrained weights."""
        options = dict(paligemma_variant="tiny_vlm", action_expert_variant="tiny_action", dtype="float32",
                       action_horizon=2, max_token_len=32, vocab_size=512, vision_patch_size=112,
                       vision_width=32, vision_layers=1, vision_heads=4, vision_mlp_dim=64,
                       detection_width=32, detection_queries=4, detection_layers=2,
                       detection_heads=4, detection_ffn_width=64, gradient_checkpointing=False)
        options.update(overrides)
        return cls(**options)
