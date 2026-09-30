"""Canonical configuration loader for the Quest3 dual-UR7e safety pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_CONFIG = REPO_ROOT / "configs" / "ur7e_dual_quest3.yaml"


def load_project_config(path: Path | str = DEFAULT_PROJECT_CONFIG) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported or missing project schema in {config_path}")
    if config.get("scene") != "ur7e_dual_quest3" or config.get("coordinate_frame") != "left_base":
        raise ValueError("The canonical project config must describe ur7e_dual_quest3 in left_base")
    import math
    safety = config.get("safety", {})
    for name in ("stale_timeout_s", "max_joint_step_rad", "inter_arm_margin_m"):
        value = float(safety[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"safety.{name} must be finite and positive")
    for name in ("collision_margin_m", "trigger_margin_m"):
        value = float(safety[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"safety.{name} must be finite and nonnegative")
    if not 0 < float(safety["cbf_alpha"]) <= 1:
        raise ValueError("safety.cbf_alpha must be in (0,1]")
    return config, config_path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
