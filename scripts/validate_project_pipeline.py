#!/usr/bin/env python3
"""Validate the canonical Quest3 → PI05 → OBB → dual CBF data contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler  # noqa: E402
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config  # noqa: E402
from safety_module.shards import open_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--shard", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--obstacle-url", default=None)
    parser.add_argument("--allow-unverified-right-tcp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_project_config(args.project_config)
    sampler = DualUR7eSurfacePointSampler(points_per_link=4, project_config=config_path)
    checks: dict[str, object] = {
        "project_config": str(config_path),
        "coordinate_frame": config["coordinate_frame"],
        "tcp_transform_verified": sampler.tcp_transform_verified,
        "right_base_to_left_base_finite": bool(np.isfinite(sampler.right_base_to_left_base).all()),
    }
    errors: list[str] = []
    if not sampler.tcp_transform_verified["right_arm"] and not args.allow_unverified_right_tcp:
        errors.append("right flange_to_active_tcp is not verified")
    if args.episode is not None:
        with h5py.File(args.episode, "r") as source:
            qpos = source["observations/qpos"]
            action = source["action"]
            arm_count = qpos.shape[1] // 6
            episode = {
                "path": str(args.episode.resolve()), "frames": qpos.shape[0], "arm_count": arm_count,
                "qpos_shape": qpos.shape, "action_shape": action.shape,
                "rgb_views": sorted(source["observations/images"]), "complete": bool(source.attrs.get("complete", False)),
            }
            checks["episode"] = episode
            if qpos.shape[1] not in (6, 12) or action.shape != qpos.shape or not episode["complete"]:
                errors.append("raw episode contract is invalid")
    if args.shard is not None:
        with open_shard(args.shard) as data:
            shard_points_per_link = int(np.asarray(data["points_per_link"]).item())
            shard_arm_count = int(np.asarray(data["arm_count"]).item())
            shard = {
                "path": str(args.shard.resolve()), "schema": str(data["schema"]),
                "coordinate_frame": str(data["coordinate_frame"]), "samples": len(data["sample_frame_indices"]),
                "qpos_shape": data["qpos"].shape, "action_chunks_shape": data["action_chunks"].shape,
                "current_points_shape": data["current_link_points"].shape,
                "target_offsets_shape": data["target_point_offsets"].shape,
                "arm_count": shard_arm_count, "points_per_link": shard_points_per_link,
            }
            checks["shard"] = shard
            actions, current, offsets = data["action_chunks"], data["current_link_points"], data["target_point_offsets"]
            if str(data["coordinate_frame"]) != "left_base" or offsets.shape[:2] != actions.shape[:2] or offsets.shape[2:] != current.shape[1:]:
                errors.append("processed shard contract is invalid")
            shard_sampler = DualUR7eSurfacePointSampler(points_per_link=shard_points_per_link, project_config=config_path)
            surface_arm_count = int(data["surface_arm_count"]) if "surface_arm_count" in data else shard_arm_count
            if str(np.asarray(data["surface_model_hash"]).item()) != shard_sampler.model_hash(surface_arm_count):
                errors.append("processed shard surface hash differs from current robot/TCP configuration")
    if args.checkpoint is not None:
        import torch

        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        checks["checkpoint"] = {
            "path": str(args.checkpoint.resolve()), "model_type": payload.get("model_type"),
            "logical_action_dim": metadata.get("logical_action_dim"), "joint_dim": metadata.get("joint_dim"),
            "action_horizon": metadata.get("action_horizon"), "point_count": len(metadata.get("selected_point_indices", [])),
        }
        if payload.get("model_type") not in ("PI05SafetyPytorch", "SurfaceActionModelPytorch") or metadata.get("coordinate_frame") != "left_base":
            errors.append("joint PI05 checkpoint contract is invalid")
    if args.obstacle_url is not None:
        with urlopen(args.obstacle_url, timeout=2.0) as response:
            snapshot = json.loads(response.read())
        checks["obstacle"] = {
            "schema": snapshot.get("schema"), "coordinate_frame": snapshot.get("coordinate_frame"),
            "frame": snapshot.get("frame"), "obbs": len(snapshot.get("obbs", [])),
        }
        from safety_module.runtime_contract import validate_snapshot
        try:
            validate_snapshot(snapshot, float(config["safety"]["stale_timeout_s"]))
        except (ValueError, RuntimeError, KeyError) as exc:
            errors.append(str(exc))
        if snapshot.get("schema") != "dual_ur7e_obstacle_snapshot_v2" or snapshot.get("coordinate_frame") != "left_base":
            errors.append("obstacle snapshot contract is invalid")
    print(json.dumps({"ok": not errors, "checks": checks, "errors": errors}, ensure_ascii=False, indent=2, default=list))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
