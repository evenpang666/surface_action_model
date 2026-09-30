#!/usr/bin/env python3
"""Serve a jointly trained Quest3 PI05 action + surface-flow checkpoint."""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "openpi" / "src", REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from safety_module.tokenizer import MODEL_IMAGE_KEYS, _Tokenizer


def _verify_cuda_runtime(torch: object, device: object) -> None:
    """Fail at startup when this PyTorch wheel has no kernels for the GPU."""
    if getattr(device, "type", None) != "cuda":
        return
    try:
        properties = torch.cuda.get_device_properties(device)
        compiled_arches = torch.cuda.get_arch_list()
        # ``to(cuda)`` alone can succeed even when the first CUDA kernel cannot
        # run. Exercise the same elementary tensor path used by image ingest.
        torch.zeros((1,), device=device, dtype=torch.float32).add_(1).cpu()
    except RuntimeError as exc:
        name = getattr(locals().get("properties"), "name", "unknown")
        capability = getattr(locals().get("properties"), "major", "?")
        minor = getattr(locals().get("properties"), "minor", "?")
        arches = locals().get("compiled_arches", [])
        raise RuntimeError(
            "The installed PyTorch CUDA build cannot execute a basic kernel on "
            f"GPU {name} (compute capability sm_{capability}{minor}); "
            f"torch={torch.__version__}, torch CUDA={torch.version.cuda}, compiled architectures={arches}. "
            "Install a PyTorch build that supports this GPU (for H100, include sm_90; "
            "for RTX 50-series, include sm_120), then restart the policy server."
        ) from exc





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    return parser.parse_args()




class Quest3JointSafetyPolicy:
    def __init__(self, checkpoint: Path, *, device_name: str, num_steps: int, tokenizer_model: Path | None) -> None:
        import torch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI05SafetyPytorch
        from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch

        self.torch = torch
        self.device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        _verify_cuda_runtime(torch, self.device)
        # Keep parameter tensors memory-mapped until ``assign=True`` hands
        # their storage directly to the module.  A conventional torch.load +
        # copy produces a state dict, an initialized model and then a CUDA
        # model at once; that can exhaust both 32 GB host RAM and 16 GB GPUs.
        # The checkpoint is a zip-format torch.save file, which supports mmap.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("model_type") not in ("PI05SafetyPytorch", "SurfaceActionModelPytorch"):
            raise ValueError(f"{checkpoint} is not a jointly trained PI05SafetyPytorch checkpoint")
        self.info = dict(payload["metadata"])
        model_class = SurfaceActionModelPytorch if payload["model_type"] == "SurfaceActionModelPytorch" else PI05SafetyPytorch
        self.model_type = payload["model_type"]
        self.measured_surface = self.info.get("point_target") == "visual"
        required = ("action_mean", "action_std", "qpos_mean", "qpos_std", "action_horizon", "logical_action_dim", "joint_dim", "selected_point_indices", "camera_map")
        missing = [key for key in required if key not in self.info]
        if missing:
            raise ValueError(f"Checkpoint metadata is missing {missing}")
        self.action_mean = np.asarray(self.info["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.info["action_std"], dtype=np.float32)
        self.qpos_mean = np.asarray(self.info["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(self.info["qpos_std"], dtype=np.float32)
        self.logical_action_dim = int(self.info["logical_action_dim"])
        self.joint_dim = int(self.info["joint_dim"])
        self.point_indices = np.asarray(self.info["selected_point_indices"], dtype=np.int64)
        self.camera_map = {str(key): str(value) for key, value in self.info["camera_map"].items()}
        config = Pi0Config(
            action_dim=32, action_horizon=int(self.info["action_horizon"]), pi05=True,
            dtype=str(self.info.get("precision", "bfloat16")),
        )
        # Constructing directly on ``meta`` prevents an otherwise-unused
        # ~6 GB CPU parameter allocation.  ``to_empty`` allocates the final
        # parameter storage only on the target device; load_state_dict then
        # copies each memory-mapped tensor in turn, rather than retaining a
        # complete extra CUDA state dict.  Some rotary buffers are not stored
        # in the checkpoint, so assigning tensor storage directly is unsafe.
        with torch.device("meta"):
            self.model = model_class(config, joint_dim=self.joint_dim)
        self.model.to_empty(device=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        from safety_module.checkpoint_loading import restore_nonpersistent_buffers

        restore_nonpersistent_buffers(self.model)
        del payload
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model.eval()
        self.tokenizer = _Tokenizer(max_len=200, model_path=tokenizer_model)
        self.num_steps = int(num_steps)
        self.metadata = {
            "model_type": self.model_type, "joint_action_and_point_flow": True,
            "coordinate_frame": self.info.get("coordinate_frame", "left_base"),
            "action_horizon": int(self.info["action_horizon"]), "logical_action_dim": self.logical_action_dim,
            "joint_dim": self.joint_dim, "point_count": len(self.point_indices), "camera_map": self.camera_map,
            "surface_model_hash": self.info.get("surface_model_hash"),
            "point_identity_version": self.info.get("point_identity_version"),
            "points_per_link": self.info.get("points_per_link"),
            "point_target": self.info.get("point_target", "fixed"),
            "action_period_s": self.info.get("action_period_s"),
        }

    def _points(self, value: object) -> np.ndarray:
        points = np.asarray(value, dtype=np.float32)
        if points.ndim == 3 and points.shape[-1] == 3:
            points = points.reshape(-1, 3)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"robot_points must have shape [K,3] or [L,P,3], got {points.shape}")
        if self.measured_surface:
            if len(points) < len(self.point_indices):
                raise ValueError("Not enough measured current surface points for this checkpoint")
            selected = points[np.linspace(0, len(points) - 1, len(self.point_indices), dtype=np.int64)]
        elif len(points) == len(self.point_indices):
            selected = points
        elif len(points) > int(self.point_indices.max(initial=-1)):
            selected = points[self.point_indices]
        else:
            raise ValueError(f"robot_points has {len(points)} rows but checkpoint requires indices through {self.point_indices.max()}")
        if not np.isfinite(selected).all():
            raise ValueError("robot_points contains non-finite values")
        return selected

    def infer(self, obs: dict) -> dict:
        torch = self.torch
        qpos = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        if qpos.shape != (self.joint_dim,) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must be finite shape [{self.joint_dim}], got {qpos.shape}")
        normalized_qpos = (qpos - self.qpos_mean) / self.qpos_std
        state = np.zeros((32,), dtype=np.float32)
        state[: self.joint_dim] = normalized_qpos
        tokens, token_mask = self.tokenizer.tokenize(str(obs.get("prompt", "")), state)
        available: dict[str, np.ndarray] = {}
        for source, target in self.camera_map.items():
            key = f"rgb_{source}"
            if key in obs:
                image = np.asarray(obs[key], dtype=np.uint8)
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"{key} must be HxWx3 RGB")
                available[target] = image
        if set(available) != set(self.camera_map.values()):
            raise KeyError(f"All checkpoint cameras are required: {[f'rgb_{name}' for name in self.camera_map]}")
        shape = next(iter(available.values())).shape
        images, masks = {}, {}
        for key in MODEL_IMAGE_KEYS:
            image = available.get(key, np.zeros(shape, dtype=np.uint8))
            images[key] = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)[None].permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
            masks[key] = torch.tensor([key in available], dtype=torch.bool, device=self.device)
        observation = SimpleNamespace(
            images=images, image_masks=masks,
            state=torch.from_numpy(state).to(self.device)[None],
            tokenized_prompt=torch.from_numpy(tokens.astype(np.int64)).to(self.device)[None],
            tokenized_prompt_mask=torch.from_numpy(token_mask).to(self.device)[None],
            token_ar_mask=None, token_loss_mask=None,
        )
        current = self._points(obs.get("current_surface", obs.get("robot_points")))
        if self.measured_surface and obs.get("surface_source") != "measured_depth_fk_gated":
            raise ValueError("Measured-surface checkpoint requires measured depth input, never synthetic FK points")
        selected_indices = np.linspace(0, len(obs["current_surface"]) - 1, len(self.point_indices), dtype=np.int64) if self.measured_surface else self.point_indices
        robot_points = torch.from_numpy(current).to(self.device)[None]
        joints = torch.from_numpy(normalized_qpos).to(self.device)[None]
        with torch.inference_mode():
            normalized_actions, offsets = self.model.sample_actions_and_point_offsets(
                self.device, observation, robot_points, joints, num_steps=self.num_steps
            )
        normalized = normalized_actions[0, :, : self.logical_action_dim].float().cpu().numpy()
        actions = normalized * self.action_std[None, :] + self.action_mean[None, :]
        point_offsets = offsets[0].float().cpu().numpy()
        return {
            "schema_version": 3 if self.model_type == "SurfaceActionModelPytorch" else 2,
            "action_chunk": actions.astype(np.float32),
            "current_surface": current,
            "surface_chunk": (current[None, ...] + point_offsets).astype(np.float32),
            "actions": actions.astype(np.float32),
            "point_offsets": point_offsets.astype(np.float32),
            "predicted_robot_points": (current[None, ...] + point_offsets).astype(np.float32),
            "selected_point_indices": selected_indices,
            "surface_model_hash": self.info.get("surface_model_hash", ""),
            "point_identity_version": self.info.get("point_identity_version", ""),
            "points_per_link": int(self.info.get("points_per_link", 0)),
        }

    def reset(self) -> None:
        return None


def main() -> None:
    args = parse_args()
    from openpi.serving import websocket_policy_server

    policy = Quest3JointSafetyPolicy(args.checkpoint, device_name=args.device, num_steps=args.num_steps, tokenizer_model=args.tokenizer_model)
    logging.info("Serving joint Quest3 PI05 safety policy on %s:%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
