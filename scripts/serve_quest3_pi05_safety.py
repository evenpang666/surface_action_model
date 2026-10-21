#!/usr/bin/env python3
"""Serve a four-RGB SAM 3 + PI0.5 action/box checkpoint (schema 5, no points)."""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "openpi" / "src", REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from safety_module.policy_contract import CAMERA_TO_MODEL, DETECTION_VIEWS, MODEL_TYPE, SCHEMA_VERSION
from safety_module.sam3_policy_data import letterbox_rgb, original_xyxy_coordinates
from safety_module.tokenizer import _Tokenizer


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=.5)
    parser.add_argument("--tokenizer-model", type=Path)
    return parser.parse_args()


class Quest3Sam3VlaPolicy:
    def __init__(self, checkpoint, *, device_name="cuda", num_steps=10, tokenizer_model=None, score_threshold=.5):
        import torch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.sam3_vla_pytorch import Sam3VlaModelPytorch
        from safety_module.checkpoint_loading import restore_nonpersistent_buffers

        if num_steps < 1 or not 0 <= score_threshold <= 1:
            raise ValueError("num_steps must be positive and score_threshold must be in [0,1]")
        self.torch = torch
        self.device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        _verify_cuda_runtime(torch, self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("model_type") != MODEL_TYPE or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Expected a schema-5 SAM 3 VLA checkpoint; old point checkpoints require migration/retraining")
        self.info = dict(payload["metadata"])
        if self.info.get("camera_map") != CAMERA_TO_MODEL or tuple(self.info.get("detection_views", ())) != DETECTION_VIEWS:
            raise ValueError("Checkpoint must use four policy RGB views and front/side-only detections")
        self.joint_dim = int(self.info["joint_dim"])
        self.action_dim = int(self.info["logical_action_dim"])
        if (self.joint_dim, self.action_dim) not in ((6, 7), (12, 14)):
            raise ValueError("Checkpoint has an invalid UR7e state/action layout")
        for field, size in (("qpos_mean", self.joint_dim), ("qpos_std", self.joint_dim),
                            ("action_mean", self.action_dim), ("action_std", self.action_dim)):
            value = np.asarray(self.info[field], np.float32)
            if value.shape != (size,) or not np.isfinite(value).all() or (field.endswith("std") and (value <= 0).any()):
                raise ValueError(f"Invalid checkpoint normalization: {field}")
            setattr(self, field, value)
        config = Pi0Config(pi05=True, action_horizon=int(self.info["action_horizon"]),
                           dtype=self.info["precision"],
                           paligemma_variant=self.info.get("paligemma_variant", "gemma_2b"),
                           action_expert_variant=self.info.get("action_expert_variant", "gemma_300m"))
        kwargs = {key: int(self.info[key]) for key in (
            "detection_queries", "detection_layers", "detection_width", "detection_heads", "detection_ffn_width")}
        with torch.device("meta"):
            self.model = Sam3VlaModelPytorch(config, **kwargs)
        self.model.to_empty(device=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        restore_nonpersistent_buffers(self.model)
        del payload
        gc.collect()
        self.model.eval()
        self.tokenizer = _Tokenizer(max_len=200, model_path=tokenizer_model)
        self.num_steps, self.score_threshold = num_steps, score_threshold
        self.detection_ready = bool(self.info.get("detection_trained"))
        self.metadata = {key: self.info[key] for key in (
            "camera_map", "joint_dim", "logical_action_dim", "action_horizon", "action_period_s")}
        self.metadata.update(model_type=MODEL_TYPE, schema_version=SCHEMA_VERSION,
                             input_views=list(CAMERA_TO_MODEL), detection_views=list(DETECTION_VIEWS),
                             detection_box_format="xyxy_original_pixels", detection_ready=self.detection_ready,
                             joint_action_and_point_flow=False, point_output=False)

    def infer(self, request):
        torch = self.torch
        qpos = np.asarray(request["qpos"], np.float32)
        if qpos.shape != (self.joint_dim,) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must be finite [{self.joint_dim}]")
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt must be nonempty")
        missing = [f"rgb_{view}" for view in CAMERA_TO_MODEL if f"rgb_{view}" not in request]
        if missing:
            raise ValueError(f"All four RGB views are required; missing {missing}")
        images, shapes = {}, {}
        for view, key in CAMERA_TO_MODEL.items():
            rgb = np.asarray(request[f"rgb_{view}"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"rgb_{view} must be uint8 HxWx3 RGB")
            shapes[view] = rgb.shape[:2]
            padded = letterbox_rgb(rgb)
            images[key] = torch.from_numpy(padded.copy()).to(self.device)[None].permute(0, 3, 1, 2).float() / 127.5 - 1
        state = np.zeros(32, np.float32)
        state[:self.joint_dim] = (qpos - self.qpos_mean) / self.qpos_std
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        observation = SimpleNamespace(
            images=images, image_masks={key: torch.ones(1, dtype=torch.bool, device=self.device) for key in images},
            state=torch.from_numpy(state).to(self.device)[None],
            tokenized_prompt=torch.from_numpy(tokens.astype(np.int64)).to(self.device)[None],
            tokenized_prompt_mask=torch.from_numpy(token_mask).to(self.device)[None],
            token_ar_mask=None, token_loss_mask=None)
        with torch.inference_mode():
            prediction = self.model.sample_actions_and_boxes(self.device, observation, num_steps=self.num_steps)
        normalized = prediction["actions"][0, :, :self.action_dim].float().cpu().numpy()
        actions = (normalized * self.action_std + self.action_mean).astype(np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("Policy produced non-finite actions")
        detections = {view: [] for view in DETECTION_VIEWS}
        if self.detection_ready:
            for index, view in enumerate(DETECTION_VIEWS):
                boxes = original_xyxy_coordinates(prediction["boxes"][0, index].float().cpu().numpy(), *shapes[view])
                scores = prediction["scores"][0, index].float().cpu().numpy()
                if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                    raise ValueError("Policy produced non-finite detections")
                detections[view] = [{"xyxy": box.tolist(), "score": float(score)}
                                    for box, score in zip(boxes, scores, strict=True)
                                    if score >= self.score_threshold and box[2] > box[0] and box[3] > box[1]]
        return {"schema_version": SCHEMA_VERSION, "actions": actions, "action_chunk": actions,
                "detections": detections, "detection_box_format": "xyxy_original_pixels",
                "detection_status": "ok" if self.detection_ready else "not_trained"}

    def reset(self):
        return None


# Preserve the Python import name while switching its model/output contract.
Quest3JointSafetyPolicy = Quest3Sam3VlaPolicy


def main():
    args = parse_args()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = Quest3Sam3VlaPolicy(args.checkpoint, device_name=args.device, num_steps=args.num_steps,
                                tokenizer_model=args.tokenizer_model, score_threshold=args.score_threshold)
    logging.info("Serving SAM 3 VLA on %s:%d", socket.gethostname(), args.port)
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
