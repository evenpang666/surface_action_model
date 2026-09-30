"""LingBot-Depth refinement for calibrated RealSense RGB-D frames.

The module intentionally imports PyTorch and LingBot-Depth only when a
``LingBotDepthRefiner`` is constructed.  This keeps the normal real-robot
tools usable on machines that only need raw D435i point clouds.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager
import os
import time
from typing import Any

import numpy as np

from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_robot_adapter import RGBDFrame


DEFAULT_LINGBOT_DEPTH_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
DEFAULT_LINGBOT_CAMERA_NAMES = ("front", "side")


@contextmanager
def _without_legacy_socks_all_proxy():
    """Allow httpx-based model loading with a legacy ``socks://`` setting.

    httpx rejects the legacy URL spelling before making a request.  Many lab
    shells also define HTTP(S)_PROXY for the same local proxy; in that case,
    temporarily hiding only ALL_PROXY retains proxy access for Hugging Face
    while avoiding the invalid fallback URL.  The caller's environment is
    restored immediately afterwards.
    """
    has_http_proxy = any(
        os.environ.get(name)
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
    )
    removed: dict[str, str] = {}
    if has_http_proxy:
        for name in ("ALL_PROXY", "all_proxy"):
            value = os.environ.get(name)
            if value and value.lower().startswith("socks://"):
                removed[name] = value
                os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.update(removed)


def add_lingbot_depth_cli_args(parser: Any) -> None:
    """Add the shared optional LingBot-Depth controls to an argparse parser."""
    parser.add_argument(
        "--lingbot-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine front/side RGB-D depth with LingBot-Depth before point-cloud fusion.",
    )
    parser.add_argument("--lingbot-model-id", default=DEFAULT_LINGBOT_DEPTH_MODEL)
    parser.add_argument(
        "--lingbot-camera-names",
        nargs="+",
        default=DEFAULT_LINGBOT_CAMERA_NAMES,
        help="Camera names used for the refined reconstruction (default: front side).",
    )
    parser.add_argument("--lingbot-device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--lingbot-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 model inference (CUDA only; disable for CPU inference).",
    )


def create_lingbot_depth_refiner_from_args(args: Any) -> LingBotDepthRefiner | None:
    """Construct a refiner only when ``--lingbot-depth`` was requested."""
    if not bool(getattr(args, "lingbot_depth", False)):
        return None
    return LingBotDepthRefiner(
        model_id=str(args.lingbot_model_id),
        camera_names=tuple(args.lingbot_camera_names),
        device=args.lingbot_device,
        use_fp16=bool(args.lingbot_fp16),
    )


def normalized_intrinsics(intrinsics: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Convert pixel-space intrinsics to the normalized LingBot-Depth format."""
    matrix = np.asarray(intrinsics, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {matrix.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("camera intrinsics fx and fy must be positive")
    normalized = matrix.copy()
    normalized[0] /= float(width)
    normalized[1] /= float(height)
    return normalized


class LingBotDepthRefiner:
    """Refine front/side RGB-D frames and return only those reconstruction views.

    Inputs must be RGB-aligned D435i depth maps expressed in metres.  The
    output keeps the same RGB pixels and image geometry, so it can be passed
    directly to :func:`real_scripts.real_robot_adapter.fuse_rgbd_frames`.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_LINGBOT_DEPTH_MODEL,
        camera_names: Sequence[str] = DEFAULT_LINGBOT_CAMERA_NAMES,
        device: str | None = None,
        use_fp16: bool = True,
        minimum_depth_m: float = 1e-4,
    ) -> None:
        names = tuple(str(name) for name in camera_names)
        if not names:
            raise ValueError("LingBot-Depth requires at least one camera name")
        if len(names) != len(set(names)):
            raise ValueError(f"LingBot-Depth camera names must be unique, got {names}")
        if minimum_depth_m <= 0.0:
            raise ValueError("minimum_depth_m must be > 0")
        try:
            import torch
            from mdm.model.v2 import MDMModel
        except ImportError as exc:
            raise ImportError(
                "LingBot-Depth is not installed. Install it in the same environment with "
                "'python -m pip install --no-deps -e ./lingbot-depth' after installing requirements/runtime.txt."
            ) from exc

        self.camera_names = names
        self.minimum_depth_m = float(minimum_depth_m)
        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_fp16 and self.device.type != "cuda":
            raise ValueError("--lingbot-fp16 requires a CUDA device; use --no-lingbot-fp16 for CPU inference")
        self.use_fp16 = bool(use_fp16)
        self.model_id = str(model_id)
        # Hugging Face Hub uses httpx.  Keep a malformed legacy SOCKS fallback
        # out of this process-local request without disabling HTTP(S) proxies.
        with _without_legacy_socks_all_proxy():
            self.model = MDMModel.from_pretrained(self.model_id).to(self.device).eval()
        # The released model defaults to nested-tensor attention, which needs
        # xFormers.  Use PyTorch SDPA when xFormers is unavailable (notably
        # CPU-only diagnostic machines) while preserving the same weights.
        try:
            from mdm.model.dinov2_rgbd.layers.attention import XFORMERS_AVAILABLE
        except ImportError:
            XFORMERS_AVAILABLE = False
        self._native_sdpa_mode = self.device.type != "cuda" or not XFORMERS_AVAILABLE
        if self._native_sdpa_mode:
            self.model.enable_pytorch_native_sdpa()
        self.last_inference_seconds = 0.0

    def _refine_one(self, frame: RGBDFrame, calibration: CameraCalibration) -> RGBDFrame:
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        depth_m = np.asarray(frame.depth_m, dtype=np.float32)
        height, width = depth_m.shape
        intrinsics = normalized_intrinsics(calibration.intrinsics, width=width, height=height)

        # LingBot accepts 0 or NaN for missing depth.  Convert non-finite D435i
        # samples to 0 while retaining the sensor's metre scale.
        depth_input = np.where(np.isfinite(depth_m) & (depth_m > 0.0), depth_m, 0.0).astype(np.float32)
        torch = self._torch
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        depth_tensor = torch.from_numpy(depth_input).to(self.device, dtype=torch.float32).unsqueeze(0)
        intrinsics_tensor = torch.from_numpy(intrinsics).to(self.device, dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            output = self.model.infer(
                image_tensor,
                depth_in=depth_tensor,
                intrinsics=intrinsics_tensor,
                use_fp16=self.use_fp16,
                # The upstream nested-token depth mask requires xFormers.
                # Standard tensor execution remains available with PyTorch
                # native SDPA while retaining RGB-D conditioning.
                enable_depth_mask=not self._native_sdpa_mode,
            )
        refined_depth = output["depth"].detach().float().cpu().numpy()
        if refined_depth.shape != (1, height, width):
            raise RuntimeError(
                f"LingBot-Depth returned depth shape {refined_depth.shape}; expected {(1, height, width)} for {frame.camera_name!r}"
            )
        refined_depth = refined_depth[0]
        refined_depth[~np.isfinite(refined_depth) | (refined_depth < self.minimum_depth_m)] = 0.0
        return RGBDFrame(
            frame.camera_name,
            rgb,
            refined_depth.astype(np.float32),
            host_timestamp_ns=frame.host_timestamp_ns,
            device_timestamp_ms=frame.device_timestamp_ms,
            frame_number=frame.frame_number,
            timestamp_domain=frame.timestamp_domain,
        )

    def refine(
        self,
        frames: Sequence[RGBDFrame],
        calibrations: dict[str, CameraCalibration],
    ) -> list[RGBDFrame]:
        """Run one synchronous refinement pass for the configured camera views."""
        by_name = {frame.camera_name: frame for frame in frames}
        missing_frames = [name for name in self.camera_names if name not in by_name]
        missing_calibrations = [name for name in self.camera_names if name not in calibrations]
        if missing_frames:
            raise KeyError(f"Missing RGB-D frames required by LingBot-Depth: {missing_frames}")
        if missing_calibrations:
            raise KeyError(f"Missing camera calibrations required by LingBot-Depth: {missing_calibrations}")

        started = time.perf_counter()
        refined = [self._refine_one(by_name[name], calibrations[name]) for name in self.camera_names]
        self.last_inference_seconds = time.perf_counter() - started
        return refined
