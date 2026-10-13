"""Read already RGB-aligned Quest3 HDF5 depth, without reprojection."""

from __future__ import annotations

import h5py
import numpy as np
import yaml

from real_scripts.camera_geometry import pixel_rays
from real_scripts.real_robot_adapter import CameraCalibration


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def recorded_intrinsics(spec: dict, *, width: int, height: int) -> np.ndarray:
    """Transform color-stream K through the recorder's centre crop and resize."""
    source_w, source_h = int(spec["width"]), int(spec["height"])
    if min(source_w, source_h, width, height) <= 0:
        raise ValueError("Camera image dimensions must be positive")
    K = np.array([[spec["fx"], 0., spec["ppx"]],
                  [0., spec["fy"], spec["ppy"]], [0., 0., 1.]], dtype=float)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    transform = spec.get("image_transform", "center_crop_resize")
    if transform not in ("none", "resize", "center_crop_resize"):
        raise ValueError(f"Unsupported recorded image transform: {transform}")
    if transform == "none" and (source_w, source_h) != (width, height):
        raise ValueError("Calibration dimensions do not match the recorded image")
    crop_w, crop_h = source_w, source_h
    if transform == "center_crop_resize":
        if source_w / source_h > width / height:
            crop_w = round(source_h * width / height)
            K[0, 2] -= (source_w - crop_w) // 2
        elif source_w / source_h < width / height:
            crop_h = round(source_w * height / width)
            K[1, 2] -= (source_h - crop_h) // 2
    if min(crop_w, crop_h) <= 0:
        raise ValueError("Recorded image crop is empty")
    K[0] *= width / crop_w
    K[1] *= height / crop_h
    return K


def _lens(spec):
    """Validate an SDK intrinsics record used by the calibration query tool."""
    coefficients = np.asarray(spec.get("distortion", spec.get("coeffs", np.zeros(5))), dtype=float)
    model = spec.get("distortion_model", "brown_conrady")
    pixel_rays(np.empty((0, 2)), np.eye(3), coefficients, model)
    return coefficients, model


class Quest3DepthReader:
    """Lazy reader for depth already expressed in recorded RGB pixels.

    Metadata must identify an aligned stream. Matching array dimensions alone
    cannot distinguish a native depth image from an aligned one.
    """

    def __init__(self, source: h5py.File, *, view: str, rgb_shape: tuple[int, int],
                 calibration: CameraCalibration, depth_unit_m: float | None = None):
        group = source[f"observations/images/{view}"]
        if "depth" not in group:
            raise ValueError(f"Missing {view} aligned depth dataset")
        self.dataset = group["depth"]
        if (self.dataset.ndim != 3 or len(self.dataset) == 0
                or tuple(self.dataset.shape[1:]) != tuple(rgb_shape)
                or self.dataset.dtype != np.uint16):
            raise ValueError("Aligned depth must be uint16 [N,H,W] matching recorded RGB dimensions")
        config = yaml.safe_load(_text(source.attrs.get("config_yaml", "{}"))) or {}
        if not isinstance(config, dict):
            raise ValueError("config_yaml must describe a mapping")
        camera = config.get("cameras", {}).get(view, {})
        topic = str(camera.get("depth_topic", ""))
        declared = _text(self.dataset.attrs.get("depth_alignment", self.dataset.attrs.get("coordinate_system", "")))
        if not declared:
            declared = _text(self.dataset.attrs.get("annotation_image", ""))
        if declared in ("native", "native_depth", "native_raw_depth") or (
                "/depth/image_rect_raw" in topic or "/depth/image_raw" in topic):
            raise ValueError("Native depth is not supported by this preprocessing path; record RGB-aligned depth")
        aligned = declared in ("aligned", "rgb", "color", "depth_to_color", "aligned_depth_to_color")
        aligned |= "aligned_depth_to_color" in topic
        if not aligned:
            raise ValueError("Depth coordinates are unknown; require recorded RGB-aligned depth metadata")
        scale = depth_unit_m if depth_unit_m is not None else self.dataset.attrs.get(
            "depth_scale_m", self.dataset.attrs.get("depth_unit_m", .001))
        self.depth_unit_m = float(scale)
        if not np.isfinite(self.depth_unit_m) or self.depth_unit_m <= 0:
            raise ValueError("Depth unit must be finite and positive")
        self.calibration = calibration
        self.metadata = {"input_mode": "recorded_aligned", "dataset": self.dataset.name,
                         "depth_unit_m": self.depth_unit_m, "output_coordinates": "color_optical_z_m",
                         "rasterization": "none", "rgb_shape": list(rgb_shape)}

    def read(self, index: int) -> np.ndarray:
        # Z16 conversion to metres is the only per-frame operation.
        return np.asarray(self.dataset[index], dtype=np.float32) * self.depth_unit_m
