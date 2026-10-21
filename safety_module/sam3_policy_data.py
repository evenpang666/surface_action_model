"""Point-free RGB/action shards and bounding-box coordinate transforms."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from safety_module.detection_annotations import load_detection_annotations, MAX_BOXES
from safety_module.policy_contract import CAMERA_TO_MODEL, POLICY_VIEWS, DETECTION_VIEWS, IMAGE_RESOLUTION
from safety_module.shards import SampleArray, open_shard


def letterbox_geometry(height, width, resolution=IMAGE_RESOLUTION):
    if min(height, width) < 1:
        raise ValueError("RGB dimensions must be positive")
    ratio = max(width / resolution, height / resolution)
    rh, rw = max(1, int(height / ratio)), max(1, int(width / ratio))
    return rh, rw, (resolution - rh) // 2, (resolution - rw) // 2


def letterbox_rgb(image):
    """Uint8 HWC -> uint8 224-square HWC, matching OpenPI's resize geometry."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB must be uint8 [H,W,3]")
    height, width = image.shape[:2]
    rh, rw, top, left = letterbox_geometry(height, width)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, (rh, rw), mode="bilinear", align_corners=False)[0]
    result = torch.zeros(3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    result[:, top:top + rh, left:left + rw] = resized
    return result.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()


def original_xyxy_coordinates(boxes, height, width):
    """Normalized padded cxcywh -> clipped xyxy in the original RGB pixels."""
    boxes = np.asarray(boxes, np.float32)
    rh, rw, top, left = letterbox_geometry(height, width)
    result = np.concatenate((boxes[..., :2] - boxes[..., 2:] / 2,
                             boxes[..., :2] + boxes[..., 2:] / 2), -1)
    result[..., (0, 2)] = (result[..., (0, 2)] * IMAGE_RESOLUTION - left) * width / rw
    result[..., (1, 3)] = (result[..., (1, 3)] * IMAGE_RESOLUTION - top) * height / rh
    result[..., (0, 2)] = result[..., (0, 2)].clip(0, width)
    result[..., (1, 3)] = result[..., (1, 3)].clip(0, height)
    return result


def load_box_targets(path, data, frames, rgb_shapes):
    """Inline normalized targets or source-HDF5 OBB labels, in fixed view order."""
    inline = ("detection_boxes", "detection_valid", "detection_reviewed")
    if any(key in data for key in inline):
        if not all(key in data for key in inline):
            raise ValueError(f"{path}: inline detection labels require {inline}")
        if "detection_box_format" not in data or str(data["detection_box_format"].item()) != "cxcywh_normalized_padded":
            raise ValueError(f"{path}: declare detection_box_format=cxcywh_normalized_padded")
        if "detection_views" not in data or tuple(data["detection_views"].tolist()) != DETECTION_VIEWS:
            raise ValueError(f"{path}: inline detection_views must be front, side")
        boxes = np.asarray(data[inline[0]], np.float32)
        valid = np.asarray(data[inline[1]], bool)
        reviewed = np.asarray(data[inline[2]], bool)
    elif "source_episode" in data:
        boxes, valid, reviewed = load_detection_annotations(
            path, data, frames, rgb_shapes, axis_aligned=True)
    else:
        boxes = np.zeros((len(frames), 2, MAX_BOXES, 4), np.float32)
        valid = np.zeros(boxes.shape[:-1], bool)
        reviewed = np.zeros((len(frames), 2), bool)
    if (boxes.ndim != 4 or boxes.shape[:2] != (len(frames), 2) or boxes.shape[-1] != 4
            or valid.shape != boxes.shape[:-1] or reviewed.shape != boxes.shape[:2]):
        raise ValueError(f"{path}: detection shapes must be [N,2,M,4], [N,2,M], [N,2]")
    if boxes.shape[2] > MAX_BOXES:
        raise ValueError(f"{path}: at most {MAX_BOXES} ground-truth boxes per view are supported")
    truth = boxes[valid]
    if (not np.isfinite(truth).all() or np.any((truth < 0) | (truth > 1))
            or np.any(truth[:, 2:] <= 0)):
        raise ValueError(f"{path}: invalid normalized detection targets")
    # Pad variable annotation counts so shards can share a DataLoader batch.
    padded = np.zeros((len(frames), 2, MAX_BOXES, 4), np.float32)
    padded_valid = np.zeros(padded.shape[:-1], bool)
    padded[:, :, :boxes.shape[2]] = boxes
    padded_valid[:, :, :boxes.shape[2]] = valid
    return padded, padded_valid, reviewed


@dataclass
class RgbActionEpisode:
    path: Path
    task_text: str
    qpos: np.ndarray
    actions: np.ndarray
    step_mask: np.ndarray
    rgb: dict
    boxes: np.ndarray
    valid: np.ndarray
    reviewed: np.ndarray


class Sam3VlaDataset(Dataset):
    def __init__(self, paths, *, tokenizer, qpos_mean, qpos_std, action_mean, action_std,
                 action_horizon=None, reviewed_only=False, include_detection=True):
        self.tokenizer = tokenizer
        self.qpos_mean, self.qpos_std = np.asarray(qpos_mean), np.asarray(qpos_std)
        self.action_mean, self.action_std = np.asarray(action_mean), np.asarray(action_std)
        self.episodes, self.index = [], []
        for path in paths:
            with open_shard(path) as data:
                required = ("task_text", "qpos", "action_chunks", "sample_frame_indices",
                            *(f"rgb_{view}" for view in POLICY_VIEWS))
                missing = [key for key in required if key not in data]
                if missing:
                    raise ValueError(f"{path}: SAM 3 VLA requires four real RGB views; missing {missing}")
                actions = np.asarray(data["action_chunks"], np.float32)
                frames = np.asarray(data["sample_frame_indices"], np.int64)
                all_qpos = np.asarray(data["qpos"], np.float32)
                if (actions.ndim != 3 or actions.shape[-1] != len(self.action_mean)
                        or (action_horizon is not None and actions.shape[1] != action_horizon)):
                    raise ValueError(f"{path}: action dimensions/horizon differ from the policy contract")
                if (frames.shape != actions.shape[:1] or np.any(frames < 0) or np.any(frames >= len(all_qpos))
                        or all_qpos.ndim != 2 or all_qpos.shape[1] != len(self.qpos_mean)):
                    raise ValueError(f"{path}: invalid sample_frame_indices/qpos")
                step_mask = np.asarray(data["action_step_mask"], bool) if "action_step_mask" in data else np.ones(actions.shape[:2], bool)
                if step_mask.shape != actions.shape[:2] or not np.isfinite(actions[step_mask]).all():
                    raise ValueError(f"{path}: invalid action targets or step mask")
                qpos = all_qpos[frames]
                if not np.isfinite(qpos).all():
                    raise ValueError(f"{path}: qpos must be finite")
                rgb = {}
                for view in POLICY_VIEWS:
                    array = data[f"rgb_{view}"]
                    if (array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8
                            or len(array) != len(all_qpos) or min(array.shape[1:3]) < 1):
                        raise ValueError(f"{path}: rgb_{view} must be frame-aligned uint8 [T,H,W,3]")
                    rgb[view] = SampleArray(array, frames=frames)
                if include_detection:
                    boxes, valid, reviewed = load_box_targets(path, data, frames, {
                        view: rgb[view].shape[1:] for view in DETECTION_VIEWS})
                else:
                    boxes = np.zeros((len(frames), 2, MAX_BOXES, 4), np.float32)
                    valid, reviewed = np.zeros(boxes.shape[:-1], bool), np.zeros((len(frames), 2), bool)
                task = str(data["task_text"].item()).strip()
                if not task:
                    raise ValueError(f"{path}: empty task_text")
            episode = RgbActionEpisode(Path(path), task, qpos, actions, step_mask, rgb, boxes, valid, reviewed)
            episode_index = len(self.episodes)
            self.episodes.append(episode)
            self.index.extend((episode_index, sample) for sample in range(len(frames))
                              if (reviewed[sample].any() if reviewed_only else step_mask[sample].any()))
        if not self.index:
            raise ValueError("No eligible RGB/action samples in dataset")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode_index, sample = self.index[index]
        ep = self.episodes[episode_index]
        state = np.zeros(32, np.float32)
        state[:len(self.qpos_mean)] = (ep.qpos[sample] - self.qpos_mean) / self.qpos_std
        tokens, token_mask = self.tokenizer.tokenize(ep.task_text, state)
        actions = np.zeros((ep.actions.shape[1], 32), np.float32)
        normalized = (ep.actions[sample] - self.action_mean) / self.action_std
        actions[:, :len(self.action_mean)] = np.where(ep.step_mask[sample, :, None], normalized, 0)
        return {"images": {CAMERA_TO_MODEL[view]: letterbox_rgb(ep.rgb[view][sample]) for view in POLICY_VIEWS},
                "image_masks": {key: np.asarray(True) for key in CAMERA_TO_MODEL.values()},
                "state": state, "tokenized_prompt": tokens.astype(np.int64), "tokenized_prompt_mask": token_mask,
                "actions": actions,
                "action_mask": ep.step_mask[sample, :, None] & (np.arange(32) < len(self.action_mean))[None],
                "detection_targets": ep.boxes[sample], "detection_valid": ep.valid[sample],
                "detection_reviewed": ep.reviewed[sample]}


def to_observation(batch, device):
    return SimpleNamespace(
        images={key: batch["images"][key].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1
                for key in CAMERA_TO_MODEL.values()},
        image_masks={key: batch["image_masks"][key].to(device=device, dtype=torch.bool) for key in CAMERA_TO_MODEL.values()},
        state=batch["state"].to(device=device, dtype=torch.float32),
        tokenized_prompt=batch["tokenized_prompt"].to(device=device, dtype=torch.long),
        tokenized_prompt_mask=batch["tokenized_prompt_mask"].to(device=device, dtype=torch.bool),
        token_ar_mask=None, token_loss_mask=None)
