"""Point-free RGB/action shards and bounding-box coordinate transforms."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .annotations import load_detection_annotations, MAX_BOXES
from .contract import CAMERA_TO_MODEL, POLICY_VIEWS, DETECTION_VIEWS, IMAGE_RESOLUTION
from .shards import SampleArray, open_shard


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
    """Load rotated boxes and align views by explicit object IDs, never position."""
    from .contract import BOX_FORMAT
    inline = ("detection_boxes", "detection_valid", "detection_reviewed", "detection_object_ids")
    n = len(frames)
    if not any(key in data for key in inline):
        if "source_episode" in data:
            return load_detection_annotations(path, data, frames, rgb_shapes)
        return (np.zeros((n, 2, MAX_BOXES, 6), np.float32), np.zeros((n, 2, MAX_BOXES), bool),
                np.zeros((n, 2), bool), np.full((n, MAX_BOXES), "", dtype="U64"))
    if not all(key in data for key in inline):
        raise ValueError(f"{path}: paired inline labels require {inline}; IDs cannot be inferred from slot order")
    if "detection_box_format" not in data or str(data["detection_box_format"].item()) != BOX_FORMAT:
        raise ValueError(f"{path}: declare detection_box_format={BOX_FORMAT}; old horizontal boxes lost OBB angles")
    if "detection_views" not in data or tuple(data["detection_views"].tolist()) != DETECTION_VIEWS:
        raise ValueError(f"{path}: inline detection_views must be front, side")
    boxes = np.asarray(data["detection_boxes"], np.float32)
    valid = np.asarray(data["detection_valid"], bool)
    reviewed = np.asarray(data["detection_reviewed"], bool)
    ids = np.asarray(data["detection_object_ids"])
    if (boxes.ndim != 4 or boxes.shape[:2] != (n, 2) or boxes.shape[-1] != 6
            or valid.shape != boxes.shape[:-1] or reviewed.shape != boxes.shape[:2] or ids.shape != valid.shape):
        raise ValueError(f"{path}: expected [N,2,M,6] boxes, [N,2,M] valid/IDs and [N,2] reviewed")
    if ids.dtype.kind not in "USiu":
        raise ValueError(f"{path}: object IDs must be strings or integers")
    # Ignore unreviewed labels entirely, including stale geometry or IDs.
    valid = valid & reviewed[..., None]
    truth = boxes[valid]
    if (not np.isfinite(truth).all() or np.any((truth[:, :2] < 0) | (truth[:, :2] > 1))
            or np.any(truth[:, 2:4] <= 0) or not np.allclose(np.linalg.norm(truth[:, 4:6], axis=-1), 1, atol=1e-3)):
        raise ValueError(f"{path}: invalid normalized rotated targets")
    padded = np.zeros((n, 2, MAX_BOXES, 6), np.float32)
    padded_valid = np.zeros(padded.shape[:-1], bool)
    names = np.full((n, MAX_BOXES), "", dtype="U64")
    for sample in range(n):
        mappings = []
        for view in range(2):
            mapping = {}
            for slot in np.flatnonzero(valid[sample, view]):
                raw = ids[sample, view, slot]
                name = (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()
                if not name or len(name) > 64 or name in mapping:
                    raise ValueError(f"{path}: empty/duplicate/long object ID in sample {sample}, view {view}")
                mapping[name] = slot
            mappings.append(mapping)
        ordered = sorted(set(mappings[0]) | set(mappings[1]))
        if len(ordered) > MAX_BOXES:
            raise ValueError(f"{path}: at most {MAX_BOXES} object pairs per frame")
        names[sample, :len(ordered)] = ordered
        for target_slot, name in enumerate(ordered):
            for view in range(2):
                if name in mappings[view]:
                    box = boxes[sample, view, mappings[view][name]].copy()
                    if box[2] < box[3]:
                        box[2], box[3] = box[3], box[2]
                        box[4:6] *= -1
                    padded[sample, view, target_slot] = box
                    padded_valid[sample, view, target_slot] = True
    return padded, padded_valid, reviewed, names


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
    object_ids: np.ndarray


class OAVlaDataset(Dataset):
    def __init__(self, paths, *, tokenizer, qpos_mean, qpos_std, action_mean, action_std,
                 action_horizon=None, reviewed_only=False, include_detection=True, padded_action_dim=32):
        self.tokenizer = tokenizer
        self.padded_action_dim = padded_action_dim
        self.qpos_mean, self.qpos_std = np.asarray(qpos_mean), np.asarray(qpos_std)
        self.action_mean, self.action_std = np.asarray(action_mean), np.asarray(action_std)
        if max(len(self.qpos_mean), len(self.action_mean)) > padded_action_dim:
            raise ValueError("State/action dimensions exceed padded_action_dim")
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
                    boxes, valid, reviewed, object_ids = load_box_targets(path, data, frames, {
                        view: rgb[view].shape[1:] for view in DETECTION_VIEWS})
                else:
                    boxes = np.zeros((len(frames), 2, MAX_BOXES, 6), np.float32)
                    valid, reviewed = np.zeros(boxes.shape[:-1], bool), np.zeros((len(frames), 2), bool)
                    object_ids = np.full((len(frames), MAX_BOXES), "", dtype="U64")
                task = str(data["task_text"].item()).strip()
                if not task:
                    raise ValueError(f"{path}: empty task_text")
            episode = RgbActionEpisode(Path(path), task, qpos, actions, step_mask, rgb, boxes, valid, reviewed, object_ids)
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
        state = np.zeros(self.padded_action_dim, np.float32)
        state[:len(self.qpos_mean)] = (ep.qpos[sample] - self.qpos_mean) / self.qpos_std
        tokens, token_mask = self.tokenizer.tokenize(ep.task_text, state)
        actions = np.zeros((ep.actions.shape[1], self.padded_action_dim), np.float32)
        normalized = (ep.actions[sample] - self.action_mean) / self.action_std
        actions[:, :len(self.action_mean)] = np.where(ep.step_mask[sample, :, None], normalized, 0)
        return {"images": {CAMERA_TO_MODEL[view]: letterbox_rgb(ep.rgb[view][sample]) for view in POLICY_VIEWS},
                "image_masks": {key: np.asarray(True) for key in CAMERA_TO_MODEL.values()},
                "state": state, "tokenized_prompt": tokens.astype(np.int64), "tokenized_prompt_mask": token_mask,
                "actions": actions,
                "action_mask": ep.step_mask[sample, :, None] & (np.arange(self.padded_action_dim) < len(self.action_mean))[None],
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
