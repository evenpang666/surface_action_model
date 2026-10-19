"""PI0.5 action + surface flow + RGB obstacle OBB detection.

The detector reuses the frozen PI0.5 image encoder. Its two views do not
occupy PI0.5's action-policy camera slots: front/side are encoded separately
with shared weights and decoded as independent sets of current-frame boxes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
from openpi.shared import image_tools


DETECTION_VIEWS = ("front", "side")
DETECTION_RESOLUTION = 224


def padded_obb_coordinates(boxes: torch.Tensor, height: int, width: int, resolution: int = DETECTION_RESOLUTION) -> torch.Tensor:
    """Pixel cx,cy,w,h,clockwise-degrees -> normalized padded image coordinates.

    The integer resize and symmetric padding mirror resize_with_pad_torch.
    Angle is encoded as cos(2 theta), sin(2 theta) for a 180-degree OBB.
    """
    ratio = max(width / resolution, height / resolution)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    top, left = (resolution - resized_h) // 2, (resolution - resized_w) // 2
    result = torch.empty((*boxes.shape[:-1], 6), dtype=boxes.dtype, device=boxes.device)
    result[..., 0] = (boxes[..., 0] * resized_w / width + left) / resolution
    result[..., 1] = (boxes[..., 1] * resized_h / height + top) / resolution
    swap = boxes[..., 2] < boxes[..., 3]
    canonical_width = torch.where(swap, boxes[..., 3], boxes[..., 2])
    canonical_height = torch.where(swap, boxes[..., 2], boxes[..., 3])
    result[..., 2] = canonical_width * resized_w / width / resolution
    result[..., 3] = canonical_height * resized_h / height / resolution
    angle = (boxes[..., 4] + swap.to(boxes.dtype) * 90) * (2 * math.pi / 180)
    result[..., 4], result[..., 5] = angle.cos(), angle.sin()
    return result


def original_obb_coordinates(boxes: torch.Tensor, height: int, width: int, resolution: int = DETECTION_RESOLUTION) -> torch.Tensor:
    """Normalized padded cx,cy,w,h,cos2θ,sin2θ -> original RGB pixel OBB."""
    ratio = max(width / resolution, height / resolution)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    top, left = (resolution - resized_h) // 2, (resolution - resized_w) // 2
    result = torch.empty((*boxes.shape[:-1], 5), dtype=boxes.dtype, device=boxes.device)
    result[..., 0] = (boxes[..., 0] * resolution - left) * width / resized_w
    result[..., 1] = (boxes[..., 1] * resolution - top) * height / resized_h
    result[..., 2] = boxes[..., 2] * resolution * width / resized_w
    result[..., 3] = boxes[..., 3] * resolution * height / resized_h
    result[..., 4] = torch.remainder(torch.atan2(boxes[..., 5], boxes[..., 4]) * (90 / math.pi), 180)
    return result


class PI05DetectionExpert(nn.Module):
    """Set predictor over PI0.5 image tokens, shared across camera views."""

    def __init__(self, image_width: int, *, width: int = 256, queries: int = 32, layers: int = 3):
        super().__init__()
        self.queries = queries
        self.image_projection = nn.Linear(image_width, width)
        self.positions = nn.Embedding(1024, width)
        self.query_tokens = nn.Embedding(queries, width)
        decoder_layer = nn.TransformerDecoderLayer(width, nhead=8, dim_feedforward=width * 4,
                                                    dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.objectness = nn.Linear(width, 1)
        self.geometry = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 6))
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = image_tokens.shape
        if count > self.positions.num_embeddings:
            raise ValueError(f"PI0.5 image has {count} patches; detector supports at most 1024")
        positions = self.positions(torch.arange(count, device=image_tokens.device))[None]
        memory = self.image_projection(image_tokens.float()) + positions
        queries = self.query_tokens.weight[None].expand(batch, -1, -1)
        hidden = self.decoder(queries, memory)
        geometry = self.geometry(hidden)
        xywh = geometry[..., :4].sigmoid()
        angle = F.normalize(geometry[..., 4:6], dim=-1, eps=1e-6)
        return self.objectness(hidden).squeeze(-1), torch.cat((xywh, angle), dim=-1)


def detection_set_loss(logits: torch.Tensor, boxes: torch.Tensor, targets: torch.Tensor,
                       valid: torch.Tensor, reviewed: torch.Tensor) -> dict[str, torch.Tensor]:
    """Hungarian matching; unreviewed frames contribute exactly zero loss."""
    from scipy.optimize import linear_sum_assignment

    if logits.shape != boxes.shape[:-1] or boxes.shape[-1] != 6:
        raise ValueError("Detection logits and boxes must have [B,2,Q] and [B,2,Q,6] shapes")
    if targets.shape[-1] != 6 or valid.shape != targets.shape[:-1] or reviewed.shape != logits.shape[:2]:
        raise ValueError("Detection target, valid, or reviewed mask shape mismatch")
    total_objectness = logits.sum() * 0
    total_box = boxes.sum() * 0
    matched = 0
    reviewed_views = 0
    for batch in range(logits.shape[0]):
        for view in range(logits.shape[1]):
            if not bool(reviewed[batch, view]):
                continue
            reviewed_views += 1
            truth = targets[batch, view][valid[batch, view]]
            labels = torch.zeros_like(logits[batch, view])
            if len(truth):
                if len(truth) > logits.shape[2]:
                    raise ValueError("More ground-truth boxes than detection queries")
                xywh_cost = torch.cdist(boxes[batch, view, :, :4].detach(), truth[:, :4], p=1)
                aspect = ((truth[:, 2] - truth[:, 3]).abs() / (truth[:, 2] + truth[:, 3]).clamp_min(1e-6)).clamp_min(0.05)
                angle_cost = (1 - boxes[batch, view, :, 4:].detach() @ truth[:, 4:].T) * aspect[None]
                score_cost = -logits[batch, view].detach().sigmoid()[:, None]
                cost = xywh_cost + 0.5 * angle_cost + 0.25 * score_cost
                rows, columns = linear_sum_assignment(cost.float().cpu().numpy())
                row = torch.as_tensor(rows, device=logits.device)
                column = torch.as_tensor(columns, device=logits.device)
                labels[row] = 1
                predicted, selected_truth = boxes[batch, view, row], truth[column]
                xywh = F.smooth_l1_loss(predicted[:, :4], selected_truth[:, :4], reduction="sum")
                matched_aspect = aspect[column]
                angle = ((1 - (predicted[:, 4:] * selected_truth[:, 4:]).sum(-1)) * matched_aspect).sum()
                total_box = total_box + xywh + 0.5 * angle
                matched += len(rows)
            classification = F.binary_cross_entropy_with_logits(logits[batch, view], labels, reduction="none")
            weights = torch.where(labels.bool(), torch.ones_like(labels), torch.full_like(labels, 0.1))
            total_objectness = total_objectness + (classification * weights).sum() / logits.shape[2]
    denominator = max(reviewed_views, 1)
    return {"detection_loss": total_objectness / denominator + total_box / max(matched, 1),
            "detection_objectness_loss": total_objectness / denominator,
            "detection_box_loss": total_box / max(matched, 1)}


class SurfaceActionDetectionModelPytorch(SurfaceActionModelPytorch):
    schema_version = 4

    def __init__(self, config, *, joint_dim: int = 12, detection_queries: int = 32):
        super().__init__(config, joint_dim=joint_dim)
        image_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.detection_expert = PI05DetectionExpert(image_width, queries=detection_queries)

    def detect(self, images: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept normalized [-1,1] NCHW RGB tensors keyed by front/side."""
        if set(images) != set(DETECTION_VIEWS):
            raise ValueError(f"Detection images must contain {DETECTION_VIEWS}")
        tokens = []
        for view in DETECTION_VIEWS:
            rgb = images[view]
            if rgb.ndim != 4 or rgb.shape[1] != 3:
                raise ValueError(f"{view} RGB must be [B,3,H,W]")
            resized = image_tools.resize_with_pad_torch(rgb.float().permute(0, 2, 3, 1), DETECTION_RESOLUTION,
                                                        DETECTION_RESOLUTION).permute(0, 3, 1, 2)
            with torch.no_grad():
                token = self.paligemma_with_expert.embed_image(resized).detach()
            tokens.append(token)
        all_tokens = torch.stack(tokens, dim=1)
        batch, views, patches, width = all_tokens.shape
        logits, boxes = self.detection_expert(all_tokens.reshape(batch * views, patches, width))
        return logits.reshape(batch, views, -1), boxes.reshape(batch, views, -1, 6)

    def compute_detection_losses(self, images, targets, valid, reviewed):
        logits, boxes = self.detect(images)
        result = detection_set_loss(logits, boxes, targets, valid, reviewed)
        result.update(detection_logits=logits, detection_boxes=boxes)
        return result

    def compute_losses(self, observation, actions, robot_points, joint_positions, target_point_offsets,
                       target_point_mask=None, *, detection_images=None, detection_targets=None,
                       detection_valid=None, detection_reviewed=None, detection_loss_weight=1.0,
                       train_stage="joint", **kwargs):
        if train_stage not in ("joint", "action_point", "detection"):
            raise ValueError(f"Unknown train stage: {train_stage}")
        if train_stage == "detection":
            if any(item is None for item in (detection_images, detection_targets, detection_valid, detection_reviewed)):
                raise ValueError("Detection stage requires images and annotation targets")
            detected = self.compute_detection_losses(detection_images, detection_targets, detection_valid, detection_reviewed)
            return {"loss": detection_loss_weight * detected["detection_loss"], **detected}
        result = super().compute_losses(observation, actions, robot_points, joint_positions,
                                        target_point_offsets, target_point_mask, **kwargs)
        if train_stage == "joint":
            if any(item is None for item in (detection_images, detection_targets, detection_valid, detection_reviewed)):
                raise ValueError("Joint stage requires detection images and annotation targets")
            detected = self.compute_detection_losses(detection_images, detection_targets, detection_valid, detection_reviewed)
            result["loss"] = result["loss"] + detection_loss_weight * detected["detection_loss"]
            result.update(detected)
        return result
