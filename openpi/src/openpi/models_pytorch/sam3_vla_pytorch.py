"""PI0.5 actions and SAM 3 boxes, with strict wrist-to-detector isolation.

Actions use all four cameras. The shared PaliGemma is run separately on a
front/side/text-only prefix for detection: slicing an all-view contextualized
prefix would leak wrist information. No point inputs, modules or outputs exist.
"""

from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.models_pytorch.sam3_detector.decoder import TransformerDecoder, TransformerDecoderLayer
from openpi.models_pytorch.sam3_detector.model_misc import DotProductScoring, MLP, inverse_sigmoid
from openpi.models_pytorch.sam3_detector.position_encoding import PositionEmbeddingSine
from safety_module.policy_contract import DETECTION_IMAGE_KEYS, POLICY_IMAGE_KEYS, SCHEMA_VERSION


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), -1)


def generalized_box_iou(first, second):
    """Pairwise GIoU for normalized cxcywh boxes, with finite empty-area math."""
    a, b = cxcywh_to_xyxy(first), cxcywh_to_xyxy(second)
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None] - intersection
    enclosing = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) -
                 torch.minimum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosing - union) / enclosing.clamp_min(1e-7)


def sam3_set_loss(prediction, targets, valid, reviewed):
    """One-to-one matching with separate global presence and local matching.

    Unreviewed views are ignored, reviewed-empty views train absence. The
    existing obstacle annotations are exhaustive sets conditioned on task text;
    this does not invent task-target/role annotations from obstacle labels.
    """
    from scipy.optimize import linear_sum_assignment

    boxes, logits, presence = prediction["boxes"], prediction["logits"], prediction["presence_logits"]
    if (targets.ndim != 4 or targets.shape[:2] != boxes.shape[:2] or targets.shape[-1] != 4
            or valid.shape != targets.shape[:-1] or reviewed.shape != boxes.shape[:2]):
        raise ValueError("Expected targets[B,2,M,4], valid[B,2,M], reviewed[B,2]")
    if not torch.isfinite(targets[valid]).all() or not torch.isfinite(boxes).all():
        raise ValueError("Detection boxes must be finite")
    if ((targets[valid] < 0) | (targets[valid] > 1)).any() or (targets[valid][..., 2:] <= 0).any():
        raise ValueError("Valid detection targets must be normalized cxcywh with positive sizes")
    box_loss, match_loss, presence_loss = boxes.sum() * 0, logits.sum() * 0, presence.sum() * 0
    matched = positive_views = reviewed_views = 0
    for batch in range(boxes.shape[0]):
        for view in range(2):
            if not bool(reviewed[batch, view]):
                continue
            reviewed_views += 1
            truth = targets[batch, view][valid[batch, view]]
            has_objects = bool(len(truth))
            presence_loss = presence_loss + F.binary_cross_entropy_with_logits(
                presence[batch, view], presence.new_tensor(float(has_objects)))
            if not has_objects:
                continue
            positive_views += 1
            if len(truth) > boxes.shape[2]:
                raise ValueError("More annotated boxes than SAM 3 object queries")
            predicted = boxes[batch, view]
            with torch.no_grad():
                cost = (5 * torch.cdist(predicted.float(), truth.float(), p=1)
                        - 2 * generalized_box_iou(predicted.float(), truth.float())
                        - logits[batch, view].float().sigmoid()[:, None])
                rows, columns = linear_sum_assignment(cost.cpu().numpy())
            rows = torch.as_tensor(rows, device=boxes.device)
            columns = torch.as_tensor(columns, device=boxes.device)
            labels = torch.zeros_like(logits[batch, view])
            labels[rows] = 1
            weights = torch.where(labels.bool(), 1., .1)
            match_loss = match_loss + (F.binary_cross_entropy_with_logits(
                logits[batch, view], labels, reduction="none") * weights).mean()
            selected, truth = predicted[rows], truth[columns]
            box_loss = box_loss + 5 * F.l1_loss(selected, truth, reduction="sum")
            box_loss = box_loss + 2 * (1 - generalized_box_iou(selected, truth).diag()).sum()
            matched += len(rows)
    box_loss = box_loss / max(matched, 1)
    match_loss = match_loss / max(positive_views, 1)
    presence_loss = presence_loss / max(reviewed_views, 1)
    return {"detection_loss": box_loss + match_loss + presence_loss,
            "detection_box_loss": box_loss, "detection_match_loss": match_loss,
            "detection_presence_loss": presence_loss}


class Sam3BoxDecoder(nn.Module):
    """Actual SAM 3 detection decoder, adapted to PaliGemma prefix features.

    The default decoder/scorer shapes match the original SAM 3 checkpoint.
    PE/fusion are replaced by PaliGemma and learned adapters; mask/tracker heads
    are absent. See sam3_detector/NOTICE.md for the upstream source revision.
    """

    def __init__(self, prefix_width, *, width=256, queries=200, layers=6, heads=8, ffn_width=2048):
        super().__init__()
        if width % 4 or width % heads or queries < 1 or layers < 1:
            raise ValueError("Invalid SAM 3 decoder dimensions")
        self.image_adapter = nn.Sequential(nn.LayerNorm(prefix_width), nn.Linear(prefix_width, width))
        self.text_adapter = nn.Sequential(nn.LayerNorm(prefix_width), nn.Linear(prefix_width, width))
        self.camera_embedding = nn.Embedding(2, width)
        self.position_encoding = PositionEmbeddingSine(width, normalize=True)
        layer = TransformerDecoderLayer(
            activation="relu", d_model=width, dim_feedforward=ffn_width, dropout=.1,
            cross_attention=nn.MultiheadAttention(width, heads, dropout=.1),
            n_heads=heads, use_text_cross_attention=True)
        self.decoder = TransformerDecoder(
            d_model=width, frozen=False, interaction_layer=None, layer=layer,
            num_layers=layers, num_queries=queries, return_intermediate=True,
            box_refine=True, dac=True, boxRPB="log", use_act_checkpoint=True,
            presence_token=True)
        self.dot_prod_scoring = DotProductScoring(width, width, prompt_mlp=MLP(
            width, ffn_width, width, 2, dropout=.1, residual=True, out_norm=nn.LayerNorm(width)))

    def _apply(self, function, recurse=True):
        # Upstream spatial caches are Python dictionaries, not registered
        # buffers. Drop them when moving/reallocating the model's tensors.
        self.position_encoding.cache.clear()
        self.decoder.coord_cache.clear()
        self.decoder.compilable_cord_cache = None
        self.decoder.compilable_stored_size = None
        return super()._apply(function, recurse=recurse)

    def load_sam3_weights(self, path: str | Path):
        """Load only native decoder/scoring weights; fail on incomplete heads.

        Adapters and camera embeddings are newly initialized. The complete SAM
        checkpoint may include unrelated PE/mask/video weights, which are ignored.
        """
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            state = state.get("model", state.get("state_dict", state))
        # Official video+image checkpoints nest the image detector here.
        if any(key.startswith("detector.") for key in state):
            state = {key.removeprefix("detector."): value for key, value in state.items()
                     if key.startswith("detector.")}
        report = {}
        for name, prefix in (("decoder", "transformer.decoder."), ("dot_prod_scoring", "dot_prod_scoring.")):
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not selected:
                raise ValueError(f"SAM 3 checkpoint contains no {prefix} tensors")
            getattr(self, name).load_state_dict(selected, strict=True)
            report[name] = len(selected)
        return report

    def forward(self, image_states, text_states, text_valid):
        batch, views, patches, _ = image_states.shape
        grid = math.isqrt(patches)
        if views != 2 or grid * grid != patches or text_valid.shape != text_states.shape[:2]:
            raise ValueError("Expected two square patch grids and aligned text tokens/mask")
        if not text_valid.any(-1).all():
            raise ValueError("SAM 3 needs at least one valid text token per sample")
        # This head stays in float32 even when the VLA runs in BF16/FP16.
        with torch.autocast(device_type=image_states.device.type, enabled=False):
            memory = self.image_adapter(image_states.float())
            camera = self.camera_embedding(torch.arange(2, device=memory.device))
            memory = memory + camera[None, :, None]
            memory = memory.reshape(batch * views, patches, -1).transpose(0, 1)
            prompt = self.text_adapter(text_states.float()).repeat_interleave(2, dim=0).transpose(0, 1)
            prompt_padding = ~text_valid.bool().repeat_interleave(2, dim=0)
            positions = self.position_encoding(memory.new_zeros(batch * views, 1, grid, grid))
            positions = positions.flatten(2).permute(2, 0, 1)
            queries = self.decoder.query_embed.weight[:, None].expand(-1, batch * views, -1)
            hidden, references, presence, _ = self.decoder(
                tgt=queries, memory=memory, pos=positions,
                spatial_shapes=torch.tensor([[grid, grid]], device=memory.device),
                valid_ratios=memory.new_ones(batch * views, 1, 2),
                level_start_index=torch.zeros(1, dtype=torch.long, device=memory.device),
                memory_text=prompt, text_attention_mask=prompt_padding,
                # Use the one-to-one queries for training and inference alike.
                # SAM 3's additional DAC losses are not part of this adaptation.
                apply_dac=False)
            hidden, references = hidden.transpose(1, 2), references.transpose(1, 2)
            all_boxes = (inverse_sigmoid(references) + self.decoder.bbox_embed(hidden)).sigmoid()
            all_logits = self.dot_prod_scoring(hidden, prompt, prompt_padding).squeeze(-1)
            all_presence = presence.squeeze(1)
            count, _, queries, _ = all_boxes.shape
            all_boxes = all_boxes.reshape(count, batch, 2, queries, 4)
            all_logits = all_logits.reshape(count, batch, 2, queries)
            all_presence = all_presence.reshape(count, batch, 2)
            return {"boxes": all_boxes[-1], "logits": all_logits[-1],
                    "presence_logits": all_presence[-1],
                    "scores": all_logits[-1].sigmoid() * all_presence[-1].sigmoid()[..., None],
                    "aux_boxes": all_boxes[:-1], "aux_logits": all_logits[:-1],
                    "aux_presence_logits": all_presence[:-1]}


class Sam3VlaModelPytorch(PI0Pytorch):
    schema_version = SCHEMA_VERSION

    def __init__(self, config, *, detection_queries=200, detection_layers=6,
                 detection_width=256, detection_heads=8, detection_ffn_width=2048):
        if not config.pi05:
            raise ValueError("SAM 3 VLA requires PI0.5")
        super().__init__(config)
        prefix_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.detection_expert = Sam3BoxDecoder(
            prefix_width, width=detection_width, queries=detection_queries, layers=detection_layers,
            heads=detection_heads, ffn_width=detection_ffn_width)
        self.paligemma_with_expert.paligemma.requires_grad_(False)

    def _preprocess_observation(self, observation, *, train=True):
        processed = preprocess_observation_pytorch(
            observation, train=train, image_keys=POLICY_IMAGE_KEYS)
        if any(not processed.image_masks[key].all() for key in POLICY_IMAGE_KEYS):
            raise ValueError("All four policy RGB views must be present; zero-filled views are not accepted")
        return (list(processed.images.values()), list(processed.image_masks.values()),
                processed.tokenized_prompt, processed.tokenized_prompt_mask, processed.state)

    def _detection_context(self, observation):
        # Select before encoding/fusion. Neither contextualized text nor visual
        # tokens from the four-view action prefix can enter the detector.
        processed = preprocess_observation_pytorch(
            observation, train=False, image_keys=DETECTION_IMAGE_KEYS)
        if any(not processed.image_masks[key].all() for key in DETECTION_IMAGE_KEYS):
            raise ValueError("Detection requires real front and side RGB")
        vlm_trainable = any(p.requires_grad for p in self.paligemma_with_expert.paligemma.parameters())
        with nullcontext() if vlm_trainable else torch.no_grad():
            prefix, padding, attention = self.embed_prefix(
                list(processed.images.values()), list(processed.image_masks.values()),
                processed.tokenized_prompt, processed.tokenized_prompt_mask)
            (context, _), _ = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(padding, attention)),
                position_ids=padding.long().cumsum(-1) - 1, inputs_embeds=[prefix, None], use_cache=False)
        text_count = processed.tokenized_prompt.shape[1]
        visual_count = context.shape[1] - text_count
        if visual_count % 2:
            raise ValueError("Detection prefix must contain equal front and side patch grids")
        return (context[:, :visual_count].reshape(context.shape[0], 2, visual_count // 2, -1),
                context[:, visual_count:], processed.tokenized_prompt_mask)

    def detect(self, observation):
        return self.detection_expert(*self._detection_context(observation))

    def compute_losses(self, observation, actions=None, *, detection_targets=None, detection_valid=None,
                       detection_reviewed=None, action_mask=None, detection_loss_weight=1.,
                       train_stage="joint", noise=None, time=None):
        if train_stage not in ("joint", "action", "detection"):
            raise ValueError(f"Unknown train stage: {train_stage}")
        if not math.isfinite(detection_loss_weight) or detection_loss_weight < 0:
            raise ValueError("detection_loss_weight must be finite and nonnegative")
        result = {}
        if train_stage != "detection":
            if actions is None or not torch.isfinite(actions).all():
                raise ValueError("Action training requires finite action targets")
            if actions.ndim != 3 or actions.shape[1:] != (self.config.action_horizon, self.config.action_dim):
                raise ValueError("Actions must match the configured horizon and padded action dimension")
            error = PI0Pytorch.forward(self, observation, actions, noise=noise, time=time)
            if action_mask is None:
                action_loss = error.mean()
            else:
                mask = action_mask.to(error).expand_as(error)
                action_loss = (error * mask).sum() / mask.sum().clamp_min(1)
            result.update(loss=action_loss, action_loss=action_loss.detach())
        if train_stage != "action":
            if any(value is None for value in (detection_targets, detection_valid, detection_reviewed)):
                raise ValueError("Detection training requires targets, valid and reviewed masks")
            prediction = self.detect(observation)
            losses = sam3_set_loss(prediction, detection_targets, detection_valid, detection_reviewed)
            aux_loss = prediction["aux_boxes"].sum() * 0
            for layer in range(len(prediction["aux_boxes"])):
                auxiliary = {key: prediction[f"aux_{key}"][layer]
                             for key in ("boxes", "logits", "presence_logits")}
                aux_loss = aux_loss + sam3_set_loss(
                    auxiliary, detection_targets, detection_valid, detection_reviewed)["detection_loss"]
            aux_loss = aux_loss / max(len(prediction["aux_boxes"]), 1)
            losses["detection_aux_loss"] = aux_loss
            losses["detection_loss"] = losses["detection_loss"] + aux_loss
            result["loss"] = result.get("loss", 0) + detection_loss_weight * losses["detection_loss"]
            result.update(losses)
        return result

    def forward(self, *args, **kwargs):
        return self.compute_losses(*args, **kwargs)

    @torch.no_grad()
    def sample_actions_and_boxes(self, device, observation, *, noise=None, num_steps=10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        actions = PI0Pytorch.sample_actions(self, device, observation, noise=noise, num_steps=num_steps)
        prediction = self.detect(observation)
        return {"actions": actions, **{key: prediction[key] for key in (
            "boxes", "scores", "presence_logits")}}
