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

from .geometry import canonical_obbs, obb_enclosures, gaussian_overlap
from .flow import ActionFlowModel, make_att_2d_masks
from .preprocessing import preprocess_observation_pytorch
from .vendor.sam3.decoder import TransformerDecoder, TransformerDecoderLayer
from .vendor.sam3.model_misc import DotProductScoring, MLP, inverse_sigmoid
from .vendor.sam3.position_encoding import PositionEmbeddingSine
from .contract import DETECTION_IMAGE_KEYS, POLICY_IMAGE_KEYS, SCHEMA_VERSION


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
    """One Hungarian assignment per *object pair*, never per image.

    Targets[B,2,M,6] share object slots across views. Valid means visible with a
    box; reviewed means exhaustive annotation. Missing boxes in unreviewed views
    provide no absence supervision. IDs themselves never become class labels.
    """
    from scipy.optimize import linear_sum_assignment
    obbs, boxes = prediction["obbs"], prediction["boxes"]
    logits, pairs, presence = prediction["visibility_logits"], prediction["pair_logits"], prediction["presence_logits"]
    if (targets.ndim != 4 or targets.shape[:2] != obbs.shape[:2] or targets.shape[-1] != 6
            or valid.shape != targets.shape[:-1] or reviewed.shape != obbs.shape[:2]):
        raise ValueError("Expected paired targets[B,2,M,6], valid[B,2,M], reviewed[B,2]")
    if valid.dtype != torch.bool or reviewed.dtype != torch.bool:
        raise ValueError("Detection valid/reviewed masks must be boolean")
    truth_values = targets[valid & reviewed[..., None]]
    if (not torch.isfinite(truth_values).all() or not torch.isfinite(obbs).all()
            or not torch.isfinite(boxes).all()):
        raise ValueError("Detection boxes must be finite")
    if (truth_values.numel() and ((truth_values[:, :2] < 0).any() or (truth_values[:, :2] > 1).any()
            or (truth_values[:, 2:4] <= 0).any()
            or not torch.allclose(truth_values[:, 4:6].norm(dim=-1), torch.ones_like(truth_values[:, 0]), atol=1e-3))):
        raise ValueError("Valid OBB targets require normalized centers, positive sizes and unit angle vectors")
    zero = sum(prediction[key].sum() * 0 for key in ("obbs", "boxes", "pair_logits", "visibility_logits", "presence_logits"))
    box_loss = angle_loss = overlap_loss = match_loss = visibility_loss = presence_loss = zero
    matched_views = reviewed_views = reviewed_samples = 0
    for batch in range(obbs.shape[0]):
        known = reviewed[batch]
        if not known.any():
            continue
        reviewed_samples += 1
        visible = valid[batch] & known[:, None]
        objects = visible.any(0).nonzero(as_tuple=True)[0]
        if len(objects) > obbs.shape[2]:
            raise ValueError("More annotated object pairs than SAM 3 object queries")
        cost = -pairs[batch].sigmoid()[:, None].expand(-1, len(objects)).clone()
        for view in range(2):
            if not known[view]:
                continue
            reviewed_views += 1
            present = bool(visible[view].any())
            presence_loss = presence_loss + F.binary_cross_entropy_with_logits(
                presence[batch, view], presence.new_tensor(float(present)))
            active = visible[view, objects]
            cost = cost + F.binary_cross_entropy_with_logits(
                logits[batch, view, :, None].expand(-1, len(objects)),
                active.float()[None].expand(obbs.shape[2], -1), reduction="none")
            if active.any():
                truth = targets[batch, view, objects[active]]
                predicted = obbs[batch, view]
                cost[:, active] += (5 * torch.cdist(predicted[:, :4].float(), truth[:, :4].float(), p=1)
                    + 1 - predicted[:, 4:6] @ truth[:, 4:6].T
                    + 2 * (1 - gaussian_overlap(predicted, truth)))
        with torch.no_grad():
            rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
        rows = torch.as_tensor(rows, device=boxes.device, dtype=torch.long)
        objects = objects[torch.as_tensor(columns, device=boxes.device, dtype=torch.long)]
        pair_labels = torch.zeros_like(pairs[batch])
        pair_labels[rows] = 1
        # With one unreviewed view, unseen objects may exist there. Do not train
        # unmatched pair slots as absent unless both views were reviewed.
        pair_weights = torch.where(pair_labels.bool(), 1., .1 if known.all() else 0.)
        match_loss = match_loss + (F.binary_cross_entropy_with_logits(
            pairs[batch], pair_labels, reduction="none") * pair_weights).sum() / pair_weights.sum().clamp_min(1)
        for view in range(2):
            if not known[view]:
                continue
            active = visible[view, objects]
            labels = torch.zeros_like(logits[batch, view])
            labels[rows[active]] = 1
            weights = torch.where(labels.bool(), 1., .1)
            visibility_loss = visibility_loss + (F.binary_cross_entropy_with_logits(
                logits[batch, view], labels, reduction="none") * weights).mean()
            if not active.any():
                continue
            selected = obbs[batch, view, rows[active]]
            truth = targets[batch, view, objects[active]]
            box_loss = box_loss + 5 * F.l1_loss(selected[:, :4], truth[:, :4], reduction="sum")
            # Keep SAM's native axis-aligned references useful for spatial RPB.
            envelope = obb_enclosures(truth)
            native = boxes[batch, view, rows[active]]
            box_loss = box_loss + F.l1_loss(native, envelope, reduction="sum")
            box_loss = box_loss + (1 - generalized_box_iou(native, envelope).diag()).sum()
            angle_loss = angle_loss + (1 - (selected[:, 4:6] * truth[:, 4:6]).sum(-1)).clamp_min(0).sum()
            overlap_loss = overlap_loss + 2 * (1 - gaussian_overlap(selected, truth).diag()).sum()
            matched_views += len(selected)
    terms = {"detection_box_loss": box_loss / max(matched_views, 1),
             "detection_angle_loss": angle_loss / max(matched_views, 1),
             "detection_overlap_loss": overlap_loss / max(matched_views, 1),
             "detection_match_loss": match_loss / max(reviewed_samples, 1),
             "detection_visibility_loss": visibility_loss / max(reviewed_views, 1),
             "detection_presence_loss": presence_loss / max(reviewed_views, 1)}
    return {"detection_loss": sum(terms.values()), **terms}


class PairQueryFusion(nn.Module):
    """Exchange two-view evidence inside each logical object slot."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=False)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, states):
        queries, views_batch, width = states.shape
        if views_batch % 2:
            raise ValueError("Paired decoder requires front/side batches")
        # [Q,B*2,D] -> [2,B*Q,D]. Attention never crosses sample boundaries.
        x = states.reshape(queries, views_batch // 2, 2, width).permute(2, 1, 0, 3).reshape(2, -1, width)
        normalized = self.norm(x)
        fused = self.output_norm(x + self.attention(normalized, normalized, normalized, need_weights=False)[0])
        return fused.reshape(2, views_batch // 2, queries, width).permute(2, 1, 0, 3).reshape_as(states)


class Sam3BoxDecoder(nn.Module):
    """Actual SAM 3 detection decoder, adapted to PaliGemma prefix features.

    The default decoder/scorer shapes match the original SAM 3 checkpoint.
    PE/fusion are replaced by PaliGemma and learned adapters; mask/tracker heads
    are absent. See vendor/sam3/NOTICE.md for the upstream source revision.
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
        self.pair_fusion = nn.ModuleList(PairQueryFusion(width, heads) for _ in range(layers))
        self.pair_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.obb_head = MLP(width, width, 6, 3)
        nn.init.zeros_(self.obb_head.layers[-1].weight)
        nn.init.zeros_(self.obb_head.layers[-1].bias)
        with torch.no_grad():
            self.obb_head.layers[-1].bias[4] = 1
            self.obb_head.layers[-1].bias[5] = .01

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
                apply_dac=False, pair_fusion=self.pair_fusion)
            hidden, references = hidden.transpose(1, 2), references.transpose(1, 2)
            all_boxes = (inverse_sigmoid(references) + self.decoder.bbox_embed(hidden)).sigmoid()
            all_logits = self.dot_prod_scoring(hidden, prompt, prompt_padding).squeeze(-1)
            all_presence = presence.squeeze(1)
            count, _, queries, _ = all_boxes.shape
            all_boxes = all_boxes.reshape(count, batch, 2, queries, 4)
            all_logits = all_logits.reshape(count, batch, 2, queries)
            all_presence = all_presence.reshape(count, batch, 2)
            view_hidden = hidden.reshape(count, batch, 2, queries, -1)
            raw = self.obb_head(view_hidden)
            center = (inverse_sigmoid(all_boxes[..., :2]) + raw[..., :2]).sigmoid()
            base_size = all_boxes[..., 2:4].clamp_min(1e-5)
            size = F.softplus(torch.log(torch.expm1(base_size)) + raw[..., 2:4])
            all_obbs = canonical_obbs(torch.cat((center, size, raw[..., 4:6]), -1))
            pair_logits = self.pair_head(view_hidden.mean(2)).squeeze(-1)
            visibility = all_logits.sigmoid() * all_presence.sigmoid()[..., None]
            values = {"boxes": all_boxes, "obbs": all_obbs, "logits": all_logits,
                      "visibility_logits": all_logits, "pair_logits": pair_logits,
                      "presence_logits": all_presence}
            result = {key: value[-1] for key, value in values.items()}
            result.update({f"aux_{key}": value[:-1] for key, value in values.items()})
            result.update(pair_scores=pair_logits[-1].sigmoid(), visibility_scores=visibility[-1],
                          scores=pair_logits[-1].sigmoid()[:, None] * visibility[-1])
            return result


class OAVLA(ActionFlowModel):
    schema_version = SCHEMA_VERSION

    def __init__(self, config):
        if not config.pi05:
            raise ValueError("SAM 3 VLA requires PI0.5")
        super().__init__(config)
        prefix_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.detection_expert = Sam3BoxDecoder(
            prefix_width, width=config.detection_width, queries=config.detection_queries, layers=config.detection_layers,
            heads=config.detection_heads, ffn_width=config.detection_ffn_width)
        self.paligemma_with_expert.paligemma.requires_grad_(not config.freeze_backbone)
        self.detection_expert.decoder.use_act_checkpoint = config.gradient_checkpointing
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def save_pretrained(self, directory, *, metadata=None):
        """Export self-contained config/weights; pretrained weights are never auto-downloaded."""
        from .checkpoint import save_pretrained
        return save_pretrained(self, directory, metadata=metadata)

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu"):
        from .checkpoint import load_pretrained
        return load_pretrained(directory, device=device)

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
            error = ActionFlowModel.forward(self, observation, actions, noise=noise, time=time)
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
                             for key in ("boxes", "obbs", "visibility_logits", "pair_logits", "presence_logits")}
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
        actions = ActionFlowModel.sample_actions(self, device, observation, noise=noise, num_steps=num_steps)
        prediction = self.detect(observation)
        return {"actions": actions, **{key: prediction[key] for key in (
            "boxes", "obbs", "scores", "pair_scores", "visibility_scores", "presence_logits")}}
