"""High-level four-RGB inference and portable preprocessing state."""

import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .contract import POLICY_VIEWS, DETECTION_VIEWS
from .data import letterbox_rgb
from .geometry import original_obb_coordinates
from .observation import Observation
from .tokenizer import ByteTokenizer, Tokenizer


class OAVLAPolicy:
    def __init__(self, model, tokenizer, *, metadata=None, score_threshold=.5, visibility_threshold=.5):
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0,1]")
        if not 0 <= visibility_threshold <= 1:
            raise ValueError("visibility_threshold must be in [0,1]")
        self.visibility_threshold = visibility_threshold
        self.model, self.tokenizer = model.eval(), tokenizer
        self.metadata = dict(metadata or {})
        self.score_threshold = score_threshold
        config = model.config
        for name, length, default in (
            ("qpos_mean", config.state_dim, 0.), ("qpos_std", config.state_dim, 1.),
            ("action_mean", config.logical_action_dim, 0.), ("action_std", config.logical_action_dim, 1.)):
            value = np.asarray(self.metadata.get(name, [default] * length), np.float32)
            if (value.shape != (length,) or not np.isfinite(value).all()
                    or name.endswith("std") and np.any(value <= 0)):
                raise ValueError(f"Invalid normalization: {name}")
            setattr(self, name, value)

    def predict(self, rgb, state, prompt, *, num_steps=10, noise=None):
        if set(rgb) != set(POLICY_VIEWS):
            raise ValueError(f"RGB must contain exactly {POLICY_VIEWS}")
        if not str(prompt).strip():
            raise ValueError("prompt must be nonempty")
        config = self.model.config
        state = np.asarray(state, np.float32)
        if state.shape != (config.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite [{config.state_dim}]")
        normalized_state = np.zeros(config.action_dim, np.float32)
        normalized_state[:config.state_dim] = (state - self.qpos_mean) / self.qpos_std
        tokens, mask = self.tokenizer.tokenize(str(prompt), normalized_state)
        if (np.any(tokens < 0) or np.any(tokens >= config.vocab_size)):
            raise ValueError("Tokenizer vocabulary does not match model configuration")
        device = next(self.model.parameters()).device
        images, shapes = {}, {}
        for view in POLICY_VIEWS:
            image = np.asarray(rgb[view])
            shapes[view] = image.shape[:2]
            padded = letterbox_rgb(image)
            images[view] = torch.from_numpy(padded.copy()).to(device)[None].permute(0, 3, 1, 2).float() / 127.5 - 1
        observation = Observation(
            images=images, state=torch.from_numpy(normalized_state).to(device)[None],
            tokenized_prompt=torch.as_tensor(tokens, device=device, dtype=torch.long)[None],
            tokenized_prompt_mask=torch.as_tensor(mask, device=device, dtype=torch.bool)[None])
        with torch.inference_mode():
            prediction = self.model.sample_actions_and_boxes(device, observation, noise=noise, num_steps=num_steps)
        actions = prediction["actions"][0, :, :config.logical_action_dim].float().cpu().numpy()
        actions = (actions * self.action_std + self.action_mean).astype(np.float32)
        if not np.isfinite(actions).all():
            raise RuntimeError("Model produced non-finite actions")
        detections, objects = {view: [] for view in DETECTION_VIEWS}, []
        trained = bool(self.metadata.get("detection_trained", False))
        if trained:
            obbs = prediction["obbs"][0].float().cpu().numpy()
            pair_scores = prediction["pair_scores"][0].float().cpu().numpy()
            visibility = prediction["visibility_scores"][0].float().cpu().numpy()
            if not all(np.isfinite(value).all() for value in (obbs, pair_scores, visibility)):
                raise RuntimeError("Model produced non-finite paired detections")
            restored = {view: original_obb_coordinates(obbs[index], *shapes[view])
                        for index, view in enumerate(DETECTION_VIEWS)}
            for query, score in enumerate(pair_scores):
                if score < self.score_threshold:
                    continue
                pair = {"pair_id": int(query), "score": float(score), "front": None, "side": None}
                for index, view in enumerate(DETECTION_VIEWS):
                    if visibility[index, query] < self.visibility_threshold:
                        continue
                    box, corners = restored[view][0][query], restored[view][1][query]
                    if min(box[2:4]) <= 0:
                        continue
                    entry = {"obb": box.tolist(), "corners": corners.tolist(),
                             "visibility_score": float(visibility[index, query]),
                             "score": float(score * visibility[index, query])}
                    pair[view] = entry
                    detections[view].append({"pair_id": int(query), **entry})
                if pair["front"] is not None or pair["side"] is not None:
                    objects.append(pair)
        return {"model_type": "oa-vla", "actions": actions, "action_chunk": actions,
                "objects": objects, "detections": detections,
                "detection_box_format": "cxcywh_angle_deg_original_pixels",
                "pair_id_scope": "current_prediction",
                "detection_status": "ok" if trained else "not_trained"}

    def infer(self, request, **kwargs):
        """Compatibility convenience for rgb_front/... plus qpos request dictionaries."""
        return self.predict({view: request[f"rgb_{view}"] for view in POLICY_VIEWS},
                            request["qpos"], request["prompt"], **kwargs)

    def save_pretrained(self, directory):
        directory = self.model.save_pretrained(directory, metadata=self.metadata)
        if isinstance(self.tokenizer, ByteTokenizer):
            settings = {"type": "byte_test_only", "max_len": self.tokenizer.max_len}
        else:
            destination = directory / "tokenizer.model"
            if self.tokenizer.model_path.resolve() != destination.resolve():
                shutil.copy2(self.tokenizer.model_path, destination)
            settings = {"type": "sentencepiece", "max_len": self.tokenizer.max_len}
        (directory / "processor.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        return directory

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu", tokenizer_model=None, score_threshold=.5, visibility_threshold=.5):
        from .model import OAVLA
        directory = Path(directory)
        model = OAVLA.from_pretrained(directory, device=device)
        settings_path = directory / "processor.json"
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {
            "type": "sentencepiece", "max_len": model.config.max_token_len}
        if settings["type"] == "byte_test_only":
            if model.config.vocab_size != 512:
                raise ValueError("Test tokenizer cannot be used with pretrained Gemma")
            tokenizer = ByteTokenizer(settings["max_len"])
        else:
            local = directory / "tokenizer.model"
            tokenizer = Tokenizer(settings["max_len"], tokenizer_model or (local if local.exists() else None))
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return cls(model, tokenizer, metadata=metadata, score_threshold=score_threshold, visibility_threshold=visibility_threshold)
