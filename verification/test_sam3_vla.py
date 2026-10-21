"""Regression coverage for point removal, camera isolation and SAM 3 training."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from safety_module.policy_contract import CAMERA_TO_MODEL, DETECTION_VIEWS, MODEL_TYPE, SCHEMA_VERSION
from safety_module.sam3_policy_data import Sam3VlaDataset, original_xyxy_coordinates, to_observation
from openpi.models_pytorch.sam3_vla_pytorch import Sam3BoxDecoder, Sam3VlaModelPytorch, sam3_set_loss


@pytest.fixture
def tiny_policy(monkeypatch):
    from transformers import PaliGemmaConfig, GemmaConfig
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch import gemma_pytorch

    def tiny_vlm_config():
        return PaliGemmaConfig(vision_config={"hidden_size": 32, "num_hidden_layers": 1,
                                               "num_attention_heads": 4, "intermediate_size": 64,
                                               "image_size": 224, "patch_size": 112})

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": tiny_vlm_config, "gemma": GemmaConfig})
    monkeypatch.setattr(torch, "compile", lambda function, **_: function)
    torch.manual_seed(42)
    config = Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                       action_horizon=2, dtype="float32")
    model = Sam3VlaModelPytorch(config, detection_queries=4, detection_layers=2,
                                detection_width=32, detection_heads=4, detection_ffn_width=64)
    observation = SimpleNamespace(
        images={key: torch.randn(1, 3, 224, 224).clamp(-1, 1) for key in CAMERA_TO_MODEL.values()},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in CAMERA_TO_MODEL.values()},
        tokenized_prompt=torch.tensor([[1, 2, 3, 0]]),
        tokenized_prompt_mask=torch.tensor([[True, True, True, False]]),
        state=torch.zeros(1, 32), token_ar_mask=None, token_loss_mask=None)
    return model, observation


def targets():
    boxes = torch.zeros(1, 2, 3, 4)
    boxes[0, 0, 0] = torch.tensor([.5, .4, .2, .3])
    valid = torch.zeros(1, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = True
    return dict(detection_targets=boxes, detection_valid=valid,
                detection_reviewed=torch.ones(1, 2, dtype=torch.bool))


def test_wrist_changes_affect_actions_but_never_detection_even_through_text(tiny_policy):
    model, observation = tiny_policy
    model.eval()
    noise = torch.randn(1, 2, 32)
    with torch.no_grad():
        first = model.sample_actions_and_boxes("cpu", observation, noise=noise.clone(), num_steps=2)
        for name in ("right_wrist", "left_wrist"):
            key = CAMERA_TO_MODEL[name]
            original = observation.images[key]
            observation.images[key] = torch.ones(1, 3, 224, 224)
            second = model.sample_actions_and_boxes("cpu", observation, noise=noise.clone(), num_steps=2)
            observation.images[key] = original
            torch.testing.assert_close(first["boxes"], second["boxes"], atol=0, rtol=0)
            torch.testing.assert_close(first["scores"], second["scores"], atol=0, rtol=0)
            torch.testing.assert_close(first["presence_logits"], second["presence_logits"], atol=0, rtol=0)
            assert not torch.allclose(first["actions"], second["actions"], atol=1e-7, rtol=0), name
    assert first["boxes"].shape == (1, 2, 4, 4)
    assert not any("surface" in name or "point" in name and "ref_point_head" not in name and "reference_points" not in name
                   for name, _ in model.named_parameters())
    assert set(first) == {"actions", "boxes", "scores", "presence_logits"}


def test_detector_gradient_cannot_reach_wrist_images_when_vlm_is_trainable(tiny_policy):
    model, observation = tiny_policy
    model.eval()
    model.paligemma_with_expert.paligemma.requires_grad_(True)
    for image in observation.images.values():
        image.requires_grad_(True)
    model.detect(observation)["logits"].sum().backward()
    for view in DETECTION_VIEWS:
        assert observation.images[CAMERA_TO_MODEL[view]].grad.abs().sum() > 0
    for view in ("right_wrist", "left_wrist"):
        assert observation.images[CAMERA_TO_MODEL[view]].grad is None


def test_joint_and_detection_only_stages_without_point_inputs(tiny_policy):
    from scripts.train_pi05_sam3_pytorch import set_train_stage
    model, observation = tiny_policy
    set_train_stage(model, "joint")
    model.train()
    result = model(observation, torch.randn(1, 2, 32), **targets())
    assert {"loss", "action_loss", "detection_loss", "detection_presence_loss"} <= set(result)
    assert not any("point" in key for key in result)
    result["loss"].backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    assert model.detection_expert.decoder.bbox_embed.layers[-1].weight.grad.abs().sum() > 0
    assert model.detection_expert.decoder.presence_token_head.layers[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
    model.zero_grad(set_to_none=True)
    set_train_stage(model, "detection")
    model(observation, train_stage="detection", **targets())["loss"].backward()
    assert model.action_out_proj.weight.grad is None
    assert model.detection_expert.image_adapter[1].weight.grad.abs().sum() > 0


def test_missing_or_masked_policy_view_fails_instead_of_zero_filling(tiny_policy):
    model, observation = tiny_policy
    observation.images.pop("base_1_rgb")
    with pytest.raises(ValueError, match="missing keys"):
        model._preprocess_observation(observation, train=False)
    observation.images["base_1_rgb"] = torch.zeros(1, 3, 224, 224)
    observation.image_masks["base_1_rgb"][:] = False
    with pytest.raises(ValueError, match="All four"):
        model._preprocess_observation(observation, train=False)


def test_unreviewed_views_have_zero_loss_and_reviewed_empty_trains_absence():
    head = Sam3BoxDecoder(16, width=32, queries=4, layers=1, heads=4, ffn_width=64)
    prediction = head(torch.randn(1, 2, 4, 16), torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    boxes = torch.zeros(1, 2, 3, 4)
    valid = torch.zeros(1, 2, 3, dtype=torch.bool)
    losses = sam3_set_loss(prediction, boxes, valid, torch.zeros(1, 2, dtype=torch.bool))
    assert all(value.item() == 0 for value in losses.values())
    losses = sam3_set_loss(prediction, boxes, valid, torch.tensor([[True, False]]))
    assert losses["detection_presence_loss"] > 0
    assert losses["detection_match_loss"] == losses["detection_box_loss"] == 0
    losses["detection_loss"].backward()
    assert head.decoder.presence_token_head.layers[-1].weight.grad.abs().sum() > 0


def test_official_sam3_weight_names_load_and_incomplete_weights_fail(tmp_path):
    first = Sam3BoxDecoder(16, width=32, queries=4, layers=1, heads=4, ffn_width=64)
    state = {"detector.transformer.decoder." + key: value for key, value in first.decoder.state_dict().items()}
    state.update({"detector.dot_prod_scoring." + key: value for key, value in first.dot_prod_scoring.state_dict().items()})
    state["tracker.unused"] = torch.ones(1)
    path = tmp_path / "official-layout.pt"
    torch.save({"model": state}, path)
    second = Sam3BoxDecoder(16, width=32, queries=4, layers=1, heads=4, ffn_width=64)
    report = second.load_sam3_weights(path)
    assert report["decoder"] == len(first.decoder.state_dict())
    for key, value in first.decoder.state_dict().items():
        torch.testing.assert_close(second.decoder.state_dict()[key], value)
    state.pop(next(key for key in state if "bbox_embed" in key))
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        second.load_sam3_weights(path)


def write_shard(path, *, include_wrist=True):
    payload = dict(task_text="avoid the obstacle", qpos=np.zeros((3, 12), np.float32),
                   action_chunks=np.ones((3, 2, 14), np.float32), sample_frame_indices=np.array([2, 0, 1]),
                   action_step_mask=np.array([[True, False], [True, True], [False, False]]),
                   detection_boxes=np.zeros((3, 2, 1, 4), np.float32),
                   detection_valid=np.zeros((3, 2, 1), bool),
                   detection_reviewed=np.array([[True, False], [False, True], [False, False]]),
                   detection_box_format="cxcywh_normalized_padded", detection_views=np.array(DETECTION_VIEWS))
    for index, view in enumerate(CAMERA_TO_MODEL):
        if include_wrist or "wrist" not in view:
            payload[f"rgb_{view}"] = np.full((3, 40 + index * 10, 80, 3), 20 * index, np.uint8)
    np.savez(path, **payload)


def dataset(path, **kwargs):
    return Sam3VlaDataset([path], tokenizer=SimpleNamespace(tokenize=lambda *_: (np.ones(4, np.int64), np.ones(4, bool))),
                          qpos_mean=np.zeros(12), qpos_std=np.ones(12), action_mean=np.zeros(14),
                          action_std=np.ones(14), action_horizon=2, **kwargs)


def test_point_free_shards_batch_different_camera_resolutions_and_keep_view_order(tmp_path):
    from torch.utils.data import DataLoader
    path = tmp_path / "episode.npz"
    write_shard(path)
    data = dataset(path)
    assert len(data) == 2
    sample = data[0]
    assert not any("point" in key or "surface" in key for key in sample)
    assert all(image.shape == (224, 224, 3) for image in sample["images"].values())
    assert not sample["action_mask"][1].any()
    assert not sample["actions"][1].any()
    batch = next(iter(DataLoader(data, batch_size=2)))
    observation = to_observation(batch, "cpu")
    assert tuple(observation.images) == tuple(CAMERA_TO_MODEL.values())
    assert observation.images["base_1_rgb"].shape == (2, 3, 224, 224)
    assert len(dataset(path, reviewed_only=True)) == 2
    write_shard(path, include_wrist=False)
    with pytest.raises(ValueError, match="four real RGB views"):
        dataset(path)


def test_obb_labels_become_aabb_and_restore_original_pixels(tmp_path):
    from safety_module.detection_annotations import load_detection_annotations
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as episode:
        episode["timestamps_ns"] = np.array([10], np.int64)
        labels = episode.require_group("annotations/obstacle_obbs")
        labels.attrs.update(schema="dual_rgb_obb_annotation_v2", annotation_image="recorded_rgb")
        labels["timestamps_ns"], labels["reviewed"] = np.array([10]), [True]
        values = {view: [{"object_id": "object", "cx": 40, "cy": 30, "width": 20,
                          "height": 10, "angle_deg": 90}] for view in DETECTION_VIEWS}
        labels.create_dataset("frames_json", data=[json.dumps({"schema": "dual_rgb_obb_annotation_v2",
            "annotation_image": "recorded_rgb", "frame_index": 0, "timestamp_ns": "10",
            "status": "objects", "boxes": values})], dtype=h5py.string_dtype())
    boxes, valid, reviewed = load_detection_annotations(tmp_path / "shard.npz", {
        "source_episode": np.asarray(str(source)), "timestamps_ns": np.array([10])},
        np.array([0]), {"front": (60, 80, 3), "side": (60, 80, 3)}, axis_aligned=True)
    assert valid[0, :, 0].all() and reviewed.all()
    actual = original_xyxy_coordinates(boxes[0, 0, :1], 60, 80)
    np.testing.assert_allclose(actual[0], [35, 20, 45, 40], atol=1e-4)


def test_joint_training_rejects_labels_only_on_samples_without_valid_actions(tmp_path, monkeypatch):
    from scripts import train_pi05_sam3_pytorch as training
    path = tmp_path / "data.npz"
    write_shard(path)
    with np.load(path) as values:
        payload = {key: values[key] for key in values.files}
    payload["detection_reviewed"] = np.array([[False, False], [False, False], [True, True]])
    np.savez(path, **payload)
    monkeypatch.setattr("sys.argv", ["train", "--dataset", str(path), "--output", str(tmp_path / "out.pt"),
                                    "--validate-only"])
    with pytest.raises(ValueError, match="at least one reviewed"):
        training.main()


def test_service_checkpoint_roundtrip_four_inputs_no_point_request(tmp_path, tiny_policy, monkeypatch):
    from scripts import serve_quest3_pi05_safety as serving
    from scripts.train_pi05_sam3_pytorch import save_checkpoint
    model, _ = tiny_policy
    path = tmp_path / "policy.pt"
    info = dict(camera_map=CAMERA_TO_MODEL, detection_views=list(DETECTION_VIEWS), detection_trained=True,
                joint_dim=12, logical_action_dim=14, action_horizon=2, action_period_s=.1,
                precision="float32", paligemma_variant="dummy", action_expert_variant="dummy",
                detection_queries=4, detection_layers=2, detection_width=32, detection_heads=4, detection_ffn_width=64,
                qpos_mean=[0.] * 12, qpos_std=[1.] * 12, action_mean=[0.] * 14, action_std=[1.] * 14)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, optimizer, 1, info)
    monkeypatch.setattr(serving, "_Tokenizer", lambda **_: SimpleNamespace(tokenize=lambda *_: (
        np.array([1, 2, 3, 0]), np.array([True, True, True, False]))))
    policy = serving.Quest3Sam3VlaPolicy(path, device_name="cpu", num_steps=1, score_threshold=0.)
    assert policy.metadata["point_output"] is False
    assert policy.metadata["model_type"] == MODEL_TYPE
    request = {"qpos": np.zeros(12), "prompt": "avoid obstacle",
               **{f"rgb_{view}": np.zeros((60, 80, 3), np.uint8) for view in CAMERA_TO_MODEL}}
    result = policy.infer(request)
    assert result["schema_version"] == SCHEMA_VERSION and result["actions"].shape == (2, 14)
    assert set(result["detections"]) == {"front", "side"}
    assert not any("point" in key or "surface" in key for key in result)
    for detection in result["detections"]["front"]:
        x0, y0, x1, y1 = detection["xyxy"]
        assert 0 <= x0 < x1 <= 80 and 0 <= y0 < y1 <= 60
    request.pop("rgb_left_wrist")
    with pytest.raises(ValueError, match="All four RGB views"):
        policy.infer(request)


def test_rgb_action_preprocess_and_real_training_entrypoint(tmp_path, tiny_policy, monkeypatch):
    from scripts import preprocess_quest3_rgb_action as preprocess
    from scripts import train_pi05_sam3_pytorch as training
    from openpi.models import pi0_config
    from safety_module.shards import open_shard

    source = tmp_path / "recording.hdf5"
    with h5py.File(source, "w") as episode:
        episode.attrs["task_description"] = "avoid obstacle"
        episode["observations/qpos"] = np.zeros((3, 12), np.float32)
        episode["action"] = np.zeros((3, 12), np.float32)
        episode["gripper_action"] = np.zeros((3, 2), np.float32)
        episode["action_valid"] = [True, True, False]
        episode["timestamps_ns"] = np.array([10, 100000010, 200000010], np.int64)
        for view in CAMERA_TO_MODEL:
            episode[f"observations/images/{view}/rgb"] = np.zeros((3, 40, 80, 3), np.uint8)
        labels = episode.require_group("annotations/obstacle_obbs")
        labels.attrs.update(schema="dual_rgb_obb_annotation_v2", annotation_image="recorded_rgb")
        labels["reviewed"] = [True, True, True]
        labels["timestamps_ns"] = episode["timestamps_ns"][:]
        labels.create_dataset("frames_json", data=[json.dumps({"schema": "dual_rgb_obb_annotation_v2",
            "annotation_image": "recorded_rgb", "frame_index": index, "timestamp_ns": str(int(timestamp)),
            "status": "no_obstacles", "boxes": {"front": [], "side": []}})
            for index, timestamp in enumerate(episode["timestamps_ns"][:])], dtype=h5py.string_dtype())
    shard = tmp_path / "data.shard"
    monkeypatch.setattr("sys.argv", ["preprocess", "--episode", str(source), "--output", str(shard), "--future-horizon", "2"])
    preprocess.main()
    with open_shard(shard) as values:
        assert all(f"rgb_{view}" in values for view in CAMERA_TO_MODEL)
        assert not any("point" in key or "surface" in key for key in values.files)
        assert values["action_step_mask"].tolist() == [[True, True], [True, False], [False, False]]
    original_config = pi0_config.Pi0Config
    monkeypatch.setattr(pi0_config, "Pi0Config", lambda **kwargs: original_config(
        **kwargs, paligemma_variant="dummy", action_expert_variant="dummy"))
    monkeypatch.setattr(training, "_Tokenizer", lambda *_: SimpleNamespace(tokenize=lambda *_: (
        np.array([1, 2, 3, 0]), np.array([True, True, True, False]))))
    checkpoint = tmp_path / "trained.pt"
    arguments = ["train", "--dataset", str(shard), "--output", str(checkpoint),
                 "--allow-random-init", "--precision", "float32", "--device", "cpu", "--epochs", "1",
                 "--max-steps", "1", "--detection-queries", "4", "--detection-layers", "1", "--no-gradient-checkpointing"]
    monkeypatch.setattr("sys.argv", arguments + ["--validate-only"])
    training.main()
    assert not checkpoint.exists()
    monkeypatch.setattr("sys.argv", arguments)
    training.main()
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    assert saved["model_type"] == MODEL_TYPE and saved["metadata"]["detection_trained"]
    assert all(not key.startswith("surface_") for key in saved["model_state_dict"])
    assert saved["metadata"]["camera_map"] == CAMERA_TO_MODEL
