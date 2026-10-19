import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from openpi.models_pytorch.surface_detection_pytorch import (
    PI05DetectionExpert, detection_set_loss, original_obb_coordinates, padded_obb_coordinates,
)
from safety_module.detection_annotations import load_detection_annotations
from safety_module.shards import open_shard


def test_detection_set_loss_ignores_unreviewed_and_updates_box_head():
    head = PI05DetectionExpert(16, width=32, queries=4, layers=1)
    logits, boxes = head(torch.randn(2, 16, 16))
    logits, boxes = logits.reshape(1, 2, 4), boxes.reshape(1, 2, 4, 6)
    target = torch.zeros(1, 2, 4, 6)
    target[0, 0, 0] = torch.tensor([0.4, 0.5, 0.2, 0.1, 1., 0.])
    valid = torch.zeros(1, 2, 4, dtype=torch.bool)
    valid[0, 0, 0] = True
    reviewed = torch.tensor([[True, False]])
    loss = detection_set_loss(logits, boxes, target, valid, reviewed)["detection_loss"]
    loss.backward()
    assert torch.isfinite(loss) and head.geometry[-1].weight.grad.abs().sum() > 0
    empty = detection_set_loss(logits.detach(), boxes.detach(), target, valid, torch.zeros_like(reviewed))
    assert empty["detection_loss"].item() == 0


def test_obbs_round_trip_between_rgb_and_padded_pi05_image():
    original = torch.tensor([[40., 30., 20., 10., 135.]], dtype=torch.float32)
    recovered = original_obb_coordinates(padded_obb_coordinates(original, 60, 80), 60, 80)
    torch.testing.assert_close(recovered, original, atol=1e-4, rtol=0)
    vertical = torch.tensor([[40., 30., 10., 20., 45.]], dtype=torch.float32)
    canonical = original_obb_coordinates(padded_obb_coordinates(vertical, 60, 80), 60, 80)
    torch.testing.assert_close(canonical, original, atol=1e-4, rtol=0)


def test_detection_annotations_align_by_frame_and_skip_unreviewed(tmp_path):
    source = tmp_path / "episode.hdf5"
    with h5py.File(source, "w") as episode:
        episode["timestamps_ns"] = np.array([10, 20, 30], np.int64)
        labels = episode.require_group("annotations/obstacle_obbs")
        labels.attrs.update(schema="dual_rgb_obb_annotation_v2", annotation_image="recorded_rgb")
        labels["timestamps_ns"] = np.array([10, 20, 30], np.int64)
        labels["reviewed"] = [True, False, True]
        frames = labels.create_dataset("frames_json", (3,), dtype=h5py.string_dtype())
        frames[0] = json.dumps({"schema": "dual_rgb_obb_annotation_v2", "annotation_image": "recorded_rgb",
                                "frame_index": 0, "timestamp_ns": "10", "status": "no_obstacles",
                                "boxes": {"front": [], "side": []}})
        frames[2] = json.dumps({"schema": "dual_rgb_obb_annotation_v2", "annotation_image": "recorded_rgb",
                                "frame_index": 2, "timestamp_ns": "30", "status": "objects",
                                "boxes": {view: [{"object_id": "tube", "cx": 40, "cy": 30, "width": 20,
                                                   "height": 10, "angle_deg": 45}] for view in ("front", "side")}})
    shard = tmp_path / "episode.npz"
    np.savez(shard, source_episode=str(source), timestamps_ns=np.array([10, 20, 30], np.int64))
    with open_shard(shard) as data:
        boxes, valid, reviewed = load_detection_annotations(
            shard, data, np.array([2, 1, 0]), {"front": (60, 80, 3), "side": (60, 80, 3)})
    assert reviewed.tolist() == [[True, True], [False, False], [True, True]]
    assert valid[0, :, 0].all() and not valid[1:].any()
    np.testing.assert_allclose(boxes[0, 0, 0, :4], [0.5, 0.5, 0.25, 0.125], atol=1e-6)
    from scripts.train_pi05_ur7e_surface_pytorch import Ur7eSurfaceDataset

    np.savez(shard, source_episode=str(source), timestamps_ns=np.array([10, 20, 30], np.int64),
             task_text="move around tube", coordinate_frame="left_base", qpos=np.zeros((3, 12), np.float32),
             action_chunks=np.ones((3, 2, 14), np.float32), action_step_mask=np.ones((3, 2), bool),
             sample_frame_indices=np.arange(3), visual_robot_current_points=np.ones((3, 2, 3), np.float32),
             visual_robot_future_offsets=np.zeros((3, 2, 2, 3), np.float32),
             visual_robot_flow_supervision_mask=np.ones((3, 2, 2), bool),
             current_surface_mask=np.ones((3, 2), bool),
             rgb_front=np.zeros((3, 60, 80, 3), np.uint8), rgb_side=np.zeros((3, 60, 80, 3), np.uint8),
             surface_model_hash="test", point_identity_version="test", points_per_link=2, arm_count=2)
    dataset = Ur7eSurfaceDataset(
        [shard], max_points=2, point_target="visual", action_horizon=2,
        tokenizer=SimpleNamespace(tokenize=lambda *_: (np.ones(4, np.int64), np.ones(4, bool))),
        qpos_mean=np.zeros(12, np.float32), qpos_std=np.ones(12, np.float32),
        action_mean=np.zeros(14, np.float32), action_std=np.ones(14, np.float32),
        include_detection=True, reviewed_only=True,
    )
    dataset.set_camera_map({"front": "base_0_rgb"})
    assert len(dataset) == 2
    assert not dataset[0]["detection_valid"].any()
    assert dataset[1]["detection_valid"][:, 0].all()


def test_three_head_joint_then_detection_stage_updates_only_detection(monkeypatch):
    from transformers import PaliGemmaConfig, GemmaConfig
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch import gemma_pytorch
    from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
    from openpi.models_pytorch.surface_detection_pytorch import SurfaceActionDetectionModelPytorch
    from safety_module.tokenizer import MODEL_IMAGE_KEYS

    def tiny_vlm_config():
        return PaliGemmaConfig(vision_config={"hidden_size": 32, "num_hidden_layers": 1,
                                               "num_attention_heads": 4, "intermediate_size": 64,
                                               "image_size": 224, "patch_size": 112})

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": tiny_vlm_config, "gemma": GemmaConfig})
    config = Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                       action_horizon=2, dtype="float32")
    old = SurfaceActionModelPytorch(config)
    model = SurfaceActionDetectionModelPytorch(config)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(key.startswith("detection_expert.") for key in missing) and not unexpected
    observation = SimpleNamespace(
        images={key: torch.zeros(1, 3, 224, 224) for key in MODEL_IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in MODEL_IMAGE_KEYS},
        tokenized_prompt=torch.ones(1, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool), state=torch.zeros(1, 32),
        token_ar_mask=None, token_loss_mask=None,
    )
    rgb = {view: torch.zeros(1, 3, 32, 64) for view in ("front", "side")}
    target = torch.zeros(1, 2, 32, 6)
    target[0, 0, 0] = torch.tensor([.5, .5, .2, .1, 1., 0.])
    valid = torch.zeros(1, 2, 32, dtype=torch.bool)
    valid[0, 0, 0] = True
    common = dict(detection_images=rgb, detection_targets=target, detection_valid=valid,
                  detection_reviewed=torch.ones(1, 2, dtype=torch.bool))
    arguments = (observation, torch.randn(1, 2, 32), torch.randn(1, 3, 3),
                 torch.zeros(1, 12), torch.randn(1, 2, 3, 3))
    joint = model(*arguments, train_stage="joint", **common)
    assert {"action_loss", "point_loss", "detection_loss"} <= set(joint)
    joint["loss"].backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    assert model.surface_dit.point_decoder[-1].weight.grad.abs().sum() > 0
    assert model.detection_expert.geometry[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("detection_expert."))
    detection = model(*arguments, train_stage="detection", **common)
    detection["loss"].backward()
    assert model.detection_expert.geometry[-1].weight.grad.abs().sum() > 0
    assert model.action_out_proj.weight.grad is None
    assert model.surface_dit.point_decoder[-1].weight.grad is None
