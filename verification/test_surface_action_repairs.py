from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from real_scripts.real_cbf_qp import OrientedBox, PointFlowConstraint, select_point_flow_constraints, project_joint_delta_qp
from safety_module.checkpoint_loading import restore_nonpersistent_buffers
from safety_module.runtime_contract import validate_prediction, validate_snapshot
from safety_module.measured_surface import MeasuredSurfaceSampler


def test_segment_collision_detects_crossing_with_safe_endpoints():
    from real_scripts.real_cbf_qp import segments_intersect_box
    box = OrientedBox(np.zeros(3), np.eye(3), np.full(3, .05))
    assert segments_intersect_box(np.array([[.1, 0, 0]]), np.array([[-.1, 0, 0]]), box).item()
    assert not segments_intersect_box(np.array([[.1, .1, 0]]), np.array([[-.1, .1, 0]]), box).item()


def test_every_frame_windows_measured_targets_and_mmap_loading(tmp_path, monkeypatch):
    import h5py
    from scripts import preprocess_quest3_hdf5 as preprocess
    from scripts.train_pi05_ur7e_surface_pytorch import Ur7eSurfaceDataset, _normalization_stats
    from safety_module.shards import open_shard

    episode = tmp_path / "episode.hdf5"
    with h5py.File(episode, "w") as source:
        source.attrs["task_description"] = "test dual surface"
        source["observations/qpos"] = np.zeros((4, 12), np.float32)
        source["observations/gripper_position"] = np.zeros((4, 2), np.float32)
        source["observations/images/front/rgb"] = np.zeros((4, 8, 8, 3), np.uint8)
        source["observations/images/front/depth"] = np.ones((4, 8, 8), np.uint16)
        source["action"] = np.full((4, 12), .001, np.float32)
        source["executed_action"] = np.full((4, 12), .001, np.float32)
        source["gripper_action"] = np.zeros((4, 2), np.float32)
        source["action_valid"] = [True, True, True, False]
        source["timestamps_ns"] = np.arange(4, dtype=np.int64) * 100000000
    measured = tmp_path / "measured.npz"
    observed = np.arange(4 * 4 * 3).reshape(4, 4, 3).astype(np.float32) / 100
    np.savez(measured, observed_points=observed, observed_visible=np.ones((4, 4), bool), source_episode=str(episode))
    output = tmp_path / "episode.shard"
    monkeypatch.setattr("sys.argv", ["preprocess", "--episode", str(episode), "--output", str(output), "--measured-flow-npz", str(measured), "--future-horizon", "2", "--points-per-link", "3"])
    preprocess.main()
    with open_shard(output) as shard:
        assert shard["action_chunks"].shape == (4, 2, 14)
        assert shard["action_step_mask"].tolist() == [[True, True], [True, True], [True, False], [False, False]]
        np.testing.assert_allclose(shard["surface_chunk"][0], observed[1:3])
        assert (tmp_path / str(shard["source_episode_relative"])).is_file()
    qm, qs, am, ast, *_ = _normalization_stats([output])
    tokenizer = SimpleNamespace(tokenize=lambda task, state: (np.zeros(4, np.int64), np.ones(4, bool)))
    dataset = Ur7eSurfaceDataset([output], max_points=4, point_target="visual", action_horizon=2, tokenizer=tokenizer,
                                qpos_mean=qm, qpos_std=qs, action_mean=am, action_std=ast)
    dataset.set_camera_map({"front": "base_0_rgb"})
    assert len(dataset) == 3
    sample = dataset[2]
    assert sample["robot_points"].shape == (4, 3)
    assert sample["target_point_offsets"].shape == (2, 4, 3)
    assert not sample["action_mask"][1].any()
    assert not sample["target_point_mask"][1].any()


def test_surface_sampling_integrates_velocity_at_every_step():
    from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
    class ConstantVelocity(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = []
        def forward(self, offsets, time, *args): self.calls.append(float(time[0])); return torch.ones_like(offsets) * .5
    model = SurfaceActionModelPytorch.__new__(SurfaceActionModelPytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=2)
    model.surface_dit = ConstantVelocity()
    torch.manual_seed(123); expected = torch.randn(1, 2, 3, 3) - .5
    torch.manual_seed(123)
    actual = model.sample_surface_offsets(torch.zeros(1, 3, 3), torch.zeros(1, 2, 12), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 32), num_steps=5)
    torch.testing.assert_close(actual, expected)
    assert model.surface_dit.calls == pytest.approx([1., .8, .6, .4, .2])


def test_meta_buffers_match_normal_rope_output():
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    config = GemmaConfig(hidden_size=16, num_attention_heads=2, head_dim=8)
    reference = GemmaRotaryEmbedding(config)
    with torch.device("meta"):
        restored = GemmaRotaryEmbedding(config)
    restored.to_empty(device="cpu")
    restored.load_state_dict(reference.state_dict())
    restore_nonpersistent_buffers(restored)
    x, positions = torch.zeros(1, 2, 8), torch.arange(2)[None]
    for actual, expected in zip(restored(x, positions), reference(x, positions)):
        torch.testing.assert_close(actual, expected)
    assert restored.original_inv_freq.device.type == "cpu"


def test_deepest_future_barrier_and_nan_rejection():
    box = OrientedBox(np.zeros(3), np.eye(3), np.full(3, .05))
    current = np.array([[.1, 0., 0.]])
    future = np.array([[[.07, 0., 0.]], [[0., 0., 0.]]])
    constraints = select_point_flow_constraints(current, future, [box], trigger_margin_m=.025)
    assert constraints[0].time_id == 1
    assert constraints[0].predicted_h == pytest.approx(-.05)
    with pytest.raises(ValueError):
        select_point_flow_constraints(current, future * np.nan, [box])


def test_uncontrollable_violation_cannot_be_hidden_by_feasible_constraint():
    items = [PointFlowConstraint(np.array([1., 0, 0]), .1, -.01, 0, 0, 0),
             PointFlowConstraint(np.array([1., 0, 0]), .1, -.01, 0, 1, 0)]
    jacobian = np.zeros((2, 3, 6)); jacobian[1, 0, 0] = 1
    delta, info = project_joint_delta_qp(np.zeros(6), jacobian, items, lower_delta=np.full(6, -.2), upper_delta=np.full(6, .2))
    assert not info["success"]
    assert not np.any(delta)


def test_runtime_rejects_expiry_bad_indices_and_nonfinite_actions():
    snap = dict(schema="dual_ur7e_obstacle_snapshot_v2", coordinate_frame="left_base", status="ok", host_timestamp_ns=time.monotonic_ns())
    validate_snapshot(snap, 1)
    snap["host_timestamp_ns"] -= 2_000_000_000
    with pytest.raises(RuntimeError): validate_snapshot(snap, 1)
    action, future = np.zeros((2, 14)), np.zeros((2, 3, 3))
    with pytest.raises(ValueError): validate_prediction(action, future, np.array([0, 0, 1]), action_dim=14, point_count=3)
    action[0, 0] = np.nan
    with pytest.raises(ValueError): validate_prediction(action, future, np.arange(3), action_dim=14, point_count=3)


def test_measured_points_are_transported_without_replacing_measurements():
    class Sampler:
        def link_points(self, q, grip):
            points = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
            return points + np.array([q[0], 0, 0])
    current = np.array([[.01, .01, .01]])
    sampler = MeasuredSurfaceSampler(Sampler(), np.zeros(6), None, current)
    np.testing.assert_allclose(sampler.link_points(np.zeros(6)), current, atol=1e-6)
    q = np.zeros(6); q[0] = .1
    np.testing.assert_allclose(sampler.link_points(q), current + [.1, 0, 0], atol=1e-6)


def test_point_dit_velocity_backward_and_point_permutation():
    from openpi.models_pytorch.surface_action_pytorch import SurfacePointDiT
    torch.manual_seed(1)
    model = SurfacePointDiT(prefix_dim=12, horizon=2, width=32, layers=2, heads=4)
    current = torch.randn(1, 3, 3); noisy = torch.randn(1, 2, 3, 3)
    prefix = torch.randn(1, 4, 12); mask = torch.ones(1, 4, dtype=torch.bool)
    action = torch.randn(1, 2, 32); t = torch.tensor([.5])
    velocity = model(noisy, t, current, prefix, mask, action)
    assert velocity.shape == noisy.shape
    order = torch.tensor([2, 0, 1])
    permuted = model(noisy[:, :, order], t, current[:, order], prefix, mask, action)
    torch.testing.assert_close(permuted, velocity[:, :, order], atol=2e-6, rtol=2e-5)
    velocity.square().mean().backward()
    assert model.point_decoder[-1].weight.grad.abs().sum() > 0


def test_joint_flow_loss_and_freeze_without_loading_large_backbone(monkeypatch):
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
    def tiny_init(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.action_in_proj = torch.nn.Linear(32, 32)
        trunk = torch.nn.Linear(2, 2)
        trunk.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=12))
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = trunk
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(32, 32)
    def tiny_forward(self, obs, actions, **kwargs):
        return self.paligemma_with_expert.gemma_expert(self.action_in_proj(actions)).square()
    monkeypatch.setattr(PI0Pytorch, "__init__", tiny_init)
    monkeypatch.setattr(PI0Pytorch, "forward", tiny_forward)
    monkeypatch.setattr(SurfaceActionModelPytorch, "_surface_context", lambda self, obs: (torch.ones(1, 2, 12), torch.ones(1, 2, dtype=torch.bool)))
    model = SurfaceActionModelPytorch(SimpleNamespace(pi05=True, action_horizon=2, action_dim=32))
    losses = model(None, torch.randn(1, 2, 32), torch.randn(1, 3, 3), torch.zeros(1, 12), torch.randn(1, 2, 3, 3), time=torch.tensor([.5]))
    losses["loss"].backward()
    assert all(not p.requires_grad and p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
    assert model.paligemma_with_expert.gemma_expert.weight.grad.abs().sum() > 0
    assert model.surface_dit.point_decoder[-1].weight.grad.abs().sum() > 0


def test_real_transformer_joint_forward_and_sampling_with_small_vision(monkeypatch):
    from transformers import PaliGemmaConfig, GemmaConfig
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch import gemma_pytorch
    from openpi.models_pytorch.surface_action_pytorch import SurfaceActionModelPytorch
    from safety_module.tokenizer import MODEL_IMAGE_KEYS

    def tiny_vlm_config():
        return PaliGemmaConfig(vision_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                                             "intermediate_size": 64, "image_size": 224, "patch_size": 112})
    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": tiny_vlm_config, "gemma": GemmaConfig})
    model = SurfaceActionModelPytorch(Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2, dtype="float32"))
    obs = SimpleNamespace(images={k: torch.zeros(1, 3, 224, 224) for k in MODEL_IMAGE_KEYS},
                          image_masks={k: torch.ones(1, dtype=torch.bool) for k in MODEL_IMAGE_KEYS},
                          tokenized_prompt=torch.ones(1, 4, dtype=torch.long), tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
                          state=torch.zeros(1, 32), token_ar_mask=None, token_loss_mask=None)
    points = torch.randn(1, 3, 3)
    losses = model(obs, torch.randn(1, 2, 32), points, torch.zeros(1, 12), torch.randn(1, 2, 3, 3))
    losses["loss"].backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
    model.eval()
    action, offsets = model.sample_actions_and_point_offsets(torch.device("cpu"), obs, points, torch.zeros(1, 12), num_steps=2)
    assert action.shape == (1, 2, 32) and offsets.shape == (1, 2, 3, 3)
    assert torch.isfinite(action).all() and torch.isfinite(offsets).all()
    with torch.device("meta"):
        reloaded = SurfaceActionModelPytorch(model.config)
    reloaded.to_empty(device="cpu")
    reloaded.load_state_dict(model.state_dict(), strict=True)
    restore_nonpersistent_buffers(reloaded)
    reloaded.eval()
    torch.manual_seed(81)
    reference = model.sample_actions_and_point_offsets(torch.device("cpu"), obs, points, torch.zeros(1, 12), num_steps=2)
    torch.manual_seed(81)
    actual = reloaded.sample_actions_and_point_offsets(torch.device("cpu"), obs, points, torch.zeros(1, 12), num_steps=2)
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected)


def test_watchdog_stops_without_a_new_planner_request():
    import threading
    from safety_module.command_watchdog import CommandWatchdog
    stopped = threading.Event()
    watchdog = CommandWatchdog(stopped.set, .03)
    try:
        watchdog.renew()
        assert stopped.wait(.5)
    finally:
        watchdog.close()


def test_invalid_calibration_blocks_before_robot_connections(tmp_path):
    from real_scripts.live_dual_ur7e_obstacle_model import load_arms
    from safety_module.project_config import DEFAULT_PROJECT_CONFIG
    import yaml
    config = yaml.safe_load(DEFAULT_PROJECT_CONFIG.read_text())
    config["safety"]["camera_calibration_valid"] = False
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError, match="calibration expired"):
        load_arms(SimpleNamespace(project_config=path))


def test_shared_tokenizer_is_used_by_both_entrypoints():
    from scripts.train_pi05_ur7e_surface_pytorch import _Tokenizer as training
    from scripts.serve_quest3_pi05_safety import _Tokenizer as serving
    assert training is serving


def test_installer_checks_sessions_before_activating(tmp_path, monkeypatch):
    import json
    import yaml
    from scripts.install_surface_calibration import main
    from safety_module.project_config import DEFAULT_PROJECT_CONFIG
    config = yaml.safe_load(DEFAULT_PROJECT_CONFIG.read_text())
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(yaml.safe_dump({"cameras": {"front": {"serial": "f"}, "side": {"serial": "s"}},
                                      "arms": {"left_arm": {}, "right_arm": {}}}))
    config["quest3"]["hardware_config"] = str(hardware)
    for key in ("front_calibration", "side_calibration", "right_front_calibration"):
        config["safety"][key] = str(tmp_path / (key + ".yaml"))
    config_path = tmp_path / "project.yaml"; config_path.write_text(yaml.safe_dump(config))
    camera = dict(camera_to_world=np.eye(4).tolist(), intrinsics=[[500, 0, 320], [0, 500, 240], [0, 0, 1]], width=640, height=480, charuco_reprojection_rms_px=.2)
    session = dict(tcp_board_probe={"rms_m": .001}, cameras={"f": camera, "s": camera})
    left = tmp_path / "left.json"; right = tmp_path / "right.json"
    left.write_text(json.dumps(session)); right.write_text(json.dumps(session))
    monkeypatch.setattr("sys.argv", ["install", "--project-config", str(config_path), "--left-session", str(left), "--right-session", str(right)])
    main()
    assert yaml.safe_load(config_path.read_text())["safety"]["camera_calibration_valid"]
    with pytest.raises(FileExistsError): main()


def test_two_view_perception_without_hardware(monkeypatch):
    import sys
    from real_scripts import live_dual_ur7e_obstacle_model as live
    from real_scripts.real_robot_adapter import RGBDFrame, CameraCalibration
    K = np.array([[100., 0, 40], [0, 100., 30], [0, 0, 1.]])
    cal = CameraCalibration("front", K, np.eye(4))
    arms = (live.Arm("left", "fake-left", np.eye(4), "2f85", np.eye(4)),
            live.Arm("right", "fake-right", np.eye(4), "2f85", np.eye(4)))
    monkeypatch.setattr(live, "load_arms", lambda args: (*arms, cal, {"camera_serial": "front", "right_base_to_left_base": np.eye(4).tolist()}))
    monkeypatch.setattr(live, "load_calibration", lambda path: (np.eye(4), K, "side", {"parent_frame": "left_base"}))
    class Receiver:
        def __init__(self, ip): pass
        def getActualQ(self): return np.zeros(6)
        def getActualTCPPose(self): return np.zeros(6)
        def disconnect(self): pass
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=Receiver))
    class Gripper:
        def __init__(self, ip, **kwargs): self.value = .3 if ip == "fake-left" else .7
        def closing_fraction(self): return self.value
        def close(self): pass
    monkeypatch.setattr(live, "ReadOnlyRobotiq2FPosition", Gripper)
    class Source:
        def __init__(self, *, cameras, **kwargs):
            assert [c.name for c in cameras] == ["front", "side"]
        def start(self): pass
        def stop(self): pass
        def get_color_calibrations(self): return {n: {"intrinsics": K.tolist()} for n in ("front", "side")}
        def read(self):
            stamp = time.monotonic_ns()
            depth = np.full((60, 80), .6, np.float32); depth[20:40, 30:50] = .55
            return {n: RGBDFrame(n, np.zeros((60, 80, 3), np.uint8), depth.copy(), host_timestamp_ns=stamp) for n in ("front", "side")}
    monkeypatch.setattr(live, "RealSenseD435iSource", Source)
    def render(*args, **kwargs):
        depth = np.zeros((60, 80), np.float32); depth[:6] = .6
        return depth
    monkeypatch.setattr(live, "render_surface_points_depth", render)
    monkeypatch.setattr("sys.argv", ["perception", "--once", "--no-browser", "--no-lingbot-depth", "--warmup-frames", "1", "--temporal-persistence-frames", "1"])
    snapshot = live.main()
    assert snapshot["camera_names"] == ["front", "side"]
    assert snapshot["gripper_position"] == pytest.approx({"left": .3, "right": .7})
    assert len(snapshot["current_surface"]) > 0
    assert snapshot["diagnostics"]["environment_points"] > 0
