from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from real_scripts.real_cbf_qp import (OrientedBox, PointFlowConstraint,
                                      project_joint_delta_qp, select_inter_arm_constraints,
                                      select_point_flow_constraints)
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


def test_rounded_box_margin_matches_euclidean_corner_clearance():
    box = OrientedBox(np.zeros(3), np.eye(3), np.asarray([.01, .01, .1]))
    current = np.asarray([[-.05, -.05, 0.]])
    legacy = select_point_flow_constraints(current, current[None], [box],
                                           collision_margin_m=.05, trigger_margin_m=.01)
    rounded = select_point_flow_constraints(current, current[None], [box],
                                            collision_margin_m=.05, trigger_margin_m=.01,
                                            rounded_margin=True)
    assert legacy[0].current_h < 0
    assert rounded[0].current_h == pytest.approx(np.hypot(.04, .04)-.05)


def test_current_cbf_barrier_does_not_make_safe_braking_infeasible():
    item = PointFlowConstraint(np.array([-1., 0., 0.]), .06, -.04, 0, 0, 0)
    jacobian = np.zeros((1, 3, 12)); jacobian[0, 0, 0] = 1.
    nominal = np.zeros(12); nominal[0] = .03
    lower, upper = np.full(12, -.05), np.full(12, .05)
    lower[0], upper[0] = .02, .03
    _, legacy = project_joint_delta_qp(nominal, jacobian, [item], lower_delta=lower,
                                       upper_delta=upper, iterations=64)
    safe, corrected = project_joint_delta_qp(nominal, jacobian, [item], lower_delta=lower,
                                             upper_delta=upper, iterations=64,
                                             use_current_barrier=True)
    assert not legacy['success']
    assert corrected['success']
    assert .02 <= safe[0] <= .03


def test_future_surface_intrusion_is_removed_by_action_projection():
    item = PointFlowConstraint(np.array([-1., 0., 0.]), .06, -.04, 0, 0, 0)
    jacobian = np.zeros((1, 3, 12)); jacobian[0, 0, 0] = 1.
    nominal = np.zeros(12); nominal[0] = .1
    safe, info = project_joint_delta_qp(
        nominal, jacobian, [item], lower_delta=np.full(12, -.1),
        upper_delta=np.full(12, .1), iterations=64,
        use_predicted_linearization=True)
    assert info['success']
    assert safe[0] == pytest.approx(.06, abs=1e-6)
    # At the nominal endpoint h=-.04; changing the action by -.04
    # raises that same future point to the clearance boundary.
    assert item.predicted_h + float(item.normal[0]) * (safe[0]-nominal[0]) >= -1e-6


def test_future_point_normal_uses_predicted_side_of_box():
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    current = np.array([[.35, 0., 0.]])
    future = np.array([[[.55, 0., 0.]]])
    old = select_point_flow_constraints(current, future, [box], normal_at_prediction=False)
    projected = select_point_flow_constraints(current, future, [box], normal_at_prediction=True)
    assert old[0].normal[0] == pytest.approx(-1.)
    assert projected[0].normal[0] == pytest.approx(1.)


def test_future_inter_arm_normal_uses_predicted_point_order():
    current = np.array([[-.1, 0., 0.], [.1, 0., 0.]])
    future = np.array([[[.01, 0., 0.], [-.01, 0., 0.]]])
    mask = np.array([True, False])
    old = select_inter_arm_constraints(current, future, mask, normal_at_prediction=False)
    projected = select_inter_arm_constraints(current, future, mask,
                                              normal_at_prediction=True)
    assert old[0].normal[0] == pytest.approx(-1.)
    assert projected[0].normal[0] == pytest.approx(1.)


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


@pytest.mark.parametrize("refinement", [False, True])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("map_mode", ["default", "on", "off"])
def test_two_view_perception_without_hardware(monkeypatch, refinement, noise, map_mode):
    import json
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
        def __init__(self, ip): self.arm = "left_arm" if ip == "fake-left" else "right_arm"
        def getTimestamp(self): return 1.
        def getActualQ(self): return np.zeros(6)
        def getActualTCPPose(self):
            from scipy.spatial.transform import Rotation
            from safety_module.project_config import load_project_config
            flange_to_tcp = np.asarray(load_project_config()[0]["robot"]["flange_to_active_tcp"][self.arm]["matrix"])
            transform = live.flange_transform(np.zeros(6)) @ flange_to_tcp
            return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]
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
        def get_stream_info(self): return {}
        def read(self):
            stamp = time.monotonic_ns()
            depth = np.full((60, 80), .6, np.float32); depth[20:40, 30:50] = .55
            # A three-pixel surface entirely between the stride=3 sample sites.
            depth[46, 70:72] = .48
            depth[47, 70] = .48
            if noise:
                # Neither pixel is on the point_stride=3 lattice. Both cameras
                # see these unsupported returns: they must not confirm each other.
                depth[13, 17] = .4
                depth[43, 62] = .4
            return {n: RGBDFrame(n, np.zeros((60, 80, 3), np.uint8), depth.copy(), host_timestamp_ns=stamp) for n in ("front", "side")}
    monkeypatch.setattr(live, "RealSenseD435iSource", Source)
    def render(*args, **kwargs):
        depth = np.zeros((60, 80), np.float32); depth[:6] = .6
        return depth
    monkeypatch.setattr(live, "render_surface_points_depth", render)
    if refinement:
        class Refiner:
            model_id, device, use_fp16 = "fake", "cpu", False
            camera_names, last_inference_seconds = ("front", "side"), 0.
            def refine(self, frames, _calibrations):
                return [RGBDFrame(f.camera_name, f.rgb, np.zeros_like(f.depth_m),
                                  host_timestamp_ns=f.host_timestamp_ns) for f in frames]
        monkeypatch.setattr(live, "create_lingbot_depth_refiner_from_args", lambda _args: Refiner())
    flags = ["perception", "--once", "--no-browser",
             "--lingbot-depth" if refinement else "--no-lingbot-depth",
             "--warmup-frames", "1", "--temporal-persistence-frames", "1"]
    if map_mode == "off":
        flags += ["--no-occupancy-map", "--point-stride", "1"]
        def forbidden(*_args, **_kwargs):
            raise AssertionError("Disabled map must not allocate, integrate or lift dense endpoints")
        for name in ("LocalOccupancyGrid", "workspace_occupancy_grid", "integrate_depth_views"):
            monkeypatch.setattr(live, name, forbidden)
        monkeypatch.setattr(live.RawDepthEvidenceFilter, "world_points", forbidden)
    else:
        flags += ["--min-cluster-points", "1000000"]
        if map_mode == "on":
            flags += ["--occupancy-map"]
    monkeypatch.setattr("sys.argv", flags)
    snapshot = live.main()
    assert snapshot["camera_names"] == ["front", "side"]
    assert snapshot["gripper_position"] == pytest.approx({"left": .3, "right": .7})
    assert len(snapshot["current_surface"]) > 0
    assert snapshot["diagnostics"]["environment_points"] > 0
    assert len(snapshot["_viewer_environment"]["points"]) >= len(snapshot["environment"]["points"])
    assert snapshot["diagnostics"]["viewer_outside_workspace_points"] >= 0
    assert snapshot["published_timestamp_ns"] >= snapshot["host_timestamp_ns"]
    assert snapshot["diagnostics"]["capture_to_publish_seconds"] >= 0
    assert snapshot["scene_observation_state"] == "observed"
    assert not snapshot["unknown_obstacles"]
    assert snapshot["diagnostics"]["table_prior"]["state"] in ("initialized", "updated")
    assert all(e["measured_points"] > 0 for e in snapshot["diagnostics"]["depth_evidence"].values())
    from real_scripts.real_cbf_qp import boxes_from_snapshot
    if map_mode == "off":
        assert snapshot["obstacle_backend"] == "clustered_obbs"
        assert snapshot["diagnostics"]["occupancy_map_enabled"] is False
        assert "occupancy_map" not in snapshot and "occupancy" not in snapshot["diagnostics"]
        assert snapshot["obbs"] and boxes_from_snapshot(snapshot).observation is None
        validate_snapshot(snapshot, 2.)
        for options in ({}, {"compact": True}, {"viewer": True}):
            payload = json.loads(live.snapshot_payload(snapshot, **options))
            assert "occupancy_map" not in payload
            assert all(box["geometry_source"] == "clustered_obbs" and "point_count" in box
                       and "voxel_count" not in box for box in payload["obbs"])
        if noise:
            assert all(e["isolated_pixels"] == 2 and e["uncertain_raw_pixels"] == 2
                       for e in snapshot["diagnostics"]["depth_evidence"].values())
        return
    assert snapshot["diagnostics"]["occupancy_map_enabled"] is True
    assert snapshot["obstacle_backend"] == "probabilistic_voxels"
    assert snapshot["occupancy_map"]["occupied_voxels"] > 0
    assert boxes_from_snapshot(snapshot).observation is not None
    from safety_module.occupancy import ObservationGrid, OCCUPIED
    observation = ObservationGrid(snapshot["occupancy_map"])
    tiny = np.array([[(70-40)/100*.48, (46-30)/100*.48, .48]])
    tiny_ids, inside = observation.indices(tiny)
    assert inside.all() and observation.state[tiny_ids[0]] == OCCUPIED
    if noise:
        from safety_module.occupancy import ObservationGrid, FREE
        view = ObservationGrid(snapshot["occupancy_map"])
        points = np.array([[(17-40)/100*.4, (13-30)/100*.4, .4],
                           [(62-40)/100*.4, (43-30)/100*.4, .4]])
        ids, inside = view.indices(points)
        assert inside.all() and np.all(view.state[ids] == FREE)
        assert snapshot["diagnostics"]["occupancy"]["uncertain_raw_voxels"] > 0
        assert all(e["isolated_pixels"] == 2 for e in snapshot["diagnostics"]["depth_evidence"].values())
        assert all(e["uncertain_raw_pixels"] == 2 for e in snapshot["diagnostics"]["depth_evidence"].values())


def test_compact_scene_preserves_capture_state_geometry_and_calibration():
    import json
    from real_scripts.live_dual_ur7e_obstacle_model import snapshot_payload
    snapshot = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                "host_timestamp_ns": 100, "published_timestamp_ns": 200, "status": "ok",
                "qpos": {"left": [0.]*6, "right": [0.]*6},
                "gripper_position": {"left": .3, "right": .7},
                "obbs": [{"center": [1., 2., 3.], "axes": np.eye(3).tolist(), "half_sizes": [.1]*3}],
                "calibration": {"sha256": "calibration-id"}, "diagnostics": {"processing_seconds": .2},
                "environment": {"points": [[0.]*3]*1000},
                "_viewer_environment": {"points": [[1.]*3]*1000},
                "robot": {"points": [[0.]*3]*1000},
                "current_surface": [[0.]*3]*1000, "rgb_front_jpeg_base64": "jpeg"}
    full = snapshot_payload(snapshot)
    viewer = snapshot_payload(snapshot, viewer=True)
    compact = snapshot_payload(snapshot, compact=True)
    decoded = json.loads(compact)
    assert decoded == {k: v for k, v in snapshot.items()
                       if k not in ("environment", "_viewer_environment", "robot", "current_surface", "rgb_front_jpeg_base64")}
    assert json.loads(full) == {k: v for k, v in snapshot.items() if k != "_viewer_environment"}
    assert json.loads(viewer)["environment"] == snapshot["_viewer_environment"]
    assert len(compact) < len(full)/10


def test_live_snapshot_defers_display_work_until_requested(monkeypatch):
    import json
    from real_scripts import live_dual_ur7e_obstacle_model as live

    points = np.asarray([[0., 0., .1], [1., 0., .1]], dtype=np.float32)
    colors = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    display = live.SnapshotDisplay(
        points[:1], colors[:1], points, colors, np.asarray([True, True]),
        points[:1], colors[:1], points[:1], np.zeros((4, 4, 3), np.uint8), 10,
    )
    snapshot = {"frame": 7, "obbs": [], "_display_data": display}
    with monkeypatch.context() as patch:
        patch.setattr(live.cv2, "imencode", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("JPEG should not be encoded for compact or viewer requests")))
        assert json.loads(live.snapshot_payload(snapshot, compact=True)) == {"frame": 7, "obbs": []}
        viewer = json.loads(live.snapshot_payload(snapshot, viewer=True))
    assert len(viewer["environment"]["points"]) == 2
    assert "current_surface" not in viewer and "rgb_front_jpeg_base64" not in viewer
    full = json.loads(live.snapshot_payload(snapshot))
    assert len(full["environment"]["points"]) == 1
    assert len(full["current_surface"]) == 1
    assert full["rgb_front_jpeg_base64"]


def test_viewer_adds_tabletop_outside_detection_workspace_only():
    from real_scripts.live_dual_ur7e_obstacle_model import viewer_environment_points

    points = np.asarray([[0., 0., 0.], [0., 1., 0.], [0., 1., -.2],
                         [0., 1., .5], [0., 1., .1]], dtype=np.float32)
    colors = np.arange(15, dtype=np.uint8).reshape(5, 3)
    workspace = np.asarray([True, False, False, False, False])
    shown, shown_colors, extra_count = viewer_environment_points(
        points, colors, workspace, np.asarray([0., 0., 1.]), 0.,
        plane_threshold_m=.012, max_height_m=.35, box_margin_m=.01,
    )

    np.testing.assert_array_equal(shown, points[[0, 1, 4]])
    np.testing.assert_array_equal(shown_colors, colors[[0, 1, 4]])
    assert extra_count == 2
    np.testing.assert_array_equal(workspace, [True, False, False, False, False])


def test_dual_arm_workspace_uses_each_calibrated_base_frame():
    from real_scripts.live_dual_ur7e_obstacle_model import dual_arm_workspace_masks

    right_to_left = np.asarray([[0., -1., 0., 2.], [1., 0., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
    points = np.asarray([[0., 0., 0.], [2., 0., 0.], [2., .4, 0.],
                         [2.4, 0., 0.], [1., 0., 0.]], dtype=np.float32)
    bounds = (-.5, .5, -.5, .5, -.1, .5)

    left, right, union = dual_arm_workspace_masks(points, bounds, bounds, right_to_left)

    np.testing.assert_array_equal(left, [True, False, False, False, False])
    np.testing.assert_array_equal(right, [False, True, True, True, False])
    np.testing.assert_array_equal(union, [True, True, True, True, False])


def test_overlapping_obstacle_boxes_merge_without_losing_covered_volume():
    from real_scripts.live_dual_ur7e_obstacle_model import (
        expanded_obb, merge_overlapping_obbs, tabletop_obb,
        upright_obb_overlap_ratio, upright_obbs_overlap,
    )

    normal = np.asarray([0., 0., 1.])

    def box(xmin, xmax, ymin=-.08, ymax=.08):
        corners = np.asarray([[x, y, z] for z in (0., .2)
                              for x in (xmin, xmax) for y in (ymin, ymax)])
        return tabletop_obb(corners, normal, 0.)

    first, second, separate = box(-.2, .1), box(0., .3), box(.55, .75)
    merged = merge_overlapping_obbs([first, second, separate], normal, 0.)

    assert len(merged) == 2
    assert merged[0].point_count == first.point_count + second.point_count
    assert not upright_obbs_overlap(*merged)
    for original in (first, second):
        local = (original.corners - merged[0].center) @ merged[0].rotation
        assert np.all(np.abs(local) <= merged[0].extents / 2 + 1e-5)
    assert len(merge_overlapping_obbs([box(-.2, 0.), box(-.05, .15), box(.1, .3)], normal, 0.)) == 1

    # Margins can make two otherwise separate boxes overlap in the published scene.
    near_a, near_b = box(0., .1), box(.115, .215)
    assert len(merge_overlapping_obbs([near_a, near_b], normal, 0.)) == 2
    inflated = [expanded_obb(item, .01) for item in (near_a, near_b)]
    assert upright_obbs_overlap(*inflated)
    assert 0 < upright_obb_overlap_ratio(*inflated) < .10
    assert len(merge_overlapping_obbs(inflated, normal, 0.)) == 2
    assert len(merge_overlapping_obbs(inflated, normal, 0., min_overlap_ratio=0.)) == 1

    # A meaningful shared volume merges; the threshold uses the smaller volume.
    deep_a, deep_b = box(0., .1), box(.075, .175)
    ratio = upright_obb_overlap_ratio(deep_a, deep_b)
    assert ratio == pytest.approx(.25, abs=1e-5)
    assert len(merge_overlapping_obbs([deep_a, deep_b], normal, 0.)) == 1
    assert len(merge_overlapping_obbs([deep_a, deep_b], normal, 0., min_overlap_ratio=ratio)) == 2
    assert upright_obb_overlap_ratio(box(-.1, .1), box(-.03, .03, -.03, .03)) == pytest.approx(1.)
    with pytest.raises(ValueError, match="overlap ratio"):
        merge_overlapping_obbs([deep_a, deep_b], normal, 0., min_overlap_ratio=1.)

    tilted_normal = np.asarray([0., 1., 1.]) / np.sqrt(2.)
    tangent = np.cross(tilted_normal, [1., 0., 0.])
    tilted_boxes = []
    for center_x, angle in ((0., 0.), (.12, np.pi / 5)):
        along = np.cos(angle) * np.asarray([1., 0., 0.]) + np.sin(angle) * tangent
        across = np.cross(tilted_normal, along)
        corners = np.asarray([center_x * np.asarray([1., 0., 0.]) + x * along + y * across + z * tilted_normal
                              for z in (0., .2) for x in (-.15, .15) for y in (-.06, .06)])
        tilted_boxes.append(tabletop_obb(corners, tilted_normal, 0.))
    tilted_merged = merge_overlapping_obbs(tilted_boxes, tilted_normal, 0.)
    assert len(tilted_merged) == 1
    for original in tilted_boxes:
        local = (original.corners - tilted_merged[0].center) @ tilted_merged[0].rotation
        assert np.all(np.abs(local) <= tilted_merged[0].extents / 2 + 1e-5)


def test_stale_track_does_not_merge_with_new_candidate():
    from real_scripts.live_dual_ur7e_obstacle_model import (
        TemporalOBBGate, tabletop_obb,
    )

    normal = np.asarray([0., 0., 1.])
    def box(xmin, xmax):
        corners = np.asarray([[x, y, z] for z in (0., .2)
                              for x in (xmin, xmax) for y in (-.08, .08)])
        return tabletop_obb(corners, normal, 0.)

    gate = TemporalOBBGate(required_hits=1, match_distance_m=.01)
    gate.update([box(-.2, .1)])
    published = gate.update([box(0., .3)])
    assert len(gate._tracks) == 2
    assert len(published) == 1
    assert len(gate.unknown_obstacles) == 1
    assert gate._tracks[0].box is None
    assert "missing_cluster" in gate.unstable_reasons


def test_large_overlap_merges_published_boxes_before_stability_gate():
    from real_scripts.live_dual_ur7e_obstacle_model import (
        TemporalOBBGate, merge_overlapping_obbs, safety_obbs_with_candidates,
        tabletop_obb, upright_obb_overlap_ratio,
    )

    normal = np.array([0., 0., 1.])
    def cloud(xmin, xmax):
        return np.array([[x, y, z] for x in np.linspace(xmin, xmax, 8)
                         for y in np.linspace(-.04, .04, 8) for z in (.04, .08)])

    first_points, second_points = cloud(0., .1), cloud(.04, .14)
    first, second = (tabletop_obb(points, normal, 0.)
                     for points in (first_points, second_points))
    assert upright_obb_overlap_ratio(first, second) == pytest.approx(.6, abs=1e-5)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.01)
    tracked = gate.update([first, second], [first_points, second_points])
    assert len(tracked) == 2
    published, _ = safety_obbs_with_candidates(tracked, [first, second],
                                                [first_points, second_points])
    merged = merge_overlapping_obbs(published, normal, 0., min_overlap_ratio=.5)
    assert len(merged) == 1
    for original in (first, second):
        local = (original.corners-merged[0].center) @ merged[0].rotation
        assert np.all(np.abs(local) <= merged[0].extents/2 + 1e-5)

    half_overlap = tabletop_obb(cloud(.05, .15), normal, 0.)
    ratio = upright_obb_overlap_ratio(first, half_overlap)
    assert ratio == pytest.approx(.5, abs=1e-5)
    assert len(merge_overlapping_obbs([first, half_overlap], normal, 0.,
                                      min_overlap_ratio=ratio)) == 2


def test_obb_geometry_follows_cluster_evidence_and_becomes_unknown_on_dropout():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    normal = np.array([0., 0., 1.])
    grid = np.array([[x, y, z] for x in np.linspace(-.04, .04, 8)
                     for y in np.linspace(-.03, .03, 8) for z in (.04, .08)])
    new_points = np.vstack((grid, grid + [.10, 0., 0.]))
    original = tabletop_obb(grid, normal, 0.)
    enlarged = tabletop_obb(new_points, normal, 0.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035,
                           point_change_distance_m=.015, point_change_fraction=.25,
                           change_confirm_frames=2)

    assert gate.update([original], [grid])[0] is original
    # A changed box with the same observed cluster has no point-cloud support.
    assert gate.update([enlarged], [grid])[0] is original
    assert not gate.unstable_reasons
    # A genuine new region must persist across frames before changing geometry.
    assert gate.update([enlarged], [new_points])[0] is original
    assert gate.unstable_reasons == ["changed_cluster"]
    assert gate.update([enlarged], [new_points])[0] is enlarged
    assert not gate.unstable_reasons
    assert gate.update([], []) == []
    assert gate._tracks[0].box is None
    assert len(gate.unknown_obstacles) == 1
    assert gate.unstable_reasons == ["missing_cluster"]


def test_obb_change_uses_current_and_previous_two_clusters():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    normal = np.array([0., 0., 1.])
    original_points = np.array([[x, y, z] for x in np.linspace(-.04, .04, 8)
                                for y in np.linspace(-.03, .03, 8) for z in (.04, .08)])
    changed_points = np.vstack((original_points, original_points + [.10, 0., 0.]))
    original = tabletop_obb(original_points, normal, 0.)
    changed = tabletop_obb(changed_points, normal, 0.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035,
                           history_frames=10)

    assert gate.update([original], [original_points]) == [original]
    assert gate.update([changed], [changed_points]) == [original]
    assert gate.update([changed], [changed_points]) == [original]
    assert gate.update([changed], [changed_points]) == [changed]
    assert gate.unstable_reasons == []
    assert [frame for frame, _ in gate._tracks[0].recent_observations] == [1, 2, 3, 4]


def test_obb_history_is_bounded_but_unknown_does_not_expire_at_window():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    points = np.array([[x, y, z] for x in np.linspace(-.03, .03, 8)
                       for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    box = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035,
                           history_frames=10)
    for _ in range(12):
        assert gate.update([box], [points]) == [box]
    assert [frame for frame, _ in gate._tracks[0].recent_observations] == list(range(3, 13))
    assert gate._tracks[0].reference_frame == 3

    for _ in range(30):
        assert gate.update([], []) == []
        assert gate.expired_tracks == 0
        assert len(gate.unknown_obstacles) == 1
        assert gate._tracks[0].box is None
    assert len(gate._tracks[0].recent_observations) == 0


def test_obbs_supported_by_persistent_merged_cluster_remain_observed():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    normal = np.array([0., 0., 1.])
    left = np.array([[x, y, z] for x in np.linspace(-.12, -.08, 8)
                     for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    right = left + [.20, 0., 0.]
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035,
                           history_frames=10)
    original_boxes = [tabletop_obb(points, normal, 0.) for points in (left, right)]
    assert gate.update(original_boxes, [left, right]) == original_boxes
    merged_points = np.vstack((left, right))
    merged_box = tabletop_obb(merged_points, normal, 0.)
    for _ in range(20):
        assert gate.update([merged_box], [merged_points]) == original_boxes
        assert not gate.unknown_obstacles
    assert gate.expired_tracks == 0
    assert len(gate._tracks) == 2


def test_missing_obb_retires_only_after_measured_free_space_persists():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb
    from real_scripts.real_robot_adapter import CameraCalibration

    points = np.array([[x, y, z] for x in np.linspace(-.06, .06, 8)
                       for y in np.linspace(-.04, .04, 8) for z in (1., 1.04)])
    box = tabletop_obb(points, np.array([0., 0., 1.]), -1.)
    calibration = CameraCalibration("test", np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
                                    np.eye(4))
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035,
                           removal_confirm_frames=3)
    gate.update([box], [points])
    foreground = [(calibration, np.full((100, 100), .8))]
    invalid = [(calibration, np.zeros((100, 100)))]
    clear = [(calibration, np.full((100, 100), 1.5))]
    assert gate.update([], [], free_space_views=foreground) == []
    assert gate.update([], [], free_space_views=invalid) == []
    assert len(gate.unknown_obstacles) == 1
    assert gate.unstable_reasons == ["missing_cluster"]
    assert gate.update([], [], free_space_views=clear) == []
    assert len(gate.unknown_obstacles) == 1
    assert gate.update([], [], free_space_views=clear) == []
    assert len(gate.unknown_obstacles) == 1
    assert gate.update([], [], free_space_views=clear) == []
    assert not gate.unknown_obstacles
    assert not gate.unstable_reasons


def test_duplicate_confirmed_obbs_coalesce_without_merged_cluster_error():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    points = np.array([[x, y, z] for x in np.linspace(-.04, .04, 8)
                       for y in np.linspace(-.03, .03, 8) for z in (.04, .08)])
    box = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    assert len(gate.update([box, box], [points, points])) == 2
    assert len(gate.update([box], [points])) == 1
    assert not gate.unstable_reasons


def test_new_obb_requires_persistent_cluster_before_safety_publication():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    points = np.array([[x, y, z] for x in np.linspace(-.03, .03, 8)
                       for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    box = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    gate = TemporalOBBGate(required_hits=3, match_distance_m=.035)
    assert gate.update([box], [points]) == []
    assert gate.unstable_reasons == ["new_cluster"]
    assert gate.update([box], [points]) == []
    assert gate.update([box], [points]) == [box]
    assert not gate.unstable_reasons


def test_one_frame_candidate_becomes_unknown_without_free_evidence():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    points = np.array([[x, y, z] for x in np.linspace(-.03, .03, 8)
                       for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    box = tabletop_obb(points, np.array([0., 0., 1.]), 0.)
    gate = TemporalOBBGate(required_hits=3, match_distance_m=.035)
    assert gate.update([box], [points]) == []
    assert gate.unstable_reasons == ["new_cluster"]
    assert gate.update([], []) == []
    assert gate.unstable_reasons == ["missing_cluster"]
    assert len(gate.unknown_obstacles) == 1


def test_transient_cluster_merge_keeps_two_observed_boxes():
    from real_scripts.live_dual_ur7e_obstacle_model import TemporalOBBGate, tabletop_obb

    normal = np.array([0., 0., 1.])
    left = np.array([[x, y, z] for x in np.linspace(-.12, -.08, 8)
                     for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    right = left + [.20, 0., 0.]
    boxes = [tabletop_obb(points, normal, 0.) for points in (left, right)]
    gate = TemporalOBBGate(required_hits=1, match_distance_m=.035)
    assert gate.update(boxes, [left, right]) == boxes

    merged_points = np.vstack((left, right))
    merged_box = tabletop_obb(merged_points, normal, 0.)
    assert gate.update([merged_box], [merged_points]) == boxes
    assert gate.unstable_reasons == ["merged_clusters"]
    assert gate.update(boxes, [left, right]) == boxes
    assert not gate.unstable_reasons


def test_safety_scene_keeps_old_boxes_and_adds_uncovered_candidate_points():
    from real_scripts.live_dual_ur7e_obstacle_model import (
        safety_obbs_with_candidates, tabletop_obb,
    )

    normal = np.array([0., 0., 1.])
    left = np.array([[x, y, z] for x in np.linspace(-.12, -.08, 8)
                     for y in np.linspace(-.02, .02, 8) for z in (.04, .08)])
    right = left + [.20, 0., 0.]
    old = [tabletop_obb(points, normal, 0.) for points in (left, right)]
    merged = np.vstack((left, right))
    merged_box = tabletop_obb(merged, normal, 0.)
    boxes, supplemental = safety_obbs_with_candidates(old, [merged_box], [merged])
    assert boxes == old
    assert supplemental == 0

    new_points = left + [.40, 0., 0.]
    new_box = tabletop_obb(new_points, normal, 0.)
    boxes, supplemental = safety_obbs_with_candidates(old, [new_box], [new_points])
    assert boxes == [*old, new_box]
    assert supplemental == 1


def test_rotated_obb_overlap_uses_footprint_area_not_axis_projection():
    from real_scripts.live_dual_ur7e_obstacle_model import UprightOBB, upright_obb_overlap_ratio

    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float32)

    def square(angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=np.float32)
        center = np.asarray([0., 0., .1], dtype=np.float32)
        extents = np.asarray([.2, .2, .2], dtype=np.float32)
        corners = center + (signs * extents / 2) @ rotation.T
        return UprightOBB(center, rotation, extents, corners, 8)

    expected = 1. - .5 * (2. - np.sqrt(2.)) ** 2
    assert upright_obb_overlap_ratio(square(0.), square(np.pi / 4)) == pytest.approx(expected, abs=1e-5)
