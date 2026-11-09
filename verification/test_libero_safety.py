"""LIBERO-Safety contracts, substep scoring and training-format integration."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"oa-vla/src"))
from obstacle_validation.libero_safety import (ConstraintMonitor, first_action, parse_args,
                                              prepare_paths, validate_metadata, rollout)
from obstacle_validation.libero_safety_geometry import CAMERAS, STATE_SEMANTICS
from obstacle_validation.oa_vla_actions import ObservationHold
from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
from obstacle_validation.ellipsoid_geometry import Ellipsoid


def metadata():
    return dict(model_type="oa-vla", model_format=2, state_dim=8, action_dim=7,
        action_semantics="libero_osc_pose", state_semantics=STATE_SEMANTICS,
        policy_views=list(CAMERAS), camera_names=CAMERAS, image_convention="opencv_top_left",
        action_period_s=.05, detection_views=["front", "side"], detection_trained=True,
        detection_box_format="cxcywh_angle_deg_original_pixels", pair_id_scope="current_prediction")


def test_panda_contract_rejects_ur_weights_wrong_cameras_and_random_safety():
    validate_metadata(metadata(), CAMERAS)
    for changes in (dict(state_dim=12, action_dim=14), dict(action_semantics=None),
                    dict(camera_names={}), dict(detection_trained=False), dict(test_only=True)):
        with pytest.raises(ValueError):
            validate_metadata({**metadata(), **changes}, CAMERAS)
    validate_metadata({**metadata(), "test_only": True}, CAMERAS, safety=False, allow_test=True)


def test_only_first_action_is_executed_without_delta_accumulation():
    first = np.arange(7)/10
    np.testing.assert_equal(first_action(dict(actions=[first, np.ones(7)*99])), first)
    with pytest.raises(ObservationHold):
        first_action(dict(actions=[np.ones(7)*2]))
    with pytest.raises(ValueError):
        first_action(dict(actions=np.zeros((2, 14))))


def test_monitor_catches_duplicate_predicates_transient_contacts_and_success_step():
    sim = SimpleNamespace(t=0)
    def step():
        sim.t += 1
    sim.step = step
    predicates = [["contact", "a"], ["contact", "b"]]
    base = SimpleNamespace(sim=sim, parsed_problem={"constraints": predicates},
                           _eval_predicate=lambda p: sim.t == (1 if p[1] == "a" else 2))
    monitor = ConstraintMonitor(SimpleNamespace(env=base))
    with monitor.track():
        sim.step(); sim.step(); sim.step()
    assert [v["index"] for v in monitor.violations] == [0, 1]
    assert sim.step is step


class SingleRobot:
    def bodies(self, q, grip):
        return [Ellipsoid(np.r_[q[:3]], np.eye(3)*.01**2, "tool", arm=0)]

    def differential(self, q, grip):
        dc = np.zeros((1, 3, 6)); dc[0, :, :3] = np.eye(3)
        return self.bodies(q, grip), dc, np.zeros((1, 3, 3, 6))

    def inter_arm_collision_pair(self, a, b):
        return False


def test_shared_mcbf_supports_six_osc_controls_and_one_exogenous_gripper():
    from real_scripts.real_cbf_qp import OrientedBox
    cbf = EllipsoidMCBF(None, margin=.005, inter_arm_margin=0, period=.05,
        robot=SingleRobot(), joint_groups=(6,), grip_dim=1,
        config=EllipsoidMCBFConfig(max_control_step_rad=.1))
    nominal = np.array([[.05, 0, 0, 0, 0, 0, .7]])
    obstacle = OrientedBox(np.array([.04, 0, 0]), np.eye(3), np.full(3, .01))
    _, info = cbf.plan(np.zeros(6), np.array([.7]), nominal, [obstacle],
                       lower=-np.ones(6), upper=np.ones(6), max_step=.1)
    assert info["steps"][0]["optimization_applied"]
    assert len(info["waypoints"][0]) == 7
    assert info["waypoints"][0][0] < .03
    assert info["waypoints"][0][-1] == .7


def test_config_isolated_and_original_libero_rejected(tmp_path):
    root = tmp_path/"benchmark"
    (root/"libero/libero/benchmark").mkdir(parents=True)
    with pytest.raises(ValueError, match="LIBERO-Safety"):
        prepare_paths(root, tmp_path/"config")
    (root/"libero/libero/benchmark/vla_safety_task_map.py").touch()
    _, paths = prepare_paths(root, tmp_path/"config")
    assert paths["assets"] == str(root/"libero/libero/assets")


def test_cli_prevents_training_on_l2_and_native_policy_safety():
    with pytest.raises(SystemExit):
        parse_args(["--libero-root", "/tmp/x", "--backend", "lerobot"])
    with pytest.raises(SystemExit):
        parse_args(["--libero-root", "/tmp/x", "--record-training", "/tmp/train", "--level", "2"])


def make_training_shard(tmp_path):
    from obstacle_validation.libero_safety_recording import TrainingRecorder
    recorder = TrainingRecorder.__new__(TrainingRecorder)
    recorder.cameras, recorder.names, recorder.horizon = dict(CAMERAS), ["hazard"], 3
    recorder.rows = [dict(rgb={k: np.zeros((16, 16, 3), np.uint8) for k in CAMERAS},
        state=np.ones(8)*i, action=np.ones(7)*i/10,
        boxes=np.tile(np.array([.5, .5, .2, .1, 1, 0]), (2, 1, 1)),
        valid=np.ones((2, 1), bool)) for i in range(2)]
    path = tmp_path/"episode.shard"
    recorder.save(path, "move the bowl")
    return path


def test_recorded_shards_are_trainable_with_profile_and_masked_chunk_tails(tmp_path):
    pytest.importorskip("torch")
    from oa_vla.training import normalization_stats
    from oa_vla.data import OAVlaDataset
    from oa_vla.tokenizer import ByteTokenizer
    path = make_training_shard(tmp_path)
    stats, contract = normalization_stats([path])
    assert contract == (3, 7, 8)
    assert stats["action_semantics"] == "libero_osc_pose"
    assert stats["camera_names"] == CAMERAS
    dataset = OAVlaDataset([path], tokenizer=ByteTokenizer(32), action_horizon=3,
        **{k: stats[k] for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")})
    item = dataset[1]
    assert item["action_mask"].sum() == 7
    assert item["detection_valid"].sum() == 2


def test_tiny_panda_training_export_load_and_real_tcp(tmp_path):
    pytest.importorskip("torch")
    import threading
    import time
    from oa_vla.train import main as train
    from oa_vla.policy import OAVLAPolicy
    from oa_vla.serve import PolicyServer, policy_metadata
    from oa_vla.socket_protocol import PolicyClient
    path = make_training_shard(tmp_path)
    output = tmp_path/"export"
    train(["--tiny", "--allow-random-init", "--device", "cpu", "--dataset", str(path), "--output", str(output),
           "--epochs", "1", "--max-steps", "1", "--stage", "joint"])
    policy = OAVLAPolicy.from_pretrained(output)
    metadata_value = policy_metadata(policy)
    assert metadata_value["state_dim"] == 8 and metadata_value["action_dim"] == 7
    assert metadata_value["action_semantics"] == "libero_osc_pose"
    assert metadata_value["test_only"]  # one-step random training never becomes a real policy
    assert metadata_value["camera_names"] == CAMERAS
    try:
        server = PolicyServer(("127.0.0.1", 0), policy, metadata=metadata_value, num_steps=1)
    except PermissionError:
        pytest.skip("Host disallows local TCP; model training/export already checked")
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        stamp = time.monotonic_ns()
        context = dict(frame_id=0, capture_ns=stamp, feedback_ns=stamp, calibration_id="test_panda",
                       views={k: dict(timestamp_ns=stamp, height=16, width=16) for k in CAMERAS})
        with PolicyClient(*server.server_address) as client:
            result = client.infer({k: np.zeros((16,16,3),np.uint8) for k in CAMERAS}, np.zeros(8),
                                  "move the bowl", context=context)
        assert np.asarray(result["actions"]).shape == (3,7)
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_normalizer_renames_action_before_lerobot_chunk_resolution_and_preserves_source(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location("normalize_safety", ROOT/"scripts/normalize_libero_safety_dataset.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, target = tmp_path/"source", tmp_path/"derived"
    (source/"meta").mkdir(parents=True)
    info = dict(codebase_version="v2.1", fps=20, total_episodes=1, features={
        "actions": dict(shape=[7]), "observation.state": dict(shape=[8]),
        "observation.image": dict(shape=[256, 256, 3]), "observation.wrist_image": dict(shape=[256, 256, 3])})
    (source/"meta/info.json").write_text(json.dumps(info))
    (source/"meta/episodes_stats.jsonl").write_text(json.dumps(dict(stats={"actions": dict(mean=[0]*7)}))+"\n")
    data = source/"data/chunk-000"; data.mkdir(parents=True)
    pq.write_table(pa.table({"actions": [[0.]*7], "observation.state": [[0.]*8]}), data/"episode_000000.parquet")
    for key in ("observation.image", "observation.wrist_image"):
        v = source/"videos/chunk-000"/key; v.mkdir(parents=True)
        (v/"episode_000000.mp4").write_bytes(b"video fixture")
    module.normalize(source, target, link_videos=True)
    assert pq.read_table(target/"data/chunk-000/episode_000000.parquet").column_names == ["action", "observation.state"]
    assert "actions" in json.loads((source/"meta/info.json").read_text())["features"]
    assert (target/"videos/chunk-000/observation.images.image/episode_000000.mp4").exists()
    assert not (target/"INCOMPLETE").exists()


def test_metric_actions_and_statistics_use_the_same_osc_scaling():
    spec = importlib.util.spec_from_file_location("normalize_safety", ROOT/"scripts/normalize_libero_safety_dataset.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    scale = np.array([.05]*3+[.5]*3+[1.])
    raw = dict(stats=dict(actions=dict(mean=scale.tolist(), std=(scale/2).tolist(), count=[100])),
               unrelated="actions")
    canonical = module.rescale_stats(module.rename_metadata(raw), scale)
    np.testing.assert_allclose(canonical["stats"]["action"]["mean"], np.ones(7))
    np.testing.assert_allclose(canonical["stats"]["action"]["std"], np.full(7, .5))
    assert canonical["stats"]["action"]["count"] == [100]
    assert raw["stats"]["actions"]["mean"] == scale.tolist()


def test_quantiles_are_computed_from_canonical_training_vectors(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location("quantiles_safety", ROOT/"scripts/compute_lerobot_quantiles.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/"meta").mkdir(); (tmp_path/"data").mkdir()
    (tmp_path/"meta/info.json").write_text(json.dumps(dict(codebase_version="v3.0", total_frames=2,
        features={"observation.state": dict(shape=[8]), "action": dict(shape=[7])})))
    (tmp_path/"meta/stats.json").write_text(json.dumps({k: dict(mean=[1]*n) for k,n in (("observation.state",8),("action",7))}))
    pq.write_table(pa.table({"observation.state": [[0.]*8, [2.]*8], "action": [[0.]*7, [4.]*7]}), tmp_path/"data/part.parquet")
    module.compute(tmp_path)
    stats = json.loads((tmp_path/"meta/stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["q01"], [.04]*7)
    np.testing.assert_allclose(stats["observation.state"]["q99"], [1.98]*8)

