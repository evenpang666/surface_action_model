"""Dataset action semantics and output sequencing without robot connections."""

import json

import h5py
import numpy as np
import pytest

from obstacle_validation.quest3_schema import NAMES
from obstacle_validation import lerobot_replay as replay
from verification.replay_test_scene import run_with_selected_scene
from real_scripts.real_cbf_qp import OrientedBox, validate_fk_rollout


@pytest.fixture(autouse=True)
def synchronous_perception_for_command_sequence_tests(monkeypatch):
    # These tests script exact read/send ordering; the concurrent cache has
    # separate tests with a worker blocked on an Event.
    monkeypatch.setattr(replay.LiveSceneReader, "start", lambda self, initial: None)


@pytest.mark.parametrize("execute_horizon,action_horizon,lengths", [
    (None, 32, [10, 10, 3]), (5, 32, [5, 5, 5, 5, 3]), (10, 4, [4, 4, 4, 4, 4, 3])])
def test_avoid_optimizes_and_consumes_only_execution_prefix(
        tmp_path, monkeypatch, execute_horizon, action_horizon, lengths):
    actions = np.zeros((23, 14)); actions[:, 0] = np.arange(1, 24)*.005
    inference_starts, optimized_lengths = [], []

    class Policy:
        period = .05
        states = np.vstack((np.zeros((1, 14)), actions[:-1]))
        source_frames = np.arange(len(actions))
        metadata = {"initial_state": np.zeros(14).tolist(), "final_joint_state": actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = actions
        def infer(self, start, horizon):
            inference_starts.append(start)
            return actions[start:start+horizon].copy()

    class Sampler:
        left_link_names = ('left',)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip): return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    original = replay.RecordedPathCBF.project_chunk
    def project_chunk(self, q, grip, prefix, *args, **kwargs):
        optimized_lengths.append(len(prefix))
        return original(self, q, grip, prefix, *args, **kwargs)

    monkeypatch.setattr(replay, 'DatasetActionPolicy', Policy)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay.RecordedPathCBF, 'project_chunk', project_chunk)
    arguments = ['replay', "--no-execute", '--dataset', str(tmp_path/'dataset'), '--mode', 'avoid', '--avoidance-controller', 'cbf',
                 '--output', str(tmp_path/'run'), '--action-horizon', str(action_horizon)]
    if execute_horizon is not None:
        arguments.extend(['--execute-horizon', str(execute_horizon)])
    report = run_with_selected_scene(replay.parser().parse_args(arguments), monkeypatch)
    assert report['completed'] and report['actions_executed'] == 23
    assert optimized_lengths == lengths
    assert inference_starts == np.cumsum([0, *lengths[:-1]]).tolist()
    assert report['execute_horizon'] == (10 if execute_horizon is None else execute_horizon)
    chunks = [json.loads(line) for line in (tmp_path/'run/chunks.jsonl').read_text().splitlines()]
    assert [chunk['planned_horizon'] for chunk in chunks] == lengths
    assert chunks[0]['inferred_horizon'] == min(action_horizon, len(actions))
    np.testing.assert_allclose(np.concatenate([chunk['dataset_actions'] for chunk in chunks]), actions)
    np.testing.assert_allclose(np.concatenate([chunk['optimized_actions'] for chunk in chunks]), actions, atol=.003)
    assert all(chunk['optimization_mode'] == 'joint_execution_prefix' for chunk in chunks)
    assert [chunk['joint_variables'] for chunk in chunks] == [length*12 for length in lengths]


@pytest.mark.parametrize("mode,controller", [("baseline", "cbf"), ("avoid", "cbf"), ("avoid", "mcbf")])
@pytest.mark.parametrize("speed", [None, .5])
def test_replay_large_targets_have_no_default_rate_limit(tmp_path, monkeypatch, mode, controller, speed):
    actions = np.zeros((3, 14))
    actions[:, 0] = [.07, .14, .21]  # Each target exceeds the former .025 rad/tick cap.
    actions[:, 12:] = [[1., 0.], [0., 1.], [1., 0.]]

    class Policy:
        period = .05
        states = np.vstack((np.zeros((1, 14)), actions[:-1]))
        source_frames = np.arange(len(actions))
        metadata = {"initial_state": np.zeros(14).tolist(), "final_joint_state": actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = actions
        def infer(self, start, horizon): return actions[start:start+horizon].copy()

    class Sampler:
        left_link_names, points_per_link = ("left",), 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip): return np.array([[[3., 3., 3.]], [[4., 4., 4.]]])

    monkeypatch.setattr(replay, "DatasetActionPolicy", Policy)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    args = ["replay", "--no-execute", "--dataset", str(tmp_path/"dataset"), "--mode", mode,
            "--avoidance-controller", controller, "--output", str(tmp_path/"run")]
    if speed is not None:
        args += ["--speed", str(speed), "--arrival-tolerance", ".001"]
    report = run_with_selected_scene(replay.parser().parse_args(args), monkeypatch)
    assert report["completed"] and report["actions_executed"] == 3
    assert report["joint_speed_limit_rad_s"] == speed
    assert report["joint_acceleration_limit_rad_s2"] is None
    assert report["restore_joint_speed_limit_rad_s"] == .5
    assert report["restore_gripper_speed_limit_fraction_s"] == .5
    assert report["gripper_speed"] == 255
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    path = [row for row in rows if row["phase"] == "path"]
    if speed is None and mode == "avoid" and controller == "mcbf":
        assert len(path) > 3
        assert max(np.max(np.abs(row["sent_joint_delta"])) for row in path) <= .040001
        for row in path:
            np.testing.assert_array_equal(row["gripper_target"], actions[row["step"],12:])
    elif speed is None:
        assert len(path) == 3
        if mode == 'baseline':
            np.testing.assert_allclose([row["sent_joint_delta"][0] for row in path], .07)
        else:
            assert all(row['sent_joint_delta'][0] > .025 for row in path)
        np.testing.assert_allclose([row["gripper_target"] for row in path], actions[:, 12:])
    else:
        assert len(path) > 3
        assert max(np.max(np.abs(row["sent_joint_delta"])) for row in path) <= speed*Policy.period+1e-9


def episode(path, *, reversed_arms=False):
    q = np.zeros((4, 12), dtype=float)
    q[:, 0] = [0., .02, .04, .06]
    q[:, 6] = [0., -.01, -.02, -.03]
    residual = np.zeros_like(q)
    residual[:-1, 0] = .001
    residual[:-1, 6] = -.001
    gripper = np.tile([.2, .7], (4, 1))
    grip_actions = np.array([[0., .7], [0., 1.], [0., 1.], [0., 1.]])
    order = list(range(6, 12)) + list(range(6)) if reversed_arms else list(range(12))
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, mode="dual", qpos_joint_names=json.dumps([NAMES[i] for i in order]),
                       action_joint_names=json.dumps([NAMES[i] for i in order]),
                       action_definition="commanded target_q - actual_q (radians)",
                       executed_action_definition="qpos[t+1] - qpos[t] (radians)")
        f["observations/qpos"] = q[:, order]
        f["observations/gripper_position"] = gripper[:, ::-1] if reversed_arms else gripper
        f["action"] = residual[:, order]
        f["executed_action"] = np.vstack((np.diff(q, axis=0), np.zeros(12)))[:, order]
        f["gripper_action"] = grip_actions[:, ::-1] if reversed_arms else grip_actions
        f["action_valid"] = [True, True, True, False]
        f["timestamps_ns"] = np.arange(4, dtype=np.uint64)*50_000_000+1_000_000_000
    return q, grip_actions


def test_latest_replay_executes_old_free_map_without_refresh_wait(tmp_path, monkeypatch):
    import time
    from safety_module.occupancy import LocalOccupancyGrid
    from safety_module.runtime_contract import validate_snapshot
    from verification.test_obstacle_validation import TwoPointSampler
    source = tmp_path/'episode.hdf5'
    episode(source)
    dataset = tmp_path/'dataset'
    replay.convert_hdf5(source, dataset)
    policy = replay.DatasetActionPolicy(dataset)
    stamp = time.monotonic_ns()-10_000_000_000
    grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    for t in (stamp-2_000_000, stamp-1_000_000, stamp):
        grid.update(timestamp_ns=t, raw_hits=[], free_ids=np.arange(grid.size))
    old = {'schema': 'dual_ur7e_obstacle_snapshot_v2', 'coordinate_frame': 'left_base',
           'status': 'ok', 'host_timestamp_ns': stamp, 'occupancy_map': grid.snapshot(), 'obbs': [],
           'qpos': {'left': policy.states[0, :6].tolist(), 'right': policy.states[0, 6:12].tolist()},
           'gripper_position': {'left': .2, 'right': .7}}
    reads = []
    def snapshot(url, max_age, **kwargs):
        assert max_age == float('inf') and kwargs.get('max_wait_s') == 0.
        assert np.isfinite(kwargs['http_timeout_s']) and kwargs['http_timeout_s'] > 0
        assert kwargs.get('after_timestamp_ns') is None
        validate_snapshot(old, max_age)
        reads.append(True)
        return old
    class Sampler(TwoPointSampler):
        calibration_digest = 'test'
        def __init__(self, **kwargs): pass
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'read_snapshot', snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *args: None)
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    args = replay.parser().parse_args(['replay', "--no-execute", '--dataset', str(dataset), '--mode', 'avoid', '--avoidance-controller', 'cbf', '--live',
                                       '--output', str(tmp_path/'run')])
    report = run_with_selected_scene(args, monkeypatch)
    assert report['completed'] and report['actions_executed'] == 3 and reads
    assert report['obstacle_age_policy'] == 'latest'
    assert report['stale_scene_pauses'] == 0
    assert 'fresh_frame_wait' not in report['runtime_timings']
    assert old['host_timestamp_ns'] == old['occupancy_map']['capture_timestamp_ns'] == stamp


def test_conversion_produces_absolute_lerobot_actions_and_grippers(tmp_path):
    source = tmp_path/"episode.hdf5"
    q, grip_actions = episode(source)
    metadata = replay.convert_hdf5(source, tmp_path/"dataset")
    policy = replay.DatasetActionPolicy(tmp_path/"dataset")
    assert metadata["schema"] == replay.SCHEMA
    assert len(policy.actions) == 3
    np.testing.assert_allclose(policy.actions[:, 0], [.021, .041, .06], atol=1e-7)
    np.testing.assert_allclose(policy.actions[:, 6], [-.011, -.021, -.03], atol=1e-7)
    np.testing.assert_allclose(policy.actions[:, 12:], grip_actions[:3])
    np.testing.assert_allclose(policy.states[0, :12], q[0])
    assert (tmp_path/"dataset/meta/info.json").exists()
    assert len(policy.infer(1, 2)) == 2


def test_conversion_normalizes_reversed_arm_order(tmp_path):
    source = tmp_path/"episode.hdf5"
    q, grip_actions = episode(source, reversed_arms=True)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    policy = replay.DatasetActionPolicy(dataset)
    np.testing.assert_allclose(policy.states[0, :12], q[0])
    np.testing.assert_allclose(policy.actions[:, 0], [.021, .041, .06], atol=1e-7)
    np.testing.assert_allclose(policy.actions[:, 6], [-.011, -.021, -.03], atol=1e-7)
    np.testing.assert_allclose(policy.actions[:, 12:], grip_actions[:3])


@pytest.mark.parametrize("fault", ["incomplete", "single_arm", "joint_names", "action_mismatch",
                                  "invalid_action", "timestamp", "definition", "nan"])
def test_conversion_rejects_malformed_episode(tmp_path, fault):
    source = tmp_path/"episode.hdf5"
    episode(source)
    with h5py.File(source, "r+") as f:
        if fault == "incomplete":
            f.attrs["complete"] = False
        elif fault == "single_arm":
            f.attrs["mode"] = "single"
        elif fault == "joint_names":
            f.attrs["qpos_joint_names"] = json.dumps(NAMES[:-1] + [NAMES[0]])
        elif fault == "action_mismatch":
            f["executed_action"][0, 0] += .1
        elif fault == "invalid_action":
            f["action_valid"][1] = False
        elif fault == "timestamp":
            f["timestamps_ns"][1] = 0
        elif fault == "definition":
            f.attrs["action_definition"] = "joint increment"
        else:
            f["observations/qpos"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        replay.convert_hdf5(source, tmp_path/"dataset")


@pytest.mark.parametrize("speed_args,expected_speed", [([], 255), (["--gripper-speed", "80"], 80)])
@pytest.mark.parametrize("restore_args,restore_speed,restore_gripper_speed", [
    ([], .5, .5), (["--restore-speed", ".1", "--gripper-speed-fraction-s", "1"], .1, 1.)])
def test_physical_replay_restores_then_sends_dataset_actions_and_grippers(
        tmp_path, monkeypatch, speed_args, expected_speed, restore_args, restore_speed, restore_gripper_speed):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    policy = replay.DatasetActionPolicy(dataset)
    state = {"q": np.full(12, -.12), "grip": np.array([0., 0.])}
    sent = []
    grip_sent = []
    grip_settings = []
    reads = 0

    class FakeCommander:
        def __init__(self, *args): pass
        def read_state(self):
            nonlocal reads
            reads += 1
            if reads == 3:
                # Feedback can change by more than the old fixed threshold
                # while a previously sent servo/gripper command is in flight.
                state["q"][0] += .012
                state["grip"][0] += .05
            return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            state["q"] = q+delta
            sent.append(state["q"].copy())
        def stop(self): pass
        def close(self): pass

    class FakeGrippers:
        def __init__(self, *args, **kwargs): grip_settings.append(kwargs)
        def send_if_changed(self, targets):
            changed = np.abs(state["grip"]-targets) > 1e-6
            state["grip"] = np.asarray(targets).copy()
            grip_sent.append(state["grip"].copy())
            return changed.tolist()
        def close(self): pass

    class FakeSampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip): return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    monkeypatch.setattr(replay, "DualCommander", FakeCommander)
    monkeypatch.setattr(replay, "DualGripperCommander", FakeGrippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", FakeSampler)
    monkeypatch.setattr(replay, "_motion_safe", lambda *args: True)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--execute", "--execute-ack", replay.EXECUTE_ACK, *speed_args, *restore_args])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"]
    assert report["runtime_timings"]["joint_command"]["calls"] == len(sent)
    assert report["runtime_timings"]["robot_feedback"]["calls"] > 0
    assert report["watchdog_stops"] == 0
    assert grip_settings == [{"speed": expected_speed, "force": 25}]
    assert report["gripper_speed"] == expected_speed and report["gripper_force"] == 25
    assert reads > 3
    assert report["actions_executed"] == len(policy.actions)
    np.testing.assert_allclose(sent[-1], policy.metadata["final_joint_state"], atol=1e-7)
    np.testing.assert_allclose(grip_sent[-1], [0., 1.])
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "restore"
    restore = [row for row in rows if row["phase"] == "restore"]
    assert len(restore) > 1
    assert report["joint_speed_limit_rad_s"] is None
    assert report["restore_joint_speed_limit_rad_s"] == restore_speed
    assert report["restore_gripper_speed_limit_fraction_s"] == restore_gripper_speed
    fractions = np.array([replay.smoothstep((index+1)/len(restore)) for index in range(len(restore))])
    np.testing.assert_allclose([row["target_qpos"] for row in restore],
                               -.12+fractions[:, None]*np.full((1, 12), .12), atol=1e-12)
    np.testing.assert_allclose([row["target_gripper"] for row in restore],
                               fractions[:, None]*np.asarray(policy.metadata["initial_state"][12:]))
    assert max(np.max(np.abs(row["joint_delta"])) for row in restore) <= restore_speed*policy.period+1e-12
    gripper_steps = np.diff(np.vstack((np.zeros(2), [row["target_gripper"] for row in restore])), axis=0)
    assert np.max(np.abs(gripper_steps)) <= restore_gripper_speed*policy.period+1e-12
    assert 0 < restore[0]["target_gripper"][1] < restore[-1]["target_gripper"][1]
    assert -.12 < restore[0]["target_qpos"][0] < restore[-1]["target_qpos"][0]
    assert rows[-len(policy.actions)]["phase"] == "path"
    np.testing.assert_allclose(rows[-1]["action"], policy.actions[-1])


def test_live_physical_restoration_interpolates_before_cbf_dataset_actions(tmp_path, monkeypatch):
    import obstacle_validation.core as core

    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    state = {"q": np.zeros(12), "grip": np.array([0., .7])}
    state["q"][0] = -.2
    state["q"][1] = .06
    sent = []
    box = {"center": [-.1, 0., 0.], "axes": np.eye(3).tolist(),
           "half_sizes": [.02, .03, .1]}

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            state["q"] = np.asarray(q)+np.asarray(delta)
            sent.append(state["q"].copy())
        def stop(self): pass
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            changed = np.abs(state["grip"]-target) > 1e-6
            state["grip"] = np.asarray(target).copy()
            return changed.tolist()
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[q[0], q[1], 0.]], [[3., 3., 3.]]])

    def snapshot(*args, **kwargs):
        return {"coordinate_frame": "left_base", "obbs": [box],
                "qpos": {"left": state["q"][:6].tolist(), "right": state["q"][6:].tolist()},
                "gripper_position": {"left": float(state["grip"][0]),
                                     "right": float(state["grip"][1])}}

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(core, "rrt_detour", lambda *a, **k: pytest.fail("Replay must never call RRT"))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--avoidance-controller", "cbf", "--live", "--execute",
                                       "--execute-ack", replay.EXECUTE_ACK,
                                       "--restore-timeout", "5",
                                       "--output", str(tmp_path/"run")])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"]
    assert report["restore_replans"] == 0 and report["restore_controller"] == "interpolation"
    assert report["gripper_speed"] == 255
    assert report["actions_executed"] == 3
    np.testing.assert_allclose(state["grip"], [0., 1.])
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    assert all(row.get("state") != "detour" for row in rows if row["phase"] == "restore")
    assert all(row['cbf'] is None and row['collision_checks'] is False for row in rows if row['phase'] == 'restore')
    assert report["restore_cbf_interventions"] == 0 and report["restore_collision_checks"] is False
    path_rows = [row for row in rows if row["phase"] == "path"]
    assert path_rows[0]["cbf"]["preview_horizon"] == 3
    assert path_rows[-1]["cbf"]["preview_horizon"] == 1
    assert report["cbf_chunk_mode"] == "joint_execution_prefix"
    assert report["chunk_plans"] == 1 and report["chunk_replans"] == 0
    assert sorted({row["chunk_step"] for row in path_rows}) == [0, 1, 2]
    assert all(row["phase"] == "restore" for row in rows[:next(
        i for i, row in enumerate(rows) if row["phase"] == "path")])
    assert sent


def test_live_replay_uses_existing_obbs_during_cluster_merge_without_pausing(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    state = {"q": np.asarray(initial[:12]).copy(), "grip": np.asarray(initial[12:]).copy()}
    events = []
    snapshot_reads = 0

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            events.append("joint_command")
            state["q"] = np.asarray(q)+np.asarray(delta)
        def stop(self): events.append("stop")
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            events.append("gripper_command")
            changed = np.abs(state["grip"]-target) > 1e-6
            state["grip"] = np.asarray(target).copy()
            return changed.tolist()
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    def changing_snapshot(*args, **kwargs):
        nonlocal snapshot_reads
        snapshot_reads += 1
        status = "ok"
        boxes = []
        if snapshot_reads in (4, 5):
            events.append("merged_scene_read")
            status = "error: unstable obstacle point clusters (merged_clusters, missing_cluster)"
            boxes = [{"center": [8., 8., 8.], "axes": np.eye(3).tolist(),
                      "half_sizes": [.1, .1, .1]}]
        return {"coordinate_frame": "left_base", "status": status, "obbs": boxes,
                "qpos": {"left": state["q"][:6].tolist(), "right": state["q"][6:].tolist()},
                "gripper_position": {"left": float(state["grip"][0]),
                                     "right": float(state["grip"][1])}}

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", changing_snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--avoidance-controller", "cbf", "--live", "--execute",
                                       "--execute-ack", replay.EXECUTE_ACK,
                                       "--output", str(tmp_path/"run")])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"]
    assert report["unstable_scene_reads"] == 2
    assert events.count("merged_scene_read") == 2
    assert "joint_command" in events[events.index("merged_scene_read"):]


def test_manual_live_replay_stops_and_replans_after_expired_presend_scene(tmp_path, monkeypatch):
    from safety_module.runtime_contract import SnapshotExpired

    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    state = {"q": np.asarray(initial[:12]).copy(), "grip": np.asarray(initial[12:]).copy()}
    state["q"][0] -= .03
    events, reads = [], [0]

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            events.append("joint_command")
            state["q"] = np.asarray(q) + np.asarray(delta)
        def stop(self): events.append("stop")
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            events.append("gripper_hold")
            changed = np.abs(state["grip"]-target) > 1e-6
            state["grip"] = np.asarray(target).copy()
            return changed.tolist()
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    class ManualFilter:
        def __init__(self, path): self.path = path

    def changing_snapshot(*args, **kwargs):
        reads[0] += 1
        if reads[0] == 3:
            events.append("expired_scene")
            raise SnapshotExpired("expired test frame")
        if kwargs.get("max_wait_s") == 3.:
            events.append("fresh_scene_wait")
        return {"coordinate_frame": "left_base", "status": "ok", "obbs": [],
                "qpos": {"left": state["q"][:6].tolist(), "right": state["q"][6:].tolist()},
                "gripper_position": {"left": float(state["grip"][0]),
                                     "right": float(state["grip"][1])}}

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "ManualVoxelFilter", ManualFilter)
    monkeypatch.setattr(replay, "read_snapshot", changing_snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--avoidance-controller", "cbf", "--live", "--execute",
                                       "--manual-selection", str(tmp_path/"selection.json"),
                                       "--obstacle-source", "external",
                                       "--execute-ack", replay.EXECUTE_ACK,
                                       "--obstacle-age-policy", "strict", "--output", str(tmp_path/"run")])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"] and report["stale_scene_pauses"] == 1
    assert report["runtime_timings"]["fresh_frame_wait"]["calls"] == 1
    expired = events.index("expired_scene")
    assert expired < events.index("stop", expired) < events.index("fresh_scene_wait")
    assert events.index("fresh_scene_wait") < events.index("joint_command", expired)


@pytest.mark.parametrize("mode", ["baseline", "avoid"])
def test_restoration_skips_geometry_and_only_avoid_checks_path(tmp_path, monkeypatch, mode):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = np.asarray(replay.DatasetActionPolicy(dataset).metadata['initial_state'])
    state = {"q": initial[:12].copy(), "grip": np.zeros(2), "path": False, "sends": [], "cbf": False}
    state['q'][0] = -.2

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state['q'].copy(), state['grip'].copy()
        def send(self, q, delta):
            assert not state['path'], 'Blocked dataset command was sent'
            assert np.max(np.abs(delta)) <= .025+1e-12
            state['sends'].append(delta.copy())
            state['q'] = q+delta
        def stop(self): pass
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            state['grip'] = np.asarray(target).copy()
            return [True, True]
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {'left': True, 'right': True}
        calibration_digest = 'test'
        left_link_names, points_per_link = ('left',), 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            assert state['path'], 'Restoration invoked FK/collision checks'
            return np.array([[[q[0], 0., 0.]], [[3., 3., 3.]]])

    def snapshot(*args, **kwargs):
        assert state['sends'], 'Live perception was read before restoration'
        np.testing.assert_allclose(state['q'], initial[:12], atol=1e-7)
        np.testing.assert_allclose(state['grip'], initial[12:], atol=1e-7)
        state['path'] = True
        return {'coordinate_frame': 'left_base', 'obbs': [
            {'center': [0., 0., 0.], 'axes': np.eye(3).tolist(), 'half_sizes': [.03, .03, .1]}]}

    def path_cbf(self, q, grip, actions, checker, **kwargs):
        assert state['path'] and not checker.state_safe(q)
        state['cbf'] = True
        raise RuntimeError('path CBF gate active')

    monkeypatch.setattr(replay, 'DualCommander', Commander)
    monkeypatch.setattr(replay, 'DualGripperCommander', Grippers)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'read_snapshot', snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *args: None)
    monkeypatch.setattr(replay.RecordedPathCBF, 'project_chunk', path_cbf)
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    args = replay.parser().parse_args(['replay', '--dataset', str(dataset), '--mode', mode, '--live',
                                       '--avoidance-controller', 'cbf',
                                       '--execute', '--execute-ack', replay.EXECUTE_ACK,
                                       '--output', str(tmp_path/'run')])
    if mode == 'avoid':
        with pytest.raises(RuntimeError, match='path CBF gate active'):
            run_with_selected_scene(args, monkeypatch)
    else:
        assert run_with_selected_scene(args, monkeypatch)['completed']
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert report['restore_controller'] == 'interpolation' and not report['restore_collision_checks']
    assert report['restore_cbf_interventions'] == 0
    assert report['actions_executed'] == (0 if mode == 'avoid' else 3)
    assert state['path'] == state['cbf'] == (mode == 'avoid')
    rows = [json.loads(line) for line in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    assert rows and all(row['cbf'] is None for row in rows)
    if mode == 'avoid':
        assert all(row['phase'] == 'restore' for row in rows)


def test_restoration_still_requires_measured_arrival_and_times_out(tmp_path, monkeypatch):
    source = tmp_path/'episode.hdf5'
    episode(source)
    dataset = tmp_path/'dataset'
    replay.convert_hdf5(source, dataset)
    initial = np.asarray(replay.DatasetActionPolicy(dataset).metadata['initial_state'])
    q = initial[:12].copy(); q[0] -= .05
    closed = []
    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return q.copy(), initial[12:].copy()
        def send(self, current, delta): pass  # Feedback deliberately never reaches the target.
        def close(self): closed.append(True)
    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target): return [False, False]
        def close(self): pass
    class Sampler:
        tcp_transform_verified = {'left': True, 'right': True}
        calibration_digest = 'test'
        left_link_names, points_per_link = ('left',), 1
        def __init__(self, **kwargs): pass
        def link_points(self, *args): pytest.fail('Restoration invoked FK')
    monkeypatch.setattr(replay, 'DualCommander', Commander)
    monkeypatch.setattr(replay, 'DualGripperCommander', Grippers)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'read_snapshot', lambda *a, **k: pytest.fail('Restoration read perception'))
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    args = replay.parser().parse_args(['replay', '--dataset', str(dataset), '--mode', 'avoid', '--avoidance-controller', 'cbf', '--live',
                                       '--execute', '--execute-ack', replay.EXECUTE_ACK,
                                       '--restore-settle-timeout', '0.000000001', '--output', str(tmp_path/'run')])
    with pytest.raises(RuntimeError, match='did not settle'):
        run_with_selected_scene(args, monkeypatch)
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert report['failure_context']['reason'] == 'restore_settle_timeout'
    assert report['actions_executed'] == 0 and closed


@pytest.mark.parametrize("phase", ["restore", "path"])
def test_cbf_continues_without_refresh_after_free_evidence_ages(tmp_path, monkeypatch, phase):
    import obstacle_validation.core as core
    from safety_module.occupancy import ObservedBoxes

    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    state = {"q": np.asarray(initial[:12]).copy(), "grip": np.asarray(initial[12:]).copy(), "frame": 100}
    if phase == "restore":
        state["q"][0] -= .03
    events = []

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            events.append("send" if state.get("perception_started") else "restore_send")
            state["q"] = q+delta
        def stop(self): events.append("stop")
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            state["grip"] = np.asarray(target).copy()
            return [False, False]
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names, points_per_link = ("left",), 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.array([[[q[0], 0., 0.]], [[3., 3., 3.]]])

    # Use the actual map consumer: old free evidence no longer blocks a sweep.
    from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid
    grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    grid.update(timestamp_ns=replay.time.monotonic_ns()-3_000_000_000,
                raw_hits=[], free_ids=np.arange(grid.size))
    observation = ObservationGrid(grid.snapshot())

    def snapshot(*args, **kwargs):
        state['perception_started'] = True
        if kwargs.get("after_timestamp_ns") is not None:
            assert kwargs["after_timestamp_ns"] == 100 and kwargs["max_wait_s"] == 3.
            events.append("new_frame")
            state["frame"] = 101
        return {"coordinate_frame": "left_base", "status": "ok", "host_timestamp_ns": state["frame"],
                "obbs": []}

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "boxes_from_json", lambda value: ObservedBoxes([], observation))
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *a: None)
    monkeypatch.setattr(core, "rrt_detour", lambda *a, **k: pytest.fail("Replay called RRT"))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--mode", "avoid", "--avoidance-controller", "cbf", "--live",
                                       "--execute", "--execute-ack", replay.EXECUTE_ACK,
                                       "--obstacle-age-policy", "strict", "--output", str(tmp_path/"run")])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"] and report["stale_scene_pauses"] == 0
    assert report["restore_replans"] == 0
    assert "new_frame" not in events and "send" in events


def test_replay_waits_for_joints_but_not_unreachable_gripper_target(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    state = {"q": np.asarray(initial[:12]).copy(), "grip": np.asarray(initial[12:]).copy()}

    class SlowCommander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta): state["q"] = q+.2*delta
        def stop(self): pass
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, targets):
            target = np.asarray(targets)
            state["grip"] += np.clip(target-state["grip"], -.01, .01)
            state["grip"][1] = min(state["grip"][1], .894)
            return [True, True]
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        def __init__(self, **kwargs): pass

    monkeypatch.setattr(replay, "DualCommander", SlowCommander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "_motion_safe", lambda *args: True)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--execute", "--execute-ack", replay.EXECUTE_ACK,
                                       "--arrival-tolerance", ".005"])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"]
    assert report["actions_executed"] == 3
    assert report["path_control_ticks"] > 3
    assert report["hold_ticks"] == report["path_control_ticks"]-3
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    path = [row for row in rows if row["phase"] == "path"]
    assert [row["step"] for row in path] == sorted(row["step"] for row in path)
    assert [row["step"] for row in path if row["action_advanced"]] == [0, 1, 2]
    assert any(row["action_advanced"] and row["gripper_error"] > args.gripper_tolerance
               for row in path)
    assert all(not row["action_advanced"] or row["target_error_rad"] <= args.arrival_tolerance
               for row in path)


def test_avoid_retries_joint_target_without_baseline_action_timeout(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)

    class Sampler:
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "_motion_safe", lambda *args: True)
    common = ["replay", "--no-execute", "--dataset", str(dataset), "--speed", ".05",
              "--arrival-tolerance", ".001", "--action-timeout", "0.00000001"]

    baseline_args = replay.parser().parse_args(
        [*common, "--mode", "baseline", "--output", str(tmp_path/"baseline")])
    with pytest.raises(RuntimeError, match="joint target within"):
        run_with_selected_scene(baseline_args, monkeypatch)
    assert json.loads((tmp_path/"baseline/report.json").read_text())["failure_context"][
        "reason"] == "dataset_action_arrival_timeout"

    avoid_args = replay.parser().parse_args(
        [*common, "--mode", "avoid", "--avoidance-controller", "cbf", "--action-horizon", "2", "--output", str(tmp_path/"avoid")])
    avoid_args.action_timeout = 0.  # This baseline-only limit is ignored by CBF replay.
    report = run_with_selected_scene(avoid_args, monkeypatch)
    assert report["completed"]
    assert report["actions_executed"] == 3
    assert report["hold_ticks"] > 0
    assert report["chunk_plans"] == 2
    chunks = [json.loads(line) for line in (tmp_path/"avoid/chunks.jsonl").read_text().splitlines()]
    assert [(chunk["start_action_index"], chunk["end_action_index_exclusive"]) for chunk in chunks] == [(0, 2), (2, 3)]
    assert sum(len(chunk["optimized_actions"]) for chunk in chunks) == 3
    rows = [json.loads(line) for line in (tmp_path/"avoid/steps.jsonl").read_text().splitlines()]
    assert all(row["cbf"]["success"] for row in rows if row["phase"] == "path")


@pytest.mark.parametrize("speed", [40, 255])
def test_gripper_writer_never_resets_activation(monkeypatch, speed):
    commands = []

    class Socket:
        def settimeout(self, *_): pass
        def sendall(self, data):
            commands.append(data.decode().strip())
        def recv(self, *_):
            if commands[-1] != "GET STA":
                raise TimeoutError("URCap does not acknowledge SET commands")
            return b"STA 3\n"
        def close(self): pass

    monkeypatch.setattr(replay.socket, "create_connection", lambda *args, **kwargs: Socket())
    writer = replay.RobotiqTargetWriter("fake", speed=speed, force=25)
    assert writer.send_if_changed(.5)
    assert not writer.send_if_changed(.5)
    writer.close()
    assert commands == ["GET STA", f"SET POS 128\nSET SPE {speed}\nSET FOR 25\nSET GTO 1"]
    assert not any("ACT" in command for command in commands)


def test_replay_rejects_stale_conversion_parameters(tmp_path):
    source = tmp_path/"episode.hdf5"
    episode(source)
    output = tmp_path/"run"
    replay.convert_hdf5(source, output/"lerobot_dataset", fps=20)
    args = replay.parser().parse_args(["replay", "--no-execute", "--hdf5", str(source), "--output", str(output), "--fps", "10"])
    with pytest.raises(ValueError, match="does not match"):
        replay.run_replay(args)


def test_baseline_sends_recorded_targets_without_perception_fk_or_constraints(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    q_actual = np.asarray(initial[:12]).copy()
    sends = []
    reads = 0

    class FakeCommander:
        def __init__(self, *args): pass
        def read_state(self): return q_actual.copy(), np.asarray(initial[12:])
        def send(self, q, delta):
            sends.append(np.asarray(delta).copy())
            q_actual[:] = q+delta
        def stop(self): pass
        def close(self): pass

    class FakeGrippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, targets): return [False, False]
        def close(self): pass

    class FakeSampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pytest.fail("baseline initialized geometry")
        def link_points(self, q, grip): return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    def snapshot(*args, **kwargs):
        nonlocal reads
        reads += 1
        pytest.fail("baseline read perception")

    monkeypatch.setattr(replay, "DualCommander", FakeCommander)
    monkeypatch.setattr(replay, "DualGripperCommander", FakeGrippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", FakeSampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay, "_motion_safe", lambda *a, **k: pytest.fail("baseline checked collision"))
    monkeypatch.setattr(replay, "RecordedPathCBF", lambda *a, **k: pytest.fail("baseline initialized CBF"))
    monkeypatch.setattr(replay, "ManualVoxelFilter", lambda *a: pytest.fail("baseline loaded selection"))
    monkeypatch.setattr(replay.ManualAnnotationFilter, "__init__",
                        lambda *a, **kw: pytest.fail("baseline loaded annotations"))
    monkeypatch.setattr(replay, "latest_manual_annotations", lambda: pytest.fail("baseline searched for annotations"))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--mode", "baseline", "--live",
                                       "--max-tracking-error", ".000001"])
    result = replay.run_replay(args)
    assert result["execution"] == "physical"
    assert result['completed'] and reads == 0
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["actions_executed"] == 3
    assert report["failure_context"] is None
    assert report["constraint_checks"] is report["action_optimization"] is False
    assert report["planning_fk_cache"] is report["tracking_error_limit_rad"] is None
    assert report["snapshot_timings"] == {}
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    path = [row for row in rows if row["phase"] == "path"]
    for row in path:
        assert row['cbf'] is None
        assert row['action'] == row['optimized_action'] == row['control_target']
    assert report['max_tracking_error_rad'] > args.max_tracking_error


def test_optimized_chunk_has_no_execution_collision_recheck(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    state = {"q": np.asarray(initial[:12]), "grip": np.asarray(initial[12:]),
             "planned": False, "blocked_once": False}
    events = []

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            events.append("send" if state["planned"] else "restore_send")
            state["q"] = q+delta
        def stop(self): events.append("stop")
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            state["grip"] = np.asarray(target).copy()
            return [True, True]
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    original_project = replay.RecordedPathCBF.project_chunk

    def project(self, *args, **kwargs):
        result = original_project(self, *args, **kwargs)
        events.append("chunk_planned")
        state["planned"] = True
        return result

    def snapshot(*args, **kwargs):
        boxes = []
        if state["planned"] and not state["blocked_once"]:
            events.append("new_obstacle")
            state["blocked_once"] = True
            boxes = [{"center": [0., 0., 0.], "axes": np.eye(3).tolist(), "half_sizes": [.1]*3}]
        return {"coordinate_frame": "left_base", "status": "ok", "obbs": boxes}

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.RecordedPathCBF, "project_chunk", project)
    def forbidden_sweep(*args, **kwargs):
        raise AssertionError('Avoid replay must not check sweeps during planning or execution')
    monkeypatch.setattr(replay, "_motion_safe", forbidden_sweep)
    monkeypatch.setattr(replay.CollisionChecker, 'edge_safe', forbidden_sweep)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--mode", "avoid", "--avoidance-controller", "cbf",
                                       "--live", "--execute", "--execute-ack", replay.EXECUTE_ACK,
                                       "--output", str(tmp_path/"run")])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"] and report["actions_executed"] == 3
    assert events.index("chunk_planned") < events.index("send") < events.index("new_obstacle")
    assert "stop" not in events[events.index("chunk_planned")+1:events.index("send")]
    assert report["chunk_replans"] == 0
    assert report["post_optimization_collision_checks"] is False
    assert report["optimization_sweep_checks"] is False
    chunks = [json.loads(line) for line in (tmp_path/'run/chunks.jsonl').read_text().splitlines()]
    assert all(chunk['sweep_validation_enabled'] is False for chunk in chunks)


def test_dynamic_rollout_uses_exact_euclidean_box_margin_at_corners():
    class Sampler:
        left_link_names = ("left",)
        points_per_link = 1
        def link_points(self, q, grip):
            return np.asarray([[[q[0], q[1], 0.]], [[3., 3., 3.]]])

    sampler = Sampler()
    box = OrientedBox(np.zeros(3), np.eye(3), np.asarray([.01, .01, .1]))
    safety = {"collision_margin_m": .05, "inter_arm_margin_m": .035}
    q = np.zeros(12); q[:2] = -.05
    away = np.zeros(12); away[:2] = -.03
    assert not validate_fk_rollout(sampler, q, np.zeros(2), np.asarray([away]), [box],
                                   point_indices=np.arange(2), left_mask=np.asarray([True, False]),
                                   collision_margin_m=.05, inter_arm_margin_m=.035)
    assert replay._motion_safe(sampler, q, np.zeros(2), away, np.zeros(2), [box], safety)
    toward = np.zeros(12); toward[:2] = .05
    assert not replay._motion_safe(sampler, q, np.zeros(2), toward, np.zeros(2), [box], safety)


@pytest.mark.parametrize("outcome", ["resume", "interrupt"])
def test_cbf_failure_retries_same_action_until_success_or_user_interrupt(tmp_path, monkeypatch, outcome):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)

    class Sampler:
        left_link_names = ("left",)
        points_per_link = 1
        calibration_digest = "test"
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    original_project = replay.RecordedPathCBF.project_chunk
    calls = []
    def project(self, q, grip, actions, checker, **kwargs):
        nominal = np.clip(actions[0, :12]-q, -kwargs['max_step'], kwargs['max_step'])
        calls.append((q.copy(), nominal.copy()))
        if len(calls) <= 4:
            return None, {"success": False, "failed_action_index": 0,
                          "failed_qpos": q.tolist(), "failed_gripper": grip.tolist(),
                          "failed_joint_lower": np.asarray(kwargs['lower'])[0].tolist(),
                          "failed_joint_upper": np.asarray(kwargs['upper'])[0].tolist(),
                          "failed_nominal_delta": nominal.tolist(),
                          "reason": "joint_chunk_optimization_failed",
                          "failed_step": {"triggered": True, "success": False,
                                          "reason": "joint_chunk_optimization_failed", "max_violation": .055}}
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return original_project(self, q, grip, actions, checker, **kwargs)
    monkeypatch.setattr(replay.RecordedPathCBF, "project_chunk", project)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--no-execute", "--dataset", str(dataset),
                                       "--mode", "avoid", "--avoidance-controller", "cbf", "--output", str(tmp_path/"run")])
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            run_with_selected_scene(args, monkeypatch)
    else:
        run_with_selected_scene(args, monkeypatch)
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["optimization_retries"] == 4
    assert report["completed"] == (outcome == "resume")
    assert report["failure_context"] is None
    context = report["last_cbf_failure"]
    assert context["reason"] == "cbf_projection_failed"
    assert context["action_index"] == 0
    assert context["cbf"]["max_violation"] == .055
    assert len(context["dataset_action"]) == 14
    assert len(context["nominal_delta"]) == 12
    assert context["obbs"] == []
    for pose, nominal in calls[1:5]:
        np.testing.assert_allclose(pose, calls[0][0])
        np.testing.assert_allclose(nominal, calls[0][1])
    retries = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()
               if json.loads(line)["phase"] == "optimization_retry"]
    assert len(retries) == 4 and all(row["step"] == 0 and not row["action_advanced"] for row in retries)


@pytest.mark.parametrize("box_x", [0.025, 0.02])
def test_baseline_preserves_recorded_gripper_closure_without_obstacle_checks(tmp_path, monkeypatch, box_x):
    source = tmp_path/"episode.hdf5"
    episode(source)
    with h5py.File(source, "r+") as f:
        f["observations/gripper_position"][:, 1] = 0.
        f["gripper_action"][:, 1] = 1.
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)

    class Sampler:
        left_link_names = ("left",)
        points_per_link = 1
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[q[6]+.04*grip[1], 0., 0.]]])

    box = {"center": [box_x, 0., 0.], "axes": np.eye(3).tolist(), "half_sizes": [.003, .1, .1]}
    snapshot = {"coordinate_frame": "left_base", "obbs": [box],
                "qpos": {"left": [0.]*6, "right": [0.]*6},
                "gripper_position": {"left": .2, "right": 0.}}
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--no-execute", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--mode", "baseline", "--live"])
    assert replay.run_replay(args)['completed']
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["failure_context"] is None
    assert report["completed"] and not report["constraint_checks"]
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    path = [row for row in rows if row["phase"] == "path"]
    assert len(path) == 3
    assert all(row["gripper_target"][1] == 1. and row["cbf"] is None for row in path)


def test_avoid_corrects_joints_and_keeps_dataset_gripper_target(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    with h5py.File(source, "r+") as f:
        f["observations/gripper_position"][:, 1] = 0.
        f["gripper_action"][:, 1] = 1.
        f["observations/qpos"][:, 6] = 0.
        f["action"][:, 6] = 0.
        f["executed_action"][:, 6] = 0.
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)

    class Sampler:
        left_link_names = ("left",)
        points_per_link = 1
        tcp_transform_verified = {"left": True, "right": True}
        calibration_digest = "test"

        def __init__(self, **kwargs): pass

        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[q[6]+.03*grip[1], 0., 0.]]])

    box = {"center": [.03, 0., 0.], "axes": np.eye(3).tolist(),
           "half_sizes": [.003, .1, .1]}
    snapshot = {"coordinate_frame": "left_base", "obbs": [box],
                "qpos": {"left": [0.]*6, "right": [0.]*6},
                "gripper_position": {"left": .2, "right": 0.}}
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--no-execute", "--dataset", str(dataset),
                                       "--output", str(tmp_path/"run"),
                                       "--mode", "avoid", "--avoidance-controller", "cbf", "--live", "--final-tolerance", ".001"])
    report = run_with_selected_scene(args, monkeypatch)
    assert report["completed"]
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    first = next(row for row in rows if row["phase"] == "path")
    assert first["cbf"]["triggered"] and first["cbf"]["success"]
    assert first["sent_joint_delta"][6] < first["action"][6]
    assert first["gripper_target"][1] == 1.
    assert first["gripper_error"] == 0.
    assert report["chunk_plans"] == 1 and report["chunk_replans"] == 0
    assert report["final_joint_error_rad"] > args.final_tolerance
    assert report["final_optimized_joint_error_rad"] <= args.final_tolerance
    advanced = [row for row in rows if row["phase"] == "path" and row["action_advanced"]]
    assert [row["chunk_step"] for row in advanced] == [0, 1, 2]
    assert all(np.max(np.abs(np.array(row["qpos"])-row["optimized_action"][:12])) <= args.arrival_tolerance
               for row in advanced)
    chunks = [json.loads(line) for line in (tmp_path/"run/chunks.jsonl").read_text().splitlines()]
    assert len(chunks) == 1 and chunks[0]["modified_action_indices"] == [0, 1, 2]
    np.testing.assert_allclose([row["optimized_action"] for row in advanced], chunks[0]["optimized_actions"])
    for row in advanced:
        assert row['gripper_target'][1] == 1.
        assert row['cbf']['joint_optimization']
