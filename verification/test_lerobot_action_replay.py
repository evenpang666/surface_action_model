"""Dataset action semantics and output sequencing without robot connections."""

import json

import h5py
import numpy as np
import pytest

from obstacle_validation.quest3_schema import NAMES
from obstacle_validation import lerobot_replay as replay
from real_scripts.real_cbf_qp import OrientedBox, validate_fk_rollout


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


def test_physical_replay_restores_then_sends_dataset_actions_and_grippers(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    policy = replay.DatasetActionPolicy(dataset)
    state = {"q": np.full(12, -.01), "grip": np.array([0., 0.])}
    sent = []
    grip_sent = []
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
        def __init__(self, *args, **kwargs): pass
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
                                       "--execute", "--execute-ack", replay.EXECUTE_ACK])
    report = replay.run_replay(args)
    assert report["completed"]
    assert reads > 3
    assert report["actions_executed"] == len(policy.actions)
    np.testing.assert_allclose(sent[-1], policy.metadata["final_joint_state"], atol=1e-7)
    np.testing.assert_allclose(grip_sent[-1], [0., 1.])
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "restore"
    assert rows[-len(policy.actions)]["phase"] == "path"
    np.testing.assert_allclose(rows[-1]["action"], policy.actions[-1])


def test_live_physical_restoration_detours_before_dataset_actions(tmp_path, monkeypatch):
    import obstacle_validation.core as core

    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    state = {"q": np.zeros(12), "grip": np.array([0., .7])}
    state["q"][0] = -.2
    sent = []
    detours = []
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

    def route(start, goal, checker, lower, upper, **kwargs):
        first, second = np.asarray(start).copy(), np.asarray(goal).copy()
        first[1] = second[1] = .12
        result = [first, second, np.asarray(goal).copy()]
        assert checker.path_safe(np.asarray([start, *result]))
        detours.append(result)
        return result

    monkeypatch.setattr(replay, "DualCommander", Commander)
    monkeypatch.setattr(replay, "DualGripperCommander", Grippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(core, "rrt_detour", route)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--live", "--execute",
                                       "--execute-ack", replay.EXECUTE_ACK,
                                       "--output", str(tmp_path/"run")])
    report = replay.run_replay(args)
    assert report["completed"]
    assert report["restore_replans"] >= 1 and detours
    assert report["actions_executed"] == 3
    np.testing.assert_allclose(state["grip"], [0., 1.])
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    assert any(row.get("state") == "detour" for row in rows if row["phase"] == "restore")
    assert any(row.get("cbf", {}).get("triggered") for row in rows if row["phase"] == "restore")
    assert report["restore_cbf_interventions"] > 0
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
                                       "--mode", "avoid", "--live", "--execute",
                                       "--execute-ack", replay.EXECUTE_ACK,
                                       "--output", str(tmp_path/"run")])
    report = replay.run_replay(args)
    assert report["completed"]
    assert report["unstable_scene_reads"] == 2
    assert events.count("merged_scene_read") == 2
    assert "joint_command" in events[events.index("merged_scene_read"):]


def test_live_restoration_rejects_occupied_recorded_start_with_context(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = np.zeros(12); initial[0] = -.2

    class Sampler:
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[q[0], 0., 0.]], [[3., 3., 3.]]])

    def snapshot(*args, **kwargs):
        return {"coordinate_frame": "left_base",
                "qpos": {"left": initial[:6].tolist(), "right": initial[6:].tolist()},
                "gripper_position": {"left": .2, "right": .7},
                "obbs": [{"center": [0., 0., 0.], "axes": np.eye(3).tolist(),
                          "half_sizes": [.03, .03, .1]}]}

    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--live",
                                       "--output", str(tmp_path/"run")])
    with pytest.raises(RuntimeError, match="Recorded initial joint/gripper state is blocked"):
        replay.run_replay(args)
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["failure_context"]["reason"] == "restore_goal_unsafe"
    assert len(report["failure_context"]["obbs"]) == 1
    assert report["actions_executed"] == 0


def test_live_restoration_moves_gripper_first_when_current_shape_blocks_goal(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    with h5py.File(source, "r+") as file:
        file["gripper_action"][:] = np.tile([.2, .7], (4, 1))
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = np.zeros(12); initial[0] = -.1

    class Sampler:
        calibration_digest = "test"
        left_link_names = ("left",)
        points_per_link = 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip):
            return np.asarray([[[q[0], .1*grip[0], 0.]], [[3., 3., 3.]]])

    def snapshot(*args, **kwargs):
        return {"coordinate_frame": "left_base",
                "qpos": {"left": initial[:6].tolist(), "right": initial[6:].tolist()},
                "gripper_position": {"left": 0., "right": .7},
                "obbs": [{"center": [0., 0., 0.], "axes": np.eye(3).tolist(),
                          "half_sizes": [.005, .005, .1]}]}

    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--live",
                                       "--output", str(tmp_path/"run")])
    report = replay.run_replay(args)
    assert report["completed"]
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    restore = [row for row in rows if row["phase"] == "restore"]
    assert restore[0]["state"] == "gripper"
    assert any(row["state"] in ("nominal", "cbf") for row in restore)
    assert rows[-1]["phase"] == "path"


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
    report = replay.run_replay(args)
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
    common = ["replay", "--dataset", str(dataset), "--speed", ".05",
              "--arrival-tolerance", ".001", "--action-timeout", "0.00000001"]

    baseline_args = replay.parser().parse_args(
        [*common, "--mode", "baseline", "--output", str(tmp_path/"baseline")])
    with pytest.raises(RuntimeError, match="joint target within"):
        replay.run_replay(baseline_args)
    assert json.loads((tmp_path/"baseline/report.json").read_text())["failure_context"][
        "reason"] == "dataset_action_arrival_timeout"

    avoid_args = replay.parser().parse_args(
        [*common, "--mode", "avoid", "--output", str(tmp_path/"avoid")])
    avoid_args.action_timeout = 0.  # This baseline-only limit is ignored by CBF replay.
    report = replay.run_replay(avoid_args)
    assert report["completed"]
    assert report["actions_executed"] == 3
    assert report["hold_ticks"] > 0
    rows = [json.loads(line) for line in (tmp_path/"avoid/steps.jsonl").read_text().splitlines()]
    assert all(row["cbf"]["success"] for row in rows if row["phase"] == "path")


def test_gripper_writer_never_resets_activation(monkeypatch):
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
    writer = replay.RobotiqTargetWriter("fake", speed=40, force=25)
    assert writer.send_if_changed(.5)
    assert not writer.send_if_changed(.5)
    writer.close()
    assert commands == ["GET STA", "SET POS 128\nSET SPE 40\nSET FOR 25\nSET GTO 1"]
    assert not any("ACT" in command for command in commands)


def test_replay_rejects_stale_conversion_parameters(tmp_path):
    source = tmp_path/"episode.hdf5"
    episode(source)
    output = tmp_path/"run"
    replay.convert_hdf5(source, output/"lerobot_dataset", fps=20)
    args = replay.parser().parse_args(["replay", "--hdf5", str(source), "--output", str(output), "--fps", "10"])
    with pytest.raises(ValueError, match="does not match"):
        replay.run_replay(args)


def test_live_replay_rechecks_new_scene_before_path_send(tmp_path, monkeypatch):
    source = tmp_path/"episode.hdf5"
    episode(source)
    dataset = tmp_path/"dataset"
    replay.convert_hdf5(source, dataset)
    initial = replay.DatasetActionPolicy(dataset).metadata["initial_state"]
    sends = []
    reads = 0

    class FakeCommander:
        def __init__(self, *args): pass
        def read_state(self): return np.asarray(initial[:12]), np.asarray(initial[12:])
        def send(self, q, delta): sends.append(np.asarray(delta).copy())
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
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip): return np.asarray([[[3., 3., 3.]], [[4., 4., 4.]]])

    def snapshot(*args, **kwargs):
        nonlocal reads
        reads += 1
        box = {"center": [0., 0., 0.], "axes": np.eye(3).tolist(), "half_sizes": [.1]*3}
        return {"coordinate_frame": "left_base", "obbs": [box] if reads >= 6 else []}

    monkeypatch.setattr(replay, "DualCommander", FakeCommander)
    monkeypatch.setattr(replay, "DualGripperCommander", FakeGrippers)
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", FakeSampler)
    monkeypatch.setattr(replay, "read_snapshot", snapshot)
    monkeypatch.setattr(replay, "validate_calibration_identity", lambda *args: None)
    monkeypatch.setattr(replay, "_motion_safe", lambda sampler, q, grip, delta, target, boxes, safety,
                        allow_unsafe_start=False: not boxes)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--mode", "baseline", "--live", "--execute", "--execute-ack", replay.EXECUTE_ACK])
    with pytest.raises(RuntimeError, match="blocked"):
        replay.run_replay(args)
    assert len(sends) == 1  # restore only; path step rejected against fresh scene
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["actions_executed"] == 0
    assert report["failure_context"]["reason"] == "joint_and_gripper_sweeps_blocked"
    assert report["gripper_hold_on_error"] == "sent_current_measured_position"


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


def test_cbf_failure_report_keeps_the_failed_action_and_scene(tmp_path, monkeypatch):
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
    monkeypatch.setattr(replay.RecordedPathCBF, "project",
                        lambda self, q, nominal, checker, **kwargs:
                        (np.zeros(12), {"triggered": True, "success": False,
                                        "max_violation": .055}))
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--mode", "avoid", "--output", str(tmp_path/"run")])
    with pytest.raises(RuntimeError, match="CBF could not project"):
        replay.run_replay(args)
    context = json.loads((tmp_path/"run/report.json").read_text())["failure_context"]
    assert context["reason"] == "cbf_projection_failed"
    assert context["action_index"] == 0
    assert context["cbf"]["max_violation"] == .055
    assert len(context["dataset_action"]) == 14
    assert len(context["nominal_delta"]) == 12
    assert context["obbs"] == []


@pytest.mark.parametrize("box_x", [0.025, 0.02])
def test_blocked_gripper_closure_does_not_advance_joints_alone(tmp_path, monkeypatch, box_x):
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
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset), "--output", str(tmp_path/"run"),
                                       "--mode", "baseline", "--live"])
    with pytest.raises(RuntimeError, match="Dataset joint/gripper action is blocked"):
        replay.run_replay(args)
    report = json.loads((tmp_path/"run/report.json").read_text())
    assert report["failure_context"]["reason"] == "gripper_sweep_blocked"
    assert report["failure_context"]["dataset_gripper_target"][1] == 1.
    assert not report["completed"]
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    path = [row for row in rows if row["phase"] == "path"]
    assert path == []


def test_avoid_corrects_joints_and_keeps_dataset_gripper_target(tmp_path, monkeypatch):
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
    args = replay.parser().parse_args(["replay", "--dataset", str(dataset),
                                       "--output", str(tmp_path/"run"),
                                       "--mode", "avoid", "--live"])
    report = replay.run_replay(args)
    assert report["completed"]
    rows = [json.loads(line) for line in (tmp_path/"run/steps.jsonl").read_text().splitlines()]
    first = next(row for row in rows if row["phase"] == "path")
    assert first["cbf"]["triggered"] and first["cbf"]["success"]
    assert first["sent_joint_delta"][6] < first["action"][6]
    assert first["gripper_target"][1] == 1.
    assert first["gripper_error"] == 0.
