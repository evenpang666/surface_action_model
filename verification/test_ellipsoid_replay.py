"""Volume geometry, coupled control and socket execution contracts."""
import json
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_validation.action_stream import (
    PROTOCOL, ProtocolError, receive_message, receive_session, send_message, send_session,
)
from obstacle_validation.ellipsoid_geometry import (
    Ellipsoid, RobotEllipsoids, fit_enclosing, fit_obstacles, plane_gap, separation, separation_gradient,
)
from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
from real_scripts.real_cbf_qp import OrientedBox
from verification.test_obstacle_validation import TwoPointSampler


def sphere(center, radius, key="", arm=0, link=0):
    return Ellipsoid(np.asarray(center, float), np.eye(3)*radius**2, key, arm, link)


@pytest.mark.parametrize("x,expected", [(3., 1.), (1.5, -.5), (0., -2.)])
def test_signed_separation_including_overlap(x, expected):
    a, b = sphere([x, 0, 0], 1), sphere([0, 0, 0], 1)
    assert separation(a, b)[0] == pytest.approx(expected, abs=1e-7)


def test_rotating_ellipsoid_gradient_includes_orientation():
    obstacle = sphere([.45, .3, 0], .06)
    def body(q):
        r = Rotation.from_rotvec([0, 0, q]).as_matrix()
        return Ellipsoid(np.zeros(3), r @ np.diag([.3**2, .04**2, .05**2]) @ r.T)
    q, eps = .3, 1e-5
    a = body(q)
    d, n, *_ = separation(a, obstacle)
    ds = ((body(q+eps).shape-body(q-eps).shape)/(2*eps))[..., None]
    gradient = separation_gradient(a, n, np.zeros((3, 1)), ds)[0]
    finite = (separation(body(q+eps), obstacle)[0]-separation(body(q-eps), obstacle)[0])/(2*eps)
    assert abs(gradient) > .02
    assert gradient == pytest.approx(finite, abs=1e-5)


def test_fit_covers_full_voxel_corners_and_retains_separate_obstacles():
    boxes = [OrientedBox(np.array([x, 0., 0.]), np.eye(3), np.array([.02, .03, .04]))
             for x in (-.2, .2)]
    fitted = fit_obstacles(boxes)
    assert len(fitted) == 2
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for box, model in zip(boxes, fitted):
        points = box.center+signs*box.half_sizes
        local = points-model.center
        assert np.max(np.einsum("ni,ij,nj->n", local, np.linalg.inv(model.shape), local)) <= 1+1e-6


class VolumeSampler:
    left_link_names = ("left",)
    def ellipsoids(self, q, grip):
        return [sphere([q[0], q[1], 0], .01, "left", 0),
                sphere([q[6], q[7], 0], .01, "right", 1)]
    def link_points(self, *args):
        pytest.fail("Online volume control must not query sampled surface points")


def test_pair_gradient_moves_both_arms_and_catches_prediction_outside_trigger():
    sampler = VolumeSampler()
    cbf = EllipsoidMCBF(sampler, margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0], q[6] = -.041, .041
    action = np.zeros((1,14)); action[0,0], action[0,6] = -.001, .001
    _, info = cbf.plan(q, np.zeros(2), action, [], lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    target = np.asarray(info["waypoints"][0][:12])
    assert info["steps"][0]["constraint_pairs"] == 1
    assert info["steps"][0]["constraint_pair_counts"] == {
        "left_obstacle": 0, "right_obstacle": 0, "inter_arm": 1}
    assert info["steps"][0]["active_constraint_counts"]["inter_arm"] == 1
    assert all(r["pair_type"] == "inter_arm" for r in info["steps"][0]["violated_pairs"])
    assert target[0] < target[6]
    assert target[6]-target[0] >= .055-1e-4
    assert not info["steps"][0]["distance_constraints_relaxed"]


@pytest.mark.parametrize("nearby", [False, True])
def test_feasible_nominal_step_never_runs_action_optimizer(monkeypatch, nearby):
    cbf = EllipsoidMCBF(TwoPointSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0] = .2
    action = np.zeros((1,14)); action[0,0] = .8; action[0,12:] = [.2,.8]
    boxes = [OrientedBox(np.array([.3,.08,0]), np.eye(3), np.full(3,.01))] if nearby else []
    monkeypatch.setattr(cbf, "_solve", lambda *a, **kw: pytest.fail("Feasible action must not run QP"))
    monkeypatch.setattr(cbf, "_guidance", lambda *a, **kw: pytest.fail("Feasible action must not detour"))
    _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12),
                       max_step=.04, previous_delta=np.full(12,-.03))
    expected = np.r_[q+np.clip(action[0,:12]-q,-.04,.04), action[0,12:]]
    np.testing.assert_array_equal(info["waypoints"][0], expected)
    assert info["joint_qp_solve_calls"] == 0 and info["modified_action_indices"] == []
    assert info["triggered_action_indices"] == []
    detail = info["steps"][0]
    assert detail["nominal_feasible"] and not detail["optimization_applied"]
    assert (detail["constraint_pairs"] > 0) == nearby


@pytest.mark.parametrize("failed_qp", [False, True])
def test_blocked_step_preserves_forward_direction_without_unrelated_motion(monkeypatch, failed_qp):
    cbf = EllipsoidMCBF(VolumeSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[6] = .1
    action = np.zeros((1,14)); action[0,6] = .3; action[0,12:] = [.3,.8]
    action[0,0] = 1e-5
    boxes = [OrientedBox(np.array([.15,0,0]), np.eye(3), np.full(3,.02))]
    backward = np.zeros(12); backward[0] = .5; backward[6] = -1
    monkeypatch.setattr(cbf, "_guidance", lambda *a: (backward, {"cached": True}))
    if failed_qp:
        monkeypatch.setattr(cbf, "_solve", lambda *a, **kw: (None, "forced failure"))
    _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12),
                       max_step=.04, previous_delta=np.full(12,.03))
    target = np.array(info["waypoints"][0])
    np.testing.assert_allclose(target[:6], action[0,:6], atol=1e-5)
    assert target[6] >= q[6]-1e-10
    assert np.max(np.abs(target[:12]-q)) <= .040001
    np.testing.assert_array_equal(target[12:], action[0,12:])
    detail = info["steps"][0]
    assert detail["optimization_applied"] and detail["active_constraints"] > 0
    assert min(detail["path_progress_rad"]) >= -1e-10
    assert detail["geodesic"]["rejected"] == "opposes_goal_or_has_no_goal_direction"
    assert (detail["fallback"] == "bounded_penalty_optimizer") == failed_qp


def test_recorded_stationary_arm_remains_available_for_actual_constraints():
    cbf = EllipsoidMCBF(VolumeSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[6] = .1
    action = np.r_[q, [.3,.8]][None]
    # The left target is unchanged, but a new obstacle enters its margin.
    # This arm is allowed to move when its real distance constraint requires it.
    boxes = [OrientedBox(np.array([.025,0,0]), np.eye(3), np.full(3,.01))]
    _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12),
                       max_step=.04)
    target = np.array(info["waypoints"][0][:12])
    assert target[0] < q[0]-.001
    np.testing.assert_allclose(target[6:], q[6:], atol=1e-6)
    assert info["steps"][0]["optimization_applied"]
    assert info["joint_qp_solve_calls"] > 0


def test_each_arm_has_its_own_unrelaxed_forward_constraint():
    cbf = EllipsoidMCBF(VolumeSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0], q[6] = -.02,.02
    action = np.zeros((1,14)); action[0,0], action[0,6] = .1,-.1
    _, info = cbf.plan(q, np.zeros(2), action, [], lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    delta = np.array(info["waypoints"][0][:12])-q
    assert delta[0] >= -1e-10 and delta[6] <= 1e-10
    assert info["steps"][0]["distance_constraints_relaxed"]
    assert len(info["steps"][0]["path_progress_rad"]) == 2
    assert min(info["steps"][0]["path_progress_rad"]) >= -1e-10


def test_clear_reached_target_stays_still_despite_nearby_geometry_and_previous_motion(monkeypatch):
    cbf = EllipsoidMCBF(VolumeSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[6] = .3
    action = np.r_[q, [.2,.8]][None]
    boxes = [OrientedBox(np.array([.12,0,0]), np.eye(3), np.full(3,.01))]
    monkeypatch.setattr(cbf, "_solve", lambda *a, **kw: pytest.fail("Reached feasible target needs no QP"))
    _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12),
                       max_step=.04, previous_delta=np.full(12,.03))
    np.testing.assert_array_equal(info["waypoints"][0][:12], q)
    assert info["joint_qp_solve_calls"] == 0


def test_leaving_constraints_immediately_restores_nominal_action(monkeypatch):
    cbf = EllipsoidMCBF(TwoPointSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0] = .2
    action = np.zeros((1,14)); action[0,0] = .8
    boxes = [OrientedBox(np.array([.26,0,0]), np.eye(3), np.full(3,.02))]
    _, blocked = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    assert blocked["steps"][0]["optimization_applied"]
    q = np.array(blocked["waypoints"][0][:12])
    monkeypatch.setattr(cbf, "_solve", lambda *a, **kw: pytest.fail("No cached avoidance on clear scene"))
    _, clear = cbf.plan(q, np.zeros(2), action, [], lower=-np.ones(12), upper=np.ones(12),
                        max_step=.04, previous_delta=np.full(12,.02))
    np.testing.assert_allclose(clear["waypoints"][0][:12], q+np.clip(action[0,:12]-q,-.04,.04), atol=0)
    assert clear["joint_qp_solve_calls"] == 0 and clear["steps"][0]["geodesic"] is None


def test_convex_obstacle_reaching_with_unchanged_gripper_commands():
    cbf = EllipsoidMCBF(TwoPointSampler(), margin=.005, inter_arm_margin=.035, period=.05,
                       config=EllipsoidMCBFConfig(guidance_refresh_s=.001))
    q = np.zeros(12); q[0] = .2
    action = np.zeros((1,14)); action[0,0] = .8; action[0,12:] = [.3,.8]
    boxes = [OrientedBox(np.array([.5,0,0]), np.eye(3), np.array([.08,.15,.1]))]
    path = []
    for _ in range(160):
        _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12), max_step=.04)
        waypoint = np.array(info["waypoints"][0])
        assert waypoint[0] >= q[0]-1e-10
        assert min(info["steps"][0]["path_progress_rad"]) >= -1e-10
        np.testing.assert_array_equal(waypoint[12:], action[0,12:])
        assert np.max(np.abs(waypoint[:12]-q)) <= .040001
        q = waypoint[:12]; path.append(q.copy())
        if np.max(np.abs(q-action[0,:12])) < .005:
            break
    assert np.max(np.abs(q-action[0,:12])) < .005
    assert np.max(np.abs(np.asarray(path)[:,1])) > .15


def test_qp_failure_uses_bounded_penalty_without_latching_action(monkeypatch):
    import osqp
    class Failed:
        def setup(self, **kwargs): pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=None, info=SimpleNamespace(status="maximum iterations reached", status_val=7))
    monkeypatch.setattr(osqp, "OSQP", Failed)
    cbf = EllipsoidMCBF(TwoPointSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    action = np.zeros((1,14)); action[0,0] = .8
    boxes = [OrientedBox(np.array([.06,0,0]), np.eye(3), np.array([.02,.02,.02]))]
    _, info = cbf.plan(np.zeros(12), np.zeros(2), action, boxes,
                       lower=-np.ones(12), upper=np.ones(12), max_step=np.inf)
    assert np.max(np.abs(info["waypoints"][0][:12])) <= .040001
    assert info["steps"][0]["fallback"] == "bounded_penalty_optimizer"
    assert not info["avoidance_bypassed"]


def test_stagnation_trades_clearance_for_original_goal_explicitly():
    cbf = EllipsoidMCBF(TwoPointSampler(), margin=.005, inter_arm_margin=.035, period=.05,
                       config=EllipsoidMCBFConfig(stagnation_steps=5, guidance_refresh_s=.001))
    q = np.zeros(12); q[0] = .2
    action = np.zeros((1,14)); action[0,0] = .5  # Deliberately inside the object.
    boxes = [OrientedBox(np.array([.5,0,0]),np.eye(3),np.array([.08,.10,.1]))]
    maximum_level, slack_seen = 0, False
    for _ in range(240):
        _, info = cbf.plan(q,np.zeros(2),action,boxes,lower=-np.ones(12),upper=np.ones(12),max_step=.04)
        q = np.asarray(info["waypoints"][0][:12])
        detail = info["steps"][0]
        maximum_level = max(maximum_level, detail["relaxation_level"])
        slack_seen |= detail["max_distance_slack_m"] > .001
        if np.linalg.norm(q-action[0,:12]) < .005:
            break
    assert np.linalg.norm(q-action[0,:12]) < .005
    assert maximum_level >= 4 and slack_seen
    assert not info["avoidance_bypassed"]


def test_embedded_source_defaults_to_no_robot_mask(monkeypatch):
    from real_scripts import replay_voxblox, live_dual_ur7e_obstacle_model as live
    monkeypatch.setattr(replay_voxblox, "require_voxblox", lambda: None)
    monkeypatch.setattr(live, "DualObstacleScene", lambda args: SimpleNamespace(report={"sha256": "test"}))
    selection = SimpleNamespace(calibration_digest="test")
    source = replay_voxblox.ReplayVoxbloxSource("configs/ur7e_dual_quest3.yaml", selection)
    assert source.args.robot_mask is False and source.thread is None
    assert source.args.occupancy_backend == "voxblox" and not source.args.voxblox_esdf
    source.close()


def hello(total=3):
    return dict(type="hello", protocol=PROTOCOL, session="test", period_s=.05, total_actions=total,
                initial_state=np.zeros(14).tolist(), action_layout="absolute_q12_gripper2", coordinate_frame="left_base")


def test_socket_sender_waits_for_execution_ack_and_sends_tail():
    a, b = socket.socketpair()
    a.settimeout(3); b.settimeout(3)
    entered, release = threading.Event(), threading.Event()
    result, errors, received = [], [], []
    policy = SimpleNamespace(actions=np.zeros((5,14)), period=.05,
                             metadata={"initial_state":np.zeros(14).tolist()})
    policy.infer = lambda start, count: policy.actions[start:start+count]
    def execute(actions, start, progress):
        received.append((start,len(actions)))
        if start == 0:
            entered.set()
            assert release.wait(2)
        progress(start)
    def sender():
        try: result.append(send_session(a, policy, chunk_size=3))
        except BaseException as e: errors.append(e)
    def receiver():
        try: result.append(receive_session(b, execute))
        except BaseException as e: errors.append(e)
    threads = [threading.Thread(target=fn) for fn in (sender, receiver)]
    try:
        for thread in threads: thread.start()
        assert entered.wait(2)
        import select
        assert not select.select([b], [], [], .05)[0]  # no pipelined second chunk
        release.set()
        for thread in threads: thread.join(3)
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert received == [(0,3),(3,2)]
        assert all(r["completed"] for r in result)
    finally:
        release.set(); a.close(); b.close()


@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "execution_failure"])
def test_socket_never_acknowledges_invalid_or_failed_chunk(bad):
    a, b = socket.socketpair()
    a.settimeout(2); b.settimeout(2)
    errors, calls = [], []
    def execute(actions, start, progress):
        calls.append(start)
        if bad == "execution_failure": raise RuntimeError("execution failed")
    def receiver():
        try: receive_session(b, execute)
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=receiver); thread.start()
    try:
        send_message(a, hello())
        assert receive_message(a)["type"] == "ready"
        actions = np.zeros((1,14)).tolist()
        if bad == "nonfinite": actions[0][0] = "NaN"
        message = dict(type="chunk", session="test", sequence=0, start=0, actions=actions)
        send_message(a, message)
        reply = receive_message(a)
        if bad == "duplicate":
            assert reply["type"] == "executed"
            send_message(a, message)
            reply = receive_message(a)
        assert reply["type"] == "error"
        thread.join(2)
        assert errors and not thread.is_alive()
        assert len(calls) == (0 if bad == "nonfinite" else 1)
    finally:
        a.close(); b.close()


def test_stream_defaults_have_no_mask_and_no_action_timeout():
    from obstacle_validation.stream_replay import parser
    args = parser().parse_args(["--manual-selection","selection.json","--output","run"])
    assert not args.robot_mask and args.action_timeout == 0


def test_each_complete_link_mesh_has_exactly_one_enclosing_ellipsoid():
    from obstacle_validation.ellipsoid_geometry import _local_robot_models
    from real_scripts.ur7e_collision_mesh import load_collision_meshes
    models = _local_robot_models()
    assert len(models) == 7
    for name, mesh in load_collision_meshes().items():
        # Check triangle interiors too: vertex-only decomposition can miss them.
        points = np.r_[np.asarray(mesh.vertices), np.asarray(mesh.triangles).mean(1)]
        fields = []
        for _, label, model in models:
            if label == name:
                local = points-model.center
                fields.append(np.einsum("ni,ij,nj->n", local, np.linalg.inv(model.shape), local))
        assert len(fields) == 1, name
        assert np.max(np.min(fields, axis=0)) <= 1+1e-5, name


@pytest.mark.parametrize("closing", [0., .37, 1.])
def test_one_gripper_ellipsoid_covers_all_components_at_actual_opening(closing):
    from obstacle_validation.ellipsoid_geometry import _local_gripper_model, _gripper_mesh_vertices
    from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_link_transforms
    body = _local_gripper_model(closing)
    assert body.key == "gripper"
    poses = robotiq_2f85_link_transforms(closing*.8)
    for name, points in _gripper_mesh_vertices():
        pose = poses[name]
        local = points @ pose[:3, :3].T+pose[:3, 3]-body.center
        assert np.max(np.einsum("ni,ij,nj->n", local, np.linalg.inv(body.shape), local)) <= 1+1e-5, name
    assert _local_gripper_model(closing) is body


@pytest.mark.parametrize("restore_failure", [False, True])
def test_socket_executor_ack_uses_post_command_feedback_and_preserves_grippers(tmp_path, monkeypatch, restore_failure):
    from contextlib import ExitStack
    from obstacle_validation import stream_replay as stream
    state = {"q": np.zeros(12), "grip": np.zeros(2), "pending": None, "events": []}
    class Sampler(TwoPointSampler):
        calibration_digest = "test"
        tcp_transform_verified = {"left": True, "right": True}
        def __init__(self, **kwargs): pass
    class Commander:
        def __init__(self, *args): pass
        def read_state(self):
            state["events"].append("feedback")
            if state["pending"] is not None and not restore_failure:
                state["q"] = state["pending"]
                state["pending"] = None
            return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            state["events"].append("send")
            state["pending"] = q+delta
        def stop(self): state["events"].append("stop")
        def close(self): state["events"].append("closed")
    gripper_commands = []
    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, grip):
            gripper_commands.append(grip.copy())
            state["grip"] = grip.copy()
        def close(self): pass
    snapshot = {"qpos": {"left": np.zeros(6).tolist(), "right": np.zeros(6).tolist()},
                "gripper_position": {"left": 0., "right": 0.}, "diagnostics": {}}
    class Scene:
        def __init__(self, *args, **kwargs): pass
        def read(self, **kwargs): return snapshot
        def boxes(self, value): return []
        def start(self, value): pass
        def close(self): pass
    hardware = tmp_path/"hardware.yaml"
    hardware.write_text("arms:\n  left_arm:\n    robot_ip: fake-left\n  right_arm:\n    robot_ip: fake-right\n")
    monkeypatch.setattr(stream, "load_project_config", lambda path: (
        {"safety": {"camera_calibration_valid": True, "stale_timeout_s": 2., "obstacle_url": "fake",
                    "collision_margin_m": .005, "inter_arm_margin_m": .035},
         "quest3": {"hardware_config": str(hardware)}}, path))
    monkeypatch.setattr(stream, "resolve_repo_path", lambda value: hardware)
    monkeypatch.setattr(stream, "DualUR7eSurfacePointSampler", Sampler)
    monkeypatch.setattr(stream, "DualCommander", Commander)
    monkeypatch.setattr(stream, "DualGripperCommander", Grippers)
    monkeypatch.setattr(stream, "LiveSceneReader", Scene)
    monkeypatch.setattr(stream, "ManualVoxelFilter", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(stream, "validate_calibration_identity", lambda *args: None)
    args = stream.parser().parse_args(["--manual-selection", str(tmp_path/"selection"),
        "--output", str(tmp_path/"out"), "--obstacle-source", "external",
        "--execute", "--execute-ack", stream.EXECUTE_ACK])
    if restore_failure:
        args.restore_timeout = .05
    actions = np.zeros((2,14)); actions[:,0] = [.08,.16]; actions[:,12:] = [[.2,.8],[.7,.1]]
    initial = np.zeros(14); initial[0] = .03; initial[12:] = [.1,.05]
    policy = SimpleNamespace(actions=actions, period=.01, metadata={"initial_state":initial.tolist()})
    policy.infer = lambda start, horizon: actions[start:start+horizon]
    errors, results = [], []
    a, b = socket.socketpair(); a.settimeout(5); b.settimeout(5)
    def receiver():
        try:
            with ExitStack() as cleanup:
                executor = stream.ReplayExecutor(args, cleanup, b)
                results.append(receive_session(b, executor.execute_chunk, start_session=executor.start))
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=receiver); thread.start()
    try:
        if restore_failure:
            with pytest.raises(ProtocolError, match="Initial-state restoration"):
                send_session(a, policy, chunk_size=1)
            thread.join(5)
            assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], TimeoutError)
            assert not results
            records = [json.loads(line) for line in (args.output/"steps.jsonl").read_text().splitlines()]
            assert records and all(r["phase"] == "restore" and r["cbf"] is None for r in records)
            chunks = [json.loads(line) for line in (args.output/"chunks.jsonl").read_text().splitlines()]
            assert [r["status"] for r in chunks] == ["received"]
            return
        reply = send_session(a, policy, chunk_size=1)
        thread.join(5)
        assert not errors and not thread.is_alive()
        assert reply["completed"] and results[0]["actions_executed"] == 2
        assert np.max(np.abs(state["q"]-actions[-1,:12])) < args.final_tolerance
        assert state["pending"] is None
        records = [json.loads(line) for line in (args.output/"steps.jsonl").read_text().splitlines()]
        restore = [r for r in records if r["phase"] == "restore"]
        path = [r for r in records if r["phase"] == "path"]
        assert restore and path and records[:len(restore)] == restore
        assert all(r["cbf"] is None and not r["collision_checks"] for r in restore)
        np.testing.assert_allclose(restore[-1]["qpos"], initial[:12], atol=args.restore_tolerance)
        np.testing.assert_allclose(restore[-1]["gripper_position"], initial[12:], atol=args.gripper_tolerance)
        assert [r["action_index"] for r in path] == sorted(r["action_index"] for r in path)
        assert set(r["action_index"] for r in path) == {0,1}
        assert all(r["cbf"]["controller"] == "ellipsoid_mcbf" for r in path)
        assert any(np.array_equal(g, initial[12:]) for g in gripper_commands)
        assert all(any(np.array_equal(r["gripper_position"], goal) for goal in actions[:,12:]) for r in path)
        events = state["events"]
        assert all(events[i+1] == "feedback" for i, value in enumerate(events[:-1]) if value == "send")
    finally:
        a.close(); b.close()


@pytest.mark.parametrize("physical", [False, True])
def test_stream_restore_uses_only_interpolation_and_feedback(tmp_path, monkeypatch, physical):
    import io
    from obstacle_validation import stream_replay as stream
    clock = [0.]
    monkeypatch.setattr(stream, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda duration: clock.__setitem__(0, clock[0]+duration)))
    executor = object.__new__(stream.ReplayExecutor)
    executor.args = stream.parser().parse_args(["--manual-selection", "selection", "--output", str(tmp_path)])
    executor.connection, peer = socket.socketpair()
    executor.q, executor.grip = np.zeros(12), np.zeros(2)
    executor.initial = np.zeros(14); executor.initial[0] = .2; executor.initial[6] = -.1
    executor.initial[12:] = [.6,.3]
    executor.period = .05
    executor.previous = np.ones(12)
    executor.report, executor.log = {}, io.StringIO()
    executor.restored = False
    forbidden = lambda *a, **kw: pytest.fail("Restoration must not read obstacles or call avoidance")
    executor.scene = SimpleNamespace(read=forbidden, boxes=forbidden)
    executor.controller = SimpleNamespace(plan=forbidden)
    hardware = {"q": executor.q.copy(), "grip": executor.grip.copy(), "stopped": False}
    if physical:
        def send(q, delta): hardware["q"] = q+delta
        def send_grip(grip): hardware["grip"] = grip.copy()
        executor.commander = SimpleNamespace(
            read_state=lambda: (hardware["q"].copy(), hardware["grip"].copy()), send=send,
            stop=lambda: hardware.update(stopped=True))
        executor.grippers = SimpleNamespace(send_if_changed=send_grip)
    else:
        executor.commander = executor.grippers = None
    progress = []
    try:
        executor._restore(lambda index, **detail: progress.append((index, detail)))
        assert executor.restored and executor.report["restore_completed"]
        assert executor.report.get("actions_executed", 0) == 0
        np.testing.assert_allclose(executor.q, executor.initial[:12], atol=executor.args.restore_tolerance)
        np.testing.assert_allclose(executor.grip, executor.initial[12:], atol=executor.args.gripper_tolerance)
        np.testing.assert_array_equal(executor.previous, np.zeros(12))
        records = [json.loads(line) for line in executor.log.getvalue().splitlines()]
        assert all(r["phase"] == "restore" and r["cbf"] is None and not r["collision_checks"] for r in records)
        assert max(np.max(np.abs(r["joint_delta"])) for r in records) <= executor.args.restore_speed*executor.period+1e-12
        grips = np.array([np.zeros(2)]+[r["target_gripper"] for r in records])
        assert np.max(np.abs(np.diff(grips, axis=0))) <= executor.args.gripper_speed_fraction_s*executor.period+1e-12
        assert all(index is None and detail["stage"] == "restore" for index, detail in progress)
        assert hardware["stopped"] == physical
    finally:
        executor.connection.close(); peer.close()


def test_stream_disconnect_detected_during_execution():
    from obstacle_validation.stream_replay import ReplayExecutor
    a, b = socket.socketpair()
    executor = object.__new__(ReplayExecutor); executor.connection = b
    a.close()
    try:
        with pytest.raises(EOFError, match="disconnected"):
            executor._check_peer()
    finally:
        b.close()
