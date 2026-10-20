"""Socket baseline must never acquire perception or instantiate avoidance."""
import json
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation import stream_replay as stream
from obstacle_validation.action_stream import send_session


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("speed", [None, .5])
def test_missing_selection_runs_direct_baseline_and_ack_after_feedback(tmp_path, monkeypatch, physical, speed):
    def forbidden(*a, **kw):
        pytest.fail("Baseline touched perception, camera calibration or collision geometry")
    for name in ("ManualVoxelFilter", "ReplayVoxbloxSource", "LiveSceneReader", "EllipsoidMCBF",
                 "DualUR7eSurfacePointSampler", "validate_calibration_identity"):
        monkeypatch.setattr(stream, name, forbidden)
    hardware = tmp_path/"hardware.yaml"
    hardware.write_text("arms:\n  left_arm:\n    robot_ip: mock-left\n  right_arm:\n    robot_ip: mock-right\n")
    # Baseline also works with unavailable camera calibration and no scene settings.
    monkeypatch.setattr(stream, "load_project_config", lambda path: (
        {"safety": {"camera_calibration_valid": False}, "quest3": {"hardware_config": str(hardware)}}, path))
    clock = [0.]
    monkeypatch.setattr(stream, "time", SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds)))
    state = {"q": np.zeros(12), "grip": np.zeros(2), "pending": None, "events": [], "commands": [], "closed": False}
    class Commander:
        def __init__(self, *a):
            assert physical
        def read_state(self):
            state["events"].append("feedback")
            if state["pending"] is not None:
                state["q"], state["pending"] = state["pending"], None
            return state["q"].copy(), state["grip"].copy()
        def send(self, q, delta):
            state["events"].append("send")
            state["pending"] = q+delta
            state["commands"].append((q.copy(), delta.copy()))
        def stop(self):
            state["events"].append("stop")
        def close(self):
            state["closed"] = True
    grips = []
    class Grippers:
        def __init__(self, *a, **kw):
            assert physical
        def send_if_changed(self, value):
            grips.append(value.copy()); state["grip"] = value.copy()
        def close(self): pass
    monkeypatch.setattr(stream, "DualCommander", Commander)
    monkeypatch.setattr(stream, "DualGripperCommander", Grippers)
    flags = ["--output", str(tmp_path/"run"), "--arrival-tolerance", ".0001", "--final-tolerance", ".0001",
             # These options must not implicitly enable perception in baseline.
             "--obstacle-source", "external", "--obstacle-url", "http://unused", "--robot-mask"]
    if physical:
        flags += ["--execute", "--execute-ack", stream.EXECUTE_ACK]
    if speed is not None:
        flags += ["--speed", str(speed)]
    args = stream.parser().parse_args(flags)
    assert args.manual_selection is None
    initial = np.zeros(14); initial[0] = .03; initial[12:] = [.1, .05]
    actions = np.zeros((3, 14)); actions[:, 0] = [.15, .25, .36]; actions[:, 6] = [-.1, -.2, -.3]
    actions[:, 12:] = [[.2, .8], [.5, .3], [.9, .1]]
    policy = SimpleNamespace(actions=actions, period=.01, metadata={"initial_state": initial.tolist()})
    policy.infer = lambda start, count: actions[start:start+count].copy()
    a, b = socket.socketpair(); a.settimeout(5.)
    monkeypatch.setattr(stream.socket, "create_connection", lambda *a, **kw: b)
    results, errors = [], []
    def receiver():
        try: results.append(stream.run(args))
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=receiver); thread.start()
    try:
        reply = send_session(a, policy, chunk_size=2)
        thread.join(5.)
        assert not thread.is_alive() and not errors
        assert reply["completed"] and results[0]["actions_executed"] == 3
        report = json.loads((args.output/"report.json").read_text())
        assert report["mode"] == report["controller"] == "baseline"
        assert report["manual_selection"] is None and report["obstacle_source"] == "disabled"
        assert not any(report[k] for k in ("perception_enabled", "avoidance_enabled", "robot_mask_enabled"))
        assert report["restore_completed"] and report["distance_relaxation_ticks"] == 0
        rows = [json.loads(line) for line in (args.output/"steps.jsonl").read_text().splitlines()]
        path = [r for r in rows if r["phase"] == "path"]
        assert all(r["cbf"] is None for r in rows)
        assert {r["action_index"] for r in path} == {0, 1, 2}
        for row in path:
            target = actions[row["action_index"]]
            np.testing.assert_array_equal(row["control_target"][12:], target[12:])
            if speed is None:
                np.testing.assert_array_equal(row["control_target"], target)
        previous = np.array(report["restored_qpos"])
        for row in path:
            target = np.array(row["control_target"][:12])
            if speed is not None:
                assert np.max(np.abs(target-previous)) <= speed*policy.period+1e-12
            previous = target
        chunks = [json.loads(line) for line in (args.output/"chunks.jsonl").read_text().splitlines()]
        assert [r["status"] for r in chunks] == ["received", "executed", "received", "executed"]
        if physical:
            assert state["closed"] and state["pending"] is None
            np.testing.assert_allclose(state["q"], actions[-1, :12], atol=args.final_tolerance)
            assert all(state["events"][i+1] == "feedback" for i, e in enumerate(state["events"][:-1]) if e == "send")
            assert all(any(np.array_equal(g, target) for g in grips) for target in actions[:, 12:])
        else:
            assert not state["commands"] and not grips
    finally:
        a.close(); b.close(); thread.join(5.)


def test_baseline_execute_still_requires_ack_before_connecting(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Connected before checking physical execution acknowledgement")
    monkeypatch.setattr(stream.socket, "create_connection", forbidden)
    args = stream.parser().parse_args(["--output", str(tmp_path), "--execute"])
    with pytest.raises(ValueError, match="execute-ack"):
        stream.run(args)
