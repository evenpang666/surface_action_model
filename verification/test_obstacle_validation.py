"""Geometry and snapshot checks shared by LeRobot replay."""

import time

import numpy as np
import pytest

from obstacle_validation.core import CollisionChecker, JointPath, PathFollower, rrt_detour, segments_within_box_margin
from obstacle_validation.runtime import boxes_from_json, read_snapshot
from real_scripts.real_cbf_qp import OrientedBox


class TwoPointSampler:
    """Two arms: one moving point in XY and one distant stationary point."""
    left_link_names = ("left",)
    points_per_link = 1

    def link_points(self, q, grip):
        return np.asarray([[[q[0], q[1], 0]], [[3, 3, 3]]])


def fixture():
    q = np.zeros((101, 12))
    q[:, 0] = np.linspace(0, 1, len(q))
    low, high = np.full(12, -.01), np.full(12, .01)
    low[:2], high[:2] = (-.2, -.6), (1.2, .6)
    path = JointPath(q, np.zeros(2), low, high, .1, .1, .3, "toy", 7)
    box = OrientedBox(np.array([.5, 0, 0]), np.eye(3), np.array([.08, .15, .1]))
    return path, CollisionChecker(TwoPointSampler(), path.gripper, [box], resolution=.02)


def test_segment_collision_catches_clear_endpoints_and_inter_arm_crossing():
    path, checker = fixture()
    assert checker.state_safe(path.q[0]) and checker.state_safe(path.q[-1])
    assert not checker.edge_safe(path.q[0], path.q[-1])

    class Crossing(TwoPointSampler):
        def link_points(self, q, grip):
            return np.array([[[q[0], 0, 0]], [[-q[0], 0, 0]]])

    check = CollisionChecker(Crossing(), np.zeros(2), [], resolution=3.)
    first, last = np.zeros(12), np.zeros(12)
    first[0], last[0] = -1, 1
    assert check.state_safe(first) and check.state_safe(last)
    assert not check.edge_safe(first, last)


def test_rrt_finds_checked_detour_instead_of_stop():
    path, checker = fixture()
    route = rrt_detour(path.q[0], path.q[-1], checker, path.lower, path.upper,
                       rng=np.random.default_rng(7), timeout_s=3.)
    assert route is not None and len(route) >= 2
    assert checker.path_safe(np.asarray([path.q[0], *route]))
    np.testing.assert_allclose(route[-1], path.q[-1])


def test_segment_margin_matches_euclidean_clearance_at_box_corners():
    box = OrientedBox(np.zeros(3), np.eye(3), np.array([.01, .01, .1]))
    # Both states are outside the round 5 cm margin, despite being inside an
    # axis-wise expanded box. Movement away from the corner must be allowed.
    start = np.array([[-.05, -.05, 0.]])
    end = np.array([[-.08, -.08, 0.]])
    assert not segments_within_box_margin(start, end, box, .05).any()
    assert not segments_within_box_margin(start, start, box, .05).any()
    # Safe endpoints can cross the rounded margin without entering the box.
    assert segments_within_box_margin(np.array([[-.06, -.02, 0.]]),
                                      np.array([[-.02, -.06, 0.]]), box, .05).all()


def test_segment_box_distance_matches_independent_numeric_minimization():
    from scipy.optimize import minimize_scalar
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(21)
    box = OrientedBox(np.array([.1, -.2, .05]), Rotation.from_euler('xyz', [.2, -.3, .7]).as_matrix(),
                      np.array([.1, .15, .05]))
    start, end = rng.uniform(-.5, .5, (2, 80, 3))
    flags = segments_within_box_margin(start, end, box, .07)
    for a, b, blocked in zip(start, end, flags):
        def distance(t):
            local = (a+t*(b-a)-box.center) @ box.axes
            return np.square(np.maximum(np.abs(local)-box.half_sizes, 0)).sum()
        result = minimize_scalar(distance, bounds=(0, 1), method='bounded', options={'xatol': 1e-12})
        minimum = min(distance(0), distance(1), result.fun)
        assert bool(blocked) == (minimum <= .07**2)


def test_baseline_blocks_but_avoid_completes_and_rejoins():
    path, checker = fixture()
    for avoid in (False, True):
        follower = PathFollower(path, avoid=avoid, detour_lower=path.lower,
                                detour_upper=path.upper, planning_timeout=3.)
        q = path.q[0].copy()
        for step in range(1000):
            delta, state = follower.next_delta(q, checker)
            assert checker.edge_safe(q, q+delta)
            if state not in ("nominal", "detour"):
                break
            q += delta
        assert state == ("completed" if avoid else "baseline_blocked")
        if avoid:
            assert follower.replans > 0
            assert np.max(np.abs(q-path.q[-1])) <= follower.tolerance


def test_impossible_detour_and_invalid_inputs_are_not_success():
    path, checker = fixture()
    follower = PathFollower(path, avoid=True, detour_lower=path.lower,
                            detour_upper=path.upper, lookahead=1)
    q = path.q[0].copy()
    q[0] = .42
    _, state = follower.next_delta(q, checker)
    assert state == "unsafe_current_state"
    with pytest.raises(ValueError):
        boxes_from_json({"coordinate_frame": "camera", "obbs": []})

    # A wall across the entire permitted XY envelope cannot be routed around.
    wall = OrientedBox(np.array([.5, 0, 0]), np.eye(3), np.array([.08, 2, 2]))
    impossible = CollisionChecker(TwoPointSampler(), path.gripper, [wall])
    assert rrt_detour(path.q[0], path.q[-1], impossible, path.lower, path.upper,
                      rng=np.random.default_rng(7), iterations=40, timeout_s=.5) is None


def snapshot(q):
    return {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
            "status": "ok", "host_timestamp_ns": time.monotonic_ns(), "obbs": [],
            "qpos": {"left": q[:6].tolist(), "right": q[6:].tolist()},
            "gripper_position": {"left": 0., "right": 0.}}


def test_stale_snapshot_rejected(monkeypatch):
    import io
    import json
    import obstacle_validation.runtime as runtime
    fresh = snapshot(np.zeros(12))
    fresh["host_timestamp_ns"] -= 10_000_000_000
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kw: io.StringIO(json.dumps(fresh)))
    with pytest.raises(RuntimeError, match="expired"):
        read_snapshot("http://unused", 1.)


def test_explicit_no_age_cutoff_keeps_network_timeout(monkeypatch):
    import io
    import json
    import obstacle_validation.runtime as runtime
    old = snapshot(np.zeros(12))
    old["host_timestamp_ns"] -= 10_000_000_000
    timeouts = []

    def open_snapshot(request, *, timeout):
        timeouts.append(timeout)
        return io.StringIO(json.dumps(old))

    monkeypatch.setattr(runtime, "urlopen", open_snapshot)
    value = read_snapshot("http://unused", float("inf"), http_timeout_s=.5)
    assert value["host_timestamp_ns"] == old["host_timestamp_ns"]
    assert timeouts == [.5]
    with pytest.raises(ValueError, match="HTTP timeout"):
        read_snapshot("http://unused", float("inf"))


def test_startup_wait_reads_new_capture_and_requests_compact_payload(monkeypatch):
    import io
    import json
    from urllib.parse import parse_qs, urlsplit
    import obstacle_validation.runtime as runtime
    old, fresh = snapshot(np.zeros(12)), snapshot(np.zeros(12))
    old["host_timestamp_ns"] -= 10_000_000_000
    replies, requests = iter([old, fresh]), []

    def open_snapshot(request, **kwargs):
        requests.append(request.full_url)
        return io.StringIO(json.dumps(next(replies)))

    monkeypatch.setattr(runtime, "urlopen", open_snapshot)
    value = read_snapshot("http://unused/snapshot.json?session=7&compact=0", 1., max_wait_s=.5)
    assert value["host_timestamp_ns"] == fresh["host_timestamp_ns"]
    assert len(requests) == 2
    assert parse_qs(urlsplit(requests[0]).query) == {"session": ["7"], "compact": ["1"]}


def test_startup_waits_for_obb_point_confirmation_but_motion_read_fails(monkeypatch):
    import io
    import json
    import obstacle_validation.runtime as runtime

    unstable = snapshot(np.zeros(12))
    unstable["status"] = "error: unstable obstacle point clusters (new_cluster)"
    stable = snapshot(np.zeros(12))
    replies = iter([unstable, stable])
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(next(replies))))
    assert read_snapshot("http://unused", 1., max_wait_s=.5)["status"] == "ok"
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(unstable)))
    with pytest.raises(runtime.ObstacleSceneUnstable, match="unstable obstacle point clusters"):
        read_snapshot("http://unused", 1.)


@pytest.mark.parametrize("reason,covered", [
    ("merged_clusters, missing_cluster", False),
    ("new_cluster", True),
    ("changed_cluster", True),
])
def test_live_replay_accepts_fresh_unstable_scene_with_all_obbs(monkeypatch, reason, covered):
    import io
    import json
    import obstacle_validation.runtime as runtime

    value = snapshot(np.zeros(12))
    value["status"] = f"error: unstable obstacle point clusters ({reason})"
    value["obbs"] = [{"center": [8., 8., 8.], "axes": np.eye(3).tolist(),
                      "half_sizes": [.1, .1, .1]}]
    value["diagnostics"] = {"obb_unstable_reasons": sorted(reason.split(", ")),
                             "published_obbs": 1,
                             "safety_obbs_cover_candidates": covered}
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(value)))
    assert len(read_snapshot("http://unused", 1., allow_tracked_obbs=True)["obbs"]) == 1
    with pytest.raises(runtime.ObstacleSceneUnstable):
        read_snapshot("http://unused", 1.)


def test_uncovered_new_cluster_and_expired_tracked_obbs_are_rejected(monkeypatch):
    import io
    import json
    import obstacle_validation.runtime as runtime

    value = snapshot(np.zeros(12))
    value["status"] = "error: unstable obstacle point clusters (new_cluster)"
    value["obbs"] = [{"center": [8., 8., 8.], "axes": np.eye(3).tolist(),
                      "half_sizes": [.1, .1, .1]}]
    value["diagnostics"] = {"obb_unstable_reasons": ["new_cluster"],
                             "published_obbs": 1}
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(value)))
    with pytest.raises(runtime.ObstacleSceneUnstable):
        read_snapshot("http://unused", 1., allow_tracked_obbs=True)
    value["diagnostics"]["safety_obbs_cover_candidates"] = True
    value["host_timestamp_ns"] -= 10_000_000_000
    with pytest.raises(runtime.SnapshotExpired):
        read_snapshot("http://unused", 1., allow_tracked_obbs=True)


@pytest.mark.parametrize("fault", ["permanently_stale", "future_clock", "service_error"])
def test_snapshot_wait_is_bounded_and_only_retries_expiry(monkeypatch, fault):
    import io
    import json
    from types import SimpleNamespace
    import obstacle_validation.runtime as runtime
    value = snapshot(np.zeros(12))
    value["host_timestamp_ns"] += 10_000_000_000 if fault == "future_clock" else -10_000_000_000
    if fault == "service_error": value["status"] = "error: disconnected camera"
    clock, requests = [0.], []
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0],
                                                       sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds)))

    def open_snapshot(*args, **kwargs):
        requests.append(True)
        return io.StringIO(json.dumps(value))

    monkeypatch.setattr(runtime, "urlopen", open_snapshot)
    with pytest.raises(RuntimeError):
        read_snapshot("http://unused", 1., max_wait_s=.2)
    assert clock[0] <= .2
    assert len(requests) == (5 if fault == "permanently_stale" else 1)


def test_recent_publication_does_not_refresh_old_capture(monkeypatch):
    import io
    import json
    import obstacle_validation.runtime as runtime
    value = snapshot(np.zeros(12))
    value["host_timestamp_ns"] -= 10_000_000_000
    value["published_timestamp_ns"] = time.monotonic_ns()
    value["diagnostics"] = {"depth_refinement": "lingbot-depth"}
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(value)))
    with pytest.raises(RuntimeError, match="no-lingbot-depth"):
        read_snapshot("http://unused", 1.)
