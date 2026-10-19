"""Latest-map reads must not wait for an in-flight detector refresh."""

import threading
import time

import pytest

from obstacle_validation.runtime import LatestSnapshotCache
from safety_module.runtime_contract import SnapshotExpired, SnapshotUnknown, validate_snapshot


def snapshot(stamp=None, **changes):
    return {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
            "host_timestamp_ns": time.monotonic_ns() if stamp is None else stamp,
            "status": "ok", **changes}


def test_control_reuses_latest_map_while_detector_fetch_is_blocked():
    initial = snapshot()
    fetching, release, finished = threading.Event(), threading.Event(), threading.Event()

    def fetch():
        fetching.set()
        release.wait(2.)
        return snapshot()

    cache = LatestSnapshotCache(fetch, lambda value: validate_snapshot(value, 2.), initial)
    result = []
    def control():
        result.extend(cache.read() for _ in range(100))
        finished.set()

    thread = threading.Thread(target=control, daemon=True)
    try:
        assert fetching.wait(1.)
        thread.start()
        assert finished.wait(1.), "Control waited for the next detector frame"
        assert not release.is_set()
        assert len(result) == 100 and all(value is initial for value in result)
    finally:
        release.set()
        thread.join(1.)
        cache.close()


def test_cached_capture_still_expires_without_waiting_for_refresh():
    release = threading.Event()
    initial = snapshot(time.monotonic_ns()-3_000_000_000)
    cache = LatestSnapshotCache(lambda: (release.wait(2.), initial)[1],
                                lambda value: validate_snapshot(value, 2.), initial)
    try:
        with pytest.raises(SnapshotExpired):
            cache.read()
    finally:
        release.set()
        cache.close()


def test_only_explicit_expired_evidence_wait_requests_require_newer_capture():
    release = threading.Event()
    initial, fresh = snapshot(), snapshot()
    fresh["host_timestamp_ns"] = initial["host_timestamp_ns"]+1
    cache = LatestSnapshotCache(lambda: (release.wait(2.), fresh)[1],
                                lambda value: validate_snapshot(value, 2.), initial)
    try:
        assert cache.read() is initial
        with pytest.raises(SnapshotExpired, match="newer capture"):
            cache.read(after_timestamp_ns=initial["host_timestamp_ns"])
        release.set()
        assert cache.read(max_wait_s=1., after_timestamp_ns=initial["host_timestamp_ns"]) is fresh
    finally:
        release.set()
        cache.close()


def test_new_unknown_scene_replaces_valid_cached_scene():
    unknown = snapshot(scene_observation_state="unknown")
    cache = LatestSnapshotCache(lambda: unknown, lambda value: validate_snapshot(value, 2.), snapshot())
    try:
        with cache.condition:
            assert cache.condition.wait_for(lambda: cache.refreshes > 0, timeout=1.)
        with pytest.raises(SnapshotUnknown):
            cache.read()
    finally:
        cache.close()


def test_live_scene_decodes_reused_snapshot_once(monkeypatch):
    from obstacle_validation import lerobot_replay as replay
    calls = []
    def decode(value):
        calls.append(value)
        return []
    monkeypatch.setattr(replay, "boxes_from_json", decode)
    reader = replay.LiveSceneReader("unused", 2.)
    first, second = snapshot(), snapshot()
    boxes = reader.boxes(first)
    assert reader.boxes(first) is boxes and len(calls) == 1
    assert reader.boxes(second) is not boxes and len(calls) == 2


def test_latest_policy_reuses_expired_capture_without_any_new_frame_wait():
    from obstacle_validation.lerobot_replay import LiveSceneReader
    old = snapshot(time.monotonic_ns()-10_000_000_000)
    release, fetching = threading.Event(), threading.Event()
    reader = LiveSceneReader("unused", 2., reuse_expired=True)
    def fetch():
        fetching.set()
        release.wait(2.)
        return old
    reader._read_direct = fetch
    reader.start(old)
    try:
        assert fetching.wait(1.)
        assert reader.read(max_wait_s=3., after_timestamp_ns=old['host_timestamp_ns']) is old
        assert reader.stale_snapshot_reads == 1 and reader.max_snapshot_age_s >= 10.
        assert not release.is_set()
    finally:
        release.set()
        reader.close()


def test_latest_policy_uses_capture_time_belief_but_unknown_and_occupied_still_block():
    import numpy as np
    from obstacle_validation.lerobot_replay import LiveSceneReader, _motion_safe
    from safety_module.occupancy import LocalOccupancyGrid
    from verification.test_obstacle_validation import TwoPointSampler
    stamp = time.monotonic_ns()-10_000_000_000
    grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    for t in (stamp-2_000_000, stamp-1_000_000, stamp):
        grid.update(timestamp_ns=t, raw_hits=np.empty((0, 3)), free_ids=np.arange(grid.size))
    old = snapshot(stamp, occupancy_map=grid.snapshot(), obbs=[])
    latest = LiveSceneReader('unused', 2., reuse_expired=True)
    strict = LiveSceneReader('unused', 2.)
    q, grip, delta = np.zeros(12), np.zeros(2), np.zeros(12)
    delta[0] = .01
    safety = {'collision_margin_m': .01, 'inter_arm_margin_m': .035}
    assert not _motion_safe(TwoPointSampler(), q, grip, delta, grip, strict.boxes(old), safety)
    assert _motion_safe(TwoPointSampler(), q, grip, delta, grip, latest.boxes(old), safety)
    assert latest.boxes(old).observation.timestamp_ns == stamp
    assert old['host_timestamp_ns'] == old['occupancy_map']['capture_timestamp_ns'] == stamp
    grid.update(timestamp_ns=stamp+1_000_000, raw_hits=[[.01, .01, .01]], free_ids=np.empty(0, np.int64))
    occupied = snapshot(grid.timestamp_ns, occupancy_map=grid.snapshot(), obbs=[])
    assert not _motion_safe(TwoPointSampler(), q, grip, delta, grip, latest.boxes(occupied), safety)
    unknown_grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    unknown_grid.update(timestamp_ns=stamp+2_000_000, raw_hits=[], free_ids=np.empty(0, np.int64))
    unknown = snapshot(unknown_grid.timestamp_ns, occupancy_map=unknown_grid.snapshot(), obbs=[])
    assert not _motion_safe(TwoPointSampler(), q, grip, delta, grip, latest.boxes(unknown), safety)
    with pytest.raises(SnapshotUnknown):
        latest._validate(snapshot(stamp, scene_observation_state='unknown'))


def test_latest_policy_retains_known_frame_on_transport_timeout():
    old = snapshot(time.monotonic_ns()-10_000_000_000)
    def fetch():
        raise TimeoutError('detector did not respond')
    cache = LatestSnapshotCache(fetch, lambda value: validate_snapshot(value, float('inf')), old,
                                reuse_on_transport_error=True)
    try:
        with cache.condition:
            assert cache.condition.wait_for(lambda: cache.refreshes > 0, timeout=1.)
        assert cache.read() is old and cache.refresh_errors > 0
    finally:
        cache.close()


def test_transport_error_does_not_hide_new_unknown_frame():
    from obstacle_validation.lerobot_replay import LiveSceneReader
    reader = LiveSceneReader('unused', 2., reuse_expired=True)
    calls = 0
    def fetch():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SnapshotUnknown('new frame contains unknown space')
        raise TimeoutError('detector did not respond')
    cache = LatestSnapshotCache(fetch,
                                reader._validate, snapshot(), reuse_on_transport_error=True)
    try:
        with cache.condition:
            assert cache.condition.wait_for(lambda: cache.refreshes >= 2, timeout=1.)
        with pytest.raises(SnapshotUnknown):
            cache.read()
    finally:
        cache.close()
