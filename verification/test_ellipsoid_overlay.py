"""Viewer ellipsoids follow the selected or complete current voxel scope."""
import json
from types import SimpleNamespace

import numpy as np

from obstacle_validation import ellipsoid_geometry as geometry
from real_scripts.ellipsoid_overlay import EllipsoidOverlay
from real_scripts.live_dual_ur7e_obstacle_model import snapshot_payload
from real_scripts.real_cbf_qp import OrientedBox
from safety_module.occupancy import LocalOccupancyGrid


def scene():
    right = np.eye(4); right[:3, 3] = [0, 1.1, .2]
    return SimpleNamespace(project={"safety": {"ellipsoid_mcbf": {"obstacle_max_span_m": .2}}},
                           arms=[SimpleNamespace(name="right", base_to_left=right)],
                           flange_to_tcp={"left": np.eye(4), "right": np.eye(4)})


def snapshot():
    value = {"coordinate_frame": "left_base", "frame": 12,
            "qpos": {"left": [.2, -.7, .8, .4, -.3, .1], "right": [-.2, -.8, .6, .2, .3, -.1]},
            "gripper_position": {"left": .2, "right": .8},
            "obbs": [{"center": [3., 0., 0.], "axes": np.eye(3).tolist(), "half_sizes": [.02]*3}],
            "viewer_filter": {"state": "applied", "obbs": [
                {"center": [.5, .1, .3], "axes": np.eye(3).tolist(), "half_sizes": [.02, .03, .04]}]}}
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for item in [*value["obbs"], *value["viewer_filter"]["obbs"]]:
        item["corners"] = (np.asarray(item["center"])+signs*np.asarray(item["half_sizes"])).tolist()
    return value


def decode(item):
    axes, radii = np.asarray(item["axes"]), np.asarray(item["radii"])
    return np.asarray(item["center"]), (axes*radii**2) @ axes.T


def test_filtered_overlay_uses_identical_replay_fit_and_caches_static_map(monkeypatch):
    value, overlay = snapshot(), EllipsoidOverlay(scene())
    original, calls = geometry.fit_obstacles, []
    def fit(boxes, **options):
        calls.append(options)
        return original(boxes, **options)
    monkeypatch.setattr(geometry, "fit_obstacles", fit)
    result = overlay.build(value, groups=["obstacles"], filtered=True)
    box = value["viewer_filter"]["obbs"][0]
    expected = original([OrientedBox(*(np.asarray(box[k]) for k in ("center", "axes", "half_sizes")))],
                        max_parts=8, max_span=.2)[0]
    center, shape = decode(result["obstacles"][0])
    np.testing.assert_allclose(center, expected.center)
    np.testing.assert_allclose(shape, expected.shape, atol=1e-12)
    assert result["scope"] == "selected_cells" and not result["margin_included"]
    assert overlay.robot is None
    again = json.loads(json.dumps(value)); again["frame"] += 1
    overlay.build(again, groups=["obstacles"], filtered=True)
    assert len(calls) == 1
    # Changes outside selection never reach the fitter or invalidate its cache.
    again["obbs"][0]["center"][0] += 10
    overlay.build(again, groups=["obstacles"], filtered=True)
    assert len(calls) == 1
    again["viewer_filter"]["obbs"][0]["center"][0] += .1
    overlay.build(again, groups=["obstacles"], filtered=True)
    assert len(calls) == 2


def test_robot_overlay_uses_actual_scene_calibration_and_articulated_grippers():
    value, overlay = snapshot(), EllipsoidOverlay(scene())
    result = overlay.build(value, groups=["robot"])
    native = overlay.robot.bodies(np.r_[value["qpos"]["left"], value["qpos"]["right"]], np.array([.2, .8]))
    assert len(native) == len(result["robot"]) == 16
    for arm in (0, 1):
        bodies = [b for b in result["robot"] if b["arm"] == arm]
        assert len({b["link"] for b in bodies}) == 8
        assert sum(b["key"].endswith("_gripper") for b in bodies) == 1
    for encoded, body in zip(result["robot"], native):
        center, shape = decode(encoded)
        np.testing.assert_allclose(center, body.center)
        np.testing.assert_allclose(shape, body.shape, atol=1e-12)
        assert encoded["key"] == body.key and encoded["arm"] == body.arm
    assert not result["obstacles"]
    value["gripper_position"]["right"] = 0
    changed = overlay.build(value, groups=["robot"])["robot"]
    assert any(not np.allclose(a["center"], b["center"]) or not np.allclose(a["radii"], b["radii"])
               for a,b in zip(changed, result["robot"]))


def test_unavailable_selection_never_falls_back_to_all_obstacles():
    value, overlay = snapshot(), EllipsoidOverlay(scene())
    value["viewer_filter"] = {"state": "missing_annotations"}
    result = overlay.build(value, groups=["obstacles"], filtered=True)
    assert not result["obstacles"] and result["messages"]
    del value["viewer_filter"]
    result = overlay.build(value, groups=["obstacles"], filtered=True)
    assert not result["obstacles"] and result["messages"]
    result = overlay.build(value, groups=["obstacles"], frozen=True, filtered=True)
    assert not result["obstacles"] and result["scope"] == "frozen_selection"
    value["manual_selection"] = {"centers": [[0, 0, .1], [1, 0, .1]],
                                 "axes": np.eye(3).tolist(), "resolution_m": .02}
    result = overlay.build(value, groups=["obstacles"], frozen=True, filtered=True)
    assert len(result["obstacles"]) == 2
    assert result["source_boxes"] == 2


def test_unfiltered_overlay_uses_all_current_voxels_and_switches_scope(monkeypatch):
    value, overlay = snapshot(), EllipsoidOverlay(scene())
    grid = LocalOccupancyGrid([0., 0., 0.], np.eye(3), [10, 1, 1], .02)
    centers = [[.01, .01, .01], [.17, .01, .01]]
    grid.update(timestamp_ns=100, raw_hits=centers, free_ids=np.empty(0, np.int64))
    value["occupancy_map"] = grid.snapshot()
    calls, original = [], geometry.fit_obstacles
    def fit(boxes, **options):
        calls.append(boxes)
        return original(boxes, **options)
    monkeypatch.setattr(geometry, "fit_obstacles", fit)
    result = overlay.build(value, groups=["obstacles"])
    assert result["scope"] == "all_cells" and result["source_boxes"] == 2
    np.testing.assert_allclose([box.center for box in calls[-1]], centers)
    # Neither the detector's padded OBB nor saved selection replaces live cells.
    assert len(result["obstacles"]) == 2
    selected = overlay.build(value, groups=["obstacles"], filtered=True)
    assert selected["scope"] == "selected_cells" and selected["source_boxes"] == 1
    np.testing.assert_allclose(calls[-1][0].center, value["viewer_filter"]["obbs"][0]["center"])
    restored = overlay.build(value, groups=["obstacles"])
    assert restored["obstacles"] == result["obstacles"]
    # Cleared voxels leave the next frame's model even without changing labels.
    grid.update(timestamp_ns=200, raw_hits=[centers[1]], free_ids=np.empty(0, np.int64))
    value["occupancy_map"] = grid.snapshot()
    current = overlay.build(value, groups=["obstacles"])
    assert current["source_boxes"] == len(current["obstacles"]) == 1
    frozen = overlay.build(value, groups=["obstacles"], frozen=True)
    assert frozen["scope"] == "frozen_all_cells" and frozen["obstacles"] == current["obstacles"]


def test_missing_robot_and_failed_fit_preserve_existing_viewer(monkeypatch):
    value, overlay = snapshot(), EllipsoidOverlay(scene())
    value["qpos"] = {}
    result = overlay.build(value, groups=["robot", "obstacles"], filtered=True)
    assert not result["robot"] and result["messages"] and result["obstacles"]
    def failed(*args, **kwargs):
        raise ValueError("fit failed")
    monkeypatch.setattr(geometry, "fit_obstacles", failed)
    value["viewer_filter"]["obbs"][0]["center"][0] += 1
    attached = overlay.attach(value, groups=["obstacles"], filtered=True)
    assert attached["ellipsoid_view"]["error"] == "ValueError: fit failed"
    assert "ellipsoid_view" not in value
    assert attached["obbs"] is value["obbs"]
    rendered = json.loads(snapshot_payload(attached, viewer=True, render_only=True))
    assert rendered["ellipsoid_view"]["error"] == "ValueError: fit failed"
