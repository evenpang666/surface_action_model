"""RGB annotations select current cells without saved 3D selections or maps."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation import manual_annotation_filter as module
from obstacle_validation import lerobot_replay as replay
from real_scripts import replay_voxblox as embedded
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.manual_obstacle_selection import ManualSelectionStore
from safety_module.project_config import DEFAULT_PROJECT_CONFIG
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED, OUTSIDE


@pytest.fixture
def annotated(tmp_path, monkeypatch):
    side = np.eye(4)
    side[0, 3] = .2
    shared = SimpleNamespace(digest="test-digest", serials={"front": "f", "side": "s"},
                             camera_to_left_base={"front": np.eye(4), "side": side})
    monkeypatch.setattr(module, "calibration_from_project", lambda project: shared)
    K = [[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]]
    color = {name: {"width": 128, "height": 96, "intrinsics": K, "distortion_model": "none"}
             for name in ("front", "side")}
    boxes = {"front": [{"object_id": "1", "cx": 64., "cy": 48., "width": 40., "height": 40., "angle_deg": 0.}],
             "side": [{"object_id": "1", "cx": 44., "cy": 48., "width": 10., "height": 20., "angle_deg": 0.}]}
    row = {"schema": "dual_rgb_obb_annotation_v2", "annotation_image": "recorded_rgb", "capture_id": "stamp",
           "camera_names": ["front", "side"], "reviewed": True, "status": "objects", "boxes": boxes,
           "cameras": {name: {"rgb_width": 128, "rgb_height": 96,
                              "stream": {"serial": shared.serials[name], "color_calibration": color[name]}}
                       for name in color}}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(row))
    grid = LocalOccupancyGrid([-.4, -.2, .9], np.eye(3), [12, 4, 4], .1)
    cameras = {name: CameraCalibration(name, K, transform, distortion_model="none")
               for name, transform in shared.camera_to_left_base.items()}
    return SimpleNamespace(path=path, row=row, grid=grid, color=color, cameras=cameras, shared=shared)


def snapshot(grid):
    return {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base", "status": "ok",
            "obstacle_backend": "voxblox", "occupancy_map": grid.snapshot(), "host_timestamp_ns": grid.timestamp_ns,
            "calibration": {"sha256": "test-digest"}, "unknown_obstacles": [], "diagnostics": {}, "obbs": []}


def test_standalone_annotations_filter_new_live_cells_without_saved_selection_or_map(annotated):
    a = annotated
    manual = module.ManualAnnotationFilter(a.path)
    inside, front_only = [-.05, .05, 1.05], [.15, .05, 1.05]
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=[inside, front_only], free_ids=np.empty(0, np.int64))
    original = snapshot(a.grid)
    filtered = manual(original)
    observed = ObservationGrid(filtered["occupancy_map"])
    assert observed.state[a.grid.point_ids([inside])[0]] == OCCUPIED
    assert observed.state[a.grid.point_ids([front_only])[0]] == OUTSIDE
    assert filtered["diagnostics"]["manual_selection"]["scope_policy"] == "dual_rgb_annotation_intersection"
    assert filtered["occupancy_map"]["occupied_voxels"] == 1
    assert ObservationGrid(original["occupancy_map"]).state[a.grid.point_ids([front_only])[0]] == OCCUPIED
    # A different live cell can enter the fixed RGB intersection after capture.
    moved = [-.05, .05, 1.15]
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=[moved], free_ids=np.empty(0, np.int64))
    filtered = manual(snapshot(a.grid))
    assert filtered["occupancy_map"]["occupied_voxels"] == 1
    np.testing.assert_allclose(filtered["obbs"][0]["center"], moved)
    assert not (a.path.parent / "selection.json").exists()
    assert not (a.path.parent / "occupancy_map.json").exists()


def test_viewer_loads_latest_annotations_and_filters_only_live_display(annotated):
    a = annotated
    directory = a.path.parent / "captures" / "new"
    directory.mkdir(parents=True)
    (directory / "annotations.json").write_text(json.dumps(a.row))
    store = ManualSelectionStore(directory.parent)
    assert store.load_latest_annotations(DEFAULT_PROJECT_CONFIG) == directory / "annotations.json"
    config = store.configuration()
    assert not config["viewer_filter_default"] and config["annotation_error"] is None
    assert config["annotation_path"] == str(directory / "annotations.json")
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=[[-.05, .05, 1.05], [.15, .05, 1.05]],
                  free_ids=np.empty(0, np.int64))
    original = {**snapshot(a.grid), "frame": 7}
    before = json.dumps(original)
    result = store.viewer_filter(original)
    assert result["state"] == "applied" and result["voxel_count"] == 1
    assert result["scope_policy"] == "dual_rgb_annotation_intersection"
    np.testing.assert_allclose(result["obbs"][0]["center"], [-.05, .05, 1.05])
    assert json.dumps(original) == before
    assert store.viewer_filter({**original, "calibration": {"sha256": "changed"}})["state"] == "annotation_invalid"


def test_viewer_reports_invalid_saved_annotations_without_breaking_detection(annotated):
    a = annotated
    directory = a.path.parent / "captures" / "new"
    directory.mkdir(parents=True)
    (directory / "annotations.json").write_text("{invalid json")
    store = ManualSelectionStore(directory.parent)
    assert store.load_latest_annotations(DEFAULT_PROJECT_CONFIG) is not None
    assert not store.configuration()["viewer_filter_default"]
    result = store.viewer_filter({**snapshot(a.grid), "frame": 1})
    assert result["state"] == "annotation_invalid" and result["error"]
    assert not result["obbs"]


def test_viewer_without_saved_annotations_keeps_unfiltered_default(tmp_path):
    store = ManualSelectionStore(tmp_path)
    assert store.load_latest_annotations(DEFAULT_PROJECT_CONFIG) is None
    assert not store.configuration()["viewer_filter_default"]


def test_legacy_annotations_use_capture_rgb_metadata_without_3d_files(annotated):
    a = annotated
    capture = {"capture_id": "stamp", "coordinate_frame": "left_base", "annotation_image": "recorded_rgb",
               "calibration": {"sha256": "test-digest", "camera_serials": a.shared.serials},
               "cameras": {name: {"rgb_width": 128, "rgb_height": 96, "intrinsics": info["intrinsics"]}
                           for name, info in a.color.items()}}
    (a.path.parent / "capture.json").write_text(json.dumps(capture))
    row = {"capture_id": "stamp", "coordinate_frame": "left_base", "annotation_image": "recorded_rgb",
           "camera_names": ["front", "side"], "annotations": a.row["boxes"]}
    a.path.write_text(json.dumps(row))
    manual = module.ManualAnnotationFilter(a.path)
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=[[-.05, .05, 1.05]], free_ids=np.empty(0, np.int64))
    assert manual(snapshot(a.grid))["occupancy_map"]["occupied_voxels"] == 1
    assert not (a.path.parent / "selection.json").exists()
    capture["capture_id"] = "other"
    (a.path.parent / "capture.json").write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="IDs differ"):
        module.ManualAnnotationFilter(a.path)


def test_matching_object_ids_are_intersected_before_combining_objects(annotated):
    a = annotated
    # Front sees the central voxel as object 1; side sees it as object 2.
    a.row["boxes"] = {name: [dict(object_id=str(i+1), cx=cx, cy=48., width=12., height=20., angle_deg=0.)
                             for i, cx in enumerate(centers)]
                      for name, centers in {"front": [64., 100.], "side": [90., 44.]}.items()}
    a.path.write_text(json.dumps(a.row))
    manual = module.ManualAnnotationFilter(a.path)
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=[[.0, .0, 1.0]], free_ids=np.empty(0, np.int64))
    # Cross-matched boxes cannot create a fictitious transparent object either.
    with pytest.raises(ValueError, match="no 3D intersection"):
        manual(snapshot(a.grid))


def test_old_standalone_without_intrinsics_requires_exact_rgb_profile(annotated):
    a = annotated
    for info in a.row["cameras"].values():
        del info["stream"]["color_calibration"]
    a.path.write_text(json.dumps(a.row))
    manual = module.ManualAnnotationFilter(a.path)
    with pytest.raises(ValueError, match="--width 128 --height 96"):
        manual.bind_color_calibrations({name: {**info, "height": 72} for name, info in a.color.items()})
    manual.prepare_reference(a.grid, a.color)
    assert manual.cell_scope.ids.size > 0 and manual.stream_size == (128, 96)
    assert manual.cell_scope.ids.size < a.grid.size


@pytest.mark.parametrize("fault", ["wrong_serial", "missing_side", "unreviewed", "no_obstacles", "coordinate_frame"])
def test_invalid_annotation_never_starts_perception(annotated, fault):
    a = annotated
    if fault == "wrong_serial": a.row["cameras"]["front"]["stream"]["serial"] = "wrist"
    elif fault == "missing_side": a.row["boxes"]["side"] = []
    elif fault == "unreviewed": a.row["reviewed"] = False
    elif fault == "no_obstacles": a.row["status"] = "no_obstacles"
    else: a.row["coordinate_frame"] = "ur_base"
    a.path.write_text(json.dumps(a.row))
    with pytest.raises(ValueError):
        module.ManualAnnotationFilter(a.path)


def test_annotation_filter_preserves_acquisition_and_calibration_failures(annotated):
    manual = module.ManualAnnotationFilter(annotated.path)
    with pytest.raises(RuntimeError, match="camera disconnected"):
        manual({"status": "error: camera disconnected"})
    current = snapshot(annotated.grid)
    current["calibration"]["sha256"] = "other"
    with pytest.raises(ValueError, match="calibration differs"):
        manual(current)


def test_annotation_embedded_tsdf_keeps_only_rgb_intersection(annotated):
    pytest.importorskip("real_scripts._voxblox_native")
    a = annotated
    manual = module.ManualAnnotationFilter(a.path)
    manual.prepare_reference(a.grid, a.color)
    pipeline = embedded.SelectedTSDFPipeline(manual, a.cameras)
    depth = np.full((96, 128), 1.05, np.float32)
    for _ in range(3):
        frames = {name: SimpleNamespace(depth_m=depth, host_timestamp_ns=time.monotonic_ns()) for name in a.cameras}
        result = pipeline.update(frames, rendered_views={n: np.zeros_like(depth) for n in frames},
                                 keep_views={n: np.ones(depth.shape, bool) for n in frames},
                                 qpos={"left": [0.]*6, "right": [0.]*6}, grippers={"left": 0., "right": 0.},
                                 calibration={"sha256": "test-digest"})
    observed = ObservationGrid(result["occupancy_map"])
    occupied = observed.centers(np.flatnonzero(observed.state == OCCUPIED))
    assert len(occupied) > 0
    assert manual.scope_mask(observed)[observed.indices(occupied)[0]].all()
    assert result["diagnostics"]["replay_voxblox"]["ellipsoid_scope"] == "dual_rgb_annotation_intersection"


def test_source_uses_annotation_rgb_profile_before_opening_hardware(annotated, monkeypatch):
    pytest.importorskip("real_scripts._voxblox_native")
    from real_scripts import live_dual_ur7e_obstacle_model as live
    monkeypatch.setattr(live, "DualObstacleScene", lambda args: SimpleNamespace(report={"sha256": "test-digest"}))
    manual = module.ManualAnnotationFilter(annotated.path)
    source = embedded.ReplayVoxbloxSource("configs/ur7e_dual_quest3.yaml", manual)
    assert (source.args.width, source.args.height) == (128, 96)
    assert source.thread is None
    source.close()


def test_annotation_workspace_preserves_the_gap_between_transformed_arm_regions():
    transform = np.eye(4)
    transform[0, 3] = 1.
    bounds = (0., .2, 0., .2, 0., .2)
    scene = SimpleNamespace(workspace_regions=[(bounds, np.eye(4)), (bounds, transform)])
    grid = embedded.annotation_reference_grid(scene, .1)
    assert grid.active[grid.point_ids([[.05, .05, .05], [1.05, .05, .05]])].all()
    assert not grid.active[grid.point_ids([[.55, .05, .05]])].any()


@pytest.mark.parametrize("option", [None, "--manual-annotations", "--annotations", "--manual-selection"])
def test_replay_annotation_option_routes_to_filter_and_embedded_provider(annotated, monkeypatch, option):
    a = annotated
    events = []
    class Provider:
        def __init__(self, config, manual, **kwargs):
            assert isinstance(manual, module.ManualAnnotationFilter)
            assert manual.path == a.path
            events.append("annotation_provider")
        def read(self, **kwargs):
            events.append("read")
            return {}
        def close(self): events.append("closed")
    monkeypatch.setattr(embedded, "ReplayVoxbloxSource", Provider)
    monkeypatch.setattr(replay, "DatasetActionPolicy", lambda path: SimpleNamespace(period=.05))
    monkeypatch.setattr(replay, "DualUR7eSurfacePointSampler", lambda **kwargs: SimpleNamespace(calibration_digest="test-digest"))
    monkeypatch.setattr(replay, "validate_obstacle_snapshot", lambda *args, **kwargs: 0.)
    def stop(*args): raise ValueError("stop before robot I/O")
    monkeypatch.setattr(replay, "validate_calibration_identity", stop)
    monkeypatch.setattr(replay, "latest_manual_annotations",
                        lambda: a.path if option is None else pytest.fail("explicit annotation was overridden"))
    flags = ["replay", "--no-execute", "--dataset", str(a.path.parent / "data"),
             "--mode", "avoid", "--output", str(a.path.parent / "run")]
    if option is not None:
        flags.extend([option, str(a.path)])
    args = replay.parser().parse_args(flags)
    with pytest.raises(ValueError, match="stop before robot"):
        replay.run_replay(args)
    assert events == ["annotation_provider", "read", "closed"]


def test_baseline_rejects_annotations_before_dataset_or_hardware(tmp_path):
    args = replay.parser().parse_args(["replay", "--no-execute", "--dataset", str(tmp_path / "data"), "--mode", "baseline",
                                       "--manual-annotations", str(tmp_path / "annotations.json"), "--output", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="only used with --mode avoid"):
        replay.run_replay(args)


def test_paired_v3_all_visible_labels_remain_usable_for_voxel_intersection(annotated):
    a = annotated
    a.row["schema"] = "dual_rgb_paired_obb_v3"
    a.row["visibility"] = {"1": {"front": True, "side": True}}
    a.path.write_text(json.dumps(a.row))
    manual = module.ManualAnnotationFilter(a.path)
    assert len(manual.annotations["front"]) == len(manual.annotations["side"]) == 1


def test_single_view_labels_do_not_silently_drop_objects_from_voxel_intersection(annotated):
    a = annotated
    a.row["schema"] = "dual_rgb_paired_obb_v3"
    a.row["visibility"] = {"1": {"front": True, "side": False}}
    a.row["boxes"]["side"] = []
    a.path.write_text(json.dumps(a.row))
    with pytest.raises(ValueError, match="every object visible in both views"):
        module.ManualAnnotationFilter(a.path)


def test_transparent_replay_tracks_occupancy_changes_without_stale_support(annotated):
    from obstacle_validation.runtime import boxes_from_json
    from obstacle_validation.ellipsoid_geometry import fit_obstacles
    a = annotated
    manual = module.ManualAnnotationFilter(a.path)
    for index, hits in enumerate(([], [[-.05, .05, 1.05]], [])):
        a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=np.asarray(hits).reshape(-1,3),
                      free_ids=np.empty(0, np.int64))
        original = snapshot(a.grid)
        before = json.dumps(original)
        result = manual(original)
        diag = result['diagnostics']['paired_rgb_obstacles']
        assert diag['transparent_count'] == (0 if hits else 1)
        assert diag['objects']['1']['occupied_voxels'] == len(hits)
        assert result['occupancy_map']['occupied_voxels'] == len(hits)
        assert len(result['obbs']) == 1 and json.dumps(original) == before
        boxes = boxes_from_json(result)
        models = fit_obstacles(boxes)
        assert len(models) == 1
        if not hits:
            assert models[0].key == 'transparent_1'
            np.testing.assert_allclose(models[0].shape, result['obbs'][0]['ellipsoid_shape'])


def test_thin_transparent_annotation_can_prepare_an_embedded_tsdf_roi(annotated):
    a = annotated
    for view in a.row['boxes']:
        a.row['boxes'][view][0].update(width=2., height=2.)
    a.path.write_text(json.dumps(a.row))
    manual = module.ManualAnnotationFilter(a.path)
    assert not manual.scope_mask(a.grid).any()  # genuinely smaller than the lattice
    manual.prepare_reference(a.grid, a.color)
    assert manual.cell_scope.ids.size > 0
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=np.empty((0,3)), free_ids=np.empty(0,np.int64))
    result = manual(snapshot(a.grid))
    assert result['diagnostics']['paired_rgb_obstacles']['transparent_objects'] == ['1']
    assert result['obbs'][0]['ellipsoid_shape']


def test_transparent_replay_has_an_effective_obstacle_qp_constraint(annotated):
    from obstacle_validation.runtime import boxes_from_json
    from obstacle_validation.ellipsoid_geometry import Ellipsoid
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
    a = annotated
    a.grid.update(timestamp_ns=time.monotonic_ns(), raw_hits=np.empty((0,3)), free_ids=np.empty(0,np.int64))
    result = module.ManualAnnotationFilter(a.path)(snapshot(a.grid))
    shape = np.asarray(result['obbs'][0]['ellipsoid_shape'])
    center = np.asarray(result['obbs'][0]['center'])
    # Start a 1 cm robot sphere 2 mm beyond the 5 mm obstacle margin.
    surface = center - shape[:,0] / np.sqrt(shape[0,0])
    robot_center = surface - [.017, 0., 0.]
    class Volumes:
        left_link_names = ('left',)
        def ellipsoids(self, q, grip):
            return [Ellipsoid(np.array([q[0], robot_center[1], robot_center[2]]), np.eye(3)*.01**2, 'left',0,0),
                    Ellipsoid(np.array([.8, .8, 0.]), np.eye(3)*.01**2, 'right',1,0)]
    planner = EllipsoidMCBF(Volumes(), margin=.005, inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0] = robot_center[0]
    action = np.r_[q, np.zeros(2)][None]; action[0,0] += .02
    _, info = planner.plan(q, np.zeros(2), action, boxes_from_json(result),
                           lower=np.full(12,-2.), upper=np.full(12,2.), max_step=.04)
    assert info['steps'][0]['active_constraint_counts']['left_obstacle'] > 0
    assert info['waypoints'][0][0] < action[0,0] - .001
    assert len(planner.obstacles) == 1 and planner.obstacles[0].key == 'transparent_1'
