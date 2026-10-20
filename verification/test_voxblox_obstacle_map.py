"""Native Voxblox TSDF fusion, ray filtering, snapshot and export regressions."""
import json
import sys
import time

import numpy as np
import pytest

from real_scripts import live_dual_ur7e_obstacle_model as live
from real_scripts.probabilistic_obstacle_map import integrate_depth_views
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_cbf_qp import boxes_from_snapshot
from real_scripts.voxblox_obstacle_map import VoxbloxOccupancyGrid
from real_scripts.voxblox_obstacle_map import require_voxblox
from safety_module.occupancy import OCCUPIED, ObservationGrid
from safety_module.runtime_contract import validate_snapshot


@pytest.fixture
def native():
    return pytest.importorskip("real_scripts._voxblox_native")


@pytest.fixture
def grid(native):
    return VoxbloxOccupancyGrid([-.5,-.5,0.], np.eye(3), (20,20,30), .05)


def plane(grid, depth=1., *, keep=True, rendered=0., hits=(), completed=(), uncertain=(), camera=None):
    camera = camera or CameraCalibration("front", np.array([[20.,0,4.],[0,20.,4.],[0,0,1.]]), np.eye(4))
    image=np.full((9,9), depth)
    integrate_depth_views(grid, [(camera,image,np.full((9,9),keep),np.full((9,9),rendered))],
                          hits, completed, uncertain_hits=uncertain, timestamp_ns=grid.timestamp_ns+1_000_000,
                          ray_stride=1)


def test_frozen_view_is_lazy_cached_and_capture_aligned(grid, monkeypatch):
    plane(grid)
    expected = grid.viewer_payload(max_voxels=40)
    frozen = grid.freeze_view(max_voxels=40)
    snapshot = {"frame": grid.frame, "_voxblox_view": frozen}
    original = VoxbloxOccupancyGrid.viewer_payload
    calls = []
    def build(view, **kwargs):
        calls.append(view.frame)
        return original(view, **kwargs)
    monkeypatch.setattr(VoxbloxOccupancyGrid, "viewer_payload", build)
    assert "voxblox_view" not in json.loads(live.snapshot_payload(snapshot, compact=True))
    assert not calls
    plane(grid, depth=1.3)
    result = live.materialize_snapshot(snapshot, viewer=True)
    assert result["voxblox_view"] == expected
    assert frozen.materialize() is result["voxblox_view"]
    assert calls == [1]


def test_cli_defaults_and_explicit_lingbot():
    assert not live.parse_args([]).lingbot_depth
    assert not live.parse_args(["--scene","single-pika"]).lingbot_depth
    assert live.parse_args(["--lingbot-depth"]).lingbot_depth
    assert live.parse_args(["--occupancy-backend","voxblox"]).occupancy_backend=="voxblox"
    for flags in (["--no-occupancy-map"], ["--voxblox-truncation-m","nan"],
                  ["--voxblox-surface-band-m","-.1"], ["--voxblox-min-weight","40"],
                  ["--voxblox-output","field.bt"], ["--voxblox-view-max-voxels","1"]):
        with pytest.raises(SystemExit):
            live.parse_args(["--occupancy-backend","voxblox",*flags])
    with pytest.raises(SystemExit):
        live.parse_args(["--voxblox-output","field.npz"])
    assert live.parse_args(["--occupancy-backend","voxblox"]).voxblox_esdf
    assert not live.parse_args(["--occupancy-backend","voxblox","--no-voxblox-esdf"]).voxblox_esdf
    for option,value in (("--voxblox-esdf-max-distance-m","nan"),
                         ("--voxblox-esdf-max-distance-m",".02"),
                         ("--voxblox-esdf-min-distance-m","0"),
                         ("--voxblox-esdf-min-distance-m",".5")):
        with pytest.raises(SystemExit):
            live.parse_args(["--occupancy-backend","voxblox",option,value])


def test_missing_native_dependency_fails_before_hardware(monkeypatch):
    monkeypatch.setitem(sys.modules,"real_scripts._voxblox_native",None)
    monkeypatch.setattr(live,"DualObstacleScene",lambda _a:pytest.fail("hardware before dependency check"))
    with pytest.raises(RuntimeError,match="scripts/build_voxblox.py"):
        live.main(["--no-robot","--occupancy-backend","voxblox","--once"])


def test_native_axis_aligned_ray_has_positive_and_negative_distance(native):
    volume=native.TsdfVolume(.05,16,.2,32.,10.,1)
    volume.integrate(np.zeros(3),np.array([[0.,0.,1.]]))
    samples=volume.observed()
    assert len(samples)>20  # Regression: upstream's zero-axis guards yielded only the start voxel.
    assert samples[:,3].min()<0 and samples[:,3].max()>0
    values=volume.query(np.array([[.025,.025,.975],[.025,.025,1.025]]))
    np.testing.assert_allclose(values[:,0],[.025,-.025],atol=1e-5)
    assert (values[:,1]>0).all()


def test_native_input_validation_precedes_upstream_checks(native):
    with pytest.raises(ValueError):
        native.TsdfVolume(0,16,.2,32.,10.,1)
    volume=native.TsdfVolume(.05,16,.2,32.,10.,1)
    with pytest.raises(ValueError):
        volume.integrate(np.zeros(3),np.array([[np.nan,0,1.]]))
    with pytest.raises(ValueError):
        volume.crop(np.ones(3),np.zeros(3),2)
    for settings in ((float('nan'),.05,.1),(2.,0.,.1),(.05,.05,.1),(2.,.05,0.)):
        with pytest.raises(ValueError):
            volume.configure_esdf(*settings)
    with pytest.raises(RuntimeError,match="not configured"):
        volume.update_esdf()


def test_outdated_native_dependency_requests_rebuild_before_hardware(native,monkeypatch):
    monkeypatch.setattr(native,"api_version",1)
    monkeypatch.setattr(live,"DualObstacleScene",lambda _a:pytest.fail("hardware before dependency check"))
    with pytest.raises(RuntimeError,match="outdated.*rebuild"):
        live.main(["--no-robot","--occupancy-backend","voxblox","--once"])


def test_plane_fuses_native_distance_and_weight_into_surface_cells(grid):
    plane(grid,hits=[[.025,.025,1.]])
    ids=grid.point_ids([[.025,.025,.975],[.025,.025,1.025]])
    np.testing.assert_allclose(grid.distance_m[ids],[.025,-.025],atol=.01)
    assert np.all(grid.weight[ids]>=grid.min_weight)
    assert np.all(grid.states()[ids]==OCCUPIED)
    assert grid.diagnostics["voxblox_integrated_rays"]==81
    assert grid.diagnostics["tsdf_surface_voxels"]>0
    assert len(grid.current_hits)>0


@pytest.mark.parametrize("depth,keep,rendered",[(0.,True,0.),(np.nan,True,0.),(1.,False,0.),(1.,True,.7),(3.,True,0.)])
def test_invalid_occluded_and_over_range_depth_never_clears_history(grid,depth,keep,rendered):
    plane(grid)
    before_distance, before_weight=grid.distance_m.copy(),grid.weight.copy()
    plane(grid,depth,keep=keep,rendered=rendered)
    assert grid.diagnostics["voxblox_integrated_rays"]==0
    np.testing.assert_allclose(grid.distance_m,before_distance,equal_nan=True)
    np.testing.assert_array_equal(grid.weight,before_weight)


def test_uncertain_endpoint_preserves_native_values_against_another_views_clearing(grid):
    plane(grid)
    point=np.array([[.025,.025,.975]])
    voxel=grid.point_ids(point)[0]
    before=grid.distance_m[voxel],grid.weight[voxel]
    plane(grid,1.4,uncertain=point)
    np.testing.assert_allclose([grid.distance_m[voxel],grid.weight[voxel]],before)
    assert grid.sources[voxel]==8


def test_completion_supplies_current_coverage_without_inventing_native_rays(grid):
    integrate_depth_views(grid,[],[],[[.025,.025,1.]],timestamp_ns=1)
    assert not grid.weight.any() and grid.volume.blocks==0
    assert len(grid.current_hits)==1
    assert grid.viewer_payload()["surface"]["shown"]==0
    assert grid.snapshot()["log_odds_semantics"]=="compatibility_state_only"


def test_rotated_world_frame_and_viewer_sampling(native):
    axes=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    grid=VoxbloxOccupancyGrid([1.5,1.5,0],axes,(20,20,30),.05)
    pose=np.eye(4);pose[:3,3]=[1.,2.,0.]
    camera=CameraCalibration("front",np.array([[20.,0,4],[0,20.,4],[0,0,1.]]),pose)
    plane(grid,camera=camera)
    view=grid.viewer_payload(max_voxels=6)
    assert sum(view[name]["shown"] for name in ("surface","field"))<=6
    assert view["surface"]["sampled"] and view["field"]["sampled"]
    for name in ("surface","field"):
        world=np.asarray(view[name]["centers"])
        ids,inside=grid.indices(world)
        assert inside.all()
        np.testing.assert_allclose(grid.distance_m[ids],view[name]["distances_m"])
        assert len(view[name]["weights"])==view[name]["shown"]
    assert view==grid.viewer_payload(max_voxels=6)


def test_snapshot_export_and_capture_aligned_consumers(grid,tmp_path):
    plane(grid)
    payload=grid.snapshot()
    observed=ObservationGrid(payload)
    assert (observed.state==OCCUPIED).sum()==len(grid.current_hits)
    grid.export(tmp_path/"field.npz",coordinate_frame="ur_base")
    with np.load(tmp_path/"field.npz") as saved:
        np.testing.assert_allclose(saved["distance_m"].ravel(),grid.distance_m,equal_nan=True)
        np.testing.assert_array_equal(saved["weight"].ravel(),grid.weight)
        np.testing.assert_allclose(saved["esdf_distance_m"].ravel(),grid.esdf_distance_m,equal_nan=True)
        np.testing.assert_array_equal(saved["esdf_observed"].ravel(),grid.esdf_observed)
        assert json.loads(str(saved["metadata"]))["esdf"]["full_euclidean_distance"]
        assert json.loads(str(saved["metadata"]))["coordinate_frame"]=="ur_base"
    import base64
    import zlib
    esdf_payload=payload["voxblox"]["esdf"]
    decoded=np.frombuffer(zlib.decompress(base64.b64decode(esdf_payload["distance_m_data"])),dtype='<f4')
    np.testing.assert_allclose(decoded,grid.esdf_distance_m,equal_nan=True)
    mask=np.frombuffer(zlib.decompress(base64.b64decode(esdf_payload["observed_data"])),dtype='u1')
    np.testing.assert_array_equal(mask,grid.esdf_observed)
    snapshot={"schema":"dual_ur7e_obstacle_snapshot_v2","coordinate_frame":"left_base",
              "obstacle_backend":"voxblox","status":"ok","host_timestamp_ns":grid.timestamp_ns,
              "occupancy_map":payload,"obbs":[],"_voxblox_view":grid.viewer_payload()}
    assert boxes_from_snapshot(snapshot).observation is not None
    with pytest.raises(ValueError,match="capture-aligned"):
        validate_snapshot({**snapshot,"host_timestamp_ns":time.monotonic_ns()},1.)
    with pytest.raises(ValueError,match="missing"):
        boxes_from_snapshot({k:v for k,v in snapshot.items() if k!="occupancy_map"})
    rendered=json.loads(live.snapshot_payload(snapshot,viewer=True,render_only=True))
    assert rendered["voxblox_view"]["capture_timestamp_ns"]==grid.timestamp_ns
    assert "voxblox_view" not in json.loads(live.snapshot_payload(snapshot,compact=True))


def test_esdf_plane_extends_beyond_tsdf_truncation_and_queries_gradient(grid):
    plane(grid)
    points=np.array([[.025,.025,.125],[.025,.025,.525],[.025,.025,1.075]])
    ids=grid.point_ids(points)
    np.testing.assert_allclose(grid.distance_m[ids[:2]],.2,atol=1e-6)
    query=grid.query_esdf(points,interpolate=False)
    assert query["observed"].all()
    np.testing.assert_allclose(query["distance_m"],[.875,.475,-.075],atol=.01)
    query=grid.query_esdf([[0.,0.,.75],[0.,0.,.9]])
    np.testing.assert_allclose(query["distance_m"],[.25,.1],atol=.01)
    np.testing.assert_allclose(query["gradient"],[[0.,0.,-1.]]*2,atol=.03)
    assert query["gradient_valid"].all()
    assert not query["saturated"].any()
    view=grid.viewer_payload(max_voxels=9)
    assert view["esdf_enabled"] and view["esdf"]["field"]["count"]>0
    assert sum(view["esdf"][name]["shown"] for name in ("surface","field"))<=9
    for name in ("surface","field"):
        group=view["esdf"][name]
        sampled=grid.query_esdf(np.asarray(group["centers"]),interpolate=False)
        np.testing.assert_allclose(sampled["distance_m"],group["distances_m"])
    json.dumps(view,allow_nan=False)


def test_native_esdf_diagonal_distance_is_full_euclidean_and_rebuild_clears_deleted_blocks(grid):
    plane(grid)
    volume=grid.volume
    samples=volume.observed()
    source=np.array([.525,.525,.925])
    target=np.array([.575,.575,.775])
    source_row=np.argmin(np.linalg.norm(samples[:,:3]-source,axis=1))
    assert np.linalg.norm(samples[source_row,:3]-source)<1e-5
    values=np.tile([.2,1.],(len(samples),1));values[source_row,0]=.025
    volume.restore(samples[:,:3],values)
    volume.update_esdf()
    esdf=volume.esdf_observed()
    row=np.argmin(np.linalg.norm(esdf[:,:3]-target,axis=1))
    assert np.linalg.norm(esdf[row,:3]-target)<1e-5
    np.testing.assert_allclose(esdf[row,3],.025+np.linalg.norm(target-source),atol=1e-5)
    volume.crop(np.array([10.,10.,10.]),np.array([11.,11.,11.]),2048)
    volume.update_esdf()
    assert volume.blocks==0 and volume.esdf_observed().shape==(0,4)


def test_esdf_unknown_inactive_boundary_and_input_validation(grid):
    plane(grid)
    unknown=grid.query_esdf([[.45,.45,.7],[2.,0.,.7]])
    assert not unknown["observed"].any()
    assert np.isnan(unknown["distance_m"]).all() and np.isnan(unknown["gradient"]).all()
    assert not unknown["gradient_valid"].any()
    grid.active[grid.point_ids([[.025,.025,.725]])]=False
    assert not grid.query_esdf([[0.,0.,.75]])["observed"].any()
    assert grid.query_esdf(np.empty((0,3)))["gradient"].shape==(0,3)
    for points in ([0.,0.,0.],[[np.inf,0.,0.]],[[np.nan,0.,0.]]):
        with pytest.raises(ValueError,match="finite world points"):
            grid.query_esdf(points)


def test_esdf_distance_cap_never_becomes_a_valid_gradient(native):
    grid=VoxbloxOccupancyGrid([-.5,-.5,0.],np.eye(3),(20,20,30),.05,esdf_max_distance_m=.4)
    plane(grid)
    query=grid.query_esdf([[0.,0.,.3]])
    np.testing.assert_allclose(query["distance_m"],.4)
    assert query["observed"].all() and query["saturated"].all()
    assert not query["gradient_valid"].any() and np.isnan(query["gradient"]).all()


def test_esdf_gradient_rotates_into_world_frame(native):
    axes=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    origin=np.array([1.,2.,3.])
    grid=VoxbloxOccupancyGrid(origin,axes,(20,20,30),.05)
    pose=np.eye(4);pose[:3,:3]=axes;pose[:3,3]=origin+np.array([.5,.5,0.])@axes.T
    camera=CameraCalibration("front",np.array([[20.,0,4],[0,20.,4],[0,0,1.]]),pose)
    plane(grid,camera=camera)
    query=grid.query_esdf(origin+np.array([[.5,.5,.75]])@axes.T)
    assert query["gradient_valid"].all()
    np.testing.assert_allclose(query["distance_m"],[.25],atol=.01)
    np.testing.assert_allclose(query["gradient"],np.array([[0.,0.,-1.]])@axes.T,atol=.03)


def test_esdf_empty_completion_only_and_disabled_states(native):
    for enabled in (True,False):
        grid=VoxbloxOccupancyGrid([-.5,-.5,0.],np.eye(3),(20,20,30),.05,esdf=enabled)
        integrate_depth_views(grid,[],[],[[.025,.025,1.]],timestamp_ns=1)
        assert not grid.esdf_observed.any() and np.isnan(grid.esdf_distance_m).all()
        assert bool(grid.viewer_payload().get("esdf"))==enabled
        if enabled:
            assert not grid.query_esdf([[0.,0.,.75]])["observed"].any()
        else:
            with pytest.raises(ValueError,match="disabled"):
                grid.query_esdf([[0.,0.,.75]])


def test_observed_negative_tsdf_interior_blocks_even_beyond_surface_band(grid):
    plane(grid)
    point = np.array([[.025, .025, 1.075]])
    ids = grid.point_ids(point)
    assert grid.distance_m[ids[0]] < -grid.surface_band_m
    assert grid.states()[ids[0]] == OCCUPIED
    assert grid.diagnostics['tsdf_interior_voxels'] > 0
    observation = ObservationGrid(grid.snapshot())
    assert not observation.segments_observed(point, point, now_ns=grid.timestamp_ns)


@pytest.mark.parametrize('maximum', [.4, 2.])
def test_snapshot_esdf_queries_match_native_grid_and_decode_once(native, maximum):
    grid = VoxbloxOccupancyGrid([-.5, -.5, 0.], np.eye(3), (20, 20, 30), .05,
                               esdf_max_distance_m=maximum)
    plane(grid)
    observation = ObservationGrid(grid.snapshot())
    assert observation._esdf_arrays is None
    points = [[0., 0., .3], [0., 0., .9], [0., 0., 1.075], [2., 0., .7]]
    for interpolate in (True, False):
        expected = grid.query_esdf(points, interpolate=interpolate)
        actual = observation.query_esdf(points, interpolate=interpolate)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)
    cached = observation._esdf_arrays
    observation.query_esdf(points)
    assert observation._esdf_arrays is cached


def test_snapshot_rejects_corrupt_observed_esdf_distance(grid):
    from safety_module.occupancy import _pack
    plane(grid)
    payload = grid.snapshot()
    distances = grid.esdf_distance_m.copy()
    distances[grid.esdf_observed] = np.nan
    payload['voxblox']['esdf']['distance_m_data'] = _pack(distances.astype('<f4'))
    with pytest.raises(ValueError, match='ESDF observations'):
        ObservationGrid(payload).query_esdf([[0., 0., .9]])


def test_mcbf_goal_uses_captured_voxblox_esdf_normal_and_keeps_occupied_check(grid):
    from obstacle_validation.mcbf import evacuate_action_target
    plane(grid)
    snapshot = {'coordinate_frame': 'left_base', 'obstacle_backend': 'voxblox',
                'occupancy_map': grid.snapshot(), 'host_timestamp_ns': grid.timestamp_ns,
                'obbs': [{'center': box.center.tolist(), 'axes': box.rotation.tolist(),
                          'half_sizes': (box.extents/2).tolist()} for box in grid.boxes()]}
    boxes = boxes_from_snapshot(snapshot)
    class Sampler:
        def link_points(self, q, grip):
            return np.array([[q[:3]], [[3., 3., 3.]]])
    action = np.zeros(14); action[:3] = [.025, .025, 1.075]
    target, info = evacuate_action_target(Sampler(), action, boxes, lower=np.full(12, -2.),
                                         upper=np.full(12, 2.), margin=.01)
    assert info['success'], info
    assert info['normal_updates'][0]['normal_source'] == 'voxblox_esdf'
    assert target[2] < 1.
    assert boxes.observation.segments_observed(target[None, :3], target[None, :3],
                                               margin_m=.01, now_ns=grid.timestamp_ns)


def test_mcbf_step_uses_native_esdf_tangent_guidance_with_voxel_sweep(grid):
    from obstacle_validation.cbf import RecordedPathCBF
    from obstacle_validation.core import CollisionChecker
    plane(grid)
    snapshot = {'coordinate_frame': 'left_base', 'obstacle_backend': 'voxblox',
                'occupancy_map': grid.snapshot(), 'host_timestamp_ns': grid.timestamp_ns,
                'obbs': [{'center': box.center.tolist(), 'axes': box.rotation.tolist(),
                          'half_sizes': (box.extents/2).tolist()} for box in grid.boxes()]}
    boxes = boxes_from_snapshot(snapshot)
    class Sampler:
        left_link_names, points_per_link = ('left',), 1
        def link_points(self, q, grip):
            return np.array([[q[:3]], [[3., 3., 3.]]])
    sampler, grip = Sampler(), np.zeros(2)
    q = np.zeros(12); q[:3] = [.025, .025, .9]
    nominal = np.zeros(12); nominal[2] = .025
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01, inter_arm_margin_m=.035,
                          trigger_margin_m=.025, alpha=1., iterations=64, modulation=True)
    delta, info = cbf.project(q, nominal, checker, lower=np.full(12, -2.), upper=np.full(12, 2.),
                              max_step=.025, previous_delta=np.zeros(12), acceleration_step=.05)
    assert info['success'] and checker.edge_safe(q, q+delta), info
    assert info['modulation']['normal_source'] == 'voxblox_esdf'
    modulation = info['modulation']
    assert np.dot(modulation['joint_direction'], delta) >= modulation['minimum_joint_progress_rad']-1e-7
    assert max(abs(delta[0]), abs(delta[1])) > .001
