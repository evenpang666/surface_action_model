"""Shared-camera, paired-voxel and feedback-gated OA-VLA execution contracts."""

import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation.oa_vla_actions import (ObservationHold, validate_server_metadata, first_action_target,
                                               validate_age_and_drift, check_optimized_step)
from obstacle_validation.oa_vla_perception import (DynamicOBBFilter, MappedObservation,
                                                  inconsistent_pairs_allow_empty_geometry)
from obstacle_validation.oa_vla_runtime import LiveRuntime, PredictionTicket, parse_args
from obstacle_validation.shared_frames import (LatestValue, Workers, Feedback, FrameBundle, freeze_frame, camera_loop)
from real_scripts.real_robot_adapter import RGBDFrame, CameraCalibration
from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, OCCUPIED

VIEWS=('front','side','right_wrist','left_wrist')


def metadata():
    return dict(model_type='oa-vla',model_format=2,state_dim=12,action_dim=14,policy_views=list(VIEWS),
                detection_views=['front','side'],detection_box_format='cxcywh_angle_deg_original_pixels',
                pair_id_scope='current_prediction',detection_trained=True,
                action_semantics='interleaved_joint_delta_gripper_target',action_source='commanded',action_period_s=.05)


def mapped_packet(*, point=(-.05,.05,1.05), q=None, frame_id=7, extra_point=True, empty=False):
    stamp=time.monotonic_ns()
    q=np.zeros(12) if q is None else np.asarray(q,float)
    state=Feedback(q,np.zeros(2),stamp,{})
    frames={view:freeze_frame(RGBDFrame(view,np.zeros((96,128,3),np.uint8),np.ones((96,128),np.float32),
                                       host_timestamp_ns=stamp,frame_number=frame_id)) for view in VIEWS}
    bundle=FrameBundle(frame_id,frames,state,'digest')
    grid=LocalOccupancyGrid([-.4,-.2,.9],np.eye(3),[12,4,4],.1)
    hits=[point]+([[.15,.05,1.05]] if extra_point else [])
    grid.update(timestamp_ns=stamp,raw_hits=np.empty((0,3)) if empty else hits,free_ids=np.empty(0,np.int64))
    snapshot=dict(schema='dual_ur7e_obstacle_snapshot_v2',coordinate_frame='left_base',status='ok',
                  host_timestamp_ns=stamp,vla_frame_id=frame_id,obstacle_backend='voxblox',
                  calibration={'sha256':'digest'},occupancy_map=grid.snapshot(),unknown_obstacles=[],
                  obbs=[],scene_observation_state='observed',diagnostics={})
    snapshot['occupancy_map']['backend']='voxblox'
    K=np.array([[100.,0.,64.],[0.,100.,48.],[0.,0.,1.]])
    side=np.eye(4);side[0,3]=.2
    cameras={'front':CameraCalibration('front',K,np.eye(4)), 'side':CameraCalibration('side',K,side)}
    return MappedObservation(bundle,snapshot,cameras),grid


def prediction(*, front=(64,48,40,40,0),side=(44,48,10,20,0)):
    return dict(model_type='oa-vla',detection_status='ok',pair_id_scope='current_prediction',
                detection_box_format='cxcywh_angle_deg_original_pixels',actions=np.zeros((2,14)).tolist(),
                objects=[dict(pair_id=0,score=.95,front={'obb':list(front),'visibility_score':.9},
                              side=None if side is None else {'obb':list(side),'visibility_score':.9})])


def test_camera_storage_is_copied_once_and_shared_read_only():
    rgb=np.zeros((4,6,3),np.uint8);depth=np.ones((4,6),np.float32)
    original=RGBDFrame('front',rgb,depth,host_timestamp_ns=time.monotonic_ns())
    frozen=freeze_frame(original)
    rgb[:]=255;depth[:]=8
    assert not frozen.rgb.any() and np.all(frozen.depth_m==1)
    with pytest.raises(ValueError):frozen.rgb[0]=1
    with pytest.raises(ValueError):frozen.depth_m[0]=1


def test_latest_channel_has_no_backlog_and_failure_beats_cached_frame():
    channel=LatestValue();channel.publish('old');channel.publish('new')
    version,value=channel.get();assert version==2 and value=='new'
    with pytest.raises(TimeoutError):channel.get(after=version,timeout=.001)
    channel.fail(ValueError('camera failed'))
    with pytest.raises(RuntimeError,match='camera failed'):channel.get()


def test_camera_worker_is_only_owner_and_stops_after_failure():
    workers=Workers();frames=workers.channel();feedback=workers.channel()
    feedback.publish(Feedback(np.zeros(12),np.zeros(2),time.monotonic_ns(),{}))
    class Source:
        starts=stops=reads=0
        def start(self):self.starts+=1
        def stop(self):self.stops+=1
        def read(self):
            self.reads+=1
            stamp=time.monotonic_ns()
            if self.reads>1:raise RuntimeError('unplugged')
            return {v:RGBDFrame(v,np.zeros((4,6,3),np.uint8),np.ones((4,6),np.float32),host_timestamp_ns=stamp) for v in VIEWS}
    source=Source()
    with pytest.raises(RuntimeError,match='unplugged'):
        camera_loop(workers,frames,feedback,source,calibration_id='digest',warmup_frames=0)
    assert source.starts==source.stops==1
    assert frames.get()[1].frame_id==0


def test_paired_filter_intersects_same_objects_not_cross_view_union():
    mapped,grid=mapped_packet()
    result=DynamicOBBFilter(mapped,prediction(),padding_pixels=0)(mapped.snapshot)
    selected=ObservationGrid(result['occupancy_map'])
    good,bad=grid.point_ids([[-.05,.05,1.05],[.15,.05,1.05]])
    assert selected.state[good]==OCCUPIED and selected.state[bad]!=OCCUPIED
    assert result['diagnostics']['vla_pair_voxels']=={'0':1}
    assert len(result['obbs'])==1
    # The source snapshot is not modified by the dynamic scope.
    assert mapped.snapshot['occupancy_map']['occupied_voxels']==2


def test_dynamic_scope_moves_to_new_object_outside_previous_crop():
    first,grid=mapped_packet()
    first_result=DynamicOBBFilter(first,prediction(),padding_pixels=0)(first.snapshot)
    # New pair locates the other cell; using the old fixed crop would miss it.
    second_result=DynamicOBBFilter(first,prediction(front=(78.3,52.8,8,8,0),side=(59.2,52.8,8,8,0)),padding_pixels=0)(first.snapshot)
    ids=grid.point_ids([[-.05,.05,1.05],[.15,.05,1.05]])
    a,b=ObservationGrid(first_result['occupancy_map']),ObservationGrid(second_result['occupancy_map'])
    assert a.state[ids[0]]==OCCUPIED and b.state[ids[1]]==OCCUPIED
    assert b.state[ids[0]]!=OCCUPIED


def test_missing_view_empty_detection_and_wrong_frame_never_become_free_scene():
    mapped,_=mapped_packet()
    with pytest.raises(ObservationHold,match='invisible'):
        DynamicOBBFilter(mapped,prediction(side=None))
    absent=prediction();absent['objects']=[]
    with pytest.raises(ObservationHold,match='empty scene'):
        DynamicOBBFilter(mapped,absent)
    wrong=copy.deepcopy(mapped.snapshot);wrong['vla_frame_id']+=1
    with pytest.raises(ValueError,match='different frames'):
        DynamicOBBFilter(mapped,prediction())(wrong)


def test_inconsistent_pair_does_not_discard_other_objects_or_measured_voxels():
    mapped,_=mapped_packet()
    pred=prediction()
    invalid=prediction(front=(10,10,2,2,0),side=(10,10,2,2,0))['objects'][0]
    invalid['pair_id']=1;pred['objects'].append(invalid)
    result = DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    diag = result['diagnostics']['paired_rgb_obstacles']
    assert diag['ignored_objects'] == ['1'] and diag['transparent_count'] == 0
    assert diag['objects']['0']['occupied_voxels'] == 1
    assert diag['objects']['1']['reason'] == 'no_3d_intersection'
    assert diag['measured_occupied_voxels'] == 2
    assert result['occupancy_map'] == mapped.snapshot['occupancy_map']
    assert sum(box['voxel_count'] for box in result['obbs']) == 2


@pytest.mark.parametrize('empty', [False,True])
def test_nonintersecting_pair_is_ignored_with_measured_map_preserved(empty):
    mapped,grid = mapped_packet(empty=empty)
    before = copy.deepcopy(mapped.snapshot)
    pred = prediction(front=(10,10,2,2,0),side=(10,10,2,2,0))
    result = DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    diag = result['diagnostics']['paired_rgb_obstacles']
    assert diag['ignored_objects'] == ['0'] and diag['transparent_count'] == 0
    assert diag['measured_voxel_scope'] == 'full_workspace_on_inconsistent_pairs'
    assert result['occupancy_map'] == before['occupancy_map']
    assert mapped.snapshot == before
    assert inconsistent_pairs_allow_empty_geometry(result,pred) is empty
    if empty:
        assert result['obbs'] == []
    else:
        assert sum(box['voxel_count'] for box in result['obbs']) == 2


def test_ignored_pair_keeps_valid_transparent_obstacles():
    mapped,_ = mapped_packet(empty=True)
    pred = prediction()
    invalid = prediction(front=(10,10,2,2,0),side=(10,10,2,2,0))['objects'][0]
    invalid['pair_id'] = 1; pred['objects'].append(invalid)
    result = DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    diag = result['diagnostics']['paired_rgb_obstacles']
    assert diag['ignored_objects'] == ['1'] and diag['transparent_objects'] == ['0']
    assert len(result['obbs']) == 1 and result['obbs'][0]['object_id'] == '0'
    assert not inconsistent_pairs_allow_empty_geometry(result,pred)


def test_malformed_geometry_errors_are_not_treated_as_inconsistent_detection(monkeypatch):
    from obstacle_validation import manual_annotation_filter
    from obstacle_validation.paired_rgb_geometry import PairedRegionError
    mapped,_ = mapped_packet(empty=True)
    def fail(*args, **kwargs):
        raise PairedRegionError('invalid geometry')
    monkeypatch.setattr(manual_annotation_filter,'transparent_obstacle',fail)
    with pytest.raises(ObservationHold,match='invalid geometry'):
        DynamicOBBFilter(mapped,prediction(),padding_pixels=0)(mapped.snapshot)


def test_empty_paired_volume_becomes_transparent_without_faking_voxels():
    mapped, grid = mapped_packet()
    # This second, thin object is supported by both RGB views but no occupied
    # centre; the first object's measured support must not hide its absence.
    pred = prediction()
    other = prediction(front=(83.,48.,2.,4.,25.), side=(64.,48.,2.,4.,25.))['objects'][0]
    other['pair_id'] = 1
    pred['objects'].append(other)
    before = copy.deepcopy(mapped.snapshot)
    result = DynamicOBBFilter(mapped, pred, padding_pixels=0)(mapped.snapshot)
    assert result['diagnostics']['vla_pair_voxels'] == {'0':1, '1':0}
    assert result['diagnostics']['paired_rgb_obstacles']['transparent_objects'] == ['1']
    transparent = [box for box in result['obbs'] if box.get('ellipsoid_shape') is not None]
    assert len(transparent) == 1 and transparent[0]['object_id'] == '1'
    assert transparent[0]['voxel_count'] == 0 and not transparent[0]['depth_measured']
    assert result['occupancy_map']['occupied_voxels'] == 1
    assert mapped.snapshot == before
    from obstacle_validation.runtime import boxes_from_json
    from obstacle_validation.ellipsoid_geometry import fit_obstacles
    models = fit_obstacles(boxes_from_json(result))
    assert len(models) == 2 and sum(model.key == 'transparent_1' for model in models) == 1


def test_entirely_empty_map_still_has_a_transparent_obstacle():
    mapped, grid = mapped_packet(empty=True)
    result = DynamicOBBFilter(mapped, prediction(), padding_pixels=0)(mapped.snapshot)
    assert result['occupancy_map']['occupied_voxels'] == 0
    assert result['diagnostics']['vla_pair_voxels'] == {'0':0}
    assert len(result['obbs']) == 1
    assert result['obbs'][0]['geometry_source'] == 'dual_rgb_transparent_region'


def test_delta_layout_reorders_joints_grippers_and_anchors_observation():
    actions=np.zeros((2,14));actions[0,:6]=.1;actions[0,6]=.7;actions[0,7:13]=-.2;actions[0,13]=.3
    actions[1]=100  # second action must not be integrated or executed
    q=np.arange(12)*.01
    result=first_action_target({'actions':actions},q,metadata()['action_semantics'],lower=np.full(12,-2.),upper=np.full(12,2.))
    np.testing.assert_allclose(result[:12],q+np.r_[np.full(6,.1),np.full(6,-.2)])
    np.testing.assert_allclose(result[12:],[.7,.3])


@pytest.mark.parametrize('field,value',[('detection_trained',False),('test_only',True),('action_semantics',None),('action_source',None),('action_period_s',None)])
def test_hardware_admission_requires_trained_explicit_contract(field,value):
    info=metadata();info[field]=value
    with pytest.raises(ValueError):validate_server_metadata(info)


def test_expired_prediction_and_state_drift_are_rejected():
    mapped,_=mapped_packet();bundle=mapped.bundle
    kwargs=dict(max_age_s=.5,max_drift_rad=.01,max_feedback_age_s=.2)
    validate_age_and_drift(bundle,bundle.feedback,**kwargs)
    with pytest.raises(ObservationHold,match='expired'):
        validate_age_and_drift(bundle,bundle.feedback,now_ns=bundle.capture_ns+1_000_000_000,**kwargs)
    drift=Feedback(bundle.feedback.q+.1,np.zeros(2),time.monotonic_ns(),{})
    with pytest.raises(ObservationHold,match='moved too far'):
        validate_age_and_drift(bundle,drift,**kwargs)


def test_optimizer_success_does_not_override_slack_or_barrier_rejection():
    target=np.zeros(14);target[0]=.01
    info=dict(success=True,steps=[dict(max_distance_slack_m=.01,minimum_predicted_barrier_after_m=.1)])
    with pytest.raises(ObservationHold,match='slack'):
        check_optimized_step(info,np.zeros(12),target,max_step=.02)
    info['steps'][0]=dict(max_distance_slack_m=0.,minimum_predicted_barrier_after_m=-.01)
    with pytest.raises(ObservationHold,match='barrier'):
        check_optimized_step(info,np.zeros(12),target,max_step=.02)


class FakeDriver:
    def __init__(self,q=None):
        self.q=np.zeros(12) if q is None else np.asarray(q,float).copy()
        self.grip=np.zeros(2);self.sent=[];self.stops=0;self.closed=False
    def read(self):return self.q.copy(),self.grip.copy(),{}
    def send(self,q,delta,grip):
        self.sent.append((q.copy(),delta.copy(),grip.copy()))
        self.q=q+delta;self.grip=grip.copy()
    def stop(self):self.stops+=1
    def close(self):self.closed=True


def runtime_for(tmp_path,mapped,driver,planner):
    args=parse_args(['--prompt','move','--output',str(tmp_path/'run'),'--execute'])
    args.max_result_age=2.;args.max_feedback_age=1.;args.control_period=.005
    runtime=LiveRuntime(args,SimpleNamespace(metadata=metadata()),driver,None,None,planner,calibration_id='digest')
    runtime.feedback.publish(Feedback(mapped.bundle.feedback.q,np.zeros(2),time.monotonic_ns(),{}))
    return runtime


@pytest.mark.parametrize('geometry', ['measured','transparent','inconsistent'])
def test_complete_voxel_ellipsoid_optimizer_to_mock_driver(tmp_path, geometry):
    from obstacle_validation.ellipsoid_geometry import Ellipsoid
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
    class Volumes:
        left_link_names=('left',)
        def ellipsoids(self,q,grip):
            return [Ellipsoid(np.array([q[0],q[1],0]),np.eye(3)*.01**2,'left',0,0),
                    Ellipsoid(np.array([q[6],q[7],0]),np.eye(3)*.01**2,'right',1,0)]
    q=np.zeros(12);q[0]=-.2;q[6]=.2
    mapped,grid=mapped_packet(q=q,empty=geometry!='measured')
    pred=(prediction(front=(10,10,2,2,0),side=(10,10,2,2,0)) if geometry=='inconsistent' else prediction())
    pred['actions'][0][0]=.001
    scoped=DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    planner=EllipsoidMCBF(Volumes(),margin=.005,inter_arm_margin=.035,period=.005)
    driver=FakeDriver(q);runtime=runtime_for(tmp_path,mapped,driver,planner)
    record=runtime.step(PredictionTicket(mapped,pred,scoped))
    assert record['status']=='executed' and len(driver.sent)==1 and driver.stops==1
    assert len(planner.obstacles)==int(geometry!='inconsistent')
    if geometry=='transparent':
        assert planner.obstacles[0].key == 'transparent_0'
        assert record['paired_geometry']['transparent_count'] == 1
    if geometry=='inconsistent':
        assert record['paired_geometry']['ignored_objects'] == ['0']
        assert record['optimized_delta'][0] > 0
    assert record['optimizer']['steps'][0]['minimum_predicted_barrier_after_m']>=0


def test_expiry_after_optimization_prevents_hardware_send(tmp_path):
    mapped,_=mapped_packet();pred=prediction()
    scoped=DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    class SlowPlanner:
        def plan(self,q,grip,actions,boxes,**kwargs):
            time.sleep(.03)
            return actions,dict(success=True,waypoints=[np.r_[q,grip].tolist()],steps=[dict(max_distance_slack_m=0.,minimum_predicted_barrier_after_m=.1)])
    driver=FakeDriver();runtime=runtime_for(tmp_path,mapped,driver,SlowPlanner());runtime.args.max_result_age=.02
    # Reset capture/feedback age immediately before the intentionally slow solve.
    with pytest.raises(ObservationHold,match='expired'):
        runtime.step(PredictionTicket(mapped,pred,scoped))
    assert not driver.sent


def test_default_runtime_host_is_local_and_dry_run():
    args=parse_args(['--prompt','move'])
    assert args.host=='127.0.0.1' and not args.execute


class SyntheticMapping:
    """Only tests use a NumPy occupancy fixture; production requires native Voxblox."""
    def __init__(self, *, empty=False):
        self.empty = empty
    def update(self,bundle):
        mapped,_=mapped_packet(q=bundle.feedback.q,frame_id=bundle.frame_id)
        grid=LocalOccupancyGrid([-.4,-.2,.9],np.eye(3),[12,4,4],.1)
        stamp=min(bundle.frames[v].host_timestamp_ns for v in ('front','side'))
        grid.update(timestamp_ns=stamp,raw_hits=np.empty((0,3)) if self.empty else [[-.05,.05,1.05]],free_ids=np.empty(0,np.int64))
        snapshot=copy.deepcopy(mapped.snapshot)
        snapshot.update(host_timestamp_ns=stamp,vla_frame_id=bundle.frame_id,occupancy_map=grid.snapshot())
        snapshot['occupancy_map']['backend']='voxblox'
        return MappedObservation(bundle,snapshot,mapped.calibrations)


class SyntheticSource:
    def __init__(self):self.starts=self.stops=self.reads=0;self.owner=None
    def start(self):self.starts+=1;self.owner=threading.get_ident()
    def get_color_calibrations(self):return {}
    def stop(self):
        assert self.owner==threading.get_ident()
        self.stops+=1
    def read(self):
        assert self.owner==threading.get_ident()
        time.sleep(.003);self.reads+=1;stamp=time.monotonic_ns()
        return {v:RGBDFrame(v,np.zeros((96,128,3),np.uint8),np.ones((96,128),np.float32),
                            host_timestamp_ns=stamp,frame_number=self.reads) for v in VIEWS}


def test_async_full_loop_keeps_capture_running_and_waits_for_step_ack(tmp_path):
    q=np.zeros(12);q[0]=-.2;q[6]=.2
    driver,source=FakeDriver(q),SyntheticSource()
    class Client:
        metadata=metadata()
        closed=False
        def __init__(self):self.contexts=[]
        def infer(self,rgb,state,prompt,*,context,encoding):
            self.contexts.append(context)
            reads=source.reads
            time.sleep(.02)
            assert source.reads>reads  # camera ownership is not blocked by inference
            result=prediction();result['actions'][0][0]=.0001
            return result
        def close(self):self.closed=True
    class Planner:
        def plan(self,q,grip,actions,boxes,**kwargs):
            delta=np.clip(actions[0,:12]-q,-kwargs['acceleration_step'],kwargs['acceleration_step'])
            return actions,dict(success=True,waypoints=[np.r_[q+delta,grip].tolist()],
                                 steps=[dict(max_distance_slack_m=0.,minimum_predicted_barrier_after_m=.1)])
    client=Client()
    args=parse_args(['--prompt','move','--output',str(tmp_path/'async'),'--max-cycles','3','--execute'])
    args.control_period=.01;args.feedback_period=.002;args.warmup_frames=0
    args.max_result_age=2.;args.max_feedback_age=.5
    runtime=LiveRuntime(args,client,driver,source,lambda info:SyntheticMapping(),Planner(),calibration_id='digest')
    report=runtime.run()
    assert report['completed'] and report['commands_sent']==3 and report['holds']==0
    assert len(driver.sent)==3 and driver.closed and client.closed
    assert source.starts==source.stops==1
    assert len(client.contexts)==3
    assert all(a['frame_id']<b['frame_id'] for a,b in zip(client.contexts,client.contexts[1:]))
    assert not any(t.is_alive() for t in runtime.workers.threads)
    records=[json.loads(line) for line in (args.output/'steps.jsonl').read_text().splitlines()]
    assert all(row['status']=='executed' for row in records)


def test_socket_failure_stops_and_releases_all_owners_without_motion(tmp_path):
    class Client:
        metadata=metadata()
        closed=False
        def infer(self,*a,**kw):raise ConnectionResetError('inference disconnected')
        def close(self):self.closed=True
    client,driver,source=Client(),FakeDriver(),SyntheticSource()
    args=parse_args(['--prompt','move','--output',str(tmp_path/'failure'),'--execute'])
    args.feedback_period=.002;args.warmup_frames=0
    runtime=LiveRuntime(args,client,driver,source,lambda info:SyntheticMapping(),None,calibration_id='digest')
    with pytest.raises(RuntimeError,match='inference disconnected'):
        runtime.run()
    assert not driver.sent and driver.closed and client.closed and source.stops==1
    assert not any(t.is_alive() for t in runtime.workers.threads)
    assert not json.loads((args.output/'report.json').read_text())['completed']


@pytest.mark.parametrize('inconsistent', [False,True])
def test_real_qp_modifies_an_obstacle_approaching_target_before_admission(tmp_path, inconsistent):
    pytest.importorskip('osqp')
    from obstacle_validation.ellipsoid_geometry import Ellipsoid
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
    class Volumes:
        left_link_names=('left',)
        def ellipsoids(self,q,grip):
            return [Ellipsoid(np.array([q[0],q[1],1.05]),np.eye(3)*.01**2,'left',0,0),
                    Ellipsoid(np.array([q[6],q[7],1.05]),np.eye(3)*.01**2,'right',1,0)]
    q=np.zeros(12);q[0]=-.2;q[1]=q[7]=.05;q[6]=.26
    mapped,_=mapped_packet(point=(.15,.05,1.05),q=q,extra_point=False)
    pred=(prediction(front=(10,10,2,2,0),side=(10,10,2,2,0)) if inconsistent else
          prediction(front=(78.3,52.8,12,12,0),side=(59.2,52.8,12,12,0)))
    pred['actions'][0][7]=-.04
    scoped=DynamicOBBFilter(mapped,pred,padding_pixels=0)(mapped.snapshot)
    driver=FakeDriver(q)
    planner=EllipsoidMCBF(Volumes(),margin=.005,inter_arm_margin=.035,period=.05)
    runtime=runtime_for(tmp_path,mapped,driver,planner)
    runtime.args.control_period=.05;runtime.args.speed=1.;runtime.args.max_joint_step=.04;runtime.args.acceleration=100.
    record=runtime.step(PredictionTicket(mapped,pred,scoped))
    assert record['optimizer']['joint_qp_solve_calls']>0
    assert -.04<record['optimized_delta'][6]<=0
    assert len(driver.sent)==1


@pytest.mark.parametrize('geometry', ['measured','transparent','inconsistent'])
def test_real_tcp_prediction_to_shared_capture_filter_optimizer_mock_outputs(tmp_path, geometry):
    try:
        import socket
        probe=socket.socket();probe.close()
    except PermissionError:
        pytest.skip('Loopback sockets prohibited by sandbox')
    from oa_vla.serve import PolicyServer
    from oa_vla.socket_protocol import PolicyClient
    from obstacle_validation.ellipsoid_geometry import Ellipsoid
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
    class Policy:
        def predict(self,rgb,state,prompt,**kwargs):
            result=(prediction(front=(10,10,2,2,0),side=(10,10,2,2,0)) if geometry=='inconsistent' else prediction())
            result['actions'][0][0]=.0001
            return result
    class Volumes:
        left_link_names=('left',)
        def ellipsoids(self,q,grip):
            return [Ellipsoid(np.array([q[0],q[1],0]),np.eye(3)*.01**2,'left',0,0),
                    Ellipsoid(np.array([q[6],q[7],0]),np.eye(3)*.01**2,'right',1,0)]
    server=PolicyServer(('127.0.0.1',0),Policy(),metadata=metadata(),num_steps=1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=PolicyClient(*server.server_address)
        q=np.zeros(12);q[0]=-.2;q[6]=.2
        driver,source=FakeDriver(q),SyntheticSource()
        args=parse_args(['--prompt','move','--output',str(tmp_path/'tcp-loop'),'--max-cycles','2','--execute'])
        args.control_period=.01;args.feedback_period=.002;args.warmup_frames=0;args.max_result_age=2.
        planner=EllipsoidMCBF(Volumes(),margin=.005,inter_arm_margin=.035,period=args.control_period)
        runtime=LiveRuntime(args,client,driver,source,lambda info:SyntheticMapping(empty=geometry!='measured'),planner,calibration_id='digest')
        report=runtime.run()
        assert report['completed'] and report['commands_sent']==2 and len(driver.sent)==2
        assert len(planner.obstacles)==int(geometry!='inconsistent') and source.starts==source.stops==1
        assert report['paired_geometry']['transparent_count'] == int(geometry=='transparent')
        if geometry=='transparent':
            assert planner.obstacles[0].key == 'transparent_0'
        if geometry=='inconsistent':
            assert report['holds'] == 0 and report['paired_geometry']['ignored_objects'] == ['0']
            assert driver.q[0] > q[0]
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


def test_local_runtime_imports_without_model_or_lerobot_dependencies(tmp_path):
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    code='''
import builtins,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
forbidden={'torch','openpi','jax','flax','pyarrow','lerobot'}
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in forbidden: raise RuntimeError('unexpected runtime dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import obstacle_validation.oa_vla_runtime
import obstacle_validation.gripper_output
import real_scripts.live_dual_ur7e_obstacle_model
assert not forbidden.intersection(sys.modules)
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(root)],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
