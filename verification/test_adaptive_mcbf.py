"""Paper mechanisms and closed-loop reaching, without connected hardware."""

from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation.adaptive_mcbf import (
    AdaptiveMCBF, AdaptiveMCBFConfig, BoxScene, geodesic_guidance, select_stage,
)
from obstacle_validation.core import CollisionChecker
from real_scripts.real_cbf_qp import OrientedBox
from safety_module.occupancy import ObservedBoxes
from verification.test_obstacle_validation import TwoPointSampler


def box(x, y, hx, hy):
    return OrientedBox(np.array([x, y, 0.]), np.eye(3), np.array([hx, hy, .2]))


def controller(**kwargs):
    return AdaptiveMCBF(TwoPointSampler(), margin=.01, alpha=1.,
                        inter_arm_margin=.035, period=.05, **kwargs)


@pytest.mark.parametrize('geometry,start,goal,minimum_y', [
    ([box(.5, 0., .08, .15)], [.2, 0.], [.8, 0.], .16),
    ([box(.65, 0., .025, .25), box(.4, .25, .25, .025), box(.4, -.25, .25, .025)],
     [.5, 0.], [.9, 0.], .28),
    ([box(.5, .17, .2, .15), box(.5, -.17, .2, .15)], [0., 0.], [1., 0.], 0.),
])
def test_feedback_reaches_original_goal_around_convex_concave_and_through_narrow_passage(
        geometry, start, goal, minimum_y):
    cbf = controller()
    q = np.zeros(12); q[:2] = start
    actions = np.zeros((1, 14)); actions[0, :2] = goal; actions[0, 12:] = [.2, .8]
    original = actions.copy()
    checker = CollisionChecker(TwoPointSampler(), np.zeros(2), geometry, margin=0.)
    points, stages = [q.copy()], []
    for _ in range(400):
        plan, info = cbf.plan(q, np.zeros(2), actions, geometry,
                              lower=np.full(12, -2.), upper=np.full(12, 2.), max_step=.025)
        np.testing.assert_array_equal(plan, original)
        step = np.asarray(info['waypoints'][0])
        assert np.max(np.abs(step[:12]-q)) <= .025+5e-5
        # Offline assertions only: no checks run in the execution pipeline.
        assert checker.edge_safe(q, step[:12])
        np.testing.assert_array_equal(step[12:], actions[0, 12:])
        q = step[:12]
        points.append(q.copy()); stages.append(info['adaptive_stage']['stage'])
        if np.max(np.abs(q-actions[0, :12])) < .001:
            break
    else:
        pytest.fail(f'Failed to reach original goal: {q[:2]}, stages={set(stages)}')
    assert np.max(np.abs(np.asarray(points)[:, 1])) >= minimum_y
    if minimum_y == 0.:
        assert 'narrow_space' in stages
        assert np.max(np.abs(np.asarray(points)[:, 1])) < .02


def test_stage_switches_change_margins_and_guidance_initialization():
    config = AdaptiveMCBFConfig()
    jac = np.zeros((3, 12)); jac[0, 0] = jac[1, 1] = 1.
    point, goal = np.zeros(3), np.array([1., 0., 0.])
    normal = select_stage(BoxScene([box(.5, 0., .1, .1)], config), point, goal, jac, .01, 1., config)
    narrow = select_stage(BoxScene([box(.5, .17, .2, .15), box(.5, -.17, .2, .15)], config),
                          point, goal, jac, .01, 1., config)
    boxes = ObservedBoxes([box(.2, 0., .05, .05)])
    boxes.source_snapshot = {'obbs': [{'velocity_m_s': [-.2, 0., 0.]}]}
    dynamic = select_stage(BoxScene(boxes, config), point, goal, jac, .01, 1., config)
    assert [s['stage'] for s in (dynamic, normal, narrow)] == ['dynamic_obstacle', 'normal', 'narrow_space']
    assert narrow['margin_m'] < normal['margin_m'] < dynamic['margin_m']
    assert dynamic['alpha'] < normal['alpha'] < narrow['alpha']
    assert narrow['guidance_initialization'] == 'zero_isoline'
    assert dynamic['guidance_initialization'] == normal['guidance_initialization'] == 'robot_state'


def test_moving_clusters_use_capture_timestamps_and_remain_static_on_cached_frame():
    config = AdaptiveMCBFConfig()
    first = ObservedBoxes([box(.3, 0., .05, .05)])
    first.source_snapshot = {'host_timestamp_ns': 1_000_000_000}
    second = ObservedBoxes([box(.28, 0., .05, .05)])
    second.source_snapshot = {'host_timestamp_ns': 1_100_000_000}
    scene = BoxScene(second, config, BoxScene(first, config))
    np.testing.assert_allclose(scene.velocities, [[-.2, 0., 0.]])
    assert scene.dynamic[0] and len(scene.static_ids) == 0
    cbf = controller()
    actions = np.zeros((1, 14)); actions[0, 0] = .5
    cbf.plan(np.zeros(12), np.zeros(2), actions, first, lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    cbf.plan(np.zeros(12), np.zeros(2), actions, second, lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    before = cbf.scene
    cbf.plan(np.zeros(12), np.zeros(2), actions, second, lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    assert cbf.scene is before
    np.testing.assert_allclose(cbf.scene.velocities, [[-.2, 0., 0.]])


@pytest.mark.parametrize('status,expected', [('maximum iterations reached', 'recorded_action'),
                                            ('primal infeasible', 'geodesic_reference')])
def test_failed_qp_moves_without_hold_or_result_collision_checks(monkeypatch, status, expected):
    import osqp
    class Solver:
        def setup(self, **kwargs): pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=None, info=SimpleNamespace(status=status, status_val=7))
    monkeypatch.setattr(osqp, 'OSQP', Solver)
    monkeypatch.setattr(CollisionChecker, 'edge_safe', lambda *a: pytest.fail('post-solve collision check'))
    cbf = controller()
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((1, 14)); actions[0, 0] = .8
    _, info = cbf.plan(q, np.zeros(2), actions, [box(.5, 0., .08, .15)],
                       lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    assert np.linalg.norm(np.asarray(info['waypoints'][0][:12])-q) > .001
    assert info['steps'][0]['fallback'] == expected
    assert not info['post_optimization_checks']


def test_solver_output_is_executed_without_residual_or_collision_gate(monkeypatch):
    import osqp
    delta = np.zeros(12); delta[0] = .02
    class Solver:
        def setup(self, **kwargs): pass
        def solve(self, **kwargs):
            # Deliberately violates the tangent inequality: the caller asked
            # for direct execution, so no residual-based rejection is allowed.
            return SimpleNamespace(x=delta.copy(), info=SimpleNamespace(status='solved', status_val=1))
    monkeypatch.setattr(osqp, 'OSQP', Solver)
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((1, 14)); actions[0, 0] = .8
    _, info = controller().plan(q, np.zeros(2), actions, [box(.5, 0., .08, .15)],
                                lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    np.testing.assert_allclose(info['waypoints'][0][:12], q+delta)
    assert info['steps'][0]['fallback'] is None


def test_stationary_base_samples_do_not_crowd_out_moving_link_barriers(monkeypatch):
    import osqp
    matrices = []
    class Sampler:
        points_per_link = 128
        left_link_names = ('base', 'tip')
        def link_points(self, q, grip):
            values = np.zeros((3, self.points_per_link, 3))
            values[0, :, 0] = .41  # Many closer, uncontrollable base samples.
            values[1, :, 0] = q[0]
            values[2] = 3.
            return values
    class Solver:
        def setup(self, **kwargs): matrices.append(kwargs['A'].toarray())
        def solve(self, **kwargs):
            return SimpleNamespace(x=np.zeros(12), info=SimpleNamespace(status='solved', status_val=1))
    monkeypatch.setattr(osqp, 'OSQP', Solver)
    cbf = AdaptiveMCBF(Sampler(), margin=.01, alpha=1., inter_arm_margin=.035, period=.05)
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((1, 14)); actions[0, 0] = .3
    cbf.plan(q, np.zeros(2), actions, [box(.5, 0., .08, .15)],
             lower=-np.ones(12), upper=np.ones(12), max_step=.025)
    # Exclude the final 12 joint-bound rows. A moving surface row must remain.
    assert np.any(matrices[0][:-12, 0] < -.9)


def test_zero_isoline_projection_and_multiple_candidate_geodesics():
    config = AdaptiveMCBFConfig()
    scene = BoxScene([box(.5, 0., .08, .15)], config)
    point, goal = np.array([.2, 0., 0.]), np.array([.8, 0., 0.])
    jacobian = np.zeros((3, 12)); jacobian[:, :3] = np.eye(3)
    stage = {'margin_m': .0025, 'guidance_initialization': 'zero_isoline'}
    guidance = geodesic_guidance(scene, point, goal, jacobian, stage, config)
    assert guidance['candidate_count'] == config.geodesic_candidates
    assert len(guidance['path']) > 2
    start = np.asarray(guidance['path'][0])
    distance, normal = scene.field(start, scene.static_ids)
    assert distance == pytest.approx(.0025, abs=1e-6)
    assert normal @ guidance['direction'] == pytest.approx(0., abs=1e-8)


def test_confinement_uses_temporary_opening_goal_and_restores_original_goal():
    geometry = [box(0., .2, .225, .025), box(0., -.2, .225, .025), box(-.2, 0., .025, .2),
                box(.2, .125, .025, .075), box(.2, -.125, .025, .075)]
    cbf = controller()
    q = np.zeros(12)
    actions = np.zeros((1, 14)); actions[0, 0] = .5
    stages = []
    for _ in range(100):
        _, info = cbf.plan(q, np.zeros(2), actions, geometry, lower=-np.ones(12), upper=np.ones(12), max_step=.025)
        stages.append(info['adaptive_stage'])
        q = np.asarray(info['waypoints'][0][:12])
        if np.max(np.abs(q-actions[0, :12])) < .001:
            break
    else:
        pytest.fail('Did not exit confinement and reach the original target')
    assert stages[0]['confined'] and stages[0]['temporary_target'] is not None
    assert not stages[-1]['confined'] and stages[-1]['temporary_target'] is None


@pytest.mark.parametrize('singular', [False, True])
def test_ee_qp_reaches_target_and_preserves_nullspace_motion_at_singularity(singular):
    from scipy.spatial.transform import Rotation
    class Sampler(TwoPointSampler):
        def tcp_transforms(self, q):
            poses = np.tile(np.eye(4), (2, 1, 1))
            if not singular:
                for arm in range(2):
                    poses[arm, :3, 3] = q[arm*6:arm*6+3]
                    poses[arm, :3, :3] = Rotation.from_rotvec(q[arm*6+3:arm*6+6]).as_matrix()
            return poses
    cbf = AdaptiveMCBF(Sampler(), margin=.01, alpha=1., inter_arm_margin=.035,
                       period=.05, optimization_space='ee')
    actions = np.zeros((1, 14)); actions[0, :12] = np.linspace(.01, .03, 12)
    _, info = cbf.plan(np.zeros(12), np.zeros(2), actions, [],
                       lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    np.testing.assert_allclose(info['waypoints'][0][:12], actions[0, :12], atol=1e-5)
    assert info['steps'][0]['solver_success']
    assert info['ee_variables'] == 12 and not info['post_optimization_checks']


def test_physical_command_sequence_uses_ellipsoids_without_planning_stops(tmp_path, monkeypatch):
    from obstacle_validation import lerobot_replay as replay
    from verification.replay_test_scene import run_with_selected_scene
    actions = np.zeros((2, 14)); actions[:, 0] = [.1, .15]
    events = []
    state = {'q': np.zeros(12), 'grip': np.zeros(2)}
    class Policy:
        period = .05
        states = np.vstack((np.zeros((1, 14)), actions[:-1]))
        source_frames = np.arange(2)
        metadata = {'initial_state': np.zeros(14).tolist(), 'final_joint_state': actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = actions
        def infer(self, start, horizon): return actions[start:start+horizon].copy()
    class Sampler(TwoPointSampler):
        tcp_transform_verified = {'left': True, 'right': True}
        calibration_digest = 'test'
        def __init__(self, **kwargs): pass
    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state['q'].copy(), state['grip'].copy()
        def send(self, q, delta):
            events.append('send')
            state['q'] = q+delta
        def stop(self): events.append('stop')
        def close(self): pass
    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target):
            state['grip'] = np.asarray(target).copy()
            return [False, False]
        def close(self): pass
    snapshot = {'coordinate_frame': 'left_base', 'status': 'ok', 'obbs': [
        {'center': [.05, 0., 0.], 'axes': np.eye(3).tolist(), 'half_sizes': [.01, .02, .1]}]}
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
    original_plan = EllipsoidMCBF.plan
    def plan(self, *args, **kwargs):
        events.append('plan')
        if events.count('plan') > 300:
            pytest.fail('Adaptive replay did not reach its goals')
        return original_plan(self, *args, **kwargs)
    monkeypatch.setattr(EllipsoidMCBF, 'plan', plan)
    monkeypatch.setattr(replay, 'DatasetActionPolicy', Policy)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'DualCommander', Commander)
    monkeypatch.setattr(replay, 'DualGripperCommander', Grippers)
    monkeypatch.setattr(replay, 'read_snapshot', lambda *a, **kw: snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *a: None)
    monkeypatch.setattr(replay, '_motion_safe', lambda *a, **kw: pytest.fail('output sweep check'))
    monkeypatch.setattr(CollisionChecker, 'edge_safe', lambda *a: pytest.fail('output collision check'))
    args = replay.parser().parse_args([
        'replay', '--dataset', str(tmp_path/'dataset'), '--output', str(tmp_path/'run'),
        '--mode', 'avoid', '--avoidance-controller', 'mcbf', '--execute', '--execute-ack', replay.EXECUTE_ACK,
        '--max-tracking-error', '.001', '--arrival-tolerance', '.001', '--final-tolerance', '.001'])
    report = run_with_selected_scene(args, monkeypatch)
    assert report['completed'] and report['actions_executed'] == 2
    first_plan, last_send = events.index('plan'), len(events)-1-events[::-1].index('send')
    assert 'stop' not in events[first_plan:last_send+1]
    assert events.count('plan') > 2
    assert report['optimization_retries'] == 0
    assert report['max_tracking_error_rad'] > args.max_tracking_error
    assert report['tracking_error_policy'] == 'diagnostic_only'
    assert report['post_optimization_checks']
