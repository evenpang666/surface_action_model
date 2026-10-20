"""Behavioral checks for coupled action-prefix optimization."""

import numpy as np

from obstacle_validation.cbf import RecordedPathCBF
from obstacle_validation.core import CollisionChecker
from real_scripts.real_cbf_qp import OrientedBox
from verification.test_obstacle_validation import TwoPointSampler


def optimize(actions, *, q=None, boxes=(), previous=None, max_step=.05, acceleration_step=np.inf,
             lower=None, upper=None, swept_safe=None, modulation=False, validate_sweeps=True):
    sampler, grip = TwoPointSampler(), np.zeros(2)
    q = np.zeros(12) if q is None else q
    checker = CollisionChecker(sampler, grip, boxes, margin=.01, inter_arm_margin=.035)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01, inter_arm_margin_m=.035,
                          trigger_margin_m=.025, alpha=1., iterations=100, modulation=modulation)
    from obstacle_validation.lerobot_replay import _motion_safe
    sweep = swept_safe or (lambda origin, opening, delta, target: _motion_safe(
        sampler, origin, opening, delta, target, boxes,
        {'collision_margin_m': .01, 'inter_arm_margin_m': .035}, True))
    result, info = cbf.project_chunk(
        q, grip, actions, checker, lower=np.full(12, -1.) if lower is None else lower,
        upper=np.full(12, 1.) if upper is None else upper, max_step=max_step,
        previous_delta=np.zeros(12) if previous is None else previous,
        acceleration_step=acceleration_step, swept_safe=sweep if validate_sweeps else None)
    return result, info, cbf, checker


def test_ten_joint_actions_reduce_acceleration_and_keep_gripper_commands():
    actions = np.zeros((10, 14)); actions[:, 0] = np.tile([.04, 0.], 5)
    actions[:, 12:] = np.tile([[0., 1.], [1., 0.]], (5, 1))
    result, info, _, _ = optimize(actions)
    assert info['success'] and info['joint_variables'] == 120, info
    assert info['mode'] == 'joint_execution_prefix' and info['joint_qp_solve_calls'] >= 1
    assert info['optimized_acceleration_squared'] < .3*info['seed_acceleration_squared']
    assert info['joint_objective_after'] <= info['joint_objective_before']+1e-9
    np.testing.assert_array_equal(result[:, 12:], actions[:, 12:])
    assert not np.allclose(result[:, 0], actions[:, 0])


def test_later_actions_affect_early_actions_in_same_joint_solve():
    first = np.zeros((10, 14)); first[:, 0] = np.arange(1, 11)*.01
    second = first.copy(); second[6:, 1] = .04
    a, ia, _, _ = optimize(first)
    b, ib, _, _ = optimize(second)
    assert ia['success'] and ib['success']
    np.testing.assert_array_equal(first[:6], second[:6])
    # A sequential optimizer cannot change action 4 in response to action 6.
    assert b[4, 1]-a[4, 1] > 1e-4


def test_chunk_boundary_uses_measured_pose_and_previous_joint_step():
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((10, 14)); actions[:, 0] = .2+np.arange(1, 11)*.01
    positive = np.zeros(12); positive[0] = .02
    negative = -positive
    a, ia, _, _ = optimize(actions, q=q, previous=positive)
    b, ib, _, _ = optimize(actions, q=q, previous=negative)
    assert ia['success'] and ib['success']
    assert a[0, 0] > b[0, 0]+.001


def test_joint_ten_action_detour_passes_all_coupled_sweeps_and_per_action_bounds():
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((10, 14)); actions[:, 0] = np.linspace(.225, .45, 10)
    boxes = [OrientedBox(np.array([.35, 0., 0.]), np.eye(3), np.array([.03, .04, .1]))]
    lower, upper = np.full((10, 12), -1.), np.full((10, 12), 1.)
    lower[:, 1], upper[:, 1] = -.15, .15
    result, info, _, checker = optimize(actions, q=q, boxes=boxes, lower=lower, upper=upper,
                                        max_step=.025, modulation=True)
    assert info['success'] and info['joint_variables'] == 120, info
    previous = q
    for waypoint, source in zip(info['waypoints'], info['waypoint_action_indices']):
        target = np.asarray(waypoint[:12])
        assert checker.edge_safe(previous, target)
        assert np.max(abs(target-previous)) <= .025+1e-7
        assert np.all(target >= lower[source]-1e-7) and np.all(target <= upper[source]+1e-7)
        previous = target
    assert np.max(abs(result[:, 1])) > .01


def test_failed_final_sweep_never_exports_a_safe_prefix_or_commits_guidance():
    actions = np.zeros((10, 14)); actions[:, 0] = np.arange(1, 11)*.01
    actions[:, 13] = np.linspace(0., 1., 10)
    # The first nine edges pass; the final fixed gripper command is blocked.
    result, info, cbf, _ = optimize(actions, swept_safe=lambda origin, opening, delta, target: target[1] < 1.,
                                   modulation=True)
    assert result is None and not info['success']
    assert info['reason'] == 'joint_chunk_optimization_failed'
    assert info['failed_action_index'] == 9
    assert not info['waypoints']
    np.testing.assert_array_equal(cbf.detour_direction, np.zeros(12))


def test_linear_acceleration_limit_includes_the_boundary_step():
    actions = np.zeros((10, 14)); actions[:, 0] = np.arange(1, 11)*.01
    result, info, _, _ = optimize(actions, acceleration_step=.003)
    assert info['success'] and not info['acceleration_relaxed'], info
    velocities = np.diff(np.vstack((np.zeros(12), result[:, :12])), axis=0)
    acceleration = np.diff(np.vstack((np.zeros(12), velocities)), axis=0)
    assert abs(acceleration).max() <= .003+1e-7


def test_direct_execution_skips_seed_and_final_sweep_checks_but_keeps_joint_qp(monkeypatch):
    from obstacle_validation import lerobot_replay as replay
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct execution must not call a sweep validator')
    monkeypatch.setattr(CollisionChecker, 'edge_safe', forbidden)
    monkeypatch.setattr(replay, '_motion_safe', forbidden)
    actions = np.zeros((10, 14)); actions[:, 0] = np.tile([.04, 0.], 5)
    result, info, cbf, _ = optimize(actions, validate_sweeps=False)
    assert info['success'] and result.shape == (10, 14), info
    assert info['joint_variables'] == 120 and info['joint_qp_solve_calls'] >= 1
    assert not info['sweep_validation_enabled']
    assert all(not step['sweep_validation_enabled'] for step in info['waypoint_steps'])
    assert 'sweep_validation' not in cbf.runtime_stats['stage_s']
    assert info['optimized_acceleration_squared'] < info['seed_acceleration_squared']


def test_infeasible_seed_guidance_rebuilds_goals_and_keeps_obstacle_bounds():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((10, 14)); actions[:, 0] = .2+np.arange(1, 11)*.003
    boxes = [OrientedBox(np.array([.27, 0., 0.]), np.eye(3), np.array([.01, .04, .1]))]
    checker = CollisionChecker(sampler, grip, boxes, margin=.01, inter_arm_margin=.035)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01, inter_arm_margin_m=.035,
                          trigger_margin_m=.025, alpha=1., iterations=100, modulation=True)
    original_seed = cbf._seed_chunk
    direction = np.zeros(12); direction[0] = 1.
    def conflicting_seed(*args, **kwargs):
        result, seed = original_seed(*args, **kwargs)
        assert seed['success']
        seed['steps'][0]['modulation'] = {'joint_direction': direction.tolist(),
                                         'minimum_joint_progress_rad': 2.}
        cbf.detour_direction = direction.copy()
        return result, seed
    cbf._seed_chunk = conflicting_seed
    events = []
    cbf.progress_callback = events.append
    result, info = cbf.project_chunk(
        q, grip, actions, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
        max_step=.05, previous_delta=np.zeros(12), acceleration_step=np.inf)
    assert info['success'] and result.shape == (10, 14), info
    assert info['guidance_fallback'] == 'relinearize_repaired_dataset_goals'
    assert info['joint_qp_solve_calls'] >= 2
    assert any(event['stage'] == 'guidance_fallback' for event in events)
    assert events[-1]['stage'] == 'complete' and events[-1]['success']
    assert all(checker.clearance(pose[:12])[0] >= .01 for pose in result)
    assert np.max(abs(np.diff(np.vstack((q, result[:, :12])), axis=0))) <= .05+1e-7
    assert all('modulation' not in step for step in info['waypoint_steps'])
    np.testing.assert_array_equal(cbf.detour_direction, np.zeros(12))


def test_opposite_face_target_repairs_relinearize_from_measured_obstacle_side():
    q = np.zeros(12); q[0] = .28
    actions = np.zeros((10, 14)); actions[:, 0] = np.linspace(.34, .36, 10)
    box = OrientedBox(np.array([.35, 0., 0.]), np.eye(3), np.array([.03, .1, .1]))
    lower, upper = np.full((10, 12), -1.), np.full((10, 12), 1.)
    # Only X can move the sampled point. Staying on the measured side is
    # feasible; switching between the repaired goals' faces is impossible.
    lower[:, 1] = upper[:, 1] = 0.
    result, info, _, checker = optimize(
        actions, q=q, boxes=[box], lower=lower, upper=upper, max_step=.2,
        modulation=True, validate_sweeps=False)
    assert info['success'], info
    repairs = np.asarray([item['target'] for item in info['target_repairs']])
    assert repairs[0, 0] < box.center[0] < repairs[-1, 0]
    assert info['linearization_reference'] == 'measured_pose_with_tracking_bounds'
    assert info['tracking_reference'] == 'repaired_dataset_goals'
    assert info['joint_qp_solve_calls'] >= 2
    assert np.all(result[:, 0] < box.center[0]-box.half_sizes[0]-.01)
    assert np.all(result[:, :12] >= lower-1e-7) and np.all(result[:, :12] <= upper+1e-7)
    np.testing.assert_array_equal(result[:, 12:], actions[:, 12:])
    origin = q
    for target in result[:, :12]:
        assert checker.edge_safe(origin, target)
        origin = target


def test_sparse_qp_handles_many_redundant_constraints_at_large_joint_offsets():
    from scipy.optimize import LinearConstraint
    from obstacle_validation.joint_chunk import _solve_sparse_qp
    center = np.full((10, 12), 5.)
    target = center.ravel()+.02
    matrix = np.tile(np.eye(120), (10, 1))
    constraints = [LinearConstraint(matrix, np.full(1200, -np.inf), np.full(1200, 5.01))]
    result = _solve_sparse_qp(np.eye(120), -target, center, center-.1, center+.1, constraints, 100)
    assert result.success, result.message
    np.testing.assert_allclose(result.x, 5.01, atol=1e-6)


def test_tool_cache_preserves_vendor_geometry_when_opening_changes(monkeypatch):
    from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler, TWO_F85_LINK_NAMES
    from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_points_in_active_tcp
    sampler = DualUR7eSurfacePointSampler(points_per_link=16)
    q = np.linspace(-.4, .6, 12)
    cached, expected = [], []
    for grip in (np.array([.13, .78]), np.array([.78, .13])):
        cached.append(sampler.link_points(q, grip))
        with monkeypatch.context() as patch:
            patch.setattr(sampler, '_tool_local_points', lambda closing: robotiq_2f85_points_in_active_tcp(
                finger_angle_rad=float(np.clip(closing, 0., 1.)*.8), samples_per_link=16).reshape(
                    len(TWO_F85_LINK_NAMES), 16, 3))
            expected.append(sampler.link_points(q, grip))
    np.testing.assert_array_equal(cached, expected)
    assert not np.array_equal(cached[0], cached[1])
