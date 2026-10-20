"""CBF safety checks used by LeRobot action replay and restoration."""

import numpy as np
import pytest

from obstacle_validation.core import CollisionChecker, JointPath, PathFollower
from obstacle_validation.cbf import CBFGoalFollower, PreviewFKCache, RecordedPathCBF, preview_action_chunk
from real_scripts.real_cbf_qp import OrientedBox
from verification.test_obstacle_validation import TwoPointSampler


def optimize_chunk(sampler, actions, boxes, *, q=None, max_step=.025):
    q = np.zeros(12) if q is None else q
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    checker = CollisionChecker(sampler, np.zeros(2), boxes, margin=.01, inter_arm_margin=.035)
    origins = []
    def sweep(origin, grip, delta, target):
        from obstacle_validation.lerobot_replay import _motion_safe
        origins.append((origin.copy(), grip.copy(), target.copy()))
        return _motion_safe(sampler, origin, grip, delta, target, boxes,
                            {"collision_margin_m": .01, "inter_arm_margin_m": .035}, True)
    optimized, info = cbf.project_chunk(
        q, np.zeros(2), actions, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
        max_step=max_step, previous_delta=np.zeros(12), acceleration_step=.05, swept_safe=sweep)
    return optimized, info, cbf, origins


def test_chunk_optimizer_corrects_every_intruding_tail_action_before_execution():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.35, 0., 0.]), np.eye(3), np.array([.03, .1, .1]))
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((6, 14)); actions[:, 0] = [.225, .25, .275, .3, .325, .35]
    optimized, info, cbf, _ = optimize_chunk(sampler, actions, [box], q=q)
    assert info['success'] and info['mode'] == 'joint_execution_prefix'
    assert info['joint_variables'] == 6*12 and info['modified_action_indices'] == list(range(6))
    assert all(info['steps'][i]['qp_solve_calls'] > 0 for i in (4, 5))
    assert info['optimized_acceleration_squared'] < info['seed_acceleration_squared']
    assert np.all(optimized[4:, 0] < .31)
    assert cbf.runtime_stats['qp_solve_calls'] >= 2
    previous = q
    for action in optimized:
        assert CollisionChecker(sampler, action[12:], [box]).edge_safe(previous, action[:12])
        previous = action[:12]


def test_chunk_optimizer_smooths_safe_goals_with_multiple_speed_bounded_steps():
    actions = np.zeros((2, 14)); actions[:, 0] = [.061, .07]
    optimized, info, cbf, _ = optimize_chunk(TwoPointSampler(), actions, [])
    assert info['success'] and info['joint_qp_solve_calls'] == 1
    assert np.max(np.abs(optimized-actions)) < .01
    assert info['optimized_acceleration_squared'] < info['seed_acceleration_squared']
    waypoints = np.asarray(info['waypoints'])
    assert info['waypoint_action_indices'] == [0, 0, 0, 1]
    assert np.abs(np.diff(np.vstack((np.zeros((1, 12)), waypoints[:, :12])), axis=0)).max() <= .025+1e-12


def test_chunk_optimizer_uses_each_gripper_shape_and_corrected_predecessor():
    class GripperSampler(TwoPointSampler):
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[q[6]+.03*grip[1], 0., 0.]]])
    actions = np.zeros((3, 14)); actions[:, 13] = [0., .7, 1.]
    box = OrientedBox(np.array([.03, 0., 0.]), np.eye(3), np.array([.003, .1, .1]))
    optimized, info, _, origins = optimize_chunk(GripperSampler(), actions, [box])
    assert info['success'] and info['modified_action_indices'] == [0, 1, 2]
    np.testing.assert_array_equal(optimized[:, 12:], actions[:, 12:])
    assert np.all(optimized[1:, 6]+.03*actions[1:, 13] < .017)
    # The last QP starts from the preceding corrected posture, not q=0.
    assert any(np.allclose(origin, optimized[1, :12]) and np.allclose(grip, [0., .7])
               for origin, grip, target in origins if np.allclose(target, [0., 1.]))


def test_joint_optimizer_can_recover_from_a_failed_greedy_tail(monkeypatch):
    sampler = TwoPointSampler()
    original = RecordedPathCBF.project
    calls = []
    def fail_last(self, *args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            return np.zeros(12), {'success': False, 'triggered': True, 'reason': 'infeasible'}
        return original(self, *args, **kwargs)
    monkeypatch.setattr(RecordedPathCBF, 'project', fail_last)
    actions = np.zeros((3, 14)); actions[:, 0] = [.01, .02, .03]
    optimized, info, cbf, _ = optimize_chunk(sampler, actions, [])
    assert optimized is not None and info['success'] and not info['seed_success']
    assert info['joint_variables'] == 3*12
    assert cbf.sampler is sampler


def test_chunk_sweeps_allow_aged_free_voxels():
    import time
    from obstacle_validation.lerobot_replay import _motion_safe
    from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, ObservedBoxes
    stamp = time.monotonic_ns()-3_000_000_000
    grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    for timestamp in (stamp-2_000_000, stamp-1_000_000, stamp):
        grid.update(timestamp_ns=timestamp, raw_hits=np.empty((0, 3)), free_ids=np.arange(grid.size))
    boxes = ObservedBoxes([], ObservationGrid(grid.snapshot()))
    q, grip = np.zeros(12), np.zeros(2)
    delta = q.copy(); delta[0] = .01
    safety = {'collision_margin_m': .01, 'inter_arm_margin_m': .035}
    assert _motion_safe(TwoPointSampler(), q, grip, delta, grip, boxes, safety,
                         observation_now_ns=stamp+10_000_000)
    assert _motion_safe(TwoPointSampler(), q, grip, delta, grip, boxes, safety)
    assert boxes.observation.last_blockage is None


def test_preview_cache_reuses_exact_overlapping_actions_and_keys_gripper():
    class Sampler(TwoPointSampler):
        calls = 0
        def link_points(self, q, grip):
            self.calls += 1
            return super().link_points(q, grip)

    sampler = Sampler()
    cache = PreviewFKCache(sampler, capacity=4)
    actions = np.zeros((4, 14)); actions[:, 0] = [.01, .02, .03, .04]
    first = preview_action_chunk(cache, np.zeros(12), actions[:3], max_step=.025)
    following = preview_action_chunk(cache, first.joint_positions[0], actions[1:], max_step=.025)
    assert (cache.hits, cache.misses, sampler.calls) == (2, 4, 4)
    expected = preview_action_chunk(TwoPointSampler(), first.joint_positions[0], actions[1:], max_step=.025)
    np.testing.assert_array_equal(following.surface_points, expected.surface_points)
    cache.link_points(actions[3, :12], np.ones(2))
    assert cache.misses == 5 and len(cache.points) == 4
    # The oldest pose was evicted, and returned FK must not be modified.
    cached = cache.link_points(actions[0, :12], np.zeros(2))
    assert cache.misses == 6 and not cached.flags.writeable


@pytest.mark.parametrize("seed", range(8))
def test_pruned_inter_arm_sweep_matches_all_pairs(seed):
    from safety_module.geometry import swept_inter_arm_safe
    rng = np.random.default_rng(seed)
    start = rng.uniform(-1., 1., (29, 3))
    for travel in (0., .01, .5, 3.):
        end = start+rng.uniform(-travel, travel, start.shape)
        first = start[:13, None]-start[None, 13:]
        change = (end[:13, None]-end[None, 13:])-first
        t = np.clip(-(first*change).sum(-1)/np.maximum((change*change).sum(-1), 1e-15), 0., 1.)
        distances = np.linalg.norm(first+t[..., None]*change, axis=-1)
        for margin in (0., .035, .3, 1.):
            expected = not np.any(distances <= margin+1e-12)
            assert swept_inter_arm_safe(start, end, 13, margin) == expected


def test_pruned_inter_arm_sweep_detects_collision_between_safe_endpoints():
    from safety_module.geometry import swept_inter_arm_safe
    start = np.array([[-1., 0., 0.], [1., 0., 0.]])
    assert not swept_inter_arm_safe(start, -start, 1, .035)
    assert not swept_inter_arm_safe(start, start, 1, 2.)


@pytest.mark.parametrize("correction,state", [([0., .003], "cbf"), ([0., 0.], "cbf_hold")])
def test_cbf_restoration_accepts_safe_lateral_steps_or_hold_without_rrt(monkeypatch, correction, state):
    import obstacle_validation.core as core

    monkeypatch.setattr(core, 'rrt_detour', lambda *a, **k: pytest.fail('CBF restoration called RRT'))
    class Projector:
        def project(self, q, nominal, checker, **kwargs):
            result = np.zeros(12); result[:2] = correction
            return result, {'success': True, 'triggered': True}

    goal = np.zeros(12); goal[0] = .5
    follower = CBFGoalFollower(goal, lower=np.full(12, -1.), upper=np.full(12, 1.),
                               dt=.05, speed=.5, acceleration=1.2, tolerance=.001, projector=Projector())
    checker = CollisionChecker(TwoPointSampler(), np.zeros(2), [])
    delta, result = follower.next_delta(np.zeros(12), checker)
    assert result == state
    np.testing.assert_allclose(delta[:2], correction)
    assert checker.edge_safe(np.zeros(12), delta)


def test_cbf_restoration_can_advance_through_aged_free_voxels(monkeypatch):
    import time
    import obstacle_validation.core as core
    from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, ObservedBoxes

    monkeypatch.setattr(core, 'rrt_detour', lambda *a, **k: pytest.fail('CBF restoration called RRT'))
    grid = LocalOccupancyGrid([-.5, -.5, -.5], np.eye(3), [10, 10, 10], .1)
    stamp = time.monotonic_ns()-3_000_000_000
    for timestamp in (stamp-2_000_000, stamp-1_000_000, stamp):
        grid.update(timestamp_ns=timestamp, raw_hits=np.empty((0, 3)), free_ids=np.arange(grid.size))
    boxes = ObservedBoxes([], ObservationGrid(grid.snapshot()))
    cbf = RecordedPathCBF(TwoPointSampler(), np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    goal = np.zeros(12); goal[0] = .2
    follower = CBFGoalFollower(goal, lower=np.full(12, -1.), upper=np.full(12, 1.),
                               dt=.05, speed=.5, acceleration=1.2, tolerance=.001, projector=cbf)
    checker = CollisionChecker(TwoPointSampler(), np.zeros(2), boxes, observation_now_ns=stamp+3_000_000_000)
    delta, state = follower.next_delta(np.zeros(12), checker)
    assert state == 'nominal' and delta[0] > 0
    assert checker.last_observation_blockage is None


def test_cbf_restoration_infeasibility_never_falls_back_to_rrt(monkeypatch):
    import obstacle_validation.core as core
    monkeypatch.setattr(core, 'rrt_detour', lambda *a, **k: pytest.fail('CBF restoration called RRT'))

    class Projector:
        def project(self, *args, **kwargs):
            return np.zeros(12), {'success': False, 'triggered': True, 'reason': 'infeasible'}

    goal = np.zeros(12); goal[0] = .5
    follower = CBFGoalFollower(goal, lower=np.full(12, -1.), upper=np.full(12, 1.),
                               dt=.05, speed=.5, acceleration=1.2, tolerance=.001, projector=Projector())
    delta, state = follower.next_delta(np.zeros(12), CollisionChecker(TwoPointSampler(), np.zeros(2), []))
    assert state == 'cbf_infeasible' and not delta.any()


def test_full_step_cbf_corrects_a_future_normal_that_cuts_a_rounded_corner():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([-.1, 0., 0.]), np.eye(3), np.array([.02, .03, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q = np.zeros(12); q[:2] = [-.07989430256591729, .03999983348247517]
    nominal = -q; nominal *= .003/max(abs(nominal))
    safe, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
                             max_step=.025, previous_delta=np.zeros(12), acceleration_step=.003)
    assert info['success'] and info['projection_method'] == 'full_step_cbf_qp'
    assert np.max(np.abs(safe)) >= np.max(np.abs(nominal))-1e-7
    assert safe[0] > 0. and checker.edge_safe(q, q+safe)


def test_chunk_preview_rolls_out_absolute_targets_from_measured_joints():
    class GripperSampler(TwoPointSampler):
        def link_points(self, q, grip):
            return np.asarray([[[q[0], grip[0], 0.]], [[3., grip[1], 3.]]])

    q = np.zeros(12)
    q[0] = .2
    actions = np.zeros((3, 14))
    actions[:, 0] = [.5, .5, .1]
    actions[:, 12:] = [[.1, .2], [.3, .4], [.5, .6]]
    preview = preview_action_chunk(GripperSampler(), q, actions, max_step=.1)
    np.testing.assert_allclose(preview.joint_positions[:, 0], [.3, .4, .3])
    np.testing.assert_allclose(preview.surface_points[:, 0, 0, 0], [.3, .4, .3])
    np.testing.assert_allclose(preview.surface_points[:, 0, 0, 1], [.1, .3, .5])
    np.testing.assert_allclose(preview.gripper_targets, actions[:, 12:])
    assert q[0] == .2


def test_chunk_tail_intrusion_triggers_qp_while_safe_first_step_is_preserved():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.45, 0., 0.]), np.eye(3), np.array([.03, .1, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .2
    actions = np.zeros((10, 14)); actions[:, 0] = np.linspace(.225, .45, 10)
    nominal = actions[0, :12]-q
    calls = []
    cbf.before_solve = lambda: calls.append(True)
    options = dict(lower=np.full(12, -1.), upper=np.full(12, 1.), max_step=.025,
                   previous_delta=np.zeros(12), acceleration_step=.05)
    _, immediate = cbf.project(q, nominal, checker, **options)
    assert not immediate['triggered'] and not calls
    preview = preview_action_chunk(sampler, q, actions, max_step=.025)
    safe, info = cbf.project(q, nominal, checker, action_preview=preview, **options)
    assert calls == [True] and info['triggered'] and info['success']
    assert info['qp_solve_calls'] == cbf.runtime_stats['qp_solve_calls'] == 1
    assert cbf.runtime_stats['project_calls'] == 2
    assert info['timing_s']['qp_solver'] > 0.
    assert info['preview_horizon'] == 10
    assert info['preview_first_obstacle_risk_step'] > 0
    assert info['preview_first_obstacle_intrusion_step'] > 0
    assert info['preview_min_obstacle_clearance_m'] < 0.
    assert info['predicted_obstacle_penetration_m'] == 0.
    np.testing.assert_allclose(safe, nominal, atol=1e-7)
    assert checker.edge_safe(q, q+safe)


def test_chunk_preview_without_future_risk_keeps_qp_inactive():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.8, 0., 0.]), np.eye(3), np.array([.03, .1, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12)
    actions = np.zeros((3, 14)); actions[:, 0] = [.02, .04, .06]
    preview = preview_action_chunk(sampler, q, actions, max_step=.025)
    cbf.before_solve = lambda: pytest.fail('Safe chunk must not invoke QP')
    _, info = cbf.project(q, actions[0, :12], checker, action_preview=preview,
                          lower=np.full(12, -1.), upper=np.full(12, 1.), max_step=.025,
                          previous_delta=np.zeros(12), acceleration_step=.003)
    assert not info['triggered']
    assert info['qp_solve_calls'] == cbf.runtime_stats['qp_solve_calls'] == 0
    assert cbf.runtime_stats['project_calls'] == 1
    assert info['timing_s']['sweep_validation'] > 0.
    assert info['preview_first_obstacle_risk_step'] is None
    assert info['preview_min_obstacle_clearance_m'] == pytest.approx(.71)


def test_chunk_tail_gripper_shape_activates_only_the_current_step_barrier():
    class GripperSampler(TwoPointSampler):
        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[q[6]+.03*grip[1], 0., 0.]]])

    sampler = GripperSampler()
    box = OrientedBox(np.array([.03, 0., 0.]), np.eye(3), np.array([.003, .1, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.005,
                          alpha=1., iterations=64)
    q = np.zeros(12)
    actions = np.zeros((2, 14)); actions[1, 13] = 1.
    preview = preview_action_chunk(sampler, q, actions, max_step=.025)
    safe, info = cbf.project(q, np.zeros(12), checker, action_preview=preview,
                             lower=np.full(12, -1.), upper=np.full(12, 1.), max_step=.025,
                             previous_delta=np.zeros(12), acceleration_step=.003)
    assert info['triggered'] and info['success']
    assert info['preview_first_obstacle_intrusion_step'] == 1
    # The tail's collision depth cannot be combined with the first step's
    # Jacobian to fabricate a required joint escape on a safe hold action.
    np.testing.assert_allclose(safe, 0., atol=1e-7)


def test_lerobot_cbf_projects_nominal_step_before_a_box():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    stops = []
    cbf.before_solve = lambda: stops.append(True)
    q = np.zeros(12); q[0] = .35
    path_q = np.stack((q, q.copy()))
    path_q[1, 0] = .40
    path = JointPath(path_q, np.zeros(2), np.full(12, -.1), np.full(12, 1.),
                     .1, .5, 10., 'toy', 7, {'kind': 'restore_start'})
    follower = PathFollower(path, avoid=True, detour_lower=path.lower, detour_upper=path.upper,
                            max_step=.05, tolerance=1e-6, cbf_projector=cbf)
    delta, state = follower.next_delta(q, checker)
    assert state == 'cbf' and stops == [True]
    assert delta[0] == pytest.approx(.05)
    assert follower.cbf_interventions == 0
    assert checker.edge_safe(q, q+delta)


def test_lerobot_cbf_redirects_an_unsafe_solver_step_without_shortening(monkeypatch):
    import obstacle_validation.cbf as cbf_module
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .35
    nominal = np.zeros(12); nominal[0] = .10
    monkeypatch.setattr(cbf_module, 'project_joint_delta_qp',
                        lambda *args, **kwargs: (nominal.copy(), {'success': True, 'triggered': True}))
    safe, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.),
                             upper=np.full(12, 1.), max_step=.1,
                             previous_delta=np.zeros(12), acceleration_step=.2)
    assert 0 <= safe[0] < .1
    assert info['success'] and info['projection_method'] == 'full_step_cbf_qp'
    assert np.max(np.abs(safe)) == pytest.approx(.1)
    assert 'fk_backtrack_fraction' not in info
    assert checker.edge_safe(q, q+safe)


def test_lerobot_cbf_can_brake_outside_nominal_acceleration_profile():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .4
    nominal = np.zeros(12); nominal[0] = .05
    previous = np.zeros(12); previous[0] = .025
    safe, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.),
                             upper=np.full(12, 1.), max_step=.05,
                             previous_delta=previous, acceleration_step=.003)
    assert info['success'] and info['acceleration_relaxed']
    assert safe[0] < .01
    assert checker.edge_safe(q, q+safe)


def test_lerobot_cbf_corrects_joints_for_commanded_gripper_shape():
    class GripperPointSampler:
        left_link_names = ('left',)
        points_per_link = 1

        def link_points(self, q, grip):
            return np.asarray([[[3., 3., 3.]], [[q[6] + .03*grip[1], 0., 0.]]])

    sampler = GripperPointSampler()
    q = np.zeros(12)
    measured_grip = np.zeros(2)
    target_grip = np.array([0., 1.])
    box = OrientedBox(np.array([.03, 0., 0.]), np.eye(3), np.array([.003, .1, .1]))
    checker = CollisionChecker(sampler, measured_grip, [box], margin=.01)
    cbf = RecordedPathCBF(sampler, measured_grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    nominal = np.zeros(12)
    nominal[6] = -.011

    def coupled_sweep(delta):
        from obstacle_validation.lerobot_replay import _motion_safe
        return _motion_safe(sampler, q, measured_grip, delta, target_grip, [box],
                            {'collision_margin_m': .01, 'inter_arm_margin_m': .035})

    corrected, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.),
                                  upper=np.full(12, 1.), max_step=.025,
                                  previous_delta=np.zeros(12), acceleration_step=.003,
                                  target_gripper=target_grip, swept_safe=coupled_sweep)
    assert info['success'] and info['predicted_obstacle_penetration_m'] > 0
    assert corrected[6] < nominal[6]
    assert coupled_sweep(corrected)


def test_lerobot_cbf_uses_future_points_to_recover_clearance():
    from obstacle_validation.lerobot_replay import _motion_safe

    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    grip = np.zeros(2)
    checker = CollisionChecker(sampler, grip, [box], margin=.01)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12)
    q[0] = .415  # 5 mm clearance: below the requested 10 mm, but outside the OBB.
    nominal = np.zeros(12)
    nominal[0] = .01  # Dataset action moves toward the obstacle.
    safety = {'collision_margin_m': .01, 'inter_arm_margin_m': .035}
    corrected, info = cbf.project(
        q, nominal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
        max_step=.025, previous_delta=np.zeros(12), acceleration_step=.003,
        swept_safe=lambda step: _motion_safe(sampler, q, grip, step, grip,
                                             [box], safety, True))
    assert info['current_state_below_clearance'] and info['success']
    assert corrected[0] < 0.  # QP changes the joint action to move outward.
    assert _motion_safe(sampler, q, grip, corrected, grip, [box], safety, True)


def test_lerobot_cbf_preserves_full_step_when_recovering_a_deficient_margin():
    from obstacle_validation.lerobot_replay import _motion_safe

    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    grip = np.zeros(2)
    checker = CollisionChecker(sampler, grip, [box], margin=.01)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12)
    q[0] = .415  # 5 mm current clearance; this tick can move at most 3 mm.
    nominal = np.zeros(12)
    nominal[0] = .003
    safety = {'collision_margin_m': .01, 'inter_arm_margin_m': .035}
    corrected, info = cbf.project(
        q, nominal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
        max_step=.003, previous_delta=np.zeros(12), acceleration_step=.003,
        swept_safe=lambda step: _motion_safe(sampler, q, grip, step, grip,
                                             [box], safety, True))
    future_clearance = checker.clearance(q+corrected)[0]
    assert info['success'] and info['projection_method'] == 'full_step_cbf_qp'
    assert np.max(np.abs(corrected)) == pytest.approx(.003)
    assert corrected[0] < 0.
    assert .005 < future_clearance < .01


@pytest.mark.parametrize("captured", [False, True])
def test_lerobot_cbf_recovers_voxel_margin_without_shortening_or_entering_occupied_cell(captured):
    from obstacle_validation.lerobot_replay import _motion_safe
    from obstacle_validation.runtime import CapturedFrameObservation
    from safety_module.occupancy import LocalOccupancyGrid, ObservationGrid, ObservedBoxes

    grid = LocalOccupancyGrid([.42, -.01, -.01], np.eye(3), [1, 1, 1], .02)
    grid.update(timestamp_ns=1, raw_hits=[[.43, 0., 0.]], free_ids=np.empty(0, np.int64))
    observation = ObservationGrid(grid.snapshot())
    if captured:
        observation = CapturedFrameObservation(observation)
    box = OrientedBox(np.array([.43, 0., 0.]), np.eye(3), np.full(3, .01))
    boxes = ObservedBoxes([box], observation)
    sampler, grip = TwoPointSampler(), np.zeros(2)
    q = np.zeros(12); q[0] = .415
    nominal = np.zeros(12); nominal[0] = .003
    safety = {'collision_margin_m': .01, 'inter_arm_margin_m': .035}
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    lower, upper = np.zeros(12), np.zeros(12)
    lower[0], upper[0] = -1., 1.
    corrected, info = cbf.project(
        q, nominal, checker, lower=lower, upper=upper,
        max_step=.003, previous_delta=np.zeros(12), acceleration_step=.003,
        swept_safe=lambda step: _motion_safe(sampler, q, grip, step, grip, boxes, safety, True))
    assert info['success'] and info['recovery_step']
    assert corrected[0] == pytest.approx(-.003)
    assert np.max(np.abs(corrected)) == pytest.approx(.003)
    assert .005 < checker.clearance(q+corrected)[0] < .01
    assert not _motion_safe(sampler, q, grip, nominal, grip, boxes, safety, True)
    assert not _motion_safe(sampler, q, grip, corrected, grip, boxes, safety, False)


def test_lerobot_cbf_rejects_action_with_no_possible_clearance_progress():
    from obstacle_validation.lerobot_replay import _motion_safe

    class FixedPointSampler:
        left_link_names = ('left',)
        points_per_link = 1

        def link_points(self, q, grip):
            return np.asarray([[[.5, 0., 0.]], [[3., 3., 3.]]])

    sampler = FixedPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    grip = np.zeros(2)
    checker = CollisionChecker(sampler, grip, [box], margin=.01)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12)
    safe, info = cbf.project(q, np.zeros(12), checker, lower=np.full(12, -1.),
                             upper=np.full(12, 1.), max_step=.025,
                             previous_delta=np.zeros(12), acceleration_step=.003,
                             swept_safe=lambda step: _motion_safe(
                                 sampler, q, grip, step, grip, [box],
                                 {'collision_margin_m': .01, 'inter_arm_margin_m': .035}, True))
    assert not info['success']
    assert info['current_state_below_clearance']
    assert info['reason'] == 'no_safe_full_joint_step'


def test_lerobot_cbf_preserves_step_when_future_face_linearization_is_infeasible():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    cbf = RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025,
                          alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .35
    nominal = np.zeros(12); nominal[0] = .2  # crosses the box centre
    safe, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.),
                             upper=np.full(12, 1.), max_step=.2,
                             previous_delta=np.zeros(12), acceleration_step=.2)
    assert info['success']
    assert info['projection_method'] == 'full_step_cbf_qp'
    assert np.max(np.abs(safe)) == pytest.approx(.2)
    assert info['predicted_obstacle_penetration_m'] > 0
    assert 0 < safe[0] < .06
    assert checker.edge_safe(q, q+safe)


def test_repeated_avoidance_preserves_step_and_passes_obstacle_without_oscillation():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))]
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .4
    goal = q.copy(); goal[0] = .8
    previous = np.zeros(12)
    poses = [q.copy()]
    for _ in range(40):
        nominal = np.clip(goal-q, -.025, .025)
        checker = CollisionChecker(sampler, grip, boxes, margin=.01, inter_arm_margin=.035)
        delta, info = cbf.project(
            q, nominal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.), max_step=.025,
            previous_delta=previous, acceleration_step=.003, target_gripper=grip)
        assert info['success'] and checker.edge_safe(q, q+delta)
        assert np.max(np.abs(nominal))-1e-7 <= np.max(np.abs(delta)) <= .025+1e-7
        assert 'fk_backtrack_fraction' not in info
        q += delta
        poses.append(q.copy())
        previous = delta
        if np.max(np.abs(q-goal)) < 1e-6:
            break
    else:
        pytest.fail('Full steps must make a detour, rather than alternate along the same face')
    assert np.max(np.abs(np.asarray(poses)[:, 1])) > .16
    np.testing.assert_allclose(q, goal, atol=1e-6)


def test_no_full_step_within_joint_limits_fails_instead_of_creeping_or_using_null_joints():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))]
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .4
    nominal = np.zeros(12); nominal[0] = .025
    lower, upper = np.full(12, -1.), np.full(12, 1.)
    lower[0] = q[0]  # No retreat; the only tangential joint is locked.
    lower[1] = upper[1] = 0.
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    delta, info = cbf.project(q, nominal, checker, lower=lower, upper=upper,
                              max_step=.025, previous_delta=np.zeros(12), acceleration_step=.003)
    assert not info['success'] and info['reason'] == 'no_safe_full_joint_step'
    np.testing.assert_array_equal(delta, np.zeros(12))


def test_chunk_preview_keeps_detour_memory_local_until_a_waypoint_is_executed():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))]
    q = np.zeros(12); q[0] = .4
    actions = np.zeros((8, 14)); actions[:, 0] = .5
    optimized, info, cbf, _ = optimize_chunk(sampler, actions, boxes, q=q)
    assert info['success'] and optimized is not None
    assert np.any(info['waypoint_steps'][0]['detour_direction'])
    np.testing.assert_array_equal(cbf.detour_direction, np.zeros(12))
    waypoints = np.asarray(info['waypoints'])[:, :12]
    assert np.all(np.max(np.abs(np.diff(np.vstack((q, waypoints)), axis=0)), axis=1) <= .025+1e-7)
    # Each already-corrected target is accepted once, instead of repeatedly
    # chasing the nominal target with its original subdivision count.
    assert len(info['waypoints']) == len(actions)


def test_full_step_redirection_preserves_inter_arm_separation():
    class NearArmSampler(TwoPointSampler):
        def link_points(self, q, grip):
            return np.asarray([[[q[0], q[1], 0.]], [[.04, 0., 0.]]])

    sampler, grip = NearArmSampler(), np.zeros(2)
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q, nominal = np.zeros(12), np.zeros(12)
    nominal[0] = .025
    checker = CollisionChecker(sampler, grip, [], inter_arm_margin=.035)
    delta, info = cbf.project(q, nominal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
                              max_step=.025, previous_delta=np.zeros(12), acceleration_step=.003)
    assert info['success'] and info['inter_arm_constraints'] > 0
    assert np.max(np.abs(delta)) == pytest.approx(.025)
    assert checker.edge_safe(q, q+delta)


def test_full_step_reselects_detour_when_old_direction_reaches_a_joint_bound():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .03, .1]))]
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q = np.zeros(12); q[:2] = [.40999, .05]
    cbf.detour_direction[1] = 1.
    nominal = np.zeros(12); nominal[:2] = [.025, -.025]
    lower, upper = np.full(12, -1.), np.full(12, 1.)
    upper[1] = q[1]
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    delta, info = cbf.project(q, nominal, checker, lower=lower, upper=upper, max_step=.025,
                              previous_delta=np.zeros(12), acceleration_step=.003)
    assert info['success'] and info['detour_reselected']
    assert np.max(np.abs(delta)) == pytest.approx(.025)
    assert q[1]+delta[1] <= upper[1]+1e-7
    assert checker.edge_safe(q, q+delta)


def test_chunk_full_steps_stay_within_each_actions_tracking_bounds():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .03, .1]))]
    cbf = RecordedPathCBF(sampler, grip, collision_margin_m=.01,
                          inter_arm_margin_m=.035, trigger_margin_m=.025, alpha=1., iterations=64)
    q = np.zeros(12); q[0] = .4
    actions = np.zeros((8, 14)); actions[:, 0] = np.arange(1, 9)*.025+.4
    lower, upper = np.full((8, 12), -1.), np.full((8, 12), 1.)
    lower[:, 0] = np.linspace(.3, .45, len(actions))
    lower[:, 1], upper[:, 1] = -.06, np.linspace(.06, .05, len(actions))
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    optimized, info = cbf.project_chunk(
        q, grip, actions, checker, lower=lower, upper=upper, max_step=.025,
        previous_delta=np.zeros(12), acceleration_step=.003,
        swept_safe=lambda origin, opening, delta, target: checker.edge_safe(origin, origin+delta))
    assert info['success'] and optimized is not None
    previous = q
    for waypoint, source in zip(info['waypoints'], info['waypoint_action_indices']):
        point = np.asarray(waypoint[:12])
        assert np.all(point >= lower[source]-1e-7) and np.all(point <= upper[source]+1e-7)
        assert checker.edge_safe(previous, point)
        previous = point
    assert all(step['joint_step_rad'] <= .025+1e-7 and step['joint_optimization']
               for step in info['waypoint_steps'])
