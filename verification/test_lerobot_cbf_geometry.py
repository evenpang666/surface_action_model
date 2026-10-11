"""CBF safety checks used by LeRobot action replay and restoration."""

import numpy as np
import pytest

from obstacle_validation.core import CollisionChecker, JointPath, PathFollower
from obstacle_validation.cbf import RecordedPathCBF
from real_scripts.real_cbf_qp import OrientedBox
from verification.test_obstacle_validation import TwoPointSampler


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


def test_lerobot_cbf_backtracks_solver_step_before_obstacle(monkeypatch):
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
    assert info['success'] and info['fk_backtrack_fraction'] < 1.
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


def test_lerobot_cbf_relaxes_recovery_when_one_tick_cannot_clear_margin():
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
    assert info['success'] and info['recovery_alpha'] == .5
    assert corrected[0] < 0.
    assert .005 < future_clearance < .01


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
    assert info['reason'] == 'recovery_future_or_sweep_unsafe'


def test_lerobot_cbf_uses_verified_prefix_when_future_face_linearization_is_infeasible():
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
    assert info['projection_method'] == 'verified_nominal_prefix'
    assert info['predicted_obstacle_penetration_m'] > 0
    assert 0 < safe[0] < .06
    assert checker.edge_safe(q, q+safe)
