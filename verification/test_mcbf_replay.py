"""MCBF replay: virtual goal evacuation, tangent progress and swept safety."""

import numpy as np
import pytest

from obstacle_validation.cbf import RecordedPathCBF
from obstacle_validation.core import CollisionChecker
from obstacle_validation.mcbf import evacuate_action_target
from real_scripts.real_cbf_qp import OrientedBox, project_joint_delta_qp
from verification.test_obstacle_validation import TwoPointSampler


def controller(sampler):
    return RecordedPathCBF(sampler, np.zeros(2), collision_margin_m=.01,
                           inter_arm_margin_m=.035, trigger_margin_m=.025,
                           alpha=1., iterations=64, modulation=True)


@pytest.mark.parametrize('space', ['joint', 'ee'])
@pytest.mark.parametrize('modulation', [False, True])
def test_qp_iteration_limit_returns_immediately_only_for_mcbf_direct_replay(monkeypatch, space, modulation):
    from types import SimpleNamespace
    from obstacle_validation import joint_chunk, cartesian_chunk

    calls = []
    def exhausted(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(x=np.full(24, np.nan), success=False, nit=4000,
                               message='maximum iterations reached',
                               minimum_tcp_jacobian_singular_value=.1)
    # Force the joint solve through OSQP, even for these obstacle-free goals.
    monkeypatch.setattr(joint_chunk, 'cho_solve', lambda *a: np.full(24, 100.))
    monkeypatch.setattr(joint_chunk, '_solve_sparse_qp', exhausted)
    monkeypatch.setattr(cartesian_chunk.CartesianChunkQP, 'solve', exhausted)
    class Sampler(TwoPointSampler):
        def tcp_transforms(self, q):
            return np.tile(np.eye(4), (2, 1, 1))
    sampler = Sampler()
    cbf = controller(sampler)
    cbf.modulation, cbf.optimization_space = modulation, space
    actions = np.zeros((2, 14)); actions[:, 0] = [.01, .02]
    result, info = cbf.project_chunk(
        np.zeros(12), np.zeros(2), actions, CollisionChecker(sampler, np.zeros(2), []),
        lower=np.full(12, -1.), upper=np.full(12, 1.), max_step=.025,
        previous_delta=np.zeros(12), acceleration_step=.05)
    assert result is None and not info['success']
    assert len(calls) == (1 if modulation else 8)
    assert info['reason'] == ('joint_chunk_iteration_limit' if modulation else 'joint_chunk_optimization_failed')
    assert info['failed_step']['solver_message'] == 'maximum iterations reached'


def test_mcbf_iteration_limit_retries_penalized_qp_without_action_release(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import osqp
    from obstacle_validation import lerobot_replay as replay
    from verification.replay_test_scene import run_with_selected_scene

    actions = np.zeros((3, 14)); actions[:, 0] = [.07, .14, .21]
    actions[:, 12:] = [[.2, .8], [.4, .6], [.6, .4]]
    starts = []
    class Policy:
        period = .05
        states = np.vstack((np.zeros((1, 14)), actions[:-1]))
        source_frames = np.arange(3)
        metadata = {'initial_state': np.zeros(14).tolist(),
                    'final_joint_state': actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = actions
        def infer(self, start, horizon):
            starts.append(start)
            return actions[start:start+horizon].copy()
    class Sampler(TwoPointSampler):
        calibration_digest = 'test'
        def __init__(self, **kwargs): pass
    original_solver = osqp.OSQP
    solve_calls = []
    class Solver:
        def __init__(self): self.delegate = original_solver()
        def setup(self, **kwargs): self.delegate.setup(**kwargs)
        def solve(self, **kwargs):
            solve_calls.append(1)
            assert len(solve_calls) < 500, "Replay must not loop behind an obsolete separating plane"
            if len(solve_calls) == 1:
                return SimpleNamespace(x=None, info=SimpleNamespace(status_val=7,
                                       status='maximum iterations reached'))
            return self.delegate.solve(**kwargs)
    monkeypatch.setattr(osqp, 'OSQP', Solver)
    monkeypatch.setattr(replay, 'DatasetActionPolicy', Policy)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    snapshot = {'coordinate_frame': 'left_base', 'status': 'ok', 'obbs': [
        {'center': [.035, 0., 0.], 'axes': np.eye(3).tolist(), 'half_sizes': [.005, .01, .1]}],
        'qpos': {'left': [0.]*6, 'right': [0.]*6}, 'gripper_position': {'left': 0., 'right': 0.}}
    monkeypatch.setattr(replay, 'read_snapshot', lambda *a, **kw: snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *a: None)
    args = replay.parser().parse_args([
        'replay', "--no-execute", '--dataset', str(tmp_path/'dataset'), '--output', str(tmp_path/'run'),
        '--mode', 'avoid', '--live', '--avoidance-controller', 'mcbf', '--speed', '.5',
        '--arrival-tolerance', '.001'])
    report = run_with_selected_scene(args, monkeypatch)
    assert report['completed'] and report['actions_executed'] == 3
    assert report['optimization_retries'] == 0
    assert starts[0] == 0 and 1 in starts and 2 in starts
    chunks = [json.loads(row) for row in (tmp_path/'run/chunks.jsonl').read_text().splitlines()]
    assert all(chunk['planned_horizon'] == 1 for chunk in chunks)
    assert chunks[0]['optimization_mode'] == 'ellipsoid_mcbf_feedback'
    np.testing.assert_array_equal(chunks[0]['optimized_actions'], actions[:1])
    rows = [json.loads(row) for row in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    released = [row for row in rows if row['phase'] == 'path' and row['step'] == 0]
    assert len(released) > 1
    assert released[0]['cbf']['distance_constraints_relaxed']
    assert not any(row['cbf']['avoidance_bypassed'] for row in released)
    for row in released:
        np.testing.assert_array_equal(row['optimized_action'], actions[0])
        assert np.max(np.abs(row['sent_joint_delta'])) <= .025+5e-5


def repair(sampler, action, boxes, **kwargs):
    return evacuate_action_target(sampler, action, boxes, lower=np.full(12, -1.),
                                  upper=np.full(12, 1.), margin=.01, **kwargs)


def test_safe_goal_is_preserved_exactly():
    action = np.linspace(0., .5, 14)
    target, info = repair(TwoPointSampler(), action, [])
    np.testing.assert_array_equal(target, action)
    assert info['success'] and not info['modified'] and info['iterations'] == 0


@pytest.mark.parametrize('angle', [0., np.pi/4])
def test_goal_inside_box_is_pushed_out_with_recomputed_normals_and_fixed_grippers(angle):
    axes = np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    box = OrientedBox(np.array([.5, 0., 0.]), axes, np.array([.12, .3, .3]))
    action = np.zeros(14); action[0], action[12:] = .52, [.2, .8]
    target, info = repair(TwoPointSampler(), action, [box])
    assert info['success'] and info['modified'] and info['iterations'] > 1
    assert CollisionChecker(TwoPointSampler(), action[12:], [box], margin=.01).state_safe(target[:12])
    np.testing.assert_array_equal(target[12:], action[12:])
    assert info['final_clearance_m'] > .01


def test_goal_push_exits_adjacent_voxel_union_without_reversing_at_internal_faces():
    boxes = [OrientedBox(np.array([x, 0., 0.]), np.eye(3), np.array([.05, .3, .3]))
             for x in (.5, .6, .7)]
    action = np.zeros(14); action[0] = .52
    target, info = repair(TwoPointSampler(), action, boxes)
    assert info['success'] and info['iterations'] >= 4
    assert target[0] > .76
    assert all(update['normal'][0] > 0 for update in info['normal_updates'])
    assert CollisionChecker(TwoPointSampler(), np.zeros(2), boxes, margin=.01).state_safe(target[:12])


def test_gripper_target_geometry_participates_in_goal_evacuation():
    class GripperSampler(TwoPointSampler):
        def link_points(self, q, grip):
            return np.array([[[3., 3., 3.]], [[q[6]+.03*grip[1], 0., 0.]]])
    box = OrientedBox(np.array([.03, 0., 0.]), np.eye(3), np.array([.01, .2, .2]))
    action = np.zeros(14); action[13] = 1.
    target, info = repair(GripperSampler(), action, [box])
    assert info['success'] and abs(target[6]) > .02
    np.testing.assert_array_equal(target[12:], [0., 1.])


def test_unreachable_goal_fails_and_invalidates_whole_chunk():
    sampler = TwoPointSampler()
    cbf = controller(sampler)
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.1, .3, .3]))
    actions = np.zeros((2, 14)); actions[:, 0] = [.1, .5]
    bounds = np.zeros((2, 12)); bounds[0, :2] = 1.
    bounds[1, 0] = .5  # No joint can move the second target out of its voxel.
    lower = -bounds.copy(); lower[1, 0] = .5
    result, info = cbf.project_chunk(
        np.zeros(12), np.zeros(2), actions, CollisionChecker(sampler, np.zeros(2), [box]),
        lower=lower, upper=bounds, max_step=.025, previous_delta=np.zeros(12),
        acceleration_step=.05, swept_safe=lambda *args: True)
    assert result is None and not info['success'] and info['failed_action_index'] == 1
    assert info['reason'] in ('target_normal_uncontrollable', 'target_push_joint_bounds_or_singularity')
    np.testing.assert_array_equal(cbf.detour_direction, np.zeros(12))


def test_iteration_limit_does_not_accept_an_occupied_goal():
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.3, .4, .4]))
    action = np.zeros(14); action[0] = .5
    target, info = repair(TwoPointSampler(), action, [box], iterations=1)
    assert target is None and not info['success'] and info['reason'] == 'target_push_iteration_limit'


def test_goal_push_reselects_normal_when_it_reaches_a_joint_limit():
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.1, .3, .3]))
    action = np.zeros(14); action[0] = .5
    lower, upper = np.zeros(12), np.zeros(12)
    lower[0], upper[0] = .3, .55
    target, info = evacuate_action_target(TwoPointSampler(), action, [box],
                                         lower=lower, upper=upper, margin=.01)
    assert info['success'], info
    assert target[0] < .39
    assert info['normal_updates'][0]['normal'][0] > 0
    assert info['normal_updates'][-1]['normal'][0] < 0


def test_tangent_constraint_is_enforced_even_without_obstacle_rows():
    row = np.array([0., 1.])
    delta, info = project_joint_delta_qp(np.array([.02, 0.]), np.zeros((1, 3, 2)), [],
                                         lower_delta=np.full(2, -.025), upper_delta=np.full(2, .025),
                                         guidance=(row, .01))
    assert info['success'] and delta[1] >= .01-1e-7


def test_mcbf_chunks_pass_head_on_obstacle_and_preserve_side_across_prefixes():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    cbf = controller(sampler)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))]
    goal = np.zeros((1, 14)); goal[0, 0] = .8
    q = np.zeros(12); q[0] = .4
    poses, modulations = [q.copy()], []
    for _ in range(35):
        checker = CollisionChecker(sampler, grip, boxes, margin=.01)
        planned, info = cbf.project_chunk(
            q, grip, goal, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
            max_step=.025, previous_delta=np.zeros(12), acceleration_step=.05,
            swept_safe=lambda origin, opening, delta, target: checker.edge_safe(origin, origin+delta))
        assert info['success'], info
        for waypoint, step in zip(info['waypoints'], info['waypoint_steps']):
            next_q = np.asarray(waypoint[:12])
            assert checker.edge_safe(q, next_q)
            assert np.max(np.abs(next_q-q)) <= .025+1e-7
            if 'modulation' in step:
                modulation = step['modulation']
                assert np.dot(modulation['joint_direction'], next_q-q) >= modulation['minimum_joint_progress_rad']-1e-7
                modulations.append(modulation)
            q = next_q
            poses.append(q.copy())
        if np.max(np.abs(q-goal[0, :12])) < 1e-6:
            break
    assert modulations and q[0] == pytest.approx(.8, abs=1e-6)
    assert max(abs(pose[1]) for pose in poses) > .16
    assert min(pose[1] for pose in poses) >= -1e-7


def test_chunk_reports_repaired_goals_and_keeps_original_input():
    sampler = TwoPointSampler()
    box = OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1]))
    actions = np.zeros((2, 14)); actions[:, 0] = [.48, .5]
    before = actions.copy()
    q = np.zeros(12); q[0] = .35
    checker = CollisionChecker(sampler, np.zeros(2), [box], margin=.01)
    planned, info = controller(sampler).project_chunk(
        q, np.zeros(2), actions, checker, lower=np.full(12, -1.), upper=np.full(12, 1.),
        max_step=.025, previous_delta=np.zeros(12), acceleration_step=.05,
        swept_safe=lambda origin, opening, delta, target: checker.edge_safe(origin, origin+delta))
    assert info['success'], info
    assert len(info['target_repairs']) == 2 and all(item['modified'] for item in info['target_repairs'])
    assert all(item['final_clearance_m'] > .01 for item in info['target_repairs'])
    assert all(checker.state_safe(action[:12]) for action in planned)
    np.testing.assert_array_equal(actions, before)


def test_mcbf_uses_opposite_tangent_when_first_side_is_blocked():
    sampler, grip = TwoPointSampler(), np.zeros(2)
    cbf = controller(sampler)
    boxes = [OrientedBox(np.array([.5, 0., 0.]), np.eye(3), np.array([.08, .15, .1])),
             OrientedBox(np.array([.38, .04, 0.]), np.eye(3), np.array([.03, .02, .1]))]
    q = np.zeros(12); q[0] = .4
    nominal = np.zeros(12); nominal[0] = .025
    checker = CollisionChecker(sampler, grip, boxes, margin=.01)
    lower = np.full(12, -1.); lower[0] = q[0]
    delta, info = cbf.project(q, nominal, checker, lower=lower, upper=np.full(12, 1.),
                              max_step=.025, previous_delta=np.zeros(12), acceleration_step=.05)
    assert info['success'] and checker.edge_safe(q, q+delta), info
    assert info['detour_reselected'] and delta[1] < 0
    modulation = info['modulation']
    assert np.dot(modulation['joint_direction'], delta) >= modulation['minimum_joint_progress_rad']-1e-7


def test_mcbf_replay_entrypoint_reaches_original_goals_and_reports_adaptive_control(tmp_path, monkeypatch):
    import json
    from obstacle_validation import lerobot_replay as replay

    actions = np.zeros((2, 14)); actions[:, 0] = [.10, .15]
    class Policy:
        period = .05
        states = np.zeros((2, 14))
        source_frames = np.arange(2)
        metadata = {'initial_state': np.zeros(14).tolist(), 'final_joint_state': actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = actions
        def infer(self, start, horizon): return actions[start:start+horizon].copy()
    class Sampler(TwoPointSampler):
        calibration_digest = 'test'
        def __init__(self, **kwargs): pass
    snapshot = {'coordinate_frame': 'left_base', 'status': 'ok', 'obbs': [
        {'center': [.05, 0., 0.], 'axes': np.eye(3).tolist(), 'half_sizes': [.01, .02, .1]}],
        'qpos': {'left': [0.]*6, 'right': [0.]*6}, 'gripper_position': {'left': 0., 'right': 0.}}
    monkeypatch.setattr(replay, 'DatasetActionPolicy', Policy)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'read_snapshot', lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *args: None)
    monkeypatch.setattr(replay.LiveSceneReader, 'start', lambda *args: None)
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    args = replay.parser().parse_args(['replay', "--no-execute", '--dataset', str(tmp_path/'dataset'),
                                      '--output', str(tmp_path/'run'), '--mode', 'avoid', '--live',
                                      '--avoidance-controller', 'mcbf', '--final-tolerance', '.001'])
    from verification.replay_test_scene import run_with_selected_scene
    report = run_with_selected_scene(args, monkeypatch)
    assert report['completed'] and report['execution'] == 'virtual'
    assert report['avoidance_controller'] == 'mcbf' and report['actions_executed'] == 2
    chunks = [json.loads(row) for row in (tmp_path/'run/chunks.jsonl').read_text().splitlines()]
    assert chunks[0]['avoidance_controller'] == 'mcbf'
    assert chunks[0]['optimization_mode'] == 'ellipsoid_mcbf_feedback'
    assert all(chunk['target_repairs'] == [] for chunk in chunks)
    assert report['controller_variant'] == 'ellipsoid_mcbf'
    assert report['effective_execute_horizon'] == 1 and report['post_optimization_checks']
    assert report['final_joint_error_rad'] <= .001
