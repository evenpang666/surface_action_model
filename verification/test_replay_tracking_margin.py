"""Tracking boundaries with imperfect physical feedback and action transitions."""

import json

import numpy as np
import pytest

from obstacle_validation import lerobot_replay as replay


@pytest.mark.parametrize("case", ["feedback_margin", "advance_guard", "tracking_excursion"])
def test_avoid_tracking_limit_with_feedback_error(tmp_path, monkeypatch, case):
    reference_states = np.zeros((2, 14)); reference_states[1, 9] = -.003
    dataset_actions = np.zeros((2, 14)); dataset_actions[:, 9] = [-.003, -.006]
    state = {"q": np.zeros(12), "path": False, "near_target_sends": 0}

    class Policy:
        period = .05
        states = reference_states
        source_frames = np.arange(2)
        metadata = {"initial_state": np.zeros(14).tolist(),
                    "final_joint_state": dataset_actions[-1, :12].tolist()}
        def __init__(self, root): self.actions = dataset_actions
        def infer(self, start, horizon): return self.actions[start:start+horizon].copy()

    class Commander:
        def __init__(self, *args): pass
        def read_state(self): return state['q'].copy(), np.zeros(2)
        def send(self, q, delta):
            target = q+delta
            if state['path'] and case == 'feedback_margin':
                target[9] += .0006
            if state['path'] and case == 'advance_guard' and abs(target[9]-.1169) < 1e-8:
                if state['near_target_sends'] == 0:
                    target[9] += .0006
                state['near_target_sends'] += 1
            state['q'] = target
        def stop(self): pass
        def close(self): pass

    class Grippers:
        def __init__(self, *args, **kwargs): pass
        def send_if_changed(self, target): return [False, False]
        def close(self): pass

    class Sampler:
        tcp_transform_verified = {'left': True, 'right': True}
        calibration_digest = 'test'
        left_link_names, points_per_link = ('left',), 1
        def __init__(self, **kwargs): pass
        def link_points(self, q, grip): return np.array([[[3., 3., 3.]], [[4., 4., 4.]]])

    def snapshot(*args, **kwargs):
        if case == 'tracking_excursion' and not state['path']:
            state['q'][9] = .121
        state['path'] = True
        return {'coordinate_frame': 'left_base', 'status': 'ok', 'obbs': []}

    def project(self, q, grip, actions, checker, **kwargs):
        optimized = actions.copy()
        # Exercise a route at the planner's permitted edge. The second case
        # emulates a previously prepared route before a next-state check.
        optimized[0, 9] = kwargs['upper'][0, 9] if case == 'feedback_margin' else .1169
        steps = [{'success': True, 'triggered': True} for _ in actions]
        return optimized, {'success': True, 'steps': steps, 'waypoint_steps': steps,
                           'waypoints': optimized.tolist(),
                           'waypoint_action_indices': list(range(len(actions))),
                           'triggered_action_indices': [0], 'modified_action_indices': [0]}

    monkeypatch.setattr(replay, 'DatasetActionPolicy', Policy)
    monkeypatch.setattr(replay, 'DualCommander', Commander)
    monkeypatch.setattr(replay, 'DualGripperCommander', Grippers)
    monkeypatch.setattr(replay, 'DualUR7eSurfacePointSampler', Sampler)
    monkeypatch.setattr(replay, 'read_snapshot', snapshot)
    monkeypatch.setattr(replay, 'validate_calibration_identity', lambda *args: None)
    monkeypatch.setattr(replay.LiveSceneReader, 'start', lambda *args: None)
    monkeypatch.setattr(replay.RecordedPathCBF, 'project_chunk', project)
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    args = replay.parser().parse_args(['replay', '--dataset', str(tmp_path/'dataset'),
                                       '--mode', 'avoid', '--avoidance-controller', 'cbf', '--live', '--execute',
                                       '--execute-ack', replay.EXECUTE_ACK,
                                       '--output', str(tmp_path/'run')])
    from verification.replay_test_scene import run_with_selected_scene
    report = run_with_selected_scene(args, monkeypatch)
    assert report['completed'] and report['actions_executed'] == 2
    rows = [json.loads(line) for line in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    path = [row for row in rows if row['phase'] == 'path']
    assert all(np.max(np.abs(np.asarray(row['qpos'])-reference_states[row['step']+1, :12])) <= .12
               for row in path if row['action_advanced'] and row['step']+1 < len(reference_states))
    assert report['tracking_error_limit_rad'] == .12
    assert report['planning_tracking_room_rad'] == pytest.approx(.0975, abs=1e-6)
    assert all(row['next_tracking_error_rad'] is None or row['next_tracking_error_rad'] <= .12
               for row in path if row['action_advanced'])
    if case == 'advance_guard':
        assert any(not row['action_advanced'] and row['next_tracking_error_rad'] > .12
                   for row in path if row['next_tracking_error_rad'] is not None)
    elif case == 'tracking_excursion':
        assert report['max_tracking_error_rad'] >= .121
        assert report['last_tracking_excursion']['joint_index'] == 9
        assert report['failure_context'] is None
    else:
        assert report['max_tracking_error_rad'] < .12
