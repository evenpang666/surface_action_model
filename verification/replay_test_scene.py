"""Selected-scene inputs for analytic replay tests without cameras or files.

These tests exercise action sequencing/optimization with synthetic OBBs. Real
selection validation and exact voxel filtering are tested in test_manual_voxel_replay
and test_replay_voxblox; do not use this helper for input-contract tests.
"""
from pathlib import Path

import numpy as np

from obstacle_validation import lerobot_replay as replay


def run_with_selected_scene(args, monkeypatch):
    if args.mode == "avoid" and not args.manual_selection:
        class SyntheticSelection:
            def __init__(self, path): self.path = path
            def __call__(self, snapshot): return snapshot
        args.manual_selection = Path(args.output)/"selection.json"
        args.obstacle_source = "external"
        monkeypatch.setattr(replay, "ManualVoxelFilter", SyntheticSelection)
        if not args.live:
            # Former offline sequence tests now use a synthetic selected live
            # scene. Their robot/snapshot sequencing tests provide their own
            # transport whenever --live was already requested.
            if not getattr(replay.DualUR7eSurfacePointSampler, "calibration_digest", None):
                monkeypatch.setattr(replay.DualUR7eSurfacePointSampler, "calibration_digest", "test", raising=False)
            monkeypatch.setattr(replay, "read_snapshot", lambda *a, **kw: {
                "coordinate_frame": "left_base", "status": "ok", "obbs": [],
                "qpos": {"left": np.zeros(6).tolist(), "right": np.zeros(6).tolist()},
                "gripper_position": {"left": 0., "right": 0.}})
            monkeypatch.setattr(replay, "validate_calibration_identity", lambda *a: None)
        monkeypatch.setattr(replay.LiveSceneReader, "start", lambda *a: None)
        monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    return replay.run_replay(args)
