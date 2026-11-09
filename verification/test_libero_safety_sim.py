"""Optional real MuJoCo/Panda adapter smoke tests using bundled Lift assets.

This is not a LIBERO-Safety score and does not require its 10.7 GB assets.
Select the official robosuite-1.4 fork on PYTHONPATH and MUJOCO_GL=egl.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

suite = pytest.importorskip("robosuite")

from obstacle_validation.libero_safety_geometry import CAMERAS, PandaGeometry, capture, SimWorkspaceTSDF
from obstacle_validation.libero_safety import parse_args, rollout, safe_action
from obstacle_validation.libero_safety_recording import TrainingRecorder
from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig


@pytest.fixture
def panda():
    env = suite.make("Lift", robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=list(CAMERAS.values()),
        camera_heights=96, camera_widths=96, camera_depths=True)
    env.parsed_problem = {"constraints": []}  # Lift has no BDDL; explicit test fixture.
    class Adapter:
        def __init__(self):
            self.env = env
        @property
        def sim(self):
            return env.sim
        @property
        def robots(self):
            return env.robots
        def reset(self):
            return env.reset()
        def set_init_state(self, state):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env._update_observables(force=True)
            return env._get_observations(force_update=True)
        def step(self, action):
            obs, reward, _, info = env.step(action)
            return obs, reward, env._check_success(), info
    try:
        adapter = Adapter()
        adapter.reset()
        yield adapter
    finally:
        env.close()


def test_real_panda_voxblox_pairs_and_shared_mcbf_preserve_physics_state(panda):
    from real_scripts.voxblox_obstacle_map import require_voxblox
    require_voxblox()
    from oa_vla.geometry import original_obb_coordinates
    obs = panda.env._get_observations(force_update=True)
    geometry = PandaGeometry(panda); geometry.bind()
    initial = panda.sim.get_state().flatten().copy()
    bundle, calibrations = capture(panda, obs, CAMERAS, 0)
    recording = TrainingRecorder(panda, CAMERAS, ["cube"], 3)
    recording.add(bundle)
    assert recording.rows[0]["valid"].all()
    mapping = SimWorkspaceTSDF(calibrations, [-.6, .6, -.6, .6, .75, 1.6], .03, 2_000_000)
    mapped = mapping.update(bundle, calibrations, geometry, CAMERAS)
    pair = {"pair_id": 0, "score": .99}
    for i, view in enumerate(("front", "side")):
        boxes, _ = original_obb_coordinates(recording.rows[0]["boxes"][i], 96, 96)
        pair[view] = dict(obb=boxes[0].tolist())
    # Segmentation supplies synthetic detector output for this test ONLY.
    prediction = dict(model_type="oa-vla", detection_status="ok", pair_id_scope="current_prediction",
        detection_box_format="cxcywh_angle_deg_original_pixels", objects=[pair])
    planner = EllipsoidMCBF(None, margin=.005, inter_arm_margin=0, period=.05, robot=geometry,
        joint_groups=(6,), grip_dim=1, config=EllipsoidMCBFConfig(max_control_step_rad=1.))
    args = SimpleNamespace(minimum_object_score=.5, obb_padding_pixels=2., max_distance_slack=1e-5)
    action, detail = safe_action(np.zeros(7), prediction, mapped, geometry, planner, args)
    assert detail["paired_geometry"]["objects"]["0"]["occupied_voxels"] > 0
    assert action[-1] == 0  # incremental gripper hold, never reinterpreted as a target
    np.testing.assert_array_equal(initial, panda.sim.get_state().flatten())


def test_panda_rollout_observes_again_after_each_tick(panda, tmp_path):
    seen = []
    def predict(rgb, state, prompt, **kwargs):
        seen.append(state.copy())
        return dict(actions=np.array([np.zeros(7), np.ones(7)]))
    # Network protocol has separate real loopback tests; injection here isolates
    # actual rendering/physics from hosts where socket and GPU permissions differ.
    args = parse_args(["--libero-root", "/tmp/unused-lift-fixture", "--mode", "baseline", "--allow-test-policy",
                       "--max-steps", "2", "--warmup-steps", "1", "--no-video", "--max-result-age", "10"])
    initial = panda.sim.get_state().flatten()
    report = rollout(panda, SimpleNamespace(infer=predict), args, CAMERAS, tmp_path/"episode",
                     task=SimpleNamespace(name="lift", language="lift cube"), initial_state=initial)
    assert report["completed"] and report["steps"] == 2 and report["holds"] == 0
    assert len(seen) == 2 and report["physics_samples"] > 3
    records = [json.loads(line) for line in (tmp_path/"episode/steps.jsonl").read_text().splitlines()]
    assert [row["frame_id"] for row in records] == [0, 1]
    assert records[1]["context"]["capture_ns"] > records[0]["context"]["capture_ns"]
    assert all(np.array_equal(row["action"], np.zeros(7)) for row in records)

