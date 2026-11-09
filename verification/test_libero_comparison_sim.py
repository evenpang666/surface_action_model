"""Opt-in real Panda tests. These are fixtures, never benchmark scores.

Run with RUN_LIBERO_COMPARISON_SIM=1, robosuite 1.4, NumPy<2, MUJOCO_GL=egl.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIBERO_COMPARISON_SIM") != "1",
    reason="Explicit real-rendering fixture opt-in required")


@pytest.fixture
def panda():
    from obstacle_validation.libero_comparison import fixture_env, parse_args
    args = parse_args(["--fixture", "--allow-test-policy", "--output", "/tmp/unused", "--resolution", "96"])
    env = fixture_env(args)
    try:
        env.reset()
        yield env
    finally:
        env.close()


def test_same_pass_depth_segmentation_and_kinematics_restore(panda):
    from obstacle_validation.libero_safety_geometry import capture, CAMERAS
    from obstacle_validation.libero_comparison_geometry import LabelDepthPerception, JointPandaGeometry
    obs = panda.env._get_observations(force_update=True)
    bundle, calibration = capture(panda, obs, CAMERAS, 0)
    perception = LabelDepthPerception(panda, CAMERAS, ["safety_obstacle"], padding=.015)
    box = perception.update(bundle, calibration)["safety_obstacle"]
    # Regression: mixing RGB-pass depth and segmentation-pass masks created
    # a spurious 45cm box extending onto the robot/table.
    np.testing.assert_allclose(box.center, [.15, .12, 1.], atol=.035)
    assert box.half_sizes.max() < .11
    geometry = JointPandaGeometry(panda, spacing=.08); geometry.bind()
    before = panda.sim.get_state().flatten().copy()
    points, jac = geometry.sample_differential()
    assert jac.shape == (len(points), 3, 7)
    body, dc, ds = geometry.eef_differential()
    assert dc.shape == (3, 7) and ds.shape == (3, 3, 7)
    assert np.linalg.eigvalsh(body.shape).min() > 0
    np.testing.assert_array_equal(before, panda.sim.get_state().flatten())


def test_grasp_samples_move_rigidly_with_eef_and_release(panda):
    from obstacle_validation.libero_comparison_geometry import JointPandaGeometry
    geometry = JointPandaGeometry(panda, spacing=.08); geometry.bind()
    base_count = len(geometry.sample_points(np.zeros(7)))
    # Fixture injects a known local held envelope; grasp-trigger logic has a
    # separate contact-sensor test, so no trained grasp policy is needed here.
    geometry.held["cube"] = np.array([[0., 0., .1], [.02, 0., .1]])
    assert geometry.held_sample_count == 2
    delta = np.r_[.02, np.zeros(6)]
    samples = geometry.sample_points(delta)
    with geometry.at(delta):
        p, r = geometry.eef_pose()
        expected = geometry.held["cube"] @ r.T+p
    assert len(samples) == base_count+2
    np.testing.assert_allclose(samples[-2:], expected)
    geometry.update_grasp(-1., {}, {})
    assert geometry.held_sample_count == 0
