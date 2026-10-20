"""VLSA Eq. (9), analytic derivatives and augmented whole-body QP controls."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_validation.ellipsoid_geometry import Ellipsoid, vlsa_barrier, separation_gradient
from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF
from verification.test_ellipsoid_replay import VolumeSampler, sphere


def sqrt_shape(shape):
    values, axes = np.linalg.eigh(shape)
    return (axes*np.sqrt(values)) @ axes.T


@pytest.mark.parametrize("angle", [0., .4, 1.2])
def test_barrier_matches_paper_virtual_surface_point_formula(angle):
    rotation = Rotation.from_rotvec([.2, angle, -.3]).as_matrix()
    a = Ellipsoid(np.array([.2, -.3, .4]), rotation @ np.diag([.2, .08, .04])**2 @ rotation.T)
    b = Ellipsoid(np.array([-.15, .1, .2]), np.diag([.1, .12, .07])**2)
    n = np.array([.5, -.2, .3]); n /= np.linalg.norm(n)
    result = vlsa_barrier(a, b, n)
    aa, ab = sqrt_shape(a.shape), sqrt_shape(b.shape)
    # Paper's unit surface point ps; the plane faces a->b, opposite our n.
    ps = -aa @ n/np.linalg.norm(aa @ n)
    mapped = np.linalg.solve(aa, ps)
    expected = (-np.linalg.norm(ab @ mapped)+(b.center-a.center) @ mapped-1)/np.linalg.norm(mapped)
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.center_distance == pytest.approx(np.linalg.norm(a.center-b.center))
    np.testing.assert_allclose(result.center_delta, a.center-b.center)


def test_direction_gradient_matches_sphere_retraction_finite_difference():
    a = Ellipsoid(np.array([.1, .2, .3]), np.diag([.2, .07, .1])**2)
    b = sphere([.5, -.1, .4], .09)
    value = vlsa_barrier(a, b, [1, 2, -1])
    np.testing.assert_allclose(value.tangent_basis.T @ value.tangent_basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(value.normal @ value.tangent_basis, 0, atol=1e-14)
    eps = 1e-6
    for k in range(2):
        shift = value.tangent_basis[:, k]*eps
        actual = (vlsa_barrier(a, b, value.normal+shift).value
                  -vlsa_barrier(a, b, value.normal-shift).value)/(2*eps)
        assert value.direction_gradient[k] == pytest.approx(actual, abs=1e-9)


def test_physical_gradient_includes_rotated_ellipsoid_center_offset():
    n = np.array([1., -.3, .2]); n /= np.linalg.norm(n)
    obstacle = sphere([.5, .2, .1], .08)
    def body(q):
        r = Rotation.from_rotvec([0, 0, q]).as_matrix()
        # The link/TCP origin is not the collision envelope's center.
        return Ellipsoid(np.array([q/10, 0, 0])+r @ np.array([.12, .03, .02]),
                         r @ np.diag([.2, .04, .08])**2 @ r.T)
    q, eps = .4, 1e-6
    a, plus, minus = body(q), body(q+eps), body(q-eps)
    dc = ((plus.center-minus.center)/(2*eps))[:, None]
    ds = ((plus.shape-minus.shape)/(2*eps))[..., None]
    derivative = separation_gradient(a, n, dc, ds)[0]
    expected = (vlsa_barrier(plus, obstacle, n).value-vlsa_barrier(minus, obstacle, n).value)/(2*eps)
    assert derivative == pytest.approx(expected, abs=1e-9)
    assert abs(derivative-n[0]/10) > .01


def test_nonzero_center_distance_does_not_certify_nonintersection():
    a, b = sphere([0, 0, 0], .2), sphere([.1, 0, 0], .2)
    value = vlsa_barrier(a, b, [-1, 0, 0])
    assert value.center_distance == pytest.approx(.1)
    assert value.value == pytest.approx(-.3)


@pytest.mark.parametrize("failure", [False, True])
def test_qp_controls_virtual_plane_with_no_gripper_or_extra_joint_commands(monkeypatch, failure):
    cbf = EllipsoidMCBF(VolumeSampler(), margin=.005, inter_arm_margin=.035, period=.05)
    boxes = []
    cbf.update_scene(boxes)
    cbf.obstacles = [sphere([.065, .04, 0], .02, "obstacle")]
    cbf.velocities = np.zeros((1, 3))
    key = ("environment", "left", "obstacle")
    cbf.normals[key] = np.array([-1., 0., 0.])  # Retained valid, nonoptimal plane.
    monkeypatch.setattr(cbf, "_guidance", lambda *a: (np.zeros(12), {"cached": True}))
    if failure:
        import osqp
        class Failed:
            def setup(self, **kwargs): pass
            def solve(self, **kwargs):
                return SimpleNamespace(x=None, info=SimpleNamespace(status_val=7, status="forced failure"))
        monkeypatch.setattr(osqp, "OSQP", Failed)
    q = np.zeros(12); q[6] = .8
    action = np.r_[q, [.3, .8]][None]; action[0, 0] = .2
    _, info = cbf.plan(q, np.zeros(2), action, boxes, lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    detail = info["steps"][0]
    assert info["joint_variables"] == 12 and info["gripper_variables"] == 0
    assert info["virtual_direction_variables"] == 2
    assert detail["barrier_model"] == "vlsa_relative_centers_support_plane"
    assert np.linalg.norm(detail["virtual_direction_delta"]) > .01
    assert np.linalg.norm(cbf.normals[key]) == pytest.approx(1.)
    assert not np.allclose(cbf.normals[key], [-1, 0, 0])
    waypoint = np.array(info["waypoints"][0])
    assert waypoint.shape == (14,)
    assert 0 < waypoint[0] <= .040001
    np.testing.assert_array_equal(waypoint[12:], action[0, 12:])
    row = detail["violated_pairs"][0]
    assert row["center_distance_m"] == pytest.approx(np.hypot(.065, .04))
    assert row["distance_m"] == pytest.approx(.035)
    assert row["barrier_h_m"] == pytest.approx(.03)
    # Nominal prediction must not silently maximize a different virtual plane.
    assert row["predicted_distance_m"] == pytest.approx(-.005)
    np.testing.assert_allclose(np.array(row["first_center_m"])-row["second_center_m"], row["relative_centers_m"])
    assert np.isfinite(row["predicted_barrier_after_m"])
    assert (detail["fallback"] == "bounded_penalty_optimizer") == failure
