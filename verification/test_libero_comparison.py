"""Numerical safety, collision accounting and independent method contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation.ellipsoid_geometry import Ellipsoid
from obstacle_validation.libero_comparison import ContactMonitor, parse_args, validate_socket
from obstacle_validation.libero_metrics import aggregate, export_comparison
from obstacle_validation.oa_vla_actions import ObservationHold
from obstacle_validation.safety_filters import (PoissonField, VLSAFilter, VLPSAFilter,
                                               box_surface, obstacle_grid, project_qp)
from real_scripts.real_cbf_qp import OrientedBox


def test_qp_keeps_safety_hard_and_only_clf_soft():
    x, detail = project_qp([1.], [[-1.]], [0.], [-1.], [1.], clf=([-1.], -.5, 1000.))
    assert x[0] <= 1e-7
    assert detail["clf_slack"] >= .5-1e-7
    with pytest.raises(ObservationHold):
        project_qp([0.], [[1.]], [2.], [-1.], [1.])


def test_poisson_discrete_pde_and_gradient():
    occupied = np.zeros((15, 13, 11), bool)
    occupied[8, 3:10, 3:8] = True
    field = PoissonField.solve(occupied, np.zeros(3), .05)
    h = field.values
    laplace = sum(np.roll(h, 1, a)+np.roll(h, -1, a)-2*h for a in range(3))/.05**2
    free = np.ones(h.shape, bool); free[[0, -1], :, :] = False
    free[:, [0, -1], :] = False; free[:, :, [0, -1]] = False
    free &= ~occupied
    np.testing.assert_allclose(laplace[free], -1., atol=2e-5)
    point = np.array([[.233, .217, .193]])
    _, grad, valid = field.query(point)
    assert valid.all() and field.residual < 1e-7
    for axis in range(3):
        d = np.eye(3)[axis][None]*1e-6
        derivative = (field.query(point+d)[0]-field.query(point-d)[0])/2e-6
        assert grad[0, axis] == pytest.approx(derivative[0], abs=1e-9)
    assert not field.query([[-1., 0, 0]])[2][0]


def test_poisson_buffer_and_solve_failure():
    occupied = np.zeros((15, 15, 15), bool); occupied[7, 7, 7] = True
    field = PoissonField.solve(occupied, np.zeros(3), .05, buffer=.11)
    assert field.values[9, 7, 7] <= 0 < field.values[10, 7, 7]
    with pytest.raises(ObservationHold):
        PoissonField.solve(occupied, np.zeros(3), .05, maxiter=1)


def test_voxelization_keeps_subvoxel_obstacle_and_surface_covers_faces():
    box = OrientedBox(np.array([.25, .25, .25]), np.eye(3), np.full(3, .005))
    occupied, _ = obstacle_grid([box], np.array([[0, 1]]*3), .1)
    assert occupied[2:4, 2:4, 2:4].all()
    points = box_surface([.1, .2, .3], .04)
    assert np.allclose(np.max(np.abs(points)/[.1, .2, .3], axis=1), 1.)


class ToyGeometry:
    held_sample_count = 0
    def eef_body(self, delta):
        return Ellipsoid(np.array([delta[0], 0, 0]), np.eye(3)*.02**2)
    def eef_differential(self):
        dc = np.zeros((3, 7)); dc[0, 0] = 1
        return self.eef_body(np.zeros(7)), dc, np.zeros((3, 3, 7))
    def sample_points(self, delta):
        return np.array([[.45+delta[0], .5, .5]])
    def sample_differential(self):
        jac = np.zeros((1, 3, 7)); jac[0, 0, 0] = 1
        return self.sample_points(np.zeros(7)), jac
    def tracking_error(self):
        jac = np.zeros((6, 7)); jac[0, 0] = 1
        return np.array([.1, 0, 0, 0, 0, 0]), jac


def test_vlsa_corrects_unsafe_nominal_and_keeps_auxiliary_out_of_robot_commands():
    geometry = ToyGeometry()
    obstacle = Ellipsoid(np.array([.08, 0, 0]), np.eye(3)*.02**2, "obstacle")
    controller = VLSAFilter(margin=.01)
    reference = np.r_[.05, np.zeros(6)]
    action, detail = controller.filter(reference, geometry, [obstacle], np.full(7, -.1), np.full(7, .1), .05)
    assert action.shape == (7,) and action[0] < .02
    assert detail["constraint_scope"] == "end_effector" and detail["virtual_variables"] == 2
    assert detail["minimum_barrier_after"] >= 0
    assert np.linalg.norm(controller.normals["obstacle"]) == pytest.approx(1.)


class AffineField:
    residual = 0.
    def __init__(self, offset, slope): self.offset, self.slope = offset, slope
    def query(self, points):
        gradient = np.tile([self.slope, 0, 0], (len(points), 1))
        return self.offset+self.slope*(points[:, 0]-.45), gradient, np.ones(len(points), bool)


def test_vlpsa_enforces_both_boolean_levels_and_injects_temporal_term():
    controller = VLPSAFilter(alpha=2.)
    geometry = ToyGeometry()
    action, detail = controller.filter(np.r_[.05, np.zeros(6)], geometry, [], -.1*np.ones(7), .1*np.ones(7), .05,
        fields=[AffineField(.01, -1.), AffineField(.005, 1.)], timestamp=1.)
    assert action[0] <= .0010001 and detail["cbf_constraints"] == 2
    assert detail["poisson_levels"] == 2
    # Moving boundary reduces h by .002 over .1s, requiring retreat.
    action2, _ = controller.filter(np.r_[.05, np.zeros(6)], geometry, [], -.1*np.ones(7), .1*np.ones(7), .05,
        fields=[AffineField(.008, -1.), AffineField(.003, 1.)], timestamp=1.1)
    assert action2[0] < action[0]


def report(episode, *, success=False, collision=False, violation=False, steps=10, **kwargs):
    return dict(suite="obstacle_avoidance", level=0, task_id=0, episode=episode, seed=0,
        completed=True, success=success, collision=collision, safety_violation=violation, steps=steps,
        termination_reason="success" if success else "timeout", test_only=False, benchmark="LIBERO-Safety", **kwargs)


def test_sr_ssr_cr_are_distinct_and_steps_conditioned_on_success():
    records = [report(0, success=True, steps=20), report(1, success=True, collision=True, steps=40),
               report(2, success=True, violation=True, steps=30), report(3, steps=100)]
    stats = aggregate(records)
    assert stats["sr_pct"] == 75 and stats["ssr_pct"] == 25 and stats["cr_pct"] == 25
    assert stats["steps_mean_success"] == 30 and stats["steps_mean_all"] == 47.5
    assert stats["steps_n_success"] == 3 and stats["paper_eligible"]
    assert stats["sr_ci95_pct"][0] < 75 < stats["sr_ci95_pct"][1]


def test_incomplete_duplicate_and_test_policy_never_look_like_paper_results(tmp_path):
    assert aggregate([report(0)], expected=2)["sr_pct"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([report(0), report(0)])
    random = {**report(0), "test_only": True}
    assert not aggregate([random])["paper_eligible"]
    export_comparison(tmp_path, {"vlsa": [random], "vlpsa": [random], "oavla": [random]}, expected=1)
    import json
    assert not json.loads((tmp_path/"comparison.json").read_text())["paper_eligible"]
    assert "test_only" in (tmp_path/"table.csv").read_text()


def test_contact_monitor_latches_robot_and_held_contacts_without_conflating_force():
    names = ["robot0_link_collision", "obstacle_g0", "held_g0"]
    model = SimpleNamespace(ngeom=3, geom_id2name=lambda i: names[i])
    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1, dist=-.001)])
    sim = SimpleNamespace(model=model, data=data)
    base = SimpleNamespace(sim=sim, parsed_problem={"constraints": [["force", "x"]]}, _eval_predicate=lambda p: True)
    env = SimpleNamespace(env=base, sim=sim)
    monitor = ContactMonitor(env, ["obstacle"], ["held"])
    monitor.sample()
    assert len(monitor.contacts) == 1 and len(monitor.violations) == 1
    data.contact = [SimpleNamespace(geom1=2, geom2=1, dist=-.001)]
    monitor.held = {"held"}; monitor.sample()
    assert len(monitor.contacts) == 2
    data.ncon = 0; monitor.sample()
    assert len(monitor.contacts) == 2


@pytest.mark.parametrize("method", ["vlsa", "vlpsa", "oavla"])
def test_method_is_independently_selectable(method):
    args = parse_args(["--fixture", "--allow-test-policy", "--output", "/tmp/unused", "--method", method])
    assert args.method == method


def test_socket_rejects_wrong_profile_and_unflagged_random_policy():
    with pytest.raises(ValueError): validate_socket({"state_dim": 12}, allow_test=True)


def test_grasp_requires_both_finger_contacts_and_injects_only_on_close():
    from obstacle_validation.libero_comparison_geometry import JointPandaGeometry
    geometry = JointPandaGeometry.__new__(JointPandaGeometry)
    geometry.spacing, geometry.held = .05, {}
    geometry.robot = SimpleNamespace(gripper=SimpleNamespace(important_geoms={"left_fingerpad": ["left"], "right_fingerpad": ["right"]}))
    geometry.eef_pose = lambda: (np.zeros(3), np.eye(3))
    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=2)])
    geometry.sim = SimpleNamespace(model=SimpleNamespace(geom_name2id=lambda n: {"left": 0, "right": 1}[n]), data=data)
    box = OrientedBox(np.array([.1, 0, 0]), np.eye(3), np.full(3, .02))
    geometry.update_grasp(1., {"cube": box}, {"cube": {2}})
    assert not geometry.held
    data.ncon = 2; data.contact.append(SimpleNamespace(geom1=1, geom2=2))
    geometry.update_grasp(1., {"cube": box}, {"cube": {2}})
    assert geometry.held_sample_count > 0
    old = geometry.held["cube"].copy()
    geometry.update_grasp(1., {}, {"cube": {2}})
    np.testing.assert_equal(geometry.held["cube"], old)
    geometry.update_grasp(-1., {}, {})
    assert not geometry.held


@pytest.mark.parametrize("mismatch", [None, "state", "protocol"])
def test_independent_run_merge_checks_protocol_and_initial_state(tmp_path, mismatch):
    import json
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("merge_methods", Path(__file__).resolve().parents[1]/"scripts/summarize_libero_safety_methods.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    roots = []
    for method in ("vlsa", "vlpsa", "oavla"):
        root = tmp_path/method; root.mkdir(); roots.append(str(root))
        config = dict(protocol="same", scenarios=[dict(episodes=[0])], arguments=dict(method=method,
            control_freq=20 if mismatch == "protocol" and method == "oavla" else 100))
        (root/"run_config.json").write_text(json.dumps(config))
        r = report(0, success=True, state_hash="different" if mismatch == "state" and method == "oavla" else "same")
        (root/f"summary_{method}.json").write_text(json.dumps(dict(episodes=[r])))
    argv = ["--runs", *roots, "--output", str(tmp_path/"merged")]
    if mismatch:
        with pytest.raises(ValueError): module.main(argv)
    else:
        module.main(argv)
        assert json.loads((tmp_path/"merged/comparison.json").read_text())["paper_eligible"]
