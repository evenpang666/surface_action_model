"""Only left/environment, right/environment and opposite-arm constraints."""
import numpy as np
import pytest

from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
from real_scripts.real_cbf_qp import OrientedBox
from verification.test_ellipsoid_replay import sphere


class OverlappingArmSampler:
    left_link_names = ("base", "shoulder", "upperarm", "gripper")

    def ellipsoids(self, q, grip):
        result = []
        for arm, offset in enumerate((0., 1.)):
            k = 6*arm
            # Base/upperarm and base/gripper envelopes deliberately overlap.
            centers = [[offset, 0, 0], [offset+.1, 0, 0],
                       [offset+.005+q[k], q[k+1], 0],
                       [offset+.006+q[k+2], q[k+3], 0]]
            result.extend(sphere(c, .03, f"{arm}_{name}", arm, i)
                          for i, (name, c) in enumerate(zip(self.left_link_names, centers)))
        return result


def controller():
    # An old config file must not silently re-enable same-arm constraints.
    return EllipsoidMCBF(OverlappingArmSampler(), margin=.005, inter_arm_margin=.035,
                         period=.05, config=EllipsoidMCBFConfig(self_collision_margin_m=.5))


def test_same_arm_overlap_never_creates_constraints_virtual_controls_or_detours(monkeypatch):
    cbf = controller()
    monkeypatch.setattr(cbf, "_solve", lambda *a, **kw: pytest.fail("Same-arm overlap must not run QP"))
    monkeypatch.setattr(cbf, "_guidance", lambda *a, **kw: pytest.fail("Same-arm overlap must not guide motion"))
    q = np.zeros(12)
    action = np.zeros((1, 14)); action[0, [0, 6]] = [.002, -.002]; action[0, 12:] = [.2, .8]
    _, info = cbf.plan(q, np.zeros(2), action, [], lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    np.testing.assert_array_equal(info["waypoints"][0], action[0])
    detail = info["steps"][0]
    assert detail["constraint_pairs"] == detail["virtual_direction_variables"] == 0
    assert detail["constraint_pair_counts"] == {"left_obstacle": 0, "right_obstacle": 0, "inter_arm": 0}
    assert not detail["optimization_applied"] and not detail["same_arm_avoidance"]
    assert not cbf.normals


@pytest.mark.parametrize("arm", [0, 1])
def test_each_arm_still_avoids_selected_obstacle_without_intra_arm_constraints(arm):
    cbf = controller()
    q = np.zeros(12)
    action = np.zeros((1, 14)); action[0, 6*arm] = .02; action[0, 12:] = [.2, .8]
    obstacle = OrientedBox(np.array([arm+.06, 0, 0]), np.eye(3), np.full(3, .01))
    _, info = cbf.plan(q, np.zeros(2), action, [obstacle], lower=-np.ones(12), upper=np.ones(12), max_step=.04)
    detail = info["steps"][0]
    kind = "left_obstacle" if arm == 0 else "right_obstacle"
    assert detail["optimization_applied"]
    assert detail["active_constraint_counts"][kind] > 0
    assert detail["constraint_pair_counts"]["inter_arm"] == 0
    assert all(r["pair_type"] == kind for r in detail["violated_pairs"])
    assert not any(k[0] == "robot" for k in cbf.normals)
    waypoint = np.array(info["waypoints"][0])
    assert waypoint[6*arm] < action[0, 6*arm]-.001
    np.testing.assert_allclose(waypoint[6*(1-arm):6*(1-arm)+6], 0, atol=1e-7)
    np.testing.assert_array_equal(waypoint[12:], action[0, 12:])
