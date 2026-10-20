"""Catch RTDE packet races and retain actionable TCP mismatch evidence."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_scripts.live_dual_ur7e_obstacle_model import validate_active_tcp
from real_scripts.rtde_state import read_rtde_tcp_state
from real_scripts.ur7e_collision_mesh import flange_transform


class Receiver:
    def __init__(self, *, advance_once=False, always_advance=False):
        self.timestamp = 1.
        self.q = np.zeros(6)
        self.advance_once, self.always_advance = advance_once, always_advance

    def getTimestamp(self):
        return self.timestamp

    def getActualQ(self):
        return self.q

    def getActualTCPPose(self):
        if self.advance_once or self.always_advance:
            self.q[0] += .03
            self.timestamp += .002
            self.advance_once = False
        matrix = flange_transform(self.q)
        return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]


def test_packet_update_retries_instead_of_reporting_false_tcp_mismatch():
    receiver = Receiver(advance_once=True)
    old_q = receiver.getActualQ().copy()
    state = read_rtde_tcp_state(receiver)
    assert state.timestamp_s == 1.002
    with pytest.raises(RuntimeError, match="Active TCP differs"):
        validate_active_tcp(old_q, state.tcp, np.eye(4))
    np.testing.assert_allclose(validate_active_tcp(state.q, state.tcp, np.eye(4)), 0., atol=1e-9)
    receiver.q[0] += .5  # Later mesh/gripper work must not change the captured pair.
    np.testing.assert_allclose(validate_active_tcp(state.q, state.tcp, np.eye(4)), 0., atol=1e-9)
    assert not state.q.flags.writeable and not state.tcp.flags.writeable


def test_continuously_crossed_packets_fail_instead_of_using_inconsistent_state():
    with pytest.raises(RuntimeError, match="one RTDE packet"):
        read_rtde_tcp_state(Receiver(always_advance=True), max_attempts=3)


@pytest.mark.parametrize("invalid", ["joint_nan", "joint_shape", "timestamp_nan", "timestamp_backwards"])
def test_invalid_readback_is_rejected(invalid):
    receiver = Receiver()
    if invalid == "joint_nan":
        receiver.getActualQ = lambda: np.full(6, np.nan)
    elif invalid == "joint_shape":
        receiver.getActualQ = lambda: np.zeros(5)
    elif invalid == "timestamp_nan":
        receiver.getTimestamp = lambda: np.nan
    else:
        times = iter([2., 1.])
        receiver.getTimestamp = lambda: next(times)
    with pytest.raises(RuntimeError, match="Invalid RTDE"):
        read_rtde_tcp_state(receiver)


def test_static_six_mm_mismatch_still_fails_with_original_joint_and_tcp_evidence():
    state = read_rtde_tcp_state(Receiver())
    measured = state.tcp.copy()
    measured[0] += .006
    with pytest.raises(RuntimeError, match="right_arm: Active TCP differs") as failure:
        validate_active_tcp(state.q, measured, np.eye(4), arm_name="right_arm",
                            controller_timestamp_s=state.timestamp_s)
    check = failure.value.active_tcp_check
    assert check["arm"] == "right_arm" and check["max_position_mm"] == 5.
    assert check["position_error_mm"] == pytest.approx(6.)
    assert check["controller_timestamp_s"] == 1.
    np.testing.assert_allclose(check["q_rad"], state.q)
    np.testing.assert_allclose(check["measured_tcp_pose"], measured)
