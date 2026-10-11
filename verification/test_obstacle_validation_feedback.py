"""LeRobot replay feedback and output failure checks without hardware."""

import sys
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_validation import runtime


def test_read_only_robotiq_reads_fragmented_lines_and_reconnects_after_drop(monkeypatch):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    connections = []

    class Socket:
        def __init__(self, chunks):
            self.chunks, self.sent, self.closed = list(chunks), [], False
        def settimeout(self, timeout): assert timeout == .25
        def sendall(self, data): self.sent.append(data)
        def recv(self, size):
            chunk = self.chunks.pop(0)
            if isinstance(chunk, Exception): raise chunk
            return chunk
        def close(self): self.closed = True

    first = Socket([b'POS ', b'17\r\n', ConnectionResetError('peer reset')])
    second = Socket([b'POS 18\n'])
    def connect(address, timeout):
        assert address == ('right-ip', 63352) and timeout == .25
        connection = (first, second)[len(connections)]
        connections.append(connection)
        return connection
    monkeypatch.setattr(live.socket, 'create_connection', connect)
    reader = live.ReadOnlyRobotiq2FPosition('right-ip', port=63352)
    assert reader.closing_fraction() == pytest.approx(17/255)
    assert reader.closing_fraction() == pytest.approx(18/255)
    assert all(sent == b'GET POS\n' for connection in connections for sent in connection.sent)
    assert first.closed
    reader.close()
    assert second.closed


@pytest.mark.parametrize('response', [b'', b'POS 256\n', b'POS -1\n', b'ERR\n'])
def test_read_only_robotiq_rejects_missing_or_invalid_position(monkeypatch, response):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    class Socket:
        def settimeout(self, timeout): pass
        def sendall(self, data): assert data == b'GET POS\n'
        def recv(self, size): return response
        def close(self): pass
    monkeypatch.setattr(live.socket, 'create_connection', lambda *args, **kwargs: Socket())
    reader = live.ReadOnlyRobotiq2FPosition('right-ip', port=63352)
    try:
        with pytest.raises((RuntimeError, ConnectionError)):
            reader.closing_fraction()
    finally:
        reader.close()


def test_read_only_robotiq_reports_failed_reconnect(monkeypatch):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    class Socket:
        def settimeout(self, timeout): pass
        def sendall(self, data): assert data == b'GET POS\n'
        def recv(self, size): raise socket.timeout('read timed out')
        def close(self): pass
    class RecoveredSocket(Socket):
        def recv(self, size): return b'POS 41\n'
    attempts = [Socket(), Socket(), RecoveredSocket()]
    monkeypatch.setattr(live.socket, 'create_connection', lambda *args, **kwargs: attempts.pop(0))
    reader = live.ReadOnlyRobotiq2FPosition('right-ip', port=63352)
    with pytest.raises(RuntimeError, match='GET POS right-ip:63352 failed after reconnect.*timed out'):
        reader.closing_fraction()
    assert reader.socket is None
    assert reader.closing_fraction() == pytest.approx(41/255)
    assert not attempts
    reader.close()


def fake_interfaces(monkeypatch, *, fail=None, reject=False):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    receivers, grippers, controls, closed = [], [], [], []

    class Receiver:
        def __init__(self, ip):
            self.ip, self.q, self.stamp, self.connected = ip, np.zeros(6), 1., True
            receivers.append(self)
        def isConnected(self): return self.connected
        def getTimestamp(self): return self.stamp
        def getActualQ(self): return self.q
        def getRobotMode(self): return 7
        def getRuntimeState(self): return 2
        def getSpeedScaling(self): return 1.
        def getTargetSpeedFraction(self): return 1.
        def disconnect(self):
            closed.append(('receiver', self.ip))
            if fail == 'close' and self.ip == 'left': raise RuntimeError('lost receiver')

    class Gripper:
        def __init__(self, ip, **kwargs):
            if fail == 'gripper' and ip == 'right': raise RuntimeError('GET POS unavailable')
            self.ip, self.value = ip, 0.
            grippers.append(self)
        def closing_fraction(self): return self.value
        def close(self): closed.append(('gripper', self.ip))

    class Control:
        def __init__(self, ip):
            if fail == 'control' and ip == 'right': raise RuntimeError('control unavailable')
            self.ip, self.targets, self.stops = ip, [], 0
            controls.append(self)
        def servoJ(self, q, *args):
            self.targets.append(q)
            return not (reject and self.ip == 'right')
        def servoStop(self, *args): self.stops += 1
        def disconnect(self): closed.append(('control', self.ip))

    monkeypatch.setitem(sys.modules, 'rtde_receive', SimpleNamespace(RTDEReceiveInterface=Receiver))
    monkeypatch.setitem(sys.modules, 'rtde_control', SimpleNamespace(RTDEControlInterface=Control))
    monkeypatch.setattr(live, 'ReadOnlyRobotiq2FPosition', Gripper)
    return SimpleNamespace(receivers=receivers, grippers=grippers, controls=controls, closed=closed)


@pytest.mark.parametrize('fault', ['disconnected', 'nonfinite', 'restarted', 'stalled', 'invalid_grip'])
def test_direct_feedback_fails_closed_without_guessing_state(monkeypatch, fault):
    rig = fake_interfaces(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    reader = runtime.DualStateReader('left', 'right')
    try:
        q, grip = reader.read()
        np.testing.assert_array_equal(q, np.zeros(12))
        if fault == 'disconnected': rig.receivers[0].connected = False
        if fault == 'nonfinite': rig.receivers[0].q[0] = np.nan
        if fault == 'restarted': rig.receivers[0].stamp = .5
        if fault == 'stalled': clock[0] = .26
        if fault == 'invalid_grip': rig.grippers[1].value = 2.
        with pytest.raises((RuntimeError, ValueError)):
            reader.read()
    finally:
        reader.close()
    assert len(rig.closed) == 4


@pytest.mark.parametrize('failure', ['gripper', 'control'])
def test_partial_output_initialization_releases_feedback_and_controllers(monkeypatch, failure):
    rig = fake_interfaces(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError):
        runtime.DualCommander('left', 'right', .1)
    assert ('receiver', 'left') in rig.closed and ('receiver', 'right') in rig.closed
    assert ('gripper', 'left') in rig.closed
    if failure == 'control':
        assert rig.controls[0].stops == 1 and ('control', 'left') in rig.closed


def test_explicit_stop_disarms_lease_and_cleanup_covers_every_connection(monkeypatch):
    rig = fake_interfaces(monkeypatch, fail='close')
    commander = runtime.DualCommander('left', 'right', .1)
    commander.send(np.zeros(12), np.full(12, .001))
    assert commander.commands_sent == 1 and commander.watchdog.deadline is not None
    commander.stop()
    assert commander.watchdog.deadline is None
    with pytest.raises(RuntimeError, match='close dual UR'):
        commander.close()
    assert len(rig.closed) == 6 and commander.watchdog.closed.is_set()


def test_one_arm_rejection_stops_both_and_is_not_counted_as_sent(monkeypatch):
    rig = fake_interfaces(monkeypatch, reject=True)
    commander = runtime.DualCommander('left', 'right', .1)
    try:
        with pytest.raises(RuntimeError, match='rejected'):
            commander.send(np.zeros(12), np.full(12, .001))
        assert commander.commands_sent == 0
        assert all(c.stops == 1 for c in rig.controls)
        assert commander.watchdog.deadline is None
    finally:
        commander.close()
