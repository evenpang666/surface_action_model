"""Passive role detection, alias handling and serial-reader integration."""
import itertools
import sys
from types import SimpleNamespace

import pytest

from real_scripts import pika_serial_discovery as discovery
from real_scripts.pika_gripper_live_reader import PikaOpeningReader


@pytest.mark.parametrize("payload,role", [({"AS5047": {}}, "sense"),
                                         ({"motor": {"Position": .5}}, "gripper"),
                                         ({"motorstatus": {"Voltage": 24}}, "gripper"),
                                         ({"unknown": 1}, None), (None, None), ("invalid", None)])
def test_classifies_streamed_firmware_payloads(payload, role):
    assert discovery.classify_payload(payload) == role


def test_candidates_prefer_persistent_alias_and_probe_one_physical_port(tmp_path, monkeypatch):
    from serial.tools import list_ports
    node = tmp_path / "ttyUSB0"
    node.touch()
    directory = tmp_path / "serial/by-id"
    directory.mkdir(parents=True)
    alias = directory / "usb-Pika-Gripper"
    alias.symlink_to(node)
    other = tmp_path / "ttyUSB81"
    other.symlink_to(node)
    monkeypatch.setattr(list_ports, "comports", lambda: [])
    monkeypatch.setattr(discovery.glob, "glob", lambda _pattern: [str(node), str(alias), str(other)])
    assert discovery.serial_candidates() == [str(alias)]


def test_probe_distinguishes_sense_motor_unknown_denied_and_failed_ports(monkeypatch):
    ports = ["sense", "motor", "unknown", "denied", "failed"]
    messages = {"sense": {"AS5047": {}}, "motor": {"motorstatus": {}}, "unknown": {}}
    opened, closed = [], []
    class Comm:
        def __init__(self, port, timeout):
            self.port = port
            assert timeout <= .25
        def connect(self):
            opened.append(self.port)
            return self.port != "failed"
        def start_reading_thread(self):
            pass
        def get_latest_data(self):
            return messages[self.port]
        def disconnect(self):
            closed.append(self.port)
        def send_command(self, *_a):
            pytest.fail("Discovery cannot send motor commands")
        def send_data(self, *_a):
            pytest.fail("Discovery cannot transmit data")
    monkeypatch.setattr(discovery, "serial_candidates", lambda: ports)
    monkeypatch.setattr(discovery, "_serial_comm_class", lambda: Comm)
    monkeypatch.setattr(discovery.os, "access", lambda port, _mode: port != "denied")
    ticks = itertools.count(0., .01)
    monkeypatch.setattr(discovery.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(discovery.time, "sleep", lambda _seconds: None)
    devices = discovery.discover_pika_serial_devices(probe_timeout_s=.1)
    assert devices.grippers == ("motor",) and devices.sense == ("sense",)
    assert devices.unknown == ("unknown", "failed") and devices.inaccessible == ("denied",)
    assert opened == closed == ["sense", "motor", "unknown", "failed"]
    assert discovery.select_gripper(devices) == "motor"


def test_no_visible_nodes_does_not_import_or_open_sdk(monkeypatch):
    monkeypatch.setattr(discovery, "serial_candidates", lambda: [])
    monkeypatch.setattr(discovery, "_serial_comm_class", lambda: pytest.fail("No port should be opened"))
    with pytest.raises(RuntimeError, match="No Pika serial device is visible"):
        discovery.detect_pika_gripper_port()


@pytest.mark.parametrize("grippers,sense,denied,message", [
    (("motor-a", "motor-b"), (), (), "Multiple Pika Grippers"),
    ((), ("sense",), (), "Sense="),
    ((), (), ("denied",), "inaccessible="),
])
def test_ambiguous_or_unidentified_ports_are_not_guessed(grippers, sense, denied, message):
    devices = discovery.PikaSerialDevices(("candidate",), grippers, sense, (), denied)
    with pytest.raises(RuntimeError, match=message):
        discovery.select_gripper(devices)


def test_auto_reader_uses_detected_gripper_without_enable_or_motion(monkeypatch):
    calls = []
    monkeypatch.setattr(discovery, "detect_pika_gripper_port", lambda: "detected-gripper")
    class SDK:
        def __init__(self, port):
            calls.append(port)
        def connect(self):
            return True
        def disconnect(self):
            calls.append("closed")
        def get_gripper_distance(self):
            return 20.
    monkeypatch.setitem(sys.modules, "pika.gripper", SimpleNamespace(Gripper=SDK))
    reader = PikaOpeningReader("auto")
    reader.connect()
    assert reader.port == "detected-gripper"
    assert reader.opening_mm() == 20.
    reader.close()
    assert calls == ["detected-gripper", "closed"]


def test_explicit_missing_device_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(discovery, "detect_pika_gripper_port", lambda: pytest.fail("Explicit path cannot fall back"))
    with pytest.raises(RuntimeError, match="Pika serial port does not exist"):
        PikaOpeningReader("/dev/nonexistent-pika-test-device").connect()
