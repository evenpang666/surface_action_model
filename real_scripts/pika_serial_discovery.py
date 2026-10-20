"""Identify Pika Gripper serial ports from passive firmware telemetry.

Uses the same motor/motorstatus vs AS5047 classification as the existing
ur7e_vla.device_discovery, without importing the policy/control runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PikaSerialDevices:
    candidates: tuple[str, ...]
    grippers: tuple[str, ...]
    sense: tuple[str, ...]
    unknown: tuple[str, ...]
    inaccessible: tuple[str, ...]


def serial_candidates():
    ports = set()
    try:
        from serial.tools import list_ports
        ports.update(p.device for p in list_ports.comports()
                     if p.device.startswith(("/dev/ttyUSB", "/dev/ttyACM")))
    except ImportError:
        pass
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*", "/dev/serial/by-path/*"):
        ports.update(glob.glob(pattern))
    # Enumerated nodes and udev aliases may refer to the same device. Prefer
    # persistent aliases while probing each physical serial endpoint once.
    aliases = {}
    def priority(port):
        return (0 if "/by-id/" in port else 1 if "/by-path/" in port else 2, port)
    for port in sorted(ports, key=priority):
        path = Path(port)
        if path.exists():
            aliases.setdefault(str(path.resolve()), port)
    return sorted(aliases.values())


def classify_payload(payload):
    if not isinstance(payload, dict):
        return None
    if "AS5047" in payload:
        return "sense"
    if "motor" in payload or "motorstatus" in payload:
        return "gripper"
    return None


def _serial_comm_class():
    sdk = REPO_ROOT / "real_scripts/ur7e_robotiq_d435i_collector/pika_sdk"
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    from pika.serial_comm import SerialComm
    return SerialComm


def discover_pika_serial_devices(*, probe_timeout_s=1.5):
    if not np.isfinite(probe_timeout_s) or probe_timeout_s <= 0:
        raise ValueError("probe_timeout_s must be finite and positive")
    candidates = serial_candidates()
    grippers, sense, unknown, inaccessible = [], [], [], []
    if not candidates:
        return PikaSerialDevices((), (), (), (), ())
    SerialComm = _serial_comm_class()
    for port in candidates:
        if not os.access(port, os.R_OK | os.W_OK):
            inaccessible.append(port)
            continue
        comm = SerialComm(port=port, timeout=min(.25, probe_timeout_s))
        role = None
        try:
            if comm.connect():
                comm.start_reading_thread()
                deadline = time.monotonic() + probe_timeout_s
                while time.monotonic() < deadline:
                    role = classify_payload(comm.get_latest_data())
                    if role is not None:
                        break
                    time.sleep(.03)
        except Exception as exc:
            print(f"[pika] Could not inspect {port}: {exc}", file=sys.stderr)
        finally:
            comm.disconnect()
        if role == "gripper":
            grippers.append(port)
        elif role == "sense":
            sense.append(port)
        else:
            unknown.append(port)
    return PikaSerialDevices(tuple(candidates), tuple(grippers), tuple(sense), tuple(unknown), tuple(inaccessible))


def select_gripper(devices):
    if not devices.candidates:
        raise RuntimeError("No Pika serial device is visible: no /dev/ttyUSB*, /dev/ttyACM* or /dev/serial/by-id nodes. "
                           "Connect the gripper controller USB serial interface and expose it to this runtime environment.")
    if len(devices.grippers) > 1:
        raise RuntimeError(f"Multiple Pika Grippers found: {list(devices.grippers)}. "
                           "Select one with --pika-gripper-port /dev/serial/by-id/...")
    if devices.grippers:
        return devices.grippers[0]
    raise RuntimeError(f"No Pika Gripper motor telemetry identified. Sense={list(devices.sense)}, "
                       f"unknown={list(devices.unknown)}, inaccessible={list(devices.inaccessible)}. "
                       "Check controller USB/power, serial access and other processes using the port; "
                       "the Sense encoder port cannot supply the gripper motor opening.")


def detect_pika_gripper_port(*, probe_timeout_s=1.5):
    return select_gripper(discover_pika_serial_devices(probe_timeout_s=probe_timeout_s))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=1.5)
    args = parser.parse_args()
    devices = discover_pika_serial_devices(probe_timeout_s=args.timeout_s)
    print(json.dumps(asdict(devices), ensure_ascii=False, indent=2))
    try:
        print(f"Pika Gripper: {select_gripper(devices)}")
    except RuntimeError as exc:
        print(f"[pika] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
