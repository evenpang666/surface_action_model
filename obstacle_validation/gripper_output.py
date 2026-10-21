"""Shared checked Robotiq outputs; independent of LeRobot/PyArrow/model packages."""

import socket
import numpy as np
from obstacle_validation.core import vector


class RobotiqTargetWriter:
    """Command an already active 2F gripper without resetting its activation."""

    def __init__(self, host: str, *, speed: int, force: int):
        self.socket = socket.create_connection((host, 63352), timeout=1.)
        self.socket.settimeout(1.)
        self.buffer = b""
        self.speed, self.force = speed, force
        self.last_target = None
        try:
            if self._query("GET STA") != "STA 3":
                raise RuntimeError(f"Robotiq gripper at {host} is not activated")
        except BaseException:
            self.close()
            raise

    def _readline(self) -> str:
        while b"\n" not in self.buffer:
            chunk = self.socket.recv(128)
            if not chunk:
                raise ConnectionError("Robotiq closed command connection")
            self.buffer += chunk
            if len(self.buffer) > 256:
                raise RuntimeError("Robotiq response exceeds 256 bytes")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()

    def _query(self, line: str) -> str:
        self.socket.sendall((line+"\n").encode("ascii"))
        return self._readline()

    def send_if_changed(self, closing_fraction: float) -> bool:
        target = float(closing_fraction)
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Gripper target must be in [0,1]")
        if self.last_target is not None and abs(target-self.last_target) < .01:
            return False
        # The installed URCap does not reliably reply to every SET line.
        # Match the working Quest3 collector: write one complete command and
        # verify actual motion through the separate receive-only GET POS reader.
        payload = (f"SET POS {round(target*255)}\n"
                   f"SET SPE {self.speed}\n"
                   f"SET FOR {self.force}\n"
                   "SET GTO 1\n")
        self.socket.sendall(payload.encode("ascii"))
        self.last_target = target
        return True

    def close(self):
        connection, self.socket = self.socket, None
        if connection is not None:
            connection.close()


class DualGripperCommander:
    """Two explicit, non-resetting Robotiq outputs."""

    def __init__(self, left_ip: str, right_ip: str, *, speed: int = 255, force: int = 25):
        if not 0 <= speed <= 255 or not 0 <= force <= 255:
            raise ValueError("Robotiq speed and force must be in [0,255]")
        self.outputs = []
        try:
            for host in (left_ip, right_ip):
                self.outputs.append(RobotiqTargetWriter(host, speed=speed, force=force))
        except BaseException:
            self.close()
            raise

    def send_if_changed(self, targets: np.ndarray) -> list[bool]:
        values = vector(targets, 2, "gripper targets")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Gripper targets must be in [0,1]")
        return [out.send_if_changed(float(target)) for out, target in zip(self.outputs, values, strict=True)]

    def close(self):
        outputs, self.outputs = self.outputs, []
        for out in outputs:
            out.close()


