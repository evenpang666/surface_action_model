"""Read PiKA's live jaw opening without enabling or commanding the motor."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import threading

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class PikaOpeningReader:
    """Read ``Gripper.get_gripper_distance()`` in millimetres over PiKA USB."""

    def __init__(self, port: str, *, max_opening_mm: float = 95.0, max_state_age_s: float | None = None) -> None:
        self.port = str(port)
        self.max_opening_mm = float(max_opening_mm)
        if self.max_opening_mm <= 0.0:
            raise ValueError("max_opening_mm must be positive")
        self._device = None
        self.max_state_age_s = max_state_age_s
        if max_state_age_s is not None and (not np.isfinite(max_state_age_s) or max_state_age_s <= 0):
            raise ValueError("max_state_age_s must be finite and positive")
        self._state_lock = threading.Lock()
        self._last_state_time = None

    def connect(self) -> None:
        if self.port.strip().lower() == "auto":
            from real_scripts.pika_serial_discovery import detect_pika_gripper_port
            self.port = detect_pika_gripper_port()
            print(f"[pika] Selected Gripper serial: {self.port} (passive motor telemetry)", flush=True)
        elif self.port.startswith("/dev/") and not Path(self.port).exists():
            raise RuntimeError(f"Pika serial port does not exist: {self.port}. "
                               "Use --pika-gripper-port auto or run python -m real_scripts.pika_serial_discovery")
        # The vendored SDK is deliberately imported only when a caller opts in.
        # No enable(), set_motor_angle(), or disable() call is made here.
        sdk_root = REPO_ROOT / "real_scripts" / "ur7e_robotiq_d435i_collector" / "pika_sdk"
        if not sdk_root.is_dir():
            raise FileNotFoundError(f"PiKA SDK directory does not exist: {sdk_root}")
        if str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        from pika.gripper import Gripper

        reader = self

        class TimestampedGripper(Gripper):
            def _data_callback(self, data):
                if reader.max_state_age_s is None:
                    return super()._data_callback(data)
                motor = data.get("motor", {}) if isinstance(data, dict) else {}
                try:
                    valid = "Position" in motor and np.isfinite(float(motor["Position"]))
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    return
                # Serialize the SDK state update and its timestamp with readers.
                with reader._state_lock:
                    super()._data_callback(data)
                    reader._last_state_time = time.monotonic()

        self._last_state_time = None
        device = TimestampedGripper(self.port)
        if not device.connect():
            raise RuntimeError(f"PiKA gripper did not connect on {self.port}")
        self._device = device

    def opening_mm(self) -> float:
        if self._device is None:
            raise RuntimeError("PiKA opening reader is not connected")
        with self._state_lock:
            if self.max_state_age_s is not None:
                if self._last_state_time is None or time.monotonic() - self._last_state_time > self.max_state_age_s:
                    raise RuntimeError("PiKA opening state is missing or stale")
            opening_mm = float(self._device.get_gripper_distance())
        if not np.isfinite(opening_mm) or not 0.0 <= opening_mm <= self.max_opening_mm:
            raise RuntimeError(
                f"PiKA opening readback {opening_mm!r} mm is outside [0, {self.max_opening_mm}]"
            )
        return opening_mm

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.disconnect()
            except Exception:
                pass
            self._device = None
