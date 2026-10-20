"""Capture joint and TCP readback from one RTDE update, without robot commands."""

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class RTDETCPState:
    q: np.ndarray
    tcp: np.ndarray
    timestamp_s: float
    read_seconds: float


def read_rtde_tcp_state(receiver, *, max_attempts=32):
    """Retry if the receive thread advances while the getters are being read.

    Getters read a continuously updated cache. Reading them near each other
    alone does not ensure they belong to the same controller packet.
    """
    began = time.perf_counter()
    for _ in range(max_attempts):
        before = float(receiver.getTimestamp())
        q = np.asarray(receiver.getActualQ(), dtype=np.float64).copy()
        tcp = np.asarray(receiver.getActualTCPPose(), dtype=np.float64).copy()
        after = float(receiver.getTimestamp())
        if (q.shape != (6,) or tcp.shape != (6,) or not np.isfinite(q).all()
                or not np.isfinite(tcp).all() or not np.isfinite([before, after]).all()
                or before <= 0 or after < before):
            raise RuntimeError("Invalid RTDE joint/TCP state or controller timestamp")
        if before == after:
            q.setflags(write=False)
            tcp.setflags(write=False)
            return RTDETCPState(q, tcp, after, time.perf_counter()-began)
    raise RuntimeError("Could not read joint/TCP state from one RTDE packet; controller updates crossed every read")
