"""Stop the last servo command if planning does not renew its lease."""
import threading
import time


class CommandWatchdog:
    def __init__(self, stop, timeout_s):
        if timeout_s <= 0:
            raise ValueError("Watchdog timeout must be positive")
        self.stop, self.timeout_s = stop, timeout_s
        self.deadline = None
        self.error = None
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def renew(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Servo watchdog failed to stop the controller") from self.error
            self.deadline = time.monotonic() + self.timeout_s

    def _run(self):
        while not self.closed.wait(min(.02, self.timeout_s / 4)):
            with self.lock:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.deadline = None
                    try:
                        self.stop()
                    except Exception as exc:
                        self.error = exc

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2.)
