"""Process signals must unwind camera owners before either replay entry exits."""

from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from obstacle_validation.runtime import replay_shutdown


CHILD = r'''
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from obstacle_validation import lerobot_replay as replay
from obstacle_validation import stream_replay as stream
from real_scripts import replay_voxblox as embedded
from real_scripts import ur7e_realsense_adapter as adapter

route, directory = sys.argv[1], Path(sys.argv[2])
sys.modules["rtde_receive"] = SimpleNamespace(RTDEReceiveInterface=None)
provider = object.__new__(embedded.ReplayVoxbloxSource)
provider.condition = threading.Condition()
provider.stop = threading.Event()
provider.thread = provider.latest = provider.error = provider.started_at = None
provider.cleanup_errors = []
provider.startup_timeout_s = 30.
provider.scene = SimpleNamespace(arms=[], calibrations={}, report={"camera_serials": {}},
                                open_grippers=lambda args: {})
provider.args = SimpleNamespace(width=640, height=360, fps=15, robot_mask=False, warmup_frames=1)
provider.manual_filter = object()

class Camera:
    def __init__(self, **kwargs): pass
    def start(self): pass
    def get_color_calibrations(self): return {}
    def read(self):
        assert not threading.current_thread().daemon
        (directory / "ready").touch()
        provider.stop.wait()
        return {}
    def stop(self):
        (directory / "stopping").touch()
        deadline = time.monotonic() + 10.
        while not (directory / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test did not release mock camera")
            time.sleep(.01)
        (directory / "stopped").touch()

adapter.RealSenseD435iSource = Camera
embedded.SelectedTSDFPipeline = lambda *args: None
scene = replay.LiveSceneReader("unused", 2., provider=provider)

if route == "lerobot":
    def body(args, cleanup, begin_cleanup):
        cleanup.callback(scene.close)
        provider.read()
    replay._run_replay = body
    replay.run_replay(None)
else:
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, timeout): pass
    class Executor:
        def __init__(self, args, cleanup, connection):
            self.report = {}
            cleanup.callback(scene.close)
        def execute_chunk(self, *args): pass
        def start(self, *args): pass
        def _write_report(self):
            (directory / "report").touch()
    stream.socket.create_connection = lambda *args, **kwargs: Connection()
    stream.ReplayExecutor = Executor
    stream.receive_session = lambda *args, **kwargs: provider.read()
    stream.run(stream.parser().parse_args(["--output", str(directory)]))
'''


def wait_for_marker(process, marker):
    deadline = time.monotonic() + 15.
    while not marker.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Replay exited before {marker.name}: {stdout}\n{stderr}")
        if time.monotonic() > deadline:
            pytest.fail(f"Replay did not reach {marker.name}")
        time.sleep(.01)


@pytest.mark.parametrize("route", ["lerobot", "stream"])
@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_signal_waits_for_camera_release_and_ignores_repeated_signals(tmp_path, route, number):
    process = subprocess.Popen([sys.executable, "-c", CHILD, route, str(tmp_path)],
                               cwd=Path(__file__).resolve().parents[1],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_marker(process, tmp_path / "ready")
        process.send_signal(number)
        wait_for_marker(process, tmp_path / "stopping")
        assert process.poll() is None
        process.send_signal(signal.SIGINT)
        process.send_signal(signal.SIGTERM)
        (tmp_path / "release").touch()
        stdout, stderr = process.communicate(timeout=15.)
        assert process.returncode == 128 + number, (stdout, stderr)
        assert (tmp_path / "stopped").exists()
        if route == "stream":
            assert (tmp_path / "report").exists()
    finally:
        (tmp_path / "release").touch()
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("terminate", [False, True])
def test_replay_restores_existing_handlers_and_protects_cleanup(terminate):
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with replay_shutdown() as begin_cleanup:
            if terminate:
                with pytest.raises(SystemExit) as exc:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                assert exc.value.code == 143
            else:
                begin_cleanup()
            assert all(signal.getsignal(number) == signal.SIG_IGN for number in previous)
    finally:
        assert all(signal.getsignal(number) == handler for number, handler in previous.items())


def test_embedded_replay_does_not_install_handlers_from_worker_thread():
    errors = []
    def run():
        try:
            with replay_shutdown() as begin_cleanup:
                begin_cleanup()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(2.)
    assert not thread.is_alive() and not errors
