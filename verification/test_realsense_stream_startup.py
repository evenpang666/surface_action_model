"""Mixed USB links must preserve RGB geometry and clean up failed startup."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


def fake_sdk(monkeypatch, *, fail_serial=None, fail_intrinsics=None, fail_stop=None):
    pipelines = []

    class Profile:
        def __init__(self, stream, width, height, fps):
            self.stream, self.w, self.h, self.rate = stream, width, height, fps

        def stream_type(self): return self.stream
        def format(self): return 'rgb8' if self.stream == 'color' else 'z16'
        def as_video_stream_profile(self): return self
        def width(self): return self.w
        def height(self): return self.h
        def fps(self): return self.rate
        def get_intrinsics(self):
            if fail_intrinsics:
                raise RuntimeError('intrinsics unavailable')
            return NS(fx=700. if self.stream == 'color' else 400.,
                      fy=701. if self.stream == 'color' else 401.,
                      ppx=self.w/2, ppy=self.h/2, coeffs=[.1, -.2, 0., 0., 0.],
                      model='distortion.brown_conrady', width=self.w, height=self.h)
        def get_extrinsics_to(self, other):
            return NS(rotation=[0., 1., 0., -1., 0., 0., 0., 0., 1.], translation=[.01, -.02, .03])

    class Device:
        def __init__(self, serial, usb): self.serial, self.usb = serial, usb
        def get_info(self, field): return self.serial if field == 'serial' else self.usb
        def supports(self, field): return True
        def query_sensors(self):
            profiles = [Profile('color', 1280, 720, 15), Profile('color', 640, 480, 15),
                        Profile('depth', 640, 480, 15), Profile('depth', 1280, 720, 5)]
            if self.usb == '3.2': profiles.append(Profile('depth', 1280, 720, 15))
            return [NS(get_stream_profiles=lambda: profiles)]

    class Config:
        def __init__(self): self.serial, self.streams = None, {}
        def enable_device(self, serial): self.serial = serial
        def enable_stream(self, stream, width, height, format_, fps):
            self.streams[stream] = (width, height, fps)

    class Frame:
        def __init__(self, data, profile): self.data, self.profile = data, profile
        def get_data(self): return self.data
        def get_profile(self): return self.profile
        def get_units(self): return .001
        def get_frame_timestamp_domain(self): return 'hardware_clock'
        def get_timestamp(self): return 1000.
        def get_frame_number(self): return 1

    class Pipeline:
        def __init__(self):
            self.config, self.stopped = None, False
            pipelines.append(self)
        def start(self, config):
            self.config = config
            if config.serial == fail_serial: raise RuntimeError("Couldn't resolve requests")
            return NS(get_stream=lambda stream: Profile(stream, *config.streams[stream]))
        def stop(self):
            self.stopped = True
            if self.config.serial == fail_stop: raise RuntimeError('USB disconnected during stop')
        def wait_for_frames(self, timeout):
            w, h, _ = self.config.streams['color']
            if 'depth' not in self.config.streams:
                return NS(get_color_frame=lambda: Frame(np.zeros((h,w,3),np.uint8),Profile('color',w,h,15)))
            dw, dh, rate = self.config.streams['depth']
            return NS(get_color_frame=lambda: Frame(np.zeros((h, w, 3), np.uint8), Profile('color', w, h, 15)),
                      get_depth_frame=lambda: Frame(np.full((dh, dw), 1100, np.uint16), Profile('depth', dw, dh, rate)))

    def align(frames):
        color = frames.get_color_frame()
        h, w = color.data.shape[:2]
        return NS(get_color_frame=lambda: color,
                  get_depth_frame=lambda: Frame(np.full((h, w), 1000, np.uint16), Profile('depth', w, h, 15)))

    devices = [Device('front-id', '3.2'), Device('side-id', '2.1')]
    sdk = NS(camera_info=NS(serial_number='serial', usb_type_descriptor='usb'),
             stream=NS(color='color', depth='depth'), format=NS(rgb8='rgb8', z16='z16'),
             context=lambda: NS(query_devices=lambda: devices), pipeline=Pipeline, config=Config,
             align=lambda stream: NS(process=align))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', sdk)
    return pipelines


def source(**kwargs):
    return RealSenseD435iSource(cameras=(D435iCameraConfig('front', 'front-id'),
                                       D435iCameraConfig('side', 'side-id')),
                               width=1280, height=720, fps=15, **kwargs)


def test_usb2_depth_fallback_preserves_rgb_calibration_and_aligned_depth(monkeypatch):
    pipelines = fake_sdk(monkeypatch)
    camera = source()
    with pytest.warns(RuntimeWarning, match="side.*USB=2.1.*640, 480, 15"):
        camera.start()
    assert pipelines[0].config.streams['depth'] == (1280, 720, 15)
    assert pipelines[1].config.streams['depth'] == (640, 480, 15)
    assert camera.get_stream_info()['side']['color'] == [1280, 720, 15]
    for frame in camera.read().values():
        assert frame.rgb.shape == (720, 1280, 3) and frame.depth_m.shape == (720, 1280)
        np.testing.assert_allclose(frame.depth_m, 1.)
    assert camera.get_color_calibrations()['side']['intrinsics'][0][0] == 700.
    camera.stop()
    assert all(p.stopped for p in pipelines)


def test_native_depth_is_copied_before_sdk_alignment(monkeypatch):
    fake_sdk(monkeypatch)
    camera = source(retain_native_depth=True)
    with pytest.warns(RuntimeWarning):
        camera.start()
    try:
        frames = camera.read()
        native = frames['side'].native_depth
        assert native.depth_units.shape == (480, 640)
        assert frames['side'].depth_m.shape == (720, 1280)
        assert native.depth_units.dtype == np.uint16
        np.testing.assert_array_equal(native.depth_units, np.full((480, 640), 1100, np.uint16))
        np.testing.assert_allclose(frames['side'].depth_m, 1.)
        assert native.depth_scale_m == .001 and native.intrinsics[0, 0] == 400.
        np.testing.assert_array_equal(native.depth_to_color[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(native.depth_to_color[:3, 3], [.01, -.02, .03])
    finally:
        camera.stop()


def test_explicit_depth_profile_is_checked_before_any_camera_starts(monkeypatch):
    pipelines = fake_sdk(monkeypatch)
    with pytest.raises(RuntimeError, match='side.*USB=2.1.*unsupported depth'):
        source(depth_width=1280, depth_height=720).start()
    assert not pipelines
    with pytest.raises(ValueError, match='both depth_width'):
        source(depth_width=640)


@pytest.mark.parametrize('fault', ['second_camera', 'intrinsics'])
def test_partial_camera_start_is_rolled_back_with_original_error(monkeypatch, fault):
    pipelines = fake_sdk(monkeypatch, fail_serial='side-id' if fault == 'second_camera' else None,
                         fail_intrinsics=fault == 'intrinsics')
    camera = source()
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError) as error:
        camera.start()
    assert pipelines[0].stopped
    assert not camera.get_stream_info()
    if fault == 'second_camera':
        assert 'side-id' in str(error.value) and 'RGB=' in str(error.value) and 'depth=' in str(error.value)
    else:
        assert 'intrinsics unavailable' in str(error.value)
    with pytest.raises(RuntimeError, match='not started'):
        camera.read()


def test_cleanup_stops_both_cameras_if_first_stop_fails(monkeypatch):
    pipelines = fake_sdk(monkeypatch, fail_stop='front-id')
    camera = source()
    with pytest.warns(RuntimeWarning): camera.start()
    with pytest.raises(RuntimeError, match='front'):
        camera.stop()
    assert all(p.stopped for p in pipelines)
    camera.stop()  # Idempotent after a device has been lost.


def test_missing_calibrated_camera_is_not_replaced_by_another_device(monkeypatch):
    pipelines = fake_sdk(monkeypatch)
    camera = RealSenseD435iSource(cameras=[D435iCameraConfig('front', 'missing-id')])
    with pytest.raises(RuntimeError, match='missing-id.*available serials'):
        camera.start()
    assert not pipelines


@pytest.mark.parametrize('failure', ['first_receiver', 'second_receiver', 'camera'])
def test_worker_initialization_error_is_published_and_reported(monkeypatch, capsys, failure):
    from real_scripts import live_dual_ur7e_obstacle_model as live
    from real_scripts.real_robot_adapter import CameraCalibration
    disconnected, stopped = [], []
    cal = CameraCalibration('front', np.eye(3), np.eye(4))
    arms = (live.Arm('left', 'left-ip', np.eye(4), '2f85', np.eye(4)),
            live.Arm('right', 'right-ip', np.eye(4), '2f85', np.eye(4)))
    monkeypatch.setattr(live, 'load_arms', lambda args: (*arms, cal, {'camera_serial': 'front-id'}))
    monkeypatch.setattr(live, 'load_calibration', lambda path: (np.eye(4), np.eye(3), 'side-id', {'parent_frame': 'left_base'}))

    class Receiver:
        def __init__(self, ip):
            self.ip = ip
            if (failure == 'first_receiver' and ip == 'left-ip'
                    or failure == 'second_receiver' and ip == 'right-ip'):
                raise RuntimeError('RTDE connection failed')
        def disconnect(self): disconnected.append(self.ip)

    class Source:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError('side-id USB=2.1 failed to start')
        def stop(self): stopped.append(True)

    class Gripper:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass

    monkeypatch.setitem(sys.modules, 'rtde_receive', NS(RTDEReceiveInterface=Receiver))
    monkeypatch.setattr(live, 'RealSenseD435iSource', Source)
    monkeypatch.setattr(live, 'ReadOnlyRobotiq2FPosition', Gripper)
    monkeypatch.setattr(sys, 'argv', ['perception', '--once', '--no-lingbot-depth', '--no-browser'])
    with pytest.raises(RuntimeError, match='Obstacle acquisition failed: error: RuntimeError:'):
        live.main()
    assert '[model] error: RuntimeError:' in capsys.readouterr().err
    if failure == 'second_receiver':
        assert disconnected == ['left-ip']
    elif failure == 'camera':
        assert set(disconnected) == {'left-ip', 'right-ip'} and stopped



def test_rgb_only_wrist_does_not_open_a_depth_stream(monkeypatch):
    pipelines = fake_sdk(monkeypatch)
    camera = RealSenseD435iSource(cameras=(D435iCameraConfig('front','front-id'),D435iCameraConfig('right_wrist','side-id')),
                                 width=640,height=480,fps=15,rgb_only_views=('right_wrist',))
    camera.start()
    try:
        assert 'depth' in pipelines[0].config.streams
        assert 'depth' not in pipelines[1].config.streams
        assert camera.get_stream_info()['right_wrist']['depth'] is None
        frames = camera.read()
        assert frames['right_wrist'].rgb.shape == (480,640,3)
        assert not frames['right_wrist'].depth_m.any()
        assert frames['front'].depth_m.max() > 0
    finally:
        camera.stop()
