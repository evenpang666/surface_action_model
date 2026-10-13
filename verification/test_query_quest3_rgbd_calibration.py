"""The optional SDK calibration inventory remains a separate hardware tool."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.query_quest3_rgbd_calibration import query_calibrations


def fake_sdk():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])

    def profile(stream, format_, width, height):
        intr = NS(width=width, height=height, fx=400., fy=400., ppx=width/2, ppy=height/2,
                  coeffs=[0.]*5, model="distortion.inverse_brown_conrady")
        result = NS(stream_type=lambda: stream, format=lambda: format_, fps=lambda: 30,
                    width=lambda: width, height=lambda: height, get_intrinsics=lambda: intr,
                    get_extrinsics_to=lambda other: NS(rotation=rotation.ravel(order="F").tolist(),
                                                       translation=[-.06, .001, .002]))
        result.as_video_stream_profile = lambda: result
        return result

    def device(serial):
        profiles = [profile("depth", "z16", 640, 480), profile("color", "rgb8", 1280, 720),
                    profile("color", "rgb8", 640, 480)]
        return NS(get_info=lambda key: serial if key == "serial" else "RealSense D455",
                  query_sensors=lambda: [NS(get_stream_profiles=lambda: profiles)],
                  first_depth_sensor=lambda: NS(get_depth_scale=lambda: .001))

    rs = NS(context=lambda: NS(query_devices=lambda: [device("side-serial"), device("front-serial")]),
            camera_info=NS(serial_number="serial", name="name"),
            stream=NS(depth="depth", color="color"), format=NS(z16="z16", rgb8="rgb8"))
    return rs, rotation


def kwargs():
    return dict(serials={"front": "front-serial", "side": "side-serial"},
                color_profiles={"front": (1280, 720, 30), "side": (640, 480, 30)},
                depth_profile=(640, 480, 30))


def test_query_matches_serial_and_restores_column_major_extrinsics():
    rs, rotation = fake_sdk()
    result = query_calibrations(rs, **kwargs())
    front, side = result["cameras"]["front"], result["cameras"]["side"]
    assert front["camera_serial"] == "front-serial"
    assert side["color_intrinsics"]["width"] == 640
    assert front["color_intrinsics"]["width"] == 1280
    assert front["color_intrinsics"]["distortion_model"] == "inverse_brown_conrady"
    np.testing.assert_array_equal(np.array(front["depth_to_color"])[:3, :3], rotation)
    np.testing.assert_allclose(np.array(front["depth_to_color"])[:3, 3], [-.06, .001, .002])


def test_query_missing_device_fails_without_substituting_another_camera():
    rs, _ = fake_sdk()
    args = kwargs()
    args["serials"]["front"] = "missing"
    with pytest.raises(RuntimeError, match="not connected"):
        query_calibrations(rs, **args)


def test_query_unsupported_recording_profile_is_not_silently_resized():
    rs, _ = fake_sdk()
    args = kwargs()
    args["depth_profile"] = (848, 480, 30)
    with pytest.raises(ValueError, match="actual native recording profile"):
        query_calibrations(rs, **args)
