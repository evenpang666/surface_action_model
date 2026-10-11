"""Coordinate-graph, FK, lens and stale-calibration regression checks."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from safety_module.dual_camera_calibration import (calibration_from_project, load_dual_camera_calibration,
                                                    validate_calibration_identity)
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config


@pytest.fixture
def bundle(tmp_path):
    """Independent source files; no dependency on the workstation's exports."""
    from scipy.spatial.transform import Rotation
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 100, degrees=True).as_matrix()
    base[:3, 3] = [.1, .7, -.2]
    front = np.eye(4)
    front[:3, :3] = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    front[:3, 3] = [.2, -.3, .8]
    relative = np.eye(4)
    relative[:3, :3] = Rotation.from_euler("xyz", [20, 30, -10], degrees=True).as_matrix()
    relative[:3, 3] = [.4, .1, .2]
    matrices = {"left/front": front, "left/side": front @ relative,
                "right/front": np.linalg.inv(base) @ front, "right/side": np.linalg.inv(base) @ front @ relative}
    references = {}
    for key, matrix in matrices.items():
        arm, view = key.split("/")
        source = tmp_path/(key.replace("/", "_")+".yaml")
        details = {"calibrated": True, "parent_frame": f"{arm}_base", "child_frame": f"{view}_camera_color_optical_frame",
                   "matrix": matrix.tolist(), "camera_serial": view, "touch_rms_mm": 1., "camera_quality_passed": True,
                   "quality_limits": {"max_touch_rms_mm": 2.5},
                   "intrinsics": {"fx": 500., "fy": 510., "ppx": 320., "ppy": 240., "distortion": [0.]*5}}
        source.write_text(yaml.safe_dump(details))
        references[key] = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    frame = lambda matrix, parent, child: {"matrix": matrix.tolist(), "parent_frame": parent, "child_frame": child}
    data = {"calibrated": True, "transform_convention": "parent_T_child maps child coordinates into parent coordinates",
            "left_base_to_right_base": frame(base, "left_base", "right_base"),
            "left_base_to_right_base_via_front": frame(base, "left_base", "right_base"),
            "left_base_to_right_base_via_side": frame(base, "left_base", "right_base"),
            "front_camera_to_side_camera": frame(relative, "front_camera_color_optical_frame", "side_camera_color_optical_frame"),
            "source_calibrations": references, "quality_limits": {"max_translation_mm": 10., "max_rotation_deg": 3.},
            "consistency": {k: {"translation_error_mm": 0., "rotation_error_deg": 0., "passed": True}
                            for k in ("base_via_front_vs_side", "cameras_via_left_vs_right")}}
    path = tmp_path/"dual.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, data, matrices, base, relative


def test_transform_direction_and_camera_closure(bundle):
    path, _, matrices, base, relative = bundle
    calibration = load_dual_camera_calibration(path, expected_serials={"front": "front", "side": "side"})
    np.testing.assert_allclose(calibration.right_base_to_left_base, base)
    np.testing.assert_allclose(calibration.camera_to_left_base["side"], matrices["left/front"] @ relative)
    point = np.array([.12, .23, .34, 1.])
    for view in ("front", "side"):
        np.testing.assert_allclose(calibration.camera_to_left_base[view] @ point,
                                   base @ matrices[f"right/{view}"] @ point, atol=1e-12)


@pytest.mark.parametrize("corruption", ["inverse", "reflection", "source_hash", "quality", "serial", "consistency"])
def test_bad_calibration_is_rejected(bundle, corruption):
    path, original, _, _, _ = bundle
    data = deepcopy(original)
    if corruption == "inverse":
        data["left_base_to_right_base"]["matrix"] = np.linalg.inv(data["left_base_to_right_base"]["matrix"]).tolist()
    elif corruption == "reflection":
        data["left_base_to_right_base"]["matrix"][0][0] *= -1
    elif corruption in ("source_hash", "quality", "serial"):
        source = path.parent/data["source_calibrations"]["left/front"]["path"]
        details = yaml.safe_load(source.read_text())
        if corruption == "quality":
            details["touch_rms_mm"] = 5.
        elif corruption == "serial":
            details["camera_serial"] = "wrong-camera"
        else:
            details["matrix"][0][3] += .1
        source.write_text(yaml.safe_dump(details))
        if corruption != "source_hash":
            data["source_calibrations"]["left/front"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        data["consistency"]["base_via_front_vs_side"]["passed"] = False
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_dual_camera_calibration(path)


def test_perception_and_sampler_share_new_graph_and_tool_fk():
    from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
    from real_scripts import live_dual_ur7e_obstacle_model as live
    from real_scripts.ur7e_collision_mesh import link_and_collision_transforms
    config, _ = load_project_config()
    calibration = calibration_from_project(config)
    left, right, front, report = live.load_arms(SimpleNamespace(project_config=DEFAULT_PROJECT_CONFIG,
                                                            front_serial=None, left_tool_from_tcp=None, right_tool_from_tcp=None))
    sampler = DualUR7eSurfacePointSampler(points_per_link=8)
    np.testing.assert_array_equal(right.base_to_left, sampler.right_base_to_left_base)
    np.testing.assert_array_equal(right.base_to_left, calibration.right_base_to_left_base)
    np.testing.assert_array_equal(front.camera_to_world, calibration.camera_to_left_base["front"])
    assert report["sha256"] == sampler.calibration_digest
    q, grip = np.linspace(-.6, .6, 12), np.array([.3, .7])
    points = sampler.link_points(q, grip)
    count = len(sampler.left_link_names)
    for arm, offset, joints, closing in ((left, 0, q[:6], grip[0]), (right, count, q[6:], grip[1])):
        local_samples = {name: sampler.ur.local_link_points[i] for i, name in enumerate(link_and_collision_transforms(joints))}
        rendered_arm = live.arm_surface_points(joints, arm.base_to_left, local_samples).reshape(7, 8, 3)
        np.testing.assert_allclose(points[offset:offset+7], rendered_arm, atol=2e-7)
        rendered_tool = live.tool_surface_points(joints, closing, arm.base_to_left,
                                                 sampler.flange_to_tcp[f"{arm.name}_arm"], samples_per_link=8)
        np.testing.assert_allclose(points[offset+7:offset+count].reshape(-1, 3), rendered_tool, atol=2e-7)


def test_active_tcp_mismatch_and_old_snapshot_are_rejected():
    from scipy.spatial.transform import Rotation
    from real_scripts.live_dual_ur7e_obstacle_model import validate_active_tcp
    from real_scripts.ur7e_collision_mesh import flange_transform
    q, tool = np.zeros(6), np.eye(4)
    matrix = flange_transform(q) @ tool
    pose = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
    validate_active_tcp(q, pose, tool)
    pose[0] += .01
    with pytest.raises(RuntimeError, match="Active TCP"):
        validate_active_tcp(q, pose, tool)
    with pytest.raises(RuntimeError, match="calibration differs"):
        validate_calibration_identity({"calibration": {"sha256": "old"}}, "new")


@pytest.mark.parametrize("model", ["brown_conrady", "inverse_brown_conrady"])
def test_lens_projection_and_depth_lifting_are_consistent(model):
    from real_scripts.camera_geometry import pixel_rays, project_pixels
    from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame, depth_to_world_points
    from real_scripts.ur7e_collision_mesh import render_surface_points_depth
    K = np.array([[100., 0, 40], [0, 100., 30], [0, 0, 1.]])
    distortion = np.array([-.06, .06, -.0003, .0002, -.02])
    pixels = np.array([[2, 3], [71, 53]], dtype=float)
    rays = pixel_rays(pixels, K, distortion, model)
    np.testing.assert_allclose(project_pixels(rays*.6, K, distortion, model), pixels, atol=1e-5)
    depth = np.zeros((60, 80), np.float32)
    depth[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = .6
    calibration = CameraCalibration("front", K, np.eye(4), distortion, model)
    frame = RGBDFrame("front", np.zeros((60, 80, 3), np.uint8), depth)
    world, _ = depth_to_world_points(frame, calibration)
    np.testing.assert_allclose(world, rays*.6, atol=1e-7)
    rendered = render_surface_points_depth(world, np.eye(4), K, width=80, height=60, splat_radius_pixels=0,
                                           distortion=distortion, distortion_model=model)
    np.testing.assert_allclose(rendered, depth, atol=1e-7)


@pytest.mark.parametrize("model", ["brown_conrady", "inverse_brown_conrady"])
def test_camera_rays_match_realsense_sdk(model):
    rs = pytest.importorskip("pyrealsense2")
    from real_scripts.camera_geometry import pixel_rays, project_pixels
    intr = rs.intrinsics()
    intr.width, intr.height = 1280, 720
    intr.fx, intr.fy, intr.ppx, intr.ppy = 648.95, 647.87, 635.88, 358.45
    intr.coeffs = [-.056, .062, -.0000178, .0001517, -.0203]
    intr.model = getattr(rs.distortion, model)
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
    pixels = np.array([[40., 60.], [1170., 660.]])
    expected = np.array([rs.rs2_deproject_pixel_to_point(intr, pixel.tolist(), 1.) for pixel in pixels])
    actual = pixel_rays(pixels, K, intr.coeffs, model)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    projected = project_pixels(expected, K, intr.coeffs, model)
    expected_pixels = [rs.rs2_project_point_to_pixel(intr, point.tolist()) for point in expected]
    np.testing.assert_allclose(projected, expected_pixels, atol=2e-4)
