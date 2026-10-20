"""Sample official PickNik Robotiq description meshes for visual alignment.

These meshes are deliberately used only by the inspection visualizer.  The
live RGB-D safety mask uses the bundled fixed collision URDF envelopes because
the controller does not publish a trustworthy 2F-85 finger angle.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np

from real_scripts.ur7e_collision_mesh import deterministic_mesh_surface_samples


VENDOR_ROOT = Path(__file__).resolve().parents[1] / "assets" / "robot_models" / "vendor"
TWO_F85_COLLISION = VENDOR_ROOT / "ros2_robotiq_gripper" / "robotiq_description" / "meshes" / "collision" / "2f_85"
EPICK_VISUAL = VENDOR_ROOT / "ros2_epick_gripper" / "epick_description" / "meshes" / "visual" / "epick_body.stl"
TWO_F85_MESH_NAMES = (
    "robotiq_base.stl",
    "ur_to_robotiq_adapter.stl",
    "left_knuckle.stl",
    "right_knuckle.stl",
    "left_finger.stl",
    "right_finger.stl",
    "left_inner_knuckle.stl",
    "right_inner_knuckle.stl",
    "left_finger_tip.stl",
    "right_finger_tip.stl",
)
TWO_F85_LINK_NAMES = tuple(f"robotiq_2f85_{name.removesuffix('.stl')}" for name in TWO_F85_MESH_NAMES)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4); result[:3, 3] = (x, y, z); return result


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle); return np.asarray(((c, 0., s, 0.), (0., 1., 0., 0.), (-s, 0., c, 0.), (0., 0., 0., 1.)))


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return ((transform[:3, :3] @ points.T).T + transform[:3, 3]).astype(np.float32)


@lru_cache(maxsize=None)
def _mesh_points_cached(path_text: str, count: int, scale: float) -> np.ndarray:
    import trimesh
    mesh = trimesh.load_mesh(path_text, process=False)
    points, _ = deterministic_mesh_surface_samples(mesh, point_count=count)
    return np.asarray(points, dtype=np.float64) * float(scale)


def _mesh_points(path: Path, count: int, *, scale: float = 1.0) -> np.ndarray:
    return _mesh_points_cached(str(path.resolve()), int(count), float(scale))


def robotiq_2f85_link_points_in_gripper_base(*, finger_angle_rad: float = 0.0, samples_per_link: int = 850) -> np.ndarray:
    """Return stable upstream 2F-85 collision samples as ``[10, P, 3]``.

    Rows are ordered by :data:`TWO_F85_LINK_NAMES`, so every point has a
    persistent physical link/sample identity across changing finger angles.
    Coordinates are in the upstream Robotiq gripper-base frame.
    """
    if not TWO_F85_COLLISION.is_dir():
        raise FileNotFoundError(f"Missing vendor 2F-85 description: {TWO_F85_COLLISION}")
    transforms = robotiq_2f85_link_transforms(finger_angle_rad)
    return np.stack([_transform(_mesh_points(TWO_F85_COLLISION / name, samples_per_link), transforms[name]) for name in TWO_F85_MESH_NAMES]).astype(np.float32)


def robotiq_2f85_link_transforms(finger_angle_rad: float = 0.0) -> dict[str, np.ndarray]:
    """Mesh-local to gripper-base poses, shared by samples and volume models."""
    q = float(finger_angle_rad)
    return {
        "robotiq_base.stl": np.eye(4),
        "ur_to_robotiq_adapter.stl": _translation(0., 0., -.011),
        "left_knuckle.stl": _translation(.03060114, 0., .05490452) @ _rotation_y(-q),
        "right_knuckle.stl": _translation(-.03060114, 0., .05490452) @ _rotation_y(q),
        "left_finger.stl": _translation(.03060114, 0., .05490452) @ _rotation_y(-q) @ _translation(.03152616, 0., -.00376347),
        "right_finger.stl": _translation(-.03060114, 0., .05490452) @ _rotation_y(q) @ _translation(-.03152616, 0., -.00376347),
        "left_inner_knuckle.stl": _translation(.0127, 0., .06142) @ _rotation_y(-q),
        "right_inner_knuckle.stl": _translation(-.0127, 0., .06142) @ _rotation_y(q),
        "left_finger_tip.stl": _translation(.03060114, 0., .05490452) @ _rotation_y(-q) @ _translation(.03152616, 0., -.00376347) @ _translation(.00563134, 0., .04718515) @ _rotation_y(q),
        "right_finger_tip.stl": _translation(-.03060114, 0., .05490452) @ _rotation_y(q) @ _translation(-.03152616, 0., -.00376347) @ _translation(-.00563134, 0., .04718515) @ _rotation_y(-q),
    }


def robotiq_2f85_points_in_active_tcp(*, finger_angle_rad: float = 0.0, samples_per_link: int = 850, base_to_active_tcp_m: float = .1493) -> np.ndarray:
    """Return the upstream 2F-85 collision meshes in an active-TCP frame.

    The upstream Xacro has its root at the gripper base.  The local conversion
    places that base behind the active TCP by the open-model fingertip distance
    (149.3 mm).  The larger measured flange-to-TCP distance includes the
    physical wrist-mount segment and must not be applied a second time.
    ``finger_angle_rad=0`` is the open Xacro pose.
    """
    root_to_tcp = _translation(0., 0., -float(base_to_active_tcp_m))
    link_points = robotiq_2f85_link_points_in_gripper_base(finger_angle_rad=finger_angle_rad, samples_per_link=samples_per_link)
    return _transform(link_points.reshape(-1, 3), root_to_tcp)


def robotiq_epick_points_in_active_tcp(*, samples: int = 3000, base_to_active_tcp_m: float = .1173) -> np.ndarray:
    """Return upstream EPick body mesh in the active-TCP frame.

    The upstream Xacro defines 102.3 mm body plus 15 mm cup.  Its total 117.3
    mm base-to-TCP length is used here; the workcell's extra 44.55 mm mount
    segment belongs between flange and EPick base, not inside the EPick mesh.
    The primitive cup remains in the collision URDF.
    """
    if not EPICK_VISUAL.is_file():
        raise FileNotFoundError(f"Missing vendor EPick description: {EPICK_VISUAL}")
    return _transform(_mesh_points(EPICK_VISUAL, samples, scale=.001), _translation(0., 0., -float(base_to_active_tcp_m)))
