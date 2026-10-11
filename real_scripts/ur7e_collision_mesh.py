"""Official UR7e collision-mesh FK helpers in the UR controller base frame."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, minimum_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLISION_ROOT = REPO_ROOT / "assets" / "robot_models" / "ur_description" / "meshes" / "ur5e" / "collision"
MESH_NAMES = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation: Rz(yaw) Ry(pitch) Rx(roll)."""
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def link_and_collision_transforms(qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``^base T_collision_mesh`` for official UR7e collision meshes.

    The transforms are the official ur7e kinematics YAML joint origins plus
    collision origins from ``ur_macro.xacro``.  ``base`` is in the controller
    base convention used by the RTDE joint angles and camera calibration.
    """
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size < 6:
        raise ValueError("qpos must contain six UR joint angles")
    q = q[:6]
    links: dict[str, np.ndarray] = {}
    links["base"] = np.eye(4, dtype=np.float64)
    shoulder = _translation(0.0, 0.0, 0.1625) @ _rotation_z(q[0])
    upperarm = shoulder @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[1])
    forearm = upperarm @ _translation(-0.425, 0.0, 0.0) @ _rotation_z(q[2])
    wrist1 = forearm @ _translation(-0.3922, 0.0, 0.1333) @ _rotation_z(q[3])
    wrist2 = wrist1 @ _translation(0.0, -0.0997, 0.0) @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[4])
    wrist3 = wrist2 @ _translation(0.0, 0.0996, 0.0) @ _rpy(np.pi / 2.0, np.pi, np.pi) @ _rotation_z(q[5])
    links.update({"shoulder": shoulder, "upperarm": upperarm, "forearm": forearm, "wrist1": wrist1, "wrist2": wrist2, "wrist3": wrist3})
    return {
        "base": links["base"],
        "shoulder": shoulder @ _rotation_z(np.pi),
        "upperarm": upperarm @ _translation(0.0, 0.0, 0.138) @ _rpy(np.pi / 2.0, 0.0, -np.pi / 2.0),
        "forearm": forearm @ _translation(0.0, 0.0, 0.007) @ _rpy(np.pi / 2.0, 0.0, -np.pi / 2.0),
        "wrist1": wrist1 @ _translation(0.0, 0.0, -0.127) @ _rotation_x(np.pi / 2.0),
        "wrist2": wrist2 @ _translation(0.0, 0.0, -0.0997),
        "wrist3": wrist3 @ _translation(0.0, -0.0005, -0.0989) @ _rotation_x(np.pi / 2.0),
    }


def flange_transform(qpos: np.ndarray) -> np.ndarray:
    """Return ``^base T_flange`` in the controller's standard-DH TCP convention.

    This is intentionally separate from :func:`link_and_collision_transforms`:
    the latter includes per-mesh URDF collision origins.  A flange-mounted
    tool (PiKA) instead needs the unmodified final standard-DH frame.  Applying
    the wrist3 collision/visual fixed rotations here would rotate a tool by a
    large, incorrect offset while leaving the flange position unchanged.
    """
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size < 6:
        raise ValueError("qpos must contain six UR joint angles")
    q = q[:6]
    # Standard DH: RotZ(theta) · TransZ(d) · TransX(a) · RotX(alpha).
    a = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
    d = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
    alpha = (np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0)
    transform = np.eye(4, dtype=np.float64)
    for theta, link_a, link_d, link_alpha in zip(q, a, d, alpha):
        transform = transform @ _rotation_z(theta) @ _translation(0.0, 0.0, link_d) @ _translation(link_a, 0.0, 0.0) @ _rotation_x(link_alpha)
    return transform


def load_collision_meshes() -> dict[str, trimesh.Trimesh]:
    return {name: trimesh.load_mesh(COLLISION_ROOT / f"{name}.stl", process=False) for name in MESH_NAMES}


def world_mesh_vertices(qpos: np.ndarray) -> dict[str, np.ndarray]:
    transforms = link_and_collision_transforms(qpos)
    result: dict[str, np.ndarray] = {}
    for name, mesh in load_collision_meshes().items():
        vertices = np.concatenate((np.asarray(mesh.vertices, dtype=np.float64), np.ones((len(mesh.vertices), 1))), axis=1)
        result[name] = (transforms[name] @ vertices.T).T[:, :3]
    return result


def mesh_surface_samples(mesh: trimesh.Trimesh, *, samples_per_face: int = 4) -> np.ndarray:
    """Return deterministic local mesh-surface samples that can be reused per frame."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    count = max(1, int(samples_per_face))
    subdivisions = 1
    while (subdivisions + 1) * (subdivisions + 2) // 2 < count:
        subdivisions += 1
    return np.concatenate(
        [
            a * triangles[:, 0] + b * triangles[:, 1] + c * triangles[:, 2]
            for a, b, c in _barycentric_lattice(subdivisions)
        ],
        axis=0,
    ).astype(np.float32)


def _barycentric_lattice(subdivisions: int) -> list[tuple[float, float, float]]:
    return [
        (
            (i + 1.0 / 3.0) / (subdivisions + 1),
            (j + 1.0 / 3.0) / (subdivisions + 1),
            1.0 - (i + 1.0 / 3.0) / (subdivisions + 1) - (j + 1.0 / 3.0) / (subdivisions + 1),
        )
        for i in range(subdivisions + 1)
        for j in range(subdivisions + 1 - i)
    ]


def collision_surface_samples(*, samples_per_face: int = 4) -> dict[str, np.ndarray]:
    """Build local UR collision samples once, for reuse in a live loop."""
    return {name: mesh_surface_samples(mesh, samples_per_face=samples_per_face) for name, mesh in load_collision_meshes().items()}


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (np.asarray(transform, dtype=np.float64)[:3, :3] @ points.T).T.astype(np.float32) + np.asarray(transform, dtype=np.float64)[:3, 3].astype(np.float32)


def deterministic_mesh_surface_samples(mesh: trimesh.Trimesh, *, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly ``point_count`` stable, area-weighted local surface points.

    Unlike ``trimesh.sample`` this function has no global RNG state.  The
    returned row index is therefore a persistent physical point identity for a
    fixed mesh file and sampler version.
    """
    if int(point_count) < 1:
        raise ValueError("point_count must be >= 1")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = areas > 1e-14
    if not np.any(valid):
        raise ValueError("Mesh has no non-degenerate triangle faces")
    valid_faces = np.flatnonzero(valid)
    cumulative_area = np.cumsum(areas[valid])
    total_area = float(cumulative_area[-1])
    sample_index = np.arange(int(point_count), dtype=np.float64)
    face_offset = (sample_index + 0.5) / float(point_count) * total_area
    selected_valid_index = np.searchsorted(cumulative_area, face_offset, side="right")
    selected_valid_index = np.minimum(selected_valid_index, len(valid_faces) - 1)
    face_indices = valid_faces[selected_valid_index]

    # Two irrational increments give a deterministic low-discrepancy sequence
    # inside each selected triangle.  Mirror the unit square into the triangle.
    u = np.mod((sample_index + 1.0) * 0.6180339887498949, 1.0)
    v = np.mod((sample_index + 1.0) * 0.4142135623730950, 1.0)
    reflected = (u + v) > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    selected = triangles[face_indices]
    points = selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])
    return points.astype(np.float32), face_indices.astype(np.int32)


class UR7eCollisionSurfacePointSampler:
    """Fixed-identity UR7e collision-surface sampler in the controller base frame.

    Each output row has the immutable identity ``(link_index, sample_index)``.
    The class currently covers official UR7e meshes only; a calibrated PiKA
    model must be appended as separate rigid/articulated links rather than
    approximated by the old centre-line gripper segment.
    """

    point_identity_version = "ur7e_collision_surface_v1"

    def __init__(self, *, points_per_link: int) -> None:
        if int(points_per_link) < 2:
            raise ValueError("points_per_link must be >= 2")
        self.points_per_link = int(points_per_link)
        self.link_names = MESH_NAMES
        meshes = load_collision_meshes()
        local_points = []
        face_indices = []
        hasher = hashlib.sha256()
        for name in self.link_names:
            mesh_path = COLLISION_ROOT / f"{name}.stl"
            hasher.update(mesh_path.read_bytes())
            points, faces = deterministic_mesh_surface_samples(meshes[name], point_count=self.points_per_link)
            local_points.append(points)
            face_indices.append(faces)
        self.local_link_points = np.stack(local_points).astype(np.float32)
        self.face_indices = np.stack(face_indices).astype(np.int32)
        link_index = np.arange(len(self.link_names), dtype=np.int32)[:, None]
        sample_index = np.arange(self.points_per_link, dtype=np.int32)[None, :]
        self.point_ids = np.stack(
            (
                np.broadcast_to(link_index, (len(self.link_names), self.points_per_link)),
                np.broadcast_to(sample_index, (len(self.link_names), self.points_per_link)),
            ),
            axis=-1,
        )
        self.mesh_model_hash = hasher.hexdigest()

    def link_points(self, qpos: np.ndarray) -> np.ndarray:
        transforms = link_and_collision_transforms(qpos)
        return np.stack(
            [_transform_points(self.local_link_points[index], transforms[name]) for index, name in enumerate(self.link_names)]
        ).astype(np.float32)


class UR7ePikaCollisionSurfacePointSampler(UR7eCollisionSurfacePointSampler):
    """UR7e surfaces plus a calibrated, rigid PiKA collision surface.

    The PiKA vendor mesh is currently treated as one rigid flange-mounted link.
    This is valid only while its moving-finger kinematics have not been
    calibrated.  It deliberately refuses an empty/example mount transform so
    a safety dataset cannot silently use an invented tool pose.
    """

    point_identity_version = "ur7e_pika_collision_surface_rigid_v1"

    def __init__(
        self,
        *,
        points_per_link: int,
        pika_mount_transform_json: Path,
        pika_full_collision_mesh: Path,
    ) -> None:
        super().__init__(points_per_link=points_per_link)
        payload = json.loads(Path(pika_mount_transform_json).read_text(encoding="utf-8"))
        transform = np.asarray(payload.get("flange_to_pika_step_frame", payload), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("PiKA mount transform must be a finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError("PiKA mount transform has an invalid final homogeneous row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("PiKA mount transform rotation must be orthonormal with determinant +1")

        pika_path = Path(pika_full_collision_mesh)
        if not pika_path.is_file():
            raise FileNotFoundError(f"PiKA collision mesh does not exist: {pika_path}")
        pika_mesh = trimesh.load_mesh(pika_path, process=False)
        pika_mesh.vertices = np.asarray(pika_mesh.vertices, dtype=np.float64) * 0.001  # source STEP/STL is millimetres
        pika_points, pika_faces = deterministic_mesh_surface_samples(pika_mesh, point_count=self.points_per_link)

        self.pika_mount_transform = transform
        self.pika_mesh = pika_mesh
        self.link_names = tuple((*MESH_NAMES, "pika_gripper_rigid"))
        self.local_link_points = np.concatenate((self.local_link_points, pika_points[None, ...]), axis=0).astype(np.float32)
        self.face_indices = np.concatenate((self.face_indices, pika_faces[None, ...]), axis=0).astype(np.int32)
        link_index = np.arange(len(self.link_names), dtype=np.int32)[:, None]
        sample_index = np.arange(self.points_per_link, dtype=np.int32)[None, :]
        self.point_ids = np.stack(
            (
                np.broadcast_to(link_index, (len(self.link_names), self.points_per_link)),
                np.broadcast_to(sample_index, (len(self.link_names), self.points_per_link)),
            ),
            axis=-1,
        )
        hasher = hashlib.sha256()
        hasher.update(self.mesh_model_hash.encode("ascii"))
        hasher.update(pika_path.read_bytes())
        hasher.update(np.asarray(transform, dtype=np.float64).tobytes())
        self.mesh_model_hash = hasher.hexdigest()

    def link_points(self, qpos: np.ndarray) -> np.ndarray:
        transforms = link_and_collision_transforms(qpos)
        arm_points = np.stack(
            [_transform_points(self.local_link_points[index], transforms[name]) for index, name in enumerate(MESH_NAMES)]
        ).astype(np.float32)
        pika_to_base = flange_transform(qpos) @ self.pika_mount_transform
        pika_points = _transform_points(self.local_link_points[-1], pika_to_base)
        return np.concatenate((arm_points, pika_points[None, ...]), axis=0).astype(np.float32)


def sample_collision_surface_points(
    qpos: np.ndarray,
    *,
    samples_per_face: int = 4,
    local_samples: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Deterministically sample the official collision surfaces in ``base``."""
    transforms = link_and_collision_transforms(qpos)
    samples = local_samples or collision_surface_samples(samples_per_face=samples_per_face)
    return np.concatenate([_transform_points(samples[name], transforms[name]) for name in MESH_NAMES], axis=0).astype(np.float32)


def sample_mesh_surface_points(mesh: trimesh.Trimesh, transform: np.ndarray, *, samples_per_face: int = 4) -> np.ndarray:
    """Deterministically sample one mesh surface after a supplied 4x4 pose."""
    return _transform_points(mesh_surface_samples(mesh, samples_per_face=samples_per_face), transform)


def render_surface_points_depth(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
    splat_radius_pixels: int = 2,
    distortion: np.ndarray | None = None,
    distortion_model: str = "brown_conrady",
) -> np.ndarray:
    """Z-buffer a base-frame point-sampled mesh into a colour-camera depth map."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    camera = (world_to_camera @ np.concatenate((points, np.ones((len(points), 1))), axis=1).T).T[:, :3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-5)
    from real_scripts.camera_geometry import project_pixels
    uv = np.zeros((len(camera), 2), dtype=float)
    uv[valid] = project_pixels(camera[valid], intrinsics, distortion, distortion_model)
    u, v = np.rint(uv).astype(np.int64).T
    valid &= (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    depth = np.full((int(height), int(width)), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v[valid], u[valid]), z[valid].astype(np.float32))
    radius = max(0, int(splat_radius_pixels))
    if radius:
        depth = minimum_filter(depth, size=2 * radius + 1, mode="constant", cval=np.inf)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def occupied_collision_voxels(
    qpos: np.ndarray,
    *,
    voxel_pitch_m: float = 0.006,
    exterior_margin_m: float = 0.015,
    extra_meshes: tuple[tuple[trimesh.Trimesh, np.ndarray], ...] = (),
) -> tuple[np.ndarray, float]:
    """Return centres of filled collision-volume voxels in the UR base frame.

    Vendor collision STLs are not perfectly watertight. ``trimesh.voxelized``
    followed by ``fill`` repairs small openings into a conservative enclosed
    volume; a Chebyshev-neighbour expansion supplies the requested exterior
    pose/depth tolerance.  This is deliberately a *volume* representation,
    not a surface z-buffer, so points inside the robot are also rejected.
    """
    pitch = float(voxel_pitch_m)
    margin = float(exterior_margin_m)
    if pitch <= 0.0 or margin < 0.0:
        raise ValueError("voxel_pitch_m must be > 0 and exterior_margin_m must be >= 0")
    transforms = link_and_collision_transforms(qpos)
    mesh_items: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    mesh_items.extend((mesh, transforms[name]) for name, mesh in load_collision_meshes().items())
    mesh_items.extend(extra_meshes)
    indices: list[np.ndarray] = []
    for mesh, transform in mesh_items:
        world_mesh = mesh.copy()
        world_mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        # PiKA meshes use millimetres at source; callers pass a metres-scaled mesh.
        filled = world_mesh.voxelized(pitch).fill()
        if len(filled.points):
            indices.append(np.rint(np.asarray(filled.points, dtype=np.float64) / pitch).astype(np.int64))
    if not indices:
        return np.zeros((0, 3), dtype=np.float32), pitch
    occupied = np.unique(np.concatenate(indices, axis=0), axis=0)
    occupied = _expand_voxel_indices(occupied, radius=int(np.ceil(margin / pitch)))
    return (occupied.astype(np.float64) * pitch).astype(np.float32), pitch


def _expand_voxel_indices(indices: np.ndarray, *, radius: int) -> np.ndarray:
    """Chebyshev-expand integer voxels without materializing N×offset arrays."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if len(indices) == 0 or radius <= 0:
        return indices
    lower = indices.min(axis=0) - radius
    upper = indices.max(axis=0) + radius
    grid = np.zeros(tuple((upper - lower + 1).tolist()), dtype=bool)
    grid[tuple((indices - lower).T)] = True
    expanded = binary_dilation(grid, structure=np.ones((2 * radius + 1,) * 3, dtype=bool))
    return np.argwhere(expanded).astype(np.int64) + lower


def mesh_local_filled_voxel_indices(mesh: trimesh.Trimesh, *, voxel_pitch_m: float) -> np.ndarray:
    """Voxelize/fill a local mesh once and return its integer voxel indices."""
    pitch = float(voxel_pitch_m)
    if pitch <= 0.0:
        raise ValueError("voxel_pitch_m must be > 0")
    filled = mesh.voxelized(pitch).fill()
    return np.rint(np.asarray(filled.points, dtype=np.float64) / pitch).astype(np.int64)


def occupied_collision_voxels_from_local_indices(
    qpos: np.ndarray,
    *,
    local_indices: dict[str, np.ndarray],
    voxel_pitch_m: float = 0.006,
    exterior_margin_m: float = 0.015,
    extra_local_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Fast live equivalent of :func:`occupied_collision_voxels`.

    STL voxelization/filling is performed at startup.  Each frame only rotates
    and translates those conservative local voxels, then applies the same
    integer-grid expansion used by the offline implementation.
    """
    pitch, margin = float(voxel_pitch_m), float(exterior_margin_m)
    if pitch <= 0.0 or margin < 0.0:
        raise ValueError("voxel_pitch_m must be > 0 and exterior_margin_m must be >= 0")
    transforms = link_and_collision_transforms(qpos)
    indices: list[np.ndarray] = []
    for name in MESH_NAMES:
        local = np.asarray(local_indices[name], dtype=np.float64) * pitch
        transformed = _transform_points(local, transforms[name])
        indices.append(np.rint(transformed / pitch).astype(np.int64))
    if extra_local_indices is not None:
        local, transform = extra_local_indices
        transformed = _transform_points(np.asarray(local, dtype=np.float64) * pitch, transform)
        indices.append(np.rint(transformed / pitch).astype(np.int64))
    occupied = np.unique(np.concatenate(indices, axis=0), axis=0)
    occupied = _expand_voxel_indices(occupied, radius=int(np.ceil(margin / pitch)))
    return (occupied.astype(np.float64) * pitch).astype(np.float32), pitch


def collision_volume_keep_mask(points_base: np.ndarray, occupied_voxel_centres: np.ndarray, *, voxel_pitch_m: float) -> np.ndarray:
    """True for points outside a precomputed filled-and-expanded collision volume."""
    points = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    centres = np.asarray(occupied_voxel_centres, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(centres) == 0:
        return np.ones(len(points), dtype=bool)
    pitch = float(voxel_pitch_m)
    occupied_indices = np.rint(centres / pitch).astype(np.int64)
    point_indices = np.rint(points / pitch).astype(np.int64)
    # Structured dtypes give NumPy a vectorized exact integer row-membership
    # test, replacing the prior Python set/tuple loop over every RGB-D point.
    row_dtype = np.dtype((np.void, point_indices.dtype.itemsize * 3))
    occupied_rows = np.ascontiguousarray(occupied_indices).view(row_dtype).ravel()
    point_rows = np.ascontiguousarray(point_indices).view(row_dtype).ravel()
    return ~np.isin(point_rows, occupied_rows, assume_unique=False)


def render_collision_depth(
    qpos: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
    samples_per_face: int = 4,
    splat_radius_pixels: int = 2,
    local_samples: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Rasterize sampled URDF collision surfaces to a conservative z-buffer."""
    points = sample_collision_surface_points(qpos, samples_per_face=samples_per_face, local_samples=local_samples)
    return render_surface_points_depth(
        points, camera_to_world, intrinsics, width=width, height=height, splat_radius_pixels=splat_radius_pixels,
    )
