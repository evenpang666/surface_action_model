"""Depth-free geometry for paired RGB obstacles, bounded by the workspace grid.

These volumes are visual hypotheses, never synthetic depth/occupancy evidence.
Pinhole OBB frusta are intersected continuously, so even sub-voxel objects work.
"""

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull

from obstacle_validation.ellipsoid_geometry import fit_enclosing
from real_scripts.camera_geometry import pixel_rays


TRANSPARENT_SOURCE = "dual_rgb_transparent_region"


class PairedRegionError(ValueError):
    """The two boxes do not define a usable volume in the workspace."""


class PairedIntersectionEmpty(PairedRegionError):
    """Valid view boxes have no volumetric intersection within the bounds."""


def _image_polygon(box, size):
    angle = np.deg2rad(box["angle_deg"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    polygon = (np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
               * [box["width"] / 2, box["height"] / 2]) @ rotation.T + [box["cx"], box["cy"]]
    # Clip rotated boxes to the actual RGB field of view before unprojection.
    for axis, bound, sign in ((0, 0., -1), (0, float(size[0]), 1),
                              (1, 0., -1), (1, float(size[1]), 1)):
        output = []
        for first, last in zip(polygon, np.roll(polygon, -1, axis=0)):
            a, b = sign * (first[axis] - bound), sign * (last[axis] - bound)
            if a <= 0:
                output.append(first)
            if (a <= 0) != (b <= 0):
                output.append(first + a / (a - b) * (last - first))
        polygon = np.asarray(output).reshape(-1, 2)
        if len(polygon) < 3:
            raise PairedRegionError("Paired RGB box has no area inside its image")
    return polygon


def _camera_planes(box, camera, size):
    polygon = _image_polygon(box, size)
    distorted = (camera.distortion_model != "none" and camera.distortion is not None
                 and np.any(camera.distortion))
    if distorted:
        # Curved distorted boundaries are approximated by an outer angular
        # envelope of densely sampled edges, padded by one original RGB pixel.
        samples = np.concatenate([a + np.linspace(0, 1, 33)[:, None] * (b - a)
                                  for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])
        xy = pixel_rays(samples, camera.intrinsics, camera.distortion, camera.distortion_model)[:, :2]
        angles = np.arange(16) * (2 * np.pi / 16)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        limits = (xy @ normals.T).max(0) + 1 / min(camera.intrinsics[0, 0], camera.intrinsics[1, 1])
        optical = np.column_stack((normals, -limits))
    else:
        xy = pixel_rays(polygon, camera.intrinsics, camera.distortion, camera.distortion_model)[:, :2]
        optical = ConvexHull(xy).equations  # nx*x/z + ny*y/z + offset <= 0
    optical = np.vstack((optical, [0., 0., -1.], [0., 0., 1.]))
    limits = np.r_[np.zeros(len(optical) - 2), -.01, 2.5]
    world = optical @ camera.camera_to_world[:3, :3].T
    limits += world @ camera.camera_to_world[:3, 3]
    norms = np.linalg.norm(world, axis=1)
    return world / norms[:, None], limits / norms


def transparent_obstacle(object_id, boxes, cameras, sizes, grid):
    """Enclose the bounded intersection in one ellipsoid, with an OBB envelope.

    The grid's oriented bounding extent supplies finite workspace bounds, not
    measured geometry. Its empty/free cells remain untouched. This deliberately
    overestimates depth and may also span gaps between configured workspaces.
    """
    planes = [_camera_planes(boxes[name], cameras[name], sizes[name]) for name in ("front", "side")]
    axes = grid.axes.T
    planes.append((np.vstack((axes, -axes)),
                   np.r_[grid.shape * grid.resolution_m + axes @ grid.origin, -axes @ grid.origin]))
    normals = np.concatenate([item[0] for item in planes])
    limits = np.concatenate([item[1] for item in planes])
    triples = np.array(list(combinations(range(len(normals)), 3)))
    matrices = normals[triples]
    valid = np.abs(np.linalg.det(matrices)) > 1e-10
    vertices = np.linalg.solve(matrices[valid], limits[triples[valid]][..., None])[..., 0]
    vertices = vertices[np.all(vertices @ normals.T <= limits + 1e-8, axis=1)]
    vertices = np.unique(np.round(vertices, 10), axis=0)
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-8) < 3:
        raise PairedIntersectionEmpty(f"Object {object_id}: paired RGB frusta have no 3D intersection inside the workspace")
    model = fit_enclosing(vertices, min_axis=.001, key=f"transparent_{object_id}")
    squared, rotation = np.linalg.eigh(model.shape)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return dict(center=model.center.tolist(), axes=rotation.tolist(), half_sizes=np.sqrt(squared).tolist(),
                object_id=str(object_id), geometry_source=TRANSPARENT_SOURCE, voxel_count=0,
                ellipsoid_shape=model.shape.tolist(), region_vertices=vertices.tolist(),
                region_method="bounded_paired_rgb_frusta", depth_measured=False)
