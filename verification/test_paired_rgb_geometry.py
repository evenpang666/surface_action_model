"""Continuous transparent-volume coverage, calibration and depth-free bounds."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_validation.paired_rgb_geometry import transparent_obstacle, PairedRegionError
from real_scripts.camera_geometry import project_pixels
from real_scripts.manual_obstacle_selection import pixels_in_obb
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_cbf_qp import OrientedBox
from safety_module.occupancy import LocalOccupancyGrid


def scene():
    K = np.array([[100.,0.,64.],[0.,100.,48.],[0.,0.,1.]])
    side = np.eye(4); side[0,3] = .2
    cameras = {name: CameraCalibration(name,K,pose,distortion_model='none')
               for name, pose in [('front',np.eye(4)),('side',side)]}
    boxes = {name:dict(cx=cx,cy=48.,width=20.,height=12.,angle_deg=20.)
             for name,cx in [('front',64.),('side',44.)]}
    sizes = {name:(128,96) for name in cameras}
    grid = LocalOccupancyGrid([-.4,-.2,.9],np.eye(3),[12,4,4],.1)
    return cameras,boxes,sizes,grid


@pytest.mark.parametrize('distortion', [False,True])
def test_ellipsoid_covers_paired_region_without_any_depth_or_occupied_cells(distortion):
    cameras,boxes,sizes,grid = scene()
    if distortion:
        cameras = {name: replace(camera, distortion_model='brown_conrady',
                                distortion=np.array([.08,-.01,.001,-.001,0.]))
                   for name,camera in cameras.items()}
    item = transparent_obstacle('glass',boxes,cameras,sizes,grid)
    random = np.random.default_rng(12)
    points = grid.origin + random.random((30000,3)) * grid.shape * grid.resolution_m
    keep = np.ones(len(points),bool)
    for name,camera in cameras.items():
        local = (points-camera.camera_to_world[:3,3]) @ camera.camera_to_world[:3,:3]
        pixels = project_pixels(local,camera.intrinsics,camera.distortion,camera.distortion_model)
        keep &= pixels_in_obb(pixels,boxes[name])
    assert keep.sum() > 50
    delta = points[keep] - item['center']
    radius = np.einsum('ni,ij,nj->n',delta,np.linalg.inv(item['ellipsoid_shape']),delta)
    assert radius.max() <= 1+1e-6
    vertices = np.asarray(item['region_vertices'])
    delta = vertices-item['center']
    assert np.einsum('ni,ij,nj->n',delta,np.linalg.inv(item['ellipsoid_shape']),delta).max() <= 1+1e-6
    assert np.all(vertices >= grid.origin-1e-8)
    assert np.all(vertices <= grid.origin+grid.shape*grid.resolution_m+1e-8)
    assert item['voxel_count'] == 0 and item['depth_measured'] is False


def test_volume_is_equivariant_to_shared_camera_and_grid_rotation():
    cameras,boxes,sizes,grid = scene()
    original = transparent_obstacle('glass',boxes,cameras,sizes,grid)
    rotation = Rotation.from_euler('xyz',[.3,.4,-.2]).as_matrix()
    shift = np.array([.5,-.7,.2])
    for camera in cameras.values():
        camera.camera_to_world[:3,:3] = rotation @ camera.camera_to_world[:3,:3]
        camera.camera_to_world[:3,3] = rotation @ camera.camera_to_world[:3,3] + shift
    transformed = LocalOccupancyGrid(rotation @ grid.origin+shift, rotation, grid.shape, grid.resolution_m)
    result = transparent_obstacle('glass',boxes,cameras,sizes,transformed)
    vertices = np.asarray(original['region_vertices']) @ rotation.T + shift
    delta = vertices - result['center']
    assert np.einsum('ni,ij,nj->n',delta,np.linalg.inv(result['ellipsoid_shape']),delta).max() <= 1+1e-6


def test_nonintersecting_views_do_not_create_an_obstacle_elsewhere():
    cameras,boxes,sizes,grid = scene()
    boxes['side']['cx'] = 100.
    with pytest.raises(PairedRegionError, match='no 3D intersection'):
        transparent_obstacle('wrong_pair',boxes,cameras,sizes,grid)


@pytest.mark.parametrize('shape', [np.diag([1.,1.,-1.]), np.eye(3)*4, np.full((3,3),np.nan)])
def test_malformed_direct_ellipsoids_cannot_bypass_box_validation(shape):
    with pytest.raises(ValueError, match='ellipsoid'):
        OrientedBox(np.zeros(3),np.eye(3),np.ones(3),shape)
