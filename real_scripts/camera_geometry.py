"""Projection and depth lifting in RealSense color optical coordinates."""

# Inverse Brown handling follows librealsense's vectorized SDK equations:
# https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp

import numpy as np


def _parameters(distortion, model):
    coefficients = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float).reshape(-1)
    if coefficients.size not in (4, 5, 8, 12, 14) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid lens distortion coefficients")
    if model not in ("none", "brown_conrady", "inverse_brown_conrady"):
        raise ValueError(f"Unsupported camera distortion model: {model}")
    return coefficients, model != "none" and bool(np.any(coefficients))


def pixel_rays(pixels, intrinsics, distortion=None, model="brown_conrady"):
    """Unproject pixels to rays with z=1, accounting for lens distortion."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    K = np.asarray(intrinsics, dtype=float)
    coefficients, distorted = _parameters(distortion, model)
    xy = (pixels - K[:2, 2])/np.array([K[0, 0], K[1, 1]])
    if distorted and len(xy):
        import cv2
        if model == "brown_conrady":
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, coefficients).reshape(-1, 2)
        else:
            if len(coefficients) != 5:
                raise ValueError("RealSense inverse Brown requires five coefficients")
            original = xy.copy()
            k1, k2, p1, p2, k3 = coefficients
            for _ in range(10):
                radius2 = (xy*xy).sum(-1)
                reciprocal = 1/(1+radius2*(k1+radius2*(k2+radius2*k3)))
                radial = xy/reciprocal[:, None]
                x, y = radial.T
                tangential = np.column_stack((2*p1*x*y+p2*(radius2+2*x*x),
                                              2*p2*x*y+p1*(radius2+2*y*y)))
                xy = (original-tangential)*reciprocal[:, None]
    return np.column_stack((xy, np.ones(len(xy))))


def project_pixels(camera_points, intrinsics, distortion=None, model="brown_conrady"):
    """Project finite points in front of a color camera, retaining raw pixels."""
    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    K = np.asarray(intrinsics, dtype=float)
    coefficients, distorted = _parameters(distortion, model)
    xy = points[:, :2]/points[:, 2:3]
    if distorted and len(points):
        import cv2
        if model == "brown_conrady":
            return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, coefficients)[0].reshape(-1, 2)
        if len(coefficients) != 5:
            raise ValueError("RealSense inverse Brown requires five coefficients")
        k1, k2, p1, p2, k3 = coefficients
        radius2 = (xy*xy).sum(-1)
        radial = 1+radius2*(k1+radius2*(k2+radius2*k3))
        x, y = (xy*radial[:, None]).T
        xy = np.column_stack((x+2*p1*x*y+p2*(radius2+2*x*x),
                              y+2*p2*x*y+p1*(radius2+2*y*y)))
    return xy*np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
