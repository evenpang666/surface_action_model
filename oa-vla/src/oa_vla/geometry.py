"""Rotated rectangle geometry; angles use (cos(2 theta), sin(2 theta))."""

import numpy as np
import torch
from torch.nn import functional as F


def canonical_obbs(boxes):
    """Torch [...,6] -> long-edge canonical boxes with unit angle vectors."""
    swap = boxes[..., 2] < boxes[..., 3]
    size = torch.stack((boxes[..., 2:4].amax(-1), boxes[..., 2:4].amin(-1)), -1)
    vector = boxes[..., 4:6]
    # A nonzero fallback prevents an undefined orientation at zero initialization.
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1
    vector = torch.where(vector.norm(dim=-1, keepdim=True) > 1e-6, vector, fallback)
    vector = F.normalize(vector, dim=-1)
    vector = torch.where(swap[..., None], -vector, vector)
    return torch.cat((boxes[..., :2], size, vector), -1)


def obb_enclosures(boxes):
    """Normalized OBB -> clipped axis-aligned cxcywh envelope for native SAM RPB."""
    c2 = boxes[..., 4].clamp(-1, 1)
    # Epsilon stabilizes derivatives at multiples of 90 degrees.
    cosine = ((1 + c2) / 2 + 1e-8).sqrt()
    sine = ((1 - c2) / 2 + 1e-8).sqrt()
    width, height = boxes[..., 2], boxes[..., 3]
    extent = torch.stack((cosine * width + sine * height, sine * width + cosine * height), -1)
    lo, hi = (boxes[..., :2] - extent / 2).clamp(0, 1), (boxes[..., :2] + extent / 2).clamp(0, 1)
    return torch.cat(((lo + hi) / 2, (hi - lo).clamp_min(1e-6)), -1)


def gaussian_overlap(first, second):
    """Pairwise Bhattacharyya coefficient for rectangle-induced Gaussians.

    This smooth rotated-overlap surrogate uses each rectangle's center and
    covariance (side variance = side_length**2 / 12). It is not polygon IoU.
    """
    def covariance(box):
        w2, h2 = box[..., 2].square(), box[..., 3].square()
        mean, delta = (w2 + h2) / 24, (w2 - h2) / 24
        return mean + delta * box[..., 4], mean - delta * box[..., 4], delta * box[..., 5]

    a1, b1, c1 = (v[:, None] for v in covariance(first))
    a2, b2, c2 = (v[None, :] for v in covariance(second))
    a, b, c = (a1 + a2) / 2, (b1 + b2) / 2, (c1 + c2) / 2
    eps = 1e-20
    determinant = (a * b - c.square()).clamp_min(eps)
    d1, d2 = (a1 * b1 - c1.square()).clamp_min(eps), (a2 * b2 - c2.square()).clamp_min(eps)
    dx = first[:, None, 0] - second[None, :, 0]
    dy = first[:, None, 1] - second[None, :, 1]
    distance = ((b * dx.square() + a * dy.square() - 2 * c * dx * dy) / determinant / 8
                + .5 * (determinant.log() - .5 * (d1.log() + d2.log())))
    return (-distance.clamp_min(0)).exp()


def pixel_obb_to_normalized(values, height, width):
    """Original pixel cx,cy,w,h,degrees -> padded normalized six-vector.

    Side lengths use one isotropic scale; integer resize rounding only affects
    center placement. This convention exactly round-trips original OBB geometry.
    """
    from .data import letterbox_geometry
    cx, cy, w, h, angle = map(float, values)
    if not np.isfinite([cx, cy, w, h, angle]).all() or not (0 <= cx < width and 0 <= cy < height) or min(w, h) < 1:
        raise ValueError("Invalid original-pixel OBB")
    if w < h:
        w, h, angle = h, w, angle + 90
    rh, rw, top, left = letterbox_geometry(height, width)
    theta = np.deg2rad(2 * angle)
    return np.array([(cx * rw / width + left) / 224, (cy * rh / height + top) / 224,
                     w / max(height, width), h / max(height, width), np.cos(theta), np.sin(theta)], np.float32)


def original_obb_coordinates(boxes, height, width):
    """Padded six-vectors -> original pixel [cx,cy,w,h,angle_deg] and corners.

    Corners are deliberately not clipped: clipping a rotated box changes its
    geometry. Consumers may clip its rendered polygon against the RGB boundary.
    """
    from .data import letterbox_geometry
    boxes = np.asarray(boxes, np.float32)
    rh, rw, top, left = letterbox_geometry(height, width)
    result = np.empty((*boxes.shape[:-1], 5), np.float32)
    result[..., 0] = (boxes[..., 0] * 224 - left) * width / rw
    result[..., 1] = (boxes[..., 1] * 224 - top) * height / rh
    result[..., 2:4] = boxes[..., 2:4] * max(height, width)
    result[..., 4] = np.rad2deg(np.arctan2(boxes[..., 5], boxes[..., 4])) / 2
    angle = np.deg2rad(result[..., 4])
    u = np.stack((np.cos(angle), np.sin(angle)), -1) * result[..., 2:3] / 2
    v = np.stack((-np.sin(angle), np.cos(angle)), -1) * result[..., 3:4] / 2
    center = result[..., :2]
    corners = np.stack((center - u - v, center + u - v, center + u + v, center - u + v), -2)
    return result, corners
