"""Euclidean distances for swept fixed surface points."""

import numpy as np
from scipy.spatial import cKDTree


def swept_inter_arm_safe(previous, current, left_count, margin):
    """Exact linear swept separation, pruning pairs by their travel bounds."""
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    travel = np.linalg.norm(current-previous, axis=-1)
    # A pair initially farther apart than margin plus both travel lengths
    # cannot meet during this segment. All remaining pairs are checked exactly.
    radii = margin+travel[:left_count]+travel[left_count:].max()+1e-12
    neighbours = cKDTree(previous[left_count:]).query_ball_point(previous[:left_count], r=radii)
    lengths = np.fromiter((len(x) for x in neighbours), dtype=int, count=left_count)
    if not lengths.any():
        return True
    left = np.repeat(np.arange(left_count), lengths)
    right = np.concatenate([x for x in neighbours if len(x)]).astype(int)+left_count
    first = previous[left]-previous[right]
    delta = (current[left]-current[right])-first
    t = np.clip(-(first*delta).sum(axis=-1)/np.maximum((delta*delta).sum(axis=-1), 1e-15), 0., 1.)
    return not np.any(np.linalg.norm(first+t[:, None]*delta, axis=-1) <= margin+1e-12)


def segments_within_aabb_margin(start, end, half_sizes, margin):
    """Test corresponding local point segments against centred axis-aligned boxes.

    ``half_sizes`` may be one (3,) box or one (N,3) box per segment. Expanded
    slabs only select candidates; rounded edges and corners use exact minimum
    Euclidean distance along each segment.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    half = np.broadcast_to(np.asarray(half_sizes, float), start.shape)
    direction = end-start
    parallel = np.abs(direction) < 1e-12
    expanded = half+margin+1e-12
    outside = np.any(parallel & (np.abs(start) > expanded), axis=-1)
    divisor = np.where(parallel, 1., direction)
    first, last = (-expanded-start)/divisor, (expanded-start)/divisor
    entry = np.where(parallel, -np.inf, np.minimum(first, last)).max(axis=-1)
    exit = np.where(parallel, np.inf, np.maximum(first, last)).min(axis=-1)
    candidate = ~outside & (np.maximum(entry, 0.) <= np.minimum(exit, 1.))
    if margin == 0 or not candidate.any():
        return candidate
    a, d, half = start[candidate], direction[candidate], half[candidate]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = np.concatenate(((half-a)/d, (-half-a)/d), axis=1)
    crossings = np.nan_to_num(crossings, nan=0., posinf=1., neginf=0.)
    boundaries = np.sort(np.column_stack((np.zeros(len(a)), np.clip(crossings, 0, 1),
                                          np.ones(len(a)))), axis=1)
    middle = (boundaries[:, :-1]+boundaries[:, 1:])/2
    positions = a[:, None, :]+middle[..., None]*d[:, None, :]
    side = np.where(positions > half[:, None, :], 1.,
                    np.where(positions < -half[:, None, :], -1., 0.))
    active = side != 0
    offset = a[:, None, :]-side*half[:, None, :]
    numerator = (offset*d[:, None, :]*active).sum(axis=-1)
    denominator = (d[:, None, :]**2*active).sum(axis=-1)
    optimum = np.clip(-numerator/np.maximum(denominator, 1e-30),
                      boundaries[:, :-1], boundaries[:, 1:])
    distances = np.maximum(np.abs(a[:, None, :]+optimum[..., None]*d[:, None, :])
                           - half[:, None, :], 0.)
    result = np.zeros_like(candidate)
    result[candidate] = (np.square(distances).sum(axis=-1) <= (margin+1e-12)**2).any(axis=-1)
    return result
