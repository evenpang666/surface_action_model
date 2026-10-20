"""Bounded probability grids and lossless observation contracts for motion checks.

The inverse sensor model follows the occupancy/log-odds convention described
by OctoMap (https://octomap.github.io/). This implementation uses a bounded
NumPy grid, rather than an octree or a dependency on ROS.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
import time
import zlib

import numpy as np

from safety_module.geometry import segments_within_aabb_margin


# Code 0 is accepted only when reading older saved maps.
UNKNOWN, FREE, OCCUPIED, OUTSIDE = 0, 1, 2, 3
MAP_SCHEMA = "local_probabilistic_occupancy_v1"
MAX_VOXELS = 2_000_000


@lru_cache(maxsize=4096)
def _tile_cuboids(tile_voxels, pattern):
    """Pack a local occupancy pattern once; reuse it across tiles and frames."""
    mask = np.frombuffer(pattern, dtype=np.uint8).reshape((tile_voxels,) * 3).astype(bool)
    cuboids = []
    while mask.any():
        low = np.array(np.unravel_index(np.flatnonzero(mask)[0], mask.shape))
        high = low + 1
        for axis in range(3):
            while high[axis] < tile_voxels:
                candidate = high.copy()
                candidate[axis] += 1
                if not mask[tuple(slice(a, b) for a, b in zip(low, candidate))].all():
                    break
                high = candidate
        mask[tuple(slice(a, b) for a, b in zip(low, high))] = False
        cuboids.append((tuple(low), tuple(high)))
    return tuple(cuboids)


def _pack(array):
    return base64.b64encode(zlib.compress(np.asarray(array).tobytes(order="C"), 1)).decode("ascii")


def _unpack(value, dtype, size):
    if not isinstance(value, str) or len(value) > size * np.dtype(dtype).itemsize * 2 + 1024:
        raise ValueError("Invalid occupancy array size")
    decoder = zlib.decompressobj()
    expected = size * np.dtype(dtype).itemsize
    data = decoder.decompress(base64.b64decode(value, validate=True), expected + 1)
    if len(data) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid compressed occupancy array")
    return np.frombuffer(data, dtype=dtype)


class GridGeometry:
    def __init__(self, origin, axes, shape, resolution_m, *, max_voxels=MAX_VOXELS):
        self.origin, self.axes = np.asarray(origin, float), np.asarray(axes, float)
        dimensions = np.asarray(shape)
        if (dimensions.shape != (3,) or not np.issubdtype(dimensions.dtype, np.integer)
                or np.any(dimensions < 1) or np.any(dimensions > max_voxels)):
            raise ValueError("Invalid occupancy grid dimensions")
        self.shape = dimensions.astype(np.int64)
        self.size = int(np.prod(self.shape))
        if (self.size > max_voxels or self.origin.shape != (3,) or self.axes.shape != (3, 3)
                or not np.isfinite(self.origin).all() or not np.isfinite(self.axes).all()
                or not np.allclose(self.axes.T @ self.axes, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.axes), 1., atol=1e-6)
                or not np.isfinite(resolution_m) or resolution_m <= 0):
            raise ValueError("Invalid occupancy grid geometry or voxel budget")
        self.resolution_m = float(resolution_m)
        self.strides = np.array([self.shape[1] * self.shape[2], self.shape[2], 1])

    def coordinates(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("Occupancy query points must be finite")
        return (points - self.origin) @ self.axes / self.resolution_m

    def indices(self, points):
        cells = np.floor(self.coordinates(points) + 1e-10).astype(np.int64)
        inside = np.all((cells >= 0) & (cells < self.shape), axis=1)
        return cells @ self.strides, inside

    def centers(self, indices):
        cells = np.column_stack(np.unravel_index(np.asarray(indices, dtype=np.int64), tuple(self.shape)))
        return self.origin + ((cells + .5) * self.resolution_m) @ self.axes.T

    def ray_cells(self, start, end, *, exclude_endpoint=False, unique_only=False):
        """Vectorized voxel traversal; clip rays to the grid before walking.

        Returns voxel IDs and original ray IDs. Invalid depth rays must be
        omitted by callers. Boundary ties advance together (zero-volume cells
        are not used as free evidence).
        """
        a, b = self.coordinates(start), self.coordinates(end)
        if len(a) == 1 and len(b) != 1:
            a = np.broadcast_to(a, b.shape)
        if a.shape != b.shape:
            raise ValueError("Ray start/end arrays do not align")
        direction = b - a
        parallel = np.abs(direction) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            first = -a / direction
            last = (self.shape - a) / direction
        entry = np.maximum(np.where(parallel, -np.inf, np.minimum(first, last)).max(axis=1), 0.)
        exit = np.minimum(np.where(parallel, np.inf, np.maximum(first, last)).min(axis=1), 1.)
        valid = (exit >= entry) & ~np.any(parallel & ((a < 0) | (a >= self.shape)), axis=1)
        valid &= np.linalg.norm(direction, axis=1) > 1e-12
        ray_ids = np.flatnonzero(valid)
        if not len(ray_ids):
            # A stationary robot point is still a collision query. It is
            # only empty for free-ray evidence, which excludes endpoints.
            if not exclude_endpoint:
                endpoint_ids, inside = self.indices(end)
                ids = endpoint_ids[inside]
                if unique_only:
                    return np.unique(ids), np.empty(0, np.int64)
                return ids, np.flatnonzero(inside)
            return np.empty(0, np.int64), np.empty(0, np.int64)
        direction, a = direction[valid], a[valid]
        entry, exit = entry[valid], exit[valid]
        position = a + direction * (entry + np.minimum(1e-10, (exit - entry) / 2))[:, None]
        cell = np.clip(np.floor(position).astype(np.int64), 0, self.shape - 1)
        endpoint = np.floor(b[valid] + 1e-10).astype(np.int64)
        sign = np.sign(direction).astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            next_crossing = (cell + (sign > 0) - a) / direction
            increments = 1 / np.abs(direction)
        next_crossing[sign == 0] = np.inf
        increments[sign == 0] = np.inf
        output, identities = [], []
        visited = np.zeros(self.size, bool) if unique_only else None
        for _ in range(int(self.shape.sum()) + 3):
            if not len(cell):
                break
            include = np.ones(len(cell), bool)
            crossing = np.minimum(np.minimum(next_crossing[:, 0], next_crossing[:, 1]), next_crossing[:, 2])
            if exclude_endpoint:
                include &= ((cell[:, 0] != endpoint[:, 0]) | (cell[:, 1] != endpoint[:, 1])
                            | (cell[:, 2] != endpoint[:, 2]))
                # A truncated endpoint inside this voxel does not prove it free.
                include &= crossing <= exit + 1e-10
            ids = (cell[include] @ self.strides).astype(np.int64)
            if unique_only:
                visited[ids] = True
            else:
                output.append(ids)
                identities.append(ray_ids[include])
            alive = crossing < exit - 1e-10
            if not alive.any():
                break
            tied = next_crossing <= crossing[:, None] + 1e-10
            cell += tied * sign
            next_crossing += np.where(tied, increments, 0.)
            # Compact once per crossing, rather than once for ray length and
            # again for grid bounds. Unique-only callers do not need ray IDs.
            inside = (alive & (cell[:, 0] >= 0) & (cell[:, 0] < self.shape[0])
                      & (cell[:, 1] >= 0) & (cell[:, 1] < self.shape[1])
                      & (cell[:, 2] >= 0) & (cell[:, 2] < self.shape[2]))
            cell, next_crossing = cell[inside], next_crossing[inside]
            if not unique_only:
                ray_ids = ray_ids[inside]
            increments, sign, endpoint, exit = increments[inside], sign[inside], endpoint[inside], exit[inside]
        else:
            raise RuntimeError("Occupancy ray traversal exceeded its grid bound")
        if not exclude_endpoint:
            endpoint_ids, inside = self.indices(end)
            if unique_only:
                visited[endpoint_ids[inside]] = True
            else:
                output.append(endpoint_ids[inside])
                identities.append(np.flatnonzero(inside))
        if unique_only:
            return np.flatnonzero(visited), np.empty(0, np.int64)
        return np.concatenate(output), np.concatenate(identities)


@dataclass(frozen=True)
class OccupancyBox:
    center: np.ndarray
    rotation: np.ndarray
    extents: np.ndarray
    corners: np.ndarray
    point_count: int


class LocalOccupancyGrid(GridGeometry):
    """One hit/miss update per voxel per complete multi-camera frame.

    Occupancy wins conflicts. Only current hits are obstacles; all other active
    cells are free. Probability and source history are retained for diagnostics.
    """

    def __init__(self, origin, axes, shape, resolution_m, *, active=None,
                 hit_probability=.7, miss_probability=.4, free_probability=.35,
                 clear_frames=3, free_ttl_s=2., decay_half_life_s=1.):
        super().__init__(origin, axes, shape, resolution_m)
        if (not np.isfinite([hit_probability, miss_probability, free_probability, free_ttl_s, decay_half_life_s]).all()
                or not .5 < hit_probability < 1 or not 0 < miss_probability < .5
                or not 0 < free_probability < .5 or not 1 <= clear_frames <= 65535
                or free_ttl_s <= 0 or decay_half_life_s <= 0):
            raise ValueError("Invalid occupancy evidence settings")
        self.active = np.ones(self.size, bool) if active is None else np.asarray(active, bool).reshape(-1)
        if len(self.active) != self.size or not self.active.any():
            raise ValueError("Occupancy domain must contain active voxels")
        self.hit_log = float(np.log(hit_probability / (1 - hit_probability)))
        self.miss_log = float(np.log(miss_probability / (1 - miss_probability)))
        self.free_log = float(np.log(free_probability / (1 - free_probability)))
        # Retain legacy settings/wire metadata for existing callers and captures.
        # Free thresholds, TTL and clear_frames no longer classify motion space.
        self.free_ttl_s = float(free_ttl_s)
        self.decay_half_life_s = float(decay_half_life_s)
        self.log_odds = np.zeros(self.size, np.float32)
        self.last_observed_ns = np.zeros(self.size, np.int64)
        self.sources = np.zeros(self.size, np.uint8)  # hit=1, miss=2, completion=4, uncertain raw=8
        self.current_hits = np.empty(0, np.int64)
        self.timestamp_ns = self.frame = 0
        self.diagnostics = {}

    def point_ids(self, points):
        ids, inside = self.indices(points)
        ids = np.unique(ids[inside])
        return ids[self.active[ids]]

    def _frame_evidence(self, raw_hits, free_ids, completion_hits, uncertain_hits):
        """Deduplicate a whole multi-view frame; hits and rejected returns veto clearing."""
        hit_ids = self.point_ids(np.asarray(raw_hits).reshape(-1, 3))
        # Rejected raw endpoints veto inferred geometry and same-frame free
        # evidence, but never create blocking cells of their own.
        uncertain_ids = np.setdiff1d(self.point_ids(np.asarray(uncertain_hits).reshape(-1, 3)), hit_ids)
        completion_ids = self.point_ids(np.asarray(completion_hits).reshape(-1, 3))
        completion_ids = np.setdiff1d(completion_ids, uncertain_ids)
        occupied = np.union1d(hit_ids, completion_ids)
        blocked = np.union1d(occupied, uncertain_ids)
        free_ids = np.asarray(free_ids)
        if free_ids.ndim != 1 or not np.issubdtype(free_ids.dtype, np.integer):
            raise ValueError("Free voxel IDs must be a one-dimensional integer array")
        if np.any(free_ids < 0) or np.any(free_ids >= self.size):
            raise ValueError("Free voxel ID exceeds the grid")
        free_ids = np.unique(free_ids)
        free_ids = free_ids[self.active[free_ids]]
        conflicts = np.intersect1d(blocked, free_ids)
        free_ids = np.setdiff1d(free_ids, blocked)
        return hit_ids, completion_ids, uncertain_ids, occupied, free_ids, conflicts

    def update(self, *, timestamp_ns, raw_hits, free_ids, completion_hits=(), uncertain_hits=()):
        if (not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns <= self.timestamp_ns):
            raise ValueError("Occupancy frames require strictly increasing capture timestamps")
        hit_ids, completion_ids, uncertain_ids, occupied, free_ids, conflicts = self._frame_evidence(
            raw_hits, free_ids, completion_hits, uncertain_hits)
        blocked = np.union1d(occupied, uncertain_ids)
        seen = np.union1d(blocked, free_ids)
        if self.timestamp_ns:
            # Decay probability history for diagnostics only.
            unobserved = self.active.copy()
            unobserved[seen] = False
            elapsed = (timestamp_ns - self.timestamp_ns) / 1e9
            self.log_odds[unobserved] *= 2 ** (-elapsed / self.decay_half_life_s)
        self.log_odds[free_ids] = np.maximum(-2., self.log_odds[free_ids] + self.miss_log)
        # Measured occupancy is immediately authoritative. Verified completion is
        # weaker evidence, but still a current obstacle, never clearing evidence.
        completed_only = np.setdiff1d(completion_ids, hit_ids)
        self.log_odds[completed_only] = np.minimum(3.5, np.maximum(0., self.log_odds[completed_only]) + .405465)
        self.log_odds[hit_ids] = np.minimum(3.5, np.maximum(0., self.log_odds[hit_ids]) + self.hit_log)
        self.log_odds[uncertain_ids] = 0.
        self.last_observed_ns[seen] = timestamp_ns
        self.sources[free_ids], self.sources[completion_ids], self.sources[hit_ids] = 2, 4, 1
        self.sources[uncertain_ids] = 8
        self.current_hits = occupied
        self.timestamp_ns, self.frame = int(timestamp_ns), self.frame + 1
        self.diagnostics = {"raw_hit_voxels": len(hit_ids), "completion_hit_voxels": len(completion_ids),
                            "uncertain_raw_voxels": len(uncertain_ids),
                            "raw_free_voxels": len(free_ids), "conflicting_voxels": len(conflicts)}

    def states(self, now_ns=None):
        now = self.timestamp_ns if now_ns is None else int(now_ns)
        if now < self.timestamp_ns:
            raise ValueError("Occupancy query precedes the last captured frame")
        states = np.full(self.size, FREE, np.uint8)
        states[~self.active] = OUTSIDE
        states[self.current_hits] = OCCUPIED
        return states

    def snapshot(self):
        states = self.states()
        ages = np.minimum((self.timestamp_ns - self.last_observed_ns + 999_999) // 1_000_000, 65535).astype("<u2")
        counts = np.bincount(states, minlength=4)
        return {"schema": MAP_SCHEMA, "coordinate_frame": "left_base", "frame": self.frame,
                "capture_timestamp_ns": self.timestamp_ns, "resolution_m": self.resolution_m,
                "origin": self.origin.tolist(), "axes": self.axes.tolist(), "shape": self.shape.tolist(),
                "encoding": "zlib+base64", "state_dtype": "uint8", "age_dtype": "uint16-le",
                "state_data": _pack(states), "age_ms_data": _pack(ages), "free_ttl_s": self.free_ttl_s,
                "log_odds_dtype": "float32-le", "log_odds_data": _pack(self.log_odds.astype("<f4")),
                "free_log_threshold": self.free_log, "decay_half_life_s": self.decay_half_life_s,
                "source_data": _pack(self.sources), "source_codes": {"raw_hit": 1, "raw_free": 2, "verified_completion": 4, "uncertain_raw": 8},
                "unknown_voxels": int(counts[UNKNOWN]), "free_voxels": int(counts[FREE]),
                "occupied_voxels": int(counts[OCCUPIED]), "outside_voxels": int(counts[OUTSIDE]),
                "historical_unknown_voxels": 0}

    def boxes(self, *, tile_voxels=4, occupied_ids=None):
        """Pack solid cuboids within bounded tiles; never span an empty gap."""
        if tile_voxels < 1 or tile_voxels > 32:
            raise ValueError("Occupancy box tiles must contain 1..32 voxels per axis")
        occupied_ids = self.current_hits if occupied_ids is None else np.asarray(occupied_ids, dtype=np.int64)
        if np.any(occupied_ids < 0) or np.any(occupied_ids >= self.size):
            raise ValueError("Occupied voxel ID exceeds the grid")
        cells = np.column_stack(np.unravel_index(occupied_ids, tuple(self.shape)))
        if not len(cells):
            return []
        tiles, local = cells // tile_voxels, cells % tile_voxels
        tile_shape = (self.shape + tile_voxels - 1) // tile_voxels
        tile_ids = np.ravel_multi_index(tiles.T, tuple(tile_shape))
        order = np.argsort(tile_ids, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(tile_ids[order])) + 1, len(order)]
        lows, highs = [], []
        signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            rows = order[begin:end]
            mask = np.zeros((tile_voxels,) * 3, np.uint8)
            mask[tuple(local[rows].T)] = 1
            pattern = _tile_cuboids(tile_voxels, mask.tobytes())
            base = tiles[rows[0]] * tile_voxels
            for low, high in pattern:
                lows.append(base + low)
                highs.append(base + high)
            if len(lows) > 8192:
                raise RuntimeError("Occupancy map exceeds the 8192-box safety geometry budget")
        lows, highs = np.asarray(lows), np.asarray(highs)
        centers = self.origin + ((lows + highs) * (.5 * self.resolution_m)) @ self.axes.T
        extents = (highs - lows) * self.resolution_m
        corners = centers[:, None, :] + (signs[None, :, :] * extents[:, None, :] / 2) @ self.axes.T
        counts = np.prod(highs - lows, axis=1)
        return [OccupancyBox(center, self.axes.copy(), extent, corner, int(count))
                for center, extent, corner, count in zip(centers, extents, corners, counts)]


class ObservationGrid(GridGeometry):
    """Read-only snapshot map whose motion checks block occupied cells only."""

    def __init__(self, payload):
        if (payload.get("schema") != MAP_SCHEMA or payload.get("coordinate_frame") != "left_base"
                or payload.get("encoding") != "zlib+base64" or payload.get("state_dtype") != "uint8"
                or payload.get("age_dtype") != "uint16-le"):
            raise ValueError("Unsupported occupancy observation contract")
        super().__init__(payload["origin"], payload["axes"], payload["shape"], payload["resolution_m"])
        self.state = _unpack(payload["state_data"], np.uint8, self.size)
        self.age_ms = _unpack(payload["age_ms_data"], "<u2", self.size)
        if payload.get("log_odds_dtype") != "float32-le":
            raise ValueError("Invalid occupancy probability encoding")
        self.log_odds = _unpack(payload["log_odds_data"], "<f4", self.size)
        self.free_log, self.decay_half_life_s = float(payload["free_log_threshold"]), float(payload["decay_half_life_s"])
        self.timestamp_ns, self.free_ttl_s = payload["capture_timestamp_ns"], float(payload["free_ttl_s"])
        if (not isinstance(self.timestamp_ns, int) or self.timestamp_ns <= 0
                or not np.isfinite([self.free_ttl_s, self.free_log, self.decay_half_life_s]).all()
                or self.free_ttl_s <= 0 or self.free_log >= 0 or self.decay_half_life_s <= 0
                or not np.isfinite(self.log_odds).all() or np.any(self.state > OUTSIDE)):
            raise ValueError("Invalid occupancy observation state/timing")
        counts = np.bincount(self.state, minlength=4)
        if any(payload[name] != int(counts[state]) for name, state in
               (("unknown_voxels", UNKNOWN), ("free_voxels", FREE),
                ("occupied_voxels", OCCUPIED), ("outside_voxels", OUTSIDE))):
            raise ValueError("Occupancy state counts disagree with its grid")
        # Preserve the v1 wire format and accept archived maps, interpreting
        # their legacy unknown cells under the current occupied-only policy.
        self.state = np.where(self.state == UNKNOWN, FREE, self.state).astype(np.uint8)
        self._esdf_payload = ((payload.get("voxblox") or {}).get("esdf")
                              if payload.get("backend") == "voxblox" else None)
        self._esdf_arrays = None
        self.last_blockage = None

    def query_esdf(self, points, *, interpolate=True):
        """Read native ESDF guidance from this captured map, if usable.

        Return None for old maps, disabled ESDF, or a manually scoped map
        whose global field has not been recomputed for the selected obstacles.
        Decode once on demand; hard collision checks still use occupied cells.
        """
        field = self._esdf_payload
        if not field or not field.get("enabled") or not field.get("guidance_usable", True):
            return None
        if self._esdf_arrays is None:
            if field.get("distance_dtype") != "float32-le" or field.get("observed_dtype") != "uint8":
                raise ValueError("Invalid snapshot ESDF encoding")
            maximum = float(field["max_distance_m"])
            distances = _unpack(field["distance_m_data"], "<f4", self.size)
            observed = _unpack(field["observed_data"], np.uint8, self.size)
            valid = observed != 0
            if (not np.isfinite(maximum) or maximum <= 0 or np.any(observed > 1)
                    or not np.isfinite(distances[valid]).all()
                    or np.any(np.abs(distances[valid]) > maximum+1e-5)):
                raise ValueError("Invalid snapshot ESDF observations/distances")
            self._esdf_arrays = distances, valid, maximum
        from safety_module.distance_field import query_esdf_grid
        distances, observed, maximum = self._esdf_arrays
        return query_esdf_grid(self, distances, observed, self.state != OUTSIDE,
                               maximum, points, interpolate=interpolate)

    def segments_observed(self, start, end, *, margin_m=0., now_ns=None,
                          allow_margin_recovery=False):
        """Check occupied cells, optionally allowing outward margin recovery.

        Recovery cannot begin on or inside an occupied cell. Every affected
        point must keep or increase its distance throughout the segment.
        """
        self.last_blockage = None
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError("Invalid occupancy motion margin")
        start, end = (np.asarray(p, float).reshape(-1, 3) for p in (start, end))
        if start.shape != end.shape:
            raise ValueError("Occupancy motion points do not align")
        radius = int(np.ceil(margin_m / self.resolution_m))
        if radius > 10:
            raise ValueError("Occupancy motion margin exceeds the supported local stencil")
        # Traverse a virtual halo too: a point centre just outside the map can
        # still sweep its safety margin through occupied boundary cells.
        shape = self.shape + 2 * radius
        traversal = GridGeometry(self.origin - np.full(3, radius * self.resolution_m) @ self.axes.T,
                                 self.axes, shape, self.resolution_m,
                                 max_voxels=int(np.prod(shape)))
        ids, ray_ids = traversal.ray_cells(start, end)
        if not len(ids):
            return True
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if now < self.timestamp_ns:
            raise ValueError("Occupancy clock differs from the motion clock")
        cells = np.column_stack(np.unravel_index(ids, tuple(shape))) - radius
        local_start, local_end = (self.coordinates(p)*self.resolution_m for p in (start, end))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    queried = cells + [dx, dy, dz]
                    inside = np.all((queried >= 0) & (queried < self.shape), axis=1)
                    # Include the start voxel in occupied-cell collision checks.
                    selected = queried[inside] @ self.strides
                    state = self.state[selected]
                    blocked = state == OCCUPIED
                    if not blocked.any():
                        continue
                    # The stencil is only a broad phase. Treating an entire
                    # neighbouring cell as within margin rounds a 10 mm guard
                    # up to whole 20 mm cells and can block every RRT edge.
                    rows = np.flatnonzero(inside)[blocked]
                    centres = (queried[rows]+.5)*self.resolution_m
                    first = local_start[ray_ids[rows]]-centres
                    last = local_end[ray_ids[rows]]-centres
                    half = np.full(3, self.resolution_m/2)
                    hit = segments_within_aabb_margin(first, last, half, margin_m)
                    if allow_margin_recovery and margin_m > 0 and hit.any():
                        # Distance to a convex AABB is nondecreasing along a
                        # segment when its initial directional derivative is
                        # nonnegative. Permit escape from the margin only;
                        # actual occupied-cell contact remains forbidden.
                        offset = first-np.clip(first, -half, half)
                        distance = np.linalg.norm(offset, axis=-1)
                        outward = np.sum(offset*(last-first), axis=-1) >= 0.
                        recoverable = (distance > 1e-12) & (distance <= margin_m) & outward
                        hit &= ~recoverable
                    if hit.any():
                        index = int(np.flatnonzero(hit)[0])
                        voxel = int(selected[blocked][index])
                        self.last_blockage = {
                            "reason": "occupied",
                            "voxel_id": voxel, "point_id": int(ray_ids[rows[index]]),
                            "capture_age_s": (now-self.timestamp_ns)/1e9,
                            "observation_age_s": (now-self.timestamp_ns)/1e9+self.age_ms[voxel]/1000}
                        return False
        return True


class ObservedBoxes(list):
    def __init__(self, boxes, observation=None):
        super().__init__(boxes)
        self.observation = observation


def observation_rollout_safe(sampler, qpos, gripper, joint_chunk, boxes,
                             *, gripper_chunk=None, substeps=4, margin_m=0., now_ns=None,
                             allow_margin_recovery=False):
    """Occupied-voxel sweep; margin recovery requires an explicit opt-in."""
    observation = getattr(boxes, "observation", None)
    if observation is None:
        return True
    q, grip = np.asarray(qpos, float).copy(), np.asarray(gripper, float).copy()
    previous = np.asarray(sampler.link_points(q, grip)).reshape(-1, 3)
    for index, delta in enumerate(joint_chunk):
        origin, opening = q.copy(), grip.copy()
        target = opening if gripper_chunk is None else np.asarray(gripper_chunk[index])
        for step in range(1, substeps + 1):
            q, grip = origin + delta * step / substeps, opening + (target - opening) * step / substeps
            current = np.asarray(sampler.link_points(q, grip)).reshape(-1, 3)
            recovery = {"allow_margin_recovery": True} if allow_margin_recovery else {}
            if not observation.segments_observed(previous, current, margin_m=margin_m,
                                                  now_ns=now_ns, **recovery):
                return False
            previous = current
    return True
