"""Strict numerical and timing boundaries shared by robot runtime callers."""

import time

import numpy as np


class SnapshotExpired(RuntimeError):
    """A valid scene whose capture timestamp is too old for execution."""


class SnapshotUnknown(RuntimeError):
    """Previously observed obstacles have neither occupied nor free evidence."""


def validate_snapshot(snapshot, max_age_s):
    if snapshot.get("schema") not in ("dual_ur7e_obstacle_snapshot_v2", "surface_action_obstacles_v3") or snapshot.get("coordinate_frame") != "left_base":
        raise ValueError("Unsupported obstacle snapshot schema/frame")
    if snapshot.get("obstacle_backend") == "probabilistic_voxels":
        from safety_module.occupancy import MAP_SCHEMA
        payload = snapshot.get("occupancy_map")
        if (not isinstance(payload, dict) or payload.get("schema") != MAP_SCHEMA
                or payload.get("capture_timestamp_ns") != snapshot.get("host_timestamp_ns")):
            raise ValueError("Probabilistic obstacle snapshot requires its capture-aligned occupancy map")
        if payload.get("historical_unknown_voxels", 0):
            raise SnapshotUnknown("Occupancy map contains previously occupied unknown voxels")
    if (snapshot.get("scene_observation_state") == "unknown"
            or snapshot.get("unknown_obstacles")):
        raise SnapshotUnknown("Obstacle scene contains unknown space; fresh measured evidence is required")
    if str(snapshot.get("status", "")).startswith(("error", "initializing")):
        raise RuntimeError(f"Obstacle service unavailable: {snapshot['status']}")
    age = (time.monotonic_ns() - int(snapshot["host_timestamp_ns"])) / 1e9
    if not np.isfinite(age) or age < 0:
        raise RuntimeError(f"Obstacle snapshot expired or clock domain differs: {age:.3f}s")
    if age > max_age_s:
        raise SnapshotExpired(f"Obstacle snapshot expired: {age:.3f}s (limit {max_age_s:.3f}s)")
    return age


def validate_prediction(actions, surface_chunk, indices, *, action_dim, point_count, horizon=None):
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != action_dim:
        raise ValueError("Invalid action chunk shape")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
        raise ValueError("Surface indices must be a nonempty integer vector")
    if np.any(indices < 0) or np.any(indices >= point_count) or len(np.unique(indices)) != len(indices):
        raise ValueError("Surface indices are duplicated or outside the current surface")
    if surface_chunk.shape != (actions.shape[0], len(indices), 3):
        raise ValueError("Action and surface horizons/point counts differ")
    if horizon is not None and actions.shape[0] != horizon:
        raise ValueError("Output horizon differs from checkpoint")
    if not np.isfinite(actions).all() or not np.isfinite(surface_chunk).all():
        raise ValueError("Non-finite policy output; refusing execution")
