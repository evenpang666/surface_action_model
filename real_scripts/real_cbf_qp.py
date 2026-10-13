"""CBF-QP projection for UR7e fixed-identity surface points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    center: np.ndarray  # [3]
    axes: np.ndarray  # [3,3], columns are box axes
    half_sizes: np.ndarray  # [3]

    def __post_init__(self):
        center, axes, half = map(np.asarray, (self.center, self.axes, self.half_sizes))
        if center.shape != (3,) or axes.shape != (3, 3) or half.shape != (3,):
            raise ValueError("Invalid OBB shapes")
        if not all(np.isfinite(x).all() for x in (center, axes, half)) or np.any(half <= 0):
            raise ValueError("OBB must be finite with positive half sizes")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
            raise ValueError("OBB axes must be orthonormal")


def boxes_from_snapshot(snapshot):
    from safety_module.occupancy import ObservationGrid, ObservedBoxes
    if snapshot.get("coordinate_frame") != "left_base":
        raise ValueError("Obstacle coordinates must explicitly be left_base")
    payload = snapshot.get("occupancy_map")
    if payload is None and snapshot.get("obstacle_backend") == "probabilistic_voxels":
        raise ValueError("Probabilistic obstacle snapshot is missing its occupancy map")
    observation = None if payload is None else ObservationGrid(payload)
    if observation is not None and observation.timestamp_ns != snapshot.get("host_timestamp_ns"):
        raise ValueError("Occupancy map and snapshot capture timestamps disagree")
    return ObservedBoxes([OrientedBox(np.asarray(item["center"], float), np.asarray(item["axes"], float),
                                      np.asarray(item["half_sizes"], float)) for item in snapshot["obbs"]],
                         observation)


@dataclass(frozen=True)
class PointFlowConstraint:
    normal: np.ndarray
    current_h: float
    predicted_h: float
    link_id: int
    point_id: int
    box_id: int
    time_id: int = 0


@dataclass(frozen=True)
class PointPairFlowConstraint:
    """Minimum-separation barrier between one point on each robot arm."""

    normal: np.ndarray
    current_h: float
    predicted_h: float
    first_point_id: int
    second_point_id: int
    time_id: int = 0


def point_jacobian_fd(
    sampler, qpos: np.ndarray, *, gripper_position: np.ndarray | None = None,
    point_indices: np.ndarray | None = None, epsilon_rad: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference d(surface point)/d(q), shape ``(...,3,J)``."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size not in (6, 12):
        raise ValueError("qpos must contain six or twelve UR joint angles")
    if epsilon_rad <= 0.0:
        raise ValueError("epsilon_rad must be positive")
    def points(value: np.ndarray) -> np.ndarray:
        try:
            result = sampler.link_points(value, gripper_position)
        except TypeError:
            result = sampler.link_points(value)
        result = np.asarray(result, dtype=np.float32)
        if point_indices is not None:
            result = result.reshape(-1, 3)[np.asarray(point_indices, dtype=np.int64)]
        return result
    jacobian = np.empty((*points(q).shape, q.size), dtype=np.float32)
    for joint in range(q.size):
        plus, minus = q.copy(), q.copy()
        plus[joint] += epsilon_rad
        minus[joint] -= epsilon_rad
        jacobian[..., joint] = (points(plus) - points(minus)) / (2.0 * epsilon_rad)
    return jacobian


def _box_signed_distance(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    delta = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(np.max(delta, axis=-1), 0.0)
    return outside + inside


def _box_gradient(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """A signed-distance subgradient, including deterministic face ties."""
    delta = np.abs(local) - half
    outside = np.maximum(delta, 0)
    signs = np.where(local >= 0, 1., -1.)
    if np.linalg.norm(outside) > 1e-12:
        return signs * outside / np.linalg.norm(outside)
    result = np.zeros(3)
    axis = int(np.argmax(delta))
    result[axis] = signs[axis]
    return result


def segments_intersect_box(start, end, box, margin=0.):
    """Exact slab test on point segments against an expanded OBB."""
    first = (np.asarray(start) - box.center) @ box.axes
    last = (np.asarray(end) - box.center) @ box.axes
    direction = last - first
    half = box.half_sizes + margin
    parallel = np.abs(direction) < 1e-12
    outside = np.any(parallel & (np.abs(first) > half), axis=-1)
    divisor = np.where(parallel, 1., direction)
    a, b = (-half - first) / divisor, (half - first) / divisor
    lower = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
    upper = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
    return ~outside & (np.maximum(lower, 0.) <= np.minimum(upper, 1.))


def validate_fk_rollout(sampler, qpos, gripper_position, joint_chunk, boxes, *, point_indices,
                        left_mask, collision_margin_m, inter_arm_margin_m, substeps=4, gripper_chunk=None):
    """Reject a corrected chunk if its local FK point rollout still collides.

    Segments are checked within each substep. This is a sampled point/rigid-link
    validation, not a proof about unobserved surfaces or dynamic obstacles.
    """
    q = np.asarray(qpos, dtype=float).copy()
    grip = np.asarray(gripper_position, dtype=float).copy()
    previous = sampler.link_points(q, grip).reshape(-1, 3)[point_indices]
    observation = getattr(boxes, "observation", None)
    for horizon, delta in enumerate(np.asarray(joint_chunk)):
        origin = q.copy()
        origin_grip = grip.copy()
        target_grip = origin_grip if gripper_chunk is None else np.asarray(gripper_chunk[horizon])
        for step in range(1, substeps + 1):
            q = origin + delta * step / substeps
            grip = origin_grip + (target_grip - origin_grip) * step / substeps
            current = sampler.link_points(q, grip).reshape(-1, 3)[point_indices]
            if not np.isfinite(current).all():
                return False
            if observation is not None and not observation.segments_observed(
                    previous, current, margin_m=collision_margin_m):
                return False
            if any(segments_intersect_box(previous, current, box, collision_margin_m).any() for box in boxes):
                return False
            # Closest approach between every left/right point pair during this segment.
            if np.any(left_mask) and np.any(~left_mask):
                r0 = previous[left_mask, None] - previous[None, ~left_mask]
                r1 = current[left_mask, None] - current[None, ~left_mask]
                dr = r1 - r0
                t = np.clip(-(r0 * dr).sum(-1) / np.maximum((dr * dr).sum(-1), 1e-15), 0, 1)
                if np.any(np.linalg.norm(r0 + t[..., None] * dr, axis=-1) <= inter_arm_margin_m):
                    return False
            previous = current
    return True


def select_point_flow_constraints(
    current_link_points: np.ndarray,
    predicted_link_points: np.ndarray,
    boxes: list[OrientedBox],
    *,
    collision_margin_m: float = 0.0,
    trigger_margin_m: float = 0.02,
    max_constraints: int = 32,
    rounded_margin: bool = False,
    normal_at_prediction: bool = False,
) -> list[PointFlowConstraint]:
    current_source = np.asarray(current_link_points, dtype=np.float32)
    predicted_source = np.asarray(predicted_link_points, dtype=np.float32)
    if current_source.ndim not in (2, 3) or current_source.shape[-1] != 3:
        raise ValueError(f"Expected current (K,3) or (L,P,3), got {current_source.shape}")
    if predicted_source.shape[1:] != current_source.shape:
        raise ValueError(f"Prediction {predicted_source.shape} does not follow current shape {current_source.shape}")
    if not predicted_source.size or not current_source.size or not all(np.isfinite(x).all() for x in (current_source, predicted_source)):
        raise ValueError("Surface trajectories must be nonempty and finite")
    if collision_margin_m < 0 or trigger_margin_m < 0 or max_constraints < 1:
        raise ValueError("Invalid constraint selection limits")
    link_shape = current_source.shape[:-1]
    current = current_source.reshape(-1, 3)
    predicted = predicted_source.reshape(predicted_source.shape[0], -1, 3)
    constraints: list[PointFlowConstraint] = []
    for box_id, box in enumerate(boxes):
        center = np.asarray(box.center, dtype=np.float32).reshape(3)
        axes = np.asarray(box.axes, dtype=np.float32).reshape(3, 3)
        # The replay collision checker uses Euclidean distance to the original
        # OBB (rounded corners), while the legacy selector expands each axis.
        # Keep the legacy behavior for existing callers unless requested.
        half = np.asarray(box.half_sizes, dtype=np.float32).reshape(3)
        if not rounded_margin:
            half = half + float(collision_margin_m)
        current_local = (current - center) @ axes
        predicted_local = (predicted - center) @ axes
        offset = float(collision_margin_m) if rounded_margin else 0.0
        distances = _box_signed_distance(predicted_local, half) - offset
        dangerous = distances <= float(trigger_margin_m)
        for flat_point_id in np.flatnonzero(dangerous.any(axis=0)):
            time_index = int(np.argmin(distances[:, flat_point_id]))
            local_current = current_local[flat_point_id]
            local_predicted = predicted_local[time_index, flat_point_id]
            normal = axes @ _box_gradient(local_predicted if normal_at_prediction else local_current, half)
            current_h = float(_box_signed_distance(local_current[None], half)[0]) - offset
            predicted_h = float(distances[time_index, flat_point_id])
            if len(link_shape) == 2:
                link_id, point_id = np.unravel_index(int(flat_point_id), link_shape)
            else:
                link_id, point_id = 0, int(flat_point_id)
            constraints.append(PointFlowConstraint(normal.astype(np.float32), current_h, predicted_h, int(link_id), int(point_id), box_id, int(time_index)))
    return sorted(constraints, key=lambda item: min(item.current_h, item.predicted_h))[: int(max_constraints)]


def select_inter_arm_constraints(
    current_points: np.ndarray,
    predicted_points: np.ndarray,
    left_point_mask: np.ndarray,
    *,
    minimum_distance_m: float = 0.035,
    trigger_margin_m: float = 0.02,
    max_constraints: int = 24,
    normal_at_prediction: bool = False,
) -> list[PointPairFlowConstraint]:
    """Select predicted left/right point pairs that violate arm separation."""
    from scipy.spatial import cKDTree

    current = np.asarray(current_points, dtype=np.float32).reshape(-1, 3)
    predicted = np.asarray(predicted_points, dtype=np.float32).reshape(len(predicted_points), -1, 3)
    left_mask = np.asarray(left_point_mask, dtype=bool).reshape(-1)
    if len(current) != len(left_mask) or predicted.shape[1:] != current.shape:
        raise ValueError("Inter-arm point arrays and left_point_mask do not align")
    if not all(np.isfinite(x).all() for x in (current, predicted)):
        raise ValueError("Inter-arm surface points must be finite")
    left_ids, right_ids = np.flatnonzero(left_mask), np.flatnonzero(~left_mask)
    if not len(left_ids) or not len(right_ids):
        return []
    threshold = float(minimum_distance_m + trigger_margin_m)
    candidates: dict[tuple[int, int], PointPairFlowConstraint] = {}
    for time_id, points in enumerate(predicted):
        neighbours = cKDTree(points[right_ids]).query_ball_point(points[left_ids], r=threshold)
        for local_left, local_rights in enumerate(neighbours):
            first = int(left_ids[local_left])
            for local_right in local_rights:
                second = int(right_ids[local_right])
                delta = points[first] - points[second]
                distance = float(np.linalg.norm(delta))
                current_delta = current[first] - current[second]
                current_distance = float(np.linalg.norm(current_delta))
                direction = delta if normal_at_prediction and distance > 1e-8 else current_delta
                normal = direction / max(float(np.linalg.norm(direction)), 1e-8)
                item = PointPairFlowConstraint(
                    normal.astype(np.float32), current_distance - minimum_distance_m,
                    distance - minimum_distance_m, first, second, time_id,
                )
                key = (first, second)
                if key not in candidates or item.predicted_h < candidates[key].predicted_h:
                    candidates[key] = item
    return sorted(candidates.values(), key=lambda item: min(item.current_h, item.predicted_h))[: int(max_constraints)]


def project_joint_delta_qp(
    nominal_delta: np.ndarray,
    jacobian: np.ndarray,
    constraints: list[PointFlowConstraint | PointPairFlowConstraint],
    *,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
    alpha: float = 1.0,
    iterations: int = 32,
    use_current_barrier: bool = False,
    use_predicted_linearization: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve ``min 0.5||dq-dq_nom||²`` with bounds and linear CBF constraints."""
    nominal = np.asarray(nominal_delta, dtype=np.float64).reshape(-1)
    low = np.asarray(lower_delta, dtype=np.float64).reshape(-1)
    high = np.asarray(upper_delta, dtype=np.float64).reshape(-1)
    if nominal.shape != low.shape or nominal.shape != high.shape:
        raise ValueError("nominal and joint-delta bounds must have the same shape")
    if np.any(low > high):
        raise ValueError("joint-delta lower bounds exceed upper bounds")
    if not all(np.isfinite(x).all() for x in (nominal, low, high, jacobian)) or not 0 < alpha <= 1:
        raise ValueError("QP inputs must be finite; alpha must be in (0, 1]")
    if use_current_barrier and use_predicted_linearization:
        raise ValueError("Choose only one CBF linearization reference")
    delta = np.clip(nominal, low, high)
    if not constraints:
        return delta.astype(np.float32), {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
    rows, rhs = [], []
    for item in constraints:
        if isinstance(item, PointPairFlowConstraint):
            flat_jacobian = np.asarray(jacobian, dtype=np.float64).reshape(-1, 3, nominal.size)
            relative = flat_jacobian[item.first_point_id] - flat_jacobian[item.second_point_id]
            grad = np.asarray(item.normal, dtype=np.float64) @ relative
        else:
            if np.asarray(jacobian).ndim == 3:
                point_jacobian = jacobian[item.point_id]
            else:
                point_jacobian = jacobian[item.link_id, item.point_id]
            grad = np.asarray(item.normal, dtype=np.float64) @ np.asarray(point_jacobian, dtype=np.float64)
        current_h, predicted_h = float(item.current_h), float(item.predicted_h)
        if not all(np.isfinite(x) for x in (current_h, predicted_h)) or not np.isfinite(grad).all():
            raise ValueError("Invalid barrier or gradient")
        if use_predicted_linearization:
            # Linearize the future clearance around the nominal action:
            # h(q+dq_nom) + grad_future @ (dq-dq_nom) >= (1-alpha) h(q).
            # This optimizes the action that caused the predicted intrusion.
            required = (1. - alpha) * current_h - predicted_h + float(grad @ nominal)
        else:
            # Legacy callers retain their original conservative constraint.
            barrier_h = current_h if use_current_barrier else min(current_h, predicted_h)
            required = -alpha * barrier_h
        if float(np.linalg.norm(grad)) <= 1e-10:
            if required > 0:
                return np.zeros_like(nominal, dtype=np.float32), {"triggered": True, "success": False, "constraint_count": len(constraints), "max_violation": required, "reason": "uncontrollable_barrier"}
            continue
        rows.append(grad)
        rhs.append(required)
    if not rows:
        return delta.astype(np.float32), {"triggered": True, "success": True, "constraint_count": len(constraints), "max_violation": 0.0}
    matrix = np.stack(rows)
    rhs_array = np.asarray(rhs)
    from scipy.optimize import Bounds, LinearConstraint, minimize

    result = minimize(
        fun=lambda value: 0.5 * float((value - nominal) @ (value - nominal)),
        x0=delta,
        jac=lambda value: value - nominal,
        method="SLSQP",
        bounds=Bounds(low, high),
        constraints=(LinearConstraint(matrix, rhs_array, np.full_like(rhs_array, np.inf)),),
        options={"maxiter": max(int(iterations), 1), "ftol": 1e-10, "disp": False},
    )
    delta = np.asarray(result.x, dtype=np.float64)
    violations = rhs_array - matrix @ delta
    max_violation = max(float(np.max(violations)), 0.0)
    return delta.astype(np.float32), {
        "triggered": True,
        "success": bool(result.success) and max_violation <= 1e-4,
        "constraint_count": len(rows),
        "max_violation": max_violation,
        "solver": "scipy_slsqp_qp",
        "solver_iterations": int(result.nit),
    }
