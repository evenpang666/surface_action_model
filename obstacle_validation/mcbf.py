"""Legacy local tangent guidance and virtual goal evacuation utilities.

Uses the tangent-constraint idea of arXiv:2502.14238, not its geodesic
planner or its convergence guarantees for arbitrary obstacle arrangements.
Live MCBF replay now uses adaptive_mcbf.py (arXiv:2609.30495 Sec. III-E).
These utilities remain for the older explicitly sweep-validated library API.
"""

from __future__ import annotations

import numpy as np

from real_scripts.real_cbf_qp import _box_gradient, _box_signed_distance, point_jacobian_fd


def _outward_normal(point, box, boxes):
    observation = getattr(boxes, "observation", None)
    query = getattr(observation, "query_esdf", None)
    result = query(np.asarray(point)[None]) if query is not None else None
    if result is not None and result["gradient_valid"][0]:
        gradient = result["gradient"][0]
        length = np.linalg.norm(gradient)
        if np.isfinite(gradient).all() and length > 1e-8:
            return gradient/length, "voxblox_esdf"
    return box.axes @ _box_gradient((point-box.center) @ box.axes, box.half_sizes), "occupied_box"


def _ray_exit(point, normal, boxes, margin):
    """Exit the connected union along a normal, including adjacent voxels.

    Expanded slabs conservatively include the rounded clearance boundary.
    This prevents opposing normals at internal voxel faces from oscillating.
    """
    intervals = []
    for box in boxes:
        local, direction = (point-box.center) @ box.axes, normal @ box.axes
        half = box.half_sizes+margin
        parallel = np.abs(direction) < 1e-12
        if np.any(parallel & (np.abs(local) > half)):
            continue
        moving = ~parallel
        if not np.any(moving):
            continue
        first = (-half[moving]-local[moving])/direction[moving]
        last = (half[moving]-local[moving])/direction[moving]
        entry, exit = float(np.minimum(first, last).max()), float(np.maximum(first, last).min())
        if entry <= exit and exit >= 0:
            intervals.append((entry, exit))
    distance = 0.
    for entry, exit in sorted(intervals):
        if entry > distance+1e-8:
            break
        distance = max(distance, exit)
    return distance+2e-5


def evacuate_action_target(sampler, action, boxes, *, lower, upper, margin, iterations=64):
    """Repeatedly push a *virtual* FK target out along obstacle normals.

    Recompute all distances and the Jacobian after every bounded joint update.
    Never treat an iteration limit or an unreachable normal as a safe target.
    Gripper commands remain fixed throughout the goal repair.
    """
    target = np.asarray(action, float).copy()
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    info = {"success": True, "modified": False, "iterations": 0,
            "original_target": target.tolist(), "normal_updates": []}
    normals = {}
    for iteration in range(iterations+1):
        points = np.asarray(sampler.link_points(target[:12], target[12:]), float)
        worst = None
        for box_id, box in enumerate(boxes):
            distances = _box_signed_distance((points-box.center) @ box.axes, box.half_sizes)
            identity = np.unravel_index(int(np.argmin(distances)), distances.shape)
            distance = float(distances[identity])
            if worst is None or distance < worst[0]:
                worst = distance, identity, box_id
        clearance = None if worst is None else worst[0]
        if iteration == 0:
            info["initial_clearance_m"] = clearance
        info.update(iterations=iteration, final_clearance_m=clearance, target=target.tolist())
        if worst is None or worst[0] > margin+1e-5:
            return target, info
        if iteration == iterations:
            break
        _, identity, box_id = worst
        box, point = boxes[box_id], points[identity]
        jacobian = point_jacobian_fd(sampler, target[:12], gripper_position=target[12:])[identity]
        if identity not in normals:
            preferred, source = _outward_normal(point, box, boxes)
            # At singularities or joint limits the nearest face may be
            # unreachable. Try other outward box-face normals before failing.
            candidates = [preferred, *(sign*box.axes[:, axis]
                                      for axis in range(3) for sign in (1., -1.))]
            for normal in candidates:
                gradient = normal @ jacobian
                trial = np.clip(target[:12]+.01*gradient/max(np.linalg.norm(gradient), 1e-12), lower, upper)
                if float(gradient @ (trial-target[:12])) > 1e-10:
                    normals[identity] = normal, source if normal is preferred else "occupied_box"
                    break
        selected = normals.get(identity)
        if selected is None:
            info.update(success=False, reason="target_normal_uncontrollable")
            return None, info
        normal, source = selected
        gradient = normal @ jacobian
        distance = _ray_exit(point, normal, boxes, margin)
        delta = distance*gradient/max(float(gradient @ gradient), 1e-12)
        delta *= min(1., .05/max(float(np.max(np.abs(delta))), 1e-12))
        updated = np.clip(target[:12]+delta, lower, upper)
        if float(gradient @ (updated-target[:12])) <= 1e-10:
            # A previously usable normal can hit a limit during evacuation.
            # Re-evaluate the other outward faces at this pose before giving
            # up; keep the union-exit direction while it remains usable.
            normals.pop(identity)
            continue
        info["normal_updates"].append({"box_id": box_id, "link_id": int(identity[0]),
                                        "point_id": int(identity[1]), "normal": normal.tolist(),
                                        "normal_source": source})
        target[:12] = updated
        info["modified"] = True
    info.update(success=False, reason="target_push_iteration_limit")
    return None, info


def tangent_guidance(sampler, q, grip, nominal, obstacles, boxes, previous_direction,
                     *, lower, upper, required_step):
    """Return phi^T J dq >= gamma using a controllable surface tangent.

    One dominant obstacle supplies the guidance; all safety barriers remain
    active. Keep the chosen side across successive actions until it clears.
    """
    if not obstacles or required_step <= 1e-9:
        return None
    points = np.asarray(sampler.link_points(q, grip), float)
    jacobian = point_jacobian_fd(sampler, q, gripper_position=grip)
    for item in sorted(obstacles, key=lambda item: min(item.current_h, item.predicted_h)):
        box = boxes[item.box_id]
        point = points[item.link_id, item.point_id]
        normal, source = _outward_normal(point, box, boxes)
        jac = jacobian[item.link_id, item.point_id]
        # Retreating from a deficient margin needs normal recovery, not a
        # compulsory tangent. Clear nominal motion can resume goal tracking.
        if float(normal @ jac @ nominal) >= -1e-10 and item.predicted_h > 1e-5:
            continue
        projection = np.eye(3)-np.outer(normal, normal)
        remembered = projection @ jac @ previous_direction
        nominal_tangent = projection @ jac @ nominal
        candidates = ([remembered] if np.linalg.norm(remembered) > 1e-9 else [])
        candidates += [nominal_tangent]
        # A head-on approach has no nominal tangent. Project controllable
        # joint columns, rather than choosing a possibly unreachable XYZ axis.
        candidates += [sign*projection @ jac[:, joint]
                       for joint in np.argsort(-np.linalg.norm(projection @ jac, axis=0))
                       for sign in (1., -1.)]
        for tangent in candidates:
            length = np.linalg.norm(tangent)
            if length <= 1e-9:
                continue
            tangent = tangent/length
            row = tangent @ jac
            scale = np.linalg.norm(row)
            if scale <= 1e-9:
                continue
            row = row/scale
            available = float(row @ np.where(row >= 0, upper, lower))
            minimum = .2*required_step
            if available < minimum:
                continue
            return (row, minimum), {"box_id": item.box_id,
                                    "normal_source": source,
                                    "tangent": tangent.tolist(), "joint_direction": row.tolist(),
                                    "minimum_joint_progress_rad": minimum}
    return None
