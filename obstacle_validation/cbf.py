"""Fixed-surface CBF projection for recorded dual-arm replay and chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
import time

import numpy as np

from real_scripts.real_cbf_qp import (PointPairFlowConstraint, _box_gradient, _box_signed_distance,
                                      point_jacobian_fd, project_joint_delta_qp,
                                      select_inter_arm_constraints, select_point_flow_constraints)


@dataclass(frozen=True)
class ActionChunkPreview:
    joint_positions: np.ndarray
    gripper_targets: np.ndarray
    surface_points: np.ndarray


class PreviewFKCache:
    """Bounded exact-pose FK cache for overlapping dataset previews."""

    def __init__(self, sampler, capacity=256):
        self.sampler, self.capacity = sampler, max(1, int(capacity))
        self.points = OrderedDict()
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def link_points(self, q, grip):
        key = (np.asarray(q, dtype=np.float64).tobytes(),
               np.asarray(grip, dtype=np.float64).tobytes())
        if key in self.points:
            self.hits += 1
            self.points.move_to_end(key)
            return self.points[key]
        self.misses += 1
        result = np.asarray(self.sampler.link_points(q, grip)).copy()
        result.setflags(write=False)
        self.points[key] = result
        if len(self.points) > self.capacity:
            self.points.popitem(last=False)
        return result


def preview_action_chunk(sampler, q, actions, *, max_step):
    """Roll out absolute dataset targets with the replay's joint speed bound.

    Gripper values are commanded closing targets, as in the executed action;
    each preview starts from measured joints rather than recorded observations.
    """
    q, actions = np.asarray(q, float), np.asarray(actions, float)
    if (q.shape != (12,) or actions.ndim != 2 or actions.shape[1] != 14
            or not len(actions) or not np.isfinite(q).all() or not np.isfinite(actions).all()
            or np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1))
            or not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("Invalid absolute action chunk preview")
    joints, points = [], []
    q = q.copy()
    for target in actions:
        direction = target[:12]-q
        q = np.where(np.abs(direction) <= max_step, target[:12],
                      q+np.clip(direction, -max_step, max_step))
        joints.append(q.copy())
        points.append(np.asarray(sampler.link_points(q, target[12:]), float))
    surfaces = np.asarray(points)
    if (surfaces.ndim != 4 or surfaces.shape[-1] != 3
            or not surfaces.shape[1] or not surfaces.shape[2] or not np.isfinite(surfaces).all()):
        raise ValueError("Invalid FK surface action chunk")
    return ActionChunkPreview(np.asarray(joints), actions[:, 12:].copy(), surfaces)


def _first_step_barrier(item, predicted, boxes, obstacle_margin, arm_margin):
    """Activate a previewed identity with a barrier for the executed first step.

    A future violation must not be paired with the first step's Jacobian: its
    distance and normal are re-evaluated at that first step before the QP.
    """
    if isinstance(item, PointPairFlowConstraint):
        flat = predicted.reshape(-1, 3)
        difference = flat[item.first_point_id]-flat[item.second_point_id]
        distance = float(np.linalg.norm(difference))
        normal = difference/distance if distance > 1e-8 else item.normal
        return replace(item, normal=normal, predicted_h=distance-arm_margin, time_id=0)
    box = boxes[item.box_id]
    local = (predicted[item.link_id, item.point_id]-box.center) @ box.axes
    return replace(item, normal=box.axes @ _box_gradient(local, box.half_sizes),
                   predicted_h=float(_box_signed_distance(local[None], box.half_sizes)[0])
                   - obstacle_margin, time_id=0)


class RecordedPathCBF:
    def __init__(self, sampler, gripper, *, collision_margin_m, inter_arm_margin_m,
                 trigger_margin_m, alpha, iterations):
        self.sampler = sampler
        self.gripper = np.asarray(gripper, dtype=float)
        self.collision_margin_m = float(collision_margin_m)
        self.inter_arm_margin_m = float(inter_arm_margin_m)
        self.trigger_margin_m = float(trigger_margin_m)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.left_count = len(sampler.left_link_names) * sampler.points_per_link
        self.before_solve = None
        self.runtime_stats = {"project_calls": 0, "qp_solve_calls": 0, "stage_s": {}}

    def project_chunk(self, q, gripper, actions, checker, *, lower, upper, max_step,
                      previous_delta, acceleration_step, swept_safe, fk_sampler=None):
        """Optimize every action in order, rolling out corrected predecessors.

        Each risky action has its own 12-variable CBF-QP at its own pose. This
        is sequential chunk optimization, not a coupled horizon-wide QP. A
        failed tail invalidates the whole plan; callers must not send its prefix.
        """
        from obstacle_validation.core import CollisionChecker
        q, grip, actions = (np.asarray(value, float).copy() for value in (q, gripper, actions))
        if (q.shape != (12,) or grip.shape != (2,) or actions.ndim != 2
                or actions.shape[1] != 14 or not len(actions)
                or not all(np.isfinite(value).all() for value in (q, grip, actions))
                or np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1))
                or not np.isfinite(max_step) or max_step <= 0):
            raise ValueError("Invalid absolute action chunk")
        previous = np.asarray(previous_delta, float).copy()
        original_sampler, original_gripper = self.sampler, self.gripper
        self.sampler = original_sampler if fk_sampler is None else fk_sampler
        optimized, steps, waypoints, waypoint_steps, waypoint_sources = [], [], [], [], []
        info = {"mode": "optimize_all_actions", "success": True, "horizon": len(actions),
                "steps": steps, "triggered_action_indices": [], "modified_action_indices": [],
                "waypoints": waypoints, "waypoint_steps": waypoint_steps,
                "waypoint_action_indices": waypoint_sources}
        self.runtime_stats["chunk_calls"] = self.runtime_stats.get("chunk_calls", 0)+1
        try:
            for index, action in enumerate(actions):
                # Preserve safe absolute goals even when reaching one requires
                # multiple speed-bounded servo steps.
                count = max(1, int(np.ceil(np.max(np.abs(action[:12]-q))/max_step)))
                action_steps = []
                for substep in range(count):
                    self.gripper = grip.copy()
                    scene = CollisionChecker(self.sampler, grip, checker.boxes, margin=checker.margin,
                                             inter_arm_margin=checker.inter_arm_margin, resolution=checker.resolution,
                                             observation_now_ns=checker.observation_now_ns)
                    nominal = np.clip(action[:12]-q, -max_step, max_step)
                    delta, step = self.project(
                        q, nominal, scene, lower=lower, upper=upper, max_step=max_step,
                        previous_delta=previous, acceleration_step=acceleration_step,
                        target_gripper=action[12:],
                        swept_safe=lambda value: swept_safe(q, grip, value, action[12:]))
                    step.update(chunk_step=index, chunk_horizon=len(actions), control_substep=substep,
                                preview_horizon=len(actions)-index, preview_qp_mode="optimize_all_actions")
                    action_steps.append(step)
                    observation = getattr(checker.boxes, "observation", None)
                    blockage = step.get("sweep_blockage") or getattr(observation, "last_blockage", None)
                    if not step["success"] or (blockage and blockage["reason"] == "expired_free"):
                        info.update(success=False, failed_action_index=index, failed_step=step,
                                    failed_qpos=q.tolist(), failed_gripper=grip.tolist(),
                                    failed_nominal_delta=nominal.tolist(),
                                    reason=step.get("reason", "expired_free" if blockage and
                                                    blockage["reason"] == "expired_free" else "cbf_projection_failed"))
                        if blockage:
                            info["sweep_blockage"] = dict(blockage)
                        return None, info
                    q = q+delta
                    grip, previous = action[12:].copy(), delta.copy()
                    waypoints.append(np.concatenate((q, grip)).tolist())
                    waypoint_steps.append(step)
                    waypoint_sources.append(index)
                summary = {**action_steps[-1], "triggered": any(item["triggered"] for item in action_steps),
                           "qp_solve_calls": sum(item["qp_solve_calls"] for item in action_steps),
                           "control_substeps": count}
                steps.append(summary)
                if summary["triggered"]:
                    info["triggered_action_indices"].append(index)
                optimized.append(np.concatenate((q, grip)))
                if not np.allclose(q, action[:12], atol=1e-7, rtol=0):
                    info["modified_action_indices"].append(index)
            return np.asarray(optimized), info
        finally:
            self.sampler, self.gripper = original_sampler, original_gripper

    def project(self, q, nominal, checker, *, lower, upper, max_step, previous_delta,
                acceleration_step, target_gripper=None, swept_safe=None, action_preview=None):
        """Project joints against the commanded future surface, then check the sweep.

        ``swept_safe`` may check the coupled joint/gripper motion. Existing
        fixed-gripper callers retain the joint-only edge check by default.
        ``action_preview`` activates identities across the full dataset chunk;
        the QP optimizes only the current joint step, re-evaluating each active
        barrier at its actual first-step surface and Jacobian.
        """
        self.runtime_stats["project_calls"] += 1
        timing = {}

        def record(stage, elapsed):
            timing[stage] = timing.get(stage, 0.)+elapsed
            totals = self.runtime_stats["stage_s"]
            totals[stage] = totals.get(stage, 0.)+elapsed

        def measured(stage, fn, *args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(stage, time.perf_counter()-started)

        q, nominal = np.asarray(q, dtype=float), np.asarray(nominal, dtype=float)
        target_gripper = (self.gripper if target_gripper is None
                          else np.asarray(target_gripper, dtype=float))
        if target_gripper.shape != (2,) or not np.isfinite(target_gripper).all():
            raise ValueError("Invalid target gripper for CBF projection")
        raw_sweep = (swept_safe if swept_safe is not None
                     else lambda step: checker.edge_safe(q, q + step))
        edge_is_safe = lambda step: measured("sweep_validation", raw_sweep, step)
        current = np.asarray(measured("current_fk", self.sampler.link_points, q, self.gripper))
        predicted = (np.asarray(measured("current_fk", self.sampler.link_points, q + nominal, target_gripper))
                     if action_preview is None else action_preview.surface_points[0])
        if current.shape != predicted.shape or current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError("CBF surface identity or shape changed during replay")
        if action_preview is not None and (
                action_preview.surface_points.shape[1:] != current.shape
                or not np.allclose(action_preview.joint_positions[0], q+nominal, atol=1e-10, rtol=0)
                or not np.allclose(action_preview.gripper_targets[0], target_gripper, atol=1e-10, rtol=0)):
            raise ValueError("Action chunk preview does not match the executed first step")
        mask = np.arange(current.reshape(-1, 3).shape[0]) < self.left_count
        selection_started = time.perf_counter()
        obstacles = select_point_flow_constraints(
            current, predicted[None], checker.boxes,
            collision_margin_m=self.collision_margin_m,
            trigger_margin_m=self.trigger_margin_m, rounded_margin=True,
            normal_at_prediction=True)
        pairs = select_inter_arm_constraints(
            current, predicted[None], mask,
            minimum_distance_m=self.inter_arm_margin_m,
            trigger_margin_m=self.trigger_margin_m, normal_at_prediction=True)
        preview_info = {}
        if action_preview is not None:
            forecast = action_preview.surface_points
            future_obstacles = select_point_flow_constraints(
                current, forecast, checker.boxes, collision_margin_m=self.collision_margin_m,
                trigger_margin_m=self.trigger_margin_m, rounded_margin=True, normal_at_prediction=True)
            future_pairs = select_inter_arm_constraints(
                current, forecast, mask, minimum_distance_m=self.inter_arm_margin_m,
                trigger_margin_m=self.trigger_margin_m, normal_at_prediction=True)
            # Keep immediate hazards even when the preview's constraint cap is
            # filled by more severe hazards in later actions.
            active = {(item.box_id, item.link_id, item.point_id): item for item in obstacles}
            for item in future_obstacles:
                active.setdefault((item.box_id, item.link_id, item.point_id), _first_step_barrier(
                    item, predicted, checker.boxes, self.collision_margin_m, self.inter_arm_margin_m))
            obstacles = list(active.values())
            active_pairs = {(item.first_point_id, item.second_point_id): item for item in pairs}
            for item in future_pairs:
                active_pairs.setdefault((item.first_point_id, item.second_point_id), _first_step_barrier(
                    item, predicted, checker.boxes, self.collision_margin_m, self.inter_arm_margin_m))
            pairs = list(active_pairs.values())
            minimum = np.full(len(forecast), np.inf)
            for box in checker.boxes:
                distances = _box_signed_distance((forecast-box.center) @ box.axes, box.half_sizes)
                minimum = np.minimum(minimum, distances.reshape(len(forecast), -1).min(axis=1))
            risk = np.flatnonzero(minimum <= self.collision_margin_m+self.trigger_margin_m)
            intrusion = np.flatnonzero(minimum <= 0.)
            preview_info = {
                "preview_horizon": len(forecast), "preview_qp_mode": "trigger_current_step",
                "preview_obstacle_constraints": len(future_obstacles),
                "preview_inter_arm_constraints": len(future_pairs),
                "preview_min_obstacle_clearance_m": (float(minimum.min())
                                                     if np.isfinite(minimum.min()) else None),
                "preview_first_obstacle_risk_step": int(risk[0]) if len(risk) else None,
                "preview_first_obstacle_intrusion_step": int(intrusion[0]) if len(intrusion) else None}
        constraints = [*obstacles, *pairs]
        record("constraint_selection", time.perf_counter()-selection_started)
        info = {"triggered": bool(constraints), "success": True,
                "timing_s": timing, "qp_solve_calls": 0,
                "constraint_count": len(constraints), "obstacle_constraints": len(obstacles),
                "inter_arm_constraints": len(pairs), "max_violation": 0.0,
                "linearization": "future_surface_points",
                "predicted_obstacle_penetration_m": max((max(0., -item.predicted_h)
                                                         for item in obstacles), default=0.),
                "predicted_inter_arm_penetration_m": max((max(0., -item.predicted_h)
                                                          for item in pairs), default=0.)}
        info.update(preview_info)
        current_obstacle, current_arm = measured("current_clearance", checker.clearance, q)
        current_below_clearance = (current_obstacle <= checker.margin
                                   or current_arm <= checker.inter_arm_margin)
        info["current_obstacle_clearance_m"] = (float(current_obstacle)
                                                 if np.isfinite(current_obstacle) else None)
        info["current_inter_arm_clearance_m"] = float(current_arm)
        info["current_state_below_clearance"] = current_below_clearance
        hard_low = np.maximum(np.asarray(lower)-q, -max_step)
        hard_high = np.minimum(np.asarray(upper)-q, max_step)
        if np.any(hard_low > hard_high):
            info.update(success=False, reason="empty_joint_delta_bounds")
            return np.zeros(12), info
        jacobian = None
        def solve_qp(*args, **kwargs):
            self.runtime_stats["qp_solve_calls"] += 1
            info["qp_solve_calls"] += 1
            return measured("qp_solver", project_joint_delta_qp, *args, **kwargs)
        # Keep the QP a few micrometres inside the requested safe set. Finite
        # difference/solver roundoff must not place a tangential step exactly
        # on the strict nonlinear sweep boundary and cause a false deadlock.
        qp_buffer = 1e-5
        qp_constraints = [replace(item, current_h=item.current_h-qp_buffer,
                                  predicted_h=item.predicted_h-qp_buffer) for item in constraints]
        info["qp_clearance_buffer_m"] = qp_buffer if constraints else 0.
        if not constraints:
            safe = nominal
        else:
            if self.before_solve is not None:
                self.before_solve()
            jacobian = measured("jacobian", point_jacobian_fd, self.sampler, q + nominal,
                                        gripper_position=target_gripper)
            low = np.maximum(hard_low, np.asarray(previous_delta)-acceleration_step)
            high = np.minimum(hard_high, np.asarray(previous_delta)+acceleration_step)
            if np.any(low > high):
                result = {"success": False, "reason": "empty_acceleration_delta_bounds"}
                safe = np.zeros_like(nominal)
            else:
                safe, result = solve_qp(
                    nominal, jacobian, qp_constraints, lower_delta=low, upper_delta=high,
                    alpha=self.alpha, iterations=self.iterations,
                    use_predicted_linearization=True)
            # Collision avoidance may require braking faster than the nominal
            # acceleration profile. Never let that soft bound exclude a safe
            # stop or detour; the hard speed and joint bounds still apply.
            if not result["success"]:
                first_failure = dict(result)
                safe, result = solve_qp(
                    nominal, jacobian, qp_constraints, lower_delta=hard_low,
                    upper_delta=hard_high, alpha=self.alpha,
                    iterations=self.iterations, use_predicted_linearization=True)
                info["acceleration_relaxed"] = True
                info["initial_solver_failure"] = first_failure
            info.update(result)
        if not np.isfinite(safe).all() or np.any(safe < hard_low-1e-6) or np.any(safe > hard_high+1e-6):
            info.update(success=False, reason="corrected_joint_delta_bounds")
        if current_below_clearance:
            if info["success"] and edge_is_safe(safe):
                info["recovery_step"] = True
                return np.asarray(safe, dtype=float), info
            # Reaching the full margin in one tick can be impossible under
            # joint-speed bounds. Relax only the recovery rate, then validate
            # actual FK progress before accepting another joint action.
            attempts = []
            for recovery_alpha in (.5, .25, .1):
                if jacobian is None or recovery_alpha >= self.alpha:
                    continue
                candidate, result = solve_qp(
                    nominal, jacobian, qp_constraints, lower_delta=hard_low,
                    upper_delta=hard_high, alpha=recovery_alpha,
                    iterations=self.iterations, use_predicted_linearization=True)
                candidate_ok = (result["success"] and np.isfinite(candidate).all()
                                and np.all(candidate >= hard_low-1e-6)
                                and np.all(candidate <= hard_high+1e-6))
                if candidate_ok:
                    candidate_ok = edge_is_safe(candidate)
                attempts.append({"alpha": recovery_alpha, "qp_success": bool(result["success"]),
                                 "verified": bool(candidate_ok)})
                if candidate_ok:
                    info.pop("reason", None)
                    info.update(result, recovery_alpha=recovery_alpha,
                                recovery_step=True, recovery_attempts=attempts)
                    return np.asarray(candidate, dtype=float), info
            nominal_prefix = np.clip(nominal, hard_low, hard_high)
            if edge_is_safe(nominal_prefix):
                info.pop("reason", None)
                info.update(success=True, projection_method="verified_nominal_recovery",
                            recovery_step=True, recovery_attempts=attempts)
                return np.asarray(nominal_prefix, dtype=float), info
            info.update(success=False, reason="recovery_future_or_sweep_unsafe",
                        recovery_attempts=attempts)
            return np.asarray(safe, dtype=float), info
        if constraints and not info["success"]:
            # A future-point linearization can be infeasible after the nominal
            # endpoint crosses an OBB face. Recover the longest verified prefix
            # of the original action, without treating the QP failure as safe.
            info["qp_failure"] = {key: value for key, value in info.items()
                                  if key in ("reason", "max_violation", "solver", "solver_iterations")}
            safe = np.clip(nominal, hard_low, hard_high)
            info.update(success=True, max_violation=0., projection_method="verified_nominal_prefix")
        if info["success"] and not edge_is_safe(safe):
            # The CBF is a local linearization; validate the full nonlinear
            # joint edge and retain only a geometrically safe action prefix.
            lower_fraction, upper_fraction = 0., 1.
            for _ in range(8):
                middle = (lower_fraction + upper_fraction) / 2.
                if edge_is_safe(middle * safe):
                    lower_fraction = middle
                else:
                    upper_fraction = middle
            safe = lower_fraction * safe
            info["fk_backtrack_fraction"] = lower_fraction
            if lower_fraction == 0.:
                observation = getattr(checker.boxes, "observation", None)
                blockage = getattr(observation, "last_blockage", None)
                if blockage:
                    info["sweep_blockage"] = dict(blockage)
                if constraints and not (blockage and blockage["reason"] == "expired_free"):
                    # A future normal can admit an endpoint whose chord first
                    # cuts inside a rounded corner. Re-solve a standard CBF at
                    # the current surface/Jacobian, then verify the full sweep.
                    current_jacobian = measured("jacobian", point_jacobian_fd, self.sampler, q,
                                                gripper_position=self.gripper)
                    gripper_flow = (np.asarray(measured("gripper_fk", self.sampler.link_points, q, target_gripper))
                                    -current if not np.array_equal(target_gripper, self.gripper)
                                    else np.zeros_like(current))
                    current_constraints = []
                    for item in constraints:
                        barrier = _first_step_barrier(item, current, checker.boxes,
                                                      self.collision_margin_m, self.inter_arm_margin_m)
                        h = barrier.predicted_h-qp_buffer
                        if isinstance(barrier, PointPairFlowConstraint):
                            jac = current_jacobian.reshape(-1, 3, len(q))
                            flow = gripper_flow.reshape(-1, 3)
                            joint_flow = (jac[barrier.first_point_id]-jac[barrier.second_point_id]) @ nominal
                            shape_flow = flow[barrier.first_point_id]-flow[barrier.second_point_id]
                        else:
                            joint_flow = current_jacobian[barrier.link_id, barrier.point_id] @ nominal
                            shape_flow = gripper_flow[barrier.link_id, barrier.point_id]
                        # Use the current outward normal, including the known
                        # gripper motion which joints must compensate for.
                        # Without this term the fallback may hold joints while
                        # the commanded fingers move into the obstacle.
                        future_h = h+float(barrier.normal @ (joint_flow+shape_flow))
                        current_constraints.append(replace(barrier, current_h=h, predicted_h=future_h))
                    candidate, result = solve_qp(
                        nominal, current_jacobian, current_constraints, lower_delta=hard_low,
                        upper_delta=hard_high, alpha=self.alpha, iterations=self.iterations,
                        use_predicted_linearization=True)
                    info["current_barrier_solver"] = dict(result)
                    if (result["success"] and np.isfinite(candidate).all()
                            and np.all(candidate >= hard_low-1e-6) and np.all(candidate <= hard_high+1e-6)):
                        fraction = 1.
                        if not edge_is_safe(candidate):
                            low, high = 0., 1.
                            for _ in range(8):
                                middle = (low+high)/2
                                if edge_is_safe(candidate*middle):
                                    low = middle
                                else:
                                    high = middle
                            fraction = low
                        if fraction > 0. and np.max(np.abs(candidate*fraction)) > 1e-8:
                            info.pop("reason", None)
                            info.update(result, projection_method="current_barrier_qp",
                                        linearization="current_surface_points", acceleration_relaxed=True,
                                        fk_backtrack_fraction=fraction)
                            return np.asarray(candidate*fraction, float), info
                blockage = getattr(observation, "last_blockage", None)
                if blockage:
                    info["sweep_blockage"] = dict(blockage)
                if edge_is_safe(np.zeros_like(safe)):
                    info.update(success=True, projection_method="verified_safe_hold")
                else:
                    info.update(success=False, reason="no_safe_joint_step_for_gripper_target")
        return np.asarray(safe, dtype=float), info


class CBFGoalFollower:
    """Restore one absolute joint target using bounded CBF-QP steps only."""

    def __init__(self, goal, *, lower, upper, dt, speed, acceleration, tolerance, projector):
        self.goal = np.asarray(goal, float).copy()
        self.lower, self.upper = np.asarray(lower, float).copy(), np.asarray(upper, float).copy()
        if (any(x.shape != (12,) or not np.isfinite(x).all()
                for x in (self.goal, self.lower, self.upper))
                or np.any(self.lower >= self.upper)
                or np.any(self.goal < self.lower) or np.any(self.goal > self.upper)
                or not all(np.isfinite(x) and x > 0 for x in (dt, speed, acceleration, tolerance))):
            raise ValueError("Invalid CBF restoration target or motion limits")
        self.max_step, self.acceleration_step = speed*dt, acceleration*dt**2
        self.tolerance, self.projector = tolerance, projector
        self.last_delta = np.zeros(12)
        self.last_cbf_info = None
        self.last_step_duration_s = None
        self.cbf_interventions = self.cbf_failures = self.hold_ticks = 0

    def next_delta(self, q, checker):
        q = np.asarray(q, float)
        if (q.shape != (12,) or not np.isfinite(q).all()
                or np.any(q < self.lower) or np.any(q > self.upper) or not checker.state_safe(q)):
            return np.zeros(12), "unsafe_current_state"
        direction = self.goal-q
        distance = float(np.max(np.abs(direction)))
        if distance <= self.tolerance:
            self.last_delta[:] = 0
            return np.zeros(12), "completed"
        ramp = np.max(np.abs(self.last_delta))+self.acceleration_step
        braking = max(self.acceleration_step, np.sqrt(2*self.acceleration_step*distance))
        nominal = direction*(min(distance, self.max_step, ramp, braking)/distance)
        started = time.monotonic()
        corrected, info = self.projector.project(
            q, nominal, checker, lower=self.lower, upper=self.upper, max_step=self.max_step,
            previous_delta=self.last_delta, acceleration_step=self.acceleration_step)
        self.last_step_duration_s = time.monotonic()-started
        self.last_cbf_info = info
        blockage = checker.last_observation_blockage
        if (np.max(np.abs(corrected)) <= 1e-8 and blockage
                and blockage["reason"] == "expired_free"):
            self.last_delta[:] = 0
            return np.zeros(12), "scene_expired"
        if info["success"] and info["triggered"] and np.max(np.abs(corrected-nominal)) > 1e-7:
            self.cbf_interventions += 1
        if not info["success"]:
            self.cbf_failures += 1
            self.last_delta[:] = 0
            blockage = checker.last_observation_blockage
            return np.zeros(12), ("scene_expired" if blockage and blockage["reason"] == "expired_free"
                                  else "cbf_infeasible")
        if not checker.edge_safe(q, q+corrected):
            self.last_delta[:] = 0
            blockage = checker.last_observation_blockage
            return np.zeros(12), ("scene_expired" if blockage and blockage["reason"] == "expired_free"
                                  else "step_blocked")
        # A valid QP correction may be lateral or temporarily away from the
        # target. Accept verified motion without an along-goal progress test.
        self.last_delta = np.asarray(corrected, float).copy()
        if np.max(np.abs(corrected)) <= 1e-8:
            self.hold_ticks += 1
            self.last_delta[:] = 0
            return np.zeros(12), "cbf_hold"
        return self.last_delta.copy(), "cbf" if info["triggered"] else "nominal"
