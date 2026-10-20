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
                 trigger_margin_m, alpha, iterations, modulation=False, optimization_space="joint"):
        self.sampler = sampler
        self.gripper = np.asarray(gripper, dtype=float)
        self.collision_margin_m = float(collision_margin_m)
        self.inter_arm_margin_m = float(inter_arm_margin_m)
        self.trigger_margin_m = float(trigger_margin_m)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.modulation = bool(modulation)
        if optimization_space not in ("joint", "ee"):
            raise ValueError("Optimization space must be joint or ee")
        self.optimization_space = optimization_space
        self.left_count = len(sampler.left_link_names) * sampler.points_per_link
        self.before_solve = None
        self.detour_direction = np.zeros(12)
        self.runtime_stats = {"project_calls": 0, "qp_solve_calls": 0, "stage_s": {}}

    def project_chunk(self, q, gripper, actions, checker, *, lower, upper, max_step,
                      previous_delta, acceleration_step, swept_safe=None, fk_sampler=None):
        """Jointly optimize targets; only explicitly supplied callbacks check sweeps."""
        from obstacle_validation.joint_chunk import optimize_joint_chunk
        return optimize_joint_chunk(
            self, q, gripper, actions, checker, lower=lower, upper=upper, max_step=max_step,
            previous_delta=previous_delta, acceleration_step=acceleration_step,
            swept_safe=swept_safe, fk_sampler=fk_sampler)

    def _seed_chunk(self, q, gripper, actions, checker, *, lower, upper, max_step,
                    previous_delta, acceleration_step, swept_safe, fk_sampler=None):
        """Build a sequential CBF route for the joint solver.

        A failed tail discards this initializer. The joint solver may instead
        start from bounded dataset goals; this method never authorizes sending
        a prefix without the subsequent coupled solve.
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
        joint_lower = np.broadcast_to(np.asarray(lower, float), (len(actions), 12))
        joint_upper = np.broadcast_to(np.asarray(upper, float), (len(actions), 12))
        if (not np.isfinite(joint_lower).all() or not np.isfinite(joint_upper).all()
                or np.any(joint_lower > joint_upper)):
            raise ValueError("Invalid action chunk joint bounds")
        original_sampler, original_gripper = self.sampler, self.gripper
        original_detour = self.detour_direction.copy()
        self.sampler = original_sampler if fk_sampler is None else fk_sampler
        optimized, steps, waypoints, waypoint_steps, waypoint_sources = [], [], [], [], []
        info = {"mode": "sequential_seed", "success": True, "horizon": len(actions),
                "controller": "mcbf" if self.modulation else "cbf", "target_repairs": [],
                "steps": steps, "triggered_action_indices": [], "modified_action_indices": [],
                "waypoints": waypoints, "waypoint_steps": waypoint_steps,
                "waypoint_action_indices": waypoint_sources}
        self.runtime_stats["chunk_calls"] = self.runtime_stats.get("chunk_calls", 0)+1
        completed = False
        try:
            for index, action in enumerate(actions):
                goal = action
                repair = None
                if self.modulation:
                    from obstacle_validation.mcbf import evacuate_action_target
                    goal, repair = evacuate_action_target(
                        self.sampler, action, checker.boxes, lower=joint_lower[index],
                        upper=joint_upper[index], margin=self.collision_margin_m)
                    info["target_repairs"].append(repair)
                    if not repair["success"]:
                        step = {"success": False, "triggered": True, "qp_solve_calls": 0,
                                "reason": repair["reason"], "target_repair": repair}
                        info.update(success=False, failed_action_index=index, failed_step=step,
                                    failed_qpos=q.tolist(), failed_gripper=grip.tolist(),
                                    failed_joint_lower=joint_lower[index].tolist(),
                                    failed_joint_upper=joint_upper[index].tolist(),
                                    failed_nominal_delta=np.clip(action[:12]-q, -max_step, max_step).tolist(),
                                    reason=repair["reason"])
                        return None, info
                # Preserve safe absolute goals even when reaching one requires
                # multiple speed-bounded servo steps.
                count = max(1, int(np.ceil(np.max(np.abs(goal[:12]-q))/max_step)))
                action_steps = []
                for substep in range(count):
                    self.gripper = grip.copy()
                    scene = CollisionChecker(self.sampler, grip, checker.boxes, margin=checker.margin,
                                             inter_arm_margin=checker.inter_arm_margin, resolution=checker.resolution,
                                             observation_now_ns=checker.observation_now_ns)
                    nominal = np.clip(goal[:12]-q, -max_step, max_step)
                    delta, step = self.project(
                        q, nominal, scene, lower=joint_lower[index], upper=joint_upper[index], max_step=max_step,
                        previous_delta=previous, acceleration_step=acceleration_step,
                        target_gripper=action[12:],
                        validate_sweeps=swept_safe is not None,
                        swept_safe=(lambda value: swept_safe(q, grip, value, action[12:]))
                        if swept_safe is not None else None)
                    step.update(chunk_step=index, chunk_horizon=len(actions), control_substep=substep,
                                preview_horizon=len(actions)-index, preview_qp_mode="sequential_seed")
                    if repair is not None:
                        step["target_repair"] = repair
                        step["triggered"] = step["triggered"] or repair["modified"]
                    action_steps.append(step)
                    observation = getattr(checker.boxes, "observation", None)
                    blockage = step.get("sweep_blockage") or getattr(observation, "last_blockage", None)
                    if not step["success"]:
                        info.update(success=False, failed_action_index=index, failed_step=step,
                                    failed_qpos=q.tolist(), failed_gripper=grip.tolist(),
                                    failed_joint_lower=joint_lower[index].tolist(),
                                    failed_joint_upper=joint_upper[index].tolist(),
                                    failed_nominal_delta=nominal.tolist(),
                                    reason=step.get("reason", "cbf_projection_failed"))
                        if blockage:
                            info["sweep_blockage"] = dict(blockage)
                        return None, info
                    q = q+delta
                    grip, previous = action[12:].copy(), delta.copy()
                    waypoints.append(np.concatenate((q, grip)).tolist())
                    waypoint_steps.append(step)
                    waypoint_sources.append(index)
                    # A detour replaces this dataset action's target. Do not
                    # spend the remaining nominal subdivision count chasing
                    # the same obstructed goal again inside the preview.
                    if step["triggered"] and not np.allclose(delta, nominal, atol=1e-7, rtol=0):
                        break
                summary = {**action_steps[-1], "triggered": (any(item["triggered"] for item in action_steps)
                                                           or bool(repair and repair["modified"])),
                           "qp_solve_calls": sum(item["qp_solve_calls"] for item in action_steps),
                           "control_substeps": len(action_steps)}
                steps.append(summary)
                if summary["triggered"]:
                    info["triggered_action_indices"].append(index)
                optimized.append(np.concatenate((q, grip)))
                if not np.allclose(q, action[:12], atol=1e-7, rtol=0):
                    info["modified_action_indices"].append(index)
            completed = True
            return np.asarray(optimized), info
        finally:
            self.sampler, self.gripper = original_sampler, original_gripper
            # Commit guidance across execution prefixes only for a fully
            # validated MCBF plan. Failed speculative tails must not leak it.
            if not self.modulation or not completed:
                self.detour_direction = original_detour

    def project(self, q, nominal, checker, *, lower, upper, max_step, previous_delta,
                acceleration_step, target_gripper=None, swept_safe=None, action_preview=None,
                validate_sweeps=True):
        """Project joint direction without shortening a moving nominal step.

        ``swept_safe`` may check the coupled joint/gripper motion. Existing
        fixed-gripper callers retain the joint-only edge check by default.
        Replay disables these checks with ``validate_sweeps=False``.
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
        edge_is_safe = ((lambda step: measured("sweep_validation", raw_sweep, step))
                        if validate_sweeps else lambda _step: True)
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
        if not validate_sweeps:
            # Without a nonlinear sweep fallback, seed the QP from the
            # current side of an obstacle. A penetrated target's far-face
            # normal can otherwise select an incompatible initial route.
            constraints = [replace(item, normal=_first_step_barrier(
                item, current, checker.boxes, self.collision_margin_m,
                self.inter_arm_margin_m).normal) for item in constraints]
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
        guidance = None
        if self.modulation:
            from obstacle_validation.mcbf import tangent_guidance
            guided = measured(
                "tangent_guidance", tangent_guidance, self.sampler, q, self.gripper, nominal,
                obstacles, checker.boxes, self.detour_direction, lower=hard_low, upper=hard_high,
                required_step=min(float(max_step), float(np.max(np.abs(nominal)))))
            if guided is not None:
                guidance, guidance_info = guided
                info["modulation"] = guidance_info
                self.detour_direction = guidance[0].copy()
        info["controller"] = "mcbf" if self.modulation else "cbf"
        jacobian = None
        def solve_qp(*args, **kwargs):
            self.runtime_stats["qp_solve_calls"] += 1
            info["qp_solve_calls"] += 1
            if guidance is not None:
                kwargs["guidance"] = guidance
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
        required_step = min(float(max_step), float(np.max(np.abs(nominal))))
        info["nominal_step_rad"] = required_step
        if guidance is None and np.any(self.detour_direction) and edge_is_safe(np.clip(nominal, hard_low, hard_high)):
            self.detour_direction[:] = 0.
        direction = self.detour_direction.copy()

        def complete_step(candidate):
            return (np.isfinite(candidate).all()
                    and np.all(candidate >= hard_low-1e-7)
                    and np.all(candidate <= hard_high+1e-7)
                    and np.max(np.abs(candidate)) >= required_step-1e-7
                    and (guidance is None or float(guidance[0] @ candidate) >= guidance[1]-1e-7)
                    and (not np.any(direction) or float(candidate @ direction) > 1e-9))

        if info["success"] and complete_step(safe) and edge_is_safe(safe):
            info["joint_step_rad"] = float(np.max(np.abs(safe)))
            info["detour_direction"] = self.detour_direction.tolist()
            if current_below_clearance:
                info["recovery_step"] = True
            return np.asarray(safe, dtype=float), info

        # A short projected step or a failed nonlinear sweep needs a different
        # direction, never a scaled prefix. Constrain at least one influential
        # joint to a full step, then verify the entire coupled FK sweep.
        if not constraints and self.before_solve is not None:
            self.before_solve()
        candidate, redirect = measured(
            "full_step_projection", self._redirect_full_step, q, nominal, checker,
            current=current, target_gripper=target_gripper, constraints=constraints,
            lower=hard_low, upper=hard_high, required_step=required_step,
            previous_delta=np.asarray(previous_delta), edge_is_safe=edge_is_safe,
            measured=measured, qp_buffer=qp_buffer, guidance=guidance)
        if not redirect["success"] and np.any(self.detour_direction):
            # A changed scene or a joint/tracking limit can close the chosen
            # side. Select another full-step direction instead of shrinking.
            if guidance is None:
                self.detour_direction[:] = 0.
            else:
                # Keep a hard tangent constraint while trying the other side.
                # A guide made infeasible by another obstacle must not force
                # retries of the same closed route forever.
                guidance = (-guidance[0], guidance[1])
                self.detour_direction = guidance[0].copy()
                info["modulation"] = {**info["modulation"],
                                      "joint_direction": guidance[0].tolist(),
                                      "tangent": (-np.asarray(info["modulation"]["tangent"])).tolist()}
            candidate, retry = measured(
                "full_step_projection", self._redirect_full_step, q, nominal, checker,
                current=current, target_gripper=target_gripper, constraints=constraints,
                lower=hard_low, upper=hard_high, required_step=required_step,
                previous_delta=np.asarray(previous_delta), edge_is_safe=edge_is_safe,
                measured=measured, qp_buffer=qp_buffer, guidance=guidance)
            retry["qp_solve_calls"] += redirect["qp_solve_calls"]
            retry["full_step_qp_attempts"] += redirect["full_step_qp_attempts"]
            redirect = retry
            info["detour_reselected"] = True
        if "reason" in info:
            info["initial_solver_failure"] = {key: info[key] for key in
                                              ("reason", "max_violation", "solver") if key in info}
        redirect["qp_solve_calls"] += info["qp_solve_calls"]
        info.pop("reason", None)
        info.update(redirect, acceleration_relaxed=True,
                    linearization="current_and_swept_surface_points",
                    projection_method="full_step_cbf_qp",
                    joint_step_rad=float(np.max(np.abs(candidate))))
        if not redirect["success"]:
            blockage = getattr(getattr(checker.boxes, "observation", None), "last_blockage", None)
            if blockage:
                info["sweep_blockage"] = dict(blockage)
        if redirect["success"] and not np.any(self.detour_direction):
            tangent = candidate-nominal*float(candidate @ nominal)/max(float(nominal @ nominal), 1e-15)
            if np.max(np.abs(tangent)) > 1e-7:
                self.detour_direction = tangent/np.max(np.abs(tangent))
        info["detour_direction"] = self.detour_direction.tolist()
        if current_below_clearance:
            info["recovery_step"] = True
        return candidate, info

    def _redirect_full_step(self, q, nominal, checker, *, current, target_gripper,
                            constraints, lower, upper, required_step, previous_delta,
                            edge_is_safe, measured, qp_buffer, guidance=None):
        """Change direction at the nominal step size, with bounded SQP attempts.

        Bounding one influential joint beyond +/- required_step is a union of
        convex QPs, rather than normalizing a projected vector after solving.
        Every candidate must pass the original nonlinear collision checks.
        """
        from scipy.optimize import Bounds, LinearConstraint, minimize

        jacobian = measured("jacobian", point_jacobian_fd, self.sampler, q,
                            gripper_position=self.gripper)
        flat_jacobian = jacobian.reshape(-1, 3, len(q))
        shape_flow = (np.asarray(measured("gripper_fk", self.sampler.link_points, q, target_gripper))-current
                      if not np.array_equal(target_gripper, self.gripper) else np.zeros_like(current))

        def gradient(item, jac):
            flat = jac.reshape(-1, 3, len(q))
            if isinstance(item, PointPairFlowConstraint):
                return item.normal @ (flat[item.first_point_id]-flat[item.second_point_id])
            return item.normal @ jac[item.link_id, item.point_id]

        def threshold(h):
            # A currently deficient margin must improve; safe points retain
            # their clearance constraint, with a small numerical buffer.
            return h+qp_buffer if h < 0. else (1.-self.alpha)*h+qp_buffer

        # Start with current outward normals. A normal on the far side of an
        # intruding endpoint can otherwise send the chord through the obstacle.
        rows, rhs = [], []
        if guidance is not None:
            rows.append(guidance[0])
            rhs.append(guidance[1])
        for item in constraints:
            barrier = _first_step_barrier(item, current, checker.boxes,
                                          self.collision_margin_m, self.inter_arm_margin_m)
            if isinstance(barrier, PointPairFlowConstraint):
                flow = shape_flow.reshape(-1, 3)
                shape = flow[barrier.first_point_id]-flow[barrier.second_point_id]
            else:
                shape = shape_flow[barrier.link_id, barrier.point_id]
            rows.append(gradient(barrier, jacobian))
            rhs.append(threshold(barrier.predicted_h)-barrier.predicted_h-float(barrier.normal @ shape))

        direction = self.detour_direction
        objective = nominal.copy()
        if np.any(direction) and required_step > 1e-8:
            # Keep the chosen side while the recorded direction is blocked.
            # Otherwise a full tangent can immediately be undone by the next
            # target projection, merely replacing tiny steps with oscillation.
            objective += 2.*required_step*direction
            rows.append(direction)
            rhs.append(.1*required_step)

        moving_arms = np.repeat(np.max(np.abs(nominal).reshape(2, 6), axis=1)
                                >= max(required_step*.01, 1e-9), 6)
        flat_flow = shape_flow.reshape(-1, 3)
        moving_arms[:6] |= np.any(np.abs(flat_flow[:self.left_count]) > 1e-8)
        moving_arms[6:] |= np.any(np.abs(flat_flow[self.left_count:]) > 1e-8)
        influential = np.linalg.norm(flat_jacobian, axis=(0, 1)) > 1e-8
        anchors = []
        if required_step <= 1e-8:
            anchors.append(None)
        else:
            for joint in np.flatnonzero(influential & moving_arms):
                for sign in (1., -1.):
                    value = sign*required_step
                    if lower[joint] <= value <= upper[joint]:
                        # Prefer the recorded direction, retaining the side of
                        # an already chosen tangent when costs are tied.
                        cost = (value-objective[joint])**2 - .1*value*previous_delta[joint]
                        anchors.append((int(joint), value, cost))
            anchors.sort(key=lambda item: item[2])
        left_mask = np.arange(current.reshape(-1, 3).shape[0]) < self.left_count
        attempts = 0
        for anchor in anchors:
            low, high = lower.copy(), upper.copy()
            if anchor is not None:
                if anchor[1] > 0:
                    low[anchor[0]] = max(low[anchor[0]], anchor[1])
                else:
                    high[anchor[0]] = min(high[anchor[0]], anchor[1])
            local_rows, local_rhs = list(rows), list(rhs)
            seed = np.clip(objective, low, high)
            for _ in range(3):
                def solve():
                    matrix = np.asarray(local_rows, float).reshape(-1, len(q))
                    linear_constraints = (() if not len(matrix) else
                                          (LinearConstraint(matrix, local_rhs, np.full(len(matrix), np.inf)),))
                    return minimize(lambda value: .5*float((value-objective) @ (value-objective)),
                                    seed, jac=lambda value: value-objective, method="SLSQP",
                                    bounds=Bounds(low, high), constraints=linear_constraints,
                                    options={"maxiter": self.iterations, "ftol": 1e-10, "disp": False})
                attempts += 1
                self.runtime_stats["qp_solve_calls"] += 1
                result = measured("qp_solver", solve)
                candidate = np.asarray(result.x, float)
                if (not result.success or not np.isfinite(candidate).all()
                        or np.any(candidate < low-1e-7) or np.any(candidate > high+1e-7)):
                    break
                if edge_is_safe(candidate):
                    return candidate, {"success": True, "max_violation": 0.,
                                       "full_step_qp_attempts": attempts,
                                       "qp_solve_calls": attempts,
                                       "solver": "scipy_slsqp_qp",
                                       "solver_iterations": int(getattr(result, "nit", 0))}
                # Add local linear constraints at the failed candidate's swept
                # surfaces. Re-solve at full length, including FK curvature and
                # moving-finger geometry, rather than bisecting the step.
                previous_count = len(local_rows)
                for fraction in (.25, .5, .75, 1.):
                    grip = self.gripper+(target_gripper-self.gripper)*fraction
                    points = np.asarray(measured("current_fk", self.sampler.link_points,
                                                 q+candidate*fraction, grip))
                    barriers = [*select_point_flow_constraints(
                        current, points[None], checker.boxes,
                        collision_margin_m=self.collision_margin_m,
                        trigger_margin_m=self.trigger_margin_m, rounded_margin=True,
                        normal_at_prediction=True), *select_inter_arm_constraints(
                        current, points[None], left_mask,
                        minimum_distance_m=self.inter_arm_margin_m,
                        trigger_margin_m=self.trigger_margin_m, normal_at_prediction=True)]
                    if not barriers:
                        continue
                    sample_jacobian = measured("jacobian", point_jacobian_fd, self.sampler,
                                               q+candidate*fraction, gripper_position=grip)
                    for item in barriers:
                        row = fraction*gradient(item, sample_jacobian)
                        local_rows.append(row)
                        local_rhs.append(threshold(item.current_h)-item.predicted_h+float(row @ candidate))
                if len(local_rows) == previous_count:
                    break
                seed = candidate
        return np.zeros_like(nominal), {"success": False, "reason": "no_safe_full_joint_step",
                                       "full_step_qp_attempts": attempts, "qp_solve_calls": attempts}


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
        if info["success"] and info["triggered"] and np.max(np.abs(corrected-nominal)) > 1e-7:
            self.cbf_interventions += 1
        if not info["success"]:
            self.cbf_failures += 1
            self.last_delta[:] = 0
            return np.zeros(12), "cbf_infeasible"
        if not checker.edge_safe(q, q+corrected):
            self.last_delta[:] = 0
            return np.zeros(12), "step_blocked"
        # A valid QP correction may be lateral or temporarily away from the
        # target. Accept verified motion without an along-goal progress test.
        self.last_delta = np.asarray(corrected, float).copy()
        if np.max(np.abs(corrected)) <= 1e-8:
            self.hold_ticks += 1
            self.last_delta[:] = 0
            return np.zeros(12), "cbf_hold"
        return self.last_delta.copy(), "cbf" if info["triggered"] else "nominal"
