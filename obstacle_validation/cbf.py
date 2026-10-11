"""One-step fixed-surface CBF projection for recorded dual-arm replay."""

from __future__ import annotations

import numpy as np

from real_scripts.real_cbf_qp import (point_jacobian_fd, project_joint_delta_qp,
                                      select_inter_arm_constraints, select_point_flow_constraints)


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

    def project(self, q, nominal, checker, *, lower, upper, max_step, previous_delta,
                acceleration_step, target_gripper=None, swept_safe=None):
        """Project joints against the commanded future surface, then check the sweep.

        ``swept_safe`` may check the coupled joint/gripper motion. Existing
        fixed-gripper callers retain the joint-only edge check by default.
        """
        q, nominal = np.asarray(q, dtype=float), np.asarray(nominal, dtype=float)
        target_gripper = (self.gripper if target_gripper is None
                          else np.asarray(target_gripper, dtype=float))
        if target_gripper.shape != (2,) or not np.isfinite(target_gripper).all():
            raise ValueError("Invalid target gripper for CBF projection")
        edge_is_safe = (swept_safe if swept_safe is not None
                        else lambda step: checker.edge_safe(q, q + step))
        current = np.asarray(self.sampler.link_points(q, self.gripper))
        predicted = np.asarray(self.sampler.link_points(q + nominal, target_gripper))
        if current.shape != predicted.shape or current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError("CBF surface identity or shape changed during replay")
        mask = np.arange(current.reshape(-1, 3).shape[0]) < self.left_count
        obstacles = select_point_flow_constraints(
            current, predicted[None], checker.boxes,
            collision_margin_m=self.collision_margin_m,
            trigger_margin_m=self.trigger_margin_m, rounded_margin=True,
            normal_at_prediction=True)
        pairs = select_inter_arm_constraints(
            current, predicted[None], mask,
            minimum_distance_m=self.inter_arm_margin_m,
            trigger_margin_m=self.trigger_margin_m, normal_at_prediction=True)
        constraints = [*obstacles, *pairs]
        info = {"triggered": bool(constraints), "success": True,
                "constraint_count": len(constraints), "obstacle_constraints": len(obstacles),
                "inter_arm_constraints": len(pairs), "max_violation": 0.0,
                "linearization": "future_surface_points",
                "predicted_obstacle_penetration_m": max((max(0., -item.predicted_h)
                                                         for item in obstacles), default=0.),
                "predicted_inter_arm_penetration_m": max((max(0., -item.predicted_h)
                                                          for item in pairs), default=0.)}
        current_below_clearance = not checker.state_safe(q)
        info["current_state_below_clearance"] = current_below_clearance
        hard_low = np.maximum(np.asarray(lower)-q, -max_step)
        hard_high = np.minimum(np.asarray(upper)-q, max_step)
        if np.any(hard_low > hard_high):
            info.update(success=False, reason="empty_joint_delta_bounds")
            return np.zeros(12), info
        jacobian = None
        if not constraints:
            safe = nominal
        else:
            if self.before_solve is not None:
                self.before_solve()
            jacobian = point_jacobian_fd(self.sampler, q + nominal,
                                        gripper_position=target_gripper)
            low = np.maximum(hard_low, np.asarray(previous_delta)-acceleration_step)
            high = np.minimum(hard_high, np.asarray(previous_delta)+acceleration_step)
            if np.any(low > high):
                result = {"success": False, "reason": "empty_acceleration_delta_bounds"}
                safe = np.zeros_like(nominal)
            else:
                safe, result = project_joint_delta_qp(
                    nominal, jacobian, constraints, lower_delta=low, upper_delta=high,
                    alpha=self.alpha, iterations=self.iterations,
                    use_predicted_linearization=True)
            # Collision avoidance may require braking faster than the nominal
            # acceleration profile. Never let that soft bound exclude a safe
            # stop or detour; the hard speed and joint bounds still apply.
            if not result["success"]:
                first_failure = dict(result)
                safe, result = project_joint_delta_qp(
                    nominal, jacobian, constraints, lower_delta=hard_low,
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
                candidate, result = project_joint_delta_qp(
                    nominal, jacobian, constraints, lower_delta=hard_low,
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
                info.update(success=False, reason="no_safe_joint_step_for_gripper_target")
        return np.asarray(safe, dtype=float), info
