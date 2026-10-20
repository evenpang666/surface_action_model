"""Coupled execution-prefix SQP with quadratic temporal regularization.

The sequential CBF rollout supplies a route seed only. Every returned plan
has passed a coupled QP over H*12 joint or Cartesian variables. Sweep validation is optional and
only runs when a caller explicitly supplies a callback.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.linalg import cho_factor, cho_solve
from scipy import sparse

from real_scripts.real_cbf_qp import (PointPairFlowConstraint, _box_signed_distance,
                                      point_jacobian_fd, select_inter_arm_constraints,
                                      select_point_flow_constraints)


def _temporal_objective(reference, q, previous, counts):
    horizon, joints = reference.shape
    size = horizon*joints
    difference = np.eye(horizon)-np.eye(horizon, k=-1)
    velocity = np.kron(difference/counts[:, None], np.eye(joints))
    offset = np.zeros((horizon, joints)); offset[0] = -q/counts[0]
    acceleration = np.kron(difference, np.eye(joints)) @ velocity
    accel_offset = difference @ offset; accel_offset[0] -= previous
    weights = np.ones((horizon, joints)); weights[-1] = 4.
    # A one-action tail has no within-chunk continuity to optimize. Preserve
    # its endpoint while still enforcing the preceding step's hard bounds.
    velocity_weight, acceleration_weight = ((.15, 1.) if horizon > 1 else (0., 0.))
    hessian = np.diag(weights.ravel())
    linear = -weights.ravel()*reference.ravel()
    hessian += velocity_weight*(velocity.T @ velocity)+acceleration_weight*(acceleration.T @ acceleration)
    linear += velocity_weight*velocity.T @ offset.ravel()+acceleration_weight*acceleration.T @ accel_offset.ravel()
    assert hessian.shape == (size, size)
    return hessian, linear, velocity, offset.ravel(), acceleration, accel_offset.ravel()


def _solve_sparse_qp(hessian, linear, current, low, high, constraints, iterations):
    """Solve the same convex QP around the warm start using sparse constraints."""
    try:
        import osqp
    except ImportError as exc:
        raise RuntimeError("Joint action optimization requires OSQP; install with python -m pip install 'osqp>=1.0,<2'") from exc
    center = current.ravel()
    matrix = sparse.vstack([*(sparse.csc_matrix(c.A) for c in constraints),
                            sparse.eye(len(center), format="csc")], format="csc")
    lower = np.concatenate([*(np.broadcast_to(c.lb, (c.A.shape[0],)) for c in constraints), low.ravel()])
    upper = np.concatenate([*(np.broadcast_to(c.ub, (c.A.shape[0],)) for c in constraints), high.ravel()])
    offset = matrix @ center
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(hessian), q=hessian @ center+linear, A=matrix,
                 l=lower-offset, u=upper-offset, verbose=False,
                 eps_abs=1e-8, eps_rel=1e-8, polishing=True,
                 eps_prim_inf=1e-9, eps_dual_inf=1e-9,
                 max_iter=max(4000, iterations*100), time_limit=2., warm_starting=True)
    solver.warm_start(x=np.zeros_like(center))
    result = solver.solve(raise_error=False)
    value = center+result.x if result.x is not None else np.full_like(center, np.nan)
    return SimpleNamespace(x=value, success=result.info.status_val in (1, 2),
                           nit=result.info.iter, message=result.info.status)


def optimize_joint_chunk(cbf, q, gripper, actions, checker, *, lower, upper, max_step,
                         previous_delta, acceleration_step, swept_safe=None, fk_sampler=None):
    q, grip, actions = (np.asarray(value, float).copy() for value in (q, gripper, actions))
    if (q.shape != (12,) or grip.shape != (2,) or actions.ndim != 2 or actions.shape[1] != 14
            or not len(actions) or not all(np.isfinite(value).all() for value in (q, grip, actions))
            or np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1))
            or not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("Invalid absolute action chunk")
    horizon, size = len(actions), len(actions)*12
    cartesian = getattr(cbf, "optimization_space", "joint") == "ee"
    low = np.broadcast_to(np.asarray(lower, float), (horizon, 12)).copy()
    high = np.broadcast_to(np.asarray(upper, float), (horizon, 12)).copy()
    previous = np.asarray(previous_delta, float)
    check_sweeps = swept_safe is not None
    if check_sweeps and not callable(swept_safe):
        raise ValueError("Sweep callback must be callable or None")
    if (previous.shape != (12,) or not np.isfinite(previous).all()
            or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high)
            or np.isnan(acceleration_step) or acceleration_step <= 0):
        raise ValueError("Invalid joint chunk bounds or acceleration")
    original_sampler, original_grip = cbf.sampler, cbf.gripper
    original_direction = cbf.detour_direction.copy()
    sampler = original_sampler if fk_sampler is None else fk_sampler
    started = time.perf_counter()
    cbf.runtime_stats["joint_chunk_calls"] = cbf.runtime_stats.get("joint_chunk_calls", 0)+1
    committed = False
    info = {"mode": "cartesian_execution_prefix" if cartesian else "joint_execution_prefix",
            "controller": "mcbf" if cbf.modulation else "cbf",
            "optimization_space": "ee" if cartesian else "joint", "gripper_variables": 0,
            "success": False, "horizon": horizon,
            "joint_variables": 0 if cartesian else size, "ee_variables": size if cartesian else 0,
            "steps": [],
            "waypoints": [], "waypoint_steps": [], "waypoint_action_indices": [],
            "triggered_action_indices": [], "modified_action_indices": [], "target_repairs": [],
            "sweep_validation_enabled": check_sweeps,
            "joint_qp_solve_calls": 0, "smoothness_weights": {"velocity": .15, "acceleration": 1.}}

    def progress(stage, **details):
        callback = getattr(cbf, "progress_callback", None)
        if callback is not None:
            callback({"stage": stage, "elapsed_s": time.perf_counter()-started, **details})

    def failure(index, reason, detail=None):
        origin = q if index == 0 else seed[index-1]
        opening = grip if index == 0 else actions[index-1, 12:]
        step = {"success": False, "triggered": True, "reason": reason,
                "qp_solve_calls": info["joint_qp_solve_calls"], "joint_optimization": not cartesian,
                "cartesian_optimization": cartesian, "optimization_space": info["optimization_space"]}
        if detail:
            step.update(detail)
        info.update(success=False, reason=reason, failed_action_index=index, failed_step=step,
                    failed_qpos=origin.tolist(), failed_gripper=opening.tolist(),
                    failed_joint_lower=low[index].tolist(), failed_joint_upper=high[index].tolist(),
                    failed_nominal_delta=np.clip(actions[index, :12]-origin, -max_step, max_step).tolist())
        return None, info

    try:
        cbf.sampler = sampler
        progress("seed")
        seeded, seed_info = cbf._seed_chunk(
            q, grip, actions, checker, lower=low, upper=high, max_step=max_step,
            previous_delta=previous, acceleration_step=acceleration_step,
            swept_safe=swept_safe, fk_sampler=sampler)
        info["seed_seconds"] = time.perf_counter()-started
        progress("seed_complete", success=seed_info["success"])
        info["target_repairs"] = seed_info.get("target_repairs", [])
        info["seed_success"] = seed_info["success"]
        if seeded is None and seed_info.get("reason", "").startswith("target_"):
            seed_info.update(mode=info["mode"], optimization_space=info["optimization_space"],
                             joint_variables=info["joint_variables"], ee_variables=info["ee_variables"],
                             gripper_variables=0, joint_qp_solve_calls=0,
                             sweep_validation_enabled=check_sweeps)
            info = seed_info
            return None, info
        if seeded is not None:
            seed = seeded[:, :12].copy()
            counts = np.maximum(1, np.bincount(seed_info["waypoint_action_indices"], minlength=horizon))
        else:
            # Joint constraints can succeed even when a greedy rollout fails.
            # Do not turn an initializer failure into a failed joint solve.
            seed = np.clip(actions[:, :12], low, high)
            counts = np.ones(horizon, dtype=int)
        seed_direction = cbf.detour_direction.copy()
        reference = seed.copy()
        info["tracking_reference"] = ("cbf_seed" if seeded is not None else "bounded_dataset_goals")
        hessian, linear, velocity, voffset, acceleration, aoffset = _temporal_objective(
            reference, q, previous, counts)
        unconstrained = cho_solve(cho_factor(hessian), -linear)
        cartesian_qp = None
        if cartesian:
            from obstacle_validation.cartesian_chunk import CartesianChunkQP
            cartesian_qp = CartesianChunkQP(sampler, actions[:, :12], q, previous, counts)
            info["tracking_reference"] = "dataset_active_tcp_poses"
            info["cartesian_objective_before"] = cartesian_qp.objective(seed)
            info["ik_method"] = "local_differential_ik"
        info["smoothness_weights"] = {"velocity": .15 if horizon > 1 else 0.,
                                      "acceleration": 1. if horizon > 1 else 0.}
        seed_velocities = (velocity @ seed.ravel()+voffset).reshape(horizon, 12)
        info["seed_acceleration_squared"] = float(np.sum((acceleration @ seed.ravel()+aoffset)**2))
        left_mask = np.arange(np.asarray(sampler.link_points(q, grip)).reshape(-1, 3).shape[0]) < cbf.left_count
        trust = min(.08, max_step*float(counts.max()))
        current = seed.copy()
        failed_edges = set()
        relaxed = False
        use_seed_guidance = True
        active_barriers = {}

        def state(index, values):
            return (q, grip) if index < 0 else (values[index], actions[index, 12:])

        def rollout(values, *, collect=False):
            waypoints, sources, failures = [], [], []
            for index in range(horizon):
                origin, opening = state(index-1, values)
                delta = (values[index]-origin)/counts[index]
                last = origin
                for substep in range(int(counts[index])):
                    endpoint = origin+(substep+1)*delta
                    if check_sweeps and not swept_safe(last, opening, delta, actions[index, 12:]):
                        failures.append(index)
                        break
                    if collect:
                        waypoints.append(np.r_[endpoint, actions[index, 12:]].tolist())
                        sources.append(index)
                    last, opening = endpoint, actions[index, 12:]
            return failures, waypoints, sources

        def point_gradient(item, jacobian):
            if isinstance(item, PointPairFlowConstraint):
                flat = jacobian.reshape(-1, 3, 12)
                return item.normal @ (flat[item.first_point_id]-flat[item.second_point_id])
            return item.normal @ jacobian[item.link_id, item.point_id]

        for iteration in range(8):
            progress("linearize", iteration=iteration+1)
            rows, thresholds = [], []
            for index in range(horizon):
                origin, opening = state(index-1, current)
                start_points = np.asarray(sampler.link_points(origin, opening))
                fractions = set(np.arange(1, 4*counts[index]+1)/(4*counts[index]))
                if index in failed_edges:
                    # Chord collisions can lie between quadrature sites. Add
                    # projected closest-approach sites for every obstacle.
                    end_points = np.asarray(sampler.link_points(current[index], actions[index, 12:]))
                    first, change = start_points.reshape(-1, 3), (end_points-start_points).reshape(-1, 3)
                    for box in checker.boxes:
                        times = np.clip(np.sum((box.center-first)*change, axis=1)
                                        / np.maximum(np.sum(change*change, axis=1), 1e-15), 0., 1.)
                        distances = _box_signed_distance((first+times[:, None]*change-box.center) @ box.axes,
                                                         box.half_sizes)
                        for point in np.argsort(distances)[:4]:
                            if distances[point] <= cbf.collision_margin_m+cbf.trigger_margin_m:
                                fractions.add(max(1e-5, float(times[point])))
                    fractions.update(np.linspace(0., 1., 17)[1:])
                start_jacobian = None
                for fraction in sorted(fractions):
                    pose = origin+(current[index]-origin)*fraction
                    # Gripper target is sent at the first control waypoint;
                    # each waypoint's swept check interpolates that command.
                    opening_fraction = min(1., fraction*counts[index])
                    sample_grip = opening+(actions[index, 12:]-opening)*opening_fraction
                    points = np.asarray(sampler.link_points(pose, sample_grip))
                    constraints = [*select_point_flow_constraints(
                        start_points, points[None], checker.boxes,
                        collision_margin_m=cbf.collision_margin_m, trigger_margin_m=cbf.trigger_margin_m,
                        rounded_margin=True, normal_at_prediction=True), *select_inter_arm_constraints(
                        start_points, points[None], left_mask, minimum_distance_m=cbf.inter_arm_margin_m,
                        trigger_margin_m=cbf.trigger_margin_m, normal_at_prediction=True)]
                    # Keep already activated identities when rebuilding from
                    # the measured pose. That pose can be outside the trigger
                    # band even though a goal would cross the same obstacle.
                    # Recompute their geometry on the new side; retaining old
                    # normals would preserve the conflicting halfspaces.
                    from obstacle_validation.cbf import _first_step_barrier
                    active = active_barriers.setdefault((index, fraction), {})
                    for item in constraints:
                        key = (("pair", item.first_point_id, item.second_point_id)
                               if isinstance(item, PointPairFlowConstraint)
                               else ("box", item.box_id, item.link_id, item.point_id))
                        active[key] = item
                    constraints = [_first_step_barrier(
                        item, points, checker.boxes, cbf.collision_margin_m, cbf.inter_arm_margin_m)
                        for item in active.values()]
                    if not constraints:
                        continue
                    jacobian = point_jacobian_fd(sampler, pose, gripper_position=sample_grip)
                    if index and start_jacobian is None:
                        start_jacobian = point_jacobian_fd(sampler, origin, gripper_position=opening)
                    for item in constraints:
                        gradient = point_gradient(item, jacobian)
                        # A sampled clearance is coupled to the preceding
                        # action's clearance, not to a frozen seed pose.
                        start_barrier = _first_step_barrier(
                            item, start_points, checker.boxes, cbf.collision_margin_m, cbf.inter_arm_margin_m)
                        h0 = start_barrier.predicted_h
                        decay = (1.-cbf.alpha*min(1., fraction*counts[index])) if h0 >= 0. else 1.
                        row = np.zeros((horizon, 12))
                        row[index] = fraction*gradient
                        if index:
                            row[index-1] = ((1-fraction)*gradient
                                           -decay*point_gradient(start_barrier, start_jacobian))
                        rhs = decay*h0+1e-5*fraction-item.predicted_h+float(row.ravel() @ current.ravel())
                        rows.append(row.ravel()); thresholds.append(rhs)
                # Retain the route seed's chosen tangent within the joint QP.
                steps = seed_info.get("steps", [])
                guidance = steps[index].get("modulation") if index < len(steps) else None
                if guidance and use_seed_guidance:
                    row = np.zeros((horizon, 12)); direction = np.asarray(guidance["joint_direction"])
                    row[index] = direction
                    if index:
                        row[index-1] = -direction
                    rows.append(row.ravel())
                    thresholds.append(guidance["minimum_joint_progress_rad"]*counts[index]
                                      +(float(direction @ q) if index == 0 else 0.))

            matrix = np.asarray(rows).reshape(-1, size)
            constraints = [LinearConstraint(velocity, -max_step-voffset, max_step-voffset)]
            # Endpoint bounds alone do not protect intermediate servo targets
            # when a recorded action needs more than one bounded control step.
            for index in np.flatnonzero(counts > 1):
                fraction = 1./counts[index]
                first_waypoint = np.zeros((12, size))
                first_waypoint[:, index*12:(index+1)*12] = fraction*np.eye(12)
                offset = np.zeros(12)
                if index:
                    first_waypoint[:, (index-1)*12:index*12] = (1-fraction)*np.eye(12)
                else:
                    offset = (1-fraction)*q
                constraints.append(LinearConstraint(first_waypoint, low[index]-offset, high[index]-offset))
            if len(matrix):
                constraints.append(LinearConstraint(matrix, thresholds, np.full(len(matrix), np.inf)))
            if np.isfinite(acceleration_step) and not relaxed:
                constraints.append(LinearConstraint(acceleration, -acceleration_step-aoffset,
                                                    acceleration_step-aoffset))
            solve_low, solve_high = np.maximum(low, current-trust), np.minimum(high, current+trust)
            if cbf.before_solve is not None:
                cbf.before_solve()
            cbf.runtime_stats["qp_solve_calls"] += 1
            info["joint_qp_solve_calls"] += 1
            solve_started = time.perf_counter()
            progress("joint_qp", iteration=iteration+1, constraints=sum(len(c.A) for c in constraints),
                     optimization_space=info["optimization_space"])
            if cartesian_qp is not None:
                result = cartesian_qp.solve(current, solve_low, solve_high, constraints, cbf.iterations)
                solver = "osqp_cartesian_qp"
                info["minimum_tcp_jacobian_singular_value"] = result.minimum_tcp_jacobian_singular_value
            elif (np.all(unconstrained >= solve_low.ravel()-1e-10)
                    and np.all(unconstrained <= solve_high.ravel()+1e-10)
                    and all(np.all(c.A @ unconstrained >= c.lb-1e-9)
                            and np.all(c.A @ unconstrained <= c.ub+1e-9) for c in constraints)):
                # This is the exact joint-QP optimum when no inequality is
                # active. Avoid iterative-solver roundoff at a feasible goal.
                result = SimpleNamespace(x=unconstrained, success=True, nit=0,
                                         message="Feasible unconstrained joint optimum")
                solver = "numpy_joint_qp"
            else:
                # Centering the cost removes a large irrelevant constant and
                # preserves precision for submillimetre clearance corrections.
                result = _solve_sparse_qp(hessian, linear, current, solve_low, solve_high,
                                          constraints, cbf.iterations)
                solver = "osqp_joint_qp"
            progress("joint_qp_complete", iteration=iteration+1, success=bool(result.success),
                     solver=solver, solver_message=str(result.message),
                     optimization_space=info["optimization_space"])
            stages = cbf.runtime_stats["stage_s"]
            stages["qp_solver"] = stages.get("qp_solver", 0.)+time.perf_counter()-solve_started
            if (cbf.modulation and not check_sweeps and not result.success
                    and str(result.message).strip().lower() == "maximum iterations reached"):
                info.update(solver=solver, sqp_iterations=iteration+1,
                            solver_iterations=int(getattr(result, "nit", 0)))
                return failure(0, "joint_chunk_iteration_limit",
                               {"solver_message": str(result.message)})
            candidate = np.asarray(result.x).reshape(horizon, 12)
            info.update(solver=solver, sqp_iterations=iteration+1,
                        solver_iterations=int(getattr(result, "nit", 0)), acceleration_relaxed=relaxed,
                        joint_linear_constraints=sum(len(c.A) for c in constraints))
            residual = max((float(np.maximum(c.lb-c.A @ candidate.ravel(), 0).max())
                            for c in constraints), default=0.)
            residual = max(residual, max((float(np.maximum(c.A @ candidate.ravel()-c.ub, 0).max())
                                          for c in constraints), default=0.))
            if (not result.success or not np.isfinite(candidate).all() or residual > 1e-6
                    or np.any(candidate < low-1e-7) or np.any(candidate > high+1e-7)):
                if np.isfinite(acceleration_step) and not relaxed:
                    relaxed = True
                    continue
                if cbf.modulation and not check_sweeps and use_seed_guidance:
                    # A greedy tangent rollout can choose incompatible sides
                    # of adjacent cells. Repeating its shifted linearization
                    # preserves that conflict forever. Keep repaired goals as
                    # the objective, but rebuild the linearization from the
                    # measured side of the obstacle. Independently evacuated
                    # goals may lie on opposite faces, so interpolating them
                    # can impose mutually incompatible clearance halfspaces.
                    # All obstacle/joint constraints stay active.
                    repaired = seed_info.get("target_repairs", [])
                    reference = np.array([repaired[i].get("target", actions[i])[:12]
                                          if i < len(repaired) else actions[i, :12]
                                          for i in range(horizon)])
                    current = np.clip(np.broadcast_to(q, reference.shape), low, high).copy()
                    hessian, linear, velocity, voffset, acceleration, aoffset = _temporal_objective(
                        reference, q, previous, counts)
                    unconstrained = cho_solve(cho_factor(hessian), -linear)
                    use_seed_guidance = False
                    seed_direction = np.zeros_like(seed_direction)
                    info["guidance_fallback"] = "relinearize_repaired_dataset_goals"
                    info["tracking_reference"] = ("dataset_active_tcp_poses" if cartesian
                                                  else "repaired_dataset_goals")
                    info["linearization_reference"] = "measured_pose_with_tracking_bounds"
                    progress("guidance_fallback", reason=str(result.message))
                    continue
                if cbf.modulation and not check_sweeps:
                    # Repaired goals can lie on opposite obstacle faces. Use
                    # the tangent as a new warm start, without reinstating the
                    # incompatible hard progress rows from the greedy seed.
                    guides = [np.asarray(step["modulation"]["joint_direction"])
                              for step in seed_info.get("steps", []) if "modulation" in step]
                    if guides:
                        direction = np.mean(guides, axis=0)
                        shifted = np.clip(current+.02*direction/max(float(np.max(np.abs(direction))), 1e-12),
                                          low, high)
                        if np.max(np.abs(shifted-current)) > 1e-7:
                            current = shifted
                            info["guidance_relinearizations"] = info.get("guidance_relinearizations", 0)+1
                            continue
                trust *= .5
                continue
            failures, waypoints, sources = rollout(candidate, collect=True)
            if failures:
                failed_edges.update(failures)
                trust *= .5
                continue
            info.update(success=True, waypoints=waypoints, waypoint_action_indices=sources,
                        max_constraint_violation=residual,
                        optimized_acceleration_squared=float(np.sum((acceleration @ candidate.ravel()+aoffset)**2)),
                        joint_objective_before=float(.5*seed.ravel() @ hessian @ seed.ravel()+linear @ seed.ravel()),
                        joint_objective_after=float(.5*candidate.ravel() @ hessian @ candidate.ravel()+linear @ candidate.ravel()))
            if cartesian_qp is not None:
                info["cartesian_objective_after"] = cartesian_qp.objective(candidate)
                # Joint-space seed costs are not the objective of this solve.
                info.pop("joint_objective_before")
                info.pop("joint_objective_after")
            for index in range(horizon):
                origin, _ = state(index-1, candidate)
                delta = (candidate[index]-origin)/counts[index]
                seed_step = seed_info.get("steps", [])
                step = dict(seed_step[index]) if index < len(seed_step) else {}
                if not use_seed_guidance:
                    step.pop("modulation", None)
                    step["detour_direction"] = seed_direction.tolist()
                    step["guidance_fallback"] = info["guidance_fallback"]
                modified = not np.allclose(candidate[index], actions[index, :12], atol=1e-7, rtol=0)
                step.update(success=True, triggered=bool(step.get("triggered", False) or modified),
                            projection_method="cartesian_chunk_qp" if cartesian else "joint_chunk_qp",
                            optimization_space=info["optimization_space"], gripper_variables=0,
                            solver=info["solver"], max_violation=residual,
                            acceleration_relaxed=relaxed,
                            linearization=("tcp_mapped_sampled_surface_points" if cartesian
                                           else "joint_sampled_surface_points"),
                            joint_optimization=not cartesian, cartesian_optimization=cartesian,
                            chunk_step=index, chunk_horizon=horizon,
                            sweep_validation_enabled=check_sweeps,
                            preview_horizon=horizon-index, preview_qp_mode=info["mode"],
                            joint_step_rad=float(np.max(np.abs(delta))),
                            nominal_step_rad=float(np.max(np.abs(seed_velocities[index]))),
                            control_substeps=int(counts[index]), qp_solve_calls=info["joint_qp_solve_calls"])
                info["steps"].append(step)
                if step["triggered"]:
                    info["triggered_action_indices"].append(index)
                if modified:
                    info["modified_action_indices"].append(index)
            info["waypoint_steps"] = [info["steps"][index] for index in sources]
            cbf.detour_direction = seed_direction
            committed = True
            return np.column_stack((candidate, actions[:, 12:])), info
        return failure(min(failed_edges, default=0), "joint_chunk_optimization_failed",
                       {"solver_message": str(result.message)})
    finally:
        cbf.sampler, cbf.gripper = original_sampler, original_grip
        if not committed:
            cbf.detour_direction = original_direction
        totals = cbf.runtime_stats["stage_s"]
        elapsed = time.perf_counter()-started
        info["planning_seconds"] = elapsed
        totals["joint_chunk_pipeline"] = totals.get("joint_chunk_pipeline", 0.)+elapsed
        progress("complete", success=bool(info["success"]), reason=info.get("reason"))
