"""Whole-body ellipsoid MCBF for goal-seeking replay, in joint coordinates.

Feasible nominal steps pass through unchanged. Only distance violations
activate the QP; its objective remains the original action. Per-arm forward
progress bounds are never relaxed. Both arms remain available when distance
constraints require motion. VLSA Eq. (9) uses relative ellipsoid centres,
shape/orientation, and a controlled separating-plane direction per pair.
Only left/environment, right/environment and left/right pairs are used.
No same-arm avoidance constraints or guidance are constructed.
Gripper targets are
exogenous, never optimization variables.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from obstacle_validation.ellipsoid_geometry import (
    RobotEllipsoids, fit_obstacles, separation, separation_gradient, vlsa_barrier,
)


@dataclass(frozen=True)
class EllipsoidMCBFConfig:
    max_control_step_rad: float = .04
    alpha_per_s: float = 20.
    smoothness_weight: float = .08
    guidance_weight: float = 8.
    goal_progress_weight: float = 20.
    collision_slack_weight: float = 20000.
    self_collision_margin_m: float = .002  # Legacy config compatibility only; unused.
    activation_distance_m: float = .08
    guidance_distance_m: float = .035
    narrow_margin_scale: float = .25
    dynamic_margin_scale: float = 1.5
    closing_speed_m_s: float = .03
    geodesic_candidates: int = 6
    geodesic_horizon: int = 6
    guidance_refresh_s: float = .5
    guidance_budget_s: float = .04
    stagnation_steps: int = 80
    max_relaxation_level: int = 6
    obstacle_max_parts: int = 8
    obstacle_max_span_m: float = .20
    goal_change_tolerance_rad: float = .001
    virtual_direction_step_rad: float = .25
    virtual_direction_weight: float = .01

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Ellipsoid MCBF {name} must be positive")
        for name in ("geodesic_candidates", "geodesic_horizon", "obstacle_max_parts",
                     "stagnation_steps", "max_relaxation_level"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")


class EllipsoidMCBF:
    def __init__(self, sampler, *, margin, alpha=1., inter_arm_margin, period,
                 optimization_space="joint", config=None):
        if optimization_space not in ("joint", "ee"):
            raise ValueError("Unknown optimization space")
        if (not np.isfinite([margin, inter_arm_margin, period]).all()
                or margin < 0 or inter_arm_margin < 0 or period <= 0):
            raise ValueError("Invalid clearance or control period")
        self.robot = RobotEllipsoids(sampler)
        self.sampler, self.margin, self.period = sampler, margin, period
        self.inter_arm_margin = inter_arm_margin
        self.optimization_space = optimization_space
        self.config = config or EllipsoidMCBFConfig()
        self.boxes = None
        self.geometry_signature = None
        self.obstacles, self.velocities = [], np.empty((0, 3))
        self.capture_time = None
        self.normals = {}
        self.direction = np.zeros(12)
        self.guide_key = None
        self.guide_at = -np.inf
        self.last_goal = None
        self.goal_reference = None
        self.goal_start = None
        self.best_goal_error = np.inf
        self.stagnant_steps = 0
        self.penalty_scale = 1.
        self.relaxation_level = 0
        self.metric = np.eye(12)
        self.virtual_reference = np.empty(0)
        self.runtime_stats = {"qp_solve_calls": 0, "stage_counts": {}, "stage_s": {}}

    def update_scene(self, boxes):
        if boxes is self.boxes:
            return
        stamp = (getattr(boxes, "source_snapshot", None) or {}).get("host_timestamp_ns")
        old, old_velocity, old_stamp = self.obstacles, self.velocities, self.capture_time
        signature = np.array([np.r_[b.center, b.axes.ravel(), b.half_sizes] for b in boxes]).tobytes()
        if signature != self.geometry_signature:
            self.obstacles = fit_obstacles(boxes, max_parts=self.config.obstacle_max_parts,
                                           max_span=self.config.obstacle_max_span_m)
            self.geometry_signature = signature
            # Fitted cluster identities may change when TSDF topology changes.
            self.normals = {k: v for k, v in self.normals.items() if k[0] == "robot"}
        self.velocities = np.zeros((len(self.obstacles), 3))
        if old and self.obstacles and stamp is not None and old_stamp is not None:
            dt = (stamp-old_stamp)/1e9
            if 0 < dt <= 2.:
                old_centers = np.array([b.center for b in old])
                centers = np.array([b.center for b in self.obstacles])
                distances, matches = cKDTree(old_centers).query(centers)
                reverse = cKDTree(centers).query(old_centers)[1]
                for i, (distance, match) in enumerate(zip(distances, matches)):
                    if reverse[match] == i and distance < .3:
                        # No fixed displacement dead band: slow per-frame
                        # motion must not be classified as a stationary object.
                        velocity = (centers[i]-old_centers[match])/dt
                        self.velocities[i] = .65*velocity+.35*old_velocity[match]
        self.boxes, self.capture_time = boxes, stamp

    def _barrier(self, a, b, key):
        normal = self.normals.get(key)
        if normal is None:
            normal = separation(a, b)[1]
        value = vlsa_barrier(a, b, normal)
        # Recover a valid plane after geometry/feedback jumps. This only
        # changes the virtual state, never moves a real goal or gripper.
        if value.value < 0:
            best = separation(a, b, normal)
            if best[0] > value.value:
                value = vlsa_barrier(a, b, best[1])
        self.normals[key] = value.normal
        return value

    def _pairs(self, bodies, dc, ds, step, grip_bodies):
        rows = []
        center_motion = np.sum(np.linalg.norm(dc, axis=1), axis=1)*step
        # Account for body rotation as well as translation in broad phase.
        shape_motion = np.sum(np.linalg.norm(ds, axis=(1, 2)), axis=1)*step
        radii = np.array([b.radius for b in bodies])
        min_axes = np.array([np.sqrt(np.linalg.eigvalsh(b.shape)[0]) for b in bodies])
        travel = center_motion+shape_motion/np.maximum(2*min_axes, 1e-6)
        travel += np.array([np.linalg.norm(g.center-b.center)+np.linalg.norm(g.shape-b.shape)/(2*m)
                            for b, g, m in zip(bodies, grip_bodies, min_axes)])
        for i, a in enumerate(bodies):
            if np.linalg.norm(dc[i])+np.linalg.norm(ds[i]) < 1e-10:
                continue
            for j, b in enumerate(self.obstacles):
                velocity = self.velocities[j]
                threshold = self.config.activation_distance_m+travel[i]+np.linalg.norm(velocity)*self.period
                if np.linalg.norm(a.center-b.center)-radii[i]-b.radius > threshold:
                    continue
                key = ("environment", a.key, b.key)
                geometry = self._barrier(a, b, key)
                d, n = geometry.value, geometry.normal
                if d > threshold:
                    continue
                grad = separation_gradient(a, n, dc[i], ds[i])
                drift = vlsa_barrier(grip_bodies[i], b, n).value-d
                rows.append({"key": key, "i": i, "j": j, "distance": d, "normal": n,
                             "pair_type": "left_obstacle" if a.arm == 0 else "right_obstacle",
                             "geometry": geometry,
                             "gradient": grad, "drift": drift, "velocity": float(n @ velocity),
                             "margin": self.margin, "environment": True})
        for i, a in enumerate(bodies):
            for j in range(i+1, len(bodies)):
                b = bodies[j]
                # Apply the arm scope BEFORE plane initialization, Jacobians,
                # virtual controls or guidance can see an overlapping envelope.
                if not self.robot.inter_arm_collision_pair(a, b):
                    continue
                if np.linalg.norm(dc[i]-dc[j])+np.linalg.norm(ds[i]-ds[j]) < 1e-10:
                    continue
                margin = self.inter_arm_margin
                threshold = margin+self.config.activation_distance_m+travel[i]+travel[j]
                if np.linalg.norm(a.center-b.center)-radii[i]-radii[j] > threshold:
                    continue
                key = ("robot", a.key, b.key)
                geometry = self._barrier(a, b, key)
                d, n = geometry.value, geometry.normal
                if d > threshold:
                    continue
                grad = (separation_gradient(a, n, dc[i], ds[i])
                        +separation_gradient(b, -n, dc[j], ds[j]))
                # For second body: -n.dc - .5 n.dS.n/r.
                drift = vlsa_barrier(grip_bodies[i], grip_bodies[j], n).value-d
                rows.append({"key": key, "i": i, "j": j, "distance": d, "normal": n,
                             "pair_type": "inter_arm",
                             "geometry": geometry,
                             "gradient": grad, "drift": drift, "velocity": 0.,
                             "margin": margin, "environment": False})
        return rows

    def _guidance(self, q, grip, goal, focus, row, lower, upper):
        cfg = self.config
        g = row["gradient"]
        projector = np.eye(12)-np.outer(g, g)/max(g @ g, 1e-18)
        now = time.monotonic()
        if (self.guide_key == row["key"] and now-self.guide_at < cfg.guidance_refresh_s
                and np.linalg.norm(self.direction) > 1e-8):
            direction = projector @ self.direction
            return direction/max(np.linalg.norm(direction), 1e-12), {"cached": True}
        _, _, vt = np.linalg.svd(g[None], full_matrices=True)
        goal_tangent = projector @ (goal-q)
        candidates = []
        if np.linalg.norm(goal_tangent) > 1e-8:
            candidates.append(goal_tangent/np.linalg.norm(goal_tangent))
        # Rank joint-space tangents by how much they move the blocking body.
        _, dc, ds = self.robot.differential(q, grip)
        mobility = [(np.linalg.norm(dc[focus] @ v)+10*np.linalg.norm(ds[focus] @ v), v)
                    for v in vt[1:]]
        for _, v in sorted(mobility, key=lambda item: item[0], reverse=True):
            candidates.extend([v, -v])
        if np.linalg.norm(self.direction) > 1e-8:
            candidates.insert(0, projector @ self.direction)
        unique = []
        for candidate in candidates:
            norm = np.linalg.norm(candidate)
            if norm > 1e-8:
                candidate = candidate/norm
                if not any(candidate @ other > .99 for other in unique):
                    unique.append(candidate)
        candidates = unique
        best, attempted = None, 0
        started = time.perf_counter()
        obstacle = self.obstacles[row["j"]] if row["environment"] else None
        level = max(row["distance"], row["margin"])
        for initial in candidates[:cfg.geodesic_candidates]:
            if np.linalg.norm(initial) < 1e-8:
                continue
            position, direction = q.copy(), initial/np.linalg.norm(initial)
            cost, n = 0., row["normal"]
            candidate_deadline = started+cfg.guidance_budget_s*(attempted+1)/2
            for _ in range(cfg.geodesic_horizon):
                bs, cj, sj = self.robot.differential(position, grip)
                a = bs[focus]
                b = obstacle if obstacle is not None else bs[row["j"]]
                distance, n, _, _ = separation(a, b, n)
                gradient = separation_gradient(a, n, cj[focus], sj[focus])
                if obstacle is None:
                    gradient += separation_gradient(b, -n, cj[row["j"]], sj[row["j"]])
                direction -= gradient*(gradient @ direction)/max(gradient @ gradient, 1e-12)
                direction /= max(np.linalg.norm(direction), 1e-12)
                advance = direction*.08
                advance -= gradient*np.clip(distance-level, -.01, .01)/max(gradient @ gradient, 1e-12)
                position = np.clip(position+advance, lower, upper)
                cost += .05*np.linalg.norm(position-goal)
                if time.perf_counter() >= candidate_deadline:
                    break
            cost += np.linalg.norm(position-goal)
            # Check candidate endpoint against all current ellipsoids, including
            # other links. Soft cost steers around competing obstacles.
            end_bodies = self.robot.bodies(position, grip)
            for other in self.obstacles:
                body = end_bodies[focus]
                if np.linalg.norm(body.center-other.center) < body.radius+other.radius+.02:
                    gap = separation(body, other)[0]
                    cost += 20*max(row["margin"]-gap, 0.)
            cost += .015*(1-initial @ self.direction)
            attempted += 1
            if best is None or cost < best[0]:
                best = cost, initial/np.linalg.norm(initial)
            if attempted >= 2 and time.perf_counter()-started >= cfg.guidance_budget_s:
                break
        direction = np.zeros(12) if best is None else best[1]
        self.direction, self.guide_key, self.guide_at = direction, row["key"], now
        return direction, {"cached": False, "candidate_count": attempted,
                           "cost": None if best is None else float(best[0])}

    def _solve(self, reference, previous, low, high, matrix, rhs, guide, progress, progress_upper, *, relaxed=False):
        import osqp
        cfg = self.config
        count, guide_count = len(rhs), int(guide is not None)
        controls = matrix.shape[1]
        virtuals = controls-12
        slack_count = count if relaxed else 0
        size = controls+slack_count+guide_count
        h = self.metric+np.eye(12)*cfg.smoothness_weight
        linear = -self.metric @ reference-cfg.smoothness_weight*previous
        h += cfg.goal_progress_weight*(progress.T @ progress)
        linear -= cfg.goal_progress_weight*progress.T @ (progress @ reference)
        extra = np.r_[np.full(virtuals, cfg.virtual_direction_weight),
                      np.full(slack_count, cfg.collision_slack_weight*self.penalty_scale),
                      np.full(guide_count, cfg.guidance_weight)]
        hessian = sparse.block_diag((h, np.diag(extra)), format="csc") if len(extra) else sparse.csc_matrix(h)
        rows, lo, hi = [], [], []
        if count:
            row = np.zeros((count, size)); row[:, :controls] = matrix
            if relaxed:
                row[:, controls:controls+count] = np.eye(count)
            rows.append(row); lo.extend(rhs); hi.extend([np.inf]*count)
        if guide is not None:
            row = np.zeros((1, size)); row[0, :12] = guide[0]; row[0, -1] = 1
            rows.append(row); lo.append(guide[1]); hi.append(np.inf)
        if len(progress):
            row = np.zeros((len(progress), size)); row[:, :12] = progress
            rows.append(row); lo.extend(np.zeros(len(progress))); hi.extend(progress_upper)
        rows.append(np.eye(size))
        direction_limit = cfg.virtual_direction_step_rad/np.sqrt(2.)
        lo.extend(np.r_[low, np.full(virtuals, -direction_limit), np.zeros(slack_count+guide_count)])
        hi.extend(np.r_[high, np.full(virtuals, direction_limit), np.full(slack_count+guide_count, np.inf)])
        solver = osqp.OSQP()
        solver.setup(P=hessian, q=np.r_[linear, -cfg.virtual_direction_weight*self.virtual_reference,
                                       np.zeros(slack_count+guide_count)],
                     A=sparse.csc_matrix(np.vstack(rows)), l=np.array(lo), u=np.array(hi),
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=2500, time_limit=.025,
                     verbose=False, polishing=False)
        result = solver.solve(raise_error=False)
        self.runtime_stats["qp_solve_calls"] += 1
        valid = (result.info.status_val in (1, 2) and result.x is not None
                 and np.isfinite(result.x).all())
        if valid:
            residual = np.vstack(rows) @ result.x
            valid = bool(np.all(residual >= np.array(lo)-5e-4) and np.all(residual <= np.array(hi)+5e-4))
        return result.x[:controls] if valid else None, str(result.info.status)

    def plan(self, q, grip, actions, boxes, *, lower, upper, max_step, previous_delta=None,
             acceleration_step=np.inf):
        began = time.perf_counter()
        q, grip, actions = (np.asarray(x, float) for x in (q, grip, actions))
        if (q.shape != (12,) or grip.shape != (2,) or actions.ndim != 2 or actions.shape[1] != 14
                or not len(actions) or not all(np.isfinite(x).all() for x in (q, grip, actions))
                or np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1))):
            raise ValueError("Expected finite dual-arm states and absolute 14-D actions")
        lower, upper = np.broadcast_to(lower, (12,)), np.broadcast_to(upper, (12,))
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower > upper) or np.isnan(max_step) or max_step <= 0):
            raise ValueError("Invalid joint bounds or step")
        action = actions[0]
        cfg = self.config
        if (self.goal_reference is None
                or np.max(np.abs(action[:12]-self.goal_reference)) > cfg.goal_change_tolerance_rad):
            self.best_goal_error, self.stagnant_steps = np.inf, 0
            self.relaxation_level = 0
            self.goal_reference, self.goal_start = action[:12].copy(), q.copy()
            self.guide_key = None
            self.direction[:] = 0
        error = float(np.linalg.norm(action[:12]-q))
        if error < self.best_goal_error-.002:
            self.best_goal_error, self.stagnant_steps = error, 0
        else:
            self.stagnant_steps += 1
        relaxation_level = max(self.relaxation_level, min(self.stagnant_steps//cfg.stagnation_steps,
                                                        cfg.max_relaxation_level))
        self.relaxation_level = relaxation_level
        self.penalty_scale = 10.**(-relaxation_level)
        self.metric = np.eye(12)
        if self.optimization_space == "ee":
            from obstacle_validation.cartesian_chunk import CartesianChunkQP
            tcp = CartesianChunkQP(self.sampler, action[None, :12], q, np.zeros(12), np.ones(1)).jacobian(q)
            self.metric += tcp.T @ np.diag(np.tile([100.,100.,100.,1.,1.,1.], 2)) @ tcp
        step = min(max_step, cfg.max_control_step_rad)
        previous = np.zeros(12) if previous_delta is None else np.asarray(previous_delta, float)
        if previous.shape != (12,) or not np.isfinite(previous).all():
            raise ValueError("Invalid previous joint increment")
        if np.isnan(acceleration_step) or acceleration_step <= 0:
            raise ValueError("Acceleration increment must be positive")
        low, high = np.maximum(lower-q, -step), np.minimum(upper-q, step)
        if np.isfinite(acceleration_step):
            low, high = np.maximum(low, previous-acceleration_step), np.minimum(high, previous+acceleration_step)
        if np.any(low > high):
            raise ValueError("Measured state and rate bounds have no feasible joint step")
        nominal = np.clip(action[:12]-q, low, high)
        goal_error = action[:12]-q
        # A zero target delta has no preferred detour direction. Do not invent
        # guidance for it, but leave every joint free for distance constraints.
        guide_arms = np.max(np.abs(goal_error.reshape(2, 6)), axis=1) > cfg.goal_change_tolerance_rad
        guidance_movable = np.repeat(guide_arms, 6)
        # Old motion must not pull a stationary/reversed target away just
        # because an unrelated distance constraint activates the QP.
        objective_previous = np.where(nominal*previous > 0,
                                      np.sign(previous)*np.minimum(np.abs(previous), np.abs(nominal)), 0.)
        progress, progress_upper = [], []
        # Keep the intended segment direction across feedback retries. Using
        # the changing error vector would forbid necessary lateral detours
        # around a convex obstacle, even though the path never goes backward.
        segment = action[:12]-self.goal_start
        for arm in range(2):
            indices = slice(6*arm, 6*arm+6)
            norm = np.linalg.norm(segment[indices])
            if norm > cfg.goal_change_tolerance_rad:
                row = np.zeros(12); row[indices] = segment[indices]/norm
                progress.append(row)
                progress_upper.append(max(float(row @ goal_error), 0.))
        progress = np.asarray(progress).reshape(-1, 12)
        progress_upper = np.asarray(progress_upper)
        self.update_scene(boxes)
        bodies, dc, ds = self.robot.differential(q, grip)
        grip_bodies = self.robot.bodies(q, action[12:])
        pairs = self._pairs(bodies, dc, ds, step, grip_bodies)
        environment = [r for r in pairs if r["environment"]]
        dynamic = any(r["distance"] < .35 and r["velocity"] > cfg.closing_speed_m_s for r in environment)
        # Opposing nearby normals indicate confinement for this link.
        narrow = any(a["i"] == b["i"] and a["normal"] @ b["normal"] < -.5
                     and a["distance"]+b["distance"] < .10
                     for i, a in enumerate(environment) for b in environment[i+1:])
        stage_name = "dynamic_obstacle" if dynamic else "narrow_space" if narrow else "normal"
        scale = cfg.dynamic_margin_scale if dynamic else cfg.narrow_margin_scale if narrow else 1.
        alpha = cfg.alpha_per_s*(.5 if dynamic else 2. if narrow else 1.)
        eta = -np.expm1(-alpha*self.period)
        for row in pairs:
            if row["environment"]:
                row["margin"] *= scale
            row["rhs"] = (-eta*(row["distance"]-row["margin"])
                          +row["velocity"]*self.period-row["drift"])
        # Candidate proximity is broad phase, not permission to modify an
        # action. Test the bounded nominal step, including gripper geometry.
        predicted = self.robot.bodies(q+nominal, action[12:]) if pairs else []
        for row in pairs:
            other = self.obstacles[row["j"]] if row["environment"] else predicted[row["j"]]
            # Evaluate exactly the SAME virtual plane as the current CBF and
            # its Jacobians. Maximizing a new normal here mixed two barriers.
            predicted_geometry = vlsa_barrier(predicted[row["i"]], other, row["normal"])
            predicted_gap = predicted_geometry.value
            predicted_gap -= row["velocity"]*self.period
            row["predicted_distance"] = predicted_gap
            row["predicted_center_distance"] = predicted_geometry.center_distance
            if predicted_gap < row["margin"]:
                row["rhs"] = max(row["rhs"], row["gradient"] @ nominal+row["margin"]-predicted_gap)
        # One VLSA sphere state per pair, with two independent tangent controls.
        # Their increments update only the separating plane, not robot joints.
        matrix = np.zeros((len(pairs), 12+2*len(pairs)))
        for index, row in enumerate(pairs):
            matrix[index, :12] = row["gradient"]
            matrix[index, 12+2*index:14+2*index] = row["geometry"].direction_gradient
        # The virtual plane must keep seeking a separating direction even
        # when its current h is positive. A zero nominal direction can leave
        # the plane on the wrong side after passing an obstacle and cause a
        # spurious avoidance loop on the way back to the recorded goal.
        self.virtual_reference = np.zeros(2*len(pairs))
        limit = cfg.virtual_direction_step_rad/np.sqrt(2.)
        for index, row in enumerate(pairs):
            gradient = row["geometry"].direction_gradient
            reference = cfg.virtual_direction_step_rad*gradient/max(np.linalg.norm(gradient), .02)
            reference = np.clip(reference, -limit, limit)
            geometry = row["geometry"]
            other = self.obstacles[row["j"]] if row["environment"] else bodies[row["j"]]
            for _ in range(8):
                normal = geometry.normal+geometry.tangent_basis @ reference
                if vlsa_barrier(bodies[row["i"]], other, normal).value >= geometry.value-1e-12:
                    break
                reference *= .5
            else:
                reference[:] = 0
            self.virtual_reference[2*index:2*index+2] = reference
        rhs = np.array([r["rhs"] for r in pairs])
        violated = [r for r in pairs if r["gradient"] @ nominal < r["rhs"]-1e-7]
        triggered = bool(violated)
        blocking = [r for r in violated if r["distance"] < r["margin"]+cfg.guidance_distance_m
                    and np.linalg.norm(r["gradient"]) > 1e-8
                    and r["gradient"] @ nominal < -1e-7]
        reference, guide, geodesic = nominal.copy(), None, None
        if blocking and not relaxation_level:
            focus = min(blocking, key=lambda r: r["distance"]-r["margin"])
            if self.last_goal is not None and np.linalg.norm(action[:12]-self.last_goal) > .3:
                self.guide_key = None
            direction, geodesic = self._guidance(q, grip, action[:12], focus["i"], focus, lower, upper)
            direction = direction*guidance_movable
            norm = np.linalg.norm(direction)
            if norm > 1e-8 and np.all(progress @ direction >= -1e-8):
                direction /= norm
                # Tangency is secondary to tracking: never replace the nominal
                # objective or request more lateral motion than goal motion.
                guide = (direction, min(step*.25, np.linalg.norm(nominal)*.25))
                self.direction = direction.copy()
            else:
                geodesic = {**geodesic, "rejected": "opposes_goal_or_has_no_goal_direction"}
                self.direction[:] = 0
                self.guide_key = None
        self.last_goal = action[:12].copy()
        calls = self.runtime_stats["qp_solve_calls"]
        relaxed, fallback = False, None
        if not triggered:
            delta, status = np.r_[nominal, self.virtual_reference], "nominal_feasible"
            self.direction[:] = 0
            self.guide_key = None
            self.stagnant_steps, self.relaxation_level = 0, 0
            relaxation_level = 0
            self.best_goal_error = error
        elif relaxation_level:
            # Persistent lack of goal progress gradually trades clearance for
            # reaching, explicitly reported; every tick still evaluates all
            # nearby bodies. No open-loop or latched bypass.
            reference, guide = nominal, None
            delta, status = self._solve(reference, objective_previous, low, high, matrix, rhs, guide, progress, progress_upper, relaxed=True)
            relaxed = True
        else:
            delta, status = self._solve(reference, objective_previous, low, high, matrix, rhs, guide, progress, progress_upper)
            relaxed = delta is None
        if delta is None:
            delta, status = self._solve(reference, objective_previous, low, high, matrix, rhs, guide, progress, progress_upper, relaxed=True)
        if delta is None:
            # Bounded minimization of the same penalized distance objective.
            # Never latch an action or discard future scene updates.
            from scipy.optimize import LinearConstraint, minimize
            def objective(x):
                violations = np.maximum(rhs-matrix @ x, 0.)
                weight = cfg.collision_slack_weight*self.penalty_scale
                tracking = x[:12]-reference
                forward_error = progress @ tracking
                guide_error = max(guide[1]-guide[0] @ x[:12], 0.) if guide is not None else 0.
                value = (.5*tracking @ self.metric @ tracking
                         +.5*cfg.smoothness_weight*np.sum((x[:12]-objective_previous)**2)
                         +.5*cfg.goal_progress_weight*(forward_error @ forward_error)
                         +.5*weight*(violations @ violations)+.5*cfg.guidance_weight*guide_error**2
                         +.5*cfg.virtual_direction_weight*np.sum((x[12:]-self.virtual_reference)**2))
                gradient = np.r_[self.metric @ tracking+cfg.smoothness_weight*(x[:12]-objective_previous)
                                  +cfg.goal_progress_weight*progress.T @ forward_error,
                                  cfg.virtual_direction_weight*(x[12:]-self.virtual_reference)]-weight*matrix.T @ violations
                if guide is not None:
                    gradient[:12] -= cfg.guidance_weight*guide_error*guide[0]
                return value, gradient
            virtuals = matrix.shape[1]-12
            limit = cfg.virtual_direction_step_rad/np.sqrt(2.)
            control_low, control_high = np.r_[low, np.full(virtuals, -limit)], np.r_[high, np.full(virtuals, limit)]
            augmented_progress = np.pad(progress, ((0, 0), (0, virtuals)))
            result = minimize(objective, np.r_[np.clip(reference, low, high), self.virtual_reference], jac=True, method="SLSQP",
                              bounds=list(zip(control_low, control_high)),
                              constraints=[LinearConstraint(augmented_progress, 0., progress_upper)] if len(progress) else (),
                              options={"maxiter": 100, "ftol": 1e-10})
            if (not np.isfinite(result.x).all() or np.any(progress @ result.x[:12] < -1e-8)
                    or np.any(progress @ result.x[:12] > progress_upper+1e-8)):
                raise RuntimeError("Ellipsoid optimization cannot preserve forward goal progress")
            delta, fallback = result.x, "bounded_penalty_optimizer"
        controls = delta.copy()
        delta = controls[:12].copy()
        delta = np.clip(delta, low, high)
        # OSQP's feasibility tolerance also applies to hard progress rows.
        # Repair only numerical residuals; a materially backward solution
        # must not reach either executor, including the fallback path.
        if triggered:
            for row, limit in zip(progress, progress_upper):
                projection = float(row @ delta)
                correction = float(np.clip(projection, 0., limit)-projection)
                if abs(correction) > 5e-4:
                    raise RuntimeError("Ellipsoid optimization returned backward or overshooting progress")
                delta += row*correction
        if np.any(delta < low-1e-8) or np.any(delta > high+1e-8):
            raise RuntimeError("Forward progress conflicts with joint/rate bounds")
        controls[:12] = delta
        violations = np.maximum(rhs-matrix @ controls, 0.)
        maximum_slack = float(violations.max(initial=0.))
        post_bodies = self.robot.bodies(q+delta, action[12:]) if triggered else predicted
        for index, row in enumerate(pairs):
            geometry = row["geometry"]
            direction_delta = controls[12+2*index:14+2*index]
            normal = geometry.normal+geometry.tangent_basis @ direction_delta
            normal /= np.linalg.norm(normal)
            self.normals[row["key"]] = normal
            other = self.obstacles[row["j"]] if row["environment"] else post_bodies[row["j"]]
            after = vlsa_barrier(post_bodies[row["i"]], other, normal).value-row["margin"]
            if row["environment"]:
                after -= normal @ self.velocities[row["j"]]*self.period
            row["predicted_barrier_after"] = float(after)
        elapsed = time.perf_counter()-began
        stage = {"stage": stage_name, "margin_m": self.margin*scale, "alpha_per_s": alpha}
        pair_types = ("left_obstacle", "right_obstacle", "inter_arm")
        step_info = {"success": True, "controller": "ellipsoid_mcbf", "solver_message": status,
                     "solver_success": fallback is None, "fallback": fallback,
                     "distance_constraints_relaxed": relaxed, "max_distance_slack_m": maximum_slack,
                     "stagnation_steps": self.stagnant_steps, "relaxation_level": relaxation_level,
                     "triggered": triggered, "nominal_feasible": not triggered,
                     "optimization_applied": triggered, "active_constraints": len(violated),
                     "same_arm_avoidance": False,
                     "constraint_pair_counts": {kind: sum(r["pair_type"] == kind for r in pairs) for kind in pair_types},
                     "active_constraint_counts": {kind: sum(r["pair_type"] == kind for r in violated) for kind in pair_types},
                     "path_progress_rad": (progress @ delta).tolist(),
                     "goal_error_before_rad": float(np.linalg.norm(goal_error)),
                     "goal_error_after_rad": float(np.linalg.norm(goal_error-delta)),
                     "violated_pairs": [{"key": r["key"], "distance_m": r["distance"],
                         "pair_type": r["pair_type"],
                         "plane_gap_m": r["distance"],
                         "barrier_h_m": r["distance"]-r["margin"],
                         "first_center_m": bodies[r["i"]].center.tolist(),
                         "second_center_m": (self.obstacles[r["j"]] if r["environment"] else bodies[r["j"]]).center.tolist(),
                         "center_distance_m": r["geometry"].center_distance,
                         "relative_centers_m": r["geometry"].center_delta.tolist(),
                         "separating_normal": r["normal"].tolist(),
                         "support_radii_m": r["geometry"].support_radii.tolist(),
                         "predicted_barrier_after_m": r["predicted_barrier_after"],
                         "predicted_center_distance_m": r["predicted_center_distance"],
                         "predicted_distance_m": r["predicted_distance"], "margin_m": r["margin"]}
                         for r in violated],
                     "barrier_model": "vlsa_relative_centers_support_plane",
                     "virtual_direction_variables": matrix.shape[1]-12,
                     "virtual_direction_delta": controls[12:].tolist(),
                     "adaptive_stage": stage, "geodesic": geodesic,
                     "detour_direction": self.direction.tolist(), "avoidance_bypassed": False,
                     "constraint_pairs": len(pairs), "robot_ellipsoids": len(bodies),
                     "obstacle_ellipsoids": len(self.obstacles), "gripper_variables": 0,
                     "post_optimization_checks": True, "sweep_validation_enabled": False}
        waypoint = np.r_[q+delta, action[12:]]
        info = {"success": True, "controller": "ellipsoid_mcbf", "mode": "ellipsoid_mcbf_feedback",
                "horizon": 1, "optimization_space": self.optimization_space, "joint_variables": 12, "ee_variables": 0,
                "gripper_variables": 0, "waypoints": [waypoint.tolist()],
                "same_arm_avoidance": False,
                "barrier_model": "vlsa_relative_centers_support_plane",
                "virtual_direction_variables": matrix.shape[1]-12,
                "steps": [step_info], "waypoint_steps": [step_info], "waypoint_action_indices": [0],
                "triggered_action_indices": [0] if triggered else [],
                "modified_action_indices": [0] if np.max(np.abs(delta-nominal)) > 1e-8 else [],
                "joint_qp_solve_calls": self.runtime_stats["qp_solve_calls"]-calls,
                "solver": "ellipsoid_osqp", "planning_seconds": elapsed, "adaptive_stage": stage,
                "avoidance_bypassed": False, "reason": None, "velocity_source": "matched_ellipsoids",
                "sweep_validation_enabled": False, "post_optimization_checks": True,
                "adaptive_feedback": True, "target_repairs": []}
        counts = self.runtime_stats["stage_counts"]
        counts[stage_name] = counts.get(stage_name, 0)+1
        self.runtime_stats["stage_s"]["pipeline"] = self.runtime_stats["stage_s"].get("pipeline", 0.)+elapsed
        return action[None].copy(), info
