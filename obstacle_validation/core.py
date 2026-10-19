"""Known joint paths, fixed-point FK collision checks and local RRT detours."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.spatial import cKDTree

from safety_module.geometry import segments_within_aabb_margin


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result


@dataclass
class JointPath:
    q: np.ndarray
    gripper: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    dt: float
    speed: float
    acceleration: float
    model_hash: str
    seed: int
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        if self.q.ndim != 2 or self.q.shape[1] != 12 or len(self.q) < 2 or not np.isfinite(self.q).all():
            raise ValueError("Joint path must be finite [T>=2,12] absolute joint positions")
        self.gripper = vector(self.gripper, 2, "gripper")
        self.lower = vector(self.lower, 12, "lower")
        self.upper = vector(self.upper, 12, "upper")
        if np.any(self.lower >= self.upper) or np.any(self.q < self.lower) or np.any(self.q > self.upper):
            raise ValueError("Path exceeds joint bounds")
        if np.any(self.gripper < 0) or np.any(self.gripper > 1):
            raise ValueError("Gripper closing fractions must lie in [0,1]")
        if not all(np.isfinite(v) and v > 0 for v in (self.dt, self.speed, self.acceleration)):
            raise ValueError("Period, speed and acceleration must be finite and positive")
        if np.max(np.abs(np.diff(self.q, axis=0))) > self.speed * self.dt + 1e-8:
            raise ValueError("Path exceeds velocity limit")

def smooth_joint_path(knots, gripper, *, dt=.05, speed=.5, acceleration=1.2,
                      durations=None, model_hash="", lower=None, upper=None, seed=7, source=None):
    """Retain every joint knot; stop at knots using bounded quintic segments."""
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or knots.shape[1] != 12 or len(knots) < 2 or not np.isfinite(knots).all():
        raise ValueError("Knots must be finite [T>=2,12]")
    if not all(np.isfinite(v) and v > 0 for v in (dt, speed, acceleration)):
        raise ValueError("Invalid motion limits")
    durations = np.full(len(knots)-1, dt) if durations is None else np.asarray(durations, dtype=float)
    if durations.shape != (len(knots)-1,) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Segment durations must be finite and positive")
    lower = knots.min(0)-.15 if lower is None else lower
    upper = knots.max(0)+.15 if upper is None else upper
    result, knot_indices = [knots[0]], [0]
    for start, end, requested in zip(knots[:-1], knots[1:], durations):
        distance = np.max(np.abs(end-start))
        duration = max(dt, requested, 1.875*distance/speed, np.sqrt(5.774*distance/acceleration))
        count = int(np.ceil(duration/dt))
        t = np.arange(1, count+1)/count
        blend = 10*t**3 - 15*t**4 + 6*t**5
        result.extend(start+blend[:, None]*(end-start))
        knot_indices.append(len(result)-1)
    source = dict(source or {})
    if source:
        source["knot_path_indices"] = knot_indices
    return JointPath(np.asarray(result), gripper, lower, upper, dt, speed, acceleration,
                     model_hash, seed, source)


def segments_within_box_margin(start, end, box, margin):
    """Exact Euclidean distance from point segments to an OBB, with round corners."""
    return segments_within_aabb_margin((np.asarray(start)-box.center) @ box.axes,
                                       (np.asarray(end)-box.center) @ box.axes,
                                       box.half_sizes, margin)


class CollisionChecker:
    """Check every selected fixed point, with segments between small FK steps."""

    def __init__(self, sampler, gripper, boxes, *, margin=.01, inter_arm_margin=.035,
                 resolution=.01, observation_now_ns=None):
        if not all(np.isfinite(v) and v >= 0 for v in (margin, inter_arm_margin)) or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Invalid collision-check settings")
        self.sampler, self.gripper, self.boxes = sampler, vector(gripper, 2, "gripper"), boxes
        self.margin, self.inter_arm_margin, self.resolution = margin, inter_arm_margin, resolution
        if observation_now_ns is not None and (not isinstance(observation_now_ns, int)
                                                or observation_now_ns <= 0):
            raise ValueError("Invalid fixed observation check time")
        # A planner checks many candidates against one immutable observation.
        # Physical execution must use a separate checker with the current time.
        self.observation_now_ns = observation_now_ns
        self.edge_rejections = {"unobserved": 0, "obstacle": 0, "inter_arm": 0}
        self.observation_rejections = {"unknown": 0, "expired_free": 0, "occupied": 0}
        self.last_observation_blockage = None
        self.left_count = len(sampler.left_link_names) * sampler.points_per_link

    def points(self, q):
        result = np.asarray(self.sampler.link_points(q, self.gripper)).reshape(-1, 3)
        if not len(result) or not np.isfinite(result).all() or self.left_count >= len(result):
            raise ValueError("Invalid dual-arm FK points")
        return result

    def clearance(self, q):
        points = self.points(q)
        obstacle = float("inf")
        for box in self.boxes:
            delta = np.abs((points - box.center) @ box.axes) - box.half_sizes
            signed = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0)
            obstacle = min(obstacle, float(signed.min()))
        arm = float(cKDTree(points[self.left_count:]).query(points[:self.left_count])[0].min())
        return obstacle, arm

    def state_safe(self, q):
        obstacle, arm = self.clearance(q)
        return obstacle > self.margin and arm > self.inter_arm_margin

    def edge_safe(self, start, end):
        start, end = vector(start, 12, "start"), vector(end, 12, "end")
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.resolution)))
        previous = self.points(start)
        observation = getattr(self.boxes, "observation", None)
        for index in range(1, steps + 1):
            current = self.points(start + (end - start) * index / steps)
            if observation is not None and not observation.segments_observed(
                    previous, current, margin_m=self.margin,
                    now_ns=self.observation_now_ns):
                self.edge_rejections["unobserved"] += 1
                blockage = getattr(observation, "last_blockage", None)
                if blockage:
                    self.last_observation_blockage = dict(blockage)
                    self.observation_rejections[blockage["reason"]] += 1
                return False
            if any(segments_within_box_margin(previous, current, box, self.margin).any() for box in self.boxes):
                self.edge_rejections["obstacle"] += 1
                return False
            # Exact minimum separation of linearly interpolated left/right points.
            first = previous[:self.left_count, None] - previous[None, self.left_count:]
            last = current[:self.left_count, None] - current[None, self.left_count:]
            delta = last - first
            t = np.clip(-(first * delta).sum(-1) / np.maximum((delta * delta).sum(-1), 1e-15), 0, 1)
            if np.any(np.linalg.norm(first + t[..., None] * delta, axis=-1) <= self.inter_arm_margin):
                self.edge_rejections["inter_arm"] += 1
                return False
            previous = current
        return True

    def path_safe(self, q):
        return self.state_safe(q[0]) and all(self.edge_safe(a, b) for a, b in zip(q[:-1], q[1:]))


def rrt_detour(start, goal, checker, lower, upper, *, rng, iterations=800,
               extension=.06, timeout_s=2.):
    """Bidirectional RRT; all tree edges and final shortcuts are FK checked.

    Returns absolute configurations excluding the start, or None. Bounds
    describe the permitted detour envelope; planning never expands it itself.
    """
    if not checker.state_safe(start) or not checker.state_safe(goal):
        return None
    if checker.edge_safe(start, goal):
        return [np.asarray(goal).copy()]
    deadline = time.monotonic() + timeout_s
    trees = [([np.asarray(start).copy()], [-1]), ([np.asarray(goal).copy()], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        node = nodes[nearest] + delta * min(1., extension / max(np.linalg.norm(delta), 1e-12))
        if not checker.edge_safe(nodes[nearest], node):
            return None
        nodes.append(node)
        parents.append(nearest)
        return len(nodes) - 1

    def branch(tree, index):
        nodes, parents = tree
        result = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return result[::-1]

    for iteration in range(iterations):
        if time.monotonic() >= deadline:
            break
        side = iteration % 2
        target = trees[1-side][0][0] if rng.random() < .2 else rng.uniform(lower, upper)
        first = extend(trees[side], target)
        if first is None:
            continue
        meeting = trees[side][0][first]
        while time.monotonic() < deadline:
            second = extend(trees[1-side], meeting)
            if second is None:
                break
            if np.linalg.norm(trees[1-side][0][second] - meeting) < 1e-9:
                a, b = branch(trees[side], first), branch(trees[1-side], second)
                route = a + b[-2::-1] if side == 0 else b + a[-2::-1]
                # Remove needless random turns, preserving checked edges.
                result, index = [], 0
                while index < len(route) - 1:
                    next_index = index + 1
                    for candidate in range(len(route)-1, index+1, -1):
                        if time.monotonic() >= deadline:
                            break
                        if checker.edge_safe(route[index], route[candidate]):
                            next_index = candidate
                            break
                    result.append(route[next_index])
                    index = next_index
                return result
    return None


class PathFollower:
    """Track a saved nominal path; replace a blocked stretch with a detour."""

    def __init__(self, path, *, avoid, detour_lower, detour_upper, lookahead=80,
                 planning_timeout=2., tolerance=.003, max_step=.015, cbf_projector=None,
                 direct_goal_cbf=False):
        self.path, self.avoid = path, avoid
        self.lower = vector(detour_lower, 12, "detour_lower")
        self.upper = vector(detour_upper, 12, "detour_upper")
        if np.any(self.lower >= self.upper) or np.any(path.q < self.lower) or np.any(path.q > self.upper):
            raise ValueError("Detour envelope must contain the nominal path")
        self.lookahead, self.planning_timeout, self.tolerance = lookahead, planning_timeout, tolerance
        self.max_step = min(max_step, path.speed * path.dt)
        self.cbf_projector = cbf_projector
        self.direct_goal_cbf = bool(direct_goal_cbf)
        self.cbf_interventions = 0
        self.cbf_failures = 0
        self.last_cbf_info = None
        self.index, self.rejoin, self.detour, self.replans = 1, None, [], 0
        self.skipped_samples = 0
        self.rng = np.random.default_rng(path.seed)
        self.last_delta = np.zeros(12)
        self.before_plan = None
        self.last_plan_duration_s = None

    @property
    def completed(self):
        return self.index >= len(self.path.q) and not self.detour

    def next_delta(self, q, checker):
        q = vector(q, 12, "q")
        self.last_cbf_info = None
        if np.any(q < self.lower) or np.any(q > self.upper) or not checker.state_safe(q):
            return np.zeros(12), "unsafe_current_state"
        while self.detour and np.max(np.abs(self.detour[0] - q)) <= self.tolerance:
            self.detour.pop(0)
            self.last_delta[:] = 0
            if not self.detour:
                self.index, self.rejoin = self.rejoin + 1, None
        if self.direct_goal_cbf:
            # Restoration optimizes each step toward the final recorded pose.
            # Intermediate interpolation knots can lie inside an obstacle and
            # must not become mandatory CBF/RRT targets.
            if np.max(np.abs(self.path.q[-1] - q)) <= self.tolerance:
                self.detour = []
                self.index = len(self.path.q)
            elif self.completed:
                self.index = len(self.path.q)-1
        else:
            while not self.detour and not self.completed and np.max(np.abs(self.path.q[self.index] - q)) <= self.tolerance:
                self.index += 1
        if self.completed:
            return np.zeros(12), "completed"
        target = self.detour[0] if self.detour else (self.path.q[-1] if self.direct_goal_cbf else self.path.q[self.index])
        def bounded_step(destination):
            direction = destination-q
            distance = np.max(np.abs(direction))
            acceleration_step = self.path.acceleration * self.path.dt**2
            ramp = np.max(np.abs(self.last_delta)) + acceleration_step
            braking = max(acceleration_step, np.sqrt(2*acceleration_step*distance))
            length = min(distance, self.max_step, ramp, braking)
            return direction * (length / max(distance, 1e-15))

        # Restoration follows bounded absolute targets; CBF adjusts each step.
        if self.avoid and self.cbf_projector is not None:
            nominal = bounded_step(target)
            corrected, info = self.cbf_projector.project(
                q, nominal, checker, lower=self.lower, upper=self.upper,
                max_step=self.max_step, previous_delta=self.last_delta,
                acceleration_step=self.path.acceleration*self.path.dt**2)
            self.last_cbf_info = info
            if info["triggered"]:
                if info["success"] and np.max(np.abs(corrected-nominal)) > 1e-7:
                    self.cbf_interventions += 1
                elif not info["success"]:
                    self.cbf_failures += 1
            progress = float(corrected @ nominal)
            enough_progress = np.max(np.abs(nominal)) <= 1e-8 or progress > .01*float(nominal @ nominal)
            if info["success"] and enough_progress and checker.edge_safe(q, q+corrected):
                self.last_delta = corrected.copy()
                return corrected, "cbf" if info["triggered"] else ("detour" if self.detour else "nominal")
            # An infeasible or stalled local projection needs a checked
            # geometric route around the obstacle to make forward progress.
            if checker.edge_safe(q, target):
                return np.zeros(12), "cbf_blocked"
        if not checker.edge_safe(q, target):
            self.detour = []
            self.last_delta[:] = 0
            if not self.avoid:
                return np.zeros(12), "baseline_blocked"
            end = min(len(self.path.q)-1, self.index + self.lookahead)
            # A bounded rejoin distance prevents jumping to the end of the test.
            goal = next((j for j in range(end, self.index-1, -1) if checker.state_safe(self.path.q[j])), None)
            if goal is None:
                return np.zeros(12), "no_safe_rejoin"
            if self.before_plan is not None:
                self.before_plan()
            plan_start = time.monotonic()
            route = rrt_detour(q, self.path.q[goal], checker, self.lower, self.upper,
                               rng=self.rng, timeout_s=self.planning_timeout)
            self.last_plan_duration_s = time.monotonic() - plan_start
            self.replans += 1
            if route is None:
                return np.zeros(12), "planning_failed"
            self.skipped_samples += max(0, goal-self.index)
            self.detour, self.rejoin = route, goal
            target = route[0]
        # A scalar ramp preserves the collision-checked straight joint edge.
        # Independent per-joint clipping can cut an RRT corner into a box.
        delta = bounded_step(target)
        if np.any(q+delta < self.lower) or np.any(q+delta > self.upper) or not checker.edge_safe(q, q+delta):
            self.last_delta[:] = 0
            return np.zeros(12), "step_blocked"
        self.last_delta = delta.copy()
        return delta, "detour" if self.detour else "nominal"
