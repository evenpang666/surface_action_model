"""Perception-based Adaptive MCBF-QP, arXiv:2609.30495, Sec. III-E.

Reproduces stage selection, stage-dependent barriers, dynamic relative
velocity and multi-candidate geodesic guidance (Eqs. 16--21). The robot
adapter uses fixed surface-point Jacobians and the selected voxel/OBB
distance field, rather than the paper's planar base and learned GPDFs.
One feedback step is returned per solve. No output collision/residual test
is used. Solver failures use a moving guidance reference (a deployment
extension, not part of the paper's feasibility/convergence assumptions).
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from real_scripts.real_cbf_qp import (
    _box_gradient, point_jacobian_fd,
    segments_intersect_box, select_inter_arm_constraints,
)


@dataclass(frozen=True)
class AdaptiveMCBFConfig:
    geodesic_candidates: int = 16
    geodesic_horizon: int = 96
    geodesic_step_m: float = .02
    guidance_speed_m_s: float = .12
    narrow_width_m: float = .10
    local_radius_m: float = .6
    dynamic_distance_m: float = .35
    closing_speed_m_s: float = .03
    cluster_gap_m: float = .005
    confinement_bins: int = 36
    dynamic_margin_scale: float = 2.
    narrow_margin_scale: float = .25
    dynamic_alpha_scale: float = .5
    narrow_alpha_scale: float = 2.
    max_control_step_rad: float = .04

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Adaptive MCBF {name} must be positive")
        for name in ('geodesic_candidates', 'geodesic_horizon', 'confinement_bins'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 4:
                raise ValueError(f"Adaptive MCBF {name} must be an integer >= 4")
        if not self.narrow_margin_scale <= 1 <= self.dynamic_margin_scale:
            raise ValueError('Adaptive margin scales must be narrow <= normal <= dynamic')
        if not self.dynamic_alpha_scale <= 1 <= self.narrow_alpha_scale:
            raise ValueError('Adaptive alpha scales must be dynamic <= normal <= narrow')


class BoxScene:
    """Connected infrastructure clusters, signed distances and tracked motion."""

    def __init__(self, boxes, config, previous=None):
        self.boxes, self.config = boxes, config
        self.timestamp = (getattr(boxes, 'source_snapshot', None) or {}).get('host_timestamp_ns')
        self.centers = np.asarray([b.center for b in boxes]).reshape(-1, 3)
        self.axes = np.asarray([b.axes for b in boxes]).reshape(-1, 3, 3)
        self.half = np.asarray([b.half_sizes for b in boxes]).reshape(-1, 3)
        self.extents = np.einsum('nij,nj->ni', np.abs(self.axes), self.half)
        payload = (getattr(boxes, 'source_snapshot', None) or {}).get('obbs', [])
        box_velocities, declared = np.zeros_like(self.centers), np.zeros(len(boxes), bool)
        for i, item in enumerate(payload[:len(boxes)]):
            value = item.get('velocity_m_s', item.get('velocity'))
            if value is not None:
                value = np.asarray(value, float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError('Obstacle velocities must be finite XYZ vectors')
                box_velocities[i], declared[i] = value, True
        moving_boxes = np.linalg.norm(box_velocities, axis=1) > config.closing_speed_m_s
        self.groups = []
        if len(boxes):
            roots = np.arange(len(boxes))
            def root(i):
                while roots[i] != i:
                    roots[i] = roots[roots[i]]
                    i = roots[i]
                return i
            radius = np.linalg.norm(self.extents, axis=1)
            tree = cKDTree(self.centers)
            for i in range(len(boxes)):
                neighbors = tree.query_ball_point(self.centers[i], radius[i]+radius.max()+config.cluster_gap_m)
                for j in neighbors:
                    if j <= i or moving_boxes[i] != moving_boxes[j]:
                        continue
                    gap = np.maximum(np.abs(self.centers[i]-self.centers[j])-self.extents[i]-self.extents[j], 0.)
                    if np.linalg.norm(gap) <= config.cluster_gap_m:
                        roots[root(j)] = root(i)
            groups = {}
            for i in range(len(boxes)):
                groups.setdefault(root(i), []).append(i)
            self.groups = [np.asarray(ids) for ids in groups.values()]
        self.group_centers = np.asarray([self.centers[ids].mean(0) for ids in self.groups]).reshape(-1, 3)
        self.velocities = np.zeros_like(self.group_centers)
        self.velocity_source = 'unobserved_static'
        # Metadata may supply measured velocities; otherwise estimate only
        # from distinct timestamped captures, never from repeated cached maps.
        explicit = False
        for i, ids in enumerate(self.groups):
            measured = ids[declared[ids]]
            if len(measured):
                self.velocities[i] = box_velocities[measured].mean(0)
                explicit = True
        if explicit:
            self.velocity_source = 'snapshot_velocity'
        elif (previous is not None and self.timestamp is not None and previous.timestamp is not None
              and len(self.groups) and len(previous.groups)):
            dt = (self.timestamp-previous.timestamp)/1e9
            if 0 < dt <= 2.:
                distances, matches = cKDTree(previous.group_centers).query(self.group_centers)
                reverse = cKDTree(self.group_centers).query(previous.group_centers)[1]
                for i, (distance, match) in enumerate(zip(distances, matches)):
                    if reverse[match] == i and .003 < distance < min(.3, dt*2.):
                        old_size, new_size = len(previous.groups[match]), len(self.groups[i])
                        if .5 <= new_size/old_size <= 2.:
                            self.velocities[i] = (self.group_centers[i]-previous.group_centers[match])/dt
                self.velocity_source = 'matched_cluster_displacement'
        self.dynamic = np.linalg.norm(self.velocities, axis=1) > config.closing_speed_m_s
        self.static_ids = np.concatenate([ids for ids, moving in zip(self.groups, self.dynamic) if not moving]) \
            if any(~self.dynamic) else np.empty(0, int)

    def distances(self, points, ids=None):
        ids = np.arange(len(self.boxes)) if ids is None else np.asarray(ids, int)
        points = np.asarray(points).reshape(-1, 3)
        if not len(ids):
            return np.full((len(points), 0), np.inf)
        local = np.einsum('pbi,bij->pbj', points[:, None]-self.centers[ids], self.axes[ids])
        delta = np.abs(local)-self.half[ids]
        return np.linalg.norm(np.maximum(delta, 0.), axis=-1)+np.minimum(delta.max(-1), 0.)

    def field(self, point, ids):
        ids = np.asarray(ids, int)
        d = self.distances(np.asarray(point)[None], ids)[0]
        nearest = int(ids[np.argmin(d)])
        box = self.boxes[nearest]
        gradient = box.axes @ _box_gradient((point-box.center) @ box.axes, box.half_sizes)
        return float(d.min()), gradient

    def blocked(self, first, last, ids, margin):
        return any(bool(segments_intersect_box(first, last, self.boxes[i], margin)) for i in ids)

    def blocked_points(self, first, last, margin):
        blocked = np.zeros(len(first), bool)
        for box in self.boxes:
            blocked |= segments_intersect_box(first, last, box, margin)
        return blocked


def _plane(jacobian):
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    basis = u[:, singular > 1e-6]
    return basis


def _confinement(scene, point, basis, config):
    # Paper Sec. III-E2 defines confinement by one infrastructure cluster.
    # Combining independent corridor walls would misclassify a through-passage
    # as an enclosure and send the robot back toward its entry.
    for ids, moving in zip(scene.groups, scene.dynamic):
        if not moving:
            confined, opening = _cluster_confinement(scene, point, basis, config, ids)
            if confined:
                return confined, opening
    return False, None


def _cluster_confinement(scene, point, basis, config, ids):
    """Paper bearing bins, in a controllable 2-D section of the surface space."""
    if basis.shape[1] < 2 or not len(scene.static_ids):
        return False, None
    angles = (np.arange(config.confinement_bins)+.5)*2*np.pi/config.confinement_bins
    directions = np.cos(angles)[:, None]*basis[:, 0]+np.sin(angles)[:, None]*basis[:, 1]
    ranges = np.full(config.confinement_bins, config.local_radius_m)
    for i, direction in enumerate(directions):
        for j in ids:
            box = scene.boxes[j]
            local = (point-box.center) @ box.axes
            ray = direction @ box.axes
            parallel = np.abs(ray) < 1e-10
            if np.any(parallel & (np.abs(local) > box.half_sizes)):
                continue
            a = (-box.half_sizes-local)/np.where(parallel, 1., ray)
            b = (box.half_sizes-local)/np.where(parallel, 1., ray)
            entry = max(0., float(np.where(parallel, -np.inf, np.minimum(a, b)).max()))
            exit = float(np.where(parallel, np.inf, np.maximum(a, b)).min())
            if entry <= exit:
                ranges[i] = min(ranges[i], entry)
    confined = np.mean(ranges < config.local_radius_m*.95) >= .9
    if not confined:
        return False, None
    # The farthest opening between adjacent occupied bearing bins is the
    # temporary target. It is discarded as soon as confinement clears.
    jumps = np.abs(ranges-np.roll(ranges, 1))
    openings = np.flatnonzero((ranges >= config.local_radius_m*.95)
                             | (jumps > config.narrow_width_m))
    if not len(openings):
        return True, None
    index = max(openings, key=lambda i: ranges[i])
    return True, point+directions[index]*ranges[index]


def select_stage(scene, point, target, jacobian, base_margin, base_alpha, config):
    basis = _plane(jacobian)
    for ids, velocity, moving in zip(scene.groups, scene.velocities, scene.dynamic):
        if moving:
            distance, normal = scene.field(point, ids)
            if distance < config.dynamic_distance_m and normal @ velocity > config.closing_speed_m_s:
                return {'stage': 'dynamic_obstacle', 'margin_m': base_margin*config.dynamic_margin_scale,
                        'alpha': base_alpha*config.dynamic_alpha_scale, 'temporary_target': None,
                        'guidance_initialization': 'robot_state'}
    confined, opening = _confinement(scene, point, basis, config)
    target_confined, _ = _confinement(scene, target, basis, config)
    narrow = confined or (target_confined and np.linalg.norm(target-point) < config.local_radius_m)
    # Inter-cluster gap indicator, restricted to the local navigation region.
    nearby = []
    direction = target-point
    length = max(np.linalg.norm(direction), 1e-12)
    direction /= length
    for ids, moving in zip(scene.groups, scene.dynamic):
        if moving:
            continue
        low = (scene.centers[ids]-scene.extents[ids]).min(0)
        high = (scene.centers[ids]+scene.extents[ids]).max(0)
        closest = np.clip(point, low, high)
        if np.linalg.norm(closest-point) < config.local_radius_m and (closest-point) @ direction > -.1:
            nearby.append((low, high))
    for i, (lo, hi) in enumerate(nearby):
        for other_lo, other_hi in nearby[i+1:]:
            gap = np.linalg.norm(np.maximum(np.maximum(lo-other_hi, other_lo-hi), 0.))
            narrow |= 0 < gap < config.narrow_width_m
    return {'stage': 'narrow_space' if narrow else 'normal',
            'margin_m': base_margin*(config.narrow_margin_scale if narrow else 1.),
            'alpha': base_alpha*(config.narrow_alpha_scale if narrow else 1.),
            'temporary_target': opening.tolist() if confined and opening is not None else None,
            'confined': bool(confined),
            'guidance_initialization': 'zero_isoline' if narrow else 'robot_state'}


def geodesic_guidance(scene, point, target, jacobian, stage, config, previous=None):
    """Eq. 18: parallel-transport candidates on an isoline; minimize path potential."""
    ids = scene.static_ids
    if not len(ids) or not scene.blocked(point, target, ids, stage['margin_m']):
        return None
    # Only the closest infrastructure cluster blocking the direct route is
    # used for geodesic guidance. All clusters still constrain the QP.
    candidates = [group for group, moving in zip(scene.groups, scene.dynamic)
                  if not moving and scene.blocked(point, target, group, stage['margin_m'])]
    ids = min(candidates, key=lambda group: scene.field(point, group)[0])
    basis = _plane(jacobian)
    projector = basis @ basis.T
    start = point.copy()
    level = max(scene.field(point, ids)[0], stage['margin_m'])
    if stage['guidance_initialization'] == 'zero_isoline':
        level = stage['margin_m']  # h=distance-margin=0.
        for _ in range(24):
            distance, normal = scene.field(start, ids)
            gradient = projector @ normal
            start -= (distance-level)*gradient/max(gradient @ gradient, 1e-12)
            if abs(distance-level) < 1e-6:
                break
    _, normal = scene.field(start, ids)
    tangent_space = projector-np.outer(projector @ normal, projector @ normal)/max(normal @ projector @ normal, 1e-12)
    eigenvalues, vectors = np.linalg.eigh(tangent_space)
    tangents = vectors[:, eigenvalues > .5]
    if not tangents.shape[1]:
        return None
    if tangents.shape[1] == 1:
        directions = [tangents[:, 0], -tangents[:, 0]]
    else:
        angles = np.arange(config.geodesic_candidates)*2*np.pi/config.geodesic_candidates
        directions = [np.cos(a)*tangents[:, 0]+np.sin(a)*tangents[:, 1] for a in angles]
    # Boundary length sets propagation scale, bounded for small voxel corners.
    diameter = np.linalg.norm((scene.centers[ids]+scene.extents[ids]).max(0)
                              -(scene.centers[ids]-scene.extents[ids]).min(0))
    beta = min(config.geodesic_step_m, max(.002, diameter/40))
    best = None
    for initial in directions:
        position, direction = start.copy(), initial.copy()
        cost, path, exit_found = 0., [start.tolist()], False
        for _ in range(config.geodesic_horizon):
            _, normal = scene.field(position, ids)
            gradient = projector @ normal
            tangent = projector @ direction-gradient*(normal @ direction)/max(normal @ gradient, 1e-12)
            norm = np.linalg.norm(tangent)
            if norm < 1e-9:
                break
            direction = tangent/norm
            position += beta*direction
            # Numerical correction preserves the chosen isoline at OBB corners.
            for _ in range(6):
                distance, normal = scene.field(position, ids)
                gradient = projector @ normal
                position -= np.clip(distance-level, -beta, beta)*gradient/max(gradient @ gradient, 1e-12)
                if abs(distance-level) < 1e-5:
                    break
            cost += beta*np.linalg.norm(position-target)
            path.append(position.tolist())
            to_goal = target-position
            if (normal @ to_goal > .05*np.linalg.norm(to_goal)
                    and not scene.blocked(position, target, ids, stage['margin_m']*.99)):
                exit_found = True
                break
        # Prefer paths that expose a route to the goal. This terminal cost is
        # the configurable goal potential c(x,goal), not an output validator.
        cost += np.linalg.norm(position-target)*(diameter+beta)
        if not exit_found:
            cost += 10.*diameter
        if previous is not None and np.linalg.norm(previous) > 1e-9:
            cost += .01*diameter*(1.-initial @ previous/np.linalg.norm(previous))
        if best is None or cost < best['cost']:
            best = {'direction': initial, 'cost': float(cost), 'path': path,
                    'exit_found': exit_found, 'candidate_count': len(directions),
                    'initialization': stage['guidance_initialization'], 'isoline_distance_m': level}
    return best


class AdaptiveMCBF:
    def __init__(self, sampler, *, margin, alpha, inter_arm_margin, period,
                 optimization_space='joint', config=None):
        self.sampler, self.margin, self.alpha = sampler, margin, alpha
        self.inter_arm_margin, self.period = inter_arm_margin, period
        self.optimization_space = optimization_space
        self.config = config or AdaptiveMCBFConfig()
        self.scene = None
        self.released_action = self.release_info = None
        self.previous_direction = np.zeros(3)
        self.runtime_stats = {'qp_solve_calls': 0, 'stage_s': {}, 'stage_counts': {}}

    def plan(self, q, grip, actions, boxes, *, lower, upper, max_step, previous_delta=None):
        started = time.perf_counter()
        q, grip, actions = (np.asarray(value, float) for value in (q, grip, actions))
        if (q.shape != (12,) or grip.shape != (2,) or actions.ndim != 2
                or actions.shape[1] != 14 or not len(actions)
                or not all(np.isfinite(v).all() for v in (q, grip, actions))):
            raise ValueError('Adaptive MCBF requires finite dual-arm states and absolute actions')
        lower, upper = np.broadcast_to(lower, (12,)), np.broadcast_to(upper, (12,))
        if np.any(lower > upper) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError('Invalid adaptive MCBF joint bounds')
        action = actions[0]
        if np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1)) or max_step <= 0 or np.isnan(max_step):
            raise ValueError('Invalid gripper commands or control step')
        if self.released_action is not None:
            if np.array_equal(action, self.released_action):
                # An iteration limit releases this complete recorded action,
                # including all speed-bounded servo ticks needed to reach it.
                # The next distinct action starts adaptive avoidance again.
                delta = np.clip(action[:12]-q, -max_step, max_step)
                step = {**self.release_info['steps'][0], 'qp_solve_calls': 0}
                info = {**self.release_info, 'waypoints': [np.r_[q+delta, action[12:]].tolist()],
                        'steps': [step], 'waypoint_steps': [step], 'joint_qp_solve_calls': 0,
                        'planning_seconds': time.perf_counter()-started}
                return action[None].copy(), info
            self.released_action = self.release_info = None
        sampler, config = self.sampler, self.config
        old = self.scene
        if old is None or boxes is not old.boxes:
            self.scene = BoxScene(boxes, config, old)
        scene = self.scene
        points = np.asarray(sampler.link_points(q, grip)).reshape(-1, 3)
        commanded_grip_points = np.asarray(sampler.link_points(q, action[12:])).reshape(-1, 3)
        gripper_drift = commanded_grip_points-points
        target_points = np.asarray(sampler.link_points(action[:12], action[12:])).reshape(-1, 3)
        step_limit = min(max_step, config.max_control_step_rad) if len(boxes) else max_step
        if not np.isfinite(step_limit):
            step_limit = max(float(np.max(np.abs(action[:12]-q))), 1e-6)
        low, high = np.maximum(lower-q, -step_limit), np.minimum(upper-q, step_limit)
        nominal = np.clip(action[:12]-q, low, high)
        jacobian = point_jacobian_fd(sampler, q, gripper_position=grip).reshape(-1, 3, 12)
        # Focus on a controllable surface point whose route is obstructed.
        focus = None
        moving = ((np.linalg.norm(jacobian, axis=(1, 2)) > 1e-7)
                  & (np.linalg.norm(target_points-points, axis=1) > 1e-8))
        possible = np.flatnonzero(moving)
        if len(possible) and len(boxes):
            blocked = scene.blocked_points(points[possible], target_points[possible], self.margin)
            possible = possible[blocked]
            best_distance = np.inf
            for batch in np.array_split(possible, max(1, int(np.ceil(len(possible)/128)))):
                if len(batch):
                    distances = scene.distances(points[batch]).min(axis=1)
                    closest = int(np.argmin(distances))
                    if distances[closest] < best_distance:
                        focus, best_distance = int(batch[closest]), float(distances[closest])
        if focus is None:
            focus = int(np.argmax(np.linalg.norm(target_points-points, axis=1)))
        stage = select_stage(scene, points[focus], target_points[focus], jacobian[focus],
                             self.margin, self.alpha, config)
        target = target_points[focus] if stage['temporary_target'] is None else np.asarray(stage['temporary_target'])
        guidance = geodesic_guidance(scene, points[focus], target, jacobian[focus], stage,
                                     config, self.previous_direction)
        reference = nominal.copy()
        rows, rhs = [], []
        # Individual cluster CBFs at the measured state. Motion contributes
        # n^T v_obstacle * dt, the discrete form of paper Eq. 17.
        for ids, obstacle_velocity in zip(scene.groups, scene.velocities):
            distances = scene.distances(points, ids)
            nearest = np.argmin(distances, axis=1)
            h = distances[np.arange(len(points)), nearest]-stage['margin_m']
            active = np.flatnonzero((h <= max(config.dynamic_distance_m,
                                             np.max(np.linalg.norm(jacobian @ nominal, axis=1))))
                                    & (np.linalg.norm(jacobian, axis=(1, 2)) > 1e-9))
            # Stratify by link so fixed-base points and a dense nearest link
            # cannot consume the constraint budget of all the other links.
            per_link = sampler.points_per_link
            active = np.asarray([i for link in np.unique(active//per_link)
                                 for i in sorted(active[active//per_link == link], key=lambda i: h[i])[:8]], int)
            for i in active:
                box = boxes[ids[nearest[i]]]
                normal = box.axes @ _box_gradient((points[i]-box.center) @ box.axes, box.half_sizes)
                row = normal @ jacobian[i]
                if np.linalg.norm(row) > 1e-9:
                    rows.append(row)
                    rhs.append(np.expm1(-stage['alpha'])*h[i]
                               +normal @ (obstacle_velocity*self.period-gripper_drift[i]))
        left_count = len(sampler.left_link_names)*sampler.points_per_link
        pairs = select_inter_arm_constraints(points.reshape(-1, 1, 3), points.reshape(1, -1, 1, 3),
                                              np.arange(len(points)) < left_count,
                                              minimum_distance_m=self.inter_arm_margin,
                                              trigger_margin_m=.025)
        for pair in pairs:
            rows.append(pair.normal @ (jacobian[pair.first_point_id]-jacobian[pair.second_point_id]))
            rhs.append(np.expm1(-stage['alpha'])*pair.current_h
                       -pair.normal @ (gripper_drift[pair.first_point_id]-gripper_drift[pair.second_point_id]))
        guide_row = None
        if guidance is not None:
            direction = guidance['direction']
            self.previous_direction = direction.copy()
            guide_row = direction @ jacobian[focus]
            guide_speed = min(config.guidance_speed_m_s*self.period,
                              max(float(guide_row @ np.where(guide_row >= 0, high, low)), 0.)*.8)
            # Follow the selected geodesic as the nominal motion while the
            # single tangent inequality enforces nonzero progress (Eq. 45).
            pinv = np.linalg.pinv(jacobian[focus], rcond=1e-5)
            desired = direction*max(guide_speed, config.guidance_speed_m_s*self.period)
            reference += pinv @ (desired-jacobian[focus] @ reference)
            reference = np.clip(reference, low, high)
            rows.append(guide_row)
            rhs.append(guide_speed)
        elif stage['temporary_target'] is not None:
            direction = target-points[focus]
            direction *= min(1., config.guidance_speed_m_s*self.period/max(np.linalg.norm(direction), 1e-12))
            reference += np.linalg.pinv(jacobian[focus], rcond=1e-5) @ (direction-jacobian[focus] @ reference)
            reference = np.clip(reference, low, high)
        else:
            self.previous_direction[:] = 0.
        hessian, linear = np.eye(12), -reference
        null_delta = np.zeros(12)
        if self.optimization_space == 'ee':
            from obstacle_validation.cartesian_chunk import CartesianChunkQP
            cartesian = CartesianChunkQP(sampler, action[None, :12], q, np.zeros(12), np.ones(1))
            tcp_jacobian = cartesian.jacobian(q)
            # QP in TCP increments, with a pseudoinverse for redundant or
            # singular configurations; hard joint bounds map into the QP.
            inverse = np.linalg.pinv(tcp_jacobian, rcond=1e-5)
            null_delta = (np.eye(12)-inverse @ tcp_jacobian) @ reference
            weights = np.tile([1e4, 1e4, 1e4, 1e2, 1e2, 1e2], 2)
            hessian = np.diag(weights)+1e-6*(inverse.T @ inverse)
            linear = -weights*(tcp_jacobian @ reference)
        else:
            inverse = np.eye(12)
        matrix = np.asarray(rows).reshape(-1, 12)
        solver_message, solver_success = 'unconstrained_nominal', True
        delta = reference
        solve_calls = 0
        if len(rows) or self.optimization_space == 'ee':
            import osqp
            solver = osqp.OSQP()
            joint_matrix = np.vstack((matrix, np.eye(12)))
            offsets = joint_matrix @ null_delta
            matrix = sparse.csc_matrix(joint_matrix @ inverse)
            solver.setup(P=sparse.csc_matrix(hessian), q=linear, A=matrix,
                         l=np.r_[rhs, low]-offsets, u=np.r_[np.full(len(rhs), np.inf), high]-offsets,
                         verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
                         time_limit=.03, polishing=False)
            result = solver.solve(raise_error=False)
            solve_calls = 1
            solver_message, solver_success = str(result.info.status), result.info.status_val in (1, 2)
            # Status controls the fallback. No collision, sweep, residual or
            # objective check is performed on the optimized output.
            if solver_success:
                delta = inverse @ result.x+null_delta
            # A failed solve keeps the moving geodesic nominal reference.
            # Iteration exhaustion preserves the requested recorded-action
            # release behavior from the replay interface.
            elif solver_message.strip().lower() == 'maximum iterations reached':
                delta = np.clip(action[:12]-q, -max_step, max_step)
                self.released_action = action.copy()
        self.runtime_stats['qp_solve_calls'] += solve_calls
        counts = self.runtime_stats['stage_counts']
        counts[stage['stage']] = counts.get(stage['stage'], 0)+1
        elapsed = time.perf_counter()-started
        self.runtime_stats['stage_s']['adaptive_pipeline'] = self.runtime_stats['stage_s'].get('adaptive_pipeline', 0.)+elapsed
        waypoint = np.r_[q+delta, action[12:]]
        step = {'success': True, 'controller': 'adaptive_mcbf', 'projection_method': 'adaptive_mcbf_qp',
                'triggered': bool(rows), 'adaptive_stage': stage, 'qp_solve_calls': solve_calls,
                'solver_message': solver_message, 'solver_success': bool(solver_success),
                'fallback': None if solver_success else ('recorded_action' if solver_message.strip().lower() == 'maximum iterations reached'
                                                        else 'geodesic_reference'),
                'avoidance_bypassed': not solver_success,
                'post_optimization_checks': False, 'sweep_validation_enabled': False,
                'detour_direction': (np.zeros(12) if guide_row is None else guide_row).tolist()}
        if guidance is not None:
            step['geodesic'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in guidance.items()}
        info = {'success': True, 'controller': 'adaptive_mcbf', 'mode': 'adaptive_mcbf_feedback',
                'horizon': 1, 'optimization_space': self.optimization_space, 'gripper_variables': 0,
                'joint_variables': 12 if self.optimization_space == 'joint' else 0,
                'ee_variables': 12 if self.optimization_space == 'ee' else 0,
                'waypoints': [waypoint.tolist()], 'waypoint_steps': [step], 'steps': [step],
                'waypoint_action_indices': [0], 'triggered_action_indices': [0] if rows else [],
                'modified_action_indices': [0] if rows else [], 'target_repairs': [],
                'joint_qp_solve_calls': solve_calls, 'solver': 'adaptive_osqp',
                'planning_seconds': elapsed, 'adaptive_stage': stage,
                'avoidance_bypassed': not solver_success,
                'reason': (None if solver_success else 'joint_chunk_iteration_limit'
                           if solver_message.strip().lower() == 'maximum iterations reached'
                           else 'adaptive_qp_solver_failure'),
                'velocity_source': scene.velocity_source, 'sweep_validation_enabled': False,
                'post_optimization_checks': False, 'adaptive_feedback': True}
        if self.released_action is not None:
            self.release_info = info
        return action[None].copy(), info
