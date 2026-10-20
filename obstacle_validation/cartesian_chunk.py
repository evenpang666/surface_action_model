"""Cartesian-coordinate QP for the dual-arm replay execution prefix.

Decision variables are TCP position/rotation-vector increments in left_base.
Local inverse kinematics maps them to the joint targets consumed by servoJ.
Joint and surface constraints are transformed into the same TCP coordinates;
neither gripper commands nor their measured positions are decision variables.
"""

from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint
from scipy.spatial.transform import Rotation


class CartesianChunkQP:
    def __init__(self, sampler, reference, q, previous, counts):
        if not callable(getattr(sampler, "tcp_transforms", None)):
            raise ValueError("EE optimization requires sampler.tcp_transforms")
        from obstacle_validation.joint_chunk import _temporal_objective
        self.sampler = sampler
        self.origin = np.asarray(sampler.tcp_transforms(q), dtype=float)
        if self.origin.shape != (2, 4, 4) or not np.isfinite(self.origin).all():
            raise ValueError("EE optimization requires two finite active TCP poses")
        self.horizon = len(reference)
        # Normalize tracking errors: 1 cm translation and 0.1 rad rotation
        # have equal cost. Temporal regularization uses these same units.
        self.scale = np.tile([100., 100., 100., 10., 10., 10.], 2*self.horizon)
        targets = np.asarray([self.coordinates(pose) for pose in reference])
        previous_tcp_delta = -self.coordinates(q-previous)
        normalized_reference = (targets.ravel()*self.scale).reshape(self.horizon, 12)
        hessian, linear, *_ = _temporal_objective(
            normalized_reference, np.zeros(12), previous_tcp_delta*self.scale[:12], counts)
        self.hessian = self.scale[:, None]*hessian*self.scale[None, :]
        self.linear = linear*self.scale

    def coordinates(self, q):
        poses = np.asarray(self.sampler.tcp_transforms(q), dtype=float)
        if poses.shape != self.origin.shape or not np.isfinite(poses).all():
            raise ValueError("Invalid TCP transforms")
        result = []
        for pose, origin in zip(poses, self.origin):
            rotation = Rotation.from_matrix(pose[:3, :3] @ origin[:3, :3].T).as_rotvec()
            # A rotation-vector chart is discontinuous at pi. Do not silently
            # map across its cut or send an arbitrary IK branch to the robot.
            if np.linalg.norm(rotation) >= np.pi-.05:
                raise ValueError("EE rotation exceeds the local rotation chart")
            result.extend(pose[:3, 3]-origin[:3, 3])
            result.extend(rotation)
        return np.asarray(result)

    def jacobian(self, q):
        q = np.asarray(q, dtype=float)
        jacobian = np.empty((12, 12))
        epsilon = 1e-5
        for joint in range(12):
            plus, minus = q.copy(), q.copy()
            plus[joint] += epsilon
            minus[joint] -= epsilon
            jacobian[:, joint] = (self.coordinates(plus)-self.coordinates(minus))/(2*epsilon)
        return jacobian

    def objective(self, joints):
        coordinates = np.concatenate([self.coordinates(q) for q in joints])
        return float(.5*coordinates @ self.hessian @ coordinates+self.linear @ coordinates)

    def solve(self, current, low, high, constraints, iterations):
        from obstacle_validation.joint_chunk import _solve_sparse_qp
        coordinates, inverse_blocks = [], []
        minimum_singular = np.inf
        for pose in current:
            coordinates.append(self.coordinates(pose))
            jacobian = self.jacobian(pose)
            # The arms are independent in FK, expressed in one common frame.
            # Refuse singular IK instead of amplifying a small TCP increment
            # into an uncontrolled joint step.
            blocks = []
            for start in (0, 6):
                block = jacobian[start:start+6, start:start+6]
                smallest = float(np.linalg.svd(block, compute_uv=False)[-1])
                minimum_singular = min(minimum_singular, smallest)
                if smallest < 1e-5:
                    return SimpleNamespace(x=current.ravel().copy(), success=False, nit=0,
                                           message="ee_inverse_kinematics_singular",
                                           minimum_tcp_jacobian_singular_value=minimum_singular)
                blocks.append(np.linalg.solve(block, np.eye(6)))
            inverse_blocks.append(block_diag(*blocks))
        inverse = block_diag(*inverse_blocks)
        center = current.ravel()
        mapped = [LinearConstraint(c.A @ inverse, c.lb-c.A @ center, c.ub-c.A @ center)
                  for c in constraints]
        # Hard joint/tracking/trust bounds constrain the recovered IK target,
        # rather than clipping it after the Cartesian solve.
        mapped.append(LinearConstraint(inverse, low.ravel()-center, high.ravel()-center))
        increment_limit = np.tile([.02, .02, .02, .1, .1, .1], 2*self.horizon)
        result = _solve_sparse_qp(
            self.hessian, self.hessian @ np.concatenate(coordinates)+self.linear,
            np.zeros_like(current), -increment_limit.reshape(current.shape),
            increment_limit.reshape(current.shape), mapped, iterations)
        result.x = center+inverse @ result.x
        result.minimum_tcp_jacobian_singular_value = minimum_singular
        return result
