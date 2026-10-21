"""Continuous volume geometry for replay. All separation values are in metres.

S is the squared-axis matrix: E={c + sqrt(S) u : ||u|| <= 1}.
The separating-plane maximization includes both bodies' rotations and works
inside overlap too. No robot surface sampling is used for online constraints.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class Ellipsoid:
    center: np.ndarray
    shape: np.ndarray
    key: str = ""
    arm: int = -1
    link: int = -1

    @property
    def radius(self):
        return float(np.sqrt(np.linalg.eigvalsh(self.shape)[-1]))

    def transformed(self, pose, *, key=None, arm=None, link=None):
        r = pose[:3, :3]
        return Ellipsoid(r @ self.center + pose[:3, 3], r @ self.shape @ r.T,
                         self.key if key is None else key,
                         self.arm if arm is None else arm, self.link if link is None else link)


def fit_enclosing(points, *, min_axis=.001, key=""):
    """PCA orientation, optimized axis lengths, verified coverage of all vertices."""
    points = np.unique(np.asarray(points, float).reshape(-1, 3), axis=0)
    if not len(points) or not np.isfinite(points).all() or min_axis <= 0:
        raise ValueError("Ellipsoid fit requires finite nonempty geometry")
    center = (points.min(0)+points.max(0))*.5
    if len(points) > 2:
        _, axes = np.linalg.eigh(np.cov(points.T))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
    else:
        axes = np.eye(3)
    local = (points-center) @ axes
    shift = (local.min(0)+local.max(0))*.5
    center += axes @ shift
    local -= shift
    half = np.maximum(np.abs(local).max(0), min_axis)
    # Diagonal inverse squared axes in normalized PCA coordinates. This convex
    # problem minimizes log volume while enclosing every supplied vertex.
    squared = (local/half)**2
    result = minimize(lambda w: -np.log(w).sum(), np.full(3, 1/3),
                      jac=lambda w: -1/w, method="SLSQP",
                      bounds=[(1e-5, 1.)]*3,
                      constraints=[{"type": "ineq", "fun": lambda w: 1-squared @ w,
                                    "jac": lambda w: -squared}],
                      options={"maxiter": 50, "ftol": 1e-8})
    w = result.x if result.success else np.full(3, 1/3)
    lengths = half/np.sqrt(w)
    # Floating point or an optimizer tolerance may slightly under-enclose.
    lengths *= max(1., float(np.sqrt(((local/lengths)**2).sum(1).max())))*(1+1e-7)
    return Ellipsoid(center, (axes*lengths**2) @ axes.T, key)


def plane_gap(a, b, normal):
    n = np.asarray(normal, float)
    n = n/max(np.linalg.norm(n), 1e-15)
    sa, sb = a.shape @ n, b.shape @ n
    ra, rb = np.sqrt(n @ sa), np.sqrt(n @ sb)
    pa, pb = a.center-sa/ra, b.center+sb/rb
    return float(n @ (pa-pb)), n, pa, pb


@dataclass
class VLSABarrier:
    value: float
    center_delta: np.ndarray
    center_distance: float
    normal: np.ndarray
    support_radii: np.ndarray
    tangent_basis: np.ndarray
    direction_gradient: np.ndarray


def vlsa_barrier(a, b, normal):
    """VLSA Sec. 4.3 Eq. (9), parameterized by its unit plane normal.

    S stores squared semi-axes, so h = n.(ca-cb)-sqrt(n.Sa.n)-sqrt(n.Sb.n).
    This uses ellipsoid CENTRES, including their link/tool frame offsets.
    For the paper's A=sqrt(Sa) and virtual point ps, n=-A^-1 ps/|A^-1 ps|;
    the inverse ps=-A n/|A n| is a smooth reparameterization of the sphere.
    The two tangent coordinates are auxiliary controls, never robot commands.
    A positive h certifies a separating plane; |ca-cb| alone cannot do so.
    """
    n = np.asarray(normal, float)
    length = np.linalg.norm(n)
    if n.shape != (3,) or not np.isfinite(n).all() or length < 1e-12:
        raise ValueError("VLSA virtual direction must be a finite nonzero 3-vector")
    n = n/length
    delta = a.center-b.center
    sa, sb = a.shape @ n, b.shape @ n
    radii = np.sqrt([n @ sa, n @ sb])
    reference = np.eye(3)[np.argmin(np.abs(n))]
    first = np.cross(n, reference); first /= np.linalg.norm(first)
    basis = np.column_stack((first, np.cross(n, first)))
    derivative = delta-sa/radii[0]-sb/radii[1]
    return VLSABarrier(float(n @ delta-radii.sum()), delta, float(np.linalg.norm(delta)),
                       n, radii, basis, derivative @ basis)


def separation(a, b, previous=None):
    """Optimize signed support-plane gap; warm-start and retain best solution.

    Positive optimum equals Euclidean separation. Negative values provide an
    overlap escape direction. A finite suboptimal plane is a lower bound,
    never an optimistic positive clearance. Multiple seeds resolve overlap.
    """
    delta = a.center-b.center
    norm = np.linalg.norm(delta)
    center_seed = delta/norm if norm > 1e-10 else np.array([1., 0., 0.])
    seeds = [center_seed]
    if previous is not None and np.isfinite(previous).all() and np.linalg.norm(previous) > .5:
        seeds.insert(0, np.asarray(previous)/np.linalg.norm(previous))
    def value(n):
        sa, sb = a.shape @ n, b.shape @ n
        ra, rb = np.sqrt(max(n @ sa, 1e-24)), np.sqrt(max(n @ sb, 1e-24))
        return -float(delta @ n-ra-rb), -delta+sa/ra+sb/rb
    best = None
    # Near coincident centres require axis seeds to avoid the wrong stationary
    # point of a long, thin body's penetration surface.
    if norm < .25*(a.radius+b.radius):
        axes = np.linalg.eigh(a.shape+b.shape)[1]
        seeds.extend([sign*axes[:, i] for i in range(3) for sign in (-1, 1)])
    for seed in seeds:
        result = minimize(value, seed, jac=True, method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda n: n @ n-1,
                                        "jac": lambda n: 2*n}],
                          options={"maxiter": 30, "ftol": 1e-10})
        for n in (seed, result.x):
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue
            candidate = plane_gap(a, b, n)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best[0] > 0 and result.success:
            break
    return best


def separation_gradient(a, normal, center_jacobian, shape_jacobian):
    """Envelope derivative, including orientation; derivatives end in joint axis."""
    n = normal
    radius = np.sqrt(n @ a.shape @ n)
    return n @ center_jacobian - .5*np.einsum("i,ijk,j->k", n, shape_jacobian, n)/radius


def box_groups(boxes, gap=.005):
    if not boxes:
        return []
    centers = np.array([b.center for b in boxes])
    extents = np.array([np.abs(b.axes) @ b.half_sizes for b in boxes])
    radii = np.linalg.norm(extents, axis=1)
    tree, roots = cKDTree(centers), np.arange(len(boxes))
    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    for i, c in enumerate(centers):
        for j in tree.query_ball_point(c, radii[i]+radii.max()+gap):
            if j > i and np.linalg.norm(np.maximum(np.abs(c-centers[j])-extents[i]-extents[j], 0)) <= gap:
                roots[root(j)] = root(i)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def fit_obstacles(boxes, *, max_parts=8, max_span=.25):
    """Preserve direct ellipsoids and partition occupied cuboids.

    Enclose all eight corners of every measured cuboid. Splits never cut a
    cuboid. Concave clusters retain local components instead of a convex hull.
    """
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    # Paired RGB volumes already enclose their entire frustum intersection.
    # Keep each one as a separate obstacle; refitting its OBB would inflate it
    # again and merging it with measured cuboids would lose object identity.
    output = [Ellipsoid(np.asarray(box.center, float), np.asarray(box.ellipsoid_shape, float),
                        key=f"transparent_{getattr(box, 'object_id', index)}")
              for index, box in enumerate(boxes) if getattr(box, "ellipsoid_shape", None) is not None]
    boxes = [box for box in boxes if getattr(box, "ellipsoid_shape", None) is None]
    for cluster, ids in enumerate(box_groups(boxes)):
        parts = [ids]
        fitted = {}
        def fit_part(part):
            key = tuple(part)
            if key not in fitted:
                vertices = np.concatenate([boxes[i].center+(signs*boxes[i].half_sizes) @ boxes[i].axes.T for i in part])
                fitted[key] = fit_enclosing(vertices)
            return fitted[key]
        while len(parts) < max_parts:
            scores = []
            for part in parts:
                if len(part) == 1:
                    scores.append(0.)
                    continue
                span = np.ptp(np.array([boxes[i].center for i in part]), axis=0).max()
                volume = 4*np.pi/3*np.sqrt(np.linalg.det(fit_part(part).shape))
                occupied = sum(8*np.prod(boxes[i].half_sizes) for i in part)
                scores.append(max(span/max_span, volume/max(occupied*3.2, 1e-12)))
            index = int(np.argmax(scores))
            part = parts[index]
            if scores[index] <= 1:
                break
            centers = np.array([boxes[i].center for i in part])
            axis = int(np.argmax(np.ptp(centers, axis=0)))
            order = np.argsort(centers[:, axis], kind="stable")
            middle = len(order)//2
            parts[index:index+1] = [[part[i] for i in order[:middle]], [part[i] for i in order[middle:]]]
        for part in parts:
            model = fit_part(part)
            model.key = f"obstacle_{cluster}_{len(output)}"
            output.append(model)
    return output


@lru_cache(maxsize=1)
def _local_robot_models():
    """One complete collision envelope per rigid UR link, including the base."""
    from real_scripts.ur7e_collision_mesh import load_collision_meshes
    return [(link, name, fit_enclosing(np.asarray(mesh.vertices), key=name))
            for link, (name, mesh) in enumerate(load_collision_meshes().items())]


@lru_cache(maxsize=1)
def _gripper_mesh_vertices():
    import trimesh
    from real_scripts.robotiq_vendor_visual_models import TWO_F85_COLLISION, TWO_F85_MESH_NAMES
    return [(name, np.asarray(trimesh.load_mesh(TWO_F85_COLLISION/name, process=False).vertices))
            for name in TWO_F85_MESH_NAMES]


@lru_cache(maxsize=128)
def _local_gripper_model(closing_fraction):
    """Enclose all articulated gripper meshes together at the exact opening.

    Cache in the gripper root frame: joint FK/differencing never refits it.
    Openings are not quantized, so the envelope covers the commanded geometry.
    """
    from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_link_transforms
    poses = robotiq_2f85_link_transforms(float(closing_fraction)*.8)
    vertices = np.concatenate([points @ poses[name][:3, :3].T+poses[name][:3, 3]
                               for name, points in _gripper_mesh_vertices()])
    return fit_enclosing(vertices, key="gripper")


class RobotEllipsoids:
    def __init__(self, sampler):
        self.sampler = sampler
        self.native = hasattr(sampler, "flange_to_tcp") and hasattr(sampler, "right_base_to_left_base")
        self.models = _local_robot_models() if self.native else None

    def bodies(self, q, grip):
        if not self.native:
            # Analytic/test adapters can supply full volume geometry directly.
            if hasattr(self.sampler, "ellipsoids"):
                return self.sampler.ellipsoids(q, grip)
            return [fit_enclosing(p, key=f"link_{i}", min_axis=.0001)
                    for i, p in enumerate(self.sampler.link_points(q, grip))]
        from real_scripts.ur7e_collision_mesh import link_and_collision_transforms, flange_transform
        output = []
        offset = np.eye(4); offset[2, 3] = -.1493
        for arm, name in enumerate(("left_arm", "right_arm")):
            base = np.eye(4) if arm == 0 else self.sampler.right_base_to_left_base
            joints = q[6*arm:6*arm+6]
            poses = link_and_collision_transforms(joints)
            tcp = flange_transform(joints) @ self.sampler.flange_to_tcp[name]
            for link, mesh, model in self.models:
                output.append(model.transformed(base @ poses[mesh], key=f"{arm}_{model.key}", arm=arm, link=link))
            gripper = _local_gripper_model(float(grip[arm]))
            output.append(gripper.transformed(base @ tcp @ offset, key=f"{arm}_gripper", arm=arm,
                                              link=len(self.models)))
        return output

    def differential(self, q, grip):
        bodies = self.bodies(q, grip)
        centers = np.array([b.center for b in bodies])
        shapes = np.array([b.shape for b in bodies])
        dc, ds = np.empty((len(bodies), 3, 12)), np.empty((len(bodies), 3, 3, 12))
        eps = 1e-5
        for k in range(12):
            moved = q.copy(); moved[k] += eps
            other = self.bodies(moved, grip)
            dc[..., k] = (np.array([b.center for b in other])-centers)/eps
            ds[..., k] = (np.array([b.shape for b in other])-shapes)/eps
        if not self.native:
            left = len(self.sampler.left_link_names)
            for i, body in enumerate(bodies):
                body.arm, body.link = int(i >= left), i if i < left else i-left
        return bodies, dc, ds

    @staticmethod
    def inter_arm_collision_pair(a, b):
        """Only left/right pairs; every same-arm pair is intentionally excluded.

        Single-link envelopes may overlap within a physically clear arm.
        They model environment and opposite-arm avoidance, not self avoidance.
        """
        return (a.arm, b.arm) in ((0, 1), (1, 0))
