"""Run VLSA / VLPSA / OA-VLA independently through the same TCP and simulator.

This is a shared-perception, seven-joint deployment adaptation, not a claim of
reproducing the authors' SafeLIBERO scores or original perception stacks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .ellipsoid_geometry import Ellipsoid, fit_enclosing
from .libero_comparison_geometry import JointPandaGeometry, LabelDepthPerception, label_geoms
from .libero_metrics import export_comparison
from .libero_safety import ConstraintMonitor, SUITES, first_action, prepare_paths, write_json
from .libero_safety_geometry import CAMERAS, capture
from .oa_vla_actions import ObservationHold
from .safety_filters import PoissonField, VLSAFilter, VLPSAFilter, obstacle_grid

METHODS = ("vlsa", "vlpsa", "oavla")


class ContactMonitor(ConstraintMonitor):
    """Latch prohibited contacts and official predicates at every physics step."""
    def __init__(self, env, obstacles, grasp_objects):
        super().__init__(env)
        self.obstacles = set().union(*label_geoms(env, obstacles).values())
        self.robot = {i for i in range(env.sim.model.ngeom)
            if (env.sim.model.geom_id2name(i) or "").startswith(("robot0_", "gripper0_"))}
        self.grasp_geoms = label_geoms(env, grasp_objects)
        self.held = set()
        self.contacts = {}

    def sample(self):
        super().sample()
        protected = self.robot | set().union(*(self.grasp_geoms[n] for n in self.held))
        sim = self.base.sim
        for contact in sim.data.contact[:sim.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0:
                continue
            if (a in self.obstacles and b in protected) or (b in self.obstacles and a in protected):
                key = tuple(sorted((a, b)))
                self.contacts.setdefault(key, dict(geoms=[sim.model.geom_id2name(g) for g in key],
                                                   substep=self.substeps, distance_m=float(contact.dist)))


class OAVLAFilter:
    """Existing full-body EllipsoidMCBF, with no VLSA/VLPSA dispatch inside it."""
    def __init__(self, geometry, margin, dt):
        from .ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
        self.planner = EllipsoidMCBF(None, margin=margin, inter_arm_margin=0., period=dt,
            robot=geometry, joint_groups=(7,), grip_dim=1,
            config=EllipsoidMCBFConfig(max_control_step_rad=1.))

    def filter(self, reference, geometry, obstacles, lower, upper, dt, *, boxes, timestamp, **unused):
        # Preserve predicted/action-selected geometry in the shared input path.
        # A fresh list causes the existing scene updater to recompute geometry.
        from safety_module.occupancy import ObservedBoxes
        scene = ObservedBoxes(list(boxes), None)
        scene.source_snapshot = {"motion_timestamp_ns": int(timestamp*1e9)+1}
        target = np.r_[reference, geometry.grip]
        try:
            _, info = self.planner.plan(np.zeros(7), geometry.grip, target[None], scene,
                lower=lower, upper=upper, max_step=float(max(np.abs(lower).max(), np.abs(upper).max())))
        except RuntimeError as exc:
            if str(exc) == "Ellipsoid optimization cannot preserve forward goal progress":
                raise ObservationHold(str(exc)) from exc
            raise
        detail = info["steps"][0]
        if (not info["success"] or detail["max_distance_slack_m"] > 1e-5
                or detail["minimum_predicted_barrier_after_m"] < -1e-5):
            raise ObservationHold("OA-VLA MCBF rejected finite control or safety slack")
        return np.asarray(info["waypoints"][0][:7]), dict(method="oavla", optimizer=info)


def validate_socket(metadata, *, allow_test=False):
    if (metadata.get("state_dim") != 8 or metadata.get("action_dim") != 7
            or metadata.get("action_semantics") != "libero_osc_pose"
            or metadata.get("state_semantics") != "libero_eef_axis_angle_gripper"
            or metadata.get("image_convention") != "opencv_top_left"
            or metadata.get("camera_names") != CAMERAS
            or metadata.get("policy_views") != list(CAMERAS)
            or metadata.get("action_period_s") != .05):
        raise ValueError("Socket must declare the four-view Panda 8-state/7-action LIBERO contract")
    if metadata.get("test_only", True) and not allow_test:
        raise ValueError("Untrained/unspecified policy needs --allow-test-policy; never paper eligible")


def ellipsoids(boxes):
    result = []
    for box in boxes:
        if box.ellipsoid_shape is not None:
            result.append(Ellipsoid(box.center, box.ellipsoid_shape, box.object_id))
        else:
            # Enclose all eight corners, rather than use the inscribed ellipsoid.
            from itertools import product
            points = np.array(list(product((-1, 1), repeat=3)))*box.half_sizes
            result.append(fit_enclosing(points @ box.axes.T+box.center, key=box.object_id))
    return result


def make_fields(boxes, args, fine_center):
    bounds = np.asarray(args.psf_bounds).reshape(3, 2)
    occupied, origin = obstacle_grid(boxes, bounds, args.psf_resolution, max_cells=args.max_grid_cells)
    buffer = args.margin+args.sample_spacing/np.sqrt(2)
    fields = [PoissonField.solve(occupied, origin, args.psf_resolution, buffer=buffer)]
    if args.psf_fine_resolution:
        fine = np.c_[fine_center-args.psf_fine_extent/2, fine_center+args.psf_fine_extent/2]
        fine[:, 0], fine[:, 1] = np.maximum(fine[:, 0], bounds[:, 0]), np.minimum(fine[:, 1], bounds[:, 1])
        occupied, origin = obstacle_grid(boxes, fine, args.psf_fine_resolution, max_cells=args.max_grid_cells)
        fields.append(PoissonField.solve(occupied, origin, args.psf_fine_resolution, buffer=buffer))
    return fields


def run_episode(env, policy, args, method, scenario, episode, initial_state, output):
    env.seed(args.seed)
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    velocity_controller = env.robots[0].controller
    velocity_controller.output_max = np.full(7, args.max_joint_velocity)
    velocity_controller.output_min = np.full(7, -args.max_joint_velocity)
    velocity_controller.action_scale = None
    geometry = JointPandaGeometry(env, spacing=args.sample_spacing)
    geometry.bind()
    monitor = ContactMonitor(env, scenario["obstacles"], scenario.get("grasp_objects", []))
    perception = LabelDepthPerception(env, CAMERAS, scenario["obstacles"]+scenario.get("grasp_objects", []),
                                       padding=args.depth_padding, max_age=args.geometry_max_age)
    controller = {"vlsa": lambda: VLSAFilter(margin=args.margin, alpha=args.cbf_alpha),
                  "vlpsa": lambda: VLPSAFilter(alpha=args.cbf_alpha),
                  "oavla": lambda: OAVLAFilter(geometry, args.margin, 1/args.control_freq)}[method]()
    dt = 1/args.control_freq
    policy_stride = args.control_freq//20
    output.mkdir(parents=True, exist_ok=False)
    report = dict(method=method, benchmark="Panda-Lift-fixture" if args.fixture else "LIBERO-Safety",
        suite=scenario["suite"], level=scenario["level"], task_id=scenario["task"], episode=episode, seed=args.seed,
        completed=False, success=False, safe_success=False, collision=False, safety_violation=False,
        steps=0, control_steps=0, policy_calls=0, holds=0, interventions=0, warmup_steps=args.warmup_steps,
        termination_reason=None, test_only=bool(policy.metadata.get("test_only", True) or args.fixture),
        control_freq=args.control_freq, state_hash=hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest())
    video = None
    began = time.monotonic()
    fields, field_signature = None, None
    try:
        if args.video:
            import imageio.v2 as imageio
            video = imageio.get_writer(output/"rollout.mp4", fps=20)
        with (output/"steps.jsonl").open("w") as log, monitor.track():
            monitor.sample()  # include forbidden contacts in the initial state
            for _ in range(args.warmup_steps):
                obs, _, done, _ = env.step(np.r_[np.zeros(7), -1.])
                report["steps"] += 1
                monitor.sample()
            nominal, boxes_by_name = None, {}
            for tick in range(args.max_steps):
                start = time.monotonic()
                record = dict(tick=tick, method=method, status="executed")
                geometry.bind()
                try:
                    if tick % policy_stride == 0 or nominal is None:
                        bundle, calibrations = capture(env, obs, CAMERAS, tick)
                        if video:
                            video.append_data(bundle.frames["front"].rgb)
                        prediction = policy.infer({k: f.rgb for k, f in bundle.frames.items()}, bundle.feedback.q,
                            scenario["language"], context=bundle.context(), encoding=args.encoding)
                        report["policy_calls"] += 1
                        record["prediction_actions"] = prediction.get("actions")
                        nominal = first_action(prediction)
                        geometry.set_reference(nominal, .05)
                        target = (geometry.target_position.copy(), geometry.target_rotation.copy())
                        boxes_by_name = perception.update(bundle, calibrations, retained_names=geometry.held)
                    else:
                        geometry.target_position, geometry.target_rotation = target
                    geometry.update_grasp(nominal[-1], boxes_by_name, monitor.grasp_geoms)
                    monitor.held = set(geometry.held)
                    boxes = [boxes_by_name[n] for n in scenario["obstacles"] if n not in geometry.held]
                    reference = geometry.nominal(dt)
                    low, high = geometry.control_bounds(dt, args.max_joint_velocity)
                    if method == "vlpsa":
                        signature = tuple(np.r_[b.center, b.half_sizes].tobytes() for b in boxes)
                        if fields is None or signature != field_signature:
                            fields = make_fields(boxes, args, geometry.eef_pose()[0])
                            field_signature = signature
                    delta, diagnostics = controller.filter(reference, geometry, ellipsoids(boxes), low, high, dt,
                        boxes=boxes, fields=fields, timestamp=float(env.sim.data.time))
                    if time.monotonic()-start > args.max_result_age:
                        raise ObservationHold("Socket/perception/filter result expired")
                    action = geometry.velocity_action(delta, dt, nominal[-1])
                    report["interventions"] += int(not np.allclose(delta, reference, atol=1e-6))
                    record.update(nominal_joint_increment=reference, joint_increment=delta, diagnostics=diagnostics)
                except ObservationHold as exc:
                    action = np.zeros(8)
                    record.update(status="hold", reason=str(exc))
                    report["holds"] += 1
                    # Failed perception must be retried rather than reusing a
                    # partially refreshed action target on the next sub-tick.
                    nominal = None
                obs, _, _, _ = env.step(action)
                # A simulator horizon's `done` is not a goal success signal.
                done = bool(env.env._check_success())
                monitor.sample()
                report["steps"] += 1; report["control_steps"] += 1
                record.update(action=action, goal_success=bool(done), collision=bool(monitor.contacts),
                              safety_violation=bool(monitor.violations), latency_s=time.monotonic()-start)
                from .libero_safety import json_value
                log.write(json.dumps(json_value(record), allow_nan=False)+"\n"); log.flush()
                # Continue after collisions so SR can measure unsafe successes.
                if done:
                    report.update(success=True, termination_reason="success")
                    break
            else:
                report["termination_reason"] = "timeout"
            report.update(completed=True)
    except Exception as exc:
        report.update(termination_reason="error", error=f"{type(exc).__name__}: {exc}")
        import traceback
        (output/"error.txt").write_text(traceback.format_exc())
    finally:
        if video:
            video.close()
        report.update(collision=bool(monitor.contacts), safety_violation=bool(monitor.violations),
            safe_success=bool(report["success"] and not monitor.contacts and not monitor.violations),
            contacts=list(monitor.contacts.values()), violations=monitor.violations,
            physics_samples=monitor.substeps, elapsed_s=time.monotonic()-began)
        write_json(output/"report.json", report)
    return report


def fixture_env(args):
    import robosuite
    from robosuite.environments.manipulation.lift import Lift
    from robosuite.models.objects import BoxObject
    class SafetyLift(Lift):
        def _load_model(self):
            super()._load_model()
            obstacle = BoxObject("safety_obstacle", size=[.035, .035, .06], joints=None,
                                 rgba=[.9, .1, .1, 1])
            body = obstacle.get_obj()
            body.set("pos", ".15 .12 1.0")
            self.model.worldbody.append(body)
            self.model.merge_assets(obstacle)
    controller = robosuite.load_controller_config(default_controller="JOINT_VELOCITY")
    controller.update(output_max=args.max_joint_velocity, output_min=-args.max_joint_velocity)
    base = SafetyLift(robots="Panda", controller_configs=controller, control_freq=args.control_freq,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=list(CAMERAS.values()), camera_heights=args.resolution, camera_widths=args.resolution,
        camera_depths=True, horizon=args.max_steps+args.warmup_steps+1, ignore_done=True)
    base.parsed_problem = {"constraints": []}
    class Adapter:
        env = base
        @property
        def sim(self): return base.sim
        @property
        def robots(self): return base.robots
        def reset(self): return base.reset()
        def seed(self, seed): np.random.seed(seed)
        def step(self, action):
            obs, reward, _, info = base.step(action)
            return obs, reward, base._check_success(), info
        def set_init_state(self, state):
            base.sim.set_state_from_flattened(state); base.sim.forward()
            base._update_observables(force=True)
            return base._get_observations(force_update=True)
        def close(self): base.close()
    return Adapter()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=(*METHODS, "all"), default="all")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--endpoints", type=Path, help='JSON {"vlsa":{"host":...,"port":...}, ...}')
    p.add_argument("--manifest", type=Path, help="JSON list of {suite,level,task,episodes,obstacles,grasp_objects}")
    p.add_argument("--libero-root", type=Path)
    p.add_argument("--config-dir", type=Path, default=Path("outputs/libero_comparison_config"))
    p.add_argument("--fixture", action="store_true", help="Panda/Lift smoke ONLY, never a LIBERO-Safety score")
    p.add_argument("--suite", choices=SUITES, default="obstacle_avoidance")
    p.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--task", type=int, choices=range(5), default=0)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--obstacles", nargs="+", default=[])
    p.add_argument("--grasp-objects", nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-test-policy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-freq", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-joint-velocity", type=float, default=.5)
    p.add_argument("--margin", type=float, default=.01)
    p.add_argument("--cbf-alpha", type=float, default=10.)
    p.add_argument("--sample-spacing", type=float, default=.04)
    p.add_argument("--depth-padding", type=float, default=.015)
    p.add_argument("--geometry-max-age", type=float, default=.5)
    p.add_argument("--psf-resolution", type=float, default=.06)
    p.add_argument("--psf-fine-resolution", type=float, default=.02, help="0 disables the fine grid")
    p.add_argument("--psf-fine-extent", type=float, default=.5)
    p.add_argument("--psf-bounds", type=float, nargs=6, default=[-1.2, 1.2, -1.2, 1.2, -.2, 2.])
    p.add_argument("--max-grid-cells", type=int, default=2_000_000)
    p.add_argument("--max-result-age", type=float, default=10.)
    p.add_argument("--socket-timeout", type=float, default=30.)
    p.add_argument("--encoding", choices=("raw", "png", "jpeg"), default="raw")
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    if args.control_freq < 20 or args.control_freq % 20:
        p.error("--control-freq must be a positive multiple of the policy's 20 Hz")
    for key in ("max_steps", "resolution", "max_joint_velocity", "cbf_alpha", "sample_spacing", "psf_resolution",
                "psf_fine_extent", "max_grid_cells", "geometry_max_age", "max_result_age", "socket_timeout"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    for key in ("warmup_steps", "margin", "depth_padding", "psf_fine_resolution"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f"{key} must be finite and nonnegative")
    bounds = np.asarray(args.psf_bounds).reshape(3, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        p.error("Invalid PSF workspace")
    if args.psf_fine_resolution and args.psf_fine_resolution >= args.psf_resolution:
        p.error("Fine PSF resolution must be smaller than coarse resolution")
    if not args.fixture and not args.libero_root:
        p.error("--libero-root is required for benchmark evaluation")
    return args


def main(argv=None):
    args = parse_args(argv)
    from oa_vla.socket_protocol import PolicyClient
    import robosuite.macros as macros
    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Expected robosuite OpenGL image convention")
    scenarios = json.loads(args.manifest.read_text()) if args.manifest else [dict(suite=args.suite,
        level=args.level, task=args.task, episodes=args.episodes,
        obstacles=["safety_obstacle"] if args.fixture else args.obstacles, grasp_objects=args.grasp_objects)]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Manifest must be a nonempty scenario list")
    identities = []
    for s in scenarios:
        if (s["suite"] not in SUITES or s["level"] not in (0, 1, 2) or s["task"] not in range(5)
                or not s["obstacles"] or not s["episodes"] or any(type(e) is not int or e < 0 for e in s["episodes"])
                or len(set(s["obstacles"])) != len(s["obstacles"])
                or set(s["obstacles"]) & set(s.get("grasp_objects", []))):
            raise ValueError("Invalid scenario/obstacle/grasp-object configuration")
        identities.extend((s["suite"], s["level"], s["task"], e) for e in s["episodes"])
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate manifest episode identity")
    if args.fixture and (len(scenarios) != 1 or not args.allow_test_policy):
        raise ValueError("Fixture requires --allow-test-policy and a single scenario")
    if not args.fixture:
        prepare_paths(args.libero_root, args.config_dir)
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    methods = METHODS if args.method == "all" else (args.method,)
    endpoints = json.loads(args.endpoints.read_text()) if args.endpoints else {}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"run_config.json", dict(arguments=vars(args), scenarios=scenarios,
        protocol="shared_visible_segmentation_rgbd_joint_velocity_v1",
        method_implementations={"vlsa": "eef_ellipsoid_auxiliary_plane_qp", "vlpsa": "poisson_fullbody_grasp_boolean_cbf_clf_qp",
                                "oavla": "existing_ellipsoid_mcbf"},
        policy_rate_hz=20, steps_definition="control ticks including warmup; success-only primary mean",
        collision_definition="nonpositive-distance MuJoCo robot/held-object contact with selected forbidden objects, every substep",
        continuation="continue after safety violations to measure unsafe goal successes",
        adaptation="Shared simulator label/depth perception; all methods use identical 7-DOF joint velocity tracking"))
    reports = {m: [] for m in methods}
    expected = len(identities)
    fixture_states = {}
    try:
        for method in methods:
            endpoint = endpoints.get(method, dict(host=args.host, port=args.port))
            policy = PolicyClient(endpoint["host"], endpoint["port"], timeout=args.socket_timeout)
            try:
                validate_socket(policy.metadata, allow_test=args.allow_test_policy)
                write_json(args.output/f"policy_{method}.json", policy.metadata)
                for scenario in scenarios:
                    scenario = dict(scenario)
                    if args.fixture:
                        scenario["language"] = "lift the cube without touching the red obstacle"
                        env = fixture_env(args)
                        states = None
                    else:
                        suite = benchmark.get_benchmark(scenario["suite"])()
                        task = suite.get_task_by_level_id(scenario["level"], scenario["task"])
                        scenario["language"] = task.language
                        states = suite.get_task_init_states_by_level_id(scenario["level"], scenario["task"])
                        if max(scenario["episodes"]) >= len(states):
                            raise ValueError("Requested episode is absent from official initial states")
                        env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path_by_level_id(
                            scenario["level"], scenario["task"])), controller="JOINT_VELOCITY", control_freq=args.control_freq,
                            camera_names=list(CAMERAS.values()), camera_heights=args.resolution,
                            camera_widths=args.resolution, camera_depths=True,
                            horizon=args.max_steps+args.warmup_steps+1, ignore_done=True)
                        env.robots[0].controller.output_max = np.full(7, args.max_joint_velocity)
                        env.robots[0].controller.output_min = np.full(7, -args.max_joint_velocity)
                    try:
                        for episode in scenario["episodes"]:
                            if args.fixture and episode not in fixture_states:
                                env.seed(args.seed+episode); env.reset()
                                fixture_states[episode] = env.sim.get_state().flatten().copy()
                            state = fixture_states[episode] if args.fixture else states[episode]
                            path = args.output/method/f"{scenario['suite']}_L{scenario['level']}_T{scenario['task']}"/f"episode_{episode:04d}"
                            report = run_episode(env, policy, args, method, scenario, episode, state, path)
                            reports[method].append(report)
                            print(json.dumps({k: report[k] for k in ("method", "episode", "termination_reason", "steps", "holds", "collision")}), flush=True)
                    finally:
                        env.close()
            finally:
                policy.close()
    finally:
        summaries = export_comparison(args.output, reports, expected=expected)
        for method, records in reports.items():
            write_json(args.output/f"summary_{method}.json", dict(**summaries[method], episodes=records))
    if not all(v["all_complete"] for v in summaries.values()):
        raise RuntimeError("Incomplete evaluation; inspect episode reports; paper metrics withheld")
    print((args.output/"table.md").read_text())


if __name__ == "__main__":
    main()
