"""Local LIBERO-Safety rollouts with OA-VLA TCP and the hardware safety pipeline."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from obstacle_validation.libero_safety_geometry import CAMERAS, STATE_SEMANTICS, IMAGE_CONVENTION
from obstacle_validation.oa_vla_actions import ObservationHold

SUITES = ("obstacle_avoidance", "obstacle_avoidance_human", "affordance", "human_safety")


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def prepare_paths(root, config_dir, assets=None):
    """Isolate LIBERO config before importing it; fail on upstream LIBERO shadows."""
    import yaml
    root = Path(root).expanduser().resolve()
    package = root / "libero" / "libero"
    if not (package / "benchmark" / "vla_safety_task_map.py").is_file():
        raise ValueError("--libero-root must be the LIBERO-Safety checkout, not upstream LIBERO")
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(benchmark_root=str(package), bddl_files=str(package/"bddl_files"),
                 init_states=str(package/"init_files"), datasets=str(root/"datasets"),
                 assets=str(Path(assets).expanduser().resolve() if assets else package/"assets"))
    if assets and (package/"assets").resolve() != Path(paths["assets"]):
        raise ValueError("Official object classes also read checkout-relative assets; link checkout/libero/libero/assets "
                         "to --assets-root first")
    config_file = config_dir/"config.yaml"
    if config_file.exists() and yaml.safe_load(config_file.read_text()) != paths:
        raise ValueError(f"Existing LIBERO config points elsewhere: {config_file}; choose another --config-dir")
    config_file.write_text(yaml.safe_dump(paths))
    if "libero.libero" in sys.modules:
        raise RuntimeError("Import LIBERO-Safety only after preparing this run's config")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)  # upstream expects a DIRECTORY
    sys.path.insert(0, str(root))
    return root, paths


def validate_metadata(metadata, camera_names, *, safety=True, allow_test=False):
    from oa_vla.socket_protocol import VIEWS
    if (metadata.get("model_type") != "oa-vla" or metadata.get("model_format") != 2
            or metadata.get("state_dim") != 8 or metadata.get("action_dim") != 7
            or metadata.get("action_semantics") != "libero_osc_pose"
            or metadata.get("state_semantics") != STATE_SEMANTICS
            or metadata.get("image_convention") != IMAGE_CONVENTION
            or metadata.get("policy_views") != list(VIEWS)
            or metadata.get("camera_names") != camera_names):
        raise ValueError("Requires a Panda/LIBERO 8-state, 7-action OA-VLA export with matching camera/image metadata; "
                         "dual-UR7e and native LeRobot/OpenPI weights cannot be relabeled")
    period = metadata.get("action_period_s")
    if not isinstance(period, (int, float)) or not math.isclose(period, .05, abs_tol=1e-6):
        raise ValueError("LIBERO-Safety profile requires 20 Hz / action_period_s=0.05")
    if metadata.get("test_only") and (safety or not allow_test):
        raise ValueError("Random test policy requires --mode baseline --allow-test-policy")
    if safety and (metadata.get("detection_trained") is not True
                   or metadata.get("detection_views") != ["front", "side"]
                   or metadata.get("pair_id_scope") != "current_prediction"
                   or metadata.get("detection_box_format") != "cxcywh_angle_deg_original_pixels"):
        raise ValueError("Safety mode requires a trained paired-OBB detector")


def first_action(prediction):
    chunk = np.asarray(prediction.get("actions"), dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or not 1 <= len(chunk) <= 1024 or not np.isfinite(chunk).all():
        raise ValueError("Expected finite [H,7] normalized OSC_POSE actions")
    if np.any(np.abs(chunk[0]) > 1+1e-6):
        raise ObservationHold("Predicted action outside normalized OSC bounds [-1,1]")
    return np.clip(chunk[0], -1, 1)


class LeRobotPolicy:
    """Native LeRobot v0.4.4 pi05 baseline; saved processors own normalization."""
    def __init__(self, path, device):
        from importlib.metadata import version
        if version("lerobot") != "0.4.4":
            raise RuntimeError("Native baseline requires the documented LeRobot v0.4.4 environment")
        import torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        features = PI05Config.from_pretrained(path)
        features.device = device
        expected = {"observation.images.image", "observation.images.image2"}
        if set(features.image_features) != expected or features.output_features["action"].shape != (7,):
            raise ValueError("Native baseline requires canonical two-camera LIBERO dataset keys and 7-D actions")
        if features.input_features["observation.state"].shape != (8,):
            raise ValueError("Native baseline requires 8-D LIBERO EEF state")
        self.torch = torch
        self.policy = PI05Policy.from_pretrained(path, config=features).to(device).eval()
        self.pre, self.post = make_pre_post_processors(features, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.metadata = dict(model_type="lerobot_pi05", checkpoint=str(path), detection_trained=False)

    def infer(self, rgb, state, prompt, **kwargs):
        torch = self.torch
        batch = {"observation.state": torch.as_tensor(state), "task": prompt}
        for key, slot in (("observation.images.image", "front"), ("observation.images.image2", "right_wrist")):
            batch[key] = torch.from_numpy(rgb[slot].copy()).permute(2, 0, 1).float()/255
        # Discard all unused chunk actions; next step must use a new observation.
        self.policy.reset()
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        return dict(actions=action.detach().cpu().numpy().reshape(1, -1))

    def close(self):
        pass


class ConstraintMonitor:
    """Evaluate every official constraint, including on the success timestep.

    Upstream _check_constraint(done) suppresses checks when done=True and folds
    duplicate predicate names into a dict. This monitor retains every predicate.
    """
    def __init__(self, env):
        self.base = env.env
        self.constraints = self.base.parsed_problem["constraints"]
        self.violations, self.substeps = [], 0

    def sample(self):
        for index, predicate in enumerate(self.constraints):
            if self.base._eval_predicate(predicate):
                entry = dict(index=index, predicate=predicate, substep=self.substeps)
                if not any(v["index"] == index for v in self.violations):
                    self.violations.append(entry)

    @contextmanager
    def track(self):
        sim = self.base.sim
        original = sim.step
        def checked_step(*args, **kwargs):
            result = original(*args, **kwargs)
            self.substeps += 1
            self.sample()
            return result
        sim.step = checked_step
        try:
            yield self
        finally:
            sim.step = original


def safe_action(nominal, prediction, mapped, geometry, planner, args):
    from obstacle_validation.oa_vla_perception import DynamicOBBFilter, inconsistent_pairs_allow_empty_geometry
    from real_scripts.real_cbf_qp import boxes_from_snapshot
    snapshot = DynamicOBBFilter(mapped, prediction, minimum_score=args.minimum_object_score,
                                padding_pixels=args.obb_padding_pixels)(mapped.snapshot)
    boxes = boxes_from_snapshot(snapshot)
    if not boxes and not inconsistent_pairs_allow_empty_geometry(snapshot, prediction):
        raise ObservationHold("No usable paired obstacle geometry")
    boxes.source_snapshot = snapshot
    # Dynamics advance by simulation time, not the time spent waiting for a GPU.
    snapshot["motion_timestamp_ns"] = int(geometry.sim.data.time*1e9)+1
    grip = geometry.gripper_target(nominal[-1])
    target = np.r_[nominal[:6], grip]
    _, info = planner.plan(np.zeros(6), geometry.grip, target[None], boxes,
                           lower=-np.ones(6), upper=np.ones(6), max_step=1.)
    waypoint = np.asarray(info["waypoints"][0])
    detail = info["steps"][0]
    if (not info.get("success") or not np.isfinite(waypoint).all() or np.any(np.abs(waypoint[:6]) > 1+1e-8)
            or not 0 <= waypoint[-1] <= 1
            or not math.isfinite(detail["max_distance_slack_m"])
            or detail["max_distance_slack_m"] > args.max_distance_slack
            or math.isnan(detail["minimum_predicted_barrier_after_m"])
            or detail["minimum_predicted_barrier_after_m"] < -1e-5):
        raise ObservationHold("MCBF rejected slack/barrier/control bounds")
    predicted_q = geometry.predicted_q(waypoint[:6])
    limits = np.asarray(geometry.robot.robot_model.joints)  # names, not joint indices
    jids = [geometry.sim.model.joint_name2id(name) for name in limits]
    ranges = geometry.sim.model.jnt_range[jids]
    if np.any(predicted_q < ranges[:, 0]) or np.any(predicted_q > ranges[:, 1]):
        raise ObservationHold("Predicted Panda joint limit violation")
    info.update(joint_variables=0, ee_variables=6, optimization_space="normalized_osc_pose")
    return np.r_[waypoint[:6], nominal[-1]], dict(optimizer=info,
        paired_geometry=snapshot["diagnostics"].get("paired_rgb_obstacles"),
        optimization_coordinates="normalized_osc_pose", geometry_coordinates="mujoco_world")


def rollout(env, policy, args, camera_names, output, *, task, initial_state):
    from obstacle_validation.libero_safety_geometry import capture, PandaGeometry, SimWorkspaceTSDF
    from obstacle_validation.ellipsoid_mcbf import EllipsoidMCBF, EllipsoidMCBFConfig
    env.reset()
    obs = env.set_init_state(initial_state)
    monitor = ConstraintMonitor(env)
    geometry = PandaGeometry(env) if args.mode == "safety" else None
    planner = EllipsoidMCBF(None, margin=args.margin, inter_arm_margin=0., period=.05,
                            robot=geometry, joint_groups=(6,), grip_dim=1,
                            config=EllipsoidMCBFConfig(max_control_step_rad=1.)) if geometry else None
    mapping, video = None, None
    recorder = None
    if args.record_training:
        from obstacle_validation.libero_safety_recording import TrainingRecorder
        recorder = TrainingRecorder(env, camera_names, args.label_obstacles, args.training_horizon)
    result = dict(completed=False, success=False, safe_success=False, safety_violation=False,
                  termination_reason=None, steps=0, holds=0, interventions=0, warmup_steps=args.warmup_steps,
                  benchmark="LIBERO-Safety", suite=args.suite, level=args.level, task_id=args.task,
                  task_name=task.name, seed=args.seed, wall_latency_s=[])
    began = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    try:
        if args.video:
            import imageio.v2 as imageio
            video = imageio.get_writer(output/"rollout.mp4", fps=20)
        with (output/"steps.jsonl").open("w") as log, monitor.track():
            # Warmup is included in contact monitoring, never hidden from scoring.
            for _ in range(args.warmup_steps):
                obs, _, _, _ = env.step(np.r_[np.zeros(6), -1.])
                if monitor.violations:
                    break
            for step in range(args.max_steps):
                if monitor.violations:
                    result["termination_reason"] = "safety_violation"
                    break
                start = time.monotonic()
                bundle, calibrations = capture(env, obs, camera_names, step)
                if recorder:
                    recorder.add(bundle)
                if video:
                    video.append_data(bundle.frames["front"].rgb)
                mapped = None
                if geometry:
                    geometry.bind()
                    if mapping is None:
                        mapping = SimWorkspaceTSDF(calibrations, args.workspace, args.voxel_resolution, args.max_grid_cells)
                    mapped = mapping.update(bundle, calibrations, geometry, camera_names)
                prediction = policy.infer({k: f.rgb for k, f in bundle.frames.items()}, bundle.feedback.q,
                                           task.language, context=bundle.context(), encoding=args.encoding)
                record = dict(step=step, frame_id=step, context=bundle.context(), status="executed",
                              prediction_actions=prediction.get("actions"), predicted_objects=prediction.get("objects"))
                try:
                    nominal = first_action(prediction)
                    action = nominal.copy()
                    if time.monotonic()-start > args.max_result_age:
                        raise ObservationHold("VLA result/capture expired")
                    if mapped:
                        action, diagnostics = safe_action(nominal, prediction, mapped, geometry, planner, args)
                        record.update(diagnostics)
                    if time.monotonic()-start > args.max_result_age:
                        raise ObservationHold("Result expired during mapping/planning")
                    record.update(nominal_action=nominal, action=action)
                    result["interventions"] += int(not np.allclose(action, nominal, atol=1e-6))
                except ObservationHold as exc:
                    # A hold still advances one simulated tick, preserving dynamic obstacles.
                    action = np.zeros(7)
                    record.update(status="hold", reason=str(exc), action=action)
                    result["holds"] += 1
                if recorder:
                    recorder.set_action(action)
                obs, _, done, info = env.step(action)
                monitor.sample()  # includes goal contact even if upstream suppresses its cost
                latency = time.monotonic()-start
                result["wall_latency_s"].append(latency)
                result["steps"] += 1
                record.update(latency_s=latency, goal_success=bool(done), cost=info.get("cost"),
                              violations=monitor.violations)
                log.write(json.dumps(json_value(record), ensure_ascii=False, allow_nan=False)+"\n")
                log.flush()
                if monitor.violations or done:
                    result.update(success=bool(done), termination_reason="safety_violation" if monitor.violations else "success")
                    break
            else:
                result["termination_reason"] = "timeout"
            result.update(completed=True, safety_violation=bool(monitor.violations),
                          safe_success=bool(result["success"] and not monitor.violations),
                          violations=monitor.violations, physics_samples=monitor.substeps)
            if recorder and result["safe_success"]:
                destination = args.record_training/f"{args.suite}_L{args.level}_task{args.task}_{output.name}_{args.output.name}.shard"
                recorder.save(destination, task.language)
                result["training_shard"] = str(destination)
            elif recorder:
                result["training_export_skipped"] = "Only complete safe successful episodes become action demonstrations"
    except BaseException as exc:
        result.update(termination_reason="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if video:
            video.close()
        result["elapsed_s"] = time.monotonic()-began
        write_json(output/"report.json", result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--assets-root", type=Path)
    p.add_argument("--config-dir", type=Path, default=Path("outputs/libero_safety_config"))
    p.add_argument("--output", type=Path, default=Path("outputs/libero_safety")/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    p.add_argument("--suite", choices=SUITES, default="obstacle_avoidance")
    p.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--task", type=int, choices=range(5), default=0)
    p.add_argument("--episodes", nargs="+", type=int, default=[0])
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("baseline", "safety"), default="safety")
    p.add_argument("--backend", choices=("oa-vla", "lerobot"), default="oa-vla")
    p.add_argument("--policy", type=Path, help="Native LeRobot pi05 pretrained_model directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--socket-timeout", type=float, default=30.)
    p.add_argument("--encoding", choices=("raw", "png", "jpeg"), default="raw")
    p.add_argument("--camera-map", type=Path, help="JSON mapping four model slots to four distinct simulator cameras")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--workspace", type=float, nargs=6, default=[-.6, .6, -.6, .6, .75, 1.6],
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    p.add_argument("--voxel-resolution", type=float, default=.02)
    p.add_argument("--max-grid-cells", type=int, default=2_000_000)
    p.add_argument("--margin", type=float, default=.01)
    p.add_argument("--max-distance-slack", type=float, default=1e-5)
    p.add_argument("--minimum-object-score", type=float, default=.5)
    p.add_argument("--obb-padding-pixels", type=float, default=2.)
    p.add_argument("--max-result-age", type=float, default=.75)
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-test-policy", action="store_true", help="Simulation baseline ONLY")
    p.add_argument("--record-training", type=Path, help="Export safe successful rollouts as four-camera OA-VLA shards")
    p.add_argument("--label-obstacles", nargs="+", help="Explicit BDDL obstacle instance names; simulator-only labels")
    p.add_argument("--training-horizon", type=int, default=8)
    p.add_argument("--check-only", action="store_true", help="Check imports, paths and task files; no model/physics/device I/O")
    a = p.parse_args(argv)
    for k in ("max_steps", "max_grid_cells", "resolution"):
        if getattr(a, k) < 1:
            p.error(f"{k} must be positive")
    for k in ("socket_timeout", "voxel_resolution", "max_result_age"):
        if not math.isfinite(getattr(a, k)) or getattr(a, k) <= 0:
            p.error(f"{k} must be finite and positive")
    for k in ("margin", "max_distance_slack", "obb_padding_pixels"):
        if not math.isfinite(getattr(a, k)) or getattr(a, k) < 0:
            p.error(f"{k} must be finite and nonnegative")
    if not 0 <= a.minimum_object_score <= 1 or a.warmup_steps < 0:
        p.error("Invalid score/warmup steps")
    bounds = np.asarray(a.workspace).reshape(3, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        p.error("Invalid workspace bounds")
    if not a.episodes or min(a.episodes) < 0 or len(set(a.episodes)) != len(a.episodes):
        p.error("Episode IDs must be unique and nonnegative")
    if a.backend == "lerobot" and (a.mode != "baseline" or a.policy is None):
        p.error("Native LeRobot requires --mode baseline --policy; it has no paired-OBB head")
    if a.allow_test_policy and (a.backend != "oa-vla" or a.mode != "baseline"):
        p.error("--allow-test-policy is only valid for OA-VLA simulation baseline")
    if a.record_training and (a.level == 2 or not a.label_obstacles or a.training_horizon < 1 or a.allow_test_policy):
        p.error("Training export needs L0/L1, --label-obstacles, a positive horizon and a trained policy; L2 is held out")
    return a


def main(argv=None):
    args = parse_args(argv)
    root, paths = prepare_paths(args.libero_root, args.config_dir, args.assets_root)
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "mujoco", "robosuite", "bddl", "wand", "skimage", "cv2", "imageio", "oa_vla", "osqp")}
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        raise RuntimeError(f"Missing inference dependencies: {missing}; install requirements/libero_safety.txt, "
                           "the benchmark's robosuite fork and oa-vla client in the active environment")
    from libero.libero import benchmark
    import libero.libero as libero_package
    if Path(libero_package.__file__).resolve().parent != root/"libero/libero":
        raise RuntimeError(f"Wrong LIBERO imported: {libero_package.__file__}")
    suite = benchmark.get_benchmark(args.suite)()
    task = suite.get_task_by_level_id(args.level, args.task)
    bddl = Path(suite.get_task_bddl_file_path_by_level_id(args.level, args.task))
    init = Path(suite._get_task_file_path(args.level, args.task, "init_states", ".pruned_init"))
    for path in (bddl, init, Path(paths["assets"])):
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark resource: {path}")
    cameras = json.loads(args.camera_map.read_text()) if args.camera_map else dict(CAMERAS)
    if set(cameras) != set(CAMERAS) or len(set(cameras.values())) != 4:
        raise ValueError("Four distinct real simulator cameras required; no image duplication")
    if args.check_only:
        if args.mode == "safety":
            from real_scripts.voxblox_obstacle_map import require_voxblox
            require_voxblox()
        print(json.dumps(dict(paths=paths, task=str(bddl), initial_states=str(init), cameras=cameras,
                              dependencies=dependencies, mode=args.mode), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    states = suite.get_task_init_states_by_level_id(args.level, args.task)
    if max(args.episodes) >= len(states):
        raise ValueError(f"Only {len(states)} initial states exist for this task")
    if args.backend == "oa-vla":
        from oa_vla.socket_protocol import PolicyClient
        policy = PolicyClient(args.host, args.port, timeout=args.socket_timeout)
    else:
        policy = LeRobotPolicy(args.policy, args.device)
    env, reports = None, []
    try:
        if args.backend == "oa-vla":
            validate_metadata(policy.metadata, cameras, safety=args.mode == "safety", allow_test=args.allow_test_policy)
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite.macros as macros
        if macros.IMAGE_CONVENTION != "opengl":
            raise ValueError("Adapter expects robosuite IMAGE_CONVENTION=opengl before its explicit vertical flip")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), controller="OSC_POSE", control_freq=20,
                                camera_names=list(cameras.values()), camera_heights=args.resolution,
                                camera_widths=args.resolution, camera_depths=True)
        env.seed(args.seed)
        if env.env.action_dim != 7:
            raise ValueError("Expected 7-D OSC_POSE environment")
        write_json(args.output/"run_config.json", dict(arguments=vars(args), cameras=cameras,
            policy=policy.metadata, libero_root=str(root), bddl=str(bddl),
            simulation_clock="physics pauses during inference", image_convention=IMAGE_CONVENTION))
        for episode in args.episodes:
            r = rollout(env, policy, args, cameras, args.output/f"episode_{episode:04d}",
                        task=task, initial_state=states[episode])
            r["episode"] = episode
            reports.append(r)
            print(f"episode={episode} reason={r['termination_reason']} holds={r['holds']}", flush=True)
    finally:
        cleanup_errors = []
        for resource in (env, policy):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        all_complete = len(reports) == len(args.episodes)
        write_json(args.output/"summary.json", dict(requested=len(args.episodes), completed=len(reports), all_complete=all_complete,
            protocol="local_paired_rgbd_extension", mode=args.mode, backend=args.backend, cleanup_errors=cleanup_errors,
            safe_success_rate=float(np.mean([r["safe_success"] for r in reports])) if all_complete else None,
            safety_violation_rate=float(np.mean([r["safety_violation"] for r in reports])) if all_complete else None,
            episodes=reports))
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"Simulation cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
