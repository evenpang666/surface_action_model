#!/usr/bin/env python3
"""Direct legacy PI05 left-arm policy executor; no viewer, OBB, or CBF service."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "openpi" / "packages" / "openpi-client" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.live_dual_ur7e_obstacle_model import ReadOnlyRobotiq2FPosition
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path

ACK = "I_UNDERSTAND_LEGACY_LEFT_POLICY_HAS_NO_VALID_CBF"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--policy-host", default="127.0.0.1"); p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    p.add_argument("--front-serial", default=None); p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480); p.add_argument("--fps", type=int, default=15)
    p.add_argument("--control-hz", type=float, default=30.0); p.add_argument("--queue-low-watermark", type=int, default=2); p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--execute", action="store_true"); p.add_argument("--execute-ack", default="")
    p.add_argument("--execute-gripper", action="store_true", help="With --execute, send legacy action dim 7 to the left 2F-85 (0=open, 1=closed).")
    p.add_argument("--gripper-speed", type=int, default=40); p.add_argument("--gripper-force", type=int, default=25)
    return p.parse_args()


def main() -> None:
    a = args()
    if a.control_hz <= 0 or a.max_steps < 1 or a.queue_low_watermark < 0: raise ValueError("control-hz/max-steps must be positive and queue-low-watermark non-negative")
    if a.execute and a.execute_ack != ACK: raise ValueError(f"Physical execution requires --execute-ack {ACK}")
    if a.execute_gripper and not a.execute: raise ValueError("--execute-gripper requires --execute")
    config, config_path = load_project_config(a.project_config)
    with resolve_repo_path(config["quest3"]["hardware_config"]).open(encoding="utf-8") as f: hw = yaml.safe_load(f)
    left_ip, right_ip = str(hw["arms"]["left_arm"]["robot_ip"]), str(hw["arms"]["right_arm"]["robot_ip"])
    serial = a.front_serial or str(hw["cameras"]["front"]["serial"])
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from rtde_receive import RTDEReceiveInterface
    import pyrealsense2 as rs

    policy = WebsocketClientPolicy(host=a.policy_host, port=a.policy_port)
    meta = policy.get_server_metadata()
    if (meta.get("joint_dim"), meta.get("logical_action_dim"), meta.get("points_per_link")) != (6, 7, 8):
        raise RuntimeError("This direct executor is only for the legacy 6-state/7-action/8-points-per-link checkpoint")
    receiver_l = RTDEReceiveInterface(left_ip)
    control = None
    gripper_control = None
    if a.execute:
        from rtde_control import RTDEControlInterface
        control = RTDEControlInterface(left_ip)  # Never opens a right-arm control channel.
        if a.execute_gripper:
            from real_scripts.ur7e_robotiq_d435i_collector.utils.robotiq_interface import RobotiqGripper
            gripper_control = RobotiqGripper(left_ip, port=63352, timeout=2.0)
            gripper_control.connect()
    gripper = None
    try: gripper = ReadOnlyRobotiq2FPosition(left_ip, port=63352)
    except Exception as exc: print(f"[warning] 2F-85 position unavailable; assuming open: {exc}")
    pipe, cfg = rs.pipeline(), rs.config(); cfg.enable_device(serial); cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    sampler = DualUR7eSurfacePointSampler(points_per_link=8, project_config=config_path)
    period = 1.0 / a.control_hz
    started = False
    try:
        pipe.start(cfg)
        started = True
        for _ in range(10): pipe.wait_for_frames()
        def observation() -> dict:
            rgb = np.ascontiguousarray(np.asarray(pipe.wait_for_frames().get_color_frame().get_data(), dtype=np.uint8))
            ql = np.asarray(receiver_l.getActualQ(), np.float32)
            grip = np.array([gripper.closing_fraction() if gripper else 0.0, 0.0], np.float32)
            qdual = np.r_[ql, np.zeros(6, np.float32)]
            points = sampler.link_points(qdual, grip).reshape(-1, 3)[: len(sampler.left_link_names) * 8]
            return {"prompt": a.prompt, "rgb_front": rgb, "qpos": ql, "robot_points": points}

        queue: deque[np.ndarray] = deque()
        last_gripper_target: float | None = None
        future = None
        step = 0
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-infer") as pool:
            future = pool.submit(policy.infer, observation())
            while step < a.max_steps:
                if future is not None and future.done():
                    result = future.result(); future = None
                    chunk = np.asarray(result["actions"], np.float32)
                    if chunk.ndim != 2 or chunk.shape[1] != 7: raise RuntimeError(f"Legacy policy returned invalid action chunk {chunk.shape}")
                    queue.extend(chunk)
                if future is None and len(queue) <= a.queue_low_watermark:
                    future = pool.submit(policy.infer, observation())
                if not queue:
                    time.sleep(.002)
                    continue
                began = time.monotonic()
                ql = np.asarray(receiver_l.getActualQ(), np.float32)
                left_action = queue.popleft()
                dual_action = np.zeros(14, np.float32); dual_action[:7] = left_action
                delta = np.clip(dual_action[:6], -0.05, 0.05)
                if control is not None and not bool(control.servoJ((ql + delta).tolist(), .10, .30, period, .10, 300)):
                    raise RuntimeError("Left UR rejected servoJ")
                gripper_sent = False
                if gripper_control is not None and (last_gripper_target is None or abs(float(left_action[6]) - last_gripper_target) >= .03):
                    gripper_control.write_position(float(np.clip(left_action[6], 0., 1.)), speed=int(np.clip(a.gripper_speed, 0, 255)), force=int(np.clip(a.gripper_force, 0, 255)))
                    last_gripper_target = float(left_action[6])
                    gripper_sent = True
                print({"step": step, "queued_actions": len(queue), "left_action": left_action.tolist(), "right_action_padded_zero": dual_action[7:].tolist(), "sent_left_delta": delta.tolist(), "left_2f85_sent": gripper_sent, "execute": bool(control)})
                step += 1
                time.sleep(max(0., period - (time.monotonic() - began)))
    finally:
        if started:
            pipe.stop()
        if gripper: gripper.close()
        receiver_l.disconnect()
        if control:
            control.servoStop(1.0); control.disconnect()
        if gripper_control:
            gripper_control.disconnect()


if __name__ == "__main__": main()
