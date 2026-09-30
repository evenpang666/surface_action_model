#!/usr/bin/env python3
"""Read-only real-time tabletop obstacle model for two UR7e arms and front and side D455 cameras.

The model uses ``quest3_collect/config/hardware_dual_teleop.yaml`` and its two
``base -> front_camera`` calibrations.  All output is expressed in ``left_base``.
It only uses :class:`rtde_receive.RTDEReceiveInterface`; it cannot command an
arm or either gripper.

The official UR7e collision meshes are rendered for both arms.  The installed
Robotiq 2F-85 uses upstream collision meshes driven by a read-only ``GET POS``
query; both arms use the articulated Robotiq 2F-85 vendor collision mesh.
Both are rendered at their configured active TCPs.  Verify the physical
mount/cup dimensions before treating the result as motion-planning geometry.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser
import socket

import numpy as np
import yaml
import cv2
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.lingbot_depth import add_lingbot_depth_cli_args, create_lingbot_depth_refiner_from_args
from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame, depth_to_world_points, robot_depth_keep_mask
from real_scripts.reconstruct_realsense_pointcloud import estimate_dominant_plane
from real_scripts.ur7e_collision_mesh import (
    collision_surface_samples,
    link_and_collision_transforms,
    render_surface_points_depth,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource
from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_points_in_active_tcp
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from real_scripts.preprocess_left_vr_hdf5_cotracker_observations import _lift_measured_tracks


VIEWER_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Dual UR7e obstacle model</title>
<script src="/plotly.min.js"></script><style>html,body,#scene{margin:0;width:100%;height:100%;overflow:hidden;background:#080a0d;color:#e6edf3;font:14px Arial}#hud{position:fixed;z-index:2;left:12px;top:10px;background:#000c;padding:10px 13px;border-radius:7px;line-height:1.6;pointer-events:none}#pause{pointer-events:auto;margin-top:6px;padding:4px 9px;border:1px solid #6c7784;border-radius:4px;background:#17212b;color:#e6edf3;cursor:pointer}.robot{color:#ff6b55}.obstacle{color:#45ff9b}.warning{color:#ffd166}</style></head>
<body><div id="hud"><b>双 UR7e 实时桌面障碍物模型（left_base）</b><br><span id="s">等待首帧…</span><br><span class="robot">红/橙：双 Robotiq 2F-85 与双臂</span> · <span class="obstacle">绿：桌面连接障碍物 OBB</span><br><span class="warning">仅 RTDE 读取；不会发送运动或夹爪命令。</span><br><button id="pause" type="button" aria-pressed="false">暂停显示（空格）</button></div><div id="scene"></div>
<script>const G=document.getElementById('scene'),S=document.getElementById('s'),P=document.getElementById('pause'),E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];let paused=false;function setPaused(value){paused=value;P.textContent=paused?'继续显示（空格）':'暂停显示（空格）';P.setAttribute('aria-pressed',String(paused))}P.addEventListener('click',()=>setPaused(!paused));document.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat){e.preventDefault();setPaused(!paused)}});function pts(g,n,s){let p=g.points,c=g.colors;return{type:'scatter3d',mode:'markers',name:n,x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),marker:{size:s,opacity:.8,color:c.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`)},hoverinfo:'skip'}}function boxes(b){let r=[];for(let k=0;k<b.length;k++){let q=b[k].corners,x=[],y=[],z=[];for(let[a,d]of E)x.push(q[a][0],q[d][0],null),y.push(q[a][1],q[d][1],null),z.push(q[a][2],q[d][2],null);r.push({type:'scatter3d',mode:'lines',name:`障碍物 ${k+1}`,x,y,z,line:{color:'#45ff9b',width:6},hoverinfo:'skip'})}return r}function draw(d){S.textContent=`帧 ${d.frame} · ${d.status} · 环境 ${d.environment.points.length} 点 · 双臂 ${d.robot.points.length} 点 · ${d.obbs.length} 个障碍物`;Plotly.react(G,[pts(d.environment,'environment',.7),pts(d.robot,'dual robots',1.1),...boxes(d.obbs)],{uirevision:'keep-orbit',showlegend:false,paper_bgcolor:'#080a0d',margin:{l:0,r:0,t:0,b:0},scene:{bgcolor:'#080a0d',aspectmode:'data',camera:{eye:{x:1.5,y:-1.5,z:1.15},up:{x:0,y:0,z:1}},xaxis:{title:'left_base X (m)'},yaxis:{title:'left_base Y (m)'},zaxis:{title:'left_base Z (m)'}}},{displaylogo:false,responsive:true,scrollZoom:true})}async function poll(){if(!paused)try{let r=await fetch('/snapshot.json?'+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();if(!paused)draw(d)}catch(e){if(!paused)S.textContent='读取快照失败，正在重试：'+e}setTimeout(poll,900)}poll();</script></body></html>"""


@dataclass(frozen=True)
class Arm:
    name: str
    ip: str
    base_to_left: np.ndarray
    tool: str
    tool_from_tcp: np.ndarray


@dataclass(frozen=True)
class UprightOBB:
    center: np.ndarray
    rotation: np.ndarray
    extents: np.ndarray
    corners: np.ndarray
    point_count: int


def expanded_obb(box: UprightOBB, margin_m: float) -> UprightOBB:
    margin = max(float(margin_m), 0.0)
    extents = np.asarray(box.extents, dtype=np.float32) + 2.0 * margin
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float32)
    corners = box.center[None, :] + (0.5 * signs * extents[None, :]) @ box.rotation.T
    return UprightOBB(box.center, box.rotation, extents, corners.astype(np.float32), box.point_count)


class ReadOnlyRobotiq2FPosition:
    """URCap position reader that is physically incapable of sending a SET command."""

    def __init__(self, host: str, *, port: int, timeout_s: float = .25) -> None:
        self.socket = socket.create_connection((host, int(port)), timeout=float(timeout_s))
        self.socket.settimeout(float(timeout_s))

    def closing_fraction(self) -> float:
        self.socket.sendall(b"GET POS\n")
        fields = self.socket.recv(1024).decode("ascii", errors="replace").strip().split()
        if len(fields) != 2 or fields[0] != "POS":
            raise RuntimeError(f"unexpected Robotiq GET POS response: {' '.join(fields)!r}")
        return float(np.clip(int(fields[1]) / 255.0, 0.0, 1.0))

    def close(self) -> None:
        self.socket.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--front-serial", default=None, help="Override front D455 serial; default is hardware config.")
    parser.add_argument("--side-serial", default=None)
    parser.add_argument("--width", type=int, default=1280); parser.add_argument("--height", type=int, default=720); parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=20); parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--samples-per-face", type=int, default=16); parser.add_argument("--absolute-tolerance-m", type=float, default=.015); parser.add_argument("--relative-tolerance", type=float, default=.015); parser.add_argument("--dilation-pixels", type=int, default=3)
    add_lingbot_depth_cli_args(parser)
    # Two fixed D455 views; wrist D435i cameras belong to policy observations.
    parser.set_defaults(lingbot_camera_names=("front", "side"), lingbot_depth=True)
    parser.add_argument("--workspace-bounds", nargs=6, type=float, default=(-0.85, 0.85, -0.85, 0.85, -0.12, 0.70), metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), help="left_base crop for point cloud and OBBs.")
    parser.add_argument("--plane-threshold-m", type=float, default=.012); parser.add_argument("--min-height-m", type=float, default=.020); parser.add_argument("--max-height-m", type=float, default=.35); parser.add_argument("--outlier-neighbour-m", type=float, default=.020)
    parser.add_argument("--cluster-eps-m", type=float, default=.010, help="Exact point-to-point DBSCAN radius; unlike voxel adjacency, this is a physical distance.")
    parser.add_argument("--cluster-min-samples", type=int, default=6, help="DBSCAN core-neighbour count, including the point itself.")
    parser.add_argument("--min-cluster-points", type=int, default=80); parser.add_argument("--attachment-distance-m", type=float, default=.050); parser.add_argument("--box-margin-m", type=float, default=.010)
    parser.add_argument("--raw-depth-support-pixels", type=int, default=2, help="LingBot pixels must be within this many pixels of raw, non-robot D455 depth; 0 permits raw pixels only.")
    parser.add_argument("--temporal-persistence-frames", type=int, default=3, help="Consecutive observations required before an obstacle OBB is shown; use 1 to disable.")
    parser.add_argument("--temporal-track-distance-m", type=float, default=.035, help="Maximum centre displacement when associating an OBB across frames.")
    parser.add_argument("--left-tool-from-tcp", type=Path, default=None, help="Optional JSON/YAML ^active_tcpT_left_tool_urdf_root; identity for the bundled URDF.")
    parser.add_argument("--right-tool-from-tcp", type=Path, default=None, help="Optional JSON/YAML ^active_tcpT_right_tool_urdf_root; identity for the bundled URDF.")
    parser.add_argument("--left-robotiq-read-port", type=int, default=63352, help="Left 2F-85 URCap port used only for GET POS.")
    parser.add_argument("--left-finger-angle-rad", type=float, default=0.0, help="Fallback 2F-85 finger angle if GET POS is unavailable; 0=open, 0.8=closed.")
    parser.add_argument("--display-max-points", type=int, default=45000); parser.add_argument("--port", type=int, default=8766); parser.add_argument("--bind-host", default="127.0.0.1"); parser.add_argument("--once", action="store_true"); parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def matrix_from_file(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if isinstance(value, dict):
        value = value.get("matrix", value.get("transform", value.get("tool_from_tcp")))
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1)):
        raise ValueError(f"{path} must contain a finite 4x4 homogeneous matrix")
    return matrix


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, str, dict]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    matrix = np.asarray(data["matrix"], dtype=np.float64)
    intr = data["intrinsics"]
    K = np.asarray(((intr["fx"], 0., intr["ppx"]), (0., intr["fy"], intr["ppy"]), (0., 0., 1.)), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], (0, 0, 0, 1)):
        raise ValueError(f"Invalid calibration matrix in {path}")
    if not data.get("calibrated", False):
        raise ValueError(f"Calibration is not marked calibrated: {path}")
    rms = float(data.get("touch_rms_mm", np.inf))
    maximum = float(data.get("quality_limits", {}).get("max_touch_rms_mm", np.inf))
    if not np.isfinite(rms) or rms > maximum:
        raise ValueError(f"Calibration touch RMS {rms:.2f} mm fails its {maximum:.2f} mm limit: {path}")
    return matrix, K, str(data.get("camera_serial", "")), data


def load_arms(args: argparse.Namespace) -> tuple[Arm, Arm, CameraCalibration, dict]:
    project, _ = load_project_config(args.project_config)
    if not project["safety"].get("camera_calibration_valid", False):
        raise RuntimeError("Front calibration expired and side is uncalibrated; install a newly validated calibration before starting live_obstacle_detection")
    config_path = resolve_repo_path(project["quest3"]["hardware_config"])
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    arms_config = config["arms"]
    left_cfg, right_cfg = arms_config["left_arm"], arms_config["right_arm"]
    left_path = resolve_repo_path(project["safety"]["front_calibration"])
    right_path = resolve_repo_path(project["safety"]["right_front_calibration"])
    T_left_base_from_front, K, serial, left_report = load_calibration(left_path)
    T_right_base_from_front, right_K, right_serial, right_report = load_calibration(right_path)
    if left_report.get("parent_frame") != "left_base" or right_report.get("parent_frame") != "right_base":
        raise ValueError("Front calibrations must explicitly name left_base and right_base")
    if serial != right_serial:
        raise ValueError(f"Calibration cameras differ: left={serial}, right={right_serial}")
    # The two independent hand-eye fits record the D455 intrinsics at different
    # instants.  Small RealSense SDK rounding/reconfiguration differences are
    # normal (the checked files differ by about 0.5 px), while their extrinsics
    # still share the same physical color optical frame.  The left fit's K is
    # used consistently with ^left_baseT_camera for projection/unprojection.
    intrinsics_delta_px = float(np.max(np.abs(K - right_K)))
    requested_serial = args.front_serial or config.get("cameras", {}).get("front", {}).get("serial")
    if requested_serial and str(requested_serial) != serial:
        raise ValueError(f"Configured front serial {requested_serial} does not match calibration serial {serial}")
    # T_left_camera maps front-camera coordinates into left_base.  Right collision
    # geometry is mapped through the shared camera frame into left_base.
    right_to_left = T_left_base_from_front @ np.linalg.inv(T_right_base_from_front)
    left = Arm("left", str(left_cfg["robot_ip"]), np.eye(4), "Robotiq 2F-85", matrix_from_file(args.left_tool_from_tcp))
    right = Arm("right", str(right_cfg["robot_ip"]), right_to_left, "Robotiq 2F-85", matrix_from_file(args.right_tool_from_tcp))
    calibration = CameraCalibration("front", K, T_left_base_from_front)
    report = {"camera_serial": serial, "left_touch_rms_mm": left_report["touch_rms_mm"], "right_touch_rms_mm": right_report["touch_rms_mm"], "calibration_intrinsics_max_delta_px": intrinsics_delta_px, "right_base_to_left_base": right_to_left.astype(float).tolist()}
    return left, right, calibration, report


def rotvec_transform(pose: np.ndarray) -> np.ndarray:
    """Convert an RTDE [x,y,z,rx,ry,rz] pose to a homogeneous matrix."""
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    theta = float(np.linalg.norm(pose[3:]))
    transform = np.eye(4, dtype=np.float64); transform[:3, 3] = pose[:3]
    if theta > 1e-12:
        axis = pose[3:] / theta
        skew = np.asarray(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]), (-axis[1], axis[0], 0.)))
        transform[:3, :3] = np.eye(3) + np.sin(theta) * skew + (1. - np.cos(theta)) * (skew @ skew)
    return transform




def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return ((transform[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + transform[:3, 3]).astype(np.float32)


def arm_surface_points(q: np.ndarray, base_to_left: np.ndarray, samples: dict[str, np.ndarray]) -> np.ndarray:
    transforms = link_and_collision_transforms(q)
    return np.concatenate([transform_points(samples[name], base_to_left @ transforms[name]) for name in samples]).astype(np.float32)


def cap(points: np.ndarray, colors: np.ndarray, limit: int) -> tuple[list, list]:
    if len(points) > limit:
        index = np.linspace(0, len(points) - 1, limit, dtype=np.int64); points, colors = points[index], colors[index]
    return points.astype(float).tolist(), colors.astype(int).tolist()


def tabletop_obb(points: np.ndarray, normal: np.ndarray, offset: float) -> UprightOBB | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3: return None
    normal = np.asarray(normal, dtype=np.float64); normal /= max(np.linalg.norm(normal), 1e-12)
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]; first = np.cross(seed, normal); first /= np.linalg.norm(first); second = np.cross(normal, first)
    basis = np.column_stack((first, second)); planar = points @ basis; centered = planar - planar.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1)); rotation = np.column_stack((basis @ vectors[:, ::-1], normal))
    if np.linalg.det(rotation) < 0: rotation[:, 1] *= -1
    local = points @ rotation; low, high = local.min(0), local.max(0); low[2] = min(low[2], -float(offset)); extents = np.maximum(high - low, 1e-4); center = rotation @ ((low + high) * .5)
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float64)
    corners = center + (0.5 * signs * extents) @ rotation.T
    return UprightOBB(center.astype(np.float32), rotation.astype(np.float32), extents.astype(np.float32), corners.astype(np.float32), int(len(points)))


def dbscan_clusters(
    points: np.ndarray,
    *,
    eps_m: float,
    min_samples: int,
    min_cluster_points: int,
) -> list[np.ndarray]:
    """Return exact point-wise DBSCAN clusters.

    The previous implementation linked *occupied voxel cells*.  Two points in
    adjacent 18 mm cells could be more than 60 mm apart yet still be joined,
    which makes a point chain bridge nearby instruments.  Here every graph edge
    is an actual Euclidean distance no greater than ``eps_m``; sparse bridge
    points are additionally rejected because they are not DBSCAN core points.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return []
    if eps_m <= 0.0:
        raise ValueError("cluster eps must be positive")
    if min_samples < 1:
        raise ValueError("cluster min samples must be at least one")
    neighbours = cKDTree(points).query_ball_point(points, r=float(eps_m), workers=-1)
    core = np.fromiter((len(group) >= int(min_samples) for group in neighbours), dtype=bool, count=len(points))
    labels = np.full(len(points), -1, dtype=np.int32)
    cluster_id = 0
    for seed in np.flatnonzero(core):
        if labels[seed] >= 0:
            continue
        labels[seed] = cluster_id
        queue: deque[int] = deque([int(seed)])
        while queue:
            point_index = queue.popleft()
            for neighbour in neighbours[point_index]:
                if labels[neighbour] >= 0:
                    continue
                labels[neighbour] = cluster_id
                if core[neighbour]:
                    queue.append(neighbour)
        cluster_id += 1
    return [indices for label in range(cluster_id) if len(indices := np.flatnonzero(labels == label)) >= int(min_cluster_points)]


def raw_depth_support_mask(depth_m: np.ndarray, robot_keep_mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    """Permit completed depth only near a measured, non-robot D455 sample."""
    raw_depth = np.asarray(depth_m, dtype=np.float32)
    keep = np.asarray(robot_keep_mask, dtype=bool)
    if raw_depth.shape != keep.shape:
        raise ValueError(f"raw depth shape {raw_depth.shape} and keep-mask shape {keep.shape} differ")
    if radius_pixels < 0:
        raise ValueError("raw depth support pixels must be non-negative")
    measured_environment = keep & np.isfinite(raw_depth) & (raw_depth > 0.0)
    if radius_pixels == 0:
        return measured_environment
    # A full 3x3 footprint avoids directionally biasing horizontal/vertical
    # gaps.  Reapply ``keep`` so completed robot pixels never gain support from
    # adjacent environment pixels.
    supported = binary_dilation(measured_environment, structure=np.ones((3, 3), dtype=bool), iterations=int(radius_pixels))
    return keep & supported


@dataclass
class _ObstacleTrack:
    center: np.ndarray
    box: UprightOBB
    consecutive_hits: int = 1
    missed_frames: int = 0


class TemporalOBBGate:
    """Show only geometrically matched OBBs observed in consecutive frames."""

    def __init__(self, *, required_hits: int, match_distance_m: float) -> None:
        if required_hits < 1:
            raise ValueError("temporal persistence frames must be at least one")
        if match_distance_m <= 0.0:
            raise ValueError("temporal track distance must be positive")
        self.required_hits = int(required_hits)
        self.match_distance_m = float(match_distance_m)
        self._tracks: list[_ObstacleTrack] = []

    def update(self, boxes: list[UprightOBB]) -> list[UprightOBB]:
        for track in self._tracks:
            track.missed_frames += 1
        candidates = sorted(
            (float(np.linalg.norm(track.center - box.center)), track_index, box_index)
            for track_index, track in enumerate(self._tracks)
            for box_index, box in enumerate(boxes)
            if np.linalg.norm(track.center - box.center) <= self.match_distance_m
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for _, track_index, box_index in candidates:
            if track_index in used_tracks or box_index in used_boxes:
                continue
            track = self._tracks[track_index]
            # A missed frame breaks persistence: stale boxes cannot become
            # confirmed merely by reappearing later.
            track.consecutive_hits = track.consecutive_hits + 1 if track.missed_frames == 1 else 1
            track.missed_frames = 0
            track.center = np.asarray(boxes[box_index].center, dtype=np.float64)
            track.box = boxes[box_index]
            used_tracks.add(track_index)
            used_boxes.add(box_index)
        for box_index, box in enumerate(boxes):
            if box_index not in used_boxes:
                self._tracks.append(_ObstacleTrack(np.asarray(box.center, dtype=np.float64), box))
        # Retain a track through one missed frame only for association.  It is
        # never emitted while missed, so no stale geometry is displayed.
        self._tracks = [track for track in self._tracks if track.missed_frames <= 1]
        return [track.box for track in self._tracks if track.missed_frames == 0 and track.consecutive_hits >= self.required_hits]


def combine_depth(depths: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(depths[0], dtype=np.float32)
    for depth in depths:
        result = np.where((result > 0) & (depth > 0), np.minimum(result, depth), np.maximum(result, depth))
    return result


def main() -> None:
    args = parse_args(); left, right, calibration, report = load_arms(args)
    project, _ = load_project_config(args.project_config)
    side_transform, side_K, side_serial, side_report = load_calibration(resolve_repo_path(project["safety"]["side_calibration"]))
    if side_report.get("parent_frame") != "left_base":
        raise ValueError("side calibration must map the camera into left_base")
    if args.side_serial and args.side_serial != side_serial:
        raise ValueError("Side camera serial differs from calibration")
    calibrations = {"front": calibration, "side": CameraCalibration("side", side_K, side_transform)}
    print(json.dumps({"mode": "read-only", "calibration": report, "left_ip": left.ip, "right_ip": right.ip}, indent=2))
    from rtde_receive import RTDEReceiveInterface
    # Model construction/download is intentionally opt-in: on a CPU-only host
    # it is not a real-time operation.  With --lingbot-depth it refines the
    # front D455 map before the environment cloud is created.
    refiner = create_lingbot_depth_refiner_from_args(args)
    if refiner is not None:
        print(json.dumps({"depth_refinement": "LingBot-Depth", "model": refiner.model_id, "device": str(refiner.device), "fp16": refiner.use_fp16, "views": refiner.camera_names}, indent=2))
    else:
        print(json.dumps({"depth_refinement": "raw D455 depth (--no-lingbot-depth selected)"}, indent=2))
    samples = collision_surface_samples(samples_per_face=args.samples_per_face)
    latest: dict = {"frame": 0, "status": "initializing", "environment": {"points": [], "colors": []}, "robot": {"points": [], "colors": []}, "obbs": []}; lock = threading.Lock(); stop = threading.Event()

    def process() -> None:
        nonlocal latest
        receivers = {left.name: RTDEReceiveInterface(left.ip), right.name: RTDEReceiveInterface(right.ip)}
        source = RealSenseD435iSource(cameras=(D435iCameraConfig("front", args.front_serial or report["camera_serial"]), D435iCameraConfig("side", args.side_serial or side_serial)), width=args.width, height=args.height, fps=args.fps)
        left_gripper: ReadOnlyRobotiq2FPosition | None = None; right_gripper: ReadOnlyRobotiq2FPosition | None = None
        finger_angle = float(np.clip(args.left_finger_angle_rad, 0., .8)); right_finger_angle = finger_angle
        gripper_status = f"2F-85 fixed angle {finger_angle:.3f} rad"
        obstacle_gate = TemporalOBBGate(
            required_hits=args.temporal_persistence_frames,
            match_distance_m=args.temporal_track_distance_m,
        )
        try:
            left_gripper = ReadOnlyRobotiq2FPosition(left.ip, port=args.left_robotiq_read_port)
            right_gripper = ReadOnlyRobotiq2FPosition(right.ip, port=args.left_robotiq_read_port)
            gripper_status = "2F-85 GET POS connected (read-only)"
        except Exception as exc:
            if left_gripper is not None: left_gripper.close()
            for receiver in receivers.values(): receiver.disconnect()
            raise RuntimeError("Both 2F-85 state readers must connect before perception starts") from exc
        frame_no = 0
        try:
            source.start()
            for name, info in source.get_color_calibrations().items():
                # Use actual color-stream intrinsics for the requested resolution.
                calibrations[name] = CameraCalibration(name, np.asarray(info["intrinsics"]), calibrations[name].camera_to_world)
            for _ in range(max(1, args.warmup_frames)): source.read()
            while not stop.is_set():
                began = time.perf_counter(); q0 = {a.name: np.asarray(receivers[a.name].getActualQ(), dtype=np.float64) for a in (left, right)}; frames = source.read(); captured = frames["front"]; q1 = {a.name: np.asarray(receivers[a.name].getActualQ(), dtype=np.float64) for a in (left, right)}
                stamps = [frame.host_timestamp_ns for frame in frames.values()]
                if (max(stamps) - min(stamps)) / 1e9 > project["safety"]["max_camera_skew_s"]:
                    raise RuntimeError("Front/side capture skew exceeded the configured limit")
                robots, robot_col, depths = [], [], []
                if left_gripper is not None:
                    try:
                        closing = left_gripper.closing_fraction(); finger_angle = .8 * closing; gripper_status = f"2F-85 GET POS={closing:.3f}, q={finger_angle:.3f} rad"
                    except Exception as exc:
                        raise RuntimeError("Left gripper state unavailable") from exc
                if right_gripper is not None:
                    try: right_finger_angle = .8 * right_gripper.closing_fraction()
                    except Exception as exc: raise RuntimeError("Right gripper state unavailable") from exc
                if left_gripper is None or right_gripper is None:
                    raise RuntimeError("Both measured gripper states are required")
                for arm, color in ((left, np.array((255, 70, 45), np.uint8)), (right, np.array((255, 165, 45), np.uint8))):
                    q = (q0[arm.name] + q1[arm.name]) * .5
                    if q.shape != (6,) or not np.isfinite(q).all(): raise RuntimeError(f"{arm.name} invalid RTDE joint state: {q}")
                    arm_points = arm_surface_points(q, arm.base_to_left, samples)
                    # getActualTCPPose is a receive-only RTDE query and tracks any
                    # UR-side active TCP setting used for the installed tool.
                    tcp = np.asarray(receivers[arm.name].getActualTCPPose(), dtype=np.float64)
                    if tcp.shape != (6,) or not np.isfinite(tcp).all(): raise RuntimeError(f"{arm.name} invalid RTDE TCP pose: {tcp}")
                    local_tool = robotiq_2f85_points_in_active_tcp(finger_angle_rad=finger_angle if arm.name == left.name else right_finger_angle)
                    tool_points = transform_points(local_tool, arm.base_to_left @ rotvec_transform(tcp) @ arm.tool_from_tcp)
                    points = np.concatenate((arm_points, tool_points)); robots.append(points); robot_col.append(np.tile(color, (len(points), 1)))
                robot = np.concatenate(robots); rcol = np.concatenate(robot_col)
                refined_frames = frames if refiner is None else {f.camera_name: f for f in refiner.refine(tuple(frames.values()), calibrations)}
                clouds, colors, measured_surface = [], [], []
                support_pixels = 0
                for name, raw in frames.items():
                    view = calibrations[name]
                    refined = refined_frames[name]
                    rendered = render_surface_points_depth(robot, view.camera_to_world, view.intrinsics, width=raw.rgb.shape[1], height=raw.rgb.shape[0], splat_radius_pixels=3)
                    raw_keep = robot_depth_keep_mask(raw.depth_m, rendered, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels)
                    refined_keep = robot_depth_keep_mask(refined.depth_m, rendered, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels)
                    raw_support = raw_depth_support_mask(raw.depth_m, raw_keep, args.raw_depth_support_pixels)
                    keep = refined_keep & raw_support
                    cloud, color = depth_to_world_points(refined, view, stride=args.point_stride, max_depth=2.5, keep_mask=keep)
                    clouds.append(cloud); colors.append(color); support_pixels += int(raw_support.sum())
                    v, u = np.nonzero(~raw_keep & (raw.depth_m > 0))
                    pixels = np.c_[u, v][::max(1, args.point_stride)]
                    observed, visible = _lift_measured_tracks(raw.depth_m, rendered, pixels, np.ones(len(pixels), bool), view, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance)
                    measured_surface.append(observed[visible])
                env, ecol = np.concatenate(clouds), np.concatenate(colors)
                surface_points = np.concatenate(measured_surface)
                xmin,xmax,ymin,ymax,zmin,zmax = args.workspace_bounds; mask = (env[:,0]>=xmin)&(env[:,0]<=xmax)&(env[:,1]>=ymin)&(env[:,1]<=ymax)&(env[:,2]>=zmin)&(env[:,2]<=zmax); env,ecol = env[mask],ecol[mask]
                local = env
                if len(local) < 80: raise RuntimeError("not enough robot-filtered points to fit the tabletop")
                normal, offset, _ = estimate_dominant_plane(local, threshold=args.plane_threshold_m, ransac_iterations=300)
                # ``left_base`` need not have its +Z along the physical table
                # normal (this dual setup is not a world frame).  The only
                # stable sign convention is: positive height faces the fixed
                # overhead camera, so tabletop objects rise toward it.
                camera_side = float(calibration.camera_to_world[:3, 3] @ normal + offset)
                if camera_side < 0:
                    normal, offset = -normal, -offset
                height = env.astype(np.float64) @ normal + offset; candidates = (height >= args.min_height_m)&(height <= args.max_height_m); op, oh = env[candidates], height[candidates]
                if len(op) > 1 and args.outlier_neighbour_m > 0:
                    distances, _ = cKDTree(op).query(op, k=2, workers=-1); valid = distances[:,1] <= args.outlier_neighbour_m; op,oh = op[valid],oh[valid]
                candidate_boxes = []; floating_components = 0; connected_components = 0
                for indices in dbscan_clusters(op, eps_m=args.cluster_eps_m, min_samples=args.cluster_min_samples, min_cluster_points=args.min_cluster_points):
                    if oh[indices].min() <= args.attachment_distance_m:
                        connected_components += 1
                        box = tabletop_obb(op[indices], normal, offset)
                        if box is not None: candidate_boxes.append(expanded_obb(box, args.box_margin_m))
                    else:
                        floating_components += 1
                display_boxes = obstacle_gate.update(candidate_boxes)
                # Safety sees new candidates immediately and retains one missed frame.
                boxes = [track.box for track in obstacle_gate._tracks]
                frame_no += 1; ep,ec = cap(env,ecol,args.display_max_points); rp,rc = cap(robot,rcol,args.display_max_points//2)
                qdelta = max(float(np.max(np.abs(q1[a.name]-q0[a.name]))) for a in (left,right))
                diagnostics = {"environment_points": int(len(env)), "object_candidate_points": int(len(op)), "table_connected_components": connected_components, "floating_components": floating_components, "candidate_obbs": len(candidate_boxes), "confirmed_obbs": len(display_boxes), "raw_depth_support_pixels": args.raw_depth_support_pixels, "raw_depth_supported_pixels": support_pixels, "table_normal_left_base": normal.astype(float).tolist(), "table_offset_m": float(offset), "depth_refinement": "lingbot-depth" if refiner is not None else "raw-d455", "depth_refinement_seconds": float(refiner.last_inference_seconds) if refiner is not None else 0.0}
                refinement = f" · LingBot {refiner.last_inference_seconds:.2f}s" if refiner is not None else ""
                ok, encoded_rgb = cv2.imencode(".jpg", captured.rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("front RGB JPEG encoding failed")
                snapshot = {
                    "schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                    "frame": frame_no, "host_timestamp_ns": int(min(stamps)),
                    "status": f"{time.perf_counter()-began:.2f}s{refinement} · {gripper_status} · 双臂 qΔ {qdelta:.4f} rad · {len(boxes)} 个桌面连接障碍物",
                    "qpos": {a.name: ((q0[a.name] + q1[a.name]) * .5).astype(float).tolist() for a in (left, right)},
                    "gripper_position": {"left": float(finger_angle / .8), "right": float(right_finger_angle / .8)},
                    "current_surface": surface_points.tolist(),
                    "surface_source": "measured_depth_fk_gated",
                    "camera_names": ["front", "side"],
                    "rgb_front_jpeg_base64": base64.b64encode(encoded_rgb.tobytes()).decode("ascii"),
                    "environment": {"points":ep,"colors":ec}, "robot":{"points":rp,"colors":rc},
                    "obbs":[{
                        "center": b.center.astype(float).tolist(), "axes": b.rotation.astype(float).tolist(),
                        "half_sizes": (b.extents * .5).astype(float).tolist(), "corners":b.corners.astype(float).tolist(),
                        "point_count": int(b.point_count),
                    } for b in boxes],
                    "calibration": report, "diagnostics": diagnostics,
                }
                with lock: latest = snapshot
                # A one-shot diagnostic must still observe enough frames to
                # exercise the configured temporal persistence gate.  The old
                # one-frame behavior could never emit an OBB with the default
                # required_hits=3 and therefore produced a misleading zero.
                if args.once and frame_no >= args.temporal_persistence_frames:
                    break
        except Exception as exc:
            with lock: latest = {**latest, "status": f"error: {type(exc).__name__}: {exc}"}
            raise
        finally:
            source.stop()
            for receiver in receivers.values(): receiver.disconnect()
            if left_gripper is not None: left_gripper.close()
            if right_gripper is not None: right_gripper.close()

    worker = threading.Thread(target=process, daemon=True); worker.start()
    if args.once:
        worker.join()
        if str(latest["status"]).startswith("error") or "schema" not in latest:
            raise RuntimeError(f"Obstacle acquisition failed: {latest['status']}")
        print(json.dumps({"schema": latest.get("schema"), "coordinate_frame": latest.get("coordinate_frame"), "frame": latest["frame"], "host_timestamp_ns": latest.get("host_timestamp_ns"), "status": latest["status"], "environment_points": len(latest["environment"]["points"]), "robot_points": len(latest["robot"]["points"]), "obbs": latest["obbs"], "diagnostics": latest.get("diagnostics", {}), "calibration": latest.get("calibration", {})}, ensure_ascii=False, indent=2)); return latest

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/snapshot.json"):
                with lock: data = json.dumps(latest, separators=(",", ":")).encode()
                if "gzip" in self.headers.get("Accept-Encoding", ""):
                    data = gzip.compress(data, compresslevel=1); compressed = True
                else: compressed = False
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
                if compressed: self.send_header("Content-Encoding", "gzip")
                self.end_headers(); self.wfile.write(data); return
            if self.path.startswith("/plotly.min.js"):
                import plotly
                payload = (Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js").read_bytes(); self.send_response(200); self.send_header("Content-Type", "application/javascript"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload); return
            payload = VIEWER_HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def log_message(self, *_): pass
    server = ThreadingHTTPServer((args.bind_host, args.port), Handler); url = f"http://127.0.0.1:{args.port}"; print(f"[model] {url}  (Ctrl+C to stop)")
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: stop.set(); server.shutdown()


if __name__ == "__main__":
    main()
