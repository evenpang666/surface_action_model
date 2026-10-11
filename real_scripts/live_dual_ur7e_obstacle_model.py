#!/usr/bin/env python3
"""Read-only real-time tabletop obstacle model for two UR7e arms and front and side D455 cameras.

The model and fixed-point FK share the validated dual-arm front/side
calibration graph. All output is expressed in ``left_base``.
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
from dataclasses import dataclass, field
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser
import socket
from urllib.parse import parse_qs, urlsplit

import numpy as np
import yaml
import cv2
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.lingbot_depth import add_lingbot_depth_cli_args, create_lingbot_depth_refiner_from_args
from real_scripts.camera_geometry import project_pixels
from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame, depth_to_world_points, robot_depth_keep_mask
from real_scripts.reconstruct_realsense_pointcloud import estimate_dominant_plane
from real_scripts.ur7e_collision_mesh import (
    collision_surface_samples,
    link_and_collision_transforms,
    render_surface_points_depth,
    flange_transform,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource
from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_points_in_active_tcp
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path
from safety_module.dual_camera_calibration import calibration_from_project, load_camera_file, transform_error
from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.preprocess_left_vr_hdf5_cotracker_observations import _lift_measured_tracks


VIEWER_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Dual UR7e obstacle model</title>
<script src="/plotly.min.js"></script><style>html,body,#scene{margin:0;width:100%;height:100%;overflow:hidden;background:#080a0d;color:#e6edf3;font:14px Arial}#hud{position:fixed;z-index:2;left:12px;top:10px;background:#000c;padding:10px 13px;border-radius:7px;line-height:1.6;pointer-events:none}#pause,#focus,#robot-toggle{pointer-events:auto;margin-top:6px;padding:4px 9px;border:1px solid #6c7784;border-radius:4px;background:#17212b;color:#e6edf3;cursor:pointer}.robot{color:#ff6b55}.obstacle{color:#45ff9b}.warning{color:#ffd166}</style></head>
<body><div id="hud"><b>双 UR7e 实时桌面障碍物模型（桌面坐标）</b><br><span id="s">等待首帧…</span><br><span id="timing"></span><br><span class="robot">红/橙：双 Robotiq 2F-85 与双臂</span> · <span class="obstacle">绿：双臂工作区内的桌面连接障碍物 OBB</span><br><span class="warning">双臂工作区外桌面点仅供显示；仅 RTDE 读取，不会发送运动或夹爪命令。</span><br><button id="focus" type="button" aria-pressed="true">当前：桌面聚焦（F）</button> <button id="robot-toggle" type="button" aria-pressed="true">隐藏机械臂（R）</button> <button id="pause" type="button" aria-pressed="false">暂停显示（空格）</button></div><div id="scene"></div>
<script>
const G=document.getElementById('scene'),S=document.getElementById('s'),T=document.getElementById('timing'),P=document.getElementById('pause'),F=document.getElementById('focus'),R=document.getElementById('robot-toggle');
const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
let paused=false,focusTable=true,showRobot=true,tableReference=null,lastSnapshot=null;
function setPaused(value){paused=value;P.textContent=paused?'继续显示（空格）':'暂停显示（空格）';P.setAttribute('aria-pressed',String(paused))}
function setFocus(value){focusTable=value;F.textContent=value?'当前：桌面聚焦（F）':'当前：全景（F）';F.setAttribute('aria-pressed',String(value));if(lastSnapshot)draw(lastSnapshot)}
function setShowRobot(value){showRobot=value;R.textContent=value?'隐藏机械臂（R）':'显示机械臂（R）';R.setAttribute('aria-pressed',String(value));if(lastSnapshot)draw(lastSnapshot)}
P.addEventListener('click',()=>setPaused(!paused));
F.addEventListener('click',()=>setFocus(!focusTable));
R.addEventListener('click',()=>setShowRobot(!showRobot));
document.addEventListener('keydown',e=>{if(e.repeat)return;if(e.code==='Space'){e.preventDefault();setPaused(!paused)}else if(e.code==='KeyF'){e.preventDefault();setFocus(!focusTable)}else if(e.code==='KeyR'){e.preventDefault();setShowRobot(!showRobot)}});
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function tableFrame(d){
  const raw=d.diagnostics?.table_normal_left_base, offset=d.diagnostics?.table_offset_m;
  if(!Array.isArray(raw)||raw.length!==3||!raw.every(Number.isFinite)||!Number.isFinite(offset))return null;
  const length=Math.hypot(...raw);
  if(length<1e-8)return null;
  const z=raw.map(v=>v/length);
  // Project left_base X onto the tabletop, then complete a right-handed frame.
  if(!tableReference)tableReference=Math.abs(z[0])<.95?[1,0,0]:[0,1,0];
  const reference=tableReference;
  const projection=dot(reference,z);
  const projected=reference.map((v,i)=>v-projection*z[i]);
  const projectedLength=Math.hypot(...projected);
  if(projectedLength<1e-8)return null;
  const x=projected.map(v=>v/projectedLength);
  const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
  return {x,y,z,offset:offset/length};
}
function tablePoint(p,frame){
  if(!frame)return p;
  return [dot(p,frame.x),dot(p,frame.y),dot(p,frame.z)+frame.offset];
}
function pts(g,name,size,frame){
  const points=g.points.map(p=>tablePoint(p,frame));
  return {type:'scatter3d',mode:'markers',name,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size,opacity:.8,color:g.colors.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`)},hoverinfo:'skip'};
}
function boxes(items,frame){
  const result=[];
  for(let k=0;k<items.length;k++){
    const corners=items[k].corners.map(p=>tablePoint(p,frame)),x=[],y=[],z=[];
    for(const [a,b] of E){x.push(corners[a][0],corners[b][0],null);y.push(corners[a][1],corners[b][1],null);z.push(corners[a][2],corners[b][2],null)}
    result.push({type:'scatter3d',mode:'lines',name:`障碍物 ${k+1}`,x,y,z,line:{color:'#45ff9b',width:6},hoverinfo:'skip'});
  }
  return result;
}
function quantile(sorted,fraction){const index=(sorted.length-1)*fraction,low=Math.floor(index),high=Math.ceil(index);return sorted[low]+(sorted[high]-sorted[low])*(index-low)}
function tableBounds(environment,frame){
  if(!frame)return null;
  const xs=[],ys=[];
  for(const point of environment.points){
    const p=tablePoint(point,frame);
    if(Math.abs(p[2])<=.025){xs.push(p[0]);ys.push(p[1])}
  }
  if(xs.length<50)return null;
  xs.sort((a,b)=>a-b);ys.sort((a,b)=>a-b);
  const xmin=quantile(xs,.01),xmax=quantile(xs,.99),ymin=quantile(ys,.01),ymax=quantile(ys,.99);
  const span=Math.max(xmax-xmin,ymax-ymin,.45)*1.15;
  const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
  return {x:[cx-span/2,cx+span/2],y:[cy-span/2,cy+span/2],z:[-.10,span-.10],width:xmax-xmin,height:ymax-ymin};
}
function overviewBounds(d,frame){
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  function include(point){
    const p=tablePoint(point,frame);
    for(let i=0;i<3;i++){if(p[i]<low[i])low[i]=p[i];if(p[i]>high[i])high[i]=p[i]}
  }
  for(const point of d.environment.points)include(point);
  if(showRobot)for(const point of d.robot.points)include(point);
  for(const box of d.obbs)for(const corner of box.corners)include(corner);
  if(!low.every(Number.isFinite)||!high.every(Number.isFinite))return null;
  const span=Math.max(...high.map((value,i)=>value-low[i]),.6)*1.1;
  const ranges=low.map((value,i)=>[(value+high[i]-span)/2,(value+high[i]+span)/2]);
  return {x:ranges[0],y:ranges[1],z:ranges[2]};
}
function draw(d){
  lastSnapshot=d;
  const frame=tableFrame(d);
  const bounds=tableBounds(d.environment,frame);
  const coverage=bounds?` · 桌面点云覆盖 ${bounds.width.toFixed(2)} × ${bounds.height.toFixed(2)} m`:'';
  const outside=d.diagnostics?.viewer_outside_workspace_points;
  const outsideText=Number.isFinite(outside)?` · 双臂工作区外桌面/物体点 ${outside}`:'';
  const left=d.diagnostics?.left_workspace_points,right=d.diagnostics?.right_workspace_points;
  const workspaces=Number.isFinite(left)&&Number.isFinite(right)?` · 左/右工作区 ${left}/${right} 点`:'';
  S.textContent=`帧 ${d.frame} · ${d.status} · 环境 ${d.environment.points.length} 点 · 双臂 ${d.robot.points.length} 点${showRobot?'':'（已隐藏）'} · ${d.obbs.length} 个障碍物${workspaces}${coverage}${outsideText}`;
  const timing=d.diagnostics||{},stages=[['采集','capture_seconds'],['深度','depth_stage_seconds'],['点云','point_cloud_seconds'],['场景','scene_analysis_seconds'],['跟踪','box_tracking_seconds']];
  const elapsed=Number.isFinite(timing.processing_seconds)?`检测 ${Math.round(timing.processing_seconds*1000)} ms`:'';
  const interval=Number.isFinite(timing.frame_interval_seconds)?` · 帧间隔 ${Math.round(timing.frame_interval_seconds*1000)} ms`:'';
  T.textContent=elapsed+interval+stages.filter(([,key])=>Number.isFinite(timing[key])).map(([label,key])=>` · ${label} ${Math.round(timing[key]*1000)} ms`).join('');
  const axis=frame?'桌面':'left_base';
  const focused=focusTable&&bounds, eye=focused?{x:.9,y:-.9,z:1.65}:{x:1.5,y:-1.5,z:1.15};
  const scene={bgcolor:'#080a0d',aspectmode:'cube',camera:{eye,up:{x:0,y:0,z:1},projection:{type:'orthographic'}},
    xaxis:{title:`${axis} X (m)`},yaxis:{title:`${axis} Y (m)`},zaxis:{title:`${axis} Z (m)`}};
  const ranges=focused?bounds:overviewBounds(d,frame);
  if(ranges){scene.xaxis.range=ranges.x;scene.yaxis.range=ranges.y;scene.zaxis.range=ranges.z}
  const traces=[pts(d.environment,'environment',.7,frame)];
  if(showRobot)traces.push(pts(d.robot,'dual robots',1.1,frame));
  traces.push(...boxes(d.obbs,frame));
  return Plotly.react(G,traces,{
    uirevision:focusTable?'table-focus-v3':'table-overview-v3',showlegend:false,paper_bgcolor:'#080a0d',margin:{l:0,r:0,t:0,b:0},scene
  },{displaylogo:false,responsive:true,scrollZoom:true});
}
async function poll(){if(!paused)try{let r=await fetch('/snapshot.json?viewer=1&'+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();if(!paused)await draw(d)}catch(e){if(!paused)S.textContent='读取快照失败，正在重试：'+e}setTimeout(poll,900)}
poll();
</script></body></html>"""


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


@dataclass(frozen=True)
class SnapshotDisplay:
    """One frame of display data, converted only for a full or viewer request."""

    environment_points: np.ndarray
    environment_colors: np.ndarray
    all_points: np.ndarray
    all_colors: np.ndarray
    table_band: np.ndarray
    robot_points: np.ndarray
    robot_colors: np.ndarray
    surface_points: np.ndarray
    front_rgb: np.ndarray
    max_points: int


def expanded_obb(box: UprightOBB, margin_m: float) -> UprightOBB:
    margin = max(float(margin_m), 0.0)
    extents = np.asarray(box.extents, dtype=np.float32) + 2.0 * margin
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float32)
    corners = box.center[None, :] + (0.5 * signs * extents[None, :]) @ box.rotation.T
    return UprightOBB(box.center, box.rotation, extents, corners.astype(np.float32), box.point_count)


def safety_obbs_with_candidates(confirmed: list[UprightOBB],
                                candidates: list[UprightOBB],
                                clusters: list[np.ndarray]) -> tuple[list[UprightOBB], int]:
    """Publish an OBB for every observed cluster, including unconfirmed ones.

    A merged candidate already covered by historical boxes does not introduce
    one large replacement box. Any observed point outside those boxes brings
    its candidate box into the safety scene immediately.
    """
    if len(candidates) != len(clusters):
        raise ValueError("Each candidate OBB needs its observed cluster")
    published = list(confirmed)
    supplemental = 0
    for candidate, points in zip(candidates, clusters):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        covered = np.zeros(len(points), dtype=bool)
        for box in published:
            local = (points-box.center) @ box.rotation
            covered |= np.all(np.abs(local) <= box.extents/2 + 1e-6, axis=1)
        if not np.all(covered):
            published.append(candidate)
            supplemental += 1
    return published, supplemental


class ReadOnlyRobotiq2FPosition:
    """URCap position reader that is physically incapable of sending a SET command."""

    def __init__(self, host: str, *, port: int, timeout_s: float = .25) -> None:
        self.host, self.port, self.timeout_s = host, int(port), float(timeout_s)
        if not 0 < self.timeout_s <= 2.:
            raise ValueError("Robotiq read timeout must be in (0, 2] seconds")
        self.socket = None
        self.buffer = b""
        self.closed = False
        self._connect()

    def _connect(self) -> None:
        connection = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        connection.settimeout(self.timeout_s)
        self.socket, self.buffer = connection, b""

    def _readline(self) -> bytes:
        while b"\n" not in self.buffer:
            chunk = self.socket.recv(128)
            if not chunk:
                raise ConnectionError("Robotiq closed the GET POS connection")
            self.buffer += chunk
            if len(self.buffer) > 128:
                raise RuntimeError("Robotiq GET POS response exceeds 128 bytes")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.strip()

    def closing_fraction(self) -> float:
        if self.closed:
            raise RuntimeError("Robotiq GET POS reader is closed")
        if self.socket is None:
            self._connect()
        try:
            return self._query_position()
        except OSError as first:
            # URCap connections can be dropped while the detector is running.
            # Reconnect once; never substitute a previous position for a live one.
            self._disconnect()
            try:
                self._connect()
                return self._query_position()
            except (OSError, RuntimeError, ValueError) as second:
                self._disconnect()
                raise RuntimeError(
                    f"GET POS {self.host}:{self.port} failed after reconnect "
                    f"({type(first).__name__}: {first}; {type(second).__name__}: {second})"
                ) from second

    def _query_position(self) -> float:
        self.socket.sendall(b"GET POS\n")
        fields = self._readline().decode("ascii", errors="replace").split()
        if len(fields) != 2 or fields[0] != "POS":
            raise RuntimeError(f"unexpected Robotiq GET POS response: {' '.join(fields)!r}")
        position = int(fields[1])
        if not 0 <= position <= 255:
            raise RuntimeError(f"Robotiq GET POS outside [0, 255]: {position}")
        return position / 255.0

    def _disconnect(self) -> None:
        connection, self.socket = self.socket, None
        self.buffer = b""
        if connection is not None:
            connection.close()

    def close(self) -> None:
        self.closed = True
        self._disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--front-serial", default=None, help="Override front D455 serial; default is hardware config.")
    parser.add_argument("--side-serial", default=None)
    parser.add_argument("--width", type=int, default=1280); parser.add_argument("--height", type=int, default=720); parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-width", type=int, default=None, help="Optional raw depth width; default selects a supported depth profile at --fps.")
    parser.add_argument("--depth-height", type=int, default=None, help="Specify together with --depth-width; depth is aligned to the RGB resolution.")
    parser.add_argument("--warmup-frames", type=int, default=20); parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--samples-per-face", type=int, default=16); parser.add_argument("--absolute-tolerance-m", type=float, default=.015); parser.add_argument("--relative-tolerance", type=float, default=.015); parser.add_argument("--dilation-pixels", type=int, default=3)
    add_lingbot_depth_cli_args(parser)
    # Two fixed D455 views; wrist D435i cameras belong to policy observations.
    parser.set_defaults(lingbot_camera_names=("front", "side"), lingbot_depth=True)
    parser.add_argument("--workspace-bounds", nargs=6, type=float, default=(-0.85, 0.85, -0.85, 1.20, -0.12, 0.70), metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), help="Local workspace bounds for each arm; both are included in tabletop detection.")
    parser.add_argument("--right-workspace-bounds", nargs=6, type=float, default=None, metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), help="Optional right_base bounds; defaults to --workspace-bounds.")
    parser.add_argument("--plane-threshold-m", type=float, default=.012); parser.add_argument("--min-height-m", type=float, default=.020); parser.add_argument("--max-height-m", type=float, default=.35); parser.add_argument("--outlier-neighbour-m", type=float, default=.020)
    parser.add_argument("--cluster-eps-m", type=float, default=.010, help="Exact point-to-point DBSCAN radius; unlike voxel adjacency, this is a physical distance.")
    parser.add_argument("--cluster-min-samples", type=int, default=6, help="DBSCAN core-neighbour count, including the point itself.")
    parser.add_argument("--min-cluster-points", type=int, default=80); parser.add_argument("--attachment-distance-m", type=float, default=.050); parser.add_argument("--box-margin-m", type=float, default=.010)
    parser.add_argument("--raw-depth-support-pixels", type=int, default=2, help="LingBot pixels must be within this many pixels of raw, non-robot D455 depth; 0 permits raw pixels only.")
    parser.add_argument("--temporal-persistence-frames", type=int, default=3, help="Consecutive observations required before an obstacle OBB is shown; use 1 to disable.")
    parser.add_argument("--temporal-track-distance-m", type=float, default=.035, help="Maximum centre displacement when associating an OBB across frames.")
    parser.add_argument("--obb-point-change-distance-m", type=float, default=.015,
                        help="Nearest-neighbour point displacement counted as cluster change (m).")
    parser.add_argument("--obb-point-change-fraction", type=float, default=.25,
                        help="Fraction of changed cluster points required before an OBB can update.")
    parser.add_argument("--obb-history-frames", type=int, default=10,
                        help="Keep only this many recent cluster observations per OBB; full snapshots are not cached.")
    parser.add_argument("--obb-change-confirm-frames", type=int, default=3,
                        help="Consecutive changed point clusters required before updating a tracked OBB; default compares the current frame with the previous two.")
    parser.add_argument("--obb-merge-overlap-ratio", type=float, default=.50,
                        help="Merge published OBBs when intersection exceeds this fraction of the smaller OBB volume.")
    parser.add_argument("--obb-removal-confirm-frames", type=int, default=3,
                        help="Consecutive frames with measured free space behind a missing cluster before removing its OBB.")
    parser.add_argument("--left-tool-from-tcp", type=Path, default=None, help="Optional JSON/YAML ^active_tcpT_left_tool_urdf_root; identity for the bundled URDF.")
    parser.add_argument("--right-tool-from-tcp", type=Path, default=None, help="Optional JSON/YAML ^active_tcpT_right_tool_urdf_root; identity for the bundled URDF.")
    parser.add_argument("--left-robotiq-read-port", type=int, default=63352, help="Left 2F-85 URCap port used only for GET POS.")
    parser.add_argument("--left-finger-angle-rad", type=float, default=0.0, help="Fallback 2F-85 finger angle if GET POS is unavailable; 0=open, 0.8=closed.")
    parser.add_argument("--display-max-points", type=int, default=12000); parser.add_argument("--port", type=int, default=8766); parser.add_argument("--bind-host", default="127.0.0.1"); parser.add_argument("--once", action="store_true"); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if (args.obb_point_change_distance_m <= 0 or not 0 < args.obb_point_change_fraction <= 1
            or args.obb_history_frames < 3
            or not 1 <= args.obb_change_confirm_frames <= args.obb_history_frames
            or args.obb_removal_confirm_frames < 1
            or not np.isfinite(args.obb_merge_overlap_ratio)
            or not 0 <= args.obb_merge_overlap_ratio < 1):
        parser.error("Invalid OBB tracking thresholds or merge overlap ratio")
    return args


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
    return load_camera_file(path)


def load_arms(args: argparse.Namespace) -> tuple[Arm, Arm, CameraCalibration, dict]:
    project, _ = load_project_config(args.project_config)
    if not project["safety"].get("camera_calibration_valid", False):
        raise RuntimeError("Camera calibration expired or disabled; validate the configured calibration before starting live_obstacle_detection")
    config_path = resolve_repo_path(project["quest3"]["hardware_config"])
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    arms_config = config["arms"]
    left_cfg, right_cfg = arms_config["left_arm"], arms_config["right_arm"]
    shared = calibration_from_project(project)
    if shared is not None:
        T_left_base_from_front, K, serial = (shared.camera_to_left_base["front"], shared.intrinsics["front"], shared.serials["front"])
        left_report, right_report = shared.source_data["left/front"], shared.source_data["right/front"]
        right_K, right_serial = load_calibration(shared.source_paths["right/front"])[1:3]
        right_to_left = shared.right_base_to_left_base
    else:
        T_left_base_from_front, K, serial, left_report = load_calibration(resolve_repo_path(project["safety"]["front_calibration"]))
        T_right_base_from_front, right_K, right_serial, right_report = load_calibration(resolve_repo_path(project["safety"]["right_front_calibration"]))
        right_to_left = T_left_base_from_front @ np.linalg.inv(T_right_base_from_front)
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
    left = Arm("left", str(left_cfg["robot_ip"]), np.eye(4), "Robotiq 2F-85", matrix_from_file(args.left_tool_from_tcp))
    right = Arm("right", str(right_cfg["robot_ip"]), right_to_left, "Robotiq 2F-85", matrix_from_file(args.right_tool_from_tcp))
    calibration = CameraCalibration("front", K, T_left_base_from_front,
                                    left_report.get("intrinsics", {}).get("distortion"))
    report = {"camera_serial": serial, "left_touch_rms_mm": left_report["touch_rms_mm"], "right_touch_rms_mm": right_report["touch_rms_mm"], "calibration_intrinsics_max_delta_px": intrinsics_delta_px, "right_base_to_left_base": right_to_left.astype(float).tolist()}
    if shared is not None:
        report.update(shared.report())
        report["camera_intrinsics"] = {name: value.tolist() for name, value in shared.intrinsics.items()}
        report["camera_distortion"] = {name: shared.source_data[f"left/{name}"]["intrinsics"].get("distortion", [0.]*5)
                                        for name in ("front", "side")}
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


def tool_surface_points(q, closing, base_to_left, flange_to_tcp, *, samples_per_link=850):
    """Same local tool mesh and FK chain as DualUR7eSurfacePointSampler."""
    local = robotiq_2f85_points_in_active_tcp(finger_angle_rad=.8*closing, samples_per_link=samples_per_link)
    return transform_points(local, base_to_left @ flange_transform(q) @ flange_to_tcp)


def validate_active_tcp(q, measured_tcp, flange_to_tcp, *, max_position_mm=5., max_rotation_deg=1., arm_name=None):
    expected = flange_transform(q) @ flange_to_tcp
    mm, degrees = transform_error(expected, rotvec_transform(measured_tcp))
    if mm > max_position_mm or degrees > max_rotation_deg:
        prefix = f"{arm_name}: " if arm_name else ""
        raise RuntimeError(f"{prefix}Active TCP differs from fixed-surface FK: {mm:.3f} mm / {degrees:.3f} deg; recheck flange_to_active_tcp")
    return mm, degrees


def cap(points: np.ndarray, colors: np.ndarray, limit: int) -> tuple[list, list]:
    if len(points) > limit:
        index = np.linspace(0, len(points) - 1, limit, dtype=np.int64); points, colors = points[index], colors[index]
    return points.astype(float).tolist(), colors.astype(int).tolist()


def points_in_bounds(points: np.ndarray, bounds: tuple[float, ...] | list[float]) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[::2] >= bounds[1::2]):
        raise ValueError("Workspace bounds must be six finite X/Y/Z min/max values with min < max")
    return np.all((points >= bounds[::2]) & (points <= bounds[1::2]), axis=1)


def dual_arm_workspace_masks(
    points_left_base: np.ndarray,
    left_bounds: tuple[float, ...] | list[float],
    right_bounds: tuple[float, ...] | list[float],
    right_base_to_left_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return left, right, and union masks without approximating the rotated right box."""
    left = points_in_bounds(points_left_base, left_bounds)
    left_base_to_right_base = np.linalg.inv(np.asarray(right_base_to_left_base, dtype=np.float64))
    right_points = points_left_base.astype(np.float64) @ left_base_to_right_base[:3, :3].T + left_base_to_right_base[:3, 3]
    right = points_in_bounds(right_points, right_bounds)
    return left, right, left | right


def viewer_environment_points(
    points: np.ndarray,
    colors: np.ndarray,
    workspace_mask: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    plane_threshold_m: float,
    max_height_m: float,
    box_margin_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Add observed tabletop points outside the detector workspace for display only."""
    height = points.astype(np.float64) @ normal + offset
    outside = (~workspace_mask) & (height >= -max(.03, 2 * plane_threshold_m)) & (height <= max_height_m + box_margin_m)
    visible = workspace_mask | outside
    return points[visible], colors[visible], int(outside.sum())


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


def upright_obbs_overlap(a: UprightOBB, b: UprightOBB) -> bool:
    """Test the tabletop footprints and height intervals of two upright boxes."""
    vertical = np.asarray(a.rotation[:, 2] + b.rotation[:, 2], dtype=np.float64)
    vertical /= max(float(np.linalg.norm(vertical)), 1e-12)
    axes = (a.rotation[:, 0], a.rotation[:, 1], b.rotation[:, 0], b.rotation[:, 1], vertical)
    for axis in axes:
        axis = np.asarray(axis, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        a_projection = np.asarray(a.corners, dtype=np.float64) @ axis
        b_projection = np.asarray(b.corners, dtype=np.float64) @ axis
        overlap = min(a_projection.max(), b_projection.max()) - max(a_projection.min(), b_projection.min())
        if overlap <= 1e-6:
            return False
    return True


def upright_obb_overlap_ratio(a: UprightOBB, b: UprightOBB) -> float:
    """Intersection volume divided by the smaller upright OBB volume."""
    if not upright_obbs_overlap(a, b):
        return 0.0

    # Both boxes are upright to the tabletop. Project their lower faces into
    # one tabletop frame and clip the two convex footprint polygons.
    basis = np.asarray(a.rotation[:, :2], dtype=np.float64)
    normal = np.asarray(a.rotation[:, 2], dtype=np.float64)
    subject = np.asarray(a.corners[:4], dtype=np.float64) @ basis
    clipper = np.asarray(b.corners[:4], dtype=np.float64) @ basis

    def cross(edge: np.ndarray, point: np.ndarray) -> float:
        return float(edge[0] * point[1] - edge[1] * point[0])

    polygon = list(subject)
    for i in range(4):
        start, end = clipper[i], clipper[(i + 1) % 4]
        edge = end - start
        clipped = []
        for j, current in enumerate(polygon):
            previous = polygon[j - 1]
            before = cross(edge, previous - start)
            after = cross(edge, current - start)
            if (before >= 0) != (after >= 0):
                clipped.append(previous + before / (before - after) * (current - previous))
            if after >= 0:
                clipped.append(current)
        polygon = clipped
        if len(polygon) < 3:
            return 0.0

    footprint = np.asarray(polygon)
    area = abs(float(np.dot(footprint[:, 0], np.roll(footprint[:, 1], -1))
                     - np.dot(footprint[:, 1], np.roll(footprint[:, 0], -1)))) * .5
    a_height = np.asarray(a.corners, dtype=np.float64) @ normal
    b_height = np.asarray(b.corners, dtype=np.float64) @ normal
    height = max(0.0, min(a_height.max(), b_height.max()) - max(a_height.min(), b_height.min()))
    smaller_volume = min(float(np.prod(a.extents)), float(np.prod(b.extents)))
    return min(1.0, area * height / max(smaller_volume, 1e-12))


def merge_overlapping_obbs(
    boxes: list[UprightOBB], normal: np.ndarray, offset: float,
    *, min_overlap_ratio: float = .10,
) -> list[UprightOBB]:
    """Enclose boxes whose intersection exceeds a fraction of the smaller box."""
    if not 0.0 <= min_overlap_ratio < 1.0:
        raise ValueError("min overlap ratio must be in [0, 1)")
    merged = list(boxes)
    while True:
        pairs = [(upright_obb_overlap_ratio(merged[i], merged[j]), i, j)
                 for i in range(len(merged)) for j in range(i + 1, len(merged))]
        if not pairs:
            return merged
        ratio, i, j = max(pairs)
        if ratio <= min_overlap_ratio:
            return merged
        covering = tabletop_obb(np.concatenate((merged[i].corners, merged[j].corners)), normal, offset)
        if covering is None:
            raise RuntimeError("Overlapping obstacle boxes could not be enclosed")
        covering = UprightOBB(covering.center, covering.rotation, covering.extents, covering.corners,
                               merged[i].point_count + merged[j].point_count)
        merged[i] = covering
        del merged[j]


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
    points: np.ndarray
    consecutive_hits: int = 1
    missed_frames: int = 0
    confirmed: bool = False
    recent_observations: deque[tuple[int, np.ndarray]] = field(default_factory=deque)
    last_observed_frame: int = 0
    reference_frame: int = 0
    change_hits: int = 0
    free_space_hits: int = 0
    last_free_space_fraction: float = 0.0


def _enclose_duplicate_boxes(first: UprightOBB, second: UprightOBB) -> UprightOBB:
    """Conservatively enclose nearly identical tracks in the first box's axes."""
    local = (np.concatenate((first.corners, second.corners)) - first.center) @ first.rotation
    low, high = local.min(axis=0), local.max(axis=0)
    center = first.center + ((low + high) / 2) @ first.rotation.T
    extents = high - low
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=float)
    corners = center + (signs * extents / 2) @ first.rotation.T
    return UprightOBB(center, first.rotation, extents, corners, first.point_count + second.point_count)


def observed_free_space_fraction(points: np.ndarray,
                                 views: list[tuple[CameraCalibration, np.ndarray]],
                                 *, depth_margin_m: float = .03) -> float:
    """Fraction of former cluster points with measured depth behind them.

    A missing or foreground depth sample does not establish free space. A 3x3
    neighborhood must have at least five valid pixels and no closer return.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return 0.0
    free = np.zeros(len(points), dtype=bool)
    for camera, depth in views:
        depth = np.asarray(depth, dtype=float)
        world_to_camera = np.linalg.inv(camera.camera_to_world)
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        in_front = np.flatnonzero(camera_points[:, 2] > .01)
        if not len(in_front):
            continue
        pixels = project_pixels(camera_points[in_front], camera.intrinsics,
                                camera.distortion, camera.distortion_model)
        u = np.rint(pixels[:, 0]).astype(int)
        v = np.rint(pixels[:, 1]).astype(int)
        inside = (u >= 1) & (u < depth.shape[1]-1) & (v >= 1) & (v < depth.shape[0]-1)
        indices = in_front[inside]
        u, v = u[inside], v[inside]
        if not len(indices):
            continue
        neighborhood = np.stack([depth[v+dv, u+du] for dv in (-1, 0, 1)
                                 for du in (-1, 0, 1)])
        valid = np.isfinite(neighborhood) & (neighborhood > 0)
        nearest = np.min(np.where(valid, neighborhood, np.inf), axis=0)
        free[indices] |= ((valid.sum(axis=0) >= 5)
                          & (nearest > camera_points[indices, 2] + depth_margin_m))
    return float(np.mean(free))


class TemporalOBBGate:
    """Keep OBB geometry fixed until its observed point cluster changes persistently.

    Only a bounded window of sampled cluster observations is retained. Missing
    confirmed boxes expire after that many frames, with measured free space
    allowed to remove them sooner.
    """

    def __init__(self, *, required_hits: int, match_distance_m: float,
                 point_change_distance_m: float = .015, point_change_fraction: float = .25,
                 change_confirm_frames: int = 3, removal_confirm_frames: int = 3,
                 history_frames: int = 10) -> None:
        if required_hits < 1:
            raise ValueError("temporal persistence frames must be at least one")
        if (not np.isfinite(match_distance_m) or match_distance_m <= 0
                or not np.isfinite(point_change_distance_m) or point_change_distance_m <= 0
                or not np.isfinite(point_change_fraction) or not 0 < point_change_fraction <= 1
                or history_frames < 3 or not 1 <= change_confirm_frames <= history_frames
                or removal_confirm_frames < 1):
            raise ValueError("Invalid OBB tracking thresholds")
        self.required_hits = int(required_hits)
        self.match_distance_m = float(match_distance_m)
        self.point_change_distance_m = float(point_change_distance_m)
        self.point_change_fraction = float(point_change_fraction)
        self.change_confirm_frames = int(change_confirm_frames)
        self.removal_confirm_frames = int(removal_confirm_frames)
        self.history_frames = int(history_frames)
        self._frame_number = 0
        self._tracks: list[_ObstacleTrack] = []
        self.unstable_reasons: list[str] = []
        self.expired_tracks = 0

    @staticmethod
    def _sample_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError("OBB cluster points must be finite and nonempty")
        return points[np.linspace(0, len(points)-1, min(len(points), 512), dtype=int)].copy()

    def _change_fraction(self, previous: np.ndarray, current: np.ndarray) -> float:
        radius = self.point_change_distance_m
        new_fraction = np.mean(cKDTree(previous).query(current)[0] > radius)
        lost_fraction = np.mean(cKDTree(current).query(previous)[0] > radius)
        return float(max(new_fraction, lost_fraction))

    def _deduplicate_tracks(self) -> None:
        unique: list[_ObstacleTrack] = []
        for track in self._tracks:
            duplicate = next((other for other in unique
                              if track.confirmed and other.confirmed
                              and np.linalg.norm(track.center - other.center) <= self.match_distance_m
                              and np.allclose(track.box.extents, other.box.extents, rtol=.15, atol=.02)
                              and self._change_fraction(track.points, other.points) < .1), None)
            if duplicate is None:
                unique.append(track)
            else:
                duplicate.box = _enclose_duplicate_boxes(duplicate.box, track.box)
                duplicate.center = np.asarray(duplicate.box.center, dtype=float)
                if track.reference_frame > duplicate.reference_frame:
                    duplicate.points = track.points
                    duplicate.reference_frame = track.reference_frame
                if track.last_observed_frame > duplicate.last_observed_frame:
                    duplicate.recent_observations = track.recent_observations
                duplicate.last_observed_frame = max(duplicate.last_observed_frame,
                                                    track.last_observed_frame)
                duplicate.missed_frames = min(duplicate.missed_frames, track.missed_frames)
                duplicate.free_space_hits = min(duplicate.free_space_hits, track.free_space_hits)
                duplicate.last_free_space_fraction = min(duplicate.last_free_space_fraction,
                                                         track.last_free_space_fraction)
        self._tracks = unique

    def update(self, boxes: list[UprightOBB], point_clusters: list[np.ndarray] | None = None,
               *, free_space_views: list[tuple[CameraCalibration, np.ndarray]] | None = None) -> list[UprightOBB]:
        if point_clusters is None:
            point_clusters = [box.corners for box in boxes]
        if len(boxes) != len(point_clusters):
            raise ValueError("Each obstacle OBB needs its own observed point cluster")
        samples = [self._sample_points(points) for points in point_clusters]
        self._frame_number += 1
        self.unstable_reasons = []
        self.expired_tracks = 0
        self._deduplicate_tracks()
        active_tracks = []
        for track in self._tracks:
            while (track.recent_observations
                   and self._frame_number - track.recent_observations[0][0] >= self.history_frames):
                track.recent_observations.popleft()
            if (track.confirmed
                    and self._frame_number - track.last_observed_frame >= self.history_frames):
                self.expired_tracks += 1
                continue
            if (track.reference_frame
                    and self._frame_number - track.reference_frame >= self.history_frames
                    and track.recent_observations):
                track.reference_frame, track.points = track.recent_observations[0]
            track.missed_frames += 1
            active_tracks.append(track)
        self._tracks = active_tracks
        candidates = []
        merged_candidates: dict[int, list[int]] = {}
        for box_index, (box, points) in enumerate(zip(boxes, samples)):
            tree = cKDTree(points)
            supporting_tracks = []
            for track_index, track in enumerate(self._tracks):
                support = float(np.mean(tree.query(track.points)[0] <= self.point_change_distance_m))
                distance = float(np.linalg.norm(track.center - box.center))
                if support >= .5 and track.confirmed:
                    supporting_tracks.append(track_index)
                if distance <= self.match_distance_m or support >= .5:
                    candidates.append((-support, distance, track_index, box_index))
            if len(supporting_tracks) >= 2:
                merged_candidates[box_index] = supporting_tracks
        candidates.sort()
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for _, _, track_index, box_index in candidates:
            if track_index in used_tracks or box_index in used_boxes or box_index in merged_candidates:
                continue
            track = self._tracks[track_index]
            # A missed frame breaks persistence: stale boxes cannot become
            # confirmed merely by reappearing later.
            track.consecutive_hits = track.consecutive_hits + 1 if track.missed_frames == 1 else 1
            track.missed_frames = 0
            track.last_observed_frame = self._frame_number
            track.free_space_hits = 0
            track.last_free_space_fraction = 0.0
            observed_points = samples[box_index]
            newly_confirmed = not track.confirmed and track.consecutive_hits >= self.required_hits
            if newly_confirmed:
                track.confirmed = True
                track.box = boxes[box_index]
                track.center = np.asarray(track.box.center, dtype=np.float64)
                track.points = observed_points
                track.reference_frame = self._frame_number
                track.change_hits = 0
            elif self._change_fraction(track.points, observed_points) >= self.point_change_fraction:
                # Compare the current cluster with the most recent consecutive
                # observations only. The default requires support from both
                # preceding frames; an isolated spike cannot move an OBB.
                track.change_hits = 1
                for frame, previous_points in reversed(track.recent_observations):
                    if frame != self._frame_number - track.change_hits:
                        break
                    if (self._change_fraction(track.points, previous_points) < self.point_change_fraction
                            or self._change_fraction(previous_points, observed_points) >= self.point_change_fraction):
                        break
                    track.change_hits += 1
                    if track.change_hits >= self.change_confirm_frames:
                        break
                if track.change_hits >= self.change_confirm_frames:
                    track.box = boxes[box_index]
                    track.center = np.asarray(track.box.center, dtype=np.float64)
                    track.points = observed_points
                    track.reference_frame = self._frame_number
                    track.change_hits = 0
                elif track.confirmed:
                    self.unstable_reasons.append("changed_cluster")
            else:
                track.change_hits = 0
            track.recent_observations.append((self._frame_number, observed_points))
            used_tracks.add(track_index)
            used_boxes.add(box_index)
        for box_index, track_indices in merged_candidates.items():
            used_boxes.add(box_index)
            self.unstable_reasons.append("merged_clusters")
            for track_index in track_indices:
                if track_index not in used_tracks:
                    self._tracks[track_index].missed_frames = 0
                    self._tracks[track_index].free_space_hits = 0
                    self._tracks[track_index].last_free_space_fraction = 0.0
                    used_tracks.add(track_index)
        for box_index, box in enumerate(boxes):
            if box_index not in used_boxes:
                points = samples[box_index]
                self._tracks.append(_ObstacleTrack(
                    np.asarray(box.center, dtype=np.float64), box, points,
                    confirmed=self.required_hits == 1,
                    recent_observations=deque([(self._frame_number, points)],
                                              maxlen=self.history_frames),
                    last_observed_frame=self._frame_number,
                    reference_frame=self._frame_number))
        # A missing confirmed cluster remains in the safety scene during the
        # recent-frame window. An unconfirmed one-frame candidate disappears
        # on its first miss.
        self._tracks = [track for track in self._tracks if track.confirmed or not track.missed_frames]
        retained = []
        for track in self._tracks:
            if track.confirmed and track.missed_frames:
                free_fraction = (observed_free_space_fraction(track.points, free_space_views)
                                 if free_space_views is not None else 0.0)
                track.last_free_space_fraction = free_fraction
                track.free_space_hits = (track.free_space_hits + 1 if free_fraction >= .6 else 0)
                if track.free_space_hits >= self.removal_confirm_frames:
                    continue
            retained.append(track)
        self._tracks = retained
        if any(not track.confirmed for track in self._tracks):
            self.unstable_reasons.append("new_cluster")
        if any(track.confirmed and track.missed_frames for track in self._tracks):
            self.unstable_reasons.append("missing_cluster")
        return [track.box for track in self._tracks if track.confirmed]


def combine_depth(depths: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(depths[0], dtype=np.float32)
    for depth in depths:
        result = np.where((result > 0) & (depth > 0), np.minimum(result, depth), np.maximum(result, depth))
    return result


def materialize_snapshot(snapshot: dict, *, viewer: bool = False,
                         include_viewer_environment: bool = False) -> dict:
    """Convert display arrays only when a client asks for them."""
    display: SnapshotDisplay | None = snapshot.get("_display_data")
    if display is None:
        return dict(snapshot)
    result = {key: value for key, value in snapshot.items() if key != "_display_data"}
    if viewer:
        visible = display.table_band
        environment = cap(display.all_points[visible], display.all_colors[visible], display.max_points)
    else:
        environment = cap(display.environment_points, display.environment_colors, display.max_points)
    robot = cap(display.robot_points, display.robot_colors, display.max_points // 2)
    result["environment"] = {"points": environment[0], "colors": environment[1]}
    result["robot"] = {"points": robot[0], "colors": robot[1]}
    if include_viewer_environment:
        visible = display.table_band
        shown = cap(display.all_points[visible], display.all_colors[visible], display.max_points)
        result["_viewer_environment"] = {"points": shown[0], "colors": shown[1]}
    if not viewer:
        result["current_surface"] = display.surface_points.tolist()
        ok, encoded_rgb = cv2.imencode(".jpg", display.front_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("front RGB JPEG encoding failed")
        result["rgb_front_jpeg_base64"] = base64.b64encode(encoded_rgb.tobytes()).decode("ascii")
    return result


def snapshot_payload(snapshot: dict, *, compact: bool = False, viewer: bool = False) -> bytes:
    """Keep safety requests small and build display JSON outside the detector."""
    if compact:
        excluded = {"environment", "_viewer_environment", "robot", "current_surface",
                    "rgb_front_jpeg_base64", "_display_data"}
        snapshot = {key: value for key, value in snapshot.items() if key not in excluded}
    else:
        snapshot = materialize_snapshot(snapshot, viewer=viewer)
        viewer_environment = snapshot.get("_viewer_environment")
        snapshot = {key: value for key, value in snapshot.items() if key != "_viewer_environment"}
        if viewer and viewer_environment is not None:
            snapshot["environment"] = viewer_environment
    return json.dumps(snapshot, separators=(",", ":"), allow_nan=False).encode()


def main() -> None:
    args = parse_args(); left, right, calibration, report = load_arms(args)
    right_workspace_bounds = args.right_workspace_bounds or args.workspace_bounds
    points_in_bounds(np.empty((0, 3)), args.workspace_bounds)
    points_in_bounds(np.empty((0, 3)), right_workspace_bounds)
    project, _ = load_project_config(args.project_config)
    if "camera_to_left_base" in report:
        side_transform = np.asarray(report["camera_to_left_base"]["side"])
        side_K = np.asarray(report["camera_intrinsics"]["side"])
        side_serial = report["camera_serials"]["side"]
    else:
        side_transform, side_K, side_serial, side_report = load_calibration(resolve_repo_path(project["safety"]["side_calibration"]))
        if side_report.get("parent_frame") != "left_base":
            raise ValueError("side calibration must map the camera into left_base")
    if args.side_serial and args.side_serial != side_serial:
        raise ValueError("Side camera serial differs from calibration")
    side_distortion = (report["camera_distortion"]["side"] if "camera_distortion" in report else
                       side_report.get("intrinsics", {}).get("distortion"))
    calibrations = {"front": calibration, "side": CameraCalibration("side", side_K, side_transform, side_distortion)}
    geometry = DualUR7eSurfacePointSampler(points_per_link=2, project_config=args.project_config)
    for arm in (left, right):
        if not np.allclose(arm.tool_from_tcp, np.eye(4), atol=1e-8):
            raise ValueError("Tool overrides would disagree with fixed-point FK; configure the shared robot model instead")
    if "sha256" in report and geometry.calibration_digest != report["sha256"]:
        raise RuntimeError("Calibration changed while constructing perception geometry")
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
    latest: dict = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                    "frame": 0, "status": "initializing", "environment": {"points": [], "colors": []},
                    "robot": {"points": [], "colors": []}, "obbs": []}; lock = threading.Lock(); stop = threading.Event()

    def process() -> None:
        nonlocal latest
        source = RealSenseD435iSource(cameras=(D435iCameraConfig("front", args.front_serial or report["camera_serial"]), D435iCameraConfig("side", args.side_serial or side_serial)), width=args.width, height=args.height, fps=args.fps,
                                    depth_width=args.depth_width, depth_height=args.depth_height)
        receivers = {}
        try:
            for arm in (left, right):
                receivers[arm.name] = RTDEReceiveInterface(arm.ip)
        except Exception:
            for receiver in receivers.values(): receiver.disconnect()
            raise
        left_gripper: ReadOnlyRobotiq2FPosition | None = None; right_gripper: ReadOnlyRobotiq2FPosition | None = None
        finger_angle = float(np.clip(args.left_finger_angle_rad, 0., .8)); right_finger_angle = finger_angle
        gripper_status = f"2F-85 fixed angle {finger_angle:.3f} rad"
        obstacle_gate = TemporalOBBGate(
            required_hits=args.temporal_persistence_frames,
            match_distance_m=args.temporal_track_distance_m,
            point_change_distance_m=args.obb_point_change_distance_m,
            point_change_fraction=args.obb_point_change_fraction,
            change_confirm_frames=args.obb_change_confirm_frames,
            removal_confirm_frames=args.obb_removal_confirm_frames,
            history_frames=args.obb_history_frames,
        )
        try:
            left_gripper = ReadOnlyRobotiq2FPosition(left.ip, port=args.left_robotiq_read_port)
            right_gripper = ReadOnlyRobotiq2FPosition(right.ip, port=args.left_robotiq_read_port)
            gripper_status = "2F-85 GET POS connected (read-only)"
        except Exception as exc:
            if left_gripper is not None: left_gripper.close()
            for receiver in receivers.values(): receiver.disconnect()
            failed_arm = "right" if left_gripper is not None else "left"
            raise RuntimeError(
                f"{failed_arm} 2F-85 GET POS {right.ip if failed_arm == 'right' else left.ip}:"
                f"{args.left_robotiq_read_port} connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        frame_no = 0
        previous_capture_ns = None
        try:
            source.start()
            stream_info = source.get_stream_info()
            print(json.dumps({"camera_streams": stream_info}, ensure_ascii=False), flush=True)
            for name, info in source.get_color_calibrations().items():
                # Use actual color-stream intrinsics for the requested resolution.
                calibrations[name] = CameraCalibration(name, np.asarray(info["intrinsics"]), calibrations[name].camera_to_world,
                                                        info.get("distortion", calibrations[name].distortion),
                                                        info.get("distortion_model", "brown_conrady"))
            for _ in range(max(1, args.warmup_frames)): source.read()
            while not stop.is_set():
                began = time.perf_counter(); q0 = {a.name: np.asarray(receivers[a.name].getActualQ(), dtype=np.float64) for a in (left, right)}; frames = source.read(); captured = frames["front"]; q1 = {a.name: np.asarray(receivers[a.name].getActualQ(), dtype=np.float64) for a in (left, right)}
                capture_done = time.perf_counter()
                stamps = [frame.host_timestamp_ns for frame in frames.values()]
                capture_timestamp_ns = min(stamps)
                frame_interval_seconds = ((capture_timestamp_ns - previous_capture_ns) / 1e9
                                          if previous_capture_ns is not None else None)
                previous_capture_ns = capture_timestamp_ns
                if (max(stamps) - min(stamps)) / 1e9 > project["safety"]["max_camera_skew_s"]:
                    raise RuntimeError("Front/side capture skew exceeded the configured limit")
                robots, robot_col, depths = [], [], []
                try:
                    failed_arm = "left"
                    closing = left_gripper.closing_fraction()
                    failed_arm = "right"
                    right_closing = right_gripper.closing_fraction()
                except Exception as exc:
                    message = f"error: {failed_arm} gripper GET POS unavailable: {type(exc).__name__}: {exc}"
                    # A partial frame cannot be used for robot masking or
                    # collision checks. Keep polling so a restored read-only
                    # connection can publish a completely new scene.
                    obstacle_gate._tracks.clear()
                    with lock:
                        latest = {"schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                                  "frame": frame_no, "status": message,
                                  "environment": {"points": [], "colors": []},
                                  "robot": {"points": [], "colors": []}, "obbs": []}
                    if args.once:
                        raise RuntimeError(message) from exc
                    print(f"[model] {message}", file=sys.stderr, flush=True)
                    continue
                finger_angle, right_finger_angle = .8 * closing, .8 * right_closing
                gripper_status = f"2F-85 GET POS={closing:.3f}, q={finger_angle:.3f} rad"
                for arm, color in ((left, np.array((255, 70, 45), np.uint8)), (right, np.array((255, 165, 45), np.uint8))):
                    q = (q0[arm.name] + q1[arm.name]) * .5
                    if q.shape != (6,) or not np.isfinite(q).all(): raise RuntimeError(f"{arm.name} invalid RTDE joint state: {q}")
                    arm_points = arm_surface_points(q, arm.base_to_left, samples)
                    # Cross-check the controller TCP against the configured FK
                    # chain; rendering uses the same chain as fixed-point FK.
                    tcp = np.asarray(receivers[arm.name].getActualTCPPose(), dtype=np.float64)
                    if tcp.shape != (6,) or not np.isfinite(tcp).all(): raise RuntimeError(f"{arm.name} invalid RTDE TCP pose: {tcp}")
                    flange_to_tcp = geometry.flange_to_tcp[f"{arm.name}_arm"]
                    # q1 is closest in time to the TCP read; q is the frame's
                    # midpoint state used consistently for both mesh surfaces.
                    validate_active_tcp(q1[arm.name], tcp, flange_to_tcp, arm_name=f"{arm.name}_arm")
                    closing = (finger_angle if arm.name == left.name else right_finger_angle)/.8
                    tool_points = tool_surface_points(q, closing, arm.base_to_left, flange_to_tcp)
                    points = np.concatenate((arm_points, tool_points)); robots.append(points); robot_col.append(np.tile(color, (len(points), 1)))
                robot = np.concatenate(robots); rcol = np.concatenate(robot_col)
                geometry_done = time.perf_counter()
                refined_frames = frames if refiner is None else {f.camera_name: f for f in refiner.refine(tuple(frames.values()), calibrations)}
                refinement_done = time.perf_counter()
                clouds, colors, measured_surface = [], [], []
                support_pixels = 0
                for name, raw in frames.items():
                    view = calibrations[name]
                    refined = refined_frames[name]
                    rendered = render_surface_points_depth(robot, view.camera_to_world, view.intrinsics,
                                                           width=raw.rgb.shape[1], height=raw.rgb.shape[0], splat_radius_pixels=3,
                                                           distortion=view.distortion, distortion_model=view.distortion_model)
                    raw_keep = robot_depth_keep_mask(raw.depth_m, rendered, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels)
                    refined_keep = (raw_keep if refiner is None else
                                    robot_depth_keep_mask(refined.depth_m, rendered,
                                                          absolute_tolerance_m=args.absolute_tolerance_m,
                                                          relative_tolerance=args.relative_tolerance,
                                                          dilation_pixels=args.dilation_pixels))
                    raw_support = raw_depth_support_mask(raw.depth_m, raw_keep, args.raw_depth_support_pixels)
                    keep = refined_keep & raw_support
                    cloud, color = depth_to_world_points(refined, view, stride=args.point_stride, max_depth=2.5, keep_mask=keep)
                    clouds.append(cloud); colors.append(color); support_pixels += int(raw_support.sum())
                    v, u = np.nonzero(~raw_keep & (raw.depth_m > 0))
                    pixels = np.c_[u, v][::max(1, args.point_stride)]
                    observed, visible = _lift_measured_tracks(raw.depth_m, rendered, pixels, np.ones(len(pixels), bool), view, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance)
                    measured_surface.append(observed[visible])
                full_env, full_ecol = np.concatenate(clouds), np.concatenate(colors)
                surface_points = np.concatenate(measured_surface)
                point_cloud_done = time.perf_counter()
                left_mask, right_mask, arm_workspace_mask = dual_arm_workspace_masks(
                    full_env, args.workspace_bounds, right_workspace_bounds, right.base_to_left,
                )
                local = full_env[left_mask]
                if len(local) < 80: raise RuntimeError("not enough robot-filtered points to fit the tabletop")
                normal, offset, _ = estimate_dominant_plane(local, threshold=args.plane_threshold_m, ransac_iterations=300)
                # ``left_base`` need not have its +Z along the physical table
                # normal (this dual setup is not a world frame).  The only
                # stable sign convention is: positive height faces the fixed
                # overhead camera, so tabletop objects rise toward it.
                camera_side = float(calibration.camera_to_world[:3, 3] @ normal + offset)
                if camera_side < 0:
                    normal, offset = -normal, -offset
                full_height = full_env.astype(np.float64) @ normal + offset
                table_band = (full_height >= -max(.03, 2 * args.plane_threshold_m)) & (full_height <= args.max_height_m + args.box_margin_m)
                mask = arm_workspace_mask & table_band
                env, ecol = full_env[mask], full_ecol[mask]
                # The viewer's tabletop band includes the detection workspace
                # plus points outside it; the band has already been calculated.
                viewer_extra_count = int((~arm_workspace_mask & table_band).sum())
                height = env.astype(np.float64) @ normal + offset; candidates = (height >= args.min_height_m)&(height <= args.max_height_m); op, oh = env[candidates], height[candidates]
                if len(op) > 1 and args.outlier_neighbour_m > 0:
                    distances, _ = cKDTree(op).query(op, k=2, workers=-1); valid = distances[:,1] <= args.outlier_neighbour_m; op,oh = op[valid],oh[valid]
                raw_boxes = []; raw_clusters = []; floating_components = 0; connected_components = 0
                for indices in dbscan_clusters(op, eps_m=args.cluster_eps_m, min_samples=args.cluster_min_samples, min_cluster_points=args.min_cluster_points):
                    if oh[indices].min() <= args.attachment_distance_m:
                        connected_components += 1
                        box = tabletop_obb(op[indices], normal, offset)
                        if box is not None:
                            raw_boxes.append(box)
                            raw_clusters.append(op[indices])
                    else:
                        floating_components += 1
                scene_analysis_done = time.perf_counter()
                # Track raw clusters separately: merging boxes before or after
                # temporal matching can create a large unsupported safety OBB.
                candidate_boxes = [expanded_obb(box, args.box_margin_m) for box in raw_boxes]
                tracked_boxes = obstacle_gate.update(
                    candidate_boxes, raw_clusters,
                    free_space_views=[(calibrations[name], frame.depth_m) for name, frame in frames.items()],
                )
                boxes, supplemental_boxes = safety_obbs_with_candidates(
                    tracked_boxes, candidate_boxes, raw_clusters)
                before_overlap_merge = len(boxes)
                # A deep geometric overlap takes precedence over temporal
                # stability of the individual tracks. Enclose both old and
                # current candidate OBBs in the published safety scene.
                boxes = merge_overlapping_obbs(
                    boxes, normal, offset,
                    min_overlap_ratio=args.obb_merge_overlap_ratio,
                )
                box_tracking_done = time.perf_counter()
                display_boxes = boxes
                frame_no += 1
                qdelta = max(float(np.max(np.abs(q1[a.name]-q0[a.name]))) for a in (left,right))
                diagnostics = {"environment_points": int(len(env)), "left_workspace_points": int((left_mask & table_band).sum()), "right_workspace_points": int((right_mask & table_band).sum()), "viewer_outside_workspace_points": viewer_extra_count, "object_candidate_points": int(len(op)), "table_connected_components": connected_components, "floating_components": floating_components, "raw_candidate_obbs": len(raw_boxes), "candidate_obbs": len(candidate_boxes), "confirmed_obbs": len(tracked_boxes), "supplemental_obbs": supplemental_boxes, "published_obbs": len(boxes), "safety_obbs_cover_candidates": True, "obb_unstable_reasons": sorted(set(obstacle_gate.unstable_reasons)), "raw_depth_support_pixels": args.raw_depth_support_pixels, "raw_depth_supported_pixels": support_pixels, "table_normal_left_base": normal.astype(float).tolist(), "table_offset_m": float(offset), "depth_refinement": "lingbot-depth" if refiner is not None else "raw-d455", "depth_refinement_seconds": float(refiner.last_inference_seconds) if refiner is not None else 0.0}
                diagnostics["obb_point_change_distance_m"] = args.obb_point_change_distance_m
                diagnostics["obb_point_change_fraction"] = args.obb_point_change_fraction
                diagnostics["obb_change_confirm_frames"] = args.obb_change_confirm_frames
                diagnostics["obb_history_frames"] = args.obb_history_frames
                diagnostics["expired_obbs"] = obstacle_gate.expired_tracks
                diagnostics["obb_removal_confirm_frames"] = args.obb_removal_confirm_frames
                diagnostics["obb_merge_overlap_ratio"] = args.obb_merge_overlap_ratio
                diagnostics["large_overlap_merges"] = before_overlap_merge - len(boxes)
                diagnostics["capture_seconds"] = capture_done - began
                diagnostics["frame_interval_seconds"] = frame_interval_seconds
                diagnostics["robot_geometry_seconds"] = geometry_done - capture_done
                diagnostics["depth_stage_seconds"] = refinement_done - geometry_done
                diagnostics["point_cloud_seconds"] = point_cloud_done - refinement_done
                diagnostics["scene_analysis_seconds"] = scene_analysis_done - point_cloud_done
                diagnostics["box_tracking_seconds"] = box_tracking_done - scene_analysis_done
                diagnostics["obb_tracks"] = [
                    {"center": track.center.astype(float).tolist(),
                     "missed_frames": track.missed_frames,
                     "last_observed_frame": track.last_observed_frame,
                     "reference_frame": track.reference_frame,
                     "cached_cluster_frames": len(track.recent_observations),
                     "free_space_fraction": round(track.last_free_space_fraction, 3),
                     "free_space_hits": track.free_space_hits}
                    for track in obstacle_gate._tracks if track.confirmed
                ]
                refinement = f" · LingBot {refiner.last_inference_seconds:.2f}s" if refiner is not None else ""
                snapshot_build_began = time.perf_counter()
                snapshot = {
                    "schema": "dual_ur7e_obstacle_snapshot_v2", "coordinate_frame": "left_base",
                    "frame": frame_no, "host_timestamp_ns": int(min(stamps)),
                    "status": (f"error: unstable obstacle point clusters ({', '.join(sorted(set(obstacle_gate.unstable_reasons)))})"
                               if obstacle_gate.unstable_reasons else
                               f"{time.perf_counter()-began:.2f}s{refinement} · {gripper_status} · 双臂 qΔ {qdelta:.4f} rad · {len(boxes)} 个桌面连接障碍物"),
                    "qpos": {a.name: ((q0[a.name] + q1[a.name]) * .5).astype(float).tolist() for a in (left, right)},
                    "gripper_position": {"left": float(finger_angle / .8), "right": float(right_finger_angle / .8)},
                    "surface_source": "measured_depth_fk_gated",
                    "camera_names": ["front", "side"],
                    "_display_data": SnapshotDisplay(env, ecol, full_env, full_ecol, table_band,
                                                     robot, rcol, surface_points, captured.rgb.copy(),
                                                     args.display_max_points),
                    "obbs":[{
                        "center": b.center.astype(float).tolist(), "axes": b.rotation.astype(float).tolist(),
                        "half_sizes": (b.extents * .5).astype(float).tolist(), "corners":b.corners.astype(float).tolist(),
                        "point_count": int(b.point_count),
                    } for b in boxes],
                    "calibration": report, "camera_streams": stream_info, "diagnostics": diagnostics,
                }
                # Capture age is the safety boundary; publication must never
                # replace it with a new timestamp for already processed data.
                snapshot["published_timestamp_ns"] = time.monotonic_ns()
                diagnostics["snapshot_build_seconds"] = time.perf_counter() - snapshot_build_began
                diagnostics["processing_seconds"] = time.perf_counter() - began
                diagnostics["capture_to_publish_seconds"] = (snapshot["published_timestamp_ns"] - min(stamps)) / 1e9
                with lock: latest = snapshot
                # A one-shot diagnostic must still observe enough frames to
                # exercise the configured temporal persistence gate.  The old
                # one-frame behavior could never emit an OBB with the default
                # required_hits=3 and therefore produced a misleading zero.
                if args.once and frame_no >= args.temporal_persistence_frames:
                    break
        finally:
            cleanup = [source.stop, *(receiver.disconnect for receiver in receivers.values())]
            cleanup.extend(gripper.close for gripper in (left_gripper, right_gripper) if gripper is not None)
            errors = []
            for close in cleanup:
                try: close()
                except Exception as exc: errors.append(exc)
            if errors:
                if sys.exc_info()[0] is None:
                    raise RuntimeError(f"Perception cleanup failed: {errors}") from errors[0]
                print(f"[model] cleanup error: {errors}", file=sys.stderr, flush=True)

    def guarded_process() -> None:
        nonlocal latest
        try:
            process()
        except Exception as exc:
            message = f"error: {type(exc).__name__}: {exc}"
            # Also catch receiver/gripper initialization failures. Never leave
            # a dead worker reporting initializing or old obstacle geometry.
            with lock:
                latest = {"schema": "dual_ur7e_obstacle_snapshot_v2",
                          "coordinate_frame": "left_base", "frame": latest.get("frame", 0),
                          "status": message, "environment": {"points": [], "colors": []},
                          "robot": {"points": [], "colors": []}, "obbs": []}
            print(f"[model] {message}", file=sys.stderr, flush=True)

    worker = threading.Thread(target=guarded_process, daemon=True)
    if args.once:
        worker.start()
        worker.join()
        if str(latest["status"]).startswith("error") or "schema" not in latest:
            raise RuntimeError(f"Obstacle acquisition failed: {latest['status']}")
        latest = materialize_snapshot(latest, include_viewer_environment=True)
        print(json.dumps({"schema": latest.get("schema"), "coordinate_frame": latest.get("coordinate_frame"), "frame": latest["frame"], "host_timestamp_ns": latest.get("host_timestamp_ns"), "status": latest["status"], "environment_points": len(latest["environment"]["points"]), "robot_points": len(latest["robot"]["points"]), "obbs": latest["obbs"], "diagnostics": latest.get("diagnostics", {}), "calibration": latest.get("calibration", {}), "camera_streams": latest.get("camera_streams", {})}, ensure_ascii=False, indent=2)); return latest

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/snapshot.json"):
                # Snapshots are replaced atomically, never modified in place.
                # JSON encoding must not hold up publication of the next frame.
                with lock: snapshot = latest
                query = parse_qs(urlsplit(self.path).query)
                compact = query.get("compact") == ["1"]
                viewer = query.get("viewer") == ["1"]
                response_began = time.perf_counter()
                data = snapshot_payload(snapshot, compact=compact, viewer=viewer)
                if "gzip" in self.headers.get("Accept-Encoding", ""):
                    data = gzip.compress(data, compresslevel=1); compressed = True
                else: compressed = False
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
                self.send_header("Server-Timing", f"snapshot;dur={(time.perf_counter() - response_began) * 1000:.1f}")
                if compressed: self.send_header("Content-Encoding", "gzip")
                self.end_headers(); self.wfile.write(data); return
            if self.path.startswith("/plotly.min.js"):
                import plotly
                payload = (Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js").read_bytes(); self.send_response(200); self.send_header("Content-Type", "application/javascript"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload); return
            payload = VIEWER_HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def log_message(self, *_): pass
    server = ThreadingHTTPServer((args.bind_host, args.port), Handler); url = f"http://127.0.0.1:{args.port}"; print(f"[model] {url}  (Ctrl+C to stop)")
    try:
        worker.start()
        if not args.no_browser: webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=10.)


if __name__ == "__main__":
    main()
