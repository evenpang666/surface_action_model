#!/usr/bin/env python3
"""Extract CoTracker-filtered *measured* left-robot surface points from HDF5.

The output points are never sampled from a robot mesh.  Mesh FK is used only
twice as a gate: to select visible measured robot pixels in frame zero, and to
reject tracker pixels whose recorded D455 depth no longer agrees with the
current robot rendering.  Valid 3-D coordinates are back-projected from the
recorded depth image and expressed in ``left_base``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import CameraCalibration, robot_depth_keep_mask  # noqa: E402
from real_scripts.ur7e_collision_mesh import render_surface_points_depth  # noqa: E402
from safety_module.shards import open_shard
from real_scripts.quest3_depth import Quest3DepthReader, recorded_intrinsics
from safety_module.dual_camera_calibration import rigid_matrix


DEFAULT_COTRACKER_REPO = Path.home() / ".cache/torch/hub/facebookresearch_co-tracker_main"
DEFAULT_COTRACKER_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/scaled_offline.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--geometry-shard", "--mesh-surface-npz", dest="mesh_surface_npz", type=Path, required=True, help="FK gate shard; mesh points are not emitted as measured observations.")
    parser.add_argument("--calibration", type=Path, required=True, help="Recorded fixed front color-camera calibration in left_base or right_base; right_base is composed with the geometry shard's base transform. Wrist cameras require per-frame FK and are not supported by this front-only extractor.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True, help="Front RGB overview with measured CoTracker points only.")
    parser.add_argument("--max-seeds", type=int, default=1024, help="Dense, stable CoTracker point IDs; each is still lifted from measured D455 depth.")
    parser.add_argument("--seed-stride", type=int, default=2)
    parser.add_argument("--tracking-scale", type=float, default=.5)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_COTRACKER_REPO)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=.015)
    parser.add_argument("--relative-depth-tolerance", type=float, default=.015)
    parser.add_argument("--depth-dilation-pixels", type=int, default=3)
    parser.add_argument("--depth-unit-m", type=float, default=None, help="Explicit metres per Z16 unit; otherwise use recorded scale (historical aligned-depth fallback: .001).")
    parser.add_argument("--render-splat-pixels", type=int, default=3)
    parser.add_argument("--visualization-max-frames", type=int, default=0, help="Number of animation frames; 0 means every recorded frame (default).")
    return parser.parse_args()


def _recorded_intrinsics(calibration_path: Path, *, width: int, height: int,
                         right_base_to_left_base: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return recorded RGB K and left_base_T_front, checking declared frames."""
    payload = yaml.safe_load(Path(calibration_path).read_text(encoding="utf-8"))
    child = payload.get("child_frame", "front_camera_color_optical_frame")
    if child != "front_camera_color_optical_frame":
        raise ValueError(f"This extractor requires a fixed front camera, got {child}; "
                         "wrist camera poses require per-frame arm FK and flange-to-camera calibration")
    parent = payload.get("parent_frame", "left_base")
    camera_to_world = rigid_matrix(payload["matrix"], str(calibration_path))
    if parent == "right_base":
        if right_base_to_left_base is None:
            raise ValueError("right_base camera calibration requires right_base_to_left_base in the geometry shard")
        camera_to_world = rigid_matrix(right_base_to_left_base, "right_base_to_left_base") @ camera_to_world
    elif parent != "left_base":
        raise ValueError(f"Unsupported fixed camera parent_frame: {parent}")
    K = recorded_intrinsics(payload["intrinsics"], width=width, height=height)
    return K, camera_to_world


def _rendered_depth(points: np.ndarray, calibration: CameraCalibration, *, height: int, width: int, splat_pixels: int) -> np.ndarray:
    return render_surface_points_depth(
        np.asarray(points, dtype=np.float32).reshape(-1, 3), calibration.camera_to_world, calibration.intrinsics,
        width=width, height=height, splat_radius_pixels=splat_pixels,
        distortion=calibration.distortion, distortion_model=calibration.distortion_model,
    )


def _seed_pixels(depth_m: np.ndarray, rendered_depth: np.ndarray, *, max_seeds: int, stride: int, absolute_tolerance_m: float, relative_tolerance: float, dilation_pixels: int) -> np.ndarray:
    environment = robot_depth_keep_mask(depth_m, rendered_depth, absolute_tolerance_m=absolute_tolerance_m, relative_tolerance=relative_tolerance, dilation_pixels=dilation_pixels)
    candidate = ~environment
    grid = np.zeros_like(candidate, dtype=bool); grid[::stride, ::stride] = True
    v, u = np.nonzero(candidate & grid)
    if len(u) == 0:
        raise RuntimeError("No measured robot pixels passed the first-frame FK depth gate")
    pixels = np.c_[u, v].astype(np.float32)
    return pixels if len(pixels) <= max_seeds else pixels[np.linspace(0, len(pixels) - 1, max_seeds, dtype=np.int64)]


def _track_cotracker(video: np.ndarray, seeds_xy: np.ndarray, *, repo: Path, checkpoint: Path, chunk_size: int, device="cuda") -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("CoTracker repository or scaled_offline checkpoint is absent; this workflow intentionally does not fall back to synthetic mesh points")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CoTracker observation extraction requires CUDA on this episode; CUDA is not available")
    model = torch.hub.load(str(repo), "cotracker3_offline", source="local", pretrained=False)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state_dict)
    model = model.to(device).eval()
    tracked_parts: list[np.ndarray] = []
    visible_parts: list[np.ndarray] = []
    current_xy = np.asarray(seeds_xy, dtype=np.float32).copy()
    start = 0
    while start < len(video):
        end = min(start + int(chunk_size), len(video))
        query = np.concatenate((np.zeros((len(current_xy), 1), dtype=np.float32), current_xy), axis=1)[None]
        video_tensor = torch.from_numpy(np.ascontiguousarray(video[start:end])).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        query_tensor = torch.from_numpy(query).to(device)
        with torch.inference_mode():
            tracks, visibility = model(video_tensor, queries=query_tensor)
        local_xy = tracks[0].detach().cpu().numpy().astype(np.float32)
        local_visible = visibility[0].detach().cpu().numpy()
        if local_visible.ndim == 3:
            local_visible = local_visible[..., 0]
        if start:
            local_xy, local_visible = local_xy[1:], local_visible[1:]
        tracked_parts.append(local_xy)
        visible_parts.append(np.asarray(local_visible, dtype=bool))
        current_xy = tracks[0, -1].detach().cpu().numpy().astype(np.float32)
        if end == len(video): break
        start = end - 1
    return np.concatenate(tracked_parts), np.concatenate(visible_parts)


def _lift_measured_tracks(depth_m: np.ndarray, rendered_depth: np.ndarray, tracks_xy: np.ndarray, tracker_visible: np.ndarray, calibration: CameraCalibration, *, absolute_tolerance_m: float, relative_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    u = np.rint(tracks_xy[:, 0]).astype(np.int64); v = np.rint(tracks_xy[:, 1]).astype(np.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    safe_u, safe_v = np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)
    measured = depth_m[safe_v, safe_u]; expected = rendered_depth[safe_v, safe_u]
    visible = np.asarray(tracker_visible, dtype=bool) & in_bounds & np.isfinite(measured) & (measured > 0.0) & (expected > 0.0)
    visible &= np.abs(measured - expected) <= float(absolute_tolerance_m) + float(relative_tolerance) * expected
    points = np.zeros((len(tracks_xy), 3), dtype=np.float32)
    if np.any(visible):
        z = measured[visible].astype(np.float64); K = calibration.intrinsics
        from real_scripts.camera_geometry import pixel_rays
        xyz = pixel_rays(np.c_[u[visible], v[visible]], K, calibration.distortion, calibration.distortion_model)*z[:, None]
        camera = np.c_[xyz, np.ones(len(z))]
        points[visible] = (calibration.camera_to_world @ camera.T).T[:, :3].astype(np.float32)
    return points, visible


def _colour(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    return tuple(int(value) for value in cv2.cvtColor(np.uint8([[[hue * 179, 205, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])


def _write_preview(path: Path, rgb: np.ndarray, tracks: np.ndarray, visible: np.ndarray) -> None:
    indices = (0, len(rgb) // 2, len(rgb) - 1); panels = []
    for index in indices:
        panel = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
        for point_id, ((u, v), valid) in enumerate(zip(tracks[index], visible[index], strict=True)):
            if valid: cv2.circle(panel, (int(round(u)), int(round(v))), 2, _colour(point_id)[::-1], -1, cv2.LINE_AA)
        label = f"frame {index + 1}/{len(rgb)} | measured CoTracker points: {int(visible[index].sum())}"
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (24, 27, 30), 1, cv2.LINE_AA)
        panels.append(panel)
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def _write_html(path: Path, points: np.ndarray, visible: np.ndarray, timestamps_ns: np.ndarray, *, max_frames: int) -> None:
    import plotly.graph_objects as go

    frame_indices = np.arange(len(points), dtype=np.int64) if int(max_frames) <= 0 else np.unique(np.linspace(0, len(points) - 1, min(int(max_frames), len(points)), dtype=np.int64))
    colours = [f"rgb{_colour(index)}" for index in range(points.shape[1])]
    colour_groups: dict[str, list[int]] = {}
    for point_id, colour in enumerate(colours):
        colour_groups.setdefault(colour, []).append(point_id)
    group_colours = list(colour_groups)
    point_group = np.empty((points.shape[1],), dtype=np.int32)
    for group_id, point_ids in enumerate(colour_groups.values()):
        point_group[point_ids] = group_id

    # The trajectory trace begins empty.  ``post_script`` below reconstructs
    # only its history up through the playback frame in the browser; this
    # avoids revealing future motion while keeping the HTML compact.
    trajectory_traces = [
        go.Scatter3d(x=[], y=[], z=[], mode="lines", name="fixed-point trajectory history", showlegend=False,
                     line={"color": colour, "width": 3}, hoverinfo="skip", connectgaps=False)
        for colour in group_colours
    ]
    trajectory_trace_indices = list(range(len(trajectory_traces)))
    current_trace_index = len(trajectory_traces)
    observations_by_id = [
        [[int(frame), *[round(float(value), 6) for value in points[frame, point_id]]] for frame in np.flatnonzero(visible[:, point_id])]
        for point_id in range(points.shape[1])
    ]

    def trace(index: int):
        keep = visible[index]
        ids = np.flatnonzero(keep)
        return [go.Scatter3d(x=points[index, ids, 0], y=points[index, ids, 1], z=points[index, ids, 2], customdata=ids.astype(np.int32), mode="markers", marker={"size": 2, "color": np.asarray(colours, dtype=object)[ids].tolist()}, name="current measured fixed points", hoverinfo="skip")]
    frames = [go.Frame(name=str(int(index)), data=trace(int(index)), traces=[current_trace_index], layout={"title": {"text": f"Measured CoTracker fixed-point flow · frame {index + 1}/{len(points)} · visible {int(visible[index].sum())}/{points.shape[1]}"}}) for index in frame_indices]
    axis_style = {"backgroundcolor": "#000000", "showbackground": True, "gridcolor": "#303030", "linecolor": "#D1D5DB", "zerolinecolor": "#D1D5DB", "tickcolor": "#D1D5DB", "tickfont": {"color": "#E5E7EB"}, "title": {"font": {"color": "#E5E7EB"}}}
    layout = {
        "title": "Measured left-robot surface points only (no mesh points)", "paper_bgcolor": "#000000", "font": {"color": "#E5E7EB"},
        "scene": {"bgcolor": "#000000", "aspectmode": "data", "xaxis": {**axis_style, "title": {"text": "left_base X (m)", "font": {"color": "#E5E7EB"}}}, "yaxis": {**axis_style, "title": {"text": "left_base Y (m)", "font": {"color": "#E5E7EB"}}}, "zaxis": {**axis_style, "title": {"text": "left_base Z (m)", "font": {"color": "#E5E7EB"}}}},
        "updatemenus": [{"type": "buttons", "showactive": False, "x": .01, "y": .02, "buttons": [
            {"label": "播放", "method": "animate", "args": [None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True}]},
            {"label": "暂停", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            {"label": "显示轨迹", "method": "restyle", "args": [{"visible": True}, trajectory_trace_indices]},
            {"label": "隐藏轨迹", "method": "restyle", "args": [{"visible": False}, trajectory_trace_indices]},
        ]}],
        "sliders": [{"active": 0, "x": .15, "len": .8, "y": .02, "steps": [
            {"label": str(int(index)), "method": "animate", "args": [[str(int(index))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
            for index in frame_indices
        ]}],
    }
    figure = go.Figure(data=[*trajectory_traces, *trace(int(frame_indices[0]))], frames=frames, layout=layout)
    post_script = """
const gd = document.getElementById('{plot_id}');
const pointCount = __POINT_COUNT__;
const groupCount = __GROUP_COUNT__;
const pointGroup = __POINT_GROUPS__;
const trajectoryTraceIndices = Array.from({length: groupCount}, (_, index) => index);
const observationsById = __OBSERVATIONS_BY_ID__;
let requestedFrame = 0;
let redrawTimer = null;

function trajectoryHistory(endFrame) {
  const xs = Array.from({length: groupCount}, () => []);
  const ys = Array.from({length: groupCount}, () => []);
  const zs = Array.from({length: groupCount}, () => []);
  const limit = Math.max(0, Math.min(__FRAME_COUNT__ - 1, endFrame));
  for (let pointId = 0; pointId < pointCount; pointId++) {
    const groupId = pointGroup[pointId], observations = observationsById[pointId];
    let lastSeen = -2;
    for (const observation of observations) {
      const frameIndex = observation[0];
      if (frameIndex > limit) break;
      if (lastSeen !== frameIndex - 1) { xs[groupId].push(null); ys[groupId].push(null); zs[groupId].push(null); }
      xs[groupId].push(observation[1]); ys[groupId].push(observation[2]); zs[groupId].push(observation[3]); lastSeen = frameIndex;
    }
  }
  return [xs, ys, zs];
}

function drawHistory(frameIndex) {
  const [x, y, z] = trajectoryHistory(frameIndex);
  Plotly.restyle(gd, {x: x, y: y, z: z}, trajectoryTraceIndices);
}
function scheduleHistory(frameIndex, immediate) {
  requestedFrame = Number(frameIndex);
  if (redrawTimer !== null) return;
  redrawTimer = setTimeout(() => {
    redrawTimer = null; drawHistory(requestedFrame);
  }, immediate ? 0 : 100);
}
function frameIndexFromEvent(eventData) {
  const raw = eventData && (eventData.name ?? (eventData.frame && eventData.frame.name));
  const value = Number(raw); return Number.isFinite(value) ? value : 0;
}
function currentSliderFrame() {
  const slider = gd._fullLayout && gd._fullLayout.sliders && gd._fullLayout.sliders[0];
  if (!slider || !slider.steps || !slider.steps.length) return 0;
  const active = Math.max(0, Math.min(slider.steps.length - 1, Number(slider.active) || 0));
  return Number(slider.steps[active].label);
}
function currentPlaybackFrame() {
  const title = gd.layout && gd.layout.title && gd.layout.title.text ? String(gd.layout.title.text) : '';
  const match = /frame\\s+(\\d+)\\//.exec(title);
  return match ? Number(match[1]) - 1 : currentSliderFrame();
}
gd.on('plotly_animated', () => scheduleHistory(currentPlaybackFrame(), false));
gd.on('plotly_sliderchange', (eventData) => scheduleHistory(Number(eventData.step.label), true));
gd.on('plotly_buttonclicked', () => scheduleHistory(currentPlaybackFrame(), true));
scheduleHistory(0, true);
""".replace("__POINT_COUNT__", str(points.shape[1])).replace("__FRAME_COUNT__", str(len(points))).replace("__GROUP_COUNT__", str(len(group_colours))).replace("__POINT_GROUPS__", json.dumps(point_group.tolist(), separators=(",", ":"))).replace("__OBSERVATIONS_BY_ID__", json.dumps(observations_by_id, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True); figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False, post_script=post_script)


def main() -> None:
    args = parse_args()
    if args.max_seeds < 1 or args.seed_stride < 1 or args.chunk_size < 2 or not 0.0 < args.tracking_scale <= 1.0 or (args.depth_unit_m is not None and (not np.isfinite(args.depth_unit_m) or args.depth_unit_m <= 0.0)):
        raise ValueError("invalid seed/tracker/depth arguments")
    with open_shard(args.mesh_surface_npz) as mesh:
        mesh_points = np.asarray(mesh["fixed_link_points"], dtype=np.float32)
        mesh_qpos = np.asarray(mesh["qpos"], dtype=np.float32)
        link_names = np.asarray(mesh["link_names"])
        surface_arm_count = int(mesh["surface_arm_count"]) if "surface_arm_count" in mesh else int(mesh["arm_count"])
        if "coordinate_frame" in mesh and str(np.asarray(mesh["coordinate_frame"]).item()) != "left_base":
            raise ValueError("The geometry shard must express both arms in left_base")
        right_base_to_left_base = np.asarray(mesh["right_base_to_left_base"]) if "right_base_to_left_base" in mesh else None
    with h5py.File(args.episode, "r") as source:
        from scripts.preprocess_quest3_hdf5 import _decode_rgb
        rgb = _decode_rgb(source["observations/images/front/rgb"])
        timestamps_ns = np.asarray(source["timestamps_ns"], dtype=np.int64)
        height, width = rgb.shape[1:3]
        K, camera_to_world = _recorded_intrinsics(args.calibration, width=width, height=height,
                                                right_base_to_left_base=right_base_to_left_base)
        world_spec = yaml.safe_load(args.calibration.read_text(encoding="utf-8"))
        color_spec = world_spec["intrinsics"]
        calibration = CameraCalibration("front", K, camera_to_world,
                                        color_spec.get("distortion", color_spec.get("coeffs")),
                                        color_spec.get("distortion_model", "brown_conrady"))
        depth_reader = Quest3DepthReader(source, view="front", rgb_shape=(height, width),
                                        calibration=calibration, depth_unit_m=args.depth_unit_m)
        if len(depth_reader.dataset) != len(rgb):
            raise ValueError("HDF5 depth and RGB frame counts disagree")
        depth0 = depth_reader.read(0)
        calibration = depth_reader.calibration
        depth_metadata = depth_reader.metadata
        depth_metadata["world_calibration_source"] = str(args.calibration.resolve())
        depth_metadata["world_coordinate_frame"] = "left_base"
        depth_unit_m = depth_reader.depth_unit_m
        K = calibration.intrinsics
    if len(rgb) == 0 or len(rgb) != len(mesh_points) or len(mesh_qpos) != len(rgb) or timestamps_ns.shape != (len(rgb),):
        raise ValueError("HDF5 and mesh-surface frame counts disagree")
    initial_rendered = _rendered_depth(mesh_points[0], calibration, height=height, width=width, splat_pixels=args.render_splat_pixels)
    seeds = _seed_pixels(depth0, initial_rendered, max_seeds=args.max_seeds, stride=args.seed_stride, absolute_tolerance_m=args.absolute_depth_tolerance_m, relative_tolerance=args.relative_depth_tolerance, dilation_pixels=args.depth_dilation_pixels)
    seed_points, seed_valid = _lift_measured_tracks(depth0, initial_rendered, seeds, np.ones(len(seeds), bool), calibration, absolute_tolerance_m=args.absolute_depth_tolerance_m, relative_tolerance=args.relative_depth_tolerance)
    seeds, seed_points = seeds[seed_valid], seed_points[seed_valid]
    _, nearest = cKDTree(mesh_points[0].reshape(-1, 3)).query(seed_points)
    track_link_ids = nearest // mesh_points.shape[2]
    if surface_arm_count == 2 and not all(any(str(link_names[link]).startswith(arm) for link in track_link_ids) for arm in ("left_", "right_")):
        raise RuntimeError("Both arms must have measured seeds in the initial view; use a recording/calibration with visibility of both arms")
    scaled_size = (int(round(width * args.tracking_scale)), int(round(height * args.tracking_scale)))
    video = np.stack([cv2.resize(frame, scaled_size, interpolation=cv2.INTER_AREA) for frame in rgb]) if args.tracking_scale != 1.0 else rgb
    tracks, tracker_visible = _track_cotracker(video, seeds * args.tracking_scale, repo=args.repo, checkpoint=args.checkpoint, chunk_size=args.chunk_size, device=args.device)
    tracks /= args.tracking_scale
    observed = np.zeros((len(rgb), len(seeds), 3), dtype=np.float32); visible = np.zeros((len(rgb), len(seeds)), dtype=bool)
    with h5py.File(args.episode, "r") as source:
        depth_reader = Quest3DepthReader(source, view="front", rgb_shape=(height, width),
                                        calibration=calibration, depth_unit_m=args.depth_unit_m)
        for index in range(len(rgb)):
            depth_m = depth_reader.read(index)
            rendered = _rendered_depth(mesh_points[index], calibration, height=height, width=width, splat_pixels=args.render_splat_pixels)
            observed[index], visible[index] = _lift_measured_tracks(depth_m, rendered, tracks[index], tracker_visible[index], calibration, absolute_tolerance_m=args.absolute_depth_tolerance_m, relative_tolerance=args.relative_depth_tolerance)
            for link in np.unique(track_link_ids):
                ids = np.flatnonzero((track_link_ids == link) & visible[index])
                if len(ids):
                    distance, _ = cKDTree(mesh_points[index, link]).query(observed[index, ids])
                    visible[index, ids] &= distance <= .04
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, source_episode=np.asarray(str(Path(args.episode).resolve())), coordinate_frame=np.asarray("left_base"), point_source=np.asarray("measured_d455_depth_lifted_cotracker_fk_gated"), mesh_points_emitted=np.asarray(False), tracks_xy=tracks.astype(np.float32), tracker_visible=tracker_visible.astype(bool), observed_points=observed, observed_visible=visible, seed_xy=seeds, timestamps_ns=timestamps_ns, intrinsics=K, camera_to_world=camera_to_world, distortion=np.asarray(calibration.distortion if calibration.distortion is not None else np.zeros(5)), distortion_model=np.asarray(calibration.distortion_model), depth_unit_m=np.asarray(depth_unit_m, dtype=np.float32), depth_alignment_metadata=np.asarray(json.dumps(depth_metadata)), rendered_depth_tolerance_m=np.asarray(args.absolute_depth_tolerance_m, dtype=np.float32), rendered_depth_relative_tolerance=np.asarray(args.relative_depth_tolerance, dtype=np.float32), tracker=np.asarray("CoTracker3 offline scaled_offline"))
    _write_preview(args.preview, rgb, tracks, visible)
    _write_html(args.visualization, observed, visible, timestamps_ns, max_frames=args.visualization_max_frames)
    print({"frames": len(rgb), "stable_seed_ids": len(seeds), "cotracker_visible": int(tracker_visible.sum()), "depth_and_fk_valid": int(visible.sum()), "mean_valid_per_frame": float(visible.sum() / len(rgb)), "output": str(args.output), "preview": str(args.preview), "visualization": str(args.visualization)})


if __name__ == "__main__":
    main()
