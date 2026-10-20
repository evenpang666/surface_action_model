#!/usr/bin/env python3
"""Standalone RGB OBB annotation: optional HDF5 file/directory, otherwise RealSense."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_scripts.batch_obstacle_annotation import (
    BatchAnnotations, annotation_schema, handler_for, png_url, validate_boxes,
)
from real_scripts.manual_obstacle_selection import timestamp_capture_directory
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path


class CameraAnnotations:
    """Frozen camera captures; annotating or saving never reads another frame."""

    def __init__(self, output_dir, capture_frame, *, camera_names=("front", "side"), camera_serials=None):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.capture_frame = capture_frame
        self.camera_names = tuple(camera_names)
        self.camera_serials = dict(camera_serials or {})
        self.lock = threading.RLock()
        self.captures = []

    def manifest(self):
        with self.lock:
            episodes = [{"episode_id": i, "relative_path": frame["filename"],
                         "frame_count": 1, "reviewed_count": int(frame["reviewed"]),
                         "first_unreviewed": None if frame["reviewed"] else 0}
                        for i, frame in enumerate(self.captures)]
            return {"mode": "camera", "camera_names": list(self.camera_names),
                    "camera_serials": self.camera_serials,
                    "task_dir": str(self.output_dir), "episodes": episodes,
                    "total_frames": len(episodes),
                    "reviewed_frames": sum(e["reviewed_count"] for e in episodes)}

    def capture(self):
        with self.lock:
            frames, streams = self.capture_frame()
            if tuple(frames) != self.camera_names:
                raise ValueError(f"Capture requires all configured cameras: {', '.join(self.camera_names)}")
            images = {name: png_url(frame.rgb[..., ::-1]) for name, frame in frames.items()}
            capture_id, directory = timestamp_capture_directory(self.output_dir)
            cameras = {name: {"rgb_width": frame.rgb.shape[1], "rgb_height": frame.rgb.shape[0],
                              "stream": streams.get(name),
                              "host_timestamp_ns": str(frame.host_timestamp_ns) if frame.host_timestamp_ns is not None else None}
                       for name, frame in frames.items()}
            metadata = {"capture_id": capture_id, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "timestamp_ns": str(time.time_ns()), "camera_names": list(frames),
                        "annotation_image": "recorded_rgb", "cameras": cameras}
            for name, frame in frames.items():
                if not cv2.imwrite(str(directory / f"{name}_rgb.png"), frame.rgb[..., ::-1]):
                    raise RuntimeError(f"Cannot save {name} RGB image")
            (directory / "capture.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            frame = {**metadata, "episode_id": len(self.captures), "frame_index": 0, "frame_count": 1,
                     "filename": capture_id, "revision": 0, "images": images,
                     "source_timestamps_ns": {name: camera["host_timestamp_ns"] for name, camera in cameras.items()},
                     "boxes": {name: [] for name in frames}, "reviewed": False, "status": None,
                     "output_dir": str(directory)}
            self.captures.append(frame)
            return self.frame(frame["episode_id"], 0)

    def frame(self, episode_id, frame_index):
        with self.lock:
            if (type(episode_id) is not int or not 0 <= episode_id < len(self.captures)
                    or type(frame_index) is not int or frame_index != 0):
                raise ValueError("Capture or frame index is out of range")
            return json.loads(json.dumps(self.captures[episode_id]))

    def save(self, body):
        with self.lock:
            frame = self.frame(body.get("episode_id"), body.get("frame_index"))
            revision = body.get("revision")
            if type(revision) is not int or revision != frame["revision"]:
                raise ValueError("Frame changed in another window; reload before saving")
            status = body.get("status", "objects")
            boxes = validate_boxes(body.get("boxes"), status, frame["cameras"])
            saved = {key: value for key, value in frame.items() if key != "images"}
            saved.update(schema=annotation_schema(frame["camera_names"]), boxes=boxes, status=status,
                         revision=revision+1, reviewed=True, updated_at_utc=datetime.now(timezone.utc).isoformat())
            # Keep labels beside the immutable RGB capture, with its readable capture time.
            directory = Path(frame["output_dir"])
            pending = directory / "annotations.json.tmp"
            pending.write_text(json.dumps(saved, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            pending.replace(directory / "annotations.json")
            self.captures[frame["episode_id"]].update(boxes=boxes, status=status, revision=revision+1, reviewed=True)
            return {"episode_id": frame["episode_id"], "frame_index": 0, "revision": revision+1,
                    "reviewed": True, "status": status, "output_path": str(directory / "annotations.json")}


def configured_camera_serials(args):
    """Use the same named hardware cameras as live_obstacle_detection, never USB order."""
    project, _ = load_project_config(args.project_config)
    hardware_path = resolve_repo_path(project["quest3"]["hardware_config"])
    with hardware_path.open(encoding="utf-8") as handle:
        hardware = yaml.safe_load(handle)
    serials = {}
    for name in ("front", "side"):
        serial = getattr(args, f"{name}_serial") or hardware.get("cameras", {}).get(name, {}).get("serial")
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError(f"Missing {name} camera serial in {hardware_path}; specify --{name}-serial")
        serials[name] = serial.strip()
    if serials["front"] == serials["side"]:
        raise ValueError("Front and side camera serials must be different")
    return serials


def camera_capture(args):
    """Create the camera source lazily; HDF5 mode never imports hardware adapters."""
    from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource

    serials = configured_camera_serials(args)
    cameras = [D435iCameraConfig(name, serials[name]) for name in ("front", "side")]
    source = RealSenseD435iSource(cameras=cameras, width=args.width, height=args.height, fps=args.fps)
    try:
        source.start()
        for _ in range(args.warmup_frames):
            source.read()
        frames = source.read()
        streams = source.get_stream_info()
        color_info = source.get_color_calibrations()
        streams = {name: {**streams[name], "color_calibration": color_info[name]} for name in ("front", "side")}
        return {name: frames[name] for name in ("front", "side")}, streams
    finally:
        source.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--hdf5", "--hdf5-path", type=Path, help="HDF5 file or dataset directory; omit for camera capture.")
    paths.add_argument("--task-dir", type=Path, help="Dataset directory (alias for --hdf5 directory).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Read-only HDF5 inventory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/manual_obstacles")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG,
                        help="Project configuration; camera serial defaults come from its hardware_config.")
    parser.add_argument("--front-serial", help="Override the configured front RealSense serial.")
    parser.add_argument("--side-serial", help="Override the configured side RealSense serial. Both views are always enabled.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--depth-unit-m", type=float, default=.001)
    parser.add_argument("--front-depth-path")
    parser.add_argument("--side-depth-path")
    args = parser.parse_args(argv)
    path = args.hdf5 or args.task_dir
    if args.inspect and path is None:
        parser.error("--inspect requires --hdf5 or --task-dir")
    if min(args.width, args.height, args.fps) <= 0 or args.warmup_frames < 0:
        parser.error("Camera dimensions/fps must be positive; warmup-frames must be nonnegative")
    try:
        serials = configured_camera_serials(args) if path is None else None
        if serials is not None:
            # Freeze the resolved mapping for this session, including subsequent captures.
            args.front_serial, args.side_serial = serials["front"], serials["side"]
        dataset = (BatchAnnotations(path, depth_unit_m=args.depth_unit_m,
                                    depth_paths={"front": args.front_depth_path, "side": args.side_depth_path})
                   if path is not None else CameraAnnotations(args.output_dir, lambda: camera_capture(args),
                                                               camera_serials=serials))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.inspect:
        print(json.dumps(dataset.manifest(), ensure_ascii=False, indent=2))
        return
    with ThreadingHTTPServer((args.host, args.port), handler_for(dataset)) as server:
        url = f"http://{args.host}:{server.server_address[1]}"
        if serials is not None:
            print(f"Camera mapping: front={serials['front']}, side={serials['side']}", flush=True)
        print(f"Open {url} (Ctrl+C to stop). " +
              ("Save writes labels into the original HDF5." if path else
               f"Capture a camera frame in the page; labels save under {dataset.output_dir}/<capture_time>/annotations.json."), flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
