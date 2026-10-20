#!/usr/bin/env python3
"""Annotate front/side RGB frames in a task's HDF5 episodes.

No camera or robot connections. Files are read lazily, and only an explicit
save appends labels to the original episode's annotations/obstacle_obbs group.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlsplit

import cv2
import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_scripts.manual_obstacle_selection import obb_mask

GROUP = "annotations/obstacle_obbs"
SCHEMA = "dual_rgb_obb_annotation_v2"
CAMERAS = ("front", "side")


def annotation_schema(camera_names):
    return SCHEMA if tuple(camera_names) == CAMERAS else "single_rgb_obb_annotation_v1"


def validate_boxes(boxes, status, cameras):
    """Validate and canonicalize RGB OBBs for both recorded and live frames."""
    if not isinstance(boxes, dict) or set(boxes) != set(cameras):
        raise ValueError(f"Boxes must contain {', '.join(cameras)}")
    canonical = {}
    for name, camera in cameras.items():
        entries = boxes[name]
        if not isinstance(entries, list) or len(entries) > 32:
            raise ValueError("At most 32 OBBs per view")
        canonical[name] = []
        for box in entries:
            if not isinstance(box, dict):
                raise ValueError("Each OBB must be an object")
            object_id = box.get("object_id")
            if not isinstance(object_id, str) or not object_id.strip() or len(object_id) > 64:
                raise ValueError("Each OBB needs a short object_id")
            obb_mask((camera["rgb_height"], camera["rgb_width"]), box)
            canonical[name].append({"object_id": object_id.strip(), **{
                key: float(box[key]) for key in ("cx", "cy", "width", "height", "angle_deg")}})
    ids = [set(box["object_id"] for box in entries) for entries in canonical.values()]
    if status == "no_obstacles":
        if any(ids):
            raise ValueError("No-obstacle frames must have empty boxes in all views")
    elif status != "objects" or not ids[0] or any(view != ids[0] for view in ids):
        raise ValueError("Use the same obstacle IDs in all views, or explicitly mark no_obstacles")
    return canonical


def decode_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def inspect_episode(path, depth_paths=None):
    """Read RGB dimensions and depth provenance without conflating their pixels."""
    with h5py.File(path, "r") as source:
        if not bool(source.attrs.get("complete", True)):
            raise ValueError(f"{path.name}: episode is incomplete")
        if "timestamps_ns" not in source or source["timestamps_ns"].ndim != 1:
            raise ValueError(f"{path.name}: requires a 1D timestamps_ns frame index")
        count = len(source["timestamps_ns"])
        if not count:
            raise ValueError(f"{path.name}: episode has no frames")
        if int(source.attrs.get("num_samples", count)) != count:
            raise ValueError(f"{path.name}: num_samples does not match timestamps_ns")
        config = yaml.safe_load(decode_text(source.attrs.get("config_yaml", "{}"))) or {}
        if not isinstance(config, dict):
            raise ValueError(f"{path.name}: config_yaml must describe a mapping")
        cameras = {}
        camera_names = tuple(name for name in CAMERAS if f"observations/images/{name}/rgb" in source)
        if "front" not in camera_names:
            raise ValueError(f"{path.name}: observations/images/front/rgb is required")
        for name in camera_names:
            rgb_path = f"observations/images/{name}/rgb"
            if rgb_path not in source or len(source[rgb_path]) != count:
                raise ValueError(f"{path.name}: {rgb_path} is required and must match timestamps_ns")
            rgb = source[rgb_path]
            if rgb.ndim == 4 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8:
                rgb_height, rgb_width = rgb.shape[1:3]
            elif rgb.ndim == 1:
                rgb_height, rgb_width = rgb_image(rgb, 0).shape[:2]
            else:
                raise ValueError(f"{path.name}: {rgb_path} must contain RGB frames or JPEG payloads")
            candidates = ([depth_paths[name]] if depth_paths and depth_paths.get(name) else
                          [f"observations/images/{name}/{suffix}" for suffix in
                           ("native_depth_units", "native_depth", "depth")])
            dataset_path = next((candidate for candidate in candidates if candidate in source), None)
            if depth_paths and depth_paths.get(name) and dataset_path is None:
                raise ValueError(f"{path.name}: requested {name} depth path does not exist: {depth_paths[name]}")
            dataset = source[dataset_path] if dataset_path else None
            if dataset is not None and (dataset.ndim != 3 or len(dataset) != count or dataset.dtype != np.uint16):
                raise ValueError(f"{path.name}: {dataset_path} must be uint16 [N,H,W] matching timestamps_ns")
            camera_config = config.get("cameras", {}).get(name, {})
            topic = str(camera_config.get("depth_topic", ""))
            declared = decode_text(dataset.attrs.get("annotation_image", dataset.attrs.get("coordinate_system", ""))) if dataset is not None else ""
            native_name = dataset_path is not None and dataset_path.rsplit("/", 1)[-1] in ("native_depth", "native_depth_units")
            native_declared = declared in ("native_raw_depth", "native_depth")
            native_topic = "/depth/image_rect_raw" in topic or "/depth/image_raw" in topic
            resized = camera_config.get("depth_record_size", camera_config.get("record_size"))
            if dataset is None:
                kind = "absent"
            elif "aligned_depth_to_color" in topic and not native_name and not native_declared:
                kind = "recorded_aligned_depth"
            elif native_name or native_declared or (native_topic and not resized):
                kind = "native_raw_depth"
            else:
                kind = "recorded_depth_unknown_coordinates"
            timestamp_path = f"source_timestamps_ns/{name}_depth"
            if timestamp_path in source and source[timestamp_path].shape != (count,):
                raise ValueError(f"{path.name}: {timestamp_path} does not match frame count")
            scale = dataset.attrs.get("depth_scale_m", dataset.attrs.get("depth_unit_m")) if dataset is not None else None
            if scale is not None and (not np.isfinite(float(scale)) or float(scale) <= 0):
                raise ValueError(f"{path.name}: invalid recorded depth scale")
            cameras[name] = {"depth_path": dataset_path, "depth_kind": kind,
                             "width": dataset.shape[2] if dataset is not None else None,
                             "height": dataset.shape[1] if dataset is not None else None,
                             "rgb_path": rgb_path, "rgb_width": rgb_width, "rgb_height": rgb_height,
                             "timestamp_path": timestamp_path if timestamp_path in source else None,
                             "source_topic": topic, "depth_scale_m": None if scale is None else float(scale)}
        reviewed = np.zeros(count, bool)
        if GROUP in source:
            group = source[GROUP]
            if group.attrs.get("schema") != annotation_schema(camera_names):
                raise ValueError(f"{path.name}: incompatible existing annotation schema; RGB coordinates require a new annotation group")
            for key in ("frames_json", "reviewed", "revision", "updated_at_utc", "frame_index", "timestamps_ns"):
                if key not in group or group[key].shape != (count,):
                    raise ValueError(f"{path.name}: annotation {key} is not frame-aligned")
            if not np.array_equal(group["timestamps_ns"][:], source["timestamps_ns"][:]):
                raise ValueError(f"{path.name}: annotation timestamps no longer match episode")
            if not np.array_equal(group["frame_index"][:], np.arange(count)):
                raise ValueError(f"{path.name}: annotation frame indices no longer match episode")
            recorded = json.loads(decode_text(group.attrs["camera_sources_json"]))
            if recorded != cameras:
                raise ValueError(f"{path.name}: annotation camera provenance no longer matches episode")
            reviewed = group["reviewed"][:]
        pending = np.flatnonzero(~reviewed)
        return {"filename": Path(path).name, "frame_count": count, "reviewed_count": int(reviewed.sum()),
                "first_unreviewed": int(pending[0]) if len(pending) else None,
                "task_description": decode_text(source.attrs.get("task_description", "")), "cameras": cameras}


def png_url(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode annotation image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def rgb_image(dataset, index):
    raw = np.asarray(dataset[index])
    if raw.ndim == 3 and raw.shape[-1] == 3:
        return raw.astype(np.uint8)
    decoded = cv2.imdecode(np.asarray(raw, np.uint8).reshape(-1), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Cannot decode recorded RGB reference")
    return decoded[..., ::-1]


class BatchAnnotations:
    def __init__(self, task_dir, *, depth_unit_m=.001, depth_paths=None):
        source = Path(task_dir).expanduser().resolve()
        if not source.exists():
            raise ValueError(f"HDF5 path does not exist: {source}")
        if source.is_file() and source.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError(f"Expected an HDF5 file: {source}")
        self.task_dir = source if source.is_dir() else source.parent
        self.paths = ([source] if source.is_file() else
                      sorted({p.resolve() for suffix in ("*.hdf5", "*.h5") for p in source.rglob(suffix)}))
        if not self.paths:
            raise ValueError(f"No HDF5 episodes under {self.task_dir}")
        for path in self.paths:
            path.relative_to(self.task_dir)
        if not np.isfinite(depth_unit_m) or depth_unit_m <= 0:
            raise ValueError("depth_unit_m must be finite and positive")
        self.depth_unit_m, self.depth_paths = float(depth_unit_m), depth_paths
        self.lock = threading.RLock()
        self.episodes = [inspect_episode(path, depth_paths) for path in self.paths]
        for path, episode in zip(self.paths, self.episodes):
            with h5py.File(path, "r") as source:
                if GROUP in source and any(c["depth_scale_m"] is None for c in episode["cameras"].values()):
                    recorded_scale = float(source[GROUP].attrs.get("fallback_depth_unit_m", np.nan))
                    if recorded_scale != self.depth_unit_m:
                        raise ValueError(f"{path.name}: resume with the original --depth-unit-m {recorded_scale}")

    def manifest(self):
        with self.lock:
            episodes = []
            for index, path in enumerate(self.paths):
                info = inspect_episode(path, self.depth_paths)
                episodes.append({**info, "episode_id": index, "relative_path": str(path.relative_to(self.task_dir))})
            return {"task_dir": str(self.task_dir), "episodes": episodes,
                    "total_frames": sum(e["frame_count"] for e in episodes),
                    "reviewed_frames": sum(e["reviewed_count"] for e in episodes)}

    def indices(self, episode_id, frame_index):
        if (type(episode_id) is not int or not 0 <= episode_id < len(self.paths)
                or type(frame_index) is not int or not 0 <= frame_index < self.episodes[episode_id]["frame_count"]):
            raise ValueError("Episode or frame index is out of range")
        return self.paths[episode_id], self.episodes[episode_id]

    def frame(self, episode_id, frame_index):
        with self.lock:
            path, info = self.indices(episode_id, frame_index)
            with h5py.File(path, "r") as source:
                references, timestamps, sources = {}, {}, {}
                for name in info["cameras"]:
                    camera = info["cameras"][name]
                    reference = rgb_image(source[camera["rgb_path"]], frame_index)
                    if reference.shape[:2] != (camera["rgb_height"], camera["rgb_width"]):
                        raise ValueError(f"{name} RGB dimensions changed within the episode")
                    references[name] = png_url(reference[..., ::-1])
                    timestamps[name] = (str(int(source[camera["timestamp_path"]][frame_index]))
                                        if camera["timestamp_path"] else None)
                    sources[name] = {**camera, "effective_depth_scale_m":
                                     (camera["depth_scale_m"] or self.depth_unit_m) if camera["depth_path"] else None}
                reviewed, revision, boxes, status = False, 0, {name: [] for name in info["cameras"]}, None
                if GROUP in source:
                    group = source[GROUP]
                    reviewed, revision = bool(group["reviewed"][frame_index]), int(group["revision"][frame_index])
                    if reviewed:
                        saved = json.loads(decode_text(group["frames_json"][frame_index]))
                        boxes, status = saved["boxes"], saved["status"]
                return {"episode_id": episode_id, "frame_index": frame_index, "frame_count": info["frame_count"],
                        "filename": str(path.relative_to(self.task_dir)), "revision": revision,
                        "timestamp_ns": str(int(source["timestamps_ns"][frame_index])),
                        "source_timestamps_ns": timestamps, "depth_sources": sources,
                        "annotation_image": "recorded_rgb", "images": references,
                        "boxes": boxes, "reviewed": reviewed, "status": status}

    def save(self, body):
        with self.lock:
            path, info = self.indices(body.get("episode_id"), body.get("frame_index"))
            index, revision = body["frame_index"], body.get("revision")
            if type(revision) is not int or revision < 0:
                raise ValueError("Frame revision is required")
            status = body.get("status", "objects")
            canonical = validate_boxes(body.get("boxes"), status, info["cameras"])
            schema = annotation_schema(info["cameras"])
            with h5py.File(path, "r+") as source:
                current = int(source[GROUP+"/revision"][index]) if GROUP in source else 0
                if current != revision:
                    raise ValueError("Frame changed in another window; reload before saving")
                if GROUP not in source:
                    group = source.require_group(GROUP)
                    group.attrs.update(schema=schema, angle_unit="degrees_clockwise", coordinate_unit="pixels",
                                       annotation_image="recorded_rgb",
                                       obb_fields="cx,cy,width,height,angle_deg,object_id",
                                       camera_sources_json=json.dumps(info["cameras"]),
                                       fallback_depth_unit_m=self.depth_unit_m,
                                       frame_alignment="dataset index i equals original episode sample index i")
                    n = info["frame_count"]
                    strings = h5py.string_dtype("utf-8")
                    group.create_dataset("frames_json", shape=(n,), dtype=strings)
                    group.create_dataset("reviewed", shape=(n,), dtype=np.bool_, fillvalue=False)
                    group.create_dataset("revision", shape=(n,), dtype=np.uint64, fillvalue=0)
                    group.create_dataset("updated_at_utc", shape=(n,), dtype=strings)
                    group.create_dataset("frame_index", data=np.arange(n, dtype=np.int64))
                    group.create_dataset("timestamps_ns", data=source["timestamps_ns"][:])
                    for name in info["cameras"]:
                        camera = info["cameras"][name]
                        if camera["timestamp_path"]:
                            group.create_dataset(f"source_timestamps_ns/{name}_depth", data=source[camera["timestamp_path"]][:])
                group = source[GROUP]
                timestamp = str(int(source["timestamps_ns"][index]))
                saved = {"schema": schema, "frame_index": index, "timestamp_ns": timestamp,
                         "source_timestamps_ns": {name: (str(int(source[camera["timestamp_path"]][index]))
                                                           if camera["timestamp_path"] else None)
                                                  for name, camera in info["cameras"].items()},
                         "annotation_image": "recorded_rgb",
                         "depth_sources": {name: {**camera, "effective_depth_scale_m":
                                                  (camera["depth_scale_m"] or self.depth_unit_m) if camera["depth_path"] else None}
                                           for name, camera in info["cameras"].items()},
                         "status": status, "boxes": canonical}
                # Reviewed is the commit marker, separating unvisited/failed saves
                # from explicit empty annotations and fully saved dual-view labels.
                group["reviewed"][index] = False
                source.flush()
                group["frames_json"][index] = json.dumps(saved, ensure_ascii=False, allow_nan=False)
                group["updated_at_utc"][index] = datetime.now(timezone.utc).isoformat()
                group["revision"][index] = current+1
                source.flush()
                group["reviewed"][index] = True
                source.flush()
            return {"episode_id": body["episode_id"], "frame_index": index, "revision": current+1,
                    "reviewed": True, "status": status}


def handler_for(dataset):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/manifest":
                    self.respond(dataset.manifest())
                elif parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    self.respond(dataset.frame(int(query["episode_id"][0]), int(query["frame_index"][0])))
                elif parsed.path == "/":
                    payload = (ROOT / "real_scripts/batch_obstacle_annotation.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.respond({"error": "Unknown endpoint"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                self.respond({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            route = urlsplit(self.path).path
            if route not in ("/api/save", "/api/capture") or (route == "/api/capture" and not hasattr(dataset, "capture")):
                self.respond({"error": "Unknown endpoint"}, 404)
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.respond({"error": "Cross-origin request rejected"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Expected a JSON body up to 64 KiB")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                self.respond(dataset.capture() if route == "/api/capture" else dataset.save(body))
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                self.respond({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def log_message(self, *_args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--depth-unit-m", type=float, default=.001,
                        help="Scale for stored uint16 depth when no scale attribute is available (default millimetres).")
    parser.add_argument("--front-depth-path")
    parser.add_argument("--side-depth-path")
    parser.add_argument("--inspect", action="store_true", help="Read-only inventory; do not start the annotation server.")
    args = parser.parse_args()
    paths = {"front": args.front_depth_path, "side": args.side_depth_path}
    try:
        dataset = BatchAnnotations(args.task_dir, depth_unit_m=args.depth_unit_m, depth_paths=paths)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    manifest = dataset.manifest()
    if args.inspect:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    print(f"{len(dataset.paths)} episodes, {manifest['total_frames']} frames, "
          f"{manifest['reviewed_frames']} reviewed. Labels write into the original HDF5 on Save.", flush=True)
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop). No cameras or robots are connected.", flush=True)
    with ThreadingHTTPServer((args.host, args.port), handler_for(dataset)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
