"""Local paired-OBB annotation server for recorded HDF5 RGB frames.

No robot or camera connection. The browser's explicit Save writes labels into
annotations/obstacle_obbs; original RGB/actions are never overwritten.
"""

import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from .annotations import GROUP, SCHEMA, LEGACY_SCHEMA, VIEWS, validate_frame, _text


def _rgb(dataset, index):
    array = np.asarray(dataset[index])
    if array.ndim == 1:
        with Image.open(BytesIO(array.astype(np.uint8).tobytes())) as image:
            array = np.asarray(image.convert("RGB"))
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Recorded RGB must be uint8 HWC or encoded image bytes")
    return array


def _png(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class PairedAnnotations:
    def __init__(self, source):
        import h5py
        source = Path(source).expanduser().resolve()
        self.task_dir = source.parent if source.is_file() else source
        self.paths = [source] if source.is_file() else sorted(set(source.rglob("*.h5")) | set(source.rglob("*.hdf5")))
        if not self.paths:
            raise ValueError("No HDF5 episodes found")
        self.lock = threading.RLock()
        self.infos = []
        for path in self.paths:
            with h5py.File(path, "r") as episode:
                stamps = episode["timestamps_ns"]
                if stamps.ndim != 1 or not len(stamps) or not bool(episode.attrs.get("complete", True)):
                    raise ValueError(f"{path}: requires complete, nonempty frame-aligned episodes")
                shapes = {}
                for view in VIEWS:
                    ds = episode[f"observations/images/{view}/rgb"]
                    if len(ds) != len(stamps):
                        raise ValueError(f"{path}: RGB/timestamps length mismatch")
                    shapes[view] = _rgb(ds, 0).shape
                if GROUP in episode:
                    group = episode[GROUP]
                    if (_text(group.attrs.get("schema", "")) not in (SCHEMA, LEGACY_SCHEMA)
                            or _text(group.attrs.get("annotation_image", "")) != "recorded_rgb"
                            or not np.array_equal(group["timestamps_ns"][:], stamps[:])
                            or any(group[key].shape != stamps.shape for key in ("reviewed", "frames_json", "revision"))):
                        raise ValueError(f"{path}: incompatible or misaligned existing annotations")
                self.infos.append(dict(frame_count=len(stamps), shapes=shapes))

    def _indices(self, episode_id, frame_index):
        if (type(episode_id) is not int or not 0 <= episode_id < len(self.paths)
                or type(frame_index) is not int or not 0 <= frame_index < self.infos[episode_id]["frame_count"]):
            raise ValueError("Episode or frame index out of range")
        return self.paths[episode_id], self.infos[episode_id]

    def manifest(self):
        import h5py
        with self.lock:
            entries = []
            for index, (path, info) in enumerate(zip(self.paths, self.infos)):
                with h5py.File(path, "r") as episode:
                    reviewed = episode[GROUP + "/reviewed"][:] if GROUP in episode else np.zeros(info["frame_count"], bool)
                pending = np.flatnonzero(~reviewed)
                entries.append(dict(episode_id=index, relative_path=str(path.relative_to(self.task_dir)),
                                    frame_count=info["frame_count"], reviewed_count=int(reviewed.sum()),
                                    first_unreviewed=int(pending[0]) if len(pending) else None))
            return dict(task_dir=str(self.task_dir), episodes=entries, total_frames=sum(e["frame_count"] for e in entries),
                        reviewed_frames=sum(e["reviewed_count"] for e in entries))

    def frame(self, episode_id, frame_index):
        import h5py
        with self.lock:
            path, info = self._indices(episode_id, frame_index)
            with h5py.File(path, "r") as episode:
                images = {}
                for view in VIEWS:
                    rgb = _rgb(episode[f"observations/images/{view}/rgb"], frame_index)
                    if rgb.shape != info["shapes"][view]:
                        raise ValueError("RGB dimensions changed within an episode")
                    images[view] = _png(rgb)
                reviewed, revision = False, 0
                boxes, visibility, status = {view: [] for view in VIEWS}, None, None
                if GROUP in episode:
                    labels = episode[GROUP]
                    reviewed, revision = bool(labels["reviewed"][frame_index]), int(labels["revision"][frame_index])
                    if reviewed:
                        row = json.loads(_text(labels["frames_json"][frame_index]))
                        if row.get("schema") == SCHEMA and not isinstance(row.get("visibility"), dict):
                            raise ValueError("Paired v3 annotations require explicit visibility")
                        boxes, visibility = validate_frame(row["boxes"], row["status"], info["shapes"], row.get("visibility"))
                        status = row["status"]
                return dict(episode_id=episode_id, frame_index=frame_index, frame_count=info["frame_count"],
                            filename=str(path.relative_to(self.task_dir)), revision=revision, reviewed=reviewed,
                            timestamp_ns=str(int(episode["timestamps_ns"][frame_index])), source_timestamps_ns={},
                            annotation_image="recorded_rgb", images=images, boxes=boxes, visibility=visibility, status=status)

    def save(self, body):
        import h5py
        with self.lock:
            path, info = self._indices(body.get("episode_id"), body.get("frame_index"))
            index, revision = body["frame_index"], body.get("revision")
            if type(revision) is not int or revision < 0:
                raise ValueError("Frame revision is required")
            boxes, visibility = validate_frame(body.get("boxes"), body.get("status"), info["shapes"], body.get("visibility"))
            now = datetime.now(timezone.utc).isoformat()
            with h5py.File(path, "r+") as episode:
                current = int(episode[GROUP + "/revision"][index]) if GROUP in episode else 0
                if current != revision:
                    raise ValueError("Frame changed in another window; reload before saving")
                if GROUP not in episode:
                    group = episode.require_group(GROUP)
                    group.attrs.update(schema=SCHEMA, annotation_image="recorded_rgb", angle_unit="degrees_clockwise",
                                       coordinate_unit="pixels", obb_fields="cx,cy,width,height,angle_deg,object_id")
                    n, strings = info["frame_count"], h5py.string_dtype("utf-8")
                    for key in ("frames_json", "updated_at_utc"):
                        group.create_dataset(key, (n,), dtype=strings)
                    group.create_dataset("reviewed", (n,), dtype=np.bool_, fillvalue=False)
                    group.create_dataset("revision", (n,), dtype=np.uint64, fillvalue=0)
                    group.create_dataset("frame_index", data=np.arange(n, dtype=np.int64))
                    group.create_dataset("timestamps_ns", data=episode["timestamps_ns"][:])
                group = episode[GROUP]
                row = dict(schema=SCHEMA, annotation_image="recorded_rgb", frame_index=index,
                           timestamp_ns=str(int(episode["timestamps_ns"][index])), status=body["status"],
                           boxes=boxes, visibility=visibility)
                group["reviewed"][index] = False
                episode.flush()
                group.attrs["schema"] = SCHEMA
                group["frames_json"][index] = json.dumps(row, ensure_ascii=False, allow_nan=False)
                group["revision"][index] = current + 1
                group["updated_at_utc"][index] = now
                episode.flush()
                group["reviewed"][index] = True
                episode.flush()
            return dict(episode_id=body["episode_id"], frame_index=index, revision=current + 1,
                        reviewed=True, status=body["status"], output_path=str(path))


def handler_for(dataset):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, status=200):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    payload = Path(__file__).with_name("annotation.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif route.path == "/api/manifest":
                    self.respond(dataset.manifest())
                elif route.path == "/api/frame":
                    params = parse_qs(route.query)
                    self.respond(dataset.frame(int(params["episode_id"][0]), int(params["frame_index"][0])))
                else:
                    self.respond(dict(error="Unknown endpoint"), 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(dict(error=str(exc)), 400)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/save":
                self.respond(dict(error="Unknown endpoint"), 404)
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.respond(dict(error="Cross-origin request rejected"), 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Expected JSON body up to 64 KiB")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                self.respond(dataset.save(body))
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(dict(error=str(exc)), 400)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="HDF5 file or directory")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    dataset = PairedAnnotations(args.dataset)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dataset))
    print(f"Paired OBB annotation: http://127.0.0.1:{server.server_port} (Save writes labels to original HDF5)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
