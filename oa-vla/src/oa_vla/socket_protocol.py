"""Bounded binary TCP protocol; no Torch, OpenPI or robot dependencies."""

from io import BytesIO
import math
import socket
import struct
import uuid

import msgpack
import numpy as np
from PIL import Image

PROTOCOL = "oa_vla_tcp_v1"
VIEWS = ("front", "side", "right_wrist", "left_wrist")
MAX_MESSAGE = 32 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
DELTA_LAYOUT = "interleaved_joint_delta_gripper_target"
ABSOLUTE_LAYOUT = "absolute_q12_gripper2"


class ProtocolError(RuntimeError):
    pass


def send_message(connection, message):
    payload = msgpack.packb(message, use_bin_type=True)
    if not 0 < len(payload) <= MAX_MESSAGE:
        raise ProtocolError("Message exceeds 32 MiB limit")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(connection):
    def read(count):
        result = bytearray()
        while len(result) < count:
            part = connection.recv(count - len(result))
            if not part:
                raise EOFError("OA-VLA socket disconnected")
            result.extend(part)
        return bytes(result)
    size, = struct.unpack("!I", read(4))
    if not 0 < size <= MAX_MESSAGE:
        raise ProtocolError("Invalid message size")
    try:
        message = msgpack.unpackb(read(size), raw=False, strict_map_key=True,
                                  max_str_len=1024 * 1024, max_bin_len=MAX_MESSAGE,
                                  max_array_len=65536, max_map_len=4096, max_ext_len=0)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("Invalid MessagePack message") from exc
    if not isinstance(message, dict) or message.get("protocol") != PROTOCOL:
        raise ProtocolError("Unsupported OA-VLA protocol")
    return message


def encode_rgb(image, *, encoding="raw", quality=90):
    image = np.asarray(image)
    if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 1 or image.shape[0] * image.shape[1] > MAX_PIXELS):
        raise ProtocolError("RGB must be bounded uint8 [H,W,3]")
    if encoding == "raw":
        payload = image.tobytes(order="C")
    elif encoding in ("png", "jpeg"):
        stream = BytesIO()
        Image.fromarray(image).save(stream, format=encoding.upper(), **({"quality": quality} if encoding == "jpeg" else {}))
        payload = stream.getvalue()
    else:
        raise ProtocolError("RGB encoding must be raw, png or jpeg")
    return dict(shape=list(image.shape), encoding=encoding, data=payload)


def decode_rgb(value):
    if not isinstance(value, dict):
        raise ProtocolError("Image payload must be a map")
    shape, payload = value.get("shape"), value.get("data")
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
            or any(type(x) is not int or x < 1 for x in shape)
            or shape[0] * shape[1] > MAX_PIXELS or not isinstance(payload, bytes)):
        raise ProtocolError("Invalid RGB shape/data")
    if value.get("encoding") == "raw":
        if len(payload) != math.prod(shape):
            raise ProtocolError("RGB byte count differs from shape")
        return np.frombuffer(payload, np.uint8).reshape(shape).copy()
    if value.get("encoding") not in ("png", "jpeg"):
        raise ProtocolError("Unsupported image encoding")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.size != (shape[1], shape[0]) or image.format not in ("PNG", "JPEG"):
                raise ProtocolError("Encoded image dimensions/format differ from declaration")
            return np.array(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot decode RGB image") from exc


def validate_context(context):
    if (not isinstance(context, dict) or type(context.get("frame_id")) is not int or context["frame_id"] < 0
            or type(context.get("capture_ns")) is not int or context["capture_ns"] <= 0
            or not isinstance(context.get("calibration_id"), str)
            or not isinstance(context.get("views"), dict) or set(context["views"]) != set(VIEWS)):
        raise ProtocolError("Invalid frame context")
    for view, info in context["views"].items():
        if (not isinstance(info, dict) or type(info.get("timestamp_ns")) is not int or info["timestamp_ns"] <= 0
                or type(info.get("height")) is not int or type(info.get("width")) is not int
                or min(info["height"], info["width"]) < 1 or info["height"] * info["width"] > MAX_PIXELS):
            raise ProtocolError(f"Invalid {view} capture metadata")
    return context


class PolicyClient:
    """One connection, one in-flight request; never automatically retries a response."""

    def __init__(self, host="127.0.0.1", port=8000, *, timeout=30., connection=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Socket timeout must be positive")
        self.connection = connection or socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            hello = receive_message(self.connection)
            if hello.get("type") != "hello" or not isinstance(hello.get("metadata"), dict):
                raise ProtocolError("Server did not send a metadata handshake")
            self.metadata = hello["metadata"]
        except BaseException:
            self.connection.close()
            raise
        self.session, self.sequence = uuid.uuid4().hex, 0

    def infer(self, rgb, state, prompt, *, context, encoding="raw"):
        validate_context(context)
        if set(rgb) != set(VIEWS):
            raise ProtocolError("Exactly four RGB views required")
        state = np.asarray(state, np.float32)
        if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
            raise ProtocolError("State must be a finite vector")
        request_id = uuid.uuid4().hex
        message = dict(protocol=PROTOCOL, type="infer", session=self.session, sequence=self.sequence,
                       request_id=request_id, context=context, state=state.tolist(), prompt=str(prompt),
                       images={view: encode_rgb(rgb[view], encoding=encoding) for view in VIEWS})
        for view, image in rgb.items():
            if list(np.asarray(image).shape[:2]) != [context["views"][view]["height"], context["views"][view]["width"]]:
                raise ProtocolError("RGB dimensions differ from capture context")
        send_message(self.connection, message)
        response = receive_message(self.connection)
        if any(response.get(key) != message[key] for key in ("session", "sequence", "request_id", "context")):
            raise ProtocolError("Response does not match this request/frame")
        self.sequence += 1
        if response.get("type") == "error":
            raise ProtocolError(f"OA-VLA inference failed: {response.get('error')}")
        if response.get("type") != "prediction" or not isinstance(response.get("prediction"), dict):
            raise ProtocolError("Missing prediction")
        return response["prediction"]

    def close(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
