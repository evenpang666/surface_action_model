"""Serve a portable OA-VLA policy over framed binary TCP; no camera/robot I/O."""

import argparse
import logging
import math
from pathlib import Path
import socket
import socketserver
import threading
import time

import numpy as np

from .socket_protocol import (PROTOCOL, VIEWS, DELTA_LAYOUT, ABSOLUTE_LAYOUT, ProtocolError,
                              send_message, receive_message, decode_rgb, validate_context)


def policy_metadata(policy, *, action_semantics=None, action_source=None, action_period=None):
    config, saved = policy.model.config, policy.metadata
    metadata = dict(test_only=bool(saved.get("test_only", False) or getattr(config, "vocab_size", None) == 512),
                    model_type="oa-vla", model_format=2, policy_views=list(VIEWS), detection_views=["front", "side"],
                    detection_box_format="cxcywh_angle_deg_original_pixels", pair_id_scope="current_prediction",
                    state_dim=config.state_dim, action_dim=config.logical_action_dim, action_horizon=config.action_horizon,
                    detection_trained=bool(saved.get("detection_trained", False)))
    for name, explicit in (("action_semantics", action_semantics), ("action_source", action_source),
                           ("action_period_s", action_period)):
        recorded = saved.get(name)
        if explicit is not None and recorded not in (None, "unknown") and explicit != recorded:
            raise ValueError(f"Explicit {name} conflicts with trained metadata")
        metadata[name] = explicit if explicit is not None else recorded
    period = metadata["action_period_s"]
    if period is not None and (not isinstance(period, (float, int)) or not math.isfinite(period) or not .002 <= period <= 2):
        raise ValueError("Action period must be 0.002..2 seconds")
    return metadata


class PolicyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, policy, *, metadata, num_steps=10, timeout=30.):
        if type(num_steps) is not int or num_steps < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid inference steps or timeout")
        self.policy, self.metadata = policy, metadata
        self.num_steps, self.timeout = num_steps, timeout
        self.inference_lock = threading.Lock()
        super().__init__(address, PolicyHandler)


class PolicyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        connection, server = self.request, self.server
        connection.settimeout(server.timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_message(connection, dict(protocol=PROTOCOL, type="hello", metadata=server.metadata))
        session, sequence = None, 0
        while True:
            request = None
            try:
                request = receive_message(connection)
                if (request.get("type") != "infer" or not isinstance(request.get("session"), str)
                        or not request["session"] or type(request.get("sequence")) is not int
                        or request["sequence"] != sequence or not isinstance(request.get("request_id"), str)
                        or not 0 < len(request["request_id"]) <= 128):
                    raise ProtocolError("Invalid inference envelope/sequence")
                session = session or request["session"]
                if session != request["session"]:
                    raise ProtocolError("Connection session changed")
                context = validate_context(request.get("context"))
                images = request.get("images")
                if not isinstance(images, dict) or set(images) != set(VIEWS):
                    raise ProtocolError("Exactly four images required")
                rgb = {view: decode_rgb(images[view]) for view in VIEWS}
                for view in VIEWS:
                    if list(rgb[view].shape[:2]) != [context["views"][view]["height"], context["views"][view]["width"]]:
                        raise ProtocolError("Image/capture context dimensions differ")
                state = np.asarray(request.get("state"), np.float32)
                if state.shape != (server.metadata["state_dim"],) or not np.isfinite(state).all():
                    raise ProtocolError("State differs from model contract")
                prompt = request.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
                    raise ProtocolError("Nonempty bounded prompt required")
                began = time.monotonic()
                with server.inference_lock:
                    prediction = server.policy.predict(rgb, state, prompt, num_steps=server.num_steps)
                # Only action arrays are converted; image/OBB coordinates retain original resolution.
                prediction = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in prediction.items()}
                response = {key: request[key] for key in ("session", "sequence", "request_id", "context")}
                send_message(connection, dict(protocol=PROTOCOL, type="prediction", **response,
                                              prediction=prediction, inference_seconds=time.monotonic() - began))
                sequence += 1
            except (EOFError, ConnectionError, socket.timeout):
                return
            except Exception as exc:
                logging.exception("OA-VLA request rejected")
                envelope = {key: request.get(key) for key in ("session", "sequence", "request_id", "context")} if isinstance(request, dict) else {}
                try:
                    send_message(connection, dict(protocol=PROTOCOL, type="error", **envelope, error=f"{type(exc).__name__}: {exc}"))
                except OSError:
                    pass
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=.5)
    parser.add_argument("--visibility-threshold", type=float, default=.5)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--action-semantics", choices=(DELTA_LAYOUT, ABSOLUTE_LAYOUT), help="Explicit contract for exports lacking action metadata")
    parser.add_argument("--action-source", choices=("commanded", "executed"))
    parser.add_argument("--action-period", type=float)
    args = parser.parse_args(argv)
    from .policy import OAVLAPolicy
    policy = OAVLAPolicy.from_pretrained(args.model, device=args.device, tokenizer_model=args.tokenizer_model,
                                        score_threshold=args.score_threshold, visibility_threshold=args.visibility_threshold)
    metadata = policy_metadata(policy, action_semantics=args.action_semantics,
                               action_source=args.action_source, action_period=args.action_period)
    with PolicyServer((args.host, args.port), policy, metadata=metadata, num_steps=args.num_steps, timeout=args.timeout) as server:
        print(f"OA-VLA TCP listening on {server.server_address}; {metadata}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
