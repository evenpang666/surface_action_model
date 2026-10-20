"""Stop-and-wait VLA simulator: send exactly one chunk, await execution ACK."""
import argparse
import json
from pathlib import Path
import socket
import struct
import uuid

import numpy as np

PROTOCOL = "dual_ur7e_action_chunk_v1"
MAX_MESSAGE = 4*1024*1024
MAX_CHUNK = 1024


class ProtocolError(RuntimeError):
    pass


def send_message(connection, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE:
        raise ProtocolError("Action message exceeds size limit")
    connection.sendall(struct.pack("!I", len(payload))+payload)


def receive_message(connection):
    def read(count):
        pieces = bytearray()
        while len(pieces) < count:
            part = connection.recv(count-len(pieces))
            if not part:
                raise EOFError("Action stream disconnected")
            pieces.extend(part)
        return bytes(pieces)
    size, = struct.unpack("!I", read(4))
    if not 0 < size <= MAX_MESSAGE:
        raise ProtocolError("Invalid action message size")
    try:
        result = json.loads(read(size), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Invalid action JSON") from exc
    if not isinstance(result, dict):
        raise ProtocolError("Action message must be an object")
    return result


def validate_hello(message):
    if message.get("type") != "hello" or message.get("protocol") != PROTOCOL:
        raise ProtocolError("Unsupported action stream protocol")
    if not isinstance(message.get("session"), str) or not message["session"]:
        raise ProtocolError("Missing stream session")
    count, period = message.get("total_actions"), message.get("period_s")
    if type(count) is not int or count < 1:
        raise ProtocolError("Invalid action count")
    if not isinstance(period, (int, float)) or not np.isfinite(period) or not .002 <= period <= 2.:
        raise ProtocolError("Invalid control period")
    initial = np.asarray(message.get("initial_state"), float)
    if initial.shape != (14,) or not np.isfinite(initial).all() or np.any((initial[12:] < 0) | (initial[12:] > 1)):
        raise ProtocolError("Invalid initial state")
    if message.get("action_layout") != "absolute_q12_gripper2" or message.get("coordinate_frame") != "left_base":
        raise ProtocolError("Action layout or coordinate frame mismatch")
    return message


def send_session(connection, policy, *, chunk_size=32, progress=None):
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_CHUNK:
        raise ValueError("chunk-size must be 1..1024")
    session = uuid.uuid4().hex
    hello = {"type": "hello", "protocol": PROTOCOL, "session": session,
             "period_s": policy.period, "total_actions": len(policy.actions),
             "initial_state": policy.metadata["initial_state"],
             "action_layout": "absolute_q12_gripper2", "coordinate_frame": "left_base"}
    validate_hello(hello)
    send_message(connection, hello)
    ready = receive_message(connection)
    if ready.get("type") != "ready" or ready.get("session") != session:
        raise ProtocolError(f"Receiver refused stream: {ready}")
    start, sequence = 0, 0
    while start < len(policy.actions):
        actions = policy.infer(start, chunk_size)
        send_message(connection, {"type": "chunk", "session": session, "sequence": sequence,
                                  "start": start, "actions": actions.tolist()})
        while True:
            reply = receive_message(connection)
            if reply.get("session") != session or reply.get("sequence") != sequence:
                raise ProtocolError("Execution acknowledgement/session is out of order")
            if reply.get("type") == "progress":
                if progress:
                    progress(reply)
                continue
            if reply.get("type") != "executed" or reply.get("next_action") != start+len(actions):
                raise ProtocolError(f"Chunk was not executed: {reply}")
            break
        if progress:
            progress(reply)
        start += len(actions)
        sequence += 1
    send_message(connection, {"type": "end", "session": session, "next_action": start})
    reply = receive_message(connection)
    if reply.get("type") != "complete" or reply.get("session") != session or reply.get("next_action") != start:
        raise ProtocolError("Missing final execution acknowledgement")
    return {"completed": True, "actions_sent": start, "chunks_sent": sequence}


def receive_session(connection, execute_chunk, *, start_session=None):
    """ACK only after execute_chunk returns; exceptions are never acknowledged."""
    hello = validate_hello(receive_message(connection))
    session, cursor, sequence = hello["session"], 0, 0
    try:
        if start_session:
            start_session(hello)
        send_message(connection, {"type": "ready", "session": session})
        while True:
            message = receive_message(connection)
            if message.get("session") != session:
                raise ProtocolError("Stream session changed")
            if message.get("type") == "end":
                if cursor != hello["total_actions"] or message.get("next_action") != cursor:
                    raise ProtocolError("Premature stream end")
                send_message(connection, {"type": "complete", "session": session, "next_action": cursor})
                return {"completed": True, "actions_executed": cursor, "chunks_executed": sequence}
            if (message.get("type") != "chunk" or type(message.get("sequence")) is not int
                    or message["sequence"] != sequence or type(message.get("start")) is not int
                    or message["start"] != cursor):
                raise ProtocolError("Duplicate or out-of-order action chunk")
            actions = np.asarray(message.get("actions"), float)
            if (actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= MAX_CHUNK
                    or not np.isfinite(actions).all() or np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1))
                    or cursor+len(actions) > hello["total_actions"]):
                raise ProtocolError("Malformed action chunk")
            def progress(index, **details):
                send_message(connection, {"type": "progress", "session": session,
                                          "sequence": sequence, "action_index": index, **details})
            execute_chunk(actions, cursor, progress)
            cursor += len(actions)
            send_message(connection, {"type": "executed", "session": session,
                                      "sequence": sequence, "next_action": cursor})
            sequence += 1
    except BaseException as exc:
        try:
            send_message(connection, {"type": "error", "session": session, "sequence": sequence,
                                      "action_index": cursor, "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--chunk-size", type=int, default=32, help="Actions per chunk (default: 32)")
    parser.add_argument("--timeout", type=float, default=120., help="No-message timeout; progress renews it")
    args = parser.parse_args(argv)
    if not np.isfinite(args.timeout) or args.timeout <= 0 or not 1 <= args.port <= 65535:
        parser.error("timeout and port must be positive and valid")
    from obstacle_validation.lerobot_replay import DatasetActionPolicy
    policy = DatasetActionPolicy(args.dataset)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port)); listener.listen(1)
        print(f"[VLA simulator] listening on {args.host}:{args.port}; chunk={args.chunk_size}", flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(args.timeout)
            result = send_session(connection, policy, chunk_size=args.chunk_size,
                                  progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
