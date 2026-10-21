#!/usr/bin/env python3
"""Query four recorded RGB frames; print actions and front/side boxes, no robot I/O."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openpi" / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(ROOT))
from safety_module.policy_contract import POLICY_VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--qpos", type=float, nargs="+", required=True)
    for view in POLICY_VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    request = {"prompt": args.prompt, "qpos": np.asarray(args.qpos, np.float32)}
    for view in POLICY_VIEWS:
        with Image.open(getattr(args, view)) as image:
            request[f"rgb_{view}"] = np.asarray(image.convert("RGB"))
    result = WebsocketClientPolicy(host=args.host, port=args.port).infer(request)
    print(json.dumps(result, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value, ensure_ascii=False))


if __name__ == "__main__":
    main()
