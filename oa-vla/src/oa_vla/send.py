"""Send four RGB files/state/prompt to an OA-VLA TCP server; no robot I/O."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from .socket_protocol import PolicyClient, VIEWS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--state", "--qpos", type=float, nargs="+", required=True)
    parser.add_argument("--encoding", choices=("raw", "png", "jpeg"), default="raw")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--output", type=Path)
    for view in VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args(argv)
    rgb = {}
    for view in VIEWS:
        with Image.open(getattr(args, view)) as image:
            rgb[view] = np.asarray(image.convert("RGB"))
    stamp = time.monotonic_ns()
    context = dict(frame_id=0, capture_ns=stamp, calibration_id="offline_files",
                   views={view: dict(timestamp_ns=stamp, height=image.shape[0], width=image.shape[1]) for view, image in rgb.items()})
    with PolicyClient(args.host, args.port, timeout=args.timeout) as client:
        prediction = client.infer(rgb, args.state, args.prompt, context=context, encoding=args.encoding)
        result = dict(server_metadata=client.metadata, context=context, prediction=prediction)
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
