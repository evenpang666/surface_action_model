"""Predict actions and paired front/side OBB from four RGB image files; no robot/network I/O."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .contract import POLICY_VIEWS
from .policy import OAVLAPolicy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--state", "--qpos", type=float, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=.5)
    parser.add_argument("--visibility-threshold", type=float, default=.5)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    for view in POLICY_VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args(argv)
    rgb = {}
    for view in POLICY_VIEWS:
        with Image.open(getattr(args, view)) as image:
            rgb[view] = np.asarray(image.convert("RGB"))
    torch.manual_seed(args.seed)
    policy = OAVLAPolicy.from_pretrained(args.model, device=args.device, tokenizer_model=args.tokenizer_model,
                                        score_threshold=args.score_threshold, visibility_threshold=args.visibility_threshold)
    result = policy.predict(rgb, args.state, args.prompt, num_steps=args.num_steps)
    serialized = json.dumps(result, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
