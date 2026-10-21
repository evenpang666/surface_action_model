#!/usr/bin/env python3
"""Compatibility entry point: train the point-free, four-view SAM 3 + VLA policy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "openpi" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Historical dataset imports remain available for old shard verification only.
from scripts.train_pi05_surface_legacy import Ur7eSurfaceDataset, _normalization_stats
from safety_module.tokenizer import _Tokenizer
from scripts.train_pi05_sam3_pytorch import main, parse_args

if __name__ == "__main__":
    main()
