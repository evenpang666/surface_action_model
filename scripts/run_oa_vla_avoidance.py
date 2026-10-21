#!/usr/bin/env python3
"""Run local shared-camera OA-VLA inference and Voxblox/ellipsoid avoidance."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_validation.oa_vla_runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
