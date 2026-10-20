#!/usr/bin/env python3
"""Single UR7e + Pika, front-only obstacle detection (receive-only)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_scripts.live_dual_ur7e_obstacle_model import main

if __name__ == "__main__":
    main(default_scene="single-pika")
