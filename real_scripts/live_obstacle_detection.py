#!/usr/bin/env python3
"""Obstacle detection: dual (default), --scene single-pika, or --no-robot reconstruction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_scripts.live_dual_ur7e_obstacle_model import main

if __name__ == "__main__":
    main()
