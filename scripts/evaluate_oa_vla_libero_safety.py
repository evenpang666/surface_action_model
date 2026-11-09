#!/usr/bin/env python3
"""Run the independent OA-VLA / native LeRobot LIBERO-Safety adapter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obstacle_validation.libero_safety import main

if __name__ == "__main__":
    main()
