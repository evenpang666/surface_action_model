#!/usr/bin/env python3
"""Extract single/dual-arm measured surface tracks using the supplied FK gate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_scripts.preprocess_left_vr_hdf5_cotracker_observations import main

if __name__ == "__main__":
    main()
