#!/usr/bin/env python3
"""Build the pinned native Voxblox TSDF/ESDF binding, without ROS or system installation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "outputs/native/voxblox")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prefix", action="append", default=[], help="Additional CMake dependency prefix, e.g. a conda root.")
    parser.add_argument("--voxblox-source", type=Path)
    parser.add_argument("--minkindr-source", type=Path)
    parser.add_argument("--pybind11-dir", type=Path, help="Directory containing pybind11Config.cmake.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cmd = ["cmake", "-S", str(ROOT / "native/voxblox"), "-B", str(args.build_dir),
           "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}"]
    prefixes = list(args.prefix)
    # Conda envs often use development libraries installed in their base env.
    for prefix in (Path(sys.prefix), Path(sys.prefix).parent.parent):
        if (prefix / "lib/cmake/glog/glog-config.cmake").is_file() and str(prefix) not in prefixes:
            prefixes.append(str(prefix))
    if prefixes:
        cmd += ["-DCMAKE_PREFIX_PATH=" + ";".join(prefixes)]
    for name, path in (("FETCHCONTENT_SOURCE_DIR_VOXBLOX_SOURCE", args.voxblox_source),
                       ("FETCHCONTENT_SOURCE_DIR_MINKINDR_SOURCE", args.minkindr_source),
                       ("pybind11_DIR", args.pybind11_dir)):
        if path is not None:
            cmd.append(f"-D{name}={path.resolve()}")
    subprocess.run(cmd, check=True)
    subprocess.run(["cmake", "--build", str(args.build_dir), "--parallel", str(args.jobs)], check=True)
    print(f"Native Voxblox binding built for {sys.executable}")

if __name__ == "__main__":
    main()
