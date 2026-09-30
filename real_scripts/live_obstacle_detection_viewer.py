#!/usr/bin/env python3
"""Open the obstacle viewer; this process never owns a camera or robot."""
import argparse
import webbrowser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    args = parser.parse_args()
    webbrowser.open(args.url)
