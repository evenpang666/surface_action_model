#!/usr/bin/env python3
"""Merge independent method runs, refusing incompatible protocols/episodes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obstacle_validation.libero_metrics import export_comparison


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    records, reference, expected = {}, None, None
    ignored = {"method", "host", "port", "endpoints", "output", "config_dir", "video", "encoding", "manifest", "libero_root"}
    state_hashes = {}
    for root in args.runs:
        config = json.loads((root/"run_config.json").read_text())
        protocol = dict(protocol=config["protocol"], scenarios=config["scenarios"],
            arguments={k: v for k, v in config["arguments"].items() if k not in ignored})
        if reference is not None and protocol != reference:
            raise ValueError(f"Evaluation protocols differ: {root}")
        reference = protocol
        count = sum(len(s["episodes"]) for s in config["scenarios"])
        expected = count if expected is None else expected
        if count != expected:
            raise ValueError("Requested episode counts differ")
        for file in sorted(root.glob("summary_*.json")):
            method = file.stem.removeprefix("summary_")
            if method in records:
                raise ValueError(f"Duplicate method run: {method}")
            reports = json.loads(file.read_text())["episodes"]
            for report in reports:
                key = tuple(report[k] for k in ("suite", "level", "task_id", "episode", "seed"))
                if key in state_hashes and state_hashes[key] != report["state_hash"]:
                    raise ValueError(f"Initial physics states differ for {key}")
                state_hashes[key] = report["state_hash"]
            records[method] = reports
    args.output.mkdir(parents=True, exist_ok=False)
    export_comparison(args.output, records, expected=expected)
    (args.output/"sources.json").write_text(json.dumps([str(p.resolve()) for p in args.runs], indent=2)+"\n")
    print((args.output/"table.md").read_text())


if __name__ == "__main__":
    main()
