#!/usr/bin/env python3
"""Add exact train-dataset q01/q99 for native LeRobot pi05 QUANTILES normalization.

Run on the canonical derived v3 dataset after migration. Temporary arrays use
disk-backed storage (~207 MB for this benchmark), not decoded video memory.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


def compute(dataset):
    import pyarrow.parquet as pq
    dataset = Path(dataset)
    info = json.loads((dataset/"meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Run the official v2.1 -> v3.0 converter first")
    stats_path = dataset/"meta/stats.json"
    stats = json.loads(stats_path.read_text())
    keys = ("observation.state", "action")
    widths = [info["features"][key]["shape"][0] for key in keys]
    if widths != [8, 7]:
        raise ValueError("Expected canonical LIBERO 8-state/7-action data")
    count = int(info["total_frames"])
    with tempfile.TemporaryDirectory(prefix="quantiles_", dir=dataset) as tmp:
        array = np.memmap(Path(tmp)/"vectors.dat", dtype=np.float32, mode="w+", shape=(count, sum(widths)))
        offset = 0
        for path in sorted((dataset/"data").rglob("*.parquet")):
            table = pq.read_table(path, columns=list(keys))
            values = np.concatenate([np.asarray(table[k].to_pylist(), np.float32) for k in keys], axis=1)
            if offset+len(values) > count or not np.isfinite(values).all():
                raise ValueError("Invalid frame count or non-finite training vectors")
            array[offset:offset+len(values)] = values
            offset += len(values)
        if offset != count:
            raise ValueError("Incomplete training dataset")
        qs = np.array([np.quantile(array[:, i], [.01, .99]) for i in range(sum(widths))]).T
        start = 0
        for key, width in zip(keys, widths):
            stats[key]["q01"], stats[key]["q99"] = qs[:, start:start+width].tolist()
            start += width
        del array
    # A temporary file and atomic replace avoid a partially written stats file.
    temporary = stats_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stats, indent=2)+"\n")
    temporary.replace(stats_path)
    return dict(frames=count, quantiles={key: {q: stats[key][q] for q in ("q01", "q99")} for key in keys})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(compute(a.dataset), indent=2))


if __name__ == "__main__":
    main()
