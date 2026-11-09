#!/usr/bin/env python3
"""Copy official LIBERO-Safety v2.1 to canonical LeRobot keys (no video decoding).

Run LeRobot's official v2.1 -> v3 converter afterwards, on this derived copy.
Renaming actions through --rename_map alone is too late for chunk timestamps.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np

RENAME = {"actions": "action", "observation.image": "observation.images.image",
          "observation.wrist_image": "observation.images.image2"}


def rename_metadata(value):
    if isinstance(value, dict):
        if len({RENAME.get(k, k) for k in value}) != len(value):
            raise ValueError("Dataset already contains colliding source/canonical keys")
        return {RENAME.get(k, k): rename_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_metadata(v) for v in value]
    return value


def rescale_stats(value, scale):
    if isinstance(value, dict):
        if "action" in value and isinstance(value["action"], dict):
            for key in ("min", "max", "mean", "std", "q01", "q99"):
                if key in value["action"]:
                    value["action"][key] = (np.asarray(value["action"][key])/scale).tolist()
        for child in value.values():
            rescale_stats(child, scale)
    elif isinstance(value, list):
        for child in value:
            rescale_stats(child, scale)
    return value


def normalize(source, destination, *, link_videos=False, action_units="normalized-osc"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and derived dataset must be separate sibling directories")
    info = json.loads((source/"meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This normalizer expects the official v2.1 snapshot; inspect newer versions before migrating")
    for key, shape in (("actions", [7]), ("observation.state", [8]),
                       ("observation.image", [256, 256, 3]), ("observation.wrist_image", [256, 256, 3])):
        if info["features"].get(key, {}).get("shape") != shape:
            raise ValueError(f"Unexpected official feature {key}")
    if info.get("fps") != 20:
        raise ValueError("Expected 20 Hz demonstrations")
    if action_units not in ("metric-delta", "normalized-osc"):
        raise ValueError("Declare metric-delta or normalized-osc")
    scale = np.array([.05]*3+[.5]*3+[1.]) if action_units == "metric-delta" else np.ones(7)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"INCOMPLETE").write_text("Do not train until normalization completes.\n")
    count = dict(parquet=0, videos=0)
    for directory in ("meta", "data", "videos"):
        for path in sorted((source/directory).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            target = destination.joinpath(*(RENAME.get(part, part) for part in relative.parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".parquet":
                table = pq.read_table(path)
                names = [RENAME.get(name, name) for name in table.column_names]
                if len(set(names)) != len(names):
                    raise ValueError(f"Colliding parquet columns in {path}")
                table = table.rename_columns(names)
                if "action" in names:
                    index = names.index("action")
                    actions = np.asarray(table["action"].to_pylist(), np.float32)/scale
                    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid action column in {path}")
                    table = table.set_column(index, table.schema.field(index),
                                             pa.array(actions.tolist(), type=table.schema.field(index).type))
                metadata = dict(table.schema.metadata or {})
                if b"huggingface" in metadata:
                    metadata[b"huggingface"] = json.dumps(rename_metadata(json.loads(metadata[b"huggingface"]))).encode()
                pq.write_table(table.replace_schema_metadata(metadata), target)
                count["parquet"] += 1
            elif path.suffix in (".json", ".jsonl"):
                if path.suffix == ".json":
                    target.write_text(json.dumps(rescale_stats(rename_metadata(json.loads(path.read_text())), scale), indent=2)+"\n")
                else:
                    with path.open() as src, target.open("w") as dst:
                        for line in src:
                            if line.strip():
                                dst.write(json.dumps(rescale_stats(rename_metadata(json.loads(line)), scale))+"\n")
            else:
                if link_videos and directory == "videos":
                    try:
                        os.link(path, target)
                    except OSError:
                        shutil.copy2(path, target)
                else:
                    shutil.copy2(path, target)
                count["videos"] += int(directory == "videos")
    expected_videos = info["total_episodes"]*2
    if count["parquet"] != info["total_episodes"] or count["videos"] != expected_videos:
        raise ValueError(f"Incomplete source: {count}; expected {info['total_episodes']} episodes / {expected_videos} videos")
    (destination/"normalization.json").write_text(json.dumps(dict(source=str(source), rename=RENAME, counts=count,
        source_action_units=action_units, action_divisor=scale.tolist(), action_semantics="libero_osc_pose"), indent=2)+"\n")
    (destination/"INCOMPLETE").unlink()
    return count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--link-videos", action="store_true", help="Hardlink immutable videos on the same filesystem; copy otherwise")
    p.add_argument("--action-units", choices=("metric-delta", "normalized-osc"), required=True,
                   help="Official snapshot stores metric deltas; convert xyz/0.05 and rotation/0.5 for default OSC")
    a = p.parse_args(argv)
    print(json.dumps(normalize(a.source, a.output, link_videos=a.link_videos, action_units=a.action_units), indent=2))


if __name__ == "__main__":
    main()
