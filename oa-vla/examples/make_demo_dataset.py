"""Create a synthetic point-free shard for offline CPU smoke tests only."""

import argparse
from pathlib import Path
import numpy as np


def make_dataset(path):
    rng = np.random.default_rng(7)
    boxes = np.zeros((3, 2, 2, 6), np.float32)
    boxes[0, :, 0] = [.35, .5, .2, .1, 1., 0.]
    boxes[0, :, 1] = [.7, .6, .2, .1, 0., 1.]
    valid = np.zeros((3, 2, 2), bool)
    valid[0] = True
    ids = np.full((3, 2, 2), "", dtype="U8")
    ids[0] = ["obj_a", "obj_b"]
    # Deliberately reverse side drawing order; IDs recover the correspondence.
    boxes[0, 1] = boxes[0, 1, ::-1]
    ids[0, 1] = ids[0, 1, ::-1]
    payload = dict(task_text="move around the obstacle", qpos=rng.normal(0, .1, (3, 12)).astype(np.float32),
                   action_chunks=rng.normal(0, .01, (3, 2, 14)).astype(np.float32),
                   sample_frame_indices=np.arange(3), action_step_mask=np.array([[True, True], [True, False], [False, False]]),
                   detection_boxes=boxes, detection_object_ids=ids, detection_valid=valid, detection_reviewed=np.ones((3, 2), bool),
                   detection_box_format="cxcywh_cos2sin2_normalized_padded", detection_views=np.array(["front", "side"]))
    for index, view in enumerate(("front", "side", "right_wrist", "left_wrist")):
        payload[f"rgb_{view}"] = rng.integers(0, 256, (3, 40 + index * 10, 80, 3), np.uint8)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(make_dataset(parser.parse_args().output))
