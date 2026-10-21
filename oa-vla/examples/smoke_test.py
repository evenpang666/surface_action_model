"""Exercise real training/export/reload/inference with a tiny random CPU model."""

from pathlib import Path
import tempfile
import numpy as np

from oa_vla import OAVLAPolicy
from oa_vla.contract import POLICY_VIEWS
from oa_vla.train import main as train
from make_demo_dataset import make_dataset


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    shard = make_dataset(root / "demo.npz")
    train(["--dataset", str(shard), "--output", str(root / "policy"), "--tiny", "--allow-random-init",
           "--device", "cpu", "--epochs", "1", "--max-steps", "1"])
    policy = OAVLAPolicy.from_pretrained(root / "policy", score_threshold=0)
    result = policy.predict({view: np.zeros((60, 80, 3), np.uint8) for view in POLICY_VIEWS},
                            np.zeros(12), "move around the obstacle", num_steps=1)
    print({"model": result["model_type"], "actions": result["actions"].shape,
           "box_views": list(result["detections"]), "status": result["detection_status"]})
