"""Create a tiny, random, offline checkpoint for socket tests, never robot execution."""

import argparse
from pathlib import Path


def create_test_checkpoint(output, *, seed=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Test checkpoint already exists: {output}")
    import torch
    from .config import OAVLAConfig
    from .model import OAVLA
    from .policy import OAVLAPolicy
    from .tokenizer import ByteTokenizer
    from .socket_protocol import DELTA_LAYOUT
    torch.manual_seed(seed)
    model = OAVLA(OAVLAConfig.tiny())
    metadata = dict(test_only=True, detection_trained=False, training_stage="untrained_test_fixture",
                    action_semantics=DELTA_LAYOUT, action_source="commanded", action_period_s=.05,
                    qpos_mean=[0.] * 12, qpos_std=[1.] * 12, action_mean=[0.] * 14, action_std=[1.] * 14,
                    initialization_note="Tiny random checkpoint for transport/shape tests; no trained action or OBB capability")
    OAVLAPolicy(model, ByteTokenizer(32), metadata=metadata).save_pretrained(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(create_test_checkpoint(args.output, seed=args.seed))


if __name__ == "__main__":
    main()
