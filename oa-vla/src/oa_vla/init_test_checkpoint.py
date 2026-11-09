"""Create a tiny, random, offline checkpoint for socket tests, never robot execution."""

import argparse
from pathlib import Path


def create_test_checkpoint(output, *, seed=0, profile="dual-ur7e"):
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
    if profile not in ("dual-ur7e", "libero-safety"):
        raise ValueError("Unknown test profile")
    state_dim, action_dim = (8, 7) if profile == "libero-safety" else (12, 14)
    model = OAVLA(OAVLAConfig.tiny(state_dim=state_dim, logical_action_dim=action_dim))
    metadata = dict(test_only=True, detection_trained=False, training_stage="untrained_test_fixture",
                    action_semantics=DELTA_LAYOUT, action_source="commanded", action_period_s=.05,
                    qpos_mean=[0.] * state_dim, qpos_std=[1.] * state_dim,
                    action_mean=[0.] * action_dim, action_std=[1.] * action_dim,
                    initialization_note="Tiny random checkpoint for transport/shape tests; no trained action or OBB capability")
    if profile == "libero-safety":
        metadata.update(action_semantics="libero_osc_pose", state_semantics="libero_eef_axis_angle_gripper",
                        image_convention="opencv_top_left",
                        camera_names=dict(front="agentview", side="sideview",
                                          right_wrist="robot0_eye_in_hand", left_wrist="frontview"))
    OAVLAPolicy(model, ByteTokenizer(32), metadata=metadata).save_pretrained(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=("dual-ur7e", "libero-safety"), default="dual-ur7e")
    args = parser.parse_args(argv)
    print(create_test_checkpoint(args.output, seed=args.seed, profile=args.profile))


if __name__ == "__main__":
    main()
