#!/usr/bin/env python3
"""Serve a Panda OA-VLA or native LeRobot pi05 checkpoint using the shared TCP.

Run on the inference machine in the checkpoint's own Python environment. The
simulator only needs the lightweight oa-vla socket client, not LeRobot/Torch.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"oa-vla/src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("lerobot", "oa-vla"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--test-only", action="store_true", help="Mark an untrained checkpoint; never paper eligible")
    p.add_argument("--normalized-osc-actions", action="store_true",
                   help="Confirm the native checkpoint was trained on normalized LIBERO OSC actions")
    args = p.parse_args(argv)
    from oa_vla.serve import PolicyServer, policy_metadata
    from obstacle_validation.libero_safety_geometry import CAMERAS
    if args.backend == "oa-vla":
        from oa_vla.policy import OAVLAPolicy
        policy = OAVLAPolicy.from_pretrained(args.model, device=args.device)
        metadata = policy_metadata(policy)
    else:
        if not args.normalized_osc_actions:
            p.error("Native pi05 needs --normalized-osc-actions after verifying dataset normalization provenance")
        import torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        config = PI05Config.from_pretrained(str(args.model))
        expected = {"observation.images.image", "observation.images.image2"}
        if (set(config.image_features) != expected or tuple(config.input_features["observation.state"].shape) != (8,)
                or tuple(config.output_features["action"].shape) != (7,)):
            raise ValueError("Native pi05 needs canonical two-camera, 8-state / 7-action LIBERO features")
        config.device = args.device
        model = PI05Policy.from_pretrained(str(args.model), config=config).to(args.device).eval()
        pre, post = make_pre_post_processors(config, pretrained_path=str(args.model),
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        class NativePolicy:
            def predict(self, rgb, state, prompt, **unused):
                batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32), "task": prompt}
                for key, slot in (("observation.images.image", "front"), ("observation.images.image2", "right_wrist")):
                    batch[key] = torch.from_numpy(rgb[slot].copy()).permute(2, 0, 1).float()/255
                model.reset()
                with torch.inference_mode():
                    action = post(model.select_action(pre(batch)))
                return dict(actions=action.cpu().numpy().reshape(1, 7))
        policy = NativePolicy()
        metadata = dict(model_type="lerobot_pi05", state_dim=8, action_dim=7, action_period_s=.05,
            action_semantics="libero_osc_pose", state_semantics="libero_eef_axis_angle_gripper",
            camera_names=CAMERAS, policy_views=list(CAMERAS), image_convention="opencv_top_left",
            checkpoint=str(args.model.resolve()), test_only=args.test_only,
            normalized_osc_actions_confirmed=True)
    metadata["test_only"] = metadata.get("test_only", False) or args.test_only
    if metadata.get("state_dim") != 8 or metadata.get("action_dim") != 7:
        raise ValueError("Dual-UR checkpoints cannot be used as Panda policies")
    with PolicyServer((args.host, args.port), policy, metadata=metadata, num_steps=args.num_steps) as server:
        print(f"LIBERO policy TCP {server.server_address}: {metadata}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
