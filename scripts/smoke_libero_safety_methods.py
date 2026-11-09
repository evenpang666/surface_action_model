#!/usr/bin/env python3
"""Real socket + untrained OA-VLA + Panda physics; never paper benchmark scores."""
import argparse
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"oa-vla/src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    from oa_vla.init_test_checkpoint import create_test_checkpoint
    from oa_vla.policy import OAVLAPolicy
    from oa_vla.serve import PolicyServer, policy_metadata
    from obstacle_validation.libero_comparison import main as evaluate
    checkpoint = args.output/"untrained_checkpoint"
    create_test_checkpoint(checkpoint, profile="libero-safety", seed=0)
    policy = OAVLAPolicy.from_pretrained(checkpoint)
    # Small random actions exercise the controls rather than reject every random
    # flow output for violating [-1,1]. This is explicit test normalization.
    policy.metadata.update(action_std=[.03]*7, initialization_note="Untrained tiny OA-VLA; action std=.03 for bounded simulation transport/control smoke")
    policy.action_std[:] = .03
    policy.save_pretrained(checkpoint)
    metadata = policy_metadata(policy)
    metadata["inference_noise"] = "deterministic_sha256_rgb_state_prompt_test_only"
    class DeterministicTestPolicy:
        def predict(self, rgb, state, prompt, **kwargs):
            import hashlib
            import torch
            seed = int.from_bytes(hashlib.sha256(rgb["front"].tobytes()+state.tobytes()+prompt.encode()).digest()[:8], "big")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn((1, policy.model.config.action_horizon, policy.model.config.action_dim), generator=generator)
            return policy.predict(rgb, state, prompt, noise=noise, **kwargs)
    with PolicyServer(("127.0.0.1", 0), DeterministicTestPolicy(), metadata=metadata, num_steps=1) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            evaluate(["--fixture", "--allow-test-policy", "--host", "127.0.0.1", "--port", str(server.server_address[1]),
                "--method", "all", "--output", str(args.output/"results"), "--episodes", *map(str, range(args.episodes)),
                "--control-freq", "20", "--warmup-steps", "2", "--max-steps", str(args.steps),
                "--resolution", "96", "--sample-spacing", ".08", "--psf-resolution", ".1",
                "--psf-fine-resolution", ".04", "--max-result-age", "60", "--no-video"])
        finally:
            server.shutdown(); thread.join(timeout=5)


if __name__ == "__main__":
    main()
