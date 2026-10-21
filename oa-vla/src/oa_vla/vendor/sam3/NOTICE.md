# SAM 3 detection subset

Source: https://github.com/facebookresearch/sam3
Revision: `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`
Retrieved: 2026-10-21

Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
Distributed under the included SAM License (`LICENSE`), not MIT.

Extracted `TransformerDecoderLayer`, `TransformerDecoder`, their scoring/MLP/
positional helpers, activation checkpoint wrapper and sine position encoder.
This is actual SAM 3 decoder code; PE, masks, video, 3D and full model-builder
imports are excluded. Policy-specific adaptations are in `../../model.py`.

Changes to the extracted decoder: remove unused ROI pooling; initialize spatial
coordinate caches lazily on the input device (no construction-time CUDA); permit
disabling activation checkpointing. The wrapper uses PyTorch's state-compatible
`nn.MultiheadAttention` for vanilla cross-attention, without optional FA3/xformers.
The decoder's learned parameter names and shapes are retained.

OA-VLA adds an optional per-layer `pair_fusion` callback before spatial reference
refinement. The native learned tensors remain unchanged; new fusion/OBB/pair
heads live in the OA-VLA wrapper and are initialized separately.
