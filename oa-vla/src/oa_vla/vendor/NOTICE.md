# Third-party sources

The action flow model and PaliGemma/action-expert stack are adapted from the
Physical Intelligence OpenPI PyTorch implementation (Apache-2.0):
https://github.com/Physical-Intelligence/openpi
The initial extraction uses this repository's patched snapshot of those files.

`transformers/` contains the repository's modified Gemma/SigLIP/PaliGemma
implementations based on Hugging Face Transformers 4.53.2 (Apache-2.0). Original
copyright headers remain. Imports are private and the PaliGemma constructor
directly instantiates these private Gemma/SigLIP classes; no global Transformers
classes or site-packages files are replaced.

Gemma-based pretrained weights are subject to the included Gemma terms. Weights
are not bundled. `sam3/` retains its separate SAM License and source revision
in `sam3/NOTICE.md`. The package's original code does not override those licenses.
