"""Shared PI0.5 observation tokenization for training and serving."""
from pathlib import Path
import os
import numpy as np

MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULT_TOKENIZER_REPOSITORY = "google/paligemma-3b-pt-224"
DEFAULT_TOKENIZER_FILENAME = "tokenizer.model"

def _cached_huggingface_tokenizer() -> Path | None:
    """Find a tokenizer even in a manually copied Hugging Face snapshot cache."""
    hub_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    snapshot_root = hub_root / "models--google--paligemma-3b-pt-224" / "snapshots"
    candidates = sorted(snapshot_root.glob(f"*/{DEFAULT_TOKENIZER_FILENAME}"))
    return candidates[-1] if candidates else None


class _Tokenizer:
    def __init__(self, max_len: int, model_path: Path | None):
        self.max_len = int(max_len)
        if model_path is None:
            model_path = _cached_huggingface_tokenizer()
        if model_path is None:
            try:
                from huggingface_hub import hf_hub_download

                model_path = Path(
                    hf_hub_download(repo_id=DEFAULT_TOKENIZER_REPOSITORY, filename=DEFAULT_TOKENIZER_FILENAME)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not obtain the PaliGemma tokenizer from the Hugging Face cache or Hub. "
                    "Check network access or pass --tokenizer-model explicitly."
                ) from exc
        import sentencepiece

        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clean = prompt.strip().replace("_", " ").replace("\n", " ")
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {clean}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self._processor.encode(text, add_bos=True)[: self.max_len]
        mask = np.zeros((self.max_len,), dtype=bool)
        mask[: len(tokens)] = True
        result = np.zeros((self.max_len,), dtype=np.int32)
        result[: len(tokens)] = tokens
        return result, mask
