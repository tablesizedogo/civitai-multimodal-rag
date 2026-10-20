"""Thin wrappers around the CLIP and sentence-transformers encoders."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .config import CLIP_MODEL, CLIP_PRETRAINED, TEXT_MODEL


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class ClipEncoder:
    def __init__(self, model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"{model_name}/{pretrained}"

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        return _l2(self.model.encode_image(batch).float().numpy())

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts)
        return _l2(self.model.encode_text(tokens).float().numpy())


class TextEncoder:
    """multilingual-e5 expects "query: " / "passage: " prefixes."""

    def __init__(self, model_name: str = TEXT_MODEL, max_seq_length: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.name = model_name
        self._e5 = "e5" in model_name.lower()

    def encode_passages(self, texts: list[str], batch_size: int = 64, progress: bool = True) -> np.ndarray:
        if self._e5:
            texts = ["passage: " + t for t in texts]
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=progress
        ).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = ["query: " + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


# Multilingual CLIP variant (XLM-R text tower) for non-English queries.
MCLIP_MODEL, MCLIP_PRETRAINED = "xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"


@lru_cache(maxsize=2)
def get_clip(model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED) -> ClipEncoder:
    return ClipEncoder(model_name, pretrained)


@lru_cache(maxsize=1)
def get_text_encoder() -> TextEncoder:
    return TextEncoder()
