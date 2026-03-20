"""SentenceTransformer embedding backend."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding via ``sentence-transformers``.

    Args:
        model_name: HuggingFace model name.
            Default ``"multi-qa-mpnet-base-cos-v1"`` (same default as
            the previous inline usage in ``LocalMemoryBackend``).
        normalize: L2-normalise output vectors.  Must be ``True`` for
            ``LocalMemoryBackend`` dedup (dot product == cosine sim).
    """

    def __init__(
        self,
        model_name: str = "multi-qa-mpnet-base-cos-v1",
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        # If the model is cached locally, avoid hitting HF Hub with a
        # potentially expired token by trying local_files_only first.
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self._normalize = normalize

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=self._normalize)

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
