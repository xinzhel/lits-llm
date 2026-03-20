"""Base class for all embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base for all embedding backends.

    Contract:
    - ``embed()`` returns L2-normalised vectors (unit length) so that
      ``dot(a, b) == cosine_similarity(a, b)``.
    - ``embedding_dim`` returns the dimensionality of the output vectors.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: Strings to embed.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), embedding_dim)``,
            dtype ``float32``, L2-normalised (each row has unit norm).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...
