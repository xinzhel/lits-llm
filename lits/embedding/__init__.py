"""Embedding subpackage — unified interface for text embedding backends.

Mirrors the ``lits.lm`` pattern: a base ABC, backend implementations,
and a ``get_embedder()`` factory that dispatches by model-name prefix.

Dispatch rules:
- ``"bedrock-embed/<model_id>"`` → :class:`BedrockEmbedder`
- Anything else → :class:`SentenceTransformerEmbedder`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseEmbedder

if TYPE_CHECKING:
    pass

__all__ = ["BaseEmbedder", "get_embedder"]


def get_embedder(model_name: str = "multi-qa-mpnet-base-cos-v1", **kwargs) -> BaseEmbedder:
    """Create an embedder by model name.

    Args:
        model_name: Model identifier string.
            - ``"bedrock-embed/<model_id>"`` dispatches to :class:`BedrockEmbedder`.
            - Anything else dispatches to :class:`SentenceTransformerEmbedder`.
        **kwargs: Passed to the backend constructor.

    Returns:
        A :class:`BaseEmbedder` instance.
    """
    if model_name.startswith("bedrock-embed/"):
        from .bedrock import BedrockEmbedder

        model_id = model_name.split("/", 1)[1]
        return BedrockEmbedder(model_id=model_id, **kwargs)
    else:
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=model_name, **kwargs)
