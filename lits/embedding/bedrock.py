"""AWS Bedrock embedding backend — supports all Bedrock embedding model families.

Model families and their API differences:

+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Family           | Request body                                 | Response body              | Native batch| Server-side normalize  |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Titan            | {"inputText": str, "dimensions": int,        | {"embedding": [...]}       | No          | Yes                    |
| (amazon.titan-*) |  "normalize": bool}                          |                            |             |                        |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Cohere           | {"texts": [...], "input_type": str}          | {"embeddings": [[...],..]} | Yes (96)    | No (client-side)       |
| (cohere.embed-*) |                                              |                            |             |                        |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import numpy as np

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class BedrockEmbedder(BaseEmbedder):
    """AWS Bedrock embedding backend for all model families.

    Supports:
    - ``amazon.titan-embed-text-v2:0`` (and other Titan models)
    - ``cohere.embed-english-v3``, ``cohere.embed-multilingual-v3``, ``cohere.embed-v4``
    - Any future Bedrock embedding model (add a new family branch)

    The model family is auto-detected from ``model_id`` prefix.  Callers
    never need to know whether the underlying model is Titan or Cohere.

    Args:
        model_id: Bedrock model identifier
            (e.g. ``"cohere.embed-english-v3"``).
        region: AWS region.  If ``None``, uses default boto3 session.
        dimensions: Output dimensionality (Titan only; Cohere ignores).
        input_type: Cohere ``input_type`` field
            (``"search_document"`` or ``"search_query"``).
            Titan ignores this.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: Optional[str] = None,
        dimensions: int = 1024,
        input_type: str = "search_document",
    ):
        import boto3

        self.model_id = model_id
        self._dimensions = dimensions
        self._input_type = input_type

        session = boto3.Session(region_name=region) if region else boto3.Session()
        self._client = session.client("bedrock-runtime")

        # Detect model family from prefix
        if model_id.startswith("cohere."):
            self._family = "cohere"
        elif model_id.startswith("amazon."):
            self._family = "titan"
        else:
            self._family = "titan"  # fallback
            logger.warning(
                "Unknown Bedrock embedding model family for '%s'; "
                "falling back to Titan API format.",
                model_id,
            )

        # Probe actual embedding dim
        self._embedding_dim: int = self._probe_dim()
        logger.info(
            "BedrockEmbedder ready: model=%s family=%s dim=%d",
            model_id,
            self._family,
            self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._call_api(texts)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _probe_dim(self) -> int:
        """Embed a short text to discover the output dimensionality."""
        vec = self._call_api(["probe"])
        return vec.shape[1]

    def _call_api(self, texts: List[str]) -> np.ndarray:
        if self._family == "cohere":
            return self._call_cohere(texts)
        return self._call_titan(texts)

    # ------------------------------------------------------------------
    # Titan (amazon.titan-embed-*)
    # ------------------------------------------------------------------

    def _call_titan(self, texts: List[str]) -> np.ndarray:
        """Titan: one ``invoke_model`` call per text, server-side normalisation."""
        vecs = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": self._dimensions,
                    "normalize": True,
                }
            )
            resp = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            vec = json.loads(resp["body"].read())["embedding"]
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Cohere (cohere.embed-*)
    # ------------------------------------------------------------------

    def _call_cohere(
        self, texts: List[str], batch_size: int = 96
    ) -> np.ndarray:
        """Cohere: native batching, client-side L2-normalisation."""
        all_vecs: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            body = json.dumps(
                {
                    "texts": batch,
                    "input_type": self._input_type,
                }
            )
            resp = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            embeddings = json.loads(resp["body"].read())["embeddings"]
            all_vecs.extend(embeddings)

        mat = np.array(all_vecs, dtype=np.float32)
        # Client-side L2-normalisation (Cohere doesn't normalise server-side)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        return mat / norms
