"""AWS Bedrock embedding backend — supports all Bedrock embedding model families.

Model families and their API differences:

+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Family           | Request body                                 | Response body              | Native batch| Server-side normalize  |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Titan            | {"inputText": str, "dimensions": int,        | {"embedding": [...]}       | No          | Yes                    |
| (amazon.titan-*) |  "normalize": bool}                          |                            |             |                        |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Cohere v3        | {"texts": [...], "input_type": str}          | {"embeddings": [[...],..]} | Yes (96)    | No (client-side)       |
| (cohere.embed-*) |                                              |                            |             |                        |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+
| Cohere v4        | {"texts": [...], "input_type": str,          | {"embeddings": [[...],..]} | Yes (96)    | No (client-side)       |
| (cohere.embed-v4)| "output_dimension": int}                     |                            |             |                        |
+------------------+----------------------------------------------+----------------------------+-------------+------------------------+

Cohere v4 notes:
- Default ``output_dimension`` is 1536 (v3 was 1024).
- Supports Matryoshka dimensions: 256, 512, 1024, 1536.
- Response format differs from v3: v4 always returns
  ``{"embeddings": {"float": [[...]]}}``, even when ``embedding_types``
  is omitted.  v3 returns ``{"embeddings": [[...]]}``.
  The parser handles both via ``isinstance(raw, dict)`` check.
- 128K token context window (v3 was 512 tokens).
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
    - ``cohere.embed-english-v3``, ``cohere.embed-multilingual-v3``
    - ``cohere.embed-v4:0`` (multimodal, Matryoshka dimensions)
    - Any future Bedrock embedding model (add a new family branch)

    The model family is auto-detected from ``model_id`` prefix.  Callers
    never need to know whether the underlying model is Titan or Cohere.

    Args:
        model_id: Bedrock model identifier
            (e.g. ``"cohere.embed-english-v3"``, ``"cohere.embed-v4:0"``).
        region: AWS region.  If ``None``, uses default boto3 session.
        dimensions: Output dimensionality.
            - Titan: passed as ``dimensions`` in request body.
            - Cohere v4: passed as ``output_dimension`` (256/512/1024/1536).
            - Cohere v3: ignored (always 1024).
        input_type: Cohere ``input_type`` field
            (``"search_document"`` or ``"search_query"``).
            Titan ignores this.
    """

    # Cohere v4 model IDs (checked via prefix match)
    _COHERE_V4_PREFIXES = ("cohere.embed-v4",)

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
            self._is_v4 = any(model_id.startswith(p) for p in self._COHERE_V4_PREFIXES)
        elif model_id.startswith("amazon."):
            self._family = "titan"
            self._is_v4 = False
        else:
            self._family = "titan"  # fallback
            self._is_v4 = False
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
        """Cohere: native batching, client-side L2-normalisation.

        For Embed v4, includes ``output_dimension`` in the request body
        to select Matryoshka dimension (256/512/1024/1536).
        """
        all_vecs: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload: dict = {
                "texts": batch,
                "input_type": self._input_type,
            }
            if self._is_v4:
                payload["output_dimension"] = self._dimensions
            body = json.dumps(payload)
            resp = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            raw_embeddings = json.loads(resp["body"].read())["embeddings"]
            # Cohere v4 returns {"embeddings": {"float": [[...]]}}
            # Cohere v3 returns {"embeddings": [[...]]}
            if isinstance(raw_embeddings, dict):
                embeddings = raw_embeddings["float"]
            else:
                embeddings = raw_embeddings
            all_vecs.extend(embeddings)

        mat = np.array(all_vecs, dtype=np.float32)
        # Client-side L2-normalisation (Cohere doesn't normalise server-side)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        return mat / norms
