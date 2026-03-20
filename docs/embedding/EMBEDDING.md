# Embedding Subpackage (`lits/embedding/`)

Unified interface for text embedding backends. Mirrors the `lits/lm/` pattern: a base ABC, backend implementations, and a `get_embedder()` factory.

## API

```python
from lits.embedding import get_embedder, BaseEmbedder
```

### `BaseEmbedder` (ABC)

| Method / Property | Signature | Description |
|---|---|---|
| `embed(texts)` | `List[str] → np.ndarray (N, D), float32` | L2-normalized embeddings |
| `embedding_dim` | `→ int` | Output dimensionality |

All implementations guarantee L2-normalized output (dot product == cosine similarity).

### `get_embedder(model_name, **kwargs)`

| Prefix | Backend | Example |
|---|---|---|
| `bedrock-embed/` | `BedrockEmbedder` | `get_embedder("bedrock-embed/cohere.embed-english-v3")` |
| *(anything else)* | `SentenceTransformerEmbedder` | `get_embedder("multi-qa-mpnet-base-cos-v1")` |

## Backends

### `SentenceTransformerEmbedder`

Wraps `sentence_transformers.SentenceTransformer`. Tries `local_files_only=True` first to avoid HF Hub auth issues with cached models.

| Param | Default | Description |
|---|---|---|
| `model_name` | `"multi-qa-mpnet-base-cos-v1"` | HuggingFace model name |
| `normalize` | `True` | L2-normalize output |

### `BedrockEmbedder`

Wraps AWS Bedrock embedding models. Auto-detects model family from `model_id` prefix.

| Family | Models | Batch | Server-side norm |
|---|---|---|---|
| Titan (`amazon.*`) | `amazon.titan-embed-text-v2:0` | No (1 text/call) | Yes |
| Cohere (`cohere.*`) | `cohere.embed-english-v3`, `cohere.embed-multilingual-v3` | Yes (up to 96) | No (client-side) |

| Param | Default | Description |
|---|---|---|
| `model_id` | `"amazon.titan-embed-text-v2:0"` | Bedrock model identifier |
| `region` | `None` | AWS region (uses default session if None) |
| `dimensions` | `1024` | Output dim (Titan only) |
| `input_type` | `"search_document"` | Cohere input type |

Dimensionality is probed at init via a single API call (`_probe_dim()`).

## Consumers

- `LocalMemoryBackend` — semantic dedup of memory units (see [docs/memory/LOCAL_MEMORY_BACKEND.md](../memory/LOCAL_MEMORY_BACKEND.md))
- `PDFClient` — document chunk indexing and similarity search

## File Layout

```
lits/embedding/
├── __init__.py              # get_embedder() + re-exports
├── base.py                  # BaseEmbedder ABC
├── sentence_transformer.py  # SentenceTransformerEmbedder
└── bedrock.py               # BedrockEmbedder
```
