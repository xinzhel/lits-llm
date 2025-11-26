from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.memory.main import Memory


def probe(collection_path: str, collection_name: str, user_id: str, limit: int = 100):
    """
    List memories using mem0's get_all with a provided user_id filter.
    This mirrors how LiTS-Mem would query inherited/cross-trajectory context.
    """
    cfg = MemoryConfig()
    cfg.vector_store.provider = "qdrant"
    cfg.vector_store.config = QdrantConfig(
        collection_name=collection_name,
        embedding_model_dims=768,
        client=None,
        host=None,
        port=None,
        path=collection_path,
        url=None,
        api_key=None,
        on_disk=True,
    )
    cfg.embedder.provider = "huggingface"
    cfg.embedder.config = {"model": "sentence-transformers/multi-qa-mpnet-base-cos-v1"}
    cfg.llm.provider = "openai"
    cfg.llm.config = {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY", "stub")}

    mem = Memory(config=cfg)
    result = mem.get_all(user_id=user_id, limit=limit)
    items = result["results"] if isinstance(result, dict) and "results" in result else result
    for item in items:
        print(f"id: {item.get('id')}")
        print(f"memory: {item.get('memory')}")
        payload_keys = {k: v for k, v in item.items() if k not in {"id", "memory"}}
        if payload_keys:
            print(f"metadata: {payload_keys}")
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a local Qdrant collection via mem0.get_all using user_id.")
    parser.add_argument("--path", required=True, help="Path to the qdrant_local directory (e.g., lits_llm/qdrant_local)")
    parser.add_argument("--name", required=True, help="Collection name (e.g., lits_mem0_3404be)")
    parser.add_argument("--user", required=True, help="user_id filter to pass to mem0.get_all")
    parser.add_argument("--limit", type=int, default=100, help="Max items to list")
    args = parser.parse_args()
    probe(args.path, args.name, args.user, args.limit)
