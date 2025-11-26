from __future__ import annotations

import argparse
import os
import sys



# python -m unit_test.probe_qdrant_collection \
#   --path qdrant_local \
#   --name lits_mem0_504b2b \
#   --limit 50
def probe(collection_path: str, collection_name: str, limit: int = 100):
    """
    List all payloads in a local Qdrant collection created by mem0.
    Does not require user_id/agent_id/run_id filters; scrolls the raw payloads.
    """

    if not os.path.isdir(collection_path):
        raise FileNotFoundError(f"Collection path does not exist: {collection_path}")

    from mem0.vector_stores.qdrant import Qdrant

    store = Qdrant(
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

    batch, next_page = store.client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    # print(f"No points found in collection '{collection_name}' at path '{collection_dir}'.")
    if not batch:
        print(f"No points found in collection '{collection_name}' at path '{collection_path}'.")
        return

    for point in batch:
        payload = point.payload or {}
        print(f"id: {point.id}")
        print(f"memory: {payload.get('data')}")
        meta = {k: v for k, v in payload.items() if k != "data"}
        if meta:
            print(f"metadata: {meta}")
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a local Qdrant collection created by mem0.")
    parser.add_argument("--path", required=True, help="Path to the qdrant_local directory (e.g., lits_llm/qdrant_local)")
    parser.add_argument("--name", required=True, help="Collection name (e.g., lits_mem0_3404be)")
    parser.add_argument("--limit", type=int, default=100, help="Max items to list")
    args = parser.parse_args()
    probe(args.path, args.name, args.limit)
