from __future__ import annotations

import datetime as _dt
import hashlib
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

try:
    from qdrant_client.models import FieldCondition, Filter, MatchValue
except Exception:  # pragma: no cover - qdrant optional for tests
    FieldCondition = Filter = MatchValue = None  # type: ignore

from .types import MemoryUnit, TrajectoryKey, ancestry_from_indices, decode_path, encode_path


class BaseMemoryBackend(ABC):
    """
    Abstract base class for memory backends.
    
    Memory backends are responsible for storing and retrieving memory units
    from a vector store. Implementations must provide methods for adding
    messages/facts and listing all units for a given search.
    """

    @abstractmethod
    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
    ) -> List[MemoryUnit]:
        """
        Add messages to the memory store.
        
        Args:
            trajectory: The trajectory key identifying the current position.
            messages: List of message dicts with 'role' and 'content' keys.
            metadata: Optional metadata to attach to stored memories.
            infer: Whether to use LLM to extract facts from messages.
            
        Returns:
            List of MemoryUnit objects that were stored.
        """
        pass

    @abstractmethod
    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        """
        List all memory units for a given search.
        
        Args:
            search_id: The search instance identifier.
            
        Returns:
            List of all MemoryUnit objects for the search.
        """
        pass


class Mem0MemoryBackend(BaseMemoryBackend):
    """

    The backend is intentionally thin: storage, deduplication, and vector operations
    are delegated to mem0.  LiTS-specific metadata (trajectory path, ancestry, etc.) is
    expected to be included in ``metadata`` by :class:`LiTSMemoryManager`.
    """

    def __init__(self, memory, scroll_batch_size: int = 256):
        self.memory = memory
        self.vector_store = getattr(memory, "vector_store", None)
        if self.vector_store is None:
            raise ValueError("mem0 Memory instance must expose `vector_store`.")
        
        self.scroll_batch_size = scroll_batch_size

    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
    ) -> List[MemoryUnit]:
        self.memory.add(
            messages=list(messages),
            user_id=trajectory.search_id,
            metadata=dict(metadata or {}),
            infer=infer,
        )
        return []

    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        filter_obj = self._build_filter(search_id=search_id)
        points = self._scroll(filter_obj)
        return [self._point_to_unit(point) for point in points]

    def _build_filter(self, search_id: str) -> Optional[Filter]:
        assert Filter is not None, "qdrant-client is required for Mem0MemoryBackend."
        must = []
        if search_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=search_id)))
        return Filter(must=must) if must else None

    def _scroll(self, filter_obj) -> List:
        points: List = []
        client = getattr(self.vector_store, "client", None)
        if client is None:
            return points
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=filter_obj,
                limit=self.scroll_batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
        return points

    def _point_to_unit(self, point) -> MemoryUnit:
        payload = getattr(point, "payload", {}) or {}
        text = payload.get("data", "")
        search_id = payload.get("user_id", "")
        origin_path = payload.get("trajectory_path") or payload.get("origin_path") or encode_path(())
        depth = payload.get("trajectory_depth") or len(decode_path(origin_path))
        ancestry_paths = payload.get("ancestry_paths") or ancestry_from_indices(decode_path(origin_path))
        created_at = payload.get("created_at")
        content_hash = payload.get("hash")
        return MemoryUnit(
            id=str(getattr(point, "id", "")),
            text=text,
            search_id=search_id,
            origin_path=origin_path,
            depth=int(depth),
            ancestry_paths=tuple(ancestry_paths),
            metadata=dict(payload),
            created_at=created_at,
            content_hash=content_hash,
        )

