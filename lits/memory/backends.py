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
    Abstract base class that describes the operations LiTS-Mem expects from a memory
    backend.  Tree-search components never interact with a backend directly; instead
    they call :class:`~lits.memory.manager.LiTSMemoryManager`, which in turn delegates
    to a backend instance injected at construction time.
    """

    def __init__(self, scroll_batch_size: int = 256):
        self.scroll_batch_size = scroll_batch_size

    @abstractmethod
    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
    ) -> List[MemoryUnit]:
        """
        Store a conversation (whether raw or action snippets) along ``trajectory``.
        Implementations may call into mem0 to run LLM-based fact extraction.
        """
        pass
        
    def add_facts(
        self,
        trajectory: TrajectoryKey,
        facts: Sequence[str],
        metadata: Optional[Dict] = None,
    ) -> List[MemoryUnit]:
        """Insert already-extracted facts for ``trajectory``."""
        if not facts:
            return []
        metadata = dict(metadata or {})
        origin_path = metadata.get("trajectory_path", trajectory.path_str)
        depth = metadata.get("trajectory_depth", trajectory.depth)
        ancestry_paths = tuple(metadata.get("ancestry_paths", trajectory.ancestry_paths))
        units: List[MemoryUnit] = []
        now = _dt.datetime.utcnow().isoformat()
        for fact in facts:
            mem_id = str(uuid.uuid4())
            payload = dict(metadata)
            payload.setdefault("data", fact)
            payload.setdefault("created_at", now)
            payload.setdefault("hash", hashlib.md5(fact.encode("utf-8")).hexdigest())
            unit = MemoryUnit(
                id=mem_id,
                text=fact,
                search_id=trajectory.search_id,
                origin_path=origin_path,
                depth=depth,
                ancestry_paths=ancestry_paths,
                metadata=payload,
                created_at=now,
                content_hash=payload.get("hash"),
            )
            units.append(unit)
        self._store.setdefault(trajectory.search_id, []).extend(units)
        return units

    @abstractmethod
    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        """Return every memory unit scoped to ``search_id``."""


class Mem0MemoryBackend(BaseMemoryBackend):
    """
    Adapter that exposes mem0's :class:`mem0.Memory` API through
    :class:`BaseMemoryBackend`.

    The backend is intentionally thin: storage, deduplication, and vector operations
    are delegated to mem0.  LiTS-specific metadata (trajectory path, ancestry, etc.) is
    expected to be included in ``metadata`` by :class:`LiTSMemoryManager`.
    """

    def __init__(self, memory, scroll_batch_size: int = 256):
        super().__init__(scroll_batch_size=scroll_batch_size)
        self.memory = memory
        self.vector_store = getattr(memory, "vector_store", None)
        if self.vector_store is None:
            raise ValueError("mem0 Memory instance must expose `vector_store`.")

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


class LocalMemoryBackend(BaseMemoryBackend):
    """
    Lightweight backend useful for unit tests and interactive notebooks.  It keeps all
    memories in Python dictionaries so the LiTS-Mem stack can be exercised without
    mem0/Qdrant dependencies.  The backend implements the same interface as
    :class:`Mem0MemoryBackend`, enabling drop-in replacement.
    """

    def __init__(self):
        super().__init__(scroll_batch_size=0)
        self._store: Dict[str, List[MemoryUnit]] = {}

    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
    ) -> List[MemoryUnit]:
        if infer:
            raise ValueError("LocalMemoryBackend cannot perform LLM-based inference.")
        facts = [msg["content"] for msg in messages]
        return self.add_facts(trajectory, facts, metadata=metadata)

    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        return list(self._store.get(search_id, []))
