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
        query_idx: Optional[int] = None,
    ) -> List[MemoryUnit]:
        """
        Add messages to the memory store.
        
        Args:
            trajectory: The trajectory key identifying the current position.
            messages: List of message dicts with 'role' and 'content' keys.
            metadata: Optional metadata to attach to stored memories.
            infer: Whether to use LLM to extract facts from messages.
            query_idx: Example index for InferenceLogger attribution.
            
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
        self._cache: Dict[str, List[MemoryUnit]] = {}
        self._dirty: set[str] = set()

    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
        query_idx: Optional[int] = None,
    ) -> List[MemoryUnit]:
        # Build qdrant payload metadata from TrajectoryKey
        mem0_metadata = dict(metadata or {})
        mem0_metadata.setdefault("trajectory_path", trajectory.path_str)
        mem0_metadata.setdefault("trajectory_depth", trajectory.depth)
        mem0_metadata.setdefault("ancestry_paths", list(trajectory.ancestry_paths))
        mem0_metadata.setdefault("memory_namespace", "lits_mem")

        self.memory.add(
            messages=list(messages),
            user_id=trajectory.search_id,
            metadata=mem0_metadata,
            infer=infer,
        )
        self._dirty.add(trajectory.search_id)
        return []

    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        if search_id not in self._cache or search_id in self._dirty:
            filter_obj = self._build_filter(search_id=search_id)
            points = self._scroll(filter_obj)
            self._cache[search_id] = [self._point_to_unit(point) for point in points]
            self._dirty.discard(search_id)
        return self._cache[search_id]

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


import json
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# -- Fact extraction prompt ------------------------------------------------

_FACT_EXTRACTION_PROMPT = """\
Extract atomic facts from the following text.
Return a JSON object with a "facts" key containing an array of strings.
Each fact should be a single, self-contained statement.
If no facts can be extracted, return {{"facts": []}}.

Text:
{text}"""


def _parse_facts_response(response_text: str) -> List[str]:
    """Parse LLM response into a list of fact strings.

    Tries JSON first, falls back to line-split with bullet stripping.
    """
    text = response_text.strip()
    # Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "facts" in data:
            return [f for f in data["facts"] if isinstance(f, str) and f.strip()]
        if isinstance(data, list):
            return [f for f in data if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "facts" in data:
                return [f for f in data["facts"] if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: line-split, strip bullets/numbers
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            lines.append(line)
    return lines


class LocalMemoryBackend(BaseMemoryBackend):
    """In-memory backend with LLM fact extraction and embedding-based dedup.

    Replaces ``Mem0MemoryBackend`` for lightweight, reproducible usage.
    No external services required — only ``sentence-transformers`` for
    embeddings and a lits LLM for fact extraction.

    Storage layout (per search_id):
        ``_units[search_id]``   — ``list[MemoryUnit]``, order-aligned with
        ``_vectors[search_id]`` — ``np.ndarray`` of shape ``(N, D)``.
        Row *i* of the matrix is the embedding for ``_units[search_id][i]``.

    Persistence (optional):
        ``save(dir_path)`` writes ``{search_id}.jsonl`` + ``{search_id}.npz``
        into *dir_path*.  ``load(dir_path)`` restores them.  The LLM and
        embedding model are NOT serialized — pass them again at construction.

    Args:
        llm: A lits LLM model (e.g. from ``get_lm()``). Used for fact
            extraction when ``add_messages(infer=True)`` is called.
        embedder: A ``BaseEmbedder`` instance (from ``lits.embedding``).
            If None, defaults to ``get_embedder()`` which creates a
            ``SentenceTransformerEmbedder("multi-qa-mpnet-base-cos-v1")``.
            Loaded eagerly at construction for fail-fast behaviour.
        dedup_threshold: Cosine similarity threshold (0–1). Facts above
            this threshold compared to an existing fact are considered
            duplicates (skip or alias).
        update_length_ratio: If a new fact is this many times longer than
            the matched existing fact, the existing fact's text and hash
            are replaced with the richer version. Default 1.3.
    """

    def __init__(
        self,
        llm,
        embedder=None,
        dedup_threshold: float = 0.85,
        update_length_ratio: float = 1.3,
    ):
        from lits.embedding import get_embedder

        self._llm = llm
        self._embedder = embedder if embedder is not None else get_embedder()
        self.dedup_threshold = dedup_threshold
        self.update_length_ratio = update_length_ratio
        # Aligned parallel structures per search_id
        self._units: Dict[str, List[MemoryUnit]] = {}
        self._vectors: Dict[str, np.ndarray] = {}  # (N, D) or empty

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, D) array."""
        return self._embedder.embed(texts)

    def _get_matrix(self, search_id: str) -> Optional[np.ndarray]:
        """Return the embedding matrix for *search_id*, or None if empty."""
        mat = self._vectors.get(search_id)
        if mat is not None and len(mat) > 0:
            return mat
        return None

    def _append(self, search_id: str, unit: MemoryUnit, vec: np.ndarray) -> None:
        """Append a unit + its embedding vector, keeping the two aligned."""
        self._units.setdefault(search_id, []).append(unit)
        mat = self._vectors.get(search_id)
        if mat is None or len(mat) == 0:
            self._vectors[search_id] = vec.reshape(1, -1)
        else:
            self._vectors[search_id] = np.vstack([mat, vec])

    def _update_unit(self, search_id: str, idx: int, text: str, vec: np.ndarray) -> None:
        """Replace a stored unit's text/hash/embedding in-place."""
        unit = self._units[search_id][idx]
        unit.text = text
        unit.content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        self._vectors[search_id][idx] = vec

    # ------------------------------------------------------------------
    # Core: _add_facts (private — called by add_messages)
    # ------------------------------------------------------------------

    def _add_facts(
        self,
        trajectory: TrajectoryKey,
        facts: List[str],
        metadata: Optional[Dict] = None,
    ) -> List[MemoryUnit]:
        """Embed facts, deduplicate against existing store, insert new ones.

        Dedup logic per fact:
        1. Cosine-compare against all existing embeddings for same search_id.
        2. If max sim >= threshold AND same trajectory/ancestor:
           a. If new fact is >update_length_ratio× longer → update existing
              unit's text and hash in-place (richer version).
           b. Else → skip (true duplicate).
        3. If max sim >= threshold BUT different trajectory → create alias
           unit copying matched unit's text/hash but with current trajectory's
           origin_path (ensures signature() overlap without mutating the
           other trajectory's unit).
        4. Else → insert as new MemoryUnit.

        Returns:
            List of newly inserted MemoryUnit objects (aliases and new facts).
            In-place updates are NOT included — the unit is already in the store.
        """
        if not facts:
            return []

        search_id = trajectory.search_id
        meta = metadata or {}
        origin_path = trajectory.path_str
        depth = trajectory.depth
        ancestry = trajectory.ancestry_paths

        # Embed all new facts at once
        new_embeddings = self._embed(facts)

        inserted: List[MemoryUnit] = []
        for i, fact_text in enumerate(facts):
            fact_text = fact_text.strip()
            if not fact_text:
                continue

            vec = new_embeddings[i]
            mat = self._get_matrix(search_id)
            units = self._units.get(search_id, [])

            if mat is not None:
                sims = mat @ vec  # cosine (already normalized)
                max_idx = int(np.argmax(sims))
                max_sim = float(sims[max_idx])

                if max_sim >= self.dedup_threshold:
                    matched = units[max_idx]
                    is_longer = len(fact_text) > len(matched.text) * self.update_length_ratio

                    # Same trajectory or ancestor
                    if matched.origin_path == origin_path or \
                       matched.inherited_by(origin_path):
                        if is_longer:
                            # Update in-place with richer text
                            self._update_unit(search_id, max_idx, fact_text, vec)
                            logger.debug(
                                f"LocalMemoryBackend: updated in-place "
                                f"(sim={max_sim:.3f}, len {len(matched.text)}→"
                                f"{len(fact_text)}): {fact_text[:80]}"
                            )
                        else:
                            logger.debug(
                                f"LocalMemoryBackend: dedup skip "
                                f"(sim={max_sim:.3f}): {fact_text[:80]}"
                            )
                        continue

                    # Different trajectory → alias (never mutate matched).
                    # Hash always copies matched to ensure signature() overlap.
                    # Text uses the richer version if available, so future
                    # trajectories dedup-matching this alias see more info.
                    alias_text = fact_text if is_longer else matched.text
                    alias = MemoryUnit(
                        id=str(uuid.uuid4()),
                        text=alias_text,
                        search_id=search_id,
                        origin_path=origin_path,
                        depth=int(depth),
                        ancestry_paths=tuple(ancestry),
                        metadata=dict(meta),
                        created_at=_dt.datetime.now().isoformat(),
                        content_hash=matched.content_hash,
                    )
                    self._append(search_id, alias, vec)
                    inserted.append(alias)
                    logger.debug(
                        f"LocalMemoryBackend: alias "
                        f"(sim={max_sim:.3f}): {fact_text[:80]}"
                    )
                    continue

            # No dedup match → new fact
            content_hash = hashlib.md5(fact_text.encode("utf-8")).hexdigest()
            unit = MemoryUnit(
                id=str(uuid.uuid4()),
                text=fact_text,
                search_id=search_id,
                origin_path=origin_path,
                depth=int(depth),
                ancestry_paths=tuple(ancestry),
                metadata=dict(meta),
                created_at=_dt.datetime.now().isoformat(),
                content_hash=content_hash,
            )
            self._append(search_id, unit, vec)
            inserted.append(unit)

        logger.debug(
            f"LocalMemoryBackend: inserted {len(inserted)}/{len(facts)} facts "
            f"for {trajectory.path_str}"
        )
        return inserted

    # ------------------------------------------------------------------
    # BaseMemoryBackend interface
    # ------------------------------------------------------------------

    def add_messages(
        self,
        trajectory: TrajectoryKey,
        messages: Sequence[Dict[str, str]],
        metadata: Optional[Dict] = None,
        infer: bool = True,
        query_idx: Optional[int] = None,
    ) -> List[MemoryUnit]:
        """Add messages to memory, optionally extracting facts via LLM.

        Args:
            trajectory: Current trajectory position.
            messages: Chat messages (role/content dicts).
            metadata: LiTS metadata. May contain ``from_phase`` (e.g. ``"expand"``).
            infer: If True, use LLM to extract atomic facts from messages.
                If False, store each message's content as a raw fact.
            query_idx: Example index for InferenceLogger attribution.

        Returns:
            List of inserted MemoryUnit objects.
        """
        if infer:
            text = "\n".join(
                m.get("content", "") for m in messages if m.get("content")
            )
            if not text.strip():
                return []

            # Construct role for InferenceLogger
            from lits.components.utils import create_role
            from_phase = (metadata or {}).get("from_phase", "")
            role = create_role("memory", query_idx, from_phase)

            prompt = _FACT_EXTRACTION_PROMPT.format(text=text)
            try:
                response = self._llm(prompt, role=role, temperature=0.0, max_new_tokens=500)
                response_text = response.text if hasattr(response, "text") else str(response)
                facts = _parse_facts_response(response_text)
            except Exception as e:
                logger.warning(f"LocalMemoryBackend: fact extraction failed: {e}")
                facts = [text]
        else:
            facts = [
                m.get("content", "") for m in messages if m.get("content", "").strip()
            ]

        return self._add_facts(trajectory, facts, metadata)

    def list_all_units(self, search_id: str) -> List[MemoryUnit]:
        """Return all stored units for a search_id."""
        return list(self._units.get(search_id, []))

    def list_trajectory_keys(self, search_id: str = None) -> set[str]:
        """Return distinct ``origin_path`` values across stored units.

        Args:
            search_id: If given, restrict to that search_id.
                If None, aggregate across all search_ids.

        Returns:
            Set of ``origin_path`` strings (e.g. ``{"q/0", "q/0/0", "q/1"}``).
        """
        if search_id is not None:
            return {u.origin_path for u in self._units.get(search_id, [])}
        return {u.origin_path for units in self._units.values() for u in units}

    # ------------------------------------------------------------------
    # Persistence: save / load
    # ------------------------------------------------------------------

    def save(self, dir_path: str) -> None:
        """Persist all search_id stores to *dir_path*.

        For each search_id writes:
            ``{search_id}.jsonl`` — one JSON object per MemoryUnit
            ``{search_id}.npz``  — embedding matrix

        Args:
            dir_path: Directory to write files into (created if needed).
        """
        from pathlib import Path
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)

        for search_id, units in self._units.items():
            # Units → jsonl
            jsonl_path = out / f"{search_id}.jsonl"
            with open(jsonl_path, "w") as f:
                for unit in units:
                    record = {
                        "id": unit.id,
                        "text": unit.text,
                        "search_id": unit.search_id,
                        "origin_path": unit.origin_path,
                        "depth": unit.depth,
                        "ancestry_paths": list(unit.ancestry_paths),
                        "metadata": unit.metadata,
                        "created_at": unit.created_at,
                        "content_hash": unit.content_hash,
                    }
                    f.write(json.dumps(record) + "\n")

            # Embeddings → npz
            mat = self._vectors.get(search_id)
            if mat is not None and len(mat) > 0:
                np.savez_compressed(out / f"{search_id}.npz", embeddings=mat)

        logger.info(f"LocalMemoryBackend: saved {len(self._units)} search(es) to {dir_path}")

    def load(self, dir_path: str) -> None:
        """Load persisted stores from *dir_path*, merging into current state.

        Args:
            dir_path: Directory containing ``{search_id}.jsonl`` and
                ``{search_id}.npz`` files.
        """
        from pathlib import Path
        src = Path(dir_path)
        if not src.is_dir():
            logger.warning(f"LocalMemoryBackend.load: {dir_path} not found")
            return

        for jsonl_file in src.glob("*.jsonl"):
            search_id = jsonl_file.stem
            units: List[MemoryUnit] = []
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    units.append(MemoryUnit(
                        id=rec["id"],
                        text=rec["text"],
                        search_id=rec["search_id"],
                        origin_path=rec["origin_path"],
                        depth=rec["depth"],
                        ancestry_paths=tuple(rec.get("ancestry_paths", ())),
                        metadata=rec.get("metadata", {}),
                        created_at=rec.get("created_at"),
                        content_hash=rec.get("content_hash"),
                    ))

            npz_file = src / f"{search_id}.npz"
            mat = None
            if npz_file.exists():
                data = np.load(npz_file)
                mat = data["embeddings"]

            self._units[search_id] = units
            if mat is not None:
                self._vectors[search_id] = mat

        logger.info(
            f"LocalMemoryBackend: loaded {len(self._units)} search(es) from {dir_path}"
        )
