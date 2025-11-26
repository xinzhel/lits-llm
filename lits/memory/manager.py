from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .backends import BaseMemoryBackend
from .config import LiTSMemoryConfig
from .retrieval import TrajectorySearchEngine
from .types import MemoryUnit, TrajectoryKey, TrajectorySimilarity


@dataclass
class AugmentedContext:
    """
    Aggregated memory bundle returned by :class:`LiTSMemoryManager`.

    The bundle is intentionally lightweight so callers across :mod:`lits.agents`,
    :mod:`lits.components`, and :mod:`lits.framework_config` can pass it around without
    worrying about backend details.  Policies typically call :meth:`to_prompt_blocks`
    when constructing LLM prompts for node expansion.
    """

    trajectory: TrajectoryKey
    inherited_units: Tuple[MemoryUnit, ...]
    retrieved_trajectories: Tuple[TrajectorySimilarity, ...]

    def selected_facts(self) -> List[str]:
        """
        Flatten the augmentation into a list of textual snippets.
        """

        snippets: List[str] = []
        for result in self.retrieved_trajectories:
            snippets.extend(unit.text for unit in result.missing_units)
        return snippets

    def to_prompt_blocks(self, include_inherited: bool = True) -> str:
        """
        Format the augmented context as a single string suitable for concatenation with
        policy prompts.  :mod:`lits.components.policy` can call this helper to assemble
        the ``<memory>`` section before invoking the LLM.
        """

        blocks: List[str] = []
        if include_inherited and self.inherited_units:
            inherited_text = "\n".join(f"- {unit.text}" for unit in self.inherited_units)
            blocks.append(f"# Inherited memories\n{inherited_text}")

        for result in self.retrieved_trajectories:
            blocks.append(result.to_prompt_section())
        return "\n\n".join(blocks).strip()


class LiTSMemoryManager:
    """
    High-level orchestrator that implements the LiTS-Mem workflow.

    The manager is attached to the search runner and invoked at three points:

    * ``record_action`` is called whenever the policy emits an action/response.
      This triggers mem0 to extract candidate facts and update the shared memory DB.
    * ``list_inherited_units`` (or ``build_augmented_context``) runs at the start of
      node expansion to recover cross-trajectory context.
    * ``search_related_trajectories`` can be invoked by evaluators (e.g., PRM/BN
      modules) to inspect provenance.

    Only this module mutates files in :mod:`lits.memory`; other subpackages interact
    exclusively through the public methods documented here.
    """

    def __init__(self, backend: BaseMemoryBackend, config: Optional[LiTSMemoryConfig] = None):
        self.backend = backend
        self.config = config or LiTSMemoryConfig()
        self.retriever = TrajectorySearchEngine(self.config)
        self._cache: Dict[str, List[MemoryUnit]] = {}
        self._cache_dirty: set[str] = set()

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------
    def record_action(
        self,
        trajectory: TrajectoryKey,
        *,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        facts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict] = None,
        infer: bool = True,
    ) -> None:
        """
        Store new information produced along ``trajectory``.  At least one of
        ``messages`` or ``facts`` must be supplied.  When ``messages`` are passed and
        ``infer`` is ``True``, the backend (typically mem0) extracts atomic facts via an
        LLM before inserting them into the vector store.
        

        Use with other lits subpackages:
        :mod:`lits.agents.tree_search.mcts` should call this method immediately after
        ``policy.expand`` so the generated action is available for subsequent nodes.
        """

        metadata = self.config.metadata_for(trajectory, metadata)

        inserted: List[MemoryUnit] = []
        if facts:
            inserted = self.backend.add_facts(trajectory, facts, metadata)
        elif messages:
            inserted = self.backend.add_messages(trajectory, messages, metadata, infer=infer)
        else:
            raise ValueError("Either `messages` or `facts` must be provided.")

        if inserted:
            cache = self._cache.setdefault(trajectory.search_id, [])
            cache.extend(inserted)
        else:
            self._cache_dirty.add(trajectory.search_id)

    # -------------------------------------------------------------------------
    # Retrieval helpers
    # -------------------------------------------------------------------------
    def list_inherited_units(self, trajectory: TrajectoryKey) -> List[MemoryUnit]:
        """
        Return the inherited memory set \(\mathsf{Mem}(t)\) for trajectory \(t\).

        In the LiTS-Mem paper, \(\mathsf{Mem}(t)\) is the set of memory units whose
        prefix paths are ancestors of \(t\) (see Eq. (1) and the definition following
        \( \mathcal{T}(m) \)). This method:
        1) fetches all memories for the search_id,
        2) keeps only those whose origin path is a prefix of ``trajectory``,
        3) sorts them by depth/created_at for deterministic consumption.

        This is the first step of “Memory Retrieval and Use” (Section 3.1): compute
        the inherited set before performing cross-trajectory search and selection.
        Results are cached per search_id and invalidated whenever ``record_action``
        inserts new memories.
        """

        units = self._ensure_cache(trajectory.search_id)
        inherited = [
            unit for unit in units if unit.inherited_by(trajectory.path_str)
        ]
        inherited.sort(key=lambda unit: (unit.depth, unit.created_at or ""))
        return inherited

    def search_related_trajectories(self, trajectory: TrajectoryKey) -> List[TrajectorySimilarity]:
        """
        Run the trajectory search stage starting from ``trajectory``.
        """

        units = self._ensure_cache(trajectory.search_id)
        inherited = [
            unit for unit in units if unit.inherited_by(trajectory.path_str)
        ]
        return self.retriever.search(trajectory, inherited, units)

    def build_augmented_context(self, trajectory: TrajectoryKey) -> AugmentedContext:
        """
        Convenience wrapper returning an :class:`AugmentedContext` object that merges
        inherited memories with cross-trajectory augmentations.  Policies can call this
        method and directly feed :meth:`AugmentedContext.to_prompt_blocks` into their
        system prompts.
        """

        inherited = tuple(self.list_inherited_units(trajectory))
        search_results = tuple(self.search_related_trajectories(trajectory))

        limited_results: List[TrajectorySimilarity] = []
        total_selected = 0
        for result in search_results:
            if total_selected >= self.config.max_augmented_memories:
                break
            limited_results.append(result)
            total_selected += len(result.missing_units)

        return AugmentedContext(
            trajectory=trajectory,
            inherited_units=inherited,
            retrieved_trajectories=tuple(limited_results),
        )

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------
    def _ensure_cache(self, search_id: str) -> List[MemoryUnit]:
        """
        Return the cached memory list for ``search_id``, refreshing it from the backend
        when necessary.

        The manager keeps a per-search cache so repeated calls to
        :meth:`list_inherited_units` or :meth:`search_related_trajectories` do not hit
        mem0/Qdrant for every node expansion.  When the cache entry is missing or marked
        dirty (e.g., because ``record_action`` added new memories without concrete IDs),
        this helper pulls the latest units from ``backend.list_all_units`` and clears the
        dirty flag before returning the list.
        """
        if search_id not in self._cache or search_id in self._cache_dirty:
            self._cache[search_id] = self.backend.list_all_units(search_id)
            self._cache_dirty.discard(search_id)
        return self._cache[search_id]
