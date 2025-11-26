from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .types import TrajectoryKey


@dataclass
class LiTSMemoryConfig:
    """
    Configuration for LiTS-Mem runtime behaviour.

    The config intentionally mirrors knobs discussed in the LiTS-Mem paper:

    * ``similarity_threshold`` gates cross-trajectory retrieval.  Higher values make the
      retriever more conservative.
    * ``max_retrieved_trajectories`` limits how many peer trajectories are surfaced for
      context augmentation per node expansion.
    * ``cardinality_ratio`` controls the cardinality-based trimming described in the
      specification.  Candidate memory sets larger than
      ``len(norm(Mem(t))) * cardinality_ratio`` are truncated before similarity is scored.
    * ``scroll_batch_size`` is forwarded to the mem0/Qdrant backend so tree-search
      policies do not need to tune vector DB fetch parameters themselves.
    * ``metadata_namespace`` provides a predictable prefix for metadata injected into
      mem0.  The namespace is useful for other LiTS components (e.g.,
      :mod:`lits.components.policy`) to locate augmented memory inside the payload.
    * ``max_augmented_memories`` keeps the generated policy context concise.

    Agents running inside :mod:`lits.agents.tree_search` instantiate
    :class:`LiTSMemoryManager` with this config and keep the returned object on the
    search context.  No additional wiring is required in other subpackages – the manager
    exposes high-level helpers for trajectory expansion, evaluation, and rollouts.
    """

    similarity_threshold: float = 0.35
    max_retrieved_trajectories: int = 3
    cardinality_ratio: float = 1.5
    scroll_batch_size: int = 256
    metadata_namespace: str = "lits_mem"
    max_augmented_memories: int = 16
    attach_depth_cap: Optional[int] = None
    include_metadata_snapshot: bool = True
    _metadata_defaults: Dict[str, str] = field(
        default_factory=lambda: {
            "trajectory_path": "trajectory_path",
            "trajectory_depth": "trajectory_depth",
            "ancestry_paths": "ancestry_paths",
            "memory_namespace": "memory_namespace",
        }
    )

    def metadata_for(self, trajectory: TrajectoryKey, extra: Optional[Dict] = None) -> Dict:
        """
        Construct metadata for storing facts along ``trajectory``.

        The resulting dictionary can be passed directly into :meth:`mem0.Memory.add`
        as ``metadata`` so that downstream modules (policies, evaluators, tool chains)
        can recover trajectory position from stored memories.

        Args:
            trajectory: Target trajectory descriptor.
            extra: Optional metadata (e.g., reward estimates) supplied by the caller.
                   These keys override LiTS defaults if duplicates exist.

        Returns:
            Dict enriched with ``trajectory_path``, ``trajectory_depth``,
            ``ancestry_paths`` and ``memory_namespace`` fields.

        Example usage:
            Within :mod:`lits.agents.tree_search.mcts`, call ::

                metadata = mem_config.metadata_for(node.trajectory, {"stage": "expand"})
                mem_backend.add_messages(node.trajectory, rollout_messages, metadata)
        
        Example output:
            {
              'trajectory_path': 'q/0', 
              'trajectory_depth': 1, 
              'ancestry_paths': ['q', 'q/0'], 
              'memory_namespace': 'lits_mem
            }
        """

        metadata = dict(extra or {})
        metadata.setdefault(self._metadata_defaults["trajectory_path"], trajectory.path_str)
        metadata.setdefault(self._metadata_defaults["trajectory_depth"], trajectory.depth)
        metadata.setdefault(self._metadata_defaults["ancestry_paths"], list(trajectory.ancestry_paths))
        metadata.setdefault(self._metadata_defaults["memory_namespace"], self.metadata_namespace)
        return metadata

    def target_cardinality(self, reference_size: int) -> int:
        """
        Compute the candidate size threshold used by the cardinality-based trimming
        stage.  The reference size equals ``len(norm(Mem(t)))`` for the current
        trajectory ``t``.
        """

        if reference_size <= 0:
            return 0
        limit = int(round(reference_size * self.cardinality_ratio))
        return max(1, limit)
