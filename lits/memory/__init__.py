"""
Foundational LiTS-Mem primitives.

This package hosts the reference implementation for cross-trajectory memory used by
LiTS tree-search agents.  The high-level entry point is :class:`LiTSMemoryManager`,
which orchestrates mem0-backed storage, trajectory-level retrieval, and policy
context augmentation as described in the LiTS-Mem section of the documentation.

Typical usage inside :mod:`lits.agents.tree_search` looks like::

    from lits.memory import LiTSMemoryManager, LiTSMemoryConfig, Mem0MemoryBackend
    from mem0 import Memory as Mem0

    mem_backend = Mem0MemoryBackend(Mem0(config=...))
    mem_manager = LiTSMemoryManager(backend=mem_backend, config=LiTSMemoryConfig())

    # During expansion:
    traj = TrajectoryKey(search_id=run_id, indices=(0, 1))
    mem_manager.record_action(
        trajectory=traj,
        messages=new_rollout_messages,
        metadata={"stage": "expand"}
    )
    context = mem_manager.build_augmented_context(traj)
    policy_prompt = tool_use_prompt + context.to_formatted_prompt()

Only the memory subpackage is modified by this change; other subpackages can call
into the public classes exposed here without needing internal knowledge.
"""

from .config import LiTSMemoryConfig
from .types import MemoryUnit, TrajectoryKey, TrajectorySimilarity
from .backends import BaseMemoryBackend, Mem0MemoryBackend
from .manager import LiTSMemoryManager, AugmentedContext

__all__ = [
    "LiTSMemoryConfig",
    "MemoryUnit",
    "TrajectoryKey",
    "TrajectorySimilarity",
    "BaseMemoryBackend",
    "Mem0MemoryBackend",
    "LiTSMemoryManager",
    "AugmentedContext",
]
