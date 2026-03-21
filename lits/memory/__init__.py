"""
Foundational LiTS-Mem primitives.

This package hosts the reference implementation for cross-trajectory memory used by
LiTS tree-search agents.  The high-level entry point is :class:`LiTSMemoryManager`,
which orchestrates backend-agnostic storage, trajectory-level retrieval, and policy
context augmentation as described in the LiTS-Mem section of the documentation.

Python API usage::

    from lits.memory import LiTSMemoryManager, LiTSMemoryConfig, LocalMemoryBackend
    from lits.lm import get_lm

    llm = get_lm("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")
    backend = LocalMemoryBackend(llm=llm)
    manager = LiTSMemoryManager(backend=backend, config=LiTSMemoryConfig())

    traj = TrajectoryKey(search_id=run_id, indices=(0, 1))
    manager.record_action(trajectory=traj, messages=[...], infer=True)
    context = manager.build_augmented_context(traj)
    prompt_block = context.to_prompt_blocks()

CLI usage (``--memory-arg`` implicitly enables memory)::

    # LocalMemoryBackend (default)
    lits-search --dataset dbbench \\
        --memory-arg backend=local \\
            model=bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0 \\
            embedding_model=bedrock-embed/cohere.embed-english-v3

    # Mem0MemoryBackend (legacy)
    lits-search --dataset dbbench \\
        --memory-arg backend=mem0 mem0_config_path=./mem0.json

Only the memory subpackage is modified by this change; other subpackages can call
into the public classes exposed here without needing internal knowledge.
"""

from .config import LiTSMemoryConfig
from .types import MemoryUnit, TrajectoryKey, TrajectorySimilarity
from .backends import BaseMemoryBackend, Mem0MemoryBackend, LocalMemoryBackend
from .manager import LiTSMemoryManager, AugmentedContext

__all__ = [
    "LiTSMemoryConfig",
    "MemoryUnit",
    "TrajectoryKey",
    "TrajectorySimilarity",
    "BaseMemoryBackend",
    "Mem0MemoryBackend",
    "LocalMemoryBackend",
    "LiTSMemoryManager",
    "AugmentedContext",
]
