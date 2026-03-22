"""Augmentor setup and callback wiring for tree search.

Provides :func:`setup_augmentors` which connects a list of
:class:`~lits.components.context_augmentor.ContextAugmentor` instances
to the tree-search loop via two callback closures:

Callback architecture::

    search loop
    ├── _expand() / _simulate()
    │   └── on_step_complete(step, node, query_idx)   ← per-step augmentors
    │       • CriticAugmentor, SQLValidator, FactMemoryAugmentor
    │       • Extracts trajectory state from node ancestry
    │       • Calls aug.analyze(traj_state, ...) for each
    │
    └── search() after backpropagation
        └── on_trajectory_complete(path, reward, query_idx)  ← per-trajectory augmentors
            • ReflectionAugmentor, SQLErrorProfiler
            • Extracts trajectory state from MCTS path
            • Calls aug.analyze(traj_state, ...) with reward

Prompt injection is handled separately: a combined ``retrieve()`` function
is registered to ``policy.set_dynamic_notes_fn()`` so that all augmentors'
context is injected into the system prompt before each action generation.
"""

import logging
from typing import List, Tuple, Callable, Optional

from ...log import log_event
from ...components.context_augmentor import (
    ContextAugmentor,
    CriticAugmentor,
    ReflectionAugmentor,
    FactMemoryAugmentor,
    SQLValidator,
    SQLErrorProfiler,
)

logger = logging.getLogger(__name__)

# Augmentor classes triggered after each step (transition complete)
_PER_STEP_TYPES = (CriticAugmentor, SQLValidator, FactMemoryAugmentor)

# Augmentor classes triggered after a full trajectory completes
_PER_TRAJECTORY_TYPES = (ReflectionAugmentor, SQLErrorProfiler)



def setup_augmentors(
    policy,
    augmentors: List[ContextAugmentor],
    query_context: dict,
) -> Tuple[Callable, Callable]:
    """Wire augmentors into the tree-search loop.

    Sets storage context on each augmentor, registers a combined
    ``retrieve()`` to the policy's dynamic notes mechanism, and builds
    two callback closures for the search loop.

    Args:
        policy: Policy instance (has ``set_dynamic_notes_fn``).
        augmentors: List of ContextAugmentor instances to wire up.
        query_context: Dict with keys:
            - ``policy_model_name`` (str): Model name for storage paths.
            - ``task_type`` (str): Task type for storage paths.
            - ``query_or_goals`` (str): The problem/question being solved.

    Returns:
        Tuple of ``(on_step_complete, on_trajectory_complete)`` callbacks.
    """
    if not augmentors:
        logger.debug("setup_augmentors: no augmentors provided, returning no-op callbacks")
        return _noop_step, _noop_trajectory

    policy_model_name = query_context.get("policy_model_name", "")
    task_type = query_context.get("task_type", "")

    # 1. Set storage context on each augmentor
    for aug in augmentors:
        aug.set_storage_context(policy_model_name, task_type)

    # 2. Register combined retrieve to policy dynamic notes
    def _combined_retrieve() -> List[str]:
        notes = []
        traj_key = query_context.get("trajectory_key", "<unset>")
        for aug in augmentors:
            try:
                result = aug.retrieve(query_context)
                if result:
                    notes.append(result)
                    log_event(
                        logger, "Memory",
                        f"_combined_retrieve: {aug.__class__.__name__} "
                        f"returned {len(result)} chars for traj_key={traj_key}",
                    )
                else:
                    log_event(
                        logger, "Memory",
                        f"_combined_retrieve: {aug.__class__.__name__} "
                        f"returned empty for traj_key={traj_key}",
                    )
            except Exception as e:
                logger.warning(
                    f"retrieve() failed for {aug.__class__.__name__}: {e}"
                )
        return notes

    policy.set_dynamic_notes_fn(_combined_retrieve)

    # 3. Classify augmentors by trigger type
    step_augmentors = [a for a in augmentors if isinstance(a, _PER_STEP_TYPES)]
    traj_augmentors = [a for a in augmentors if isinstance(a, _PER_TRAJECTORY_TYPES)]

    logger.info(
        f"setup_augmentors: {len(step_augmentors)} per-step, "
        f"{len(traj_augmentors)} per-trajectory augmentors"
    )

    # 4. Build on_step_complete closure
    def on_step_complete(step, node, query_idx, **kwargs):
        """Called after each child node transition in _expand().

        Extracts trajectory state from node ancestry and invokes
        per-step augmentors.
        """
        traj_state = _extract_trajectory_state_up_to(node)
        query_or_goals = query_context.get("query_or_goals", "")
        traj_key_str = (
            node.trajectory_key.path_str if node.trajectory_key else None
        )

        for aug in step_augmentors:
            try:
                aug.analyze(
                    traj_state,
                    query_idx=query_idx,
                    query_or_goals=query_or_goals,
                    trajectory_key=traj_key_str,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"on_step_complete: {aug.__class__.__name__}.analyze() "
                    f"failed: {e}"
                )

    # 5. Build on_trajectory_complete closure
    def on_trajectory_complete(path, reward, query_idx, **kwargs):
        """Called after backpropagation in search().

        Extracts trajectory state from the MCTS path and invokes
        per-trajectory augmentors.
        """
        traj_state = _extract_trajectory_state(path)
        query_or_goals = query_context.get("query_or_goals", "")
        traj_key_str = (
            path[-1].trajectory_key.path_str
            if path and path[-1].trajectory_key
            else None
        )

        for aug in traj_augmentors:
            try:
                aug.analyze(
                    traj_state,
                    query_idx=query_idx,
                    query_or_goals=query_or_goals,
                    trajectory_key=traj_key_str,
                    reward=reward,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"on_trajectory_complete: {aug.__class__.__name__}.analyze() "
                    f"failed: {e}"
                )

    return on_step_complete, on_trajectory_complete


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_trajectory_state_up_to(node) -> list:
    """Walk the parent chain from *node* to root, collecting steps.

    Args:
        node: An MCTSNode (or SearchNode) with ``.parent`` and ``.step``.

    Returns:
        List of steps in root-to-node order.
    """
    steps = []
    current = node
    while current is not None:
        if hasattr(current, "step") and current.step is not None:
            steps.append(current.step)
        current = current.parent
    steps.reverse()
    return steps


def _extract_trajectory_state(path) -> list:
    """Extract steps from an MCTS path (root-to-leaf node list).

    Args:
        path: List of MCTSNode from root to leaf.

    Returns:
        List of steps (nodes with non-None step attribute).
    """
    steps = []
    for node in path:
        if hasattr(node, "step") and node.step is not None:
            steps.append(node.step)
    return steps


def _noop_step(step, node, query_idx, **kwargs):
    """No-op step callback when no augmentors are configured."""
    pass


def _noop_trajectory(path, reward, query_idx, **kwargs):
    """No-op trajectory callback when no augmentors are configured."""
    pass
