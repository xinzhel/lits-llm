"""FactMemoryAugmentor: LiTS-Mem fact extraction and cross-trajectory retrieval.

Wraps :class:`~lits.memory.manager.LiTSMemoryManager` as a ContextAugmentor
subclass, bridging the memory subsystem into the unified augmentor pipeline.

Unlike other augmentors (Critic, Reflection, SQLValidator), FactMemoryAugmentor
does NOT call an LLM directly — the memory backend handles fact extraction
(either via mem0 or LocalMemoryBackend's LLM extraction).  Therefore:

    - ``base_model=None`` (no LLM needed at this layer; the backend owns the LLM)
    - ``analyze()`` is overridden directly (not ``_analyze()``) because the
      flow is fundamentally different: extract messages from the last step,
      pass them to ``memory_manager.record_action()``, and return a summary
      ContextUnit.
    - ``store()`` is a no-op — recording happens inside ``analyze()`` via
      ``memory_manager.record_action()``.
    - ``retrieve()`` delegates to ``memory_manager.build_augmented_context()``
      and returns ``AugmentedContext.to_prompt_blocks()``.
    - ``_buffer`` is NOT used — facts are persisted by the backend during
      ``analyze()``, so buffering would be redundant logging noise.

Usage::

    # Programmatic construction (e.g., in tests or custom scripts)
    from lits.memory.manager import LiTSMemoryManager
    from lits.memory.backends import LocalMemoryBackend  # or Mem0MemoryBackend

    backend = LocalMemoryBackend(llm=my_llm)
    manager = LiTSMemoryManager(backend)
    fact_aug = FactMemoryAugmentor(memory_manager=manager)

    # In MCTS on_step_complete callback:
    unit = fact_aug.analyze(traj_state, trajectory_key=traj_key, query_idx=0)

    # In policy prompt construction:
    prompt_block = fact_aug.retrieve(query_context={"trajectory_key": traj_key})

CLI usage (via ``--memory-arg``, see ``docs/cli/search.md``)::

    # Local backend (default) — in-process embeddings + LLM fact extraction
    lits-search --dataset math500 --search_framework rest \\
        --memory-arg backend=local model=bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0

    # Local with Bedrock Cohere embedder
    lits-search --dataset math500 --search_framework rest \\
        --memory-arg backend=local \\
            model=bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \\
            embedding_model=bedrock-embed/cohere.embed-english-v3

    # Mem0 backend — delegates to mem0 library
    lits-search --dataset math500 --search_framework rest \\
        --memory-arg backend=mem0 mem0_config_path=./mem0_config.json

``--memory-arg`` implicitly enables memory. ``setup_memory_manager()`` in
``lits/cli/search.py`` constructs the backend and wires it into
``LiTSMemoryManager``; ``FactMemoryAugmentor`` is backend-agnostic.
"""

import logging
from typing import Optional, Dict, Any

from . import ContextAugmentor, ContextUnit
from ...log import log_event
from ...memory.manager import LiTSMemoryManager, AugmentedContext
from ...memory.types import TrajectoryKey, normalize_trajectory_key

logger = logging.getLogger(__name__)


def _extract_messages_from_step(step) -> list[Dict[str, str]]:
    """Extract chat-format messages from the last step of a trajectory.

    Tries several Step attributes in order of preference:
        1. ``step.messages`` — already in ``[{role, content}]`` format
        2. ``step.verb_step()`` — tool-use steps with verbalized output
        3. ``str(step)`` — fallback

    Returns:
        List of message dicts suitable for ``memory_manager.record_action()``.
    """
    if hasattr(step, "messages") and step.messages:
        return list(step.messages)

    # Build a single assistant message from the step content
    if hasattr(step, "verb_step"):
        content = step.verb_step()
    elif hasattr(step, "action") and step.action:
        content = str(step.action)
    else:
        content = str(step)

    if not content:
        return []

    return [{"role": "assistant", "content": content}]


class FactMemoryAugmentor(ContextAugmentor):
    """LiTS-Mem fact extraction and cross-trajectory memory retrieval.

    Wraps :class:`~lits.memory.manager.LiTSMemoryManager` to provide
    atomic fact extraction and cross-trajectory augmentation through the
    unified ContextAugmentor interface.

    This augmentor overrides ``analyze()`` directly because its flow
    differs fundamentally from the dict-with-issues pattern used by
    other augmentors. It does NOT use ``_buffer`` — facts are persisted
    by the backend during ``record_action()``, so buffering would be
    redundant.

    Args:
        memory_manager: A configured :class:`LiTSMemoryManager` instance.
        persist: Persistence mode. Default True — facts are persisted
            by the backend inside ``analyze()`` via ``record_action()``.
        include_inherited: Whether ``retrieve()`` includes inherited
            (ancestor) memories in the prompt block. Default False,
            since inherited context is typically already in the prompt.
    """

    evaluator_type = "fact_memory"

    def __init__(
        self,
        memory_manager: LiTSMemoryManager,
        persist=True,
        include_inherited: bool = False,
        **kwargs,
    ):
        super().__init__(
            base_model=None,
            persist=persist,
            **kwargs,
        )
        self.memory_manager = memory_manager
        self.include_inherited = include_inherited

    # ------------------------------------------------------------------
    # analyze: override directly (NOT _analyze)
    # ------------------------------------------------------------------

    def analyze(self, traj_state, **kwargs) -> Optional[ContextUnit]:
        """Extract facts from trajectory steps and record them in memory.

        Two modes controlled by ``batch`` kwarg:

        - ``batch=False`` (default): Incremental mode — extracts messages
          from ``traj_state[-1]`` only.  Used by tree search where steps
          arrive one at a time.
        - ``batch=True``: Batch mode — extracts messages from ALL steps
          and passes them as a single text to the LLM for fact extraction.
          Used by chain pass@N where the full trajectory is available.
          Produces better facts (LLM sees full context) and is cheaper
          (1 LLM call vs N).

        Args:
            traj_state: Trajectory (list of Steps).
            **kwargs:
                trajectory_key: TrajectoryKey or ``q/...`` string.
                query_idx (int): Example index for logging.
                batch (bool): If True, extract facts from all steps at once.

        Returns:
            ContextUnit summarizing what was recorded, or None if
            no extractable content.
        """
        if traj_state is None or len(traj_state) == 0:
            return None

        trajectory_key = kwargs.get("trajectory_key")
        query_idx = kwargs.get("query_idx", -1)
        from_phase = kwargs.get("from_phase", "")
        batch = kwargs.get("batch", False)

        # Resolve TrajectoryKey
        if isinstance(trajectory_key, TrajectoryKey):
            traj_key_obj = trajectory_key
        elif isinstance(trajectory_key, str) and trajectory_key:
            assert query_idx is not None, (
                "FactMemoryAugmentor.analyze: query_idx is required when "
                "trajectory_key is a string (needed to construct search_id)"
            )
            search_id = f"q_{query_idx}"
            traj_key_obj = TrajectoryKey.from_path(search_id, trajectory_key)
        else:
            logger.debug("FactMemoryAugmentor.analyze: no trajectory_key, skipping")
            return None

        # Extract messages
        if batch:
            # Batch mode: all steps concatenated
            messages = []
            for step in traj_state:
                messages.extend(_extract_messages_from_step(step))
        else:
            # Incremental mode: last step only
            messages = _extract_messages_from_step(traj_state[-1])

        if not messages:
            logger.debug("FactMemoryAugmentor.analyze: no messages extracted, skipping")
            return None

        # Record via memory manager (backend extracts facts internally)
        self.memory_manager.record_action(
            traj_key_obj,
            messages=messages,
            metadata={"from_phase": from_phase},
            infer=True,
            query_idx=query_idx,
        )

        n_steps = len(traj_state) if batch else 1
        content_summary = "; ".join(
            m.get("content", "")[:100] for m in messages[:3]
        )
        unit = ContextUnit(
            content=f"[fact_memory] Recorded from {n_steps} step(s): {content_summary}",
            source=self.evaluator_type,
            trajectory_key=normalize_trajectory_key(trajectory_key),
            query_id=query_idx if query_idx is not None else -1,
            metadata={"n_messages": len(messages), "n_steps": n_steps, "step_idx": None if batch else len(traj_state)},
        )
        logger.debug(
            f"FactMemoryAugmentor: recorded {len(messages)} messages "
            f"from {n_steps} step(s) at {traj_key_obj.path_str}"
        )
        return unit

    # ------------------------------------------------------------------
    # store: no-op (recording happens in analyze via memory_manager)
    # ------------------------------------------------------------------

    def store(self, unit: ContextUnit, **kwargs) -> None:
        """No-op. Facts are persisted by mem0 inside analyze()."""
        pass

    # ------------------------------------------------------------------
    # retrieve: cross-trajectory augmented context
    # ------------------------------------------------------------------

    def retrieve(self, query_context=None, **kwargs) -> str:
        """Retrieve cross-trajectory memory context for prompt injection.

        Delegates to ``memory_manager.build_augmented_context()`` and
        formats the result via ``AugmentedContext.to_prompt_blocks()``.

        Args:
            query_context: Dict with 'trajectory_key' (TrajectoryKey or
                str). Optionally 'query_idx' for search_id inference.

        Returns:
            Formatted prompt block string, or "" if no context available.
        """
        if not query_context:
            return ""

        trajectory_key = query_context.get("trajectory_key")
        if trajectory_key is None:
            return ""

        # Resolve TrajectoryKey
        if isinstance(trajectory_key, TrajectoryKey):
            traj_key_obj = trajectory_key
        elif isinstance(trajectory_key, str) and trajectory_key:
            query_idx = query_context.get("query_idx")
            assert query_idx is not None, (
                "FactMemoryAugmentor.retrieve: query_context must contain "
                "'query_idx' (int) when trajectory_key is a string "
                "(needed to construct search_id)"
            )
            search_id = f"q_{query_idx}"
            traj_key_obj = TrajectoryKey.from_path(search_id, trajectory_key)
        else:
            return ""

        context: AugmentedContext = self.memory_manager.build_augmented_context(
            traj_key_obj
        )
        result = context.to_prompt_blocks(include_inherited=self.include_inherited)
        log_event(
            logger, "Memory",
            f"retrieve: traj_key={traj_key_obj.path_str}, "
            f"search_id={traj_key_obj.search_id}, "
            f"inherited={len(context.inherited_units)}, "
            f"retrieved_trajs={len(context.retrieved_trajectories)}, "
            f"result_len={len(result)}",
        )
        return result

    # ------------------------------------------------------------------
    # flush_buffer: override to skip jsonl write (mem0 is the store)
    # ------------------------------------------------------------------

    def flush_buffer(self, **kwargs) -> None:
        """No-op. Facts are persisted by the backend during analyze()."""
        pass
