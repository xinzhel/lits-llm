"""ReflectionAugmentor: LATS-style reflection from failed trajectories.

Generates reflections from failed trajectories (reward below threshold)
and injects them into the policy prompt for subsequent iterations.
This implements the LATS reflection mechanism as a ContextAugmentor
subclass, with persistent storage (jsonl) and a write-back buffer.

Key differences from CriticAugmentor:
    - Trigger: per-trajectory (on_trajectory_complete), not per-step
    - Input: complete failed trajectory, not current prefix
    - Semantics: post-hoc retrospection ("why did this fail?"),
      not in-progress guidance ("what to do next?")
    - History access: cross_trajectory by default (shared across
      iterations within one search), matching LATS original behaviour

Usage:
    reflection = ReflectionAugmentor(base_model=lm)

    # Tool-use tasks
    reflection = ReflectionAugmentor(base_model=lm, task_type="tool_use")

    # Custom failure threshold
    reflection = ReflectionAugmentor(base_model=lm, reward_threshold=0.5)
"""

import logging
from typing import Optional, Dict, Any, List

from . import ContextAugmentor, ContextUnit
from ...lm.base import DETERMINISTIC_TEMPERATURE
from ...memory.types import normalize_trajectory_key

logger = logging.getLogger(__name__)

REFLECTION_PROMPT_LANGUAGE_GROUNDED = (
    "You are an expert problem solver. Given a failed attempt at solving "
    "a math or science problem, diagnose the likely reason for failure "
    "and devise a new, concise, high-level plan that avoids the same "
    "mistakes. Be specific about what went wrong and what to do differently."
)

REFLECTION_PROMPT_TOOL_USE = (
    "You are reviewing a failed agent trajectory for a tool-use task. "
    "The agent attempted to solve a task by invoking tools (e.g. SQL "
    "queries, web searches, API calls) but did not reach a correct "
    "answer. Diagnose the likely reason for failure — focus on tool "
    "usage mistakes, incorrect assumptions, or missing steps — and "
    "suggest a concise, high-level plan to avoid the same mistakes."
)

_DEFAULT_PROMPTS = {
    "language_grounded": REFLECTION_PROMPT_LANGUAGE_GROUNDED,
    "tool_use": REFLECTION_PROMPT_TOOL_USE,
}

# ---------------------------------------------------------------------------
# Original LATS reflection prompts, keyed by benchmark name.
# Source: OOS-git/LanguageAgentTreeSearch/{hotpot,webshop,programming}/
#
# Each prompt expects a ``{trajectory}`` placeholder containing the failed
# trajectory text.  The LLM is asked to append its reflection after the
# trailing "Reflection:" tag.
# ---------------------------------------------------------------------------

LATS_REFLECTION_PROMPTS: dict[str, str] = {}

# -- hotpotqa (Docstore API / Wikipedia QA) --------------------------------
# From hotpot/hotpot.py  ``reflection_prompt``
LATS_REFLECTION_PROMPTS["hotpotqa"] = (
    "You are an advanced reasoning agent that can improve based on self "
    "refection. You will be given a previous reasoning trial in which you "
    "were given access to an Docstore API environment and a question to "
    "answer. You were unsuccessful in answering the question either because "
    "you guessed the wrong answer with Finish[<answer>], or you used up "
    "your set number of reasoning steps. In a few sentences, Diagnose a "
    "possible reason for failure and devise a new, concise, high level plan "
    "that aims to mitigate the same failure. Use complete sentences."
)

# -- webshop (web shopping) ------------------------------------------------
# From webshop/prompt.py  ``reflection_prompt`` (system instruction only,
# few-shot examples stripped)
LATS_REFLECTION_PROMPTS["webshop"] = (
    "You are an advanced reasoning agent that can improve based on self "
    "refection. You will be given a previous reasoning trial in which you "
    "were given access to an shopping website and a specific type of item "
    "to buy. You were given access to relevant context and a item to "
    "purchase. You were unsuccessful in buying the correct item either "
    "because you did not find an item meeting all of the required "
    "specifications or because you did not select the correct item. The "
    "ideal score is 1.0, and anything less is incorrect. In a few "
    "sentences, Diagnose a possible reason for failure and devise a new, "
    "concise, high level plan that aims to mitigate the same failure. "
    "Use complete sentences."
)

# -- humaneval (code generation / programming) -----------------------------
# From programming/generators/py_generate.py
# ``PY_SELF_REFLECTION_CHAT_INSTRUCTION`` (chat mode) and
# ``PY_SELF_REFLECTION_COMPLETION_INSTRUCTION`` (completion mode).
# The chat instruction is the primary one used in LATS MCTS.
LATS_REFLECTION_PROMPTS["humaneval"] = (
    "You are a Python programming assistant. You will be given a function "
    "implementation and a series of unit test results. Your goal is to "
    "write a few sentences to explain why your implementation is wrong as "
    "indicated by the tests. You will need this as guidance when you try "
    "again later. Only provide the few sentence description in your "
    "answer, not the implementation."
)


def _build_reflection_message(traj_state, query_or_goals: str, task_type: str,
                              reward: Optional[float] = None) -> str:
    """Build the user message for reflection generation.

    Includes the full trajectory and optionally the reward score so the
    LLM knows how badly the attempt failed.

    Args:
        traj_state: Iterable of Step objects.
        query_or_goals: The problem/task being solved.
        task_type: "language_grounded" or "tool_use".
        reward: Terminal reward (if available).

    Returns:
        Formatted user message string.
    """
    if task_type == "tool_use":
        parts = [f"Task: {query_or_goals}\n"]
        parts.append("Failed trajectory:")
        for idx, step in enumerate(traj_state):
            parts.append(f"--- Step {idx + 1} ---")
            if hasattr(step, "verb_step"):
                parts.append(step.verb_step())
            else:
                parts.append(str(step))
    else:
        parts = [f"Question: {query_or_goals}\n"]
        parts.append("Failed attempt:")
        for idx, step in enumerate(traj_state):
            action_text = step.action if hasattr(step, "action") else str(step)
            parts.append(f"Step {idx + 1}: {action_text}")

    if reward is not None:
        parts.append(f"\nReward score: {reward}")

    parts.append("\nDiagnose the failure and suggest a better approach.")
    return "\n".join(parts)


def _is_failed_path(reward: Optional[float], threshold: float = 0.3) -> bool:
    """Determine whether a trajectory is considered failed.

    A trajectory is failed if its reward is below the threshold, or if
    no reward is available (e.g. depth-limit termination).

    Args:
        reward: Terminal reward. None means unknown (treated as failed).
        threshold: Reward below this value counts as failure.

    Returns:
        True if the trajectory should trigger reflection.
    """
    if reward is None:
        return True
    return reward < threshold


class ReflectionAugmentor(ContextAugmentor):
    """LATS-style reflection from failed trajectories.

    Generates reflections when a trajectory ends with low reward,
    accumulates them in a write-back buffer, and injects them into
    subsequent iterations via retrieve().

    The buffer flushes to jsonl when it reaches flush_threshold or
    when flush_buffer() is called explicitly (e.g. at search end).

    Args:
        base_model: Chat LLM for generating reflections.
        task_type: "language_grounded" (default) or "tool_use".
        reflection_prompt: Custom system prompt. If None, auto-selected
            based on task_type.
        max_reflections: Maximum reflections to inject into prompt.
            Matches LATS default of 3.
        flush_threshold: Buffer size that triggers auto-flush to jsonl.
        reward_threshold: Reward below this triggers reflection.
        persist: Persistence mode (True / False / "auto").
        history_access: Set of access levels for retrieve filtering.
            Default {"cross_trajectory"} = shared across iterations
            within one search, matching LATS original behaviour.
    """

    evaluator_type = "reflection"

    def __init__(
        self,
        base_model,
        task_type: str = "language_grounded",
        reflection_prompt: str = None,
        max_reflections: int = 3,
        flush_threshold: int = 5,
        reward_threshold: float = 0.3,
        persist=True,
        history_access=None,
        **kwargs,
    ):
        super().__init__(
            base_model=base_model,
            persist=persist,
            require_chat_model=True,
            flush_threshold=flush_threshold,
            **kwargs,
        )
        self.task_type = task_type
        self.reflection_prompt = reflection_prompt or _DEFAULT_PROMPTS.get(
            task_type, REFLECTION_PROMPT_LANGUAGE_GROUNDED
        )
        self.max_reflections = max_reflections
        self.reward_threshold = reward_threshold
        self.history_access = history_access or {"cross_trajectory"}

    # ------------------------------------------------------------------
    # analyze: failed trajectory → ContextUnit with reflection
    # ------------------------------------------------------------------

    def _analyze(self, traj_state, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate reflection from a failed trajectory.

        Only produces a reflection if the trajectory is considered failed
        (reward below threshold). The reflection is also appended to
        self._buffer as a write-back cache entry.

        Args:
            traj_state: Complete trajectory (list of Steps).
            **kwargs:
                query_or_goals (str): The problem/task.
                query_idx (int): Example index for logging.
                from_phase (str): Algorithm phase for logging.
                reward (float): Terminal reward of this trajectory.
                trajectory_key (str): Trajectory identifier.

        Returns:
            Dict with 'issues' key containing the reflection text,
            or None if trajectory is not failed or empty.
        """
        query_or_goals = kwargs.get("query_or_goals", "")
        query_idx = kwargs.get("query_idx")
        from_phase = kwargs.get("from_phase", "")
        reward = kwargs.get("reward")
        trajectory_key = kwargs.get("trajectory_key")

        if traj_state is None or len(traj_state) == 0:
            logger.debug("ReflectionAugmentor._analyze: empty traj_state, skipping")
            return None

        if not _is_failed_path(reward, self.reward_threshold):
            logger.debug(
                f"ReflectionAugmentor._analyze: reward={reward} >= "
                f"threshold={self.reward_threshold}, skipping"
            )
            return None

        user_message = _build_reflection_message(
            traj_state, query_or_goals, self.task_type, reward=reward
        )

        # LLM call
        self.base_model.sys_prompt = self.reflection_prompt
        output = self._call_model(
            user_message,
            query_idx=query_idx,
            from_phase=from_phase,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_new_tokens=1024,
        )
        reflection = output.text.strip()

        if not reflection:
            return None

        return {
            "issues": reflection,
            "reward": reward,
            "n_steps": len(traj_state),
        }

    # ------------------------------------------------------------------
    # retrieve: return recent reflections for policy prompt injection
    # ------------------------------------------------------------------

    def retrieve(self, query_context=None, **kwargs) -> str:
        """Return recent reflections formatted for prompt injection.

        Reads from both persisted jsonl and the in-memory _buffer,
        applies history-access filtering, and truncates to
        max_reflections (most recent kept).

        Args:
            query_context: Dict with 'trajectory_key', 'query_id',
                'policy_model_name', 'task_type'.

        Returns:
            Formatted reflection string, or "" if none available.
        """
        if not query_context:
            return ""

        traj_key = query_context.get("trajectory_key")
        query_id = query_context.get("query_id")

        # Collect persisted entries from jsonl
        persisted_units = self._load_persisted_units(query_context)

        # Merge persisted + buffer (buffer entries not yet flushed)
        all_entries = persisted_units + list(self._buffer)

        # Layer 1: history-access filter
        candidates = self._filter_by_history_access(
            history_access=self.history_access,
            trajectory_key=normalize_trajectory_key(traj_key),
            query_id=query_id,
            entries=all_entries,
        )

        # Layer 2: truncate to max_reflections (keep most recent)
        recent = candidates[-self.max_reflections:]

        if not recent:
            return ""

        parts = ["Previous failed attempts and reflections:\n"]
        for i, unit in enumerate(recent, 1):
            parts.append(f"[Reflection {i}]\n{unit.content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_persisted_units(self, query_context: dict) -> List[ContextUnit]:
        """Load persisted reflection records from jsonl as ContextUnits.

        Args:
            query_context: Must contain 'policy_model_name' and 'task_type'.

        Returns:
            List of ContextUnit reconstructed from jsonl records.
        """
        pmn = query_context.get("policy_model_name", "")
        tt = query_context.get("task_type", "")
        if not pmn or not tt:
            return []

        records = self.load_results(pmn, tt, evaluator_type=self.evaluator_type)
        units = []
        for rec in records:
            content = rec.get("content", "")
            if not content:
                # Legacy format: try 'issues' field
                issues = rec.get("issues", [])
                if isinstance(issues, str):
                    content = issues
                elif isinstance(issues, list):
                    content = "\n".join(str(i) for i in issues if i)
            if not content:
                continue
            units.append(ContextUnit(
                content=content,
                source=rec.get("source", self.evaluator_type),
                trajectory_key=rec.get("trajectory_key", ""),
                query_id=rec.get("query_id", -1),
                metadata={
                    k: v for k, v in rec.items()
                    if k not in ("content", "source", "trajectory_key",
                                 "query_id", "evaluator_type", "timestamp",
                                 "issues")
                },
            ))
        return units

    def _should_persist_unit(self, unit: ContextUnit) -> bool:
        """In auto mode, persist reflections that have substantive content."""
        return bool(unit.content and len(unit.content) > 20)
