"""CriticAugmentor: per-step critic feedback for tree search.

Extracts the generate_critic logic from ConcatTransition into a
ContextAugmentor subclass. The critic analyzes the current trajectory
and produces advice that is injected into the policy's system prompt
via the dynamic_notes mechanism.

Supports both language-grounded (math/science reasoning) and tool-use
(SQL, web, PDF) task types. The task_type controls which system prompt
and message formatting are used.

Usage:
    # Language-grounded (math reasoning)
    critic = CriticAugmentor(base_model=lm)

    # Tool-use (SQL, web search, etc.)
    critic = CriticAugmentor(base_model=lm, task_type="tool_use")

    # Custom prompt
    critic = CriticAugmentor(base_model=lm, critic_prompt="Your custom prompt")
"""

import logging
from typing import Optional, Dict, Any

from . import ContextAugmentor, ContextUnit
from ...lm.base import DETERMINISTIC_TEMPERATURE
from ...memory.types import normalize_trajectory_key

logger = logging.getLogger(__name__)

CRITIC_PROMPT_LANGUAGE_GROUNDED = (
    "Given a science or math problem and a corresponding solution "
    "that may be incomplete, your task is to give some advice on how "
    "to solve the problem based on current steps or what to consider next."
)

CRITIC_PROMPT_TOOL_USE = (
    "You are reviewing an agent's tool-use trajectory for a given task. "
    "The trajectory consists of thought-action-observation steps where "
    "the agent reasons, invokes tools (e.g. SQL queries, web searches, "
    "API calls), and receives observations. Based on the trajectory so "
    "far, provide brief, actionable advice on what the agent should "
    "consider or correct in the next step. Focus on tool usage mistakes, "
    "missing information, or strategy improvements."
)

# Map task_type to default prompt
_DEFAULT_PROMPTS = {
    "language_grounded": CRITIC_PROMPT_LANGUAGE_GROUNDED,
    "tool_use": CRITIC_PROMPT_TOOL_USE,
}


def _build_user_message(traj_state, query_or_goals: str, task_type: str) -> str:
    """Build the user message from trajectory state.

    For language_grounded: "Question: ... Step 1: ... Step N: ..."
    For tool_use: "Task: ... " followed by each step's verb_step() output,
    which includes <think>/<action>/<observation> XML tags.

    Args:
        traj_state: Iterable of Step objects (ThoughtStep or ToolUseStep).
        query_or_goals: The problem/question being solved.
        task_type: "language_grounded" or "tool_use".

    Returns:
        Formatted user message string.
    """
    if task_type == "tool_use":
        parts = [f"Task: {query_or_goals}\n"]
        for idx, step in enumerate(traj_state):
            parts.append(f"--- Step {idx + 1} ---")
            if hasattr(step, "verb_step"):
                parts.append(step.verb_step())
            else:
                parts.append(str(step))
        return "\n".join(parts)
    else:
        # language_grounded: original generate_critic format
        msg = f"Question: {query_or_goals}\n"
        for idx, step in enumerate(traj_state):
            action_text = step.action if hasattr(step, "action") else str(step)
            msg += f"Step {idx + 1}: {action_text}\n"
        return msg


class CriticAugmentor(ContextAugmentor):
    """Per-step critic that analyzes trajectory progress and produces advice.

    Mirrors the old ConcatTransition.generate_critic() behaviour for
    language-grounded tasks, and extends it to tool-use tasks with an
    appropriate prompt and message format.

    The task_type controls:
    - System prompt (math/science vs tool-use)
    - User message format (plain steps vs XML-tagged steps)

    retrieve() returns the most recent critic for the current
    trajectory_key so the policy can inject it as a dynamic note.

    Args:
        base_model: Chat LLM for generating critic feedback.
        task_type: "language_grounded" (default) or "tool_use".
        critic_prompt: Custom system prompt. If None, auto-selected
            based on task_type.
        persist: Persistence mode (True / False / "auto").
        history_access: Set of access levels for retrieve filtering.
            Default {"cross_step"} = only same trajectory + same query.
    """

    evaluator_type = "critic"

    def __init__(
        self,
        base_model,
        task_type: str = "language_grounded",
        critic_prompt: str = None,
        persist="auto",
        history_access=None,
        **kwargs,
    ):
        super().__init__(
            base_model=base_model,
            persist=persist,
            require_chat_model=True,
            **kwargs,
        )
        self.task_type = task_type
        self.critic_prompt = critic_prompt or _DEFAULT_PROMPTS.get(
            task_type, CRITIC_PROMPT_LANGUAGE_GROUNDED
        )
        self.history_access = history_access or {"cross_step"}

    # ------------------------------------------------------------------
    # analyze: trajectory state → ContextUnit with critic advice
    # ------------------------------------------------------------------

    def _analyze(self, traj_state, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate critic feedback for the current trajectory state.

        Args:
            traj_state: Iterable of steps (ThoughtStep or ToolUseStep).
            **kwargs:
                query_or_goals (str): The problem/question being solved.
                query_idx (int): Example index for logging.
                from_phase (str): Algorithm phase for logging.

        Returns:
            Dict with 'issues' key containing the critic advice string,
            or None if traj_state is empty.
        """
        query_or_goals = kwargs.get("query_or_goals", "")
        query_idx = kwargs.get("query_idx")
        from_phase = kwargs.get("from_phase", "")

        if traj_state is None or len(traj_state) == 0:
            return None

        user_message = _build_user_message(
            traj_state, query_or_goals, self.task_type
        )

        # LLM call
        self.base_model.sys_prompt = self.critic_prompt
        output = self._call_model(
            user_message,
            query_idx=query_idx,
            from_phase=from_phase,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_new_tokens=1024,
        )
        advice = output.text.strip()

        if not advice:
            return None

        return {"issues": advice}

    # ------------------------------------------------------------------
    # _should_persist_unit: persist="auto" hook
    # ------------------------------------------------------------------

    def _should_persist_unit(self, unit: ContextUnit) -> bool:
        """In auto mode, persist only non-trivial critic advice."""
        if not unit.content or unit.content.lower() == "no critic":
            return False
        return True

    # ------------------------------------------------------------------
    # retrieve: return latest critic for current trajectory
    # ------------------------------------------------------------------

    def retrieve(self, query_context=None, **kwargs) -> str:
        """Return the most recent critic advice for the current trajectory.

        Uses _filter_by_history_access to select relevant units from
        _buffer, then returns the last one (most recent).

        Args:
            query_context: Dict with 'trajectory_key' and 'query_id'.

        Returns:
            Critic advice string, or "" if none available.
        """
        if not query_context:
            return ""

        traj_key = query_context.get("trajectory_key")
        query_id = query_context.get("query_id")

        candidates = self._filter_by_history_access(
            history_access=self.history_access,
            trajectory_key=normalize_trajectory_key(traj_key),
            query_id=query_id,
        )

        if not candidates:
            return ""

        # Return the most recent critic (last in buffer order)
        latest = candidates[-1]
        return f"Advice: {latest.content}"
