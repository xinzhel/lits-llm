import copy
import logging
from typing import Optional

from ..base import Transition
from ..structures import ToolUseState, ToolUseStep
from ...agents.utils import execute_tool_action
from ..structures import log_state

logger = logging.getLogger(__name__)


class ToolUseTransition(Transition[ToolUseState, ToolUseStep, str]):
    """Transition model that materializes tool observations for ToolUsePolicy-driven search."""

    def __init__(self, tools: list, observation_on_error: str = "Tool execution failed."):
        self.tools = tools
        self.observation_on_error = observation_on_error

    def init_state(self) -> ToolUseState:
        """Start each search trace with an empty tool-use history."""
        return ToolUseState()

    def step(
        self,
        example: str,
        state: ToolUseState,
        action: ToolUseStep,
        example_idx: Optional[int] = None,
        from_phase: str = "",
    ):
        """Append the sampled ToolUseStep and execute the associated tool if needed."""
        new_state = ToolUseState(state.copy())
        step = copy.deepcopy(action)

        if step.action and step.observation is None:
            try:
                observation = execute_tool_action(step.action, self.tools)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Tool execution failed for example %s: %s", example_idx, exc)
                observation = f"{self.observation_on_error} Reason: {exc}"
            if not isinstance(observation, str):
                observation = str(observation)
            step.observation = observation
        new_state.append(step)
        log_state(logger, new_state, header="ToolUseTransition.step")
        return new_state, {"confidence": 1.0}

    def is_terminal(
        self,
        state: ToolUseState,
        example: Optional[str] = None,
        fast_reward: Optional[float] = None,
        example_idx: Optional[int] = None,
        from_phase: str = "",
    ) -> bool:
        """Stop expansion once the latest step provides a final answer."""
        if not state:
            return False
        last = state[-1]
        return bool(last.get_answer())
