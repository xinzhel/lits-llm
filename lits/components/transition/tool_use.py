import copy
import logging
from typing import Optional

from ..base import Transition
from ...structures import ToolUseState, ToolUseStep, ToolUseAction, log_state
from ...tools.utils import execute_tool_action

logger = logging.getLogger(__name__)


class ToolUseTransition(Transition[ToolUseState, ToolUseAction]):
    """Transition model that materializes tool observations for ToolUsePolicy-driven search.
    
    This transition receives a ToolUseAction (extracted from ToolUseStep by the policy),
    executes it via execute_tool_action, and constructs a complete ToolUseStep with
    the observation to append to the state.
    """

    def __init__(self, tools: list, observation_on_error: str = "Tool execution failed."):
        self.tools = tools
        self.observation_on_error = observation_on_error

    def init_state(self) -> ToolUseState:
        """Start each search trace with an empty tool-use history."""
        return ToolUseState()

    def step(
        self,
        state: ToolUseState,
        action: ToolUseAction,
        query_or_goals: str=None,
        query_idx: Optional[int] = None,
        from_phase: str = "",
    ):
        """Execute the tool action and append the resulting ToolUseStep to state.
        
        Args:
            state: Current ToolUseState (trajectory of ToolUseSteps)
            action: ToolUseAction to execute (extracted from ToolUseStep by policy)
            query_or_goals: Optional query/goal context
            query_idx: Optional query index for logging
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            Tuple of (new_state, aux_dict) where:
            - new_state: Updated ToolUseState with executed step appended
            - aux_dict: Auxiliary data (e.g., confidence)
        """
        # Create new state by copying the existing state
        # ToolUseState extends list, so we copy and then wrap in ToolUseState
        new_state = ToolUseState()
        new_state.extend(state)
        
        # Execute the tool action to get observation
        observation = None
        if action:
            try:
                observation = execute_tool_action(str(action), self.tools)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Tool execution failed for example %s: %s", query_idx, exc)
                observation = f"{self.observation_on_error} Reason: {exc}"
            if not isinstance(observation, str):
                observation = str(observation)
        
        # Construct a ToolUseStep with the action and observation
        step = ToolUseStep(
            action=action,
            observation=observation
        )
        
        new_state.append(step)
        log_state(logger, new_state, header="ToolUseTransition.step")
        return new_state, {"confidence": 1.0}

    def is_terminal(
        self,
        state: ToolUseState,
        query_or_goals: Optional[str] = None,
        fast_reward: Optional[float] = None,
        query_idx: Optional[int] = None,
        from_phase: str = "",
    ) -> bool:
        """Stop expansion once the latest step provides a final answer."""
        if not state:
            return False
        last = state[-1]
        return bool(last.get_answer())
