import copy
import logging
from typing import Optional

from ..base import Transition
from ...structures import ToolUseState, ToolUseStep, ToolUseAction, log_state
from ...structures.tool_use import BaseToolUseStep
from ...tools.utils import execute_tool_action
from ...tools import ToolServerDownError
from ...log import log_event

logger = logging.getLogger(__name__)


class ToolUseTransition(Transition[ToolUseState, ToolUseAction]):
    """Transition model that materializes tool observations for ToolUsePolicy-driven search.
    
    This transition receives a ToolUseAction (extracted from ToolUseStep by the policy),
    executes it via execute_tool_action, and constructs a complete ToolUseStep with
    the observation to append to the state.
    """

    def __init__(
        self,
        tools: list,
        observation_on_error: str = "Tool execution failed.",
        tool_failure_threshold: int = 3,
    ):
        """Args:
            tools: Available tool instances.
            observation_on_error: Observation surfaced to the LLM when a tool
                raises a non-network exception.
            tool_failure_threshold: Number of `ToolServerDownError`s that
                must accumulate before `step()` re-raises to abort the run.
                The counter increments on `ToolServerDownError`, resets on a
                successful tool call, and is unaffected by other exceptions
                or by `init_state()`.
        """
        self.tools = tools
        self.observation_on_error = observation_on_error
        self.tool_failure_threshold = tool_failure_threshold
        self.consecutive_server_down: int = 0

    def init_state(self, **kwargs) -> ToolUseState:
        """Return a fresh empty trajectory.

        Note: does not reset `consecutive_server_down`. The counter persists
        across examples so that an unhealthy backend (visible only as
        intermittent failures spread across examples) still trips the breaker.
        """
        return ToolUseState()

    def step(
        self,
        state: ToolUseState,
        step_or_action,
        query_or_goals: str=None,
        query_idx: Optional[int] = None,
        from_phase: str = "",
    ):
        """Execute the tool step/action and append the resulting ToolUseStep to state.
        
        This method handles three cases:
        1. ToolUseStep with action: Execute the action and add observation
        2. ToolUseStep with answer: Append the answer step directly (terminal)
        3. ToolUseStep with error: Append the error step directly
        
        Args:
            state: Current ToolUseState (trajectory of ToolUseSteps)
            step_or_action: ToolUseStep (from policy) or ToolUseAction to execute
            query_or_goals: Optional query/goal context
            query_idx: Optional query index for logging
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            Tuple of (new_state, aux_dict) where:
            - new_state: Updated ToolUseState with executed step appended
            - aux_dict: Auxiliary data (e.g., confidence)
        """
        # Create new state by copying the existing state
        new_state = ToolUseState()
        new_state.extend(state)
        
        # step_or_action should be a BaseToolUseStep (ToolUseStep or NativeToolUseStep)
        assert isinstance(step_or_action, BaseToolUseStep), \
            f"Expected BaseToolUseStep, got {type(step_or_action)}"
        
        step = step_or_action
        
        # Case 1: Step has an answer (terminal) - append directly
        if step.answer is not None:
            log_event(logger, "TRANSITION", f"Step has answer, appending directly: {step.answer}", level="debug")
            new_state.append(step)
            log_state(logger, new_state, header="ToolUseTransition.step (answer)")
            return new_state, {"confidence": 1.0}
        
        # Case 2: Step has an error - append directly
        if step.error is not None:
            log_event(logger, "TRANSITION", f"Step has error, appending directly: {step.error}", level="debug")
            new_state.append(step)
            log_state(logger, new_state, header="ToolUseTransition.step (error)")
            return new_state, {"confidence": 0.0}
        
        # Case 3: Handle cases where no action and answer are provided
        elif step.answer is None and step.action is None:
            log_event(logger, "TRANSITION", "No action or answer provided", level="warning")
            step.observation = (
                "Assistant output did not provide an action or answer, or it did not follow the required "
                "format and could not be parsed. Please STRICTLY follow the format required in the system prompt."
            ) 
            new_state.append(step)  
            return new_state, {"confidence": 0.0}
        
        # Case 3: Step has an action - execute it
        action = step.action
        
        # Execute the tool action to get observation
        observation = None
        if action:
            # Call pre_step hook on all tools (e.g., KG tools rebuild variable tracker)
            for tool in self.tools:
                if hasattr(tool, 'pre_step'):
                    tool.pre_step(new_state)
            try:
                observation = execute_tool_action(str(action), self.tools)
            except ToolServerDownError as e:
                self.consecutive_server_down += 1
                counter = self.consecutive_server_down
                threshold = self.tool_failure_threshold
                reason_repr = repr(e.original_exc) if e.original_exc is not None else str(e)

                if counter >= threshold:
                    log_event(
                        logger,
                        "TOOL_SERVER_DOWN",
                        f"example={query_idx} TRIPPED after {counter} consecutive failures (cross-example), "
                        f"tool={e.tool_name} last_reason={reason_repr}",
                        level="warning",
                    )
                    step.observation = (
                        f"Tool server unreachable. Final attempt before circuit-breaker "
                        f"abort. Reason: {e.original_exc or e}"
                    )
                    new_state.append(step)
                    log_state(logger, new_state, header="ToolUseTransition.step (server-down trip)")
                    raise
                else:
                    step.observation = (
                        f"Tool server unreachable (attempt {counter}/{threshold}). "
                        f"Reason: {e.original_exc or e}"
                    )
                    new_state.append(step)
                    log_state(logger, new_state, header="ToolUseTransition.step (server-down)")
                    return new_state, {"confidence": 0.0}
            except Exception as exc:
                # Counter unchanged: not evidence of backend health either way.
                logger.exception("Tool execution failed for example %s: %s", query_idx, exc)
                observation = f"{self.observation_on_error} Reason: {exc}"
            else:
                # Successful call resets the counter.
                self.consecutive_server_down = 0
            if not isinstance(observation, str):
                observation = str(observation)
        
        # Update the step with the observation and append to state
        step.observation = observation
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
