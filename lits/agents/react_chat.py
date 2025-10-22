import logging
from pathlib import Path
from typing import Optional

from lits.agents.utils import execute_tool_action
from lits.components.policy.tool_use import ToolUsePolicy
from lits.components.structures import ToolUseState, ToolUseStep

logger = logging.getLogger(__name__)

class ReActChat:
    """Implements a ReAct-style reasoning-and-acting loop for tool-augmented LLMs.

    The model receives a system prompt describing the reasoning format:

    Question → Thought → Action → Observation → Thought → … → Final Answer"""
    
    def __init__(
        self,
        policy: ToolUsePolicy,
        max_iter: int = 10,
    ):
        self.policy = policy
        self.max_iter = max_iter

    def run(self, query, example_idx=None, from_phase: str = "", checkpoint_path: Optional[str] = None):
        """Run the ReAct reasoning-and-acting loop."""
        state = None
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                query, state = ToolUseState.load(str(checkpoint_file))
                logger.debug("Resuming conversation from checkpoint: %s", checkpoint_file)
        if state is None:
            state = ToolUseState()

        logger.debug("Initial user query:\n%s\n", query)
        start_iter = len(state)
        for i in range(start_iter, self.max_iter):
            logger.debug("\n ======== Iteration %d ========\n", i)
            step = self.get_step(query, state, example_idx=example_idx, from_phase=from_phase)
            state.append(step)

            if step.observation is not None:
                obs_content = step.observation if isinstance(step.observation, str) else str(step.observation)
                obs_message = f"<observation>\n{obs_content.strip()}\n</observation>"
                logger.debug(">>>>>>>>> Tool observation: %s <<<<<<<<<<", obs_message)

            if checkpoint_path:
                state.save(checkpoint_path, query)
                logger.debug("\nCheckpoint saved to %s \n", checkpoint_path)
            if step.answer is not None:
                break
        return state

    def get_step(self, query: str, state: ToolUseState, example_idx=None, from_phase: str = "") -> ToolUseStep:
        """Elicit the next <think>/<action>/<answer> block and run tools when requested."""
        steps = self.policy.get_actions(
            query,
            state,
            n_actions=1,
            example_idx=example_idx,
            from_phase=from_phase,
        )
        if not steps:
            raise RuntimeError("ToolUsePolicy returned no candidate actions.")
        step = steps[0]
        assistant_text = step.assistant_message or step.verb_step()
        logger.debug(">>>>>>>>> Assistant raw output:\n%s <<<<<<<<<<", assistant_text)

        if step.action:
            step.observation = execute_tool_action(step.action, self.policy.tools)
            if isinstance(step.observation, str) and "ConnectTimeoutError" in step.observation:
                raise Exception(f"Network issue or server-side error: {step.observation}")

        if step.answer is None and step.action is None:
            logger.warning("Either action or answer must be provided in assistant output.")
            step.observation = (
                "Assistant output did not provide an action or answer, or it did not follow the required "
                "format and could not be parsed. Please STRICTLY follow the format required in the system prompt."
            )

        return step

