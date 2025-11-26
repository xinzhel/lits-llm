import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from ...tools import execute_tool_action
from ...components.policy.tool_use import ToolUsePolicy
from ...structures import ToolUseState, ToolUseStep
from ...lm import HfChatModel, InferenceLogger, get_lm
from ...framework_config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, PACKAGE_VERSION
from ..base import BaseConfig

    
logger = logging.getLogger(__name__)

@dataclass
class ReactChatConfig(BaseConfig):
    """Persistence helper mirroring other LiTS configs."""

    model_name: Optional[str] = None
    max_length: Optional[int] = None
    enable_think: bool = True
    gpu_device: Optional[str] = None
    
    exclude_think_when_verb: bool = False
    secret_token: str = None
    client_host: str = None
    client_port: int = None
    timeout: int = 30

    def to_dict(self):
        return asdict(self)
    
def resume_tool_use_state(checkpoint_path):
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        query, state = ToolUseState.load(str(checkpoint_file))
        logger.debug("\n\n\n\nResuming conversation !!!!!!!!!!")
    else:
        state = ToolUseState()
        logger.debug("\n\n\n\nStarting ReAct evaluation !!!!!!!!!")
    return state

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
        if checkpoint_path:
            state = resume_tool_use_state(checkpoint_path)
        else:
            state = ToolUseState()

        logger.debug("Initial user query:\n%s\n", query)
        start_iter = len(state)
        for i in range(start_iter, self.max_iter):
            logger.debug("\n ======== Iteration %d ========\n", i)
            step = self.get_step(query, state, example_idx=example_idx, from_phase=from_phase)
            state.append(step)

            if getattr(step, "error", None) is not None:
                break
            
            if getattr(step, "observation", None) is not None:
                obs_content = step.observation if isinstance(step.observation, str) else str(step.observation)
                obs_message = f"<observation>\n{obs_content.strip()}\n</observation>"
                logger.debug(">>>>>>>>> Tool observation: %s <<<<<<<<<<", obs_message)

            if checkpoint_path:
                state.save(checkpoint_path, query)
                logger.debug("\nCheckpoint saved to %s \n", checkpoint_path)
            if getattr(step, "answer", None) is not None:
                break
        return state

    def get_step(self, query: str, state: ToolUseState, example_idx=None, from_phase: str = "") -> ToolUseStep:
        """Elicit the next <think>/<action>/<answer> block and run tools when requested."""
        steps = self.policy.get_actions(
            state,
            query=query,
            n_actions=1,
            query_idx=example_idx,
            from_phase=from_phase,
        )
        if not steps:
            raise RuntimeError("ToolUsePolicy returned no candidate generation.")
        
        step = steps[0]
        if step.error:
            return step
        assistant_text = step.assistant_message or step.verb_step()
        logger.debug(">>>>>>>>> Assistant raw output:\n%s <<<<<<<<<<", assistant_text)

        if step.action:
            step.observation = execute_tool_action(str(step.action), self.policy.tools)

        if step.answer is None and step.action is None:
            logger.warning("Either action or answer must be provided in assistant output.")
            step.observation = (
                "Assistant output did not provide an action or answer, or it did not follow the required "
                "format and could not be parsed. Please STRICTLY follow the format required in the system prompt."
            )

        return step
