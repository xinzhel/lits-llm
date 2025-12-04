import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from ...components.policy.tool_use import ToolUsePolicy
from ...components.transition.tool_use import ToolUseTransition
from ...structures import ToolUseState, ToolUseStep
from ...lm import HfChatModel, InferenceLogger, get_lm
from ...framework_config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, PACKAGE_VERSION
from ..base import BaseConfig

    
logger = logging.getLogger(__name__)

@dataclass
class ReactChatConfig(BaseConfig):
    """
    Configuration for ReAct-style reasoning and acting agent.
    
    Inherits common attributes from BaseConfig:
        - model_name: Language model name
        - gpu_device: GPU device identifier
        - max_length: Maximum token length for generation
        - max_steps: Maximum number of reasoning iterations (default: 10)
    """
    enable_think: bool = True
    exclude_think_when_verb: bool = False
    timeout: int = 30
    
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

    Question → Thought → Action → Observation → Thought → … → Final Answer
    
    This implementation follows the LiTS framework's separation of concerns:
    - Policy: Generates actions (ToolUseStep with action field)
    - Transition: Executes actions and produces observations
    """
    
    def __init__(
        self,
        policy: ToolUsePolicy,
        transition: ToolUseTransition,
        max_iter: int = 10,
    ):
        """Initialize ReActChat with policy and transition components.
        
        Args:
            policy: ToolUsePolicy that generates actions based on state
            transition: ToolUseTransition that executes actions and produces observations
            max_iter: Maximum number of reasoning iterations
        """
        self.policy = policy
        self.transition = transition
        self.max_iter = max_iter

    def run(self, query, query_idx=None, from_phase: str = "", checkpoint_path: Optional[str] = None):
        """Run the ReAct reasoning-and-acting loop.
        
        Args:
            query: The user's question or task
            query_idx: Optional query index for logging
            from_phase: Description of algorithm phase (for logging)
            checkpoint_path: Optional path to save/load checkpoints
        
        Returns:
            ToolUseState: Final state containing the trajectory of steps
        """
        if checkpoint_path:
            state = resume_tool_use_state(checkpoint_path)
        else:
            state = self.transition.init_state()

        logger.debug("Initial user query:\n%s\n", query)
        start_iter = len(state)
        for i in range(start_iter, self.max_iter):
            logger.debug("\n ======== Iteration %d ========\n", i)
            state = self.update_state(query, state, query_idx=query_idx, from_phase=from_phase)
            
            if checkpoint_path:
                state.save(checkpoint_path, query)
                logger.debug("\nCheckpoint saved to %s \n", checkpoint_path)
            if getattr(state[-1], "answer", None) is not None:
                break
        return state

    def update_state(self, query: str, state: ToolUseState, query_idx=None, from_phase: str = "") -> ToolUseStep:
        """
        
        This method follows the LiTS framework pattern:
        1. Policy generates a ToolUseStep with action (but no observation)
        2. Transition executes the action and produces observation
        3. Update and Return the state 
        Args:
            query: The user's question or task
            state: Current ToolUseState (trajectory of steps)
            query_idx: Optional query index for logging
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            ToolUseState
        """
        # Step 1: Policy generates action
        steps = self.policy.get_actions(
            state,
            query=query,
            n_actions=1,
            query_idx=query_idx,
            from_phase=from_phase,
        )
        if not steps:
            raise RuntimeError("ToolUsePolicy returned no candidate generation.")
        
        step = steps[0]

        
        assistant_text = step.assistant_message or step.verb_step()
        logger.debug(">>>>>>>>> Assistant raw output:\n%s <<<<<<<<<<", assistant_text)

         # Step 2.0: Error
        if step.error:
            state.append(step)
            return state
        
        # Step 2.1: Transition executes action if present
        if step.action:
            # Use transition to execute the action and get observation
            new_state, aux = self.transition.step(
                state=state,
                action=step.action,
                query_or_goals=query,
                query_idx=query_idx,
                from_phase=from_phase
            )
            # Extract the executed step (last one in new_state)
            executed_step = new_state[-1]
            # Preserve think and answer from original step
            executed_step.think = step.think
            executed_step.answer = step.answer
            executed_step.assistant_message = step.assistant_message
            state.append(executed_step)
        # Step 2.2
        elif step.answer is not None:
            state.append(step)
            pass
        # Step 2.3: Handle cases where no action is provided
        elif step.answer is None and step.action is None:
            logger.warning("Either action or answer must be provided in assistant output.")
            step.observation = (
                "Assistant output did not provide an action or answer, or it did not follow the required "
                "format and could not be parsed. Please STRICTLY follow the format required in the system prompt."
            ) 
            state.append(step)       
        else:
            raise Exception
        
        return state
