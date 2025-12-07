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
        policy_model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        step_evaluators: Optional[list] = None,
        trajectory_evaluators: Optional[list] = None,
    ):
        """Initialize ReActChat with policy and transition components.
        
        Args:
            policy: ToolUsePolicy that generates actions based on state
            transition: ToolUseTransition that executes actions and produces observations
            max_iter: Maximum number of reasoning iterations
            policy_model_name: Optional policy model name (for callbacks like SQL validation)
            task_type: Optional task type (for callbacks like SQL validation)
            step_evaluators: Optional list of step-level evaluators (e.g., SQLValidator)
                that validate each generated step
            trajectory_evaluators: Optional list of trajectory-level evaluators 
                (e.g., SQLErrorProfiler) that analyze the complete trajectory
        """
        self.policy = policy
        self.transition = transition
        self.max_iter = max_iter
        self.policy_model_name = policy_model_name
        self.task_type = task_type
        self.step_evaluators = step_evaluators or []
        self.trajectory_evaluators = trajectory_evaluators or []
        
        # Setup learning loop if evaluators provided
        if self.step_evaluators or self.trajectory_evaluators:
            self._setup_learning_loop()
    
    def _setup_learning_loop(self):
        """Setup the learning loop with dynamic notes and validation.
        
        This creates a feedback loop where:
        1. Past issues are loaded and injected into the prompt (input enhancement)
        2. New generations are validated at step-level (SQLValidator)
        3. Complete trajectories are analyzed (SQLErrorProfiler)
        4. Next iteration uses the newly saved issues
        """
        all_evaluators = self.step_evaluators + self.trajectory_evaluators
        
        # Setup dynamic notes function (input enhancement)
        def get_dynamic_notes():
            """Load past issues from all evaluators and format as notes."""
            all_notes = ""
            for evaluator in all_evaluators:
                try:
                    notes = evaluator.load_eval_as_prompt(
                        self.policy_model_name,
                        self.task_type,
                        max_items=5
                    )
                    if notes:
                        all_notes = all_notes + notes
                except Exception as e:
                    logger.error(f"Error loading notes from {evaluator.__class__.__name__}: {e}")
            return all_notes
        
        self.policy.set_dynamic_notes_fn(get_dynamic_notes)
        logger.info(f"Dynamic notes function set with {len(all_evaluators)} evaluator(s)")
        
        # Setup post-generation validation function for step-level evaluators
        if self.step_evaluators:
            def validate_steps(steps, context):
                """Validate generated steps with step-level evaluators.
                
                Includes state history as context for more informed validation.
                """
                # Build context string from state history (action-observation pairs, no think)
                state = context.get('state')
                state_context = None
                if state and len(state) > 0:
                    # Temporarily set exclude_think_when_verb to True
                    original_exclude_think = ToolUseStep.exclude_think_when_verb
                    ToolUseStep.exclude_think_when_verb = True
                    try:
                        state_context = state.render_history()
                    finally:
                        # Restore original setting
                        ToolUseStep.exclude_think_when_verb = original_exclude_think
                
                for evaluator in self.step_evaluators:
                    try:
                        for step in steps:
                            evaluator.evaluate(
                                step,
                                context=state_context,  # Pass state history as context
                                query_idx=context.get('query_idx'),
                                policy_model_name=context.get('policy_model_name'),
                                task_type=context.get('task_type')
                            )
                    except Exception as e:
                        logger.error(f"Error in {evaluator.__class__.__name__}.evaluate(): {e}")
            
            self.policy.set_post_generation_fn(validate_steps)
            logger.info(f"Step-level validation set with {len(self.step_evaluators)} evaluator(s)")

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
            if len(state) > 0 and getattr(state[-1], "answer", None) is not None:
                break
            logger.debug("\n ======== Iteration %d ========\n", i)
            state = self.update_state(query, state, query_idx=query_idx, from_phase=from_phase)
            
            if checkpoint_path:
                state.save(checkpoint_path, query)
                logger.debug("\nCheckpoint saved to %s \n", checkpoint_path)
        
        # Evaluate complete trajectory with trajectory-level evaluators
        if self.trajectory_evaluators:
            self._evaluate_trajectory(state, query_idx)
        
        return state
    
    def _evaluate_trajectory(self, state, query_idx):
        """Evaluate the complete trajectory with trajectory-level evaluators.
        
        Args:
            state: Complete ToolUseState trajectory
            query_idx: Query index for logging
        """
        for evaluator in self.trajectory_evaluators:
            try:
                evaluator.evaluate(
                    state,
                    query_idx=query_idx,
                    policy_model_name=self.policy_model_name,
                    task_type=self.task_type
                )
                logger.debug(f"Trajectory evaluated by {evaluator.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error in {evaluator.__class__.__name__}.evaluate(): {e}")

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
        # Pass policy_model_name and task_type for post-generation callbacks
        steps = self.policy.get_actions(
            state,
            query=query,
            n_actions=1,
            query_idx=query_idx,
            from_phase=from_phase,
            policy_model_name=self.policy_model_name,
            task_type=self.task_type,
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
