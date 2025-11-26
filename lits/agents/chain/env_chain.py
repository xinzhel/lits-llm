import logging
from pathlib import Path
from typing import Optional, Callable, Optional, List
from dataclasses import dataclass, asdict
from ..base import BaseConfig
from ...components.policy.env_grounded import EnvPolicy
from ...components.base import Transition
from ...structures.env_grounded import EnvState, EnvStep

logger = logging.getLogger(__name__)

@dataclass
class EnvChainConfig(BaseConfig):
    """Configuration for environment-grounded chain agent."""
    
    model_name: Optional[str] = None
    max_length: Optional[int] = None
    gpu_device: Optional[str] = None
    temperature: float = 0.8
    max_steps: int = 10
    
    def to_dict(self):
        return asdict(self)


def resume_env_state(checkpoint_path):
    """Resume environment state from checkpoint."""
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        query, state = EnvState.load(str(checkpoint_file))
        logger.debug("\n\n\n\nResuming environment chain !!!!!!!!!!")
    else:
        state = None
        logger.debug("\n\n\n\nStarting environment chain evaluation !!!!!!!!!")
    return state


class EnvChain:
    """
    Implements a chain-like invocation of environment-grounded policy.
    
    This agent iteratively:
    1. Generates an action using the policy
    2. Executes the action via the world model (transition)
    3. Checks if the goal is reached
    4. Continues until goal is reached or max steps exceeded
    
    Typical Use Case:
        Planning tasks where an agent needs to generate a sequence of actions
        to reach a goal state (e.g., BlocksWorld, logistics, robotics).
    
    Args:
        policy: EnvPolicy instance for action generation.
        world_model: Transition instance for state transitions.
        goal_check: Callable that takes (state, query) and returns bool indicating
            if the goal is reached.
        max_steps: Maximum number of steps before termination (default: 10).
    
    Example:
        >>> def goal_check(state, query):
        ...     # Check if current state satisfies goal
        ...     return is_goal_satisfied(state.env_state, query)
        >>> 
        >>> agent = EnvChain(
        ...     policy=env_policy,
        ...     world_model=blocks_world_model,
        ...     max_steps=15
        ... )
        >>> 
        >>> final_state = agent.run(
        ...     query="stack A on B",
        ...     init_state_str=init_state_str
        ... )
    """
    
    def __init__(
        self,
        policy: EnvPolicy,
        world_model: Transition,
        max_steps: int = 10,
    ):
        """
        Initialize the environment chain agent.
        
        Args:
            policy: EnvPolicy for generating actions.
            world_model: Transition model for executing actions and updating state.
            max_steps: Maximum number of action steps (default: 10).
        """
        self.policy = policy
        self.world_model = world_model
        self.max_steps = max_steps

    def run(
        self,
        goals: List[str],
        verb_goals: str,
        init_state_str: str,
        example_idx: Optional[int] = None,
        from_phase: str = "",
        checkpoint_path: Optional[str] = None
    ) -> EnvState:
        """
        Run the environment chain to generate a sequence of actions.
        
        Args:
            goals: Goal description or query (e.g., "stack A on B").
            init_state_str: the environment state.
            example_idx: Optional index for logging/tracking.
            from_phase: Description of current phase (e.g., 'planning', 'execution').
            checkpoint_path: Optional path to save/load checkpoints.
        
        Returns:
            Final EnvState after goal is reached or max steps exceeded.
        
        The returned state contains the full action history and can be used to
        extract the action sequence or evaluate the solution.
        """
        logger.debug("Starting EnvChain with goals:\n%s\n", verb_goals)
        logger.debug("Initial state string:\n%s\n", init_state_str)
        
        # Validate inputs
        assert isinstance(goals, list) and len(goals) > 0, "Goals must be a non-empty list."
        assert isinstance(init_state_str, str) and len(init_state_str) > 0, "Initial state string must be a non-empty string."
        assert isinstance(verb_goals, str) and len(verb_goals) > 0, "Verb goals must be a non-empty string."
        for goal in goals:
            assert goal in verb_goals, f"Goal '{goal}' not found in verb_goals string"
            
        # Initialize or resume state
        if checkpoint_path:
            state = resume_env_state(checkpoint_path)
            if state is None:
                state = self.world_model.init_state(init_state_str)
        else:
            state = self.world_model.init_state(init_state_str)
        
        # Ensure history is initialized
        if state.history is None:
            state.history = []
        
        start_step = len(state)
        for step_idx in range(start_step, self.max_steps):
            logger.debug("\n ======== Step %d ========\n", step_idx)
            
            # Check if goal is reached
            if self.world_model.goal_check(goals, state.env_state, )[0]:
                logger.info("Goal reached at step %d!", step_idx)
                break
            
            # Generate action
            steps = self.policy.get_actions(
                state,
                n_actions=1,
                query=verb_goals,
                query_idx=example_idx,
                from_phase=from_phase,
            )
            
            if not steps:
                logger.error("Policy returned no actions")
                break
            
            step = steps[0]
            
            # Handle errors
            if step.error:
                logger.error("Error in action generation: %s", step.error)
                break
            
            logger.debug("Selected action: %s", step.action)
            
            # Execute action via world model
            try:
                next_state, aux_data = self.world_model.step(
                    state=state,
                    action=step.action,
                    goals=goals,
                ) # tuple[EnvState, dict]
                
                assert isinstance(next_state, EnvState), "World model step must return EnvState"
                assert isinstance(aux_data, dict), "World model step must return aux_data as dict"
                assert 'goal_reached' in aux_data, "aux_data must contain 'goal_reached' key"
                
                # CRITICAL: Preserve and accumulate history across state transitions
                # The world model creates a new EnvState object, so we must explicitly
                # carry over the history from the previous state and add the new step
                
                # Ensure next_state has history initialized
                if next_state.history is None:
                    next_state.history = []
                
                # Copy all previous history to the new state
                # (world model might not preserve it)
                next_state.history = state.history.copy()
                
                # Add the current step to the accumulated history
                next_state.add_step(step)
                
                state = next_state
                logger.debug("New state:\n%s\n", state.env_state)
                logger.debug("Trajectory length: %d steps\n", len(state.history))
                
            except Exception as e:
                logger.error("Error in world model step: %s", e, exc_info=True)
                break
            
            # Save checkpoint if requested
            if checkpoint_path:
                state.save(checkpoint_path, verb_goals)
                logger.debug("\nCheckpoint saved to %s\n", checkpoint_path)
        
        # Final goal check
        if self.world_model.is_terminal(goals, state):
            logger.info("Successfully reached goal!")
        else:
            logger.warning("Max steps reached without achieving goal")
        
        return state

    def extract_action_sequence(self, state: EnvState) -> list:
        """
        Extract the sequence of actions from the final state.
        
        Args:
            state: Final EnvState after running the agent.
        
        Returns:
            List of action strings representing the action sequence.
        """
        if state.history:
            return [str(step.action) for step in state.history if step.action]
        return []
