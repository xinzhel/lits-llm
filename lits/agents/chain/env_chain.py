import logging
from pathlib import Path
from typing import Optional, Callable, Optional, List
from dataclasses import dataclass, asdict
from ..base import BaseConfig
from .base import ChainAgent, ChainConfig
from ...components.policy.env_grounded import EnvGroundedPolicy
from ...components.base import Transition
from ...structures.env_grounded import EnvState, EnvStep

logger = logging.getLogger(__name__)

@dataclass
class EnvChainConfig(ChainConfig):
    """
    Configuration for environment-grounded chain agent.
    """
    max_steps: int = 30  # Override default

class EnvChain(ChainAgent[EnvState]):
    """
    Implements a chain-like invocation of environment-grounded policy.
    """
    
    def __init__(
        self,
        policy: EnvGroundedPolicy,
        world_model: Transition,
        max_steps: int = 10,
    ):
        super().__init__(max_steps=max_steps)
        self.policy = policy
        self.world_model = world_model

    def run(
        self,
        query_or_goals: str,
        init_state_str: str,
        query_idx: Optional[int] = None,
        from_phase: str = "",
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        override: bool = False
    ) -> EnvState:
        """
        Run the environment chain to generate a sequence of actions.
        
        Args:
            init_state_str: the environment state.
            query_idx: Optional index for logging/tracking.
            from_phase: Description of current phase (e.g., 'planning', 'execution').
            checkpoint_path: Optional path to save/load checkpoints.
            override: If True, ignore existing checkpoints and start fresh.
        
        Returns:
            Final EnvState after goal is reached or max steps exceeded.
        
        The returned state contains the full action history and can be used to
        extract the action sequence or evaluate the solution.
        """
        logger.debug("Starting EnvChain with goals:\n%s\n", query_or_goals)
        logger.debug("Initial state string:\n%s\n", init_state_str)
        
        # Validate inputs
        assert isinstance(init_state_str, str) and len(init_state_str) > 0, "Initial state string must be a non-empty string."
        assert isinstance(query_or_goals, str) and len(query_or_goals) > 0, "Verb goals must be a non-empty string."
            
        # Initialize or resume state
        checkpoint_path = self.get_checkpoint_path(checkpoint_dir, query_idx, checkpoint_path)
        
        state = None
        if checkpoint_path and not override:
            state = self.resume_state(checkpoint_path, EnvState)
            
        if state is None:
            state = self.world_model.init_state(init_state_str)
        else:
            # If we resumed, we might want to check if we are already done or where we are
            pass
        
        # Ensure history is initialized
        if state.history is None:
            state.history = []
        
        start_step = len(state)
        for step_idx in range(start_step, self.max_steps):
            logger.debug("\n ======== Step %d ========\n", step_idx)
            
            # Check if goal is reached
            if self.world_model.goal_check(query_or_goals, state.env_state, )[0]:
                logger.info("Goal reached at step %d!", step_idx)
                break
            
            # Generate action
            steps = self.policy.get_actions(
                state,
                n_actions=1,
                query=query_or_goals,
                query_idx=query_idx,
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
                    step_or_action=step.action,
                    query_or_goals=query_or_goals,
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
                # Update step with next state information
                step.next_state = next_state.env_state
                next_state.add_step(step)
                
                state = next_state
                logger.debug("New state:\n%s\n", state.env_state)
                logger.debug("Trajectory length: %d steps\n", len(state.history))
                
            except Exception as e:
                logger.error("Error in world model step: %s", e, exc_info=True)
                break
            
            # Save checkpoint if requested
            if checkpoint_path:
                state.save(checkpoint_path, query_or_goals)
                logger.debug("\nCheckpoint saved to %s\n", checkpoint_path)
        
        # Final goal check
        if self.world_model.is_terminal(query_or_goals, state):
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
