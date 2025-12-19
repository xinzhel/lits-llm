from typing import List, Optional
from ..base import Policy
from ...structures.env_grounded import EnvState, EnvStep, EnvAction
import logging

logger = logging.getLogger(__name__)
class EnvGroundedPolicy(Policy):
    """
    Environment-grounded policy that generates valid actions for planning tasks.
    
    This policy uses an LLM to select actions from a set of valid actions generated
    by the environment. It ensures all generated actions are valid by validating
    against the environment's action space and retrying invalid generations.
    
    Typical Use Case:
        Planning tasks where actions must be valid in the current environment state
        (e.g., BlocksWorld, logistics, robotics). The policy prompts an LLM to choose
        from valid actions, ensuring feasibility.
    
    Args:
        base_model: Language model for action selection. If None, returns all valid actions.
        usr_prompt_spec: Dict containing prompt templates. Must include "policy" key with
            placeholders: <init_state>, <goals>, <action>.
        generate_all_actions: Callable that takes env_state (str) and returns list of
            valid action strings for that state.
        goal_reward_default: Default reward for non-terminal states (default: 0.0).
        goal_reached_reward: Reward when goal is reached (default: 100).
        **kwargs: Optional Policy parameters (n_actions, temperature, top_k, top_p, etc.).
    
    Example:
        >>> # Define action generator for BlocksWorld
        >>> def generate_all_actions(blocks_state):
        ...     # Parse state and return valid actions like ["unstack A from B", ...]
        ...     return valid_actions
        >>> 
        >>> # Define prompt template
        >>> prompts = {
        ...     "policy": "State: <init_state>\\nGoals: <goals>\\nValid actions:\\n<action>\\nSelect one:"
        ... }
        >>> 
        >>> # Create policy
        >>> policy = EnvGroundedPolicy(
        ...     base_model=base_model,
        ...     usr_prompt_spec=None,
        ...     generate_all_actions=generate_all_actions,
        ...     goal_reached_reward=100,
        ...     goal_reward_default=0.0
        ... )
        >>> 
        >>> # Initialize environment state
        >>> world_model = ...
        >>> state = world_model.init_state(problem_instance)
        >>> 
        >>> # Generate actions
        >>> actions = policy.get_actions(
        ...     state,
        ...     n_actions=3,
        ...     query="stack A on B",
        ...     from_phase='expansion'
        ... )
        >>> # Returns: [EnvStep(action=StringAction(...), reward=0.0), ...]
    
    Notes:
        - If base_model is None, returns all valid actions without LLM selection.
        - Invalid LLM generations are retried until a valid action is selected.
        - All returned actions are guaranteed to be valid in the current state.
    """
    
    # Interface category for this policy type
    TASK_TYPE: str = "env_grounded"
    
    def __init__(
        self,
        base_model,  # Required parameter from parent
        generate_all_actions,  # Function to generate all valid actions
        usr_prompt_spec:str = None,  # Required parameter from parent
        goal_reward_default: float = 0.,  # Subclass-specific parameter
        goal_reached_reward: float = 100,  # Subclass-specific parameter
        **kwargs  # Optional parent parameters (n_actions, temperature, top_k, top_p, etc.)
    ) -> None:
        """
        Initialize the environment-grounded policy.
        
        Args:
            base_model: Language model for action selection. Pass None to return all valid actions.
            usr_prompt_spec: prompt template with placeholders: <init_state>, <goals>, <action>.
            generate_all_actions: Function(env_state: str) -> List[str] that returns valid
                action strings for the given environment state.
            goal_reward_default: Reward for non-terminal states (default: 0.0).
            goal_reached_reward: Reward when goal state is reached (default: 100).
            **kwargs: Optional parameters passed to parent Policy class (n_actions, temperature,
                top_k, top_p, max_steps, etc.).
        """
        super().__init__(
            base_model=base_model,
            usr_prompt_spec=usr_prompt_spec,
            **kwargs
        )
        self.generate_all_actions = generate_all_actions
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        # Verify policy prompts for blocksworld
        if self.task_name == "blocksworld":
            assert self.usr_prompt_spec.startswith("I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\nPick up a block\nUnstack a block"), \
                f"EnvGroundedPolicy usr_prompt_spec does not match expected BlocksWorld prompt. Got:\n{self.usr_prompt_spec}"


    def _create_error_steps(self, n_actions: int, error_msg: str) -> List[EnvStep]:
        """Create EnvStep error steps for EnvGroundedPolicy."""
        return [EnvStep(action=EnvAction(""), error=error_msg) for _ in range(n_actions)]

    def _build_system_prompt(self):
        """ Unused for EnvGroundedPolicy since completion (not chat) LLM is used. Just for compatibility with parent signature."""
        return ""
    
    def _get_actions(
        self,
        state: EnvState,
        n_actions: int,
        temperature: float,
        query: Optional[str] = None,
        from_phase: str = '',
        **kwargs  # Ignores unused parameters like example, critic, etc.
    ) -> List[EnvStep]:
        """
        Generate n_actions valid actions for the given environment state.
        
        This method:
        1. Generates all valid actions for the current state using generate_all_actions
        2. If base_model exists, prompts it to select actions from valid options
        3. Validates LLM outputs and retries until valid actions are selected
        4. Returns EnvStep objects with selected actions
        
        Args:
            state: Current environment state (EnvState with env_state string).
            n_actions: Number of actions to generate.
            temperature: Sampling temperature for LLM generation.
            query: Optional goal description or query context (e.g., "stack A on B").
            from_phase: Description of current search phase (e.g., 'expansion', 'simulation').
            **kwargs: Additional arguments (ignored, for compatibility with parent signature).
        
        Returns:
            List of EnvStep objects, each containing:
                - action: EnvAction (StringAction) with the selected action string
                - reward: 0.0 (default, actual rewards computed by world model)
                - error: None (unless action generation fails)
        
        Behavior:
            - If base_model is None: Returns all valid actions as EnvSteps
            - If base_model exists: Samples n_actions from valid actions using LLM
            - Invalid LLM outputs are rejected and regeneration is attempted
        
        Example:
            >>> state = EnvState(step_idx=0, env_state="on(A, B), on(B, table)", ...)
            >>> steps = policy._get_actions(state, n_actions=2, temperature=0.8, query="clear A")
            >>> # Returns: [EnvStep(action=StringAction("unstack A from B"), reward=0.0), ...]
        """
        from ..utils import create_role
        query_idx = kwargs.get("query_idx", None)
        valid_actions = self.generate_all_actions(state.env_state)
        actions_selected = []
        if self.base_model:
            for _ in range(n_actions):
                options = '\t'+'\n\t'.join(valid_actions)
                prompt = self.usr_prompt_spec.replace("<init_state>", state.env_state)\
                            .replace("<goals>", query).replace("<action>", options)
                
                valid_gen = False
                while not valid_gen:
                    role = create_role("policy", query_idx, from_phase)
                    gen_action = self.base_model(prompt, temperature=temperature, from_phase=from_phase, role=role).text.strip()
                    if gen_action in valid_actions:
                        # find another from valid actions if duplicated
                        if gen_action in actions_selected:
                            other_valid_actions = [a for a in valid_actions if a not in actions_selected] 
                            if other_valid_actions:
                                gen_action = other_valid_actions[0]
                            else:
                                logger.warning("All valid actions have been selected, allowing duplicates.")
    
                        actions_selected.append(gen_action)
                        valid_gen = True
        else:
            actions_selected = valid_actions
        
        steps = [EnvStep(action=EnvAction(action_str)) for action_str in actions_selected]
        return steps