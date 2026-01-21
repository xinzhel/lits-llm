from typing import List, Optional, Union
from ..base import Policy
from ...structures.env_grounded import EnvState, EnvStep, EnvAction
from ...prompts.prompt import PromptTemplate
import logging

logger = logging.getLogger(__name__)
class EnvGroundedPolicy(Policy):
    """
    Environment-grounded policy that generates valid actions for planning tasks.
    
    This policy uses an LLM to generate actions for environment-grounded tasks.
    It supports two modes of operation:
    
    1. **Finite action space** (e.g., BlocksWorld): 
       - `generate_all_actions` returns all valid actions
       - Validation uses exact match against the action list
       - LLM selects from the provided options
    
    2. **Infinite action space** (e.g., Crosswords):
       - `generate_all_actions` is None (no enumerable actions)
       - `validate_action` validates LLM-generated actions
       - LLM generates actions freely, validated post-hoc
    
    Action Space Contract:
        The Transition class can define either or both:
        
        - `generate_all_actions(env_state) -> List[str]`: Returns valid action strings.
          Used to populate {actions} in the prompt. If None, {actions} is empty.
        
        - `validate_action(env_state, action) -> bool`: Validates LLM-generated action.
          If None and generate_all_actions is provided, uses exact match.
    
    Args:
        base_model: Language model for action selection. If None, returns all valid actions.
        generate_all_actions: Optional callable that takes env_state (str) and returns list of
            valid action strings. If None, the prompt won't include action options.
        task_prompt_spec: Optional system prompt for chat-style LLM calls.
        usr_prompt_spec: PromptTemplate with placeholders: {init_state}, {goals}, {actions}.
        validate_action: Optional callable(env_state, action) -> bool that validates
            LLM-generated actions. If None, uses exact match against generate_all_actions().
        goal_reward_default: Default reward for non-terminal states (default: 0.0).
        goal_reached_reward: Reward when goal is reached (default: 100).
        **kwargs: Optional Policy parameters (n_actions, temperature, top_k, top_p, etc.).
    
    Example (Finite action space - BlocksWorld):
        >>> policy = EnvGroundedPolicy(
        ...     base_model=model,
        ...     generate_all_actions=BlocksWorldTransition.generate_actions,
        ...     usr_prompt_spec=prompt_template,
        ... )
    
    Example (Infinite action space - Crosswords):
        >>> policy = EnvGroundedPolicy(
        ...     base_model=model,
        ...     validate_action=CrosswordsTransition.validate_action,
        ...     usr_prompt_spec=prompt_template,
        ... )
    """
    
    # Interface category for this policy type
    TASK_TYPE: str = "env_grounded"
    
    def __init__(
        self,
        base_model,  # Required parameter from parent
        generate_all_actions=None,  # Optional: function to generate action hints/options
        task_prompt_spec: Union[PromptTemplate, str] = None,  # System prompt for chat-style
        usr_prompt_spec: Union[PromptTemplate, str] = None,  # PromptTemplate or string
        validate_action = None,  # Optional: callable(env_state, action) -> bool
        goal_reward_default: float = 0.,  # Subclass-specific parameter
        goal_reached_reward: float = 100,  # Subclass-specific parameter
        **kwargs  # Optional parent parameters (n_actions, temperature, top_k, top_p, etc.)
    ) -> None:
        """
        Initialize the environment-grounded policy.
        
        Args:
            base_model: Language model for action selection. Pass None to return all valid actions.
            generate_all_actions: Optional function(env_state: str) -> List[str] that returns
                action strings (or placeholder hints) for the given environment state.
                If None, the prompt will not include action options (for infinite action spaces).
            task_prompt_spec: Optional system prompt for chat-style LLM calls. If provided,
                enables chat API with system message. Can be a string or PromptTemplate.
            usr_prompt_spec: PromptTemplate with placeholders: {init_state}, {goals}, {actions}.
                Can also be a string template for backward compatibility (will be converted).
            validate_action: Optional function(env_state: str, action: str) -> bool that
                validates LLM-generated actions. If None and generate_all_actions is provided,
                uses exact match against generate_all_actions() output.
            goal_reward_default: Reward for non-terminal states (default: 0.0).
            goal_reached_reward: Reward when goal state is reached (default: 100).
            **kwargs: Optional parameters passed to parent Policy class (n_actions, temperature,
                top_k, top_p, max_steps, etc.).
        """
        super().__init__(
            base_model=base_model,
            task_prompt_spec=task_prompt_spec,
            usr_prompt_spec=usr_prompt_spec,
            **kwargs
        )
        self.generate_all_actions = generate_all_actions
        self.validate_action = validate_action
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        # Note: Domain-specific prompt validation removed to enable component reusability
        # across different env_grounded tasks (BlocksWorld, Crosswords, etc.)

    def _create_error_steps(self, n_actions: int, error_msg: str) -> List[EnvStep]:
        """Create EnvStep error steps for EnvGroundedPolicy."""
        return [EnvStep(action=EnvAction(""), error=error_msg) for _ in range(n_actions)]

    def _build_system_prompt(self) -> str:
        """Build system prompt for chat-style LLM calls.
        
        Returns the task_prompt_spec if provided, enabling chat API with system message.
        If task_prompt_spec is None, returns empty string (completion-style).
        
        Returns:
            System prompt string, or empty string if not using chat-style.
        """
        if self.task_prompt_spec is None:
            return ""
        
        # Handle PromptTemplate or string
        if isinstance(self.task_prompt_spec, PromptTemplate):
            # For system prompts, typically no variables needed
            # But if variables are present, they should be filled by caller
            return self.task_prompt_spec.template
        return str(self.task_prompt_spec)
    
    def _is_valid_action(self, env_state: str, action: str, valid_actions: List[str]) -> bool:
        """Check if an action is valid.
        
        Uses validate_action if provided, otherwise falls back to exact match.
        
        Args:
            env_state: Current environment state string
            action: Action string to validate
            valid_actions: List of valid actions from generate_all_actions() (may be empty)
        
        Returns:
            True if action is valid, False otherwise
        """
        if self.validate_action is not None:
            return self.validate_action(env_state, action)
        return action in valid_actions

    def _build_prompt(self, env_state: str, query: str, valid_actions: List[str]) -> str:
        """Build the prompt for action generation.
        
        Args:
            env_state: Current environment state string
            query: Goal description or query context
            valid_actions: List of valid actions (may be empty for infinite action spaces)
        
        Returns:
            Formatted prompt string
        """
        options = '\t'+'\n\t'.join(valid_actions) if valid_actions else ''
        
        if isinstance(self.usr_prompt_spec, PromptTemplate):
            return self.usr_prompt_spec.format(
                init_state=env_state,
                goals=query,
                actions=options
            )
        # Backward compatibility: support string templates with <placeholder> syntax
        return self.usr_prompt_spec.replace("<init_state>", env_state)\
                    .replace("<goals>", query).replace("<action>", options)

    def _parse_llm_response(self, response: str, valid_actions: List[str]) -> str:
        """Parse LLM response, handling numeric index responses.
        
        Args:
            response: Raw LLM response string
            valid_actions: List of valid actions for index lookup
        
        Returns:
            Parsed action string
        """
        action = response.strip()
        
        # Try to parse as index if LLM returns a number and we have valid_actions
        if action.isdigit() and valid_actions:
            idx = int(action)
            if 1 <= idx <= len(valid_actions):
                action = valid_actions[idx - 1]  # 1-indexed
                logger.debug(f"Parsed numeric response as 1-indexed action: {action}")
            elif 0 <= idx < len(valid_actions):
                action = valid_actions[idx]  # 0-indexed fallback
                logger.debug(f"Parsed numeric response as 0-indexed action: {action}")
        
        return action

    def _handle_duplicate(self, action: str, selected_actions: List[str], valid_actions: List[str]) -> str:
        """Handle duplicate action selection by finding an alternative.
        
        Args:
            action: The action that may be a duplicate
            selected_actions: Actions already selected
            valid_actions: All valid actions available
        
        Returns:
            Original action or an alternative if duplicate and alternatives exist
        """
        if action not in selected_actions:
            return action
        
        if valid_actions:
            alternatives = [a for a in valid_actions if a not in selected_actions]
            if alternatives:
                return alternatives[0]
            logger.warning("All valid actions have been selected, allowing duplicates.")
        
        return action

    def _generate_action_with_retry(
        self,
        env_state: str,
        query: str,
        valid_actions: List[str],
        selected_actions: List[str],
        base_temperature: float,
        allow_duplicates: bool,
        max_retries: int = 3,
        temperature_increment: float = 0.2
    ) -> EnvStep:
        """Generate a single valid action with retry and temperature escalation on duplicates.
        
        When a duplicate action is detected (and allow_duplicates=False), this method
        retries with progressively higher temperature to encourage diverse generation.
        This is particularly useful for infinite action spaces (e.g., Crosswords) where
        there's no valid_actions list to fall back to.
        
        Fallback behavior when max_retries exhausted:
        - Finite action space (valid_actions non-empty): Returns first unselected valid action
        - Infinite action space (valid_actions empty): Returns last attempted action with error
        
        Args:
            env_state: Current environment state string
            query: Goal description or query context
            valid_actions: List of valid actions (may be empty for infinite action spaces)
            selected_actions: Actions already selected (for duplicate detection)
            base_temperature: Initial sampling temperature
            allow_duplicates: Whether to allow duplicate actions
            max_retries: Maximum retry attempts for duplicates (default: 3)
            temperature_increment: Temperature increase per retry (default: 0.2)
        
        Returns:
            EnvStep with valid action, or error step if all retries exhausted
        """
        prompt = self._build_prompt(env_state, query, valid_actions)
        temperature = base_temperature
        last_action = None  # Track last action (invalid or duplicate)
        
        for attempt in range(1, max_retries + 1):
            response = self._call_model(prompt, temperature=temperature).text
            action = self._parse_llm_response(response, valid_actions)
            last_action = action  # Always track the last attempted action
            
            # Check validity first
            if not self._is_valid_action(env_state, action, valid_actions):
                logger.warning(f"Invalid action (attempt {attempt}/{max_retries}): '{action}'")
                temperature = min(temperature + temperature_increment, 1.0)
                continue
            
            # Check for duplicates (only if not allowing duplicates)
            if not allow_duplicates and action in selected_actions:
                logger.warning(
                    f"Duplicate action detected (attempt {attempt}/{max_retries}): '{action}'. "
                    f"Retrying with temperature={temperature + temperature_increment:.2f}"
                )
                temperature = min(temperature + temperature_increment, 1.0)
                continue
            
            # Valid and non-duplicate action found
            return EnvStep(action=EnvAction(action))
        
        # All retries exhausted - create fallback step with error
        logger.error(f"Max retries ({max_retries}) exceeded for action generation.")
        
        # Finite action space: fall back to first unselected valid action
        if valid_actions:
            unselected = [a for a in valid_actions if a not in selected_actions]
            fallback_action = unselected[0] if unselected else valid_actions[0]
            return EnvStep(
                action=EnvAction(fallback_action),
                error=f"LLM failed after {max_retries} retries; using fallback action"
            )
        
        # Infinite action space: return last attempted action with terminal error
        return EnvStep(
            action=EnvAction(last_action) if last_action else EnvAction(""),
            error=f"Validation failed after {max_retries} retries",
            terminate=True  # Signal agent to stop this trajectory
        )

    def _generate_single_action(
        self,
        env_state: str,
        query: str,
        valid_actions: List[str],
        selected_actions: List[str],
        temperature: float,
        allow_duplicates: bool,
        max_retries: int = 3,
        temperature_increment: float = 0.2
    ) -> EnvStep:
        """Generate a single valid action using the LLM.
        
        This is a convenience wrapper around _generate_action_with_retry.
        For infinite action spaces, uses temperature escalation to avoid duplicates.
        
        Args:
            env_state: Current environment state string
            query: Goal description or query context
            valid_actions: List of valid actions (may be empty)
            selected_actions: Actions already selected (for duplicate handling)
            temperature: Sampling temperature
            allow_duplicates: Whether to allow duplicate actions
            max_retries: Maximum retry attempts for invalid/duplicate actions
            temperature_increment: Temperature increase per retry (default: 0.2)
        
        Returns:
            EnvStep with valid action or error
        """
        return self._generate_action_with_retry(
            env_state=env_state,
            query=query,
            valid_actions=valid_actions,
            selected_actions=selected_actions,
            base_temperature=temperature,
            allow_duplicates=allow_duplicates,
            max_retries=max_retries,
            temperature_increment=temperature_increment
        )

    def _get_actions(
        self,
        state: EnvState,
        n_actions: int,
        temperature: float,
        query: Optional[str] = None,
        from_phase: str = '',
        **kwargs
    ) -> List[EnvStep]:
        """Generate n_actions valid actions for the given environment state.
        
        Args:
            state: Current environment state (EnvState with env_state string).
            n_actions: Number of actions to generate.
            temperature: Sampling temperature for LLM generation.
            query: Optional goal description or query context.
            from_phase: Description of current search phase (for logging).
            **kwargs: Additional arguments (allow_duplicates, etc.).
        
        Returns:
            List of EnvStep objects with actions or errors.
        """
        allow_duplicates = kwargs.get("allow_duplicates", False)
        current_env_state = state.env_state
        logger.debug(f"Current env_state (len={len(state)} steps):\n{current_env_state[:200]}...")
        
        # Get valid actions if generate_all_actions is provided
        valid_actions = []
        if self.generate_all_actions is not None:
            valid_actions = self.generate_all_actions(current_env_state)
        
        # No base_model - return all valid actions
        if not self.base_model:
            return [EnvStep(action=EnvAction(a)) for a in valid_actions]
        
        # Generate n_actions using LLM
        steps = []
        for _ in range(n_actions):
            selected_actions = [s.action.action_str for s in steps if s.action]
            step = self._generate_single_action(
                env_state=current_env_state,
                query=query,
                valid_actions=valid_actions,
                selected_actions=selected_actions,
                temperature=temperature,
                allow_duplicates=allow_duplicates
            )
            steps.append(step)
        
        return steps