from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Tuple, Optional
from ..structures import StateT, ActionT, StepT, Step
from ..lm.base import DETERMINISTIC_TEMPERATURE
import logging
logger = logging.getLogger(__name__)

class Transition(ABC, Generic[StateT, ActionT]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init_state(self) -> StateT: ...

    @abstractmethod
    def step(self, example: str, state: StateT, action: ActionT) -> Union[StateT, Tuple[StateT, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool: ...


class LlmTransition(Transition, Generic[StateT, ActionT]):
    """
    Base class for LLM-based transitions that use prompts.
    
    This class provides prompt management for transitions that use LLMs
    to generate or validate state transitions.
    
    Args:
        base_model: The LLM model to use for generation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_type: Task type identifier (e.g., 'math_qa', 'tool_use') for loading
            task-specific prompts from the registry
        usr_prompt_spec: User message specification. Used to construct the user message
            content. Alternative to task_prompt_spec for different prompt injection needs.
        **kwargs: Additional arguments passed to parent class
    
    Note:
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
        - Priority: task_prompt_spec > usr_prompt_spec > registry
    """
    def __init__(
        self,
        base_model,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_type: Optional[str] = None,
        usr_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.task_type = task_type
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        if task_prompt_spec is None:
            task_prompt_spec = PromptRegistry.get('transition', agent_name, task_type)
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('transition', agent_name, task_type)
        
        # Store prompts
        self.task_prompt_spec = task_prompt_spec
        self.usr_prompt_spec = usr_prompt_spec
            
    def _get_agent_name(self) -> str:
        """
        Infer agent name from class name.
        
        Examples:
            RAPTransition -> 'rap'
            BlocksWorldTransition -> 'blocksworld'
        """
        class_name = self.__class__.__name__
        # Remove 'Transition' suffix
        if class_name.endswith('Transition'):
            class_name = class_name[:-len('Transition')]
        # Convert to lowercase
        return class_name.lower()
    

    
class Policy(ABC, Generic[StateT, ActionT]):
    """
    Abstract base class for policy implementations. This class provides the framework for generating actions given a state.
    
    Args:
        base_model: The LLM model to use for action generation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_type: Task type identifier (e.g., 'math_qa', 'tool_use') for loading
            task-specific prompts from the registry
        usr_prompt_spec: User message specification. Used to construct the user message
            content. Alternative to task_prompt_spec for different prompt injection needs.
        n_actions: Number of actions to generate per policy call
        max_length: Maximum total sequence length for generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        reward_alpha: Weight for reward in action selection
        reward_confidence_default: Default confidence when reward is unavailable
        depth_limit: Maximum depth for tree search
        force_terminating_on_depth_limit: Force termination at depth limit
    
    Note:
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
    
    ## Guide on Subclass Implementation:
    Subclass `__init__` methods should only explicitly declare required parent parameters (e.g., `base_model`, `task_prompt_spec`) 
    and use `**kwargs` for optional ones, reducing redundancy and preventing default value mismatches. Same for `_get_actions`
    
    Example:
    ```
    class BWPolicy(Policy):
        def __init__(
            self,
            base_model,  # Required parameter from parent
            task_prompt_spec: str,  # Required parameter from parent
            goal_reward_default: float = 0.,  # Subclass-specific parameter
            goal_reached_reward: float = 100,  # Subclass-specific parameter
            **kwargs  # Optional parent parameters (n_actions, temperature, top_k, top_p, etc.)
        ) -> None:
            super().__init__(
                base_model=base_model,
                task_prompt_spec=task_prompt_spec,
                **kwargs
            )
            self.goal_reward_default = goal_reward_default
            self.goal_reached_reward = goal_reached_reward

        def _get_actions(
            self,
            state: BWState,
            n_actions: int,
            temperature: float,
            **kwargs  # Ignores unused parameters like example, critic, etc.
        ) -> List[BWAction]:
            blocks_state = state.blocks_state
            return generate_all_actions(blocks_state)

    ```
    
    """
    def __init__(
        self,
        base_model,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_type: Optional[str] = None,
        usr_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        n_actions: int = 4,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        reward_alpha: float = 0.5,
        reward_confidence_default: float = 0.8,
        depth_limit: int = 5,
        force_terminating_on_depth_limit: bool = True
    ) -> None:
        super().__init__()
        # Model configuration
        self.base_model = base_model
        self.max_length= max_length
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Policy configuration
        self.n_actions = n_actions
        self.task_type = task_type
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        if task_prompt_spec is None:
            task_prompt_spec = PromptRegistry.get('policy', agent_name, task_type)
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('policy', agent_name, task_type)
        
        # Store prompts
        self.task_prompt_spec = task_prompt_spec 
        self.usr_prompt_spec = usr_prompt_spec
        
        # Tree search configuration
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
    
    def _get_agent_name(self) -> str:
        """
        Infer agent name from class name.
        
        Examples:
            RAPPolicy -> 'rap'
            ReStPolicy -> 'rest'
            ToolUsePolicy -> 'tool_use'
        """
        class_name = self.__class__.__name__
        # Remove 'Policy' suffix
        if class_name.endswith('Policy'):
            class_name = class_name[:-len('Policy')]
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name
        
    def get_actions(
        self,
        state: StateT,
        query: Optional[str] = None,
        n_actions: Optional[int] = None,
        query_idx: Optional[int] = None,
        critic: Optional[str] = None,
        from_phase: str = "",
        *args,
        **kwargs
    ) -> List[StepT]:
        """
        Generate actions for the given state.
        
        This is a robust wrapper that handles:
        - Depth limit detection and temperature adjustment
        - Exception handling with error step generation
        - Validation of outputs
        - Logging

        Args:
            state: Current state or trajectory to condition the policy.
            query: Optional context or example (not needed for all tasks).
            n_actions: Number of actions to generate. If None, uses self.n_actions
                      or 1 if at depth limit.
            query_idx: Optional index for logging or batching.
            critic: Optional critic for action evaluation.
            from_phase: Description of the current algorithm phase.
            *args, **kwargs: Additional arguments passed to _get_actions.

        Return:
            List of Step objects with length exactly n_actions.
        """

        # Determine if we're at the depth limit
        at_depth_limit = self._is_at_depth_limit(state)
        
        # Set number of actions and temperature
        n_actions = self._determine_n_actions(n_actions, at_depth_limit)
        temperature = self._determine_temperature(at_depth_limit)
        
        # Log the policy call
        self._log_policy_call_start(state, n_actions)
        
        
        # Generate actions with error handling
        try:
            outputs = self._get_actions(
                state=state,
                n_actions=n_actions,
                temperature=temperature,
                query=query,
                at_depth_limit=at_depth_limit,
                query_idx=query_idx,
                critic=critic,
                from_phase=from_phase,
                *args,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in _get_actions: {e}", exc_info=True)
            outputs = self._create_error_steps(n_actions, str(e))

        # Validate outputs
        if len(outputs) != n_actions:
            logging.warning(f"Expected {n_actions} actions, but got {len(outputs)}")
        assert all(isinstance(output, Step) for output in outputs), "All outputs must be instances of Step or its subclasses"
        
        # Log the results
        self._log_policy_call_end(outputs, n_actions)

        return outputs
    
    def _create_error_steps(self, n_actions: int, error_msg: str) -> List[StepT]:
        """Create error steps when _get_actions fails."""
        # TODO: Do not put all the n_actions as error steps, consider partial results. To do this, may need to put for loop here instead of in _get_actions and define a _get_action method.
        return [Step(error=error_msg) for _ in range(n_actions)]
    
    def _is_at_depth_limit(self, state: StateT) -> bool:
        """Check if the state has reached the depth limit."""
        if not self.force_terminating_on_depth_limit:
            return False
        return len(state) + 1 >= self.depth_limit
    
    def _determine_n_actions(
        self,
        n_actions: Optional[int],
        at_depth_limit: bool
    ) -> int:
        """Determine the number of actions to generate."""
        if n_actions is not None:
            return n_actions
        return 1 if at_depth_limit else self.n_actions
    
    def _determine_temperature(self, at_depth_limit: bool) -> float:
        """Determine the temperature for generation."""
        return DETERMINISTIC_TEMPERATURE if at_depth_limit else self.temperature
        
    def _log_policy_call_start(self, state: StateT, n_actions: int) -> None:
        """Log the start of a policy call."""
        logger.debug(f"\n{'='*70}")
        logger.debug(f">>> Policy Call: Generating {n_actions} actions (BEGIN)")
        logger.debug(f"{'='*70}")
        logger.debug(f"State: {state}")
    
    def _log_policy_call_end(self, outputs: List[StepT], n_actions: int) -> None:
        """Log the end of a policy call with results."""
        logger.debug(f"\nGenerated Actions:")
        for idx, output in enumerate(outputs):
            logger.debug(f"  [{idx}] {output}")
        logger.debug(f"{'='*70}")
        logger.debug(f">>> Policy Call: Generating {n_actions} actions (END)")
        logger.debug(f"{'='*70}\n")

    
    @abstractmethod
    def _get_actions(
        self,
        state: StateT,
        n_actions: int,
        temperature: float,
        **kwargs 
    ) -> List[StepT]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_actions method"
        )

class RewardModel(ABC, Generic[StateT, ActionT]):
    """
    Abstract base class for reward model implementations.
    
    Reward models evaluate the quality of actions or states in tree search.
    They can evaluate actions before execution (fast_reward) or after (reward).
    
    Args:
        base_model: The LLM model to use for reward evaluation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_type: Task type identifier (e.g., 'math_qa', 'tool_use') for loading
            task-specific prompts from the registry
        usr_prompt_spec: User message specification. Used to construct the user message
            content. Alternative to task_prompt_spec for different prompt injection needs.
        max_length: Maximum total sequence length for generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        reward_alpha: Weight for combining different reward signals
        reward_confidence_default: Default confidence when evaluation is uncertain
    
    Note:
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
        - Priority: task_prompt_spec > usr_prompt_spec > registry
    """
    def __init__(
        self,
        base_model,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_type: Optional[str] = None,
        usr_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        max_length=None,
        max_new_tokens=None,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        reward_alpha=0.5,
        reward_confidence_default=0.8
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.task_type = task_type
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        if task_prompt_spec is None:
            task_prompt_spec = PromptRegistry.get('reward', agent_name, task_type)
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('reward', agent_name, task_type)
        
        # Store prompts
        self.task_prompt_spec = task_prompt_spec
        self.usr_prompt_spec = usr_prompt_spec
        self.max_length= max_length
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # for evaluator
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        
    def fast_reward(self, example, example_idx, state, action, from_phase="") -> tuple[float, dict]:
        """
        Generate a reward for an action without executing it.
        
        This method evaluates the potential usefulness/quality of an action based only on
        the current state and the proposed action, without actually executing the action
        to observe its outcome. This is useful for:
        
        - Tasks where action execution is expensive (e.g., env_grounded tasks)
        - Reasoning tasks where we can evaluate thought quality before execution (e.g., math_qa with RAP)
        - Pruning unpromising actions early in tree search
        
        Args:
            example: The problem/question being solved
            example_idx: Index of the example (for logging)
            state: Current state before action execution
            action: Proposed action to evaluate
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            Tuple of (reward, auxiliary_dict) where:
            - reward: Float score indicating action quality
            - auxiliary_dict: Additional metrics (e.g., {'r_useful': probability})
        
        Note:
            This differs from `reward()` which evaluates after action execution.
        """
        logger.debug("\n>>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (BEGIN) <<<<<<<<<")

        fast_reward = self._fast_reward(example, example_idx, state, action, from_phase=from_phase)

        fast_reward = self.calculate_reward(fast_reward)

        logger.debug(f"fast_reward: {fast_reward}")
        logger.debug(">>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (END) <<<<<<<<<\n")

        return fast_reward, {'r_useful': float(useful_prob)}

    def _get_agent_name(self) -> str:
        """
        Infer agent name from class name.
        
        Examples:
            RapPRM -> 'rap'
            GenerativePRM -> 'generative'
            SelfConsistencyRM -> 'self_consistency'
        """
        class_name = self.__class__.__name__
        # Remove 'PRM' or 'RM' suffix
        if class_name.endswith('PRM'):
            class_name = class_name[:-3]
        elif class_name.endswith('RM'):
            class_name = class_name[:-2]
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name
    
    @abstractmethod
    def _fast_reward(self, example, example_idx, state, action, from_phase="") -> float:
        raise NotImplementedError("_fast_reward is not implemented for RewardModel")
    
    @abstractmethod
    def calculate_reward(self, fast_reward: float) -> float:
        raise NotImplementedError("calculate_reward is not implemented for RewardModel")

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
