from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, List, Union, Tuple, Optional, Callable
from ..structures import StateT, ActionT, StepT, Step
from ..lm.base import DETERMINISTIC_TEMPERATURE
from ..lm import OpenAIChatModel, BedrockChatModel, HfChatModel, HfModel, LanguageModel
import logging
import re
logger = logging.getLogger(__name__)

class Transition(ABC, Generic[StateT, ActionT]):
    """Base class for transition models (world models) in tree search.
    
    Transition models define how states evolve in response to actions.
    They are responsible for:
    1. Initializing the starting state via init_state()
    2. Computing the next state given current state and action via step()
    3. Determining if a state is terminal via is_terminal()
    
    init_state_kwargs Convention:
    -----------------------------
    The init_state() method receives kwargs from the dataset example.
    Different task types pass different fields:
    
    | Task Type         | Expected kwargs           | Description                          |
    |-------------------|---------------------------|--------------------------------------|
    | env_grounded      | init_state_str: str       | Initial environment state description|
    | language_grounded | (none)                    | Returns empty list                   |
    | tool_use          | (none)                    | Returns empty ToolUseState           |
    
    When implementing a custom Transition:
    - Extract only the kwargs you need using kwargs.get('key_name')
    - Ignore unknown kwargs for forward compatibility
    - Raise ValueError with helpful message if required kwargs are missing
    
    Example:
        class MyTransition(Transition):
            def init_state(self, **kwargs) -> MyState:
                # Extract what you need, ignore the rest
                init_str = kwargs.get('init_state_str')
                if init_str is None:
                    raise ValueError("MyTransition requires 'init_state_str' in init_state_kwargs")
                return MyState(init_str)
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init_state(self, **kwargs) -> StateT:
        """Initialize and return the initial state.
        
        This method is called by tree search algorithms (MCTS, BFS) at the start
        of each search. The kwargs come from the dataset example dict.
        
        Args:
            **kwargs: Example-specific data from the dataset. Contents depend on task type:
                      - env_grounded: expects 'init_state_str' (str) - initial state description
                      - language_grounded: no kwargs needed, returns empty trajectory
                      - tool_use: no kwargs needed, returns empty ToolUseState
                      Subclasses should extract what they need and ignore the rest.
        
        Returns:
            The initial state for tree search
            
        Raises:
            ValueError: If required kwargs are missing (implementation-specific)
        """
        ...

    @abstractmethod
    def step(self, state: StateT, step_or_action, *arg, **kwargs) -> Union[StateT, Tuple[StateT, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param step_or_action: Step or Action to execute. Policies return Steps which may
            contain actions, answers, or errors. Transitions should handle all cases.
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: StateT, *arg, **kwargs) -> bool: ...


class LlmTransition(Transition, Generic[StateT, ActionT]):
    """
    Base class for LLM-based transitions that use prompts.
    
    This class provides prompt management for transitions that use LLMs
    to generate or validate state transitions.
    
    Class Attributes:
        TASK_TYPE: Interface category for this transition (e.g., 'language_grounded', 'tool_use', 'env_grounded').
            Subclasses should override this to declare their interface category.
    
    Args:
        base_model: The LLM model to use for generation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_name: Task name identifier (e.g., 'gsm8k', 'blocksworld', 'mapeval-sql') for loading
            task-specific prompts from the registry. This is the prompt lookup key.
        usr_prompt_spec: User message specification. Used to construct the user message
            content. Alternative to task_prompt_spec for different prompt injection needs.
        **kwargs: Additional arguments passed to parent class
    
    Note:
        - TASK_TYPE: Interface category (set as class constant in subclasses)
        - task_name: Prompt lookup key (passed as parameter)
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
        - Priority: task_prompt_spec > usr_prompt_spec > registry
    """
    
    # Interface category for this transition type (subclasses should override)
    TASK_TYPE: str = None
    
    def __init__(
        self,
        base_model,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_name: Optional[str] = None,
        usr_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.task_name = task_name
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        # Get task_type from class attribute TASK_TYPE for fallback lookup
        task_type = getattr(self.__class__, 'TASK_TYPE', None)
        
        if task_prompt_spec is None:
            task_prompt_spec = PromptRegistry.get('transition', agent_name, task_name, task_type)
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('transition', agent_name, task_name, task_type)
        
        # Store prompts
        self.task_prompt_spec = task_prompt_spec
        self.usr_prompt_spec = usr_prompt_spec
        
        # Context attributes for _call_model() helper (set by step() or subclass methods)
        self._query_idx: Optional[int] = None
        self._from_phase: str = ""
            
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
    
    def _get_llm_role(self) -> str:
        """
        Return the LLM role prefix for this transition.
        
        This is used by _call_model() to construct the role string for inference logging.
        Subclasses can override this to use a different role prefix.
        
        Returns:
            str: The role prefix, default is "dynamics"
        """
        return "dynamics"
    
    def _call_model(self, prompt: str, **kwargs):
        """
        Call the base model with auto-constructed role from stored context.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `step()`) and calls `self.base_model()`. Subclasses can use this
        in `_step()` without manually passing query_idx or from_phase.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments passed to base_model (e.g., temperature, max_new_tokens)
        
        Returns:
            The model response object
        
        Example:
            ```python
            def _step(self, state, action, query_or_goals, **kwargs):
                prompt = self._build_prompt(state, action)
                response = self._call_model(prompt, temperature=0.0)
                return self._parse_response(response)
            ```
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        return self.base_model(prompt, role=role, **kwargs)
    
    def _batch_call_model(self, prompts: list, **kwargs):
        """
        Call the base model's batch_generate with auto-constructed role from stored context.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `step()`) and calls `self.base_model.batch_generate()`. Subclasses can use
        this in `_step()` for batch generation without manually passing query_idx or from_phase.
        
        Args:
            prompts: List of prompts to send to the model
            **kwargs: Additional arguments passed to batch_generate (e.g., temperature, max_new_tokens)
        
        Returns:
            List of model response strings
        
        Example:
            ```python
            def _step(self, state, action, query_or_goals, **kwargs):
                prompts = [self._build_prompt(state, action)] * n_samples
                outputs = self._batch_call_model(prompts, temperature=0.8)
                return self._aggregate_outputs(outputs)
            ```
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        return self.base_model.batch_generate(prompts, role=role, **kwargs)
    
    def _sample_binary_output(self, user_message: str, sample_size: int, target: str, contrast: str, role_prefix: Optional[str] = None, **kwargs):
        """
        Call the base model's sample_binary_output with auto-constructed role from stored context.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `step()` or `is_terminal()`) and calls `self.base_model.sample_binary_output()`.
        
        Args:
            user_message: The user message to send to the model
            sample_size: Number of samples to generate
            target: Target token (e.g., "yes", "complete")
            contrast: Contrast token (e.g., "no", "incomplete")
            role_prefix: Optional custom role prefix. If None, uses `_get_llm_role()`.
            **kwargs: Additional arguments passed to sample_binary_output
        
        Returns:
            Dict with counts for each token type
        
        Example:
            ```python
            def _is_terminal(self, state, query_or_goals, **kwargs):
                user_message = self._build_terminal_prompt(state, query_or_goals)
                answer_samples = self._sample_binary_output(user_message, sample_size=10, target="yes", contrast="no")
                return answer_samples['yes'] / 10 > 0.8
            ```
        """
        from .utils import create_role
        role_name = role_prefix if role_prefix is not None else self._get_llm_role()
        role = create_role(role_name, self._query_idx, self._from_phase)
        return self.base_model.sample_binary_output(user_message, sample_size=sample_size, target=target, contrast=contrast, role=role, **kwargs)
    
    def step(self, state: StateT, step_or_action, query_or_goals: str, query_idx: Optional[int] = None, from_phase: str = "", **kwargs) -> Union[StateT, Tuple[StateT, dict]]:
        """
        Execute a transition step. This is the public interface called by tree search algorithms.
        
        This method stores context for `_call_model()` helper and delegates to `_step()`.
        Subclasses should implement `_step()` instead of overriding this method.
        
        Args:
            state: The current state
            step_or_action: Step or Action to execute
            query_or_goals: The problem/question being solved
            query_idx: Index of the example (for logging)
            from_phase: Description of algorithm phase (for logging)
            **kwargs: Additional arguments passed to _step()
        
        Returns:
            The next state and optionally an auxiliary data dict
        """
        # Store context for _call_model() helper
        self._query_idx = query_idx
        self._from_phase = from_phase
        
        return self._step(state, step_or_action, query_or_goals, **kwargs)
    
    def _step(self, state: StateT, step_or_action, query_or_goals: str, **kwargs) -> Union[StateT, Tuple[StateT, dict]]:
        """
        Internal step implementation. Subclasses should override this method.
        
        This method is called by `step()` after context has been stored. Subclasses
        can use `self._call_model()` without passing query_idx or from_phase.
        
        Args:
            state: The current state
            step_or_action: Step or Action to execute
            query_or_goals: The problem/question being solved
            **kwargs: Additional arguments
        
        Returns:
            The next state and optionally an auxiliary data dict
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _step()")
    
    def is_terminal(self, state: StateT, query_or_goals: str, query_idx: Optional[int] = None, from_phase: str = "", **kwargs) -> bool:
        """
        Check if a state is terminal. This is the public interface called by tree search algorithms.
        
        This method stores context for `_call_model()` helper and delegates to `_is_terminal()`.
        Subclasses should implement `_is_terminal()` instead of overriding this method.
        
        Args:
            state: The current state
            query_or_goals: The problem/question being solved
            query_idx: Index of the example (for logging)
            from_phase: Description of algorithm phase (for logging)
            **kwargs: Additional arguments passed to _is_terminal()
        
        Returns:
            True if the state is terminal, False otherwise
        """
        # Store context for _call_model() helper
        self._query_idx = query_idx
        self._from_phase = from_phase
        
        return self._is_terminal(state, query_or_goals, **kwargs)
    
    def _is_terminal(self, state: StateT, query_or_goals: str, **kwargs) -> bool:
        """
        Internal terminal check implementation. Subclasses should override this method.
        
        This method is called by `is_terminal()` after context has been stored. Subclasses
        can use `self._call_model()` without passing query_idx or from_phase.
        
        Args:
            state: The current state
            query_or_goals: The problem/question being solved
            **kwargs: Additional arguments
        
        Returns:
            True if the state is terminal, False otherwise
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _is_terminal()")
    

    
class Policy(ABC, Generic[StateT, ActionT]):
    """
    Abstract base class for policy implementations. This class provides the framework for generating actions given a state.
    
    Class Attributes:
        TASK_TYPE: Interface category for this policy (e.g., 'language_grounded', 'tool_use', 'env_grounded').
            Subclasses should override this to declare their interface category.
    
    Args:
        base_model: The LLM model to use for action generation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_name: Task name identifier (e.g., 'gsm8k', 'blocksworld', 'mapeval-sql') for loading
            task-specific prompts from the registry. This is the prompt lookup key.
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
        max_steps: Maximum depth/steps for tree search
        force_terminating_on_depth_limit: Force termination at max_steps
    
    Note:
        - TASK_TYPE: Interface category (set as class constant in subclasses)
        - task_name: Prompt lookup key (passed as parameter)
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
    
    Dynamic Notes Injection:
        Policies support injecting dynamic notes from external sources (memory, database, files)
        into the system prompt. This is useful for:
        - Adding context from cross-trajectory memory
        - Including user preferences or past errors
        - Injecting task-specific hints or constraints
        
        Usage:
            ```python
            # Define a function that returns notes
            def get_memory_notes() -> List[str]:
                return memory_backend.get_relevant_memories()
            
            # Set the function on the policy
            policy.set_dynamic_notes_fn(get_memory_notes)
            
            # Notes will be automatically appended to system prompt during generation
            ```
        
        The notes are formatted as bullet points and appended to the system prompt:
            ```
            [Base system prompt]
            
            Additional Notes:
            * note1
            * note2
            * note3
            ```
    
    LLM Call Helper:
        Subclasses can use `_call_model(prompt, **kwargs)` to call the LLM without manually
        constructing the role string. The helper auto-constructs the role from stored context
        (`_query_idx`, `_from_phase`) set by `get_actions()`.
        
        Usage in subclass `_get_actions()`:
            ```python
            def _get_actions(self, state, n_actions, temperature, **kwargs):
                prompt = self._build_prompt(state)
                # No need to pass query_idx or from_phase - they're auto-injected
                response = self._call_model(prompt, temperature=temperature)
                return self._parse_response(response)
            ```
    
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
    
    # Interface category for this policy type (subclasses should override)
    TASK_TYPE: str = None
    
    def __init__(
        self,
        base_model,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_name: Optional[str] = None,
        usr_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        n_actions: int = 4,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        reward_alpha: float = 0.5,
        reward_confidence_default: float = 0.8,
        max_steps: int = 5,
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
        self.task_name = task_name
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        # Get task_type from class attribute TASK_TYPE for fallback lookup
        task_type = getattr(self.__class__, 'TASK_TYPE', None)
        
        logger.debug(f"Policy.__init__: agent_name={agent_name}, task_name={task_name}, task_type={task_type}")
        logger.debug(f"Policy.__init__: task_prompt_spec (before registry) type={type(task_prompt_spec)}, value={task_prompt_spec}")
        
        if task_prompt_spec is None:
            # Try task_name first, then fallback to TASK_TYPE
            task_prompt_spec = PromptRegistry.get('policy', agent_name, task_name, task_type)
            if task_prompt_spec is None:
                logger.warning(f"Policy.__init__: task_prompt_spec not found for task_name='{task_name}' or task_type='{task_type}'")
            else:
                logger.debug(f"Policy.__init__: task_prompt_spec loaded from registry, type={type(task_prompt_spec)}")
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('policy', agent_name, task_name, task_type)
            logger.debug(f"Policy.__init__: usr_prompt_spec loaded from registry, type={type(usr_prompt_spec)}, value={usr_prompt_spec}")
        
        # Store prompts
        self.task_prompt_spec = task_prompt_spec 
        self.usr_prompt_spec = usr_prompt_spec
        
        logger.debug(f"Policy.__init__: Final task_prompt_spec type={type(self.task_prompt_spec)}")
        logger.debug(f"Policy.__init__: Final usr_prompt_spec type={type(self.usr_prompt_spec)}")
        
        # Tree search configuration
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.max_steps = max_steps
        self.reward_alpha = reward_alpha
        
        # Dynamic notes injection callback
        self._dynamic_notes_fn: Optional[Callable[[], List[str]]] = None
        
        # Post-generation callback for action validation/processing
        self._post_generation_fn: Optional[Callable[[List[StepT], dict], None]] = None
        
        # LLM call callback for intercepting/logging/modifying LLM calls
        self._llm_call_fn: Optional[Callable[..., Any]] = None
        
        # Current memory context for LiTS-Mem integration (set per get_actions call)
        self._current_memory_context = None
        
        # Context attributes for _call_model() helper (set by get_actions())
        self._query_idx: Optional[int] = None
        self._from_phase: str = ""
    
    def _get_agent_name(self, first_word: bool = False) -> str:
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
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        if first_word:
            return name.split('_')[0]
        return name
    
    def _get_llm_role(self) -> str:
        """
        Return the LLM role prefix for this policy.
        
        This is used by _call_model() to construct the role string for inference logging.
        Subclasses can override this to use a different role prefix.
        
        Returns:
            str: The role prefix, default is "policy"
        """
        return "policy"
    
    def _call_model(self, prompt: str, **kwargs):
        """
        Call the base model with auto-constructed role from stored context.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `get_actions()`) and calls `self.base_model()`. Subclasses can use this
        in `_get_actions()` without manually passing query_idx or from_phase.
        
        If `_llm_call_fn` is set, it's called with (prompt, response, **call_context).
        The callback can return a modified response or None to keep the original.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments passed to base_model (e.g., temperature, max_new_tokens)
        
        Returns:
            The model response object (possibly modified by hook)
        
        Example:
            ```python
            def _get_actions(self, state, n_actions, temperature, **kwargs):
                prompt = self._build_prompt(state)
                response = self._call_model(prompt, temperature=temperature)
                return self._parse_response(response)
            ```
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        response = self.base_model(prompt, role=role, **kwargs)
        
        # Invoke callback if set
        if self._llm_call_fn is not None:
            fn_result = self._llm_call_fn(
                prompt=prompt,
                response=response,
                query_idx=self._query_idx,
                from_phase=self._from_phase,
                **kwargs
            )
            # If callback returns something, use it as the new response
            if fn_result is not None:
                response = fn_result
        
        return response
    
    def set_dynamic_notes_fn(self, fn: Callable[[], List[str]]) -> None:
        """
        Set a callback function to retrieve dynamic notes for system prompt injection.
        
        The callback function should return a list of strings that will be formatted
        as bullet points and appended to the system prompt during construction.
        
        Args:
            fn: A callable that takes no arguments and returns List[str] of notes
        
        Example:
            ```python
            def get_memory_notes() -> List[str]:
                return [
                    "User prefers concise answers",
                    "Previous error: division by zero in step 3",
                    "Context: working on algebra problems"
                ]
            
            policy.set_dynamic_notes_fn(get_memory_notes)
            ```
        """
        self._dynamic_notes_fn = fn
        logger.debug(f"Dynamic notes function set for {self.__class__.__name__}")
    
    def set_post_generation_fn(self, fn: Callable[[List[StepT], dict], None]) -> None:
        """
        Set a callback function to process/validate actions after generation.
        
        This callback is invoked after actions are generated but before they are
        returned. It can be used for validation, logging, or side effects like
        saving validation results.
        
        Args:
            fn: A callable that takes (steps, context) where:
                - steps: List[StepT] - Generated steps to process
                - context: dict - Context information (query, query_idx, state, etc.)
        
        Example:
            ```python
            def validate_sql(steps: List[ToolUseStep], context: dict):
                for step in steps:
                    result = validator.validate(
                        step,
                        query_idx=context.get('query_idx'),
                        policy_model_name=context.get('policy_model_name'),
                        task_name=context.get('task_name')
                    )
                    if result and not result['is_valid']:
                        logger.warning(f"Invalid SQL: {result['issue']}")
            
            policy.set_post_generation_fn(validate_sql)
            ```
        """
        self._post_generation_fn = fn
        logger.debug(f"Post-generation function set for {self.__class__.__name__}")
    
    def set_llm_call_fn(self, fn: Callable[..., Any]) -> None:
        """
        Set a callback to intercept LLM calls for logging, caching, or response modification.
        
        The callback is invoked after each `_call_model()` with full context:
            fn(prompt=str, response=obj, query_idx=int, from_phase=str, **kwargs)
        
        The callback can:
        - Return None: Keep original response (use for logging/side effects)
        - Return a value: Replace the response (use for mocking/caching/modification)
        
        Args:
            fn: Callable that receives (prompt, response, query_idx, from_phase, **kwargs)
        
        Example - Logging with prompt hash:
            ```python
            import hashlib
            records = []
            
            def log_calls(prompt, response, **kwargs):
                records.append({
                    "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:12],
                    "output": response.text,
                    "temperature": kwargs.get('temperature'),
                })
                return None  # Keep original response
            
            policy.set_llm_call_fn(log_calls)
            ```
        
        Example - Response caching:
            ```python
            cache = {}
            
            def cache_fn(prompt, response, **kwargs):
                key = hashlib.md5(prompt.encode()).hexdigest()
                if key not in cache:
                    cache[key] = response
                return cache[key]  # Return cached response
            
            policy.set_llm_call_fn(cache_fn)
            ```
        
        Example - Mock responses for testing:
            ```python
            from unittest.mock import MagicMock
            
            def mock_fn(prompt, response, **kwargs):
                mock = MagicMock()
                mock.text = "mocked action"
                return mock
            
            policy.set_llm_call_fn(mock_fn)
            ```
        """
        self._llm_call_fn = fn
        logger.debug(f"LLM call function set for {self.__class__.__name__}")
    
    def _get_dynamic_notes(self) -> str:
        """
        Retrieve and format dynamic notes from the callback function.
        
        Returns:
            Formatted string with bullet points, or empty string if no notes available.
            Format: "\n\nAdditional Notes:\n* note1\n* note2\n* note3"
        """
        if self._dynamic_notes_fn is None:
            return ""
        
        try:
            notes = self._dynamic_notes_fn()
            if not notes:
                return ""
            
            return f"\n\nYou MUST AVOID the following mistakes:\n\n{notes}"
        except Exception as e:
            logger.error(f"Error retrieving dynamic notes: {e}", exc_info=True)
            return ""
    
    def _format_memory_context(self, memory_context) -> str:
        """
        Format memory context from LiTS-Mem for prompt injection.
        
        This helper method formats an AugmentedContext object into a string suitable
        for prepending to policy prompts. It wraps the memory context in XML-style
        tags for clear delineation in the prompt.
        
        Args:
            memory_context: An AugmentedContext object from LiTSMemoryManager, or None.
                           If None or empty, returns empty string.
        
        Returns:
            Formatted string with memory context wrapped in <memory> tags, or empty
            string if no memory context is available.
        
        Example output::
        
            <memory>
            Trajectory q/1 (score=0.75)
            - Consider substitution u = x^2
            - Apply power rule after substitution
            </memory>
        
        Note:
            By default, inherited memories are excluded (include_inherited=False) since
            prefix actions are already in the policy prompt via TrajectoryState.render_history().
            Cross-trajectory memories (from similar trajectories) are included to provide
            new insights that the current trajectory doesn't have.
        """
        if memory_context is None:
            return ""
        
        # Format memory context, excluding inherited memories by default
        # since they're already in the prompt via state history
        prompt_blocks = memory_context.to_prompt_blocks(include_inherited=False)
        
        if not prompt_blocks:
            return ""
        
        return f"\n<memory>\n{prompt_blocks}\n</memory>\n"
    
    def set_system_prompt(self) -> None:
        """
        Set the system prompt for the base model based on the task prompt specification.
        
        This method is called every time get_action is invoked, in case of dynamic system prompt construction.
        It automatically appends dynamic notes if a notes function has been set via set_dynamic_notes_fn().
        It also includes memory context from LiTS-Mem if available (set via _current_memory_context).
        """
        
        if isinstance(self.base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)):
            if self.task_prompt_spec:
                # Build base system prompt from subclass implementation
                base_prompt = self._build_system_prompt()
                # Append dynamic notes if available
                dynamic_notes = self._get_dynamic_notes()
                # Append memory context if available
                memory_context_str = self._format_memory_context(self._current_memory_context)
                self.base_model.sys_prompt = base_prompt + dynamic_notes + memory_context_str
            else:
                logger.warning("Chat Model but no system prompt constructed since `task_prompt_spec` is None ")
        else:
            if self.task_prompt_spec:
                logger.warning("task_prompt_spec exists but base_model does not support system prompts.")
         
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        Build the base system prompt for the LLM.
        
        Subclasses should implement this method to construct the system prompt
        from task_prompt_spec. Dynamic notes will be automatically appended by
        set_system_prompt(), so subclasses don't need to handle that.
        
        Returns:
            The base system prompt string (without dynamic notes)
        
        Example implementation:
            ```python
            def _build_system_prompt(self) -> str:
                return self.task_prompt_spec
            ```
        
        Note:
            Do NOT call self._get_dynamic_notes() in this method. Dynamic notes
            are automatically appended by set_system_prompt().
        """
        raise NotImplementedError()
        
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
        - Memory context injection from LiTS-Mem

        Args:
            state: Current state or trajectory to condition the policy.
            query: Optional context or example (not needed for all tasks).
            n_actions: Number of actions to generate. If None, uses self.n_actions
                      or 1 if at depth limit.
            query_idx: Optional index for logging or batching.
            critic: Optional critic for action evaluation.
            from_phase: Description of the current algorithm phase.
            *args, **kwargs: Additional arguments passed to _get_actions.
                - memory_context: Optional AugmentedContext from LiTS-Mem for
                  cross-trajectory memory augmentation.

        Return:
            List of Step objects with length exactly n_actions.
        """
        
        # Extract and store memory context for system prompt injection
        self._current_memory_context = kwargs.pop('memory_context', None)
        
        # Store context attributes for _call_model() helper
        self._query_idx = query_idx
        self._from_phase = from_phase
        
        self.set_system_prompt()

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
            if "run aws sso login" in str(e):
                raise Exception("Run `aws sso login`")
            
            raise e
            # Log the error with full traceback
            # logger.error(
            #     f"Error in {self.__class__.__name__}._get_actions(): {e}",
            #     exc_info=True,
            #     extra={
            #         'policy_class': self.__class__.__name__,
            #         'n_actions': n_actions,
            #         'query_idx': query_idx,
            #         'from_phase': from_phase
            #     }
            # )
            # # Create error steps to allow graceful continuation
            # outputs = self._create_error_steps(n_actions, str(e))

        # Validate outputs
        if len(outputs) != n_actions:
            logging.warning(f"Expected {n_actions} actions, but got {len(outputs)}")
        assert all(isinstance(output, Step) for output in outputs), "All outputs must be instances of Step or its subclasses"
        
        # Log the results
        self._log_policy_call_end(outputs, n_actions)
        
        # Execute post-generation callback if set
        if self._post_generation_fn is not None:
            try:
                context = {
                    'query': query,
                    'query_idx': query_idx,
                    'state': state,
                    'n_actions': n_actions,
                    'temperature': temperature,
                    'critic': critic,
                    'from_phase': from_phase,
                    'policy_model_name': kwargs.get('policy_model_name'),
                    'task_name': kwargs.get('task_name'),
                }
                self._post_generation_fn(outputs, context)
            except Exception as e:
                logger.error(
                    f"Error in post-generation callback: {e}",
                    exc_info=True,
                    extra={'policy_class': self.__class__.__name__}
                )

        # Clear memory context after use to avoid leaking to subsequent calls
        self._current_memory_context = None
        
        return outputs
    
    @abstractmethod
    def _create_error_steps(self, n_actions: int, error_msg: str) -> List[StepT]:
        """
        Create error steps when _get_actions fails with an exception.
        
        This method is called by get_actions() when _get_actions() raises an exception,
        allowing the policy to gracefully handle errors by returning valid Step objects
        with error information instead of crashing the entire search.
        
        IMPORTANT: Do NOT add logging in this method. Error logging is already handled
        by the base class get_actions() method before calling this method. This method
        should only create and return the appropriate Step objects.
        
        When called:
        - During policy.get_actions() execution
        - Only when _get_actions() raises an exception (e.g., LLM API failure, parsing error)
        - After the error has been logged by get_actions()
        - Before returning to the tree search algorithm
        
        Args:
            n_actions: Number of error steps to create (matches requested action count)
            error_msg: The exception message to include in error steps
        
        Returns:
            List of Step objects (specific subclass type) with error field set
        
        Example implementations:
            - ConcatPolicy: return [ThoughtStep(action="", error=error_msg) for _ in range(n_actions)]
            - RAPPolicy: return [SubQAStep(sub_question="", sub_answer="", confidence=0.0, error=error_msg) for _ in range(n_actions)]
            - ToolUsePolicy: return [ToolUseStep(action=None, observation=None, answer=None, error=error_msg) for _ in range(n_actions)]
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _create_error_steps")
    
    def _is_at_depth_limit(self, state: StateT) -> bool:
        """Check if the state has reached the max_steps limit."""
        if not self.force_terminating_on_depth_limit:
            return False
        return len(state) + 1 >= self.max_steps
    
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
    
    Class Attributes:
        TASK_TYPE: Interface category for this reward model (e.g., 'language_grounded', 'tool_use', 'env_grounded').
            Subclasses should override this to declare their interface category.
    
    Args:
        base_model: The LLM model to use for reward evaluation
        task_prompt_spec: System prompt specification (instructions, format, etc.)
            Can be a string, dict, or PromptTemplate. Used to construct the system message.
        task_name: Task name identifier (e.g., 'gsm8k', 'blocksworld', 'mapeval-sql') for loading
            task-specific prompts from the registry. This is the prompt lookup key.
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
        - TASK_TYPE: Interface category (set as class constant in subclasses)
        - task_name: Prompt lookup key (passed as parameter)
        - task_prompt_spec: For system-level instructions (system prompt)
        - usr_prompt_spec: For user-level content (user message)
        - Priority: task_prompt_spec > usr_prompt_spec > registry
    
    LLM Call Helpers:
        Subclasses can use `_call_model(prompt, **kwargs)` or `_call_model_logits(prompt, tokens, **kwargs)`
        to call the LLM without manually constructing the role string. The helpers auto-construct
        the role from stored context (`_query_idx`, `_from_phase`) set by `fast_reward()`.
        
        Usage in subclass `_fast_reward()`:
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                prompt = self._build_prompt(state, action, query)
                # No need to pass query_idx or from_phase - they're auto-injected
                logits = self._call_model_logits(prompt, ["Yes", "No"])
                probs = np.exp(logits) / np.sum(np.exp(logits))
                return float(probs[0])
            ```
    """
    
    # Interface category for this reward model type (subclasses should override)
    TASK_TYPE: str = None
    
    def __init__(
        self,
        base_model: LanguageModel,
        task_prompt_spec: Optional[Union[str, dict, 'PromptTemplate']] = None,
        task_name: Optional[str] = None,
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
        self.task_name = task_name
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        # Get task_type from class attribute TASK_TYPE for fallback lookup
        task_type = getattr(self.__class__, 'TASK_TYPE', None)
        
        if task_prompt_spec is None:
            logger.debug(f"Task prompt spec not provided, loading from registry for agent '{agent_name}', task_name='{task_name}', task_type='{task_type}'")
            task_prompt_spec = PromptRegistry.get('reward', agent_name, task_name, task_type)
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('reward', agent_name, task_name, task_type)
        
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
        
        # Context attributes for _call_model() helper (set by fast_reward()/reward())
        self._query_idx: Optional[int] = None
        self._from_phase: str = ""
        
    def fast_reward(self, state, action_or_step, query_or_goals, query_idx, from_phase="") -> tuple[float, dict]:
        """
        Generate a reward for an action without executing it.
        
        This method evaluates the potential usefulness/quality of an action based only on
        the current state and the proposed action, without actually executing the action
        to observe its outcome. This is useful for:
        
        - Tasks where action execution is expensive (e.g., env_grounded tasks)
        - Reasoning tasks where we can evaluate thought quality before execution (e.g., language_grounded with RAP)
        - Pruning unpromising actions early in tree search
        
        Args:
            query_or_goals: The problem/question being solved
            query_idx: Index of the example (for logging)
            state: Current state before action execution
            action_or_step: Proposed action or step to evaluate
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            Tuple of (reward, auxiliary_dict) where:
            - reward: Float score indicating action quality
            - auxiliary_dict: Additional metrics from _fast_reward (e.g., {'r_useful': probability})
        
        Note:
            This differs from `reward()` which evaluates after action execution.
        """
        logger.debug("\n>>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (BEGIN) <<<<<<<<<")
        if self.task_prompt_spec:
            self.base_model.sys_prompt = self.task_prompt_spec
        
        # Store context attributes for _call_model() helper
        self._query_idx = query_idx
        self._from_phase = from_phase
        
        # Call _fast_reward which may return float or (float, dict)
        result = self._fast_reward(state, action_or_step, query_or_goals, query_idx, from_phase=from_phase)
        
        # Handle both return types from _fast_reward
        if isinstance(result, tuple):
            raw_reward, details = result
        else:
            raw_reward = result
            details = {}
        
        # Apply calculate_reward transformation
        fast_reward = self.calculate_reward(raw_reward)

        logger.debug(f"fast_reward: {fast_reward}")
        logger.debug(">>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (END) <<<<<<<<<\n")

        return fast_reward, details

    def _get_agent_name(self, first_word=False) -> str:
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
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        if first_word:
            return name.split('_')[0]
        return name
    
    def _get_llm_role(self) -> str:
        """
        Return the LLM role prefix for this reward model.
        
        This is used by _call_model() to construct the role string for inference logging.
        Subclasses can override this to use a different role prefix.
        
        Default behavior returns "rm" (reward model) which is the standard role for
        reward models that evaluate action quality.
        
        Common role prefixes for reward models:
        - "rm": For general reward model evaluation (default)
        - "prm_language": For language-grounded process reward models
        - "prm_env": For environment-grounded process reward models
        - "prm_tool": For tool-use process reward models
        - "evaluator_correctness": For correctness evaluation
        - "evaluator_usefulness": For usefulness evaluation
        - "evaluator_tooluse": For tool-use evaluation
        
        Returns:
            str: The role prefix, default is "rm"
        """
        return "rm"
    
    def _call_model(self, prompt: str, **kwargs):
        """
        Call the base model with auto-constructed role from stored context.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `fast_reward()`) and calls `self.base_model()`. Subclasses can use this
        in `_fast_reward()` without manually passing query_idx or from_phase.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments passed to base_model (e.g., temperature, max_new_tokens)
        
        Returns:
            The model response object
        
        Example:
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                prompt = self._build_prompt(state, action, query)
                response = self._call_model(prompt, temperature=0.0)
                return self._parse_response(response)
            ```
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        return self.base_model(prompt, role=role, **kwargs)
    
    def _call_model_logits(self, prompt: str, candidates: list[str], **kwargs):
        """
        Call the base model's get_next_token_logits with auto-constructed role.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `fast_reward()`) and calls `self.base_model.get_next_token_logits()`.
        Subclasses can use this in `_fast_reward()` for logit-based evaluation.
        
        Args:
            prompt: The prompt to send to the model
            candidates: List of candidate tokens to get logits for (e.g., ["Yes", "No"])
            **kwargs: Additional arguments passed to get_next_token_logits
        
        Returns:
            The logits array for the specified tokens
        
        Example:
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                prompt = self._build_prompt(state, action, query)
                logits = self._call_model_logits(prompt, ["Yes", "No"])
                probs = np.exp(logits) / np.sum(np.exp(logits))
                return float(probs[0])
            ```
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        return self.base_model.get_next_token_logits(prompt, candidates, role=role, **kwargs)
    
    def _call_model_with_role(self, prompt: str, role_prefix: str, **kwargs):
        """
        Call the base model with a custom role prefix.
        
        This helper is for subclasses that need multiple different role types
        (e.g., "evaluator_correctness", "evaluator_usefulness") within the same
        _fast_reward() method. It constructs the role string from the provided
        role_prefix and stored context (`_query_idx`, `_from_phase`).
        
        Args:
            prompt: The prompt to send to the model
            role_prefix: Custom role prefix (e.g., "evaluator_correctness")
            **kwargs: Additional arguments passed to base_model
        
        Returns:
            The model response object
        
        Example:
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                # Use different roles for different evaluation types
                correctness = self._call_model_with_role(prompt, "evaluator_correctness")
                usefulness = self._call_model_with_role(prompt, "evaluator_usefulness")
            ```
        """
        from .utils import create_role
        role = create_role(role_prefix, self._query_idx, self._from_phase)
        return self.base_model(prompt, role=role, **kwargs)
    
    def _call_model_logits_with_role(self, prompt: str, candidates: list[str], role_prefix: str, **kwargs):
        """
        Call the base model's get_next_token_logits with a custom role prefix.
        
        This helper is for subclasses that need multiple different role types
        for logit-based evaluation within the same _fast_reward() method.
        
        Args:
            prompt: The prompt to send to the model
            candidates: List of candidate tokens to get logits for (e.g., ["Yes", "No"])
            role_prefix: Custom role prefix (e.g., "evaluator_logits")
            **kwargs: Additional arguments passed to get_next_token_logits
        
        Returns:
            The logits array for the specified tokens
        
        Example:
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                logits = self._call_model_logits_with_role(prompt, ["1", "0"], "evaluator_logits")
            ```
        """
        from .utils import create_role
        role = create_role(role_prefix, self._query_idx, self._from_phase)
        return self.base_model.get_next_token_logits(prompt, candidates, role=role, **kwargs)
    
    def _sample_binary_output(self, user_message: str, sample_size: int, target: str, contrast: str, unknown: str, **kwargs):
        """
        Call the base model's sample_binary_output with auto-constructed role.
        
        This helper method constructs the role string from `_query_idx` and `_from_phase`
        (set by `fast_reward()`) and calls `self.base_model.sample_binary_output()`.
        
        Args:
            user_message: The user message to send to the model
            sample_size: Number of samples to generate
            target: Target token (e.g., "good")
            contrast: Contrast token (e.g., "bad")
            unknown: Unknown token (e.g., "unknown")
            **kwargs: Additional arguments passed to sample_binary_output
        
        Returns:
            Dict with counts for each token type
        """
        from .utils import create_role
        role = create_role(self._get_llm_role(), self._query_idx, self._from_phase)
        return self.base_model.sample_binary_output(
            user_message=user_message,
            sample_size=sample_size,
            target=target,
            contrast=contrast,
            unknown=unknown,
            role=role,
            **kwargs
        )
    
    @abstractmethod
    def _fast_reward(self, state, action, query, query_idx, from_phase="") -> float:
        """Evaluate action quality without executing it. Subclasses must implement this.
        
        This is the core evaluation logic called by fast_reward(). Implementations can
        use the helper methods `_call_model()` or `_call_model_logits()` which auto-construct
        the role string from stored context (`_query_idx`, `_from_phase`).
        
        Args:
            state: Current state before action execution
            action: Proposed action to evaluate
            query: The problem/question being solved (query_or_goals)
            query_idx (int): Index of the current example. Used for:
                - Tracking which example an LLM call belongs to
                - Constructing role strings via create_role(llm_role, query_idx, from_phase)
                - Debugging and log analysis
            from_phase (str): Algorithm phase description. Used for:
                - Inference logging to distinguish LLM calls from different search phases
                - Common values: 'expand', 'simulate', 'continuation', 'sort'
                - Passed to create_role() to construct role like 'evaluator_logits_3_expand'
        
        Returns:
            float or tuple[float, dict]: Reward score, optionally with auxiliary info dict.
            
            Return Types:
            1. float: Simple reward score (details dict will be empty {})
            2. tuple[float, dict]: Reward score with auxiliary details
            
            The auxiliary dict serves two purposes:
            1. **Primary**: Provide additional metrics needed by the reward() method.
               For example, RapPRM returns {'r_useful': score, 'r_correct': score} which
               are then passed to reward() for final reward calculation.
            2. **Secondary**: Record evaluation details in search nodes for tracking/debugging.
               The dict is stored in node.fast_reward_details and can be used for analysis,
               logging, or visualization of the search process.
        
        Example using _call_model_logits() helper (recommended):
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                prompt = self._build_prompt(state, action, query)
                # No need to pass query_idx or from_phase - they're auto-injected
                logits = self._call_model_logits(prompt, ["Yes", "No"])
                probs = np.exp(logits) / np.sum(np.exp(logits))
                return float(probs[0])
            ```
        
        Example using create_role() directly (legacy):
            ```python
            def _fast_reward(self, state, action, query, query_idx, from_phase=""):
                from ..utils import create_role
                output = self.base_model(prompt, role=create_role("evaluator_logits", query_idx, from_phase))
                score = parse_score(output)
                return score, {"r_useful": score, "raw_output": output.text}
            ```
        """
        raise NotImplementedError("_fast_reward is not implemented for RewardModel")
    
    @abstractmethod
    def calculate_reward(self, fast_reward: float) -> float:
        raise NotImplementedError("calculate_reward is not implemented for RewardModel")

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
