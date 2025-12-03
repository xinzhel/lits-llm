from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Tuple, Optional, Callable
from ..structures import StateT, ActionT, StepT, Step
from ..lm.base import DETERMINISTIC_TEMPERATURE
from ..lm import OpenAIChatModel, BedrockChatModel, HfChatModel, HfModel
import logging
logger = logging.getLogger(__name__)

class Transition(ABC, Generic[StateT, ActionT]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init_state(self) -> StateT: ...

    @abstractmethod
    def step(self, state: StateT, action: ActionT, *arg, **kwargs) -> Union[StateT, Tuple[StateT, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
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
        max_steps: Maximum depth/steps for tree search
        force_terminating_on_depth_limit: Force termination at max_steps
    
    Note:
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
        self.task_type = task_type
        
        # Load prompts from registry if not provided
        from ..prompts.registry import PromptRegistry
        agent_name = self._get_agent_name()
        
        logger.debug(f"Policy.__init__: agent_name={agent_name}, task_type={task_type}")
        logger.debug(f"Policy.__init__: task_prompt_spec (before registry) type={type(task_prompt_spec)}, value={task_prompt_spec}")
        
        if task_prompt_spec is None:
            task_prompt_spec = PromptRegistry.get('policy', agent_name, task_type)
            logger.debug(f"Policy.__init__: task_prompt_spec loaded from registry, type={type(task_prompt_spec)}, value={task_prompt_spec}")
        
        if usr_prompt_spec is None:
            usr_prompt_spec = PromptRegistry.get_usr('policy', agent_name, task_type)
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
            
            formatted_notes = "\n".join(f"* {note}" for note in notes)
            return f"\n\nAdditional Notes:\n{formatted_notes}"
        except Exception as e:
            logger.error(f"Error retrieving dynamic notes: {e}", exc_info=True)
            return ""
    
    def set_system_prompt(self) -> None:
        """
        Set the system prompt for the base model based on the task prompt specification.
        
        This method is called every time get_action is invoked, in case of dynamic system prompt construction.
        It automatically appends dynamic notes if a notes function has been set via set_dynamic_notes_fn().
        """
        
        if isinstance(self.base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)):
            if self.task_prompt_spec:
                # Build base system prompt from subclass implementation
                base_prompt = self._build_system_prompt()
                # Append dynamic notes if available
                dynamic_notes = self._get_dynamic_notes()
                self.base_model.sys_prompt = base_prompt + dynamic_notes
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
            # Log the error with full traceback
            logger.error(
                f"Error in {self.__class__.__name__}._get_actions(): {e}",
                exc_info=True,
                extra={
                    'policy_class': self.__class__.__name__,
                    'n_actions': n_actions,
                    'query_idx': query_idx,
                    'from_phase': from_phase
                }
            )
            # Create error steps to allow graceful continuation
            outputs = self._create_error_steps(n_actions, str(e))

        # Validate outputs
        if len(outputs) != n_actions:
            logging.warning(f"Expected {n_actions} actions, but got {len(outputs)}")
        assert all(isinstance(output, Step) for output in outputs), "All outputs must be instances of Step or its subclasses"
        
        # Log the results
        self._log_policy_call_end(outputs, n_actions)

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
            logger.debug(f"Task prompt spec not provided, loading from registry for agent '{agent_name}' and task '{task_type}'")
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
        
    def fast_reward(self, state, action, query, query_idx, from_phase="") -> tuple[float, dict]:
        """
        Generate a reward for an action without executing it.
        
        This method evaluates the potential usefulness/quality of an action based only on
        the current state and the proposed action, without actually executing the action
        to observe its outcome. This is useful for:
        
        - Tasks where action execution is expensive (e.g., env_grounded tasks)
        - Reasoning tasks where we can evaluate thought quality before execution (e.g., math_qa with RAP)
        - Pruning unpromising actions early in tree search
        
        Args:
            query: The problem/question being solved
            query_idx: Index of the example (for logging)
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

        fast_reward = self._fast_reward(state, action, query, query_idx, from_phase=from_phase)

        fast_reward = self.calculate_reward(fast_reward)

        logger.debug(f"fast_reward: {fast_reward}")
        logger.debug(">>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (END) <<<<<<<<<\n")

        return fast_reward, {'r_useful': float(fast_reward)}

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
    def _fast_reward(self, state, action,  query, query_idx, from_phase="") -> float:
        raise NotImplementedError("_fast_reward is not implemented for RewardModel")
    
    @abstractmethod
    def calculate_reward(self, fast_reward: float) -> float:
        raise NotImplementedError("calculate_reward is not implemented for RewardModel")

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
