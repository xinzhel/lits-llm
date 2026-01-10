"""Component Registry for LiTS framework.

This module provides a decorator-based registration system for LiTS components
(Transition, Policy, RewardModel) across all task types (env_grounded, 
language_grounded, tool_use).

Usage:
    from lits.components.registry import register_transition, register_policy, register_reward_model
    
    # Register a custom Transition
    @register_transition("my_task", task_type="language_grounded")
    class MyTransition(LlmTransition):
        ...
    
    # Register a custom Policy
    @register_policy("my_task", task_type="language_grounded")
    class MyPolicy(ConcatPolicy):
        ...
    
    # Register a custom RewardModel
    @register_reward_model("my_task", task_type="language_grounded")
    class MyRewardModel(GenerativePRM):
        ...

Lookup:
    from lits.components.registry import ComponentRegistry
    
    TransitionCls = ComponentRegistry.get_transition("my_task")
    PolicyCls = ComponentRegistry.get_policy("my_task")
    RewardCls = ComponentRegistry.get_reward_model("my_task")
"""

from typing import Dict, Type, Optional, Callable, List, Literal
import logging

logger = logging.getLogger(__name__)

# Type alias for task types
TaskType = Literal['env_grounded', 'language_grounded', 'tool_use']


class ComponentRegistry:
    """Central registry for all task type components.
    
    Provides decorator-based registration and lookup for:
    - Transition classes (world models) - all task types
    - Policy classes - all task types
    - RewardModel classes - all task types
    
    For env_grounded tasks, Transition classes include:
    - goal_check() static method - checks if goals are met
    - generate_actions() static method - generates valid actions
    
    Also tracks task_type for each registered benchmark to support
    automatic task type inference.
    
    Lookup Strategy:
        For components (Transition, Policy, RewardModel), the registry supports
        two lookup modes:
        1. By benchmark_name (e.g., 'blocksworld', 'gsm8k') - for task-specific overrides
        2. By task_type (e.g., 'env_grounded', 'language_grounded') - for default implementations
        
        component_factory.py will:
        1. First try to get a benchmark-specific component
        2. Fall back to task_type default if not found
        3. Fall back to built-in hardcoded defaults if neither found
    """
    
    # Internal storage - keyed by benchmark_name or task_type
    _transitions: Dict[str, Type] = {}
    _policies: Dict[str, Type] = {}
    _reward_models: Dict[str, Type] = {}
    _task_types: Dict[str, str] = {}  # benchmark_name -> task_type
    
    @classmethod
    def register_transition(cls, name: str, task_type: str = 'env_grounded') -> Callable:
        """Decorator to register a Transition class.
        
        Args:
            name: Benchmark name (e.g., 'blocksworld', 'robot_arm') or task_type for defaults
            task_type: Task type for this benchmark ('env_grounded', 'language_grounded', 'tool_use')
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If a Transition with the same name is already registered
        
        Example:
            # Register for specific benchmark
            @ComponentRegistry.register_transition("robot_arm", task_type="env_grounded")
            class RobotArmTransition(LlmTransition):
                ...
            
            # Register as default for task_type
            @ComponentRegistry.register_transition("language_grounded", task_type="language_grounded")
            class CustomConcatTransition(LlmTransition):
                ...
        """
        def decorator(transition_cls: Type) -> Type:
            if name in cls._transitions:
                raise ValueError(
                    f"Transition '{name}' is already registered as {cls._transitions[name].__name__}. "
                    f"Cannot register {transition_cls.__name__}. "
                    f"Use a different name or call ComponentRegistry.clear() first."
                )
            cls._transitions[name] = transition_cls
            cls._task_types[name] = task_type
            logger.debug(f"Registered Transition '{name}' (task_type={task_type}): {transition_cls.__name__}")
            return transition_cls
        return decorator
    
    @classmethod
    def register_policy(cls, name: str, task_type: Optional[str] = None) -> Callable:
        """Decorator to register a Policy class.
        
        Args:
            name: Benchmark name or task_type for defaults
            task_type: Task type (optional, for benchmark-specific registrations)
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If a Policy with the same name is already registered
        
        Example:
            # Register custom policy for gsm8k
            @ComponentRegistry.register_policy("gsm8k", task_type="language_grounded")
            class CustomGSM8KPolicy(ConcatPolicy):
                ...
        """
        def decorator(policy_cls: Type) -> Type:
            if name in cls._policies:
                raise ValueError(
                    f"Policy '{name}' is already registered as {cls._policies[name].__name__}. "
                    f"Cannot register {policy_cls.__name__}. "
                    f"Use a different name or call ComponentRegistry.clear() first."
                )
            cls._policies[name] = policy_cls
            if task_type is not None:
                cls._task_types[name] = task_type
            logger.debug(f"Registered Policy '{name}' (task_type={task_type}): {policy_cls.__name__}")
            return policy_cls
        return decorator
    
    @classmethod
    def register_reward_model(cls, name: str, task_type: Optional[str] = None) -> Callable:
        """Decorator to register a RewardModel class.
        
        Args:
            name: Benchmark name or task_type for defaults
            task_type: Task type (optional, for benchmark-specific registrations)
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If a RewardModel with the same name is already registered
        
        Example:
            # Register custom reward model for math tasks
            @ComponentRegistry.register_reward_model("math500", task_type="language_grounded")
            class CustomMathPRM(GenerativePRM):
                ...
        """
        def decorator(reward_cls: Type) -> Type:
            if name in cls._reward_models:
                raise ValueError(
                    f"RewardModel '{name}' is already registered as {cls._reward_models[name].__name__}. "
                    f"Cannot register {reward_cls.__name__}. "
                    f"Use a different name or call ComponentRegistry.clear() first."
                )
            cls._reward_models[name] = reward_cls
            if task_type is not None:
                cls._task_types[name] = task_type
            logger.debug(f"Registered RewardModel '{name}' (task_type={task_type}): {reward_cls.__name__}")
            return reward_cls
        return decorator
    
    @classmethod
    def get_transition(cls, name: str, fallback_task_type: Optional[str] = None) -> Type:
        """Look up a registered Transition class.
        
        Args:
            name: Benchmark name
            fallback_task_type: Task type to try if benchmark-specific not found
        
        Returns:
            The registered Transition class
        
        Raises:
            KeyError: If no Transition is found for the name or fallback_task_type
        
        Note:
            For env_grounded tasks, the returned class includes goal_check() and
            generate_actions() as static methods.
        """
        if name in cls._transitions:
            return cls._transitions[name]
        
        if fallback_task_type is not None and fallback_task_type in cls._transitions:
            return cls._transitions[fallback_task_type]
        
        available = list(cls._transitions.keys())
        raise KeyError(
            f"Transition '{name}' not found in registry. "
            f"Available transitions: {available}. "
            f"Did you forget to import the module containing @register_transition('{name}')?"
        )
    
    @classmethod
    def get_policy(cls, name: str, fallback_task_type: Optional[str] = None) -> Type:
        """Look up a registered Policy class.
        
        Args:
            name: Benchmark name
            fallback_task_type: Task type to try if benchmark-specific not found
        
        Returns:
            The registered Policy class
        
        Raises:
            KeyError: If no Policy is found for the name or fallback_task_type
        """
        if name in cls._policies:
            return cls._policies[name]
        
        if fallback_task_type is not None and fallback_task_type in cls._policies:
            return cls._policies[fallback_task_type]
        
        available = list(cls._policies.keys())
        raise KeyError(
            f"Policy '{name}' not found in registry. "
            f"Available policies: {available}. "
            f"Did you forget to import the module containing @register_policy('{name}')?"
        )
    
    @classmethod
    def get_reward_model(cls, name: str, fallback_task_type: Optional[str] = None) -> Type:
        """Look up a registered RewardModel class.
        
        Args:
            name: Benchmark name
            fallback_task_type: Task type to try if benchmark-specific not found
        
        Returns:
            The registered RewardModel class
        
        Raises:
            KeyError: If no RewardModel is found for the name or fallback_task_type
        """
        if name in cls._reward_models:
            return cls._reward_models[name]
        
        if fallback_task_type is not None and fallback_task_type in cls._reward_models:
            return cls._reward_models[fallback_task_type]
        
        available = list(cls._reward_models.keys())
        raise KeyError(
            f"RewardModel '{name}' not found in registry. "
            f"Available reward models: {available}. "
            f"Did you forget to import the module containing @register_reward_model('{name}')?"
        )
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names.
        
        Returns:
            List of all benchmark names that have been registered with any component.
        """
        return list(cls._task_types.keys())
    
    @classmethod
    def list_by_task_type(cls, task_type: str) -> List[str]:
        """List benchmark names registered with a specific task_type.
        
        Args:
            task_type: The task type to filter by ('env_grounded', 'language_grounded', 'tool_use')
        
        Returns:
            List of benchmark names registered with the specified task_type.
        """
        return [name for name, tt in cls._task_types.items() if tt == task_type]
    
    @classmethod
    def get_task_type(cls, name: str) -> Optional[str]:
        """Get the task_type for a registered benchmark.
        
        Args:
            name: Benchmark name
        
        Returns:
            The task_type string, or None if not found.
        """
        return cls._task_types.get(name)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations.
        
        This is primarily useful for testing to reset the registry state
        between test cases.
        """
        cls._transitions.clear()
        cls._policies.clear()
        cls._reward_models.clear()
        cls._task_types.clear()
        logger.debug("ComponentRegistry cleared")


# Module-level decorator aliases for convenience
def register_transition(name: str, task_type: str = 'env_grounded') -> Callable:
    """Decorator to register a Transition class.
    
    This is a module-level alias for ComponentRegistry.register_transition().
    
    Args:
        name: Benchmark name (e.g., 'blocksworld', 'robot_arm')
        task_type: Task type for this benchmark (default: 'env_grounded')
    
    Returns:
        Decorator function
    
    Example:
        from lits.components.registry import register_transition
        
        @register_transition("robot_arm", task_type="env_grounded")
        class RobotArmTransition(EnvGroundedTransition):
            ...
    """
    return ComponentRegistry.register_transition(name, task_type)


def register_policy(name: str, task_type: Optional[str] = None) -> Callable:
    """Decorator to register a Policy class.
    
    This is a module-level alias for ComponentRegistry.register_policy().
    
    Args:
        name: Benchmark name or task_type for defaults
        task_type: Task type (optional)
    
    Returns:
        Decorator function
    
    Example:
        from lits.components.registry import register_policy
        
        @register_policy("gsm8k", task_type="language_grounded")
        class CustomGSM8KPolicy(ConcatPolicy):
            ...
    """
    return ComponentRegistry.register_policy(name, task_type)


def register_reward_model(name: str, task_type: Optional[str] = None) -> Callable:
    """Decorator to register a RewardModel class.
    
    This is a module-level alias for ComponentRegistry.register_reward_model().
    
    Args:
        name: Benchmark name or task_type for defaults
        task_type: Task type (optional)
    
    Returns:
        Decorator function
    
    Example:
        from lits.components.registry import register_reward_model
        
        @register_reward_model("math500", task_type="language_grounded")
        class CustomMathPRM(GenerativePRM):
            ...
    """
    return ComponentRegistry.register_reward_model(name, task_type)
