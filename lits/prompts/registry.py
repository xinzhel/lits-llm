"""
Centralized prompt registry for LLM-based components.

This module provides a registry system for managing prompts across different
components (Policy, RewardModel, Transition) and task types.

Lookup priority in get()/get_usr():
1. task_name (benchmark-specific, e.g., 'blocksworld')
2. task_type (from component's TASK_TYPE, e.g., 'language_grounded', 'env_grounded', 'tool_use')
3. 'default'

Decorator API:
- register_system_prompt(component, agent, prompt_key): Register system prompts
- register_user_prompt(component, agent, prompt_key): Register user prompts

Note: prompt_key is a lookup key that can be matched by task_name or task_type during retrieval.
It can be a benchmark name (e.g., 'blocksworld', 'crosswords') or a task type (e.g., 'language_grounded').
"""

from typing import Optional, Dict, Any, Union, Callable
from .prompt import PromptTemplate


class PromptRegistry:
    """
    Centralized registry for managing prompts across components and task types.
    
    Usage:
        # Register a prompt for a task type
        PromptRegistry.register('policy', 'rap', 'language_grounded', prompt_spec)
        
        # Get a prompt (tries task_name first, then task_type, then default)
        prompt = PromptRegistry.get('policy', 'rap', task_name='gsm8k', task_type='language_grounded')
    """
    
    _registry: Dict[str, Dict[str, Dict[str, Any]]] = {
        'policy': {},
        'reward': {},
        'transition': {}
    }
    
    _usr_registry: Dict[str, Dict[str, Dict[str, Any]]] = {
        'policy': {},
        'reward': {},
        'transition': {}
    }
    
    @classmethod
    def register(
        cls,
        component_type: str,
        agent_name: str,
        prompt_key: Optional[str],
        prompt_spec: Union[str, Dict, PromptTemplate]
    ):
        """
        Register a prompt for a specific component, agent, and prompt key.
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Name of the agent (e.g., 'rap', 'concat', 'tool_use')
            prompt_key: Lookup key for the prompt. Can be:
                - A benchmark name (e.g., 'blocksworld', 'crosswords') for benchmark-specific prompts
                - A task type (e.g., 'language_grounded', 'env_grounded') for task-type-level prompts
                - None for default prompts
            prompt_spec: Prompt specification (string, dict, or PromptTemplate)
        """
        if component_type not in cls._registry:
            cls._registry[component_type] = {}
        
        if agent_name not in cls._registry[component_type]:
            cls._registry[component_type][agent_name] = {}
        
        key = prompt_key if prompt_key else 'default'
        cls._registry[component_type][agent_name][key] = prompt_spec
    
    @classmethod
    def get(
        cls,
        component_type: str,
        agent_name: str,
        task_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[Union[str, Dict, PromptTemplate]]:
        """
        Get a prompt from the registry with fallback support.
        
        Lookup priority:
        1. task_name (benchmark-specific, e.g., 'blocksworld')
        2. task_type (from component's TASK_TYPE, e.g., 'language_grounded')
        3. 'default'
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Name of the agent
            task_name: Benchmark name (e.g., 'gsm8k', 'blocksworld')
            task_type: Component's TASK_TYPE (e.g., 'language_grounded', 'env_grounded')
        
        Returns:
            Prompt specification or None if not found
        """
        if component_type not in cls._registry:
            return None
        
        if agent_name not in cls._registry[component_type]:
            return None
        
        agent_prompts = cls._registry[component_type][agent_name]
        
        # Priority 1: Try task_name (benchmark-specific)
        if task_name and task_name in agent_prompts:
            return agent_prompts[task_name]
        
        # Priority 2: Try task_type (from component's TASK_TYPE)
        if task_type and task_type in agent_prompts:
            return agent_prompts[task_type]
        
        # Priority 3: Fall back to default
        if 'default' in agent_prompts:
            return agent_prompts['default']
        
        return None
    
    @classmethod
    def list_registered(cls, component_type: Optional[str] = None) -> Dict:
        """
        List all registered prompts.
        
        Args:
            component_type: Optional filter by component type
        
        Returns:
            Dictionary of registered prompts
        """
        if component_type:
            return cls._registry.get(component_type, {})
        return cls._registry
    
    @classmethod
    def register_usr(
        cls,
        component_type: str,
        agent_name: str,
        prompt_key: Optional[str],
        usr_prompt_spec: Union[Dict, PromptTemplate]
    ):
        """
        Register a usr_prompt_spec (user message template).
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Agent identifier (e.g., 'rap', 'tool_use')
            prompt_key: Lookup key for the prompt. Can be:
                - A benchmark name (e.g., 'blocksworld', 'crosswords') for benchmark-specific prompts
                - A task type (e.g., 'language_grounded', 'env_grounded') for task-type-level prompts
                - None for default prompts
            usr_prompt_spec: User prompt specification (dict or PromptTemplate, NOT string)
        """
        if component_type not in cls._usr_registry:
            raise ValueError(f"Invalid component_type: {component_type}")
        
        if agent_name not in cls._usr_registry[component_type]:
            cls._usr_registry[component_type][agent_name] = {}
        
        key = prompt_key if prompt_key else 'default'
        cls._usr_registry[component_type][agent_name][key] = usr_prompt_spec
    
    @classmethod
    def get_usr(
        cls,
        component_type: str,
        agent_name: str,
        task_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[Union[Dict, PromptTemplate]]:
        """
        Get a usr_prompt_spec from the registry with fallback support.
        
        Lookup priority:
        1. task_name (benchmark-specific, e.g., 'blocksworld')
        2. task_type (from component's TASK_TYPE, e.g., 'language_grounded')
        3. 'default'
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Agent identifier
            task_name: Benchmark name (e.g., 'gsm8k', 'blocksworld')
            task_type: Component's TASK_TYPE (e.g., 'language_grounded', 'env_grounded')
        
        Returns:
            User prompt specification or None if not found
        """
        if component_type not in cls._usr_registry:
            return None
        
        if agent_name not in cls._usr_registry[component_type]:
            return None
        
        agent_prompts = cls._usr_registry[component_type][agent_name]
        
        # Priority 1: Try task_name (benchmark-specific)
        if task_name and task_name in agent_prompts:
            return agent_prompts[task_name]
        
        # Priority 2: Try task_type (from component's TASK_TYPE)
        if task_type and task_type in agent_prompts:
            return agent_prompts[task_type]
        
        # Priority 3: Fall back to default
        if 'default' in agent_prompts:
            return agent_prompts['default']
        
        return None
    
    @classmethod
    def clear(cls):
        """Clear all registered prompts (useful for testing)."""
        cls._registry = {
            'policy': {},
            'reward': {},
            'transition': {}
        }
        cls._usr_registry = {
            'policy': {},
            'reward': {},
            'transition': {}
        }


# Module-level decorator functions for prompt registration

def register_system_prompt(
    component: str,
    agent: str,
    prompt_key: Optional[str] = None
) -> Callable:
    """Decorator to register a system prompt (task_prompt_spec).
    
    The decorated function is called immediately and its return value is
    registered with PromptRegistry.register().
    
    Args:
        component: Component type ('policy', 'reward', 'transition')
        agent: Agent name (e.g., 'concat', 'generative', 'rap')
        prompt_key: Lookup key for the prompt. Can be:
            - A benchmark name (e.g., 'blocksworld', 'crosswords') for benchmark-specific prompts
            - A task type (e.g., 'language_grounded', 'env_grounded') for task-type-level prompts
            - None for default prompts
    
    Returns:
        Decorator function
    
    Return Format:
        The decorated function can return any type. The component that consumes
        the prompt is responsible for handling the type appropriately.
        Common patterns:
        - str: Simple system prompt text
        - Dict: Structured prompt with multiple fields
        - Custom objects: For complex prompt configurations
    
    Example:
        @register_system_prompt("policy", "rap", "my_math_task")
        def my_math_system_prompt():
            return "You are solving math problems step by step..."
    """
    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        prompt_spec = func()
        PromptRegistry.register(component, agent, prompt_key, prompt_spec)
        return func
    return decorator


def register_user_prompt(
    component: str,
    agent: str,
    prompt_key: Optional[str] = None
) -> Callable:
    """Decorator to register a user prompt (usr_prompt_spec).
    
    The decorated function is called immediately and its return value is
    registered with PromptRegistry.register_usr().
    
    Args:
        component: Component type ('policy', 'reward', 'transition')
        agent: Agent name (e.g., 'concat', 'generative', 'rap')
        prompt_key: Lookup key for the prompt. Can be:
            - A benchmark name (e.g., 'blocksworld', 'crosswords') for benchmark-specific prompts
            - A task type (e.g., 'language_grounded', 'env_grounded') for task-type-level prompts
            - None for default prompts
    
    Returns:
        Decorator function
    
    Return Format:
        The decorated function can return any type. The component that consumes
        the prompt is responsible for handling the type appropriately.
        Common patterns:
        - Dict[str, str]: Template dictionary with format keys
        - str: Simple user prompt template
        - Custom objects: For complex prompt configurations
    
    Example:
        @register_user_prompt("policy", "rap", "my_math_task")
        def my_math_user_prompt():
            return {"question_format": "Problem: {question}"}
    """
    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        usr_prompt_spec = func()
        PromptRegistry.register_usr(component, agent, prompt_key, usr_prompt_spec)
        return func
    return decorator


def load_default_prompts():
    """
    Load all default prompts from lits.prompts into the registry.
    
    This function is called automatically when the package is imported.
    
    Note: Prompts are registered under task_type (e.g., 'language_grounded', 'env_grounded')
    not benchmark names. The component's TASK_TYPE is used for lookup.
    
    Note: RAP prompts are no longer in core - they're in lits_benchmark.formulations.rap.
    Import that module to register RAP prompts.
    """
    # Import prompt modules
    try:
        from .policy import concat as concat_policy
        from .policy import tool_use as tool_use_policy
        from .policy import env_grounded as env_grounded_policy  # Fallback prompts
        from .reward import generative as generative_reward
        from .reward import env_grounded as env_grounded_reward  # Fallback prompts
        
        # Register policy prompts
        # Concat policy for language_grounded tasks
        if hasattr(concat_policy, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('policy', 'concat', 'language_grounded', concat_policy.task_prompt_spec_math_qa)
        
        # ToolUse policy (default for all tool_use tasks)
        if hasattr(tool_use_policy, 'task_prompt_spec'):
            PromptRegistry.register('policy', 'tool_use', None, tool_use_policy.task_prompt_spec)

        # Register reward prompts
        # Generative reward for language_grounded tasks
        if hasattr(generative_reward, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('reward', 'generative', 'language_grounded', generative_reward.task_prompt_spec_math_qa)
        
    except ImportError as e:
        # Gracefully handle missing prompt modules
        import logging
        logging.warning(f"Could not load some default prompts: {e}")


# Load default prompts when module is imported
load_default_prompts()
