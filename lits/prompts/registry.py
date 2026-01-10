"""
Centralized prompt registry for LLM-based components.

This module provides a registry system for managing prompts across different
components (Policy, RewardModel, Transition) and task types.

Lookup priority in get()/get_usr():
1. task_name (benchmark-specific, e.g., 'blocksworld')
2. task_type (from component's TASK_TYPE, e.g., 'language_grounded', 'env_grounded', 'tool_use')
3. 'default'

Decorator API:
- register_system_prompt(component, agent, task_type): Register system prompts
- register_user_prompt(component, agent, task_type): Register user prompts
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
        task_type: Optional[str],
        prompt_spec: Union[str, Dict, PromptTemplate]
    ):
        """
        Register a prompt for a specific component, agent, and task type.
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Name of the agent (e.g., 'rap', 'concat', 'tool_use')
            task_type: Task type (e.g., 'language_grounded', 'env_grounded', 'tool_use') or None for default
            prompt_spec: Prompt specification (string, dict, or PromptTemplate)
        """
        if component_type not in cls._registry:
            cls._registry[component_type] = {}
        
        if agent_name not in cls._registry[component_type]:
            cls._registry[component_type][agent_name] = {}
        
        key = task_type if task_type else 'default'
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
        task_type: Optional[str],
        usr_prompt_spec: Union[Dict, PromptTemplate]
    ):
        """
        Register a usr_prompt_spec (user message template).
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Agent identifier (e.g., 'rap', 'tool_use')
            task_type: Task type (e.g., 'language_grounded') or None for default
            usr_prompt_spec: User prompt specification (dict or PromptTemplate, NOT string)
        """
        if component_type not in cls._usr_registry:
            raise ValueError(f"Invalid component_type: {component_type}")
        
        if agent_name not in cls._usr_registry[component_type]:
            cls._usr_registry[component_type][agent_name] = {}
        
        key = task_type if task_type else 'default'
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
    task_type: Optional[str] = None
) -> Callable:
    """Decorator to register a system prompt (task_prompt_spec).
    
    The decorated function is called immediately and its return value is
    registered with PromptRegistry.register().
    
    Args:
        component: Component type ('policy', 'reward', 'transition')
        agent: Agent name (e.g., 'concat', 'generative', 'rap')
        task_type: Task type or benchmark name (e.g., 'language_grounded', 'blocksworld')
    
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
        PromptRegistry.register(component, agent, task_type, prompt_spec)
        return func
    return decorator


def register_user_prompt(
    component: str,
    agent: str,
    task_type: Optional[str] = None
) -> Callable:
    """Decorator to register a user prompt (usr_prompt_spec).
    
    The decorated function is called immediately and its return value is
    registered with PromptRegistry.register_usr().
    
    Args:
        component: Component type ('policy', 'reward', 'transition')
        agent: Agent name (e.g., 'concat', 'generative', 'rap')
        task_type: Task type or benchmark name (e.g., 'language_grounded', 'blocksworld')
    
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
        PromptRegistry.register_usr(component, agent, task_type, usr_prompt_spec)
        return func
    return decorator


def load_default_prompts():
    """
    Load all default prompts from lits.prompts into the registry.
    
    This function is called automatically when the package is imported.
    
    Note: Prompts are registered under task_type (e.g., 'language_grounded', 'env_grounded')
    not benchmark names. The component's TASK_TYPE is used for lookup.
    """
    # Import prompt modules
    try:
        from .policy import rap as rap_policy
        from .policy import concat as concat_policy
        from .policy import tool_use as tool_use_policy
        from .policy import blocksworld as blocksworld_policy
        from .reward import rap as rap_reward
        from .reward import generative as generative_reward
        from .reward import blocksworld as blocksworld_reward
        from .transition import rap as rap_transition
        from .transition import blocksworld as blocksworld_transition
        
        # Register policy prompts
        # RAP policy for language_grounded tasks (gsm8k, math500, spart_yn)
        if hasattr(rap_policy, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('policy', 'rap', 'language_grounded', rap_policy.task_prompt_spec_math_qa)
        if hasattr(rap_policy, 'usr_prompt_spec_math_qa'):
            PromptRegistry.register_usr('policy', 'rap', 'language_grounded', rap_policy.usr_prompt_spec_math_qa)
        
        # EnvGrounded policy for blocksworld (benchmark-specific)
        if hasattr(blocksworld_policy, 'usr_prompt_spec'):
            PromptRegistry.register_usr('policy', 'env_grounded', 'blocksworld', blocksworld_policy.usr_prompt_spec)
        
        # Concat policy for language_grounded tasks
        if hasattr(concat_policy, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('policy', 'concat', 'language_grounded', concat_policy.task_prompt_spec_math_qa)
        
        # ToolUse policy (default for all tool_use tasks)
        if hasattr(tool_use_policy, 'task_prompt_spec'):
            PromptRegistry.register('policy', 'tool_use', None, tool_use_policy.task_prompt_spec)

        # Register reward prompts
        # RAP reward for language_grounded tasks
        if hasattr(rap_reward, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('reward', 'rap', 'language_grounded', rap_reward.task_prompt_spec_math_qa)
        
        # Generative reward for language_grounded tasks
        if hasattr(generative_reward, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('reward', 'generative', 'language_grounded', generative_reward.task_prompt_spec_math_qa)
        
        # EnvGrounded reward for blocksworld (benchmark-specific)
        if hasattr(blocksworld_reward, 'task_prompt_spec_blocksworld'):
            PromptRegistry.register('reward', 'env_grounded', 'blocksworld', blocksworld_reward.task_prompt_spec_blocksworld)
        if hasattr(blocksworld_reward, 'usr_prompt_spec_blocksworld'):
            PromptRegistry.register_usr('reward', 'env_grounded', 'blocksworld', blocksworld_reward.usr_prompt_spec_blocksworld)
                
        # Register transition prompts
        # RAP transition (default and language_grounded)
        if hasattr(rap_transition, 'task_prompt_spec'):
            PromptRegistry.register('transition', 'rap', None, rap_transition.task_prompt_spec)
        if hasattr(rap_transition, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('transition', 'rap', 'language_grounded', rap_transition.task_prompt_spec_math_qa)
        if hasattr(rap_transition, 'usr_prompt_spec_math_qa'):
            PromptRegistry.register_usr('transition', 'rap', 'language_grounded', rap_transition.usr_prompt_spec_math_qa)
        
        # BlocksWorld transition (benchmark-specific)
        if hasattr(blocksworld_transition, 'task_prompt_spec'):
            PromptRegistry.register('transition', 'blocksworld', None, blocksworld_transition.task_prompt_spec)
        if hasattr(blocksworld_transition, 'usr_prompt_spec'):
            PromptRegistry.register_usr('transition', 'blocksworld', None, blocksworld_transition.usr_prompt_spec)
        
    except ImportError as e:
        # Gracefully handle missing prompt modules
        import logging
        logging.warning(f"Could not load some default prompts: {e}")


# Load default prompts when module is imported
load_default_prompts()
