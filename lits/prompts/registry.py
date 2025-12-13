"""
Centralized prompt registry for LLM-based components.

This module provides a registry system for managing prompts across different
components (Policy, RewardModel, Transition) and task types.
"""

from typing import Optional, Dict, Any, Union
from .prompt import PromptTemplate


class PromptRegistry:
    """
    Centralized registry for managing prompts across components and task types.
    
    Usage:
        # Register a prompt
        PromptRegistry.register('policy', 'rap', 'math_qa', prompt_spec)
        
        # Get a prompt
        prompt = PromptRegistry.get('policy', 'rap', 'math_qa')
        
        # Inject custom prompt
        PromptRegistry.register('policy', 'custom_agent', 'my_task', my_prompt)
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
            agent_name: Name of the agent (e.g., 'rap', 'rest', 'tool_use')
            task_type: Task type (e.g., 'math_qa', 'tool_use') or None for default
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
        task_type: Optional[str] = None
    ) -> Optional[Union[str, Dict, PromptTemplate]]:
        """
        Get a prompt from the registry.
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Name of the agent
            task_type: Task type or None
        
        Returns:
            Prompt specification or None if not found
        """
        if component_type not in cls._registry:
            return None
        
        if agent_name not in cls._registry[component_type]:
            return None
        
        agent_prompts = cls._registry[component_type][agent_name]
        
        # Try task-specific first, then fall back to default
        key = task_type if task_type else 'default'
        if key in agent_prompts:
            return agent_prompts[key]
        
        # Fall back to default if task-specific not found
        if task_type and 'default' in agent_prompts:
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
            task_type: Task type (e.g., 'math_qa') or None for default
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
        task_type: Optional[str] = None
    ) -> Optional[Union[Dict, PromptTemplate]]:
        """
        Get a usr_prompt_spec from the registry.
        
        Args:
            component_type: 'policy', 'reward', or 'transition'
            agent_name: Agent identifier
            task_type: Task type or None for default
        
        Returns:
            User prompt specification or None if not found
        """
        if component_type not in cls._usr_registry:
            return None
        
        if agent_name not in cls._usr_registry[component_type]:
            return None
        
        agent_prompts = cls._usr_registry[component_type][agent_name]
        
        # Try task-specific first, then fall back to default
        key = task_type if task_type else 'default'
        if key in agent_prompts:
            return agent_prompts[key]
        
        # Fall back to default if task-specific not found
        if task_type and 'default' in agent_prompts:
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


def load_default_prompts():
    """
    Load all default prompts from lits.prompts into the registry.
    
    This function is called automatically when the package is imported.
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
        # if hasattr(rap_policy, 'task_prompt_spec'):
        #     PromptRegistry.register('policy', 'rap', None, rap_policy.task_prompt_spec)
        if hasattr(rap_policy, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('policy', 'rap', 'math_qa', rap_policy.task_prompt_spec_math_qa)
        if hasattr(rap_policy, 'usr_prompt_spec_math_qa'):
            PromptRegistry.register_usr('policy', 'rap', 'math_qa', rap_policy.usr_prompt_spec_math_qa)
        if hasattr(blocksworld_policy, 'usr_prompt_spec'):
            PromptRegistry.register_usr('policy', 'env_grounded', "blocksworld", blocksworld_policy.usr_prompt_spec)
        
        # if hasattr(concat_policy, 'task_prompt_spec'):
        #     PromptRegistry.register('policy', 'rest', None, concat_policy.task_prompt_spec)
        if hasattr(concat_policy, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('policy', 'concat', 'math_qa', concat_policy.task_prompt_spec_math_qa)
        
        if hasattr(tool_use_policy, 'task_prompt_spec'):
            PromptRegistry.register('policy', 'tool_use', None, tool_use_policy.task_prompt_spec)

        # Register reward prompts
        # if hasattr(rap_reward, 'task_prompt_spec'):
        #     PromptRegistry.register('reward', 'rap', None, rap_reward.task_prompt_spec)
        if hasattr(rap_reward, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('reward', 'rap', 'math_qa', rap_reward.task_prompt_spec_math_qa)
        if hasattr(generative_reward, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('reward', 'generative', 'math_qa', generative_reward.task_prompt_spec_math_qa)
        if hasattr(blocksworld_reward, 'task_prompt_spec_blocksworld'):
            PromptRegistry.register('reward', 'env_grounded', "blocksworld", blocksworld_reward.task_prompt_spec_blocksworld)
        if hasattr(blocksworld_reward, 'usr_prompt_spec_blocksworld'):
            PromptRegistry.register_usr('reward', 'env_grounded', "blocksworld", blocksworld_reward.usr_prompt_spec_blocksworld)
                
        # Register transition prompts
        if hasattr(rap_transition, 'task_prompt_spec'):
            PromptRegistry.register('transition', 'rap', None, rap_transition.task_prompt_spec)
        if hasattr(rap_transition, 'task_prompt_spec_math_qa'):
            PromptRegistry.register('transition', 'rap', 'math_qa', rap_transition.task_prompt_spec_math_qa)
        if hasattr(rap_transition, 'usr_prompt_spec_math_qa'):
            PromptRegistry.register_usr('transition', 'rap', 'math_qa', rap_transition.usr_prompt_spec_math_qa)
        
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
