"""Agent Registry for LiTS framework.

Provides decorator-based registration for search algorithms (MCTS, BFS, custom).
AI/NLP researchers can register new algorithms that automatically inherit:
- Task-agnostic data structures (Action → Step → State → Node)
- Shared subprocedures from lits.agents.tree.common
- Component interfaces (Policy, Transition, RewardModel)
"""

from typing import Dict, Type, Callable, Optional, List
import logging

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for search algorithms.
    
    Enables AI/NLP researchers to register custom search algorithms
    that work with any task type through LiTS's task-agnostic interfaces.
    
    Example:
        @register_search("my_algorithm")
        def my_search(query, query_idx, config, world_model, policy, reward_model, **kwargs):
            # Use shared subprocedures from lits.agents.tree.common
            from lits.agents.tree.common import _world_modeling, create_child_node
            ...
    """
    
    _searches: Dict[str, Callable] = {}
    _configs: Dict[str, Type] = {}  # algorithm_name -> Config class
    
    @classmethod
    def register_search(cls, name: str, config_class: Optional[Type] = None) -> Callable:
        """Decorator to register a search algorithm.
        
        Args:
            name: Algorithm name (e.g., 'mcts', 'bfs', 'beam_search')
            config_class: Optional config dataclass for the algorithm
        
        Returns:
            Decorator function
        """
        def decorator(search_func: Callable) -> Callable:
            if name in cls._searches:
                logger.warning(f"Search '{name}' already registered, overwriting")
            cls._searches[name] = search_func
            if config_class is not None:
                cls._configs[name] = config_class
            logger.debug(f"Registered search algorithm '{name}'")
            return search_func
        return decorator
    
    @classmethod
    def get_search(cls, name: str) -> Callable:
        """Look up a registered search algorithm."""
        if name not in cls._searches:
            raise KeyError(f"Search '{name}' not found. Available: {list(cls._searches.keys())}")
        return cls._searches[name]
    
    @classmethod
    def get_config_class(cls, name: str) -> Optional[Type]:
        """Get the config class for a search algorithm."""
        return cls._configs.get(name)
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithm names."""
        return list(cls._searches.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._searches.clear()
        cls._configs.clear()


# Module-level decorator alias
def register_search(name: str, config_class: Optional[Type] = None) -> Callable:
    """Decorator to register a search algorithm.
    
    Example:
        from lits.agents.registry import register_search
        from lits.agents.tree.common import _world_modeling, create_child_node
        
        @register_search("greedy_search")
        def greedy_search(query, query_idx, config, world_model, policy, reward_model, **kwargs):
            root = SearchNode(state=world_model.init_state(), action=query, parent=None)
            # ... implement greedy search using shared subprocedures
    """
    return AgentRegistry.register_search(name, config_class)
