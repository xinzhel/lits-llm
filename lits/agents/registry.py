"""Agent Registry for LiTS framework.

Provides decorator-based registration for search algorithms (MCTS, BFS, custom).
AI/NLP researchers register ``BaseTreeSearch`` subclasses via ``@register_search``;
the registry wraps each class into a callable that instantiates + runs the search.

Notes:
    ``TYPE_CHECKING`` is purely for IDE/type-checker visibility.  It adds zero
    runtime cost and zero circular import risk — the guarded import block never
    executes at runtime.

    ``from __future__ import annotations`` makes all annotations strings
    (lazy-evaluated).  Defensive: if a maintainer later adds
    ``target: type[BaseTreeSearch]`` in a signature, it won't trigger a
    runtime ``NameError`` from the ``TYPE_CHECKING``-only import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type, Callable, Optional, List
import logging

if TYPE_CHECKING:
    from lits.agents.tree.search_base import BaseTreeSearch

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for search algorithms.

    Enables AI/NLP researchers to register custom search algorithms
    that work with any task type through LiTS's task-agnostic interfaces.

    Example::

        from lits.agents.tree.search_base import BaseTreeSearch, SearchResult

        @register_search("greedy_best_first", config_class=GreedyConfig)
        class GreedyBestFirst(BaseTreeSearch):
            def search(self, query, query_idx) -> SearchResult:
                ...
    """

    _searches: Dict[str, Callable] = {}
    _configs: Dict[str, Type] = {}  # algorithm_name -> Config class

    @classmethod
    def register_search(cls, name: str, config_class: Optional[Type] = None) -> Callable:
        """Decorator to register a ``BaseTreeSearch`` subclass.

        The decorated class is wrapped into a callable stored in
        ``cls._searches[name]``.  The callable accepts the same arguments
        as the old function-based interface so that ``main_search.py``
        works unchanged::

            search_fn = AgentRegistry.get_search("mcts")
            result = search_fn(query, idx, config, world_model, policy, reward_model, ...)

        The decorator returns the original class (not the wrapper), so
        ``MCTSSearch`` in module scope is still the class itself.

        Args:
            name: Algorithm name (e.g., ``'mcts'``, ``'bfs'``)
            config_class: Optional config dataclass for the algorithm.
                          If ``None``, CLI falls back to ``BaseSearchConfig``.
        """
        def decorator(target):
            # Lazy import to avoid circular dependency at module load time.
            from lits.agents.tree.search_base import BaseTreeSearch as _BTS

            if not (isinstance(target, type) and issubclass(target, _BTS)):
                raise TypeError(
                    f"@register_search expects a BaseTreeSearch subclass, "
                    f"got {type(target)}"
                )

            if name in cls._searches:
                logger.warning(f"Search '{name}' already registered, overwriting")

            # Wrap class into a callable matching the old function signature.
            def wrapper(query, query_idx, config, world_model, policy,
                        reward_model, bn_evaluator=None, **kwargs):
                instance = target(
                    config=config,
                    world_model=world_model,
                    policy=policy,
                    reward_model=reward_model,
                    bn_evaluator=bn_evaluator,
                    **kwargs,
                )
                return instance.run(query, query_idx)

            cls._searches[name] = wrapper

            if config_class is not None:
                cls._configs[name] = config_class

            logger.debug(f"Registered search algorithm '{name}'")
            return target  # preserve the class in module namespace
        return decorator

    @classmethod
    def get_search(cls, name: str) -> Callable:
        """Look up a registered search algorithm."""
        if name not in cls._searches:
            raise KeyError(
                f"Search '{name}' not found. Available: {list(cls._searches.keys())}"
            )
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
    """Decorator to register a ``BaseTreeSearch`` subclass.

    Example::

        from lits.agents.registry import register_search
        from lits.agents.tree.search_base import BaseTreeSearch, SearchResult

        @register_search("greedy_best_first")
        class GreedyBestFirst(BaseTreeSearch):
            def search(self, query, query_idx) -> SearchResult:
                ...
    """
    return AgentRegistry.register_search(name, config_class)
