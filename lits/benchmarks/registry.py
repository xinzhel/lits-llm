"""Benchmark Registry for LiTS framework.

This module provides a decorator-based registration system for dataset loaders
and task type inference. It enables automatic discovery of datasets and their
associated task types.

Usage:
    from lits.benchmarks.registry import register_dataset, load_dataset, infer_task_type
    
    # Register a dataset loader
    @register_dataset("my_dataset", task_type="language_grounded")
    def load_my_dataset(split: str = "test", **kwargs):
        ...
    
    # Load a dataset by name
    data = load_dataset("my_dataset", split="train")
    
    # Infer task type from dataset name
    task_type = infer_task_type("my_dataset")  # Returns "language_grounded"

The registry is the single source of truth for task type inference.
Datasets must be registered via @register_dataset() and their modules
imported (e.g., via --include) before use.
"""

from typing import Dict, Callable, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class BenchmarkRegistry:
    """Registry for dataset loaders and task type inference.
    
    Provides:
    - Dataset loader registration and lookup
    - Task type inference from dataset names
    - Resource (tool-use benchmark) registration and lookup
    
    This registry is separate from ComponentRegistry because:
    - Datasets are not components (Transition, Policy, RewardModel)
    - Dataset loaders have flexible signatures (via **kwargs)
    - Task type inference is a distinct concern from component lookup
    """
    
    # Internal storage
    _datasets: Dict[str, Callable[..., List[Dict]]] = {}
    _dataset_task_types: Dict[str, str] = {}
    _resources: Dict[str, Callable[..., Dict[str, Any]]] = {}
    
    @classmethod
    def register_dataset(cls, name: str, task_type: Optional[str] = None) -> Callable:
        """Decorator to register a dataset loader function.
        
        The registered function can have any signature - additional parameters
        are passed via **kwargs when calling load_dataset().
        
        Args:
            name: Dataset name (e.g., 'blocksworld', 'gsm8k', 'math500')
            task_type: Task type (optional, e.g., 'env_grounded', 'language_grounded', 'tool_use')
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If a dataset with the same name is already registered
        
        Example:
            @BenchmarkRegistry.register_dataset("math500", task_type="language_grounded")
            def load_math500(levels: list = None, split: str = "test"):
                # levels is passed via kwargs when invoked
                ...
        """
        def decorator(loader_func: Callable[..., List[Dict]]) -> Callable[..., List[Dict]]:
            if name in cls._datasets:
                raise ValueError(
                    f"Dataset '{name}' is already registered as {cls._datasets[name].__name__}. "
                    f"Cannot register {loader_func.__name__}. "
                    f"Use a different name or call BenchmarkRegistry.clear() first."
                )
            cls._datasets[name] = loader_func
            if task_type is not None:
                cls._dataset_task_types[name] = task_type
            logger.debug(f"Registered dataset '{name}' (task_type={task_type}): {loader_func.__name__}")
            return loader_func
        return decorator
    
    @classmethod
    def get_dataset(cls, name: str) -> Optional[Callable]:
        """Look up a registered dataset loader function.
        
        Returns the function itself, not the result. Caller invokes with **kwargs.
        
        Args:
            name: Dataset name
        
        Returns:
            The registered loader function, or None if not found
        """
        return cls._datasets.get(name)
    
    @classmethod
    def load_dataset(cls, name: str, **kwargs) -> List[Dict]:
        """Load dataset by name with optional parameters.
        
        Looks up the dataset loader in the registry and invokes it with kwargs.
        
        Args:
            name: Dataset name
            **kwargs: Additional parameters passed to the loader function
        
        Returns:
            List of dataset examples (typically List[Dict])
        
        Raises:
            KeyError: If no loader is found for the dataset name
        
        Note:
            Built-in datasets (blocksworld, gsm8k, etc.) must be registered first
            by importing their modules. For example:
                import lits_benchmark.blocksworld  # Registers blocksworld dataset
                data = load_dataset("blocksworld")
        
        Example:
            # Load with default parameters
            data = BenchmarkRegistry.load_dataset("blocksworld")
            
            # Load with custom parameters
            data = BenchmarkRegistry.load_dataset("math500", levels=[1, 2, 3])
        """
        loader = cls.get_dataset(name)
        if loader is not None:
            return loader(**kwargs)
        
        available = list(cls._datasets.keys())
        raise KeyError(
            f"Dataset '{name}' not found in registry. "
            f"Available datasets: {available}. "
            f"Did you forget to import the module containing @register_dataset('{name}')?"
        )
    
    @classmethod
    def infer_task_type(cls, dataset_name: str) -> str:
        """Infer task type from dataset name.
        
        Looks up the task_type registered via @register_dataset(name, task_type=...).
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Task type string: 'env_grounded', 'language_grounded', or 'tool_use'
        
        Raises:
            ValueError: If the dataset name is not recognized
        
        Example:
            task_type = BenchmarkRegistry.infer_task_type("blocksworld")
            # Returns "env_grounded"
        """
        if dataset_name in cls._dataset_task_types:
            return cls._dataset_task_types[dataset_name]
        
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Register it with @register_dataset('{dataset_name}', task_type='...') "
            f"and ensure the module is imported via --include."
        )
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names.
        
        Returns:
            List of all dataset names that have been registered.
        """
        return list(cls._datasets.keys())
    
    @classmethod
    def list_by_task_type(cls, task_type: str) -> List[str]:
        """List dataset names registered with a specific task_type.
        
        Args:
            task_type: The task type to filter by
        
        Returns:
            List of dataset names registered with the specified task_type.
        """
        return [name for name, tt in cls._dataset_task_types.items() if tt == task_type]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations.
        
        This is primarily useful for testing to reset the registry state
        between test cases.
        """
        cls._datasets.clear()
        cls._dataset_task_types.clear()
        cls._resources.clear()
        logger.debug("BenchmarkRegistry cleared")
    
    # --- Resource registry (tool-use benchmarks) ---
    
    @classmethod
    def register_resource(cls, name: str) -> Callable:
        """Decorator to register a tool-use resource loader function.
        
        The registered function should return a dict with:
        - "tools": list of BaseTool instances
        - "tool_context": str describing the tools
        
        Dataset examples are loaded separately via @register_dataset / load_dataset().
        
        Args:
            name: Resource name (e.g., 'mapeval-sql')
        
        Returns:
            Decorator function
        
        Example:
            @BenchmarkRegistry.register_resource("mapeval-sql")
            def load_mapeval_sql_resource(**kwargs):
                tools = build_tools("mapeval-sql", **kwargs)
                return {"tools": tools, "tool_context": "..."}
        """
        def decorator(loader_func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
            if name in cls._resources:
                raise ValueError(
                    f"Resource '{name}' is already registered as {cls._resources[name].__name__}. "
                    f"Cannot register {loader_func.__name__}."
                )
            cls._resources[name] = loader_func
            logger.debug(f"Registered resource '{name}': {loader_func.__name__}")
            return loader_func
        return decorator
    
    @classmethod
    def load_resource(cls, name: str, **kwargs) -> Dict[str, Any]:
        """Load a tool-use resource by name.
        
        Args:
            name: Resource name
            **kwargs: Additional parameters (e.g., db_host, db_port)
        
        Returns:
            Dict with "tools" and "tool_context"
        
        Raises:
            KeyError: If no resource loader is found
        """
        loader = cls._resources.get(name)
        if loader is not None:
            return loader(**kwargs)
        
        available = list(cls._resources.keys())
        raise KeyError(
            f"Resource '{name}' not found in registry. "
            f"Available resources: {available}. "
            f"Did you forget to import the module containing @register_resource('{name}')?"
        )
    
    @classmethod
    def has_resource(cls, name: str) -> bool:
        """Check if a resource is registered for the given name.
        
        Args:
            name: Resource name
        
        Returns:
            True if a resource loader is registered
        """
        return name in cls._resources


# Module-level function aliases for convenience
def register_dataset(name: str, task_type: Optional[str] = None) -> Callable:
    """Decorator to register a dataset loader function.
    
    This is a module-level alias for BenchmarkRegistry.register_dataset().
    
    Args:
        name: Dataset name (e.g., 'blocksworld', 'gsm8k', 'math500')
        task_type: Task type (optional, e.g., 'env_grounded', 'language_grounded', 'tool_use')
    
    Returns:
        Decorator function
    
    Example:
        from lits.benchmarks.registry import register_dataset
        
        @register_dataset("my_dataset", task_type="language_grounded")
        def load_my_dataset(split: str = "test"):
            ...
    """
    return BenchmarkRegistry.register_dataset(name, task_type)


def load_dataset(name: str, **kwargs) -> List[Dict]:
    """Load dataset by name with optional parameters.
    
    This is a module-level alias for BenchmarkRegistry.load_dataset().
    
    Args:
        name: Dataset name
        **kwargs: Additional parameters passed to the loader function
    
    Returns:
        List of dataset examples (typically List[Dict])
    
    Note:
        Built-in datasets must be registered first by importing their modules:
            import lits_benchmark.blocksworld  # Registers blocksworld
            data = load_dataset("blocksworld")
    
    Example:
        from lits.benchmarks.registry import load_dataset
        
        # Load with default parameters (after importing the dataset module)
        data = load_dataset("blocksworld")
        
        # Load with custom parameters
        data = load_dataset("math500", levels=[1, 2, 3])
    """
    return BenchmarkRegistry.load_dataset(name, **kwargs)


def infer_task_type(dataset_name: str) -> str:
    """Infer task type from dataset name.
    
    This is a module-level alias for BenchmarkRegistry.infer_task_type().
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Task type string: 'env_grounded', 'language_grounded', or 'tool_use'
    
    Example:
        from lits.benchmarks.registry import infer_task_type
        
        task_type = infer_task_type("blocksworld")
        # Returns "env_grounded"
    """
    return BenchmarkRegistry.infer_task_type(dataset_name)


def register_resource(name: str) -> Callable:
    """Decorator to register a tool-use resource loader function.
    
    This is a module-level alias for BenchmarkRegistry.register_resource().
    
    Args:
        name: Resource name (e.g., 'mapeval-sql', 'clue')
    
    Returns:
        Decorator function
    
    Example:
        from lits.benchmarks.registry import register_resource
        
        @register_resource("mapeval-sql")
        def load_mapeval_sql_resource(**kwargs):
            return {"tools": [...], "tool_context": "..."}
    """
    return BenchmarkRegistry.register_resource(name)


def load_resource(name: str, **kwargs) -> Dict[str, Any]:
    """Load a tool-use resource by name.
    
    This is a module-level alias for BenchmarkRegistry.load_resource().
    
    Args:
        name: Resource name
        **kwargs: Additional parameters (e.g., db_host, db_port)
    
    Returns:
        Dict with "tools" and "tool_context"
    """
    return BenchmarkRegistry.load_resource(name, **kwargs)


def has_resource(name: str) -> bool:
    """Check if a resource is registered for the given name.
    
    This is a module-level alias for BenchmarkRegistry.has_resource().
    
    Args:
        name: Resource name
    
    Returns:
        True if a resource loader is registered
    """
    return BenchmarkRegistry.has_resource(name)
