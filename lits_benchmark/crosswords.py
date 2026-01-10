"""Crosswords benchmark module for mini crossword puzzles.

This module provides dataset loading for the crosswords env_grounded task.
The actual CrosswordsTransition implementation is in lits/components/transition/crosswords.py.

Note: This is a placeholder implementation. The full crosswords domain will be
implemented in the env-grounded-reusability spec.
"""

from typing import List, Dict
from lits.benchmarks.registry import register_dataset


@register_dataset("crosswords", task_type="env_grounded")
def load_crosswords(data_file: str = None, **kwargs) -> List[Dict]:
    """Load mini crossword puzzles.
    
    Args:
        data_file: Path to the crossword data file (e.g., "mini0505.json").
                   If None, returns a placeholder example.
        **kwargs: Additional arguments for future extensibility.
    
    Returns:
        List of dictionaries with 'init_state_str' and 'query_or_goals' fields.
    
    Note:
        This is a placeholder implementation. The full crosswords dataset loading
        will be implemented in the env-grounded-reusability spec.
    """
    # Placeholder implementation - returns empty list until full implementation
    # The actual implementation will load from mini0505.json or similar
    if data_file is None:
        # Return placeholder example for testing
        return []
    
    # TODO: Implement actual crossword loading from data_file
    # Expected format:
    # [
    #     {
    #         "init_state_str": "_ _ _ _ _\n_ _ _ _ _\n_ _ _ _ _\n_ _ _ _ _\n_ _ _ _ _",
    #         "query_or_goals": "H1: ..., H2: ..., V1: ..., V2: ..."
    #     },
    #     ...
    # ]
    raise NotImplementedError(
        f"Crosswords data loading from file '{data_file}' not yet implemented. "
        "This will be added in the env-grounded-reusability spec."
    )
