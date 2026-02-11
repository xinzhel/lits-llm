"""Transition components for LiTS.

This module provides transition (world model) implementations for different task types:
- EnvGroundedTransition: Base class for env_grounded tasks (BlocksWorld, Crosswords, etc.)
- ConcatTransition: Language-grounded transition for sequential reasoning
- ToolUseTransition: Tool-use transition for ReAct-style agents

Domain-specific transitions are in lits_benchmark (external package):
    from lits_benchmark.blocksworld import BlocksWorldTransition
    from lits_benchmark.crosswords import CrosswordsTransition

For RAP (Reasoning via Planning) transition:
    from lits_benchmark.formulations.rap import RAPTransition
"""

from .env_grounded import EnvGroundedTransition
from .concat import ConcatTransition
from .tool_use import ToolUseTransition

__all__ = [
    "EnvGroundedTransition",
    "ConcatTransition",
    "ToolUseTransition",
]
