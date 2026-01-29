"""Transition components for LiTS.

This module provides transition (world model) implementations for different task types:
- EnvGroundedTransition: Base class for env_grounded tasks (BlocksWorld, Crosswords, etc.)
- BlocksWorldTransition: BlocksWorld planning domain implementation
- ConcatTransition: Language-grounded transition for sequential reasoning
- ToolUseTransition: Tool-use transition for ReAct-style agents

For RAP (Reasoning via Planning) transition, import from external formulation:
    from lits_benchmark.formulations.rap import RAPTransition
"""

from .env_grounded import EnvGroundedTransition
from .blocksworld import BlocksWorldTransition
from .concat import ConcatTransition
from .tool_use import ToolUseTransition

__all__ = [
    "EnvGroundedTransition",
    "BlocksWorldTransition",
    "ConcatTransition",
    "ToolUseTransition",
]
