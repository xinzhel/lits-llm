"""LiTS Components: Modular building blocks for reasoning agents.

This module provides the core components for building reasoning agents:
- Policy: Action generation
- Transition: State transitions and action execution
- Reward: Action/state evaluation
- Verbal Evaluator: LLM-based validation and assessment
"""

from .verbal_evaluator import SQLValidator

__all__ = [
    "SQLValidator",
]
