from typing import NamedTuple

from ...agents.tree_search.type_registry import register_type
from .base import ActionT, Step, State


@register_type
class SubQAStep(Step):
    """RAP-style sub-question step capturing the decomposition state."""

    sub_question: ActionT
    sub_answer: str
    confidence: float

    def get_action(self) -> ActionT:
        return self.sub_question

    def get_answer(self) -> str:
        return self.sub_answer


@register_type
class ThoughtStep(Step):
    """General reasoning step used by concatenation-style policies."""

    action: ActionT

    def get_action(self) -> ActionT:
        return self.action

    def get_answer(self) -> str:
        return self.action
