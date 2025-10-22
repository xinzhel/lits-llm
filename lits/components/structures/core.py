from typing import NamedTuple

from lits.agents.search.type_registry import register_type
from lits.components.structures.base import PolicyAction


@register_type
class SubQAStep(NamedTuple):
    """RAP-style sub-question step capturing the decomposition state."""

    sub_question: PolicyAction
    sub_answer: str
    confidence: float

    def get_action(self) -> PolicyAction:
        return self.sub_question

    def get_answer(self) -> str:
        return self.sub_answer


@register_type
class ThoughtStep(NamedTuple):
    """General reasoning step used by concatenation-style policies."""

    action: PolicyAction

    def get_action(self) -> PolicyAction:
        return self.action

    def get_answer(self) -> str:
        return self.action
