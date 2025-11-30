from typing import NamedTuple
from dataclasses import dataclass

from ..type_registry import register_type
from .base import ActionT, Step, State, TrajectoryState


@register_type
@dataclass
class SubQAStep(Step):
    """RAP-style sub-question step capturing the decomposition state."""

    sub_question: ActionT = ""
    sub_answer: str = ""
    confidence: float = 0.0

    def get_action(self) -> ActionT:
        return self.sub_question

    def get_answer(self) -> str:
        return self.sub_answer


@register_type
@dataclass
class ThoughtStep(Step):
    """General reasoning step used by concatenation-style policies."""

    action: ActionT = ""

    def get_action(self) -> ActionT:
        return self.action

    def get_answer(self) -> str:
        return self.action

class StepConcatState(TrajectoryState[ThoughtStep]):
    """State represented as a concatenation of steps."""

    def get_steps(self) -> list[ThoughtStep]:
        return self
    
    