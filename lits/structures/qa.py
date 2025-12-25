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

    def verb_step(self) -> str:
        """Return a string representation of the sub-question and answer."""
        return f"Sub-question: {self.sub_question}\nSub-answer: {self.sub_answer}"

    def to_messages(self) -> list[dict]:
        """Convert the step into chat messages (assistant asks, then answers)."""
        return [
            {"role": "assistant", "content": f"Sub-question: {self.sub_question}"},
            {"role": "assistant", "content": f"Sub-answer: {self.sub_answer}"}
        ]


@register_type
@dataclass
class ThoughtStep(Step):
    """General reasoning step used by concatenation-style policies."""

    action: ActionT = ""

    def get_action(self) -> ActionT:
        return self.action

    def get_answer(self) -> str:
        return self.action

    def verb_step(self) -> str:
        """Return a string representation of the thought/action."""
        return f"Thought: {self.action}"

    def to_messages(self) -> list[dict]:
        """Convert the step into a chat message."""
        return [{"role": "assistant", "content": str(self.action)}]
    