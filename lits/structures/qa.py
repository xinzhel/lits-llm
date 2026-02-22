from typing import NamedTuple
from dataclasses import dataclass

from ..type_registry import register_type
from .base import ActionT, Step, State, TrajectoryState, StringAction


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
    
    def to_dict(self) -> dict:
        """Serialize the step for checkpointing, including the action field."""
        data = super().to_dict()
        data["action"] = self.action
        return data

    @classmethod
    def verbalize_state(cls, question: str, state: list["ThoughtStep"]) -> str:
        """Verbalize ThoughtStep state into a prompt string.
        
        Delegates to verbalize_concat_state for consistency with existing code.
        
        Args:
            question: The original question
            state: List of ThoughtStep objects
        
        Returns:
            Formatted string for LLM prompt
        """
        from lits.components.utils import verbalize_concat_state
        return verbalize_concat_state(question, state)
    