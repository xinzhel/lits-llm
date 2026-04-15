""" Tool use step and state representations."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Optional
from .base import Step, State, Action, TrajectoryState, StringAction
from ..type_registry import register_type
from ..utils import make_tag_extractor

logger = logging.getLogger(__name__)

_DEFAULT_THINK_EXTRACTOR = make_tag_extractor("think")
_DEFAULT_ACTION_EXTRACTOR = make_tag_extractor("action")
_DEFAULT_OBSERVATION_EXTRACTOR = make_tag_extractor("observation")
_DEFAULT_ANSWER_EXTRACTOR = make_tag_extractor("answer")


def _extract_first(extractor: Callable[[str], list], message: str):
    """Return the first non-empty value produced by an extractor."""
    try:
        results = extractor(message)
    except Exception as exc:
        logger.warning(
            "Extractor %s failed on message: %s (type=%s)",
            extractor,
            message,
            type(message),
        )
        raise exc
    if not results or results[0] is None:
        return None
    return results[0].strip()


@register_type
class ToolUseAction(StringAction):
    """Action type for tool use - wraps a JSON string representing a tool call."""
    pass


@register_type
@dataclass
class BaseToolUseStep(Step):
    """Base step for tool use — shared fields for both text-based and native tool use.

    Subclasses:
    - ``ToolUseStep``: text-based (XML tag parsing, ``assistant_message: str``)
    - ``NativeToolUseStep``: native tool use (structured dict, ``assistant_message_dict: dict``)
    """

    action: Optional[ToolUseAction] = None
    observation: Optional[str] = None
    answer: Optional[str] = None

    def get_action(self):
        return self.action

    def get_observation(self):
        return self.observation

    def get_answer(self):
        return self.answer

    def to_dict(self) -> dict:
        """Serialize shared fields for checkpointing."""
        data = {"__type__": self.__class__.__name__}
        if self.action is not None:
            data["action"] = str(self.action)
        if self.observation is not None:
            data["observation"] = self.observation
        if self.answer is not None:
            data["answer"] = self.answer
        if self.error is not None:
            data["error"] = self.error
        return data


@register_type
@dataclass
class ToolUseStep(BaseToolUseStep):
    """Text-based ReAct step: thought, tool invocation via XML tags, observation, and answer.

    Uses XML tag extractors to parse ``<think>``, ``<action>``, ``<answer>`` from
    assistant text. This is the original text-based tool use implementation.
    """

    think: str = ""
    assistant_message: Optional[str] = None

    _think_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_THINK_EXTRACTOR
    _action_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_ACTION_EXTRACTOR
    _observation_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_OBSERVATION_EXTRACTOR
    _answer_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_ANSWER_EXTRACTOR

    exclude_think_when_verb: ClassVar[bool] = False # whether to exclude think when verbalizing the step


    def _identity_key(self) -> tuple:
        if self.assistant_message:
            return (self.assistant_message,)
        return (self.think, self.action, self.answer)

    def __hash__(self) -> int:
        return hash(self._identity_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolUseStep):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def _verb_assistant_content(self) -> str:
        """Return the assistant-generated portion (<think>, <action>, <answer>) for this step."""
        if self.assistant_message and not self.exclude_think_when_verb:
            return self.assistant_message.strip()
        parts = []
        if self.think and not self.exclude_think_when_verb:
            parts.append(f"<think>\n{self.think.strip()}\n</think>")
        if self.action:
            parts.append(f"<action>\n{str(self.action).strip()}\n</action>")
        if self.answer:
            parts.append(f"<answer>\n{self.answer.strip()}\n</answer>")
        return "\n".join(parts).strip()

    def _verb_observation_content(self) -> Optional[str]:
        """Return the observation message formatted as a user turn."""
        if self.observation is None:
            return None
        obs_content = self.observation if isinstance(self.observation, str) else str(self.observation)
        return f"<observation>\n{obs_content.strip()}\n</observation>"

    def verb_step(self) -> str:
        """Verbalize the step into text format."""
        text = ""
        assistant_text = self._verb_assistant_content()
        if assistant_text:
            text += assistant_text.rstrip() + "\n"
        observation_text = self._verb_observation_content()
        if observation_text:
            text += observation_text.rstrip() + "\n"
        return text.strip()
    
    def to_messages(self) -> list[dict]:
        """Convert the step into a list of chat messages."""
        messages = []
        assistant_text = self._verb_assistant_content()
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        observation_text = self._verb_observation_content()
        if observation_text:
            messages.append({"role": "user", "content": observation_text})
        return messages

    def to_dict(self) -> dict:
        """Serialize the step for checkpointing."""
        data = super().to_dict()
        # save either assistant_message or think, but not both, because assistant_message can be used to reconstruct think
        if self.assistant_message is not None:
            data["assistant_message"] = self.assistant_message
        elif self.think:
            data["think"] = self.think
        return data

    @classmethod
    def configure_extractors(
        cls,
        *,
        think_extractor: Optional[Callable[[str], list]] = None,
        action_extractor: Optional[Callable[[str], list]] = None,
        observation_extractor: Optional[Callable[[str], list]] = None,
        answer_extractor: Optional[Callable[[str], list]] = None,
    ) -> None:
        """Override how think/action/observation/answer spans are extracted from assistant text."""
        if think_extractor is not None:
            cls._think_extractor = think_extractor
        if action_extractor is not None:
            cls._action_extractor = action_extractor
        if observation_extractor is not None:
            cls._observation_extractor = observation_extractor
        if answer_extractor is not None:
            cls._answer_extractor = answer_extractor

    @classmethod
    def from_dict(cls, payload: dict) -> "ToolUseStep":
        """Rebuild a step from serialized data."""
        assistant_message = payload.get("assistant_message")
        if assistant_message:
            step = cls.from_assistant_message(assistant_message)
        else:
            action_str = payload.get("action")
            step = cls(
                think=payload.get("think", ""),
                action=ToolUseAction(action_str) if action_str else None,
                answer=payload.get("answer"),
                error=payload.get("error"),
            )
        step.observation = payload.get("observation")
        # Override answer with saved value if present (e.g., resolve_answer
        # may have replaced "#3" with resolved entity names post-run)
        if "answer" in payload and payload["answer"] is not None:
            step.answer = payload["answer"]
        return step

    @classmethod
    def from_assistant_message(cls, message: str) -> "ToolUseStep":
        """Parse a raw assistant turn into a ToolUseStep using the configured extractors."""
        message = message.strip()
        think = _extract_first(cls._think_extractor, message) 
        action_str = _extract_first(cls._action_extractor, message)
        # observation = _extract_first(cls._observation_extractor, message) # observation is not parsed from assistant message, but from tool execution result
        observation = None
        answer = _extract_first(cls._answer_extractor, message)
        return cls(
            think=think,
            action=ToolUseAction(action_str) if action_str else None,
            observation=observation,
            answer=answer,
            assistant_message=message,
        )


@register_type
@dataclass
class NativeToolUseStep(BaseToolUseStep):
    """Step for native tool use API (structured dict, not text parsing).

    Uses LLM's raw assistant message (``assistant_message_dict``) directly
    instead of XML tag parsing. Supports multi-turn conversation via
    ``user_message`` field.

    Fields:
        assistant_message_dict: LLM's raw assistant message as provider-specific dict.
            Stored as-is from ``ToolCallOutput.raw_message`` and replayed directly
            in ``_build_messages()`` — no manual reconstruction needed.
        user_message: Pure user turn text for multi-turn conversation history.
            A step with only ``user_message`` set represents a user message
            (no tool call, no answer).
    """

    assistant_message_dict: Optional[dict] = None
    user_message: Optional[str] = None
    tool_use_id: Optional[str] = None  # Provider-agnostic tool call ID (from ToolCall.id)

    def to_messages(self) -> list[dict]:
        """Convert step to Converse API message list.

        Returns 0-2 messages depending on step type:
        - user_message step: 1 user message
        - tool call step: 1 assistant message (from assistant_message_dict)
          (tool result is built by Policy using base_model.format_tool_result)
        - answer step: 1 assistant message with text
        """
        messages = []
        if self.user_message:
            messages.append({"role": "user", "content": [{"text": self.user_message}]})
        if self.assistant_message_dict:
            messages.append(self.assistant_message_dict)
        elif self.answer:
            messages.append({"role": "assistant", "content": [{"text": self.answer}]})
        return messages

    def verb_step(self) -> str:
        """Verbalize for logging."""
        if self.user_message:
            return f"[USER] {self.user_message}"
        if self.action:
            return f"[TOOL_CALL] {self.action}"
        if self.answer:
            return f"[ANSWER] {self.answer[:100]}..."
        return "[EMPTY STEP]"

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        data = super().to_dict()
        if self.assistant_message_dict is not None:
            data["assistant_message_dict"] = self.assistant_message_dict
        if self.user_message is not None:
            data["user_message"] = self.user_message
        if self.tool_use_id is not None:
            data["tool_use_id"] = self.tool_use_id
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "NativeToolUseStep":
        """Rebuild from serialized data."""
        action_str = payload.get("action")
        return cls(
            action=ToolUseAction(action_str) if action_str else None,
            observation=payload.get("observation"),
            answer=payload.get("answer"),
            error=payload.get("error"),
            assistant_message_dict=payload.get("assistant_message_dict"),
            user_message=payload.get("user_message"),
            tool_use_id=payload.get("tool_use_id"),
        )


from ..type_registry import register_state

@register_state
class ToolUseState(TrajectoryState[ToolUseStep]):
    """State container for tool-use traces; each entry is a ToolUseStep."""

    def get_final_answer(self):
        """Return the answer from the latest step if available."""
        if not self:
            return None
        last = self[-1]
        if getattr(last, "answer", None) is not None:
            return last.answer
        if getattr(last, "assistant_message", None):
            # extractor = last._answer_extractor #  WRONG since Python binding _answer_extractor as a method when accessed on a ToolUseStep instance.
            extractor = type(last)._answer_extractor 
            return _extract_first(extractor, last.assistant_message)
        return None
