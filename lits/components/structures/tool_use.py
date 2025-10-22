import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Optional

from lits.agents.search.type_registry import register_type
from lits.agents.utils import make_tag_extractor

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
@dataclass
class ToolUseStep:
    """Single ReAct step capturing thought, tool invocation, observation, and answer."""

    think: str = ""
    action: Optional[str] = None
    observation: Optional[str] = None
    answer: Optional[str] = None
    assistant_message: Optional[str] = None

    _think_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_THINK_EXTRACTOR
    _action_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_ACTION_EXTRACTOR
    _observation_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_OBSERVATION_EXTRACTOR
    _answer_extractor: ClassVar[Callable[[str], list]] = _DEFAULT_ANSWER_EXTRACTOR


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

    def get_action(self):
        return self.action

    def get_observation(self):
        return self.observation

    def get_answer(self):
        return self.answer

    def assistant_content(self) -> str:
        """Return the assistant-visible portion (<think>, <action>, <answer>) for this step."""
        if self.assistant_message:
            return self.assistant_message.strip()
        parts = []
        if self.think:
            parts.append(f"<think>\n{self.think.strip()}\n</think>")
        if self.action:
            parts.append(f"<action>\n{self.action.strip()}\n</action>")
        if self.answer:
            parts.append(f"<answer>\n{self.answer.strip()}\n</answer>")
        return "\n".join(parts).strip()

    def observation_message(self) -> Optional[str]:
        """Return the observation message formatted as a user turn."""
        if self.observation is None:
            return None
        obs_content = self.observation if isinstance(self.observation, str) else str(self.observation)
        return f"<observation>\n{obs_content.strip()}\n</observation>"

    def verb_step(self) -> str:
        """Verbalize the step into text format."""
        text = ""
        assistant_text = self.assistant_content()
        if assistant_text:
            text += assistant_text.rstrip() + "\n"
        observation_text = self.observation_message()
        if observation_text:
            text += observation_text.rstrip() + "\n"
        return text.strip()

    def to_dict(self) -> dict:
        """Serialize the step for checkpointing."""
        data = {}
        if self.action is not None:
            data["action"] = self.action
        if self.observation is not None:
            data["observation"] = self.observation
        if self.answer is not None:
            data["answer"] = self.answer
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
            step = cls(
                think=payload.get("think", ""),
                action=payload.get("action"),
                answer=payload.get("answer"),
            )
        step.observation = payload.get("observation")
        
        return step

    @classmethod
    def from_assistant_message(cls, message: str) -> "ToolUseStep":
        """Parse a raw assistant turn into a ToolUseStep using the configured extractors."""
        message = message.strip()
        think = _extract_first(cls._think_extractor, message) 
        action = _extract_first(cls._action_extractor, message)
        # observation = _extract_first(cls._observation_extractor, message) # observation is not parsed from assistant message, but from tool execution result
        observation = None
        answer = _extract_first(cls._answer_extractor, message)
        return cls(
            think=think,
            action=action,
            observation=observation,
            answer=answer,
            assistant_message=message,
        )


class ToolUseState(list[ToolUseStep]):
    """State container for tool-use traces; each entry is a ToolUseStep."""

    def render_history(self) -> str:
        return "\n".join([step.verb_step() for step in self])

    def to_dict(self) -> list[dict]:
        """Serialize the entire state as a list of steps."""
        return [step.to_dict() for step in self]

    @classmethod
    def from_dict(cls, payload: list[dict]) -> "ToolUseState":
        """Create a ToolUseState from serialized steps."""
        state = cls()
        for step_data in payload:
            state.append(ToolUseStep.from_dict(step_data))
        return state

    def to_messages(self, initial_query: str) -> list[dict]:
        """Reconstruct the chat message sequence from the stored steps."""
        messages = [{"role": "user", "content": initial_query}]
        for step in self:
            assistant_text = step.assistant_content()
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            logger.debug(">>>>>>>>>> ToolUseStep (Begin) <<<<<<<<<< ")
            logger.debug(f"Assistant text: {assistant_text}")
            logger.debug(">>>>>>>>>> ToolUseStep ( End ) <<<<<<<<<< ")
            
            observation_text = step.observation_message()
            if observation_text:
                messages.append({"role": "user", "content": observation_text})
        return messages

    def get_final_answer(self):
        """Return the answer from the latest step if available."""
        if not self:
            return None
        last = self[-1]
        if last.answer is not None:
            return last.answer
        if last.assistant_message:
            # extractor = last._answer_extractor #  WRONG since Python binding _answer_extractor as a method when accessed on a ToolUseStep instance.
            extractor = type(last)._answer_extractor 
            return _extract_first(extractor, last.assistant_message)
        return None

    def save(self, path: str, query: str) -> None:
        """Persist the state and originating query for later resumption."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"query": query, "steps": self.to_dict()}
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> tuple[str, "ReactState"]:
        """Load a saved state and associated query."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "query" not in payload:
            raise ValueError("Checkpoint is missing the original query.")
        steps_payload = payload.get("steps", [])
        state = cls.from_dict(steps_payload)
        return payload["query"], state
