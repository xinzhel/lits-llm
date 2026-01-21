from dataclasses import dataclass, field
from typing import Any, Optional, List
from .base import Step, State, StringAction, TrajectoryState
from ..type_registry import register_type

EnvAction = StringAction

@register_type
@dataclass
class EnvStep(Step):
    """Environment interaction step - just the action taken."""
    
    action: Optional[EnvAction] = None
    next_state: Optional[str] = None # Optional: resulting state from this action
    
    def get_action(self) -> EnvAction:
        return self.action
    
    def to_dict(self) -> dict:
        data = super().to_dict()  # This includes __type__ from base Step class
        if self.action is not None:
            data["action"] = str(self.action)
        if self.next_state is not None:
            data["next_state"] = self.next_state
        return data
    
    @classmethod
    def from_dict(cls, payload: dict) -> "EnvStep":
        """Rebuild an EnvStep from serialized data."""
        action_str = payload.get("action")
        return cls(
            action=EnvAction(action_str) if action_str else None,
            next_state=payload.get("next_state"),
            error=payload.get("error"),
            terminate=payload.get("terminate", False),
        )

    def verb_step(self) -> str:
        return f"Action: {self.action}\nState: {self.next_state}"

    def to_messages(self) -> list[dict]:
        """Convert the step into chat messages (action and resulting state)."""
        messages = []
        if self.action is not None:
            messages.append({"role": "assistant", "content": f"Action: {self.action}"})
        if self.next_state is not None:
            messages.append({"role": "user", "content": f"Observation: {self.next_state}"})
        return messages

from ..type_registry import register_state

@register_state
@dataclass  
class EnvState(TrajectoryState[EnvStep]):
    """
    State that represents an environment snapshot at a point in time.
    
    This state tracks both the current environment snapshot and the full history
    of steps taken to reach this state.
    """
    init_state: str = ""
    
    @property
    def env_state(self) -> str:
        if len(self) > 0 and self[-1].next_state:
            return self[-1].next_state
        return self.init_state

    @property
    def last_env_state(self) -> str:
        if len(self) > 1 and self[-2].next_state:
            return self[-2].next_state
        return self.init_state

    @property
    def step_idx(self) -> int:
        return len(self)
    
    def to_dict(self) -> dict:
        """Serialize the environment state snapshot with full history."""
        data = super().to_dict()
        data["init_state"] = self.init_state
        return data
    
    @classmethod
    def from_dict(cls, payload: dict) -> "EnvState":
        """Rebuild an EnvState from serialized data."""
        state = super().from_dict(payload)
        state.init_state = payload.get("init_state", "")
        return state
    
    def save(self, path: str, query: str) -> None:
        """Persist the environment state with full history and originating query."""
        from pathlib import Path
        import json
        
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"query": query, "state": self.to_dict()}
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> tuple[str, "EnvState"]:
        """Load a saved environment state with full history and associated query."""
        from pathlib import Path
        import json
        
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "query" not in payload:
            raise ValueError("Checkpoint is missing the original query.")
        state_payload = payload.get("state", {})
        state = cls.from_dict(state_payload)
        return payload["query"], state

    def __repr__(self) -> str:
        """
        Show step count and truncated current env_state for clearer debugging.
        
        Default dataclass __repr__ only shows init_state, which is misleading in logs
        since it always displays the initial state regardless of trajectory progress.
        """
        current = self.env_state
        truncated = current[:100] + "..." if len(current) > 100 else current
        # Escape newlines for single-line repr
        truncated = truncated.replace("\n", "\\n")
        return f"EnvState(steps={len(self)}, env_state='{truncated}')"
    
