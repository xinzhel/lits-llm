from dataclasses import dataclass, field
from typing import Any, Optional, List
from .base import Step, State, StringAction
from ..type_registry import register_type

EnvAction = StringAction

@register_type
@dataclass
class EnvStep(Step):
    """Environment interaction step - just the action taken."""
    
    action: Optional[EnvAction] = None
    reward: float = 0.0  # Optional: reward from this transition
    
    def get_action(self) -> EnvAction:
        return self.action
    
    def to_dict(self) -> dict:
        data = super().to_dict()  # This includes __type__ from base Step class
        if self.action is not None:
            data["action"] = str(self.action)
        if self.reward != 0.0:
            data["reward"] = self.reward
        return data
    
    @classmethod
    def from_dict(cls, payload: dict) -> "EnvStep":
        """Rebuild an EnvStep from serialized data."""
        action_str = payload.get("action")
        return cls(
            action=EnvAction(action_str) if action_str else None,
            reward=payload.get("reward", 0.0),
            error=payload.get("error"),
        )

@dataclass  
class EnvState(State):
    """
    State that represents an environment snapshot at a point in time.
    
    This state tracks both the current environment snapshot and the full history
    of steps taken to reach this state.
    
    Attributes:
        step_idx: Current step index (0-based)
        last_env_state: Previous environment state string
        env_state: Current environment state string
        buffered_action: Action that will be/was taken from this state
        history: List of EnvStep objects representing the trajectory
    """
    step_idx: int
    last_env_state: str
    env_state: str
    buffered_action: EnvAction
    history: Optional[List[EnvStep]] = None
    
    def __post_init__(self):
        """Initialize history if not provided."""
        if self.history is None:
            self.history = []
    
    def __len__(self):
        """Return the step index (which step we're at)."""
        return self.step_idx
    
    def __str__(self):
        """Return the environment state."""
        return self.env_state
    
    def add_step(self, step: EnvStep) -> None:
        """Add a step to the history."""
        self.history.append(step)
    
    def to_dict(self) -> dict:
        """Serialize the environment state snapshot with full history."""
        return {
            "step_idx": self.step_idx,
            "last_env_state": self.last_env_state,
            "env_state": self.env_state,
            "buffered_action": str(self.buffered_action) if self.buffered_action else None,
            "history": [step.to_dict() for step in self.history] if self.history else [],
        }
    
    @classmethod
    def from_dict(cls, payload: dict) -> "EnvState":
        """Rebuild an EnvState from serialized data."""
        from ..type_registry import TYPE_REGISTRY
        
        action_str = payload.get("buffered_action")
        
        # Deserialize history
        history = []
        for step_data in payload.get("history", []):
            if "__type__" in step_data:
                step_type_name = step_data["__type__"]
                step_class = TYPE_REGISTRY.get(step_type_name, EnvStep)
                step_data_without_type = {k: v for k, v in step_data.items() if k != "__type__"}
                if hasattr(step_class, "from_dict"):
                    step = step_class.from_dict(step_data_without_type)
                else:
                    step = step_class(**step_data_without_type)
            else:
                # Fallback for old format without __type__
                step = EnvStep.from_dict(step_data)
            history.append(step)
        
        return cls(
            step_idx=payload["step_idx"],
            last_env_state=payload["last_env_state"],
            env_state=payload["env_state"],
            buffered_action=EnvAction(action_str) if action_str else None,
            history=history,
        )
    
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
    
