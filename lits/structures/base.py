from typing import Generic, List, Tuple, TypeVar, Union
from dataclasses import dataclass
import json
from pathlib import Path
from abc import ABC, abstractmethod

@dataclass
class Step(ABC):
    # error attribute to capture any errors during step generation
    error: Union[None, str] = None
    # terminate flag: if True, this error should stop trajectory generation
    terminate: bool = False
    
    def to_dict(self) -> dict:
        """Serialize the step for checkpointing."""
        data = {"__type__": self.__class__.__name__}

        if self.error is not None:
            data["error"] = self.error
        if self.terminate:
            data["terminate"] = self.terminate
        return data
    
    def verb_step(self) -> str:
        """Return a string representation of the step for logging."""
        raise NotImplementedError("Subclasses must implement verb_step method.")
    
    def to_messages(self) -> list[dict]:
        """Convert the step into a list of chat messages."""
        raise NotImplementedError("Subclasses must implement to_messages method.")

@dataclass
class State:
    """Base state class - marker for all state types. 
    The following two states are defined to distinguish between trajectory-based states (that accumulate steps) and 
    environment snapshot states (that track step index). """
    pass


from ..type_registry import register_state, register_type

@register_state
@dataclass
class TrajectoryState(State, list):
    """State that accumulates steps as a trajectory. Supports `len()` to return number of accumulated steps"""
    
    default_step: Step = None
    
    def render_history(self) -> str:
        return "\n".join([step.verb_step() for step in self])
    
    
    def to_messages(self, initial_query: str) -> list[dict]:
        """Reconstruct the chat message sequence from the stored steps."""
        messages = [{"role": "user", "content": initial_query}]
        for step in self:
            step_messages = step.to_messages()
            messages.extend(step_messages)
        return messages
    
    def get_steps(self) -> list["Step"]:
        return self
    
    def to_dict(self) -> dict:
        """Serialize the entire state as a dict with type information."""
        return {
            "__type__": self.__class__.__name__,
            "steps": [step.to_dict() for step in self]
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrajectoryState":
        """
        Create a TrajectoryState from serialized dict with type information.
        
        Each step dict must contain a "__type__" key that maps to a registered Step subclass.
        """
        from ..type_registry import TYPE_REGISTRY
        
        # Handle both old format (list) and new format (dict with steps)
        if isinstance(payload, list):
            steps_data = payload
        else:
            steps_data = payload.get("steps", [])
        
        state = cls()
        for step_data in steps_data:
            step_type_name = step_data.get("__type__", cls.default_step)
            # Extract the type information
            if step_type_name is None:
                raise ValueError(
                    f"Step data missing '__type__' field. Ensure Step.to_dict() includes type information. "
                    f"Got: {step_data}"
                )
            
            
            step_class = TYPE_REGISTRY.get(step_type_name)
            
            if step_class is None:
                raise ValueError(
                    f"Unknown step type '{step_type_name}'. Ensure it is registered via @register_type decorator. "
                    f"Available types: {list(TYPE_REGISTRY.keys())}"
                )
            
            # Create a copy without __type__ for the step's from_dict method
            step_data_without_type = {k: v for k, v in step_data.items() if k != "__type__"}
            
            # Deserialize using the appropriate step class
            if hasattr(step_class, "from_dict"):
                step = step_class.from_dict(step_data_without_type)
            else:
                # Fallback: try to instantiate directly
                step = step_class(**step_data_without_type)
            
            state.append(step)
        
        return state
    
    def save(self, path: str, query: str, **kwargs) -> None:
        """Persist the state and originating query for later resumption."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.to_dict()
        
        # Build state dictionary including kwargs
        state_dict = {
            **self.to_dict(),
            "query": query,
            **kwargs,  # merge in any additional fields
        }
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> tuple[str, "TrajectoryState"]:
        """Load a saved state and associated query."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "query" not in payload:
            raise ValueError("Checkpoint is missing the original query.")
        query = payload["query"]
        state = cls.from_dict(payload)
        return query, state


@dataclass
class Action:
    """Base action marker class. Actions represent operations that can be taken in a state."""
    pass

@register_type
@dataclass
class StringAction(Action):
    action_str: str
    
    def __str__(self):
        return self.action_str

StateT = TypeVar("StateT", bound=State) # 泛型类型变量：必须是 State 或其子类
ActionT = TypeVar("ActionT", bound=Action)
StepT = TypeVar("StepT", bound=Step)

@dataclass
class Trace(Generic[StepT]):
    steps: List[StepT]

    def add(self, step: StepT):
        self.steps.append(step)
