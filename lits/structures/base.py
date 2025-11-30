from typing import Generic, List, Tuple, TypeVar, Union
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class Step:
    # error attribute to capture any errors during step generation
    error: Union[None, str] = None
    
    def to_dict(self) -> dict:
        """Serialize the step for checkpointing."""
        data = {"__type__": self.__class__.__name__}

        if self.error is not None:
            data["error"] = self.error
        return data

@dataclass
class State:
    """Base state class - marker for all state types. 
    The following two states are defined to distinguish between trajectory-based states (that accumulate steps) and 
    environment snapshot states (that track step index). """
    pass


@dataclass
class TrajectoryState(State, list):
    """State that accumulates steps as a trajectory. Supports `len()` to return number of accumulated steps"""
    def get_steps(self) -> list["TrajectoryState"]:
        return self
    
    def to_dict(self) -> list[dict]:
        """Serialize the entire state as a list of steps."""
        return [step.to_dict() for step in self]

    @classmethod
    def from_dict(cls, payload: list[dict]) -> "TrajectoryState":
        """
        Create a TrajectoryState from serialized steps using the type registry.
        
        Each step dict must contain a "__type__" key that maps to a registered Step subclass.
        """
        from ..type_registry import TYPE_REGISTRY
        
        state = cls()
        for step_data in payload:
            # Extract the type information
            if "__type__" not in step_data:
                raise ValueError(
                    f"Step data missing '__type__' field. Ensure Step.to_dict() includes type information. "
                    f"Got: {step_data}"
                )
            
            step_type_name = step_data["__type__"]
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
    
    def save(self, path: str, query: str) -> None:
        """Persist the state and originating query for later resumption."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"query": query, "steps": self.to_dict()}
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> tuple[str, "TrajectoryState"]:
        """Load a saved state and associated query."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "query" not in payload:
            raise ValueError("Checkpoint is missing the original query.")
        steps_payload = payload.get("steps", [])
        state = cls.from_dict(steps_payload)
        return payload["query"], state


@dataclass
class Action:
    """Base action marker class. Actions represent operations that can be taken in a state."""
    pass

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
