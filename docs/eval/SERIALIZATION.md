# Step and State Serialization in LiTS

This document describes the serialization and deserialization system for Step and State objects in the LiTS framework.

## Overview

LiTS uses a type registry system to serialize and deserialize Step and State objects to/from JSON format. This enables:
- Checkpointing and resuming search processes
- Saving terminal nodes for evaluation
- Persisting trajectories for analysis
- Cross-process communication

## Type Registries

### TYPE_REGISTRY (Steps and Actions)

The `TYPE_REGISTRY` in `lits/type_registry.py` stores Step subclasses for serialization/deserialization:

```python
from lits.type_registry import register_type

@register_type
@dataclass
class MyStep(Step):
    action: str
    observation: str = None
    
    def to_dict(self) -> dict:
        return {
            "__type__": self.__class__.__name__,
            "action": self.action,
            "observation": self.observation
        }
    
    @classmethod
    def from_dict(cls, payload: dict) -> "MyStep":
        return cls(
            action=payload["action"],
            observation=payload.get("observation")
        )
```

### STATE_REGISTRY (States)

The `STATE_REGISTRY` stores State subclasses for serialization/deserialization:

```python
from lits.type_registry import register_state

@register_state
class MyState(TrajectoryState):
    pass
```

## Serialization Process

### Step Serialization

Each Step class must implement `to_dict()` and `from_dict()` methods:

1. **Type Information**: `to_dict()` must include `"__type__": self.__class__.__name__`
2. **Registration**: Use `@register_type` decorator to register in TYPE_REGISTRY
3. **Optional Fields**: Handle None values gracefully

**Example (ToolUseStep)**:
```python
@register_type
@dataclass
class ToolUseStep(Step):
    think: str = ""
    action: Optional[ToolUseAction] = None
    observation: Optional[str] = None
    answer: Optional[str] = None
    assistant_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        data = {"__type__": self.__class__.__name__}
        if self.action is not None:
            data["action"] = str(self.action)
        if self.observation is not None:
            data["observation"] = self.observation
        if self.answer is not None:
            data["answer"] = self.answer
        if self.assistant_message is not None:
            data["assistant_message"] = self.assistant_message
        elif self.think:
            data["think"] = self.think
        if self.error is not None:
            data["error"] = self.error
        return data
    
    @classmethod
    def from_dict(cls, payload: dict) -> "ToolUseStep":
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
        return step
```

### State Serialization

#### TrajectoryState (Automatic)

`TrajectoryState` provides default `to_dict()` and `from_dict()` implementations:

1. **Automatic Implementation**: Inherits serialization from base class
2. **Type Information**: Automatically includes `"__type__"` and `"steps"` fields
3. **Registration**: Use `@register_state` decorator

**TrajectoryState Format**:
```json
{
  "__type__": "ToolUseState",
  "steps": [
    {
      "__type__": "ToolUseStep",
      "action": "...",
      "observation": "..."
    }
  ]
}
```

**Example (ToolUseState)**:
```python
@register_state
class ToolUseState(TrajectoryState[ToolUseStep]):
    """State container for tool-use traces."""
    
    def render_history(self) -> str:
        return "\n".join([step.verb_step() for step in self])
```

#### EnvState (Custom)

For snapshot-based states (non-trajectory), implement custom serialization:

```python
@register_state
@dataclass  
class EnvState(State):
    step_idx: int
    last_env_state: str
    env_state: str
    buffered_action: EnvAction
    history: Optional[List[EnvStep]] = None
    
    def to_dict(self) -> dict:
        return {
            "__type__": self.__class__.__name__,
            "step_idx": self.step_idx,
            "last_env_state": self.last_env_state,
            "env_state": self.env_state,
            "buffered_action": str(self.buffered_action) if self.buffered_action else None,
            "history": [step.to_dict() for step in self.history] if self.history else [],
        }
    
    @classmethod
    def from_dict(cls, payload: dict) -> "EnvState":
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
                step = EnvStep.from_dict(step_data)
            history.append(step)
        
        return cls(
            step_idx=payload["step_idx"],
            last_env_state=payload["last_env_state"],
            env_state=payload["env_state"],
            buffered_action=EnvAction(action_str) if action_str else None,
            history=history,
        )
```

## Deserialization Process

### _serialize_obj Function

The `_serialize_obj` function in `lits/structures/trace.py` handles serialization with priority order:

1. **Custom to_dict()**: Checks for `to_dict()` method first (for State/Step subclasses)
2. **Dataclasses**: Uses `asdict()` for dataclass objects
3. **NamedTuples**: Uses `_asdict()` for named tuples
4. **Collections**: Recursively serializes lists, tuples, and dicts
5. **Primitives**: Returns primitives as-is

```python
def _serialize_obj(obj):
    # Check for custom to_dict() method first
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    # Handle dataclasses
    if is_dataclass(obj):
        data = asdict(obj)
        data["__type__"] = type(obj).__name__
        return data
    # Handle collections and primitives...
```

### _deserialize_obj Function

The `_deserialize_obj` function handles deserialization with registry lookup:

1. **Check STATE_REGISTRY**: For objects with `"__type__"` (State types)
2. **Check TYPE_REGISTRY**: For Step objects with `"__type__"`
3. **Recursive**: Deserializes nested objects and lists

```python
def _deserialize_obj(payload):
    if isinstance(payload, dict) and "__type__" in payload:
        typ = payload.get("__type__")
        
        # Check STATE_REGISTRY first for State types
        from ..type_registry import STATE_REGISTRY
        if typ in STATE_REGISTRY:
            state_class = STATE_REGISTRY[typ]
            if hasattr(state_class, "from_dict"):
                return state_class.from_dict(payload)
        
        # Handle Step types from TYPE_REGISTRY
        payload_copy = dict(payload)
        payload_copy.pop("__type__")
        ctor = TYPE_REGISTRY.get(typ)
        if ctor is None:
            raise ValueError(f"Unknown step type '{typ}'. Ensure it is registered.")
        return ctor(**{k: _deserialize_obj(v) for k, v in payload_copy.items()})
    
    # Handle lists and primitives...
```

## Built-in Types

### Registered Step Types

- `ToolUseStep`: Tool use actions with observations/answers/errors
- `EnvStep`: Environment interaction steps with rewards
- `SubQAStep`: Sub-question and answer pairs for mathematical reasoning
- `ThoughtStep`: Sequential reasoning steps

### Registered State Types

- `TrajectoryState`: Base class for step sequences
- `ToolUseState`: Tool use trajectories
- `EnvState`: Environment state snapshots with history

## Usage Examples

### Serializing a TrajectoryState

```python
from lits.structures import ToolUseState, ToolUseStep, ToolUseAction

# Create state with steps
state = ToolUseState()
action = ToolUseAction('{"tool": "search", "query": "LiTS"}')
step = ToolUseStep(action=action, observation="Result")
state.append(step)

# Serialize
serialized = state.to_dict()
# Result: {"__type__": "ToolUseState", "steps": [...]}
```

### Deserializing a State

```python
from lits.structures.trace import _deserialize_obj

# Deserialize
deserialized_state = _deserialize_obj(serialized)
# Result: ToolUseState instance with ToolUseStep objects
```

### Saving and Loading States

```python
# Save state with query
state.save("checkpoint.json", query="What is LiTS?")

# Load state with query
query, loaded_state = ToolUseState.load("checkpoint.json")
```

### Node Serialization (Tree Search)

```python
from lits.agents.tree.node import SearchNode

# SearchNode automatically serializes state, action, and step
node_dict = node.to_dict()
# Includes: state, action, step (if present), is_terminal, fast_reward, etc.

# Deserialize node
node = SearchNode.from_dict(node_dict)
```

## Best Practices

### For Step Classes

1. **Always use @register_type**: Ensures discoverability
2. **Include __type__ in to_dict()**: Required for deserialization
3. **Handle None values**: Check for optional fields in from_dict()
4. **Use base class to_dict()**: Call `super().to_dict()` when extending

```python
@register_type
@dataclass
class MyStep(Step):
    field1: str
    field2: Optional[str] = None
    
    def to_dict(self) -> dict:
        data = super().to_dict()  # Gets __type__ and error from base
        data["field1"] = self.field1
        if self.field2 is not None:
            data["field2"] = self.field2
        return data
    
    @classmethod
    def from_dict(cls, payload: dict) -> "MyStep":
        return cls(
            field1=payload["field1"],
            field2=payload.get("field2"),
            error=payload.get("error")
        )
```

### For State Classes

1. **Use @register_state**: Enables automatic deserialization
2. **Inherit from TrajectoryState**: Gets to_dict/from_dict for free
3. **Override only if needed**: Default implementation handles most cases

```python
@register_state
class MyState(TrajectoryState[MyStep]):
    default_step = "MyStep"
```

### For Custom Serialization

1. **Check _serialize_obj first**: Handles to_dict() methods automatically
2. **Use TYPE_REGISTRY**: For dynamic type resolution
3. **Preserve type information**: Always include __type__ field

## Troubleshooting

### Common Errors

**"Unknown step type 'X'"**:
- Ensure class is decorated with `@register_type`
- Import the module containing the class before deserialization

**"Missing __type__ field"**:
- Ensure `to_dict()` includes `"__type__": self.__class__.__name__`

**"State serialization returns empty dict"**:
- Check if State class has proper `to_dict()` implementation
- Ensure State is registered with `@register_state`

### Debugging Tips

1. **Check registries**: Print `TYPE_REGISTRY` and `STATE_REGISTRY` contents
2. **Validate JSON**: Ensure serialized data contains `__type__` fields
3. **Test round-trip**: Serialize then deserialize to verify correctness

```python
# Debug registries
from lits.type_registry import TYPE_REGISTRY, STATE_REGISTRY
print("Steps:", list(TYPE_REGISTRY.keys()))
print("States:", list(STATE_REGISTRY.keys()))

# Test round-trip
original = MyStep(field1="value")
serialized = original.to_dict()
deserialized = MyStep.from_dict(serialized)
assert original == deserialized
```

## Migration Guide

### From v0.2.4 to v0.2.5

1. **State serialization format changed**:
   - Old: `[step1_dict, step2_dict, ...]`
   - New: `{"__type__": "StateClass", "steps": [...]}`

2. **Add @register_state decorators**:
   ```python
   # Before
   class MyState(TrajectoryState):
       pass
   
   # After
   @register_state
   class MyState(TrajectoryState):
       pass
   ```

3. **Update imports**:
   ```python
   from lits.type_registry import register_state
   ```

4. **Backward compatibility**: TrajectoryState.from_dict() handles both old (list) and new (dict) formats

## Architecture Notes

### Why Two Registries?

- **TYPE_REGISTRY**: For Step and Action types (used during step deserialization)
- **STATE_REGISTRY**: For State types (used during state deserialization)

This separation allows:
- Clear distinction between step-level and state-level serialization
- Different deserialization logic for trajectory vs snapshot states
- Better error messages and type safety

### Serialization Priority

The `_serialize_obj` function checks in this order:
1. Custom `to_dict()` method (highest priority)
2. Dataclass with `asdict()`
3. NamedTuple with `_asdict()`
4. Collections (list, tuple, dict)
5. Primitives (str, int, float, bool, None)

This ensures custom serialization logic is always respected.

### Node Serialization

SearchNode serialization includes:
- `state`: Full state object (serialized via _serialize_obj)
- `action`: Action taken to reach this node
- `step`: Full step object (v0.2.5+, includes action + observation + answer)
- Metadata: `is_terminal`, `fast_reward`, `bn_score`, etc.

The `step` attribute was added in v0.2.5 to preserve complete step information during tree search, enabling better evaluation and analysis.

### When to Use @dataclass Decorator

**Q: Should I add @dataclass decorator for State subclasses?**

**A: It depends on the State type:**

#### TrajectoryState Subclasses: NO @dataclass

```python
@register_state
class ToolUseState(TrajectoryState[ToolUseStep]):
    """State container for tool-use traces."""
    pass
```

**Why not?**
- TrajectoryState inherits from `list` and gets `__init__()` from list
- No instance fields to manage (only class-level `default_step`)
- Always instantiated empty: `ToolUseState()`
- Adding `@dataclass` can cause conflicts with list inheritance

#### Snapshot-Based State Subclasses: YES @dataclass

```python
@register_state
@dataclass  
class EnvState(State):
    step_idx: int
    last_env_state: str
    env_state: str
    buffered_action: EnvAction
    history: Optional[List[EnvStep]] = None
```

**Why yes?**
- Has explicit instance fields that need initialization
- Needs automatic `__init__()`, `__repr__()`, `__eq__()` generation
- Doesn't inherit from list or other collection types

#### What Does @dataclass Provide?

The `@dataclass` decorator automatically generates:

1. **`__init__()`**: Constructor accepting all fields as parameters
2. **`__repr__()`**: String representation for debugging
3. **`__eq__()`**: Equality comparison based on field values
4. **Field type hints**: Enforces type annotations
5. **Default values**: Handles default field values correctly
6. **`__post_init__()`**: Hook for custom initialization logic

**Example - Manual vs @dataclass:**

```python
# WITHOUT @dataclass - manual implementation:
class EnvState(State):
    def __init__(self, step_idx: int, last_env_state: str, env_state: str, 
                 buffered_action: EnvAction, history: Optional[List[EnvStep]] = None):
        self.step_idx = step_idx
        self.last_env_state = last_env_state
        self.env_state = env_state
        self.buffered_action = buffered_action
        self.history = history if history is not None else []
    
    def __repr__(self):
        return f"EnvState(step_idx={self.step_idx}, ...)"
    
    def __eq__(self, other):
        if not isinstance(other, EnvState):
            return False
        return (self.step_idx == other.step_idx and ...)

# WITH @dataclass - Python generates all that:
@dataclass
class EnvState(State):
    step_idx: int
    last_env_state: str
    env_state: str
    buffered_action: EnvAction
    history: Optional[List[EnvStep]] = None
```

#### Summary

- **TrajectoryState subclasses**: Don't use `@dataclass` (inherit list behavior)
- **Custom State subclasses with fields**: Use `@dataclass` (need field management)
- **Empty marker subclasses**: No `@dataclass` needed (no fields to manage)
