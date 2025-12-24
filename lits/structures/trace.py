"""Utilities for logging, replaying, and serializing heterogeneous LangTree states."""

import json
import logging
from typing import Iterable
from dataclasses import is_dataclass, asdict

from ..type_registry import TYPE_REGISTRY

def _serialize_obj(obj):
    """
    Serialize an object to a JSON-safe dict for polymorphic subclass serialization.
    
    This function is designed for types with multiple subclasses (e.g., Step has
    ThoughtStep, SubQAStep, ToolUseStep; State has TrajectoryState, QAState).
    It adds a ``__type__`` field that records the concrete class name, enabling
    ``_deserialize_obj`` to reconstruct the correct subclass during deserialization.
    
    When to use this function:
        - Types with multiple polymorphic subclasses that need runtime type dispatch
        - Types registered in TYPE_REGISTRY or STATE_REGISTRY
    
    When NOT to use this function:
        - Types with a single implementation (e.g., TrajectoryKey) - use explicit
          serialization to avoid TYPE_REGISTRY dependency and keep code simpler
    
    Args:
        obj: The object to serialize.
    
    Returns:
        A JSON-safe representation with ``__type__`` field for polymorphic types.
    """
    # Check for custom to_dict() method first (for State subclasses)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    # Handle dataclasses (like ThoughtStep, SubQAStep)
    if is_dataclass(obj):
        data = asdict(obj)
        data["__type__"] = type(obj).__name__
        return data
    # Handle NamedTuples
    if hasattr(obj, "_asdict"):
        data = obj._asdict()
        data["__type__"] = type(obj).__name__
        return data
    # list/tuple?
    if isinstance(obj, (list, tuple)):
        return [_serialize_obj(item) for item in obj]
    # dict?
    if isinstance(obj, dict):
        return {key: _serialize_obj(value) for key, value in obj.items()}
    # primitives?
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # raise TypeError(f"Cannot JSON-serialize {type(obj)}")
    return obj


def _deserialize_obj(payload):
    """
    Deserialize a JSON payload back to the original polymorphic object.
    
    This function reconstructs polymorphic types by looking up the concrete class
    from the ``__type__`` field in TYPE_REGISTRY or STATE_REGISTRY. The registries
    must contain the class registered via ``@register_type`` or ``@register_state``.
    
    When to use this function:
        - Types with multiple polymorphic subclasses that need runtime type dispatch
        - Types registered in TYPE_REGISTRY or STATE_REGISTRY
    
    When NOT to use this function:
        - Types with a single implementation (e.g., TrajectoryKey) - use explicit
          deserialization to avoid TYPE_REGISTRY dependency and keep code simpler
    
    Args:
        payload: The JSON payload to deserialize.
    
    Returns:
        The reconstructed object with the correct concrete type.
    
    Raises:
        ValueError: If ``__type__`` references an unknown type not in the registries.
    """
    if isinstance(payload, dict) and "__type__" in payload:
        typ = payload.get("__type__")
        if typ == "SubResult": # backwards compatibility for legacy serialization
            typ = "SubQAStep"
        
        # Check STATE_REGISTRY first for State types
        from ..type_registry import STATE_REGISTRY
        if typ in STATE_REGISTRY:
            state_class = STATE_REGISTRY[typ]
            if hasattr(state_class, "from_dict"):
                return state_class.from_dict(payload)
            # If no from_dict, return raw payload
            return payload
        
        # Handle Step types from TYPE_REGISTRY
        payload_copy = dict(payload)
        payload_copy.pop("__type__")
        ctor = TYPE_REGISTRY.get(typ)
        if ctor is None:
            raise ValueError(f"Unknown step type '{typ}'. Ensure it is registered via @register_type.")
        return ctor(**{k: _deserialize_obj(v) for k, v in payload_copy.items()})
    if isinstance(payload, list):
        return [_deserialize_obj(item) for item in payload]
    return payload


def deserialize_state(payload: Iterable):
    """Rebuild a state from a serialized representation."""
    return [_deserialize_obj(step) for step in payload]


def log_state(logger: logging.Logger, state, header: str, level: int = logging.DEBUG) -> None:
    """Emit a structured log record capturing a heterogeneous state."""
    if not logger.isEnabledFor(level):
        return
    
    if hasattr(state, "render_history"):
        content = state.render_history()
    else:
        content = str(state)
        
    logger.log(level, "%s\n%s", header, content)

__all__ = [
    "deserialize_state",
    "log_state"
]
