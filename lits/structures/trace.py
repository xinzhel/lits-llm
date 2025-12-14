"""Utilities for logging, replaying, and serializing heterogeneous LangTree states."""

import json
import logging
from typing import Iterable
from dataclasses import is_dataclass, asdict

from ..type_registry import TYPE_REGISTRY

def _serialize_obj(obj):
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
