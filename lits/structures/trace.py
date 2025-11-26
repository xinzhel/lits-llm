"""Utilities for logging, replaying, and serializing heterogeneous LangTree states."""

import json
import logging
from typing import Iterable

from ..type_registry import TYPE_REGISTRY

def _serialize_obj(obj):
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


def serialize_state(state) -> list:
    """Convert a state (list of steps) into a JSON-serializable payload."""
    return [_serialize_obj(step) for step in state]


def _deserialize_obj(payload):
    if isinstance(payload, dict) and "__type__" in payload:
        payload = dict(payload)
        typ = payload.pop("__type__")
        if typ == "SubResult": # backwards compatibility for legacy serialization
            typ = "SubQAStep"
        ctor = TYPE_REGISTRY.get(typ)
        if ctor is None:
            raise ValueError(f"Unknown step type '{typ}'. Ensure it is registered via @register_type.")
        return ctor(**{k: _deserialize_obj(v) for k, v in payload.items()})
    if isinstance(payload, list):
        return [_deserialize_obj(item) for item in payload]
    return payload

def deserialize_state(payload: Iterable):
    """Rebuild a state from a serialized representation produced by ``serialize_state``."""
    return [_deserialize_obj(step) for step in payload]


def log_state(logger: logging.Logger, state, header: str, level: int = logging.DEBUG) -> None:
    """Emit a structured log record capturing a heterogeneous state."""
    if not logger.isEnabledFor(level):
        return
    serialized = serialize_state(state)
    logger.log(level, "%s %s", header, json.dumps(serialized, ensure_ascii=False))

__all__ = [
    "serialize_state",
    "deserialize_state",
    "log_state"
]
