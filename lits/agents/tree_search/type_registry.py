"""Shared registry for serializable search step types."""

TYPE_REGISTRY = {}


def register_type(cls):
    """Decorator registering a class for SearchNode serialization/deserialization."""
    TYPE_REGISTRY[cls.__name__] = cls
    return cls
