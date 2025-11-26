"""Shared registry for serializable step types."""

TYPE_REGISTRY = {}


def register_type(cls):
    """
    Decorator registering a Step subclass for serialization/deserialization.
    
    This allows the State.from_dict() method to dynamically instantiate the
    correct Step subclass based on the "__type__" field in serialized data.
    
    Usage:
        @register_type
        @dataclass
        class MyStep(Step):
            ...
    
    The class will be registered in TYPE_REGISTRY with its __name__ as the key.
    """
    TYPE_REGISTRY[cls.__name__] = cls
    return cls
