"""Shared registry for serializable step and state types."""

TYPE_REGISTRY = {}
STATE_REGISTRY = {}


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


def register_state(cls):
    """
    Decorator registering a State subclass for serialization/deserialization.
    
    This allows _deserialize_obj() to dynamically instantiate the correct
    State subclass based on the "__type__" field in serialized data.
    
    Usage:
        @register_state
        class MyState(TrajectoryState):
            ...
    
    The class will be registered in STATE_REGISTRY with its __name__ as the key.
    """
    STATE_REGISTRY[cls.__name__] = cls
    return cls
