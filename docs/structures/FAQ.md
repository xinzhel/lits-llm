# Structures FAQ

## Q1: What is the difference between `State` and `StateT`?

`State` is a concrete class (the base class for all state types), while `StateT` is a type variable (a placeholder for generics).

| | `State` | `StateT` |
|---|---|---|
| What it is | Concrete class | Type variable (placeholder) |
| Definition | `class State: ...` | `StateT = TypeVar("StateT", bound=State)` |
| Can create instances | ✓ `State()` | ✗ Cannot |
| Purpose | Define data structures, create objects | Preserve type information in function signatures |

## Q2: Why do base classes use `StateT` instead of `State` in method signatures?

Using `StateT` (generics) allows subclasses to preserve their specific type information for IDE autocompletion and static type checking.

**Without generics** (`state: State`):
```python
class Policy:
    def _get_actions(self, state: State, ...): ...

class MyPolicy(Policy):
    def _get_actions(self, state, ...):
        state.get_valid_moves()  # IDE doesn't know this method exists
```

**With generics** (`state: StateT`):
```python
class Policy(Generic[StateT, StepT]):
    def _get_actions(self, state: StateT, ...): ...

class MyPolicy(Policy[MyGameState, MyStep]):
    def _get_actions(self, state: MyGameState, ...):
        state.get_valid_moves()  # IDE correctly suggests this method ✓
```

Generics let subclasses "tell" the IDE what the concrete type is, enabling proper autocompletion for subclass-specific methods.

## Q3: Do I need to use generics? What if I ignore them?

No, generics are optional. They only affect static type checking (mypy/pyright) and IDE autocompletion. At runtime, both approaches work identically.

Use generics when:
- You want IDE autocompletion for custom State/Step methods
- You use mypy/pyright for type checking
- You're building a framework where users will create subclasses

Skip generics when:
- You're writing quick prototypes or research code
- You don't use static type checkers
- You're familiar with the codebase and don't need IDE hints
