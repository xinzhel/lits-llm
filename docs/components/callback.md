# Dynamic Notes Injection via Callbacks

## Overview

The Policy component supports injecting dynamic notes from external sources (memory systems, databases, files) into the system prompt at runtime. This enables context-aware action generation by incorporating relevant information that changes during agent execution.

## Use Cases

- **Cross-trajectory Memory**: Inject learnings from previous problem-solving attempts
- **User Preferences**: Include user-specific preferences or constraints
- **Error Context**: Add information about past errors to avoid repeating mistakes
- **Task-specific Hints**: Dynamically provide relevant hints based on current state
- **Adaptive Guidance**: Adjust agent behavior based on performance metrics

## How It Works

1. Define a callback function that returns `List[str]` of notes
2. Register the callback with `policy.set_dynamic_notes_fn(callback)`
3. Notes are automatically retrieved and appended to the system prompt during each `get_actions()` call
4. Notes are formatted as bullet points at the end of the system prompt

## API Reference

### `Policy.set_dynamic_notes_fn(fn: Callable[[], List[str]])`

Register a callback function to provide dynamic notes for system prompt injection.

**Parameters:**
- `fn`: A callable that takes no arguments and returns `List[str]` of note strings

**Example:**
```python
def get_notes() -> List[str]:
    return ["Note 1", "Note 2", "Note 3"]

policy.set_dynamic_notes_fn(get_notes)
```

### `Policy._get_dynamic_notes() -> str`

Internal method that retrieves and formats notes from the callback. Returns an empty string if no callback is set or if the callback returns an empty list.

**Format:**
```
Additional Notes:
* note1
* note2
* note3
```

## Basic Usage

### Simple Static Notes

```python
from lits.components.policy.concat import ConcatPolicy
from lits.lm import get_lm

# Create policy
model = get_lm("gpt-4")
policy = ConcatPolicy(
    base_model=model,
    task_prompt_spec="You are a helpful math tutor.",
    n_actions=4
)

# Define notes function
def get_static_notes() -> List[str]:
    return [
        "User prefers step-by-step explanations",
        "Show all intermediate calculations",
        "Verify answers when possible"
    ]

# Register callback
policy.set_dynamic_notes_fn(get_static_notes)

# Notes will be automatically appended to system prompt
actions = policy.get_actions(state, query="Solve: 2x + 5 = 13")
```

**Resulting System Prompt:**
```
You are a helpful math tutor.

Additional Notes:
* User prefers step-by-step explanations
* Show all intermediate calculations
* Verify answers when possible
```

## Integration with Memory Systems

### Using mem0 Backend

```python
from lits.memory.manager import MemoryManager
from lits.components.policy.tool_use import ToolUsePolicy

# Initialize memory manager
memory_manager = MemoryManager(
    backend_type="mem0",
    config={
        "user_id": "user_123",
        "agent_id": "math_agent"
    }
)

# Create policy
policy = ToolUsePolicy(
    base_model=model,
    tools=tools,
    task_type="tool_use"
)

# Define memory-based notes function
def get_memory_notes() -> List[str]:
    # Retrieve relevant memories
    memories = memory_manager.search(
        query="math problem solving preferences",
        limit=3
    )
    return [f"Memory: {m['text']}" for m in memories]

# Register callback
policy.set_dynamic_notes_fn(get_memory_notes)
```

### Custom Memory Backend

```python
class CustomMemoryBackend:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_relevant_context(self, user_id: str, task_type: str) -> List[str]:
        """Query database for relevant context."""
        results = self.db.query(
            "SELECT note FROM user_context WHERE user_id = ? AND task_type = ?",
            (user_id, task_type)
        )
        return [row['note'] for row in results]

# Initialize backend
memory = CustomMemoryBackend(db_connection)

# Create notes function with closure
user_id = "user_456"
task_type = "math_qa"

def get_db_notes() -> List[str]:
    return memory.get_relevant_context(user_id, task_type)

policy.set_dynamic_notes_fn(get_db_notes)
```

## Advanced Patterns

### State-dependent Notes

```python
def get_state_dependent_notes(state, query) -> Callable[[], List[str]]:
    """Factory function that creates a notes callback with state context."""
    def notes_fn() -> List[str]:
        notes = []
        
        # Add depth-based hints
        if len(state) > 3:
            notes.append("Consider wrapping up the solution")
        
        # Add query-specific hints
        if "quadratic" in query.lower():
            notes.append("Remember to check for two solutions")
        
        # Add error-based hints
        if hasattr(state, 'errors') and state.errors:
            notes.append(f"Previous error: {state.errors[-1]}")
        
        return notes
    
    return notes_fn

# Update callback before each action generation
for step in range(max_steps):
    policy.set_dynamic_notes_fn(get_state_dependent_notes(state, query))
    actions = policy.get_actions(state, query=query)
    # ... process actions
```

### Multi-source Notes Aggregation

```python
class NotesAggregator:
    """Aggregate notes from multiple sources."""
    
    def __init__(self):
        self.sources = []
    
    def add_source(self, name: str, fn: Callable[[], List[str]]):
        """Register a notes source."""
        self.sources.append((name, fn))
    
    def get_all_notes(self) -> List[str]:
        """Retrieve and combine notes from all sources."""
        all_notes = []
        for name, fn in self.sources:
            try:
                notes = fn()
                # Prefix with source name for clarity
                all_notes.extend([f"[{name}] {note}" for note in notes])
            except Exception as e:
                logger.warning(f"Failed to get notes from {name}: {e}")
        return all_notes

# Setup aggregator
aggregator = NotesAggregator()

# Add multiple sources
aggregator.add_source("memory", lambda: memory_manager.get_recent(limit=2))
aggregator.add_source("preferences", lambda: user_preferences.get_active())
aggregator.add_source("errors", lambda: error_tracker.get_recent_errors(limit=1))

# Register aggregated callback
policy.set_dynamic_notes_fn(aggregator.get_all_notes)
```

### Conditional Notes with Caching

```python
class CachedNotesProvider:
    """Provide notes with caching to avoid repeated expensive queries."""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache = None
        self.cache_time = None
        self.ttl = ttl_seconds
    
    def _fetch_notes(self) -> List[str]:
        """Expensive operation to fetch notes."""
        # Query database, API, etc.
        return expensive_query()
    
    def get_notes(self) -> List[str]:
        """Get notes with caching."""
        import time
        current_time = time.time()
        
        # Return cached notes if still valid
        if self.cache and self.cache_time:
            if current_time - self.cache_time < self.ttl:
                return self.cache
        
        # Fetch fresh notes
        self.cache = self._fetch_notes()
        self.cache_time = current_time
        return self.cache

provider = CachedNotesProvider(ttl_seconds=30)
policy.set_dynamic_notes_fn(provider.get_notes)
```

## Error Handling

The framework handles errors gracefully:

```python
def failing_notes_fn() -> List[str]:
    raise ValueError("Database connection failed")

policy.set_dynamic_notes_fn(failing_notes_fn)

# Error is logged but doesn't crash the agent
# Empty notes are returned, and execution continues
actions = policy.get_actions(state, query=query)  # Works fine
```

**Error Behavior:**
- Exceptions in the callback are caught and logged
- Empty string is returned (no notes appended)
- Agent execution continues normally
- Full traceback is logged for debugging

## Best Practices

### 1. Keep Notes Concise

```python
# Good: Concise, actionable notes
def get_notes() -> List[str]:
    return [
        "User prefers brief explanations",
        "Previous error: forgot to simplify fractions"
    ]

# Avoid: Verbose, redundant notes
def get_notes() -> List[str]:
    return [
        "The user has indicated in their profile that they prefer brief explanations without too much detail",
        "In the previous attempt, the agent made an error where it forgot to simplify the fractions properly"
    ]
```

### 2. Limit Number of Notes

```python
def get_notes() -> List[str]:
    all_notes = fetch_all_relevant_notes()
    # Limit to top 5 most relevant
    return all_notes[:5]
```

### 3. Use Descriptive Prefixes

```python
def get_notes() -> List[str]:
    return [
        "Preference: Show intermediate steps",
        "Constraint: Use only basic arithmetic",
        "Hint: Check for edge cases",
        "Error: Previous division by zero at step 3"
    ]
```

### 4. Update Dynamically During Search

```python
# In tree search loop
for node in search_nodes:
    # Update notes based on current context
    policy.set_dynamic_notes_fn(
        lambda: get_context_notes(node.state, node.depth)
    )
    actions = policy.get_actions(node.state, query=query)
```

### 5. Test Callback Functions

```python
def test_notes_callback():
    """Test that notes callback works correctly."""
    notes = get_memory_notes()
    
    assert isinstance(notes, list)
    assert all(isinstance(note, str) for note in notes)
    assert len(notes) <= 10  # Reasonable limit
    assert all(len(note) < 200 for note in notes)  # Not too long
```

## Implementation Details

### System Prompt Construction Flow

1. `policy.get_actions()` is called
2. `policy.set_system_prompt()` is invoked
3. `policy._build_system_prompt()` returns base prompt (subclass implementation)
4. `policy._get_dynamic_notes()` retrieves and formats notes
5. Base prompt + dynamic notes are set as `model.sys_prompt`
6. Model generates actions with the enhanced prompt

### Format Specification

Notes are appended with the following format:

```python
def _get_dynamic_notes(self) -> str:
    if self._dynamic_notes_fn is None:
        return ""
    
    notes = self._dynamic_notes_fn()
    if not notes:
        return ""
    
    formatted_notes = "\n".join(f"* {note}" for note in notes)
    return f"\n\nAdditional Notes:\n{formatted_notes}"
```

### Subclass Implementation

Subclasses only need to implement `_build_system_prompt()` to return their base prompt. Dynamic notes are automatically handled:

```python
class MyCustomPolicy(Policy):
    def _build_system_prompt(self) -> str:
        # Just return the base prompt
        # Dynamic notes are automatically appended by set_system_prompt()
        return self.task_prompt_spec
```

## Testing

See `lits_llm/unit_test/components/test_dynamic_notes.py` for comprehensive test examples:

```bash
cd lits_llm
python unit_test/components/test_dynamic_notes.py
```

**Test Coverage:**
- Basic notes injection
- Empty notes handling
- No callback set (default behavior)
- Memory backend integration
- Error handling and recovery
- Multiple sources aggregation

## Related Components

- **Memory Manager**: `lits.memory.manager.MemoryManager` - Cross-trajectory memory backend
- **Policy Base**: `lits.components.base.Policy` - Base policy class
- **Prompt Registry**: `lits.prompts.registry.PromptRegistry` - Prompt template management

## See Also

- [Memory System Documentation](../memory/README.md)
- [Policy Component Guide](./policy/README.md)
- [Prompt Management](../prompts/README.md)
