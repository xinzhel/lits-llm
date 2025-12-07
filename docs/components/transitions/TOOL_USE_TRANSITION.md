# ToolUseTransition

## Type Signature

```python
class ToolUseTransition(Transition[ToolUseState, ToolUseAction]):
    """Transition model that materializes tool observations for ToolUsePolicy-driven search."""
```

**Key Points:**
- **StateT**: `ToolUseState` - A trajectory of `ToolUseStep` objects
- **ActionT**: `ToolUseAction` - A JSON string representing a tool call
- **Input**: Receives `ToolUseStep` objects from the policy (not just actions)
- **Handles**: Actions, answers, errors, and malformed outputs

## Interface Methods

### `__init__(tools, observation_on_error)`

Initialize the transition with available tools.

**Parameters:**
- `tools` (list): List of tool instances (e.g., `BaseTool` subclasses)
- `observation_on_error` (str): Error message prefix when tool execution fails (default: "Tool execution failed.")

**Example:**
```python
from lits.components.transition.tool_use import ToolUseTransition
from lits.tools import CalculatorTool, SearchTool

tools = [CalculatorTool(), SearchTool()]
transition = ToolUseTransition(
    tools=tools,
    observation_on_error="Tool execution failed."
)
```

### `init_state() -> ToolUseState`

Initialize an empty state for starting a new trajectory.

**Returns:**
- `ToolUseState`: Empty trajectory state

**Example:**
```python
state = transition.init_state()
assert len(state) == 0
```

### `step(state, step_or_action, ...) -> Tuple[ToolUseState, dict]`

Execute a tool step and return the updated state. Handles four cases:

1. **Action step**: Execute the tool action and add observation
2. **Answer step**: Append directly without execution (terminal)
3. **Error step**: Append directly with confidence=0.0
4. **Malformed step**: Add error message as observation

**Parameters:**
- `state` (ToolUseState): Current trajectory state
- `step_or_action` (ToolUseStep): Step from policy containing action/answer/error
- `query_or_goals` (str, optional): Query context for logging
- `query_idx` (int, optional): Query index for logging
- `from_phase` (str, optional): Algorithm phase description (e.g., "expand", "simulate")

**Returns:**
- Tuple of `(new_state, aux_dict)` where:
  - `new_state` (ToolUseState): Updated state with executed step appended
  - `aux_dict` (dict): Auxiliary data with confidence score

**Case Handling:**

**Case 1: Action Step**
```python
# Policy generates step with action
step = ToolUseStep(
    think="I need to calculate 2+2",
    action=ToolUseAction('{"tool": "calculator", "args": {"expression": "2+2"}}')
)

# Transition executes action and adds observation
new_state, aux = transition.step(state, step)
# Result: step.observation = "4", confidence = 1.0
```

**Case 2: Answer Step (Terminal)**
```python
# Policy generates step with answer
step = ToolUseStep(
    think="The calculation is complete",
    answer="The answer is 4"
)

# Transition appends directly without execution
new_state, aux = transition.step(state, step)
# Result: step appended as-is, confidence = 1.0
```

**Case 3: Error Step**
```python
# Policy generates step with error
step = ToolUseStep(
    error="Failed to parse action from assistant message"
)

# Transition appends directly
new_state, aux = transition.step(state, step)
# Result: step appended as-is, confidence = 0.0
```

**Case 4: Malformed Step**
```python
# Policy generates step without action or answer
step = ToolUseStep(
    think="Some reasoning..."
    # No action or answer
)

# Transition adds error observation
new_state, aux = transition.step(state, step)
# Result: step.observation = "Assistant output did not provide...", confidence = 0.0
```

### `is_terminal(state, ...) -> bool`

Check if the state is terminal (has reached a final answer).

**Parameters:**
- `state` (ToolUseState): Current trajectory state
- `query_or_goals` (str, optional): Query context
- `fast_reward` (float, optional): Fast reward score (unused in this implementation)
- `query_idx` (int, optional): Query index for logging
- `from_phase` (str, optional): Algorithm phase description

**Returns:**
- `bool`: True if the last step contains a final answer, False otherwise

**Logic:**
```python
if not state:
    return False
last = state[-1]
return bool(last.get_answer())
```

**Example:**
```python
# State without answer
state = ToolUseState([
    ToolUseStep(action=action1, observation="Result: 4")
])
assert not transition.is_terminal(state, query_or_goals="What is 2+2?")

# State with answer
state.append(ToolUseStep(answer="The answer is 4"))
assert transition.is_terminal(state, query_or_goals="What is 2+2?")
```


