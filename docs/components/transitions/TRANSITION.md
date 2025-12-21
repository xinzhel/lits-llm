# Transition Models

Transition models (also called "world models") define how states evolve in response to actions during tree search. They are responsible for executing actions, updating states, and determining when a trajectory is complete.

## Overview

The `Transition` abstract base class defines three core methods that all transition implementations must provide:

1. **`init_state(**kwargs)`**: Initialize the starting state for a new search trajectory
2. **`step(state, step_or_action, ...)`**: Execute an action and return the next state
3. **`is_terminal(state, ...)`**: Determine if a state represents a terminal/goal state

## Key Design Principles

### 1. Flexible Input Handling

Transitions receive `step_or_action` which can be either:
- **Step objects** (e.g., `ToolUseStep`, `EnvStep`) - Contains action + metadata from policy
- **Action objects** (e.g., `ToolUseAction`, `EnvAction`) - Raw action only

**Best Practice:** Always handle both cases by extracting the action when needed:

```python
def step(self, state, step_or_action, ...):
    # Extract action from Step if needed
    if isinstance(step_or_action, EnvStep):
        action = step_or_action.action
    else:
        action = step_or_action
    
    # Execute action...
```

### 2. Consistent Method Signatures

All transition methods should accept `**kwargs` for forward compatibility with tree search algorithms that may pass additional parameters:

```python
def step(self, state, step_or_action, query_or_goals=None, query_idx=None, from_phase="", **kwargs):
    # Implementation...
    
def is_terminal(self, state, query_or_goals=None, fast_reward=None, query_idx=None, from_phase="", **kwargs):
    # Implementation...
```

**Common kwargs passed by tree search:**
- `query_or_goals` (str): The problem/question being solved
- `query_idx` (int): Example index for logging/tracking
- `from_phase` (str): Algorithm phase ("expand", "simulate", "continuation")
- `fast_reward` (float): Pre-computed reward score (may be used for termination)

### 3. Return Auxiliary Data

The `step()` method should return `(new_state, aux_dict)` where `aux_dict` contains metadata useful for reward calculation:

```python
def step(self, state, step_or_action, ...):
    # Execute action and create new state
    new_state = self._execute_action(state, action)
    
    # Return auxiliary data for reward model
    aux = {
        "confidence": 1.0,
        "goal_reached": self.goal_check(query_or_goals, new_state)
    }
    return new_state, aux
```

### 4. init_state_kwargs Convention

The `init_state()` method receives kwargs from the dataset example. Different task types pass different fields:

| Task Type     | Expected kwargs           | Description                          |
|---------------|---------------------------|--------------------------------------|
| env_grounded  | `init_state_str` (str)    | Initial environment state description|
| math_qa       | (none)                    | Returns empty list                   |
| tool_use      | (none)                    | Returns empty ToolUseState           |

**Implementation pattern:**
```python
def init_state(self, **kwargs):
    # Extract what you need, ignore the rest
    init_str = kwargs.get('init_state_str')
    if init_str is None:
        raise ValueError("MyTransition requires 'init_state_str' in kwargs")
    return MyState(init_str)
```

## Common Pitfalls and Solutions

### Pitfall 1: Missing Parameters in Method Signatures

**Problem:** Tree search algorithms pass `query_idx` and `from_phase` to all transition methods, but your implementation doesn't accept them.

**Error:**
```
TypeError: step() got an unexpected keyword argument 'query_idx'
```

**Solution:** Add `**kwargs` to all method signatures:
```python
def step(self, state, step_or_action, query_or_goals=None, query_idx=None, from_phase="", **kwargs):
    pass

def is_terminal(self, state, query_or_goals=None, **kwargs):
    pass
```

### Pitfall 2: Wrong Parameter Order

**Problem:** Different transition implementations use inconsistent parameter ordering.

**Solution:** Follow the standard order from the base class:
```python
# Correct order
def step(self, state, step_or_action, query_or_goals, query_idx=None, from_phase=""):
    pass

def is_terminal(self, state, query_or_goals, **kwargs):
    pass
```

### Pitfall 3: Not Handling Step Objects

**Problem:** Policy returns `Step` objects but transition expects raw `Action` objects.

**Error:**
```
AssertionError: Action must be of type EnvAction
```

**Solution:** Extract action from Step when needed:
```python
def step(self, state, step_or_action, ...):
    if isinstance(step_or_action, EnvStep):
        action = step_or_action.action
    else:
        action = step_or_action
    # Now use action...
```

### Pitfall 4: Incorrect init_state_kwargs

**Problem:** Tree search passes dataset example dict to `init_state()`, but your implementation doesn't extract the right fields.

**Error:**
```
KeyError: 'init_state_str'
```

**Solution:** Use `.get()` with clear error messages:
```python
def init_state(self, **kwargs):
    init_str = kwargs.get('init_state_str')
    if init_str is None:
        raise ValueError(
            f"{self.__class__.__name__}.init_state() requires 'init_state_str' in kwargs. "
            f"Pass the example dict or init_state_str from the dataset."
        )
    return EnvState(init_state=init_str)
```

## Implementation Examples

### Example 1: ToolUseTransition

**Type Signature:**
```python
class ToolUseTransition(Transition[ToolUseState, ToolUseAction]):
    """Transition model that materializes tool observations for ToolUsePolicy-driven search."""
```

**Key Points:**
- **StateT**: `ToolUseState` - A trajectory of `ToolUseStep` objects
- **ActionT**: `ToolUseAction` - A JSON string representing a tool call
- **Input**: Receives `ToolUseStep` objects from the policy (not just actions)
- **Handles**: Actions, answers, errors, and malformed outputs

### ToolUseTransition Interface Methods

#### `__init__(tools, observation_on_error)`

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

#### `init_state() -> ToolUseState`

Initialize an empty state for starting a new trajectory.

**Returns:**
- `ToolUseState`: Empty trajectory state

**Example:**
```python
state = transition.init_state()
assert len(state) == 0
```

#### `step(state, step_or_action, ...) -> Tuple[ToolUseState, dict]`

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

#### `is_terminal(state, ...) -> bool`

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




### Example 2: BlocksWorldTransition

**Type Signature:**
```python
class BlocksWorldTransition(LlmTransition[EnvState, EnvAction]):
    """BlocksWorld Transition for environment-grounded planning tasks."""
    
    TASK_TYPE: str = "env_grounded"
```

**Key Points:**
- **StateT**: `EnvState` - Contains current blocks configuration
- **ActionT**: `EnvAction` - Natural language action (e.g., "put the red block on the green block")
- **Uses LLM**: Generates state updates by prompting an LLM to describe changes
- **init_state_kwargs**: Requires `init_state_str` from dataset

#### Key Implementation Details

**1. Handling init_state_kwargs:**
```python
def init_state(self, **kwargs) -> EnvState:
    """Initialize the world model.
    
    Args:
        **kwargs: Must include 'init_state_str' - the initial state description
                  from the dataset (e.g., "the red block is on the table...")
    """
    state_str = kwargs.get('init_state_str')
    if state_str is None:
        raise ValueError(
            "BlocksWorldTransition.init_state() requires 'init_state_str' in kwargs. "
            "Pass the example dict or init_state_str from the dataset."
        )
    return EnvState(init_state=state_str)
```

**2. Extracting action from Step:**
```python
def step(self, state: EnvState, step_or_action, query_or_goals: str, 
         query_idx: Optional[int] = None, from_phase: str = "") -> tuple[EnvState, dict]:
    """Take a step in the world model."""
    assert isinstance(query_or_goals, str), "query_or_goals must be str"
    
    # Extract action from EnvStep if needed
    if isinstance(step_or_action, EnvStep):
        action = step_or_action.action
    else:
        action = step_or_action
    
    # Create new state
    new_state = copy.deepcopy(state)
    env_state = new_state.env_state
    
    # Update blocks using LLM
    env_state = self.update_blocks(env_state, action, query_idx=query_idx, from_phase=from_phase)
    
    # Create new step and append
    new_step = EnvStep(action=action, next_state=env_state)
    new_state.append(new_step)
    
    return new_state, {"goal_reached": self.goal_check(query_or_goals, env_state)}
```

**3. is_terminal checks goal achievement only:**

The `is_terminal()` method should only check if the **goal/answer is reached** (task-level termination). Search-level termination (max_steps) is handled separately by `_is_terminal_with_depth_limit()` in the tree search algorithms.

```python
def is_terminal(self, state: EnvState, query_or_goals: str, **kwargs) -> bool:
    """Check if goal is reached.
    
    Note: 
    - **kwargs accepts fast_reward, query_idx, from_phase from tree search.
    - max_steps termination is handled by tree search algorithms, NOT here.
    """
    if self.goal_check(query_or_goals, state.env_state)[0]:
        return True
    return False
```

**Important:** Do NOT check `max_steps` in `is_terminal()`. The separation of concerns is:
- `Transition.is_terminal()`: Checks if the **goal/answer is achieved** (task-level)
- `_is_terminal_with_depth_limit()` in tree search: Checks if **max_steps is reached** (search-level)

**4. Using query_idx and from_phase for logging:**
```python
def update_blocks(self, env_state: str, action: EnvAction, 
                  query_idx: int = None, from_phase: str = '') -> str:
    """Update the block states with the action using LLM."""
    from ..utils import create_role
    
    # Construct prompt
    prompt_template = self._get_prompt_template(str(action))
    world_update_prompt = prompt_template.format(env_state, str(action).capitalize() + ".")
    
    # Call LLM with proper role for inference logging
    self.base_model.sys_prompt = self.task_prompt_spec
    world_output = self.base_model(
        world_update_prompt, 
        new_line_stop=True, 
        role=create_role("dynamics", query_idx, from_phase),
        temperature=DETERMINISTIC_TEMPERATURE
    ).text.strip()
    
    # Apply changes to state
    new_state = apply_change(world_output, env_state)
    return new_state
```

## Best Practices Summary

1. **Always accept `**kwargs`** in `step()` and `is_terminal()` for forward compatibility
2. **Handle both Step and Action inputs** by checking type and extracting action when needed
3. **Use consistent parameter ordering**: `state`, `step_or_action`, `query_or_goals`, `query_idx`, `from_phase`
4. **Extract init_state_kwargs carefully** with `.get()` and clear error messages
5. **Return auxiliary data** from `step()` for reward model consumption
6. **Use `query_idx` and `from_phase`** for proper inference logging via `create_role()`
7. **Follow the init_state_kwargs convention** for your task type (see table above)
8. **Separation of concerns for termination**:
   - `is_terminal()` should ONLY check goal/answer achievement (task-level)
   - Do NOT check `max_steps` in `is_terminal()` - this is handled by tree search algorithms via `_is_terminal_with_depth_limit()`

## Testing Your Transition

When implementing a new transition, test these scenarios:

1. **Basic execution**: Does `step()` correctly update state?
2. **Step vs Action**: Can it handle both `Step` and raw `Action` inputs?
3. **Terminal detection**: Does `is_terminal()` correctly identify goal states?
4. **init_state_kwargs**: Does `init_state()` handle missing/incorrect kwargs gracefully?
5. **Integration**: Does it work with MCTS/BFS tree search algorithms?

**Example test:**
```python
# Test with Step input
step = EnvStep(action=EnvAction("pick up red block"))
new_state, aux = transition.step(state, step, query_or_goals="stack blocks", query_idx=0, from_phase="expand")

# Test with Action input
action = EnvAction("pick up red block")
new_state, aux = transition.step(state, action, query_or_goals="stack blocks")

# Test terminal detection
assert not transition.is_terminal(state, query_or_goals="stack blocks")
# ... reach goal ...
assert transition.is_terminal(goal_state, query_or_goals="stack blocks", fast_reward=0.9)
```
