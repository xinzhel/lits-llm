# LiTS Framework Design

## Design Rationale

LiTS (Language Inference via Tree Search) is designed around a modular architecture that separates concerns between reasoning, execution, and evaluation. This separation enables flexible composition of different algorithms and components.

### Core Principles

1. **Separation of Concerns**: Policy (reasoning), Transition (execution), and RewardModel (evaluation) are independent components
2. **Composability**: Components can be mixed and matched to create different agents
3. **Reusability**: Same components work across chain agents and tree search algorithms
4. **Type Safety**: Generic types ensure compile-time correctness

### Architecture Layers

The framework is organized into three main layers:

**Agents Layer**: High-level algorithms that orchestrate components
- `tree.MCTS/BFS`: Tree search algorithms (RAP, ReST-MCTS, LATS, ToT)
- `chain.ReAct`: Sequential reasoning agent
- `chain.EnvChain`: Environment-grounded task execution

**Components Layer**: Modular building blocks
- `Policy`: Generates actions from states
- `Transition`: Executes actions and produces new states
- `RewardModel`: Evaluates action quality

**Structures Layer**: Data types for states, actions, and steps
- `Action`: Base action type (StringAction, ToolUseAction, etc.)
- `Step`: Single reasoning/execution step
- `State`: Trajectory or environment snapshot
- `Node`: Tree search node (MCTSNode, BFSNode)

## Component Compatibility

### Policy + Transition + RewardModel Combinations

Different task types require different component combinations:

#### Tool Use Tasks (e.g., ReAct, Function Calling)

**Components:**
- **Policy**: `ToolUsePolicy` - Generates tool calls with reasoning
- **Transition**: `ToolUseTransition` - Executes tools and captures observations
- **RewardModel**: TBD (optional for tree search)

**Flow:**
```
Policy: state → ToolUseStep(think, action)
Transition: (state, action) → (new_state, observation)
RewardModel: (state, action) → reward
```

**Example:**
```python
policy = ToolUsePolicy(base_model=model, tools=tools)
transition = ToolUseTransition(tools=tools)
agent = ReActChat(policy=policy, transition=transition)
```

#### Reasoning Tasks (e.g., Math QA, RAP)

**Components:**
- **Policy**: `RapPolicy` - Generates sub-questions
- **Transition**: `RapTransition` - Answers sub-questions using LLM
- **RewardModel**: `RapQAPRM` - Evaluates sub-question usefulness

**Flow:**
```
Policy: state → SubQAStep(sub_question)
Transition: (state, sub_question) → (new_state, sub_answer)
RewardModel: (state, sub_question) → usefulness_score
```

#### Sequential Reasoning (e.g., ReST, Chain-of-Thought)

**Components:**
- **Policy**: `ConcatPolicy` - Generates reasoning steps
- **Transition**: `ConcatTransition` - Appends steps to trajectory
- **RewardModel**: `GenerativePRM` or `SelfConsistencyRM`

**Flow:**
```
Policy: state → ThoughtStep(action)
Transition: (state, action) → new_state
RewardModel: (state, action) → correctness_score
```

#### Environment-Grounded Tasks (e.g., BlocksWorld, Robotics)

**Components:**
- **Policy**: `EnvGroundedPolicy` - Generates valid actions for environment
- **Transition**: `BlocksWorldTransition` or custom - Simulates environment dynamics
- **RewardModel**: TBD (goal-based reward)

**Flow:**
```
Policy: state → EnvAction
Transition: (state, action) → (new_state, env_reward)
RewardModel: (state, action) → goal_progress
```

### Type Compatibility Matrix

| Agent Type | Policy | Transition | State Type | Action Type |
|------------|--------|------------|------------|-------------|
| ReActChat | ToolUsePolicy | ToolUseTransition | ToolUseState | ToolUseAction |
| RAP (MCTS) | RapPolicy | RapTransition | SubQAState | SubQAStep |
| ReST (MCTS) | ConcatPolicy | ConcatTransition | ThoughtState | ThoughtStep |
| EnvChain | EnvGroundedPolicy | Custom | EnvState | EnvAction |

## Key Design Patterns

### 1. Policy Returns Steps, Transition Receives Steps (v0.2.5+)

**Pattern:**
```python
# Policy generates full steps
steps = policy.get_actions(state, ...)  # Returns List[Step]

# Tree search stores full steps on nodes
for step in steps:
    action = step.get_action()  # Extract action for node identity
    node = SearchNode(action=action, ...)
    node.step = step  # Store full step for transition

# Transition receives full steps
new_state, aux = transition.step(state, node.step, ...)
```

**Rationale:** Transitions need full step context to handle special cases (answers, errors, malformed outputs) without requiring logic in agents.

### 2. Transition Handles Multiple Step Types

**Pattern:**
```python
# In ToolUseTransition.step()
def step(self, state, step_or_action, ...):
    step = step_or_action
    
    # Case 1: Answer step (terminal) - append directly
    if step.answer is not None:
        new_state.append(step)
        return new_state, {"confidence": 1.0}
    
    # Case 2: Error step - append directly
    if step.error is not None:
        new_state.append(step)
        return new_state, {"confidence": 0.0}
    
    # Case 3: Malformed step - add error observation
    if step.action is None and step.answer is None:
        step.observation = "Assistant output did not provide action or answer..."
        new_state.append(step)
        return new_state, {"confidence": 0.0}
    
    # Case 4: Action step - execute and add observation
    observation = execute_tool_action(step.action, self.tools)
    step.observation = observation
    new_state.append(step)
    return new_state, {"confidence": 1.0}
```

**Rationale:** Transition owns all step handling logic, keeping agents clean and focused on orchestration.

### 3. Chain Agents Pass Full Steps to Transition

**Pattern:**
```python
# In ReActChat.update_state()
policy_step = policy.get_actions(...)[0]

# Pass full step to transition (handles action/answer/error)
new_state, aux = transition.step(state, policy_step, ...)

return new_state
```

**Rationale:** Transition receives full step context (think, assistant_message) and handles all cases internally, eliminating special case logic in agents.

### 4. Tree Search Uses Actions as Node Identity

**Pattern:**
```python
# Tree search stores actions in nodes
node = SearchNode(state=None, action=action, parent=parent)

# Later, transition materializes the state
if node.state is None:
    node.state, aux = transition.step(parent.state, node.action, ...)
```

**Rationale:** Actions uniquely identify transitions; states are materialized lazily for efficiency.

## Component Interface Contracts

### Policy Interface

```python
class Policy(ABC, Generic[StateT, ActionT]):
    def get_actions(self, state: StateT, ...) -> List[StepT]:
        """Generate candidate actions for the given state.
        
        Returns List of Step objects (not raw actions).
        Each Step contains the action plus metadata (think, confidence, etc.)
        """
```

**Contract:**
- Input: Current state
- Output: List of Step objects with actions
- Must NOT execute actions or modify environment
- Must implement `_create_error_steps()` for error handling

### Transition Interface

```python
class Transition(ABC, Generic[StateT, ActionT]):
    def step(self, state: StateT, step_or_action, ...) -> Tuple[StateT, dict]:
        """Execute step/action and return new state.
        
        Args:
            state: Current state
            step_or_action: Step object from policy (may contain action, answer, or error)
        
        Returns (new_state, auxiliary_dict)
        """
    
    def is_terminal(self, state: StateT, ...) -> bool:
        """Check if state is terminal."""
```

**Contract:**
- Input: State and step_or_action (Step object from policy, or Action for backward compatibility)
- Output: New state and auxiliary data
- Must handle action steps (execute), answer steps (append), error steps (append), and malformed steps
- Must NOT generate new actions

### RewardModel Interface

```python
class RewardModel(ABC, Generic[StateT, ActionT]):
    def fast_reward(self, state: StateT, action: ActionT, ...) -> Tuple[float, dict]:
        """Evaluate action without execution."""
    
    def reward(self, state: StateT, action: ActionT, **kwargs) -> Tuple[float, dict]:
        """Evaluate action after execution."""
```

**Contract:**
- Input: State and action
- Output: Reward score and auxiliary metrics
- `fast_reward`: Evaluates before execution (for pruning)
- `reward`: Evaluates after execution (for scoring)

## Common Pitfalls

### ❌ Wrong: Handling Answer/Error Logic in Agents

```python
# WRONG - special case logic scattered in agent code
class ReActChat:
    def update_state(self, query, state, ...):
        step = policy.get_actions(...)[0]
        
        # ❌ Agent handles answer/error cases
        if step.answer:
            state.append(step)
            return state
        if step.error:
            state.append(step)
            return state
        
        # Only then call transition
        new_state, aux = transition.step(state, step.action, ...)
```

### ✅ Correct: Transition Handles All Cases

```python
# CORRECT - transition handles all step types
class ReActChat:
    def update_state(self, query, state, ...):
        step = policy.get_actions(...)[0]
        
        # ✅ Transition handles action/answer/error/malformed
        new_state, aux = transition.step(state, step, ...)
        return new_state
```

### ❌ Wrong: Policy Executes Actions

```python
# WRONG - violates separation of concerns
class ToolUsePolicy(Policy):
    def get_actions(self, state, ...):
        action = self.generate_action(state)
        observation = execute_tool_action(action, self.tools)  # ❌
        return [ToolUseStep(action=action, observation=observation)]
```

### ✅ Correct: Policy Only Generates

```python
# CORRECT - policy only generates
class ToolUsePolicy(Policy):
    def get_actions(self, state, ...):
        action = self.generate_action(state)
        return [ToolUseStep(action=action, observation=None)]  # ✅
```

### ❌ Wrong: Extracting Only Actions for Transition

```python
# WRONG - loses step context (pre-v0.2.5 pattern)
def _world_modeling(query, node, transition_model, ...):
    action = node.action  # Only action, no step context
    node.state, aux = transition_model.step(node.parent.state, action, ...)
```

### ✅ Correct: Passing Full Steps to Transition

```python
# CORRECT - preserves step context (v0.2.5+ pattern)
def _world_modeling(query, node, transition_model, ...):
    step_or_action = getattr(node, 'step', node.action)  # Full step if available
    node.state, aux = transition_model.step(node.parent.state, step_or_action, ...)
```

## Extension Points

### Adding New Task Types

1. Define State and Action types
2. Implement Policy for action generation
3. Implement Transition for execution
4. (Optional) Implement RewardModel for evaluation
5. Use with existing agents (ReActChat, MCTS, BFS)

### Adding New Agents

1. Implement agent loop (e.g., new search algorithm)
2. Use existing Policy/Transition/RewardModel interfaces
3. Follow the pattern: Policy → extract action → Transition

### Adding New Reward Models

1. Inherit from `RewardModel` base class
2. Implement `_fast_reward()` and `reward()` methods
3. Use with existing tree search algorithms

## See Also

- [Component Base Classes](../lits/components/base.py) - Abstract interfaces
- [Tree Search Guide](agents/TREE_SEARCH_GUIDE.md) - Using tree search algorithms
- [ToolUseTransition](components/transitions/TOOL_USE_TRANSITION.md) - Tool execution example
- [RewardModel Interface](components/REWARD_MODEL_INTERFACE.md) - Evaluation interface
