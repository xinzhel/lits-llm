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

### 1. Policy Returns Steps, Transition Receives Actions

**Pattern:**
```python
# Policy generates full steps
steps = policy.get_actions(state, ...)  # Returns List[Step]

# Tree search extracts actions
for step in steps:
    action = step.get_action()  # Extract action from step
    node = SearchNode(action=action, ...)

# Transition receives actions
new_state, aux = transition.step(state, action, ...)
```

**Rationale:** Steps contain reasoning (think, confidence), but transitions only need actions to execute.

### 2. Transition Constructs Complete Steps

**Pattern:**
```python
# In ToolUseTransition.step()
def step(self, state, action, ...):
    # Execute action
    observation = execute_tool_action(action, self.tools)
    
    # Construct complete step
    step = ToolUseStep(action=action, observation=observation)
    
    # Update state
    new_state = ToolUseState()
    new_state.extend(state)
    new_state.append(step)
    
    return new_state, aux
```

**Rationale:** Transition owns the execution logic and constructs the complete step with results.

### 3. Chain Agents Preserve Policy Reasoning

**Pattern:**
```python
# In ReActChat.get_step()
policy_step = policy.get_actions(...)[0]

if policy_step.action:
    new_state, aux = transition.step(state, policy_step.action, ...)
    executed_step = new_state[-1]
    
    # Preserve reasoning from policy
    executed_step.think = policy_step.think
    executed_step.answer = policy_step.answer
    
    return executed_step
```

**Rationale:** Policy's reasoning (think) should be preserved even though transition reconstructs the step.

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
    def step(self, state: StateT, action: ActionT, ...) -> Tuple[StateT, dict]:
        """Execute action and return new state.
        
        Returns (new_state, auxiliary_dict)
        """
    
    def is_terminal(self, state: StateT, ...) -> bool:
        """Check if state is terminal."""
```

**Contract:**
- Input: State and action (NOT Step)
- Output: New state and auxiliary data
- Must execute action and capture results
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

### ❌ Wrong: Transition Receives Step

```python
# WRONG - violates type contract
class ToolUseTransition(Transition[ToolUseState, ToolUseStep]):
    def step(self, state, step: ToolUseStep, ...):
        # This breaks tree search compatibility
```

### ✅ Correct: Transition Receives Action

```python
# CORRECT - follows type contract
class ToolUseTransition(Transition[ToolUseState, ToolUseAction]):
    def step(self, state, action: ToolUseAction, ...):
        # Tree search can pass extracted actions
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
