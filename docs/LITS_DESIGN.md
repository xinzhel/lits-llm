# LiTS Framework Design

## Section 1: Design Rationale

LiTS (Language Inference via Tree Search) is designed around a modular architecture that separates concerns between reasoning, execution, and evaluation. This separation enables flexible composition of different algorithms and components.

### Section 1.1: Core Principles

1. **Separation of Concerns**: Policy (reasoning), Transition (execution), and RewardModel (evaluation) are independent components
2. **Composability**: Components can be mixed and matched to create different agents
3. **Reusability**: Same components work across chain agents and tree search algorithms
4. **Type Safety**: Generic types ensure compile-time correctness

### Section 1.2: Architecture Layers

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

## Section 2: Extending LiTS

LiTS uses a decorator-based registration system that lets you extend the framework without modifying core code. All decorators are available from a unified import:

```python
from lits.registry import (
    # Components
    register_transition, register_policy, register_reward_model,
    # Prompts
    register_system_prompt, register_user_prompt,
    # Datasets
    register_dataset, load_dataset, infer_task_type
)
```

### Section 2.1: Adding a New Planning Domain (env_grounded)

Domain experts can add new planning domains by implementing a single Transition class. This section provides a complete guide for implementing custom `EnvGroundedTransition` subclasses.

#### Required Methods

| Method | Type | Purpose |
|--------|------|---------|
| `goal_check()` | Static | Check if goals are met, return `(bool, float)` |
| `generate_actions()` | Static | Generate valid actions from state, return `List[str]` |
| `init_state()` | Instance | Initialize state from dataset example kwargs |
| `_step()` | Instance | Execute action and return new state |

#### Minimal Example

```python
from lits.registry import register_transition
from lits.components.transition.env_grounded import EnvGroundedTransition
from lits.structures.env_grounded import EnvState, EnvStep, EnvAction
from typing import Tuple, List

@register_transition("robot_arm", task_type="env_grounded")
class RobotArmTransition(EnvGroundedTransition):
    """All domain-specific logic in one class."""
    
    @staticmethod
    def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
        """Check if robot reached target position.
        
        Args:
            query_or_goals: Goal description (e.g., "move arm to position (5, 3)")
            env_state: Current state string (e.g., "arm at (0, 0)")
        
        Returns:
            (goal_reached, progress): Boolean and progress score 0.0-1.0
        """
        target = parse_target(query_or_goals)
        current = parse_position(env_state)
        distance = compute_distance(target, current)
        reached = distance < 0.01
        progress = max(0.0, 1.0 - distance / 10.0)
        return reached, progress
    
    @staticmethod
    def generate_actions(env_state: str) -> List[str]:
        """Return valid robot movements for current state."""
        return ["move_up", "move_down", "move_left", "move_right", "grip", "release"]
    
    def init_state(self, **kwargs) -> EnvState:
        """Initialize state from dataset example.
        
        Args:
            **kwargs: Dataset example fields. Must include 'init_state_str'.
        
        Returns:
            Initial EnvState for tree search
        """
        init_str = kwargs.get('init_state_str')
        if init_str is None:
            raise ValueError("RobotArmTransition requires 'init_state_str' in kwargs")
        return EnvState(init_state=init_str)
    
    def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> Tuple[EnvState, dict]:
        """Execute action and update state.
        
        Args:
            state: Current EnvState
            step_or_action: EnvStep or EnvAction to execute
            query_or_goals: Goal description
        
        Returns:
            (new_state, aux_dict): Updated state and auxiliary info
        """
        import copy
        
        # Extract action from EnvStep if needed
        action = step_or_action.action if isinstance(step_or_action, EnvStep) else step_or_action
        
        # Create new state
        new_state = copy.deepcopy(state)
        new_env_state = self._apply_action(new_state.env_state, action)
        
        # Create step and append to state
        new_step = EnvStep(action=action, next_state=new_env_state)
        new_state.append(new_step)
        
        # Check goal and return
        goal_reached, progress = self.goal_check(query_or_goals, new_env_state)
        return new_state, {"goal_reached": (goal_reached, progress)}
    
    def _apply_action(self, env_state: str, action: EnvAction) -> str:
        """Domain-specific action execution logic."""
        # Implement your state update logic here
        ...
```

#### Method Signatures

**`goal_check()` - Static Method**
```python
@staticmethod
def goal_check(*args, **kwargs) -> Tuple[bool, float]:
    """
    Returns:
        (goal_reached, progress): 
            - goal_reached: True if all goals satisfied
            - progress: Score from 0.0 (no progress) to 1.0 (complete)
    
    Common signatures:
        - String-based: goal_check(query_or_goals: str, env_state: str)
        - Structured: goal_check(goals: List[Dict], state: np.ndarray)
    """
```

**`generate_actions()` - Static Method**
```python
@staticmethod
def generate_actions(*args, **kwargs) -> List[str]:
    """
    Returns:
        List of valid action strings for the current state
    
    Common signatures:
        - String-based: generate_actions(env_state: str)
        - Structured: generate_actions(state: Dict, constraints: List = None)
    """
```

**`init_state()` - Instance Method**
```python
def init_state(self, **kwargs) -> EnvState:
    """
    Args:
        **kwargs: Fields from dataset example dict. Extract what you need.
    
    Returns:
        Initial EnvState for tree search
    
    Convention:
        - env_grounded tasks expect 'init_state_str' in kwargs
        - Raise ValueError with helpful message if required kwargs missing
    """
```

**`_step()` - Instance Method**
```python
def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> Tuple[EnvState, dict]:
    """
    Args:
        state: Current state
        step_or_action: EnvStep (from policy) or EnvAction
        query_or_goals: Goal description string
    
    Returns:
        (new_state, aux_dict): 
            - new_state: Updated EnvState with new step appended
            - aux_dict: Must include 'goal_reached' key with (bool, float) tuple
    """
```

#### Using LLM in Transitions

If your domain requires LLM calls (e.g., for state updates), use the `_call_model()` helper:

```python
def _step(self, state, step_or_action, query_or_goals, **kwargs):
    # Build prompt for LLM
    prompt = f"Given state: {state.env_state}\nAction: {action}\nWhat is the new state?"
    
    # Call LLM with auto-constructed role for logging
    response = self._call_model(prompt, temperature=0.0)
    new_env_state = response.text.strip()
    
    # ... rest of step logic
```

Once registered, set `benchmark_name="robot_arm"` in your config and tree search works automatically.

#### Understanding EnvState vs env_state

LiTS distinguishes between two related but different concepts:

| Concept | Type | Purpose | Scope |
|---------|------|---------|-------|
| `EnvState` | Class | Framework's trajectory container | Framework-level |
| `env_state` | Property | Domain-specific environment snapshot | Domain-level |

**`EnvState`** is the framework's state container that:
- Stores the full trajectory history (list of `EnvStep` objects)
- Tracks `init_state` (initial environment description)
- Provides `env_state` property to access current snapshot
- Handles serialization/deserialization for checkpointing

**`env_state`** (accessed via `EnvState.env_state`) is the domain-specific snapshot:
- Represents the current environment at a single point in time
- Default type is `str` for text-based domains (BlocksWorld, Crosswords)
- Can be extended to other types for structured domains

```python
# EnvState is the framework container
state = EnvState(init_state="block A on table, block B on A")

# env_state is the domain-specific snapshot (str by default)
current_snapshot: str = state.env_state  # "block A on table, block B on A"

# Static methods operate on the snapshot, not the container
actions = MyTransition.generate_actions(state.env_state)  # Pass str
goal_met, progress = MyTransition.goal_check(goals, state.env_state)  # Pass str
```

**Why this separation?**
1. **Simplicity for domain experts**: Static methods only need the current snapshot, not trajectory history
2. **Flexibility**: Different domains can use different snapshot representations
3. **Efficiency**: Action generation doesn't need to process full trajectory

**Extending to other data types:**

For domains with structured state (robotics, games), you can:
1. Store serialized state in `EnvState.init_state` and `EnvStep.next_state` (both `str`)
2. Parse/serialize in your static methods
3. Or subclass `EnvState` to override the `env_state` property

```python
# Option 1: Serialize structured state to string
@register_transition("robot_arm", task_type="env_grounded")
class RobotArmTransition(EnvGroundedTransition):
    @staticmethod
    def generate_actions(env_state: str) -> List[str]:
        # Parse JSON string to structured data
        state_dict = json.loads(env_state)
        position = np.array(state_dict["position"])
        # Generate actions based on structured state
        return ["move_up", "move_down", ...]

# Option 2: Custom EnvState subclass (advanced)
@dataclass
class RobotState(EnvState):
    @property
    def env_state(self) -> np.ndarray:
        # Return structured data instead of string
        return np.array(json.loads(super().env_state))
```

#### Extension Scenarios for env_grounded Tasks

LiTS uses a **registry-first with fallback** pattern for env_grounded components. This provides flexibility for different extension scenarios:

| Scenario | What to Register | Policy | RewardModel |
|----------|------------------|--------|-------------|
| **Simple** | Transition only | Generic `EnvGroundedPolicy` | Generic `EnvGroundedPRM` |
| **Custom action selection** | Transition + Policy | Custom subclass | Generic `EnvGroundedPRM` |
| **Custom reward shaping** | Transition + RewardModel | Generic `EnvGroundedPolicy` | Custom subclass |
| **Full customization** | All three | Custom subclass | Custom subclass |

**Scenario 1: Simple Domain (Transition Only)**

Most domains only need a custom Transition. The generic Policy and RewardModel work out of the box:

```python
# Just register Transition - generic Policy/RewardModel are used automatically
@register_transition("puzzle_game", task_type="env_grounded")
class PuzzleTransition(EnvGroundedTransition):
    @staticmethod
    def goal_check(query, env_state): ...
    @staticmethod
    def generate_actions(env_state): ...
    def _step(self, state, action, query, **kwargs): ...

# In config: benchmark_name="puzzle_game"
# Factory uses: PuzzleTransition + EnvGroundedPolicy + EnvGroundedPRM
```

**Scenario 2: Custom Action Selection (Transition + Policy)**

When you need domain-specific action filtering, prioritization, or sampling:

```python
@register_transition("robot_arm", task_type="env_grounded")
class RobotArmTransition(EnvGroundedTransition):
    # ... standard methods ...

@register_policy("robot_arm", task_type="env_grounded")
class RobotArmPolicy(EnvGroundedPolicy):
    """Safety-aware action selection for robotics."""
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # Get all valid actions
        valid_actions = self.generate_all_actions(state.env_state)
        
        # Filter out unsafe actions (domain-specific logic)
        safe_actions = [a for a in valid_actions if self._is_safe(a, state)]
        
        # Prioritize actions that move toward goal
        prioritized = self._prioritize_by_heuristic(safe_actions, kwargs.get('query'))
        
        # Return top n_actions
        return [EnvStep(action=EnvAction(a)) for a in prioritized[:n_actions]]
```

**Scenario 3: Custom Reward Shaping (Transition + RewardModel)**

When you need domain-specific reward signals beyond goal checking:

```python
@register_transition("logistics", task_type="env_grounded")
class LogisticsTransition(EnvGroundedTransition):
    # ... standard methods ...

@register_reward_model("logistics", task_type="env_grounded")
class LogisticsPRM(EnvGroundedPRM):
    """Reward model with efficiency and constraint penalties."""
    
    def __init__(self, base_model, task_name, **kwargs):
        super().__init__(base_model, task_name, **kwargs)
        self.efficiency_weight = kwargs.get('efficiency_weight', 0.3)
        self.constraint_penalty = kwargs.get('constraint_penalty', -10.0)
    
    def fast_reward(self, state, action, **kwargs):
        base_reward, aux = super().fast_reward(state, action, **kwargs)
        
        # Add efficiency bonus (fewer steps = better)
        efficiency_bonus = self.efficiency_weight * (1.0 / (len(state) + 1))
        
        # Add constraint penalty (e.g., weight limits, time windows)
        constraint_score = self._check_constraints(state, action)
        penalty = self.constraint_penalty if constraint_score < 0 else 0
        
        return base_reward + efficiency_bonus + penalty, aux
```

**Scenario 4: Full Customization (All Three)**

For complex domains requiring complete control:

```python
@register_transition("autonomous_vehicle", task_type="env_grounded")
class AVTransition(EnvGroundedTransition):
    """Simulates vehicle dynamics with physics engine."""
    # ... custom physics simulation ...

@register_policy("autonomous_vehicle", task_type="env_grounded")
class AVPolicy(EnvGroundedPolicy):
    """Uses learned policy network for action generation."""
    def __init__(self, base_model, task_name, policy_network, **kwargs):
        super().__init__(base_model, task_name, **kwargs)
        self.policy_network = policy_network
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # Use neural network instead of LLM for action generation
        action_probs = self.policy_network(state.env_state)
        return self._sample_actions(action_probs, n_actions, temperature)

@register_reward_model("autonomous_vehicle", task_type="env_grounded")
class AVPRM(EnvGroundedPRM):
    """Uses learned value network for reward estimation."""
    def __init__(self, base_model, task_name, value_network, **kwargs):
        super().__init__(base_model, task_name, **kwargs)
        self.value_network = value_network
    
    def fast_reward(self, state, action, **kwargs):
        # Use neural network instead of LLM for reward estimation
        return self.value_network(state.env_state, action), {}
```

**How the Factory Resolves Components:**

```python
# In component_factory.py
def create_components_env_grounded(benchmark_name, ...):
    # 1. Transition: Required - must be registered
    TransitionCls = ComponentRegistry.get_transition(benchmark_name)  # Raises if not found
    
    # 2. Policy: Optional - falls back to EnvGroundedPolicy
    try:
        PolicyCls = ComponentRegistry.get_policy(benchmark_name)
    except KeyError:
        PolicyCls = EnvGroundedPolicy  # Default
    
    # 3. RewardModel: Optional - falls back to EnvGroundedPRM
    try:
        RewardModelCls = ComponentRegistry.get_reward_model(benchmark_name)
    except KeyError:
        RewardModelCls = EnvGroundedPRM  # Default
    
    # Instantiate and return
    ...
```

### Section 2.2: Customizing Reasoning Components (language_grounded)

Researchers can register custom Policy, Transition, or RewardModel classes:

```python
from lits.registry import register_policy, register_reward_model
from lits.components.policy.concat import ConcatPolicy
from lits.components.reward.generative import GenerativePRM

@register_policy("my_math_task", task_type="language_grounded")
class CustomMathPolicy(ConcatPolicy):
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        actions = super()._get_actions(state, n_actions, temperature, **kwargs)
        return self._filter_invalid_math_steps(actions)

@register_reward_model("my_math_task", task_type="language_grounded")
class CustomMathPRM(GenerativePRM):
    ...
```

### Section 2.3: Registering Datasets

```python
from lits.registry import register_dataset

@register_dataset("my_dataset", task_type="env_grounded")
def load_my_dataset(split="test", **kwargs):
    # Load and return dataset
    ...
```

### Section 2.4: Registering Custom Prompts

```python
from lits.registry import register_system_prompt, register_user_prompt

# prompt_key (3rd argument) can be a benchmark name (e.g., 'blocksworld') 
# or task type (e.g., 'language_grounded')
@register_system_prompt("policy", "concat", "my_math_task")
def my_system_prompt():
    return "You are solving math problems step by step..."

@register_user_prompt("policy", "concat", "my_math_task")
def my_user_prompt():
    return {"question_format": "Problem: {question}"}
```

Both decorators accept any return type—the component that consumes the prompt handles the type.

## Section 3: Component Compatibility

### Section 3.1: Policy + Transition + RewardModel Combinations

Different task types require different component combinations:

#### Section 3.1.1: Tool Use Tasks (e.g., ReAct, Function Calling)

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

#### Section 3.1.2: Reasoning Tasks (e.g., Math QA, RAP)

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

#### Section 3.1.3: Sequential Reasoning (e.g., ReST, Chain-of-Thought)

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

#### Section 3.1.4: Environment-Grounded Tasks (e.g., BlocksWorld, Robotics)

**Components:**
- **Policy**: `EnvGroundedPolicy` - Generates valid actions for environment
- **Transition**: `BlocksWorldTransition` or custom - Simulates environment dynamics
- **RewardModel**: `EnvGroundedPRM` - Evaluates action quality via LLM

**Flow:**
```
Policy: state → EnvStep(action)
Transition: (state, action) → (new_state with next_state, goal_reached)
RewardModel: (state, action, query) → action_score
```

**Example:**
```python
# After registering RobotArmTransition (see Section 2.1)
from lits.components.registry import ComponentRegistry

TransitionCls = ComponentRegistry.get_transition("robot_arm")
goal_check = TransitionCls.goal_check  # Static method
generate_actions = TransitionCls.generate_actions  # Static method

policy = EnvGroundedPolicy(
    base_model=model,
    task_name='robot_arm',
    generate_all_actions=generate_actions
)
transition = TransitionCls(
    base_model=model,
    task_name='robot_arm',
    max_steps=10
)
```

### Section 3.2: Component Compatibility by TASK_TYPE and Method

Components define their interface category via the `TASK_TYPE` class constant:

| TASK_TYPE | Method | Policy | Transition | RewardModel | State/Step Types | LLM Type | Notes |
|-----------|--------|--------|------------|-------------|------------------|----------|-------|
| **env_grounded** | Chain (`EnvChain`) | `EnvGroundedPolicy` | `BlocksWorldTransition` | — | `EnvState` / `EnvStep` | Chat | Sequential execution without search |
| **env_grounded** | Tree (RAP / REST / BFS) | `EnvGroundedPolicy` | `BlocksWorldTransition` | `EnvGroundedPRM` | `EnvState` / `EnvStep` | Chat | **Same components for all tree methods** - only search settings differ |
| **tool_use** | Chain (`ReActChat`) | `ToolUsePolicy` | `ToolUseTransition` | — | `ToolUseState` / `ToolUseAction` | Chat | Sequential tool execution with observations |
| **tool_use** | Tree (REST / BFS) | `ToolUsePolicy` | `ToolUseTransition` | `ToolUsePRM` | `ToolUseState` / `ToolUseAction` | Chat | Same components as chain + RewardModel |
| **language_grounded** | Tree (RAP) | `RAPPolicy` | `RAPTransition` | `RapPRM` | `SubQAState` / `SubQAStep` | **Completion** | Requires completion model, sub-question decomposition |
| **language_grounded** | Tree (REST / BFS) | `ConcatPolicy` | `ConcatTransition` | `GenerativePRM` | `ThoughtState` / `ThoughtStep` | Chat | Chain-of-thought style reasoning |

### Section 3.3: TASK_TYPE vs task_name

| Concept | Purpose | Example Values | Where Used |
|---------|---------|----------------|------------|
| **TASK_TYPE** (class constant) | Defines the interface category a component implements | `'env_grounded'`, `'tool_use'`, `'language_grounded'`, `None` | Component class definitions |
| **task_name** (constructor param) | Key for prompt registry lookup | `'blocksworld'`, `'gsm8k'`, `'mapeval-sql'` | Component instantiation |

**Example:**
```python
class EnvGroundedPolicy(Policy):
    TASK_TYPE = "env_grounded"  # Interface category
    
    def __init__(self, base_model, task_name: str, ...):
        # task_name (e.g., 'blocksworld') used for prompt lookup
        super().__init__(base_model, task_name=task_name, ...)
```

This separation allows:
- **TASK_TYPE**: Used by factory functions to select appropriate component classes, and as fallback for prompt lookup
- **task_name**: Used by components to load task-specific prompts from the registry

**Why no chain method for language_grounded?** Language-grounded QA tasks (math reasoning, multi-hop QA) don't require interrupting the reasoning chain for external execution. Chain-of-thought can be generated in a single LLM inference. Tree search is used only when exploring multiple reasoning paths is beneficial.

**Key Insight:** For environment-grounded and tool-use tasks, chain and tree methods share the same Policy and Transition components. Tree search simply adds a RewardModel for evaluating and selecting among multiple candidate paths. For env_grounded tasks, all tree search variants (RAP, REST, BFS) use identical components—only the search hyperparameters differ:

| Setting | RAP (MCTS) | REST (MCTS) | BFS |
|---------|------------|-------------|-----|
| Algorithm | MCTS with UCB | MCTS with value estimation | Beam search |
| `n_iterations` | Higher (100+) | Medium (50) | N/A |
| `beam_width` | N/A | N/A | Configurable (5-20) |
| `depth_limit` | Configurable | Configurable | Configurable |
| Reward aggregation | Backpropagation | Backpropagation | Cumulative |

### Section 3.4: Task-Instance-Specific Components

Most components work across multiple task instances within a `TASK_TYPE`. However, some components are specific to a single task instance and set `TASK_TYPE = None`:

| Component | TASK_TYPE | Task Instance | Why Task-Specific |
|-----------|-----------|---------------|-------------------|
| `BlocksWorldTransition` | `None` | `blocksworld` | Implements BlocksWorld-specific state parsing and goal checking |

**Why `TASK_TYPE = None`?**

Setting `TASK_TYPE = None` prevents the prompt registry from falling back to a generic task type prompt when `task_name` lookup fails:

1. **Prevents format mismatches** - Task-instance-specific components often expect specific output formats
2. **Forces explicit prompt registration** - Developers must register prompts under the specific `task_name`
3. **Enables extensibility** - Components can be adapted for new tasks beyond predefined categories

## Section 4: Key Design Patterns

### Section 4.1: Policy Returns Steps, Transition Receives Steps (v0.2.5+)

```python
# Policy generates full steps
steps = policy.get_actions(state, ...)  # Returns List[Step]

# Tree search stores full steps on nodes
for step in steps:
    action = step.get_action()
    node = SearchNode(action=action, ...)
    node.step = step

# Transition receives full steps
new_state, aux = transition.step(state, node.step, ...)
```

**Rationale:** Transitions need full step context to handle special cases (answers, errors, malformed outputs) without requiring logic in agents.

### Section 4.2: Transition Handles Multiple Step Types

```python
def step(self, state, step_or_action, ...):
    step = step_or_action
    
    # Case 1: Answer step (terminal)
    if step.answer is not None:
        new_state.append(step)
        return new_state, {"confidence": 1.0}
    
    # Case 2: Error step
    if step.error is not None:
        new_state.append(step)
        return new_state, {"confidence": 0.0}
    
    # Case 3: Malformed step
    if step.action is None and step.answer is None:
        step.observation = "Assistant output did not provide action or answer..."
        new_state.append(step)
        return new_state, {"confidence": 0.0}
    
    # Case 4: Action step - execute
    observation = execute_tool_action(step.action, self.tools)
    step.observation = observation
    new_state.append(step)
    return new_state, {"confidence": 1.0}
```

**Rationale:** Transition owns all step handling logic, keeping agents clean and focused on orchestration.

### Section 4.3: Chain Agents Pass Full Steps to Transition

```python
# In ReActChat.update_state()
policy_step = policy.get_actions(...)[0]
new_state, aux = transition.step(state, policy_step, ...)
return new_state
```

**Rationale:** Transition receives full step context and handles all cases internally.

### Section 4.4: Tree Search Uses Actions as Node Identity

```python
# Tree search stores actions in nodes
node = SearchNode(state=None, action=action, parent=parent)

# Later, transition materializes the state
if node.state is None:
    node.state, aux = transition.step(parent.state, node.action, ...)
```

**Rationale:** Actions uniquely identify transitions; states are materialized lazily for efficiency.

## Section 5: Component Interface Contracts

### Section 5.1: Policy Interface

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

### Section 5.2: Transition Interface

```python
class Transition(ABC, Generic[StateT, ActionT]):
    def step(self, state: StateT, step_or_action, ...) -> Tuple[StateT, dict]:
        """Execute step/action and return new state."""
    
    def is_terminal(self, state: StateT, ...) -> bool:
        """Check if state is terminal."""
```

**Contract:**
- Input: State and step_or_action (Step object from policy, or Action for backward compatibility)
- Output: New state and auxiliary data
- Must handle action steps (execute), answer steps (append), error steps (append), and malformed steps
- Must NOT generate new actions

### Section 5.3: RewardModel Interface

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

## Section 6: Common Pitfalls

### Section 6.1: Handling Answer/Error Logic in Agents

```python
# ❌ WRONG - special case logic scattered in agent code
class ReActChat:
    def update_state(self, query, state, ...):
        step = policy.get_actions(...)[0]
        if step.answer:
            state.append(step)
            return state
        if step.error:
            state.append(step)
            return state
        new_state, aux = transition.step(state, step.action, ...)
```

```python
# ✅ CORRECT - transition handles all step types
class ReActChat:
    def update_state(self, query, state, ...):
        step = policy.get_actions(...)[0]
        new_state, aux = transition.step(state, step, ...)
        return new_state
```

### Section 6.2: Policy Executes Actions

```python
# ❌ WRONG - violates separation of concerns
class ToolUsePolicy(Policy):
    def get_actions(self, state, ...):
        action = self.generate_action(state)
        observation = execute_tool_action(action, self.tools)  # ❌
        return [ToolUseStep(action=action, observation=observation)]
```

```python
# ✅ CORRECT - policy only generates
class ToolUsePolicy(Policy):
    def get_actions(self, state, ...):
        action = self.generate_action(state)
        return [ToolUseStep(action=action, observation=None)]  # ✅
```

### Section 6.3: Extracting Only Actions for Transition

```python
# ❌ WRONG - loses step context (pre-v0.2.5 pattern)
def _world_modeling(query, node, transition_model, ...):
    action = node.action
    node.state, aux = transition_model.step(node.parent.state, action, ...)
```

```python
# ✅ CORRECT - preserves step context (v0.2.5+ pattern)
def _world_modeling(query, node, transition_model, ...):
    step_or_action = getattr(node, 'step', node.action)
    node.state, aux = transition_model.step(node.parent.state, step_or_action, ...)
```

## Section 7: Advanced: Prompt Registry

LiTS uses a centralized registry to manage prompts for all LLM-based components. This section covers advanced prompt customization beyond the basic decorator usage shown in Section 2.4.

### Section 7.1: Prompt Types

Components use two types of prompts:
- **`task_prompt_spec`** (System Prompt): Instructions, format specifications, few-shot examples
- **`usr_prompt_spec`** (User Template): Structures for formatting query-specific information

### Section 7.2: Registry Lookup Priority

When a component is instantiated, prompts are loaded with the following priority:

1. **Explicit parameter** - If `task_prompt_spec` or `usr_prompt_spec` is passed directly
2. **task_name lookup** - Registry lookup by `task_name` (e.g., `'blocksworld'`)
3. **TASK_TYPE lookup** - Registry lookup by component's `TASK_TYPE` (e.g., `'language_grounded'`) - **skipped if TASK_TYPE is None**
4. **Default** - Registry lookup with key `'default'`

```python
# Example: GenerativePRM with TASK_TYPE = "language_grounded"
evaluator = GenerativePRM(
    base_model=model,
    task_name='gsm8k'  # Lookup order: 'gsm8k' → 'language_grounded' → 'default'
)

# Example: BlocksWorldTransition with TASK_TYPE = None
transition = BlocksWorldTransition(
    base_model=model,
    task_name='blocksworld'  # Lookup order: 'blocksworld' → 'default' (no TASK_TYPE fallback)
)
```

### Section 7.3: Direct Prompt Injection

For one-off experiments, inject prompts directly without registration:

```python
from lits.components.policy.rap import RAPPolicy

# Inject custom system prompt
policy = RAPPolicy(
    base_model=model,
    task_prompt_spec="Custom system instructions...",
    task_name='gsm8k'  # usr_prompt_spec loaded from registry
)

# Inject both prompts
policy = RAPPolicy(
    base_model=model,
    task_prompt_spec="Custom system...",
    usr_prompt_spec={'format': 'custom'}
)
```

### Section 7.4: Programmatic Registration

For dynamic registration (e.g., in scripts), use class methods directly:

```python
from lits.prompts.registry import PromptRegistry

# prompt_key can be a benchmark name (e.g., 'blocksworld') or task type (e.g., 'language_grounded')
PromptRegistry.register('policy', 'rap', 'my_task', 'Custom instructions...')
PromptRegistry.register_usr('policy', 'rap', 'my_task', {'format': 'custom'})

policy = RAPPolicy(base_model=model, task_name='my_task')
```

### Section 7.5: Adding New Prompts to the Framework

#### Step 1: Create Prompt File

```python
# lits/prompts/policy/my_agent.py

task_prompt_spec_language_grounded = """
Your system instructions here...
"""

usr_prompt_spec_language_grounded = {
    'question_format': 'Question: {question}',
    'answer_format': 'Answer: {answer}'
}
```

#### Step 2: Register in load_default_prompts()

```python
# In lits/prompts/registry.py
def load_default_prompts():
    from .policy import my_agent
    
    # prompt_key can be a benchmark name or task type
    if hasattr(my_agent, 'task_prompt_spec_language_grounded'):
        PromptRegistry.register(
            'policy', 'my_agent', 'language_grounded',  # prompt_key='language_grounded'
            my_agent.task_prompt_spec_language_grounded
        )
```

#### Step 3: Implement Component

```python
from ..base import Policy

class MyAgentPolicy(Policy):
    TASK_TYPE = "language_grounded"
    
    def _get_agent_name(self) -> str:
        return 'my_agent'  # Must match registry key
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # self.task_prompt_spec and self.usr_prompt_spec are auto-loaded
        ...
```

### Section 7.6: Registered Prompts Reference

#### Policy Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `concat` | `language_grounded` | ✓ | — |
| `env_grounded` | `blocksworld` | — | ✓ |
| `tool_use` | `default` | ✓ | — |

#### Reward Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | — |
| `generative` | `language_grounded` | ✓ | — |
| `env_grounded` | `blocksworld` | ✓ | ✓ |

#### Transition Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `rap` | `default` | ✓ | — |
| `blocksworld` | `default` | ✓ | ✓ |

### Section 7.7: Best Practices

**Use `task_prompt_spec` for:**
- System-level instructions that don't change per query
- Output format specifications
- Few-shot examples

**Use `usr_prompt_spec` for:**
- Query-specific formatting with placeholders
- Action-specific templates
- State-dependent formatting

**Type Guidelines:**
```python
# ✓ task_prompt_spec: string, dict, or PromptTemplate
task_prompt_spec = "Instructions..."
task_prompt_spec = {'instruction': '...', 'examples': [...]}

# ✓ usr_prompt_spec: dict or PromptTemplate (NOT string)
usr_prompt_spec = {'question': 'Q: {q}', 'answer': 'A: {a}'}

# ✗ BAD: usr_prompt_spec as plain string
usr_prompt_spec = "Q: {question}"  # Use dict instead!
```

## See Also

- [Component Base Classes](../lits/components/base.py) - Abstract interfaces
- [Prompt Registry](../lits/prompts/registry.py) - Prompt management
- [Tree Search Guide](agents/TREE_SEARCH_GUIDE.md) - Using tree search algorithms
- [ToolUseTransition](components/transitions/TOOL_USE_TRANSITION.md) - Tool execution example
- [RewardModel Interface](components/REWARD_MODEL_INTERFACE.md) - Evaluation interface
