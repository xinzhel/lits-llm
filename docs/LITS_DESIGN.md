# LiTS Framework Design

## Section 1: Design Rationale

LiTS (Language Inference via Tree Search) is designed around a modular architecture that separates concerns between reasoning, execution, and evaluation. This separation enables flexible composition of different algorithms and components.

### Section 1.1: Core Principles

1. **Separation of Concerns**: Policy (reasoning), Transition (execution), and RewardModel (evaluation) are independent components
2. **Composability**: Components can be mixed and matched to create different agents
3. **Reusability**: Same components work across chain agents and tree search algorithms
4. **Type Safety**: Generic types ensure compile-time correctness

### Section 1.2: Target Users

LiTS serves two primary user groups with different goals:

| User Type | Goal | What They Don't Need to Know |
|-----------|------|------------------------------|
| **Domain experts** (non-AI/NLP) | Plug-and-play inference with domain-specific implementation | Search algorithm internals and agent component wiring |
| **AI/NLP researchers** | Connect novel inference methods they devise to downstream tasks, or explore new reasoning formulations (e.g., critique-refine, hypothesis-test) | Domain-specific semantics and evaluation criteria |

### Section 1.3: Task Types (MDP Formulations)

LiTS supports domains that can be formulated as Markov Decision Processes:

| Task Type | State Space | Action Space | Examples |
|-----------|-------------|--------------|----------|
| **env_grounded** | Physical/symbolic state | Discrete domain actions | BlocksWorld, robotics, games |
| **language_grounded** | Text context | Reasoning steps | Math reasoning, multi-hop QA |
| **tool_use** | Context + tool state | Tool calls + reasoning | SQL, web search, APIs |

### Section 1.4: Architecture Layers

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

## Section 2: Domain-Specific Injection

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

### Section 2.1: Injection Points by Task Type

Different task types require different injection points:

| Task Type | Required Injections | Optional Injections |
|-----------|---------------------|---------------------|
| **env_grounded** | Transition (`goal_check`, `generate_actions`, `init_state`, `_step`), Prompt | RewardModel (domain-specific shaping), Policy (custom action filtering) |
| **tool_use** | Tools, Prompt | RewardModel |
| **language_grounded** | Prompt, Dataset, Evaluation logic | RewardModel (e.g., ThinkPRM), Policy, Transition, custom Step/State structures |

#### Injection Point Details

| Component | What to Inject | Examples | LLM Usage |
|-----------|----------------|----------|-----------|
| **Prompt** | Domain context | "You are playing Crosswords", few-shot examples | Always |
| **Tools** | Domain-specific capabilities (tool_use only) | SQL executor, web search API, PDF parser | N/A (execution) |
| **Transition** | State semantics and update logic (env_grounded only) | BlocksWorld: regex-parsed block positions; Crosswords: `"h5 words"` format | Optional (for reasoning-based state updates) |
| **RewardModel** | Domain-specific reward shaping | Goal progress, execution success, constraint satisfaction | Optional (LLM-as-judge has self-preference bias; prefer verifiable signals) |
| **Policy** | Custom action filtering (env_grounded with enumerable action spaces) | Safety filtering, heuristic prioritization | Optional |
| **Dataset** | Task instances (language_grounded) | Question-answer pairs with metadata | N/A |
| **Evaluation** | Answer extraction and comparison (language_grounded) | Numeric comparison, string matching | N/A |

#### When to Use LLM in Each Component

| Scenario | Recommended Approach |
|----------|---------------------|
| **Enumerable action space** (e.g., BlocksWorld: 4-8 valid moves) | `generate_actions()` returns all; LLM selects in Policy |
| **Large but structured action space** (e.g., SQL with schema) | LLM generates candidates; `validate_action()` filters |
| **Open-ended action space** (e.g., free-form reasoning) | LLM generates; no enumeration possible |
| **Deterministic state transitions** (e.g., game rules) | No LLM in Transition |
| **Reasoning-based transitions** (e.g., BlocksWorld state update via natural language) | LLM in Transition |
| **Verifiable rewards** (e.g., goal check, SQL execution success) | Domain-specific reward; avoid LLM |
| **Subjective evaluation** (e.g., reasoning quality) | LLM-based RM (with caveats about self-preference bias) |

### Section 2.2: Adding a New Planning Domain (env_grounded)

Domain experts can add new planning domains by implementing a single Transition class. The generic `EnvGroundedPolicy` and `EnvGroundedPRM` work out of the box.

#### Required Methods

| Method | Type | Purpose |
|--------|------|---------|
| `goal_check()` | Static | Check if goals are met, return `(bool, float)` |
| `init_state()` | Instance | Initialize state from dataset example kwargs |
| `_step()` | Instance | Execute action and return new state |

#### Optional Methods

| Method | Type | Purpose |
|--------|------|---------|
| `generate_actions()` | Static | Generate valid actions from state (for enumerable action spaces) |
| `validate_action()` | Static | Validate LLM-generated action (for open-ended action spaces) |

**Action Space Contract:**
- **Enumerable action space** (e.g., BlocksWorld): Implement `generate_actions()` to return all valid actions
- **Open-ended action space** (e.g., Crosswords): Implement `validate_action()` to validate LLM output

#### Method Signatures

**`goal_check()` - Static Method**
```python
@staticmethod
def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
    """Check if goals are satisfied.
    
    Args:
        query_or_goals: Goal description string
        env_state: Current environment snapshot (str) - NOT the EnvState container!
                   This is the domain-specific state representation.
                   Access via state.env_state property in instance methods.
    
    Returns:
        (goal_reached, progress): 
            - goal_reached: True if all goals satisfied
            - progress: Score from 0.0 (no progress) to 1.0 (complete)
    """
```

**`generate_actions()` - Static Method**
```python
@staticmethod
def generate_actions(env_state: str) -> List[str]:
    """Generate valid actions for current state.
    
    Args:
        env_state: Current environment snapshot (str) - domain-specific format.
                   e.g., "block A on table, block B on A" for BlocksWorld.
                   Parse this string to extract state information.
    
    Returns:
        List of valid action strings for the current state
    """
```

**`init_state()` - Instance Method**
```python
def init_state(self, **kwargs) -> EnvState:
    """Initialize state from dataset example.
    
    Args:
        **kwargs: Fields from dataset example dict. Extract what you need.
                  Convention: env_grounded tasks expect 'init_state_str' in kwargs.
    
    Returns:
        EnvState: Framework's trajectory container initialized with init_state.
                  EnvState stores trajectory history; access current snapshot
                  via the env_state property.
    
    Raises:
        ValueError: With helpful message if required kwargs missing
    """
```

**`_step()` - Instance Method**
```python
def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> Tuple[EnvState, dict]:
    """Execute action and return new state.
    
    Args:
        state: EnvState container (trajectory history + current snapshot).
               Use state.env_state to get current snapshot (str) for domain logic.
        step_or_action: EnvStep (from policy) or EnvAction to execute
        query_or_goals: Goal description string
    
    Returns:
        (new_state, aux_dict): 
            - new_state: Updated EnvState with new step appended
            - aux_dict: Must include 'goal_reached' key with (bool, float) tuple
    """
```

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
        """Check if robot reached target position."""
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
        """Initialize state from dataset example."""
        init_str = kwargs.get('init_state_str')
        if init_str is None:
            raise ValueError("RobotArmTransition requires 'init_state_str' in kwargs")
        return EnvState(init_state=init_str)
    
    def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> Tuple[EnvState, dict]:
        """Execute action and update state."""
        import copy
        action = step_or_action.action if isinstance(step_or_action, EnvStep) else step_or_action
        new_state = copy.deepcopy(state)
        new_env_state = self._apply_action(new_state.env_state, action)
        new_step = EnvStep(action=action, next_state=new_env_state)
        new_state.append(new_step)
        goal_reached, progress = self.goal_check(query_or_goals, new_env_state)
        return new_state, {"goal_reached": (goal_reached, progress)}
    
    def _apply_action(self, env_state: str, action: EnvAction) -> str:
        """Domain-specific action execution logic."""
        ...
```

Once registered, set `benchmark_name="robot_arm"` in your config and tree search works automatically.

#### Using LLM in Transitions

If your domain requires LLM calls (e.g., for reasoning-based state updates), use the `_call_model()` helper:

```python
def _step(self, state, step_or_action, query_or_goals, **kwargs):
    prompt = f"Given state: {state.env_state}\nAction: {action}\nWhat is the new state?"
    response = self._call_model(prompt, temperature=0.0)
    new_env_state = response.text.strip()
    # ... rest of step logic
```

#### Extension Scenarios for env_grounded Tasks

LiTS uses a **registry-first with fallback** pattern:

| Scenario | What to Register | Policy | RewardModel |
|----------|------------------|--------|-------------|
| **Simple** | Transition only | Generic `EnvGroundedPolicy` | Generic `EnvGroundedPRM` |
| **Custom action selection** | Transition + Policy | Custom subclass | Generic `EnvGroundedPRM` |
| **Custom reward shaping** | Transition + RewardModel | Generic `EnvGroundedPolicy` | Custom subclass |
| **Full customization** | All three | Custom subclass | Custom subclass |

**Scenario 1: Simple Domain (Transition Only)**

```python
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

**Scenario 2: Custom Reward Shaping**

```python
@register_reward_model("robot_arm", task_type="env_grounded")
class RobotArmPRM(EnvGroundedPRM):
    """Reward model with distance-based progress and collision penalty."""
    
    def _fast_reward(self, state, action, query, query_idx, from_phase=""):
        # Distance-based reward (domain-specific, no LLM needed)
        target = parse_target(query)
        current = parse_position(state.env_state)
        distance_reward = 1.0 - (compute_distance(target, current) / 10.0)
        
        # Collision penalty
        collision_penalty = -1.0 if self._near_obstacle(current) else 0.0
        
        return max(0.0, distance_reward + collision_penalty), {}
```

**How the Factory Resolves Components:**

```python
def create_components_env_grounded(benchmark_name, ...):
    # 1. Transition: Required - must be registered
    TransitionCls = ComponentRegistry.get_transition(benchmark_name)
    
    # 2. Policy: Optional - falls back to EnvGroundedPolicy
    try:
        PolicyCls = ComponentRegistry.get_policy(benchmark_name)
    except KeyError:
        PolicyCls = EnvGroundedPolicy
    
    # 3. RewardModel: Optional - falls back to EnvGroundedPRM
    try:
        RewardModelCls = ComponentRegistry.get_reward_model(benchmark_name)
    except KeyError:
        RewardModelCls = EnvGroundedPRM
```

### Section 2.3: Adding Tool-Use Tasks (tool_use)

For tool-use tasks, domain experts only need to define **tools** and **prompts**. The generic `ToolUsePolicy` and `ToolUseTransition` handle all orchestration.

```python
from lits.tools import Tool

class SQLExecutor(Tool):
    name = "execute_sql"
    description = "Execute SQL query against the database"
    
    def run(self, query: str) -> str:
        return self.client.execute(query)

# Use with generic components
tools = [SQLExecutor(client=db_client)]
policy = ToolUsePolicy(base_model=model, tools=tools)
transition = ToolUseTransition(tools=tools)
agent = ReActChat(policy=policy, transition=transition)
```

**Key insight:** `ToolUseTransition` is domain-agnostic—it simply calls `execute_tool_action(action, tools)` and handles answer/error/action cases uniformly. Domain-specific logic lives entirely in the tool implementations.

### Section 2.4: Adding Reasoning Tasks (language_grounded)

For language-grounded tasks (math reasoning, QA), LiTS provides multiple reasoning formulations:

#### Built-in Formulations

| Formulation | Step Type | Policy/Transition | Use Case |
|-------------|-----------|-------------------|----------|
| **Thought Concatenation** | `ThoughtStep(action=reasoning_text)` | `ConcatPolicy`/`ConcatTransition` | ReST, Chain-of-Thought |
| **Sub-QA Decomposition** | `SubQAStep(sub_question, sub_answer)` | `RAPPolicy`/`RAPTransition` | RAP-style reasoning |

#### Minimal Setup (Thought Concatenation)

For most tasks, inject **prompts** and **datasets**. The generic components are task-agnostic:

```python
from lits.registry import register_dataset

@register_dataset("my_math_task", task_type="language_grounded")
def load_my_math_task(split="test", **kwargs):
    # Load and return dataset with 'question' and 'answer' fields
    ...
```

**Prompt injection:**
```python
@register_system_prompt("policy", "concat", "my_math_task")
def my_system_prompt():
    return "You are solving math problems step by step..."
```

#### Custom Reasoning Formulations (Advanced)

AI/NLP researchers can define entirely new reasoning formulations by creating custom Step/State structures:

| Formulation | Step Structure | Example Use Case |
|-------------|----------------|------------------|
| **Critique-Refine** | `CritiqueStep(draft, critique, refined)` | Self-improvement reasoning |
| **Hypothesis-Test** | `HypothesisStep(hypothesis, evidence, verdict)` | Scientific reasoning |
| **Plan-Execute** | `PlanStep(plan, execution_result)` | Multi-step planning |
| **Multi-perspective** | `PerspectiveStep(perspective_id, reasoning)` | Ensemble reasoning |

To implement a custom formulation:
1. Define custom `Step` and `State` dataclasses in `lits/structures/`
2. Implement corresponding `Policy` and `Transition` classes
3. Register with `@register_policy` and `@register_transition`

### Section 2.5: Registering Custom Prompts

```python
from lits.registry import register_system_prompt, register_user_prompt

# prompt_key can be a benchmark name (e.g., 'blocksworld') or task type (e.g., 'language_grounded')
@register_system_prompt("policy", "concat", "my_math_task")
def my_system_prompt():
    return "You are solving math problems step by step..."

@register_user_prompt("policy", "concat", "my_math_task")
def my_user_prompt():
    return {"question_format": "Problem: {question}"}
```

For detailed guidance on adding prompts to the framework, see [Adding Prompts Guide](prompts/ADDING_PROMPTS.md).

## Section 3: Component Compatibility

### Section 3.1: Task Type Comparison

| Aspect | env_grounded | tool_use | language_grounded |
|--------|--------------|----------|-------------------|
| **Transition** | Custom (domain logic) | Generic (`ToolUseTransition`) | Generic (`ConcatTransition`) or custom |
| **Policy** | Generic + `generate_actions` | Generic (`ToolUsePolicy`) | Generic (`ConcatPolicy`) or custom |
| **Primary injection** | Transition class | Tools | Prompts + Dataset |
| **State representation** | String (domain-parsed) | Tool call history | Reasoning steps |
| **Plug-and-play level** | Medium (implement Transition) | High (define tools only) | Highest (prompts + data) |

### Section 3.2: Component Compatibility by TASK_TYPE and Method

Components define their interface category via the `TASK_TYPE` class constant:

| TASK_TYPE | Method | Policy | Transition | RewardModel | State/Step Types | Notes |
|-----------|--------|--------|------------|-------------|------------------|-------|
| **env_grounded** | Chain (`EnvChain`) | `EnvGroundedPolicy` | Custom | — | `EnvState` / `EnvStep` | Sequential execution |
| **env_grounded** | Tree (RAP/REST/BFS) | `EnvGroundedPolicy` | Custom | `EnvGroundedPRM` | `EnvState` / `EnvStep` | Same components for all tree methods |
| **tool_use** | Chain (`ReActChat`) | `ToolUsePolicy` | `ToolUseTransition` | — | `ToolUseState` / `ToolUseStep` | Sequential tool execution |
| **tool_use** | Tree (REST/BFS) | `ToolUsePolicy` | `ToolUseTransition` | `ToolUsePRM` | `ToolUseState` / `ToolUseStep` | Chain + RewardModel |
| **language_grounded** | Tree (RAP) | `RAPPolicy` | `RAPTransition` | `RapPRM` | `SubQAState` / `SubQAStep` | Sub-question decomposition |
| **language_grounded** | Tree (REST/BFS) | `ConcatPolicy` | `ConcatTransition` | `GenerativePRM` | `ThoughtState` / `ThoughtStep` | Chain-of-thought |

**Key Insight:** For env_grounded and tool_use tasks, chain and tree methods share the same Policy and Transition. Tree search adds a RewardModel for evaluating candidate paths. For env_grounded, all tree variants (RAP, REST, BFS) use identical components—only search hyperparameters differ.

### Section 3.3: TASK_TYPE vs task_name

| Concept | Purpose | Example Values | Where Used |
|---------|---------|----------------|------------|
| **TASK_TYPE** (class constant) | Interface category | `'env_grounded'`, `'tool_use'`, `'language_grounded'`, `None` | Component class definitions |
| **task_name** (constructor param) | Prompt registry lookup key | `'blocksworld'`, `'gsm8k'`, `'mapeval-sql'` | Component instantiation |

```python
class EnvGroundedPolicy(Policy):
    TASK_TYPE = "env_grounded"  # Interface category
    
    def __init__(self, base_model, task_name: str, ...):
        # task_name (e.g., 'blocksworld') used for prompt lookup
        super().__init__(base_model, task_name=task_name, ...)
```

**Why no chain method for language_grounded?** Language-grounded QA tasks don't require interrupting the reasoning chain for external execution. Chain-of-thought can be generated in a single LLM inference. Tree search is used only when exploring multiple reasoning paths is beneficial.

### Section 3.4: Task-Instance-Specific Components

Some components are specific to a single task instance and set `TASK_TYPE = None`:

| Component | TASK_TYPE | Task Instance | Why Task-Specific |
|-----------|-----------|---------------|-------------------|
| `BlocksWorldTransition` | `None` | `blocksworld` | Implements BlocksWorld-specific state parsing and goal checking |

Setting `TASK_TYPE = None` prevents the prompt registry from falling back to a generic task type prompt, forcing explicit prompt registration under the specific `task_name`.

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

### Section 4.3: Tree Search Uses Actions as Node Identity

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

LiTS uses a centralized registry to manage prompts for all LLM-based components.

### Section 7.1: Prompt Types

- **`task_prompt_spec`** (System Prompt): Instructions, format specifications, few-shot examples
- **`usr_prompt_spec`** (User Template): Structures for formatting query-specific information

### Section 7.2: Registry Lookup Priority

1. **Explicit parameter** - If `task_prompt_spec` or `usr_prompt_spec` is passed directly
2. **task_name lookup** - Registry lookup by `task_name` (e.g., `'blocksworld'`, `'crosswords'`)
3. **TASK_TYPE lookup** - Registry lookup by component's `TASK_TYPE` (e.g., `'env_grounded'`) - **skipped if TASK_TYPE is None**
4. **Default** - Registry lookup with key `'default'`

```python
# EnvGroundedPolicy with TASK_TYPE = "env_grounded"
policy = EnvGroundedPolicy(
    base_model=model,
    task_name='crosswords'  # Lookup: 'crosswords' → 'env_grounded' (fallback) → 'default'
)

# BlocksWorldTransition with TASK_TYPE = None
transition = BlocksWorldTransition(
    base_model=model,
    task_name='blocksworld'  # Lookup: 'blocksworld' → 'default' (no TASK_TYPE fallback)
)
```

### Section 7.3: Direct Prompt Injection

For one-off experiments, inject prompts directly without registration:

```python
policy = RAPPolicy(
    base_model=model,
    task_prompt_spec="Custom system instructions...",
    task_name='gsm8k'  # usr_prompt_spec loaded from registry
)
```

### Section 7.4: Programmatic Registration

```python
from lits.prompts.registry import PromptRegistry

PromptRegistry.register('policy', 'rap', 'my_task', 'Custom instructions...')
PromptRegistry.register_usr('policy', 'rap', 'my_task', {'format': 'custom'})

policy = RAPPolicy(base_model=model, task_name='my_task')
```

### Section 7.5: Best Practices

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

# ✓ usr_prompt_spec: dict or PromptTemplate (NOT string)
usr_prompt_spec = {'question': 'Q: {q}', 'answer': 'A: {a}'}

# ✗ BAD: usr_prompt_spec as plain string
usr_prompt_spec = "Q: {question}"  # Use dict instead!
```

## See Also

- [Data Structures Guide](structures/STRUCTURES.md) - EnvState vs env_state, custom Step/State design
- [Adding Prompts Guide](prompts/ADDING_PROMPTS.md) - How to add new prompts to the framework
- [Component Base Classes](../lits/components/base.py) - Abstract interfaces
- [Prompt Registry](../lits/prompts/registry.py) - Prompt management
- [Tree Search Guide](agents/TREE_SEARCH_GUIDE.md) - Using tree search algorithms
- [ToolUseTransition](components/transitions/TOOL_USE_TRANSITION.md) - Tool execution example
- [RewardModel Interface](components/REWARD_MODEL_INTERFACE.md) - Evaluation interface
