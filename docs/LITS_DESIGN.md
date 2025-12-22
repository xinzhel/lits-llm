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

## Section 2: Component Compatibility

### Section 2.1: Policy + Transition + RewardModel Combinations

Different task types require different component combinations:

#### Section 2.1.1: Tool Use Tasks (e.g., ReAct, Function Calling)

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

#### Section 2.1.2: Reasoning Tasks (e.g., Math QA, RAP)

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

#### Section 2.1.3: Sequential Reasoning (e.g., ReST, Chain-of-Thought)

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

#### Section 2.1.4: Environment-Grounded Tasks (e.g., BlocksWorld, Robotics)

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
from lits_benchmark.blocksworld import goal_check, generate_all_actions

policy = EnvGroundedPolicy(
    base_model=model,
    task_name='blocksworld',  # task_name for prompt registry lookup
    generate_all_actions=generate_all_actions
)
transition = BlocksWorldTransition(
    base_model=model,
    task_name='blocksworld',
    goal_check=goal_check,
    max_steps=10
)
evaluator = EnvGroundedPRM(
    base_model=eval_model,
    task_name='blocksworld'
)

# Use with tree search (RAP, REST, or BFS - all use same components)
result = bfs_topk(query, query_idx, config, transition, policy, evaluator)
```

### Section 2.2: Component Compatibility by TASK_TYPE and Method

Components define their interface category via the `TASK_TYPE` class constant:

| TASK_TYPE | Method | Policy | Transition | RewardModel | State/Step Types | LLM Type | Notes |
|-----------|--------|--------|------------|-------------|------------------|----------|-------|
| **env_grounded** | Chain (`EnvChain`) | `EnvGroundedPolicy` | `BlocksWorldTransition` | — | `EnvState` / `EnvStep` | Chat | Sequential execution without search |
| **env_grounded** | Tree (RAP / REST / BFS) | `EnvGroundedPolicy` | `BlocksWorldTransition` | `EnvGroundedPRM` | `EnvState` / `EnvStep` | Chat | **Same components for all tree methods** - only search settings differ |
| **tool_use** | Chain (`ReActChat`) | `ToolUsePolicy` | `ToolUseTransition` | — | `ToolUseState` / `ToolUseAction` | Chat | Sequential tool execution with observations |
| **tool_use** | Tree (REST / BFS) | `ToolUsePolicy` | `ToolUseTransition` | `ToolUsePRM` | `ToolUseState` / `ToolUseAction` | Chat | Same components as chain + RewardModel |
| **language_grounded** | Tree (RAP) | `RAPPolicy` | `RAPTransition` | `RapPRM` | `SubQAState` / `SubQAStep` | **Completion** | Requires completion model, sub-question decomposition |
| **language_grounded** | Tree (REST / BFS) | `ConcatPolicy` | `ConcatTransition` | `GenerativePRM` | `ThoughtState` / `ThoughtStep` | Chat | Chain-of-thought style reasoning |

### Section 2.3: TASK_TYPE vs task_name

The framework distinguishes between two concepts:

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

**Why no chain method for language_grounded?** Language-grounded QA tasks (math reasoning, multi-hop QA) don't require interrupting the reasoning chain for external execution. Chain-of-thought can be generated in a single LLM inference due to the sequential nature of language models. Tree search is used only when exploring multiple reasoning paths is beneficial.

**Key Insight:** For environment-grounded and tool-use tasks, chain and tree methods share the same Policy and Transition components. Tree search simply adds a RewardModel for evaluating and selecting among multiple candidate paths. For env_grounded tasks, all tree search variants (RAP, REST, BFS) use identical components—only the search hyperparameters differ:

| Setting | RAP (MCTS) | REST (MCTS) | BFS |
|---------|------------|-------------|-----|
| Algorithm | MCTS with UCB | MCTS with value estimation | Beam search |
| `n_iterations` | Higher (100+) | Medium (50) | N/A |
| `beam_width` | N/A | N/A | Configurable (5-20) |
| `depth_limit` | Configurable | Configurable | Configurable |
| Reward aggregation | Backpropagation | Backpropagation | Cumulative |

For QA tasks, RAP requires fundamentally different components because it uses sub-question decomposition with a completion-style LLM, while REST/BFS use chain-of-thought with chat models.

### Section 2.4: Task-Instance-Specific Components

Most components are designed to work across multiple task instances within a `TASK_TYPE`. However, some components are specific to a single task instance and set `TASK_TYPE = None`:

| Component | TASK_TYPE | Task Instance | Why Task-Specific |
|-----------|-----------|---------------|-------------------|
| `BlocksWorldTransition` | `None` | `blocksworld` | Implements BlocksWorld-specific state parsing and goal checking |

**Why `TASK_TYPE = None` for task-instance-specific components?**

Setting `TASK_TYPE = None` prevents the prompt registry from falling back to a generic task type prompt when `task_name` lookup fails. This is important because:

1. **Prevents format mismatches** - Task-instance-specific components often expect specific output formats that generic prompts don't provide
2. **Forces explicit prompt registration** - Developers must register prompts under the specific `task_name`
3. **Enables extensibility** - Components with `TASK_TYPE = None` can be adapted for new tasks beyond predefined categories by registering appropriate prompts

**Example:**
```python
class BlocksWorldTransition(LlmTransition):
    # TASK_TYPE is None to prevent fallback to generic 'env_grounded' prompts
    # Prompts must be registered under task_name='blocksworld'
    TASK_TYPE: str = None
```

## Section 3: Prompt Registry System

LiTS uses a centralized registry to manage prompts for all LLM-based components. This enables:
- Task-specific prompt customization
- Easy prompt injection for experiments
- Consistent prompt management across components

### Section 3.1: Prompt Types

Components use two types of prompts:
- **`task_prompt_spec`** (System Prompt): Instructions, format specifications, few-shot examples
- **`usr_prompt_spec`** (User Template): Structures for formatting query-specific information

### Section 3.2: Registry Lookup Priority

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

### Section 3.3: Registered Prompts

#### Section 3.3.1: Policy Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `concat` | `language_grounded` | ✓ | — |
| `env_grounded` | `blocksworld` | — | ✓ |
| `tool_use` | `default` | ✓ | — |

#### Section 3.3.2: Reward Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | — |
| `generative` | `language_grounded` | ✓ | — |
| `env_grounded` | `blocksworld` | ✓ | ✓ |

#### Section 3.3.3: Transition Prompts
| Agent | Task Type | System Prompt | User Template |
|-------|-----------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `rap` | `default` | ✓ | — |
| `blocksworld` | `default` | ✓ | ✓ |

### Section 3.4: Prompt Injection Methods

#### Section 3.4.1: Direct Injection

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

#### Section 3.4.2: Registry Injection

```python
from lits.prompts.registry import PromptRegistry

# Register for a new task
PromptRegistry.register('policy', 'rap', 'my_task', 'Custom instructions...')
PromptRegistry.register_usr('policy', 'rap', 'my_task', {'format': 'custom'})

# Now components with task_name='my_task' use these prompts
policy = RAPPolicy(base_model=model, task_name='my_task')
```

### Section 3.5: Adding New Prompts

#### Step 1: Create Prompt File

```python
# lits/prompts/policy/my_agent.py

# System prompt for language_grounded tasks
task_prompt_spec_language_grounded = """
Your system instructions here...
"""

# User template for language_grounded tasks
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
    
    if hasattr(my_agent, 'task_prompt_spec_language_grounded'):
        PromptRegistry.register(
            'policy', 'my_agent', 'language_grounded',
            my_agent.task_prompt_spec_language_grounded
        )
    
    if hasattr(my_agent, 'usr_prompt_spec_language_grounded'):
        PromptRegistry.register_usr(
            'policy', 'my_agent', 'language_grounded',
            my_agent.usr_prompt_spec_language_grounded
        )
```

#### Step 3: Implement Component

```python
from ..base import Policy

class MyAgentPolicy(Policy):
    TASK_TYPE = "language_grounded"  # Used for prompt fallback lookup
    
    def _get_agent_name(self) -> str:
        return 'my_agent'  # Must match registry key
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # self.task_prompt_spec and self.usr_prompt_spec are auto-loaded
        ...
```

### Section 3.6: Best Practices

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

**When to use `TASK_TYPE = None`:**
- Component requires task-instance-specific prompts with specific output formats
- Component's parsing logic depends on prompt-generated text structure
- You want to prevent accidental fallback to generic prompts

## Section 4: Key Design Patterns

### Section 4.1: Policy Returns Steps, Transition Receives Steps (v0.2.5+)

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

### Section 4.2: Transition Handles Multiple Step Types

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

### Section 4.3: Chain Agents Pass Full Steps to Transition

**Pattern:**
```python
# In ReActChat.update_state()
policy_step = policy.get_actions(...)[0]

# Pass full step to transition (handles action/answer/error)
new_state, aux = transition.step(state, policy_step, ...)

return new_state
```

**Rationale:** Transition receives full step context (think, assistant_message) and handles all cases internally, eliminating special case logic in agents.

### Section 4.4: Tree Search Uses Actions as Node Identity

**Pattern:**
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

```python
# ✅ CORRECT - transition handles all step types
class ReActChat:
    def update_state(self, query, state, ...):
        step = policy.get_actions(...)[0]
        
        # ✅ Transition handles action/answer/error/malformed
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
    action = node.action  # Only action, no step context
    node.state, aux = transition_model.step(node.parent.state, action, ...)
```

```python
# ✅ CORRECT - preserves step context (v0.2.5+ pattern)
def _world_modeling(query, node, transition_model, ...):
    step_or_action = getattr(node, 'step', node.action)  # Full step if available
    node.state, aux = transition_model.step(node.parent.state, step_or_action, ...)
```

## Section 7: Extension Points

### Section 7.1: Adding New Task Types

1. Define State and Action types
2. Implement Policy for action generation (set `TASK_TYPE` class constant, or `None` for task-instance-specific)
3. Implement Transition for execution (set `TASK_TYPE` class constant, or `None` for task-instance-specific)
4. (Optional) Implement RewardModel for evaluation (set `TASK_TYPE` class constant)
5. Register prompts in `PromptRegistry` with your `task_name` or `TASK_TYPE`
6. Use with existing agents (ReActChat, MCTS, BFS)

### Section 7.2: Adding New Agents

1. Implement agent loop (e.g., new search algorithm)
2. Use existing Policy/Transition/RewardModel interfaces
3. Follow the pattern: Policy → extract action → Transition

### Section 7.3: Adding New Reward Models

1. Inherit from `RewardModel` base class
2. Implement `_fast_reward()` and `reward()` methods
3. Use with existing tree search algorithms

## See Also

- [Component Base Classes](../lits/components/base.py) - Abstract interfaces
- [Prompt Registry](../lits/prompts/registry.py) - Prompt management
- [Tree Search Guide](agents/TREE_SEARCH_GUIDE.md) - Using tree search algorithms
- [ToolUseTransition](components/transitions/TOOL_USE_TRANSITION.md) - Tool execution example
- [RewardModel Interface](components/REWARD_MODEL_INTERFACE.md) - Evaluation interface
