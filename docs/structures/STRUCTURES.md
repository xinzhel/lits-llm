# LiTS Data Structures

This document explains the data structures used in LiTS and the design philosophies behind them.

## Overview: Two Design Philosophies

LiTS uses two distinct approaches to data structure design depending on the task type:

| Philosophy | Task Type | Key Idea | Example |
|------------|-----------|----------|---------|
| **Semantic-agnostic containers** | env_grounded | Framework provides generic containers; domain experts inject meaning through interpretation | `EnvState` stores strings; BlocksWorld parses with regex |
| **Formulation-specific structures** | language_grounded, tool_use | Structure fields encode the reasoning formulation's design logic | `SubQAStep(sub_question, sub_answer)` reflects RAP's decomposition approach |

## Section 1: Environment-Grounded Structures

### EnvState vs env_state

LiTS distinguishes between two related but different concepts:

| Concept | Type | Purpose | Scope |
|---------|------|---------|-------|
| `EnvState` | Class | Framework's trajectory container | Framework-level |
| `env_state` | Property | Domain-specific environment snapshot | Domain-level |

**`EnvState`** is the framework's state container that:
- Stores the full trajectory history (list of `EnvStep` objects)
- Tracks `init_state` (initial environment description as string)
- Provides `env_state` property to access current snapshot
- Handles serialization/deserialization for checkpointing

**`env_state`** (accessed via `EnvState.env_state`) is the domain-specific snapshot:
- Represents the current environment at a single point in time
- Default type is `str` for text-based domains
- Domain experts inject semantics via static methods that parse this string

```python
# EnvState is the framework container
state = EnvState(init_state="block A on table, block B on A")

# env_state is the domain-specific snapshot (str by default)
current_snapshot: str = state.env_state  # "block A on table, block B on A"

# Static methods operate on the snapshot, not the container
actions = BlocksWorldTransition.generate_actions(state.env_state)  # Parses string
goal_met, progress = BlocksWorldTransition.goal_check(goals, state.env_state)
```

### Why String-Based Snapshots?

1. **Simplicity for domain experts**: Static methods only need the current snapshot, not trajectory history
2. **Flexibility**: Different domains can use different snapshot representations
3. **Efficiency**: Action generation doesn't need to process full trajectory
4. **Serialization**: Strings are trivially serializable for checkpointing

### EnvStep Structure

```python
@dataclass
class EnvStep(Step):
    action: Optional[EnvAction] = None      # The action taken
    next_state: Optional[str] = None        # Resulting state snapshot (str)
```

The `next_state` field stores the environment snapshot after executing the action—again as a string that domain experts interpret.

### Extending to Structured Domains

For domains with structured state (robotics, games), serialize to string:

```python
# Store as JSON string in generic EnvState
init_state = json.dumps({"position": [0, 0], "gripper": "open"})
state = EnvState(init_state=init_state)

# Parse in domain methods
@staticmethod
def generate_actions(env_state: str) -> List[str]:
    data = json.loads(env_state)  # Domain-specific interpretation
    position = data["position"]
    gripper = data["gripper"]
    # Generate actions based on structured state
    ...
```

Alternative: Subclass `EnvState` to override the `env_state` property:

```python
@dataclass
class RobotState(EnvState):
    @property
    def env_state(self) -> dict:
        # Return structured data instead of string
        return json.loads(super().env_state)
```

## Section 2: Language-Grounded Structures

Unlike env_grounded, language-grounded structures are **formulation-specific**—their fields encode the reasoning approach's design logic.

### Built-in Formulations

| Formulation | Step Type | Fields | Design Logic |
|-------------|-----------|--------|--------------|
| **Thought Concatenation** | `ThoughtStep` | `action: str` | Each step is a reasoning fragment; concatenate to form chain-of-thought |
| **Sub-QA Decomposition** | `SubQAStep` | `sub_question: str, sub_answer: str` | Decompose problem into sub-questions; answer each to build toward final answer |

**ThoughtStep** (ReST, Chain-of-Thought):
```python
@dataclass
class ThoughtStep(Step):
    action: str = ""  # A single reasoning step text
```

**SubQAStep** (RAP):
```python
@dataclass  
class SubQAStep(Step):
    sub_question: str = ""  # Decomposed sub-question
    sub_answer: str = ""    # Answer to the sub-question
```

### Custom Reasoning Formulations

AI/NLP researchers can define entirely new reasoning formulations by creating custom Step structures. The field design should reflect the reasoning logic:

| Formulation | Step Structure | Fields | Design Logic |
|-------------|----------------|--------|--------------|
| **Sub-QA Decomposition** | `SubQAStep` | `sub_question, sub_answer` | Decompose into sub-questions; answer each |
| **Self-Refine** | `RefineStep` | `draft, feedback, refined` | Generate → critique → refine iteratively |
| **Hypothesis-Test** | `HypothesisStep` | `hypothesis, evidence, verdict` | Propose → gather evidence → conclude |
| **Plan-Execute** | `PlanStep` | `plan, execution_result` | Create plan → execute and observe |

### Example 1: RAP (Sub-QA Decomposition)

RAP (Reasoning via Planning, Hao et al. 2023) decomposes complex questions into sub-questions, answering each to build toward the final answer.

**Step Structure:**
```python
@dataclass
class SubQAStep(Step):
    """Step for RAP-style sub-question decomposition."""
    sub_question: str = ""    # Decomposed sub-question
    sub_answer: str = ""      # Answer to the sub-question
    confidence: float = 0.0   # Confidence score for tree search
    
    def get_action(self):
        return self.sub_question  # Sub-question is the "action"
    
    def get_answer(self):
        return self.sub_answer
    
    def verb_step(self) -> str:
        return f"Q: {self.sub_question}\nA: {self.sub_answer}"
```

**Policy generates sub-questions:**
```python
class RAPPolicy(Policy):
    TASK_TYPE = "language_grounded"
    
    def _get_actions(self, state, query, n_actions, **kwargs) -> list[SubQAStep]:
        # LLM generates sub-questions based on current state
        prompt = self._build_prompt(state, query)
        outputs = self.base_model.generate(prompt, n=n_actions)
        # Return steps with sub_question filled; sub_answer filled by Transition
        return [SubQAStep(sub_question=q, sub_answer="", confidence=0.0) for q in outputs]
```

**Transition answers sub-questions:**
```python
class RAPTransition(LlmTransition):
    TASK_TYPE = "language_grounded"
    
    def _step(self, state, step_or_action, query, **kwargs):
        sub_question = step_or_action.sub_question if isinstance(step_or_action, SubQAStep) else step_or_action
        
        # LLM answers the sub-question
        prompt = f"Given context:\n{verbalize_state(state)}\n\nAnswer: {sub_question}"
        answer = self._call_model(prompt)
        
        new_state = state.copy()
        new_state.append(SubQAStep(sub_question=sub_question, sub_answer=answer, confidence=1.0))
        return new_state, {"confidence": 1.0}
```

### Example 2: Self-Refine (Iterative Refinement)

Self-Refine (Madaan et al. 2023) iteratively improves outputs through self-feedback: generate draft → critique → refine → repeat.

**Step Structure:**
```python
from dataclasses import dataclass
from lits.structures.base import Step
from lits.type_registry import register_type

@register_type
@dataclass
class RefineStep(Step):
    """Step for Self-Refine iterative improvement."""
    draft: str = ""           # Current attempt/solution
    feedback: str = ""        # Self-critique identifying issues
    refined: str = ""         # Improved version after feedback
    iteration: int = 0        # Refinement iteration number
    is_satisfactory: bool = False  # Whether refinement should stop
    
    def get_action(self):
        return self.refined if self.refined else self.draft
    
    def verb_step(self) -> str:
        parts = [f"[Iteration {self.iteration}]"]
        parts.append(f"Draft: {self.draft}")
        if self.feedback:
            parts.append(f"Feedback: {self.feedback}")
        if self.refined:
            parts.append(f"Refined: {self.refined}")
        return "\n".join(parts)
    
    def to_messages(self) -> list[dict]:
        messages = [{"role": "assistant", "content": self.draft}]
        if self.feedback:
            messages.append({"role": "user", "content": f"Feedback: {self.feedback}"})
        if self.refined:
            messages.append({"role": "assistant", "content": self.refined})
        return messages
```

**State Container:**
```python
@register_state
class RefineState(TrajectoryState[RefineStep]):
    """State tracking refinement iterations."""
    
    def get_current_solution(self) -> str:
        """Return the latest refined output."""
        if not self.steps:
            return ""
        last_step = self.steps[-1]
        return last_step.refined if last_step.refined else last_step.draft
    
    def is_converged(self) -> bool:
        """Check if refinement has converged."""
        return self.steps and self.steps[-1].is_satisfactory
```

**Policy generates drafts and feedback:**
```python
class RefinePolicy(Policy):
    TASK_TYPE = "language_grounded"
    
    def _get_actions(self, state: RefineState, query, n_actions, **kwargs) -> list[RefineStep]:
        iteration = len(state.steps)
        
        if iteration == 0:
            # Initial generation
            drafts = self.base_model.generate(f"Solve: {query}", n=n_actions)
            return [RefineStep(draft=d, iteration=0) for d in drafts]
        else:
            # Generate feedback on current solution
            current = state.get_current_solution()
            feedback_prompt = f"Critique this solution:\n{current}\n\nIdentify specific issues:"
            feedbacks = self.base_model.generate(feedback_prompt, n=n_actions)
            return [RefineStep(draft=current, feedback=f, iteration=iteration) for f in feedbacks]
```

**Transition refines based on feedback:**
```python
class RefineTransition(LlmTransition):
    TASK_TYPE = "language_grounded"
    
    def _step(self, state: RefineState, step: RefineStep, query, **kwargs):
        new_state = state.copy()
        
        if not step.feedback:
            # First iteration: just store the draft
            new_state.append(step)
        else:
            # Refine based on feedback
            refine_prompt = f"Original: {step.draft}\nFeedback: {step.feedback}\n\nImproved solution:"
            refined = self._call_model(refine_prompt)
            
            # Check if satisfactory (could use LLM or heuristic)
            is_satisfactory = self._check_satisfactory(step.draft, refined, step.feedback)
            
            step.refined = refined
            step.is_satisfactory = is_satisfactory
            new_state.append(step)
        
        return new_state, {"is_satisfactory": step.is_satisfactory}
    
    def is_terminal(self, state: RefineState, **kwargs) -> bool:
        return state.is_converged() or len(state.steps) >= self.max_iterations
```

**Using with Tree Search:**
```python
# Self-Refine can be used with tree search to explore multiple refinement paths
from lits.agents.tree import bfs

# Each branch explores different feedback → refinement trajectories
result = bfs(
    query="Solve this math problem: ...",
    policy=RefinePolicy(base_model=model),
    transition=RefineTransition(base_model=model),
    reward_model=RefineRewardModel(),  # Scores refinement quality
    n_actions=3,  # 3 different feedback options per iteration
    max_depth=5,  # Up to 5 refinement iterations
)
```

### Key Design Principles for Custom Formulations

1. **Fields reflect reasoning logic**: Each field should correspond to a distinct phase in your reasoning process
2. **`get_action()` returns the primary output**: What the search algorithm uses as the "action"
3. **`verb_step()` for debugging**: Human-readable representation for logging
4. **`to_messages()` for LLM context**: Convert to chat format for multi-turn prompting
5. **Register with `@register_type`**: Required for serialization/checkpointing

## Section 3: Tool-Use Structures

Tool-use structures are **formulation-specific**—they encode the ReAct-style interleaved reasoning and action pattern.

### ToolUseStep

```python
@dataclass
class ToolUseStep(Step):
    think: str = ""                          # Reasoning before action
    action: Optional[ToolUseAction] = None   # Tool call (JSON)
    observation: Optional[str] = None        # Tool execution result
    answer: Optional[str] = None             # Final answer (terminal)
    assistant_message: Optional[str] = None  # Raw LLM output for reconstruction
```

**Field semantics:**
- `think`: The reasoning that led to the action decision
- `action`: Tool call specification (parsed from LLM output)
- `observation`: Result from tool execution (filled by Transition)
- `answer`: Final answer when reasoning is complete (terminal state)

### ToolUseState

```python
class ToolUseState(TrajectoryState[ToolUseStep]):
    """State container for tool-use traces; each entry is a ToolUseStep."""
    
    def get_final_answer(self):
        """Return the answer from the latest step if available."""
        ...
```

### Message Conversion

ToolUseStep provides methods to convert to chat messages for LLM context:

```python
step.to_messages()  # Returns list of {"role": "assistant/user", "content": ...}
step.verb_step()    # Returns text representation
```

## Section 4: Creating Custom Structures

### Step 1: Define the Step Dataclass

```python
from dataclasses import dataclass
from typing import Optional
from lits.structures.base import Step
from lits.type_registry import register_type

@register_type  # Required for serialization
@dataclass
class MyCustomStep(Step):
    """Custom step for my reasoning formulation."""
    
    # Define fields that reflect your reasoning logic
    field1: str = ""
    field2: Optional[str] = None
    
    def get_action(self):
        """Return the primary action/output of this step."""
        return self.field1
    
    def verb_step(self) -> str:
        """Return text representation for logging/display."""
        return f"Field1: {self.field1}\nField2: {self.field2}"
    
    def to_messages(self) -> list[dict]:
        """Convert to chat messages for LLM context."""
        messages = []
        if self.field1:
            messages.append({"role": "assistant", "content": self.field1})
        if self.field2:
            messages.append({"role": "user", "content": self.field2})
        return messages
    
    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "__type__": self.__class__.__name__,
            "field1": self.field1,
            "field2": self.field2,
        }
    
    @classmethod
    def from_dict(cls, payload: dict) -> "MyCustomStep":
        """Deserialize from checkpoint."""
        return cls(
            field1=payload.get("field1", ""),
            field2=payload.get("field2"),
        )
```

### Step 2: Define the State Container

```python
from lits.structures.base import TrajectoryState
from lits.type_registry import register_state

@register_state  # Required for serialization
class MyCustomState(TrajectoryState[MyCustomStep]):
    """State container for my custom reasoning traces."""
    
    # Add any state-level properties or methods
    def get_summary(self) -> str:
        """Example: summarize the trajectory."""
        return " → ".join(step.field1 for step in self)
```

### Step 3: Register with Type Registry

The `@register_type` and `@register_state` decorators automatically register your structures for serialization. This enables:
- JSON checkpointing during tree search
- Loading saved trajectories for evaluation
- Polymorphic deserialization (loading the correct subclass)

### Step 4: Implement Corresponding Components

Create Policy and Transition that work with your custom structures:

```python
from lits.components.base import Policy, Transition

class MyCustomPolicy(Policy):
    TASK_TYPE = "language_grounded"  # or a new task type
    
    def _get_actions(self, state: MyCustomState, ...) -> List[MyCustomStep]:
        # Generate steps using your custom structure
        ...

class MyCustomTransition(Transition):
    TASK_TYPE = "language_grounded"
    
    def _step(self, state: MyCustomState, step: MyCustomStep, ...) -> Tuple[MyCustomState, dict]:
        # Process the step and update state
        ...
```

## Summary: Design Philosophy by Task Type

| Task Type | Container Philosophy | Who Defines Semantics | Example |
|-----------|---------------------|----------------------|---------|
| **env_grounded** | Semantic-agnostic | Domain expert via string parsing | BlocksWorld parses "block A on B" |
| **language_grounded** | Formulation-specific | AI/NLP researcher via field design | `SubQAStep(sub_question, sub_answer)` |
| **tool_use** | Formulation-specific | Framework (ReAct pattern) | `ToolUseStep(think, action, observation)` |

## See Also

- [LITS_DESIGN.md](../LITS_DESIGN.md) - Framework architecture overview
- [Base structures](../../lits/structures/base.py) - Step, State, Action base classes
- [Type registry](../../lits/type_registry.py) - Serialization registration
