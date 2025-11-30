# RewardModel Interface Guide

## Overview

The `RewardModel` class provides a modular interface for evaluating actions in tree search algorithms. It supports two evaluation modes:

1. **Fast Reward** - Evaluate actions without execution (for pruning)
2. **Full Reward** - Evaluate actions after execution (for scoring)

## Base Class

```python
from lits.components.base import RewardModel

class RewardModel(ABC, Generic[StateT, ActionT]):
    def __init__(
        self,
        base_model,
        task_prompt_spec,
        max_length=None,
        max_new_tokens=None,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        reward_alpha=0.5,
        reward_confidence_default=0.8
    ):
        ...
```

### Parameters

- `base_model`: LLM instance for generating evaluations
- `task_prompt_spec`: Unified prompt template or dictionary containing textual snippets
  - Can be a string template
  - Can be a dict with keys like `{'instruction': '...', 'examples': '...', 'format': '...'}`
  - Replaces legacy attributes like `useful_prompt_dict` (RapPRM) and `eval_instruction` (GenerativePRM)
- `max_length`: Maximum sequence length
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_k`, `top_p`: Sampling parameters
- `reward_alpha`: Weight for reward mixing
- `reward_confidence_default`: Default confidence score

## Key Methods

### 1. `fast_reward()` - Evaluate Without Execution

```python
def fast_reward(
    self,
    example,
    example_idx,
    state,
    action,
    from_phase=""
) -> tuple[float, dict]:
    """
    Generate a reward for an action without executing it.
    
    This method evaluates the potential usefulness/quality of an action based only on
    the current state and the proposed action, without actually executing the action
    to observe its outcome.
    
    Use cases:
    - Tasks where action execution is expensive (e.g., env_grounded tasks)
    - Reasoning tasks where we can evaluate thought quality before execution (e.g., math_qa with RAP)
    - Pruning unpromising actions early in tree search
    
    Args:
        example: The problem/question being solved
        example_idx: Index of the example (for logging)
        state: Current state before action execution
        action: Proposed action to evaluate
        from_phase: Description of algorithm phase (for logging)
    
    Returns:
        Tuple of (reward, auxiliary_dict) where:
        - reward: Float score indicating action quality
        - auxiliary_dict: Additional metrics (e.g., {'r_useful': probability})
    """
```

**Example Usage:**

```python
# Evaluate multiple candidate actions without executing them
state = current_state
candidate_actions = policy.get_actions(state)

for action in candidate_actions:
    reward, aux = reward_model.fast_reward(
        example=question,
        example_idx=0,
        state=state,
        action=action
    )
    print(f"Action: {action}, Fast Reward: {reward}, Useful Prob: {aux['r_useful']}")
```

### 2. `reward()` - Evaluate After Execution

```python
@abstractmethod
def reward(self, state, action, **kwargs) -> tuple[float, dict]:
    """
    Evaluate an action after it has been executed.
    
    This method assesses the quality of a state transition after the action
    has been applied and the new state observed.
    
    Args:
        state: The resulting state after action execution
        action: The action that was executed
        **kwargs: Additional context (e.g., example, example_idx)
    
    Returns:
        Tuple of (reward, auxiliary_dict)
    """
```

**Example Usage:**

```python
# Execute action and evaluate the result
next_state = world_model.step(state, action)
reward, aux = reward_model.reward(
    state=next_state,
    action=action,
    example=question,
    example_idx=0
)
```

### 3. Abstract Methods to Implement

When creating a custom reward model, implement these methods:

```python
@abstractmethod
def _fast_reward(self, example, example_idx, state, action, from_phase="") -> float:
    """
    Internal method to compute usefulness probability.
    
    Returns:
        Float between 0 and 1 indicating action usefulness
    """
    raise NotImplementedError

@abstractmethod
def calculate_reward(self, fast_reward: float) -> float:
    """
    Convert usefulness probability to reward score.
    
    Args:
        fast_reward: Probability from _fast_reward
    
    Returns:
        Reward score (can be any range depending on your reward scheme)
    """
    raise NotImplementedError

@abstractmethod
def reward(self, state, action, **kwargs) -> tuple[float, dict]:
    """
    Evaluate action after execution.
    """
    raise NotImplementedError
```

## Built-in Reward Models

### 1. GenerativePRM (Process Reward Model)

Evaluates reasoning steps using a generative LLM:

```python
from lits.components.reward.generative import GenerativePRM

reward_model = GenerativePRM(
    base_model=eval_model,
    task_prompt_spec={
        'instruction': 'Evaluate if this reasoning step is correct...',
        'format': 'Answer with Yes or No.'
    },
    n_for_correctness=5,
    n_for_usefulness=5
)
```

### 2. RapPRM (RAP Process Reward Model)

Specialized for Reasoning via Planning:

```python
from lits.components.reward.rap import RapPRM

reward_model = RapPRM(
    base_model=eval_model,
    task_prompt_spec={
        'useful_prompt': 'Is this step useful for solving the problem?',
        'correct_prompt': 'Is this step logically correct?'
    }
)
```

### 3. RLHFlowPRM (Outcome Reward Model)

Uses pre-trained reward models:

```python
from lits.components.reward.rlhflow import RLHFlowPRM

reward_model = RLHFlowPRM(
    base_model=prm_model,
    task_prompt_spec=None  # Uses model's built-in prompting
)
```

### 4. SelfConsistencyRM

Evaluates via self-consistency voting:

```python
from lits.components.reward.sc import SelfConsistencyRM

reward_model = SelfConsistencyRM(
    base_model=eval_model,
    task_prompt_spec={'instruction': 'Solve this problem...'},
    n_samples=10
)
```

## Complete Example

```python
from lits.lm import get_lm
from lits.components.reward.generative import GenerativePRM

# Load model
eval_model = get_lm("Qwen/Qwen2.5-7B-Instruct", device="cuda")

# Create reward model with unified task_prompt_spec
reward_model = GenerativePRM(
    base_model=eval_model,
    task_prompt_spec={
        'instruction': '''Evaluate if the reasoning step is correct and useful.
        
Problem: {problem}
Current reasoning: {state}
Next step: {action}

Is this step correct and useful?''',
        'format': 'Answer: Yes/No\nConfidence: 0.0-1.0'
    },
    n_for_correctness=5,
    n_for_usefulness=5,
    reward_alpha=0.5
)

# Use in tree search
state = initial_state
actions = policy.get_actions(state)

# Fast evaluation (without execution)
for action in actions:
    fast_r, aux = reward_model.fast_reward(
        example=problem,
        example_idx=0,
        state=state,
        action=action
    )
    print(f"Fast reward: {fast_r}, Useful prob: {aux['r_useful']}")

# Full evaluation (after execution)
best_action = max(actions, key=lambda a: reward_model.fast_reward(...)[0])
next_state = world_model.step(state, best_action)
full_r, aux = reward_model.reward(
    state=next_state,
    action=best_action,
    example=problem
)
print(f"Full reward: {full_r}")
```

## Design Principles

### 1. Unified `task_prompt_spec`

All reward models use `task_prompt_spec` for prompt configuration:

**Before (inconsistent):**
```python
# RapPRM used useful_prompt_dict
rap_prm = RapPRM(useful_prompt_dict={'prompt': '...'})

# GenerativePRM used eval_instruction
gen_prm = GenerativePRM(eval_instruction='...')
```

**After (unified):**
```python
# All use task_prompt_spec
rap_prm = RapPRM(task_prompt_spec={'useful_prompt': '...', 'correct_prompt': '...'})
gen_prm = GenerativePRM(task_prompt_spec={'instruction': '...', 'format': '...'})
```

### 2. Separation of Concerns

- `fast_reward()` - Public API with logging
- `_fast_reward()` - Internal implementation
- `calculate_reward()` - Probability to reward conversion
- `reward()` - Post-execution evaluation

### 3. Flexibility

The `task_prompt_spec` can be:
- A string template
- A dictionary of prompt components
- None (for models with built-in prompting)

Subclasses decide how to use it based on their needs.

## When to Use Fast Reward vs Full Reward

| Scenario | Use Fast Reward | Use Full Reward |
|----------|----------------|-----------------|
| Pruning unpromising actions | ✓ | |
| Expensive action execution | ✓ | |
| Need actual outcome | | ✓ |
| Final node scoring | | ✓ |
| MCTS simulation | ✓ | |
| BFS expansion | ✓ | |
| Terminal state evaluation | | ✓ |

## Common Patterns

### Pattern 1: Two-Stage Evaluation

```python
# Stage 1: Fast evaluation for pruning
candidates = policy.get_actions(state, n_actions=10)
fast_scores = [
    reward_model.fast_reward(example, 0, state, a)[0]
    for a in candidates
]

# Keep top-k
top_k = sorted(zip(candidates, fast_scores), key=lambda x: x[1], reverse=True)[:3]

# Stage 2: Full evaluation after execution
for action, _ in top_k:
    next_state = world_model.step(state, action)
    full_reward, aux = reward_model.reward(next_state, action)
    # Use full_reward for final decision
```

### Pattern 2: Reward Mixing

```python
# Combine fast and full rewards
fast_r, _ = reward_model.fast_reward(example, 0, state, action)
next_state = world_model.step(state, action)
full_r, _ = reward_model.reward(next_state, action)

# Mix with alpha
alpha = reward_model.reward_alpha
mixed_reward = alpha * fast_r + (1 - alpha) * full_r
```

## Testing Your Reward Model

```python
# Test fast_reward
state = test_state
action = test_action
reward, aux = reward_model.fast_reward(
    example="Test problem",
    example_idx=0,
    state=state,
    action=action
)
assert 0 <= aux['r_useful'] <= 1, "Useful prob should be in [0, 1]"
assert isinstance(reward, float), "Reward should be float"

# Test reward
next_state = world_model.step(state, action)
reward, aux = reward_model.reward(next_state, action)
assert isinstance(reward, float), "Reward should be float"
```

## See Also

- [Policy Interface](POLICY_INTERFACE.md) - For action generation
- [World Model Interface](WORLD_MODEL_INTERFACE.md) - For state transitions
- [Tree Search Algorithms](TREE_SEARCH.md) - For using reward models in search
