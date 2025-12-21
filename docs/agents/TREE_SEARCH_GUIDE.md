# Tree Search Agents User Guide

This guide explains how to use tree search agents (MCTS and BFS) in LITS-LLM for complex reasoning tasks.

## Overview

LITS-LLM provides two main tree search algorithms:
- **MCTS (Monte Carlo Tree Search)**: Used by RAP and ReST methods
- **BFS (Breadth-First Search)**: Beam search with configurable beam size

Both algorithms share a **unified interface**, making it easy to switch between methods without changing your code.

## Quick Start

### Basic Usage

```python
from lits.agents.tree.mcts import mcts
from lits.agents.tree.bfs import bfs_topk
from lits.agents.tree.common import extract_answers_from_terminal_nodes

# Configure search
search_config = BaseSearchConfig(
    reasoning_method="rest",  # or "rap", "bfs"
    max_steps=10,
    n_actions=3
)

# Run search (identical interface for both)
# Note: Use query_idx instead of example_idx (v0.2.5+)
result = mcts(
    example=question,
    query_idx=0,
    mcts_search_config=search_config,
    world_model=world_model,
    policy=policy,
    reward_model=evaluator,
    bn_evaluator=None
)

# Or use BFS (same signature)
result = bfs_topk(
    question=question,
    query_idx=0,
    search_config=search_config,
    world_model=world_model,
    policy=policy,
    evaluator=evaluator,
    bn_evaluator=None
)

# Terminal nodes are automatically saved to checkpoint files
# Post-processing (answer extraction) is done separately using eval_search.py
```

## Configuration

### Common Configuration (BaseSearchConfig)

All search methods inherit from `BaseSearchConfig`:

```python
from lits.agents.tree.base import BaseSearchConfig

config = BaseSearchConfig(
    reasoning_method="rest",      # "rest", "rap", or "bfs"
    package_version="v0.2.5",
    
    # Common attributes
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    gpu_device="cuda:0",
    max_length=32768,
    max_steps=10,                 # Maximum search depth
    
    # Action generation
    n_actions=3,                  # Number of actions per node
    
    # Termination
    force_terminating_on_depth_limit=True,
    terminate_on_terminal_node=True,
    
    # Evaluation
    enable_think_policy=True,
    enable_think_eval=True
)
```

### ExperimentConfig for Full Experiments

For running complete experiments with dataset slicing and result management:

```python
from examples.search_config import ExperimentConfig

config = ExperimentConfig(
    # Dataset and models
    dataset_name="gsm8k",
    policy_model_name="meta-llama/Llama-3.1-8B-Instruct",
    eval_model_name="meta-llama/Llama-3.1-8B-Instruct",
    reasoning_method="bfs",
    
    # Search parameters
    n_actions=3,
    max_steps=10,
    
    # Dataset slicing (v0.2.5+)
    offset=0,                     # Starting index for dataset slicing
    limit=100,                    # Number of examples to evaluate (None = all from offset)
    eval_idx=[],                  # Specific indices to evaluate (overrides offset/limit)
    
    # Continuation (optional)
    add_continuation=False,
    bn_method=None,               # "entropy", "sc", or "direct"
    
    # Logging
    verbose=True,
    package_version="v0.2.5"
)

# Dataset slicing examples:
# Evaluate first 100 examples:
config = ExperimentConfig(..., offset=0, limit=100)

# Evaluate examples 100-200:
config = ExperimentConfig(..., offset=100, limit=100)

# Evaluate all examples from index 50 onwards:
config = ExperimentConfig(..., offset=50, limit=None)

# Evaluate specific indices (overrides offset/limit):
config = ExperimentConfig(..., eval_idx=[0, 5, 10, 15])
```

### BFS-Specific Configuration

```python
from lits.agents.tree.bfs import BFSConfig

bfs_config = BFSConfig(
    reasoning_method="bfs",
    max_steps=10,
    n_actions=3,
    beam_size=5  # BFS-specific: beam width
)
```

### MCTS-Specific Configuration

MCTS uses `BaseSearchConfig` with additional parameters for exploration, simulation, and continuation.

### MCTS Termination Behavior

MCTS has multiple termination controls that operate at different levels:

#### Separation of Concerns

There are two distinct types of termination in MCTS:

1. **Task-level termination** (`Transition.is_terminal()`): Checks if the **goal/answer is achieved**
   - Implemented in your world model (e.g., `BlocksWorldTransition.is_terminal()`)
   - Should ONLY check goal achievement, NOT max_steps

2. **Search-level termination** (`_is_terminal_with_depth_limit()`): Checks if **max_steps is reached**
   - Handled automatically by tree search algorithms
   - Controlled by `max_steps` and `force_terminating_on_depth_limit` config

**Important:** Do NOT check `max_steps` in your `Transition.is_terminal()` method. This is handled by the search algorithm.

#### Termination Configuration Parameters

```python
config = BaseSearchConfig(
    # Search depth limit
    max_steps=10,
    force_terminating_on_depth_limit=True,  # Force stop at max_steps
    
    # Terminal node handling during selection
    terminate_on_terminal_node=True,  # Stop iteration when selecting a terminal node
    
    # Early termination for feasibility checking
    terminate_on_first_solution=False,  # Stop MCTS when first solution is found
)
```

#### Parameter Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `terminate_on_terminal_node` | `True` | When selecting a previously-visited terminal node, stop the current iteration. Set to `False` to continue exploring other branches. |
| `terminate_on_first_solution` | `False` | When any path reaches a terminal node (goal achieved) during simulation, immediately stop MCTS. Useful for feasibility checking where you only care if a solution exists, not finding the optimal one. |

#### Use Cases

**Finding optimal solutions (default):**
```python
config = BaseSearchConfig(
    terminate_on_terminal_node=True,   # Default
    terminate_on_first_solution=False,  # Default - explore all iterations
)
# MCTS will run all n_iters iterations to find the best solution
```

**Feasibility checking (stop on first solution):**
```python
config = BaseSearchConfig(
    terminate_on_first_solution=True,  # Stop as soon as any solution is found
)
# Useful for environment-grounded tasks where you only need to verify
# that a valid solution exists, not find the optimal one
```

**Full exploration (never stop early):**
```python
config = BaseSearchConfig(
    terminate_on_terminal_node=False,  # Continue even when selecting terminal nodes
    terminate_on_first_solution=False,  # Run all iterations
)
# Maximum exploration - useful for collecting diverse solutions
```

## Result Structure

Both MCTS and BFS return results with a unified structure:

### Common Attributes

```python
# Both MCTSResult and BFSResult have:
result.root                      # Root node of search tree
result.terminal_nodes_collected  # All terminal nodes (for post-processing)
```

### MCTS-Specific Attributes

```python
result.cum_reward                # Cumulative reward of best path
result.trace                     # (states, actions) tuple
result.trace_of_nodes            # Best path as list of nodes
result.trace_in_each_iter        # Paths from each iteration
result.unselected_terminal_paths_during_simulate  # Unselected paths
```

### BFS-Specific Attributes

```python
result.buckets_with_terminal  # All nodes organized by depth
```

## Post-Processing

### Extract Answers from Terminal Nodes

The `extract_answers_from_terminal_nodes()` function processes terminal nodes to extract answers:

```python
from lits.agents.tree.common import extract_answers_from_terminal_nodes

vote_answers, answer_rewards, best_node, trace = extract_answers_from_terminal_nodes(
    terminal_nodes_collected=result.terminal_nodes_collected,
    retrieve_answer=retrieve_answer_fn,
    question=question
)

# vote_answers: Dict[str, int] - answer -> vote count
# answer_rewards: Dict[str, List[float]] - answer -> list of rewards
# best_node: Node with highest reward
# trace: Path from root to best_node
```

### Answer Retrieval Function

You need to provide a function to extract answers from node states:

```python
def retrieve_answer(state, question):
    """Extract answer from the final state."""
    if state and len(state) > 0:
        last_step = state[-1]
        # Extract answer from last step
        if hasattr(last_step, 'answer'):
            return last_step.answer
        # Or parse from action text
        action = last_step.get('action', '')
        if 'answer is' in action.lower():
            # Extract answer...
            return extracted_answer
    return None
```

## Visualization

Both MCTS and BFS results can be visualized using the same tools:

```python
from lits.visualize import visualize_mcts_result, visualize_bfs_result

# Visualize MCTS
visualize_mcts_result(mcts_result, 'output/mcts_tree', format='pdf')

# Visualize BFS
visualize_bfs_result(bfs_result, 'output/bfs_tree', format='pdf')

# Or use unified interface
from lits.visualize import get_tree_from_result, plot_save_tree

paths = get_tree_from_result(result, idx=0, full_dataset=dataset)
plot_save_tree(paths, 'output/tree', format='pdf')
```

See [TREE_VISUALIZATION.md](../TREE_VISUALIZATION.md) for details.

## Complete Example

```python
from lits.agents.tree.mcts import mcts
from lits.agents.tree.bfs import bfs_topk
from lits.agents.tree.base import BaseSearchConfig
from lits.agents.tree.common import extract_answers_from_terminal_nodes
from lits.visualize import buckets_to_paths

# 1. Configure search
config = BaseSearchConfig(
    reasoning_method="rest",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_steps=10,
    n_actions=3
)

# 2. Run search
result = mcts(
    example=question,
    example_idx=0,
    mcts_search_config=config,
    world_model=world_model,
    policy=policy,
    reward_model=evaluator
)

# 3. Post-process: extract answers
vote_answers, answer_rewards, best_node, trace = extract_answers_from_terminal_nodes(
    terminal_nodes_collected=result.terminal_nodes_collected,
    retrieve_answer=retrieve_answer_fn,
    question=question
)

# 4. Get final answer
if vote_answers:
    final_answer = max(vote_answers, key=lambda a: vote_answers[a])
    print(f"Answer: {final_answer}")
    print(f"Votes: {vote_answers}")
    print(f"Rewards: {answer_rewards}")

# 5. Save paths for later analysis
if hasattr(result, 'trace_in_each_iter'):
    # MCTS: use iteration traces
    paths = [result.trace_of_nodes] + result.trace_in_each_iter
else:
    # BFS: convert buckets to paths
    paths = buckets_to_paths(result.buckets_with_terminal)

# Save to file
from lits.eval import TreeToJsonl
saver = TreeToJsonl(run_id='', root_dir='./results')
saver.append_result(paths)

# 6. Visualize
from lits.visualize import visualize_mcts_result
visualize_mcts_result(result, 'output/tree', format='pdf')
```

## Switching Between Methods

The unified interface makes it easy to switch between MCTS and BFS:

```python
def run_search(question, method="mcts"):
    """Run search with specified method."""
    config = BaseSearchConfig(
        reasoning_method=method,
        max_steps=10,
        n_actions=3
    )
    
    if method in ['rap', 'rest']:
        result = mcts(question, 0, config, world_model, policy, evaluator)
    elif method == 'bfs':
        result = bfs_topk(question, 0, config, world_model, policy, evaluator)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Post-processing is identical for both
    vote_answers, _, _, _ = extract_answers_from_terminal_nodes(
        terminal_nodes_collected=result.terminal_nodes_collected,
        retrieve_answer=retrieve_answer_fn,
        question=question
    )
    
    return max(vote_answers, key=lambda a: vote_answers[a])

# Use MCTS
answer = run_search(question, method="rest")

# Switch to BFS - no code changes needed!
answer = run_search(question, method="bfs")
```

## Advanced Topics

### Custom World Models

Implement the `Transition` interface:

```python
from lits.components.base import Transition

class MyWorldModel(Transition):
    def init_state(self):
        """Initialize the starting state."""
        return []
    
    def step(self, example, state, action, query_idx=None, from_phase=""):
        """Execute action and return new state.
        
        Note: Use query_idx instead of example_idx (v0.2.5+)
        """
        new_state = state + [action]
        aux = {'confidence': 1.0}
        return new_state, aux
    
    def is_terminal(self, state, example, fast_reward=None, query_idx=None, from_phase=""):
        """Check if state is terminal.
        
        Note: Use query_idx instead of example_idx (v0.2.5+)
        """
        return len(state) >= 5  # Example: terminate after 5 steps
```

### Custom Policies

Implement the `Policy` interface (v0.2.5+ requires returning Step objects):

```python
from lits.components.base import Policy
from lits.structures import ThoughtStep

class MyPolicy(Policy):
    def _create_error_steps(self, n_actions: int, error_msg: str):
        """Create error steps when _get_actions fails."""
        return [ThoughtStep(action="", error=error_msg) for _ in range(n_actions)]
    
    def _get_actions(self, query, state, n_actions, temperature, at_depth_limit, query_idx, critic=None, from_phase="", **kwargs):
        """Generate n_actions for the given state.
        
        Returns:
            List of Step objects (e.g., ThoughtStep, SubQAStep, ToolUseStep)
        """
        steps = []
        for i in range(n_actions):
            action_text = self.generate_action(state, query)
            # Wrap action in appropriate Step type
            step = ThoughtStep(action=action_text)
            steps.append(step)
        return steps
```

**Important Changes in v0.2.5:**
- Policies must return `Step` objects (e.g., `ThoughtStep`, `SubQAStep`), not raw strings
- Must implement `_create_error_steps()` for graceful error handling
- Use `query_idx` parameter instead of `example_idx` for consistency

### Custom Reward Models

Implement the `RewardModel` interface:

```python
from lits.components.base import RewardModel

class MyRewardModel(RewardModel):
    def _fast_reward(self, example, query_idx, state, action, from_phase=""):
        """Evaluate action quality without execution.
        
        Note: Use query_idx instead of example_idx (v0.2.5+)
        """
        # Quick heuristic evaluation
        score = self.evaluate_action(action)
        return score, {'method': 'heuristic'}
    
    def reward(self, state, action, **kwargs):
        """Compute final reward after execution."""
        # Combine fast reward with state confidence
        fast_reward = kwargs.get('usefulness', 0.5)
        confidence = kwargs.get('confidence', 1.0)
        return fast_reward * confidence
    
    def calculate_reward(self, fast_reward: float) -> float:
        """Transform fast reward to final reward."""
        return fast_reward
```

## Best Practices

1. **Start with small max_steps**: Begin with `max_steps=5` and increase gradually
2. **Tune n_actions**: More actions = better exploration but slower search
3. **Use appropriate beam_size for BFS**: Larger beam = more thorough but slower
4. **Monitor terminal nodes**: Check `len(result.terminal_nodes_collected)` to ensure search finds solutions
5. **Visualize results**: Use visualization tools to understand search behavior
6. **Save paths**: Always save paths for later analysis and debugging

## Troubleshooting

### No Terminal Nodes Found

```python
if len(result.terminal_nodes_collected) == 0:
    print("No terminal nodes found!")
    # Possible solutions:
    # 1. Increase max_steps
    # 2. Check world_model.is_terminal() logic
    # 3. Verify policy generates valid actions
```

### Poor Answer Quality

```python
# Check vote distribution
print(f"Vote distribution: {vote_answers}")
print(f"Reward distribution: {answer_rewards}")

# If votes are scattered:
# 1. Increase n_actions for better exploration
# 2. Tune reward model
# 3. Check if world model is deterministic
```

### Slow Search

```python
# Profile search time
import time
start = time.time()
result = mcts(...)
print(f"Search took {time.time() - start:.2f}s")

# Optimization strategies:
# 1. Reduce max_steps
# 2. Reduce n_actions
# 3. Use smaller model
# 4. Enable GPU acceleration
```

## Related Documentation

- [Tree Visualization Guide](../TREE_VISUALIZATION.md) - Visualizing search trees
- [Config Refactoring](../../unit_test/test_config_refactoring.py) - Configuration system tests
- [Main Search Example](../../examples/main_search.py) - Complete working example
- [CHANGELOG](../../CHANGELOG.md) - Version history and breaking changes

## API Reference

### Functions

- `mcts(example, example_idx, mcts_search_config, world_model, policy, reward_model, bn_evaluator=None)` - Run MCTS search
- `bfs_topk(question, example_idx, search_config, world_model, policy, evaluator, bn_evaluator=None)` - Run BFS search
- `extract_answers_from_terminal_nodes(terminal_nodes_collected, retrieve_answer, question)` - Extract answers from terminal nodes
- `buckets_to_paths(buckets_with_terminal)` - Convert BFS buckets to paths

### Classes

- `BaseSearchConfig` - Base configuration for all search methods
- `BFSConfig` - BFS-specific configuration
- `MCTSResult` - MCTS search result
- `BFSResult` - BFS search result

For detailed API documentation, see the source code docstrings.
