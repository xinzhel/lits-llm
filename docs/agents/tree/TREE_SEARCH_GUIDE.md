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

### Command Line Interface (CLI)

The recommended way to run tree search experiments is via `main_search.py` with CLI arguments:

```bash
# Basic usage
python main_search.py --dataset math500 --search_framework rest

# With custom search parameters
python main_search.py --dataset gsm8k --search_framework rest \
    --search-arg n_iters=100 n_actions=5 max_steps=10

# With component parameters (e.g., ThinkPRM reward model)
python main_search.py --dataset math500 \
    --component-arg reward_model_type=thinkprm thinkprm_endpoint=my-endpoint

# Environment-grounded tasks with custom modules
python main_search.py --dataset crosswords --include lits_benchmark.crosswords \
    --search-arg n_actions=3 max_steps=10 n_iters=30 \
    --dataset-arg data_file=crosswords/data/mini0505.json

# Show all available parameters
python main_search.py --help-config
```

#### CLI Flag Reference

| Flag | Purpose | Example |
|------|---------|---------|
| `--dataset` | Dataset name | `--dataset math500` |
| `--search_framework` | Framework (rest, rap, tot_bfs) | `--search_framework rest` |
| `--search-arg` | Search algorithm params | `--search-arg n_iters=50 n_actions=3` |
| `--component-arg` | Component params | `--component-arg think_for_correctness=true` |
| `--dataset-arg` | Dataset loader kwargs | `--dataset-arg levels=1,2,3` |
| `--var` | Execution vars (not saved) | `--var offset=0 limit=50` |
| `--include` | Custom module/package include | `--include lits_benchmark.crosswords` |
| `--override` | Override existing results | `--override` |
| `--dry-run` | Test dataset loading | `--dry-run` |
| `--help-config` | Show all parameters | `--help-config` |

Note: CLI flags use singular form (`--search-arg`, `--component-arg`) while internal Python variables use plural (`search_args`, `component_args`). This follows standard argparse convention where each flag invocation adds one argument to the collection.

See `examples/run_configs.sh` for complete CLI examples.

### ExperimentConfig (Programmatic)

For programmatic configuration in Python:

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

### ExperimentConfig (Programmatic)

For programmatic configuration in Python:

```python
from lits.config import ExperimentConfig

config = ExperimentConfig(
    # Dataset and models
    dataset="gsm8k",
    policy_model_name="meta-llama/Llama-3.1-8B-Instruct",
    eval_model_name="meta-llama/Llama-3.1-8B-Instruct",
    search_framework="rest",  # "rest", "rap", "tot_bfs"
    search_algorithm="mcts",  # "mcts" or "bfs"
    
    # Search parameters (passed to algorithm)
    search_args={
        "n_actions": 3,
        "max_steps": 10,
        "n_iters": 50,
        "roll_out_steps": 2,
    },
    
    # Component parameters (passed to from_config())
    component_args={
        "think_for_correctness": True,
        "n_for_correctness": 2,
    },
    
    # Dataset slicing
    offset=0,
    limit=100,
    eval_idx=[],  # Specific indices (overrides offset/limit)
)

# Get merged parameters with defaults
search_args = config.get_search_args()
component_args = config.get_component_args()

# Create search config for algorithm
search_config = config.create_search_config()
```

#### Parameter Categories

Parameters are organized into categories:

| Category | CLI Flag | Description |
|----------|----------|-------------|
| Search Args | `--search-arg` | Algorithm params: n_iters, n_actions, max_steps, termination |
| Component Args | `--component-arg` | Component params: reward_model_type, think_for_correctness |
| Dataset Args | `--dataset-arg` | Dataset loader kwargs: data_file, levels |
| Execution Vars | `--var` | Script-level: offset, limit (not saved to config) |

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

## External Formulations

LITS supports external formulations via the `--include` flag. This allows custom search frameworks to be developed outside the core package while still integrating seamlessly with the CLI and component factory.

### Using External Formulations

External formulations are imported at runtime before component creation:

```bash
python main_search.py \
    --include lits_benchmark.formulations.rap \
    --search_framework rap \
    --dataset gsm8k \
    --policy_model_name "meta-llama/Llama-3-8B-Instruct" \
    --search-arg n_actions=3 \
    --search-arg max_steps=10
```

The `--include` flag triggers Python's import mechanism, which executes the module's `__init__.py` and registers components with `ComponentRegistry`.

### Example: RAP Formulation

RAP (Reasoning via Planning) is provided as an external formulation in `lits_benchmark/formulations/rap/`:

```
lits_benchmark/formulations/rap/
├── __init__.py       # Registration + exports
├── policy.py         # RAPPolicy with @register_policy("rap")
├── transition.py     # RAPTransition with @register_transition("rap")
├── reward.py         # RapPRM with @register_reward_model("rap")
├── structures.py     # SubQAStep (RAP-specific Step subclass)
├── utils.py          # verbalize_rap_state(), retrieve_answer_from_last_step()
└── prompts.py        # RAP prompts (policy + reward)
```

### Creating Custom Formulations

To create your own formulation:

1. **Register components with matching names**: Use the same name for all three decorators, matching your `--search_framework` value:

```python
# my_formulation/__init__.py
from lits.components.registry import (
    register_policy, register_transition, register_reward_model
)
from lits.components.base import Policy, Transition, RewardModel

@register_policy("my_formulation")
class MyPolicy(Policy):
    """My custom policy.
    
    Config Args (via --search-arg):
        n_actions: Number of actions to generate (default: 3)
        my_param: Custom parameter (default: 10)
    """
    
    @classmethod
    def from_config(cls, base_model, search_args, component_args, **kwargs):
        return cls(
            base_model=base_model,
            n_actions=search_args.get("n_actions", 3),
            my_param=search_args.get("my_param", 10),
            **kwargs
        )
    
    # ... implement _get_actions(), _create_error_steps()

@register_transition("my_formulation")
class MyTransition(Transition):
    """My custom transition."""
    
    @classmethod
    def from_config(cls, base_model, search_args, component_args, **kwargs):
        return cls(base_model=base_model, **kwargs)
    
    # ... implement step(), is_terminal()

@register_reward_model("my_formulation")
class MyRewardModel(RewardModel):
    """My custom reward model.
    
    Config Args (via --component-arg):
        reward_weight: Weight for reward calculation (default: 1.0)
    """
    
    @classmethod
    def from_config(cls, base_model, search_args, component_args, **kwargs):
        return cls(
            base_model=base_model,
            reward_weight=component_args.get("reward_weight", 1.0),
        )
    
    # ... implement _fast_reward(), reward()
```

2. **Implement `from_config()` for each component**: The factory calls `from_config()` with `search_args` and `component_args` dicts. Components extract their own parameters.

3. **Add "Config Args" docstrings**: Parameters documented in "Config Args" sections appear in `--help-config` output.

4. **Use via CLI**:

```bash
python main_search.py \
    --include my_formulation \
    --search_framework my_formulation \
    --dataset my_dataset \
    --search-arg my_param=20 \
    --component-arg reward_weight=0.5
```

### Custom Step Types

If your formulation needs a custom state representation, define a Step subclass:

```python
from lits.structures.base import Step
from lits.type_registry import register_type

@register_type
class MyStep(Step):
    """Custom step for my formulation."""
    
    my_field: str = ""
    
    @classmethod
    def verbalize_state(cls, question: str, state: list) -> str:
        """Convert state to string for answer extraction."""
        # Custom verbalization logic
        return "\n".join(step.my_field for step in state)
    
    def get_answer(self) -> str:
        """Extract answer from this step."""
        return self.my_field
```

The `verbalize_state()` classmethod is used by `get_fn_retrieve_answer()` for answer extraction from terminal nodes.

### How Component Lookup Works

When `create_components_language_grounded()` is called:

1. It normalizes the framework name (`tot_bfs` → `bfs`)
2. Looks up all three components in `ComponentRegistry` by name
3. If found, calls `from_config()` on each
4. If not found, falls back to built-in Concat components (only for `rest`/`tot_bfs`/`bfs`)
5. For unknown frameworks not in registry, raises a helpful error

This means custom formulations are first-class citizens - they use the same code path as built-in frameworks.

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
- [Main Search Example](../../examples/main_search.py) - Complete working example
- [CLI Examples](../../examples/run_configs.sh) - CLI command examples for different settings
- [CHANGELOG](../../CHANGELOG.md) - Version history and breaking changes

## Adding Custom Components to CLI Help

When you create custom components (Policy, Transition, RewardModel), you can make their parameters appear in `--help-config` by:

1. **Register with ComponentRegistry** (optional but recommended)
2. **Add a "Config Args" section to the class docstring**

### Docstring Format

The `--help-config` flag parses "Config Args" sections from class docstrings. Use this format:

```python
class MyCustomRewardModel(RewardModel):
    """My custom reward model description.
    
    Config Args (via --component-arg):
        my_param: Description of the parameter (default: value)
        another_param: Another parameter description (default: 10)
        complex_param: Multi-line descriptions are supported
            by indenting continuation lines with 8+ spaces
    
    Other docstring sections (Args, Returns, etc.) are ignored.
    """
    
    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        return cls(
            base_model=base_model,
            my_param=component_args.get('my_param', 'default_value'),
            another_param=component_args.get('another_param', 10),
        )
```

### Key Requirements

1. **Section header**: Must start with `Config Args` (case-sensitive)
2. **Parameter format**: `param_name: description (default: value)`
3. **Indentation**: Parameters must be indented (4 spaces recommended)
4. **Continuation lines**: Use 8+ spaces for multi-line descriptions

### For Search Configs

Search algorithm parameters use the same format but with `--search-arg`:

```python
@dataclass
class MySearchConfig(BaseSearchConfig):
    """My custom search configuration.
    
    Config Args (via --search-arg):
        my_search_param: Description (default: 5)
        exploration_weight: UCT exploration weight (default: 1.0)
    """
    my_search_param: int = 5
    exploration_weight: float = 1.0
```

### Automatic Discovery

Components are discovered via:
1. **ComponentRegistry**: Registered policies, transitions, and reward models
2. **Built-in fallback**: Core components (GenerativePRM, ThinkPRM, ConcatPolicy, ConcatTransition)

To register a custom component:

```python
from lits.components.registry import register_reward_model

@register_reward_model("my_task", task_type="language_grounded")
class MyCustomRewardModel(RewardModel):
    """..."""
```

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

## FAQ

### Q: Can I use my own LLM implementation instead of LITS's built-in models?

Yes. Components receive `base_model` as a parameter and only require it to be callable with signature `base_model(prompt, **kwargs) -> Output`. You have two options:

**Option 1: Write a Python script (recommended)**
```python
from lits.lm.base import Output

def my_custom_llm(prompt, **kwargs):
    # Your custom logic - call any API, local model, etc.
    response = my_api_call(prompt)
    return Output(text=response)

# Pass to components
policy = RAPPolicy(base_model=my_custom_llm, ...)
```

**Option 2: Subclass LanguageModel**
```python
from lits.lm.base import LanguageModel, Output

class MyCustomLM(LanguageModel):
    def __call__(self, prompt, **kwargs):
        response = self.my_api.generate(prompt)
        return Output(text=response)

model = MyCustomLM()
policy = RAPPolicy(base_model=model, ...)
```

**Note:** The CLI (`main_search.py`) uses `get_lm()` which supports built-in providers (HuggingFace, Bedrock, OpenAI, TGI). For custom LLM logic not covered by these, write a Python script instead of using CLI.

**Future:** A registry pattern for custom LLM providers (similar to ComponentRegistry) may be added to enable CLI usage without modifying source code.

### Q: Why do I only see one checkpoint file per example even with n_iters=30?

This typically happens due to **mode collapse** in action generation. When the LLM generates the same actions repeatedly:

1. The tree degenerates to a single path (no branching diversity)
2. UCT selection always picks the same terminal node
3. When `terminate_on_terminal_node=false`, selecting a terminal node triggers `continue` which skips checkpoint saving
4. Only iteration 0 saves a checkpoint (during initial tree construction)

**Solutions:**
- Increase temperature in policy to encourage diverse action generation
- Check if your prompt encourages varied responses
- Verify the LLM isn't deterministically producing identical outputs

### Q: What's the difference between `terminate_on_terminal_node` and `terminate_on_first_solution`?

These control different aspects of MCTS termination:

| Parameter | Triggers On | Effect |
|-----------|-------------|--------|
| `terminate_on_terminal_node=true` | Selecting ANY terminal node (depth-limited OR goal-reached) | Breaks out of current iteration, moves to next iteration |
| `terminate_on_first_solution=true` | `node.is_terminal=True` from state itself (goal achieved) | Immediately stops entire MCTS search |

**Key distinction:**
- `terminate_on_terminal_node` checks if the **selected node** during UCT selection is terminal (either from reaching max_steps or from goal achievement)
- `terminate_on_first_solution` only triggers when a node's state indicates the **actual goal is achieved** (via `Transition.is_terminal()`)

**Example scenarios:**

```python
# Scenario 1: Feasibility checking (BlocksWorld)
# Stop as soon as any valid plan is found
config = BaseSearchConfig(
    terminate_on_first_solution=True,  # Stop when goal achieved
    terminate_on_terminal_node=True,   # Default
)

# Scenario 2: Finding best solution (Crosswords)
# Explore all iterations to find highest-reward solution
config = BaseSearchConfig(
    terminate_on_first_solution=False,  # Run all iterations
    terminate_on_terminal_node=False,   # Continue exploring even when selecting terminal nodes
)
```

### Q: Why does my MCTS stop after iteration 0 even with `terminate_on_first_solution=false`?

Check if `terminate_on_terminal_node=true` (the default). When this is enabled:
- If UCT selection picks a terminal node, the iteration breaks immediately
- If all paths lead to terminal nodes, every iteration breaks after selection
- This can make it appear that only iteration 0 ran

Set `terminate_on_terminal_node=false` to continue exploring other branches when a terminal node is selected.
