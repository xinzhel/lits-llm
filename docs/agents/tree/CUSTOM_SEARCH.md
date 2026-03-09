# Custom Search Algorithms

Register custom search algorithms via `@register_search` to reuse LiTS's task-agnostic data structures and shared subprocedures.

## Registration

The CLI invokes search functions with a fixed positional signature. Your function must accept these arguments in order:

```python
def my_search(
    query_or_goals,     # Question or goal list
    query_idx,          # Example index (int)
    config,             # Your config dataclass instance
    world_model,        # Transition instance
    policy,             # Policy instance
    reward_model,       # RewardModel instance (evaluator)
    bn_evaluator=None,  # Optional bottleneck evaluator
    **kwargs,           # init_state_kwargs, checkpoint_dir, memory_manager
):
```

Register with `@register_search`:

```python
from dataclasses import dataclass
from lits.agents import register_search
from lits.agents.tree import BaseSearchConfig

@dataclass
class MyConfig(BaseSearchConfig):
    beam_width: int = 5
    temperature: float = 1.0

@register_search("my_algorithm", config_class=MyConfig)
def my_search(query_or_goals, query_idx, config, world_model, policy, reward_model, bn_evaluator=None, **kwargs):
    ...
```

Config fields are automatically available via `--search-arg`:

```bash
lits-search --include my_package \
    --search-algorithm my_algorithm \
    --search-arg beam_width=10 \
    --search-arg temperature=0.5
```

## Shared Subprocedures

All functions operate on task-agnostic data structures (`Action → Step → State → Node`), so the same algorithm works across language, environment, and tool-use tasks.

```python
from lits.agents.tree import (
    # Node management
    create_child_node,          # Create child with proper trajectory_key
    SearchNode, MCTSNode,       # Node classes

    # Core operations
    _world_modeling,            # Run transition + assign reward + check terminal
    _assign_fast_reward,        # Score node via RewardModel (without transition)
    _sample_actions_with_existing,  # Generate actions via Policy, reuse existing children

    # Termination
    _is_terminal_with_depth_limit,
    extract_answers_from_terminal_nodes,
)
```

### `_world_modeling(query, query_idx, node, transition_model, reward_model, from_phase)`

Executes the full world model step on a node:
1. `transition_model.step()` → sets `node.state` and `node.step.observation`
2. `_assign_fast_reward()` → sets `node.fast_reward` (if not already assigned)
3. `reward_model.reward()` → sets `node.reward`
4. `transition_model.is_terminal()` → sets `node.is_terminal`

Idempotent: no-op if `node.state is not None`.

### `_assign_fast_reward(node, reward_model, query, query_idx, from_phase)`

Scores a node via `reward_model.fast_reward()` without running transition. Sets `node.fast_reward` and `node.fast_reward_details`. Asserts `node.fast_reward == -1` (prevents double-assignment).

### `create_child_node(node_class, parent, action, step, child_index)`

Creates a child node with proper parent linkage and `trajectory_key` for logging.

## Example: Greedy Best-First Search

```python
from dataclasses import dataclass
from lits.agents import register_search
from lits.agents.tree import (
    BaseSearchConfig, SearchNode,
    create_child_node, _world_modeling,
)

@dataclass
class GreedyConfig(BaseSearchConfig):
    n_actions: int = 3
    max_steps: int = 10

@register_search("greedy_best_first", config_class=GreedyConfig)
def greedy_best_first(query, query_idx, config, world_model, policy, reward_model, **kwargs):
    root = SearchNode(state=world_model.init_state(query), action=query, parent=None)
    frontier = [root]

    for _ in range(config.max_steps):
        if not frontier:
            break
        node = max(frontier, key=lambda n: n.fast_reward if n.fast_reward != -1 else float('-inf'))
        frontier.remove(node)

        if node.is_terminal:
            return node

        steps = policy.get_actions(node.state, query, n_actions=config.n_actions)
        for i, step in enumerate(steps):
            child = create_child_node(SearchNode, node, step.get_action(), step, i)
            _world_modeling(query, query_idx, child, world_model, reward_model, "expand")
            if not child.is_terminal:
                frontier.append(child)

    return max(frontier, key=lambda n: n.fast_reward, default=root)
```

## Lookup and Invocation

```python
from lits.agents import AgentRegistry

# List registered algorithms
AgentRegistry.list_algorithms()  # ['mcts', 'bfs', 'greedy_best_first']

# Look up by name
search_fn = AgentRegistry.get_search("greedy_best_first")
config_cls = AgentRegistry.get_config_class("greedy_best_first")
```
