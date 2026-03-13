# Backpropagation

Module-level functions in `lits/agents/tree/mcts.py`. After each MCTS iteration completes a rollout (select → expand → simulate), backpropagation updates Q-values along the path from leaf to root.

Two modes are available: **cumulative** and **decay**. The choice is governed by `MCTSConfig.backprop_mode` (`"cumulative"` or `"decay"`).

## Cumulative Mode: `_back_propagate()`

```python
def _back_propagate(path: list[MCTSNode], backprop_reward_func) -> float
```

Standard MCTS backpropagation. Each rollout appends one value to every node's `cum_rewards` list.

### Algorithm

Traverses the path in reverse (leaf → root). At each node:
1. Collects `node.reward` into a running list of per-step rewards.
2. Applies `backprop_reward_func` (default: `np.mean`) to the rewards accumulated so far (from current node to leaf).
3. Appends the result to `node.cum_rewards`.
4. Increments `node.visit_count`.

### Two-level aggregation (`per_node` mode only)

In `per_node` mode there are two aggregation stages:

| Stage | Function | When | What it aggregates |
|-------|----------|------|--------------------|
| Within-rollout | `backprop_reward_func` (config field) | During backprop | Per-step rewards from node to leaf within a single rollout |
| Across-rollout | `cross_rollout_q_func` (node field) | During UCT select | All values in `cum_rewards` across multiple rollouts |

In `terminal` mode, `backprop_reward_func` is not used — every node receives the raw leaf reward. Only `cross_rollout_q_func` applies (across-rollout aggregation).

The Q-value used by UCT is:
```python
node.Q = node.cross_rollout_q_func(node.cum_rewards)  # e.g., max([0.15, 0.25, 0.10]) = 0.25
```

### Per-node backpropagation

Given a path of 4 nodes with per-step rewards `[r0, r1, r2, r3]` and `backprop_reward_func = np.mean`:

```
Node 3 (leaf):  cum_rewards.append(mean([r3]))
Node 2:         cum_rewards.append(mean([r2, r3]))
Node 1:         cum_rewards.append(mean([r1, r2, r3]))
Node 0 (root):  cum_rewards.append(mean([r0, r1, r2, r3]))
```

After 5 iterations through this node, `len(cum_rewards) == 5` and `visit_count == 5`.

Given a path with 11 nodes (root to leaf) and individual rewards:

Rewards (leaf → root): `[0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, -1]`

If `backprop_reward_func = np.mean`, the cumulative rewards appended to each node are:

```
Leaf (depth 10):  mean([0.25])                                                       = 0.25
Node (depth 9):   mean([0.25, 0.25])                                                 = 0.25
Node (depth 8):   mean([0.25, 0.25, 0.0])                                            = 0.167
Node (depth 7):   mean([0.25, 0.25, 0.0, 0.25])                                      = 0.188
Node (depth 6):   mean([0.25, 0.25, 0.0, 0.25, 0.0])                                 = 0.15
Node (depth 5):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25])                           = 0.167
Node (depth 4):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0])                      = 0.143
Node (depth 3):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25])                = 0.156
Node (depth 2):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0])           = 0.139
Node (depth 1):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25])     = 0.15
Root (depth 0):   mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, -1]) = 0.045
```

Each node's `cum_rewards` list grows with each backpropagation (one value per rollout).
UCT then computes Q-values as `cross_rollout_q_func(cum_rewards)`, which defaults to `np.mean`.

Note: When `backprop_reward_func = np.mean` and `cross_rollout_q_func = np.mean`, Q becomes the mean of means:
- First mean: aggregates rewards within a single rollout (backpropagation)
- Second mean: aggregates across multiple rollouts (UCT selection)

### When to use

Appropriate when rollouts are i.i.d. — no cross-trajectory memory, so all rollouts are equally informative. Used by standard MCTS, RAP, ReST-MCTS* baselines.

## Decay Mode: `_back_propagate_decay()`

```python
def _back_propagate_decay(path: list[MCTSNode], gamma: float) -> float
```

Exponential recency-weighted backpropagation from LLaMA-Berry / Empirical-MCTS (Zhang et al., NAACL 2025).

### Algorithm

Starting from the leaf, walks to root. At each node:
1. If first visit: `cum_rewards = [child_q]`
2. Otherwise: `cum_rewards[0] = (1 - gamma) * cum_rewards[0] + gamma * child_q`
3. Increments `visit_count`.

For the leaf node, `child_q = leaf.reward`. For all other nodes, `child_q` is the updated Q of the child just processed.

### Key differences from cumulative

| Property | Cumulative | Decay |
|----------|-----------|-------|
| `len(cum_rewards)` | Grows with each rollout | Always 1 |
| `visit_count` needed? | Redundant (`== len(cum_rewards)`) | Required (can't infer from list length) |
| Rollout weighting | Equal (or determined by `cross_rollout_q_func`) | Exponential recency: recent rollouts dominate |
| `cross_rollout_q_func` role | Aggregates across rollouts | Effectively bypassed (single value) |

### When to use

Appropriate when rollouts are **not** i.i.d. — specifically when cross-trajectory memory (LATS reflection, LiTS-Mem fact sharing, etc.) makes later iterations better-informed than earlier ones. The decay factor `gamma` controls how aggressively old rollouts are down-weighted.

Config: `MCTSConfig.backprop_mode = "decay"`, `MCTSConfig.decay_gamma = 0.5` (default).

## How Q-values feed into UCT selection

In `_uct_select`, the UCT score for each child is:

```
score = child.Q + w_exp * sqrt(log(parent.visit_count) / max(1, child.visit_count))
```

Where `child.Q` is:
- Cumulative mode: `cross_rollout_q_func(cum_rewards)` — e.g., `max` or `mean` over the list
- Decay mode: `cum_rewards[0]` — the single maintained value

The `visit_count` field is used for the exploration term in both modes, ensuring correct UCT behavior regardless of `len(cum_rewards)`.

## Configuration

Relevant `MCTSConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `backprop_mode` | `"cumulative"` | `"cumulative"` or `"decay"` |
| `backprop_broadcast_mode` | `"per_node"` | `"per_node"` or `"terminal"` |
| `backprop_reward_func` | `np.mean` | Within-rollout aggregation (`per_node` broadcast mode only) |
| `cross_rollout_q_func` | `max` | Across-rollout aggregation (both broadcast modes; bypassed in decay) |
| `decay_gamma` | `0.5` | Recency weight (decay mode only). Higher = more weight on new rollout |

## Two orthogonal dimensions

Backpropagation is configured along two independent axes:

| Dimension | Field | Options | Determines |
|-----------|-------|---------|------------|
| Cross-rollout aggregation | `backprop_mode` | `cumulative` / `decay` | How multiple rollouts' values are combined at each node |
| Reward broadcast | `backprop_broadcast_mode` | `per_node` / `terminal` | What reward value each node receives during a single rollout |

### Reward broadcast modes

`per_node` (default): each node computes its own value from its position to the leaf. Different-depth nodes get different values. Appropriate when rewards are subjective LM scores (tool-use, language-grounded tasks).

`terminal`: the leaf node's reward is appended identically to every ancestor's `cum_rewards`. Appropriate for env-grounded tasks where an objective terminal signal is available (e.g., environment reward, test-case pass rate). Across-rollout aggregation is handled by `cross_rollout_q_func` at UCT selection time, just like `per_node` mode.

To reproduce LATS backpropagation (Zhou et al., ICML 2024), set `backprop_broadcast_mode="terminal"` and `cross_rollout_q_func=np.mean`. The LATS formula `V(s) = (V(s)*(N-1) + r) / N` is a running mean, which is mathematically equivalent to `mean(cum_rewards)` when each rollout appends the raw terminal reward.

### Choosing the right combination

| Task type | Memory | `backprop_mode` | `backprop_broadcast_mode` |
|-----------|--------|-----------------|---------------------------|
| Tool-use / language-grounded | No | `cumulative` | `per_node` |
| Tool-use / language-grounded | Yes (LATS, LiTS-Mem) | `decay` | `per_node` |
| Env-grounded | No | `cumulative` | `terminal` |
| Env-grounded | Yes | `decay` | `terminal` |
