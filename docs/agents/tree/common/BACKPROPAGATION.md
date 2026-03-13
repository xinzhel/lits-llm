# Backpropagation

Module-level functions in `lits/agents/tree/mcts.py`. After each MCTS iteration completes a rollout (select → expand → simulate), backpropagation updates Q-values along the path from leaf to root.

Two modes are available: **cumulative** and **decay**. The choice is governed by `MCTSConfig.backprop_mode` (`"cumulative"` or `"decay"`).

## Cumulative Mode: `_back_propagate()`

```python
def _back_propagate(path: list[MCTSNode], cum_reward_func) -> float
```

Standard MCTS backpropagation. Each rollout appends one value to every node's `cum_rewards` list.

### Algorithm

Traverses the path in reverse (leaf → root). At each node:
1. Collects `node.reward` into a running list of per-step rewards.
2. Applies `cum_reward_func` (default: `np.mean`) to the rewards accumulated so far (from current node to leaf).
3. Appends the result to `node.cum_rewards`.
4. Increments `node.visit_count`.

### Two-level aggregation

There are two aggregation stages:

| Stage | Function | When | What it aggregates |
|-------|----------|------|--------------------|
| Within-rollout | `cum_reward_func` (config: `cum_reward`) | During backprop | Per-step rewards from node to leaf within a single rollout |
| Across-rollout | `calc_q` | During UCT select | All values in `cum_rewards` across multiple rollouts |

The Q-value used by UCT is:
```python
node.Q = node.calc_q(node.cum_rewards)  # e.g., max([0.15, 0.25, 0.10]) = 0.25
```

### Example

Given a path of 4 nodes with per-step rewards `[r0, r1, r2, r3]` and `cum_reward_func = np.mean`:

```
Node 3 (leaf):  cum_rewards.append(mean([r3]))
Node 2:         cum_rewards.append(mean([r2, r3]))
Node 1:         cum_rewards.append(mean([r1, r2, r3]))
Node 0 (root):  cum_rewards.append(mean([r0, r1, r2, r3]))
```

After 5 iterations through this node, `len(cum_rewards) == 5` and `visit_count == 5`.

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
| Rollout weighting | Equal (or determined by `calc_q`) | Exponential recency: recent rollouts dominate |
| `calc_q` role | Aggregates across rollouts | Effectively bypassed (single value) |

### When to use

Appropriate when rollouts are **not** i.i.d. — specifically when cross-trajectory memory (LATS reflection, LiTS-Mem fact sharing, etc.) makes later iterations better-informed than earlier ones. The decay factor `gamma` controls how aggressively old rollouts are down-weighted.

Config: `MCTSConfig.backprop_mode = "decay"`, `MCTSConfig.decay_gamma = 0.5` (default).

## How Q-values feed into UCT selection

In `_uct_select`, the UCT score for each child is:

```
score = child.Q + w_exp * sqrt(log(parent.visit_count) / max(1, child.visit_count))
```

Where `child.Q` is:
- Cumulative mode: `calc_q(cum_rewards)` — e.g., `max` or `mean` over the list
- Decay mode: `cum_rewards[0]` — the single maintained value

The `visit_count` field is used for the exploration term in both modes, ensuring correct UCT behavior regardless of `len(cum_rewards)`.

## Configuration

Relevant `MCTSConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `backprop_mode` | `"cumulative"` | `"cumulative"` or `"decay"` |
| `cum_reward` | `np.mean` | Within-rollout aggregation (cumulative mode only) |
| `calc_q` | `max` | Across-rollout aggregation (cumulative mode; bypassed in decay) |
| `decay_gamma` | `0.5` | Recency weight (decay mode only). Higher = more weight on new rollout |
