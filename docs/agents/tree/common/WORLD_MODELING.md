# `_world_modeling()`

Module-level function in `lits/agents/tree/common.py`. Materializes a node's state by running the transition model, then assigns reward and checks terminality. The single entry point for "execute an action and observe the result."

## Signature

```python
def _world_modeling(
    query_or_goals, query_idx, node, transition_model, reward_model,
    from_phase="expand",
)
```

## What It Does

| Step | Call | Control | Output |
|------|------|---------|--------|
| Idempotency guard | `if node.state is not None: return` | always | skips everything if state already set |
| State transition | `transition_model.step(parent.state, step_or_action)` | `node.state is None` | sets `node.state`, `node.state_conf` |
| Parent state integrity | `assert node_state_copy == node.parent.state` | always | ensures transition doesn't mutate parent |
| Fast reward (if unset) | `_assign_fast_reward(node)` | `node.fast_reward == -1` | sets `node.fast_reward`, `node.fast_reward_details` |
| Final reward | `reward_model.reward(parent.state, action, fast_reward, ...)` | `hasattr(node, "reward")` | sets `node.reward` (float) |
| Terminal check | `transition_model.is_terminal(node.state, ...)` | always after state set | sets `node.is_terminal` |

## Key Properties

- **Idempotent**: The `if node.state is not None: return` guard means calling `_world_modeling` multiple times on the same node is safe and free. This is critical because multiple code paths may call it on the same node (e.g., `_expand` with `transition_before_evaluate=True` materializes children, then `_simulate` or `search()` calls `_world_modeling` again on the selected child — no-op).

- **`_assign_fast_reward` subsumption**: If `fast_reward == -1` when `_world_modeling` runs, it calls `_assign_fast_reward` internally. This means after `_world_modeling` completes, `fast_reward` is always assigned. This is the V(s') path — the reward model sees the observation from `transition_model.step()`.

- **Parent state safety**: A `deepcopy` + assert ensures `transition_model.step()` doesn't accidentally mutate the parent's state. This is defensive against transition implementations that modify state in-place.

## Relationship with `_assign_fast_reward`

| Function | When used | Reward model sees | Semantics |
|----------|-----------|-------------------|-----------|
| `_assign_fast_reward(node)` | `_expand` with `transition_before_evaluate=False` | `parent.state + step` (observation=None) | Q(s,a) |
| `_world_modeling(node)` | `_expand` with `transition_before_evaluate=True`, `_simulate`, `search()` main loop, continuation State Confidence | `parent.state + step` (observation present) | V(s') |

`_world_modeling` is a superset: it runs transition, then calls `_assign_fast_reward` if needed, then computes final reward and terminal check. `_assign_fast_reward` is the "score only" subset.

## Callers

| Caller | Phase | Purpose |
|--------|-------|---------|
| `MCTSSearch.search()` — before `_expand` | `"expand"` | materialize selected leaf's state |
| `MCTSSearch.search()` — before `_simulate` | `"expand"` | ensure leaf has state before rollout |
| `_expand()` per child (when `transition_before_evaluate=True`) | from caller | V(s'): transition + score each child |
| `_simulate()` — after child selection | `"simulate"` | materialize selected child during rollout |
| `BFSSearch.search()` — per node | `"expand"` | materialize each node before expansion |
| `BFSSearch.search()` — per child after expand | `"expand"` | materialize all children (BFS explores all) |
| `_continuation()` — loop start | `"continuation"` | materialize current node if state is None |
| `_continuation()` — State Confidence gate | `"continuation"` | materialize child to get `state_conf` |
