# `_expand()`

Module-level function in `lits/agents/tree/mcts.py`. Expands a node by generating candidate child nodes, optionally scoring and simulating them.

## Signature

```python
def _expand(
    query_or_goals, query_idx, node, policy, n_actions, reward_model,
    world_model=None, assign_rewards=True,
    from_phase="expand", memory_context=None,
    transition_before_evaluate=False,
)
```

## Why Not Simply Create Children with Action + Reward + State?

A naive expand would generate children and eagerly compute everything (action, reward, state) for each. `_expand` deliberately defers or skips expensive operations to avoid unnecessary costs:

- **Reward scoring is optional** (`assign_rewards=False`): Continuation's BN Eval branch generates multiple children purely for bottleneck evaluation — the `bn_evaluator` handles scoring separately, so calling `reward_model.fast_reward()` on each child would be wasted LLM calls.

- **State simulation is deferred by default** (`transition_before_evaluate=False`): In standard MCTS, `_expand` generates `n_actions` children but only one gets selected (via UCT in the main loop, or `simulate_choice` in rollout). Running transition on all children would mean `n_actions` tool executions (SQL queries, web requests) when only 1 is needed. State is materialized later by `_world_modeling()` on the selected child only.

- **Action generation reuses existing children** (`_sample_actions_with_existing`): When `_expand` is called multiple times on the same node (e.g., simulate after expand), it keeps existing children and only generates the delta from the policy — avoiding redundant LLM calls.

- **`_world_modeling` is idempotent**: The `if node.state is not None: return` guard means downstream calls (in `_simulate`, `search()` main loop) never double-execute transitions on children that already have state.

The `assign_rewards` and `transition_before_evaluate` flags give callers fine-grained control over which costs to pay:

| Scenario | `assign_rewards` | `transition_before_evaluate` | Cost per child |
|---|:-:|:-:|---|
| BN Eval (no reward needed) | `False` | — | policy only |
| Standard MCTS Q(s,a) | `True` | `False` | policy + reward LLM call |
| LATS V(s') for tool-use | `True` | `True` | policy + transition + reward LLM call |

## What It Does

| Step | Call | Control | Output |
|------|------|---------|--------|
| Sample actions | `_sample_actions_with_existing()` | always | `list[Step]` — reuses existing children, generates remaining from policy |
| Create child nodes | `create_child_node()` per step | always | `MCTSNode` with `action`, `step`, `trajectory_key`, phase flag |
| Repeat detection | `step.get_action()` / `step.terminate` | always | sets `child.is_terminal_for_repeat` |
| Reward scoring | `_assign_fast_reward(child)` | `assign_rewards` and not `transition_before_evaluate` | Q(s,a): scores `parent.state + step` without observation |
| State simulation + reward | `_world_modeling(child)` | `assign_rewards` and `transition_before_evaluate` | V(s'): runs transition, then scores with observation |
| Backfill unscored children | same as reward/state rows above | `assign_rewards` and `child.fast_reward == -1` | scores children from prior `assign_rewards=False` calls |

## Callers

| Caller | `assign_rewards` | `world_model` | `transition_before_evaluate` | Phase |
|--------|:-:|:-:|:-:|:-:|
| `MCTSSearch.search()` main loop | `True` | `self.world_model` | from config | `"expand"` |
| `_simulate()` rollout loop | `True` | `world_model` | from config | `"simulate"` |
| continuation Fast Reward (`threshold_alpha`) | `True` | `world_model` | from config | `"continuation"` |
| continuation BN Eval entropy/sc (with `threshold_gamma1`) | `True` | `world_model` | from config | `"continuation"` |
| continuation BN Eval entropy/sc (no `threshold_gamma1`) | `False` | — | — | `"continuation"` |
| continuation BN Eval direct | `False` | — | — | `"continuation"` |
