# `_continuation()`

Greedy chain-forward expansion in `lits/agents/tree/continuation.py`. Starting from a selected leaf, repeatedly expand one child, evaluate it against a quality gate, and advance — producing a linear trace appended to the search tree.

## Signature

```python
def _continuation(
    query_or_goals,
    query_idx,
    node: SearchNode,
    world_model: Transition,
    policy: Policy,
    reward_model: RewardModel,
    expand_func: callable,
    world_modeling_func: callable,
    bn_evaluator=None,
    depth_limit: int = 999999,
    threshold_alpha: float = None,    # fast_reward threshold
    threshold_conf: float = None,     # state confidence threshold
    threshold_gamma: float = None,    # BN evaluator threshold
    threshold_gamma1: float = None,   # BN pre-filter threshold
    n_actions_for_bne: int = None,    # actions for BN evaluation
    on_step: callable = None,
    transition_before_evaluate: bool = False,  # V(s') vs Q(s,a)
) -> list[SearchNode]
```

## What It Does

Each iteration of the continuation loop runs a subset of these steps depending on which quality gate is active. At most one gate should be active per run (except `threshold_gamma1` which is a sub-gate of BN Eval).

| Step | Call | Control | Output |
|------|------|---------|--------|
| Materialize node state | `world_modeling_func(node)` | `node.state is None` | sets `node.state`, checks `is_terminal` |
| Depth/terminal check | `_is_terminal_with_depth_limit()` | always | break if terminal or depth limit |
| **Gate 1: Fast Reward** | `expand_func(..., n_actions=1, assign_rewards=True)` | `threshold_alpha` | expand 1 child, break if `fast_reward < threshold_alpha` |
| **Gate 2a: BN Eval entropy/sc (with pre-filter)** | `expand_func(..., n_actions_for_bne, assign_rewards=True)` | `threshold_gamma` + `threshold_gamma1` | expand N children with scoring, filter by `fast_reward >= threshold_gamma1`, then `bn_evaluator.evaluate()` |
| **Gate 2b: BN Eval entropy/sc (no pre-filter)** | `expand_func(..., n_actions_for_bne, assign_rewards=False)` | `threshold_gamma` | expand N children without scoring, all pass to `bn_evaluator.evaluate()` |
| **Gate 2c: BN Eval direct** | `expand_func(..., n_actions=1, assign_rewards=False)` | `threshold_gamma` | expand 1 child, `bn_evaluator.evaluate()` directly |
| **Gate 3: State Confidence** | `world_modeling_func(child)` | `threshold_conf` | run transition on child, break if `state_conf < threshold_conf` |
| Advance frontier | `node = child` | always (if no gate broke) | append child to trace, call `on_step` |

## `expand_func` Calls

| Call site | `assign_rewards` | `world_model` | `transition_before_evaluate` | `n_actions` |
|-----------|:-:|:-:|:-:|:-:|
| Fast Reward (`threshold_alpha`) | `True` | passed | forwarded | 1 |
| BN Eval entropy/sc + `threshold_gamma1` | `True` | passed | forwarded | `n_actions_for_bne` |
| BN Eval entropy/sc, no `threshold_gamma1` | `False` | — | — | `n_actions_for_bne` |
| BN Eval direct | `False` | — | — | 1 |

## `transition_before_evaluate` Interaction

Five places where reward evaluation can happen in continuation, and how `transition_before_evaluate` affects each:

| Path | Mechanism | `transition_before_evaluate` effect |
|------|-----------|-------------------------------------|
| Fast Reward gate | `expand_func` with `assign_rewards=True` | `_expand` handles V(s') vs Q(s,a) internally |
| BN Eval + `threshold_gamma1` | `expand_func` with `assign_rewards=True` | same as above — no direct `reward_model.fast_reward()` bypass |
| BN Eval, no `threshold_gamma1` | `assign_rewards=False` | unaffected — no reward scoring |
| BN Eval direct | `assign_rewards=False` | unaffected |
| State Confidence | `world_modeling_func()` directly | naturally V(s') — always runs transition first |

## Config-to-Parameter Mapping

| Config field | Parameter |
|---|---|
| `reward_alpha` | `threshold_alpha` |
| `reward_beta` | `threshold_conf` |
| `reward_gamma` | `threshold_gamma` |
| `reward_gamma1` | `threshold_gamma1` |
| `n_actions_for_bne` | `n_actions_for_bne` |
| `max_steps` | `depth_limit` |
| `add_continuation` | caller checks before calling |
| `transition_before_evaluate` | `transition_before_evaluate` |

## Callers

| Caller | `expand_func` | `transition_before_evaluate` | Notes |
|--------|---------------|:-:|-------|
| `MCTSSearch.search()` | `_expand` | `config.transition_before_evaluate` | after `_select()`, before main `_expand()` |
| `BFSSearch.search()` | `_expand_with_existing` | not passed (defaults `False`) | BFS naturally does transition-before-evaluate; flag is MCTS-specific |
