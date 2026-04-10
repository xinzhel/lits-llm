# MCTS Search Loop: Phase Analysis & Extension Guide

`mcts.py::MCTSSearch.search()` orchestrates the MCTS loop: Select → Continuation → Expand → Simulate → Backpropagate. This document maps every phase call, explains the lazy-evaluation safeguards, and describes how to extend MCTS by subclassing.

## Phase Call Map

Each MCTS iteration executes phases in this order. Indented items are conditional.

```
for each iteration:
    _select(root)                          # UCT walk to leaf
    ├─ if terminal → continue/break
    │
    ├─ if add_continuation:
    │     _continuation(leaf, expand_func=_expand, ...)
    │     ├─ if terminal → continue/break
    │
    ├─ if leaf.state is None:              # SAFEGUARD-1
    │     _world_modeling(leaf)
    │     ├─ if terminal → continue/break
    │
    ├─ _expand(leaf, n_actions, ...)       # generate children
    │
    ├─ if leaf.state is None:              # SAFEGUARD-2
    │     _world_modeling(leaf)
    │     ├─ if terminal → continue/break
    │
    ├─ _simulate(leaf, ...)                # rollout
    │     └─ per rollout step:
    │         _expand(node, n_actions=1)
    │         _world_modeling(selected_child)  # SAFEGUARD-3
    │
    └─ _back_propagate(path)
```

## Lazy Evaluation & World-Modeling Safeguards

The MCTS loop uses lazy evaluation: transition (`_world_modeling`) and reward (`_assign_fast_reward`) are deferred until actually needed. Multiple safeguard calls to `_world_modeling` appear throughout the loop. These are safe because `_world_modeling` is **idempotent** — the `if node.state is not None: return` guard makes repeated calls free.

### Why Lazy?

In standard MCTS, `_expand` generates N children but only 1 gets selected for simulate. Running transition on all N would mean N tool executions (SQL queries, API calls) when only 1 is needed. Lazy evaluation defers transition to the point where a child is actually selected.

### Safeguard Inventory

| ID | Location | Condition | Purpose | When it's a no-op |
|----|----------|-----------|---------|-------------------|
| SAFEGUARD-1 | `search()`, before `_expand` | `path[-1].state is None` | Selected leaf (from `_select`) may be an unvisited child with no state. Expand needs parent state to generate actions. | When leaf was already visited (state materialized in a prior iteration). |
| SAFEGUARD-2 | `search()`, before `_simulate` | `path[-1].state is None` | After expand, the expanded node itself may still lack state if `transition_before_evaluate=False` (expand only scored Q(s,a), didn't run transition). Simulate needs state. | When `transition_before_evaluate=True` (expand already ran transition on all children, including the node itself if it was a child). Actually, this guards the *parent* node, not children — it's for the case where expand was called on a node whose state was set by SAFEGUARD-1, so this is almost always a no-op. |
| SAFEGUARD-3 | `_simulate()`, per rollout step | always called | After expand picks 1 child in simulate, that child needs state for the next rollout step. | When `transition_before_evaluate=True` in expand (child already has state). |
| SAFEGUARD-4 | `_continuation()`, loop start | `node.state is None` | Continuation needs state to expand. | When the node was already materialized by a prior phase. |
| SAFEGUARD-5 | `_expand()`, Step 4 backfill | `child.fast_reward == -1` | Children from a prior `assign_rewards=False` call (e.g., continuation BN Eval) may need rewards later. | When all children already have rewards. |

### Impact on `_interleaved_expand`

`_interleaved_expand` runs transition on every child immediately (to get observation for sibling awareness). The analysis below assumes `_interleaved_expand` is used **everywhere** — both in the main expand phase and as the `expand_func` passed to `_continuation`. This is the case when using `SiblingAwareMCTSSearch`, which passes `self._do_expand` (= `_interleaved_expand`) to continuation.

**Pre-expand safeguards (become no-ops):**
- SAFEGUARD-1: `_select` can only land on nodes created by a prior expand. Since `_interleaved_expand` materializes every child's state, `_select` never reaches a node with `state=None`.
- SAFEGUARD-4 (continuation loop start) guards the node that continuation just advanced to (`node = child`). In standard MCTS, whether this child has state depends on which quality gate ran:
  - **State Confidence gate**: explicitly calls `world_modeling_func(child)` before advancing — child has state.
  - **Fast Reward gate** and **BN Eval gate**: do NOT call `_world_modeling` on the child before advancing. With standard `_expand` (lazy), the child has no state, so SAFEGUARD-4 is necessary at the next iteration.
  
  With `_interleaved_expand` as continuation's `expand_func`, all children get state during the expand loop regardless of which gate is active. SAFEGUARD-4 is always a no-op.
- These safeguards are still worth keeping as defensive guards — they protect against mixed-mode scenarios (e.g., simulate phase using standard `_expand` while expand phase uses interleaved) and cost nothing when the guard fires (idempotent).

**Post-expand safeguards (become no-ops):**
- SAFEGUARD-2 and SAFEGUARD-3: after interleaved expand, **all children already have state and rewards** (transition ran on each child during the loop). The idempotent guard fires, no extra cost.
- SAFEGUARD-5 (backfill): all children already have `fast_reward` assigned, so the backfill loop is a no-op.

### Coupling with Memory / Augmentor Callbacks

The `on_step_complete` callback (used by `FactMemoryAugmentor`, `SQLValidator`, `CriticAugmentor`) fires after each child is created in `_expand`. Its data quality depends on whether transition has run:

- `FactMemoryAugmentor.analyze()` calls `_extract_messages_from_step(traj_state[-1])`, which uses `step.verb_step()`. For `ToolUseStep`, `verb_step()` includes `step.observation`. Since `transition/tool_use.py` mutates `step.observation` in-place during `transition_model.step()`, the same `step` object referenced by `node.step` gets updated.

| Expand mode | `transition_before_evaluate` | `step.observation` at `on_step_complete` time | Memory completeness |
|---|:-:|---|---|
| Standard `_expand` | `False` | `None` (transition deferred) | Incomplete — action only, no observation |
| Standard `_expand` | `True` | Present (`_world_modeling` ran before callback) | Complete |
| `_interleaved_expand` | N/A (always transitions) | Present (transition runs per child in loop) | Complete |

This means `_interleaved_expand` guarantees complete memory recording regardless of config, while standard `_expand` only provides complete memory when `transition_before_evaluate=True`.


## Extending MCTS via Subclassing

`MCTSSearch` exposes phase methods that subclasses can override to customize behavior without modifying the core search loop.

### Override Pattern

Each MCTS phase is dispatched through a `self._do_*()` method:

| Phase | Method | Default implementation | Override for |
|-------|--------|----------------------|--------------|
| Expand | `self._do_expand(...)` | calls module-level `_expand()` | sibling-aware expand, progressive widening |
| Simulate | `self._do_simulate(...)` | calls module-level `_simulate()` | custom rollout policy, depth-limited rollout |
| Backpropagate | `self._do_backpropagate(...)` | calls `_back_propagate()` or `_back_propagate_decay()` based on config | custom value aggregation |

`_select` is not wrapped because selection strategy is controlled by config (`w_exp`, `cross_rollout_q_func`) rather than algorithmic variation. If needed, it can be wrapped similarly.

### Example: Sibling-Aware MCTS

```python
@register_search("mcts_sibling_aware", config_class=MCTSConfig)
class SiblingAwareMCTSSearch(MCTSSearch):
    """MCTS with interleaved sibling-aware expansion.
    
    Overrides _do_expand to use _interleaved_expand, which runs
    transition after each child so subsequent siblings see the
    full action+observation of prior siblings.
    """
    
    def _do_expand(self, query, query_idx, node, **kwargs):
        from .common import _interleaved_expand
        _interleaved_expand(
            MCTSNode, query, query_idx, node,
            self.policy, n_actions=self.policy.n_actions,
            world_model=self.world_model,
            reward_model=self.reward_model,
            **kwargs,
        )
```

### Continuation Compatibility

`_continuation()` accepts `expand_func` as a parameter. When wrapping expand in a method, pass `self._do_expand` as `expand_func` so continuation also uses the overridden expand:

```python
_continuation(
    ...,
    expand_func=self._do_expand,
    ...
)
```

## Reward Scoring Modes

Two modes control when reward is computed relative to transition:

| Mode | `transition_before_evaluate` | Reward sees | Semantics | Use case |
|------|:--:|---|---|---|
| Q(s,a) | `False` | `parent.state + step` (no observation) | action quality estimate | Fast scoring, math reasoning |
| V(s') | `True` | `parent.state + step + observation` | next-state value estimate | Tool-use tasks where observation matters |

With `_interleaved_expand`, transition always runs before the next sibling's policy call (to provide observation). This is inherently V(s')-like for sibling awareness, but the reward scoring mode for the children themselves still respects `transition_before_evaluate`.
