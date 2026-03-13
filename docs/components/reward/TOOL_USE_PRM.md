# ToolUsePRM

Process Reward Model for tool-use task evaluation.

## Type Signature

```python
class ToolUsePRM(RewardModel[ToolUseState, ToolUseAction]):
    TASK_TYPE: str = "tool_use"

    def __init__(
        self,
        base_model,
        tools: List,
        task_prompt_spec: Optional[str] = None,
        max_rollout_steps: int = 0,
        save_rollouts_dir: Optional[str] = None,
        **kwargs
    ): ...
```

## Interface Methods

```python
@property
def requires_transition_before_evaluate(self) -> bool
```
Returns `True` when `max_rollout_steps == 0` (direct scoring mode). MCTS auto-infers
`transition_before_evaluate=True` from this property during `_setup()`, so users do not
need to set the flag manually.

```python
def fast_reward(
    self,
    state: ToolUseState,
    step_or_action: ToolUseStep,
    query_or_goals: str,
    query_idx: Optional[int] = None,
    from_phase: str = ""
) -> Tuple[float, dict]
```
Scores a proposed step. Returns `(score, details)` where score ∈ [0, 1].

```python
def reward(
    self,
    state: ToolUseState,
    action: ToolUseAction,
    fast_reward: float,
    confidence: float,
    **kwargs
) -> float
```
Combines `fast_reward` and transition confidence: `fast_reward^α × confidence^(1-α)`.

```python
def calculate_reward(self, fast_reward: float, r_conf: Optional[float] = None) -> float
```
Pure calculation helper for the reward formula.

## Evaluation Modes

### `max_rollout_steps=0` (default) — Direct LM Scoring

Single LLM call scores the partial trajectory (state + proposed step). No internal rollout.
In MCTS, the simulation loop drives rollout and calls this function at each step as a
heuristic. Corresponds to the LATS evaluation step (§4.2).

### `max_rollout_steps>0` — Self-Contained Rollout

The reward model itself completes the trajectory with real tool execution (up to N steps)
before scoring. Use when MCTS simulation is disabled or for standalone evaluation.

## Evaluation Timing: V(s') vs Q(s, a)

The timing of when `fast_reward` is called relative to the transition determines what the
reward model actually estimates.

### Q(s, a) — Score Before Transition

Default MCTS behavior (`transition_before_evaluate=False`). The reward model scores the
proposed action *before* the transition executes it:

- `_expand()` → policy generates step (think + action, `observation=None`) → `_assign_fast_reward()` → later `_world_modeling()`
- The LLM evaluates: "given this state and this proposed action, how promising is it?"
- `step.observation` is `None` at scoring time

This is a **Q(s, a)** estimate — the value of taking action *a* in state *s*, without
seeing the outcome.

### V(s') — Score After Transition

LATS-aligned behavior (`transition_before_evaluate=True`). The transition runs first to
obtain environmental feedback, then the reward model scores with the observation present:

- `_expand()` → policy generates step → `_world_modeling()` (transition fills observation) → `_assign_fast_reward()`
- The LLM evaluates: "given this trajectory including the tool's response, how good is this state?"
- `step.observation` is populated at scoring time

This is a **V(s')** estimate — the value of the resulting state *s'* after executing the
action.

### Parameter Semantics Under V(s') Mode

When `transition_before_evaluate=True`, `_world_modeling` calls `transition.step()` then
`_assign_fast_reward()` in sequence. Both receive the **same Python object** (`node.step`):

```
_world_modeling(child):
    step_or_action = node.step              # object reference
    transition.step(parent.state, step_or_action)  # mutates step.observation in-place
    _assign_fast_reward(child):
        step_or_action = node.step          # same object, now has observation
        reward_model.fast_reward(parent.state, step_or_action)
```

So `fast_reward` always receives:
- `state` = **parent state** (trajectory history before this step), not the post-transition state
- `step_or_action` = the current `ToolUseStep`, whose `observation` field is already populated by the transition

Inside `_fast_reward` (direct scoring mode), the scoring context is constructed as:
```python
scoring_state = deepcopy(parent_state) + deepcopy(step_or_action)
```
This gives the LLM the full trajectory: historical steps + current step with observation.
The two parameters are needed separately because the scoring prompt is built from the
concatenated trajectory, not from `node.state` directly.

### When to Use Which

| Setting | Estimates | Best for |
|---------|-----------|----------|
| `transition_before_evaluate=False` | Q(s, a) | Language-grounded tasks where transition = state concatenation, so Q(s,a) ≈ V(s') |
| `transition_before_evaluate=True` | V(s') | Tool-use tasks where observation carries critical information (SQL results, API responses) |

For tool-use tasks, V(s') is strictly more informative — the LLM can see whether the SQL
query returned useful data, whether the API call succeeded, etc. The LATS paper (§4.2)
uses V(s'): *"Our key distinction from ToT is that we obtain this value after obtaining
the environmental feedback."*

The tradeoff: V(s') requires running transition for *all* candidate children during
expansion (`n_actions` calls), not just the selected one. For tool-use tasks where
transition involves real tool execution, this increases cost proportionally.

## Backpropagation and Terminal Reward

Tool-use tasks have no objective terminal reward signal. Tools return observations (SQL
results, API responses, error messages), not reward scores. The `@register_evaluator`
mechanism requires ground truth and is post-hoc only (evaluation time, not search time) —
using it during search would be oracle cheating.

Therefore tool-use MCTS uses **path-aggregate backpropagation**: each node's `backprop_reward_func`
is computed from the per-step LM scores along its path. This differs from LATS on
env-grounded tasks (e.g., WebShop) where `env.step()` returns an objective reward that
can be broadcast to all ancestors.

Note: tool execution feedback (success/failure, error messages) is already present in
`step.observation`. The LM naturally incorporates this when scoring — a trajectory with
a SQL syntax error in the observation will receive a lower score than one with valid
results. No separate "tool feedback extraction" mechanism is needed.
