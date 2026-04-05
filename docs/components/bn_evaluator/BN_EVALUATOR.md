# BN Evaluator

Branching Necessity (BN) evaluator — a continuation quality gate for tree search.

During continuation (greedy chain-forward after MCTS/BFS selects a leaf), the BN
evaluator scores whether sampled candidate actions converge.  High score means the
policy agrees on a single next step; low score means genuine branching diversity
exists and continuation should stop.

## Architecture

```
                    BNEvaluatorBase (ABC)
                    ├── eval_method: str
                    ├── state_verbalizer: (query, state) → str
                    └── evaluate(query, state, actions) → (score, canonical_action)
                           │
            ┌──────────────┼──────────────┬──────────────┐
            │              │              │              │
       ExactMatchSC   LLMSemanticSC   EntropySC     DirectLLM
       (no LLM)       (paper BN-SC2)  (paper BN-SC1) (single-action)
       eval_method     eval_method     eval_method    eval_method
         = "sc"          = "sc"        = "entropy"    = "direct"
```

## Eval Methods

| `bn_method` (CLI) | Class | `eval_method` | LLM? | Paper Name | Description |
|---|---|---|---|---|---|
| `sc_exact` | `ExactMatchSC` | `sc` | No | — | Exact string majority vote |
| `sc` | `LLMSemanticSC` | `sc` | Yes | BN-SC2 | LLM pairwise semantic overlap |
| `entropy` | `EntropySC` | `entropy` | Yes | BN-SC1 | LLM clustering + Shannon entropy |
| `direct` | `DirectLLM` | `direct` | Yes | — | Single-action necessity score (1–4) |

`continuation.py` branches on `eval_method`:
- `"sc"` or `"entropy"` → expand `n_actions_for_bne` children, call `evaluate()` on the batch, commit to canonical action if `bn_score ≥ threshold_gamma`
- `"direct"` → expand 1 child, call `evaluate()` on it, gate on `threshold_gamma`

## Task-Type Compatibility

| Eval Method | Math QA | Tool-Use (KGQA, DBBench) | Env-Grounded |
|---|---|---|---|
| `sc_exact` | ✗ (actions are natural language) | ✓ (structured strings) | ✓ (identical action repr) |
| `sc` (LLM) | ✓ | ✓ (overkill if actions are exact matches) | ✗ (use `sc_exact`) |
| `entropy` | ✓ | ✓ (overkill if actions are exact matches) | ✗ |
| `direct` | ✓ | ✓ | ✓ |

## State Verbalizer

The base class accepts a `state_verbalizer: Callable[[str, State], str]` to decouple
task-specific prompt rendering from the evaluator logic.  LLM-based evaluators use it
to build the context string; `ExactMatchSC` ignores it (no prompt needed).

Built-in verbalizers:
- `verbalize_concat_state(query, state)` — for ReST/BFS math QA
- `verbalize_rap_state(query, state)` — for RAP sub-question decomposition
- `EnvState.render_history()` — for environment-grounded tasks
- `_default_state_verbalizer` — generic fallback (`query + state.render_history()`)

## CLI Usage

```bash
# ExactMatchSC — no LLM loaded for BN evaluation
--search-arg bn_method=sc_exact --search-arg reward_gamma=0.5

# LLM semantic SC (BN-SC2)
--search-arg bn_method=sc --search-arg reward_gamma=0.5

# Entropy clustering (BN-SC1)
--search-arg bn_method=entropy --search-arg reward_gamma=0.5

# Direct LLM scoring
--search-arg bn_method=direct --search-arg reward_gamma=0.5
```

Common companion args:
- `reward_gamma` — BN score threshold (stop continuation if `bn_score < gamma`)
- `n_actions_for_bne` — number of candidate actions to sample for sc/entropy methods
- `reward_gamma1` — optional reward pre-filter before BN evaluation
- `bn_model_name` — use a separate model for BN evaluation (default: reuse base model)

## ExactMatchSC

Pure string-match self-consistency.  No LLM call.  Uses `collections.Counter` to
find the majority action and returns `count / total` as the BN score.

```python
from lits.components.bn_evaluator import ExactMatchSC

evaluator = ExactMatchSC()
assert evaluator.eval_method == "sc"

# All identical → perfect consensus
evaluator.evaluate("q", None, ["a", "a", "a"])
# (1.0, 'a')

# 2-of-3 majority
evaluator.evaluate("q", None, ["a", "a", "b"])
# (0.667, 'a')

# All different → weakest consensus
evaluator.evaluate("q", None, ["a", "b", "c"])
# (0.333, 'a')

# Empty / whitespace-only actions are filtered out
evaluator.evaluate("q", None, ["a", "", "  ", "a"])
# (1.0, 'a')
```

Factory shortcut — `create_bn_evaluator` returns `ExactMatchSC()` for
`bn_method=sc_exact` and skips all LLM model loading.

## Action Type Handling

`continuation.py` passes `child_node.action` to `evaluate()`, which may be a plain
`str` (from `ThoughtStep.get_action()` in math QA) or a `ToolUseAction`/`StringAction`
object (from tool-use / env-grounded tasks).  All evaluators normalize actions to
`str` via `str(a)` before processing, so callers do not need to convert beforehand.
