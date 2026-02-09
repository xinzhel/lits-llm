# InferenceLogger

Tracks token usage and timing for LLM calls during tree search experiments.

## Role Format

```
{component}_{query_idx}_{phase}
```

Examples: `policy_0_expand`, `prm_env_1_simulate`, `dynamics_2_continuation`

- **component**: policy, prm, prm_env, dynamics, evaluator, bn_eval, bn_entropy
- **query_idx**: dataset example index
- **phase**: expand, simulate, continuation

## Basic Usage

```python
from lits.lm.base import InferenceLogger

# Create logger (writes to {root_dir}/inferencelogger.log)
logger = InferenceLogger(root_dir="results/")

# With run_id (writes to {root_dir}/inferencelogger_{run_id}.log)
logger = InferenceLogger(run_id="exp_001", root_dir="results/")
# -> results/inferencelogger_exp_001.log

# Log a call (typically done automatically by LanguageModel)
logger.update_usage(
    input_tokens=1500,
    output_tokens=200,
    batch=False,
    batch_size=1,
    role="policy_0_expand",
    running_time=1.2
)
```

## Contextual Metadata (`log_context`)

Attaches key-value fields to all LLM call records within a `with` block. Useful for tagging records with search-specific information (iteration index, trajectory key) without modifying individual call sites.

### On `InferenceLogger`

```python
with inference_logger.log_context(iteration=3, trajectory_key="0/1/2"):
    # All update_usage() calls inside this block get iteration=3 and trajectory_key="0/1/2"
    policy.get_actions(...)
```

### On `BaseTreeSearch`

Convenience wrapper that returns `contextlib.nullcontext()` when no inference logger is available:

```python
class MCTSSearch(BaseTreeSearch):
    def search(self, query, query_idx):
        for idx_iter in range(self.config.n_iters):
            path = _select(...)
            with self.log_context(iteration=idx_iter):
                _expand(...)
                _simulate(...)
```

### Nesting

Inner blocks add fields; the outer block's state is restored on exit:

```python
with self.log_context(iteration=3):
    with self.log_context(trajectory_key="0/1/2"):
        _expand(...)   # records get iteration=3 + trajectory_key="0/1/2"
    _back_propagate(...)  # records get iteration=3 only
```

### Sample log record with extra fields

```json
{"timestamp": "02-09 10:30:15", "role": "policy_5_expand", "input_tokens": 1200, "output_tokens": 45, "batch": false, "batch_size": 1, "num_flatten_calls": 0, "running_time": 0.8, "iteration": 3, "trajectory_key": "5_1738000000/0/1/2"}
```

## Query Methods

### Single-group aggregation (returns one dict)

```python
# All records
logger.get_metrics_by_role()

# Filter by exact role
logger.get_metrics_by_role(role="policy_0_expand")

# Filter by prefix (prm matches prm_env_*)
logger.get_metrics_by_prefix("prm")

# Filter by example ID
logger.get_metrics_by_example_id(example_id=0)

# Filter by subtext
logger.get_metrics_by_subtext("expand")
logger.get_metrics_by_subtexts(["expand", "policy"], occurrence="all")
```

### Multi-group aggregation (returns dict of dicts)

```python
# Group by component (policy, prm_env, dynamics)
logger.get_metrics_by_component()

# Group by phase (expand, simulate, continuation)
logger.get_metrics_by_phase()

# Group by instance (query_idx)
logger.get_metrics_by_instance()
```

### Print helpers

```python
logger.print_metrics_for_all_role_prefixes()
logger.print_metrics_for_mcts_phases()
```

## Report Generation

```python
from lits.eval.inference_report import generate_report

report = generate_report(log_dir="results/run_0.2.5")
print(report)
```

Output:
```
================================================================================
                        INFERENCE USAGE REPORT
================================================================================

SUMMARY:
  Total Calls:    2,007
  Input Tokens:   1.96M
  Output Tokens:  198543
  Est. Cost:      $8.86

BY COMPONENT:
┌──────────┬───────┬───────────┬────────────┬───────┬─────────┐
│   Name   │ Calls │ Input Tok │ Output Tok │  Cost │ % Total │
├──────────┼───────┼───────────┼────────────┼───────┼─────────┤
│ prm_env  │  837  │   1.24M   │   173395   │ $6.32 │  63.2%  │
│  policy  │  837  │   422934  │   10300    │ $1.42 │  21.6%  │
│ dynamics │  333  │   298661  │   14848    │ $1.12 │  15.2%  │
└──────────┴───────┴───────────┴────────────┴───────┴─────────┘
```

## Log File Format

JSONL file at `{root_dir}/inferencelogger.log`:

```json
{"timestamp": "12-19 15:56:24", "role": "policy_0_expand", "input_tokens": 437, "output_tokens": 15, "batch": false, "batch_size": 1, "num_flatten_calls": 0, "running_time": 1.2}
```
