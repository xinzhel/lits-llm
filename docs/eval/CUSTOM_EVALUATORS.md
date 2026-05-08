# Custom Evaluators

Register dataset-specific evaluation functions so `lits-eval` uses your comparison logic instead of the default numeric matcher.

## Why

The default `lits-eval` accuracy check uses `eval_output(type="number_exact")`, which works for math QA but not for benchmarks that need float tolerance, set comparison, or special value normalization (e.g., DBBench, KG F1).

## Usage

```python
from lits.benchmarks.registry import register_evaluator

@register_evaluator("my_dataset")
def evaluate_my_dataset(predicted, ground_truth) -> bool:
    """Return True if predicted matches ground_truth."""
    return predicted.strip().lower() == ground_truth.strip().lower()
```

`lits-eval` automatically picks up the registered evaluator when evaluating results for `my_dataset`. No CLI flags needed — just ensure the module is imported (via `--include`).

## How It Works

`lits-eval` uses a three-way dispatch for accuracy comparison:

1. `env_grounded` tasks — goal check (applied during answer extraction, then exact string match)
2. Registered evaluator — `@register_evaluator("dataset_name")` if available
3. Default — `eval_output(type="number_exact")` for math QA

## Example: DBBench

DBBench needs float tolerance (±0.01), set comparison for multi-value answers, and special value normalization (`None` → `"0"`, `NaN` → `"0"`):

```python
from lits.benchmarks.registry import register_evaluator

@register_evaluator("dbbench")
def evaluate_dbbench(predicted_answer, ground_truth) -> bool:
    # Float tolerance for numeric answers
    # Set comparison for multi-value results
    # Special value normalization (None, NaN, Inf → "0")
    ...
```

Run evaluation:

```bash
lits-eval --result_dir results/dbbench_rest/run_0 \
    --include lits_benchmark.dbbench
```

## API

```python
from lits.benchmarks.registry import register_evaluator, get_evaluator, has_evaluator

# Register
@register_evaluator("name")
def my_eval(predicted, ground_truth) -> bool: ...

# Query
has_evaluator("name")       # True / False
get_evaluator("name")       # callable or None
```

## Dataset Loader Convention

For `lits-eval` compatibility, dataset loaders should include:

- `"question"` key — used by `lits-search` CLI to extract the query
- `"answer"` key — used by `lits-eval` to look up ground truth

These can be aliases of domain-specific fields (e.g., `"question"` = `"description"`, `"answer"` = `"label"` for DBBench).

## Evaluator Signature Contract

```python
@register_evaluator("my_dataset")
def my_evaluator(predicted: str, ground_truth: Any) -> Union[bool, float]:
    ...
```

- `predicted` — always a `str` (the agent's committed answer from the checkpoint).
- `ground_truth` — **the raw value from `example["answer"]`** in your dataset loader. It is passed directly without any type conversion. If your dataset loader returns a dict (e.g., `{"names": [...], "ids": [...]}`), your evaluator receives that dict. If it returns a string, your evaluator receives a string.
- Return `bool` (correct/incorrect) or `float` (e.g., F1 score where 1.0 = exact match).

This means the evaluator and dataset loader must agree on the `ground_truth` type. For example:

| Dataset | `example["answer"]` type | Evaluator handles |
|---------|--------------------------|-------------------|
| KGQA | `dict` (`{"names": [...], "ids": [...]}`) | Extracts IDs, does set comparison after splitting predicted by comma |
| DBBench | `str` or `list[str]` | Float tolerance, set comparison |
| Math QA | `str` (numeric) | Exact string match |
