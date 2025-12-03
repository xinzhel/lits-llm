# Incremental Evaluation

Add new evaluation perspectives without re-running expensive LLM evaluations.

## Overview

When evaluating agent performance, you often want to add new evaluation criteria after initial evaluation. Incremental evaluation allows you to evaluate only the new perspectives, saving time and API costs.

**Example scenario:**
- Initial evaluation: correctness, completeness
- Later: add spatial_correctness perspective
- Without incremental evaluation: re-evaluate all 3 perspectives (wasteful)
- With incremental evaluation: evaluate only spatial_correctness (efficient)

## Quick Start

### Step 1: Initial Evaluation

```python
from lits.eval import GeneralEvaluator, ResultDictToCSV
from lits.lm import get_lm

base_model = get_lm("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
result_saver = ResultDictToCSV(run_id='eval', root_dir='results')

evaluator = GeneralEvaluator(
    base_model=base_model,
    eval_perspectives=[
        {'eval_id': 'correctness', 'description': 'Is the answer correct?', 'options': ['yes', 'no']},
        {'eval_id': 'completeness', 'description': 'Is the answer complete?', 'options': ['yes', 'no']}
    ]
)

for idx, example in enumerate(dataset):
    evaluator.evaluate(
        solution=example['solution'],
        question=example['question'],
        truth=example['answer'],
        result_saver=result_saver,
        identifier_value=idx
    )
```

### Step 2: Add New Perspective

```python
# Add new perspective to evaluator
evaluator = GeneralEvaluator(
    base_model=base_model,
    eval_perspectives=[
        {'eval_id': 'correctness', ...},
        {'eval_id': 'completeness', ...},
        {'eval_id': 'spatial_correctness', ...}  # NEW
    ]
)

# Evaluate only the new perspective
for idx, example in enumerate(dataset):
    evaluator.evaluate_incremental(
        solution=example['solution'],
        question=example['question'],
        truth=example['answer'],
        result_saver=result_saver,
        identifier_value=idx,
        eval_only_perspectives=['spatial_correctness']
    )
```

The CSV is automatically updated with the new column, preserving existing evaluations.

## Usage Patterns

### Pattern 1: Add Single Perspective

```python
# Initial evaluation
evaluator_v1 = GeneralEvaluator(base_model, perspectives=[p1, p2])
for idx, ex in enumerate(dataset):
    evaluator_v1.evaluate(..., result_saver=saver, identifier_value=idx)

# Add new perspective
evaluator_v2 = GeneralEvaluator(base_model, perspectives=[p1, p2, p3])
for idx, ex in enumerate(dataset):
    evaluator_v2.evaluate_incremental(
        ..., result_saver=saver, identifier_value=idx,
        eval_only_perspectives=['p3']
    )
```

### Pattern 2: Add Multiple Perspectives

```python
evaluator = GeneralEvaluator(base_model, perspectives=[p1, p2, p3, p4, p5])
for idx, ex in enumerate(dataset):
    evaluator.evaluate_incremental(
        ..., result_saver=saver, identifier_value=idx,
        eval_only_perspectives=['p3', 'p4', 'p5']
    )
```

### Pattern 3: Auto-detect Missing Perspectives

```python
# Automatically evaluate only missing perspectives
evaluator = GeneralEvaluator(base_model, perspectives=[p1, p2, p3, p4])
for idx, ex in enumerate(dataset):
    evaluator.evaluate_incremental(
        ..., result_saver=saver, identifier_value=idx,
        eval_only_perspectives=None  # Auto-detect
    )
```

### Pattern 4: Batch Update

```python
# Evaluate all examples first, then batch update CSV
new_results = []
for idx, ex in enumerate(dataset):
    result = evaluate_new_perspective(ex)
    new_results.append({
        'identifier_value': idx,
        'column_updates': result
    })

# Batch update (more efficient)
saver.update_columns_batch(row_identifier='idx', updates=new_results)
```

## API Reference

### evaluate_incremental()

```python
evaluator.evaluate_incremental(
    solution: str,
    question: str,
    truth: str,
    others: str,
    result_saver: ResultDictToCSV,
    identifier_value: Any,
    eval_only_perspectives: List[str] = None,  # None = auto-detect missing
    row_identifier: str = 'idx'
) -> Dict[str, str]
```

Evaluates only missing or specified perspectives. Returns empty dict if all perspectives already evaluated.

### update_column()

```python
result_saver.update_column(
    row_identifier: str,
    identifier_value: Any,
    column_updates: Dict[str, Any]
)
```

Updates specific columns for a single row. Uses pandas for efficient CSV manipulation.

### update_columns_batch()

```python
result_saver.update_columns_batch(
    row_identifier: str,
    updates: List[Dict[str, Any]]
)
```

Batch updates multiple rows. More efficient than calling `update_column()` multiple times.

## Related Documentation

- [Result Savers](./RESULT_SAVERS.md) - ResultDictToCSV documentation
- [General Evaluator](./GENERAL_EVALUATOR.md) - Full evaluator API
- [VERIS Example](../../examples/veris/eval.py) - Complete working example
