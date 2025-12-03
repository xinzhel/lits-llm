# Result Savers

Result savers provide immediate write-through persistence for evaluation results, search trees, and other experimental outputs. All result savers inherit from `BaseResults` and support immediate file writing (即时保存) to prevent data loss.

## Core Features

### Immediate Write-Through (即时保存)
All result savers write data to disk immediately on each `append_result()` call, ensuring:
- **No data loss**: Results are persisted even if the process crashes
- **Memory efficiency**: No need to accumulate all results in memory
- **Progress tracking**: Partial results available during long-running experiments
- **Resumability**: Can inspect intermediate results without waiting for completion

### Unified Interface
All result savers share a common interface:
```python
class BaseResults(ABC):
    def __init__(self, run_id: str, root_dir: str, override: bool, ext: str)
    def load_results(self, filepath: str) -> Any
    def append_result(self, result: Any) -> None
```

## Available Result Savers
**Use appropriate saver for data type**:
   - Simple predictions → `ResultToTxtLine`
   - Structured metrics → `ResultDictToCSV`
   - Complex nested data → `ResultDictToJsonl`
   - Search trees → `TreeToJsonl`

### ResultDictToCSV
**Purpose**: Save evaluation results as CSV with immediate write-through

**Use Case**: Evaluation metrics, experiment results, performance benchmarks

**Key Features**:
- Immediate CSV write on each `append_result()` call
- Auto-generates header from first result
- Column exclusion for verbose fields (e.g., trajectories)
- Compatible with pandas for analysis

**Example**:
```python
from lits.eval import ResultDictToCSV

# Initialize with immediate write-through
saver = ResultDictToCSV(
    run_id='eval',
    root_dir='results/model_name',
    override=False,
    exclude_columns=['trajectory', 'verbose_output']
)

# Each append immediately writes to CSV file
for idx, example in enumerate(dataset):
    result = evaluate(example)
    saver.append_result(result)  # Immediately saved to disk
    # If process crashes here, previous results are already saved
```

**Parameters**:
- `run_id`: Identifier for the result file
- `root_dir`: Directory to save CSV file
- `override`: If True, overwrite existing file; if False, append to it
- `exclude_columns`: List of column names to exclude from CSV

**File Format**: `{root_dir}/resultdicttocsv_{run_id}.csv`

### ResultDictToJsonl
**Purpose**: Save structured results as JSON Lines with immediate write-through

**Use Case**: Complex nested results, results with variable structure

**Key Features**:
- Each line is a complete JSON object
- Supports nested dictionaries and lists
- Can parse reasoning from `<think>` tags
- Immediate write on each append

**Example**:
```python
from lits.eval import ResultDictToJsonl

saver = ResultDictToJsonl(
    run_id='predictions',
    root_dir='results',
    override=True
)

# Append structured results
saver.append_result({
    'idx': 0,
    'prediction': 'yes',
    'reasoning': 'Based on the evidence...',
    'metadata': {'model': 'gpt-4', 'temperature': 0.7}
})
```

**File Format**: `{root_dir}/resultdicttojsonl_{run_id}.jsonl`

### TreeToJsonl
**Purpose**: Save search trees (MCTS/BFS) as JSON Lines with immediate write-through

**Use Case**: Tree search results, reasoning traces, path visualization

**Key Features**:
- Serializes complete search trees with node relationships
- Preserves parent-child structure
- Supports both SearchNode and MCTSNode
- Each line represents one task's complete tree

**Example**:
```python
from lits.eval import TreeToJsonl

saver = TreeToJsonl(
    run_id='',
    root_dir='results/bfs_gsm8k',
    override=False
)

# Save paths from tree search
paths = [trace_of_nodes] + trace_in_each_iter
saver.append_result(paths)  # Immediately saved
```

**File Format**: `{root_dir}/treetojsonl_{run_id}.jsonl`

### ResultToTxtLine
**Purpose**: Save simple text predictions with immediate write-through

**Use Case**: Classification labels, simple predictions, answer strings

**Key Features**:
- One prediction per line
- Minimal overhead
- Easy to parse

**Example**:
```python
from lits.eval import ResultToTxtLine

saver = ResultToTxtLine(
    run_id='predictions',
    root_dir='results',
    override=True
)

# Append text predictions
for prediction in predictions:
    saver.append_result(prediction)  # Immediately written
```

**File Format**: `{root_dir}/resulttotxtline_{run_id}.txt`


## Related Documentation

- [General Evaluator](./GENERAL_EVALUATOR.md) - Multi-perspective evaluation framework
- [Tree Search Guide](../agents/TREE_SEARCH_GUIDE.md) - Using TreeToJsonl for search results
- [Evaluation Metrics](../metrics/INFERENCE_METRICS.md) - Computing cost metrics
