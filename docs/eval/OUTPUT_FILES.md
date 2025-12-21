# Output Files Reference

This document describes all files generated during tree search experiments using `main_search.py`.

## Directory Structure

```
results/{model_name}/{dataset}_{method}_{config}/run_{version}/
├── checkpoints/                    # Incremental MCTS rollout paths (v0.2.5+)
│   ├── {query_idx}_{iter}.json    # Path from iteration {iter}
│   └── {query_idx}_result.json    # Final selected path
├── terminal_nodes/                 # Terminal nodes for post-processing
│   └── terminal_nodes_{query_idx}.json
├── treetojsonl.jsonl              # Complete search trees
├── treetojsonl_unselected_simulate.jsonl  # Unselected paths (MCTS only)
├── {method}_config.json           # Search configuration
├── {run_id}.log                   # Execution logs
└── inferencelogger.log            # LLM inference metrics
```

## File Descriptions

### Configuration Files

#### `{method}_config.json`
**Purpose**: Complete search configuration for reproducibility

**Content**: All hyperparameters and settings used for the search run
- Algorithm parameters (n_iters, w_exp, max_steps, etc.)
- Model names and settings
- Termination conditions
- Continuation settings
- Package version

**Example**:
```json
{
    "reasoning_method": "rap",
    "package_version": "0.2.5",
    "policy_model_name": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "n_iters": 10,
    "max_steps": 6,
    "n_actions": 3,
    "w_exp": 1.0,
    "cum_reward": "np.mean",
    "calc_q": "max"
}
```

**Use Cases**:
- Reproduce experiments with identical settings
- Compare configurations across runs
- Document experimental setup in papers

---

### Search Results

#### `treetojsonl.jsonl`
**Purpose**: Complete search trees for all examples

**Format**: JSON Lines - one line per example, each containing all paths explored

**Content**: 
- For MCTS: `[trace_of_nodes] + trace_in_each_iter`
- For BFS: All paths from `buckets_to_paths(buckets_with_terminal)`

**Structure**:
```json
[
  [  // Path 1 (nodes from root to leaf)
    {"id": 0, "state": {...}, "action": "...", "reward": 0.5, ...},
    {"id": 1, "state": {...}, "action": "...", "reward": 0.3, ...}
  ],
  [  // Path 2
    {"id": 0, "state": {...}, "action": "...", "reward": 0.5, ...},
    {"id": 2, "state": {...}, "action": "...", "reward": 0.7, ...}
  ]
]
```

**Use Cases**:
- Visualize search tree structure
- Analyze exploration patterns
- Debug search behavior
- Generate figures for papers

**Related Tools**: `lits.visualize.visualize_tree()`

---

#### `treetojsonl_unselected_simulate.jsonl` (MCTS only)
**Purpose**: Paths that were generated during simulation but not selected

**Content**: Unselected terminal paths from `_simulate()` phase

**Use Cases**:
- Analyze alternative solutions
- Study exploration vs exploitation
- Understand why certain paths were rejected

---

### Terminal Nodes (v0.2.5+)

#### `terminal_nodes/terminal_nodes_{query_idx}.json`
**Purpose**: All terminal nodes for post-search evaluation

**Content**:
```json
{
  "terminal_nodes": [
    {"id": 5, "state": {...}, "action": "...", "reward": 0.8, ...},
    {"id": 12, "state": {...}, "action": "...", "reward": 0.6, ...}
  ],
  "query": "Original question text",
  "query_idx": 0
}
```

**Use Cases**:
- Answer extraction without reconstructing tree
- Voting across multiple terminal states
- Post-hoc evaluation with different metrics
- Debugging final states

**Related Script**: `eval_search.py` loads these for evaluation

---

### Checkpoints (v0.2.5+)

#### `checkpoints/{query_idx}_{iter}.json`
**Purpose**: Incremental rollout paths saved after each MCTS iteration

**Content**: Serialized path (list of nodes) from root to leaf for iteration `{iter}`

**Structure**:
```json
[
  {"id": 0, "state": {...}, "action": "...", "cum_rewards": [0.5, 0.6], ...},
  {"id": 1, "state": {...}, "action": "...", "cum_rewards": [0.3], ...},
  {"id": 3, "state": {...}, "action": "...", "cum_rewards": [], ...}
]
```

**Use Cases**:
- Monitor search progress in real-time
- Resume interrupted runs
- Debug specific iterations
- Analyze convergence behavior

**Parameters**:
- `checkpoint_dir`: Directory to save checkpoints
- `override_checkpoint`: Whether to overwrite existing files (default: True)

---

#### `checkpoints/{query_idx}_result.json`
**Purpose**: Final selected path based on `output_strategy`

**Content**: The best path selected by the search algorithm

**Use Cases**:
- Quick access to final answer
- Compare final vs intermediate paths
- Validate output strategy

---

### Logs

#### `{run_id}.log`
**Purpose**: Detailed execution logs

**Content**:
- Search algorithm steps (select, expand, simulate, backpropagate)
- Node creation and evaluation
- Termination conditions
- Error messages and warnings
- Timing information

**Log Levels**:
- `DEBUG`: Detailed step-by-step execution
- `INFO`: High-level progress
- `WARNING`: Potential issues
- `ERROR`: Failures

**Use Cases**:
- Debug search behavior
- Understand why certain decisions were made
- Diagnose errors
- Performance profiling

---

#### `inferencelogger.log`
**Purpose**: LLM inference metrics and token usage

**Content**:
- Model calls (policy, reward, transition)
- Token counts (input/output)
- Latency per call
- Cost estimates
- Cumulative statistics

**Example**:
```
[2024-12-19 10:30:15] Policy call - Input: 150 tokens, Output: 50 tokens, Latency: 1.2s
[2024-12-19 10:30:17] Reward call - Input: 200 tokens, Output: 10 tokens, Latency: 0.5s
Total tokens: 410, Total cost: $0.0082
```

**Use Cases**:
- Track inference costs
- Optimize prompt lengths
- Compare model efficiency
- Budget planning

---

## Workflow

### 1. Run Search
```bash
python main_search.py
```

**Generates**:
- `{method}_config.json` - Configuration
- `treetojsonl.jsonl` - Search trees
- `terminal_nodes/*.json` - Terminal nodes
- `checkpoints/*.json` - Incremental paths (if enabled)
- `*.log` - Execution logs

### 2. Monitor Progress
```bash
# Watch checkpoints in real-time
watch -n 5 'ls -lh results/run_0.2.5/checkpoints/'

# Check latest iteration
cat results/run_0.2.5/checkpoints/0_9.json | jq '.[].action'

# Monitor inference costs
tail -f results/run_0.2.5/inferencelogger.log
```

### 3. Evaluate Results
```bash
python eval_search.py \
  --result_dir results/run_0.2.5 \
  --benchmark_name gsm8k \
  --policy_model_name gpt-4
```

**Uses**:
- `terminal_nodes/*.json` - For answer extraction
- `{method}_config.json` - For configuration context

---

## File Size Considerations

**Large Files** (may need compression):
- `treetojsonl.jsonl` - Grows with tree size and dataset size
- `{run_id}.log` - Can be very large with DEBUG logging

**Small Files**:
- `{method}_config.json` - Few KB
- `terminal_nodes/*.json` - KB to MB per example
- `checkpoints/*.json` - KB to MB per iteration

**Tips**:
- Use `gzip` for long-term storage of large JSONL files
- Set log level to INFO for production runs
- Use `override_checkpoint=False` to avoid re-saving existing checkpoints

---

## Version History

### v0.2.5+
- Added `checkpoints/` directory for incremental MCTS paths
- Added `terminal_nodes/` directory for post-processing
- Added `override_checkpoint` parameter

### v0.2.0+
- Unified `treetojsonl.jsonl` format for MCTS and BFS
- Added `inferencelogger.log` for token tracking

---

## Related Documentation

- [Result Savers](./RESULT_SAVERS.md) - File writing utilities
- [Serialization](./SERIALIZATION.md) - Node serialization format
- [Incremental Evaluation](./INCREMENTAL_EVALUATION.md) - Checkpoint-based evaluation
- [Tree Search Guide](../agents/TREE_SEARCH_GUIDE.md) - Using tree search algorithms
