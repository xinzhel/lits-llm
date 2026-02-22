# Tree Search CLI

Two-stage workflow for running tree search experiments:
1. `main_search.py` - Run tree search and save terminal nodes
2. `eval_search.py` - Evaluate results from checkpoint files

## main_search.py

Run tree search experiments with various reasoning methods (RAP, ReST, BFS).

### Basic Usage

```bash
python main_search.py --dataset <dataset> --search_framework <framework> [options]
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset name (e.g., math500, gsm8k, blocksworld) |
| `--search_framework` | Search framework: rest, rap, tot_bfs |
| `--policy-model` | Policy model name |
| `--eval-model` | Evaluation model (defaults to policy model) |
| `--include` | Python module(s)/package(s) for custom component registration |
| `--search-arg` | Search algorithm params (e.g., `n_iters=50 n_actions=3`) |
| `--component-arg` | Component params (e.g., `think_for_correctness=true`) |
| `--dataset-arg` | Dataset loader kwargs (e.g., `levels=[5]`) |
| `--var` | Script variables: `offset=N limit=M` |
| `--override` | Clean and overwrite existing results |
| `--dry-run` | Print first dataset element and exit |
| `--help-config` | Show all available parameters |

### Output Files

```
results/{run_id}/
├── paths_*.jsonl              # Search paths for visualization
├── terminal_nodes/*.json      # Terminal nodes for evaluation
├── config.json                # Saved configuration (includes import_modules, dataset_kwargs)
├── execution.log              # Execution logs
└── inference_usage.log        # Token usage metrics
```

### Examples

See `run_configs.sh` for complete examples. Key patterns:

```bash
# ReST-MCTS* on math500 (level 5 only)
python main_search.py \
    --dataset math500 \
    --search_framework rest \
    --policy-model "Qwen/Qwen3-32B-AWQ" \
    --dataset-arg levels=[5] \
    --search-arg roll_out_steps=2 n_iters=50 n_actions=3 max_steps=10 \
    --var offset=0 limit=50

# RAP on math500 with TGI (requires completion model for logits)
python main_search.py \
    --dataset math500 \
    --search_framework rap \
    --include lits_benchmark.formulations.rap \
    --policy-model "tgi:///meta-llama/Meta-Llama-3-8B" \
    --dataset-arg levels=[5] \
    --search-arg roll_out_steps=10000 n_iters=10 n_confidence=3

# env_grounded task (blocksworld)
python main_search.py \
    --dataset blocksworld \
    --include lits_benchmark.blocksworld \
    --search-arg max_steps=6 roll_out_steps=6 terminate_on_first_solution=true
```

## eval_search.py

Evaluate tree search results from checkpoint files.

### Basic Usage

```bash
python eval_search.py \
    --result_dir <path> \
    --dataset_name <dataset> \
    --eval_model_name <model>
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--result_dir` | Directory containing terminal_nodes/ |
| `--dataset_name` | Dataset name (must match main_search.py) |
| `--eval_model_name` | Model for answer extraction |
| `--offset` | Dataset offset (default: 0) |
| `--limit` | Dataset limit (default: all) |
| `--include` | Override auto-loaded import_modules |

### Auto-Loading

`eval_search.py` auto-loads from `config.json`:
- `import_modules` - Custom component modules
- `dataset_kwargs` - Dataset filtering (e.g., `levels=[5]`)

### Examples

```bash
# Evaluate math500 results (auto-loads dataset_kwargs from config)
python eval_search.py \
    --result_dir Meta-Llama-3-8B_results/math500_mcts/run_0.2.5 \
    --dataset_name math500 \
    --eval_model_name "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

# Evaluate blocksworld (auto-loads import_modules from config)
python eval_search.py \
    --result_dir claude35v1_results/blocksworld_rap/run_0.2.5 \
    --dataset_name blocksworld \
    --eval_model_name "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
```

## Supported Task Types

| Task Type | Datasets | Evaluation Method |
|-----------|----------|-------------------|
| `language_grounded` | gsm8k, math500, spart_yn | Answer extraction with voting |
| `env_grounded` | blocksworld, crosswords | Goal satisfaction checking |
| `tool_use` | mapeval, mapeval-sql, clue | Answer extraction from tool outputs |

## QA
### Question 1: How is a component argument  (e.g., `max_new_tokens=1024`) paased to `Policy.from_config` (e.g., ConcatPolicy)？

1. CLI: `--component-arg max_new_tokens=1024`
2. `cli/search.py` line 280: `config.component_args.update(parse_component_args(cli_args))`
3. `cli/search.py` line 340: `create_components(..., config=config)`
4. `factory.py` line 296: `component_args = config.get_component_args()`
5. `factory.py` line 85: `PolicyCls.from_config(..., component_args=component_args, ...)`
6. `concat.py` line 68: `max_new_tokens=component_args.get('max_new_tokens')`

## Limitations (TODO)

- **Tool-use dataset registration**: Currently tool-use datasets (mapeval, clue) are not registered via `@register_dataset`. Users cannot add custom tool-use benchmarks through the registry. Consider renaming to "benchmark" since it includes tool configuration, not just data.
- **Unified dataset loading**: `load_dataset()` from registry is only used for `language_grounded` and `env_grounded` tasks. Tool-use tasks still use `load_resource()` directly.
