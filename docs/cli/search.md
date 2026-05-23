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
| `--memory-arg` | Memory backend params (e.g., `backend=local model=... embedding_model=...`) |

### Memory Parameters (`--memory-arg`)

Passing `--memory-arg` enables the memory/augmentor subsystem. Shared by `lits-search` and `lits-chain`.

| Key | Description | Default |
|-----|-------------|---------|
| `backend` | Backend type: `local` or `mem0` | `local` |
| `model` | LLM for fact extraction (LocalMemoryBackend) | `bedrock/us.anthropic.claude-sonnet-4-6` |
| `embedding_model` | Embedder name (LocalMemoryBackend) | `multi-qa-mpnet-base-cos-v1` |
| `dedup_threshold` | Cosine similarity threshold for dedup (0–1) | `0.85` |
| `update_length_ratio` | Replace existing fact if new is N× longer | `1.3` |
| `mem0_config_path` | Path to JSON config for Mem0MemoryBackend | — |
| `augmentors` | Comma-separated augmentor names to activate | `fact` |
| `skip_similarity_filtering` | If `true`, retrieve ALL prior facts without similarity search (chain pass@N) | `false` |

**Augmentors:**

- `fact` (default): `FactMemoryAugmentor` — cross-trajectory fact extraction and retrieval via `LiTSMemoryManager`. Per-step trigger.
- `reflection`: `ReflectionAugmentor` — LLM-generated reflection on failed trajectories. Per-trajectory trigger. Requires a `base_model`.

Note: `--memory-arg` currently configures three layers via a flat key-value dict (`memory_kwargs`):
1. Memory backend: `backend`, `model`, `embedding_model`, `dedup_threshold`, `update_length_ratio`, `mem0_config_path`
2. Augmentor selection: `augmentors`
3. Augmentor behavior: `skip_similarity_filtering`

A future refactor may split these into separate CLI flags (e.g., `--augmentor-arg`).

**Backend comparison:**

- `local` (default): In-process, no external services. Uses `lits.embedding` for embeddings and a lits LLM (`model`) for fact extraction. Reproducible, fast startup.
- `mem0`: Delegates to mem0 library. Requires `mem0_config_path` pointing to a JSON file with mem0 provider config (llm, embedder, vector_store sections).

**`--memory-arg` CLI examples:**

```bash
# Minimal: local backend with defaults (fact augmentor, similarity search)
lits-search --memory-arg backend=local

# Local backend with explicit model and embedder
lits-search --memory-arg backend=local \
    model=bedrock/us.anthropic.claude-sonnet-4-6 \
    embedding_model=bedrock-embed/cohere.embed-english-v3

# Local backend with custom dedup/update thresholds
lits-search --memory-arg backend=local \
    dedup_threshold=0.9 \
    update_length_ratio=1.5

# Mem0 backend
lits-search --memory-arg backend=mem0 \
    mem0_config_path=./mem0_config.json

# Fact + reflection augmentors (requires base_model for reflection)
lits-search --memory-arg backend=local \
    augmentors=fact,reflection

# Chain pass@N: skip similarity filtering (return all prior facts)
lits-chain --memory-arg backend=local \
    skip_similarity_filtering=true

# All arguments combined (chain pass@N with reflection)
lits-chain --memory-arg backend=local \
    model=bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0 \
    embedding_model=multi-qa-mpnet-base-cos-v1 \
    dedup_threshold=0.85 \
    update_length_ratio=1.3 \
    augmentors=fact,reflection \
    skip_similarity_filtering=true
```

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

# With memory (local backend, default embedder)
python main_search.py \
    --dataset math500 \
    --search_framework rest \
    --memory-arg backend=local model=bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0

# With memory (local backend, Bedrock Cohere embedder)
python main_search.py \
    --dataset blocksworld \
    --include lits_benchmark.blocksworld \
    --memory-arg backend=local \
        model=bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \
        embedding_model=bedrock-embed/cohere.embed-english-v3 \
        dedup_threshold=0.9

# With memory (mem0 backend)
python main_search.py \
    --dataset math500 \
    --search_framework rest \
    --memory-arg backend=mem0 mem0_config_path=./mem0_config.json
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

## Tool-Use Task Hooks: `prepare_tool_state` and `resolve_answer`

Tool-use benchmarks (KGQA, mapeval-sql, etc.) register two optional hooks via the dataset's `tool_use_spec` dict:

| Hook | When it runs | Purpose | SPARQL/IO side effects |
|------|--------------|---------|------------------------|
| `prepare_tool_state(example)` | Once per example, before search | Inject example-specific context into a shared tool state (e.g., resolve KGQA entity names → Freebase IDs and seed the `KGState.entities` map) | Yes (entity lookup queries) |
| `resolve_answer(raw_answer, state)` | After search, on each terminal node | Convert symbolic answer references like `#3` to concrete entity names by replaying the trajectory and executing the symbolic program | Yes (`final_execute` SPARQL per terminal) |

### Resume Behavior

`lits-search` skips an example whose `terminal_nodes/terminal_nodes_<idx>.json` already exists. In that case `resolve_answer` is also skipped — the saved file is already in its final form from the original run, so re-resolving would be wasted SPARQL traffic and would re-emit the same warnings for any unresolvable answers (see "Unresolvable answers" below).

`prepare_tool_state` runs unconditionally per example today, even on skip. This is harmless (its SPARQL calls are cached) but represents a small per-skipped-example cost during resume; tracked as future work.

### Unresolvable Answers ("variable #N not found")

Agents sometimes hallucinate variable references that were never created — e.g., the LLM writes `<answer>#3</answer>` after only producing `#0..#2` via `get_neighbors`. `resolve_answer` cannot resolve such answers and emits:

```
WARNING resolve_answer: variable #3 not found (only 3 variables)
```

The raw `#3` is left in the saved file unchanged, so downstream evaluators see the literal `#3` and score it as wrong. This is an agent-quality signal, not a framework bug; the warnings appear in the original run's log, not just on resume.

## QA

### Question 1: How is a component argument passed to `Policy.from_config`?

Example: `--component-arg max_new_tokens=1024`

```
CLI --component-arg max_new_tokens=1024
  → config.component_args.update(parse_component_args(cli_args))  # search.py:280
  → create_components(..., config=config)                          # search.py:340
  → component_args = config.get_component_args()                   # factory.py:296
  → PolicyCls.from_config(..., component_args=component_args)      # factory.py:85
  → max_new_tokens=component_args.get('max_new_tokens')            # concat.py:68
```

### Question 2: How is a component argument saved to config.json?

Example: `--component-arg max_new_tokens=1024`

```
CLI --component-arg max_new_tokens=1024
  → config.component_args.update(parse_component_args(cli_args))  # search.py:280
  → config.save_config(result_dir)                                 # search.py:305
  → config.to_dict() returns {"component_args": config.get_component_args(), ...}
  → json.dump(...)
```

`config.to_dict()` calls `get_component_args()` which merges defaults with CLI overrides, so config.json contains the complete component_args (including `max_new_tokens=1024`).

### Question 3: How do LLM generation parameters flow to model calls?

Generation parameters (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `max_length`) are handled at two levels:

**Model-level (context window):**
```
--search-arg max_length=32768
  → load_models(..., max_length=max_length)           # search.py:445
  → get_lm(..., max_length=max_length)                # loader.py:135
  → OpenAIChatModel/HfChatModel stores self.max_length
```

**Component-level (per-call generation params):**
```
--component-arg max_new_tokens=1024 temperature=0.8
  → Policy.__init__(..., max_new_tokens=1024, temperature=0.8)
  → Policy._call_model(prompt)                        # base.py:600
  → auto-injects: kwargs['max_new_tokens'] = self.max_new_tokens
  → base_model(prompt, max_new_tokens=1024, ...)      # openai_chat.py:90
```

Key points:
- `max_length`: Total context window (prompt + output), set at model instantiation
- `max_new_tokens`: Max tokens to generate per call, set on component (Policy/RewardModel)
- Component params override model defaults (e.g., Policy's `max_new_tokens=1024` overrides model's default `512`)
- Different components can have different generation params while sharing the same model

### Question 4: How does `--memory-arg` flow to the memory backend?

```
CLI --memory-arg backend=local model=bedrock/... augmentors=fact,reflection
  → cli_args.memory_args = ["backend=local", "model=bedrock/...", "augmentors=fact,reflection"]
  → memory_kwargs = parse_memory_args(cli_args)           # args.py → {"backend": "local", "model": "bedrock/...", "augmentors": "fact,reflection"}
  → setup_memory_manager(run_logger, memory_kwargs)       # Layer 1: backend + manager
    → _create_local_backend(memory_kwargs, run_logger)
      → llm = get_lm(memory_kwargs["model"])
      → embedder = get_embedder(memory_kwargs["embedding_model"])
      → LocalMemoryBackend(llm, embedder, dedup_threshold, update_length_ratio)
    → LiTSMemoryManager(backend=backend)
  → create_augmentors(manager, memory_kwargs, ...)        # Layer 2: augmentor assembly
    → FactMemoryAugmentor(memory_manager=manager)         # "fact"
    → ReflectionAugmentor(base_model=base_model)          # "reflection"
```

For `backend=mem0`:
```
  → _create_mem0_backend(config, memory_kwargs, run_logger)
    → loads JSON from memory_kwargs["mem0_config_path"]
    → Memory.from_config(mem0_config)
    → Mem0MemoryBackend(memory)
```

## Limitations (TODO)

- **Tool-use dataset registration**: Currently tool-use datasets (mapeval, clue) are not registered via `@register_dataset`. Users cannot add custom tool-use benchmarks through the registry. Consider renaming to "benchmark" since it includes tool configuration, not just data.
- **Unified dataset loading**: `load_dataset()` from registry is only used for `language_grounded` and `env_grounded` tasks. Tool-use tasks still use `load_resource()` directly.
