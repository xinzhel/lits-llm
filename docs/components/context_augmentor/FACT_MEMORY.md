# FactMemoryAugmentor

## Overview

`FactMemoryAugmentor` extracts atomic facts from agent trajectories and stores them in a vector-indexed memory backend (`LiTSMemoryManager`). On subsequent trajectories for the same task, retrieved facts are injected into the policy's system prompt, giving the agent access to environmental knowledge discovered in prior attempts.

## Recording Modes

The `batch` kwarg controls when facts are extracted:

| Mode | `batch` | When called | What it sees | Use case |
|------|---------|-------------|--------------|----------|
| Incremental | `False` (default) | After each step | Last step only | Tree search (steps arrive one at a time) |
| Batch | `True` | After trajectory completes | All steps | Chain pass@N (full trajectory available) |

**Batch mode** produces better facts (LLM sees full trajectory context) and is cheaper (1 LLM call vs N). In `lits-chain` pass@N, `_analyze_trajectory()` always uses `batch=True`.

**Incremental mode** is used by `lits-search` (MCTS/BFS) where the augmentor is called after each tree node expansion.

## Memory Scope

Facts are **cross-trajectory, within-example**:

```
Example 0:
  attempt 0 → extract facts → store in memory
  attempt 1 → retrieve facts from attempt 0 → inject into prompt → extract → store
  attempt 2 → retrieve facts from attempts 0+1 → inject → extract → store
  ...
  _clear_memory() ← clears all facts

Example 1:
  attempt 0 → empty memory (fresh start)
  ...
```

Different examples do not share facts. Within the same example, facts accumulate across attempts.

## CLI Usage

```bash
# Fact extraction with local backend, Sonnet for extraction
lits-chain \
    --include lits_benchmark.dbbench \
    --dataset dbbench \
    --cfg n_attempts=5 --cfg temperature=0.9 \
    --memory-arg backend=local augmentors=fact skip_similarity_filtering=true batch=true \
    --memory-arg model=bedrock/us.anthropic.claude-sonnet-4-6 \
    --output-dir results/dbbench_fact/run_v1
```

Key `--memory-arg` options for fact extraction:

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `local` | Memory backend (`local` or `mem0`) |
| `augmentors` | `fact` | Augmentor type(s), comma-separated |
| `model` | Sonnet 4.6 | LLM for fact extraction (LocalMemoryBackend) |
| `skip_similarity_filtering` | `false` | If `true`, retrieve all facts (no cosine filtering) |
| `batch` | `false` | Passed through to `_analyze_trajectory` (chain always uses `True`) |

## What Gets Extracted

The LLM receives the full trajectory (in batch mode) and extracts atomic facts. Quality depends on the task type:

**Good facts** (DBBench — concrete environmental knowledge):
- "Table `Jiu-Jitsu Championships Results` has columns: Result, Opponent, Method, Event, Notes"
- "Column `Method` contains values like 'Decision', 'Points (11 x 0)', 'Submission (armbar)'"
- "Query `SELECT Notes FROM ... WHERE Method = 'Decision'` returned 'Women +60kg Bronze'"

**Poor facts** (Terminal-Bench — generic paraphrases):
- "The function is called `run_tasks`"
- "The task involves implementing a scheduler"

The hypothesis is that fact extraction helps most on tasks with concrete, reusable environmental knowledge (database schemas, API responses) and least on tasks where the knowledge is procedural (coding strategies, debugging approaches).

## Code References

- `fact_memory.py::FactMemoryAugmentor` — augmentor implementation
- `chain.py::_analyze_trajectory` — calls `analyze(batch=True)` after each attempt
- `chain.py::_clear_memory` — clears memory between examples
- `search.py::setup_memory_manager` — creates `LocalMemoryBackend`
- `search.py::create_augmentors` — assembles augmentor list from CLI kwargs
- `augmentor_setup.py::wire_retrieval_to_policy` — hooks `retrieve()` into policy prompt
