# CLI–Registry Protocol

How `lits-chain` and `lits-search` consume registered datasets, resources, and evaluators.

## Dataset (`@register_dataset`)

Returns `List[Dict]`. Each dict must have `question` (str). Other keys are benchmark-specific and opaque to the CLI.

| Key | Type | Required | Used by | Notes |
|-----|------|----------|---------|-------|
| `question` | str | yes | chain, search | Complete query string ready for the LLM |
| `answer` | str \| list | yes (for eval) | `lits-eval` | Gold answer for evaluator |

Benchmark-specific keys (e.g., `entities`, `db_id`) are opaque to the CLI — they're only accessed by `prepare_tool_state` or the evaluator.

## Resource (`@register_resource`)

Returns `Dict`. Consumed by `create_tool_use_agent` (chain) and `create_components` (search).

| Key | Type | Required | Consumed by | Notes |
|-----|------|----------|-------------|-------|
| `tools` | list[BaseTool] | yes | chain, search | Tool instances for the agent |
| `tool_context` | str | yes | chain, search | Prepended to system prompt via `{tool_context}` placeholder |
| `prepare_tool_state` | Callable[[dict], None] | no | chain, search | Per-example tool state reset (see below) |
| `resolve_answer` | Callable[[str, State], str] | no | chain, search | Post-run answer resolution (see below) |

## Evaluator (`@register_evaluator`)

```python
def evaluate(predicted_answer, ground_truth) -> bool | float
```

- `bool`: binary correctness (backward-compatible)
- `float` (0.0–1.0): continuous score (e.g., set-based F1). `lits-eval` tracks both accuracy (`score == 1.0`) and mean score.

## Existing implementations

| Benchmark | Dataset keys | Resource extras | Evaluator type |
|-----------|-------------|-----------------|----------------|
| dbbench | `question`, `answer`, `db_id` | — | bool |
| mapeval-sql | `question`, `answer` | — | bool |
| kgqa | `question`, `answer`, `entities` | `prepare_tool_state` | float (F1) |

---

## Advanced: Stateful Tools

Some benchmarks require tools whose internal state changes per-example. For instance, KGQA tools share a `KGState` that maps entity names to Freebase IDs — and each question has different entities. The CLI itself should not know about these internals.

The `prepare_tool_state` callback solves this cleanly:

- The **dataset loader** produces a self-contained `question` string (the CLI never needs to reformat it).
- The **resource** returns a `prepare_tool_state(example) -> None` callback that mutates shared tool state before each example.
- The **CLI** calls `prepare_tool_state(example)` in the per-example loop, then uses `example["question"]` as the query — no conditional branching.

```
┌─────────────────────────────────────────────────────────┐
│  CLI per-example loop                                   │
│                                                         │
│  for example in dataset:                                │
│      if prepare_tool_state:        # optional callback     │
│          prepare_tool_state(example)   # side effect only  │
│      query = example["question"]    # always from here  │
│      agent.run(query, ...)                              │
└─────────────────────────────────────────────────────────┘
```

### Example: KGQA

The KGQA benchmark has 7 tools sharing a `KGState` with a mutable `entities` dict. Each question maps different entity names to Freebase IDs.

**Dataset loader** (`kgqa.py::load_kgqa`): formats the query at load time.

```python
formatted_question = f"Question: {entry['question']}\nEntities: [{', '.join(entity_names)}]"
examples.append({"question": formatted_question, "entities": entry["entities"], ...})
```

**Resource** (`kgqa.py::load_kgqa_resource`): returns a `prepare_tool_state` that resets `KGState`.

```python
def prepare_tool_state(example: dict) -> None:
    kg_state.entities = example["entities"]   # update entity→ID map
    kg_state.variables.clear()                # reset variable tracker
    # clear AgentBench API caches to avoid cross-example leakage
```

The CLI doesn't know about `KGState`, entities, or Freebase IDs. It just calls the callback and uses the question string.

### `resolve_answer` — post-run answer resolution

Some benchmarks produce symbolic answers that need resolution against an external service. For example, KGQA agents output variable references (`#3`) that must be executed as SPARQL queries to get actual entity names.

The `resolve_answer` callback handles this:

```python
def resolve_answer(raw_answer: str, state: TrajectoryState) -> str:
    """Resolve symbolic answer to concrete value. Return raw_answer if no resolution needed."""
```

- Called after `agent.run()` if the state has a final answer.
- `state` is always a deserialized `TrajectoryState` object (both chain and search CLIs ensure this).
- Replays the trajectory to reconstruct internal state, then resolves the answer.
- The resolved answer overwrites the checkpoint so `lits-eval` sees concrete values.
- If absent, the CLI uses the raw answer as-is.
