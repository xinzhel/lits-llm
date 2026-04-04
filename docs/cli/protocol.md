# CLI–Registry Protocol

How `lits-chain` and `lits-search` consume registered datasets, resources, and evaluators.

## Dataset (`@register_dataset`)

Returns `List[Dict]`. Each dict must have `question` (str). Other keys are benchmark-specific and opaque to the CLI.

| Key | Type | Required | Used by | Notes |
|-----|------|----------|---------|-------|
| `question` | str | yes | chain, search | Complete query string ready for the LLM |
| `answer` | str \| list | yes (for eval) | `lits-eval` | Gold answer for evaluator |

Benchmark-specific keys (e.g., `entities`, `db_id`) are opaque to the CLI — they're only accessed by `prepare_example` or the evaluator.

## Resource (`@register_resource`)

Returns `Dict`. Consumed by `create_tool_use_agent` (chain) and `create_components` (search).

| Key | Type | Required | Consumed by | Notes |
|-----|------|----------|-------------|-------|
| `tools` | list[BaseTool] | yes | chain, search | Tool instances for the agent |
| `tool_context` | str | yes | chain, search | Prepended to system prompt via `{tool_context}` placeholder |
| `prepare_example` | Callable[[dict], None] | no | chain, search | Per-example tool state reset (see below) |

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
| kgqa | `question`, `answer`, `entities` | `prepare_example` | float (F1) |

---

## Advanced: Stateful Tools

Some benchmarks require tools whose internal state changes per-example. For instance, KGQA tools share a `KGState` that maps entity names to Freebase IDs — and each question has different entities. The CLI itself should not know about these internals.

The `prepare_example` callback solves this cleanly:

- The **dataset loader** produces a self-contained `question` string (the CLI never needs to reformat it).
- The **resource** returns a `prepare_example(example) -> None` callback that mutates shared tool state before each example.
- The **CLI** calls `prepare_example(example)` in the per-example loop, then uses `example["question"]` as the query — no conditional branching.

```
┌─────────────────────────────────────────────────────────┐
│  CLI per-example loop                                   │
│                                                         │
│  for example in dataset:                                │
│      if prepare_example:        # optional callback     │
│          prepare_example(example)   # side effect only  │
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

**Resource** (`kgqa.py::load_kgqa_resource`): returns a `prepare_example` that resets `KGState`.

```python
def prepare_example(example: dict) -> None:
    kg_state.entities = example["entities"]   # update entity→ID map
    kg_state.variables.clear()                # reset variable tracker
    # clear AgentBench API caches to avoid cross-example leakage
```

The CLI doesn't know about `KGState`, entities, or Freebase IDs. It just calls the callback and uses the question string.
