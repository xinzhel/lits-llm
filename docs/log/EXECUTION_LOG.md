# Execution Log Guide

How to read and monitor `execution.log` during tree search and chain agent runs.

## Log Location

```
{result_dir}/execution.log
```

The first line always records the exact CLI command and working directory for reproducibility.

## Real-Time Monitoring

### MCTS tree structure (per-iteration snapshot)

```bash
# Live follow (use result_dir path)
tail -f {result_dir}/execution.log | grep -A 30 "\[MCTS\] Iteration"

# One-shot check
grep -A 30 "\[MCTS\] Iteration" {result_dir}/execution.log
```

Example output:

```
[MCTS] Iteration 0/10 (example=14) | terminals=3 | early_stop=first_solution
Root
├── get_relations("Kodak EasyShare M753") r=0.75 d=1 [cont]
│   └── get_neighbors("Kodak EasyShare M753") r=0.76 d=2 [cont]
│       └── get_relations("canon, inc.") r=0.66 d=3
│           ├── get_neighbors(...) r=0.70 d=4
│           └── get_neighbors(...) r=0.76 d=4 [sim]
│               └── intersection(...) r=0.81 d=5 [sim]
│                   └── (no action) r=0.76 d=6 ★
└── ...
```

<!-- TODO: add real example from KGQA run -->

### Node flags

| Flag | Meaning |
|------|---------|
| `[cont]` | Created during continuation phase (chain-forward) |
| `[sim]` | Selected/created during simulate phase |
| `★` | Terminal node (agent produced final answer) |
| (no flag) | Created during expand phase (default) |

A node can have multiple flags (e.g., `[cont] [sim]`) if it was created in continuation and later selected for simulation.

### Errors (exclude false positives from `error=None` in ToolUseStep)

```bash
grep -i "error\|exception" execution.log | grep -v "error=None"
```

### Progress tracking

```bash
# Which iteration is running
grep "\[MCTS\] Iteration\|\[MCTS\] End" execution.log

# Count completed examples
grep -c "\[MCTS\] End" execution.log

# Tree snapshot for latest iteration
grep -A 30 "\[MCTS\] Iteration" execution.log | tail -35
```

### Chain agent (ReAct)

```bash
# Monitor per-example progress
tail -f execution.log | grep "Processing example\|Resolved answer"
```

## Log Structure

<!-- TODO: document log entry format, levels, components -->

### Key log prefixes

| Prefix | Source | Level |
|--------|--------|-------|
| `[MCTS] Iteration` | `mcts.py` | INFO |
| `[MCTS] End` | `mcts.py` | INFO |
| `[BFS]` | `bfs.py` | INFO |
| `[Continuation]` | `continuation.py` | DEBUG |
| `[TRANSITION]` | `tool_use.py` | DEBUG |

## Related Files

- `inferencelogger.log` — per-LLM-call token usage with component/phase tags
- `config.json` — saved experiment configuration
- `terminal_nodes/` — serialized terminal nodes for evaluation
- `checkpoints/` — chain agent per-example checkpoints
