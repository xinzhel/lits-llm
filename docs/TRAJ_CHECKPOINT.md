# Trajectory Checkpoints as Training Data

LiTS tree search saves incremental checkpoints that double as structured training data for process reward models (PRM), step-level RLHF, and rejection sampling.

## What Gets Saved

Every MCTS/BFS iteration writes a checkpoint JSON containing:

| Field | Description |
|-------|-------------|
| `state` | Full reasoning trace (list of steps up to this node) |
| `action` | The candidate action taken |
| `step` | Structured step: `think` → `action` → `observation` |
| `fast_reward` | Per-step process reward (correctness + usefulness scores) |
| `cum_rewards` | Cumulative reward trajectory along the path |
| `is_terminal` | Whether this path reached a final answer |

## Output Structure

```
my_results/
├── checkpoints/
│   ├── 0_0.json          # query 0, iteration/depth 0
│   ├── 0_1.json          # query 0, iteration/depth 1
│   └── ...               # one file per iteration (MCTS) or depth level (BFS)
├── terminal_nodes/
│   └── terminal_nodes_0.json   # all completed paths for query 0
├── config.json
└── inferencelogger.log
```

Checkpointing is provided by `BaseTreeSearch.save_checkpoint()`, so any search algorithm that inherits from it can use the same mechanism.

## Training Data Use Cases

### PRM Training
Each node provides a `(state, action, reward)` tuple at every reasoning step — not just final-answer correctness. Parse `checkpoints/*.json` to extract step-level labels.

### Step-Level RLHF / DPO
MCTS naturally produces paired trajectories: successful and failed branches from the same parent node, with reward signals at every step. These are ready-made preference pairs.

### Rejection Sampling / Best-of-N
`terminal_nodes/` contains all completed paths with cumulative rewards. Filter by reward threshold, extract reasoning traces.

## Scale Estimate

Running MCTS on 500 problems with `n_iters=50, n_actions=3` produces ~75,000 search nodes with step-level reward annotations — a PRM training dataset as a byproduct of search.

## Example

```bash
lits-search --include my_benchmark \
    --dataset my_math \
    --policy-model "openai/gpt-4o-mini" \
    --search-arg n_iters=50 n_actions=3 max_steps=10 \
    -o my_results
```

Checkpoints are saved automatically. No extra flags needed.
