# LiTS — Language Inference via Tree Search

A modular Python framework for building LLM agents with tree search (MCTS, BFS) and chain reasoning (ReAct), supporting multi-provider LLMs and tool use.

## Why LiTS?

| Concern | Challenge | LiTS Solution |
|---------|-----------|---------------|
| **Reusability** | Reimplementing search algorithms for each new task | Task-agnostic data structures (`Action → Step → State → Node`) that hide search procedures from task-specific logic |
| **Extensibility** | Adding new tasks requires modifying many files | Modular components (`Policy`, `Transition`, `RewardModel`) + decorator-based registry — add a task by registering prompts and a transition |
| **Observability** | Tree search is expensive and hard to debug | Built-in `InferenceLogger` tracks token usage at component, instance, and search-phase levels; incremental checkpointing for fault tolerance |

## Installation

```bash
pip install -e .          # editable install
pip install -e .[dev]     # with dev extras
```

Requires Python >= 3.11.

## Quick Start — CLI

LiTS provides four CLI commands installed via `pip install`:

```bash
lits-search       # Run tree search experiments
lits-eval         # Evaluate tree search results
lits-chain        # Run chain agents (ReAct, EnvChain)
lits-eval-chain   # Evaluate chain results
```

All example CLI commands below assume you are in the `demos/` directory, which contains `lits_benchmark` (example benchmarks) and sample data files:

<!-- ```
demos/                   # Demo data and example benchmarks
├── lits_benchmark/     # Benchmark implementations (importable via --include)
├── blocksworld/        # BlocksWorld data files
├── crosswords/         # Crosswords data files
└── demo_results/       # Pre-run results for evaluation demos

lits_benchmark/          # Example benchmarks (in demos/)
├── formulations/       # Custom frameworks (RAP)
├── math_qa.py          # GSM8K, MATH500
├── blocksworld.py      # BlocksWorld
├── crosswords.py       # Crosswords
└── mapeval.py          # MapEval (SQL tool use)
``` -->

```bash
cd demos
```

### Run MCTS on MATH500

```bash
lits-search --include lits_benchmark.math_qa \
    --dataset math500 \
    --policy concat --transition concat --reward generative \
    --search-arg n_iters=50 n_actions=3 max_steps=10 \
    --var limit=5
```

### Swap to RAP (different components, same algorithm)

```bash
lits-search --include lits_benchmark.math_qa lits_benchmark.formulations.rap \
    --dataset math500 \
    --policy rap --transition rap --reward rap \
    --search-arg n_iters=10 n_confidence=3
```

### Swap to BFS (different algorithm, same components)

```bash
lits-search --include lits_benchmark.math_qa \
    --dataset math500 \
    --cfg search_algorithm=bfs \
    --policy concat --transition concat --reward generative \
    --search-arg roll_out_steps=2 n_actions=3 max_steps=10
```

### Environment-grounded task (BlocksWorld)

```bash
lits-search --include lits_benchmark.blocksworld \
    --dataset blocksworld \
    --transition blocksworld \
    --search-arg max_steps=6 n_iters=50
```

### Tool-use task (MapEval-SQL)

```bash
lits-search --include lits_benchmark.mapeval \
    --dataset mapeval-sql
```

No component flags needed — the framework auto-selects tool-use components.

### Evaluate results

```bash
lits-eval --result_dir <result_dir>
```

### Dry run (validate config without inference)

```bash
lits-search --include lits_benchmark.math_qa \
    --dataset math500 --dry-run
```


## Quick Start — Python API

Tree search algorithms are class-based, inheriting from `BaseTreeSearch`:

```python
from lits.agents.tree.mcts import MCTSSearch
from lits.agents.tree.base import BaseSearchConfig
from lits.lm import get_lm

# Load model
model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

# Configure search
config = BaseSearchConfig(
    max_steps=10,
    n_actions=3,
    n_iters=50,
)

# Create search instance with components
search = MCTSSearch(
    config=config,
    policy=policy,           # generates candidate actions
    world_model=transition,  # executes actions, produces new states
    reward_model=reward,     # evaluates action quality
)

# Run search
result = search.run(query="What is 25 * 17?", query_idx=0)

# Extract answers from terminal nodes
from lits.agents.tree.common import extract_answers_from_terminal_nodes
vote_answers, answer_rewards, best_node, trace = extract_answers_from_terminal_nodes(
    terminal_nodes_collected=result.terminal_nodes_collected,
    retrieve_answer=retrieve_answer_fn,
    question="What is 25 * 17?"
)
```

### ReAct Agent (tool use)

```python
from lits.agents import create_tool_use_agent

agent = create_tool_use_agent(tools=tool_list, max_iter=50)
state = agent.run(query="Find restaurants near Sydney Opera House")
```

### Supported LLM Providers

```python
from lits.lm import get_lm

model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")  # AWS Bedrock
model = get_lm("openai/gpt-4")                                         # OpenAI
model = get_lm("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")           # HuggingFace
model = get_lm("groq/llama-3.1-8b-instant")                            # Groq
model = get_lm("tgi:///meta-llama/Meta-Llama-3-8B")                    # TGI
```

## Architecture

Three core component abstractions compose into agents:

```
Policy          →  generates candidate actions from states
Transition      →  executes actions, produces new states
RewardModel     →  evaluates action quality (optional)
```

Search frameworks bundle these with an algorithm:

| Framework | Algorithm | Components |
|-----------|-----------|------------|
| ReST-MCTS* | MCTS | ConcatPolicy + ConcatTransition + GenerativePRM |
| RAP | MCTS | RAPPolicy + RAPTransition + RapPRM |
| ToT-BFS | BFS | ConcatPolicy + ConcatTransition + GenerativePRM |

### Extending with Custom Components

Register components via decorators — no core code changes needed:

```python
from lits.components.registry import register_transition, register_dataset

@register_transition("my_domain")
class MyTransition(Transition):
    def step(self, example, state, action, **kwargs):
        ...
    def is_terminal(self, state, example, **kwargs):
        ...

@register_dataset("my_dataset", task_type="env_grounded")
def load_my_dataset(**kwargs):
    ...
```

Then use via CLI:

```bash
lits-search --include my_package \
    --dataset my_dataset --transition my_domain
```

## Task Types

| Task Type | State Space | Examples |
|-----------|-------------|----------|
| `language_grounded` | Text context | Math reasoning (GSM8K, MATH500) |
| `env_grounded` | Symbolic/physical state | BlocksWorld, Crosswords |
| `tool_use` | Context + tool state | SQL queries, web search, APIs |

## Project Structure

```
lits/                    # Core framework
├── agents/             # MCTS, BFS, ReAct, EnvChain
├── components/         # Policy, Transition, RewardModel
├── lm/                 # Multi-provider LLM interface
├── structures/         # State, Action, Step, Node
├── cli/                # CLI entry points
├── eval/               # Evaluation utilities
└── tools/              # Tool implementations
```

## Documentation
<!-- ```
docs/                    # Documentation
├── agents/             # Agent guides
├── components/         # Component API reference
├── cli/                # CLI reference
└── LITS_DESIGN.md      # Architecture overview
``` -->
- [Architecture Overview](docs/LITS_DESIGN.md)
- [Tree Search Guide](docs/agents/TREE_SEARCH_GUIDE.md)
- [CLI Reference](docs/cli/search.md)
- [ReAct Agent](docs/agents/ReAct.md)
- [Component API](docs/components/)
- [Tree Visualization](docs/TREE_VISUALIZATION.md)

## License

Apache License 2.0
