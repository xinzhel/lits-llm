# LiTS — Language Inference via Tree Search

A modular Python framework for LLM reasoning and planning with tree search (e.g., MCTS, BFS and other custom search algorithms) and chain reasoning (e.g.,ReAct).

## Why LiTS?

| Concern | Challenge | LiTS Solution |
|---------|-----------|---------------|
| **Reusability** | Reimplementing search algorithms for each new task | Task-agnostic data structures (`Action → Step → State → Node`) that hide search procedures from task-specific logic |
| **Extensibility** | Adding new tasks requires modifying many files | Modular components (`Policy`, `Transition`, `RewardModel`) + decorator-based registry — add a task by registering prompts and a transition |
| **Observability** | Tree search is expensive and hard to debug | Built-in `InferenceLogger` tracks token usage at component, instance, and search-phase levels; incremental checkpointing for fault tolerance |

## Installation

```bash
pip install lits-llm          # from PyPI
# or
pip install -e .              # editable install from source
```

Requires Python >= 3.11.

## Quick Start — 5-Minute Demo

Three steps: install, configure an LLM, run a search and see the artifacts.

### 1. Install

```bash
pip install lits-llm
```

### 2. Configure an LLM provider

LiTS needs an LLM for policy (action generation) and reward (scoring). Pick one provider:

**OpenAI** — set `OPENAI_API_KEY`:
```bash
export OPENAI_API_KEY="sk-..."
MODEL="openai/gpt-4o-mini"
```

**AWS Bedrock** — configure AWS credentials ([SSO](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html) or env vars):
```bash
aws sso login --profile default   # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
MODEL="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
```

**Groq** (free tier available) — set `GROQ_API_KEY`:
```bash
export GROQ_API_KEY="gsk_..."
MODEL="groq/llama-3.1-8b-instant"
```

**Local HuggingFace** — no API key needed (for testing only; prefer GPU or Apple Silicon for larger LLMs):
```bash
MODEL="Qwen/Qwen2.5-0.5B-Instruct"  # auto-downloads from HuggingFace
# pass device via --search-arg: device=cuda (default), device=mps (Apple Silicon), or device=cpu
```

**Any OpenAI-compatible API** (vLLM, Together AI, Fireworks, etc.):
```bash
export OPENAI_API_KEY="your-key"           # use "EMPTY" for local vLLM
export OPENAI_API_BASE="http://localhost:8000/v1"  # vLLM / provider's base URL
MODEL="openai/meta-llama/Llama-3-8B-Instruct"
```

### 3. Run MCTS on a math problem

Save this as `my_benchmark.py` (the `.py` extension matters) in your working directory:

```python
from lits.registry import register_dataset

@register_dataset("my_math", task_type="language_grounded")
def load_my_math(**kwargs):
    return [
        {
            "question": (
                "The proper divisors of 12 are 1, 2, 3, 4 and 6. "
                "A proper divisor of an integer $N$ is a positive divisor of $N$ "
                "that is less than $N$. What is the sum of the proper divisors "
                "of the sum of the proper divisors of 284?"
            ),
            "answer": "284",
        }
    ]
```

Then run MCTS:

```bash
lits-search --include my_benchmark \
    --dataset my_math \
    --policy-model "$MODEL" \
    --search-arg roll_out_steps=2 n_iters=50 force_terminating_on_depth_limit=false n_actions=3 max_steps=10 \
    -o demo_results --override
```

### 4. Evaluate

```bash
lits-eval --result_dir demo_results
```

### What you should see

```
demo_results/
├── checkpoints/           # Intermediate tree states per iteration
├── terminal_nodes/        # All terminal nodes found
├── config.json            # Full config (reproducible)
├── execution.log          # Execution log
├── inferencelogger.log    # Per-call token usage with component/phase tags
├── llm_calls.jsonl        # Raw LLM call records
└── eval.log               # Accuracy + inference usage report
```

### Validate config without LLM calls (no API key needed)

```bash
lits-search --include my_benchmark \
    --dataset my_math --dry-run
```

This prints the resolved components, dataset info, and first example — useful for checking your setup before a real run.

## CLI Commands

```bash
lits-search       # Run tree search (MCTS, BFS)
lits-eval         # Evaluate tree search results
lits-chain        # Run chain agents (ReAct, EnvChain)
lits-eval-chain   # Evaluate chain results
```

## 🎬 2.5-Minute Demo Video

See the full walkthrough — MCTS on math, Crosswords (environment-grounded), and evaluation:

👉 https://youtu.be/nRGX43YrR3I

The commands demonstrated in the video are listed below for direct copy-paste.

## More CLI Examples

All commands below assume `cd demos` and a configured `$MODEL` (see Quick Start above).

### MCTS on MATH500

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


## Quick Start — Python API

Tree search algorithms are class-based, inheriting from `BaseTreeSearch`:

```python
from lits.agents.tree.mcts import MCTSSearch, MCTSConfig
from lits.lm import get_lm

# Load model
model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

# Configure search
config = MCTSConfig(
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
