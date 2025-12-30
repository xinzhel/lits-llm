# LITS-LLM

A modular toolkit for LLM-based search, planning, and tool-use workflows.

## Overview

LITS-LLM is a production-ready wrapper around LITS (Language Inference via Tree Search) that implements modular LLM search algorithms including Tree-of-Thoughts and Reasoning via Planning (RAP).

**Key Features:**
- Modular components for planning, reasoning, and tool orchestration
- Seamless hand-off between reactive (LLM-as-a-function) and deliberative (tree search) modes
- Extensible interface for custom tools, memory backends, and evaluation loops
- Built-in telemetry hooks for observability and benchmarking
- Unified interface for multiple LLM providers (HuggingFace, AWS Bedrock, OpenAI, etc.)

## Installation

```bash
# Normal install
pip install .

# Editable mode (changes take effect instantly)
pip install -e .

# Editable mode with development extras
pip install -e .[dev]
```

**Install from Test PyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple lits-llm==0.2.2
```

## Quick Start

### Basic Usage

See `lits_llm/examples/math_qa/main_search.py` for a complete example:

```python
main(
    dataset_name="math500", 
    model_name=model_name,
    eval_model_name=eval_model_name, 
    reasoning_method="bfs", 
    add_continuation=False, 
    bn_method=None, 
    bn_model_name=bn_model_name, 
    eval_idx=eval_idx
)
```

**Parameters:**
- `dataset_name`: "math500", "gsm8k"
- `model_name`: Model for search
- `eval_model_name`: Model for evaluation
- `reasoning_method`: "bfs", "rap", "rest"
- `add_continuation`: Enable chaining in search
- `bn_method`: "direct", "entropy" (sc1), "sc" (sc2)
- `bn_model_name`: Model for continuation
- `eval_idx`: Example indices to evaluate (default: all 100)

### LLM Interface

Unified interface for loading different LLM types:

```python
from lits.lm import get_lm

# HuggingFace models
model = get_lm("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")

# AWS Bedrock models (requires AWS credentials)
model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

# OpenAI models
model = get_lm("openai/gpt-4")

# Use the model
output = model("Hello, how are you?")
print(output.text)
```

**Supported LLM Providers:**
- HuggingFace models (e.g., `Qwen/*`, `meta-llama/*`)
- AWS Bedrock (prefix: `bedrock/`)
- OpenAI (prefix: `openai/`)
- Azure OpenAI (prefix: `azure_openai/`)
- Moonshot (prefix: `moonshot/`)
- Groq (prefix: `groq/`)

### RewardModel Interface

The package provides a modular reward model interface for evaluating actions in tree search:

```python
from lits.components.base import RewardModel

class MyRewardModel(RewardModel):
    def __init__(self, base_model, task_prompt_spec, **kwargs):
        super().__init__(base_model, task_prompt_spec, **kwargs)
    
    def _fast_reward(self, example, example_idx, state, action, from_phase=""):
        # Evaluate action without execution
        return usefulness_score
    
    def reward(self, state, action, **kwargs):
        # Evaluate action after execution
        return reward_value, auxiliary_dict
```

**Key Methods:**
- `fast_reward()` - Evaluate action quality without execution (for pruning)
- `reward()` - Evaluate action after execution (for scoring)
- `task_prompt_spec` - Unified prompt template/dictionary for all reward models

For detailed documentation, see [RewardModel Interface Guide](docs/REWARD_MODEL_INTERFACE.md).

### Prompt Injection

LITS-LLM supports flexible prompt customization for all LLM-based components (Policy, RewardModel, Transition). You can inject custom prompts directly or use the registry system for shared prompts across components.

For detailed documentation, see [Prompt Injection Design](docs/PROMPT_INJECTION_DESIGN.md).

### Tree Visualization

LITS-LLM provides unified visualization tools for analyzing MCTS and BFS search trees:

```python
from lits.visualize import visualize_mcts_result, visualize_bfs_result, get_tree_from_result

# Visualize MCTS result
mcts_result = mcts(question, idx, config, world_model, policy, evaluator)
visualize_mcts_result(mcts_result, 'output/mcts_tree', format='pdf')

# Visualize BFS result
bfs_result = bfs_topk(question, idx, config, world_model, policy, 
                      evaluator, retrieve_answer, return_buckets=True)
visualize_bfs_result(bfs_result, 'output/bfs_tree', format='pdf')

# Unified interface for both
paths = get_tree_from_result(result, idx, full_dataset)
plot_save_tree(paths, 'output/tree', format='pdf')
```

**Requirements:**
```bash
pip install anytree
# For PDF/PNG rendering, also install system graphviz:
# macOS:   brew install graphviz
# Ubuntu:  sudo apt-get install graphviz
```

**Unified Result Structure:**
Both `MCTSResult` and `BFSResult` share common attributes:
- `trace_of_nodes` - Best path from root to terminal node
- `root` - Root node of the search tree

For examples, see `lits_llm/unit_test/test_visualization_demo.py`.

## Documentation

- [Prompt Injection Design](docs/PROMPT_INJECTION_DESIGN.md) - Guide on customizing prompts for components
- [RewardModel Interface Guide](docs/REWARD_MODEL_INTERFACE.md) - Comprehensive guide on implementing and using reward models
- [AWS Bedrock Setup & Inference Profiles](docs/AWS_BEDROCK_INFERENCE_PROFILES.md) - Detailed guide on using AWS Bedrock models, inference profiles, and AWS configuration

## Development

### Running Tests

```bash
# Run all tests
cd lits_llm/unit_test
python test_load_models.py

# Or with pytest
pytest test_load_models.py -v -s
```

<!-- ### Building & Distribution

**Build package:**
```bash
rm -rf dist lits_llm.egg-info
python -m build
```

**Upload to Test PyPI:**
```bash
pip install twine
twine upload --repository testpypi dist/*
```

**Upload to PyPI:**
```bash
twine upload dist/*
``` -->

## Project Structure

```
lits_llm/
├── lits/                   # Core package
│   ├── agents/            # Agent implementations (MCTS, BFS, chains)
│   ├── components/        # Modular components (policy, evaluator, world model)
│   ├── lm/               # LLM interface (HF, Bedrock, OpenAI)
│   ├── structures/       # State and step structures
│   └── benchmarks/       # Evaluation utilities
├── examples/             # Example scripts and configs
├── unit_test/           # Unit tests
├── docs/                # Documentation
└── README.md           # This file
```

## License

Apache License 2.0 (see `LICENSE` file for details)

## Roadmap

- [ ] CLI tools (`lits-llm init`, `lits-llm run`, `lits-llm eval`)
- [ ] Additional search algorithms
- [ ] Enhanced memory backends
- [ ] More evaluation benchmarks
- [ ] Web UI for visualization
