# LITS-LLM

Personal agent toolkit for modular LLM search, planning, and tool-use workflows.

## Overview
- Production-ready wrapper around LITS (Language Inference via Tree Search) for autonomous agent flows.
- Ships as a reusable Python package with batteries-included prompts, tools, and evaluation utilities.

## Key Features
- Modular components for planning, reasoning, and tool orchestration.
- Seamless hand-off between reactive (LLM-as-a-function) and deliberative (tree search) modes.
- Extensible interface for plugging in custom tools, memory backends, and evaluation loops.
- Built-in telemetry hooks for observability and benchmarking.

## Installation
### PyPI (recommended)
```bash
pip install lits-llm
```

### Local editable install
```bash
pip install -e .
```

## Quickstart
TBA
<!-- ```python
``` -->

## CLI Usage
<!-- - `lits-llm init` scaffolds a new project with opinionated defaults.
- `lits-llm run --config path/to/config.yaml` executes an agent locally.
- `lits-llm eval --dataset math500` benchmarks reasoning strategies. -->
TBA

## Roadmap
TBA

## License
- Apache License 2.0 (see `LICENSE` file for details).


## Distribution on Pypi 

Upload to Test PyPI
```bash
pip install twine
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            lits-llm==0.2.1
```

Upload to official PyPI
```bash
twine upload dist/*
```