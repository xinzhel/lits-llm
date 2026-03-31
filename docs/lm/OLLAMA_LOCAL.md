# Local LLM via Ollama

Run open-source models locally through Ollama's OpenAI-compatible API.

## Prerequisites

```bash
brew install ollama
ollama serve          # start the server (runs on http://localhost:11434)
```

## Pull a Model

```bash
ollama pull qwen3:0.6b          # tiny, for testing (~522 MB)
ollama pull qwen3:32b           # strong dense model, Q4_K_M (~20 GB)
ollama pull qwen3:235b-a22b     # strongest MoE, Q3_K_L (~112 GB, needs 128 GB unified memory)
```

See all available models at [ollama.com/library](https://ollama.com/library).

## Usage

```python
from lits.lm import get_lm

model = get_lm("ollama/qwen3:0.6b")
output = model("What is 2+2?", temperature=0.7, max_new_tokens=512)
print(output.text)
```

The `ollama/` prefix routes through `OpenAIChatModel` with `base_url=http://localhost:11434/v1`. No new dependencies required — it reuses the `openai` package already in core.

## Override Defaults

```python
# Custom host (e.g., remote Ollama server)
model = get_lm("ollama/qwen3:32b", base_url="http://remote-host:11434/v1")

# System prompt
model = get_lm("ollama/qwen3:32b", sys_prompt="You are a math tutor.")
```

## Context Length

Ollama manages context length server-side. To change it, create a Modelfile:

```
FROM qwen3:32b
PARAMETER num_ctx 32768
```

Then:

```bash
ollama create qwen3-32k -f Modelfile
```

Use `ollama/qwen3-32k` in `get_lm()`.

## CLI Usage

```bash
lits-search --include lits_benchmark.math_qa \
    --dataset math500 \
    --policy-model "ollama/qwen3:32b" \
    --search-arg n_iters=10 n_actions=3
```

## Hardware Guidelines

| Model | Quantization | Memory | Recommended For |
|-------|-------------|--------|-----------------|
| qwen3:0.6b | Q4_K_M | ~0.5 GB | Testing, CI |
| qwen3:32b | Q4_K_M | ~20 GB | Production on 32+ GB machines |
| qwen3:235b-a22b | Q3_K_L | ~112 GB | 128 GB Apple Silicon (M4/M5 Max/Ultra) |

The 235B MoE model activates only 22B parameters per token, so inference speed is closer to a 22B dense model despite the large memory footprint.
