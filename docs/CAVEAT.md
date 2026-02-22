# Known Caveats

## Running lits-eval with Custom Benchmarks

When running `lits-eval` on results from custom benchmarks (e.g., `lits_benchmark.math_qa`), you need to ensure the benchmark module is importable. Set `PYTHONPATH` to include the `demos` directory:

```bash
# From workspace root
PYTHONPATH=lits_llm/demos lits-eval \
    --result_dir examples/results/model_results/dataset_mcts/run_name \
    --dataset_name math500 \
    --eval_model_name "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
```

The `import_modules` and `dataset_kwargs` are auto-loaded from `config.json` in the result directory, so you typically don't need to specify them manually.

**Output behavior:**
- By default, `lits-eval` shows a progress bar and concise summary
- Detailed logs are saved to `eval.log` in the result directory
- Use `-v` or `--verbose` for detailed console output

## max_new_tokens Must Be Set for HuggingFace Models

When using local HuggingFace models (not API-based like Bedrock/OpenAI), you **must** set `max_new_tokens` via `--component-arg` to prevent runaway generation.

Without this limit, models may generate tens of thousands of tokens in a single call, causing:
- Extremely long inference times (hours per call)
- GPU memory exhaustion
- Apparent "hangs" with no progress

**Why prompt instructions aren't enough:**

The ConcatPolicy prompt includes "output ONLY ONE step within 1000 tokens", but this is just a suggestion to the model. Models (especially smaller or quantized ones) may ignore this instruction entirely. The only reliable way to enforce token limits is via `max_new_tokens` at the code level.

**Example of the problem:**
```json
{"output_tokens": 32102, "running_time": 5554.31}  // 1.5 hours for one call!
```

**Solution:**
```bash
lits-search ... \
    --component-arg max_new_tokens=512
```

**Recommended values:**
- Policy model: `max_new_tokens=512` (reasoning steps are typically short)
- Reward model with thinking: `max_new_tokens=500`
- Reward model without thinking: `max_new_tokens=50`

This caveat does not apply to API-based models (Bedrock, OpenAI) which have built-in limits.
