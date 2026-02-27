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


## vLLM with AWQ Models: Use `awq_marlin` Quantization

When serving AWQ-quantized models (e.g., `Qwen/Qwen3-32B-AWQ`) with vLLM, use `--quantization awq_marlin` instead of `--quantization awq`.

**Performance difference:**
- `awq`: ~4 tokens/s (uses generic AWQ implementation)
- `awq_marlin`: ~38 tokens/s (uses optimized Marlin kernels)

vLLM will log a warning if you use `awq` when `awq_marlin` is available:
```
Detected that the model can run with awq_marlin, however you specified quantization=awq explicitly, so forcing awq. Use quantization=awq_marlin for faster inference
```

**Correct vLLM startup:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --quantization awq_marlin \
    --port 8000 \
    --max-model-len 8192
```

## Qwen3 Thinking Mode and vLLM

Qwen3 models have a built-in "thinking mode" that generates `<think>...</think>` tokens before the actual response. This significantly increases generation time and token count.

**To disable thinking mode with vLLM:**

1. Set `enable_think_policy=false` and `enable_think_eval=false` in search args:
```bash
lits-search ... \
    --search-arg enable_think_policy=false enable_think_eval=false
```
* 看日志：`'<think>Okay, let's tackle this crossword puzzle'` - 这是 reward model 的输出，说明 
    * thinking的prompt tokens是包含在max_new_tokens以内的。
    * 而prompt里说的字数限制，模型不考虑<think>里的（把它当作内部思考的感觉）

2. LITS will automatically pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to vLLM for non-OpenAI endpoints.

**Trade-off:** Disabling thinking mode improves speed but may reduce output quality. Qwen3 without thinking may:
- Generate shorter, less accurate responses
- Fail to follow format requirements (e.g., generating 4-letter words instead of 5-letter words for crosswords)

**Recommendation:** Test with thinking mode disabled first. If output quality is unacceptable, re-enable thinking mode and accept the slower inference speed.

## vLLM and transformers Version Conflicts

vLLM 0.15.1+ requires `transformers>=4.56.0`, which conflicts with `autoawq` (used for direct HuggingFace AWQ loading).

**Symptoms:**
```
ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'
```

**Solution:** When using vLLM, uninstall `autoawq` since vLLM has built-in AWQ support:
```bash
pip install -e .  # Install LITS
pip uninstall -y autoawq autoawq-kernels
pip install vllm
```

**Note:** This means you cannot use both vLLM and direct HuggingFace AWQ loading in the same environment. Choose one approach:
- **vLLM serving:** Fast inference via OpenAI-compatible API (no autoawq needed)
- **Direct HuggingFace:** Slower but simpler setup (requires autoawq, incompatible with vLLM)
