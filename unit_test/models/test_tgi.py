"""Test TGI model with inference logger for token usage tracking.

Requires:
    export TGI_ENDPOINT=http://localhost:8080

Run:
    ```
    cd lits_llm/unit_test/models
    python test_tgi.py
    ```
"""
import sys
sys.path.append('../..')

from lits.lm import get_lm
from lits.lm.base import InferenceLogger


def test_basic_generation():
    """Test basic text generation."""
    logger = InferenceLogger(run_id='tgi_test', root_dir='./results', override=True)
    m = get_lm('tgi:///meta-llama/Meta-Llama-3-8B', inference_logger=logger)
    
    result = m('2+2=', max_new_tokens=10)
    print(f"Output: {result.text}")
    print(f"Metrics: {logger.get_metrics_by_role()}")


def test_get_next_token_logits():
    """Test get_next_token_logits for token classification.
    
    Note: Uses grammar constraints, requires N API calls for N candidates.
    Single-character tokens may return logprob=0.0 due to TGI behavior.
    """
    m = get_lm('tgi:///meta-llama/Meta-Llama-3-8B')
    
    # Test with Yes/No (multi-character tokens work well)
    prompt = "Is 2+2=4? Answer Yes or No:"
    candidates = ["Yes", "No"]
    
    logprobs = m.get_next_token_logits(prompt, candidates)
    
    print(f"Prompt: {prompt}")
    print(f"Candidates: {candidates}")
    print(f"Logprobs: {logprobs}")
    
    # Convert to probabilities
    import math
    probs = [math.exp(lp) if lp > float('-inf') else 0.0 for lp in logprobs]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    print(f"Normalized probs: {dict(zip(candidates, probs))}")


def test_get_top5_logits():
    """Test get_top5_logits for fast top-5 token retrieval.
    
    Single API call, but limited to top 5 tokens.
    """
    m = get_lm('tgi:///meta-llama/Meta-Llama-3-8B')
    
    prompt = "2+2="
    
    # Get all top 5 tokens
    top5 = m.get_top5_logits(prompt)
    print(f"Prompt: {prompt}")
    print(f"Top 5 tokens: {top5}")
    
    # Get logprobs for specific candidates
    candidates = ["4", "5", "6"]
    logprobs = m.get_top5_logits(prompt, candidates)
    print(f"Candidates {candidates} logprobs: {logprobs}")
    print(f"Note: -inf means candidate not in top 5")


if __name__ == "__main__":
    print("=== Test basic generation ===")
    test_basic_generation()
    print()
    print("=== Test get_next_token_logits (grammar-based) ===")
    test_get_next_token_logits()
    print()
    print("=== Test get_top5_logits ===")
    test_get_top5_logits()
