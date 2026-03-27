"""Test semantic diversity functions: cluster_by_embedding, judge_semantic_equivalence,
and the full get_diversity_stats pipeline with semantic_dedup.

Source: llm_call_logger.py::cluster_by_embedding, judge_semantic_equivalence, get_diversity_stats
"""
import hashlib
from lits.embedding import get_embedder
from lits.lm import get_lm
from lits.eval.llm_call_logger import (
    cluster_by_embedding, judge_semantic_equivalence, get_diversity_stats,
)

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_all():
    embedder = get_embedder("multi-qa-mpnet-base-cos-v1")
    llm = get_lm(MODEL_NAME)

    # ── 1. cluster_by_embedding ──────────────────────────────────────────
    outputs = ["Let x = 5", "Set x to 5", "Define x as 5",
               "Compute y = x^2", "Calculate the square of x to get y"]
    clusters = cluster_by_embedding(outputs, embedder, threshold=0.7)
    print(f"Clusters: {[[outputs[i] for i in c] for c in clusters]}")
    breakpoint()  # inspect: clusters — expect 2 groups: {0,1,2} and {3,4}

    # ── 2. judge_semantic_equivalence ────────────────────────────────────
    pairs = [("Let x = 5", "Set x to 5"), ("Let x = 5", "Compute y = x^2")]
    verdicts = judge_semantic_equivalence(pairs, llm)
    print(f"Judge verdicts: {verdicts}")
    breakpoint()  # inspect: verdicts — expect [True, False]

    # ── 3. get_diversity_stats with semantic_dedup ───────────────────────
    prompt = "Solve: what is 2+3?"
    ph = hashlib.md5(prompt.encode()).hexdigest()[:12]
    records = [
        {"prompt_hash": ph, "output": "Let x = 2+3 = 5"},
        {"prompt_hash": ph, "output": "Set x = 2+3, so x = 5"},
        {"prompt_hash": ph, "output": "The answer is 5"},
    ]
    stats = get_diversity_stats(
        records,
        semantic_dedup={"embedder": embedder, "llm": llm, "threshold": 0.7},
    )
    prompt_stats = stats["by_prompt"][ph]
    print(f"Total: {prompt_stats['total']}, Unique: {prompt_stats['unique']}")
    print(f"Outputs: {prompt_stats['outputs']}")
    breakpoint()  # inspect: prompt_stats — unique should be < 3 if dedup merges equivalent steps


if __name__ == "__main__":
    test_all()
