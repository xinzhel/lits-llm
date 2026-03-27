"""
LLM Call Logger for analyzing generation patterns in tree search.

This module provides:
1. `create_llm_call_logger()` - Factory to create a callback that appends to JSONL file
2. `load_llm_calls()` - Load records from JSONL file
3. `get_diversity_stats()` - Compute diversity statistics from records
4. `print_diversity_report()` - Print formatted report

The functional design enables:
- Incremental logging (each call appended immediately, crash-safe)
- Decoupled analysis (can analyze any saved log file)
- Simple integration (just pass callback to policy.set_llm_call_fn)

Usage:
    from lits.eval.llm_call_logger import (
        create_llm_call_logger, load_llm_calls, print_diversity_report,
        normalize_crosswords_action, parse_crosswords_correct_actions,
        make_crosswords_correctness_checker
    )
    
    # Create callback that appends to file
    log_llm_call = create_llm_call_logger(f"{result_dir}/llm_calls.jsonl")
    policy.set_llm_call_fn(log_llm_call)
    
    # Run search (logs saved incrementally)...
    
    # Analyze after all instances complete
    records = load_llm_calls(f"{result_dir}/llm_calls.jsonl")
    print_diversity_report(
        records,
        normalize_fn=normalize_crosswords_action,
        is_correct=make_crosswords_correctness_checker(
            parse_crosswords_correct_actions(query_or_goals)
        )
    )
"""
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional, Callable, Dict, List, Tuple, TYPE_CHECKING
import logging

import numpy as np

if TYPE_CHECKING:
    from lits.embedding.base import BaseEmbedder

module_logger = logging.getLogger(__name__)


# =============================================================================
# Callback Factory
# =============================================================================

def create_llm_call_logger(path: str) -> Callable[..., None]:
    """Create a callback function that logs LLM calls to a JSONL file.
    
    Each call is appended immediately (incremental, crash-safe).
    
    Args:
        path: Path to JSONL file for logging
    
    Returns:
        Callback function for Policy.set_llm_call_fn()
    
    Example:
        log_llm_call = create_llm_call_logger("llm_calls.jsonl")
        policy.set_llm_call_fn(log_llm_call)
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def log_llm_call(prompt, response: Any, **kwargs) -> None:
        """Append LLM call record to JSONL file."""
        output = response.text if hasattr(response, 'text') else str(response)
        
        # prompt can be str (concat/CoT) or list (tool_use messages format)
        prompt_str = json.dumps(prompt) if isinstance(prompt, list) else prompt
        
        record = {
            "prompt_hash": hashlib.md5(prompt_str.encode()).hexdigest()[:12],
            "output": output,
            "output_hash": hashlib.md5(output.encode()).hexdigest()[:8],
            "temperature": kwargs.get('temperature'),
            "query_idx": kwargs.get('query_idx'),
            "from_phase": kwargs.get('from_phase', ''),
        }
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return None  # Keep original response
    
    return log_llm_call


# =============================================================================
# Loading
# =============================================================================

def load_llm_calls(path: str) -> List[Dict]:
    """Load LLM call records from JSONL file.
    
    Args:
        path: Path to JSONL file
    
    Returns:
        List of record dicts
    """
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    module_logger.info(f"Loaded {len(records)} records from {path}")
    return records


# =============================================================================
# Normalization Functions (Task-Specific)
# =============================================================================

def normalize_crosswords_action(output: str) -> Optional[str]:
    """Normalize crosswords action output to canonical form.
    
    Handles variations like:
    - "h1. TASK_" -> "h1. task"
    - "h1. tasks" -> "h1. tasks"
    - "should be h1. tasks" -> "h1. tasks"
    - "h1.tasks" -> "h1. tasks"
    
    Args:
        output: Raw LLM output string
    
    Returns:
        Normalized action string "pos. word" (lowercase), or None if unparseable
    """
    patterns = [
        r'([hv][1-5])\.\s*([a-zA-Z_]{1,5})',  # "h1. word" or "h1.word"
        r'([hv][1-5])\s+([a-zA-Z_]{1,5})',     # "h1 word"
        r'([hv][1-5]):\s*([a-zA-Z_]{1,5})',    # "h1: word"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            pos = match.group(1).lower()
            word = match.group(2).lower().rstrip('_')
            if word and word.replace('_', ''):
                return f"{pos}. {word}"
    
    return None


def parse_crosswords_correct_actions(query_or_goals: str) -> Dict[str, str]:
    """Parse crosswords ground truth into position -> word mapping.
    
    Args:
        query_or_goals: Ground truth answers (10 words, newline-separated)
            Format: h1, h2, h3, h4, h5, v1, v2, v3, v4, v5
    
    Returns:
        Dict mapping position to correct word, e.g., {'h1': 'agend', 'v1': 'amass', ...}
    """
    answers = [a.strip().lower() for a in query_or_goals.strip().split('\n') if a.strip()]
    if len(answers) != 10:
        return {}
    
    correct = {}
    for i in range(5):
        correct[f'h{i+1}'] = answers[i]
    for i in range(5):
        correct[f'v{i+1}'] = answers[i+5]
    
    return correct


def make_crosswords_correctness_checker(
    correct_actions: Dict[str, str],
) -> Callable[[str], bool]:
    """Build a predicate that checks whether a normalized crosswords output is correct.

    Args:
        correct_actions: Dict mapping position to correct word,
            e.g., ``{'h1': 'agend', 'v1': 'amass', ...}``.
            Typically from :func:`parse_crosswords_correct_actions`.

    Returns:
        A callable ``(normalized_output: str) -> bool``.
    """
    def _is_correct(norm_out: str) -> bool:
        match = re.match(r'([hv][1-5])\.\s*(\w+)', norm_out)
        if not match:
            return False
        pos, word = match.groups()
        return correct_actions.get(pos) == word

    return _is_correct


# =============================================================================
# Semantic Deduplication
# =============================================================================

def cluster_by_embedding(
    outputs: List[str],
    embedder: "BaseEmbedder",
    threshold: float = 0.65,
) -> List[List[int]]:
    """Group output indices into clusters by embedding cosine similarity.

    Uses a two-pass approach:
    1. Embed all outputs (L2-normalised, so dot product = cosine sim).
    2. Build connected components via union-find: if sim(i, j) > threshold,
       merge their sets.  Transitivity is handled by the union-find structure
       (if sim(a,b) > t AND sim(b,c) > t, then {a,b,c} form one cluster).

    Args:
        outputs: Raw output strings to cluster.
        embedder: A :class:`BaseEmbedder` whose ``embed()`` returns
            L2-normalised vectors.
        threshold: Cosine-similarity threshold for merging (default 0.65).

    Returns:
        List of clusters, each cluster a list of indices into *outputs*.
        Every index appears in exactly one cluster.  Singletons are included.
    """
    n = len(outputs)
    if n == 0:
        return []

    # ── Embed ────────────────────────────────────────────────────────────
    vecs = embedder.embed(outputs)  # (n, dim), L2-normalised

    # ── Pairwise cosine sim via dot product ──────────────────────────────
    sim_matrix = vecs @ vecs.T  # (n, n)

    # ── Union-Find ───────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                union(i, j)

    # ── Collect clusters ─────────────────────────────────────────────────
    clusters_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(i)

    return list(clusters_map.values())


_SEMANTIC_EQ_PROMPT = (
    "You are a strict semantic equivalence judge.\n"
    "Two reasoning steps are equivalent ONLY if they express the same "
    "intent, operate on the same entities, and reach the same conclusion. "
    "Differences in wording or phrasing are fine, but steps that involve "
    "different operations, different subjects, or different outcomes are "
    "NOT equivalent.\n\n"
    "Step A: {a}\n"
    "Step B: {b}\n\n"
    "Are Step A and Step B semantically equivalent? Answer only YES or NO."
)


def judge_semantic_equivalence(
    pairs: List[Tuple[str, str]],
    llm,
) -> List[bool]:
    """Ask an LLM whether each pair of reasoning steps is semantically equivalent.

    One LLM call per pair (simple, no batching tricks).  The prompt asks for
    a YES/NO answer; we parse the first line of the response.

    Args:
        pairs: List of (step_a, step_b) string pairs.
        llm: Any callable matching the ``lits.lm`` interface
            (``llm(prompt, temperature=...) -> Output`` with ``.text``).

    Returns:
        List of bools, same length as *pairs*.  ``True`` means the LLM
        judged the pair semantically equivalent.
    """
    results: List[bool] = []
    for a, b in pairs:
        prompt = _SEMANTIC_EQ_PROMPT.format(a=a, b=b)
        response = llm(prompt, temperature=0.0)
        first_line = response.text.strip().split("\n")[0].upper()
        results.append("YES" in first_line)
    return results


# =============================================================================
# Analysis Functions
# =============================================================================

def _apply_semantic_dedup(
    output_counts: Dict[str, int],
    semantic_dedup: Dict,
) -> Dict[str, int]:
    """Merge semantically equivalent keys in *output_counts* using two-stage pipeline.

    Stage 1: cluster keys by embedding similarity.
    Stage 2: within each cluster, LLM judge confirms equivalence on all pairs;
             merge confirmed-equivalent keys into the first key (canonical).

    Args:
        output_counts: {normalized_output: count} mapping.
        semantic_dedup: Dict with keys ``embedder``, ``llm``, ``threshold`` (float).

    Returns:
        New output_counts dict with equivalent keys merged.
    """
    keys = list(output_counts.keys())
    if len(keys) <= 1:
        return dict(output_counts)

    embedder = semantic_dedup["embedder"]
    llm = semantic_dedup["llm"]
    threshold = semantic_dedup.get("threshold", 0.9)

    # Stage 1: embedding clusters
    clusters = cluster_by_embedding(keys, embedder, threshold)

    merged: Dict[str, int] = {}
    for cluster_indices in clusters:
        if len(cluster_indices) == 1:
            k = keys[cluster_indices[0]]
            merged[k] = output_counts[k]
            continue

        # Stage 2: LLM judge on all pairs within cluster
        cluster_keys = [keys[i] for i in cluster_indices]
        pairs = [
            (cluster_keys[i], cluster_keys[j])
            for i in range(len(cluster_keys))
            for j in range(i + 1, len(cluster_keys))
        ]
        verdicts = judge_semantic_equivalence(pairs, llm)

        # Union-find to merge confirmed-equivalent keys
        n = len(cluster_keys)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if verdicts[pair_idx]:
                    union(i, j)
                pair_idx += 1

        # Collect into canonical groups (first key in group = canonical)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        for members in groups.values():
            canonical = cluster_keys[members[0]]
            total = sum(output_counts[cluster_keys[m]] for m in members)
            merged[canonical] = total

    return merged


def get_diversity_stats(
    records: List[Dict],
    normalize_fn: Optional[Callable[[str], Optional[str]]] = None,
    is_correct: Optional[Callable[[str], bool]] = None,
    semantic_dedup: Optional[Dict] = None,
) -> Dict:
    """Compute diversity statistics grouped by prompt.
    
    Args:
        records: List of LLM call records (from load_llm_calls)
        normalize_fn: Optional function to normalize outputs before comparison.
        is_correct: Optional predicate that returns True if a normalized
            output is correct.  Task-agnostic — for crosswords, use
            :func:`make_crosswords_correctness_checker`.
        semantic_dedup: Optional dict with keys ``embedder`` (BaseEmbedder),
            ``llm`` (LanguageModel), ``threshold`` (float, default 0.65).
            When provided, semantically equivalent outputs are merged
            via the two-stage embedding + LLM judge pipeline.
    
    Returns:
        dict with total_calls, unique_prompts, and per-prompt stats
    """
    by_prompt = defaultdict(list)
    for r in records:
        by_prompt[r['prompt_hash']].append(r['output'])
    
    stats = {
        "total_calls": len(records),
        "unique_prompts": len(by_prompt),
        "by_prompt": {}
    }
    
    for prompt_hash, outputs in by_prompt.items():
        # Normalize outputs if function provided
        normalized = [normalize_fn(o) if normalize_fn else o for o in outputs]
        
        # Count occurrences
        output_counts = defaultdict(int)
        for norm_out in normalized:
            if norm_out is not None:
                output_counts[norm_out] += 1
        
        # Semantic dedup: merge equivalent outputs via embedding + LLM judge
        if semantic_dedup is not None and len(output_counts) > 1:
            output_counts = _apply_semantic_dedup(output_counts, semantic_dedup)
        
        # Determine correct outputs
        correct_outputs = set()
        if is_correct:
            for norm_out in output_counts.keys():
                if is_correct(norm_out):
                    correct_outputs.add(norm_out)
        
        # Calculate stats
        total = len(outputs)
        unique_all = len(output_counts)
        unique_correct = len([o for o in output_counts if o in correct_outputs])
        unique_incorrect = len([o for o in output_counts if o not in correct_outputs])
        correct_count = sum(output_counts[o] for o in correct_outputs)
        incorrect_count = total - correct_count
        
        duplicate_rate = (total - unique_all) / total if total > 0 else 0
        correct_duplicate_rate = (correct_count - unique_correct) / correct_count if correct_count > 0 else 0
        incorrect_duplicate_rate = (incorrect_count - unique_incorrect) / incorrect_count if incorrect_count > 0 else 0
        
        output_list = [
            (out, count, out in correct_outputs)
            for out, count in sorted(output_counts.items(), key=lambda x: -x[1])
        ]
        
        stats["by_prompt"][prompt_hash] = {
            "total": total,
            "unique": unique_all,
            "unique_correct": unique_correct,
            "unique_incorrect": unique_incorrect,
            "duplicate_rate": duplicate_rate,
            "correct_duplicate_rate": correct_duplicate_rate,
            "incorrect_duplicate_rate": incorrect_duplicate_rate,
            "correct_count": correct_count,
            "outputs": output_list,
        }
    
    return stats


def print_diversity_report(
    records: List[Dict],
    normalize_fn: Optional[Callable[[str], Optional[str]]] = None,
    is_correct: Optional[Callable[[str], bool]] = None,
    semantic_dedup: Optional[Dict] = None,
) -> None:
    """Log a formatted diversity analysis report to file.

    Args:
        records: List of LLM call records
        normalize_fn: Optional function to normalize outputs
        is_correct: Optional predicate passed through to get_diversity_stats
        semantic_dedup: Optional dict passed through to get_diversity_stats
    """
    stats = get_diversity_stats(records, normalize_fn, is_correct, semantic_dedup)

    module_logger.info(f"{'='*70}")
    module_logger.info("LLM Call Diversity Report")
    module_logger.info(f"{'='*70}")
    module_logger.info(f"Unique states visited: {stats['unique_prompts']}")
    module_logger.info(f"Avg. policy calls per state: {stats['total_calls'] / stats['unique_prompts']:.1f}")

    # Overall stats
    total_outputs = sum(s['total'] for s in stats['by_prompt'].values())
    total_unique = sum(s['unique'] for s in stats['by_prompt'].values())
    total_correct = sum(s['correct_count'] for s in stats['by_prompt'].values())
    total_incorrect = total_outputs - total_correct
    total_unique_correct = sum(s['unique_correct'] for s in stats['by_prompt'].values())
    total_unique_incorrect = sum(s['unique_incorrect'] for s in stats['by_prompt'].values())

    overall_dup_rate = (total_outputs - total_unique) / total_outputs if total_outputs > 0 else 0
    correct_dup_rate = (total_correct - total_unique_correct) / total_correct if total_correct > 0 else 0
    incorrect_dup_rate = (total_incorrect - total_unique_incorrect) / total_incorrect if total_incorrect > 0 else 0

    module_logger.info(f"Dup. rate (all): {overall_dup_rate:.1%}")
    if is_correct:
        module_logger.info(f"Dup. rate (correct): {correct_dup_rate:.1%}")
        module_logger.info(f"Dup. rate (incorrect): {incorrect_dup_rate:.1%}")

    # Per-prompt breakdown
    module_logger.info("Per-prompt breakdown:")
    header = f"{'Prompt':<12} {'Total':<6} {'Uniq':<6} {'Dup%':<8}"
    if is_correct:
        header += f" {'Corr':<6} {'CorrDup%':<9} {'IncDup%':<8}"
    module_logger.info(header)
    module_logger.info("-" * len(header))

    for prompt_hash, s in sorted(stats['by_prompt'].items(),
                                  key=lambda x: x[1]['incorrect_duplicate_rate'],
                                  reverse=True):
        row = f"{prompt_hash:<12} {s['total']:<6} {s['unique']:<6} {s['duplicate_rate']:.1%}"
        if is_correct:
            corr_dup = f"{s['correct_duplicate_rate']:.1%}" if s['correct_count'] > 0 else "N/A"
            row += f"   {s['correct_count']:<6} {corr_dup:<9} {s['incorrect_duplicate_rate']:.1%}"
        module_logger.info(row)

    module_logger.info(f"{'='*70}")
