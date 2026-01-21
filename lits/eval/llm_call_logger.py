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
        normalize_crosswords_action, parse_crosswords_correct_actions
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
        correct_actions=parse_crosswords_correct_actions(query_or_goals)
    )
"""
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional, Callable, Dict, List
import logging

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
    
    def log_llm_call(prompt: str, response: Any, **kwargs) -> None:
        """Append LLM call record to JSONL file."""
        output = response.text if hasattr(response, 'text') else str(response)
        
        record = {
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:12],
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


# =============================================================================
# Analysis Functions
# =============================================================================

def get_diversity_stats(
    records: List[Dict],
    normalize_fn: Optional[Callable[[str], Optional[str]]] = None,
    correct_actions: Optional[Dict[str, str]] = None
) -> Dict:
    """Compute diversity statistics grouped by prompt.
    
    Args:
        records: List of LLM call records (from load_llm_calls)
        normalize_fn: Optional function to normalize outputs before comparison.
        correct_actions: Optional dict mapping position to correct word.
    
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
        
        # Determine correct outputs
        correct_outputs = set()
        if correct_actions and normalize_fn:
            for norm_out in output_counts.keys():
                match = re.match(r'([hv][1-5])\.\s*(\w+)', norm_out)
                if match:
                    pos, word = match.groups()
                    if correct_actions.get(pos) == word:
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
    correct_actions: Optional[Dict[str, str]] = None
) -> None:
    """Print a formatted diversity analysis report.
    
    Args:
        records: List of LLM call records
        normalize_fn: Optional function to normalize outputs
        correct_actions: Optional dict mapping position to correct word
    """
    stats = get_diversity_stats(records, normalize_fn, correct_actions)
    
    print(f"\n{'='*70}")
    print("LLM Call Diversity Report")
    print(f"{'='*70}")
    print(f"Unique states visited: {stats['unique_prompts']}")
    print(f"Avg. policy calls per state: {stats['total_calls'] / stats['unique_prompts']:.1f}")
    print()
    
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
    
    print(f"Dup. rate (all): {overall_dup_rate:.1%}")
    if correct_actions:
        print(f"Dup. rate (correct): {correct_dup_rate:.1%}")
        print(f"Dup. rate (incorrect): {incorrect_dup_rate:.1%}")
    print()
    
    # Per-prompt breakdown
    print("Per-prompt breakdown:")
    header = f"{'Prompt':<12} {'Total':<6} {'Uniq':<6} {'Dup%':<8}"
    if correct_actions:
        header += f" {'Corr':<6} {'CorrDup%':<9} {'IncDup%':<8}"
    print(header)
    print("-" * len(header))
    
    for prompt_hash, s in sorted(stats['by_prompt'].items(), 
                                  key=lambda x: x[1]['incorrect_duplicate_rate'], 
                                  reverse=True):
        row = f"{prompt_hash:<12} {s['total']:<6} {s['unique']:<6} {s['duplicate_rate']:.1%}"
        if correct_actions:
            corr_dup = f"{s['correct_duplicate_rate']:.1%}" if s['correct_count'] > 0 else "N/A"
            row += f"   {s['correct_count']:<6} {corr_dup:<9} {s['incorrect_duplicate_rate']:.1%}"
        print(row)
    
    print(f"{'='*70}\n")
