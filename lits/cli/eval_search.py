"""
lits-eval: Evaluate tree search results from checkpoint files.

Usage:
    lits-eval --result_dir results/math500_rest/run_0.2.10 --dataset_name math500 \
        --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    lits-eval --result_dir results/blocksworld_rap/run_0.2.10 --dataset_name blocksworld \
        --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    lits-eval --help

Auto-loads import_modules and dataset_kwargs from config.json in result_dir.
See docs/cli/search.md for full CLI documentation.
"""

import sys
import os
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.common import extract_answers_from_terminal_nodes
from lits.benchmarks.registry import load_dataset, has_resource, has_evaluator, get_evaluator
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules, load_config_from_result_dir
from lits.components.utils import get_fn_retrieve_answer
from lits.lm import get_lm
from lits.eval.inference_report import generate_report
from lits.log import setup_logging
from lits.structures import ToolUseState, ToolUseStep  # Import to register types
from lits.structures.env_grounded import EnvState, EnvStep  # Import to register types

try:
    import botocore.exceptions
except ImportError:
    botocore = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def load_terminal_nodes_from_file(filepath: Path) -> Dict[str, Any]:
    """
    Load terminal nodes from a checkpoint file.
    
    Args:
        filepath: Path to terminal_nodes_{query_idx}.json file
    
    Returns:
        Dictionary containing terminal_nodes, query, and query_idx
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle empty terminal_nodes list
    if not data['terminal_nodes']:
        return {
            'terminal_nodes': [],
            'query': data['query'],
            'query_idx': data['query_idx']
        }
    
    # Deserialize nodes using from_dict
    # Determine node type from the data
    if 'cum_rewards' in data['terminal_nodes'][0]:
        node_class = MCTSNode
    else:
        node_class = SearchNode
    
    terminal_nodes = [node_class.from_dict(node_dict) for node_dict in data['terminal_nodes']]
    
    return {
        'terminal_nodes': terminal_nodes,
        'query': data['query'],
        'query_idx': data['query_idx']
    }


def get_goal_check_from_registry(dataset_name: str) -> Callable:
    """Get goal_check function from registry for env_grounded tasks.
    
    Args:
        dataset_name: Name of the registered benchmark
        
    Returns:
        The goal_check static method from the registered Transition class
        
    Raises:
        KeyError: If the benchmark is not registered
    """
    try:
        TransitionCls = ComponentRegistry.get_transition(dataset_name)
    except KeyError as e:
        available = ComponentRegistry.list_by_task_type("env_grounded")
        raise KeyError(
            f"Benchmark '{dataset_name}' not found in registry. "
            f"Available env_grounded benchmarks: {available}. "
            f"Did you forget to import the module containing @register_transition('{dataset_name}')?"
        ) from e
    
    if not hasattr(TransitionCls, 'goal_check'):
        raise AttributeError(
            f"Transition class '{TransitionCls.__name__}' for benchmark '{dataset_name}' "
            f"does not have a 'goal_check' static method."
        )
    
    return TransitionCls.goal_check


def is_env_grounded_task(dataset_name: str) -> bool:
    """Check if dataset is an env_grounded task using registry."""
    env_grounded_benchmarks = ComponentRegistry.list_by_task_type("env_grounded")
    return dataset_name.lower() in [b.lower() for b in env_grounded_benchmarks]


def _discover_checkpoint_files(checkpoints_dir: Path) -> tuple:
    """Scan a checkpoints directory and classify its contents.

    Returns:
        ``(files, is_pass_at_n, n_attempts)`` where *files* is a sorted
        list of ``Path`` objects (only recognized checkpoint patterns),
        *is_pass_at_n* is ``True`` when ``{idx}_a{attempt}.json`` files
        are present, and *n_attempts* is the max attempt count (1 for
        single-attempt runs).
    """
    all_json = sorted(checkpoints_dir.glob("*.json"), key=lambda f: f.stem)
    files = []
    max_attempt = -1
    for f in all_json:
        try:
            _idx, attempt = _parse_checkpoint_filename(f)
        except ValueError:
            continue  # e.g. "pass_at_n_summary.json", "3_a2_incomplete.json"
        files.append(f)
        if attempt is not None and attempt > max_attempt:
            max_attempt = attempt

    is_pass_at_n = max_attempt >= 0
    n_attempts = (max_attempt + 1) if is_pass_at_n else 1  # attempts are 0-indexed
    return files, is_pass_at_n, n_attempts


def _parse_checkpoint_filename(filepath: Path) -> tuple:
    """Extract ``(query_idx, attempt)`` from a checkpoint filename.

    Returns:
        ``(query_idx: int, attempt: int | None)``.

    Raises:
        ``ValueError`` if the filename doesn't match any known pattern.

    Supported patterns::

        "3_a2.json"             → (3, 2)   pass@N checkpoint
        "3.json"                → (3, None) single-attempt chain checkpoint
        "terminal_nodes_3.json" → (3, None) tree search result
    """
    import re
    filename = filepath.stem
    # "3_a2" → pass@N checkpoint, example idx = 3, attempt = 2
    m = re.match(r'^(\d+)_a(\d+)$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "3" → single-attempt chain checkpoint, example idx = 3
    if re.match(r'^\d+$', filename):
        return int(filename), None
    # "terminal_nodes_3" → tree search result, example idx = 3
    m = re.match(r'^terminal_nodes_(\d+)$', filename)
    if m:
        return int(m.group(1)), None
    raise ValueError(f"Unrecognized checkpoint filename: {filepath.name}")


def _should_include_file(filepath: Path, offset: int, end_idx: int = None) -> bool:
    """Check if a checkpoint file's example index falls within ``[offset, end_idx)``.

    Only includes files matching known checkpoint patterns (see
    ``_parse_checkpoint_filename``). Files with extra suffixes
    (e.g. ``1_a2_incomplete.json``) are excluded.
    """
    try:
        idx, _attempt = _parse_checkpoint_filename(filepath)
    except ValueError:
        return False
    if idx < offset:
        return False
    if end_idx is not None and idx >= end_idx:
        return False
    return True


def _check_answer(pred: str, truth, custom_evaluator=None, llm_evaluator=None,
                  llm_eval_mode: str = "binary",
                  eval_logger=None, label: str = "") -> tuple:
    """Evaluate a single predicted answer against ground truth.

    Tries evaluators in priority order:
    1. ``custom_evaluator`` (dataset-specific, e.g. dbbench float tolerance)
    2. ``llm_evaluator`` fallback (LLM-as-judge for verbose answers)
    3. Exact string match

    Args:
        pred: Predicted answer string.
        truth: Ground truth (any type — passed directly to custom_evaluator;
            converted to str only for the fallback exact-match path).
        custom_evaluator: Registered evaluator function, or *None*.
        llm_evaluator: ``GeneralEvaluator`` instance, or *None*.
        llm_eval_mode: ``"binary"`` (yes/no) or ``"f1"`` (float score).
        eval_logger: Logger for debug messages (optional).
        label: Human-readable label for log lines (e.g. ``"[3_a2]"``).

    Returns:
        ``(correct, score)`` where *correct* is bool and *score* is
        ``Optional[float]`` (non-None when the evaluator returns a
        continuous score, e.g. F1 mode or a float-returning custom
        evaluator).
    """
    score = None

    if custom_evaluator:
        try:
            result = custom_evaluator(pred, truth)
            if isinstance(result, float):
                score = result
                correct = (result == 1.0)
            else:
                correct = bool(result)
        except Exception as e:
            correct = False
            if eval_logger:
                eval_logger.debug(f"  {label} custom evaluator failed: {e}")
        if not correct and llm_evaluator:
            correct, score = _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, score)
            if eval_logger:
                eval_logger.debug(f"  {label} LLM fallback: correct={correct}, score={score}")
        return correct, score

    if llm_evaluator:
        correct, score = _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, score)
        if eval_logger:
            eval_logger.debug(f"  {label} LLM evaluator: correct={correct}, score={score}")
        return correct, score

    return (pred == str(truth)), None


def _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, prior_score=None):
    """Run LLM evaluator. Returns ``(correct, score)``."""
    if llm_eval_mode == "f1":
        llm_score = llm_evaluator.check_score(pred, truth)
        final = max(prior_score, llm_score) if prior_score is not None else llm_score
        return (final == 1.0), final
    else:
        return llm_evaluator.check_correct(pred, truth), prior_score


def _create_llm_evaluator(base_model, llm_eval_mode: str, eval_logger):
    """Create an LLM-based evaluator if mode is not ``"none"``.

    Returns:
        A ``GeneralEvaluator`` instance, or *None*.
    """
    if llm_eval_mode == "none":
        return None
    from lits.eval.general_eval import GeneralEvaluator
    if llm_eval_mode == "f1":
        evaluator = GeneralEvaluator(
            base_model=base_model,
            eval_perspectives=[{
                "eval_id": "score",
                "output_type": "float",
                "description": (
                    "Compute the F1 score between the predicted and ground truth answer sets. "
                    "F1 = 2 * precision * recall / (precision + recall), where "
                    "precision = (correct predictions) / (total predictions), "
                    "recall = (correct predictions) / (total ground truth elements). "
                    "Output 0.0 if no overlap, 1.0 if exact match. "
                    "Penalize both missing elements (low recall) and extra wrong elements (low precision)."
                ),
            }],
        )
        eval_logger.info("LLM-based F1 score evaluator enabled")
    else:
        evaluator = GeneralEvaluator(
            base_model=base_model,
            eval_perspectives=[{
                "eval_id": "correct",
                "description": (
                    "Does the predicted answer contain the correct value? "
                    "Ignore formatting differences, extra explanation, or markdown. "
                    "Focus only on whether the core answer value matches."
                ),
                "options": ["yes", "no"],
            }],
        )
        eval_logger.info("LLM-based binary evaluator enabled")
    return evaluator


def _save_eval_results(results: dict, result_dir: "Path", eval_logger):
    """Write ``eval_results.json`` to *result_dir*.

    Both the single-attempt and pass@N evaluation paths call this so
    there is exactly one save location for every eval run.
    """
    path = result_dir / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    eval_logger.info(f"Evaluation results saved to {path}")


def evaluate_from_checkpoints(
    result_dir: str,
    dataset_name: str,
    eval_model_name: str,
    offset: int = 0,
    limit: int = None,
    dataset_kwargs: dict = None,
    input_price_per_m: float = None,
    output_price_per_m: float = None,
    verbose: bool = False,
    llm_eval_mode: str = "binary",
):
    """
    Evaluate tree search results from checkpoint files.
    
    Args:
        result_dir: Directory containing terminal_nodes_*.json files
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'math500', 'blocksworld')
        eval_model_name: Model name used for answer extraction
        offset: Dataset offset used during search
        limit: Dataset limit used during search
        dataset_kwargs: Dataset-specific arguments (loaded from config)
        verbose: If True, print detailed output to console
    """
    result_dir = Path(result_dir)
    
    # Setup logging - only log to file, not console (unless verbose)
    eval_logger = setup_logging(
        run_id="eval",
        result_dir=result_dir,
        add_console_handler=verbose,
        verbose=True
    )
    
    from lits.cli import log_command
    log_command(eval_logger)
    
    # Determine task type using registry
    is_env_grounded = is_env_grounded_task(dataset_name)
    is_tool_use = has_resource(dataset_name)
    
    # Load dataset for ground truths (not needed for env_grounded tasks)
    if is_env_grounded:
        # For env_grounded tasks, get goal_check from registry
        goal_check = get_goal_check_from_registry(dataset_name)
        full_dataset = None
        eval_logger.info(f"Evaluating {dataset_name} results (goal checking via registry)")
    elif is_tool_use:
        # Tool-use datasets are registered via @register_dataset, same as other task types
        full_dataset = load_dataset(dataset_name, **(dataset_kwargs or {}))
        # NOTE: Do NOT slice the dataset here. The query_idx in terminal node files
        # refers to the original dataset index, not the sliced index.
    else:
        dataset_kwargs = dataset_kwargs or {}
        full_dataset = load_dataset(dataset_name, **dataset_kwargs)
        # NOTE: Do NOT slice the dataset here. The query_idx in terminal node files
        # refers to the index within the (possibly filtered) dataset used during search.
        # We need the same filtered dataset to look up ground truths by query_idx.
    
    # Create a minimal model instance for answer retrieval (not needed for env_grounded)
    if not is_env_grounded:
        base_model = get_lm(eval_model_name)
    
    # Get answer retrieval function
    if is_env_grounded:
        # For environment-grounded tasks, check goal satisfaction
        def retrieve_answer_from_env_node(node, query):
            """Extract final state and check goal satisfaction."""
            if hasattr(node, 'state') and node.state:
                state = node.state
                if isinstance(state, EnvState):
                    # Get the final env_state (after all steps)
                    final_env_state = state.env_state
                    is_reached, score = goal_check(query, final_env_state)
                    return "correct" if is_reached else "incorrect"
            return "incorrect"
        retrieve_answer = retrieve_answer_from_env_node
    elif is_tool_use:
        # For tool use, extract answer from the step (stored in node)
        def retrieve_answer_from_tool_use_node(node, query):
            # Check if node has a step with an answer
            if hasattr(node, 'step') and node.step:
                step = node.step
                if hasattr(step, 'answer') and step.answer:
                    return step.answer
                # Try to get answer from step dict if not deserialized
                if isinstance(step, dict) and 'answer' in step:
                    return step['answer']
            # Fallback: check state
            if hasattr(node, 'state') and node.state:
                if isinstance(node.state, list) and len(node.state) > 0:
                    last_step = node.state[-1]
                    if isinstance(last_step, dict) and 'answer' in last_step:
                        return last_step['answer']
            return ""
        retrieve_answer = retrieve_answer_from_tool_use_node
    else:
        retrieve_answer = get_fn_retrieve_answer(base_model)
    
    # Find result files: terminal_nodes/ (tree search) or checkpoints/ (chain)
    terminal_nodes_dir = result_dir / "terminal_nodes"
    checkpoints_dir = result_dir / "checkpoints"
    use_chain_checkpoints = False
    is_pass_at_n = False

    if terminal_nodes_dir.exists():
        terminal_node_files = sorted(terminal_nodes_dir.glob("terminal_nodes_*.json"),
                                     key=lambda f: int(f.stem.split('_')[-1]))
        if not terminal_node_files:
            eval_logger.error(f"No terminal node files found in {terminal_nodes_dir}")
            return
        eval_logger.info(f"Found {len(terminal_node_files)} terminal node files")
    elif checkpoints_dir.exists():
        use_chain_checkpoints = True
        terminal_node_files, is_pass_at_n, n_attempts = _discover_checkpoint_files(checkpoints_dir)
        if not terminal_node_files:
            eval_logger.error(f"No checkpoint files found in {checkpoints_dir}")
            return
        if is_pass_at_n:
            eval_logger.info(f"Found {len(terminal_node_files)} pass@N checkpoint files (up to {n_attempts} attempts)")
        else:
            eval_logger.info(f"Found {len(terminal_node_files)} chain checkpoint files")
    else:
        eval_logger.error(f"Neither terminal_nodes/ nor checkpoints/ found in {result_dir}")
        return
    
    # Filter terminal node files by offset and limit based on query_idx
    # The query_idx in terminal node files corresponds to the original dataset index
    end_idx = offset + limit if limit is not None else None
    
    filtered_files = [f for f in terminal_node_files if _should_include_file(f, offset, end_idx)]
    eval_logger.info(f"Processing {len(filtered_files)} files (offset={offset}, limit={limit})")

    # --- Setup evaluators (shared by both single-attempt and pass@N) ---
    from lits.components.utils import eval_output

    custom_evaluator = get_evaluator(dataset_name) if has_evaluator(dataset_name) else None
    if custom_evaluator:
        eval_logger.info(f"Using registered evaluator for '{dataset_name}'")

    llm_evaluator = None
    if is_tool_use and llm_eval_mode != "none":
        llm_evaluator = _create_llm_evaluator(base_model, llm_eval_mode, eval_logger)

    # --- Evaluate every checkpoint file (single-attempt and pass@N alike) ---
    # Each file produces one (query_idx, correct, score) result.
    # For pass@N files ({idx}_a{attempt}.json), query_idx is parsed from the
    # filename prefix; for single files ({idx}.json), it's the whole stem.
    EvalPoint = dict  # {"idx": int, "attempt": int|None, "correct": bool, "score": float}
    eval_points: List[EvalPoint] = []
    soft_scores = []  # env_grounded word-level scores
    eval_scores = []  # continuous scores from float-returning evaluators

    eval_logger.info("=" * 40)
    eval_logger.info("Detailed comparison:")

    for filepath in tqdm(filtered_files, desc="Evaluating", unit="file"):
        try:
            # --- Parse query_idx and attempt from filename ---
            query_idx, attempt = _parse_checkpoint_filename(filepath)

            # --- Extract predicted answer ---
            if use_chain_checkpoints:
                from lits.structures.tool_use import ToolUseState
                _query, state = ToolUseState.load(str(filepath))
                ground_truth = full_dataset[query_idx]['answer']
                answer_pred = state.get_final_answer() or ""
            elif is_env_grounded:
                data = load_terminal_nodes_from_file(filepath)
                terminal_nodes = data['terminal_nodes']
                _query = data['query']
                ground_truth = "correct"
                if terminal_nodes:
                    def get_best_reward(node):
                        if hasattr(node, 'cum_rewards') and node.cum_rewards:
                            return max(node.cum_rewards) if isinstance(node.cum_rewards, list) else node.cum_rewards
                        return -float('inf')
                    best_node = sorted(terminal_nodes, key=get_best_reward, reverse=True)[0]
                    if hasattr(best_node, 'state') and best_node.state and isinstance(best_node.state, EnvState):
                        is_reached, env_score = goal_check(_query, best_node.state.env_state)
                        answer_pred = "correct" if is_reached else "incorrect"
                        soft_scores.append(env_score)
                    else:
                        answer_pred = "incorrect"
                        soft_scores.append(0.0)
                else:
                    answer_pred = "incorrect"
                    soft_scores.append(0.0)
            elif is_tool_use:
                data = load_terminal_nodes_from_file(filepath)
                terminal_nodes = data['terminal_nodes']
                _query = data['query']
                ground_truth = full_dataset[query_idx]['answer']
                answer_pred = retrieve_answer(terminal_nodes[0], _query) if terminal_nodes else ""
                if "mapeval" in dataset_name:
                    ground_truth = "Option " + str(ground_truth)
            else:
                data = load_terminal_nodes_from_file(filepath)
                terminal_nodes = data['terminal_nodes']
                _query = data['query']
                ground_truth = full_dataset[query_idx]['answer']
                vote_answers, *_ = extract_answers_from_terminal_nodes(
                    terminal_nodes_collected=terminal_nodes,
                    retrieve_answer=retrieve_answer,
                    query=_query,
                )
                answer_pred = max(vote_answers, key=lambda a: vote_answers[a]) if vote_answers else ''

            # --- Score the prediction ---
            label = f"[{query_idx}_a{attempt}]" if attempt is not None else f"[{query_idx}]"
            if is_env_grounded:
                correct = (answer_pred == ground_truth)
                score = None
            elif custom_evaluator or llm_evaluator:
                correct, score = _check_answer(
                    answer_pred, ground_truth,
                    custom_evaluator=custom_evaluator,
                    llm_evaluator=llm_evaluator,
                    llm_eval_mode=llm_eval_mode,
                    eval_logger=eval_logger,
                    label=label,
                )
                if score is not None:
                    eval_scores.append(score)
            else:
                try:
                    correct = eval_output(str(ground_truth), answer_pred, type="number_exact")
                except (AssertionError, ValueError):
                    correct = (answer_pred == str(ground_truth))
                score = None

            eval_logger.info(f"  {label} Pred='{str(answer_pred)[:80]}', Truth='{str(ground_truth)[:80]}', Correct={correct}")
            eval_points.append({"idx": query_idx, "attempt": attempt, "correct": correct, "score": 1.0 if correct else 0.0})

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # Auth errors are global — fail fast instead of silently skipping all files
            if botocore is not None and isinstance(e, (
                botocore.exceptions.SSOError,
                botocore.exceptions.UnauthorizedSSOTokenError,
                botocore.exceptions.TokenRetrievalError,
            )):
                raise
            eval_logger.error(f"Error processing {filepath}: {e}")
            eval_logger.error(traceback.format_exc())

    eval_logger.info("=" * 40)

    # --- Aggregate results ---
    if is_pass_at_n:
        # Group by example_idx, compute pass@1 (attempt 0) and pass@N (any)
        from collections import defaultdict
        groups = defaultdict(list)
        for pt in eval_points:
            groups[pt["idx"]].append(pt)
        for idx in groups:
            groups[idx].sort(key=lambda p: p["attempt"] if p["attempt"] is not None else 0)

        examples = []
        n_pass_1 = 0
        n_pass_n = 0
        for idx in sorted(groups.keys()):
            attempts = [p["score"] for p in groups[idx]]
            p1 = attempts[0] == 1.0 if attempts else False
            pn = any(s == 1.0 for s in attempts)
            if p1: n_pass_1 += 1
            if pn: n_pass_n += 1
            examples.append({"idx": idx, "attempts": attempts, "pass_at_1": p1, "pass_at_n": pn})

        total = len(groups)
        eval_results = {
            "n_attempts": n_attempts,
            "temperature": None,
            "pass_at_1": n_pass_1 / total if total else 0,
            "pass_at_n": n_pass_n / total if total else 0,
            "examples": examples,
            "dataset_name": dataset_name,
            "result_dir": str(result_dir),
        }

        eval_logger.info("=" * 60)
        eval_logger.info(f"Pass@N Evaluation Results: {result_dir.name}")
        eval_logger.info(f"Dataset: {dataset_name}, Examples: {total}, Attempts: {n_attempts}")
        eval_logger.info(f"pass@1: {eval_results['pass_at_1']:.1%} ({n_pass_1}/{total})")
        eval_logger.info(f"pass@{n_attempts}: {eval_results['pass_at_n']:.1%} ({n_pass_n}/{total})")
        eval_logger.info("=" * 60)

        print()
        print("=" * 60)
        print(f"  Pass@N Evaluation: {result_dir.name}")
        print("=" * 60)
        print(f"  Dataset:    {dataset_name}")
        print(f"  Examples:   {total}")
        print(f"  Attempts:   {n_attempts}")
        print(f"  pass@1:     {eval_results['pass_at_1']:.1%} ({n_pass_1}/{total})")
        print(f"  pass@{n_attempts}:     {eval_results['pass_at_n']:.1%} ({n_pass_n}/{total})")
        print("=" * 60)

    else:
        # Single-attempt: flat accuracy
        correct_count = sum(1 for p in eval_points if p["correct"])
        total_count = len(eval_points)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        soft_accuracy = (sum(soft_scores) / len(soft_scores)) if (is_env_grounded and soft_scores) else None
        mean_score = (sum(eval_scores) / len(eval_scores)) if eval_scores else None

        eval_results = {
            "n_attempts": 1,
            "temperature": None,
            "accuracy": accuracy,
            "examples": [{"idx": p["idx"], "correct": p["correct"], "score": p["score"]} for p in eval_points],
            "dataset_name": dataset_name,
            "result_dir": str(result_dir),
        }
        if soft_accuracy is not None:
            eval_results["soft_accuracy"] = soft_accuracy
        if mean_score is not None:
            eval_results["mean_score"] = mean_score

        eval_logger.info("=" * 80)
        eval_logger.info(f"Evaluation Results for {result_dir.name}")
        eval_logger.info(f"Dataset: {dataset_name}, Examples: {total_count}, Correct: {correct_count}")
        eval_logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if mean_score is not None:
            eval_logger.info(f"Mean Score: {mean_score:.4f} ({mean_score*100:.2f}%)")
        if soft_accuracy is not None:
            eval_logger.info(f"Soft Accuracy: {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
        eval_logger.info("=" * 80)

        # Token usage report
        report_kwargs = {}
        if input_price_per_m is not None:
            report_kwargs["input_price_per_m"] = input_price_per_m
        if output_price_per_m is not None:
            report_kwargs["output_price_per_m"] = output_price_per_m
        report = generate_report(str(result_dir), **report_kwargs)
        eval_logger.info(report)

        print()
        print("=" * 60)
        print(f"  Evaluation Results: {result_dir.name}")
        print("=" * 60)
        print(f"  Dataset:    {dataset_name}")
        print(f"  Examples:   {total_count}")
        print(f"  Correct:    {correct_count}")
        print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        if mean_score is not None:
            print(f"  Mean Score: {mean_score:.4f} ({mean_score*100:.2f}%)")
        if soft_accuracy is not None:
            print(f"  Soft Acc:   {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
        print("=" * 60)
        print()
        print(report)

    _save_eval_results(eval_results, result_dir, eval_logger)
    return eval_results


def main() -> int:
    """Entry point for lits-eval command.
    
    Evaluates tree search results from checkpoint files. Auto-loads
    import_modules and dataset_kwargs from config.json in result_dir.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Evaluate tree search results from checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate results (auto-loads dataset_name, eval_model_name, import_modules from config)
  lits-eval --result_dir demo_results
  
  # Explicit overrides
  lits-eval --result_dir claude35v1_results/blocksworld_rap/run_0.2.10 \\
      --dataset_name blocksworld --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
  
  # Evaluate custom benchmark (import module to register Transition)
  lits-eval --result_dir results/robot_arm_rap/run_0.2.10 \\
      --dataset_name robot_arm --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \\
      --include my_project.robot_arm
"""
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing terminal_nodes/ subdirectory")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (e.g., gsm8k, math500, blocksworld). Auto-loaded from config.json if not specified.")
    parser.add_argument("--eval_model_name", type=str, default=None, help="Model name used for answer extraction. Auto-loaded from config.json if not specified.")
    parser.add_argument("--offset", type=int, default=0, help="Dataset offset used during search")
    parser.add_argument("--limit", type=int, default=None, help="Dataset limit used during search")
    parser.add_argument(
        "--include",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s)/package(s) to include for custom component registration. Auto-loaded from config if not specified."
    )
    parser.add_argument("--input-price", type=float, default=None, help="Price per 1M input tokens (for cost estimation)")
    parser.add_argument("--output-price", type=float, default=None, help="Price per 1M output tokens (for cost estimation)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output to console (default: progress bar + summary only)")
    parser.add_argument("--llm-eval", choices=["binary", "f1", "none"], default="binary",
                        help="LLM fallback evaluator mode: 'binary' (default, yes/no), 'f1' (float F1 score), 'none' (disable LLM fallback)")
    
    args = parser.parse_args()
    
    # Validate result_dir exists
    if not Path(args.result_dir).exists():
        print(f"Error: Directory not found: {args.result_dir}", file=sys.stderr)
        return 1
    
    # Load config from result_dir if available (for auto-loading import_modules)
    config = load_config_from_result_dir(args.result_dir, config_filename="config.json")
    
    # Determine import_modules: CLI args override config (top-level field)
    import_modules = args.import_modules
    if not import_modules and config.get("import_modules"):
        import_modules = config["import_modules"]
        print(f"Auto-loaded import_modules from config: {import_modules}")
    
    # Load dataset_kwargs from config (top-level field)
    dataset_kwargs = config.get("dataset_kwargs", {})
    if dataset_kwargs:
        print(f"Auto-loaded dataset_kwargs from config: {dataset_kwargs}")
    
    # Resolve dataset_name: CLI > config > error
    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = config.get("dataset") or config.get("benchmark")
        if dataset_name:
            print(f"Auto-loaded dataset_name from config: {dataset_name}")
        else:
            print("Error: --dataset_name not specified and not found in config.json", file=sys.stderr)
            return 1

    # Resolve eval_model_name: CLI > config (eval_model_name or policy_model_name) > error
    eval_model_name = args.eval_model_name
    if not eval_model_name:
        eval_model_name = config.get("eval_model_name") or config.get("policy_model_name")
        if eval_model_name:
            print(f"Auto-loaded eval_model_name from config: {eval_model_name}")
        else:
            print("Error: --eval_model_name not specified and not found in config.json", file=sys.stderr)
            return 1
    
    # Import custom modules to trigger registration
    try:
        import_custom_modules(import_modules)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    try:
        evaluate_from_checkpoints(
            result_dir=args.result_dir,
            dataset_name=dataset_name,
            eval_model_name=eval_model_name,
            offset=args.offset,
            limit=args.limit,
            dataset_kwargs=dataset_kwargs,
            input_price_per_m=args.input_price,
            output_price_per_m=args.output_price,
            verbose=args.verbose,
            llm_eval_mode=args.llm_eval,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    
    eval_log = Path(args.result_dir) / "eval.log"
    if eval_log.exists():
        print(f"Evaluation log saved to: {eval_log}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
