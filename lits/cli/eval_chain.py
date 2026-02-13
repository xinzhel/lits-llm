"""
lits-eval-chain: Evaluate environment-grounded chain agent results.

Evaluates chain agent execution results by checking if the final state
satisfies the goal conditions. Uses the goal_check function from the
registered Transition class via ComponentRegistry.

Usage:
    lits-eval-chain --result_dir results/blocksworld_chain/run_0.2.10
    lits-eval-chain --benchmark crosswords --result_dir results/crosswords_chain/run_0.2.10
    lits-eval-chain --list --include lits_benchmark.blocksworld
    lits-eval-chain --help

Auto-loads import_modules and benchmark from config.json in result_dir.
See docs/cli/search.md for full CLI documentation.
"""

import sys
import os
import json
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from lits.structures.env_grounded import EnvState
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules, load_config_from_result_dir
from lits.eval.inference_report import generate_report
from lits.log import setup_logging

logger = logging.getLogger(__name__)


def get_goal_check(benchmark_name: str) -> Callable:
    """Get the goal_check function for a benchmark from the registry.
    
    Args:
        benchmark_name: Name of the registered benchmark (e.g., 'blocksworld', 'crosswords')
        
    Returns:
        The goal_check static method from the registered Transition class
        
    Raises:
        KeyError: If the benchmark is not registered
        AttributeError: If the Transition class doesn't have a goal_check method
    """
    try:
        TransitionCls = ComponentRegistry.get_transition(benchmark_name)
    except KeyError as e:
        available = ComponentRegistry.list_by_task_type("env_grounded")
        raise KeyError(
            f"Benchmark '{benchmark_name}' not found in registry. "
            f"Available env_grounded benchmarks: {available}. "
            f"Did you forget to import the module containing @register_transition('{benchmark_name}')?"
        ) from e
    
    if not hasattr(TransitionCls, 'goal_check'):
        raise AttributeError(
            f"Transition class '{TransitionCls.__name__}' for benchmark '{benchmark_name}' "
            f"does not have a 'goal_check' static method. "
            f"EnvGroundedTransition subclasses must implement goal_check()."
        )
    
    return TransitionCls.goal_check


def evaluate_results(benchmark_name: str, result_dir: str) -> Dict[str, Any]:
    """
    Evaluate chain agent results from checkpoint files.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'blocksworld', 'crosswords')
        result_dir: Run directory containing checkpoints/ subdirectory and config
        
    Returns:
        Dictionary with evaluation results including accuracy metrics
    """
    result_path = Path(result_dir)
    checkpoint_dir = result_path / "checkpoints"
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return {"error": f"Directory not found: {checkpoint_dir}"}

    # Get goal_check function from registry
    try:
        goal_check = get_goal_check(benchmark_name)
    except (KeyError, AttributeError) as e:
        logger.error(str(e))
        return {"error": str(e)}

    # Setup logging with file output
    eval_logger = setup_logging(
        run_id="eval",
        result_dir=result_path,
        add_console_handler=True,
        verbose=True
    )

    json_files = sorted(checkpoint_dir.glob("*.json"))
    if not json_files:
        eval_logger.warning(f"No JSON files found in {checkpoint_dir}")
        return {"error": "No checkpoint files found"}

    total = 0
    correct = 0
    predictions = []
    soft_scores = []  # For word-level accuracy (crosswords)
    
    eval_logger.info(f"Evaluating benchmark: {benchmark_name}")
    eval_logger.info(f"Found {len(json_files)} checkpoint files in {checkpoint_dir}")

    for file_path in tqdm(json_files, desc=f"Evaluating {benchmark_name}"):
        try:
            # Load the checkpoint - EnvState.load returns (query, state)
            query, state = EnvState.load(str(file_path))
            
            # Check if goal is reached - goal_check returns (is_reached, score)
            is_reached, score = goal_check(query, state.env_state)
            
            total += 1
            if is_reached:
                correct += 1
            
            predictions.append({
                "file": file_path.name,
                "is_reached": is_reached,
                "score": score
            })
            soft_scores.append(score)
                
            eval_logger.debug(f"File: {file_path.name}, Reached: {is_reached}, Score: {score}")
            
        except Exception as e:
            eval_logger.error(f"Error processing {file_path}: {e}")
            eval_logger.error(traceback.format_exc())

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    soft_accuracy = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
    
    eval_logger.info("=" * 80)
    eval_logger.info(f"Evaluation Results for {benchmark_name} - {result_path.name}")
    eval_logger.info(f"Total Examples: {total}")
    eval_logger.info(f"Correct: {correct}")
    eval_logger.info(f"Accuracy (exact match): {accuracy:.2%}")
    eval_logger.info(f"Soft Accuracy (word-level): {soft_accuracy:.2%}")
    eval_logger.info("=" * 80)
    
    # Log token usage metrics
    eval_logger.info("Token Usage Metrics:")
    report = generate_report(str(result_path))
    eval_logger.info(report)
    
    # Build results dictionary
    eval_results = {
        "benchmark": benchmark_name,
        "result_dir": str(result_dir),
        "total_examples": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "soft_accuracy": soft_accuracy,
        "soft_scores": soft_scores,
        "predictions": predictions
    }
    
    # Save evaluation results
    eval_results_file = result_path / "eval_results.json"
    with open(eval_results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    eval_logger.info(f"Evaluation results saved to {eval_results_file}")
    
    return eval_results


def list_available_benchmarks() -> None:
    """Print available env_grounded benchmarks."""
    benchmarks = ComponentRegistry.list_by_task_type("env_grounded")
    print("Available env_grounded benchmarks:")
    for name in benchmarks:
        TransitionCls = ComponentRegistry.get_transition(name)
        has_goal_check = hasattr(TransitionCls, 'goal_check')
        status = "✓" if has_goal_check else "✗ (missing goal_check)"
        print(f"  - {name}: {TransitionCls.__name__} {status}")


def main() -> int:
    """Entry point for lits-eval-chain command.
    
    Evaluates chain agent results from checkpoint files. Auto-loads
    import_modules and benchmark from config.json in result_dir.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Evaluate environment-grounded chain agent results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate results (auto-loads config from result_dir)
  lits-eval-chain --result_dir claude35v1_results/blocksworld_chain/run_0.2.10
  
  # Evaluate with explicit benchmark (overrides config)
  lits-eval-chain --benchmark blocksworld --result_dir results/blocksworld_chain/run_0.2.10
  
  # Evaluate custom benchmark (import module to register Transition)
  lits-eval-chain --benchmark robot_arm --include my_project.robot_arm --result_dir results/robot_arm/run_0.2.10
  
  # List available benchmarks (including custom ones)
  lits-eval-chain --include my_project.robot_arm --list
"""
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Name of the benchmark (e.g., 'blocksworld', 'crosswords'). Auto-loaded from config if not specified."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Path to run directory containing checkpoints/ and config (e.g., results/blocksworld_chain/run_0.2.10)"
    )
    parser.add_argument(
        "--include",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s)/package(s) to include for custom Transition registration. Auto-loaded from config if not specified."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available env_grounded benchmarks and exit"
    )
    
    args = parser.parse_args()
    
    # Load config from result_dir if available (for auto-loading import_modules and benchmark)
    config = {}
    if args.result_dir:
        if not Path(args.result_dir).exists():
            print(f"Error: Directory not found: {args.result_dir}", file=sys.stderr)
            return 1
        config = load_config_from_result_dir(args.result_dir, config_filename="config.json")
    
    # Determine import_modules: CLI args override config
    import_modules = args.import_modules
    if not import_modules and config.get("import_modules"):
        import_modules = config["import_modules"]
        print(f"Auto-loaded import_modules from config: {import_modules}")
    
    # Import custom modules to trigger registration
    try:
        import_custom_modules(import_modules)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.list:
        list_available_benchmarks()
        return 0
    
    if args.result_dir is None:
        parser.error("--result_dir is required when not using --list")
    
    # Determine benchmark: CLI args override config, default to 'blocksworld'
    benchmark = args.benchmark
    if not benchmark:
        benchmark = config.get("benchmark", "blocksworld")
        if config.get("benchmark"):
            print(f"Auto-loaded benchmark from config: {benchmark}")
    
    try:
        results = evaluate_results(benchmark, args.result_dir)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    
    if "error" in results:
        print(f"Error: {results['error']}", file=sys.stderr)
        return 1
    
    print(f"\n{benchmark} Accuracy (exact match): {results['accuracy']:.2%} ({results['correct_count']}/{results['total_examples']})")
    print(f"{benchmark} Soft Accuracy (word-level): {results['soft_accuracy']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
