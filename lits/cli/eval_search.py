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

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.common import extract_answers_from_terminal_nodes
from lits.benchmarks.registry import load_dataset, TOOL_USE_DATASETS
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules, load_config_from_result_dir
from lits.components.utils import get_fn_retrieve_answer
from lits.lm import get_lm
from lits.eval.inference_report import generate_report
from lits.log import setup_logging
from lits.structures import ToolUseState, ToolUseStep  # Import to register types
from lits.structures.env_grounded import EnvState, EnvStep  # Import to register types

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


def evaluate_from_checkpoints(
    result_dir: str,
    dataset_name: str,
    eval_model_name: str,
    offset: int = 0,
    limit: int = None,
    dataset_kwargs: dict = None
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
    """
    result_dir = Path(result_dir)
    
    # Setup logging
    eval_logger = setup_logging(
        run_id="eval",
        result_dir=result_dir,
        add_console_handler=True,
        verbose=True
    )
    
    # Determine task type using registry
    is_env_grounded = is_env_grounded_task(dataset_name)
    is_tool_use = dataset_name in TOOL_USE_DATASETS
    
    # Load dataset for ground truths (not needed for env_grounded tasks)
    if is_env_grounded:
        # For env_grounded tasks, get goal_check from registry
        goal_check = get_goal_check_from_registry(dataset_name)
        full_dataset = None
        eval_logger.info(f"Evaluating {dataset_name} results (goal checking via registry)")
    elif is_tool_use:
        # Load only the examples, not the tools (no database connection needed)
        from lits_benchmark import load_dataset_examples
        full_dataset = load_dataset_examples(dataset_name)
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
    
    # Find all terminal node files in terminal_nodes subdirectory
    terminal_nodes_dir = result_dir / "terminal_nodes"
    if not terminal_nodes_dir.exists():
        eval_logger.error(f"Terminal nodes directory not found: {terminal_nodes_dir}")
        return
    
    terminal_node_files = sorted(terminal_nodes_dir.glob("terminal_nodes_*.json"))
    
    if not terminal_node_files:
        eval_logger.error(f"No terminal node files found in {result_dir}")
        return
    
    eval_logger.info(f"Found {len(terminal_node_files)} terminal node files")
    
    # Filter terminal node files by offset and limit based on query_idx
    # The query_idx in terminal node files corresponds to the original dataset index
    end_idx = offset + limit if limit is not None else None
    
    def should_include_file(filepath):
        """Check if file's query_idx falls within [offset, offset+limit) range."""
        # Extract query_idx from filename: terminal_nodes_{query_idx}.json
        filename = filepath.stem  # terminal_nodes_0, terminal_nodes_10, etc.
        try:
            idx = int(filename.split('_')[-1])
            if idx < offset:
                return False
            if end_idx is not None and idx >= end_idx:
                return False
            return True
        except ValueError:
            return True  # Include files with non-standard names
    
    filtered_files = [f for f in terminal_node_files if should_include_file(f)]
    eval_logger.info(f"Processing {len(filtered_files)} files (offset={offset}, limit={limit})")
    
    # Process each file and extract answers
    predictions = []
    ground_truths = []
    soft_scores = []  # For env_grounded tasks: word-level accuracy scores
    
    for filepath in filtered_files:
        try:
            # Load terminal nodes
            data = load_terminal_nodes_from_file(filepath)
            terminal_nodes = data['terminal_nodes']
            query = data['query']
            query_idx = data['query_idx']
            
            # Get ground truth based on task type
            if is_env_grounded:
                # For BlocksWorld, ground truth is always "correct" if goal is reached
                ground_truth = "correct"
            else:
                ground_truth = str(full_dataset[query_idx]['answer'])
            
            # Extract answer
            if is_env_grounded:
                # For environment-grounded, check if any terminal node reached the goal
                if terminal_nodes:
                    # Sort by cumulative reward (descending) to get best node first
                    def get_best_reward(node):
                        if hasattr(node, 'cum_rewards') and node.cum_rewards:
                            return max(node.cum_rewards) if isinstance(node.cum_rewards, list) else node.cum_rewards
                        return -float('inf')
                    sorted_nodes = sorted(terminal_nodes, key=get_best_reward, reverse=True)
                    # Check the best terminal node (highest cumulative reward)
                    best_node = sorted_nodes[0]
                    if hasattr(best_node, 'state') and best_node.state:
                        state = best_node.state
                        if isinstance(state, EnvState):
                            final_env_state = state.env_state
                            is_reached, score = goal_check(query, final_env_state)
                            answer_pred = "correct" if is_reached else "incorrect"
                            soft_scores.append(score)
                        else:
                            answer_pred = "incorrect"
                            soft_scores.append(0.0)
                    else:
                        answer_pred = "incorrect"
                        soft_scores.append(0.0)
                else:
                    answer_pred = "incorrect"
                    soft_scores.append(0.0)
                    eval_logger.warning(f"Query {query_idx}: No terminal nodes found")
            elif is_tool_use:
                # Direct extraction from node
                answer_pred = retrieve_answer(terminal_nodes[0], query) if terminal_nodes else ""
                ground_truth = "Option " + ground_truth if "mapeval" in dataset_name else ground_truth
            else:
                # Use voting for QA tasks
                vote_answers, answer_reward_d, best_node, trace_of_nodes = extract_answers_from_terminal_nodes(
                    terminal_nodes_collected=terminal_nodes,
                    retrieve_answer=retrieve_answer,
                    query=query
                )
                # Get prediction
                if len(vote_answers) > 0:
                    answer_pred = max(vote_answers, key=lambda answer: vote_answers[answer])
                else:
                    answer_pred = ''
            
            predictions.append(answer_pred)
            ground_truths.append(ground_truth)
            
            # Use info level for first few examples to help debug
            if len(predictions) < 5:
                eval_logger.info(f"Query {query_idx}: Pred='{answer_pred}', Truth='{ground_truth}'")
            else:
                eval_logger.debug(f"Query {query_idx}: Pred={answer_pred}, Truth={ground_truth}")
            
        except Exception as e:
            eval_logger.error(f"Error processing {filepath}: {e}")
            eval_logger.error(traceback.format_exc())
            continue
    
    # Calculate accuracy
    # For QA tasks (math), use eval_output for proper number comparison
    # For env_grounded tasks, use exact string match ("correct"/"incorrect")
    from lits.components.utils import eval_output
    
    correct_count = 0
    eval_logger.info("=" * 40)
    eval_logger.info("Detailed comparison (first 10):")
    for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        if is_env_grounded:
            # Exact string match for env_grounded tasks
            correct = (pred == truth)
        else:
            # Use eval_output for number comparison in QA tasks
            try:
                correct = eval_output(truth, pred, type="number_exact")
            except (AssertionError, ValueError) as e:
                # Fallback to exact match if eval_output fails
                correct = (pred == truth)
                if i < 10:
                    eval_logger.info(f"  [{i}] eval_output failed: {e}")
        if i < 10:
            eval_logger.info(f"  [{i}] Pred='{pred}' (type={type(pred).__name__}), Truth='{truth}' (type={type(truth).__name__}), Correct={correct}")
        if correct:
            correct_count += 1
    eval_logger.info("=" * 40)
    
    total_count = len(predictions)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Calculate soft accuracy for env_grounded tasks
    soft_accuracy = None
    if is_env_grounded and soft_scores:
        soft_accuracy = sum(soft_scores) / len(soft_scores)
    
    eval_logger.info("=" * 80)
    eval_logger.info(f"Evaluation Results for {result_dir.name}")
    eval_logger.info(f"Dataset: {dataset_name}")
    eval_logger.info(f"Total Examples: {total_count}")
    eval_logger.info(f"Correct: {correct_count}")
    eval_logger.info(f"Accuracy (exact match): {accuracy:.4f} ({accuracy*100:.2f}%)")
    if soft_accuracy is not None:
        eval_logger.info(f"Soft Accuracy (word-level): {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
    eval_logger.info("=" * 80)
    
    # Log token usage metrics from existing inference log
    eval_logger.info("Token Usage Metrics:")
    report = generate_report(str(result_dir))
    eval_logger.info(report)
    
    # Save evaluation results
    eval_results = {
        'dataset_name': dataset_name,
        'result_dir': str(result_dir),
        'total_examples': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'predictions': predictions,
        'ground_truths': ground_truths
    }
    
    # Add soft accuracy for env_grounded tasks
    if soft_accuracy is not None:
        eval_results['soft_accuracy'] = soft_accuracy
        eval_results['soft_scores'] = soft_scores
    
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
  # Evaluate results (auto-loads import_modules from config)
  lits-eval --result_dir claude35v1_results/blocksworld_rap/run_0.2.10 \\
      --dataset_name blocksworld --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
  
  # Evaluate custom benchmark (import module to register Transition)
  lits-eval --result_dir results/robot_arm_rap/run_0.2.10 \\
      --dataset_name robot_arm --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \\
      --import my_project.robot_arm
"""
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing terminal_nodes/ subdirectory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., gsm8k, math500, blocksworld)")
    parser.add_argument("--eval_model_name", type=str, required=True, help="Model name used for answer extraction")
    parser.add_argument("--offset", type=int, default=0, help="Dataset offset used during search")
    parser.add_argument("--limit", type=int, default=None, help="Dataset limit used during search")
    parser.add_argument(
        "--import",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s) to import for custom component registration. Auto-loaded from config if not specified."
    )
    
    args = parser.parse_args()
    
    # Validate result_dir exists
    if not Path(args.result_dir).exists():
        print(f"Error: Directory not found: {args.result_dir}", file=sys.stderr)
        return 1
    
    # Load config from result_dir if available (for auto-loading import_modules)
    config = load_config_from_result_dir(args.result_dir, config_filename="config.json")
    
    # Determine import_modules: CLI args override config
    import_modules = args.import_modules
    if not import_modules and config.get("import_modules"):
        import_modules = config["import_modules"]
        print(f"Auto-loaded import_modules from config: {import_modules}")
    
    # Load dataset_kwargs from config
    dataset_kwargs = config.get("dataset_kwargs", {})
    if dataset_kwargs:
        print(f"Auto-loaded dataset_kwargs from config: {dataset_kwargs}")
    
    # Import custom modules to trigger registration
    try:
        import_custom_modules(import_modules)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    try:
        evaluate_from_checkpoints(
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            eval_model_name=args.eval_model_name,
            offset=args.offset,
            limit=args.limit,
            dataset_kwargs=dataset_kwargs
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
