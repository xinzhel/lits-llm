"""
lits-search: Tree search experiment CLI entry point.

Usage:
    lits-search --dataset math500 --search_framework rest --var limit=10
    lits-search --dataset blocksworld --include lits_benchmark.blocksworld --search-arg n_iters=10
    lits-search --help
    lits-search --help-config

Two-Stage Workflow:
1. Run lits-search to perform tree search and save terminal nodes
2. Run lits-eval to evaluate results from checkpoint files

See docs/cli/search.md for full CLI documentation.
"""

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from huggingface_hub import login

from lits.lm.base import InferenceLogger, VALID_ROLES_PREFIX, log_final_metrics
from lits.eval import TreeToJsonl, ResultDictToJsonl, _slice_dataset
from lits.eval.llm_call_logger import create_llm_call_logger, load_llm_calls, print_diversity_report
from lits.agents import AgentRegistry
from lits.log import setup_logging
from lits.utils.sys_utils import is_running_in_jupyter
from lits.benchmarks.registry import infer_task_type, load_dataset, load_resource, has_resource
from lits.config import ExperimentConfig
from lits.lm import configure_hf_model_logging, setup_inference_logging, load_models
from lits.components.factory import create_components, create_bn_evaluator
from lits.components.base import Transition, Policy, RewardModel
from lits.components.bn_evaluator import BNEvaluator
from lits.memory.manager import LiTSMemoryManager
from lits.memory.backends import Mem0MemoryBackend
from lits.memory.config import LiTSMemoryConfig
from lits.registry import import_custom_modules
from lits.cli import (
    parse_experiment_args, apply_config_overrides, parse_dataset_kwargs,
    parse_script_vars, parse_search_args, parse_component_args, print_config_help,
)

logger = logging.getLogger(__name__)


def setup_result_savers(search_algorithm: str, result_dir: str, override: bool):
    """Setup result savers based on search algorithm.

    Both MCTS and BFS use TreeToJsonl to save paths consistently.
    MCTS additionally saves unselected simulation paths.
    If override is True, cleans up stale files from previous runs.
    """
    # Clean up stale files if override is specified
    if override:
        result_path = Path(result_dir)

        # Clean checkpoints directory
        checkpoints_dir = result_path / "checkpoints"
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*"):
                f.unlink()

        # Clean terminal_nodes directory
        terminal_nodes_dir = result_path / "terminal_nodes"
        if terminal_nodes_dir.exists():
            for f in terminal_nodes_dir.glob("*"):
                f.unlink()

        # Clean treetojsonl*.jsonl files
        for f in result_path.glob("treetojsonl*.jsonl"):
            f.unlink()

    result_saver = TreeToJsonl(run_id='', root_dir=result_dir, override=override)

    # MCTS saves unselected simulation paths as a secondary output
    result_saver_unselected = None
    if search_algorithm == "mcts":
        result_saver_unselected = TreeToJsonl(
            run_id='unselected_simulate', root_dir=result_dir, override=override
        )

    return result_saver, result_saver_unselected


def setup_memory_manager(config: ExperimentConfig, run_logger):
    """Initialize memory manager if memory is enabled in config.

    Args:
        config: ExperimentConfig with enable_memory and memory_config fields
        run_logger: Logger instance for debug output

    Returns:
        LiTSMemoryManager instance if memory is enabled, None otherwise

    Raises:
        ValueError: If memory is enabled but configuration is invalid

    Example memory_config structure::

        {
            "llm": {
                "provider": "aws_bedrock",
                "config": {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0"}
            },
            "embedder": {
                "provider": "aws_bedrock",
                "config": {"model": "amazon.titan-embed-text-v2:0"}
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": "./qdrant_local"}
            },
            "lits_mem": {
                "similarity_threshold": 0.35,
                "max_retrieved_trajectories": 3
            }
        }
    """
    if not config.enable_memory:
        return None

    try:
        from mem0 import Memory

        mem0_config = config.memory_config or {}

        if mem0_config:
            if not isinstance(mem0_config, dict):
                raise ValueError(
                    f"memory_config must be a dictionary, got {type(mem0_config).__name__}"
                )
            recommended_sections = ["llm", "embedder", "vector_store"]
            missing_sections = [s for s in recommended_sections if s not in mem0_config]
            if missing_sections:
                run_logger.warning(
                    f"memory_config is missing recommended sections: {missing_sections}. "
                    "mem0 will use default providers which may require additional setup."
                )
        else:
            run_logger.warning(
                "Memory enabled but no memory_config provided. "
                "Using default mem0 configuration which may require OpenAI API key."
            )

        memory = Memory.from_config(mem0_config) if mem0_config else Memory()
        backend = Mem0MemoryBackend(memory)

        lits_mem_config_dict = mem0_config.get("lits_mem", {})
        lits_mem_config = LiTSMemoryConfig(**lits_mem_config_dict) if lits_mem_config_dict else LiTSMemoryConfig()

        memory_manager = LiTSMemoryManager(backend=backend, config=lits_mem_config)
        run_logger.info("Memory manager initialized successfully")
        return memory_manager

    except ImportError as e:
        raise ValueError(
            f"Memory is enabled but mem0 is not installed. "
            f"Install with: pip install mem0ai. Error: {e}"
        )
    except TypeError as e:
        raise ValueError(
            f"Invalid memory configuration. Check that all config values are valid. Error: {e}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to initialize memory manager with provided configuration. Error: {e}"
        )


def save_terminal_nodes(algo_output, query_or_goals, query_idx, result_dir, run_logger):
    """Save terminal nodes separately for post-search evaluation.

    Terminal nodes are saved to result_dir/terminal_nodes/terminal_nodes_{query_idx}.json
    for use by lits-eval to perform answer extraction and voting without needing
    to reconstruct the full search tree.

    Args:
        algo_output: MCTSResult or BFSResult containing terminal_nodes_collected
        query_or_goals: Original query/question
        query_idx: Query index
        result_dir: Directory to save results
        run_logger: Logger instance
    """
    terminal_nodes_data = {
        'terminal_nodes': [node.to_dict() for node in algo_output.terminal_nodes_collected],
        'query': query_or_goals,
        'query_idx': query_idx
    }

    terminal_nodes_dir = Path(result_dir) / "terminal_nodes"
    terminal_nodes_dir.mkdir(parents=True, exist_ok=True)
    terminal_nodes_file = terminal_nodes_dir / f"terminal_nodes_{query_idx}.json"

    with open(terminal_nodes_file, 'w') as f:
        json.dump(terminal_nodes_data, f, indent=2)

    run_logger.debug(f"Saved {len(algo_output.terminal_nodes_collected)} terminal nodes to {terminal_nodes_file}")


def run_tree_search(
    query_or_goals: str, query_idx: int, search_config, world_model: Transition, policy: Policy,
    evaluator: RewardModel, bn_evaluator: BNEvaluator, result_saver: TreeToJsonl, result_dir: str,
    result_saver_unselected=None, init_state_kwargs: dict = None,
    memory_manager: Optional[LiTSMemoryManager] = None,
    search_algorithm: str = "mcts",
    run_logger=None,
):
    """Run tree search and save results to checkpoint files.

    This function only performs tree search and saves results. Answer extraction
    and evaluation should be done separately as post-processing using the saved
    checkpoint files.

    Args:
        query_or_goals: Question to solve
        query_idx: Example index for logging/tracking
        search_config: Search configuration (MCTSConfig or BFSConfig)
        world_model: World model
        policy: Policy
        evaluator: Evaluator/reward model
        bn_evaluator: BN evaluator (optional)
        result_saver: Result saver (TreeToJsonl for both MCTS and BFS)
        result_dir: Directory to save results
        result_saver_unselected: Unselected paths saver (MCTS only)
        init_state_kwargs: Optional kwargs passed to world_model.init_state().
                           For env_grounded tasks, should include 'init_state_str'.
        memory_manager: Optional LiTSMemoryManager for cross-trajectory memory.
                       If provided, enables memory retrieval before expansion and
                       recording of actions after expansion.
        search_algorithm: Search algorithm to use ("mcts", "bfs", or custom registered algorithm)
        run_logger: Logger instance (falls back to module logger if None)

    Returns:
        None (all results saved to checkpoint files)
    """
    _logger = run_logger or logger

    # Look up search function from registry (unified invocation)
    search_fn = AgentRegistry.get_search(search_algorithm)

    search_kwargs = {
        "init_state_kwargs": init_state_kwargs,
        "checkpoint_dir": result_dir,
        "memory_manager": memory_manager,
    }

    algo_output = search_fn(
        query_or_goals, query_idx, search_config, world_model,
        policy, evaluator, bn_evaluator, **search_kwargs
    )

    # Save unselected paths if available (MCTS-specific attribute)
    if result_saver_unselected and hasattr(algo_output, 'unselected_terminal_paths_during_simulate'):
        result_saver_unselected.append_result(algo_output.unselected_terminal_paths_during_simulate)

    # Convert to paths for saving — all results support to_paths()
    paths = algo_output.to_paths()
    result_saver.append_result(paths)

    # Save terminal nodes separately for post-search evaluation
    save_terminal_nodes(algo_output, query_or_goals, query_idx, result_dir, _logger)


def main() -> int:
    """Entry point for lits-search command.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    # Default config — CLI flags override these values
    config = ExperimentConfig(
        dataset="math500",
        policy_model_name="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        eval_model_name="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        search_framework="rest",
        search_algorithm="mcts",
        search_args={"n_actions": 3},
        component_args={},
        offset=0,
        limit=None,
        eval_idx=[],
        override_log_result=False,
    )

    # Parse CLI arguments
    cli_args = parse_experiment_args(description="Run tree search experiments (MCTS, BFS, RAP)")

    # Handle --help-config flag
    if cli_args.help_config:
        print_config_help()
        return 0

    config = apply_config_overrides(config, cli_args)

    # Apply explicit CLI flags (take precedence over --cfg)
    if cli_args.dataset:
        config.dataset = cli_args.dataset
    if cli_args.search_framework:
        config.search_framework = cli_args.search_framework
    if cli_args.policy:
        config.policy = cli_args.policy
    if cli_args.transition:
        config.transition = cli_args.transition
    if cli_args.reward:
        config.reward = cli_args.reward

    # Apply model flags (take precedence over --cfg)
    if cli_args.policy_model:
        config.policy_model_name = cli_args.policy_model
    if cli_args.eval_model:
        config.eval_model_name = cli_args.eval_model
    elif cli_args.policy_model:
        config.eval_model_name = cli_args.policy_model
    if cli_args.transition_model:
        config.search_args["transition_model_name"] = cli_args.transition_model
    if cli_args.bn_model:
        config.search_args["bn_model_name"] = cli_args.bn_model

    # Map --override flag
    if cli_args.override:
        config.override_log_result = True

    # Map --output-dir flag
    if cli_args.output_dir:
        config.output_dir = cli_args.output_dir

    # Parse script-level variables
    script_vars = parse_script_vars(cli_args, {'offset': config.offset, 'limit': config.limit})
    config.offset = script_vars['offset']
    config.limit = script_vars['limit']

    # Import custom modules to trigger registration
    if cli_args.import_modules:
        import_custom_modules(cli_args.import_modules)
        print(f"Imported custom modules: {cli_args.import_modules}")

    # Apply --search-arg and --component-arg from CLI
    if cli_args.search_args:
        config.search_args.update(parse_search_args(cli_args))
    if cli_args.component_args:
        config.component_args.update(parse_component_args(cli_args))

    # Infer task type and load dataset early (needed for --dry-run before any side effects)
    task_type = infer_task_type(config.dataset)
    task_name = config.dataset
    dataset_kwargs = parse_dataset_kwargs(cli_args)

    # Load tool use spec if applicable (skip DB connection in dry-run mode)
    if has_resource(config.dataset) and not cli_args.dry_run:
        if os.path.exists("mapeval/.env"):
            load_dotenv("mapeval/.env")
        tool_use_spec = load_resource(config.dataset)
        is_tool_use = True
    elif has_resource(config.dataset):
        tool_use_spec = None
        is_tool_use = True
    else:
        tool_use_spec = None
        is_tool_use = False

    # Load dataset
    if task_type == "tool_use":
        full_dataset = load_dataset(config.dataset, **dataset_kwargs)
    elif task_type == "language_grounded":
        full_dataset = load_dataset(config.dataset, **dataset_kwargs)
    elif task_type == "env_grounded":
        if config.dataset == "blocksworld" and not dataset_kwargs:
            dataset_kwargs = {
                'config_file': "blocksworld/bw_data_bw_config.yaml",
                'domain_file': "blocksworld/bw_data_generated_domain.pddl",
                'data_file': 'blocksworld/bw_data_step_6.json'
            }
        # Validate dataset file paths exist before loading
        for key, path in dataset_kwargs.items():
            if key.endswith('_file') and not os.path.exists(path):
                print(f"Error: dataset file not found: {path} (from --dataset-arg {key}={path})")
                print(f"Current working directory: {os.getcwd()}")
                print("Hint: run from the directory containing the data files, or use --dataset-arg to specify paths.")
                return 1
        full_dataset = load_dataset(config.dataset, **dataset_kwargs)

    # Dry-run mode: print config + first element, then exit with no side effects
    if cli_args.dry_run:
        from lits.components.factory import resolve_component_names

        names = resolve_component_names(task_type, config)
        print(f"\n=== Dry Run Mode ===")
        print(f"Dataset: {config.dataset}")
        print(f"Task type: {task_type}")
        print(f"Search algorithm: {config.search_algorithm}")
        print(f"Components: policy={names['policy']}, transition={names['transition']}, reward={names['reward']}")
        print(f"Dataset size: {len(full_dataset)}")
        print(f"\nFirst element:")
        print(json.dumps(full_dataset[0], indent=2, default=str))
        return 0

    # --- Everything below only runs for real execution ---

    # Setup directories and save config
    configure_hf_model_logging()
    run_id, result_dir = config.setup_directories(is_running_in_jupyter())

    # Store import_modules and dataset_kwargs in config for reproducibility
    if cli_args.import_modules:
        config.import_modules = cli_args.import_modules
    if dataset_kwargs:
        config.dataset_kwargs = dataset_kwargs
    config.save_config(result_dir)

    # Create SearchConfig for the search algorithm (pure search params only)
    search_config = config.create_search_config()
    if cli_args.import_modules:
        search_config.import_modules = cli_args.import_modules
    if dataset_kwargs:
        search_config.dataset_kwargs = dataset_kwargs

    # Login to Hugging Face (only if HF_TOKEN is set)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load models
    search_args = config.get_search_args()
    base_model, eval_model, terminal_model, terminate_ORM = load_models(
        policy_model_name=config.policy_model_name,
        eval_model_name=config.eval_model_name,
        search_framework=config.search_framework or "rest",
        task_type=task_type,
        device=search_args.get("device", "cuda"),
        max_length=search_args.get("max_length", 32768),
        enable_think_policy=search_config.enable_think_policy,
        enable_think_eval=search_config.enable_think_eval,
        enable_think_terminal_gen=search_config.enable_think_terminal_gen,
        transition_model_name=search_args.get("transition_model_name"),
        terminate_ORM_name=search_args.get("terminate_ORM_name"),
        terminate_constraints=search_args.get("terminate_constraints", ["binary_sampling"]),
        is_tool_use=is_tool_use,
        model_verbose=config.model_verbose
    )

    # Setup logging
    run_logger = setup_logging(
        "execution", result_dir,
        add_console_handler=True, verbose=config.verbose,
        override=config.override_log_result
    )
    inference_logger = setup_inference_logging(
        base_model, eval_model, terminal_model, terminate_ORM,
        result_dir, config.override_log_result
    )

    # Create components
    world_model, policy, evaluator = create_components(
        task_type=task_type,
        task_name=task_name,
        base_model=base_model,
        eval_base_model=eval_model,
        terminal_model=terminal_model,
        tool_use_spec=tool_use_spec,
        config=config
    )

    # Set save directory for ToolUsePRM rollouts if applicable
    if hasattr(evaluator, 'save_rollouts_dir'):
        rollouts_dir = Path(result_dir) / "tool_use_prm_rollouts"
        evaluator.save_rollouts_dir = str(rollouts_dir)
        run_logger.info(f"ToolUsePRM will save rollouts to: {rollouts_dir}")

    # Create BN evaluator if needed
    component_args = config.get_component_args()
    bn_evaluator = create_bn_evaluator(
        base_model=base_model,
        search_args=search_args,
        component_args=component_args,
        search_framework=config.search_framework,
        device=search_args.get("device", "cuda"),
        enable_think_policy=search_config.enable_think_policy,
        model_verbose=config.model_verbose,
        inference_logger=inference_logger,
        task_type=task_type
    )

    # Setup LLM call logger for diversity analysis (env_grounded tasks only)
    # Skip for language_grounded tasks - duplicate detection is less meaningful for free-form text
    llm_calls_path = None
    if task_type != "language_grounded":
        llm_calls_path = f"{result_dir}/llm_calls.jsonl"
        log_llm_call = create_llm_call_logger(llm_calls_path)
        policy.set_llm_call_fn(log_llm_call)

    # Setup result savers
    result_saver, result_saver_unselected = setup_result_savers(
        config.search_algorithm, result_dir, config.override_log_result
    )

    # Initialize memory manager if enabled
    memory_manager = setup_memory_manager(config, run_logger)

    # Slice dataset
    if config.eval_idx:
        full_dataset = [full_dataset[i] for i in config.eval_idx]
    else:
        full_dataset = _slice_dataset(full_dataset, offset=config.offset, limit=config.limit)

    # Run experiments
    begin_time = time.time()

    for query_idx, example in tqdm(enumerate(full_dataset, start=config.offset)):
        if config.eval_idx and query_idx not in config.eval_idx:
            run_logger.debug(f"Skipping example {query_idx}")
            continue

        if task_type == "env_grounded":
            query_or_goals = example.get("query_or_goals", example.get("question", ""))
        else:
            query_or_goals = example["question"]

        run_tree_search(
            query_or_goals=query_or_goals,
            query_idx=query_idx,
            search_config=search_config,
            world_model=world_model,
            policy=policy,
            evaluator=evaluator,
            bn_evaluator=bn_evaluator,
            result_saver=result_saver,
            result_dir=result_dir,
            result_saver_unselected=result_saver_unselected,
            init_state_kwargs=example,
            memory_manager=memory_manager,
            search_algorithm=config.search_algorithm,
            run_logger=run_logger,
        )

    end_time = time.time()
    run_logger.info(f"Total time: {end_time - begin_time}")

    # Log final metrics
    log_final_metrics(run_logger, base_model.inference_logger)

    # Log diversity report to file (env_grounded tasks only)
    if llm_calls_path:
        try:
            records = load_llm_calls(llm_calls_path)
            if task_type == "env_grounded" and config.dataset == "crosswords":
                from lits.eval.llm_call_logger import normalize_crosswords_action, parse_crosswords_correct_actions
                # Aggregate correct actions from all examples
                all_correct_actions = {}
                for ex in full_dataset:
                    correct = parse_crosswords_correct_actions(ex.get('query_or_goals', ''))
                    all_correct_actions.update(correct)
                print_diversity_report(
                    records,
                    normalize_fn=normalize_crosswords_action,
                    correct_actions=all_correct_actions if all_correct_actions else None
                )
            else:
                print_diversity_report(records)
        except FileNotFoundError:
            run_logger.warning(f"No LLM calls logged to {llm_calls_path}")

    run_logger.info(f"Tree search complete. Terminal nodes saved to {result_dir}/terminal_nodes/")
    run_logger.info("Run lits-eval to extract answers and compute accuracy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
