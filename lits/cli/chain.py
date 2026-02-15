"""
lits-chain: Environment-grounded chain agent CLI entry point.

Runs chain-based reasoning for environment-grounded tasks like BlocksWorld.
Unlike tree search (lits-search), this uses a sequential chain agent that
executes actions step-by-step without branching.

Usage:
    lits-chain --dataset blocksworld --include lits_benchmark.blocksworld
    lits-chain --dataset crosswords --include lits_benchmark.crosswords \
        --dataset-arg data_file=crosswords/data/mini0505.json
    lits-chain --dry-run --dataset blocksworld --include lits_benchmark.blocksworld
    lits-chain --help

Two-Stage Workflow:
1. Run lits-chain to execute chain agent and save checkpoints
2. Run lits-eval-chain to evaluate results from checkpoint files

See docs/cli/search.md for full CLI documentation.
"""

import sys
import os
import json
import traceback
import logging

from dotenv import load_dotenv, find_dotenv

from lits.agents.main import create_env_chain_agent
from lits.agents.chain.env_chain import EnvChainConfig
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules
from lits.lm import get_lm
from lits.eval import _slice_dataset
from lits.log import setup_logging
from lits.framework_config import PACKAGE_VERSION
from lits.benchmarks.registry import load_dataset
from lits.cli import (
    parse_experiment_args, apply_config_overrides,
    parse_dataset_kwargs, parse_script_vars,
)

logger = logging.getLogger(__name__)




def main() -> int:
    """Entry point for lits-chain command.
    
    Runs an environment-grounded chain agent on a registered benchmark.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    # Default config — CLI flags override these values
    config = EnvChainConfig(
        reasoning_method="env_chain",
        package_version=PACKAGE_VERSION,
        dataset="blocksworld",
        policy_model_name="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_steps=30,
        goal_reached_reward=100.0,
        goal_reward_default=0.0,
    )

    # Parse CLI arguments and apply config overrides
    cli_args = parse_experiment_args(description="Run environment-grounded chain agent")
    config = apply_config_overrides(config, cli_args)

    # Apply explicit CLI flags (take precedence over --cfg)
    if cli_args.dataset:
        config.dataset = cli_args.dataset
    if cli_args.transition:
        config.transition = cli_args.transition

    # Apply model flags
    if cli_args.policy_model:
        config.policy_model_name = cli_args.policy_model

    # Parse script-level variables (not part of algorithm config)
    script_vars = parse_script_vars(cli_args, {'offset': 0, 'limit': None})
    offset = script_vars['offset']
    limit = script_vars['limit']
    override = cli_args.override

    # Import custom modules to trigger registration
    if cli_args.import_modules:
        config.import_modules = cli_args.import_modules
        try:
            import_custom_modules(cli_args.import_modules)
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Imported custom modules: {cli_args.import_modules}")

    # Get Transition class from registry
    # Explicit --transition flag takes precedence, otherwise fall back to dataset name
    benchmark_name = config.dataset
    transition_key = getattr(config, 'transition', None) or benchmark_name
    try:
        TransitionCls = ComponentRegistry.get_transition(transition_key)
    except KeyError:
        available = ComponentRegistry.list_by_task_type("env_grounded")
        print(f"Error: Transition '{transition_key}' not found in registry.", file=sys.stderr)
        print(f"Available env_grounded benchmarks: {available}", file=sys.stderr)
        print(f"Did you forget to use --include to load the module containing "
              f"@register_transition('{transition_key}')?", file=sys.stderr)
        return 1

    # Get goal_check and generate_actions from Transition class
    if not hasattr(TransitionCls, 'goal_check'):
        print(f"Error: Transition class '{TransitionCls.__name__}' does not have "
              f"a 'goal_check' static method.", file=sys.stderr)
        return 1
    # Note: generate_actions is optional (for infinite action spaces like crosswords)
    # validate_action is optional (for finite action spaces like blocksworld)

    goal_check = TransitionCls.goal_check
    generate_all_actions = getattr(TransitionCls, 'generate_actions', None)
    validate_action = getattr(TransitionCls, 'validate_action', None)

    # Load dataset kwargs from CLI --dataset-arg or config
    dataset_kwargs = parse_dataset_kwargs(cli_args)

    # Merge with config.dataset_kwargs (CLI takes precedence)
    if config.dataset_kwargs:
        merged_kwargs = {**config.dataset_kwargs, **dataset_kwargs}
        dataset_kwargs = merged_kwargs

    # Default kwargs for known datasets (backwards compatibility)
    if benchmark_name == "blocksworld" and not dataset_kwargs:
        dataset_kwargs = {
            'config_file': "blocksworld/bw_data_bw_config.yaml",
            'domain_file': "blocksworld/bw_data_generated_domain.pddl",
            'data_file': 'blocksworld/bw_data_step_6.json'
        }

    try:
        full_dataset = load_dataset(benchmark_name, **dataset_kwargs)
    except KeyError:
        print(f"Error: No dataset loader registered for '{benchmark_name}'.", file=sys.stderr)
        print("Please register a dataset loader using @register_dataset decorator.", file=sys.stderr)
        return 1

    # Dry-run mode: print first dataset element and exit with no side effects
    if cli_args.dry_run:
        print(f"\n=== Dry Run Mode ===")
        print(f"Benchmark: {benchmark_name}")
        print(f"Transition: {TransitionCls.__name__}")
        print(f"Dataset size: {len(full_dataset)}")
        print(f"\nFirst element:")
        print(json.dumps(full_dataset[0], indent=2, default=str))
        return 0

    # --- Everything below only runs for real execution ---

    # Setup directories
    run_id = f"{benchmark_name}_chain"
    result_dir = config.setup_directories(run_id)
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Store dataset_kwargs in config for reproducibility
    if dataset_kwargs:
        config.dataset_kwargs = dataset_kwargs

    # Save config
    config.save_config(result_dir)

    # Setup logging
    run_logger = setup_logging(
        run_id="execution",
        result_dir=result_dir,
        add_console_handler=True,
        verbose=True,
        override=override
    )

    run_logger.info(f"Loaded {len(full_dataset)} examples from {benchmark_name} dataset")

    # Load model
    base_model = get_lm(
        config.policy_model_name,
        device="cuda",
        enable_thinking=True,
        sys_prompt=None,
        verbose=True
    )

    # Setup inference logging
    from lits.lm import setup_inference_logging
    setup_inference_logging(
        base_model, None, None, None,
        result_dir, override
    )

    # Create transition model using the registry-looked-up Transition class
    world_model = TransitionCls(
        base_model=base_model,
        goal_check=goal_check,
        max_steps=config.max_steps
    )

    # Create agent using factory function
    agent = create_env_chain_agent(
        base_model=base_model,
        generate_all_actions=generate_all_actions,
        validate_action=validate_action,
        world_model=world_model,
        task_name=benchmark_name,
        max_steps=config.max_steps,
        goal_reached_reward=config.goal_reached_reward,
        goal_reward_default=config.goal_reward_default,
        root_dir=result_dir
    )

    # Run agent on dataset
    selected_examples = _slice_dataset(full_dataset, offset, limit)
    run_logger.info(f"Running on {len(selected_examples)} examples (offset={offset}, limit={limit})")

    try:
        for example_idx, example in enumerate(selected_examples, start=offset):
            run_logger.info(f"Processing example {example_idx}")
            state = agent.run(
                query_or_goals=example["query_or_goals"],
                init_state_str=example["init_state_str"],
                query_idx=example_idx,
                checkpoint_dir=checkpoint_dir,
                override=override
            )
    except Exception as e:
        run_logger.error(f"Error during chain execution: {e}")
        traceback.print_exc()
        return 1

    run_logger.info(f"Chain agent complete. Checkpoints saved to {checkpoint_dir}")
    run_logger.info(f"Run evaluation: lits-eval-chain --result_dir {result_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
