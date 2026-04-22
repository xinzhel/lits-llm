"""
lits-chain: Chain agent CLI entry point (env_grounded + tool-use).

Runs chain-based reasoning for both environment-grounded tasks (BlocksWorld,
Crosswords) and tool-use tasks (DBBench, MapEval).  For env_grounded tasks it
uses ``EnvChain``; for tool-use tasks it uses ``ReActChat``.

Usage:
    # env_grounded
    lits-chain --dataset blocksworld --include lits_benchmark.blocksworld
    lits-chain --dataset crosswords --include lits_benchmark.crosswords \
        --dataset-arg data_file=crosswords/data/mini0505.json

    # tool-use
    lits-chain --dataset dbbench --dataset-arg database=wikisql \
        --include lits_benchmark.dbbench \
        --policy-model bedrock/us.anthropic.claude-sonnet-4-6

    lits-chain --dry-run --dataset dbbench --dataset-arg database=wikisql \
        --include lits_benchmark.dbbench
    lits-chain --help

    # tool-use with cross-trajectory memory (pass@N + memory)
    lits-chain --dataset terminal_bench --include lits_benchmark.terminal_bench \
        --policy-model bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0 \
        --cfg native=True --cfg n_attempts=5 \
        --memory-arg backend=local \
        --var limit=10

Two-Stage Workflow:
1. Run lits-chain to execute chain agent and save checkpoints
2. Evaluate:
   - env_grounded → lits-eval-chain
   - tool-use     → lits-eval
"""

import sys
import os
import json
import traceback
import logging
from contextlib import nullcontext
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv

from lits.agents.main import create_env_chain_agent, create_tool_use_agent
from lits.agents.chain.env_chain import EnvChainConfig
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules
from lits.lm import get_lm
from lits.eval import _slice_dataset
from lits.log import setup_logging
from lits.benchmarks.registry import load_dataset, has_resource, load_resource
from lits.cli import (
    parse_experiment_args, apply_config_overrides,
    parse_dataset_kwargs, parse_script_vars,
    parse_memory_args,
    log_command,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentor helpers for pass@N + cross-trajectory memory
# ---------------------------------------------------------------------------


def _analyze_trajectory(
    augmentors: List["ContextAugmentor"],
    state,
    example_idx: int,
    attempt: int,
    run_logger,
):
    """Extract facts from a completed trajectory via all augmentors.

    Calls each augmentor once with the full trajectory and ``batch=True``.
    Per-step augmentors (e.g., FactMemoryAugmentor) use batch mode to
    process all steps in a single LLM call.  Per-trajectory augmentors
    (e.g., ReflectionAugmentor) ignore the batch kwarg.

    Trajectory key format: ``q/{attempt}`` — each attempt is a root-level
    branch in the TrajectoryKey tree.  ``search_id`` is ``q_{example_idx}``.

    Args:
        augmentors: List of ContextAugmentor instances.
        state: ToolUseState (list of steps) from agent.run().
        example_idx: Example index.
        attempt: Attempt number within pass@N.
        run_logger: Logger.
    """
    if state is None or len(state) == 0:
        return

    traj_key = f"q/{attempt}"
    for aug in augmentors:
        try:
            aug.analyze(
                list(state),
                trajectory_key=traj_key,
                query_idx=example_idx,
                batch=True,
            )
        except Exception as e:
            logger.warning(f"  {type(aug).__name__}.analyze() failed: {e}")

    run_logger.info(
        f"  Augmentors: analyzed attempt {attempt} "
        f"({len(state)} steps, {len(augmentors)} augmentors)"
    )


def _clear_memory(manager: "LiTSMemoryManager", run_logger, example_idx: int):
    """Clear all stored facts between examples.

    Memory is per-example: attempts within the same example share memory,
    but different examples start fresh.

    Args:
        manager: LiTSMemoryManager whose backend storage is cleared.
        run_logger: Logger.
        example_idx: Next example index (for logging).
    """
    backend = manager.backend
    backend._units.clear()
    backend._vectors.clear()
    run_logger.info(f"  Memory cleared for example {example_idx}")


def main() -> int:
    """Entry point for lits-chain command.

    Supports both env_grounded (EnvChain) and tool-use (ReActChat) tasks.
    Task type is auto-detected via ``has_resource()`` — if a resource loader
    is registered for the dataset, it is treated as tool-use.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    # Default config — CLI flags override these values.
    # We start with EnvChainConfig (superset of ChainConfig) and switch to
    # ReactChatConfig for tool-use after detection.
    config = EnvChainConfig(
        dataset="blocksworld",
        policy_model_name="bedrock/us.anthropic.claude-sonnet-4-6",
        max_steps=30,
        goal_reached_reward=100.0,
        goal_reward_default=0.0,
    )

    # Parse CLI arguments and apply config overrides
    cli_args = parse_experiment_args(description="Run chain agent (env_grounded + tool-use)")
    config = apply_config_overrides(config, cli_args)

    # Apply explicit CLI flags (take precedence over --cfg)
    if cli_args.dataset:
        config.dataset = cli_args.dataset
    if cli_args.transition:
        config.transition = cli_args.transition

    # Apply model flags
    if cli_args.policy_model:
        config.policy_model_name = cli_args.policy_model
    if cli_args.eval_model:
        config.eval_model_name = cli_args.eval_model

    # Map --output-dir and --root-dir flags
    if cli_args.output_dir:
        config.output_dir = cli_args.output_dir
    if cli_args.root_dir:
        config.root_dir = cli_args.root_dir

    # Parse script-level variables (not part of algorithm config)
    script_vars = parse_script_vars(cli_args, {'offset': 0, 'limit': None})
    offset = script_vars['offset']
    limit = script_vars['limit']
    override = cli_args.override

    # Parse memory args (--memory-arg backend=local model=... etc.)
    memory_kwargs = parse_memory_args(cli_args) if cli_args.memory_args else None

    # Import custom modules to trigger registration
    if cli_args.import_modules:
        config.import_modules = cli_args.import_modules
        try:
            import_custom_modules(cli_args.import_modules)
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Imported custom modules: {cli_args.import_modules}")

    # --- Task-type detection (1.1) ---
    benchmark_name = config.dataset
    is_tool_use = has_resource(benchmark_name)

    # Load dataset kwargs from CLI --dataset-arg or config
    dataset_kwargs = parse_dataset_kwargs(cli_args)
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

    # --- Dry-run mode ---
    if cli_args.dry_run:
        print(f"\n=== Dry Run Mode ===")
        print(f"Benchmark: {benchmark_name}")
        print(f"Task type: {'tool_use' if is_tool_use else 'env_grounded'}")
        print(f"Dataset size: {len(full_dataset)}")
        print(f"\nFirst element:")
        print(json.dumps(full_dataset[0], indent=2, default=str))
        return 0

    # --- Real execution below ---

    if is_tool_use:
        return _run_tool_use(config, benchmark_name, full_dataset, dataset_kwargs,
                             offset, limit, override, memory_kwargs=memory_kwargs)
    else:
        return _run_env_grounded(config, benchmark_name, full_dataset, dataset_kwargs,
                                 offset, limit, override)


# ---------------------------------------------------------------------------
# Tool-use path (ReActChat)
# ---------------------------------------------------------------------------

def _run_tool_use(config, benchmark_name, full_dataset, dataset_kwargs,
                  offset, limit, override, memory_kwargs=None) -> int:
    """Run ReActChat on a tool-use dataset (e.g. dbbench, mapeval)."""

    # Load tool-use resource (starts DB containers, etc.)
    if os.path.exists("mapeval/.env"):
        load_dotenv("mapeval/.env")
    tool_use_spec = load_resource(benchmark_name)

    # Setup directories
    run_id = f"{benchmark_name}_chain"
    result_dir = config.setup_directories(run_id)
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config for reproducibility (after agent creation, since
    # create_tool_use_agent also writes a ReactChatConfig to result_dir)
    config.dataset = benchmark_name
    if dataset_kwargs:
        config.dataset_kwargs = dataset_kwargs

    # Setup logging
    run_logger = setup_logging(
        run_id="execution",
        result_dir=result_dir,
        add_console_handler=True,
        verbose=True,
        override=override
    )
    log_command(run_logger)
    run_logger.info(f"Loaded {len(full_dataset)} examples from {benchmark_name} dataset")

    # pass@N: auto-set temperature when n_attempts > 1
    n_attempts = getattr(config, "n_attempts", 1)
    if n_attempts > 1 and config.temperature == 0.0:
        config.temperature = 0.9  # follows Golubev et al. ICML 2025
        run_logger.info(f"Auto-set temperature=0.9 for pass@{n_attempts} (n_attempts>1)")

    # Create agent via factory (handles model loading, policy, transition)
    agent = create_tool_use_agent(
        tools=tool_use_spec["tools"],
        tool_context=tool_use_spec.get("tool_context", ""),
        task_name=benchmark_name,
        model_name=config.policy_model_name,
        max_iter=config.max_steps,
        root_dir=result_dir,
        override_logger=override,
        native=getattr(config, "native", False),
        temperature=config.temperature,
    )

    # Save experiment config (supplements ReactChatConfig saved by create_tool_use_agent
    # with experiment metadata: dataset, import_modules, dataset_kwargs, eval_model, root_dir)
    config.save_config(result_dir)

    # --- Augmentor setup (cross-trajectory fact transfer for pass@N) ---
    memory_manager = None
    augmentors = []
    # Mutable query_context dict — updated per-attempt, read by _combined_retrieve
    query_context = {}
    if memory_kwargs is not None:
        from lits.cli.search import setup_memory_manager, create_augmentors
        from lits.agents.tree.augmentor_setup import wire_retrieval_to_policy
        memory_manager = setup_memory_manager(run_logger, memory_kwargs)
        augmentors = create_augmentors(memory_manager, memory_kwargs, run_logger=run_logger)
        wire_retrieval_to_policy(agent.policy, augmentors, query_context)

    # Run agent on dataset
    selected_examples = _slice_dataset(full_dataset, offset, limit)
    run_logger.info(f"Running on {len(selected_examples)} examples (offset={offset}, limit={limit})")

    # Per-example callbacks from resource (e.g., KG entity injection, answer resolution)
    prepare_tool_state = tool_use_spec.get("prepare_tool_state")
    resolve_answer = tool_use_spec.get("resolve_answer")
    verify_fn = tool_use_spec.get("verify")

    try:
        for example_idx, example in enumerate(selected_examples, start=offset):
            run_logger.info(f"Processing example {example_idx}")
            query = example["question"]

            # Clear memory between examples (different tasks don't share memory)
            if memory_manager is not None:
                _clear_memory(memory_manager, run_logger, example_idx)

            for attempt in range(n_attempts):
                attempt_id = f"{example_idx}_a{attempt}" if n_attempts > 1 else example_idx
                if n_attempts > 1:
                    run_logger.info(f"  Attempt {attempt+1}/{n_attempts} for example {example_idx}")

                # Skip completed attempts on resume
                if not override:
                    cp_file = os.path.join(checkpoint_dir, f"{attempt_id}.json")
                    reward_file = os.path.join(checkpoint_dir, f"{attempt_id}_reward.json")

                    # Fully complete: reward file exists (agent ran + verify ran)
                    if os.path.exists(reward_file):
                        run_logger.info(f"  Skipping {attempt_id} (completed)")
                        continue

                    # Partially complete: checkpoint has answer but no reward
                    if os.path.exists(cp_file):
                        try:
                            with open(cp_file) as _f:
                                cp_data = json.load(_f)
                            has_answer = any(s.get("answer") for s in cp_data.get("steps", []))
                        except (json.JSONDecodeError, KeyError):
                            has_answer = False

                        if has_answer and verify_fn is None:
                            # No verify needed, answer = complete
                            run_logger.info(f"  Skipping {attempt_id} (completed, no verify)")
                            continue
                        if has_answer and verify_fn is not None:
                            # Has answer but verify didn't run — need to re-run
                            run_logger.info(f"  Re-running {attempt_id} (answer exists but verify missing)")

                if prepare_tool_state is not None:
                    prepare_tool_state(example)

                # Update query_context for memory retrieval (read by _combined_retrieve)
                # Uses tree-compatible path: q/{attempt} (attempt as root-level branch)
                if memory_manager is not None:
                    query_context["trajectory_key"] = f"q/{attempt}"
                    query_context["query_idx"] = example_idx

                # Log attempt context for InferenceLogger (separate from role)
                log_ctx = {}
                if n_attempts > 1:
                    log_ctx["attempt"] = attempt
                with agent.policy.base_model.inference_logger.log_context(**log_ctx) if log_ctx else nullcontext():
                    state = agent.run(
                        query=query,
                        query_idx=attempt_id,
                        checkpoint_dir=checkpoint_dir,
                        override=True,  # always fresh for env-stateful tasks (Docker container is new)
                    )

                # Post-run answer resolution (e.g., KG variable → entity names via SPARQL)
                if resolve_answer is not None and state is not None:
                    raw_answer = state.get_final_answer()
                    if raw_answer:
                        resolved = resolve_answer(raw_answer, state)
                        if resolved != raw_answer:
                            state[-1].answer = resolved
                            checkpoint_path = os.path.join(checkpoint_dir, f"{attempt_id}.json")
                            state.save(checkpoint_path, query)
                            run_logger.info(f"Resolved answer: '{raw_answer}' → '{resolved}'")

                # Post-run verification for environment-based benchmarks (e.g., Terminal-Bench).
                # Must run while the container is still alive (before prepare_tool_state stops it).
                if verify_fn is not None:
                    try:
                        reward = verify_fn(example)
                        run_logger.info(f"Verification for {attempt_id}: reward={reward}")
                        # Save reward alongside checkpoint
                        reward_path = os.path.join(checkpoint_dir, f"{attempt_id}_reward.json")
                        with open(reward_path, "w") as rf:
                            import json as _json
                            _json.dump({"example_idx": example_idx, "attempt": attempt, "task_id": example.get("task_id", ""), "reward": reward}, rf)
                    except Exception as ve:
                        run_logger.warning(f"Verification failed for {attempt_id}: {ve}")

                # Extract facts from trajectory via augmentors for next attempt
                if augmentors and state is not None:
                    _analyze_trajectory(augmentors, state, example_idx, attempt, run_logger)

    except Exception as e:
        run_logger.error(f"Error during ReAct execution: {e}")
        traceback.print_exc()
        return 1

    run_logger.info(f"ReAct chain complete. Checkpoints saved to {checkpoint_dir}")

    # Generate pass@N summary if n_attempts > 1
    if n_attempts > 1:
        import json as _json
        from collections import defaultdict
        attempt_rewards = defaultdict(list)  # example_idx -> [reward_0, reward_1, ...]
        for f in sorted(os.listdir(checkpoint_dir)):
            if f.endswith("_reward.json"):
                with open(os.path.join(checkpoint_dir, f)) as fh:
                    d = _json.load(fh)
                    attempt_rewards[d["example_idx"]].append(d["reward"])

        summary = {"n_attempts": n_attempts, "examples": [], "temperature": config.temperature}
        n_pass_1 = 0
        n_pass_n = 0
        for idx in sorted(attempt_rewards.keys()):
            rewards = attempt_rewards[idx]
            p1 = rewards[0] > 0 if rewards else False
            pn = any(r > 0 for r in rewards)
            if p1: n_pass_1 += 1
            if pn: n_pass_n += 1
            summary["examples"].append({
                "idx": idx,
                "attempts": rewards,
                "pass_at_1": p1,
                "pass_at_n": pn,
            })
        total = len(attempt_rewards)
        summary["pass_at_1"] = n_pass_1 / total if total else 0
        summary["pass_at_n"] = n_pass_n / total if total else 0

        summary_path = os.path.join(checkpoint_dir, "pass_at_n_summary.json")
        with open(summary_path, "w") as sf:
            _json.dump(summary, sf, indent=2)
        run_logger.info(f"pass@1={summary['pass_at_1']:.1%}, pass@{n_attempts}={summary['pass_at_n']:.1%} ({n_pass_n}/{total})")
        run_logger.info(f"Summary saved to {summary_path}")

    run_logger.info(f"Run evaluation: lits-eval --result_dir {result_dir} "
                    f"--dataset_name {benchmark_name} --include {' '.join(config.import_modules or [])}")
    return 0


# ---------------------------------------------------------------------------
# Env-grounded path (EnvChain) — unchanged logic, extracted to helper
# ---------------------------------------------------------------------------

def _run_env_grounded(config, benchmark_name, full_dataset, dataset_kwargs,
                      offset, limit, override) -> int:
    """Run EnvChain on an env_grounded dataset (e.g. blocksworld, crosswords)."""

    # Get Transition class from registry
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

    if not hasattr(TransitionCls, 'goal_check'):
        print(f"Error: Transition class '{TransitionCls.__name__}' does not have "
              f"a 'goal_check' static method.", file=sys.stderr)
        return 1

    goal_check = TransitionCls.goal_check
    generate_all_actions = getattr(TransitionCls, 'generate_actions', None)
    validate_action = getattr(TransitionCls, 'validate_action', None)

    # Setup directories
    run_id = f"{benchmark_name}_chain"
    result_dir = config.setup_directories(run_id)
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Store dataset_kwargs in config for reproducibility
    if dataset_kwargs:
        config.dataset_kwargs = dataset_kwargs
    config.save_config(result_dir)

    # Setup logging
    run_logger = setup_logging(
        run_id="execution",
        result_dir=result_dir,
        add_console_handler=True,
        verbose=True,
        override=override
    )
    log_command(run_logger)
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
    setup_inference_logging(base_model, root_dir=result_dir, override=override)

    # Create transition model
    world_model = TransitionCls(
        base_model=base_model,
        goal_check=goal_check,
        max_steps=config.max_steps
    )

    # Create agent
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
