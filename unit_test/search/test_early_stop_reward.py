"""Test early_stop_reward parameter on MCTS.

Runs MCTS on 1 DBBench wikitq example with terminate_on_first_solution=True
and two different early_stop_reward thresholds to verify the reward gate works.

Usage (from lits_llm/):
    PYTHONPATH="$PYTHONPATH:$(pwd)/demos" python -m unit_test.search.test_early_stop_reward

Expected behavior:
    - Low threshold (0.5): stops early (few iterations), because first terminal reward > 0.5
    - High threshold (0.99): does NOT stop early (runs all iterations), because no terminal reward > 0.99
"""

import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../demos'))

from dotenv import load_dotenv
load_dotenv()

from lits.lm import get_lm
from lits.benchmarks.registry import load_resource, load_dataset, infer_task_type
from lits.components.factory import create_components
from lits.config import ExperimentConfig
from lits.agents.tree.mcts import MCTSSearch, MCTSConfig
from lits.log import setup_logging


MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"


def run_mcts_with_early_stop(early_stop_reward, n_iters=5):
    """Run MCTS on 1 wikitq example and return iteration count."""
    import lits_benchmark.dbbench  # trigger registration

    tool_use_spec = load_resource("dbbench", database="wikitq")
    dataset = load_dataset("dbbench", database="wikitq")
    example = dataset[0]

    base_model = get_lm(MODEL)
    eval_model = get_lm(MODEL)

    config = ExperimentConfig(
        dataset="dbbench",
        policy_model_name=MODEL,
        eval_model_name=MODEL,
        search_args={"n_actions": 3, "max_steps": 10},
        component_args={"max_eval_rollout_steps": 0},
    )

    world_model, policy, evaluator = create_components(
        task_type="tool_use", task_name="dbbench",
        base_model=base_model, eval_base_model=eval_model,
        terminal_model=None, tool_use_spec=tool_use_spec, config=config,
    )

    search_config = MCTSConfig(
        policy_model_name=MODEL, eval_model_name=MODEL,
        n_actions=3, max_steps=10, n_iters=n_iters,
        roll_out_steps=2, w_exp=1.0,
        transition_before_evaluate=True,
        simulate_strategy="max",
        force_terminating_on_depth_limit=False,
        terminate_on_terminal_node=True,
        terminate_on_first_solution=True,
        early_stop_reward=early_stop_reward,
        output_trace_in_each_iter=True,
    )

    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging("test", temp_dir, add_console_handler=False)

        searcher = MCTSSearch(
            config=search_config,
            world_model=world_model,
            policy=policy,
            reward_model=evaluator,
            bn_evaluator=None,
            init_state_kwargs=example,
        )
        output = searcher.run(query=example["question"], query_idx=0)

        n_actual_iters = len(output.trace_in_each_iter)
        n_terminals = len(output.terminal_nodes_collected)
        return n_actual_iters, n_terminals
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    n_iters = 5

    # --- Test 1: low threshold (should stop early) ---
    print(f"[Test 1] early_stop_reward=0.5, n_iters={n_iters}")
    iters_low, terms_low = run_mcts_with_early_stop(0.5, n_iters=n_iters)
    print(f"  Iterations used: {iters_low}, Terminal nodes: {terms_low}")
    breakpoint()  # inspect: iters_low should be < n_iters if a terminal with reward >= 0.5 was found

    # --- Test 2: high threshold (should NOT stop early) ---
    print(f"\n[Test 2] early_stop_reward=0.99, n_iters={n_iters}")
    iters_high, terms_high = run_mcts_with_early_stop(0.99, n_iters=n_iters)
    print(f"  Iterations used: {iters_high}, Terminal nodes: {terms_high}")
    breakpoint()  # inspect: iters_high should be == n_iters (no terminal has reward >= 0.99)

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"  Low threshold (0.5):  {iters_low}/{n_iters} iterations")
    print(f"  High threshold (0.99): {iters_high}/{n_iters} iterations")
    print(f"  Early stop worked: {iters_low < iters_high}")


if __name__ == "__main__":
    main()
