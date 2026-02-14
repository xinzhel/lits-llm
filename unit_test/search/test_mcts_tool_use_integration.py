"""Test MCTS integration with tool-use components for mapeval-sql.

This test verifies that main_search.py can successfully run MCTS with tool-use
components (ToolUsePolicy, ToolUseTransition, ToolUsePRM) on the mapeval-sql dataset.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../examples'))

from dotenv import load_dotenv
# Load environment variables from mapeval example directory for database configuration
load_dotenv("../../examples/mapeval/.env")
# Also load main .env for API keys
load_dotenv("../../.env")

from lits.lm import get_lm
from lits.benchmarks.registry import load_resource, load_dataset, infer_task_type, has_resource
from lits.agents.tree.mcts import mcts, MCTSConfig
from lits.log import setup_logging
import tempfile
import shutil


def test_mcts_tool_use_integration():
    """Test that MCTS runs successfully with tool-use components for mapeval-sql."""
    print("\n=== Testing MCTS with tool-use components ===")

    # Configuration
    benchmark_name = "mapeval-sql"
    policy_model_name = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    eval_model_name = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Verify this is a tool-use dataset
    assert has_resource(benchmark_name), \
        f"{benchmark_name} should be registered as a resource"
    print(f"✓ {benchmark_name} is registered as a tool-use resource")

    # Load tool-use specification (tools + tool_context only)
    tool_use_spec = None
    db_available = False
    try:
        tool_use_spec = load_resource(benchmark_name)
        db_available = True
        print(f"✓ Loaded tool_use_spec with {len(tool_use_spec['tools'])} tools")
    except Exception as e:
        print(f"⚠ Failed to load tool_use_spec: {e}")
        print("  Database not configured - will test component creation only")

        # Create a minimal mock tool_use_spec for testing component creation
        from lits.tools.base import BaseTool
        from pydantic import BaseModel, Field

        class MockToolInput(BaseModel):
            query: str = Field(description="Query string")

        class MockTool(BaseTool):
            name: str = "mock_tool"
            description: str = "A mock tool for testing"
            args_schema: type[BaseModel] = MockToolInput

            def _run(self, query: str) -> str:
                return f"Mock result for: {query}"

        tool_use_spec = {
            "tools": [MockTool()],
            "tool_context": "Mock tool context for testing",
        }
        print(f"✓ Created mock tool_use_spec for testing")

    # Verify tool_use_spec structure (no "examples" — those come from load_dataset)
    assert "tools" in tool_use_spec, "tool_use_spec should have 'tools' key"
    assert "tool_context" in tool_use_spec, "tool_use_spec should have 'tool_context' key"
    print(f"✓ tool_use_spec has correct structure")

    # Load dataset examples separately via load_dataset
    full_dataset = load_dataset(benchmark_name)
    assert len(full_dataset) > 0, "Dataset should have examples"
    assert "question" in full_dataset[0], "Examples should have 'question' key"
    print(f"✓ Loaded {len(full_dataset)} examples via load_dataset()")

    # Load models
    print(f"Loading models...")
    base_model = get_lm(policy_model_name)
    eval_base_model = get_lm(eval_model_name)
    print(f"✓ Loaded policy model: {policy_model_name}")
    print(f"✓ Loaded eval model: {eval_model_name}")

    # Create components using component factory
    from lits.components.factory import create_components
    from lits.config import ExperimentConfig

    config = ExperimentConfig(
        dataset=benchmark_name,
        policy_model_name=policy_model_name,
        eval_model_name=eval_model_name,
        search_framework="rest",
        search_args={"n_actions": 2, "max_steps": 5},
        component_args={"max_eval_rollout_steps": 3},
        offset=0,
        limit=1,
    )

    task_type = infer_task_type(benchmark_name)
    assert task_type == "tool_use", f"Expected task_type='tool_use', got '{task_type}'"
    print(f"✓ Inferred task_type: {task_type}")

    # Create components
    try:
        world_model, policy, evaluator = create_components(
            task_type=task_type,
            task_name=benchmark_name,
            base_model=base_model,
            eval_base_model=eval_base_model,
            terminal_model=None,
            tool_use_spec=tool_use_spec,
            config=config,
        )
        print(f"✓ Created components:")
        print(f"  - world_model: {type(world_model).__name__}")
        print(f"  - policy: {type(policy).__name__}")
        print(f"  - evaluator: {type(evaluator).__name__}")
    except Exception as e:
        print(f"✗ Failed to create components: {e}")
        raise

    # Verify component types
    from lits.components.transition.tool_use import ToolUseTransition
    from lits.components.policy.tool_use import ToolUsePolicy
    from lits.components.reward.tool_use import ToolUsePRM

    assert isinstance(world_model, ToolUseTransition)
    assert isinstance(policy, ToolUsePolicy)
    assert isinstance(evaluator, ToolUsePRM)
    print(f"✓ All components have correct types")

    # Create MCTS config
    search_config = MCTSConfig(
        reasoning_method="rest",
        policy_model_name=policy_model_name,
        eval_model_name=eval_model_name,
        n_actions=2,
        max_steps=5,
        n_iters=3,
        roll_out_steps=2,
        w_exp=1.0,
        force_terminating_on_depth_limit=False,
        terminate_on_terminal_node=True,
        output_trace_in_each_iter=True,
    )
    print(f"✓ Created MCTS config with {search_config.n_iters} iterations")

    # Setup temporary logging
    temp_dir = tempfile.mkdtemp()
    try:
        logger = setup_logging("test_mcts_tool_use", temp_dir, add_console_handler=False)

        # Get first example from dataset
        example = full_dataset[0]
        question = example["question"]
        print(f"\n✓ Testing with question: {question[:100]}...")

        # Only run MCTS if database is available
        if db_available:
            print(f"\nRunning MCTS...")
            try:
                algo_output = mcts(
                    query_or_goals=question,
                    query_idx=0,
                    mcts_search_config=search_config,
                    world_model=world_model,
                    policy=policy,
                    reward_model=evaluator,
                    bn_evaluator=None,
                )
                print(f"✓ MCTS completed successfully")
            except Exception as e:
                print(f"✗ MCTS failed: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Verify output structure
            assert hasattr(algo_output, 'terminal_nodes_collected')
            assert hasattr(algo_output, 'trace_of_nodes')
            assert hasattr(algo_output, 'trace_in_each_iter')

            print(f"✓ MCTS output has correct structure")
            print(f"  - terminal_nodes: {len(algo_output.terminal_nodes_collected)}")
            print(f"  - trace_of_nodes: {len(algo_output.trace_of_nodes)}")
            print(f"  - trace_in_each_iter: {len(algo_output.trace_in_each_iter)}")

            assert len(algo_output.terminal_nodes_collected) > 0
            print(f"✓ Collected {len(algo_output.terminal_nodes_collected)} terminal nodes")

            for i, node in enumerate(algo_output.terminal_nodes_collected):
                assert hasattr(node, 'state') and node.state is not None
            print(f"✓ All terminal nodes have valid states")
        else:
            print(f"\n⚠ Skipping MCTS execution (database not available)")
            print(f"  Component creation and configuration verified successfully")

        print("\n=== Test passed ===\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mcts_tool_use_integration()
    print("\n✅ MCTS tool-use integration test passed!")
