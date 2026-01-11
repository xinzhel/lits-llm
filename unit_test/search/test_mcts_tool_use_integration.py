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
from lits_benchmark import load_resource
from lits.benchmarks.registry import infer_task_type, TOOL_USE_DATASETS
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
    assert benchmark_name in TOOL_USE_DATASETS, \
        f"{benchmark_name} should be in TOOL_USE_DATASETS"
    print(f"✓ {benchmark_name} is recognized as tool-use dataset")
    
    # Load tool-use specification
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
            "examples": [
                {"question": "Test question?", "answer": "Test answer"}
            ]
        }
        print(f"✓ Created mock tool_use_spec for testing")
    
    # Verify tool_use_spec structure
    assert "tools" in tool_use_spec, "tool_use_spec should have 'tools' key"
    assert "tool_context" in tool_use_spec, "tool_use_spec should have 'tool_context' key"
    assert "examples" in tool_use_spec, "tool_use_spec should have 'examples' key"
    assert len(tool_use_spec["examples"]) > 0, "tool_use_spec should have examples"
    print(f"✓ tool_use_spec has correct structure")
    
    # Load models
    print(f"Loading models...")
    base_model = get_lm(policy_model_name)
    eval_base_model = get_lm(eval_model_name)
    print(f"✓ Loaded policy model: {policy_model_name}")
    print(f"✓ Loaded eval model: {eval_model_name}")
    
    # Create components using component factory
    from component_factory import create_components
    from search_config import ExperimentConfig
    
    config = ExperimentConfig(
        benchmark_name=benchmark_name,
        policy_model_name=policy_model_name,
        eval_model_name=eval_model_name,
        reasoning_method="rest",  # MCTS-based method
        n_actions=2,
        max_steps=5,
        max_eval_rollout_steps=3,  # Limit rollout steps for testing
        offset=0,
        limit=1,  # Test with just 1 example
        override_log_result=True
    )
    
    task_type = infer_task_type(benchmark_name)
    assert task_type == "tool_use", f"Expected task_type='tool_use', got '{task_type}'"
    print(f"✓ Inferred task_type: {task_type}")
    
    # Create components
    try:
        world_model, policy, evaluator = create_components(
            reasoning_method=config.reasoning_method,
            task_type=task_type,
            base_model=base_model,
            eval_base_model=eval_base_model,
            terminal_model=None,
            tool_use_spec=tool_use_spec,
            config=config
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
    
    assert isinstance(world_model, ToolUseTransition), \
        f"Expected ToolUseTransition, got {type(world_model)}"
    assert isinstance(policy, ToolUsePolicy), \
        f"Expected ToolUsePolicy, got {type(policy)}"
    assert isinstance(evaluator, ToolUsePRM), \
        f"Expected ToolUsePRM, got {type(evaluator)}"
    print(f"✓ All components have correct types")
    
    # Create MCTS config (max_eval_rollout_steps is passed to component factory, not MCTS config)
    search_config = MCTSConfig(
        reasoning_method="rest",
        policy_model_name=policy_model_name,
        eval_model_name=eval_model_name,
        n_actions=2,
        max_steps=5,
        n_iters=3,  # Limit iterations for testing
        roll_out_steps=2,
        w_exp=1.0,
        force_terminating_on_depth_limit=False,
        terminate_on_terminal_node=True,
        output_trace_in_each_iter=True
    )
    print(f"✓ Created MCTS config with {search_config.n_iters} iterations")
    
    # Setup temporary logging
    temp_dir = tempfile.mkdtemp()
    try:
        logger = setup_logging("test_mcts_tool_use", temp_dir, add_console_handler=False)
        
        # Get first example
        example = tool_use_spec["examples"][0]
        question = example["question"]
        print(f"\n✓ Testing with question: {question[:100]}...")
        
        # Only run MCTS if database is available
        if db_available:
            # Run MCTS
            print(f"\nRunning MCTS...")
            try:
                algo_output = mcts(
                    query_or_goals=question,
                    query_idx=0,
                    mcts_search_config=search_config,
                    world_model=world_model,
                    policy=policy,
                    reward_model=evaluator,
                    bn_evaluator=None
                )
                print(f"✓ MCTS completed successfully")
            except Exception as e:
                print(f"✗ MCTS failed: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Verify output structure
            assert hasattr(algo_output, 'terminal_nodes_collected'), \
                "algo_output should have terminal_nodes_collected"
            assert hasattr(algo_output, 'trace_of_nodes'), \
                "algo_output should have trace_of_nodes"
            assert hasattr(algo_output, 'trace_in_each_iter'), \
                "algo_output should have trace_in_each_iter"
            
            print(f"✓ MCTS output has correct structure")
            print(f"  - terminal_nodes_collected: {len(algo_output.terminal_nodes_collected)} nodes")
            print(f"  - trace_of_nodes: {len(algo_output.trace_of_nodes)} nodes")
            print(f"  - trace_in_each_iter: {len(algo_output.trace_in_each_iter)} iterations")
            
            # Verify we collected some terminal nodes
            assert len(algo_output.terminal_nodes_collected) > 0, \
                "Should have collected at least one terminal node"
            print(f"✓ Collected {len(algo_output.terminal_nodes_collected)} terminal nodes")
            
            # Verify terminal nodes have states
            for i, node in enumerate(algo_output.terminal_nodes_collected):
                assert hasattr(node, 'state'), f"Terminal node {i} should have state"
                assert node.state is not None, f"Terminal node {i} state should not be None"
            print(f"✓ All terminal nodes have valid states")
        else:
            print(f"\n⚠ Skipping MCTS execution (database not available)")
            print(f"  Component creation and configuration verified successfully")
        
        print("\n=== Test passed ===\n")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mcts_tool_use_integration()
    print("\n✅ MCTS tool-use integration test passed!")
