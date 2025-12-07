"""Test component factory for tool-use MCTS integration."""

import sys
import os

# Add parent directory to path for lits imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add examples directory for component_factory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../examples'))

from component_factory import create_rest_bfs_components_tool_use, create_components
from lits.lm import get_lm
from lits.components.policy.tool_use import ToolUsePolicy
from lits.components.transition.tool_use import ToolUseTransition
from lits.components.reward.tool_use import ToolUsePRM


def test_create_rest_bfs_components_tool_use():
    """Test that create_rest_bfs_components_tool_use creates correct component types."""
    print("\n=== Testing create_rest_bfs_components_tool_use ===")
    
    # Create mock models
    MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    base_model = get_lm(MODEL_NAME)
    eval_base_model = get_lm(MODEL_NAME)
    
    # Create mock tool_use_spec
    from lits.tools.base import BaseTool
    from pydantic import BaseModel
    
    class MockToolArgs(BaseModel):
        query: str = "test"
    
    class MockTool(BaseTool):
        name = "mock_tool"
        description = "A mock tool for testing"
        args_schema = MockToolArgs
        
        def __init__(self):
            super().__init__(client=None)
        
        def _run(self, **kwargs):
            return "Mock result"
    
    tool_use_spec = {
        "tools": [MockTool()],
        "tool_context": "This is a test context"
    }
    
    # Create components
    world_model, policy, evaluator = create_rest_bfs_components_tool_use(
        base_model=base_model,
        eval_base_model=eval_base_model,
        tool_use_spec=tool_use_spec,
        task_type="tool_use",
        n_actions=3,
        max_steps=5,
        force_terminating_on_depth_limit=True,
        max_length=2048,
        max_rollout_steps=5
    )
    
    # Verify component types
    assert isinstance(world_model, ToolUseTransition), \
        f"Expected ToolUseTransition, got {type(world_model)}"
    assert isinstance(policy, ToolUsePolicy), \
        f"Expected ToolUsePolicy, got {type(policy)}"
    assert isinstance(evaluator, ToolUsePRM), \
        f"Expected ToolUsePRM, got {type(evaluator)}"
    
    # Verify ToolUsePolicy received tools
    assert policy.tools == tool_use_spec["tools"], \
        "Policy should receive tools from tool_use_spec"
    # Note: tool_context is used to format the prompt but not stored as an attribute
    
    # Verify ToolUseTransition received tools
    assert world_model.tools == tool_use_spec["tools"], \
        "Transition should receive tools from tool_use_spec"
    
    # Verify ToolUsePRM received eval_base_model and max_rollout_steps
    assert evaluator.base_model == eval_base_model, \
        "ToolUsePRM should receive eval_base_model, not base_model"
    assert evaluator.tools == tool_use_spec["tools"], \
        "ToolUsePRM should receive tools from tool_use_spec"
    assert evaluator.max_rollout_steps == 5, \
        "ToolUsePRM should receive max_rollout_steps parameter"
    
    print("✓ All component types are correct")
    print("✓ ToolUsePolicy received tools")
    print("✓ ToolUseTransition received tools")
    print("✓ ToolUsePRM received eval_base_model and max_rollout_steps")
    print("\n=== Test passed ===\n")


def test_create_components_error_handling():
    """Test error handling when tool_use_spec is None."""
    print("\n=== Testing error handling for missing tool_use_spec ===")
    
    MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    base_model = get_lm(MODEL_NAME)
    eval_base_model = get_lm(MODEL_NAME)
    
    # Create a mock config object
    class MockConfig:
        n_actions = 3
        max_steps = 5
        force_terminating_on_depth_limit = True
        max_length = 2048
        max_rollout_steps = 5
    
    config = MockConfig()
    
    # Test that ValueError is raised when tool_use_spec is None
    try:
        create_components(
            reasoning_method="rest",
            task_type="tool_use",
            base_model=base_model,
            eval_base_model=eval_base_model,
            terminal_model=None,
            tool_use_spec=None,
            config=config
        )
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "tool_use_spec is required" in str(e), \
            f"Expected error message about tool_use_spec, got: {e}"
        print(f"✓ Correct error raised: {e}")
    
    print("\n=== Test passed ===\n")


if __name__ == "__main__":
    test_create_rest_bfs_components_tool_use()
    test_create_components_error_handling()
    print("\n✅ All tests passed!")
