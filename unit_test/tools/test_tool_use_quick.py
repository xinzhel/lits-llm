"""Quick test for ToolUsePolicy registry loading"""
import sys
sys.path.append('..')

from lits.components.policy.tool_use import ToolUsePolicy
from lits.prompts.registry import PromptRegistry
from lits.prompts.prompt import PromptTemplate

# Mock tools
class MockTool:
    def __init__(self, name):
        self.name = name

mock_tools = [MockTool("tool1"), MockTool("tool2")]

# Test 1: Check registry
print("Test 1: Check registry")
registry_prompt = PromptRegistry.get('policy', 'tool_use', None)
print(f"Registry prompt type: {type(registry_prompt)}")
print(f"Is PromptTemplate: {isinstance(registry_prompt, PromptTemplate)}")

# Test 2: Create policy without explicit prompt
print("\nTest 2: Create policy without explicit prompt")
policy = ToolUsePolicy(base_model=None, tools=mock_tools, tool_context="Test")
print(f"Policy task_prompt_spec type: {type(policy.task_prompt_spec)}")
print(f"Is string: {isinstance(policy.task_prompt_spec, str)}")
print(f"Length: {len(policy.task_prompt_spec) if policy.task_prompt_spec else 0}")
if policy.task_prompt_spec:
    print(f"Contains tool1: {'tool1' in policy.task_prompt_spec}")
    print(f"Contains tool2: {'tool2' in policy.task_prompt_spec}")
    print("✓ SUCCESS")
else:
    print("✗ FAILED: task_prompt_spec is None")
