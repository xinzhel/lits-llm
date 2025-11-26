"""
Unit tests for all LLM-based Policy components with prompt injection system.

Tests all policy implementations in lits.components.policy:
- RAPPolicy
- ToolUsePolicy
- ConcatPolicy
- EnvGroundedPolicy
"""

import sys
sys.path.append('../..')

from lits.components.policy.rap import RAPPolicy
from lits.components.policy.tool_use import ToolUsePolicy
from lits.components.policy.concat import ConcatPolicy
from lits.components.policy.env_grounded import EnvGroundedPolicy
from lits.prompts.registry import PromptRegistry, load_default_prompts


def test_rap_policy_prompt_loading():
    """Test RAPPolicy prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: RAPPolicy Prompt Loading")
    print("="*70)
    
    # Clear registry
    PromptRegistry.clear()
    
    # Register test prompts
    PromptRegistry.register('policy', 'rap', None, "Default RAP policy prompt")
    PromptRegistry.register('policy', 'rap', 'math_qa', "Math QA RAP policy prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    policy1 = RAPPolicy(base_model=None, task_type=None)
    assert policy1.task_prompt_spec == "Default RAP policy prompt"
    print(f"✓ Loaded default prompt: {policy1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    policy2 = RAPPolicy(base_model=None, task_type='math_qa')
    assert policy2.task_prompt_spec == "Math QA RAP policy prompt"
    print(f"✓ Loaded math_qa prompt: {policy2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit RAP prompt"
    policy3 = RAPPolicy(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert policy3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {policy3.task_prompt_spec}")
    
    # Test 4: usr_prompt_spec parameter (independent from task_prompt_spec)
    print("\nTest 4: usr_prompt_spec parameter (independent)")
    usr_prompt = {'question_prefix': 'Q{idx}:', 'answer_prefix': 'A{idx}:'}
    policy4 = RAPPolicy(base_model=None, usr_prompt_spec=usr_prompt, task_type='math_qa')
    assert policy4.usr_prompt_spec == usr_prompt
    assert policy4.task_prompt_spec == "Math QA RAP policy prompt"  # Loaded from registry
    print(f"✓ usr_prompt_spec set independently: {policy4.usr_prompt_spec}")
    print(f"✓ task_prompt_spec loaded from registry: {policy4.task_prompt_spec[:30]}...")
    
    # Test 5: Both prompts loaded from registry
    print("\nTest 5: Both prompts loaded from registry")
    PromptRegistry.register_usr('policy', 'rap', 'math_qa', {'user_key': 'user_value'})
    policy5 = RAPPolicy(base_model=None, task_type='math_qa')
    assert policy5.task_prompt_spec == "Math QA RAP policy prompt"
    assert policy5.usr_prompt_spec == {'user_key': 'user_value'}
    print(f"✓ Both prompts loaded from registry")
    print(f"  task_prompt_spec: {policy5.task_prompt_spec[:30]}...")
    print(f"  usr_prompt_spec: {policy5.usr_prompt_spec}")
    
    # Clean up and reload defaults for next test
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All RAPPolicy prompt loading tests passed")


def test_tool_use_policy_prompt_loading():
    """Test ToolUsePolicy prompt loading with different methods.
    
    Note: ToolUsePolicy loads task_prompt_spec from registry and formats it with
    tool_context and tools information.
    """
    print("\n" + "="*70)
    print("TEST: ToolUsePolicy Prompt Loading")
    print("="*70)
    
    # Import to trigger registry loading
    from lits.prompts.policy import tool_use as tool_use_prompts
    from lits.prompts.prompt import PromptTemplate
    
    # Mock tools list
    class MockTool:
        def __init__(self, name):
            self.name = name
            self.description = f"Description of {name}"
    
    mock_tools = [MockTool("tool1"), MockTool("tool2")]
    
    # Test 1: Load from registry and format with tools
    print("\nTest 1: Load from registry and format with tools")
    policy1 = ToolUsePolicy(base_model=None, tools=mock_tools, tool_context="Test context")
    print("Type: ", type(policy1.task_prompt_spec))
    assert policy1.task_prompt_spec is not None, "task_prompt_spec should be loaded from registry"
    assert isinstance(policy1.task_prompt_spec, str), f"Expected str, got {type(policy1.task_prompt_spec)}"
    assert "tool1" in policy1.task_prompt_spec
    assert "tool2" in policy1.task_prompt_spec
    print(f"✓ Loaded from registry and formatted (length: {len(policy1.task_prompt_spec)})")
    print(f"✓ Contains tool names: tool1, tool2  (length: {len(policy1.task_prompt_spec)})")
    print(f"✓ Contains tool names: tool1, tool2")
    
    # Test 2: Verify registry contains PromptTemplate
    print("\nTest 2: Verify registry contains PromptTemplate")
    registry_prompt = PromptRegistry.get('policy', 'tool_use', None)
    assert registry_prompt is not None
    assert isinstance(registry_prompt, PromptTemplate)
    print(f"✓ Registry contains PromptTemplate: {type(registry_prompt).__name__}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit ToolUse prompt"
    policy3 = ToolUsePolicy(base_model=None, task_prompt_spec=custom_prompt, tools=mock_tools)
    assert policy3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt overrides registry: {policy3.task_prompt_spec}")
    
    # Test 4: usr_prompt_spec parameter
    print("\nTest 4: usr_prompt_spec parameter")
    usr_prompt = "User message prompt"
    policy4 = ToolUsePolicy(base_model=None, usr_prompt_spec=usr_prompt, tools=mock_tools)
    assert policy4.usr_prompt_spec == usr_prompt
    print(f"✓ usr_prompt_spec set: {policy4.usr_prompt_spec}")
    
    print("\n✓ All ToolUsePolicy prompt loading tests passed")


def test_concat_policy_prompt_loading():
    """Test ConcatPolicy prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: ConcatPolicy Prompt Loading")
    print("="*70)
    
    # Register test prompts (don't clear - keep defaults)
    PromptRegistry.register('policy', 'concat', None, "Default Concat policy prompt")
    PromptRegistry.register('policy', 'concat', 'math_qa', "Math QA Concat policy prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    policy1 = ConcatPolicy(base_model=None, task_type=None)
    assert policy1.task_prompt_spec == "Default Concat policy prompt"
    print(f"✓ Loaded default prompt: {policy1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    policy2 = ConcatPolicy(base_model=None, task_type='math_qa')
    assert policy2.task_prompt_spec == "Math QA Concat policy prompt"
    print(f"✓ Loaded math_qa prompt: {policy2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit Concat prompt"
    policy3 = ConcatPolicy(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert policy3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {policy3.task_prompt_spec}")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All ConcatPolicy prompt loading tests passed")


def test_env_grounded_policy_prompt_loading():
    """Test EnvGroundedPolicy prompt loading with different methods.
    
    Note: EnvGroundedPolicy requires task_prompt_spec as a required parameter (dict).
    """
    print("\n" + "="*70)
    print("TEST: EnvGroundedPolicy Prompt Loading")
    print("="*70)
    
    # Mock action generator
    def mock_generate_all_actions(env_state):
        return []
    
    # Test 1: Dictionary prompt (required format)
    print("\nTest 1: Dictionary prompt (required format)")
    dict_prompt = {"policy": "Test policy prompt with <init_state>, <goals>, <action>"}
    policy1 = EnvGroundedPolicy(
        base_model=None,
        task_prompt_spec=dict_prompt,
        generate_all_actions=mock_generate_all_actions
    )
    assert policy1.task_prompt_spec == dict_prompt
    print(f"✓ Dictionary prompt works: {list(policy1.task_prompt_spec.keys())}")
    
    # Test 2: Different prompt structure
    print("\nTest 2: Different prompt structure")
    custom_prompt = {
        "policy": "Custom policy template",
        "additional_key": "Additional value"
    }
    policy2 = EnvGroundedPolicy(
        base_model=None,
        task_prompt_spec=custom_prompt,
        generate_all_actions=mock_generate_all_actions
    )
    assert policy2.task_prompt_spec == custom_prompt
    print(f"✓ Custom prompt structure works: {list(policy2.task_prompt_spec.keys())}")
    
    # Test 3: usr_prompt_spec parameter
    print("\nTest 3: usr_prompt_spec parameter")
    main_prompt = {"policy": "Main prompt"}
    usr_prompt = {"user_instruction": "User instruction"}
    policy3 = EnvGroundedPolicy(
        base_model=None,
        task_prompt_spec=main_prompt,
        usr_prompt_spec=usr_prompt,
        generate_all_actions=mock_generate_all_actions
    )
    assert policy3.usr_prompt_spec == usr_prompt
    print(f"✓ usr_prompt_spec set: {policy3.usr_prompt_spec}")
    
    print("\n✓ All EnvGroundedPolicy prompt loading tests passed")


def test_policy_agent_name_inference():
    """Test that agent name inference works for all policy components."""
    print("\n" + "="*70)
    print("TEST: Policy Agent Name Inference")
    print("="*70)
    
    # Create policies with dummy prompts
    rap_policy = RAPPolicy(base_model=None, task_prompt_spec="test")
    tool_policy = ToolUsePolicy(base_model=None, task_prompt_spec="test", tools=[])
    concat_policy = ConcatPolicy(base_model=None, task_prompt_spec="test")
    
    def mock_generate_all_actions(env_state):
        return []
    env_policy = EnvGroundedPolicy(
        base_model=None,
        task_prompt_spec="test",
        generate_all_actions=mock_generate_all_actions
    )
    
    # Test agent name inference
    assert rap_policy._get_agent_name() == 'rap'
    assert tool_policy._get_agent_name() == 'tool_use'
    assert concat_policy._get_agent_name() == 'concat'
    assert env_policy._get_agent_name() == 'env_grounded'
    
    print("✓ RAPPolicy -> 'rap'")
    print("✓ ToolUsePolicy -> 'tool_use'")
    print("✓ ConcatPolicy -> 'concat'")
    print("✓ EnvGroundedPolicy -> 'env_grounded'")
    
    print("\n✓ All policy agent name inference tests passed")


def test_policy_independent_prompt_loading():
    """Test that task_prompt_spec and usr_prompt_spec are loaded independently."""
    print("\n" + "="*70)
    print("TEST: Policy Independent Prompt Loading")
    print("="*70)
    
    # Register both types of prompts
    PromptRegistry.register('policy', 'rap', 'test_task', 'System prompt for test_task')
    PromptRegistry.register_usr('policy', 'rap', 'test_task', {'user_key': 'user_value'})
    
    # Test 1: Both loaded from registry
    print("\nTest 1: Both prompts loaded from registry")
    policy1 = RAPPolicy(base_model=None, task_type='test_task')
    assert policy1.task_prompt_spec == 'System prompt for test_task'
    assert policy1.usr_prompt_spec == {'user_key': 'user_value'}
    print(f"✓ task_prompt_spec: {policy1.task_prompt_spec}")
    print(f"✓ usr_prompt_spec: {policy1.usr_prompt_spec}")
    
    # Test 2: Override task_prompt_spec, usr_prompt_spec from registry
    print("\nTest 2: Override task_prompt_spec, usr_prompt_spec from registry")
    policy2 = RAPPolicy(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system prompt'
    )
    assert policy2.task_prompt_spec == 'Custom system prompt'
    assert policy2.usr_prompt_spec == {'user_key': 'user_value'}
    print(f"✓ task_prompt_spec overridden: {policy2.task_prompt_spec}")
    print(f"✓ usr_prompt_spec from registry: {policy2.usr_prompt_spec}")
    
    # Test 3: Override usr_prompt_spec, task_prompt_spec from registry
    print("\nTest 3: Override usr_prompt_spec, task_prompt_spec from registry")
    policy3 = RAPPolicy(
        base_model=None,
        task_type='test_task',
        usr_prompt_spec={'custom_key': 'custom_value'}
    )
    assert policy3.task_prompt_spec == 'System prompt for test_task'
    assert policy3.usr_prompt_spec == {'custom_key': 'custom_value'}
    print(f"✓ task_prompt_spec from registry: {policy3.task_prompt_spec}")
    print(f"✓ usr_prompt_spec overridden: {policy3.usr_prompt_spec}")
    
    # Test 4: Override both
    print("\nTest 4: Override both prompts")
    policy4 = RAPPolicy(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system',
        usr_prompt_spec={'custom': 'user'}
    )
    assert policy4.task_prompt_spec == 'Custom system'
    assert policy4.usr_prompt_spec == {'custom': 'user'}
    print(f"✓ Both prompts overridden")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All independent prompt loading tests passed")


def run_all_tests():
    """Run all policy component tests."""
    print("\n" + "="*70)
    print("LLM POLICY COMPONENTS UNIT TESTS")
    print("="*70)
    
    tests = [
        test_rap_policy_prompt_loading,
        test_tool_use_policy_prompt_loading,
        test_concat_policy_prompt_loading,
        test_env_grounded_policy_prompt_loading,
        test_policy_agent_name_inference,
        test_policy_independent_prompt_loading,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
