"""
Unit tests for Policy and RewardModel prompt registry integration.

Tests the prompt loading and injection for Policy and RewardModel classes.


```
cd unit_test/components
python test_policy_reward_prompts.py
```
"""

import sys
sys.path.append('../..')

from lits.components.base import Policy, RewardModel
from lits.prompts.registry import PromptRegistry
from lits.structures import State, Step


def test_policy_prompt_loading():
    """Test Policy loads prompts from registry."""
    print("\n" + "="*70)
    print("TEST: Policy Prompt Loading")
    print("="*70)
    
    # Create test policy class
    class TestPolicy(Policy):
        def _get_actions(self, state, n_actions, temperature, **kwargs):
            return [Step() for _ in range(n_actions)]
        
        def _get_agent_name(self):
            return 'test'
    
    # Register test prompts
    default_prompt = "Default test policy prompt"
    math_qa_prompt = "Math QA specific test policy prompt"
    
    PromptRegistry.register('policy', 'test', None, default_prompt)
    PromptRegistry.register('policy', 'test', 'math_qa', math_qa_prompt)
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    policy1 = TestPolicy(
        base_model=None,
        task_type=None
    )
    assert policy1.task_prompt_spec == default_prompt
    print(f"✓ Loaded default prompt: {policy1.task_prompt_spec[:50]}...")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    policy2 = TestPolicy(
        base_model=None,
        task_type='math_qa'
    )
    assert policy2.task_prompt_spec == math_qa_prompt
    print(f"✓ Loaded math_qa prompt: {policy2.task_prompt_spec[:50]}...")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit prompt"
    policy3 = TestPolicy(
        base_model=None,
        task_prompt_spec=custom_prompt,
        task_type='math_qa'
    )
    assert policy3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {policy3.task_prompt_spec}")
    
    # Test 4: usr_prompt_spec parameter
    print("\nTest 4: usr_prompt_spec parameter")
    usr_prompt = {"instruction": "Custom instruction"}
    policy4 = TestPolicy(
        base_model=None,
        usr_prompt_spec=usr_prompt,
        task_type='math_qa'
    )
    assert policy4.usr_prompt_spec == usr_prompt
    print(f"✓ usr_prompt_spec set: {policy4.usr_prompt_spec}")
    
    # Test 5: Both prompts from registry
    print("\nTest 5: Both prompts loaded from registry")
    PromptRegistry.register('policy', 'test', 'both_test', "System prompt")
    PromptRegistry.register_usr('policy', 'test', 'both_test', {'user_template': 'value'})
    policy5 = TestPolicy(
        base_model=None,
        task_type='both_test'
    )
    assert policy5.task_prompt_spec == "System prompt"
    assert policy5.usr_prompt_spec == {'user_template': 'value'}
    print(f"✓ Both prompts loaded from registry")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ All Policy prompt loading tests passed")


def test_reward_model_prompt_loading():
    """Test RewardModel loads prompts from registry."""
    print("\n" + "="*70)
    print("TEST: RewardModel Prompt Loading")
    print("="*70)
    
    # Create test reward model class
    class TestRewardModel(RewardModel):
        def _fast_reward(self, example, example_idx, state, action, from_phase=""):
            return 0.5
        
        def calculate_reward(self, useful_prob):
            return useful_prob
        
        def reward(self, state, action, **kwargs):
            return 0.5, {}
        
        def _get_agent_name(self):
            return 'test'
    
    # Register test prompts
    default_prompt = "Default test reward prompt"
    math_qa_prompt = "Math QA specific test reward prompt"
    
    PromptRegistry.register('reward', 'test', None, default_prompt)
    PromptRegistry.register('reward', 'test', 'math_qa', math_qa_prompt)
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    reward1 = TestRewardModel(
        base_model=None,
        task_type=None
    )
    assert reward1.task_prompt_spec == default_prompt
    print(f"✓ Loaded default prompt: {reward1.task_prompt_spec[:50]}...")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    reward2 = TestRewardModel(
        base_model=None,
        task_type='math_qa'
    )
    assert reward2.task_prompt_spec == math_qa_prompt
    print(f"✓ Loaded math_qa prompt: {reward2.task_prompt_spec[:50]}...")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = {"eval_instruction": "Custom evaluation"}
    reward3 = TestRewardModel(
        base_model=None,
        task_prompt_spec=custom_prompt,
        task_type='math_qa'
    )
    assert reward3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {reward3.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ All RewardModel prompt loading tests passed")


def test_agent_name_inference():
    """Test agent name inference for Policy and RewardModel."""
    print("\n" + "="*70)
    print("TEST: Agent Name Inference")
    print("="*70)
    
    # Test Policy name inference
    class RAPPolicy(Policy):
        def _get_actions(self, state, n_actions, temperature, **kwargs):
            return [Step() for _ in range(n_actions)]
    
    class ReStPolicy(Policy):
        def _get_actions(self, state, n_actions, temperature, **kwargs):
            return [Step() for _ in range(n_actions)]
    
    class ToolUsePolicy(Policy):
        def _get_actions(self, state, n_actions, temperature, **kwargs):
            return [Step() for _ in range(n_actions)]
    
    rap_policy = RAPPolicy(base_model=None, task_prompt_spec="test")
    rest_policy = ReStPolicy(base_model=None, task_prompt_spec="test")
    tool_policy = ToolUsePolicy(base_model=None, task_prompt_spec="test")
    
    assert rap_policy._get_agent_name() == 'rap'
    assert rest_policy._get_agent_name() == 're_st'  # CamelCase conversion
    assert tool_policy._get_agent_name() == 'tool_use'
    
    print("✓ RAPPolicy -> 'rap'")
    print("✓ ReStPolicy -> 're_st'")
    print("✓ ToolUsePolicy -> 'tool_use'")
    
    # Test RewardModel name inference
    class RapPRM(RewardModel):
        def _fast_reward(self, example, example_idx, state, action, from_phase=""):
            return 0.5
        def calculate_reward(self, useful_prob):
            return useful_prob
        def reward(self, state, action, **kwargs):
            return 0.5, {}
    
    class GenerativePRM(RewardModel):
        def _fast_reward(self, example, example_idx, state, action, from_phase=""):
            return 0.5
        def calculate_reward(self, useful_prob):
            return useful_prob
        def reward(self, state, action, **kwargs):
            return 0.5, {}
    
    rap_prm = RapPRM(base_model=None, task_prompt_spec="test")
    gen_prm = GenerativePRM(base_model=None, task_prompt_spec="test")
    
    assert rap_prm._get_agent_name() == 'rap'
    assert gen_prm._get_agent_name() == 'generative'
    
    print("✓ RapPRM -> 'rap'")
    print("✓ GenerativePRM -> 'generative'")
    
    print("\n✓ Agent name inference tests passed")


def test_prompt_injection():
    """Test that users can inject custom prompts."""
    print("\n" + "="*70)
    print("TEST: Prompt Injection")
    print("="*70)
    
    class MyPolicy(Policy):
        def _get_actions(self, state, n_actions, temperature, **kwargs):
            return [Step() for _ in range(n_actions)]
        def _get_agent_name(self):
            return 'my_agent'
    
    # Method 1: Direct injection
    print("\nMethod 1: Direct injection via task_prompt_spec")
    custom_prompt1 = "My custom prompt for specific instance"
    policy1 = MyPolicy(
        base_model=None,
        task_prompt_spec=custom_prompt1
    )
    assert policy1.task_prompt_spec == custom_prompt1
    print(f"✓ Direct injection works: {policy1.task_prompt_spec}")
    
    # Method 2: Registry injection
    print("\nMethod 2: Registry injection for all instances")
    custom_prompt2 = "My custom prompt for all instances"
    PromptRegistry.register('policy', 'my_agent', 'my_task', custom_prompt2)
    
    policy2 = MyPolicy(
        base_model=None,
        task_type='my_task'
    )
    assert policy2.task_prompt_spec == custom_prompt2
    print(f"✓ Registry injection works: {policy2.task_prompt_spec}")
    
    # Method 3: Override registry with direct
    print("\nMethod 3: Direct overrides registry")
    override_prompt = "Override prompt"
    policy3 = MyPolicy(
        base_model=None,
        task_prompt_spec=override_prompt,
        task_type='my_task'
    )
    assert policy3.task_prompt_spec == override_prompt
    print(f"✓ Direct overrides registry: {policy3.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ Prompt injection tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("POLICY & REWARD MODEL PROMPT TESTS")
    print("="*70)
    
    tests = [
        test_policy_prompt_loading,
        test_reward_model_prompt_loading,
        test_agent_name_inference,
        test_prompt_injection,
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
