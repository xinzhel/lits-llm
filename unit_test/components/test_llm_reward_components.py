"""
Unit tests for all LLM-based RewardModel components with prompt injection system.

Tests all reward model implementations in lits.components.reward:
- RapPRM
- GenerativePRM
- SelfConsistencyRM
- RLHFlowPRM
"""

import sys
sys.path.append('../..')

from lits.components.reward.rap import RapPRM
from lits.components.reward.generative import GenerativePRM
from lits.components.reward.sc import SelfConsistencyRM
from lits.components.reward.rlhflow import RLHFlowPRM
from lits.prompts.registry import PromptRegistry, load_default_prompts


def test_rap_prm_prompt_loading():
    """Test RapPRM prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: RapPRM Prompt Loading")
    print("="*70)
    
    # Clear registry
    PromptRegistry.clear()
    
    # Register test prompts
    PromptRegistry.register('reward', 'rap', None, "Default RAP reward prompt")
    PromptRegistry.register('reward', 'rap', 'math_qa', "Math QA RAP reward prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    reward1 = RapPRM(base_model=None, task_type=None)
    assert reward1.task_prompt_spec == "Default RAP reward prompt"
    print(f"✓ Loaded default prompt: {reward1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    reward2 = RapPRM(base_model=None, task_type='math_qa')
    assert reward2.task_prompt_spec == "Math QA RAP reward prompt"
    print(f"✓ Loaded math_qa prompt: {reward2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = {'eval_instruction': 'Custom evaluation instruction'}
    reward3 = RapPRM(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert reward3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {reward3.task_prompt_spec}")
    
    # Test 4: usr_prompt_spec parameter
    print("\nTest 4: usr_prompt_spec parameter")
    usr_prompt = {'user_instruction': 'Custom user instruction'}
    reward4 = RapPRM(base_model=None, usr_prompt_spec=usr_prompt)
    assert reward4.usr_prompt_spec == usr_prompt
    print(f"✓ usr_prompt_spec set: {reward4.usr_prompt_spec}")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All RapPRM prompt loading tests passed")


def test_generative_prm_prompt_loading():
    """Test GenerativePRM prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: GenerativePRM Prompt Loading")
    print("="*70)
    
    # Clear registry
    PromptRegistry.clear()
    
    # Register test prompts
    PromptRegistry.register('reward', 'generative', None, "Default Generative reward prompt")
    PromptRegistry.register('reward', 'generative', 'math_qa', "Math QA Generative reward prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    reward1 = GenerativePRM(base_model=None, task_type=None)
    assert reward1.task_prompt_spec == "Default Generative reward prompt"
    print(f"✓ Loaded default prompt: {reward1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    reward2 = GenerativePRM(base_model=None, task_type='math_qa')
    assert reward2.task_prompt_spec == "Math QA Generative reward prompt"
    print(f"✓ Loaded math_qa prompt: {reward2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit Generative prompt"
    reward3 = GenerativePRM(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert reward3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {reward3.task_prompt_spec}")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All GenerativePRM prompt loading tests passed")


def test_self_consistency_rm_prompt_loading():
    """Test SelfConsistencyRM prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: SelfConsistencyRM Prompt Loading")
    print("="*70)
    
    # Clear registry
    PromptRegistry.clear()
    
    # Register test prompts
    PromptRegistry.register('reward', 'self_consistency', None, "Default SC reward prompt")
    PromptRegistry.register('reward', 'self_consistency', 'math_qa', "Math QA SC reward prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    reward1 = SelfConsistencyRM(base_model=None, task_type=None)
    assert reward1.task_prompt_spec == "Default SC reward prompt"
    print(f"✓ Loaded default prompt: {reward1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    reward2 = SelfConsistencyRM(base_model=None, task_type='math_qa')
    assert reward2.task_prompt_spec == "Math QA SC reward prompt"
    print(f"✓ Loaded math_qa prompt: {reward2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit SC prompt"
    reward3 = SelfConsistencyRM(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert reward3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {reward3.task_prompt_spec}")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All SelfConsistencyRM prompt loading tests passed")


def test_rlhflow_prm_prompt_loading():
    """Test RLHFlowPRM prompt loading with different methods."""
    print("\n" + "="*70)
    print("TEST: RLHFlowPRM Prompt Loading")
    print("="*70)
    
    # Clear registry
    PromptRegistry.clear()
    
    # Register test prompts
    PromptRegistry.register('reward', 'rlhflow', None, "Default RLHFlow reward prompt")
    PromptRegistry.register('reward', 'rlhflow', 'math_qa', "Math QA RLHFlow reward prompt")
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    reward1 = RLHFlowPRM(base_model=None, task_type=None)
    assert reward1.task_prompt_spec == "Default RLHFlow reward prompt"
    print(f"✓ Loaded default prompt: {reward1.task_prompt_spec}")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    reward2 = RLHFlowPRM(base_model=None, task_type='math_qa')
    assert reward2.task_prompt_spec == "Math QA RLHFlow reward prompt"
    print(f"✓ Loaded math_qa prompt: {reward2.task_prompt_spec}")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit RLHFlow prompt"
    reward3 = RLHFlowPRM(base_model=None, task_type='math_qa', task_prompt_spec=custom_prompt)
    assert reward3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {reward3.task_prompt_spec}")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All RLHFlowPRM prompt loading tests passed")


def test_reward_agent_name_inference():
    """Test that agent name inference works for all reward model components."""
    print("\n" + "="*70)
    print("TEST: RewardModel Agent Name Inference")
    print("="*70)
    
    # Create reward models with dummy prompts
    rap_reward = RapPRM(base_model=None, task_prompt_spec="test")
    gen_reward = GenerativePRM(base_model=None, task_prompt_spec="test")
    sc_reward = SelfConsistencyRM(base_model=None, task_prompt_spec="test")
    rlhflow_reward = RLHFlowPRM(base_model=None, task_prompt_spec="test")
    
    # Test agent name inference
    assert rap_reward._get_agent_name() == 'rap'
    assert gen_reward._get_agent_name() == 'generative'
    assert sc_reward._get_agent_name() == 'self_consistency'
    assert rlhflow_reward._get_agent_name() == 'rlhflow'
    
    print("✓ RapPRM -> 'rap'")
    print("✓ GenerativePRM -> 'generative'")
    print("✓ SelfConsistencyRM -> 'self_consistency'")
    print("✓ RLHFlowPRM -> 'rlhflow'")
    
    print("\n✓ All reward model agent name inference tests passed")


def test_reward_independent_prompt_loading():
    """Test that task_prompt_spec and usr_prompt_spec are loaded independently."""
    print("\n" + "="*70)
    print("TEST: RewardModel Independent Prompt Loading")
    print("="*70)
    
    # Register both types of prompts
    PromptRegistry.register('reward', 'rap', 'test_task', 'System prompt for test_task')
    PromptRegistry.register_usr('reward', 'rap', 'test_task', {'eval_key': 'eval_value'})
    
    # Test 1: Both loaded from registry
    print("\nTest 1: Both prompts loaded from registry")
    reward1 = RapPRM(base_model=None, task_type='test_task')
    assert reward1.task_prompt_spec == 'System prompt for test_task'
    assert reward1.usr_prompt_spec == {'eval_key': 'eval_value'}
    print(f"✓ task_prompt_spec: {reward1.task_prompt_spec}")
    print(f"✓ usr_prompt_spec: {reward1.usr_prompt_spec}")
    
    # Test 2: Override task_prompt_spec, usr_prompt_spec from registry
    print("\nTest 2: Override task_prompt_spec, usr_prompt_spec from registry")
    reward2 = RapPRM(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system prompt'
    )
    assert reward2.task_prompt_spec == 'Custom system prompt'
    assert reward2.usr_prompt_spec == {'eval_key': 'eval_value'}
    print(f"✓ task_prompt_spec overridden: {reward2.task_prompt_spec}")
    print(f"✓ usr_prompt_spec from registry: {reward2.usr_prompt_spec}")
    
    # Test 3: Override usr_prompt_spec, task_prompt_spec from registry
    print("\nTest 3: Override usr_prompt_spec, task_prompt_spec from registry")
    reward3 = RapPRM(
        base_model=None,
        task_type='test_task',
        usr_prompt_spec={'custom_eval': 'custom_value'}
    )
    assert reward3.task_prompt_spec == 'System prompt for test_task'
    assert reward3.usr_prompt_spec == {'custom_eval': 'custom_value'}
    print(f"✓ task_prompt_spec from registry: {reward3.task_prompt_spec}")
    print(f"✓ usr_prompt_spec overridden: {reward3.usr_prompt_spec}")
    
    # Test 4: Override both
    print("\nTest 4: Override both prompts")
    reward4 = RapPRM(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system',
        usr_prompt_spec={'custom': 'eval'}
    )
    assert reward4.task_prompt_spec == 'Custom system'
    assert reward4.usr_prompt_spec == {'custom': 'eval'}
    print(f"✓ Both prompts overridden")
    
    # Clean up and reload defaults
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All independent prompt loading tests passed")


def test_reward_structured_prompts():
    """Test that reward models support structured (dict) prompts."""
    print("\n" + "="*70)
    print("TEST: RewardModel Structured Prompts")
    print("="*70)
    
    # Test with dictionary prompts
    print("\nTest 1: Dictionary prompt for RapPRM")
    dict_prompt = {
        'eval_instruction': 'Evaluate the reasoning step',
        'output_format': 'Score: [0.0-1.0]',
        'criteria': ['correctness', 'usefulness']
    }
    reward1 = RapPRM(base_model=None, task_prompt_spec=dict_prompt)
    assert reward1.task_prompt_spec == dict_prompt
    assert 'eval_instruction' in reward1.task_prompt_spec
    print(f"✓ Dictionary prompt works: {list(reward1.task_prompt_spec.keys())}")
    
    # Test with GenerativePRM
    print("\nTest 2: Dictionary prompt for GenerativePRM")
    gen_prompt = {
        'correctness_instruction': 'Check if the step is correct',
        'usefulness_instruction': 'Check if the step is useful'
    }
    reward2 = GenerativePRM(base_model=None, task_prompt_spec=gen_prompt)
    assert reward2.task_prompt_spec == gen_prompt
    print(f"✓ Dictionary prompt works: {list(reward2.task_prompt_spec.keys())}")
    
    print("\n✓ All structured prompt tests passed")


def run_all_tests():
    """Run all reward model component tests."""
    print("\n" + "="*70)
    print("LLM REWARD MODEL COMPONENTS UNIT TESTS")
    print("="*70)
    
    tests = [
        test_rap_prm_prompt_loading,
        test_generative_prm_prompt_loading,
        test_self_consistency_rm_prompt_loading,
        test_rlhflow_prm_prompt_loading,
        test_reward_agent_name_inference,
        test_reward_independent_prompt_loading,
        test_reward_structured_prompts,
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
