"""
Unit tests for LlmTransition and its subclasses.

Tests the prompt registry integration and parameter handling for:
- BlocksWorldTransition
- RAPTransition
"""

import sys
sys.path.append('../..')

from lits_benchmark.blocksworld import BlocksWorldTransition
from lits.components.transition.rap import RAPTransition
from lits.prompts.registry import PromptRegistry


def test_prompt_registry_integration():
    """Test that PromptRegistry is properly integrated."""
    print("\n" + "="*70)
    print("TEST: Prompt Registry Integration")
    print("="*70)
    
    # Register a test prompt
    test_prompt = {"test_key": "test_value"}
    PromptRegistry.register('transition', 'rap', 'test_task', test_prompt)
    
    # Verify it was registered
    retrieved = PromptRegistry.get('transition', 'rap', 'test_task')
    assert retrieved == test_prompt, f"Expected {test_prompt}, got {retrieved}"
    print("✓ Prompt registration and retrieval works")
    
    # Test listing
    all_prompts = PromptRegistry.list_registered('transition')
    assert 'rap' in all_prompts, "RAP should be in registered prompts"
    print(f"✓ Registered transition prompts: {list(all_prompts.keys())}")
    
    # Clean up
    PromptRegistry.clear()
    print("✓ Registry cleared")


def test_rap_transition_prompt_loading():
    """Test RAPTransition loads prompts correctly."""
    print("\n" + "="*70)
    print("TEST: RAPTransition Prompt Loading")
    print("="*70)
    
    # Register test prompts
    default_prompt = "Default RAP transition prompt"
    math_qa_prompt = "Math QA specific RAP transition prompt"
    
    PromptRegistry.register('transition', 'rap', None, default_prompt)
    PromptRegistry.register('transition', 'rap', 'math_qa', math_qa_prompt)
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt (task_type=None)")
    transition1 = RAPTransition(
        base_model=None,  # Mock model
        task_type=None
    )
    assert transition1.task_prompt_spec == default_prompt
    print(f"✓ Loaded default prompt: {transition1.task_prompt_spec[:50]}...")
    
    # Test 2: Load task-specific prompt
    print("\nTest 2: Load task-specific prompt (task_type='math_qa')")
    transition2 = RAPTransition(
        base_model=None,
        task_type='math_qa'
    )
    assert transition2.task_prompt_spec == math_qa_prompt
    print(f"✓ Loaded math_qa prompt: {transition2.task_prompt_spec[:50]}...")
    
    # Test 3: Explicit prompt overrides registry
    print("\nTest 3: Explicit prompt overrides registry")
    custom_prompt = "Custom explicit prompt"
    transition3 = RAPTransition(
        base_model=None,
        task_prompt_spec=custom_prompt,
        task_type='math_qa'
    )
    assert transition3.task_prompt_spec == custom_prompt
    print(f"✓ Explicit prompt takes precedence: {transition3.task_prompt_spec}")
    
    # Test 4: usr_prompt_spec parameter
    print("\nTest 4: usr_prompt_spec parameter")
    usr_prompt = {"subquestion_prefix": "Q{idx}.{sub_idx}: ", "answer_prefix": "A{idx}.{sub_idx}: "}
    transition4 = RAPTransition(
        base_model=None,
        usr_prompt_spec=usr_prompt,
        task_type='math_qa'
    )
    assert transition4.usr_prompt_spec == usr_prompt
    print(f"✓ usr_prompt_spec set: {transition4.usr_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ All RAPTransition prompt loading tests passed")


def test_blocksworld_transition_prompt_loading():
    """Test BlocksWorldTransition loads prompts correctly."""
    print("\n" + "="*70)
    print("TEST: BlocksWorldTransition Prompt Loading")
    print("="*70)
    
    # Register test prompts
    default_prompt = {
        'world_update_pickup': 'Pickup template',
        'world_update_putdown': 'Putdown template'
    }
    
    PromptRegistry.register('transition', 'blocksworld', None, default_prompt)
    
    # Mock goal_check function
    def mock_goal_check(goals, env_state):
        return False, 0.0
    
    # Test 1: Load default prompt
    print("\nTest 1: Load default prompt")
    transition = BlocksWorldTransition(
        base_model=None,
        goal_check=mock_goal_check,
        task_type=None
    )
    assert transition.task_prompt_spec == default_prompt
    print(f"✓ Loaded default prompt with keys: {list(transition.task_prompt_spec.keys())}")
    
    # Test 2: Explicit prompt overrides
    print("\nTest 2: Explicit prompt overrides")
    custom_prompt = {'world_update_pickup': 'Custom pickup'}
    transition2 = BlocksWorldTransition(
        base_model=None,
        goal_check=mock_goal_check,
        task_prompt_spec=custom_prompt
    )
    assert transition2.task_prompt_spec == custom_prompt
    print(f"✓ Custom prompt used: {transition2.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ All BlocksWorldTransition prompt loading tests passed")


def test_agent_name_inference():
    """Test that agent names are correctly inferred from class names."""
    print("\n" + "="*70)
    print("TEST: Agent Name Inference")
    print("="*70)
    
    # Register prompts with inferred names
    PromptRegistry.register('transition', 'rap', None, "RAP prompt")
    PromptRegistry.register('transition', 'blocksworld', None, "BlocksWorld prompt")
    
    # Mock goal_check
    def mock_goal_check(goals, env_state):
        return False, 0.0
    
    # Test RAPTransition -> 'rap'
    rap_trans = RAPTransition(base_model=None)
    assert rap_trans._get_agent_name() == 'rap'
    print("✓ RAPTransition -> 'rap'")
    
    # Test BlocksWorldTransition -> 'blocksworld'
    bw_trans = BlocksWorldTransition(base_model=None, goal_check=mock_goal_check)
    assert bw_trans._get_agent_name() == 'blocksworld'
    print("✓ BlocksWorldTransition -> 'blocksworld'")
    
    # Clean up
    PromptRegistry.clear()
    print("\n✓ Agent name inference tests passed")


def test_transition_independent_prompt_loading():
    """Test that task_prompt_spec and usr_prompt_spec are loaded independently."""
    print("\n" + "="*70)
    print("TEST: Transition Independent Prompt Loading")
    print("="*70)
    
    # Register both types of prompts
    PromptRegistry.register('transition', 'rap', 'test_task', 'System prompt for test_task')
    PromptRegistry.register_usr('transition', 'rap', 'test_task', {'user_key': 'user_value'})
    
    # Test 1: Both loaded from registry
    print("\nTest 1: Both prompts loaded from registry")
    transition1 = RAPTransition(base_model=None, task_type='test_task')
    assert transition1.task_prompt_spec == 'System prompt for test_task'
    assert transition1.usr_prompt_spec == {'user_key': 'user_value'}
    print(f"✓ task_prompt_spec: {transition1.task_prompt_spec}")
    print(f"✓ usr_prompt_spec: {transition1.usr_prompt_spec}")
    
    # Test 2: Override task_prompt_spec, usr_prompt_spec from registry
    print("\nTest 2: Override task_prompt_spec, usr_prompt_spec from registry")
    transition2 = RAPTransition(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system prompt'
    )
    assert transition2.task_prompt_spec == 'Custom system prompt'
    assert transition2.usr_prompt_spec == {'user_key': 'user_value'}
    print(f"✓ task_prompt_spec overridden: {transition2.task_prompt_spec}")
    print(f"✓ usr_prompt_spec from registry: {transition2.usr_prompt_spec}")
    
    # Test 3: Override usr_prompt_spec, task_prompt_spec from registry
    print("\nTest 3: Override usr_prompt_spec, task_prompt_spec from registry")
    transition3 = RAPTransition(
        base_model=None,
        task_type='test_task',
        usr_prompt_spec={'custom_key': 'custom_value'}
    )
    assert transition3.task_prompt_spec == 'System prompt for test_task'
    assert transition3.usr_prompt_spec == {'custom_key': 'custom_value'}
    print(f"✓ task_prompt_spec from registry: {transition3.task_prompt_spec}")
    print(f"✓ usr_prompt_spec overridden: {transition3.usr_prompt_spec}")
    
    # Test 4: Override both
    print("\nTest 4: Override both prompts")
    transition4 = RAPTransition(
        base_model=None,
        task_type='test_task',
        task_prompt_spec='Custom system',
        usr_prompt_spec={'custom': 'user'}
    )
    assert transition4.task_prompt_spec == 'Custom system'
    assert transition4.usr_prompt_spec == {'custom': 'user'}
    print(f"✓ Both prompts overridden")
    
    # Clean up
    PromptRegistry.clear()
    load_default_prompts()
    print("\n✓ All independent prompt loading tests passed")


def test_backward_compatibility():
    """Test that old code still works."""
    print("\n" + "="*70)
    print("TEST: Backward Compatibility")
    print("="*70)
    
    # Old style: passing task_prompt_spec directly
    prompt = "Old style prompt"
    transition = RAPTransition(
        base_model=None,
        task_prompt_spec=prompt
    )
    assert transition.task_prompt_spec == prompt
    print("✓ Old style task_prompt_spec parameter still works")
    
    # Old style: usr_msg_dict (now usr_prompt_spec)
    usr_dict = {"key": "value"}
    transition2 = RAPTransition(
        base_model=None,
        usr_prompt_spec=usr_dict
    )
    assert transition2.usr_prompt_spec == usr_dict
    print("✓ usr_prompt_spec parameter works")
    
    print("\n✓ Backward compatibility maintained")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("LLM TRANSITION UNIT TESTS")
    print("="*70)
    
    tests = [
        test_prompt_registry_integration,
        test_rap_transition_prompt_loading,
        test_blocksworld_transition_prompt_loading,
        test_agent_name_inference,
        test_transition_independent_prompt_loading,
        test_backward_compatibility,
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
