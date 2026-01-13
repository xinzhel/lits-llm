"""Unit tests for Crosswords prompt registration.

Tests that crosswords-specific prompts are correctly registered and retrievable
via PromptRegistry for EnvGroundedPolicy and EnvGroundedPRM.

Run:
    python unit_test/prompts/test_crosswords_prompts.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import crosswords module to trigger prompt registration
import lits_benchmark.crosswords

from lits.prompts.registry import PromptRegistry
from lits.prompts.prompt import PromptTemplate


def test_policy_user_prompt_registration():
    """Test that policy user prompt is registered as PromptTemplate."""
    print("\n" + "="*60)
    print("TEST: Policy User Prompt Registration")
    print("="*60)
    
    policy_usr = PromptRegistry.get_usr('policy', 'env_grounded', task_name='crosswords')
    assert policy_usr is not None, 'Policy user prompt not found'
    assert isinstance(policy_usr, PromptTemplate), f'Expected PromptTemplate, got {type(policy_usr)}'
    print('✓ Policy user prompt registered as PromptTemplate')
    
    # Test formatting
    formatted = policy_usr.format(
        init_state='Current Board:\n_____\n_____\n_____\n_____\n_____',
        goals='AGEND\nMOTOR\nARTSY\nSALLE\nSLEER',
        actions='h1. _____\nh2. _____'
    )
    assert 'Current Board' in formatted
    assert 'h1. _____' in formatted
    print('✓ Policy user prompt formats correctly')
    
    print("\n✓ Policy user prompt test passed")


def test_reward_system_prompt_registration():
    """Test that reward system prompt is registered."""
    print("\n" + "="*60)
    print("TEST: Reward System Prompt Registration")
    print("="*60)
    
    reward_sys = PromptRegistry.get('reward', 'env_grounded', task_name='crosswords')
    assert reward_sys is not None, 'Reward system prompt not found'
    assert isinstance(reward_sys, str), f'Expected str, got {type(reward_sys)}'
    assert 'EVALUATION CRITERIA' in reward_sys
    assert 'good' in reward_sys.lower()
    assert 'bad' in reward_sys.lower()
    print('✓ Reward system prompt registered')
    print('✓ Contains evaluation criteria')
    
    print("\n✓ Reward system prompt test passed")


def test_reward_user_prompt_registration():
    """Test that reward user prompt is registered with correct placeholders."""
    print("\n" + "="*60)
    print("TEST: Reward User Prompt Registration")
    print("="*60)
    
    reward_usr = PromptRegistry.get_usr('reward', 'env_grounded', task_name='crosswords')
    assert reward_usr is not None, 'Reward user prompt not found'
    assert isinstance(reward_usr, str), f'Expected str, got {type(reward_usr)}'
    assert '<init_state>' in reward_usr, 'Missing <init_state> placeholder'
    assert '<action>' in reward_usr, 'Missing <action> placeholder'
    print('✓ Reward user prompt registered')
    print('✓ Contains required placeholders')
    
    print("\n✓ Reward user prompt test passed")


def test_prompt_lookup_by_task_name():
    """Test that prompts can be looked up by task_name='crosswords'."""
    print("\n" + "="*60)
    print("TEST: Prompt Lookup by Task Name")
    print("="*60)
    
    # Policy prompts
    policy_usr = PromptRegistry.get_usr('policy', 'env_grounded', task_name='crosswords')
    assert policy_usr is not None, 'Policy user prompt lookup failed'
    print('✓ Policy user prompt found via task_name=crosswords')
    
    # Reward prompts
    reward_sys = PromptRegistry.get('reward', 'env_grounded', task_name='crosswords')
    assert reward_sys is not None, 'Reward system prompt lookup failed'
    print('✓ Reward system prompt found via task_name=crosswords')
    
    reward_usr = PromptRegistry.get_usr('reward', 'env_grounded', task_name='crosswords')
    assert reward_usr is not None, 'Reward user prompt lookup failed'
    print('✓ Reward user prompt found via task_name=crosswords')
    
    print("\n✓ Prompt lookup test passed")


if __name__ == "__main__":
    test_policy_user_prompt_registration()
    test_reward_system_prompt_registration()
    test_reward_user_prompt_registration()
    test_prompt_lookup_by_task_name()
    
    print("\n" + "="*60)
    print("✓ All crosswords prompt tests passed!")
    print("="*60)
