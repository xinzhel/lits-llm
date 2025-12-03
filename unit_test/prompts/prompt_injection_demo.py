"""
Demonstration of the Prompt Injection System in LITS-LLM

This example shows different ways to inject custom prompts into
Policy, RewardModel, and Transition components.
"""

import sys
sys.path.append('..')

from lits.lm import get_lm
from lits.components.policy.rap import RAPPolicy
from lits.components.reward.rap import RapPRM
from lits.components.transition.rap import RAPTransition
from lits.prompts.registry import PromptRegistry


def demo_direct_injection():
    """Demo 1: Direct prompt injection via task_prompt_spec"""
    print("\n" + "="*70)
    print("DEMO 1: Direct Prompt Injection")
    print("="*70)
    
    # Note: Using None for base_model in demo (would use actual model in production)
    model = None
    
    # Direct string prompt
    custom_prompt = "Think carefully and solve this problem step by step."
    policy = RAPPolicy(
        base_model=model,
        task_prompt_spec=custom_prompt
    )
    
    print(f"✓ Policy prompt: {policy.task_prompt_spec}")
    
    # Direct dictionary prompt
    custom_reward_prompt = {
        'eval_instruction': 'Evaluate if this reasoning step is correct',
        'output_format': 'Confidence: [0.0-1.0]'
    }
    reward_model = RapPRM(
        base_model=model,
        task_prompt_spec=custom_reward_prompt
    )
    
    print(f"✓ RewardModel prompt: {reward_model.task_prompt_spec}")


def demo_registry_injection():
    """Demo 2: Registry-based prompt injection"""
    print("\n" + "="*70)
    print("DEMO 2: Registry-Based Prompt Injection")
    print("="*70)
    
    # Clear registry first
    PromptRegistry.clear()
    
    # Register default prompts
    PromptRegistry.register(
        component_type='policy',
        agent_name='rap',
        task_type=None,
        prompt='Default RAP policy prompt for all tasks'
    )
    
    # Register task-specific prompts
    PromptRegistry.register(
        component_type='policy',
        agent_name='rap',
        task_type='math_qa',
        prompt='Specialized RAP prompt for math QA tasks'
    )
    
    PromptRegistry.register(
        component_type='reward',
        agent_name='rap',
        task_type='math_qa',
        prompt='Evaluate mathematical reasoning correctness'
    )
    
    model = None
    
    # Load default prompt
    policy1 = RAPPolicy(base_model=model, task_type=None)
    print(f"✓ Default policy prompt: {policy1.task_prompt_spec}")
    
    # Load task-specific prompt
    policy2 = RAPPolicy(base_model=model, task_type='math_qa')
    print(f"✓ Math QA policy prompt: {policy2.task_prompt_spec}")
    
    # Load reward model prompt
    reward = RapPRM(base_model=model, task_type='math_qa')
    print(f"✓ Math QA reward prompt: {reward.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()


def demo_priority_system():
    """Demo 3: Prompt priority system"""
    print("\n" + "="*70)
    print("DEMO 3: Prompt Priority System")
    print("="*70)
    
    # Register a default prompt
    PromptRegistry.register(
        component_type='policy',
        agent_name='rap',
        task_type='math_qa',
        prompt='Registry prompt for math QA'
    )
    
    model = None
    
    # Priority 1: Direct task_prompt_spec (highest)
    policy1 = RAPPolicy(
        base_model=model,
        task_type='math_qa',
        task_prompt_spec='Direct prompt overrides registry'
    )
    print(f"✓ Priority 1 (direct): {policy1.task_prompt_spec}")
    
    # Priority 2: usr_prompt_spec
    policy2 = RAPPolicy(
        base_model=model,
        task_type='math_qa',
        usr_prompt_spec='Alternative parameter prompt'
    )
    print(f"✓ Priority 2 (usr_prompt_spec): {policy2.task_prompt_spec}")
    
    # Priority 3: Registry (task-specific)
    policy3 = RAPPolicy(
        base_model=model,
        task_type='math_qa'
    )
    print(f"✓ Priority 3 (registry): {policy3.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()


def demo_agent_name_inference():
    """Demo 4: Automatic agent name inference"""
    print("\n" + "="*70)
    print("DEMO 4: Agent Name Inference")
    print("="*70)
    
    model = None
    
    # Create components with dummy prompts
    rap_policy = RAPPolicy(base_model=model, task_prompt_spec="test")
    rap_reward = RapPRM(base_model=model, task_prompt_spec="test")
    rap_transition = RAPTransition(base_model=model, task_prompt_spec="test")
    
    # Show inferred agent names
    print(f"✓ RAPPolicy -> '{rap_policy._get_agent_name()}'")
    print(f"✓ RapPRM -> '{rap_reward._get_agent_name()}'")
    print(f"✓ RAPTransition -> '{rap_transition._get_agent_name()}'")
    
    print("\nAgent names are automatically inferred from class names:")
    print("  - RAPPolicy -> 'rap'")
    print("  - ToolUsePolicy -> 'tool_use'")
    print("  - GenerativePRM -> 'generative'")
    print("  - SelfConsistencyRM -> 'self_consistency'")


def demo_multi_component_workflow():
    """Demo 5: Complete workflow with multiple components"""
    print("\n" + "="*70)
    print("DEMO 5: Multi-Component Workflow")
    print("="*70)
    
    # Clear and setup registry
    PromptRegistry.clear()
    
    # Register prompts for a complete math QA workflow
    PromptRegistry.register(
        component_type='policy',
        agent_name='rap',
        task_type='math_qa',
        prompt='Generate reasoning steps to solve the math problem'
    )
    
    PromptRegistry.register(
        component_type='reward',
        agent_name='rap',
        task_type='math_qa',
        prompt='Evaluate if the reasoning step is mathematically sound'
    )
    
    PromptRegistry.register(
        component_type='transition',
        agent_name='rap',
        task_type='math_qa',
        prompt='Update the problem state with the new reasoning step'
    )
    
    model = None
    
    # Create all components with task_type
    policy = RAPPolicy(base_model=model, task_type='math_qa')
    reward_model = RapPRM(base_model=model, task_type='math_qa')
    transition = RAPTransition(base_model=model, task_type='math_qa')
    
    print("✓ Created complete RAP workflow for math_qa:")
    print(f"  Policy prompt: {policy.task_prompt_spec}")
    print(f"  Reward prompt: {reward_model.task_prompt_spec}")
    print(f"  Transition prompt: {transition.task_prompt_spec}")
    
    # Clean up
    PromptRegistry.clear()


def demo_backward_compatibility():
    """Demo 6: Backward compatibility with legacy code"""
    print("\n" + "="*70)
    print("DEMO 6: Backward Compatibility")
    print("="*70)
    
    model = None
    
    # Old style: task_instruction parameter (Policy only)
    policy = RAPPolicy(
        base_model=model,
        task_instruction="Legacy task instruction parameter"
    )
    
    print(f"✓ Legacy task_instruction: {policy.task_instruction}")
    print(f"✓ Also available as task_prompt_spec: {policy.task_prompt_spec}")
    print("\nOld code using task_instruction still works!")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("LITS-LLM PROMPT INJECTION SYSTEM DEMO")
    print("="*70)
    
    demos = [
        demo_direct_injection,
        demo_registry_injection,
        demo_priority_system,
        demo_agent_name_inference,
        demo_multi_component_workflow,
        demo_backward_compatibility,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Direct injection (task_prompt_spec) has highest priority")
    print("2. Registry allows sharing prompts across components")
    print("3. task_type enables task-specific prompt loading")
    print("4. Agent names are automatically inferred from class names")
    print("5. Backward compatibility maintained with legacy parameters")
    print("\nFor more details, see:")
    print("  - README.md (Prompt Injection System section)")
    print("  - docs/PROMPT_INJECTION_DESIGN.md")
    print("  - unit_test/test_policy_reward_prompts.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
