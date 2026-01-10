"""Unified Registry API for LiTS framework.

This module provides a single import entry point for all registry decorators
and functions across the LiTS framework. It re-exports:

1. Component registration decorators (from lits.components.registry):
   - register_transition: Register Transition classes
   - register_policy: Register Policy classes
   - register_reward_model: Register RewardModel classes
   - ComponentRegistry: Central registry class for components

2. Prompt registration decorators (from lits.prompts.registry):
   - register_system_prompt: Register system prompts
   - register_user_prompt: Register user prompts
   - PromptRegistry: Central registry class for prompts

3. Dataset registration functions (from lits.benchmarks.registry):
   - register_dataset: Register dataset loader functions
   - load_dataset: Load datasets by name
   - infer_task_type: Infer task type from dataset name
   - BenchmarkRegistry: Central registry class for datasets

Usage:
    # Import all decorators from a single location
    from lits.registry import (
        register_transition,
        register_policy,
        register_reward_model,
        register_dataset,
        register_system_prompt,
        register_user_prompt,
    )
    
    # Register a custom env_grounded Transition
    @register_transition("robot_arm", task_type="env_grounded")
    class RobotArmTransition(EnvGroundedTransition):
        @staticmethod
        def goal_check(target, current):
            ...
        
        @staticmethod
        def generate_actions(state):
            ...
    
    # Register a dataset loader
    @register_dataset("robot_arm", task_type="env_grounded")
    def load_robot_arm_data(config_file: str):
        ...
    
    # Register custom prompts for language_grounded tasks
    @register_system_prompt("policy", "concat", "my_math_task")
    def my_math_system_prompt():
        return "You are solving math problems step by step..."
    
    @register_user_prompt("policy", "concat", "my_math_task")
    def my_math_user_prompt():
        return {"question_format": "Problem: {question}"}

Lookup:
    from lits.registry import ComponentRegistry, BenchmarkRegistry, PromptRegistry
    
    # Look up components
    TransitionCls = ComponentRegistry.get_transition("robot_arm")
    PolicyCls = ComponentRegistry.get_policy("my_task")
    
    # Load datasets
    data = load_dataset("robot_arm", config_file="config.yaml")
    
    # Infer task type
    task_type = infer_task_type("robot_arm")  # Returns "env_grounded"
    
    # Look up prompts
    prompt = PromptRegistry.get("policy", "concat", task_name="my_math_task")
"""

# Component registration decorators
from lits.components.registry import (
    register_transition,
    register_policy,
    register_reward_model,
    ComponentRegistry,
)

# Prompt registration decorators
from lits.prompts.registry import (
    register_system_prompt,
    register_user_prompt,
    PromptRegistry,
)

# Dataset registration functions
from lits.benchmarks.registry import (
    register_dataset,
    load_dataset,
    infer_task_type,
    BenchmarkRegistry,
)

__all__ = [
    # Component decorators
    "register_transition",
    "register_policy",
    "register_reward_model",
    "ComponentRegistry",
    # Prompt decorators
    "register_system_prompt",
    "register_user_prompt",
    "PromptRegistry",
    # Dataset functions
    "register_dataset",
    "load_dataset",
    "infer_task_type",
    "BenchmarkRegistry",
]
