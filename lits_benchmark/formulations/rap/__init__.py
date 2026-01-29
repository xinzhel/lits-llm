"""RAP (Reasoning via Planning) formulation for LITS.

This module registers RAP components with ComponentRegistry when imported,
enabling the RAP search framework for sub-question decomposition tasks.

Usage:
    python main_search.py --import lits_benchmark.formulations.rap --search_framework rap ...

Components registered:
- RAPPolicy: Generates candidate sub-questions
- RAPTransition: Executes sub-questions with confidence estimation
- RapPRM: Evaluates sub-question usefulness

Example:
    python main_search.py \\
        --import lits_benchmark.formulations.rap \\
        --search_framework rap \\
        --dataset gsm8k \\
        --policy_model_name "meta-llama/Llama-3-8B-Instruct" \\
        --search-arg n_actions=3 \\
        --search-arg max_steps=10
"""

# Import structures first (needed by components)
from .structures import SubQAStep

# Import components to trigger registration with ComponentRegistry
from .policy import RAPPolicy
from .transition import RAPTransition
from .reward import RapPRM

# Import prompts for external use
from .prompts import (
    task_prompt_spec_math_qa,
    usr_prompt_spec_math_qa,
    reward_prompt_spec_math_qa,
)

__all__ = [
    # Structures
    'SubQAStep',
    # Components
    'RAPPolicy',
    'RAPTransition',
    'RapPRM',
    # Prompts
    'task_prompt_spec_math_qa',
    'usr_prompt_spec_math_qa',
    'reward_prompt_spec_math_qa',
]
