"""Generic fallback prompts for EnvGroundedPolicy.

These prompts serve as fallback when benchmark-specific prompts are not registered.
They use generic language that works across all env_grounded benchmarks.

Lookup priority in PromptRegistry:
1. task_name (benchmark-specific, e.g., 'blocksworld', 'crosswords')
2. task_type='env_grounded' (this fallback)
3. 'default'

Example:
    If 'blocksworld' has its own prompt registered, it will be used.
    If a new benchmark 'my_puzzle' doesn't register prompts, this fallback is used.
"""

from lits.prompts.prompt import PromptTemplate
from lits.prompts.registry import register_user_prompt


@register_user_prompt('policy', 'env_grounded', 'env_grounded')
def env_grounded_policy_usr_prompt():
    """Generic user prompt for EnvGroundedPolicy.
    
    This fallback prompt works across all env_grounded benchmarks by using
    generic placeholders that are filled by the transition's state representation.
    
    Placeholders:
        - {init_state}: Current environment state (rendered by transition)
        - {goals}: Goal description or target state
        - {actions}: List of valid actions from current state
    
    Returns:
        PromptTemplate with generic env_grounded format
    """
    return PromptTemplate(
        template="""I am solving a planning problem where I need to reach a goal state from an initial state.

[CURRENT STATE]
{init_state}

[GOAL]
{goals}

[AVAILABLE ACTIONS]
{actions}

Choose ONE action from the available actions that will help reach the goal.
Output ONLY the action, nothing else.""",
        input_variables=["init_state", "goals", "actions"]
    )
