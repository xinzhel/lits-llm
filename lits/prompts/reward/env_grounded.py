"""Generic fallback prompts for EnvGroundedPRM (Process Reward Model).

These prompts serve as fallback when benchmark-specific prompts are not registered.
They use generic language that works across all env_grounded benchmarks.

Lookup priority in PromptRegistry:
1. task_name (benchmark-specific, e.g., 'blocksworld', 'crosswords')
2. task_type='env_grounded' (this fallback)
3. 'default'

Note: EnvGroundedPRM uses sample_binary_output with tokens "good", "bad", "unknown".
The prompts must instruct the LLM to output one of these three words.
"""

from lits.prompts.registry import register_system_prompt, register_user_prompt


@register_system_prompt('reward', 'env_grounded', 'env_grounded')
def env_grounded_reward_task_prompt():
    """Generic system prompt for EnvGroundedPRM.
    
    This fallback prompt instructs the LLM to evaluate whether a proposed action
    is good, bad, or unknown based on the current state and goal.
    
    Returns:
        str: System prompt for action evaluation
    """
    return """You are evaluating actions in a planning problem.

## EVALUATION CRITERIA

### 1. ACTION VALIDITY
Is the action valid given the current state?
- **good**: Action is valid and can be executed
- **bad**: Action is invalid or cannot be executed
- **unknown**: Cannot determine validity

### 2. GOAL PROGRESS
Does the action make progress toward the goal?
- **good**: Action moves closer to the goal state
- **bad**: Action moves away from the goal or is counterproductive
- **unknown**: Cannot determine if action helps reach the goal

### 3. EFFICIENCY
Is the action part of a reasonable solution path?
- **good**: Action is efficient and doesn't create unnecessary steps
- **bad**: Action creates unnecessary work or undoes previous progress
- **unknown**: Cannot determine efficiency

## OUTPUT FORMAT
Respond with ONLY one word: good, bad, or unknown"""


@register_user_prompt('reward', 'env_grounded', 'env_grounded')
def env_grounded_reward_usr_prompt():
    """Generic user prompt for EnvGroundedPRM.
    
    Placeholders (using <placeholder> syntax for EnvGroundedPRM compatibility):
        - <init_state>: Current environment state
        - <goals>: Goal description or target state
        - <action>: Proposed action to evaluate
    
    Returns:
        str: User prompt template with placeholders
    """
    return """[CURRENT STATE]
<init_state>

[GOAL]
<goals>

[PROPOSED ACTION]
<action>

[EVALUATION]
Is this action good, bad, or unknown?"""
