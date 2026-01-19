# Adding New Prompts to the Framework

This guide covers how to add new prompts to LiTS for custom tasks or components.

## Step 1: Create Prompt File

Create a new file in the appropriate prompts directory:

```python
# lits/prompts/policy/my_agent.py

task_prompt_spec_language_grounded = """
Your system instructions here...
"""

usr_prompt_spec_language_grounded = {
    'question_format': 'Question: {question}',
    'answer_format': 'Answer: {answer}'
}
```

## Step 2: Register in load_default_prompts()

Add registration in the prompt registry initialization:

```python
# In lits/prompts/registry.py
def load_default_prompts():
    from .policy import my_agent
    
    # prompt_key can be a benchmark name (e.g., 'blocksworld') 
    # or task type (e.g., 'language_grounded')
    if hasattr(my_agent, 'task_prompt_spec_language_grounded'):
        PromptRegistry.register(
            'policy', 'my_agent', 'language_grounded',
            my_agent.task_prompt_spec_language_grounded
        )
    
    if hasattr(my_agent, 'usr_prompt_spec_language_grounded'):
        PromptRegistry.register_usr(
            'policy', 'my_agent', 'language_grounded',
            my_agent.usr_prompt_spec_language_grounded
        )
```

## Step 3: Implement Component

Create the component that uses the prompts:

```python
from ..base import Policy

class MyAgentPolicy(Policy):
    TASK_TYPE = "language_grounded"
    
    def _get_agent_name(self) -> str:
        return 'my_agent'  # Must match registry key
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # self.task_prompt_spec and self.usr_prompt_spec are auto-loaded
        ...
```

## Registered Prompts Reference

### Policy Prompts

| Agent | Prompt Key | System Prompt | User Template |
|-------|------------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `concat` | `language_grounded` | ✓ | — |
| `env_grounded` | `env_grounded` (fallback) | — | ✓ |
| `env_grounded` | `blocksworld` | — | ✓ |
| `tool_use` | `default` | ✓ | — |

### Reward Prompts

| Agent | Prompt Key | System Prompt | User Template |
|-------|------------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | — |
| `generative` | `language_grounded` | ✓ | — |
| `env_grounded` | `env_grounded` (fallback) | ✓ | ✓ |
| `env_grounded` | `blocksworld` | ✓ | ✓ |

### Transition Prompts

| Agent | Prompt Key | System Prompt | User Template |
|-------|------------|---------------|---------------|
| `rap` | `language_grounded` | ✓ | ✓ |
| `rap` | `default` | ✓ | — |
| `blocksworld` | `default` | ✓ | ✓ |

## See Also

- [LITS_DESIGN.md](../LITS_DESIGN.md) - Framework architecture overview
- [Prompt Registry](../../lits/prompts/registry.py) - Registry implementation
