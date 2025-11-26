# Prompt Injection Design

## Overview

This document describes the design for prompt management and injection in LITS-LLM, allowing users to customize prompts for different agents and task types.

LITS-LLM uses a dual-registry system to manage two types of prompts:
- **System Prompts** (`task_prompt_spec`): Instructions, format specifications, and examples
- **User Message Templates** (`usr_prompt_spec`): Structures for formatting user-specific information

## Design Principles

1. **Centralized Registry** - All prompts managed in separate registries for system and user prompts
2. **Explicit Over Implicit** - Users can see and control all prompts
3. **Task-Specific Overrides** - Support task-specific prompt variations
4. **Easy Injection** - Simple API for custom prompts
5. **Separation of Concerns** - System prompts and user message templates are managed independently
6. **Type Safety** - usr_prompt_spec must be dict or PromptTemplate (not string) to ensure proper structure

## Architecture

### 1. Prompt Registry

The registry provides two separate storage systems for system prompts and user message templates.

### 2. Component Base Classes

All LLM-based components (Policy, RewardModel, LlmTransition) support independent loading of both prompt types from their respective registries.

### 3. Prompt Loading Behavior

- `task_prompt_spec` and `usr_prompt_spec` are loaded **independently**
- No priority between them - they serve different purposes
- If not explicitly provided, both are loaded from their respective registries
- Explicit values always override registry values

## Registered Prompts

### System Prompts (task_prompt_spec)

#### Policy
- rap/math_qa: `lits.prompts.policy.rap.task_prompt_spec_math_qa`
- rest/math_qa: `lits.prompts.policy.rest.task_prompt_spec_math_qa`
- tool_use/default: `lits.prompts.policy.tool_use.task_prompt_spec`

#### Transition
- rap/math_qa: `lits.prompts.transition.rap.task_prompt_spec_math_qa`
- blocksworld/default: `lits.prompts.transition.blocksworld.task_prompt_spec`

### User Message Templates (usr_prompt_spec)

#### Policy
- rap/math_qa: `lits.prompts.policy.rap.usr_prompt_spec_math_qa`

#### Transition
- rap/math_qa: `lits.prompts.transition.rap.usr_prompt_spec_math_qa`
- blocksworld/default: `lits.prompts.transition.blocksworld.usr_prompt_spec`

## Usage Guide

### Method 1: Direct Injection

Pass prompts directly when creating components:

```python
from lits.lm import get_lm
from lits.components.policy.rap import RAPPolicy

model = get_lm("Qwen/Qwen2.5-7B-Instruct")

# Inject custom system prompt
custom_system = """Given a question, decompose it into sub-questions.
For each sub-question, provide a complete answer."""

policy = RAPPolicy(
    base_model=model,
    task_prompt_spec=custom_system,
    task_type='math_qa'  # usr_prompt_spec loaded from registry
)

# Inject custom user template
custom_user_template = {
    'question_prefix': 'Q{idx}: {question}',
    'subquestion_prefix': 'Q{idx}.{sub_idx}:',
    'answer_prefix': 'A{idx}.{sub_idx}:'
}

policy = RAPPolicy(
    base_model=model,
    usr_prompt_spec=custom_user_template,
    task_type='math_qa'  # task_prompt_spec loaded from registry
)

# Inject both prompts
policy = RAPPolicy(
    base_model=model,
    task_prompt_spec=custom_system,
    usr_prompt_spec=custom_user_template
)
```

### Method 2: Registry Injection

Register prompts globally for reuse across components:

```python
from lits.prompts.registry import PromptRegistry

# Register system prompt
PromptRegistry.register(
    component_type='policy',
    agent_name='rap',
    task_type='my_custom_task',
    prompt_spec='Custom system instructions...'
)

# Register user template
PromptRegistry.register_usr(
    component_type='policy',
    agent_name='rap',
    task_type='my_custom_task',
    usr_prompt_spec={
        'question_format': 'Question: {question}',
        'answer_format': 'Answer: {answer}'
    }
)

# Now all RAP policies for 'my_custom_task' use these prompts
policy = RAPPolicy(base_model=model, task_type='my_custom_task')
```

### Method 3: Override Registry for Specific Instance

```python
# Register default prompts
PromptRegistry.register('policy', 'rap', 'math_qa', default_system_prompt)
PromptRegistry.register_usr('policy', 'rap', 'math_qa', default_user_template)

# Override for specific instance
policy = RAPPolicy(
    base_model=model,
    task_type='math_qa',
    task_prompt_spec=custom_system_prompt  # Overrides registry
)
```

### Method 4: Task-Type Selection

Specify task type to automatically load task-specific prompts:

```python
# Loads math_qa-specific prompts from registry
policy = RAPPolicy(base_model=model, task_type='math_qa')

# Loads tool_use-specific prompts
from lits.components.policy.tool_use import ToolUsePolicy
policy = ToolUsePolicy(base_model=model, tools=my_tools, task_type='tool_use')
```

## Code Examples

### Example 1: Using Default Prompts

```python
from lits.lm import get_lm
from lits.components.policy.rap import RAPPolicy

model = get_lm("Qwen/Qwen2.5-7B-Instruct")

# Both task_prompt_spec and usr_prompt_spec loaded from registry
policy = RAPPolicy(base_model=model, task_type='math_qa')

# Access the prompts
print("System prompt:", policy.task_prompt_spec)
print("User template:", policy.usr_prompt_spec)
```

### Example 2: Custom System Prompt with Default User Template

```python
custom_system = """Given a question, please decompose it into sub-questions.
For each sub-question, please answer it in a complete sentence."""

# usr_prompt_spec will be loaded from registry
policy = RAPPolicy(
    base_model=model,
    task_prompt_spec=custom_system,
    task_type='math_qa'
)
```

### Example 3: Custom User Template with Default System Prompt

```python
custom_user_template = {
    'question_prefix': 'Problem {idx}: {question}',
    'subquestion_prefix': 'Step {idx}.{sub_idx}:',
    'answer_prefix': 'Solution {idx}.{sub_idx}:',
    'overall_question_prefix': 'Final Answer:'
}

# task_prompt_spec will be loaded from registry
policy = RAPPolicy(
    base_model=model,
    usr_prompt_spec=custom_user_template,
    task_type='math_qa'
)
```

### Example 4: Working with Transitions

```python
from lits.components.transition.rap import RAPTransition

# Both prompts loaded from registry
transition = RAPTransition(base_model=model, task_type='math_qa')

# Access prompts
print("System prompt:", transition.task_prompt_spec)
print("User template:", transition.usr_prompt_spec)

# Custom prompts
transition = RAPTransition(
    base_model=model,
    task_prompt_spec="Custom transition instructions",
    usr_prompt_spec={'format': 'custom format'}
)
```

### Example 5: BlocksWorld with Custom Templates

```python
from lits.components.transition.blocksworld import BlocksWorldTransition

# Custom user templates for different actions
custom_templates = {
    'world_update_pickup': "Custom pickup template: {}",
    'world_update_putdown': "Custom putdown template: {}",
    'world_update_stack': "Custom stack template: {}",
    'world_update_unstack': "Custom unstack template: {}"
}

def goal_check(goals, env_state):
    return False, 0.0

transition = BlocksWorldTransition(
    base_model=model,
    goal_check=goal_check,
    usr_prompt_spec=custom_templates
)
```

## Adding New Prompts

### Step 1: Create Prompt File

Create a new file in the appropriate prompts directory:

```python
# lits/prompts/policy/my_agent.py

# System prompt (default)
task_prompt_spec = """
Your default system instructions here...
Include examples, format specifications, etc.
"""

# Task-specific system prompt
task_prompt_spec_math_qa = """
Math-specific system instructions...
Include math-specific examples and guidelines.
"""

# User message template (task-specific)
usr_prompt_spec_math_qa = {
    'question_format': 'Question {idx}: {question}',
    'answer_format': 'Answer {idx}: {answer}',
    'step_format': 'Step {idx}.{sub_idx}: {step}'
}
```

### Step 2: Register in load_default_prompts()

Add registration code in `lits/prompts/registry.py`:

```python
def load_default_prompts():
    try:
        from .policy import my_agent
        
        # Register system prompts
        if hasattr(my_agent, 'task_prompt_spec'):
            PromptRegistry.register(
                'policy', 'my_agent', None, 
                my_agent.task_prompt_spec
            )
        
        if hasattr(my_agent, 'task_prompt_spec_math_qa'):
            PromptRegistry.register(
                'policy', 'my_agent', 'math_qa',
                my_agent.task_prompt_spec_math_qa
            )
        
        # Register user templates
        if hasattr(my_agent, 'usr_prompt_spec_math_qa'):
            PromptRegistry.register_usr(
                'policy', 'my_agent', 'math_qa',
                my_agent.usr_prompt_spec_math_qa
            )
            
    except ImportError as e:
        logging.warning(f"Could not load prompts: {e}")
```

### Step 3: Implement Component

Create your component class:

```python
# lits/components/policy/my_agent.py

from ..base import Policy

class MyAgentPolicy(Policy):
    def _get_agent_name(self) -> str:
        return 'my_agent'  # Must match registry key
    
    def _get_actions(self, state, n_actions, temperature, **kwargs):
        # Use self.task_prompt_spec for system message
        system_message = self.task_prompt_spec
        
        # Use self.usr_prompt_spec for user message formatting
        user_message = self.usr_prompt_spec['question_format'].format(
            idx=1,
            question=kwargs.get('query', '')
        )
        
        # Generate actions using the prompts
        ...
```

### Step 4: Use Your Component

```python
from lits.components.policy.my_agent import MyAgentPolicy

# Prompts automatically loaded from registry
policy = MyAgentPolicy(base_model=model, task_type='math_qa')

# Or with custom prompts
policy = MyAgentPolicy(
    base_model=model,
    task_prompt_spec="Custom instructions",
    usr_prompt_spec={'format': 'custom'}
)
```

## Best Practices

### 1. When to Use task_prompt_spec vs usr_prompt_spec

**Use `task_prompt_spec` for:**
- System-level instructions that don't change per query
- Task descriptions and objectives
- Output format specifications
- Few-shot examples
- General guidelines and constraints

**Use `usr_prompt_spec` for:**
- Query-specific formatting that changes per input
- Dynamic content structure with placeholders
- Action-specific templates
- State-dependent formatting

### 2. Type Guidelines

```python
# ✓ GOOD: task_prompt_spec can be string
task_prompt_spec = "Solve the problem step by step..."

# ✓ GOOD: task_prompt_spec can be dict
task_prompt_spec = {
    'instruction': 'Solve problems...',
    'examples': ['Example 1...', 'Example 2...']
}

# ✓ GOOD: task_prompt_spec can be PromptTemplate
from lits.prompts.prompt import PromptTemplate
task_prompt_spec = PromptTemplate("Template with {placeholders}")

# ✓ GOOD: usr_prompt_spec as dict
usr_prompt_spec = {
    'question_prefix': 'Q: {question}',
    'answer_prefix': 'A: {answer}'
}

# ✓ GOOD: usr_prompt_spec as PromptTemplate
usr_prompt_spec = PromptTemplate({
    'question': 'Q: {question}',
    'answer': 'A: {answer}'
})

# ✗ BAD: usr_prompt_spec should NOT be plain string
usr_prompt_spec = "Q: {question}"  # Use dict instead!
```

### 3. Registry Organization

```python
# Register default prompts (task_type=None)
PromptRegistry.register('policy', 'my_agent', None, default_system)
PromptRegistry.register_usr('policy', 'my_agent', None, default_user)

# Register task-specific prompts
PromptRegistry.register('policy', 'my_agent', 'math_qa', math_system)
PromptRegistry.register_usr('policy', 'my_agent', 'math_qa', math_user)

# Task-specific prompts take precedence over defaults
policy = MyAgentPolicy(base_model=model, task_type='math_qa')
# Uses math_system and math_user

policy = MyAgentPolicy(base_model=model, task_type=None)
# Uses default_system and default_user
```

### 4. Testing Custom Prompts

```python
# Test with custom prompts before registering
test_policy = RAPPolicy(
    base_model=model,
    task_prompt_spec=my_custom_system,
    usr_prompt_spec=my_custom_user
)

# Run tests...

# If tests pass, register for production use
PromptRegistry.register('policy', 'rap', 'my_task', my_custom_system)
PromptRegistry.register_usr('policy', 'rap', 'my_task', my_custom_user)
```

## Benefits

1. **Separation of Concerns** - System instructions and user formatting are managed independently
2. **Type Safety** - usr_prompt_spec enforces structured templates (dict/PromptTemplate)
3. **Flexibility** - Override either prompt type independently without affecting the other
4. **Discoverability** - `PromptRegistry.list_registered()` shows all available prompts
5. **Maintainability** - Centralized prompt management with clear organization
6. **Testability** - Easy to test with different prompt combinations
7. **Reusability** - Register once, use across multiple component instances
