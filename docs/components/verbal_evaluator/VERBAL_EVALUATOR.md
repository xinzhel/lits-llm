# Verbal Evaluator Framework

## Overview

The Verbal Evaluator framework provides LLM-based validation and analysis of generated content (SQL queries, code, etc.) with automatic issue tracking and policy feedback generation.

**Key Features:**
- Step-level and trajectory-level evaluation
- Automatic issue tracking to `~/.lits_llm/verbal_evaluator/`
- Policy feedback generation from historical issues
- Unified `evaluate()` interface across all evaluators
- Simple file-based persistence using `ResultDictToJsonl`

## Architecture

### Base Class: VerbalEvaluator

All verbal evaluators inherit from the abstract `VerbalEvaluator` base class, which provides:

- **Unified file management**: All evaluators for the same policy/task save to the same file
- **Automatic metadata**: Adds `evaluator_type` and `timestamp` to all saved results
- **Result filtering**: Load results by evaluator type
- **Common interface**: Abstract `evaluate()` and `load_eval_as_prompt()` methods

```python
from lits.components.verbal_evaluator.base import VerbalEvaluator

class MyEvaluator(VerbalEvaluator):
    def evaluate(self, input, **kwargs) -> Optional[str]:
        """Evaluate input and return issue string."""
        # Implementation
        
    def load_eval_as_prompt(self, policy_model_name, task_type, **kwargs) -> str:
        """Load saved evaluations as prompt."""
        # Implementation
```

### Unified Interface

All verbal evaluators implement a common `evaluate()` method:

```python
def evaluate(self, input, **kwargs) -> Optional[str]:
    """Evaluate input and return a single issue string for policy feedback.
    
    Returns:
        Single string describing the issue and how to avoid it, or None if no issues
    """
```

This unified interface allows policies to use any evaluator consistently:

```python
# Works with any evaluator
issue = evaluator.evaluate(input, policy_model_name="gpt-4", task_type="spatial_qa")
if issue:
    policy.base_model.sys_prompt += f"\n\n**Note:** {issue}"
```

### Unified Issue Storage

**Key Feature:** All evaluators for the same policy model and task type save to the **same file**.

File location:
```
~/.lits_llm/verbal_evaluator/resultdicttojsonl_{model}_{task}.jsonl
```

Each record contains:
```json
{
  "evaluator_type": "sqlvalidator",
  "query_idx": 0,
  "issue": "Description of what went wrong and why",
  "timestamp": "2024-12-03 10:30:45",
  ...additional evaluator-specific fields...
}
```

**Benefits:**
- Single file per policy/task combination
- Easy to track all issues in one place
- `evaluator_type` field distinguishes which evaluator generated each result
- Simplified file management and loading

## Current Implementations

### 1. SQLValidator

**Purpose:** Step-level validation of individual SQL queries

**Input:** Single `ToolUseStep` containing a SQL action

**Output:** Validation result with specific issue

**Use Case:** Real-time validation during policy execution

**Example:**
```python
from lits.components.verbal_evaluator import SQLValidator
from lits.lm import get_lm

# Initialize
llm = get_lm("gpt-4")
validator = SQLValidator(
    base_model=llm,
    sql_tool_names=['sql_db_query', 'sql_query']
)

# Evaluate a step
issue = validator.evaluate(
    step,
    context="PostGIS database with psr_point, psr_polygon tables",
    user_intent="Find priority sites",
    query_idx=0,
    policy_model_name="gpt-4",
    task_type="spatial_qa"
)

if issue:
    print(f"Issue detected: {issue}")
    # Issue automatically saved to file
```

**Validation Criteria:**
- SQL syntax correctness
- Semantic validity (table/column references)
- Spatial commonsense (PostGIS functions, CRS logic)
- Safety (no destructive operations)
- Intent alignment

**Output Format:**
```python
# evaluate() returns:
"Using geometry in EPSG:4326 with meter-level distance thresholds causes incorrect results..."

# validate() returns full dict:
{
    'is_valid': False,
    'score': 0.3,
    'reasoning': "...",
    'issue': "Using geometry in EPSG:4326...",
    'query_idx': 0
}
```

### 2. SQLErrorProfiler

**Purpose:** Trajectory-level analysis of SQL error patterns

**Input:** Entire `TrajectoryState` (sequence of steps)

**Output:** Structured error profile with pattern analysis

**Use Case:** Post-hoc analysis to identify recurring issues

**Example:**
```python
from lits.components.verbal_evaluator import SQLErrorProfiler
from lits.structures import ToolUseState

# Initialize
profiler = SQLErrorProfiler(base_model=llm)

# Load trajectory from checkpoint
query, state = ToolUseState.load("checkpoint.json")

# Evaluate trajectory
issue = profiler.evaluate(
    state,
    query_idx=0,
    policy_model_name="gpt-4",
    task_type="spatial_qa"
)

if issue:
    print(f"Pattern detected: {issue}")
```

**Analysis Includes:**
- Both successful and failed steps
- Error type classification
- Principle-based issues (what + why)
- Comparison of what worked vs. what failed

**Output Format:**
```python
# evaluate() returns:
"Schema mismatch errors: Querying non-existent tables due to lack of schema validation..."

# profile_trajectory() returns full dict:
{
    'error_type': 'Schema mismatch errors',
    'issues': [
        "Querying non-existent tables due to lack of schema validation",
        "Using 'geometry' column which failed, instead of 'geom' which succeeded in Step 5"
    ],
    'query_idx': 0
}
```

## Comparison

| Feature | SQLValidator | SQLErrorProfiler |
|---------|-------------|------------------|
| **Scope** | Single step | Entire trajectory |
| **Input** | ToolUseStep | TrajectoryState |
| **Timing** | Real-time | Post-hoc |
| **Focus** | Specific query | Error patterns |
| **Analysis** | Syntax, semantics | Patterns, comparisons |
| **Output** | Specific issue | Generalized insights |
| **Use Case** | Prevent errors | Learn from errors |

## Loading Issues for Policy Feedback

Both evaluators provide methods to load historical issues:

### SQLValidator

```python
# Load recent issues
feedback = validator.load_issues_as_prompt(
    policy_model_name="gpt-4",
    task_type="spatial_qa",
    max_issues=10
)

# Inject into policy
policy.base_model.sys_prompt += "\n\n" + feedback
```

Output format:
```
**Previous SQL Errors to Avoid:**

1. Using geometry in EPSG:4326 with meter-level distance thresholds...
2. Querying non-existent tables without schema validation...
```

### SQLErrorProfiler

```python
# Load recent profiles
feedback = profiler.load_profiles_as_prompt(
    policy_model_name="gpt-4",
    task_type="spatial_qa",
    max_profiles=5
)

# Inject into policy
policy.base_model.sys_prompt += "\n\n" + feedback
```

Output format:
```
**Previous SQL Error Patterns to Avoid:**

1. Error Type: Schema mismatch errors
   Key Issues:
   - Querying non-existent tables due to lack of schema validation
   - Using incorrect column names from assumptions about structure

2. Error Type: Spatial CRS errors
   Key Issues:
   - Using geometry with meter-based distances in EPSG:4326
```

## Integration Patterns

### Pattern 1: Real-time Validation

```python
# During policy execution
for step in policy.get_actions(state, query):
    # Validate before execution
    issue = validator.evaluate(step, context=schema, 
                               policy_model_name=model_name,
                               task_type=task_type)
    
    if issue:
        logger.warning(f"Invalid action: {issue}")
        continue  # Skip invalid action
    
    # Execute valid action
    new_state = transition.step(state, step.action)
```

### Pattern 2: Post-hoc Learning

```python
# After trajectory completion
issue = profiler.evaluate(final_state,
                         policy_model_name=model_name,
                         task_type=task_type)

if issue:
    # Load accumulated feedback
    feedback = profiler.load_profiles_as_prompt(model_name, task_type)
    
    # Update policy for next run
    policy.base_model.sys_prompt += "\n\n" + feedback
```

### Pattern 3: Hybrid Approach

```python
# Combine both evaluators
validator = SQLValidator(base_model=llm, sql_tool_names=tools)
profiler = SQLErrorProfiler(base_model=llm)

# Real-time validation during execution
for step in steps:
    issue = validator.evaluate(step, ...)
    if issue:
        handle_invalid_step(step, issue)

# Post-hoc analysis after completion
trajectory_issue = profiler.evaluate(final_state, ...)
if trajectory_issue:
    update_policy_with_patterns(trajectory_issue)
```

## File Organization

**Unified Storage:** All evaluators for the same policy/task save to one file.

```
~/.lits_llm/verbal_evaluator/
├── resultdicttojsonl_gpt-4_spatial_qa.jsonl          # Both SQLValidator AND SQLErrorProfiler
├── resultdicttojsonl_claude_tool_use.jsonl           # Both evaluators
└── resultdicttojsonl_llama-3_math_qa.jsonl           # Both evaluators
```

**File naming:** `resultdicttojsonl_{model_name}_{task_type}.jsonl`

Files are automatically created and appended to based on:
- Policy model name (e.g., "gpt-4", "claude-3.5-sonnet")
- Task type (e.g., "spatial_qa", "tool_use")

**Record format:**
```json
{
  "evaluator_type": "sqlvalidator",
  "query_idx": 0,
  "issue": "Using geometry in EPSG:4326...",
  "is_valid": false,
  "score": 0.3,
  "timestamp": "2024-12-03 10:30:45"
}
```

```json
{
  "evaluator_type": "sqlerrorprofiler",
  "query_idx": 1,
  "error_type": "CRS mismatch errors",
  "issues": ["Using geometry with meter-based distances..."],
  "timestamp": "2024-12-03 10:35:12"
}
```

The `evaluator_type` field distinguishes which evaluator generated each result.

## Best Practices

### 1. Use Descriptive Model/Task Names

```python
# Good
policy_model_name = "gpt-4-turbo"
task_type = "spatial_qa_postgis"

# Avoid
policy_model_name = "model1"
task_type = "task"
```

### 2. Provide Rich Context

```python
# Good
context = """
PostGIS database with tables:
- psr_point(gid, geom, address, issue)
- psr_polygon(gid, geom, address, issue)
Geometry column: 'geom' (not 'geometry')
CRS: EPSG:4283
"""

# Avoid
context = "PostGIS database"
```

### 3. Regular Feedback Updates

```python
# Load feedback periodically
if iteration % 10 == 0:
    feedback = validator.load_issues_as_prompt(model_name, task_type)
    policy.base_model.sys_prompt = base_prompt + "\n\n" + feedback
```

### 4. Monitor Issue Files

```python
from pathlib import Path

issue_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model}_{task}.jsonl"

if issue_file.exists():
    with open(issue_file) as f:
        issue_count = sum(1 for _ in f)
    print(f"Total issues logged: {issue_count}")
```

## Extending the Framework

To create a new verbal evaluator, inherit from `VerbalEvaluator`:

### 1. Inherit from Base Class

```python
from lits.components.verbal_evaluator.base import VerbalEvaluator
from typing import Optional

class MyCustomEvaluator(VerbalEvaluator):
    """Custom evaluator for analyzing X."""
    
    def __init__(self, base_model, custom_param: str, **kwargs):
        # Initialize parent
        super().__init__(
            base_model=base_model,
            temperature=kwargs.get('temperature', 0.0),
            max_new_tokens=kwargs.get('max_new_tokens', 500)
        )
        
        self.custom_param = custom_param
        # evaluator_type is automatically set to 'mycustomevaluator'
```

### 2. Implement Required Methods

```python
    def evaluate(self, input, **kwargs) -> Optional[str]:
        """Evaluate input and return issue string.
        
        Args:
            input: The input to evaluate
            query_idx: Optional query index
            policy_model_name: Policy model name (for saving)
            task_type: Task type (for saving)
            **kwargs: Additional parameters
        
        Returns:
            Single issue string, or None if no issues
        """
        # Your evaluation logic
        result = self._analyze(input)
        
        # Save if policy info provided
        if result and kwargs.get('policy_model_name') and kwargs.get('task_type'):
            save_result = {
                'issue': result['issue'],
                'custom_field': result.get('custom_field')
            }
            self._save_eval(
                save_result,
                kwargs.get('query_idx'),
                kwargs['policy_model_name'],
                kwargs['task_type']
            )
        
        # Return single issue string
        return result.get('issue') if result else None
    
    def load_eval_as_prompt(
        self,
        policy_model_name: str,
        task_type: str,
        max_items: int = 10
    ) -> str:
        """Load saved evaluations as prompt.
        
        Args:
            policy_model_name: Policy model name
            task_type: Task type
            max_items: Maximum number of items to include
        
        Returns:
            Formatted prompt string
        """
        # Load results filtered by this evaluator type
        results = self.load_results(
            policy_model_name,
            task_type,
            evaluator_type=self.evaluator_type
        )
        
        if not results:
            return ""
        
        # Get recent items
        recent = results[-max_items:]
        
        # Format as prompt
        prompt_parts = ["**Previous Issues:**\n"]
        for idx, record in enumerate(recent, 1):
            issue = record.get('issue', '')
            if issue:
                prompt_parts.append(f"{idx}. {issue}\n")
        
        return "\n".join(prompt_parts)
```

### 3. Benefits of Inheriting from VerbalEvaluator

The base class automatically provides:

- **`_get_result_saver()`**: Unified file management
- **`_save_eval()`**: Automatic metadata addition (evaluator_type, timestamp)
- **`load_results()`**: Load and filter results by evaluator type
- **`evaluator_type`**: Automatic type identification from class name

### 4. Example: Complete Custom Evaluator

```python
from lits.components.verbal_evaluator.base import VerbalEvaluator
from typing import Optional, Dict, Any

class CodeStyleEvaluator(VerbalEvaluator):
    """Evaluates code style and best practices."""
    
    STYLE_PROMPT = "You are a code style expert. Evaluate code for PEP8 compliance..."
    
    def __init__(self, base_model, **kwargs):
        super().__init__(base_model=base_model, **kwargs)
        self.base_model.sys_prompt = self.STYLE_PROMPT
    
    def evaluate(
        self,
        code: str,
        query_idx: Optional[int] = None,
        policy_model_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[str]:
        """Evaluate code style."""
        # Call LLM
        response = self.base_model(
            f"Evaluate this code:\n```python\n{code}\n```",
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        
        # Parse response
        result = self._parse_response(response.text)
        
        # Save if policy info provided
        if result and policy_model_name and task_type:
            save_result = {
                'issue': result['issue'],
                'severity': result.get('severity', 'medium')
            }
            self._save_eval(save_result, query_idx, policy_model_name, task_type)
        
        return result.get('issue')
    
    def load_eval_as_prompt(self, policy_model_name, task_type, max_issues=10):
        results = self.load_results(policy_model_name, task_type, 
                                    evaluator_type=self.evaluator_type)
        
        if not results:
            return ""
        
        recent = results[-max_issues:]
        prompt = "**Code Style Issues to Avoid:**\n"
        for idx, r in enumerate(recent, 1):
            prompt += f"{idx}. {r['issue']} (severity: {r.get('severity', 'N/A')})\n"
        
        return prompt
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        # Your parsing logic
        return {'issue': text[:200], 'severity': 'medium'}
```

### 5. Usage

```python
# Create evaluator
evaluator = CodeStyleEvaluator(base_model=llm)

# Evaluate
issue = evaluator.evaluate(
    code="def foo( x,y ):\n  return x+y",
    query_idx=0,
    policy_model_name="gpt-4",
    task_type="code_gen"
)

# Load feedback
feedback = evaluator.load_eval_as_prompt("gpt-4", "code_gen")
policy.base_model.sys_prompt += "\n\n" + feedback
```

The evaluator will automatically:
- Save to `~/.lits_llm/verbal_evaluator/resultdicttojsonl_gpt-4_code_gen.jsonl`
- Add `evaluator_type: "codestyleevaluator"` to each record
- Share the file with other evaluators for the same policy/task

## See Also

- [SQL Validator API](./SQL_VALIDATOR.md)
- [SQL Error Profiler API](./SQL_ERROR_PROFILER.md)
- [Result Savers](../../eval/RESULT_SAVERS.md)
